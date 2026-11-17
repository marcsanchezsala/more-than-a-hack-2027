"""Learned emissions + Viterbi possession model v1.1.

Same architecture as model_v10 (two LightGBM classifiers trained only on the 10 open external
matches, 3-state Viterbi), now labelling BAR-ATH on its own: model_v09's dead-ball mask is gone.

  - training data closer to BAR-ATH, every step keyed on the tracking and never on a label:
    the restart taker carries the dead ball (carry_ball), the ball is lost in few long gaps
    instead of many short ones (ball_gaps), and the tracked-player count is not a feature;
  - features for the home/away decision, which is where the error was (--bench): 0.2 and 0.5 s
    windows, touches seen as kinks of the ball velocity, ball velocity along the attack, players
    of each team around the ball, team shape;
  - models are trained on the N=25 label convention (possessions shorter than 1 s absorbed);
  - validation is leave-one-PROVIDER-out as well as leave-one-match-out, against the raw labels
    and two conventions, plus label-free checks that BAR-ATH behaves like a match of a provider
    seen in training and not like one of a provider never seen;
  - before the tagged kick-off of each half the ball is dead (laws of the game, as model_v07-v09).

No BAR-ATH label (public window included) is read to fit or choose anything.

  python model_v11.py --bench   # evidence tables on the 10 external matches + transfer checks
  python model_v11.py --domain  # label-free comparison of BAR-ATH and the degraded external data
  python model_v11.py           # train on all 10, label BAR-ATH -> evaluation/results/model_v11.csv
"""

import io
import os
import time
import argparse
import warnings
import contextlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

import evaluation.possession_metrics as pm
from games import GAME, GAMES, half_video_blocks, require, source_frames
from starter.baseline import team_of_each_track
from starter.load_smart_tagging import load as load_tags
from bench_external.run import extract_features_from_arrays, predict_match_v09, predict_game_v09
from bench_external.fetch import ALL_MATCHES, DFL_MATCHES, CACHE_DIR
from bench_external.degrade import degrade_match, slow_noise, BALL_NOISE_SIGMA
from bench_external.convention import smooth_label_convention

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
GT_CSV = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

SEED = 42
CONV_N = 25                 # label convention the models are trained on (frames, see --bench)
SUBSAMPLE = 3               # train on every 3rd frame: neighbouring frames are near-duplicates
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                  deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)
# Viterbi transition costs (nats), from a leave-one-match-out grid on the external matches (v1.0).
COSTS = dict(p_switch=20.0, p_enter_d=40.0, p_exit_d=40.0)
STATES = np.array(["H", "A", "D"], dtype=object)
WINDOWS = (25, 75, 250)     # frames, rolling windows of the base features
# Ball visibility depends on play intensity: in BAR-ATH the ball is lost mostly while play is stopped.
SPEED_BINS = np.array([0.8, 1.1, 1.4, 1.7, 2.0, 2.4, 2.8])   # m/s, mean player speed over 3 s
# DFL freezes the ball (identical position) during stoppages; BAR-ATH never does. A slow 1 m drift on
# frozen frames brings low-intensity ball speed to BAR-ATH's level (label-free, see RESULTS_v1.0).
FROZEN_DRIFT = 1.0          # metres
# Ball smoothing and aerial-ball detection
SMOOTH_MAX_GAP = 125        # frames; longer ball gaps are not bridged
AERIAL_WINDOW = 14          # frames
AERIAL_SPEED = 9.5          # m/s
AERIAL_LINEARITY = 0.94     # straight-line displacement / path length
# Dead-ball carry (label-free: keyed on the frozen ball, never on a label).
CARRY_MIN = 50              # frames; shorter frozen runs are left alone
CARRY_FRAC = 0.35           # mean share of a stoppage the taker carries the ball; matches BAR-ATH (--domain)
CARRY_OFFSET = 0.4          # metres, ball-to-carrier wander
CARRY_GAP = 25              # frames the ball is lost while it is picked up
GAP_SMOOTH = 300            # frames, correlation of the ball-dropout field (model_v10: 50)
KINK_LAG = 5                # frames; velocity change is measured over +-KINK_LAG
KINK_DV = 4.0               # m/s; a larger change next to a player is a touch


def runs(mask):
    """[start, end) of every True run."""
    d = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


# ---------------------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------------------
SWAP_KEYS = [("min_dist_h", "min_dist_a"), ("mean_speed_h", "mean_speed_a"), ("centroid_hx", "centroid_ax")] + \
            [(f"{k}_h", f"{k}_a") for k in ("cnt3", "cnt6", "d2", "sx", "sy")]
GROUPS = {"short", "kink", "bvatt", "shape"}     # feature groups added in v11 (ablation switch)


def _roll(x, w, future):
    s = pd.Series(np.asarray(x, dtype=np.float32))
    if future:
        return s[::-1].rolling(w, min_periods=1).mean()[::-1].to_numpy(np.float32)
    return s.rolling(w, min_periods=1).mean().to_numpy(np.float32)


def _frames_since(mask, cap=500, future=False):
    """Frames since the last True (or until the next one when future=True), capped."""
    m = mask[::-1] if future else mask
    idx = np.where(m, np.arange(len(m)), -10 ** 9)
    out = np.minimum(np.arange(len(m)) - np.maximum.accumulate(idx), cap).astype(np.float32)
    return out[::-1] if future else out


def smooth_ball(bx, by, bvis, max_gap=SMOOTH_MAX_GAP, dt=0.04):
    """Ball track with gaps up to max_gap bridged linearly, then a 5-frame moving average.
    Returns (x, y, speed, visible after bridging)."""
    n = len(bx)
    smooth_x, smooth_y, smooth_vis = bx.copy(), by.copy(), bvis.copy()
    for s, e in runs(~smooth_vis):
        if e - s <= max_gap and s > 0 and e < n:
            x0, y0, x1, y1 = smooth_x[s - 1], smooth_y[s - 1], smooth_x[e], smooth_y[e]
            if not np.isnan(x0) and not np.isnan(x1):
                alpha = np.linspace(0, 1, e - s + 2)[1:-1]
                smooth_x[s:e] = x0 + alpha * (x1 - x0)
                smooth_y[s:e] = y0 + alpha * (y1 - y0)
                smooth_vis[s:e] = True
    smooth_x = pd.Series(smooth_x).interpolate(limit=5).bfill().ffill().rolling(5, center=True, min_periods=1).mean().values
    smooth_y = pd.Series(smooth_y).interpolate(limit=5).bfill().ffill().rolling(5, center=True, min_periods=1).mean().values
    dx, dy = np.gradient(smooth_x) / dt, np.gradient(smooth_y) / dt
    speed = np.sqrt(dx * dx + dy * dy)
    speed[~smooth_vis] = np.nan
    return smooth_x, smooth_y, speed, smooth_vis


def aerial_ball(sx, sy, speed, vis, window=AERIAL_WINDOW):
    """Fast, straight ball flight: players underneath it are not touching it."""
    dx, dy = np.diff(sx), np.diff(sy)
    path = pd.Series(np.sqrt(dx * dx + dy * dy)).rolling(window, min_periods=window).sum().values
    disp = np.sqrt((sx[window:] - sx[:-window]) ** 2 + (sy[window:] - sy[:-window]) ** 2)
    linearity = np.full(len(sx), np.nan, dtype=np.float32)
    valid = (path[window - 1:] > 0) & ~np.isnan(disp)
    linearity[window:] = np.where(valid, disp / np.maximum(path[window - 1:], 1e-4), np.nan)
    return (speed > AERIAL_SPEED) & (linearity > AERIAL_LINEARITY) & vis


def load_barath_arrays(game=GAME):
    """BAR-ATH tracking in the arrays the external matches use: ball (N,3), players (N,S,3), team (S,)."""
    meta, raw = require(["atd_meta", "atd_raw"], game)
    n = source_frames("tactical", game)
    team = team_of_each_track(meta)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    players = np.full((n, len(team), 3), np.nan, dtype=np.float32)
    with open(raw) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            if not 0 <= i < n:
                continue
            for k, grp in enumerate(parts[1].split(";")):
                if grp[0] != "N":
                    players[i, k] = [float(v) for v in grp.split(",")[:3]]
            if parts[2][0] != "N":
                ball[i] = [float(v) for v in parts[2].split(",")[:3]]
    scale = np.array(GAMES[game]["pitch"], dtype=np.float32)
    ball[:, :2] *= scale
    players[..., :2] *= scale
    return ball, players, team


def frame_features(ball, players, team, pitch):
    """Per-frame physical signals plus pressure around the ball and the shape of each team."""
    F = extract_features_from_arrays(ball, players, team, pitch)
    px, py = players[..., 0], players[..., 1]
    ok = np.isfinite(px) & np.isfinite(py)
    d = np.hypot(px - F["ball_x"][:, None], py - F["ball_y"][:, None])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # frames with a whole team untracked
        for t, s in ((0, "h"), (1, "a")):
            sel = ok & (team == t)
            dt = np.where(sel & np.isfinite(d), d, np.inf)
            F[f"cnt3_{s}"] = (dt < 3).sum(1).astype(np.float32)
            F[f"cnt6_{s}"] = (dt < 6).sum(1).astype(np.float32)
            d2 = np.partition(dt, 1, axis=1)[:, 1]
            F[f"d2_{s}"] = np.where(np.isfinite(d2), d2, np.nan).astype(np.float32)
            F[f"sx_{s}"] = np.nanstd(np.where(sel, px, np.nan), axis=1).astype(np.float32)
            F[f"sy_{s}"] = np.nanstd(np.where(sel, py, np.nan), axis=1).astype(np.float32)
    return F


def swap_teams(F):
    G = dict(F)
    for h, a in SWAP_KEYS:
        G[h], G[a] = F[a], F[h]
    return G


def _roll_nan(x, w, future):
    s = pd.Series(np.asarray(x, dtype=np.float32))
    r = s[::-1].rolling(w, min_periods=1).mean()[::-1] if future else s.rolling(w, min_periods=1).mean()
    return r.to_numpy(np.float32)


def ml_features(F, blocks, pitch):
    """Per-frame feature matrix. Same code for BAR-ATH and the external matches.

    base:  ball-player distances, ball visibility and speed, aerial ball, player speeds, ball
           position along home's attacking direction, and 1 / 3 / 10 s rolling means (no
           tracked-player count: it differs between providers);
    short: the same at 0.2 and 0.5 s, to resolve short possessions;
    kink:  touches seen as a change of ball velocity (over +-5 frames, the lag at which BAR-ATH and
           the external providers agree) next to a player of one team;
    bvatt: ball velocity along home's attacking direction;
    shape: players of each team around the ball, second-nearest player, spread of each team.
    """
    L, W = pitch
    n = F["n_frames"]
    vis, oob, dh, da = F["ball_vis"], F["oob"], F["min_dist_h"], F["min_dist_a"]
    sx, sy, kbs, svis = smooth_ball(F["ball_x"], F["ball_y"], vis)
    aerial = aerial_ball(sx, sy, kbs, vis)
    touch_h = vis & ~oob & (dh < 1.5) & ~(da < 2.0)
    touch_a = vis & ~oob & (da < 1.5) & ~(dh < 2.0)
    diff = np.clip(np.nan_to_num(da - dh, nan=0.0), -10, 10)

    # Ball position along home's attacking direction, per half (from the team centroids)
    bx_att = np.full(n, np.nan, dtype=np.float32)
    adv = np.zeros(n, dtype=np.float32)
    for first, last in blocks:
        sl = slice(first, last + 1)
        sign = 1.0 if np.nanmean(F["centroid_hx"][sl]) < np.nanmean(F["centroid_ax"][sl]) else -1.0
        bx_att[sl] = sign * (sx[sl] - L / 2)
        mid = 0.5 * (pd.Series(F["centroid_hx"][sl]).interpolate().bfill().ffill()
                     + pd.Series(F["centroid_ax"][sl]).interpolate().bfill().ffill())
        adv[sl] = sign * np.gradient(mid.rolling(50, center=True, min_periods=1).mean().to_numpy()) * 25.0

    cols = {
        "dh": dh, "da": da, "diff": diff, "dmin": np.fmin(dh, da),
        "vis": vis, "oob": oob, "kbs": kbs, "aerial": aerial,
        "spd": F["player_speed"], "p90": F["p90_speed"],
        "spd_h": F["mean_speed_h"], "spd_a": F["mean_speed_a"],
        "bx_att": bx_att, "by_edge": W / 2 - np.abs(sy - W / 2), "bx_edge": L / 2 - np.abs(sx - L / 2),
        "adv": adv,
        "since_vis": _frames_since(vis), "until_vis": _frames_since(vis, future=True),
        "since_th": _frames_since(touch_h), "since_ta": _frames_since(touch_a),
        "until_th": _frames_since(touch_h, future=True), "until_ta": _frames_since(touch_a, future=True),
    }
    for w in WINDOWS:
        for fut in (False, True):
            tag = f"{'f' if fut else 'p'}{w}"
            cols[f"vis_{tag}"] = _roll(vis, w, fut)
            cols[f"diff_{tag}"] = _roll(diff, w, fut)
            cols[f"kbs_{tag}"] = _roll(kbs, w, fut)
            cols[f"spd_{tag}"] = _roll(F["player_speed"], w, fut)
            cols[f"p90_{tag}"] = _roll(F["p90_speed"], w, fut)
            cols[f"oob_{tag}"] = _roll(oob, w, fut)
            cols[f"th_{tag}"] = _roll(touch_h, w, fut)
            cols[f"ta_{tag}"] = _roll(touch_a, w, fut)

    vx, vy = np.gradient(sx) * 25.0, np.gradient(sy) * 25.0
    vx[~svis] = np.nan
    vy[~svis] = np.nan
    both = [(w, fut, f"{'f' if fut else 'p'}{w}") for w in (12, 25, 75) for fut in (False, True)]

    if "short" in GROUPS:
        for w in (5, 12):
            for fut in (False, True):
                tag = f"{'f' if fut else 'p'}{w}"
                cols[f"diff_{tag}"] = _roll(diff, w, fut)
                cols[f"th_{tag}"] = _roll(touch_h, w, fut)
                cols[f"ta_{tag}"] = _roll(touch_a, w, fut)
    if "kink" in GROUPS:
        k = KINK_LAG
        dv = np.full(n, np.nan, dtype=np.float32)
        dv[k:-k] = np.hypot(vx[2 * k:] - vx[:-2 * k], vy[2 * k:] - vy[:-2 * k])
        kink = vis & ~oob & (np.nan_to_num(dv) > KINK_DV)
        kh = kink & (dh < 2.0) & ~(da < dh)
        ka = kink & (da < 2.0) & ~(dh < da)
        cols.update(dv=dv, since_kh=_frames_since(kh), since_ka=_frames_since(ka),
                    until_kh=_frames_since(kh, future=True), until_ka=_frames_since(ka, future=True))
        for w, fut, tag in both:
            cols[f"kh_{tag}"] = _roll(kh, w, fut)
            cols[f"ka_{tag}"] = _roll(ka, w, fut)
    if "bvatt" in GROUPS:
        bv = np.full(n, np.nan, dtype=np.float32)
        for first, last in blocks:
            sl = slice(first, last + 1)
            sign = 1.0 if np.nanmean(F["centroid_hx"][sl]) < np.nanmean(F["centroid_ax"][sl]) else -1.0
            bv[sl] = sign * vx[sl]
        cols["bv_att"] = bv
        for w, fut, tag in both:
            cols[f"bvatt_{tag}"] = _roll_nan(bv, w, fut)
    if "shape" in GROUPS:
        for key in ("cnt3", "cnt6", "d2", "sx", "sy"):
            cols[f"{key}_h"], cols[f"{key}_a"] = F[f"{key}_h"], F[f"{key}_a"]
        for key in ("sx", "sy"):
            dlt = F[f"{key}_h"] - F[f"{key}_a"]
            cols[f"d{key}"] = dlt
            cols[f"d{key}_p75"] = _roll_nan(dlt, 75, False)
            cols[f"d{key}_f75"] = _roll_nan(dlt, 75, True)
        for w in (12, 25):
            for fut in (False, True):
                cols[f"dcnt3_{'f' if fut else 'p'}{w}"] = _roll(F["cnt3_h"] - F["cnt3_a"], w, fut)
    names = list(cols)
    return np.column_stack([np.asarray(cols[c], dtype=np.float32) for c in names]), names


def features_both(F, blocks, pitch):
    X, names = ml_features(F, blocks, pitch)
    Xs, _ = ml_features(swap_teams(F), blocks, pitch)
    return X, Xs, names


# ---------------------------------------------------------------------------------------------
# External data
# ---------------------------------------------------------------------------------------------
def carry_ball(ball, players, frozen, rng, frac=CARRY_FRAC):
    """Let the restart taker carry the ball for the last part of each frozen (dead) stretch."""
    out = ball.copy()
    n = len(ball)
    for s, e in runs(frozen):
        if e - s < CARRY_MIN or e >= n or frac <= 0:
            continue
        d = np.hypot(players[e, :, 0] - ball[e, 0], players[e, :, 1] - ball[e, 1])
        u = rng.random()
        if not np.isfinite(d).any():
            continue
        k = int(np.nanargmin(d))                       # the player who restarts play
        t0 = e - int(min(1.0, 2 * frac * u) * (e - s))
        if t0 >= e:
            continue
        out[t0:e, :2] = players[t0:e, k, :2] + slow_noise(rng, e - t0, CARRY_OFFSET)
        out[max(s, t0 - CARRY_GAP):t0, :2] = np.nan
    return out


def intensity_bin(F):
    s = pd.Series(F["player_speed"]).rolling(75, center=True, min_periods=1).mean().to_numpy()
    return np.digitize(np.nan_to_num(s), SPEED_BINS)


def ball_miss_curve(F, blocks):
    """P(ball missing | play-intensity bin) over graded frames. Label-free."""
    g = np.zeros(F["n_frames"], dtype=bool)
    for first, last in blocks:
        g[first:last + 1] = True
    b = intensity_bin(F)
    return np.array([1 - F["ball_vis"][g & (b == k)].mean() if (g & (b == k)).any() else 0.0
                     for k in range(len(SPEED_BINS) + 1)])


def ball_gaps(clean, blocks, target, rng):
    """Frames where the ball is removed, with the probability that makes this match's ball-missing
    curve (by play intensity) match BAR-ATH's. The dropout field is correlated over GAP_SMOOTH = 300
    frames (v1.0: 50): BAR-ATH loses the ball in few long gaps, and with 50 the share of missing
    time spent in gaps of 10 s or more was 0.70 against 0.85 (--domain)."""
    own = ball_miss_curve(clean, blocks)
    q = np.clip(1 - (1 - target) / np.maximum(1 - own, 1e-6), 0, 1)
    z = gaussian_filter1d(rng.normal(size=clean["n_frames"]), GAP_SMOOTH)
    return norm.cdf(z / z.std()) < q[intensity_bin(clean)]


def load_external(match_id, idx, target, carry=CARRY_FRAC):
    """Degraded features + raw labels for one external match (deterministic, label-free).

    Players: degrade.py dropouts and jitter. Ball: noise, drift where the source froze it, the
    restart taker carrying it at the end of frozen stretches (carry_ball), then long gaps
    (ball_gaps). Labels in the N=25 convention; they are never used to degrade.
    """
    d = np.load(os.path.join(CACHE_DIR, f"{match_id}.npz"))
    blocks = [(int(b[0]), int(b[1])) for b in d["blocks"]]
    pitch = tuple(float(v) for v in d["pitch"]) if "pitch" in d else (105.0, 68.0)
    raw = d["labels"].astype(object)
    n = len(raw)
    rng = np.random.default_rng(SEED + idx * 79)
    _, d_players = degrade_match(d["ball"], d["players"], d["team"],
                                 np.full(n, "H", dtype=object), blocks, seed=SEED + idx * 79)
    clean = extract_features_from_arrays(d["ball"], d["players"], d["team"], pitch)
    d_ball = d["ball"].copy()
    d_ball[:, :2] += slow_noise(rng, n, BALL_NOISE_SIGMA)
    step = np.hypot(*np.diff(d["ball"][:, :2], axis=0, prepend=d["ball"][:1, :2]).T)
    is_frozen = np.nan_to_num(step, nan=1.0) < 1e-4
    frozen = gaussian_filter1d(is_frozen.astype(float), 12)
    d_ball[:, :2] += slow_noise(rng, n, FROZEN_DRIFT) * frozen[:, None]
    d_ball = carry_ball(d_ball, d["players"], is_frozen, rng, carry)
    d_ball[ball_gaps(clean, blocks, target, rng), :2] = np.nan
    F = frame_features(d_ball, d_players, d["team"], pitch)
    X, Xs, names = features_both(F, blocks, pitch)
    return dict(match=match_id, X=X, Xs=Xs, names=names, raw=raw, blocks=blocks, pitch=pitch, F=F,
                y=smooth_label_convention(raw, blocks, min_spell_frames=CONV_N),
                y50=smooth_label_convention(raw, blocks, min_spell_frames=50),
                d_ball=d_ball, d_players=d_players, team=d["team"])


def load_all_external(target, carry=CARRY_FRAC):
    return [load_external(m, i, target, carry) for i, m in enumerate(ALL_MATCHES)]


def barath_target(game=GAME):
    """BAR-ATH features (same code path as the external matches), ball-missing curve, graded blocks."""
    blocks = [(f, l) for _, f, l in half_video_blocks(game)]
    F = frame_features(*load_barath_arrays(game), GAMES[game]["pitch"])
    return F, ball_miss_curve(F, blocks), blocks


# ---------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------
def _train_frames(m):
    return np.flatnonzero(np.isin(m["y"], ["H", "A", "D"]))[::SUBSAMPLE]


def fit(train):
    """(alive model, team model). The team model sees every live frame twice, teams mirrored."""
    Xa, ya, Xt, yt = [], [], [], []
    for m in train:
        g = _train_frames(m)
        Xa.append(m["X"][g]); ya.append(m["y"][g] != "D")
        live = g[m["y"][g] != "D"]
        Xt += [m["X"][live], m["Xs"][live]]
        yt += [m["y"][live] == "H", m["y"][live] == "A"]
    alive = lgb.LGBMClassifier(**LGB_PARAMS).fit(np.vstack(Xa), np.concatenate(ya))
    team = lgb.LGBMClassifier(**LGB_PARAMS).fit(np.vstack(Xt), np.concatenate(yt))
    return alive, team


def player_cols(names):
    return [i for i, k in enumerate(names) if k.split("_")[0] in ("spd", "p90", "adv")]


def fit_players(train):
    """Alive model that sees player speeds only. Never used to label: it is the second opinion
    behind the label-free shift indicator (shift_indicator)."""
    cols = player_cols(train[0]["names"])
    X = np.vstack([m["X"][_train_frames(m)][:, cols] for m in train])
    y = np.concatenate([m["y"][_train_frames(m)] != "D" for m in train])
    return lgb.LGBMClassifier(**LGB_PARAMS).fit(X, y), cols


def shift_indicator(p_full, p_players):
    """% of frames where the players-only model is sure the ball is dead and the full model calls
    it alive. Needs no labels; low when the ball behaves as in training (see --bench)."""
    return 100.0 * np.mean((p_players < 0.1) & (p_full > 0.5))


def log_emissions(models, X, Xs):
    """Per-frame log P(H), log P(A), log P(D). P(home) averages the frame and its team-mirrored copy."""
    alive, team = models
    p_live = np.clip(alive.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
    p_h = 0.5 * (team.predict_proba(X)[:, 1] + team.predict_proba(Xs)[:, 0])
    p_h = np.clip(p_h, 1e-4, 1 - 1e-4)
    return np.column_stack([np.log(p_live) + np.log(p_h), np.log(p_live) + np.log1p(-p_h), np.log1p(-p_live)])


def viterbi(E, p_switch, p_enter_d, p_exit_d):
    """Plain 3-state Viterbi (H, A, D) with constant transition costs."""
    T = np.array([[0.0, -p_switch, -p_enter_d],
                  [-p_switch, 0.0, -p_enter_d],
                  [-p_exit_d, -p_exit_d, 0.0]])
    n = len(E)
    back = np.empty((n, 3), dtype=np.int8)
    dp = E[0].copy()
    for t in range(1, n):
        s = dp[:, None] + T
        back[t] = s.argmax(0)
        dp = s.max(0) + E[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = dp.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(E, blocks, n, costs=COSTS):
    out = np.full(n, "X", dtype=object)
    for first, last in blocks:
        out[first:last + 1] = STATES[viterbi(E[first:last + 1], **costs)]
    return out


def force_dead(E, dead):
    """Keep the learned H/A emissions but take alive/dead from another model (--bench comparison)."""
    E = E.copy()
    E[dead, :2] = -1e9
    E[~dead, 2] = -1e9
    return E


def score(y, pred):
    r = pm.evaluate(pm.label_codes(y), pm.label_codes(pred))
    return r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"]


# ---------------------------------------------------------------------------------------------
# External benchmark: leave-one-match-out and leave-one-provider-out
# ---------------------------------------------------------------------------------------------
def is_dfl(m):
    return m["match"] in DFL_MATCHES


def out_of_fold(data):
    """Per scheme and match: emissions and the shift indicator, from models that never saw the
    match (lomo) or its provider (lopo)."""
    folds = {"lomo": [([o for o in data if o is not m], [m]) for m in data],
             "lopo": [([o for o in data if is_dfl(o) != grp], [m for m in data if is_dfl(m) == grp])
                      for grp in (True, False)]}
    oof = {}
    for scheme, parts in folds.items():
        oof[scheme] = {}
        for train, test in parts:
            models = fit(train)
            players, cols = fit_players(train)
            for m in test:
                g = np.isin(m["raw"], ["H", "A", "D"])
                ind = shift_indicator(models[0].predict_proba(m["X"][g])[:, 1],
                                      players.predict_proba(m["X"][g][:, cols])[:, 1])
                oof[scheme][m["match"]] = (log_emissions(models, m["X"], m["Xs"]), ind)
    return oof


VARIANTS = {
    "v09": lambda m, E: m["p09"],
    "learned": lambda m, E: decode(E, m["blocks"], len(m["y"])),
    "v09 dead mask": lambda m, E: decode(force_dead(E, m["p09"] == "D"), m["blocks"], len(m["y"])),
}


def bench():
    t0 = time.time()
    F, target, bl = barath_target()
    data = load_all_external(target)
    with contextlib.redirect_stdout(io.StringIO()):
        for m in data:
            m["p09"] = predict_match_v09(m["d_ball"], m["d_players"], m["team"], m["blocks"], m["pitch"])
    oof = out_of_fold(data)
    print(f"{len(data)} matches, {time.time() - t0:.0f}s. lomo = leave one match out, lopo = leave one provider out")
    print(f"spellF1 against the raw labels and the N=25 / N=50 conventions; models are trained on N={CONV_N}\n")
    print(f"{'':<28}{'raw':>7}{'N25':>7}{'N50':>7}{'macroF1':>9}{'aliveDead':>10}{'D% pred-true':>14}")
    for scheme in oof:
        for name, f in VARIANTS.items():
            rows = []
            for m in data:
                p = f(m, oof[scheme][m["match"]][0])
                g = np.isin(m["raw"], ["H", "A", "D"])
                s = score(m["y"], p)
                rows.append([score(m["raw"], p)[0], s[0], score(m["y50"], p)[0], s[1], s[2],
                             100 * ((p[g] == "D").mean() - (m["raw"][g] == "D").mean())])
            rows = np.array(rows)
            for label, sel in (("DFL", [is_dfl(m) for m in data]), ("Metrica", [not is_dfl(m) for m in data]),
                               ("all 10", [True] * len(data))):
                r = rows[np.array(sel)].mean(0)
                print(f"{scheme + ' ' + name + ' ' + label:<28}{r[0]:>7.3f}{r[1]:>7.3f}{r[2]:>7.3f}{r[3]:>9.3f}{r[4]:>10.3f}{r[5]:>+14.1f}")
        print()

    # Label-free: does BAR-ATH behave like a match of a provider seen in training?
    models = fit(data)
    players, cols = fit_players(data)
    X, _ = ml_features(F, bl, GAMES[GAME]["pitch"])
    g = live_blocks_mask(F["n_frames"], bl, pre_kickoff(bl))
    print("label-free transfer checks, min .. max over matches")
    print("  shift: % frames where a players-only alive model is sure dead and the full model says alive")
    print(f"  {'':<38}{'shift':>12}{'D% learned - v09':>20}{'alive/dead agreement %':>25}")
    rng = lambda v: f"{min(v):.1f} .. {max(v):.1f}"
    for scheme, label in (("lomo", "same provider in training"), ("lopo", "provider never seen")):
        for prov in (True, False):
            sel = [m for m in data if is_dfl(m) == prov]
            c = np.array([vs_v09(decode(oof[scheme][m["match"]][0], m["blocks"], len(m["y"])), m["p09"],
                                 np.isin(m["raw"], ["H", "A", "D"])) for m in sel])
            print(f"  {label + ', ' + ('DFL' if prov else 'Metrica'):<38}{rng([oof[scheme][m['match']][1] for m in sel]):>12}"
                  f"{rng(c[:, 0]):>20}{rng(c[:, 1]):>25}")
    lab = label_barath(models, F, bl)
    c = vs_v09(lab, predict_game_v09(GAME), lab != "X")
    ind = shift_indicator(models[0].predict_proba(X[g])[:, 1], players.predict_proba(X[g][:, cols])[:, 1])
    print(f"  {'BAR-ATH':<38}{ind:>12.1f}{c[0]:>20.1f}{c[1]:>25.1f}")


def vs_v09(pred, p09, g):
    """(D% of pred minus D% of model_v09, % of frames where both agree on alive/dead)."""
    return (100 * ((pred[g] == "D").mean() - (p09[g] == "D").mean()),
            100 * ((pred[g] == "D") == (p09[g] == "D")).mean())


# ---------------------------------------------------------------------------------------------
# Label-free comparison of BAR-ATH and the degraded external data
# ---------------------------------------------------------------------------------------------
def bin_stats(F, blocks):
    """Per play-intensity bin: P(ball missing), P(player within 2 m | ball visible), median ball
    speed, P(velocity change > KINK_DV | ball visible)."""
    g = live_blocks_mask(F["n_frames"], blocks, [])
    b, vis = intensity_bin(F), F["ball_vis"]
    near = np.fmin(F["min_dist_h"], F["min_dist_a"]) < 2
    sx, sy, kbs, svis = smooth_ball(F["ball_x"], F["ball_y"], vis)
    vx, vy = np.gradient(sx) * 25.0, np.gradient(sy) * 25.0
    dv = np.full(len(sx), np.nan)
    dv[KINK_LAG:-KINK_LAG] = np.hypot(vx[2 * KINK_LAG:] - vx[:-2 * KINK_LAG], vy[2 * KINK_LAG:] - vy[:-2 * KINK_LAG])
    out = []
    for k in range(len(SPEED_BINS) + 1):
        m, mv = g & (b == k), g & (b == k) & vis
        out.append([1 - vis[m].mean(), near[mv].mean(), np.nanmedian(kbs[mv]), np.mean(dv[mv] > KINK_DV)]
                   if mv.any() else [np.nan] * 4)
    return np.array(out)


def domain():
    F, target, bl = barath_target()
    rows = {"BAR-ATH": bin_stats(F, bl)}
    for carry in (0.0, CARRY_FRAC):
        ext = [load_external(m, i, target, carry) for i, m in enumerate(ALL_MATCHES)]
        st = [bin_stats(e["F"], e["blocks"]) for e in ext]
        rows[f"DFL, carry {carry}"] = np.nanmean([s for s, e in zip(st, ext) if is_dfl(e)], axis=0)
        rows[f"Metrica, carry {carry}"] = np.nanmean([s for s, e in zip(st, ext) if not is_dfl(e)], axis=0)
    print("play-intensity bin: stopped -> intense (mean player speed over 3 s)")
    for j, title in enumerate(["P(ball missing)", "P(player within 2 m | ball visible)", "median ball speed, m/s",
                               f"P(ball velocity change > {KINK_DV} m/s | ball visible)"]):
        print(title)
        for name, v in rows.items():
            print(f"  {name:<20}" + " ".join(f"{x:6.2f}" for x in v[:, j]))


# ---------------------------------------------------------------------------------------------
# BAR-ATH submission
# ---------------------------------------------------------------------------------------------
def pre_kickoff(blocks, game=GAME):
    """[first graded frame, tagged initial kick-off) of each half: dead ball by the laws of the game.
    Same rule as model_v07-v09; the kick-off comes from the automatic Smart Tagging."""
    tags = load_tags(game)
    out = []
    for first, last in blocks:
        ko = min((t["start_frame"] for t in tags if t["code"] == "SET PIECES"
                  and (t["tags"].get("Type") or "").strip().lower() == "kick off (start)"
                  and first <= t["start_frame"] <= last), default=first)
        out.append((first, ko))
    return out


def live_blocks_mask(n, blocks, exclude):
    g = np.zeros(n, dtype=bool)
    for first, last in blocks:
        g[first:last + 1] = True
    for first, stop in exclude:
        g[first:stop] = False
    return g


def label_barath(models, F, bl, game=GAME):
    X, Xs, _ = features_both(F, bl, GAMES[game]["pitch"])
    E = log_emissions(models, X, Xs)
    for first, stop in pre_kickoff(bl, game):
        E[first:stop, :2] = -1e9
    return decode(E, bl, F["n_frames"])


def build(out_csv, game=GAME):
    t0 = time.time()
    F, target, bl = barath_target(game)
    models = fit(load_all_external(target))
    print(f"Trained on {len(ALL_MATCHES)} external matches in {time.time() - t0:.0f}s")
    n = F["n_frames"]
    labels = label_barath(models, F, bl, game)
    pd.DataFrame({"frame": np.arange(n), "label": labels}).to_csv(out_csv, index=False)
    spells = [e - s for b in bl for v in "HA" for s, e in runs(labels[b[0]:b[1] + 1] == v)]
    graded = labels[labels != "X"]
    print(f"Saved {out_csv}: {len(spells)} possessions, mean {np.mean(spells) / 25:.1f}s, "
          f"D {100 * (graded == 'D').mean():.1f}%")
    if os.path.exists(GT_CSV):  # sanity check only; never used to choose anything
        gt = pd.read_csv(GT_CSV)["label"].values
        r = pm.evaluate(pm.label_codes(gt), pm.label_codes(labels))
        print(f"Public window (sanity only): spellF1 {r['spell_f1_mean']:.4f}  macroF1 {r['macro_f1']:.4f}  "
              f"poss {r['n_pred_spells']}/{r['n_gt_spells']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Possession model v1.1 (learned emissions + Viterbi)")
    ap.add_argument("--bench", action="store_true", help="external benchmark: lomo + lopo, three label conventions")
    ap.add_argument("--domain", action="store_true", help="label-free comparison of BAR-ATH and the degraded external data")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "model_v11.csv"))
    a = ap.parse_args()
    if a.bench:
        bench()
    elif a.domain:
        domain()
    else:
        build(a.out)
