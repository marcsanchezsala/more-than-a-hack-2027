"""Learned emissions + Viterbi possession model v1.0.

Replaces model_v09's hand-written emission scores with two LightGBM classifiers trained ONLY on the
10 open external matches (7 DFL/IDSSE + 3 Metrica samples), degraded to look like BAR-ATH tracking:
  - alive model: P(ball in play | features), trained on every graded frame;
  - team model:  P(home | features, ball in play), trained on live frames plus their home/away
    mirror so the classifier is symmetric by construction.
On BAR-ATH the alive/dead decision is taken from model_v09 (the learned alive model does not
transfer to BAR-ATH stoppages); the learned team model decides H/A inside live play.
Features are the same per-frame physical signals model_v09 extracts (nearest home/away distance,
Kalman ball speed, out-of-bounds, player speeds and counts, touches) plus past/future windows.
The three Viterbi transition costs are chosen by leave-one-match-out on the external matches;
no BAR-ATH label (public window included) is read to fit or choose anything.

  python model_v10.py --bench   # leave-one-match-out on the 10 external matches, v10 vs v09
  python model_v10.py           # train on all 10, label BAR-ATH -> evaluation/results/model_v10.csv
"""

import io
import os
import sys
import tempfile
import contextlib
import time
import argparse
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

import model_v09 as v9
import evaluation.possession_metrics as pm
from games import GAME, GAMES, half_video_blocks
from bench_external.run import extract_features_from_arrays, predict_match_v09
from bench_external.fetch import ALL_MATCHES, DFL_MATCHES, CACHE_DIR
from bench_external.degrade import degrade_match, slow_noise, BALL_NOISE_SIGMA
from bench_external.convention import smooth_label_convention

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
GT_CSV = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

SEED = 42
CONV_N = 50          # external label convention, same as bench_external/calibrate_physics.py
SUBSAMPLE = 3        # train on every 3rd frame: neighbouring frames are near-duplicates
WINDOWS = (25, 75, 250)
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                  deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)
# Viterbi transition costs (nats). Chosen by leave-one-match-out grid on the external matches.
DEFAULT_COSTS = dict(p_switch=20.0, p_enter_d=40.0, p_exit_d=40.0)

# Ball visibility depends on play intensity: in BAR-ATH the ball is lost mostly while play is stopped.
SPEED_BINS = np.array([0.8, 1.1, 1.4, 1.7, 2.0, 2.4, 2.8])   # m/s, mean player speed over 3 s
GAP_SMOOTH = 50                                               # frames, correlation of the dropout field
# DFL freezes the ball (identical position) during stoppages; BAR-ATH never does. A slow 1 m drift on
# frozen frames brings low-intensity ball speed to BAR-ATH's level (label-free, see RESULTS_v1.0).
FROZEN_DRIFT = 1.0                                            # metres

SWAP_KEYS = [("min_dist_h", "min_dist_a"), ("mean_speed_h", "mean_speed_a"), ("centroid_hx", "centroid_ax")]


# ---------------------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------------------
def swap_teams(F):
    G = dict(F)
    for h, a in SWAP_KEYS:
        G[h], G[a] = F[a], F[h]
    return G


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


def ml_features(F, blocks, pitch):
    """Per-frame feature matrix from a model_v09 feature dict. Same code for BAR-ATH and external."""
    L, W = pitch
    n = F["n_frames"]
    vis, oob = F["ball_vis"], F["oob"]
    dh, da = F["min_dist_h"], F["min_dist_a"]
    sx, sy, kbs, _ = v9.kalman_smooth_ball(F["ball_x"], F["ball_y"], vis)
    aerial = v9.detect_aerial_ballistics(sx, sy, kbs, vis)

    touch_h = vis & ~oob & (dh < 1.5) & ~(da < 2.0)
    touch_a = vis & ~oob & (da < 1.5) & ~(dh < 2.0)
    diff = np.clip(np.nan_to_num(da - dh, nan=0.0), -10, 10)

    # Ball position along home's attacking direction, per half (centroid sign, as in model_v09)
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
        "n_players": F["n_players"], "spd": F["player_speed"], "p90": F["p90_speed"],
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
            cols[f"np_{tag}"] = _roll(F["n_players"], w, fut)
    names = list(cols)
    X = np.column_stack([np.asarray(cols[k], dtype=np.float32) for k in names])
    return X, names


def features_both(F, blocks, pitch):
    X, names = ml_features(F, blocks, pitch)
    Xs, _ = ml_features(swap_teams(F), blocks, pitch)
    return X, Xs, names


# ---------------------------------------------------------------------------------------------
# External data
# ---------------------------------------------------------------------------------------------
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


def load_external(match_id, idx, target_curve):
    """Degraded features + convention labels for one external match (deterministic).

    Players: degrade.py dropouts and jitter. Ball: noise, drift where the source froze it, then removed with the probability that
    makes this match's ball-missing curve (by play intensity) match target_curve, measured on
    BAR-ATH tracking. Labels are never used to degrade (degrade.py's D-only drift would leak them).
    """
    d = np.load(os.path.join(CACHE_DIR, f"{match_id}.npz"))
    blocks = [(int(b[0]), int(b[1])) for b in d["blocks"]]
    pitch = tuple(float(v) for v in d["pitch"]) if "pitch" in d else (105.0, 68.0)
    labels = d["labels"].astype(object)
    n = len(labels)
    rng = np.random.default_rng(SEED + idx * 79)
    _, d_players = degrade_match(d["ball"], d["players"], d["team"],
                                 np.full(n, "H", dtype=object), blocks, seed=SEED + idx * 79)
    clean = extract_features_from_arrays(d["ball"], d["players"], d["team"], pitch)
    own = ball_miss_curve(clean, blocks)
    q = np.clip(1 - (1 - target_curve) / np.maximum(1 - own, 1e-6), 0, 1)
    z = gaussian_filter1d(rng.normal(size=n), GAP_SMOOTH)
    d_ball = d["ball"].copy()
    d_ball[:, :2] += slow_noise(rng, n, BALL_NOISE_SIGMA)
    step = np.hypot(*np.diff(d["ball"][:, :2], axis=0, prepend=d["ball"][:1, :2]).T)
    frozen = gaussian_filter1d((np.nan_to_num(step, nan=1.0) < 1e-4).astype(float), 12)
    d_ball[:, :2] += slow_noise(rng, n, FROZEN_DRIFT) * frozen[:, None]
    d_ball[norm.cdf(z / z.std()) < q[intensity_bin(clean)], :2] = np.nan
    F = extract_features_from_arrays(d_ball, d_players, d["team"], pitch)
    X, Xs, names = features_both(F, blocks, pitch)
    y = smooth_label_convention(labels, blocks, min_spell_frames=CONV_N)
    return dict(match=match_id, X=X, Xs=Xs, y=y, blocks=blocks, pitch=pitch,
                d_ball=d_ball, d_players=d_players, team=d["team"])


def load_all_external(target_curve):
    return [load_external(m, i, target_curve) for i, m in enumerate(ALL_MATCHES)]


def barath_target(game=GAME):
    F = v9.extract_features(game)
    return F, ball_miss_curve(F, [(f, l) for _, f, l in half_video_blocks(game)])


# ---------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------
def fit(train):
    Xa, ya, Xt, yt = [], [], [], []
    for m in train:
        g = np.flatnonzero(np.isin(m["y"], ["H", "A", "D"]))[::SUBSAMPLE]
        Xa.append(m["X"][g]); ya.append(m["y"][g] != "D")
        live = g[m["y"][g] != "D"]
        Xt += [m["X"][live], m["Xs"][live]]
        yt += [m["y"][live] == "H", m["y"][live] == "A"]
    alive = lgb.LGBMClassifier(**LGB_PARAMS).fit(np.vstack(Xa), np.concatenate(ya))
    team = lgb.LGBMClassifier(**LGB_PARAMS).fit(np.vstack(Xt), np.concatenate(yt))
    return alive, team


def log_emissions(models, X, Xs):
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


def force_dead(E, dead):
    """Keep the learned H/A emissions but take alive/dead from another model (dead: bool mask)."""
    E = E.copy()
    E[dead, :2] = -1e9
    E[~dead, 2] = -1e9
    return E


def decode(E, blocks, n, costs):
    out = np.full(n, "X", dtype=object)
    for first, last in blocks:
        out[first:last + 1] = v9.STATE_LABELS[viterbi(E[first:last + 1], **costs)]
    return out


# ---------------------------------------------------------------------------------------------
# External benchmark (leave-one-match-out)
# ---------------------------------------------------------------------------------------------
def score(y, pred):
    r = pm.evaluate(pm.label_codes(y), pm.label_codes(pred))
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def bench(grid=True):
    t0 = time.time()
    _, target = barath_target()
    print("BAR-ATH P(ball missing | intensity bin):", np.round(target, 2))
    data = load_all_external(target)
    print(f"Loaded + featurised {len(data)} matches in {time.time() - t0:.0f}s")

    oof = {}
    for m in data:
        models = fit([o for o in data if o is not m])
        oof[m["match"]] = log_emissions(models, m["X"], m["Xs"])
        print(f"  LOMO fit -> {m['match']} ({time.time() - t0:.0f}s)")

    def mean_spell(costs):
        return np.mean([score(m["y"], decode(oof[m["match"]], m["blocks"], len(m["y"]), costs))[0] for m in data])

    costs = dict(DEFAULT_COSTS)
    if grid:
        print("\nTransition-cost grid (mean LOMO spellF1 over 10 matches):")
        best = -1
        for ps, pe, px in itertools.product([10, 20, 40], [20, 40, 80], [20, 40, 80]):
            c = dict(p_switch=ps, p_enter_d=pe, p_exit_d=px)
            s = mean_spell(c)
            print(f"  {c} -> {s:.4f}")
            if s > best:
                best, costs = s, c
        print(f"Best: {costs} -> {best:.4f}")

    print(f"\nspellF1 / macroF1 per match. hybrid = v09 alive/dead + v10 team model")
    print(f"{'match':<18} {'v09':>15} {'v10':>15} {'hybrid':>15}")
    rows = []
    for m in data:
        n = len(m["y"])
        p10 = decode(oof[m["match"]], m["blocks"], n, costs)
        p09 = predict_match_v09(m["d_ball"], m["d_players"], m["team"], m["blocks"], m["pitch"])
        phy = decode(force_dead(oof[m["match"]], p09 == "D"), m["blocks"], n, costs)
        sc = [score(m["y"], p)[:2] for p in (p09, p10, phy)]
        rows.append(np.ravel(sc))
        print(f"{m['match']:<18} " + " ".join(f"{a:>7.4f}/{b:.4f}" for a, b in sc))
    r = np.mean(rows, axis=0)
    print(f"{'MEAN':<18} " + " ".join(f"{r[i]:>7.4f}/{r[i + 1]:.4f}" for i in (0, 2, 4)))
    return costs


# ---------------------------------------------------------------------------------------------
# BAR-ATH submission
# ---------------------------------------------------------------------------------------------
def build(out_csv, costs=DEFAULT_COSTS, dead="v09", game=GAME):
    t0 = time.time()
    F, target = barath_target(game)
    models = fit(load_all_external(target))
    print(f"Trained on {len(ALL_MATCHES)} external matches in {time.time() - t0:.0f}s")

    n = F["n_frames"]
    blocks = half_video_blocks(game)                     # (half, first, last)
    bl = [(f, l) for _, f, l in blocks]
    X, Xs, _ = features_both(F, bl, GAMES[game]["pitch"])
    E = log_emissions(models, X, Xs)
    learned = decode(E, bl, n, costs)
    if dead == "learned":
        labels = learned
    else:
        labels = None

    # Alive/dead comes from model_v09 (its own defaults, Smart Tagging included): the learned
    # alive model does not transfer to BAR-ATH stoppages (see docs/RESULTS_v1.0.md). Only the H/A
    # decision inside live play is learned.
    # ponytail: re-runs model_v09 end to end (~6 s); a learned, domain-adapted alive model replaces this.
    with tempfile.TemporaryDirectory() as tmp:  # needed in both modes: agreement is reported
        v09_csv = os.path.join(tmp, "v09.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            v9.build_and_evaluate(game=game, out_csv=v09_csv)
        v09_labels = pd.read_csv(v09_csv)["label"].to_numpy()
    if labels is None:
        labels = decode(force_dead(E, v09_labels == "D"), bl, n, costs)
    g = v09_labels != "X"
    print(f"Learned alive/dead vs v09: D {100 * (learned[g] == 'D').mean():.1f}% vs "
          f"{100 * (v09_labels[g] == 'D').mean():.1f}%, agreement {100 * ((learned[g] == 'D') == (v09_labels[g] == 'D')).mean():.1f}%")
    pd.DataFrame({"frame": np.arange(n), "label": labels}).to_csv(out_csv, index=False)
    spells = [r for b in bl for r in v9.extract_possession_spells(labels[b[0]:b[1] + 1]) if r[0] in ("H", "A")]
    graded = labels[labels != "X"]
    print(f"Saved {out_csv}: {len(spells)} possessions, mean {np.mean([r[3] for r in spells]) / 25:.1f}s, "
          f"D {100 * (graded == 'D').mean():.1f}%")

    if os.path.exists(GT_CSV):  # sanity check only; never used to choose anything
        gt = pd.read_csv(GT_CSV)["label"].values
        s, mac, npred, ngt = score(gt, labels)
        print(f"Public window (sanity only): spellF1 {s:.4f}  macroF1 {mac:.4f}  poss {npred}/{ngt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Possession model v1.0 (learned emissions + Viterbi)")
    ap.add_argument("--bench", action="store_true", help="leave-one-match-out on the external matches")
    ap.add_argument("--no-grid", action="store_true", help="with --bench: skip the transition-cost grid")
    ap.add_argument("--dead", choices=["v09", "learned"], default="v09",
                    help="alive/dead from model_v09 (hybrid) or from the learned alive model")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "model_v10.csv"))
    a = ap.parse_args()
    if a.bench:
        bench(grid=not a.no_grid)
    else:
        build(a.out, dead=a.dead)
