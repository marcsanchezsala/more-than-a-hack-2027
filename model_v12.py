"""Learned emissions + Viterbi possession model v1.2.

model_v11 plus two label-free changes aimed at the one thing v11 got wrong on BAR-ATH: a visible ball
that moves during a stoppage was not read as dead ball, because in the degraded DFL training data a
dead ball barely moves (--domain: median speed and velocity-change rate in the four lowest
play-intensity bins well below BAR-ATH's).

  1. Dead-ball play (dead_ball_play, replaces v11's carry_ball): during each frozen-ball stretch of
     DFL the ball is carried, passed between nearby players and returned to the player who restarts.
     Its four constants are chosen by --calibrate against BAR-ATH's tracking statistics (no label).
  2. Alive/dead features: a classifier separating BAR-ATH frames from degraded external frames ranks
     the features that give the domain away (--domain); the ball-kinematics ones among the top are
     removed from the alive/dead model. The team model keeps every feature.

No BAR-ATH label (public window included) is read to fit or choose anything.

Outcome (docs/RESULTS_v1.2.md): neither change, alone or together, meets v1.2's acceptance criteria,
so model_v11 stays the submission and this file produces no results CSV of record.

  python model_v12.py --calibrate          # grid for the dead-ball play constants (label-free)
  python model_v12.py --domain             # bin statistics + domain classifier
  python model_v12.py --bench [--play v11|v12] [--alive all|domain]
  python model_v12.py                      # label BAR-ATH -> evaluation/results/model_v12.csv
"""

import io
import os
import time
import argparse
import itertools
import contextlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import gaussian_filter1d

import model_v09 as v9
import model_v10 as v10
import model_v11 as v11
import evaluation.possession_metrics as pm
from games import GAME, GAMES
from model_v11 import (runs, frame_features, features_both, ml_features, ball_gaps, barath_target, bin_stats,
                       is_dfl, decode, score, vs_v09, pre_kickoff, live_blocks_mask, fit_players,
                       shift_indicator, CARRY_MIN, CARRY_OFFSET, CARRY_GAP, CONV_N, SEED)
from bench_external.run import extract_features_from_arrays, predict_match_v09
from bench_external.fetch import ALL_MATCHES, DFL_MATCHES, CACHE_DIR
from bench_external.degrade import degrade_match, slow_noise, BALL_NOISE_SIGMA
from bench_external.convention import smooth_label_convention

RESULTS_DIR = v11.RESULTS_DIR
GT_CSV = v11.GT_CSV

# Dead-ball play (--calibrate; label-free, keyed on the frozen ball).
PLAY_FRAC = 0.7             # mean share of a frozen stretch in which the ball is in someone's hands/feet
PASS_RATE = 0.5             # passes per second of that time at 1 m/s mean player speed
PASS_SPEED = 9.0            # m/s
PASS_GAMMA = 2.0            # pass rate scales as (mean player speed over the stretch / 1 m/s) ** PASS_GAMMA
PLAY_MAX = 40.0             # seconds; the ball is in play at most this long before the restart
PASS_R = 20.0               # metres; passes go between players this close to the restart taker (not tuned)
P_RETURN = 0.5              # chance a pass goes back to the player who passed it last (not tuned)
PLAY = (PLAY_FRAC, PASS_RATE, PASS_SPEED, PASS_GAMMA, PLAY_MAX)
# Alive/dead features (--domain): ball-kinematics features among the DOMAIN_TOP features that best
# separate BAR-ATH from the degraded external data are left out of the alive/dead model.
DOMAIN_TOP = 10
BALL_KIN = ("kbs", "dv", "aerial", "bv_att", "bvatt", "kh", "ka", "since_kh", "since_ka", "until_kh", "until_ka")


def is_ball_kin(name):
    return name.split("_")[0] in BALL_KIN or name in BALL_KIN


# ---------------------------------------------------------------------------------------------
# External data
# ---------------------------------------------------------------------------------------------
def dead_ball_play(ball, players, frozen, rng, frac, rate, speed, gamma, cap):
    """For the last part of each frozen (dead) stretch (a share 2*frac*U of it, at most cap seconds)
    the ball is in play among nearby players: carried, passed (Poisson, rate per second times
    (mean player speed / 1 m/s) ** gamma) between players within PASS_R of the restart taker, often
    returned to the passer, and finally carried by the taker to the restart. v11's carry_ball is
    the case rate = 0, cap = inf."""
    out = ball.copy()
    n = len(ball)
    for s, e in runs(frozen):
        if e - s < CARRY_MIN or e >= n or frac <= 0:
            continue
        pe = players[e, :, :2]
        d = np.hypot(pe[:, 0] - ball[e, 0], pe[:, 1] - ball[e, 1])
        u = rng.random()
        if not np.isfinite(d).any():
            continue
        taker = int(np.nanargmin(d))
        t0 = e - min(int(min(1.0, 2 * frac * u) * (e - s)), int(cap * 25))
        if t0 >= e:
            continue
        near = np.flatnonzero(np.hypot(*(pe - pe[taker]).T) < PASS_R)          # taker included
        with np.errstate(all="ignore"):
            spd = np.nan_to_num(np.nanmean(players[t0:e, :, 2]), nan=1.0)
        lam = rate * max(spd, 0.0) ** gamma * (e - t0) / 25.0
        k = min(rng.poisson(lam), e - t0 - 1) if len(near) > 1 else 0
        chain = [taker]                                                          # built backwards
        for _ in range(k):
            ret = len(chain) > 1 and rng.random() < P_RETURN
            chain.append(chain[-2] if ret else int(rng.choice(near[near != chain[-1]])))
        chain = chain[::-1]
        cut = np.sort(rng.choice(np.arange(t0 + 1, e), size=k, replace=False)) if k else []
        bounds = [t0, *cut, e]
        pos = np.concatenate([players[a:b, h, :2] for h, a, b in zip(chain, bounds[:-1], bounds[1:])])
        for i in range(1, len(chain)):                   # pass: straight flight into the receiver
            a, b = bounds[i] - t0, bounds[i + 1] - t0
            src = pos[a - 1].copy()
            dist = np.nan_to_num(np.hypot(*(pos[min(b - 1, a + 12)] - src)))
            fl = int(np.clip(round(dist / speed * 25), 3, b - a))
            pos[a:a + fl] = src + (pos[a + fl - 1] - src) * (np.arange(1, fl + 1) / fl)[:, None]
        out[t0:e, :2] = pos + slow_noise(rng, e - t0, CARRY_OFFSET)
        out[max(s, t0 - CARRY_GAP):t0, :2] = np.nan
    return out


def _degrade_parts(match_id, idx, target):
    """Everything in v11.load_external that does not depend on the dead-ball play."""
    d = np.load(os.path.join(CACHE_DIR, f"{match_id}.npz"))
    blocks = [(int(b[0]), int(b[1])) for b in d["blocks"]]
    pitch = tuple(float(v) for v in d["pitch"]) if "pitch" in d else (105.0, 68.0)
    n = len(d["labels"])
    rng = np.random.default_rng(SEED + idx * 79)
    _, d_players = degrade_match(d["ball"], d["players"], d["team"],
                                 np.full(n, "H", dtype=object), blocks, seed=SEED + idx * 79)
    clean = extract_features_from_arrays(d["ball"], d["players"], d["team"], pitch)
    d_ball = d["ball"].copy()
    d_ball[:, :2] += slow_noise(rng, n, BALL_NOISE_SIGMA)
    step = np.hypot(*np.diff(d["ball"][:, :2], axis=0, prepend=d["ball"][:1, :2]).T)
    is_frozen = np.nan_to_num(step, nan=1.0) < 1e-4
    frozen = gaussian_filter1d(is_frozen.astype(float), 12)
    d_ball[:, :2] += slow_noise(rng, n, v10.FROZEN_DRIFT) * frozen[:, None]
    gaps = ball_gaps(clean, blocks, target, rng)
    return dict(match=match_id, idx=idx, d=d, blocks=blocks, pitch=pitch, d_players=d_players,
                d_ball=d_ball, is_frozen=is_frozen, gaps=gaps)


def _play(p, play=PLAY):
    """Degraded ball after the dead-ball play. Own RNG stream, so the gaps do not depend on it."""
    rng = np.random.default_rng(SEED + p["idx"] * 79 + 1)
    ball = dead_ball_play(p["d_ball"], p["d"]["players"], p["is_frozen"], rng, *play)
    ball[p["gaps"], :2] = np.nan
    return ball


def load_external(match_id, idx, target):
    p = _degrade_parts(match_id, idx, target)
    d_ball, raw = _play(p), p["d"]["labels"].astype(object)
    F = frame_features(d_ball, p["d_players"], p["d"]["team"], p["pitch"])
    X, Xs, names = features_both(F, p["blocks"], p["pitch"])
    return dict(match=match_id, X=X, Xs=Xs, names=names, raw=raw, blocks=p["blocks"], pitch=p["pitch"], F=F,
                y=smooth_label_convention(raw, p["blocks"], min_spell_frames=CONV_N),
                y50=smooth_label_convention(raw, p["blocks"], min_spell_frames=50),
                d_ball=d_ball, d_players=p["d_players"], team=p["d"]["team"])


_v11_load = v11.load_all_external      # use() replaces v11's name; keep the original


def load_all_external(target, play="v12"):
    if play == "v11":
        return _v11_load(target)
    return [load_external(m, i, target) for i, m in enumerate(ALL_MATCHES)]


# ---------------------------------------------------------------------------------------------
# Calibration of the dead-ball play (label-free)
# ---------------------------------------------------------------------------------------------
GRID = dict(frac=(0.5, 0.7, 0.9), rate=(0.35, 0.5, 0.7), speed=(9.0, 13.0), gamma=(1.0, 2.0),
            cap=(10.0, 20.0, 40.0, 1e9))


def _grid_match(args):
    m, i, target = args
    p = _degrade_parts(m, i, target)
    out = {}
    for c in itertools.product(*GRID.values()):
        F = extract_features_from_arrays(_play(p, c), p["d_players"], p["d"]["team"], p["pitch"])
        out[c] = bin_stats(F, p["blocks"])
    return out


def calibrate():
    """Dead-ball play constants: bin statistics of degraded DFL against BAR-ATH's. Loss = mean
    relative error of median ball speed and velocity-change rate over the 4 lowest bins; a setting
    is admissible if P(ball missing) and P(player within 2 m | visible) stay within 0.05 of BAR-ATH
    in every bin (v11's own worst deviation)."""
    F, target, bl = barath_target()
    ref = bin_stats(F, bl)
    with ProcessPoolExecutor(len(DFL_MATCHES)) as ex:
        per = list(ex.map(_grid_match, [(m, i, target) for i, m in enumerate(ALL_MATCHES) if m in DFL_MATCHES]))
    rows = []
    for c in itertools.product(*GRID.values()):
        st = np.nanmean([r[c] for r in per], axis=0)
        loss = np.mean(np.abs(st[:4, 2:] - ref[:4, 2:]) / ref[:4, 2:])
        dev = np.abs(st[:, :2] - ref[:, :2]).max(0)
        rows.append((loss, c, dev, st))
    rows.sort(key=lambda r: (r[2].max() > 0.05 + 1e-9, r[0]))
    print(f"BAR-ATH bins 0-3: speed {np.round(ref[:4, 2], 2)}  change rate {np.round(ref[:4, 3], 2)}")
    print(f"{'frac':>5}{'rate':>6}{'speed':>6}{'gamma':>6}{'cap':>6}{'loss':>7}{'max|dMiss|':>11}{'max|dNear|':>11}"
          f"  speed 0-3 / change rate 0-3")
    for loss, c, dev, st in rows[:12]:
        print(f"{c[0]:>5}{c[1]:>6}{c[2]:>6}{c[3]:>6}{min(c[4], 999):>6.0f}{loss:>7.3f}{dev[0]:>11.3f}{dev[1]:>11.3f}  "
              f"{np.round(st[:4, 2], 2)} / {np.round(st[:4, 3], 2)}  near-ref {np.round(st[:, 1] - ref[:, 1], 2)}")


# ---------------------------------------------------------------------------------------------
# Domain classifier: which features tell BAR-ATH from the degraded external data (label-free)
# ---------------------------------------------------------------------------------------------
_barath_cache = {}


def barath(game=GAME):
    if game not in _barath_cache:
        _barath_cache[game] = barath_target(game)
    return _barath_cache[game]


def _domain_frames(data, game=GAME):
    """(X, is_barath, half) on graded frames: all of BAR-ATH's (bar the pre-kickoff), the same
    number from the external matches spread evenly. half = 0/1 = first/second half, for the CV."""
    F, _, bl = barath(game)
    Xb, names = ml_features(F, bl, GAMES[game]["pitch"])
    gb = np.flatnonzero(live_blocks_mask(F["n_frames"], bl, pre_kickoff(bl, game)))[::v10.SUBSAMPLE]
    hb = (gb > bl[0][1]).astype(int)
    ext, he = [], []
    step = max(1, sum(int(np.isin(m["raw"], ["H", "A", "D"]).sum()) for m in data) // len(gb))
    for m in data:
        g = np.flatnonzero(np.isin(m["raw"], ["H", "A", "D"]))[::step]
        ext.append(m["X"][g]); he.append((g > m["blocks"][0][1]).astype(int))
    X = np.vstack([Xb[gb]] + ext)
    return X, np.r_[np.ones(len(gb), bool), np.zeros(sum(map(len, he)), bool)], np.r_[hb, *he], names


def domain_rank(data, cols=None):
    """Features ranked by gain in a BAR-ATH-vs-external classifier, and its AUC (trained on one half
    of every match, tested on the other, both ways). cols: subset to use (default all)."""
    from sklearn.metrics import roc_auc_score
    X, y, half, names = _domain_frames(data)
    cols = list(range(X.shape[1])) if cols is None else cols
    X = X[:, cols]
    prm = dict(v10.LGB_PARAMS, n_estimators=150)
    auc = np.mean([roc_auc_score(y[half != h], lgb.LGBMClassifier(**prm).fit(X[half == h], y[half == h])
                                 .predict_proba(X[half != h])[:, 1]) for h in (0, 1)])
    clf = lgb.LGBMClassifier(**prm, importance_type="gain").fit(X, y)
    order = np.argsort(-clf.feature_importances_, kind="stable")
    imp = clf.feature_importances_ / clf.feature_importances_.sum()
    return auc, [(names[cols[i]], imp[i]) for i in order]


def alive_cols(data):
    """Alive/dead model features: all but the ball-kinematics ones among the DOMAIN_TOP features
    that best separate BAR-ATH from the degraded external data."""
    names = data[0]["names"]
    _, rank = domain_rank(data)
    drop = {k for k, _ in rank[:DOMAIN_TOP] if is_ball_kin(k)}
    return [i for i, k in enumerate(names) if k not in drop]


# ---------------------------------------------------------------------------------------------
# Models: v11's, with the alive/dead model restricted to a column subset
# ---------------------------------------------------------------------------------------------
class OnCols:
    """A fitted classifier that reads only some columns of the full feature matrix."""

    def __init__(self, model, cols):
        self.model, self.cols = model, cols

    def predict_proba(self, X):
        return self.model.predict_proba(X[:, self.cols])


_v11_fit = v11.fit


def fit(train, cols):
    """v11.fit; with cols, the alive/dead model is refitted on those columns (team model as v11's)."""
    alive, team = _v11_fit(train)
    if cols is None:
        return alive, team
    g = [v11._train_frames(m) for m in train]
    Xa = np.vstack([m["X"][i][:, cols] for m, i in zip(train, g)])
    ya = np.concatenate([m["y"][i] != "D" for m, i in zip(train, g)])
    return OnCols(lgb.LGBMClassifier(**v10.LGB_PARAMS).fit(Xa, ya), cols), team


def use(play="v12", alive="domain"):
    """Point model_v11's bench/build pipeline at v12's training data and alive/dead features, so
    v12 is measured by exactly the code that measured v11. model_v11.py itself is not changed."""
    state = {}

    def load(target):
        data = load_all_external(target, play)
        state["cols"] = alive_cols(data) if alive == "domain" else None
        if state["cols"] is not None:
            names = data[0]["names"]
            print(f"alive/dead model: {len(state['cols'])} of {len(names)} features; dropped "
                  f"{sorted(set(names) - {names[i] for i in state['cols']})}")
        return data

    v11.barath_target = barath
    v11.load_all_external = load
    v11.fit = lambda train: fit(train, state["cols"])


# ---------------------------------------------------------------------------------------------
# --domain
# ---------------------------------------------------------------------------------------------
def domain():
    F, target, bl = barath()
    rows = {"BAR-ATH": bin_stats(F, bl)}
    sets = {}
    for play in ("v11", "v12"):
        sets[play] = load_all_external(target, play)
        st = [bin_stats(e["F"], e["blocks"]) for e in sets[play]]
        for prov in (True, False):
            rows[f"{'DFL' if prov else 'Metrica'}, {play} play"] = np.nanmean(
                [s for s, e in zip(st, sets[play]) if is_dfl(e) == prov], axis=0)
    print("play-intensity bin: stopped -> intense (mean player speed over 3 s)")
    for j, title in enumerate(["P(ball missing)", "P(player within 2 m | ball visible)", "median ball speed, m/s",
                               f"P(ball velocity change > {v11.KINK_DV} m/s | ball visible)"]):
        print(title)
        for name, v in rows.items():
            print(f"  {name:<20}" + " ".join(f"{x:6.2f}" for x in v[:, j]))
    print("\ndomain classifier BAR-ATH vs degraded external (AUC: trained on one half, tested on the other)")
    for play, data in sets.items():
        auc, rank = domain_rank(data)
        cols = alive_cols(data)
        auc2, _ = domain_rank(data, cols)
        print(f"  {play} play: AUC all features {auc:.3f}, alive/dead features {auc2:.3f} "
              f"({len(data[0]['names']) - len(cols)} dropped)")
        for k, w in rank[:15]:
            print(f"    {k:<14}{w:6.3f}{'  ball kinematics' if is_ball_kin(k) else ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Possession model v1.2")
    ap.add_argument("--calibrate", action="store_true", help="grid for the dead-ball play constants (label-free)")
    ap.add_argument("--domain", action="store_true", help="bin statistics and the domain classifier")
    ap.add_argument("--bench", action="store_true", help="model_v11's benchmark on v12's data and features")
    ap.add_argument("--play", choices=["v11", "v12"], default="v12", help="dead-ball simulation in the training data")
    ap.add_argument("--alive", choices=["all", "domain"], default="domain", help="alive/dead model features")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "model_v12.csv"))
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
    elif a.domain:
        domain()
    else:
        use(a.play, a.alive)
        v11.bench() if a.bench else v11.build(a.out)
