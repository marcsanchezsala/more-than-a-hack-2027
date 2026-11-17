"""External benchmark runner for football possession inference.

Runs a model pipeline (model_v09 or model_v10) on the 7 DFL / IDSSE matches and
3 Metrica sample matches with synthetic degradation, scored against official ground truth
using evaluation/possession_metrics.py.
For model_v10, matches are evaluated using Leave-One-Match-Out (LOMO) cross-validation.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evaluation.possession_metrics as pm
from bench_external.fetch import DFL_MATCHES, METRICA_MATCHES, ALL_MATCHES, CACHE_DIR
from bench_external.degrade import degrade_match


def extract_features_from_arrays(ball, players, team, pitch=(105.0, 68.0)):
    """Extract standard physical kinematics, player counts, and p90 speed from per-frame arrays."""
    pitch_l, pitch_w = pitch
    n_frames = len(ball)

    b_vis = np.isfinite(ball[:, 0]) & np.isfinite(ball[:, 1])
    bx = np.where(b_vis, ball[:, 0], np.nan).astype(np.float32)
    by = np.where(b_vis, ball[:, 1], np.nan).astype(np.float32)
    bs = ball[:, 2].astype(np.float32)

    px = players[..., 0]
    py = players[..., 1]
    ps = players[..., 2]
    p_vis = np.isfinite(px) & np.isfinite(py)
    p_has_s = np.isfinite(ps)

    is_h = (team == 0)
    is_a = (team == 1)

    n_players = p_vis.sum(axis=1).astype(np.int16)

    count_h = (p_vis & is_h).sum(axis=1)
    sum_hx = np.where(p_vis & is_h, px, 0.0).sum(axis=1)
    centroid_hx = np.where(count_h > 0, sum_hx / np.maximum(count_h, 1), np.nan).astype(np.float32)

    count_a = (p_vis & is_a).sum(axis=1)
    sum_ax = np.where(p_vis & is_a, px, 0.0).sum(axis=1)
    centroid_ax = np.where(count_a > 0, sum_ax / np.maximum(count_a, 1), np.nan).astype(np.float32)

    count_hs = (p_vis & is_h & p_has_s).sum(axis=1)
    sum_hs = np.where(p_vis & is_h & p_has_s, ps, 0.0).sum(axis=1)
    mean_speed_h = np.where(count_hs > 0, sum_hs / np.maximum(count_hs, 1), np.nan).astype(np.float32)

    count_as = (p_vis & is_a & p_has_s).sum(axis=1)
    sum_as = np.where(p_vis & is_a & p_has_s, ps, 0.0).sum(axis=1)
    mean_speed_a = np.where(count_as > 0, sum_as / np.maximum(count_as, 1), np.nan).astype(np.float32)

    player_speed = np.full(n_frames, np.nan, dtype=np.float32)
    both_s = ~np.isnan(mean_speed_h) & ~np.isnan(mean_speed_a)
    only_h = ~np.isnan(mean_speed_h) & np.isnan(mean_speed_a)
    only_a = np.isnan(mean_speed_h) & ~np.isnan(mean_speed_a)
    player_speed[both_s] = 0.5 * (mean_speed_h[both_s] + mean_speed_a[both_s])
    player_speed[only_h] = mean_speed_h[only_h]
    player_speed[only_a] = mean_speed_a[only_a]

    p90_speed = np.full(n_frames, np.nan, dtype=np.float32)
    for i in range(0, n_frames, 5):
        row_s = ps[i, p_vis[i] & p_has_s[i]]
        if len(row_s) > 0:
            p90_speed[i] = np.percentile(row_s, 85)
    p90_speed = pd.Series(p90_speed).interpolate(limit=10).bfill().ffill().values.astype(np.float32)

    dx = px - bx[:, None]
    dy = py - by[:, None]
    d2 = dx * dx + dy * dy
    d2_h = np.where(p_vis & is_h, d2, np.inf)
    d2_a = np.where(p_vis & is_a, d2, np.inf)
    min_sq_h = np.min(d2_h, axis=1)
    min_sq_a = np.min(d2_a, axis=1)
    min_dist_h = np.where(b_vis & (min_sq_h < np.inf), np.sqrt(min_sq_h), np.nan).astype(np.float32)
    min_dist_a = np.where(b_vis & (min_sq_a < np.inf), np.sqrt(min_sq_a), np.nan).astype(np.float32)

    oob = (bx < 0) | (bx > pitch_l) | (by < 0) | (by > pitch_w)

    return {
        "n_frames": n_frames,
        "ball_vis": b_vis,
        "ball_x": bx,
        "ball_y": by,
        "ball_s": bs,
        "min_dist_h": min_dist_h,
        "min_dist_a": min_dist_a,
        "mean_speed_h": mean_speed_h,
        "mean_speed_a": mean_speed_a,
        "player_speed": player_speed,
        "p90_speed": p90_speed,
        "n_players": n_players,
        "oob": oob,
        "centroid_hx": centroid_hx,
        "centroid_ax": centroid_ax,
    }


def predict_match_v09(
    ball,
    players,
    team,
    blocks,
    pitch=(105.0, 68.0),
    p_switch=70.0,
    p_enter_d=55.0,
    p_exit_d=15.0,
    max_d_gain=1.4,
    tackle_bonus=8.0,
):
    """Run model_v09 physical pipeline on a match."""
    import model_v09 as v9
    features = extract_features_from_arrays(ball, players, team, pitch)
    n_frames = features["n_frames"]

    s_interp = pd.Series(features["player_speed"]).interpolate(limit=250).bfill().ffill()
    roll_speed_fwd = s_interp.iloc[::-1].rolling(40, min_periods=1).mean().iloc[::-1].values

    p90_interp = pd.Series(features["p90_speed"]).interpolate(limit=250).bfill().ffill()
    roll_p90_fwd = p90_interp.iloc[::-1].rolling(40, min_periods=1).mean().iloc[::-1].values

    smooth_x, smooth_y, kalman_bs, _ = v9.kalman_smooth_ball(
        features["ball_x"], features["ball_y"], features["ball_vis"]
    )
    is_aerial = v9.detect_aerial_ballistics(smooth_x, smooth_y, kalman_bs, features["ball_vis"])

    defl_h, defl_a = v9.detect_touches(
        features["ball_vis"], features["oob"], is_aerial,
        features["min_dist_h"], features["min_dist_a"], kalman_bs, n_frames
    )
    pass_bridges = v9.build_pass_bridges_v5(
        features["ball_vis"], features["min_dist_h"], features["min_dist_a"], features["oob"]
    )

    prior_h = np.zeros(n_frames, dtype=np.float32)
    prior_a = np.zeros(n_frames, dtype=np.float32)
    prior_d = np.zeros(n_frames, dtype=np.float32)
    priors = (prior_h, prior_a, prior_d)

    can_gain_h = (
        (features["ball_vis"] & (~features["oob"]) & (~is_aerial) & (~defl_h) & (~np.isnan(features["min_dist_h"])) & (features["min_dist_h"] < max_d_gain))
    )
    can_gain_a = (
        (features["ball_vis"] & (~features["oob"]) & (~is_aerial) & (~defl_a) & (~np.isnan(features["min_dist_a"])) & (features["min_dist_a"] < max_d_gain))
    )

    dh_all = features["min_dist_h"]
    da_all = features["min_dist_a"]
    bv_all = features["ball_vis"]
    oob_all = features["oob"]
    tackle_h_all = bv_all & (~oob_all) & (~is_aerial) & (~defl_h) & (~np.isnan(dh_all)) & (~np.isnan(da_all)) & (dh_all < 1.2) & (dh_all < da_all - 0.5)
    tackle_a_all = bv_all & (~oob_all) & (~is_aerial) & (~defl_a) & (~np.isnan(dh_all)) & (~np.isnan(da_all)) & (da_all < 1.2) & (da_all < dh_all - 0.5)

    can_restart = (
        (~np.isnan(kalman_bs) & (kalman_bs > 2.8))
        | (roll_p90_fwd > 2.85)
        | (roll_speed_fwd > 2.20)
    )

    base_trans = np.zeros((3, 3), dtype=np.float64)
    base_trans[0, 1] = -p_switch
    base_trans[1, 0] = -p_switch
    base_trans[0, 2] = -p_enter_d
    base_trans[1, 2] = -p_enter_d
    base_trans[2, 0] = -p_exit_d
    base_trans[2, 1] = -p_exit_d

    full_labels = np.full(n_frames, "X", dtype=object)
    for half_idx, (first, last) in enumerate(blocks):
        T = last - first + 1
        c_res_half = can_restart[first : last + 1]
        c_gh_half = can_gain_h[first : last + 1]
        c_ga_half = can_gain_a[first : last + 1]
        tack_h_half = tackle_h_all[first : last + 1]
        tack_a_half = tackle_a_all[first : last + 1]
        oob_half = features["oob"][first : last + 1]
        rps_half = roll_speed_fwd[first : last + 1]
        np_half = features["n_players"][first : last + 1]

        chx = features["centroid_hx"][first : last + 1]
        cax = features["centroid_ax"][first : last + 1]
        att_dir_h = 1.0 if np.nanmean(chx) < np.nanmean(cax) else -1.0
        c_mid = 0.5 * (
            pd.Series(chx).interpolate().bfill().ffill()
            + pd.Series(cax).interpolate().bfill().ffill()
        )
        c_mid_smooth = c_mid.rolling(50, center=True, min_periods=1).mean().values
        adv_vel_h = att_dir_h * np.gradient(c_mid_smooth) * 25.0

        emissions = v9.compute_half_emissions(first, last, features, priors, pass_bridges, kalman_bs, roll_speed_fwd, adv_vel_h)
        path = v9.viterbi_decode_v7(emissions, base_trans, c_res_half, c_gh_half, c_ga_half, tack_h_half, tack_a_half, tackle_bonus=tackle_bonus)
        raw_half_labels = v9.STATE_LABELS[path]
        cleaned = v9.clean_spurious_segments_v7(raw_half_labels, oob_half, rps_half, np_half)
        full_labels[first : last + 1] = cleaned

    block_tuples = [(idx, int(b[0]), int(b[1])) for idx, b in enumerate(blocks)]
    snapped = v9.snap_transitions(
        full_labels, features["min_dist_h"], features["min_dist_a"], kalman_bs, features["oob"], block_tuples
    )
    return snapped


def predict_game_v09(game):
    """model_v09's labels for a provided game: a baseline to compare against, never used to label."""
    import io, tempfile, contextlib
    import model_v09 as v9
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        out = os.path.join(tmp, "v09.csv")
        v9.build_and_evaluate(game=game, out_csv=out)
        return pd.read_csv(out)["label"].to_numpy()


def predict_match_v10(ball, players, team, blocks, pitch=(105.0, 68.0), team_log_odds=None, team_weight=0.5):
    """Run model_v10 pipeline with injected learned team layer."""
    import model_v09 as v9, model_v10 as v10
    features = extract_features_from_arrays(ball, players, team, pitch)
    n_frames = features["n_frames"]

    s_interp = pd.Series(features["player_speed"]).interpolate(limit=250).bfill().ffill()
    roll_speed_fwd = s_interp.iloc[::-1].rolling(40, min_periods=1).mean().iloc[::-1].values

    p90_interp = pd.Series(features["p90_speed"]).interpolate(limit=250).bfill().ffill()
    roll_p90_fwd = p90_interp.iloc[::-1].rolling(40, min_periods=1).mean().iloc[::-1].values

    smooth_x, smooth_y, kalman_bs, _ = v9.kalman_smooth_ball(
        features["ball_x"], features["ball_y"], features["ball_vis"]
    )
    is_aerial = v9.detect_aerial_ballistics(smooth_x, smooth_y, kalman_bs, features["ball_vis"])

    defl_h, defl_a = v9.detect_touches(
        features["ball_vis"], features["oob"], is_aerial,
        features["min_dist_h"], features["min_dist_a"], kalman_bs, n_frames
    )
    pass_bridges = v9.build_pass_bridges_v5(
        features["ball_vis"], features["min_dist_h"], features["min_dist_a"], features["oob"]
    )

    prior_h = np.zeros(n_frames, dtype=np.float32)
    prior_a = np.zeros(n_frames, dtype=np.float32)
    prior_d = np.zeros(n_frames, dtype=np.float32)
    priors = (prior_h, prior_a, prior_d)

    can_gain_h = (
        (features["ball_vis"] & (~features["oob"]) & (~is_aerial) & (~defl_h) & (~np.isnan(features["min_dist_h"])) & (features["min_dist_h"] < 1.6))
    )
    can_gain_a = (
        (features["ball_vis"] & (~features["oob"]) & (~is_aerial) & (~defl_a) & (~np.isnan(features["min_dist_a"])) & (features["min_dist_a"] < 1.6))
    )

    dh_all = features["min_dist_h"]
    da_all = features["min_dist_a"]
    bv_all = features["ball_vis"]
    oob_all = features["oob"]
    tackle_h_all = bv_all & (~oob_all) & (~is_aerial) & (~defl_h) & (~np.isnan(dh_all)) & (~np.isnan(da_all)) & (dh_all < 1.2) & (dh_all < da_all - 0.5)
    tackle_a_all = bv_all & (~oob_all) & (~is_aerial) & (~defl_a) & (~np.isnan(dh_all)) & (~np.isnan(da_all)) & (da_all < 1.2) & (da_all < dh_all - 0.5)

    can_restart = (
        (~np.isnan(kalman_bs) & (kalman_bs > 2.8))
        | (roll_p90_fwd > 2.85)
        | (roll_speed_fwd > 2.20)
    )

    base_trans = np.zeros((3, 3), dtype=np.float64)
    base_trans[0, 1] = -70.0
    base_trans[1, 0] = -70.0
    base_trans[0, 2] = -55.0
    base_trans[1, 2] = -55.0
    base_trans[2, 0] = -15.0
    base_trans[2, 1] = -15.0

    full_labels = np.full(n_frames, "X", dtype=object)
    for half_idx, (first, last) in enumerate(blocks):
        T = last - first + 1
        c_res_half = can_restart[first : last + 1]
        c_gh_half = can_gain_h[first : last + 1]
        c_ga_half = can_gain_a[first : last + 1]
        tack_h_half = tackle_h_all[first : last + 1]
        tack_a_half = tackle_a_all[first : last + 1]
        oob_half = features["oob"][first : last + 1]
        rps_half = roll_speed_fwd[first : last + 1]
        np_half = features["n_players"][first : last + 1]

        chx = features["centroid_hx"][first : last + 1]
        cax = features["centroid_ax"][first : last + 1]
        att_dir_h = 1.0 if np.nanmean(chx) < np.nanmean(cax) else -1.0
        c_mid = 0.5 * (
            pd.Series(chx).interpolate().bfill().ffill()
            + pd.Series(cax).interpolate().bfill().ffill()
        )
        c_mid_smooth = c_mid.rolling(50, center=True, min_periods=1).mean().values
        adv_vel_h = att_dir_h * np.gradient(c_mid_smooth) * 25.0

        emissions = v10.compute_half_emissions(
            first, last, features, priors, pass_bridges, kalman_bs, roll_speed_fwd, adv_vel_h,
            team_log_odds=team_log_odds, team_weight=team_weight
        )
        path = v9.viterbi_decode_v7(emissions, base_trans, c_res_half, c_gh_half, c_ga_half, tack_h_half, tack_a_half, tackle_bonus=8.0)
        raw_half_labels = v9.STATE_LABELS[path]
        cleaned = v9.clean_spurious_segments_v7(raw_half_labels, oob_half, rps_half, np_half)
        full_labels[first : last + 1] = cleaned

    block_tuples = [(idx, int(b[0]), int(b[1])) for idx, b in enumerate(blocks)]
    snapped = v9.snap_transitions(
        full_labels, features["min_dist_h"], features["min_dist_a"], kalman_bs, features["oob"], block_tuples
    )
    return snapped


def evaluate_benchmark(model_name="model_v09", matches=ALL_MATCHES, seed=42, team_weight=0.5):
    """Run benchmark across matches and compute official metrics."""
    import model_v09 as v9, model_v10 as v10
    print("=" * 85)
    print(f"BENCHMARK RUN: {model_name} on {len(matches)} external matches")
    print("=" * 85)

    # Pre-extract degraded data and ML features if evaluating model_v10 (for LOMO training)
    precomputed = {}
    if model_name == "model_v10":
        print("Pre-computing degraded tracking and ML features for Leave-One-Match-Out...")
        for idx, match_id in enumerate(matches):
            npz_path = os.path.join(CACHE_DIR, f"{match_id}.npz")
            d = np.load(npz_path)
            ball = d["ball"]
            players = d["players"]
            team = d["team"]
            gt_labels = d["labels"]
            blocks = [(int(b[0]), int(b[1])) for b in d["blocks"]]
            pitch = tuple(d["pitch"]) if "pitch" in d else (105.0, 68.0)

            match_seed = seed + idx * 79
            d_ball, d_players = degrade_match(ball, players, team, gt_labels, blocks, seed=match_seed)
            _, _, kalman_bs, _ = v9.kalman_smooth_ball(d_ball[:, 0], d_ball[:, 1], np.isfinite(d_ball[:, 0]))

            b_vis = np.isfinite(d_ball[:, 0]) & np.isfinite(d_ball[:, 1])
            bx = np.where(b_vis, d_ball[:, 0], np.nan)
            by = np.where(b_vis, d_ball[:, 1], np.nan)

            ml_feat = v10.compute_ml_features(bx, by, b_vis, d_players[..., 0], d_players[..., 1], team, kalman_bs)
            precomputed[match_id] = {
                "d_ball": d_ball,
                "d_players": d_players,
                "team": team,
                "labels": gt_labels,
                "blocks": blocks,
                "pitch": pitch,
                "kalman_bs": kalman_bs,
                "ml_feat": ml_feat,
            }

    records = []
    t_start = time.time()

    for idx, match_id in enumerate(matches):
        t0 = time.time()
        if model_name == "model_v10":
            # Train Leave-One-Match-Out on all matches EXCEPT match_id
            train_matches = [m for m in matches if m != match_id]
            X_train_list, y_train_list = [], []
            for m in train_matches:
                md = precomputed[m]
                m_lab = md["labels"]
                m_X = md["ml_feat"]
                live_mask = np.isin(m_lab, ["H", "A"])
                sub = np.flatnonzero(live_mask)[::5]
                sub_X = m_X[sub]
                sub_y = (m_lab[sub] == "H").astype(np.int32)
                sub_X_swap = v10.swap_ha_features(sub_X)
                sub_y_swap = 1 - sub_y

                X_train_list.extend([sub_X, sub_X_swap])
                y_train_list.extend([sub_y, sub_y_swap])

            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)

            clf = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.08,
                num_leaves=31,
                min_child_samples=50,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
            clf.fit(X_train, y_train)

            td = precomputed[match_id]
            probs = clf.predict_proba(td["ml_feat"])
            p_h = np.clip(probs[:, 1], 1e-4, 1.0 - 1e-4)
            team_log_odds = np.log(p_h / (1.0 - p_h))

            pred_labels = predict_match_v10(
                td["d_ball"], td["d_players"], td["team"], td["blocks"], td["pitch"],
                team_log_odds=team_log_odds, team_weight=team_weight
            )
            gt_labels = td["labels"]
        else:
            npz_path = os.path.join(CACHE_DIR, f"{match_id}.npz")
            d = np.load(npz_path)
            ball = d["ball"]
            players = d["players"]
            team = d["team"]
            gt_labels = d["labels"]
            blocks = [(int(b[0]), int(b[1])) for b in d["blocks"]]
            pitch = tuple(d["pitch"]) if "pitch" in d else (105.0, 68.0)

            match_seed = seed + idx * 79
            d_ball, d_players = degrade_match(ball, players, team, gt_labels, blocks, seed=match_seed)
            pred_labels = predict_match_v09(d_ball, d_players, team, blocks, pitch)

        elapsed = time.time() - t0

        gt_codes = pm.label_codes(gt_labels)
        pred_codes = pm.label_codes(pred_labels)
        res = pm.evaluate(gt_codes, pred_codes)

        group = "DFL" if match_id in DFL_MATCHES else "Metrica"
        rec = {
            "group": group,
            "match": match_id,
            "spellF1": res["spell_f1_mean"],
            "macroF1": res["macro_f1"],
            "aliveDeadF1": res["alive_dead_f1"],
            "pred_poss": res["n_pred_spells"],
            "gt_poss": res["n_gt_spells"],
            "elapsed": elapsed,
        }
        records.append(rec)
        print(f"  {group:<7} {match_id:<18} | spellF1: {rec['spellF1']:.4f} | macroF1: {rec['macroF1']:.4f} | AliveDead: {rec['aliveDeadF1']:.4f} | Poss: {rec['pred_poss']:3d}/{rec['gt_poss']:3d} ({elapsed:.1f}s)")

    # Summary table
    print("\n" + "=" * 85)
    print(f"{'Group':<8} {'Match':<20} | {'spellF1':<8} | {'macroF1':<8} | {'Alive/Dead':<10} | {'Poss (P/GT)':<12}")
    print("-" * 85)
    for r in records:
        print(f"{r['group']:<8} {r['match']:<20} | {r['spellF1']:>8.4f} | {r['macroF1']:>8.4f} | {r['aliveDeadF1']:>10.4f} | {r['pred_poss']:>4d} / {r['gt_poss']:<4d}")

    print("-" * 85)
    dfl_recs = [r for r in records if r["group"] == "DFL"]
    met_recs = [r for r in records if r["group"] == "Metrica"]

    if dfl_recs:
        mean_dfl_spell = np.mean([r["spellF1"] for r in dfl_recs])
        mean_dfl_macro = np.mean([r["macroF1"] for r in dfl_recs])
        print(f"{'DFL Mean (7 matches)':<29} | {mean_dfl_spell:>8.4f} | {mean_dfl_macro:>8.4f} |")
    if met_recs:
        mean_met_spell = np.mean([r["spellF1"] for r in met_recs])
        mean_met_macro = np.mean([r["macroF1"] for r in met_recs])
        print(f"{'Metrica Mean (3 matches)':<29} | {mean_met_spell:>8.4f} | {mean_met_macro:>8.4f} |")
    if records:
        all_spell = np.mean([r["spellF1"] for r in records])
        all_macro = np.mean([r["macroF1"] for r in records])
        print(f"{'OVERALL Mean (10 matches)':<29} | {all_spell:>8.4f} | {all_macro:>8.4f} |")
    print("=" * 85)
    print(f"Total benchmark elapsed: {time.time() - t_start:.1f}s.\n")

    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run external benchmark.")
    parser.add_argument("--model", default="model_v09", choices=["model_v09", "model_v10"], help="Model to evaluate")
    parser.add_argument("--team-weight", type=float, default=0.5, help="Team weight for model_v10")
    parser.add_argument("--dfl-only", action="store_true", help="Run only 7 DFL matches")
    parser.add_argument("--metrica-only", action="store_true", help="Run only 3 Metrica matches")
    args = parser.parse_args()

    matches = ALL_MATCHES
    if args.dfl_only:
        matches = DFL_MATCHES
    elif args.metrica_only:
        matches = METRICA_MATCHES

    evaluate_benchmark(model_name=args.model, matches=matches, team_weight=args.team_weight)
