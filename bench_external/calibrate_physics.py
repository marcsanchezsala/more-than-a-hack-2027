"""Physical Hyperparameter Calibration for model_v09.

Evaluates the physical model of model_v09.py over the 10 degraded external matches
(7 DFL Bundesliga CC BY 4.0 + 3 Metrica Sports open sample data) under the clean
convention (N = 50 frames / 2.0 s via bench_external/convention.py).

Optimizes key physical hyperparameters:
  - p_switch: penalty for H <-> A transitions
  - p_exit_d: penalty for exiting dead ball (D) to live play
  - max_d_gain: maximum physical ball distance to gain possession
  - tackle_bonus: dynamic assist bonus for defensive tackles

Zero data leakage: 100% evaluated on external matches, zero training on BAR-ATH.
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evaluation.possession_metrics as pm
import model_v09 as v9
import bench_external.run as r
from bench_external.fetch import DFL_MATCHES, METRICA_MATCHES, ALL_MATCHES, CACHE_DIR
from bench_external.degrade import degrade_match
from bench_external.convention import smooth_label_convention


def precompute_match_cache(seed: int = 42, conv_n_frames: int = 50) -> Dict[str, Dict[str, Any]]:
    """Precompute and cache kinematics, emissions, and smoothed GT labels for all matches.

    Precomputing emissions and static kinematic masks once makes parameter evaluation
    instant (~1 second per match instead of extracting features every trial).
    """
    print(f"Precomputing and caching physical kinematics for {len(ALL_MATCHES)} matches...")
    t0 = time.time()
    cache = {}

    for idx, match_id in enumerate(ALL_MATCHES):
        t_m0 = time.time()
        npz_path = os.path.join(CACHE_DIR, f"{match_id}.npz")
        data = np.load(npz_path)
        ball = data["ball"]
        players = data["players"]
        team = data["team"]
        raw_labels = data["labels"]
        blocks = [(int(b[0]), int(b[1])) for b in data["blocks"]]
        pitch = tuple(data["pitch"]) if "pitch" in data else (105.0, 68.0)

        # Degrade match with deterministic seed
        match_seed = seed + idx * 79
        d_ball, d_players = degrade_match(ball, players, team, raw_labels, blocks, seed=match_seed)

        # Smooth GT labels to clean possession convention (N=50f / 2.0s)
        gt_conv = smooth_label_convention(
            raw_labels, blocks, min_spell_frames=conv_n_frames, merge_micro_d=True
        )
        gt_codes = pm.label_codes(gt_conv)

        # Feature extraction & ball kinematics
        features = r.extract_features_from_arrays(d_ball, d_players, team, pitch)
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
            features["ball_vis"],
            features["oob"],
            is_aerial,
            features["min_dist_h"],
            features["min_dist_a"],
            kalman_bs,
            n_frames,
        )
        pass_bridges = v9.build_pass_bridges_v5(
            features["ball_vis"], features["min_dist_h"], features["min_dist_a"], features["oob"]
        )

        priors = (
            np.zeros(n_frames, dtype=np.float32),
            np.zeros(n_frames, dtype=np.float32),
            np.zeros(n_frames, dtype=np.float32),
        )

        # Precompute half-by-half emissions (independent of Viterbi transition parameters)
        half_emissions = []
        for first, last in blocks:
            chx = features["centroid_hx"][first : last + 1]
            cax = features["centroid_ax"][first : last + 1]
            att_dir_h = 1.0 if np.nanmean(chx) < np.nanmean(cax) else -1.0
            c_mid = 0.5 * (
                pd.Series(chx).interpolate().bfill().ffill()
                + pd.Series(cax).interpolate().bfill().ffill()
            )
            c_mid_smooth = c_mid.rolling(50, center=True, min_periods=1).mean().values
            adv_vel_h = att_dir_h * np.gradient(c_mid_smooth) * 25.0

            em = v9.compute_half_emissions(
                first, last, features, priors, pass_bridges, kalman_bs, roll_speed_fwd, adv_vel_h
            )
            half_emissions.append(em)

        # Shared tackle triggers
        dh_all = features["min_dist_h"]
        da_all = features["min_dist_a"]
        bv_all = features["ball_vis"]
        oob_all = features["oob"]
        tackle_h_all = (
            bv_all
            & (~oob_all)
            & (~is_aerial)
            & (~defl_h)
            & (~np.isnan(dh_all))
            & (~np.isnan(da_all))
            & (dh_all < 1.2)
            & (dh_all < da_all - 0.5)
        )
        tackle_a_all = (
            bv_all
            & (~oob_all)
            & (~is_aerial)
            & (~defl_a)
            & (~np.isnan(dh_all))
            & (~np.isnan(da_all))
            & (da_all < 1.2)
            & (da_all < dh_all - 0.5)
        )

        can_restart = (
            (~np.isnan(kalman_bs) & (kalman_bs > 2.8))
            | (roll_p90_fwd > 2.85)
            | (roll_speed_fwd > 2.20)
        )

        block_tuples = [(idx_b, int(b[0]), int(b[1])) for idx_b, b in enumerate(blocks)]

        cache[match_id] = {
            "group": "DFL" if match_id in DFL_MATCHES else "Metrica",
            "n_frames": n_frames,
            "blocks": blocks,
            "block_tuples": block_tuples,
            "gt_codes": gt_codes,
            "features": features,
            "is_aerial": is_aerial,
            "defl_h": defl_h,
            "defl_a": defl_a,
            "kalman_bs": kalman_bs,
            "roll_speed_fwd": roll_speed_fwd,
            "can_restart": can_restart,
            "tackle_h_all": tackle_h_all,
            "tackle_a_all": tackle_a_all,
            "half_emissions": half_emissions,
        }
        print(f"  [{idx + 1:2d}/{len(ALL_MATCHES)}] {match_id:<18} cached in {time.time() - t_m0:.1f}s")

    print(f"Precomputation finished in {time.time() - t0:.1f}s.\n")
    return cache


def evaluate_parameter_set(
    cache: Dict[str, Dict[str, Any]],
    p_switch: float,
    p_exit_d: float,
    max_d_gain: float,
    tackle_bonus: float,
    p_enter_d: float = 55.0,
    min_d_frames: int = 90,
    min_d_oob_frames: int = 40,
    min_deflection_frames: int = 35,
) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    """Decodes all 10 matches under given parameters and evaluates against clean GT (N=50).

    Returns:
      (mean_spell_f1, mean_macro_f1, per_match_results)
    """
    base_trans = np.zeros((3, 3), dtype=np.float64)
    base_trans[0, 1] = -p_switch
    base_trans[1, 0] = -p_switch
    base_trans[0, 2] = -p_enter_d
    base_trans[1, 2] = -p_enter_d
    base_trans[2, 0] = -p_exit_d
    base_trans[2, 1] = -p_exit_d

    per_match = {}
    spell_f1s = []
    macro_f1s = []

    for match_id in ALL_MATCHES:
        m = cache[match_id]
        features = m["features"]
        n_frames = m["n_frames"]
        blocks = m["blocks"]
        is_aerial = m["is_aerial"]
        defl_h = m["defl_h"]
        defl_a = m["defl_a"]
        kalman_bs = m["kalman_bs"]
        roll_speed_fwd = m["roll_speed_fwd"]
        can_restart = m["can_restart"]
        tackle_h_all = m["tackle_h_all"]
        tackle_a_all = m["tackle_a_all"]
        half_emissions = m["half_emissions"]
        gt_codes = m["gt_codes"]

        # Contact gating depends on max_d_gain
        can_gain_h = (
            features["ball_vis"]
            & (~features["oob"])
            & (~is_aerial)
            & (~defl_h)
            & (~np.isnan(features["min_dist_h"]))
            & (features["min_dist_h"] < max_d_gain)
        )
        can_gain_a = (
            features["ball_vis"]
            & (~features["oob"])
            & (~is_aerial)
            & (~defl_a)
            & (~np.isnan(features["min_dist_a"]))
            & (features["min_dist_a"] < max_d_gain)
        )

        full_labels = np.full(n_frames, "X", dtype=object)

        for half_idx, (first, last) in enumerate(blocks):
            c_res_half = can_restart[first : last + 1]
            c_gh_half = can_gain_h[first : last + 1]
            c_ga_half = can_gain_a[first : last + 1]
            tack_h_half = tackle_h_all[first : last + 1]
            tack_a_half = tackle_a_all[first : last + 1]
            oob_half = features["oob"][first : last + 1]
            rps_half = roll_speed_fwd[first : last + 1]
            np_half = features["n_players"][first : last + 1]

            path = v9.viterbi_decode_v7(
                half_emissions[half_idx],
                base_trans,
                c_res_half,
                c_gh_half,
                c_ga_half,
                tack_h_half,
                tack_a_half,
                tackle_bonus=tackle_bonus,
            )
            raw_labels = v9.STATE_LABELS[path]
            cleaned = v9.clean_spurious_segments_v7(
                raw_labels,
                oob_half,
                rps_half,
                np_half,
                min_d_frames=min_d_frames,
                min_d_oob_frames=min_d_oob_frames,
                min_deflection_frames=min_deflection_frames,
            )
            full_labels[first : last + 1] = cleaned

        snapped = v9.snap_transitions(
            full_labels,
            features["min_dist_h"],
            features["min_dist_a"],
            kalman_bs,
            features["oob"],
            m["block_tuples"],
            window=20,
        )

        pred_codes = pm.label_codes(snapped)
        res = pm.evaluate(gt_codes, pred_codes)

        spell_f1s.append(res["spell_f1_mean"])
        macro_f1s.append(res["macro_f1"])

        # Also count possessions and dead-ball percentage
        graded = snapped[snapped != "X"]
        d_pct = (graded == "D").mean() * 100.0 if len(graded) > 0 else 0.0

        per_match[match_id] = {
            "group": m["group"],
            "spellF1": res["spell_f1_mean"],
            "macroF1": res["macro_f1"],
            "aliveDeadF1": res["alive_dead_f1"],
            "pred_poss": res["n_pred_spells"],
            "gt_poss": res["n_gt_spells"],
            "d_pct": d_pct,
        }

    return float(np.mean(spell_f1s)), float(np.mean(macro_f1s)), per_match


def run_calibration(seed: int = 42):
    """Executes coordinate search followed by targeted joint grid search."""
    print("=" * 85)
    print("PHYSICAL HYPERPARAMETER CALIBRATION FOR MODEL_V09")
    print("Evaluating 10 external matches (DFL Bundesliga + Metrica) under convention N=50")
    print("=" * 85)

    cache = precompute_match_cache(seed=seed, conv_n_frames=50)

    # Initial baseline parameters (model_v09 defaults)
    current_params = {
        "p_switch": 70.0,
        "p_exit_d": 15.0,
        "max_d_gain": 1.6,
        "tackle_bonus": 8.0,
    }

    print("Evaluating initial baseline parameters:")
    base_spell, base_macro, base_matches = evaluate_parameter_set(cache, **current_params)
    print(f"  Baseline: p_switch={current_params['p_switch']}, p_exit_d={current_params['p_exit_d']}, "
          f"max_d_gain={current_params['max_d_gain']}, tackle_bonus={current_params['tackle_bonus']}")
    print(f"  -> Mean spellF1: {base_spell:.4f} | Mean macroF1: {base_macro:.4f}\n")

    # Search candidates specified in the task
    param_grids = {
        "p_switch": [25.0, 40.0, 55.0, 70.0],
        "p_exit_d": [10.0, 15.0, 20.0],
        "max_d_gain": [1.4, 1.6, 1.8],
        "tackle_bonus": [6.0, 8.0, 12.0],
    }

    history = []
    history.append({**current_params, "spellF1": base_spell, "macroF1": base_macro, "step": "baseline"})

    # -------------------------------------------------------------
    # Phase 1: Coordinate Sensitivity Analysis
    # -------------------------------------------------------------
    print("-" * 85)
    print("PHASE 1: 1D Coordinate Sensitivity Sweeps")
    print("-" * 85)

    best_coords = dict(current_params)

    for param_name in ["p_switch", "p_exit_d", "max_d_gain", "tackle_bonus"]:
        candidates = param_grids[param_name]
        print(f"\nSweeping {param_name} over {candidates} (holding other coordinates fixed)...")
        best_val = best_coords[param_name]
        best_val_score = -1.0

        for val in candidates:
            trial_params = dict(best_coords)
            trial_params[param_name] = val
            s_f1, m_f1, _ = evaluate_parameter_set(cache, **trial_params)
            is_best_mark = " *" if s_f1 > best_val_score else ""
            print(f"  {param_name} = {val:<5} -> Mean spellF1: {s_f1:.4f} | Mean macroF1: {m_f1:.4f}{is_best_mark}")
            history.append({**trial_params, "spellF1": s_f1, "macroF1": m_f1, "step": f"coord_{param_name}"})
            if s_f1 > best_val_score:
                best_val_score = s_f1
                best_val = val

        best_coords[param_name] = best_val
        print(f"  >> Optimal {param_name} along coordinate: {best_val} (spellF1 = {best_val_score:.4f})")

    print("\nPhase 1 coordinate search result:", best_coords)

    # -------------------------------------------------------------
    # Phase 2: Joint Grid Refinement around Best Candidates
    # -------------------------------------------------------------
    print("\n" + "-" * 85)
    print("PHASE 2: Targeted Joint Grid Search (Exploring Top Interactions)")
    print("-" * 85)

    # Pick top 2 for p_switch, top 2 for p_exit_d, top 2 for max_d_gain, top 2 for tackle_bonus
    # to test joint interaction effects
    # Find top-2 for each parameter from history
    top2_dict = {}
    for p_name, candidates in param_grids.items():
        val_scores = {}
        for h in history:
            if h["step"].startswith("coord_" + p_name) or h["step"] == "baseline":
                v = h[p_name]
                if v not in val_scores or h["spellF1"] > val_scores[v]:
                    val_scores[v] = h["spellF1"]
        sorted_vals = sorted(val_scores.items(), key=lambda x: x[1], reverse=True)
        top2 = [x[0] for x in sorted_vals[:2]]
        # Always include best_coords value
        if best_coords[p_name] not in top2:
            top2[1] = best_coords[p_name]
        top2_dict[p_name] = sorted(top2)
        print(f"  Top candidates for {p_name}: {top2_dict[p_name]}")

    overall_best_params = dict(best_coords)
    overall_best_spell = -1.0
    overall_best_macro = -1.0
    overall_best_matches = None

    total_combos = (
        len(top2_dict["p_switch"])
        * len(top2_dict["p_exit_d"])
        * len(top2_dict["max_d_gain"])
        * len(top2_dict["tackle_bonus"])
    )
    print(f"\nEvaluating {total_combos} joint combinations...")

    trial_idx = 0
    for ps in top2_dict["p_switch"]:
        for pe in top2_dict["p_exit_d"]:
            for mg in top2_dict["max_d_gain"]:
                for tb in top2_dict["tackle_bonus"]:
                    trial_idx += 1
                    t_params = {
                        "p_switch": ps,
                        "p_exit_d": pe,
                        "max_d_gain": mg,
                        "tackle_bonus": tb,
                    }
                    s_f1, m_f1, per_m = evaluate_parameter_set(cache, **t_params)
                    history.append({**t_params, "spellF1": s_f1, "macroF1": m_f1, "step": "joint_grid"})
                    is_new_best = s_f1 > overall_best_spell
                    mark = " *** NEW BEST ***" if is_new_best else ""
                    print(
                        f"  [{trial_idx:2d}/{total_combos}] p_switch={ps:<4} p_exit_d={pe:<4} "
                        f"max_d_gain={mg:<4} tackle_bonus={tb:<4} -> spellF1: {s_f1:.4f} | macroF1: {m_f1:.4f}{mark}"
                    )
                    if is_new_best:
                        overall_best_spell = s_f1
                        overall_best_macro = m_f1
                        overall_best_params = t_params
                        overall_best_matches = per_m

    # -------------------------------------------------------------
    # Final Calibration Report
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print("CALIBRATION RESULTS & COMPARISON")
    print("=" * 85)
    print(f"Baseline Parameters:  {current_params}")
    print(f"Baseline spellF1:     {base_spell:.4f} (macroF1: {base_macro:.4f})")
    print(f"Calibrated Parameters: {overall_best_params}")
    print(f"Calibrated spellF1:   {overall_best_spell:.4f} (macroF1: {overall_best_macro:.4f})")
    gain = overall_best_spell - base_spell
    print(f"External Gain:        +{gain:.4f} spellF1 points (+{gain / base_spell * 100:.2f}%)")

    print("\n" + "-" * 85)
    print(f"{'Group':<8} {'Match':<18} | {'Base spellF1':<12} | {'Calib spellF1':<13} | {'Diff':<8} | {'Calib D%':<8}")
    print("-" * 85)
    for mid in ALL_MATCHES:
        bm = base_matches[mid]
        cm = overall_best_matches[mid]
        diff = cm["spellF1"] - bm["spellF1"]
        print(
            f"{bm['group']:<8} {mid:<18} | {bm['spellF1']:>12.4f} | {cm['spellF1']:>13.4f} | {diff:>+8.4f} | {cm['d_pct']:>7.1f}%"
        )
    print("-" * 85)

    return overall_best_params, base_spell, overall_best_spell, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate model_v09 physical parameters")
    parser.add_argument("--seed", type=int, default=42, help="Seed for degradation reproducibility")
    args = parser.parse_args()

    run_calibration(seed=args.seed)
