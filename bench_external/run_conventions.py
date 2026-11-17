"""Evaluation across label-convention smoothing thresholds.

Compares model_v09 and model_v10 (Leave-One-Match-Out) under:
  1. Raw convention (N = 0 frames)
  2. Deflection smoothing N = 25 frames (1.0 s)
  3. Deflection smoothing N = 50 frames (2.0 s)
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
import model_v09 as v9
import model_v10 as v10
import bench_external.run as r
from bench_external.fetch import DFL_MATCHES, METRICA_MATCHES, ALL_MATCHES, CACHE_DIR
from bench_external.degrade import degrade_match
from bench_external.convention import smooth_label_convention


def run_convention_benchmark(seed=42, team_weight=0.5):
    print("=" * 90, flush=True)
    print("STARTING EXTERNAL LABEL CONVENTION BENCHMARK", flush=True)
    print("Matches: 7 DFL Bundesliga (CC BY 4.0) + 3 Metrica Sports", flush=True)
    print("Conventions to evaluate: Raw (N=0), N=25 frames (1.0s), N=50 frames (2.0s)", flush=True)
    print("=" * 90, flush=True)

    t0_all = time.time()

    # Step 1: Pre-load and degrade all matches once
    print("Loading and degrading tracking data for all 10 matches...", flush=True)
    match_data = {}
    for idx, match_id in enumerate(ALL_MATCHES):
        npz_path = os.path.join(CACHE_DIR, f"{match_id}.npz")
        d = np.load(npz_path)
        ball = d["ball"]
        players = d["players"]
        team = d["team"]
        gt_labels_raw = d["labels"]
        blocks = [(int(b[0]), int(b[1])) for b in d["blocks"]]
        pitch = tuple(d["pitch"]) if "pitch" in d else (105.0, 68.0)

        match_seed = seed + idx * 79
        d_ball, d_players = degrade_match(ball, players, team, gt_labels_raw, blocks, seed=match_seed)
        _, _, kalman_bs, _ = v9.kalman_smooth_ball(d_ball[:, 0], d_ball[:, 1], np.isfinite(d_ball[:, 0]))

        b_vis = np.isfinite(d_ball[:, 0]) & np.isfinite(d_ball[:, 1])
        bx = np.where(b_vis, d_ball[:, 0], np.nan)
        by = np.where(b_vis, d_ball[:, 1], np.nan)

        ml_feat = v10.compute_ml_features(bx, by, b_vis, d_players[..., 0], d_players[..., 1], team, kalman_bs)
        match_data[match_id] = {
            "d_ball": d_ball,
            "d_players": d_players,
            "team": team,
            "raw_labels": gt_labels_raw,
            "blocks": blocks,
            "runs": [(int(b[0]), int(b[1]) + 1) for b in blocks],
            "pitch": pitch,
            "kalman_bs": kalman_bs,
            "ml_feat": ml_feat,
        }

    # Step 2: Generate model_v09 predictions once (v09 is purely physical and label-independent)
    print("Computing model_v09 predictions across all 10 matches...", flush=True)
    v09_preds = {}
    for match_id in ALL_MATCHES:
        md = match_data[match_id]
        v09_preds[match_id] = r.predict_match_v09(
            md["d_ball"], md["d_players"], md["team"], md["blocks"], md["pitch"]
        )

    conventions = [
        ("Raw (N=0)", 0),
        ("Deflection (N=25 / 1.0s)", 25),
        ("Deflection (N=50 / 2.0s)", 50),
    ]

    summary_rows = []

    for conv_name, N_frames in conventions:
        print("\n" + "=" * 90, flush=True)
        print(f"EVALUATING CONVENTION: {conv_name}", flush=True)
        print("=" * 90, flush=True)

        # Apply convention smoothing to GT labels
        conv_gt = {}
        for match_id in ALL_MATCHES:
            md = match_data[match_id]
            if N_frames > 0:
                conv_gt[match_id] = smooth_label_convention(
                    md["raw_labels"], md["blocks"], min_spell_frames=N_frames, merge_micro_d=True
                )
            else:
                conv_gt[match_id] = md["raw_labels"].copy()

        # Print GT stats under this convention
        dfl_spells = []
        met_spells = []
        dfl_d_pct = []
        for match_id in ALL_MATCHES:
            lbl = conv_gt[match_id]
            runs = match_data[match_id]["runs"]
            sp = pm.possession_spells(pm.label_codes(lbl), runs)
            graded = lbl[lbl != "X"]
            d_pct = (graded == "D").mean() * 100.0
            if match_id in DFL_MATCHES:
                dfl_spells.append(len(sp))
                dfl_d_pct.append(d_pct)
            else:
                met_spells.append(len(sp))

        print(f"GT Profile under {conv_name}:", flush=True)
        print(f"  DFL Mean Spells:     {np.mean(dfl_spells):.1f} (range {min(dfl_spells)} - {max(dfl_spells)})", flush=True)
        print(f"  Metrica Mean Spells: {np.mean(met_spells):.1f} (range {min(met_spells)} - {max(met_spells)})", flush=True)
        print(f"  DFL Mean D%:         {np.mean(dfl_d_pct):.1f}%", flush=True)

        # 1. Score model_v09
        v09_scores = {}
        for match_id in ALL_MATCHES:
            gt_codes = pm.label_codes(conv_gt[match_id])
            pred_codes = pm.label_codes(v09_preds[match_id])
            res = pm.evaluate(gt_codes, pred_codes)
            v09_scores[match_id] = res

        # 2. Train and score model_v10 under LOMO (trained on smoothed convention labels)
        print("Training model_v10 under LOMO cross-validation...", flush=True)
        v10_scores = {}
        for match_id in ALL_MATCHES:
            train_matches = [m for m in ALL_MATCHES if m != match_id]
            X_train_list, y_train_list = [], []
            for m in train_matches:
                md = match_data[m]
                m_lab = conv_gt[m]
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

            td = match_data[match_id]
            probs = clf.predict_proba(td["ml_feat"])
            p_h = np.clip(probs[:, 1], 1e-4, 1.0 - 1e-4)
            team_log_odds = np.log(p_h / (1.0 - p_h))

            pred_labels_v10 = r.predict_match_v10(
                td["d_ball"], td["d_players"], td["team"], td["blocks"], td["pitch"],
                team_log_odds=team_log_odds, team_weight=team_weight
            )

            gt_codes = pm.label_codes(conv_gt[match_id])
            pred_codes = pm.label_codes(pred_labels_v10)
            v10_scores[match_id] = pm.evaluate(gt_codes, pred_codes)

        # Print per-match comparison under this convention
        print(f"\nResults Table ({conv_name}):", flush=True)
        print(f"{'Match':<18} | {'v09 spellF1':<12} | {'v10 spellF1':<12} | {'Delta':<8} | {'v09 macro':<10} | {'v10 macro':<10} | {'v10 Won?':<8}", flush=True)
        print("-" * 90, flush=True)

        dfl_v09_f1, dfl_v10_f1 = [], []
        met_v09_f1, met_v10_f1 = [], []
        dfl_v09_macro, dfl_v10_macro = [], []
        dfl_wins, met_wins = 0, 0

        for match_id in ALL_MATCHES:
            s09 = v09_scores[match_id]["spell_f1_mean"]
            s10 = v10_scores[match_id]["spell_f1_mean"]
            m09 = v09_scores[match_id]["macro_f1"]
            m10 = v10_scores[match_id]["macro_f1"]
            delta = s10 - s09
            won = delta >= 0.0

            if match_id in DFL_MATCHES:
                dfl_v09_f1.append(s09)
                dfl_v10_f1.append(s10)
                dfl_v09_macro.append(m09)
                dfl_v10_macro.append(m10)
                if won:
                    dfl_wins += 1
            else:
                met_v09_f1.append(s09)
                met_v10_f1.append(s10)
                if won:
                    met_wins += 1

            won_str = "YES" if won else "NO"
            print(f"{match_id:<18} | {s09:>12.4f} | {s10:>12.4f} | {delta:>+8.4f} | {m09:>10.4f} | {m10:>10.4f} | {won_str:<8}", flush=True)

        print("-" * 90, flush=True)
        dfl_mean_09 = np.mean(dfl_v09_f1)
        dfl_mean_10 = np.mean(dfl_v10_f1)
        met_mean_09 = np.mean(met_v09_f1)
        met_mean_10 = np.mean(met_v10_f1)
        all_mean_09 = np.mean(dfl_v09_f1 + met_v09_f1)
        all_mean_10 = np.mean(dfl_v10_f1 + met_v10_f1)

        print(f"DFL Mean (7)       | {dfl_mean_09:>12.4f} | {dfl_mean_10:>12.4f} | {dfl_mean_10 - dfl_mean_09:>+8.4f} | Won {dfl_wins}/7 DFL matches", flush=True)
        print(f"Metrica Mean (3)   | {met_mean_09:>12.4f} | {met_mean_10:>12.4f} | {met_mean_10 - met_mean_09:>+8.4f} | Won {met_wins}/3 Metrica matches", flush=True)
        print(f"Overall Mean (10)  | {all_mean_09:>12.4f} | {all_mean_10:>12.4f} | {all_mean_10 - all_mean_09:>+8.4f} | Won {dfl_wins + met_wins}/10 Overall", flush=True)

        summary_rows.append({
            "convention": conv_name,
            "dfl_v09_spell": dfl_mean_09,
            "dfl_v10_spell": dfl_mean_10,
            "dfl_spell_delta": dfl_mean_10 - dfl_mean_09,
            "dfl_wins": dfl_wins,
            "met_v09_spell": met_mean_09,
            "met_v10_spell": met_mean_10,
            "met_spell_delta": met_mean_10 - met_mean_09,
            "met_wins": met_wins,
            "all_v09_spell": all_mean_09,
            "all_v10_spell": all_mean_10,
            "all_spell_delta": all_mean_10 - all_mean_09,
            "all_wins": dfl_wins + met_wins,
        })

    print("\n" + "=" * 90, flush=True)
    print("FINAL SUMMARY COMPARISON ACROSS CONVENTIONS:", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Convention':<26} | {'DFL v09':<8} | {'DFL v10':<8} | {'DFL D':<7} | {'DFL Won':<8} | {'Met v09':<8} | {'Met v10':<8} | {'Met Won':<8} | {'All v09':<8} | {'All v10':<8}", flush=True)
    print("-" * 115, flush=True)
    for r_row in summary_rows:
        print(f"{r_row['convention']:<26} | {r_row['dfl_v09_spell']:>8.4f} | {r_row['dfl_v10_spell']:>8.4f} | {r_row['dfl_spell_delta']:>+7.4f} | {r_row['dfl_wins']:>4d}/7   | {r_row['met_v09_spell']:>8.4f} | {r_row['met_v10_spell']:>8.4f} | {r_row['met_wins']:>4d}/3   | {r_row['all_v09_spell']:>8.4f} | {r_row['all_v10_spell']:>8.4f}", flush=True)
    print("=" * 90, flush=True)
    print(f"Total run time: {time.time() - t0_all:.1f}s", flush=True)
    return summary_rows


if __name__ == "__main__":
    run_convention_benchmark()
