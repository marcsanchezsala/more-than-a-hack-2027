import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evaluation.possession_metrics as pm
import bench_external.run as r
from bench_external.fetch import DFL_MATCHES, METRICA_MATCHES, ALL_MATCHES, CACHE_DIR
from bench_external.degrade import degrade_match

def run_step1_and_step2():
    print("=" * 80)
    print("STEP 2: BAR-ATH Public Ground Truth (5-Minute Public Window Aggregate Stats)")
    print("=" * 80)
    gt_pub_df = pm.read_label_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
    pub_codes = pm.label_codes(gt_pub_df)
    pub_runs = pm.valid_runs(pub_codes)
    pub_spells = pm.possession_spells(pub_codes, pub_runs)
    pub_lens = np.array([(b - a) / 25.0 for a, b, k in pub_spells])
    pub_graded = pub_codes[pub_codes != pm.X]
    pub_live = np.isin(pub_codes, [pm.H, pm.A]).sum()
    pub_live_min = pub_live / 1500.0
    pub_d_pct = (pub_graded == pm.D).mean() * 100.0
    pub_poss_per_live_min = len(pub_spells) / pub_live_min if pub_live_min > 0 else 0.0

    b1_pub = (pub_lens < 1.0).sum()
    b2_pub = ((pub_lens >= 1.0) & (pub_lens < 3.0)).sum()
    b3_pub = ((pub_lens >= 3.0) & (pub_lens < 10.0)).sum()
    b4_pub = (pub_lens >= 10.0).sum()

    print(f"Total graded frames: {len(pub_graded)} (5.00 min)")
    print(f"Total live frames:   {pub_live} ({pub_live_min:.2f} live min)")
    print(f"Dead ball % D:       {pub_d_pct:.2f}%")
    print(f"Total spells:        {len(pub_spells)}")
    print(f"Possessions / live m:{pub_poss_per_live_min:.2f}")
    print(f"Mean spell duration: {pub_lens.mean():.2f} s")
    print(f"Median duration:     {np.median(pub_lens):.2f} s")
    print(f"Duration breakdown:  <1s: {b1_pub} ({b1_pub/len(pub_spells)*100:.1f}%), "
          f"1-3s: {b2_pub} ({b2_pub/len(pub_spells)*100:.1f}%), "
          f"3-10s: {b3_pub} ({b3_pub/len(pub_spells)*100:.1f}%), "
          f">=10s: {b4_pub} ({b4_pub/len(pub_spells)*100:.1f}%)")
    print()

    print("=" * 80)
    print("STEP 1: 10 External Matches (GT vs Model_v09 Predictions on Degraded Tracking)")
    print("=" * 80)

    results = []
    seed = 42

    for idx, match_id in enumerate(ALL_MATCHES):
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
        pred_labels = r.predict_match_v09(d_ball, d_players, team, blocks, pitch)

        gt_codes = pm.label_codes(gt_labels)
        pred_codes = pm.label_codes(pred_labels)
        runs = [(int(b[0]), int(b[1]) + 1) for b in blocks]

        gt_spells = pm.possession_spells(gt_codes, runs)
        pred_spells = pm.possession_spells(pred_codes, runs)
        overlaps = pm.spell_overlaps(gt_spells, pred_spells)
        matched_flags = pm.matched_gt_flags(overlaps, len(gt_spells), 0.5)

        gt_lens = np.array([(b - a) / 25.0 for a, b, k in gt_spells])
        pred_lens = np.array([(b - a) / 25.0 for a, b, k in pred_spells])

        b1_gt = gt_lens < 1.0
        b2_gt = (gt_lens >= 1.0) & (gt_lens < 3.0)
        b3_gt = (gt_lens >= 3.0) & (gt_lens < 10.0)
        b4_gt = gt_lens >= 10.0

        b1_pred = pred_lens < 1.0
        b2_pred = (pred_lens >= 1.0) & (pred_lens < 3.0)
        b3_pred = (pred_lens >= 3.0) & (pred_lens < 10.0)
        b4_pred = pred_lens >= 10.0

        rec_all = matched_flags.sum() / len(gt_spells) if len(gt_spells) else 0.0
        rec_b1 = matched_flags[b1_gt].sum() / b1_gt.sum() if b1_gt.sum() else 0.0
        rec_b2 = matched_flags[b2_gt].sum() / b2_gt.sum() if b2_gt.sum() else 0.0
        rec_b3 = matched_flags[b3_gt].sum() / b3_gt.sum() if b3_gt.sum() else 0.0
        rec_b4 = matched_flags[b4_gt].sum() / b4_gt.sum() if b4_gt.sum() else 0.0

        gt_graded = gt_codes[gt_codes != pm.X]
        pred_graded = pred_codes[pred_codes != pm.X]
        live_min_gt = np.isin(gt_codes, [pm.H, pm.A]).sum() / 1500.0
        live_min_pred = np.isin(pred_codes, [pm.H, pm.A]).sum() / 1500.0

        d_gt = np.mean(gt_graded == pm.D) * 100.0
        d_pred = np.mean(pred_graded == pm.D) * 100.0

        group = "DFL" if match_id in DFL_MATCHES else "Metrica"
        res_rec = {
            "group": group,
            "match": match_id,
            "gt_spells": len(gt_spells),
            "pred_spells": len(pred_spells),
            "gt_live_min": live_min_gt,
            "pred_live_min": live_min_pred,
            "gt_poss_per_live_min": len(gt_spells) / live_min_gt if live_min_gt > 0 else 0.0,
            "pred_poss_per_live_min": len(pred_spells) / live_min_pred if live_min_pred > 0 else 0.0,
            "gt_mean_dur": gt_lens.mean(),
            "pred_mean_dur": pred_lens.mean(),
            "gt_med_dur": np.median(gt_lens),
            "pred_med_dur": np.median(pred_lens),
            "gt_d_pct": d_gt,
            "pred_d_pct": d_pred,
            "rec_all": rec_all,
            "rec_b1": rec_b1,
            "rec_b2": rec_b2,
            "rec_b3": rec_b3,
            "rec_b4": rec_b4,
            "n_gt_b1": int(b1_gt.sum()),
            "n_gt_b2": int(b2_gt.sum()),
            "n_gt_b3": int(b3_gt.sum()),
            "n_gt_b4": int(b4_gt.sum()),
            "n_pred_b1": int(b1_pred.sum()),
            "n_pred_b2": int(b2_pred.sum()),
            "n_pred_b3": int(b3_pred.sum()),
            "n_pred_b4": int(b4_pred.sum()),
            "matched_b1": int(matched_flags[b1_gt].sum()),
            "matched_b2": int(matched_flags[b2_gt].sum()),
            "matched_b3": int(matched_flags[b3_gt].sum()),
            "matched_b4": int(matched_flags[b4_gt].sum()),
        }
        results.append(res_rec)

        print(f"[{group}] {match_id}:")
        print(f"  GT:   spells={res_rec['gt_spells']:3d} | poss/live_m={res_rec['gt_poss_per_live_min']:4.1f} | mean={res_rec['gt_mean_dur']:5.2f}s | med={res_rec['gt_med_dur']:5.2f}s | D%={res_rec['gt_d_pct']:4.1f}%")
        print(f"  Pred: spells={res_rec['pred_spells']:3d} | poss/live_m={res_rec['pred_poss_per_live_min']:4.1f} | mean={res_rec['pred_mean_dur']:5.2f}s | med={res_rec['pred_med_dur']:5.2f}s | D%={res_rec['pred_d_pct']:4.1f}%")
        print(f"  Recall@0.5 by dur:")
        print(f"    <1s:   {rec_b1*100:5.1f}% ({res_rec['matched_b1']}/{res_rec['n_gt_b1']}) | pred_count={res_rec['n_pred_b1']}")
        print(f"    1-3s:  {rec_b2*100:5.1f}% ({res_rec['matched_b2']}/{res_rec['n_gt_b2']}) | pred_count={res_rec['n_pred_b2']}")
        print(f"    3-10s: {rec_b3*100:5.1f}% ({res_rec['matched_b3']}/{res_rec['n_gt_b3']}) | pred_count={res_rec['n_pred_b3']}")
        print(f"    >=10s: {rec_b4*100:5.1f}% ({res_rec['matched_b4']}/{res_rec['n_gt_b4']}) | pred_count={res_rec['n_pred_b4']}")
        print(f"    ALL:   {rec_all*100:5.1f}% ({matched_flags.sum()}/{len(gt_spells)})")
        print()

    # Aggregate summaries
    df_res = pd.DataFrame(results)
    print("=" * 80)
    print("SUMMARY BY GROUP (GT vs Model_v09 Pred):")
    print("=" * 80)
    for grp in ["DFL", "Metrica"]:
        sub = df_res[df_res["group"] == grp]
        print(f"--- {grp} (n={len(sub)}) ---")
        print(f"  GT poss/live_m:   {sub['gt_poss_per_live_min'].mean():.2f}  vs  Pred poss/live_m:   {sub['pred_poss_per_live_min'].mean():.2f}")
        print(f"  GT spells:        {sub['gt_spells'].mean():.1f}   vs  Pred spells:        {sub['pred_spells'].mean():.1f}")
        print(f"  GT mean duration: {sub['gt_mean_dur'].mean():.2f}s vs  Pred mean duration: {sub['pred_mean_dur'].mean():.2f}s")
        print(f"  GT med duration:  {sub['gt_med_dur'].mean():.2f}s  vs  Pred med duration:  {sub['pred_med_dur'].mean():.2f}s")
        print(f"  GT D%:            {sub['gt_d_pct'].mean():.1f}%   vs  Pred D%:            {sub['pred_d_pct'].mean():.1f}%")
        print(f"  Recall@0.5 by dur:")
        tot_b1_gt = sub['n_gt_b1'].sum()
        tot_b2_gt = sub['n_gt_b2'].sum()
        tot_b3_gt = sub['n_gt_b3'].sum()
        tot_b4_gt = sub['n_gt_b4'].sum()
        tot_b1_m = sub['matched_b1'].sum()
        tot_b2_m = sub['matched_b2'].sum()
        tot_b3_m = sub['matched_b3'].sum()
        tot_b4_m = sub['matched_b4'].sum()
        tot_all_m = tot_b1_m + tot_b2_m + tot_b3_m + tot_b4_m
        tot_all_gt = tot_b1_gt + tot_b2_gt + tot_b3_gt + tot_b4_gt
        print(f"    <1s:   {tot_b1_m/tot_b1_gt*100:5.1f}% ({tot_b1_m}/{tot_b1_gt}) | Pred count: {sub['n_pred_b1'].sum()}")
        print(f"    1-3s:  {tot_b2_m/tot_b2_gt*100:5.1f}% ({tot_b2_m}/{tot_b2_gt}) | Pred count: {sub['n_pred_b2'].sum()}")
        print(f"    3-10s: {tot_b3_m/tot_b3_gt*100:5.1f}% ({tot_b3_m}/{tot_b3_gt}) | Pred count: {sub['n_pred_b3'].sum()}")
        print(f"    >=10s: {tot_b4_m/tot_b4_gt*100:5.1f}% ({tot_b4_m}/{tot_b4_gt}) | Pred count: {sub['n_pred_b4'].sum()}")
        print(f"    ALL:   {tot_all_m/tot_all_gt*100:5.1f}% ({tot_all_m}/{tot_all_gt})")
        print()

if __name__ == "__main__":
    run_step1_and_step2()
