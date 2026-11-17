# model_v12 (v1.2): not adopted

**Outcome: no variant meets the acceptance criteria. model_v11 stays the submission.**
`evaluation/results/model_v11.csv` and `model_v11.py` are unchanged; no `model_v12.csv` was produced.

v1.2 tried to fix one failure of v11. When a visible ball moves during a stoppage, v11 does not label
it dead ball. The cause was measured without labels (`model_v11.py --domain`): in degraded DFL, a
dead ball hardly moves. Two changes were built in `model_v12.py`, which imports model_v11 and does
not modify it:

1. **Dead-ball play** in the DFL degradation (`dead_ball_play`, which replaces `carry_ball`).
2. **Alive/dead features chosen by a domain classifier.** Ball-kinematics features that give away
   BAR-ATH vs the external data are dropped from the alive/dead model only. The team model is not
   changed.

```bash
python model_v12.py --calibrate                   # grid for the dead-ball play constants (label-free, ~3 min)
python model_v12.py --domain                      # bin statistics + domain classifier (~2 min)
python model_v12.py --bench --play v12|v11 --alive domain|all   # ~10-12 min each
```

`--bench` runs `model_v11.bench` unchanged. `model_v12.use()` points its data loader and its `fit`
at the v12 versions. `--play v11 --alive all` reproduces v11's published numbers exactly (0.691 /
0.807 / shift 1.2), so all four variants are measured by the same code.

## Acceptance (`--bench`, each change alone and both together)

| | lomo spellF1 N25 (≥ 0.68) | lopo alive/dead F1, DFL (> 0.807) | BAR-ATH shift (1.2 → 0.1–0.7) | D% learned − v09 | agreement with v09 | passes |
|---|---|---|---|---|---|---|
| v11 (`--play v11 --alive all`) | 0.691 | 0.807 | 1.2 | −6.0 | 91.2% | (reference) |
| 1. dead-ball play only | **0.669** ✗ | 0.805 ✗ | **0.3** ✓ | −5.0 ✓ | 93.0% ✓ | no |
| 2. domain features only | 0.689 ✓ | 0.810 (+0.003) ≈ | 1.2 ✗ | −5.9 ✓ | 91.1% ✓ | no |
| 1 + 2 | **0.667** ✗ | 0.805 ✗ | **0.4** ✓ | −5.5 ✓ | 92.4% ✓ | no |

In-domain ranges, from v11's own bench: shift 0.1–0.7, D% −9.5 .. −2.7, agreement 87.6–93.2%.

Full tables (lomo = leave one match out, lopo = leave one provider out, learned model):

| variant | scheme | raw | N25 | N50 | macroF1 | alive/dead F1 | D% pred − true |
|---|---|---|---|---|---|---|---|
| v11 | lomo, all 10 | 0.640 | 0.691 | 0.715 | 0.895 | 0.955 | +2.2 |
| | lopo, DFL | 0.453 | 0.487 | 0.497 | 0.693 | 0.807 | −19.7 |
| | lopo, Metrica | 0.675 | 0.730 | 0.744 | 0.863 | 0.921 | +5.1 |
| play only | lomo, all 10 | 0.619 | 0.669 | 0.691 | 0.884 | 0.947 | +2.1 |
| | lopo, DFL | 0.443 | 0.473 | 0.486 | 0.687 | 0.805 | −20.4 |
| | lopo, Metrica | 0.656 | 0.712 | 0.732 | 0.859 | 0.916 | +7.3 |
| features only | lomo, all 10 | 0.638 | 0.689 | 0.713 | 0.894 | 0.954 | +2.2 |
| | lopo, DFL | 0.454 | 0.487 | 0.498 | 0.698 | 0.810 | −19.3 |
| | lopo, Metrica | 0.675 | 0.731 | 0.745 | 0.863 | 0.921 | +5.3 |
| both | lomo, all 10 | 0.617 | 0.667 | 0.688 | 0.884 | 0.947 | +2.3 |
| | lopo, DFL | 0.444 | 0.475 | 0.486 | 0.687 | 0.805 | −20.4 |
| | lopo, Metrica | 0.656 | 0.712 | 0.731 | 0.858 | 0.915 | +7.2 |

Label-free checks with the dead-ball play (both changes):

| | shift | D% learned − v09 | agreement with v09 |
|---|---|---|---|
| provider in training, DFL | 0.1 .. 0.6 | −7.9 .. −0.3 | 85.8 .. 90.9% |
| provider in training, Metrica | 0.1 .. 0.2 | −5.7 .. −2.6 | 92.5 .. 93.1% |
| provider never seen, Metrica | 0.1 .. 0.3 | −5.8 .. −2.9 | 92.9 .. 93.8% |
| provider never seen, DFL | 2.7 .. 5.3 | −28.6 .. −18.9 | 70.7 .. 79.3% |
| **BAR-ATH** | **0.4** | **−5.5** | **92.4%** |

### Reading

- **The dead-ball play does what it was built to do on BAR-ATH.** By the label-free checks, BAR-ATH
  now looks like an in-domain match: shift 1.2 → 0.3–0.4, closer agreement with v09. It does not
  help the failure it was also expected to help, DFL learned from Metrica (lopo 0.807 → 0.805).
- **It costs 0.022 on the lomo bench, which is the blocking criterion.** Part of that drop comes
  from a harder test set, not a worse model. With the play switched on, the degraded external
  matches are also the test data, and they now contain moving dead balls. model_v09, which learns
  nothing, falls on them too (lomo N25 0.454 → 0.439). So about 0.015 of the 0.022 is the test set,
  and the comparison is not like for like. The criterion is defined on the bench as it stands,
  though, and on that bench the change fails.
- **The domain features hardly matter.** Among the 10 features that best separate the domains,
  only 1–2 are ball kinematics (`dv`, plus `since_kh` with the dead-ball play). The domain is given
  away mainly by ball position (`bx_edge`, `by_edge`, `bx_att`) and team shape (`sx`, `sy`, `dsx`).
  Dropping those 1–2 features changes nothing outside noise (lopo +0.003, lomo −0.002, shift 1.2).

## 1. Dead-ball play (label-free calibration)

Applies to each frozen-ball stretch of DFL of at least 2 s (`CARRY_MIN`). The definition of
"frozen" does not change: same position in consecutive frames, never a label. For the last part of
the stretch, a share of 2·0.7·U, at most 40 s, the ball:

- is carried by a player;
- is passed between players within 20 m of the restart taker, as a Poisson process at
  0.5 × (mean player speed / 1 m/s)² passes per second, at 9 m/s;
- goes back to the previous passer half of the time;
- ends with the taker. v11's carry is the case with 0 passes and no cap.

The play draws from its own RNG stream, so the ball gaps are identical with and without it.

`--calibrate` searched 144 settings on the 7 DFL matches. Loss = mean relative error of median
ball speed and velocity-change rate in bins 0–3 against BAR-ATH. A setting was admissible if
P(ball missing) and P(player < 2 m | visible) stay within 0.05 of BAR-ATH in every bin, the same
worst deviation v11 has. Loss went from about 0.40 (v11) to 0.107.

| play-intensity bin | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| **median ball speed, m/s** | | | | | | | | |
| BAR-ATH | 1.32 | 1.81 | 2.62 | 4.22 | 5.04 | 5.62 | 6.31 | 7.01 |
| DFL, v11 | 1.13 | 1.42 | 1.88 | 2.64 | 4.08 | 5.32 | 6.17 | 6.81 |
| DFL, v12 | 1.41 | 1.91 | 2.38 | 3.23 | 4.78 | 5.66 | 6.37 | 6.88 |
| **P(velocity change > 4 m/s \| visible)** | | | | | | | | |
| BAR-ATH | 0.12 | 0.20 | 0.23 | 0.28 | 0.34 | 0.35 | 0.36 | 0.32 |
| DFL, v11 | 0.02 | 0.06 | 0.13 | 0.21 | 0.27 | 0.28 | 0.28 | 0.25 |
| DFL, v12 | 0.09 | 0.17 | 0.23 | 0.27 | 0.31 | 0.30 | 0.29 | 0.26 |
| **P(ball missing)** | | | | | | | | |
| BAR-ATH | 0.88 | 0.71 | 0.56 | 0.35 | 0.18 | 0.13 | 0.10 | 0.10 |
| DFL, v11 | 0.90 | 0.72 | 0.59 | 0.36 | 0.23 | 0.15 | 0.10 | 0.12 |
| DFL, v12 | 0.86 | 0.72 | 0.59 | 0.37 | 0.20 | 0.16 | 0.11 | 0.12 |
| **P(player < 2 m \| visible)** | | | | | | | | |
| BAR-ATH | 0.53 | 0.50 | 0.49 | 0.51 | 0.52 | 0.54 | 0.51 | 0.51 |
| DFL, v11 | 0.53 | 0.51 | 0.48 | 0.51 | 0.54 | 0.54 | 0.53 | 0.56 |
| DFL, v12 | 0.56 | 0.49 | 0.52 | 0.55 | 0.56 | 0.55 | 0.53 | 0.56 |

Metrica is not touched: it has no frozen stretches, and its rows do not move.

What did not work in the calibration:

| attempt | result |
|---|---|
| Fixed pass rate (no speed scaling, no cap) | Only one admissible setting. Bins 2–3 stayed at 2.1 / 3.0 m/s |
| Higher active share without a cap (frac 0.7–0.9) | Change rates matched, but P(near) in bin 0 was +0.11 to +0.14: the ball spent too long at someone's feet in long stoppages |
| Matching bin 3 speed (4.22) | Not reached (best 3.23). 57% of DFL's bin 3 is live play, which this change does not touch |

P(near) is not spoiled by the worst-bin measure (0.05, same as v11), but it gets worse in bins 2–3:
+0.03/+0.04, against −0.01/0.00 with v11.

## 2. Domain classifier

Built from frames of BAR-ATH, all graded frames except the pre-kickoff, every 3rd one, and an equal
number spread evenly over the 10 degraded external matches. A LightGBM classifier is trained on the
first halves and tested on the second halves, then the other way round. It sees all 126 v11
features. Ranking is by gain on all frames. Domain membership is the only target: no possession
label is involved.

| training data | AUC, all features | AUC, without the dropped ones | dropped (ball kinematics in the top 10) |
|---|---|---|---|
| v11 play | 0.812 | 0.799 | `dv` |
| v12 play | 0.819 | 0.805 | `dv`, `since_kh` |

Top features (v12 play, share of gain): `bx_edge` 0.085, `by_edge` 0.075, `bx_att` 0.074, `sy_a`
0.051, `dsx_f75` 0.049, `sx_h` 0.043, `dv` 0.043, `sx_a` 0.040, `dsx_p75` 0.035, `since_kh` 0.027.
The remaining ball-kinematics features each carry about 0.02 (`kbs_f250`, `since_ka`, `until_ka`,
`until_kh`).

Under the brief, only ball kinematics may be removed. The two signals that give the domain away
most strongly fall outside it:

- **Ball position near the edges.** This points to a difference in pitch geometry or coordinates
  between BAR-ATH and the external providers, which is worth checking before any further feature
  work.
- **Team spread,** which only the team model's features carry.

`DOMAIN_TOP = 10` was set before the bench and was not tuned.

## 3. Set-piece evidence from Smart Tagging (optional): not done

The plan was to validate it on Metrica by simulating SET PIECES tags from Metrica's events. That
test would be circular: `bench_external/fetch.py` builds Metrica's alive/dead labels from those same
events, and a `SET PIECE` event is by definition the dead→alive transition. Any "dead ball before
the tag" evidence would score itself. Also, `Sample_Game_3`'s events JSON is deleted after
conversion, so only 2 of the 3 matches have events locally. A meaningful validation needs set-piece
times from a source that is independent of the labels, such as DFL's own event feed against its
`BallStatus`.

## Rules

- No BAR-ATH label was read. The calibration and the domain classifier use BAR-ATH tracking
  statistics and domain membership only, printed as aggregates.
- The public 5-minute window was not used at any point. No v1.2 variant was adopted, so there is no
  public-window figure to report.
- Nothing was committed. `model_v11.py` and `evaluation/results/model_v11.csv` are unchanged.

## Possible next steps

- **Compare on equal test data.** Score both training sets (v11 and v12 play) against both degraded
  test sets. The current lomo criterion mixes a model change with a test-set change.
- **Look at the pitch geometry and edge distances** behind `bx_edge`/`by_edge`, the strongest
  domain signal, before more ball-kinematics work.
