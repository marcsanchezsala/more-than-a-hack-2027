# model_v10 (v1.0)

Two LightGBM classifiers trained only on the 10 open external matches (7 DFL/IDSSE + 3 Metrica),
degraded to look like BAR-ATH tracking: an alive/dead model and a home/away model (trained with
mirrored teams, so it is symmetric). Their log-probabilities feed a 3-state Viterbi.

```bash
python model_v10.py --bench           # leave-one-match-out on the 10 external matches, v09 vs v10
python model_v10.py                   # submission (hybrid) -> evaluation/results/model_v10.csv
python model_v10.py --dead learned    # fully learned variant, for comparison
```

## External benchmark (leave-one-match-out, convention N=50, BAR-ATH-matched degradation)

Each match is scored by models trained on the other 9. Same degraded data and labels for all three.

| | spellF1 | macroF1 |
|---|---|---|
| model_v09 | 0.481 | 0.803 |
| v10, fully learned | **0.667** | **0.896** |
| v10 hybrid (v09 alive/dead + learned H/A) | 0.572 | 0.835 |

The learned team model beats v09 on all 10 matches. Viterbi costs: the grid over
p_switch x p_enter_d x p_exit_d is flat once p_switch=20; the centre (20, 40, 40) is used.

## The degradation, and two leaks it had

`bench_external/degrade.py` (used by v09's calibration) is not used for the ball any more. Measured
label-free on the tracking itself, binned by play intensity (mean player speed over 3 s):

| | P(ball missing), stopped -> intense play | ball speed when visible, stopped play |
|---|---|---|
| BAR-ATH | 0.88 -> 0.10 | 1.3 m/s |
| Metrica samples (same vendor) | 0.81 -> 0.02 | 0.1 m/s |
| DFL | 0.00 everywhere | **0.00 m/s** (TRACAB freezes the ball) |

1. **Ball dropouts.** degrade.py removes the ball uniformly at random (and adds 41% on top of
   Metrica's own dropouts). In BAR-ATH the ball is lost mostly while play is stopped. v10 removes
   the ball with the probability that makes each match's curve match BAR-ATH's, per intensity bin.
2. **Labels in the features.** degrade.py adds ball drift only on frames labelled D. v10 instead adds
   a 1 m slow drift where the source froze the ball (identical position frame to frame), which is
   detectable without labels and brings stopped-play ball speed to BAR-ATH's level.

## Why the submission is still the hybrid

The learned alive/dead model does not fully transfer to BAR-ATH: on the full match it calls 40.2% of
graded frames dead against 50.9% for v09 (87.6% agreement). The disagreements (16 min, median 3.9 s
segments) have the ball visible, slow players and moderate ball speed; they are not explained by
Smart Tagging set pieces. So `model_v10.csv` keeps v09's dead-ball mask exactly and only re-decides
H/A inside live play. Full match: 260 possessions, mean 12.0 s, 50.9% D. Reproduces byte for byte.

## Next

Close the remaining alive/dead domain gap (what BAR-ATH stopped play with a visible ball looks like)
so the fully learned variant can be submitted.
