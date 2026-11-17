# model_v11 (v1.1)

Two LightGBM classifiers (alive/dead, home/away) trained only on 10 open external matches that are
degraded to look like BAR-ATH tracking, feeding a 3-state Viterbi. Same architecture as v1.0; the
difference is that v1.1 labels BAR-ATH on its own. v1.0 still needed model_v09's dead-ball mask
because its learned alive/dead model did not transfer; that dependency is gone.

```bash
python bench_external/fetch.py   # once: download + cache the 10 external matches
python model_v11.py              # train on all 10, label BAR-ATH -> evaluation/results/model_v11.csv (~80 s)
python model_v11.py --bench      # external evidence + label-free transfer checks (~7 min)
python model_v11.py --domain     # label-free comparison of BAR-ATH and the degraded external data
```

Environment: `requirements-model.txt` (Python 3.12). Two consecutive runs give the same file byte
for byte; `evaluation/validate_submission.py` passes. Full match: 292 possessions, mean 12.0 s,
44.9% of graded frames dead.

## External benchmark

`--bench`. spellF1 of each external match, predicted by models that never saw that match (lomo) or
never saw its provider (lopo), against the raw provider labels and against the same labels with
possessions shorter than 1 s (N25) or 2 s (N50) absorbed. All variants see the same degraded data.

| leave one match out, all 10 | raw | N25 | N50 | macroF1 | alive/dead F1 | D% predicted - true |
|---|---|---|---|---|---|---|
| model_v09 | 0.422 | 0.454 | 0.480 | 0.793 | 0.892 | +7.8 |
| v09 dead-ball mask + v11 H/A model (the v1.0 recipe) | 0.558 | 0.600 | 0.614 | 0.835 | 0.892 | +7.8 |
| **v11, fully learned** | **0.640** | **0.691** | **0.715** | **0.895** | **0.955** | +2.2 |

By provider (lomo, v11): DFL 0.614 / 0.662 / 0.685, Metrica 0.701 / 0.760 / 0.784.

| leave one provider out | raw | N25 | N50 | alive/dead F1 | D% predicted - true |
|---|---|---|---|---|---|
| v11, tested on DFL, trained on the 3 Metrica matches | 0.453 | 0.487 | 0.497 | 0.807 | -19.7 |
| v11, tested on Metrica, trained on the 7 DFL matches | 0.675 | 0.730 | 0.744 | 0.921 | +5.1 |
| v09 dead-ball mask + v11 H/A, tested on DFL | 0.521 | 0.559 | 0.571 | 0.896 | +6.1 |

The learned alive/dead model is the best option when the target looks like the training data and
the worst when it does not (DFL from three Metrica matches). So the decision to ship it rests on
whether BAR-ATH looks like the training data, which is checked without labels below.

## Is BAR-ATH in domain? (label-free)

Printed at the end of `--bench`. Three statistics that need no ground truth, computed on every
external match in both regimes and on BAR-ATH:

| | shift | D% learned - v09 | alive/dead agreement with v09 |
|---|---|---|---|
| provider in training, DFL (7 matches) | 0.1 .. 0.6 | -9.5 .. -2.7 | 87.6 .. 92.7% |
| provider in training, Metrica (3) | 0.3 .. 0.5 | -6.2 .. -3.8 | 91.8 .. 92.2% |
| provider never seen, Metrica (3) | 0.4 .. 0.7 | -7.8 .. -5.1 | 91.3 .. 93.2% |
| provider never seen, DFL (7), where the model fails | 3.0 .. 6.1 | -29.1 .. -22.9 | 69.8 .. 75.8% |
| **BAR-ATH** | **1.2** | **-6.0** | **91.2%** |

*shift* = % of frames where an alive model that only sees player speeds is sure the ball is dead
and the full model says alive. The other two columns compare the learned labels with model_v09,
a physics model that does not depend on the training data and whose bias is known (+6 to +12 points
of dead ball on every external match).

BAR-ATH disagrees with v09 exactly as the matches the model handles well do, and nothing like the
failure case. With v1.0's training data the same gap was -10.7 (40.2% vs 50.9%). The shift value is
the one caveat: 1.2 is above every in-domain match and below the failure range.

## What changed from v1.0

Training data (`--domain` prints the per-intensity-bin tables; everything is keyed on the tracking,
never on a label):

1. **Dead-ball carry.** DFL parks the ball at the restart spot for the whole stoppage. In BAR-ATH a
   visible ball has a player within 2 m about half the time at every play intensity (0.53 with
   play stopped); degraded DFL had 0.21. Now the player who will restart carries the ball for part
   of each frozen stretch: 0.53.
2. **Long ball gaps.** BAR-ATH loses the ball in few long gaps (142 per 100 min, median 10.7 s, 85%
   of the missing time in gaps of 10 s or more). v1.0's dropout produced 326 gaps with median 3.5 s
   (70%). The dropout field is now correlated over 300 frames instead of 50: 88%, with the
   per-intensity missing curve still matched (BAR-ATH 0.88 -> 0.10, DFL 0.90 -> 0.12).
3. **Tracked-player count removed** from the features. It is the signal that most separates
   BAR-ATH from the external matches (17.8 tracked players with play stopped vs ~21) and it carries
   no information in them.

H/A model. An oracle decomposition on v1.0 (lomo, N50: 0.660 as is, 0.705 with perfect alive/dead,
0.914 with perfect H/A) showed the team decision is where the error is, and short possessions in
particular (recall 0.14 for 0-2 s, 0.34 for 2-4 s, 0.63 for 4-8 s). Added: 0.2 s and 0.5 s windows;
touches seen as changes of ball velocity next to a player; ball velocity along the attacking
direction; players of each team within 3 and 6 m of the ball, second-nearest player, team spread.
With perfect alive/dead, spellF1 went from 0.601 / 0.655 / 0.714 to 0.685 / 0.740 / 0.770
(raw / N25 / N50). The velocity change is measured over +-5 frames: at +-2 it fired 3-5 times less
often on BAR-ATH than on the external data, at +-5 the rates agree.

Labels. Models are trained on the N25 convention instead of N50: better against raw and N25, equal
against N50. Viterbi costs unchanged (20 / 40 / 40); the grid is flat.

Rule. Before the tagged kick-off of each half the ball is dead (laws of the game; the kick-off
frame comes from the automatic Smart Tagging, as in model_v07-v09).

(The oracle decomposition and the feature ablation were one-off measurements; they are not part of
`--bench`.)

## Tried and dropped

| | result |
|---|---|
| Soft fusion: learned alive/dead log-odds + v09's call weighted by its measured likelihood ratios | lomo N25 0.691 -> 0.687 (w=0.5) -> 0.679 (w=1); lopo +0.003. No gain, dropped |
| Logit average of the full and the players-only alive models | does not rescue lopo |
| Co-training with pseudo-labels on the unseen provider | lomo -0.018, lopo +0.035 .. +0.053; not adopted |
| Players-only alive model as the labeller | more robust across providers, worse in domain |
| Team-shape features in the players-only model | lopo alive/dead 0.869 vs 0.905 without |
| Markov-chain gap model (two versions) | right gap counts, flattened the per-intensity missing curve |
| Tuning the Viterbi costs | flat |

## Technical-document items

**Assets used.** ATD tracking (ball and player positions, team of each track) and the Smart Tagging
kick-off tags. Video, homography and broadcast feed are not used.

**Training data.** No label of BAR-ATH is read to fit or choose anything. The labelled data is:

| | matches | graded time | labels |
|---|---|---|---|
| DFL / IDSSE (Bassek et al. 2025, CC BY 4.0, Hugging Face `pysport/idsse-data`) | 7 | ~11 h | official DFL `BallStatus` (alive/dead) and `BallPossession` (home/away), per frame |
| Metrica Sports sample data (github.com/metrica-sports/sample-data) | 3 | ~5 h | rebuilt per frame from Metrica's hand-tagged event data (`bench_external/fetch.py`) |

**Possession rules.** Learned from those labels, with possessions shorter than 1 s absorbed into
their neighbours (`bench_external/convention.py`, N=25), so a deflection or a single contested touch
does not change the team in possession. There is no hand-written rule for loose balls or
challenges: the line is wherever the providers' labels put it. The ball is dead from the stoppage
until the restart is taken, and before each half's kick-off.
A change of team costs 20 nats and entering or leaving dead ball 40 in the Viterbi, so a label only
changes on sustained evidence.

**Limitations.**
- The model depends on the degradation making external data look like BAR-ATH. Where that fails it
  fails badly (DFL from Metrica: -19.7 points of dead ball). The label-free checks say BAR-ATH is
  not that case, but they are indirect.
- Short possessions are still the weak point, and the organisers' convention for them is unknown:
  the same predictions score 0.640 to 0.715 depending on it.
- 10 training matches from two providers. More providers would be the most direct improvement.
- Feature extraction reuses model_v09 code (Kalman ball smoother, aerial-ball detector) whose
  constants were set during v0.x development, part of which looked at the public window. They are
  inputs to models trained on external data only, not decisions.

**Public 5-minute window** (17 possessions, 4 stoppages; sanity check only, nothing was chosen on
it): spellF1 0.612, macroF1 0.848, 17 predicted possessions for 17 true, 3 of 4 stoppages found.
model_v10: 0.594, 0.926, 14 for 17; its dead-ball mask is model_v09's, which was developed on that
window. During development the window did one thing: a first v1.1 build missed two of the four
stoppages, which prompted the label-free comparison of ball-gap lengths (item 2 above). The fix was
chosen and sized on those label-free statistics and on the external benchmark.
