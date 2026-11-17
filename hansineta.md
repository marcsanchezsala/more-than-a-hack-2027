# Technical document — ball possession from automatic tracking

**More than a Hack 2027, Phase 1** · Team: hansineta · Submission: `model_v11.py` → `hansineta.csv`

## 1. Approach

The labels come from two gradient-boosted classifiers (LightGBM), one for alive/dead ball and one
for home/away possession. Their per-frame log-probabilities are decoded by a 3-state Viterbi (H, A, D)
over each half. Both classifiers are trained **only on 10 open external matches**. Those matches are
first degraded so that their tracking looks like the BAR-ATH tracking: the same pattern of ball
loss, noise and drift. Nothing about BAR-ATH is labelled or learned from labels.

**Why this design.** The task is a sequence-labelling problem with an awkward test set. There is one
match, the ball is invisible on 41% of the graded frames, and there are no labels except a
5-minute window that the rules reserve for pipeline checks. That rules out learning from the game
itself. It leaves two options: hand-written physics rules, or a model trained on other matches.

- **Hand-written rules (v0.1–v0.9).** Our earlier versions were physics rules inside an HMM. We
  measured them on the external matches and found they mark dead ball 6–12 points too often on every
  match. They also miss many short possessions. Every rule was set by looking at the one game, which
  makes them hard to trust.
- **Learned model (v1.0–v1.1).** A model trained on labelled external matches does much better on
  them (table below). The risk is that it does not transfer to BAR-ATH. Most of the work in v1.1 is
  about measuring and reducing that risk without labels.

The Viterbi is used because the metric (`spellF1`) rewards whole possessions, not single frames.
A change of team costs 20 nats and entering or leaving dead ball costs 40. So a label only changes
when the evidence for the change lasts, and a single noisy frame cannot split a possession in two.

**External benchmark.** Each match is labelled by models trained without it (leave-one-match-out).
The scores are spellF1 against the raw provider labels, and against the same labels with possessions
shorter than 1 s (N25) or 2 s (N50) absorbed:

| | raw | N25 | N50 | macroF1 | alive/dead F1 |
|---|---|---|---|---|---|
| physics HMM (model_v09) | 0.422 | 0.454 | 0.480 | 0.793 | 0.892 |
| v09 dead-ball mask + learned H/A (v1.0) | 0.558 | 0.600 | 0.614 | 0.835 | 0.892 |
| **learned alive/dead + learned H/A (v1.1, submitted)** | **0.640** | **0.691** | **0.715** | **0.895** | **0.955** |

## 2. Assets

| asset | used | how |
|---|---|---|
| ATD tracking (tactical camera) | yes | ball and player positions, team of each track. All features come from it |
| Smart Tagging | yes, one tag type | the "Kick Off (Start)" tag of each half. Before it the ball is dead |
| tactical / broadcast video | no | the tracking already carries what we need |
| homography | no | positions are already in pitch coordinates |
| broadcast-camera tracking | no | the tactical camera covers the whole pitch |
| other Smart Tagging codes | tried, dropped | set-piece tags as dead-ball evidence. It could not be validated without circularity (§6) |

## 3. Training data

No label of the provided game is used to fit, tune or select anything. All labelled data is open:

| dataset | matches | graded time | labels | who produced them |
|---|---|---|---|---|
| DFL / IDSSE Bundesliga data (Bassek et al. 2025, CC BY 4.0), Hugging Face `pysport/idsse-data`, revision `a715a38` | 7 | ~11 h | per frame: `BallStatus` (alive/dead) and `BallPossession` (home/away), shipped with the position data | the data provider (DFL / Sportec Solutions), as published in the dataset. We add nothing |
| Metrica Sports sample data, github.com/metrica-sports/sample-data, commit `e706dd5` | 3 | ~5 h | per frame, rebuilt by `bench_external/fetch.py` from Metrica's event data: the team of the last on-ball event (pass, shot, recovery, set piece…), dead from a ball out, a goal, a shot off target or a foul until the next event | events tagged by Metrica Sports. The conversion to frames is ours and is deterministic |

We labelled nothing by hand, on any game. `python bench_external/fetch.py` downloads both datasets
at the pinned revisions and converts them.

**Degradation (`model_v11.load_external`).** Each external match is turned into something that
looks like the BAR-ATH tracking. Every step is driven by the tracking itself, never by a label:

1. **Players.** Dropouts and position jitter (`bench_external/degrade.py`).
2. **Ball noise and drift.** The ball gets slow-varying noise. Where the source freezes the ball
   (the same position frame after frame), it also gets a slow drift.
3. **Dead-ball carry.** DFL leaves the ball at the restart spot for the whole stoppage. Now, for part
   of each frozen stretch, the player nearest the restart carries it. This brings P(player within
   2 m | ball visible) with play stopped from 0.21 to 0.53. BAR-ATH: 0.53.
4. **Ball loss.** The ball is removed with the probability that makes each match's ball-missing
   curve match BAR-ATH's, bin by bin of play intensity (mean player speed). BAR-ATH: 0.88 missing
   with play stopped, 0.10 in intense play. The removal is correlated over 300 frames, so the ball
   is lost in few long gaps, as in BAR-ATH. 85–88% of the missing time is in gaps of 10 s or more.

**Features (126).** All come from positions only:

- distances from the ball to the nearest and second-nearest player of each team, and their trends
  over 0.2 to 10 s windows, looking back and ahead;
- touches, seen as changes of ball velocity next to a player;
- ball speed, aerial-ball detection, the ball's distance to the lines;
- player speeds and the shape of each team;
- how long the ball has been missing.

The team model sees every frame twice, once with the teams swapped, so it cannot learn a bias for
either side. The tracked-player count is deliberately left out. It is the signal that most separates
BAR-ATH from the external data, and it carries no information about possession.

## 4. Possession rules

The line is wherever the providers' labels put it. We write no rule for loose balls or challenges.
On top of that:

- **Short possessions.** Possessions shorter than 1 s are absorbed into their neighbours in the
  training labels (`bench_external/convention.py`, N=25). A deflection or one contested touch does
  not change the team in possession. A loose ball stays with the team that had it until the other
  team gains control.
- **Dead ball.** The ball is dead from the stoppage until the restart is taken.
- **Before kick-off.** Before each half's tagged kick-off the ball is dead, by the laws of the game.
  This is the only hand-written rule. It uses no frame constant: the kick-off frame comes from Smart
  Tagging.
- **Changes of state.** The Viterbi costs (20 / 40 / 40 nats) were chosen on the external matches.
  The cost grid is flat around them.
- **X.** It is written only outside the graded halves (`games.half_video_blocks`).

## 5. Does it transfer to BAR-ATH? (label-free)

The learned alive/dead model has one known failure. Trained on the 3 Metrica matches and tested on
DFL (a provider it never saw), it predicts about 20 points too little dead ball. So before shipping
it we checked, without labels, whether BAR-ATH looks like the matches it handles well or like that
failure. `python model_v11.py --bench` prints three measures for every external match under both
regimes, and for BAR-ATH:

- **shift:** % of frames where an alive model that sees only player speeds is sure the ball is dead
  and the full model says alive;
- **D% vs v09:** how much less dead ball the model predicts than the physics model v09. v09 needs no
  training data and its bias is known;
- **agreement with v09:** % of frames where both agree on alive/dead.

| | shift | D% learned − v09 | agreement with v09 |
|---|---|---|---|
| provider in training (10 matches) | 0.1 .. 0.6 | −9.5 .. −2.7 | 87.6 .. 92.7% |
| provider never seen, DFL, where the model fails | 3.0 .. 6.1 | −29.1 .. −22.9 | 69.8 .. 75.8% |
| **BAR-ATH** | **1.2** | **−6.0** | **91.2%** |

BAR-ATH behaves like the matches the model handles well. The shift value is the one caveat: above
every in-domain match, but well below the failure range.

## 6. What we tried and dropped

| | result |
|---|---|
| Physics HMM with hand-set rules (v0.1–v0.9) | over-predicts dead ball by 6–12 points on every external match. Replaced |
| v09 dead-ball mask + learned team model (v1.0) | 0.600 vs 0.691 N25. Replaced |
| Soft fusion of the learned alive/dead with v09 | no gain (0.691 → 0.687) |
| Co-training with pseudo-labels on an unseen provider | −0.018 in domain |
| Players-only alive/dead model | more robust across providers, worse in domain |
| Markov-chain ball-gap model | right gap counts, but distorts the per-intensity missing curve |
| Simulated dead-ball play (passes and dribbles during stoppages) + domain-classifier feature selection (v1.2) | BAR-ATH moves fully in domain (shift 0.3), but in-domain spellF1 drops (0.691 → 0.667), including on the Metrica matches, which it does not touch. Not adopted (`docs/RESULTS_v1.2.md`) |
| Set-piece tags as dead-ball evidence | not built. Validating it on Metrica would be circular: Metrica's labels come from the same events |

## 7. Limitations

- **Transfer.** Everything depends on the degradation making the external data look like BAR-ATH.
  Where it does not, the model fails badly (DFL from Metrica only: −20 points of dead ball). The
  checks in §5 say BAR-ATH is not that case, but they are indirect.
- **Moving dead ball.** A dead ball that is visible and moving (a quick throw-in, the ball passed
  back to the taker) is sometimes called alive. The training data has too few such stoppages.
  Simulating them (v1.2) fixed the domain gap but cost accuracy elsewhere.
- **Short possessions and boundaries.** Short possessions remain the weak point, and transitions are
  often placed 0.5–1 s off. The organisers' convention for very short possessions is unknown: the
  same predictions score 0.640–0.715 depending on it.
- **Training data.** 10 matches from two providers. A third provider would be the most direct
  improvement.
- **Inherited code.** The ball smoother and the aerial-ball detector in model_v11 come from model_v09,
  with the same constants. Those were set during v0.x development, and part of that
  development looked at the public window. In v1.1 they only produce inputs to models trained on
  external data. They decide no label.

**Public 5-minute window.** Reported as a pipeline check only; nothing was chosen on it.

- **v1.1:** spellF1 0.612, macroF1 0.848, 17 possessions predicted for 17 true. The dead ball is
  under-predicted (recall 0.64, precision 0.996), mostly in one moving-ball stoppage.
- **How we used it.** During development the window did one thing. A first v1.1 build missed two
  of its four stoppages. That prompted the label-free comparison of ball-gap lengths behind
  degradation step 4. The fix was sized on label-free statistics and on the external benchmark.

## 8. Reproduction

Python 3.12, Windows x86-64 (the submitted file was generated there; another OS may differ), packages pinned in `requirements-model.txt`. Full commands are in the
[README](README.md#reproduce-the-submission).

```bash
pip install -r requirements-model.txt
export MTAH_DATA_DIR=/path/to/match/data
python bench_external/fetch.py                   # external matches, pinned revisions
python model_v11.py --out hansineta.csv        # ~80 s
python evaluation/validate_submission.py hansineta.csv
```

Training is deterministic: fixed seeds, and LightGBM with `deterministic=True`, `force_row_wise=True`
and 4 threads. Two runs give the same file byte for byte. On Windows, two consecutive runs gave this hash. The earlier Linux build (hash dc78d429...67d4) was also reproduced from scratch (fresh environment,
fresh download). SHA-256 of the file:
`c0488981e5c70ca8b4204a268a266937cff8c2ce645c69c024f3b049aef60043`.

The development history, with the numbers of every version, is in `docs/RESULTS_v*.md`.

**Reference.** Bassek, M., Rein, R., Weber, A., Memmert, D. (2025). *An integrated dataset of
spatiotemporal and event data in elite soccer.* Scientific Data. doi:10.6084/m9.figshare.28196177.
