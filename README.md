# Who has the ball? — team hansineta, More than a Hack 2027 (Phase 1)

**Team hansineta:** Oscar Lu, Jaume Majó, Marc Sánchez, David Vàzquez.

The hansineta entry to [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027)
(Metrica Sports, FC Barcelona, Mobile World Capital Barcelona). For every frame of one LaLiga match
(BAR-ATH, 199,319 frames) the task is to say who has the ball: `H` home, `A` away, `D` dead ball,
`X` not graded. The challenge, data and scoring are described in the organisers' documents, kept in
this repository: [CHALLENGE.md](CHALLENGE.md), [DATA.md](DATA.md), [EVALUATION.md](EVALUATION.md),
[SUBMISSION.md](SUBMISSION.md), [RULES.md](RULES.md), [FAQ.md](FAQ.md).

**The submission is produced by [`model_v11.py`](model_v11.py).** How it works, what it was trained
on and where it fails: **[hansineta.md](hansineta.md)**.

## In short

Two LightGBM classifiers estimate, frame by frame, whether the ball is alive and which team has it.
A 3-state Viterbi turns those estimates into whole possessions. Both classifiers are trained only
on **10 open external matches**: 7 DFL Bundesliga, 3 Metrica Sports samples. Before training, those
matches are degraded to look like the BAR-ATH tracking: the ball goes missing in long gaps, mostly
while play is stopped, and drifts while it is dead. **No label of the provided game is used to train,
tune or choose anything.**

**Results on the external benchmark (10 open matches, not BAR-ATH).** Each external match is
labelled by models trained on the other 9 (leave one match out), on the same degraded data.
Mean over the 10 matches, against the providers' labels with possessions shorter than 1 s absorbed:

| | spellF1 | macroF1 | alive/dead F1 |
|---|---|---|---|
| hand-written physics HMM (v0.9) | 0.454 | 0.793 | 0.892 |
| v0.9 dead-ball mask + v1.1 team model (the v1.0 recipe) | 0.600 | 0.835 | 0.892 |
| **fully learned (v1.1, submitted)** | **0.691** | **0.895** | **0.955** |

**Results on the public 5-minute window of BAR-ATH** (7,500 graded frames, 17 possessions). A
pipeline check, not used to train or tune; how it was used during development is in
[hansineta.md §7](hansineta.md#7-limitations). v0.9 was developed while looking at
this window, so its figures here are optimistic:

| | spellF1 | macroF1 | alive/dead F1 | possessions (pred / true) |
|---|---|---|---|---|
| hand-written physics HMM (v0.9) | 0.607 | 0.918 | 0.988 | 14 / 17 |
| v0.9 dead-ball mask + v1.0 team model (v1.0) | 0.594 | 0.926 | 0.988 | 14 / 17 |
| **fully learned (v1.1, submitted)** | **0.612** | **0.848** | **0.958** | **17 / 17** |

## Reproduce the submission

**Requirements.** Python 3.12 on Windows x86-64 (the submitted file was generated there; a run on another OS may differ); packages pinned in
[`requirements-model.txt`](requirements-model.txt). About 300 MB of disk for the external data cache
(more during the download). Tested on a 16-core, 32 GB machine: about 2.5 min to download the
external data and 70 s to build.

```bash
# 1. environment (pip works too: python3.12 -m venv .venv && .venv/bin/pip install -r requirements-model.txt)
uv venv -p 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-model.txt
source .venv/bin/activate

# 2. the match data (not in this repository): unpack it into ./data or point at it
export MTAH_DATA_DIR=/path/to/mtah2027

# 3. the 10 external matches, downloaded at pinned revisions and cached in external_data/cache/
python bench_external/fetch.py

# 4. train and label BAR-ATH (~80 s)
python model_v11.py --out hansineta.csv

# 5. check
python evaluation/validate_submission.py hansineta.csv
sha256sum hansineta.csv
# c0488981e5c70ca8b4204a268a266937cff8c2ce645c69c024f3b049aef60043
```

The pipeline is deterministic: fixed seeds; LightGBM with `deterministic=True`, `force_row_wise=True`
and a fixed 4 threads; external data pinned to exact revisions. Two consecutive Windows runs gave the hash above. On Linux (earlier build, hash dc78d429...67d4) we also checked it from scratch, with a
fresh environment and a fresh download, and it reproduces the file byte for byte.

Run without `--out`, the file goes to `evaluation/results/model_v11.csv`. Then
`python evaluation/run_evaluation.py` scores it against the public window, next to the baselines.

### Evidence behind the design

```bash
python model_v11.py --bench    # external benchmark, label-free transfer checks (~7 min)
python model_v11.py --domain   # BAR-ATH vs the degraded external data, label-free
```

`--bench` scores v0.9, v1.0 and v1.1 on the external matches in two ways:

- leaving one match out;
- leaving one whole provider out, which is the hard case.

It also prints three label-free measures. They show whether BAR-ATH behaves like the matches the
model handles well or like the case where it fails.

## Repository layout

```
model_v11.py               the submission: degradation, features, training, decoding (self-contained)
model_v10.py               v1.0: v0.9 dead-ball mask + learned team model (history)
model_v09.py               v0.9 physics HMM (--bench baseline)
bench_external/
  fetch.py                 downloads and converts the 10 external matches (pinned revisions)
  degrade.py               player dropouts and jitter, ball noise
  convention.py            absorbs possessions shorter than N frames in the external labels
  run.py                   per-frame physical features shared by every model
hansineta.md      the technical document
docs/RESULTS_v*.md         numbers and decisions of every version, v0.1 to v1.2
requirements-model.txt     pinned environment for the model

games.py, evaluation/, starter/     from the organisers' starter kit (scorer, validator, loaders)
CHALLENGE.md … SUBMISSION.md        the organisers' documents
```

Not part of the pipeline, kept as history:

- `model_v07.py`, `model_v08.py`, `archive/`: earlier versions. v0.4–v0.6 contain a frame-tied
  constant and must not be used.
- `model_v12.py`: a later experiment that was not adopted (`docs/RESULTS_v1.2.md`).
- `audit_full_match.py`, `bench_external/{calibrate_physics,diagnose,run_conventions}.py`: tools
  used for v0.9.
- `docs/EDA_REPORT.md`: earlier analysis
  and documents.

## Licence

Code and documentation: MIT, see [LICENSE](LICENSE). The starter kit is © Metrica Sports. The match
data is not covered by it. It is licensed by the organisers for this competition only and is not
included. The external datasets keep their own licences:

- DFL / IDSSE data: CC BY 4.0. Bassek, M., Rein, R., Weber, A., Memmert, D. (2025). *An integrated
  dataset of spatiotemporal and event data in elite soccer.* Scientific Data.
- Metrica Sports sample data: see
  [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data).
