# Data

Data files needed to reproduce **Experiment 2 (Upworthy classifier threshold tuning)**.
Both files are loaded by `src/black_box_upworthy.py`; the runner
(`jobs/run_bo_upworthy.py`) and the analysis notebook
(`notebooks/02_upworthy_bo.ipynb`) work out of the box without any path flags
when these files are present.

## Files

### `upworthy/categories_raw.csv` (~6 MB)

BART zero-shot category predictions for every Upworthy headline. Produced once,
upstream, by running `facebook/bart-large-mnli` against a fixed candidate label
set. One row per headline; columns include the predicted top label and its
confidence score.

The Bayesian Optimisation loop **does not run BART** — that inference step is
already done. BO only tunes the *post-processing* thresholds (confidence cutoff
and minimum category size) applied to these saved predictions.

### `upworthy/confirmatory_clean.csv` (~5 MB)

Headline-level click-through rates from the Upworthy Research Archive
(confirmatory split, post-cleaning). Trimmed to the two columns the BO
objective actually reads:

| Column | Meaning |
|---|---|
| `headline` | The Upworthy headline text (used as the join key with `categories_raw.csv`) |
| `log_ctr`  | $\log(\text{clicks} / \text{impressions})$ — the response variable for the ANOVA |

The full 30-column version (with all the headline-level features used in a
parallel STAT 230A project) is not needed here.

## Provenance

Both CSVs originate from the companion repo
[`A-B-testing-analysis-upworthy`](https://github.com/ricardo-pc/A-B-testing-analysis-upworthy)
(STAT 230A, Spring 2026). The BO experiment treats them as fixed inputs.

Experiment 1 (NCF) does **not** ship its training data — MovieLens 1M is a
public dataset hosted by GroupLens (~5 MB) and reproducing the experiment
would require ~5.5 GPU hours regardless. The trial-by-trial outcomes are
recorded in `results/ncf/trials.csv` and the notebook reads from there.
