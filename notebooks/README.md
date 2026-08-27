# Running the study on a Colab T4

`run_colab.py` is the driver, kept as a script rather than a `.ipynb` so it
diffs as text and carries no stale embedded outputs. Split it into cells at the
`STAGE` comments.

## Cell 1 — setup, ~3 min

```python
from google.colab import drive; drive.mount('/content/drive')
```

then the contents of `SETUP`. Both the test suite and the synthetic demo must
pass before any GPU-minute is spent; neither needs a GPU or a download, and
between them they check every metric path against planted ground truth.

## Cell 2 — preflight, seconds

```python
preflight()
```

The surface-shortcut audit. `FAIL*` on `verbosity` and `topic_science` is
expected and declared. Any other failure means the concept set is broken; fix it
before running a model.

## Cell 3 — positive control, ~5 min

```python
stage1_control("Qwen/Qwen2.5-7B-Instruct")
```

**Stop here if it returns False.** A model where sentiment will not steer scores
low controllability on every concept, which puts the whole model in the danger
zone and looks exactly like the paper's finding. `aggregate` will withhold it
(P9) but there is no point generating the data.

If it fails, check in this order: the chat template is being applied; the
coefficient grid reaches far enough in RMS units; and whether the fluency
ceiling is hit before any behaviour change — `ceiling_reason` on the result says
which of the two triggered.

## Cell 4 — full sweep, ~1.5 h per 7B

```python
stage2_full("Qwen/Qwen2.5-7B-Instruct")
```

Results write per concept as they finish, so a dead runtime loses at most one
concept. Re-running re-reads the activation cache rather than recomputing.

Watch two numbers in the output: the judge parse-failure rate (a judge that
cannot be parsed returns a constant, and a constant reads as an immovable
concept — `LLMJudgeScorer` raises past 20%) and Krippendorff alpha between the
judges.

## Cell 5 — repeat for each model

One model per session. `MODELS` lists the four families plus the Qwen size
sweep for Experiment 6. The small Qwens are minutes each.

## Cell 6 — aggregate, CPU only

```bash
!python scripts/run_experiment.py --aggregate-only --out-dir {OUT_DIR}
```

Withholds any model whose control failed, builds the gap map with CI-excluded
danger-zone membership and a concept-clustered correlation CI, then runs the
preregistered primary test and the BH-corrected exploratory analyses.

## The human rater

`stage2_full` writes `human_labels.csv` with 100 sampled outputs. Fill the
`score` column with values in [0, 1]; leave blanks for anything you cannot
judge, since `read_labeling_sheet` returns NaN and the alpha computation handles
missing cells. This is the third judge the objection ledger promises.

## Budget

Measured from the work volume (1,752 activation texts and 1,980
prompt-generations of 64 tokens per model under the band protocol):

| Model | Full sweep |
| --- | --- |
| 7-9B, 4-bit | ~1.5 h |
| 3B | ~45 min |
| 1.5B | ~30 min |
| 0.5B | ~15 min |

About 6 hours of T4 time for all seven, before downloads and retries. Colab free
tier throttles well before that in one day, so plan on one model per session.
