# Running the study on Kaggle (T4 x2)

`run_kaggle.py` holds the paths and the model list. The stages themselves live
in `lbi/driver.py` so they are covered by tests and reviewable as code rather
than as cell output.

## Before the first cell

Three settings, and the study cannot run without any of them:

1. **Settings -> Accelerator -> GPU T4 x2**
2. **Settings -> Internet -> On.** Off by default, and needs phone
   verification. Without it `git clone` and every model download fail.
3. **Add-ons -> Secrets -> `HF_TOKEN`** if you want Llama or Gemma. Both are
   gated: accept the licence on huggingface.co with the same account first, or
   the download 403s after you have already spent session time.

Start with `Qwen/Qwen2.5-7B-Instruct`, which is ungated, so a licence problem
cannot be confused with a pipeline problem on the first run.

## Save Version runs every cell

Kaggle's **Save Version** re-executes the notebook top to bottom under
papermill. There is no interactive skipping, so any cell that can raise will
eventually fail a run -- and the notebook is reported as failed even if every
check before it passed. Optional steps must degrade to a printed note.

That is what killed Version 1: an unguarded
`UserSecretsClient().get_secret("HF_TOKEN")` for a Qwen run that needed no
token. Use `hf_login_if_available()` from `run_kaggle.py` instead.

Running the sweep *as* a Save Version batch job is the better workflow anyway:
it runs headless for up to twelve hours and persists `/kaggle/working` when it
finishes, instead of depending on a browser tab staying open.

## Persistence

`/kaggle/working` is the only directory that survives, and only if you **Save
Version** at the end of the session. `/kaggle/temp` does not.

Per-concept results are small JSON. At the end of each model's session, Save
Version, then attach that notebook's output as an input dataset to the next
session so `--aggregate-only` can see every model at once.

## The two GPUs: study on 0, fixed judge on 1

A 7-9B in 4-bit needs about 6 GB and fits one T4, so the second card is not
needed to hold the model under study. It holds the **judge** instead, and that
is not an optimisation -- it is a correctness requirement.

The judge must be the same model for every model under study. If each studied
model scores its own outputs, the measuring instrument changes with the
condition. Within-model normalisation protects the correlation and Figure 1,
but danger-zone membership is decided on **raw** controllability by design
(P6, R10), so a stricter judge on one model can put a concept in the danger
zone for a reason that has nothing to do with steering.

```python
from lbi.driver import load_judge, stage1_control, stage2_full
judge = load_judge()                     # Qwen2.5-1.5B-Instruct, fp16, cuda:1
ok = stage1_control(lm, OUT_DIR, CACHE_DIR, judge_lm=judge)
```

Omitting `judge_lm` falls back to self-judging and prints a warning. That is
fine for a smoke test and not fine for anything reported. Every result file
records `judge_model` and `judge_is_self`, so a dataset accidentally scored by
two different judges is detectable afterwards rather than assumed away.

## Budget

**Measured on a real run, not estimated.** The positive control alone -- one
concept, full P4 band -- took **21.7 minutes** on Qwen2.5-7B 4-bit on
2026-08-29. The earlier table here said ~1.5 h for a whole model; that was
derived for single-layer steering and the band protocol is four sweeps, so it
was low by about 2.4x.

| Model | One concept | Full sweep (10 concepts) |
| --- | --- | --- |
| 7-9B, 4-bit | ~22 min | **~3.6 h** |
| 3B | ~9 min | ~1.5 h |
| 1.5B | ~6 min | ~1 h |
| 0.5B | ~3 min | ~30 min |

Seven models is roughly **25 hours**, not 6, before downloads, retries and the
gauntlet. That is at or over a weekly GPU quota, so the model list is a budget
decision and not a free choice. Options if it does not fit: drop the 3B from the
scale sweep, trim the coefficient grid from nine points to seven, or cut
`max_new_tokens` from 64 to 48. Do not silently drop the band -- that is P4.

## The human rater

`stage2_full` writes `human_labels.csv` with 100 sampled outputs. Fill the
`score` column with values in [0, 1] and leave blanks for anything you cannot
judge: blanks come back as NaN and the alpha computation handles missing cells,
so a partial sheet is usable and must not be padded with guesses. This is the
third judge the objection ledger promises.
