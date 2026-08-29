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

## The two GPUs

A 7-9B model in 4-bit needs about 6 GB, so it fits on one T4 and `load_model`
pins device 0. The second card is spare capacity rather than a requirement — it
is not needed to fit anything in this study, so do not spend time sharding.

## Budget

Measured from the work volume (1,752 activation texts and 1,980
prompt-generations of 64 tokens per model under the band protocol):

| Model | Full sweep |
| --- | --- |
| 7-9B, 4-bit | ~1.5 h |
| 3B | ~45 min |
| 1.5B | ~30 min |
| 0.5B | ~15 min |

About 6 hours for all seven before downloads and retries, against a weekly GPU
quota. One model per session.

## The human rater

`stage2_full` writes `human_labels.csv` with 100 sampled outputs. Fill the
`score` column with values in [0, 1] and leave blanks for anything you cannot
judge: blanks come back as NaN and the alpha computation handles missing cells,
so a partial sheet is usable and must not be padded with guesses. This is the
third judge the objection ledger promises.
