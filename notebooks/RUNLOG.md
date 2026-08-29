# Run log

Append-only. Every Kaggle/Colab session gets an entry: what was run, what came
back, and what the next session should do. The point is that a new session can
pick up without re-deriving context, and that a result is never remembered more
favourably than it happened.

Format: date, environment, commands, results, what broke, next action.

---

## 2026-08-27 — Kaggle, T4 x2, Version 1

**Environment.** Kaggle "Latest Container Image", Python 3.12, GPU T4 x2
(2 x Tesla T4, 15360 MiB each). Internet on. Repo cloned at commit `787655f`.

**Commands.**

```bash
!git clone -q https://github.com/ArnavLifelessCoder/legible-but-immovable.git /kaggle/working/lbi-repo
%cd /kaggle/working/lbi-repo
!pip -q install -U bitsandbytes accelerate transformers
!python -m pytest tests/ -q
!python scripts/demo_synthetic.py
# then: preflight(), then the HF login cell
```

**Results.**

| Step | Outcome |
| --- | --- |
| pip install | OK, ~15 s |
| `pytest tests/ -q` | **125 passed** in 62.53 s |
| `demo_synthetic.py` | **ALL CHECKS PASSED**, ground truth recovered |
| `preflight()` | 8/10 PASS, `topic_science` and `verbosity` FAIL* as declared |
| HF login cell | **FAILED** — see below |

Demo detail worth keeping: the danger-zone line now reads *"unsteerable under
tested interventions (4 not yet through the gauntlet)"* rather than
"readable-but-immovable". That is the R21 terminology change working — the
synthetic demo never runs the gauntlet, so nothing is entitled to the stronger
word. Primary test on synthetic data: partial rho = 0.525 [-0.127, 0.863].

**What broke.**

```
BackendError: Unexpected response from the service.
Response: {'errors': ['No user secrets exist for kernel id 132377725 and
label HF_TOKEN.'], 'error': {'code': 5}, 'wasSuccessful': False}
```

Cause is the notebook, not the environment. The HF-login cell called
`UserSecretsClient().get_secret("HF_TOKEN")` unguarded. **Kaggle's Save Version
runs every cell top to bottom under papermill**, so "just skip that cell for
the Qwen run" is not a thing that can happen in a committed notebook: any cell
that can raise will eventually kill a run. The cell now degrades to a printed
note when the secret is absent.

Nothing was lost — the failure came after the CPU checks and before any model
download, so no GPU time was spent.

**Next action.** Re-run with the guarded login cell, then cell 5
(`stage1_control` on `Qwen/Qwen2.5-7B-Instruct`). That is the first real
model of the study and the go/no-go for everything after it. Record the
control controllability, the fluency `ceiling_reason`, and the judge
parse-failure rate here.

**Still true going in.** No real model has been run. Every number in the repo
so far is synthetic and validates plumbing only.

---

## 2026-08-29 — Kaggle, T4 x2, nb1 — FIRST REAL MODEL. Control FAILED.

**Environment.** Same image. Repo at `685d8d6`. No `HF_TOKEN` secret; the
guarded login printed its note and continued, as intended.

**Commands.** Cells 1-5 as in `README.md`, model `Qwen/Qwen2.5-7B-Instruct`,
4-bit.

**Results.**

| Step | Outcome |
| --- | --- |
| `pytest tests/ -q` | 125 passed, 58 s |
| `demo_synthetic.py` | ALL CHECKS PASSED |
| `preflight()` | 8/10, two declared |
| `hf_login_if_available()` | printed note, continued — fix from Version 1 works |
| model load | 28 layers, d_model 3584, cuda, ~110 s |
| **`stage1_control`** | **controllability = 0.000, FAIL (floor 0.10)** |

```
control controllability = 0.000 (floor 0.1)
fluency ceiling: no breakage in swept range
judge parse-failure rate: 0.0%
```

**Reading the failure.** Exactly 0.000, with all nine coefficients usable and a
span of 6.0 RMS units, means `dose_response_auc` integrated a curve where the
behaviour score equalled the baseline at *every* coefficient. Not small — zero.
Combined with a 0.0% parse-failure rate, the most likely cause is a judge that
returns the same parseable number every time. A constant judge produces a
perfectly flat curve, and a perfectly flat curve is exactly what the danger zone
is defined by.

Second candidate, independent of the first: `st.generate` was feeding raw
instruction text to an instruct model with no chat template applied. That is
base-model prompting — the model continues the instruction instead of following
it — so the text being scored was not the behaviour the concept is about.

**Output that was missing.** The result file could not distinguish these. It
recorded `coeff`, `behavior`, `perplexity`, `broken` and dropped the
generations, which existed in memory. 1.7 KB of numbers and no text. Fixed:
`samples`, `repetition` and the behaviour CI are now persisted.

**Timing correction.** The control alone took **~21.7 min** (264.7 s to
1565.4 s), not the ~5 min estimated. The estimate was for single-layer
steering; the control runs the full P4 band, so it is four sweeps. This scales:
**a 10-concept model is ~3.6 h, not ~1.5 h**, and seven models is ~25 h against
a weekly quota, not ~6 h. The budget table in `README.md` was wrong and is
corrected.

**Changes made in response.**

1. `st.generate` applies the tokenizer chat template when one exists, stripping
   a leading `"User: "` so transcript-style prompts are not double-wrapped.
2. Per-coefficient `samples`, `repetition` and behaviour CI are written to the
   result JSON.
3. `run_steering` warns when a judge returns one distinct value across an
   entire sweep — the failure the parse-guard cannot see.
4. `lbi.driver.diagnose_steering` prints raw generations and raw judge replies
   at three coefficients, about two minutes instead of a 20-minute re-run.

**Next action.** Do **not** re-run the sweep. Run `diagnose_steering(lm)` and
read the text. It separates the three causes: sensible text that will not move,
garbage text, or a constant judge.
