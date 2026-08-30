# Run log

Append-only. Every Kaggle/Colab session gets an entry: what was run, what came
back, and what the next session should do. The point is that a new session can
pick up without re-deriving context, and that a result is never remembered more
favourably than it happened.

Format: date, environment, commands, results, what broke, next action.

---

## 2026-08-27 -- Kaggle, T4 x2, Version 1

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
| HF login cell | **FAILED** -- see below |

Demo detail worth keeping: the danger-zone line now reads *"unsteerable under
tested interventions (4 not yet through the gauntlet)"* rather than
"readable-but-immovable". That is the R21 terminology change working -- the
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

Nothing was lost -- the failure came after the CPU checks and before any model
download, so no GPU time was spent.

**Next action.** Re-run with the guarded login cell, then cell 5
(`stage1_control` on `Qwen/Qwen2.5-7B-Instruct`). That is the first real
model of the study and the go/no-go for everything after it. Record the
control controllability, the fluency `ceiling_reason`, and the judge
parse-failure rate here.

**Still true going in.** No real model has been run. Every number in the repo
so far is synthetic and validates plumbing only.

---

## 2026-08-29 -- Kaggle, T4 x2, nb1 -- FIRST REAL MODEL. Control FAILED.

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
| `hf_login_if_available()` | printed note, continued -- fix from Version 1 works |
| model load | 28 layers, d_model 3584, cuda, ~110 s |
| **`stage1_control`** | **controllability = 0.000, FAIL (floor 0.10)** |

```
control controllability = 0.000 (floor 0.1)
fluency ceiling: no breakage in swept range
judge parse-failure rate: 0.0%
```

**Reading the failure.** Exactly 0.000, with all nine coefficients usable and a
span of 6.0 RMS units, means `dose_response_auc` integrated a curve where the
behaviour score equalled the baseline at *every* coefficient. Not small -- zero.
Combined with a 0.0% parse-failure rate, the most likely cause is a judge that
returns the same parseable number every time. A constant judge produces a
perfectly flat curve, and a perfectly flat curve is exactly what the danger zone
is defined by.

Second candidate, independent of the first: `st.generate` was feeding raw
instruction text to an instruct model with no chat template applied. That is
base-model prompting -- the model continues the instruction instead of following
it -- so the text being scored was not the behaviour the concept is about.

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
   entire sweep -- the failure the parse-guard cannot see.
4. `lbi.driver.diagnose_steering` prints raw generations and raw judge replies
   at three coefficients, about two minutes instead of a 20-minute re-run.

**Next action.** Do **not** re-run the sweep. Run `diagnose_steering(lm)` and
read the text. It separates the three causes: sensible text that will not move,
garbage text, or a constant judge.

---

## 2026-08-29 -- Kaggle, nb2 -- DIAGNOSTIC. Steering works. Root cause found.

**Command.** `diagnose_steering(lm)` on Qwen2.5-7B, repo at `49412e2`, layer 14,
single-layer add, 2 prompts, coeffs -3 / 0 / +3. ~3.5 min total.

**Result: the harness is fine.** Text moves with the coefficient and the judge
tracks it.

| coeff | judge mean | text |
| --- | --- | --- |
| -3.0 | 0.30 | "...dealing with poor delivery services..." |
| 0.0 | 0.75 | "an impressive exhibit on ancient Egyptian artifacts" |
| +3.0 | 1.00 | "was delighted to share the news with you! It was an honor" |

Implied controllability from these three points: **0.175**, against a floor of
0.10. Judge replies were `'0.6'`, `'0'`, `'1'`, `'0.5'`, `'1'`, `'1'` -- varying,
not constant, so the constant-judge hypothesis is **rejected**.

**Root cause of nb1: the missing chat template.** nb1 fed raw instructions to
an instruct model, which continued rather than followed them; the outputs were
uniform enough that the judge returned the same score everywhere and the curve
was exactly flat. nb2 differs by having the template applied. Steering itself
was never broken.

**Two defects still visible in the text.**

1. *Assistant deflection.* Two of the sampled generations were "As Qwen, I am
   an AI assistant ... do not have personal experiences". Four of sentiment's
   six eval prompts presupposed the assistant's own past ("the museum exhibit
   you saw yesterday"). A deflection carries no sentiment for steering to move
   or a judge to score, so the control was partly measuring refusals. Rewritten
   as writing tasks. One prompt in `rudeness` had the same defect.
2. *The lexicon judge was inert.* It returned its no-hit neutral 0.5 on five of
   six outputs because "impressive", "delighted" and "intricate" were not in the
   sentiment lists. A second judge pinned at 0.5 makes reported agreement
   meaningless rather than merely weak. Lists widened.

**On changing prompts after a failed control.** Recorded deliberately. The
change is a fix to an observable instrument defect -- the model declining to
answer -- and not a search for prompts that produce a better number; no
controllability figure was consulted in choosing the replacements, the same
rule was applied to every concept rather than to the control alone, and this
entry exists so the change cannot be quietly absorbed. The prompt set should be
frozen after the next control run and not touched again.

**Next action.** Re-run `stage1_control` on Qwen2.5-7B with the new prompts.
Expect roughly 0.10-0.18: the diagnostic used three coefficients, while the
real grid has nine, and the small ones contribute little movement and drag the
mean down. **Delete any stale `*sentiment.json` first** or `resume=True` will
skip the concept and keep nb1's 0.000.

---

## 2026-08-30 -- nb3 analysed, three fixes. Control 0.033, still FAIL.

**Result.** Control 0.033 (floor 0.10), ceiling `perplexity > 2.0x baseline at
coeff -2`. Ran on pre-`8a24826` code, so the fixed judge was **not** in it --
the result file has no `judge_model` key.

The chat-template fix worked: the curve moves now.

| coeff | behaviour | ppl | usable |
| --- | --- | --- | --- |
| -3.0 | 0.383 | 153.4 | no |
| -2.0 | 0.517 | 506.2 | no |
| -1.0 | 0.900 | 4.9 | yes |
| 0.0 | 0.967 | 5.5 | yes |
| 1.0 | 1.000 | 5.2 | yes |
| 2.0 | 0.825 | 94.8 | no |
| 3.0 | 0.625 | 69.2 | no |

Baseline perplexity 5.46, threshold 10.9.

**The perplexity gate is correct, not overtuned.** At -2 the sample reads
`"...long gone by nowhenremembrighterlandsogenesisagowhenuttauf.Re"`. That is
word salad. The 2.0x ratio was not touched and should not be.

**A hypothesis that was wrong, recorded because it was acted on.** Before
seeing the data I believed the ceiling was sign-asymmetric -- that negative
steering broke while positive stayed clean -- and was ready to make
`find_ceiling` per-sign. The curve says both signs break at |2| (506 and 95
against 10.9). Per-sign ceilings would have changed nothing. The data was
requested before the change; it should be.

**Three real problems, all fixed in this pass.**

1. *P2 was violated.* `train_probes` chose the layer by test AUROC and then
   reported that same AUROC as readability. PLAN P2 says the layer is chosen on
   a validation split disjoint from train and test, and **never** on test
   AUROC. This is objection 2 in the ledger, and it was documented and
   unimplemented -- the same failure class as P4, P6, P7 and P9. Now a
   three-way family split: `default_splits` returns validation and test
   families, selection happens on validation, readability is the test AUROC at
   the selected layer.
2. *Ties broke toward layer 0.* The probe reported `best_layer=1` of 28 at
   AUROC 1.000. With several layers saturated, `max()` returns the earliest.
   P4 then anchored the steering band there, giving layers 0-6 -- and an
   intervention early enough to move behaviour drives perplexity from 5 to 506,
   which is the cliff above. Ties now break toward the middle of the network.
   nb2's diagnostic steered at layer 14 and produced readable text at ±3, same
   model, same direction method.
3. *The baseline was saturated.* The judge scored unsteered output at 0.967, so
   positive steering had at most 0.033 of headroom by construction and the
   whole measurement leaned on one direction. Caused by my own previous prompt
   rewrite: asking a helpful assistant to "review" something gets a favourable
   review. Prompts are now neutral descriptions.

**Missing output, again.** `probe.summary()` dropped `per_layer`, so there was
no way to tell whether layer 1 was a genuine peak or one of many ties. The
per-layer AUROC curve is now persisted as `probe_layers`.

**Third prompt change, recorded.** Same standard as last time: driven by an
observable defect (baseline saturation visible in the curve, not in any
controllability figure), applied to the concept uniformly. This is the last one
-- freeze the prompt set after the next control run.

**Next action.** Re-run the control as **nb4**, on `Qwen/Qwen2.5-7B-Instruct`,
with `judge_lm=load_judge()`. Record `best_layer` and the `probe_layers` curve:
if selection still lands early, the problem is the concept and not the
tie-break.
