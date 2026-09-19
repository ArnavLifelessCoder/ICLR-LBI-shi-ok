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

---

## 2026-08-30 -- nb4 -- KILLED AT 12h. Zero output. Private repo.

**Cost.** 43200 s of GPU quota, exit code 137, 0 B of output. Nothing was run.

```
43209.5s  Timeout waiting for execute reply (43200s).
43209.5s  Username for 'https://github.com': ^C
43209.7s  [Errno 2] No such file or directory: '/kaggle/working/lbi-repo'
```

**Cause.** `ICLR-LBI-shi-ok` was created **private**. Cloning a private repo
over HTTPS without credentials prompts for a username. Under papermill nothing
can answer the prompt, so the cell did not fail -- it blocked until Kaggle's
twelve-hour ceiling. Every earlier notebook cloned the old repo, which is
public, which is why this appeared only after the move.

Confirmed: `ICLR-LBI-shi-ok` returns HTTP 404 unauthenticated,
`legible-but-immovable` returns 200.

**Fixes.** Repo goes public. And cell 1 now sets `GIT_TERMINAL_PROMPT=0` so git
exits with an error in seconds instead of waiting for input that cannot arrive.
The flag is the seatbelt; public is the fix. A step that can block on stdin has
no place in a batch notebook, which is the same lesson as the unguarded
`get_secret` in Version 1 -- and this time it cost twelve hours rather than two
minutes.

**Budget.** Roughly 12 h of a ~30 h weekly quota is gone for nothing. The
seven-model plan needs ~25 h. Until the quota resets, prioritise: one control
per model beats one full sweep.

**Next action.** Make the repo public, then re-run the nb4 control cells
unchanged. The three fixes from `c7b2ddc` are still untested against a real
model.

---

## 2026-09-01 -- Kaggle, nb4 -- CONTROL PASSES. First reportable numbers.

**Environment.** Repo at `c7b2ddc`, made public first (nb4's predecessor died
on a private-repo clone prompt). Qwen2.5-7B-Instruct 4-bit on cuda:0, judge
Qwen2.5-1.5B-Instruct fp16 on cuda:1. Control alone: **11.8 min**, faster than
nb3's 21.7 because breakage no longer forces the widest sweeps.

```
judge: Qwen/Qwen2.5-1.5B-Instruct (fixed)
control controllability = 0.309 (floor 0.1)
fluency ceiling: perplexity > 2.0x baseline at coeff 3
judge parse-failure rate: 0.0%
PASS -- steering works on this model, run the sweep.
```

**All three fixes from `c7b2ddc` did what they were meant to.**

| | nb3 | nb4 |
| --- | --- | --- |
| `best_layer` | 1 | **13** |
| `baseline_behavior` | 0.967 | **0.825** |
| controllability | 0.033 | **0.309** |
| judge | self | fixed, recorded |

The layer curve settles the tie-break question: **25 of 28 layers sit at
validation AUROC 1.000**. Argmax really was returning the earliest of a mass of
ties, and layer 1 was never a peak.

**Honest caveat that belongs in Methods.** Because validation AUROC saturates
across almost the whole network, it does not discriminate between layers. The
preregistered tie-break -- nearest the middle -- is what actually selects layer
13, not validation performance. That is a defensible prior and it must be
stated as a prior rather than presented as a data-driven choice. Layer 3 is the
cautionary case: validation AUROC 1.000, test AUROC 0.188.

**Dose-response is strongly asymmetric.** Baseline 0.825 leaves little room
upward; the signal is almost entirely on the negative side (0.825 -> 0.083 at
coeff -2). Worth a sentence in the paper: the control demonstrates steering
works, and it does so mainly in one direction.

**A new bug the passing curve exposed.** `find_ceiling` walked by `|coeff|`
against one scalar and `mark_broken` thresholded on `|coeff|`, so the point
that *breached* the gate could still be counted. Here -3.0 had perplexity 9.59
(under the 12.96 threshold) and set `max_usable = 3.0`; +3.0 then measured
14.57, breached, stopped the walk -- and `|3.0| > 3.0` is false, so it went into
the AUC anyway. Replaced by `mark_broken_by_fluency`, which marks each point on
its own perplexity and repetition and then closes over each sign separately.
`bootstrap_curve_ci` now takes the exact usable coefficient set rather than a
magnitude, so the interval and the point estimate cannot describe different
sets.

Recomputed on nb4's own curve: **0.309 -> 0.338**. It rises, because dropping
+3.0 shrinks the coefficient span more than it removes area. The control passes
either way, which is worth stating: the fix was not outcome-motivated and did
not change the verdict.

**Next action.** Re-run the control once on `c7b2ddc`+ceiling-fix to get a
number produced by the corrected code, then the full ten-concept sweep on this
model (~3.6 h). Budget note: roughly 12 h of the weekly quota was lost to the
private-repo hang, so prefer one control per model over one full sweep until it
resets.

---

## 2026-09-01 -- Kaggle, nb5 -- Control 0.338 PASS. Sweep died on the judge.

**Control, on the corrected ceiling code:** 0.338, exactly the number predicted
when the ceiling fix was written. PASS. Resume then skipped sentiment in the
sweep, as intended.

**The sweep got four concepts in and the judge broke it.**

```
WARNING [rudeness]:   judge returned 0.0 for all 54 generations
WARNING [sycophancy]: judge returned 0.0 for all 54 generations
JudgeParseError: honesty: 2/6 unparsed. Samples: ['20', '2+2=4']
```

`'2+2=4'` is the diagnosis: the 1.5B judge was **answering the text instead of
scoring it**. And a judge that returns 0.0 to all 54 generations produces
controllability exactly 0.000, which is precisely what the danger zone selects
for. Had the parse error not stopped the run, rudeness and sycophancy would
have been reported as readable-but-immovable on the strength of a judge that
said "no" to everything.

Completed before the crash: sentiment (skipped, from the control), formality,
rudeness, sycophancy, refusal. **rudeness and sycophancy are unusable.**

**Fix: read the judge's logits instead of parsing its text.** Every
`behavior_question` in the concept set opens with a yes/no question, so
`LogitJudgeScorer` builds "«question» / Text: ... / Answer Yes or No", runs one
forward pass, and returns P(Yes) over the Yes/No token mass. This removes the
whole failure class at once:

* nothing to parse, so no `JudgeParseError` and no arithmetic answers;
* continuous in [0, 1], so a partial behaviour shift registers instead of
  rounding to 0 or 1 -- the coarseness was silently costing signal on every
  concept, not just the two that collapsed;
* one forward pass rather than a decoding loop, so it is also faster.

**Second fix: a degenerate judge can no longer reach the gap map.**
`SteeringResult.judge_degenerate` is set when a whole sweep returns one
distinct score, persisted, and `aggregate` drops those points with a printed
reason. A warning was not enough -- the number it warns about is exactly the
number the danger zone is defined by.

**Next action.** Re-run the sweep from scratch with the logit judge. Delete
`results/` first: the four completed concepts were scored by the generation
judge and are not comparable to anything produced from here on, and resume
would silently keep them.

---

## 2026-09-02 -- Kaggle, nb5 -- FULL SWEEP COMPLETE. Danger zone empty.

**Environment.** Repo `c239033`, Qwen2.5-7B-Instruct 4-bit on cuda:0, fixed
logit judge Qwen2.5-1.5B-Instruct fp16 on cuda:1. Control 11 min, sweep 1h45,
**1h55 total** -- well under the 3.6 h estimate, because the logit judge is one
forward pass instead of a decoding loop.

**Control:** 0.215 (floor 0.10), PASS. This is the number of record; 0.338 was
the generation judge and is now historical.

| concept | read | read CI | ctrl | ctrl CI | layer | base | gauntlet |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sentiment (control) | 1.000 | [1.000,1.000] | 0.215 | [0.115,0.339] | 13 | 0.98 | -- |
| rudeness | 0.797 | [0.627,0.938] | 0.153 | [0.080,0.259] | 15 | 0.81 | -- |
| certainty | 1.000 | [1.000,1.000] | 0.123 | [0.094,0.220] | 13 | 0.49 | -- |
| factuality | 1.000 | [1.000,1.000] | 0.120 | [0.088,0.177] | 13 | 0.74 | -- |
| sycophancy | 1.000 | [1.000,1.000] | 0.118 | [0.083,0.202] | 3 | 0.43 | -- |
| honesty | 1.000 | [1.000,1.000] | 0.081 | [0.070,0.176] | 20 | 0.44 | -- |
| verbosity | 1.000 | [1.000,1.000] | 0.050 | [0.029,0.088] | 13 | 0.62 | moved (add) |
| refusal | 1.000 | [1.000,1.000] | 0.047 | [0.043,0.108] | 13 | 0.31 | moved (clamp) |
| formality | 0.992 | [0.964,1.000] | 0.034 | [0.021,0.070] | 13 | 0.91 | moved (clamp) |
| topic_science | 0.944 | [0.771,1.000] | 0.019 | [0.014,0.029] | 13 | 0.95 | **survived** |

**The danger zone is empty.** No concept satisfies P6. Nearest misses:

* `topic_science` -- controllability CI clears (0.029 <= 0.05) but the
  readability CI runs to 0.771, below the 0.9 floor. It is also one of the two
  declared surface-confounded concepts, so its readability could not have been
  reported as clean even if the CI had held.
* `formality`, `refusal`, `verbosity` -- readability CIs clear, controllability
  CIs run to 0.070 / 0.108 / 0.088, all above 0.05. All three then *moved*
  under the gauntlet (clamp or add), so none is unsteerable either.

The one gauntlet survivor is `topic_science`, and it is surface-confounded.
That cannot headline anything.

**The readability axis is saturated and this is the real finding of the run.**
Seven of ten concepts sit at exactly 1.000 and the spread is
[0.797, 0.944, 0.992, 1.000 x7]. A near-constant x cannot correlate with
anything: the gap map returns Spearman rho = 0.231 with a cluster-bootstrap CI
of [-0.592, 0.897], which is uninformative rather than null. The minimal-pair
construction that made the surface audit pass -- disjoint markers, matched
lengths -- also made every concept trivially linearly separable. The audit and
the readability axis were traded against each other and nobody noticed until
there were ten real numbers to look at.

**Judge agreement is poor.** Krippendorff alpha = 0.132 over 3240 items,
between the logit judge and the lexicon scorer. That is not "two judges with
reported agreement" in any useful sense; objection 14 is currently unanswered.

**H1: uninformative, and the verdict string said the opposite.** Partial
Spearman = -0.709, CI [-0.962, 0.340]. The old verdict checked `abs(rho) >= 0.7`
and printed *"output overlap predicts controllability; geometry explains the
gap"* -- from an interval that includes zero, with a sign opposite to the one
Section 2.5 predicts. Fixed: the verdict now requires the CI to exclude zero,
and reports a strong negative estimate as contradicting H1 rather than
supporting it. On this data it now reads UNINFORMATIVE. LOO R^2 = -0.380.

**Next action.** See the assessment in RESULTS.md. The blocking problem is the
saturated readability axis, not the pipeline: the pipeline is now working.

---

## 2026-09-02 -- Kaggle, nb6 -- TWO MODELS COMPLETE. 20 points.

**Environment.** Repo at `9f86309`. Qwen2.5-7B-Instruct then
Mistral-7B-Instruct-v0.3, both 4-bit on cuda:0, judge Qwen2.5-1.5B-Instruct
fp16 on cuda:1 loaded once and shared. 4h21m total. `del lm` between models
freed cuda:0 to 10.0 GB both times, so the two-model-per-session pattern works.

**Both controls pass.** Qwen 0.215, Mistral 0.163, floor 0.10. Neither model is
withheld under P9.

**Danger zone: one point.** `topic_science@Mistral`, confirmed immovable
against all six interventions. It replicates on Qwen (0.019 vs 0.024, gauntlet
survived on both), so the effect is real and not model-specific. It is also one
of the two concepts declared surface-confounded, so its readability is the
number the audit says not to trust. The claim has an occupant it cannot use.

**`refusal` is the near-miss worth chasing.** Not surface-confounded,
0.047 / 0.038 across the two models, survived the gauntlet on Mistral, and
misses P6 only on the CI: controllability upper bound 0.080 against a 0.05
threshold.

**H1 uninformative.** partial rho = -0.430, CI [-0.729, 0.070]. The verdict
logic added in `26ae787` reported this correctly as uninformative rather than
announcing support from a point estimate; the earlier code would have said
"geometry explains the gap". Sign is also opposite to Section 2.5's prediction.
E1 p_BH 0.907, E2 p_BH 0.907, ridge LOO R^2 -0.033. All null.

**Gap map.** 20 points, Spearman 0.119, CI [-0.233, 0.505]. Much tighter than
nb5's [-0.592, 0.897] but still spanning zero.

**Two problems the fuller dataset makes unavoidable.**

1. *Readability is saturated.* Sixteen of twenty points at exactly AUROC 1.000.
   The primary x-axis has almost no variance, so the headline correlation is
   uninformative rather than null. Selectivity is available as a declared
   secondary axis and does have spread.
2. *Judge agreement is 0.132 (Qwen) and 0.309 (Mistral).* Objection 14 is
   unanswered. The lexicon second judge returns its neutral 0.5 on too much
   free text, and `human_labels.csv` is still unfilled.

P3's flag fired on four of twenty points: Qwen certainty 0.631, rudeness 0.644,
sycophancy 0.615, Mistral verbosity 0.601.

**Next action.** Decide the framing before writing the abstract. Both controls
pass so the kill branch is closed; the dissociation has one confirmed instance
and it is confounded; the geometry pivot is unavailable because H1 and both
exploratory features are null. What is defensible today is a characterisation
paper. Technically the highest-value next runs are a third model family and
more eval prompts for `refusal`, in that order.

---

## 2026-09-03 -- Kaggle, nb7 -- FOUR MODELS COMPLETE. 40 points. H1 crosses.

**Environment.** Repo at `eba69b4`. Llama-3.1-8B-Instruct then gemma-2-9b-it,
both 4-bit on cuda:0, judge Qwen2.5-1.5B-Instruct fp16 on cuda:1 loaded once and
shared. Same two-model-per-session pattern as nb6.

**First attempt died on Gemma access, but lost nothing.** Llama ran to
completion (all 10 concepts, ~2h55), then Gemma failed 4 s after start with
`GatedRepoError: 403 ... Access to model google/gemma-2-9b-it is restricted`.
The `HF_TOKEN` account had not accepted the Gemma licence. The failure came at
the tokenizer download, before any Gemma compute, so the whole cost was Llama
finishing successfully. Fix was account-side: accept the licence on the exact
`-it` repo (the base `gemma-2-9b` repo is a separate gate), confirmed with a
one-file `hf_hub_download(..., "config.json")` before relaunching. Second run
completed both models.

**All four controls now pass; two of them narrowly.**

| Model | Control | Floor |
| --- | --- | --- |
| Qwen2.5-7B (nb6) | 0.215 | 0.10 |
| Mistral-7B (nb6) | 0.163 | 0.10 |
| Llama-3.1-8B | 0.101 | 0.10 |
| gemma-2-9b-it | 0.110 | 0.10 |

Llama passes by 0.001. Genuine pass, thin margin; belongs in Methods.

**The notebook aggregated only 20 points, not 40.** Same nb6-copy failure as
before: `cp /kaggle/input/nb6-lbi/results/*.json` returned "No such file or
directory", so the notebook never had Qwen or Mistral present and its own gap
map covered Llama and Gemma alone. The 40-point map was rebuilt locally by
combining the nb6 and nb7 result files (44 files: 10 concepts + 1 control x 4
models) and re-running `--aggregate-only`. Saved to
`results nb7/combined_40point/`. **Fix for next time: attach the nb6 result
dataset under the exact path the copy cell expects, or the aggregate silently
runs on a subset.**

**Danger zone: two points, both `topic_science`** (Mistral and Gemma), each
confirmed immovable against all six interventions. Gemma's is the new one:
readability 1.000 [1.000, 1.000], controllability 0.007 [0.006, 0.013]. Qwen and
Llama place `topic_science` just outside, both because the readability CI dips
under 0.9 (Qwen 0.771, Llama 0.875). So the concept is the least controllable on
all four models and immovable on the two whose readability CI happens to hold,
and it is the surface-confounded concept every time. The occupant the claim
cannot use now appears on two model families.

**`refusal` holds as the clean near-miss across all four.** 0.047 / 0.038 /
0.033 / 0.038. Survived the gauntlet on Llama as well as Mistral. Still misses
P6 only on the controllability CI upper bound (0.108 / 0.080 / 0.107 / 0.146,
all above 0.05). Not chased, per the CI arithmetic in CONTEXT.md.

**H1 crosses from uninformative to a moderate contradiction.** With 40 points:
partial rho = -0.451, CI [-0.723, -0.041]. The interval now **excludes zero**,
on the wrong side. Section 2.5 predicts more output overlap means more
controllability; the data says less, at moderate strength. This is a real
negative result about mechanism M1, not weak support. nb6 had -0.430 with the CI
spanning zero (uninformative); the two new models tightened it past zero.

**Verdict-logic bug found and fixed while reading this result.** `primary_test`
gated its wrong-sign guard behind `abs(rho) >= 0.7`. A moderate wrong-signed
estimate whose CI excludes zero fell through to a sign-blind branch and printed
"suggestive but not strong; report as suggestive, not confirmation" -- which is
wrong twice: the CI excludes zero (a resolved result, not suggestive) and the
sign contradicts H1. The chain is now sign-aware at every magnitude: any
CI-excludes-zero wrong-signed result reads as a contradiction, with a
modest-support branch for the right-signed moderate case. Two behavioral
regression tests added (the prior coverage was source-inspection only, which is
why the gate slipped through). Suite at 143. Committed and pushed to iclr as
`4d6ef41`.

**Exploratory and predictor still null.** E1 participation ratio rho = -0.092,
p_BH 0.833. E2 low-variance PC alignment rho = 0.023, p_BH 0.833. Ridge LOO
R^2 = 0.001. Gap map Spearman = 0.135, CI [-0.239, 0.509], still spanning zero.

**Judge agreement, all four.** Qwen 0.132, Mistral 0.309, Llama 0.300, Gemma
0.149. Poor throughout. Objection 14 remains open; `human_labels.csv` still
empty.

**Two saturation facts, now unavoidable.** 31 of 40 points at exactly AUROC
1.000, so the primary x-axis has almost no variance. P3 control-probe flag fires
on 5 of 40 (Qwen certainty/rudeness/sycophancy, Mistral verbosity, Llama
refusal).

**Next action.** The data-collection phase is effectively done for the
characterisation paper: four model families, 40 points, controls all passing,
the dissociation confirmed on a confounded concept and a clean near-miss that
sits on the P6 boundary. The two things that would change the paper are neither
of them more sweeps: (1) fill `human_labels.csv` to answer objection 14 on judge
agreement, which is hand-labelling not GPU; (2) decide the abstract framing,
given H1 is now a moderate contradiction rather than a null. A fifth model
family is optional upside, not a blocker. Do **not** re-run for the map; it is
complete.

---

## 2026-09-04 -- No GPU. Write-up: docs aligned to the inverted-H1 arc, abstract drafted.

**No run.** Desk session on the 40-point map from nb7. The point was to fix the
framing and confirm nothing needs re-running, not to collect data.

**Audit first: nothing needs a re-run.** Confirmed the H1 sign-gate bug was a
reporting bug, not a data bug -- the per-concept readability, controllability and
geometry numbers are untouched by it, so re-aggregating the existing files with
the fixed code is the complete fix. Verified the committed
`results nb7/combined_40point/geometry_predictor.json` carries the corrected
verdict (not a stale pre-fix copy), the gap map has 40 points and two danger-zone
occupants, and Gemma is not quietly wrong (eager-attention guard present in
`extraction.py:93`, judge not degenerate, generations fluent). Full suite 143
passed. The recurring nb6-copy miss is a Kaggle dataset-attach step, not a code
bug, and the notebook is not in the repo; documented the fix in CONTEXT.md
instead.

**Framing changes, all pushed to iclr.**

| Commit | Change |
| --- | --- |
| `4d6ef41` | H1 verdict made sign-aware at every magnitude; two behavioral regression tests; 40-point artifacts saved |
| `2f7dcc4` | RESULTS and RUNLOG updated for 40 points |
| `a76b0c1` | Section 2.5 gains a dated block: preregistered M1 prediction left verbatim, observed inversion framed as a directional contradiction, `topic_science` named as the driver, lexicality reading offered |
| `302d74a` | Section 1 thesis paragraph aligned: drops "predictable from geometry", states the arc as "the obvious geometric predictor points the wrong way, and here is why" |
| `f5effb9` | CONTEXT.md refreshed to the four-model / 40-point state |
| `62b9f2b` | `ABSTRACT.md` drafted to the inverted-H1 arc (main ~215 words + short form + claim-to-evidence map) |

**The arc, settled across all four docs.** Detection and control are
uncorrelated (Spearman 0.14, CI includes zero); legible concepts resist the
six-method gauntlet; the first-order geometric predictor is inverted (partial
rho -0.451, CI [-0.723, -0.041] excluding zero on the side opposite Section
2.5); `topic_science` drives it as the highest-overlap and least-controllable
concept on every model; lexicality is the offered reading. Section 1, Section
2.5, RESULTS, CONTEXT and the abstract now all tell this one story.

**Next action.** Two gates remain before the abstract is submission-ready, and
neither is a run. (1) Fill `human_labels.csv` and add a validated
`ClassifierScorer` as the third rater: judge agreement is 0.13 / 0.31 / 0.30 /
0.15 across the four models, and the abstract's controllability claims rest on
the judge. Until objection 14 is answered or explicitly scoped in Limitations,
do not treat the controllability numbers as final (noted in ABSTRACT.md). (2)
Write the paper body to the arc the abstract now states. A fifth model family
stays optional upside; if run, attach the four existing result datasets so
`--aggregate-only` covers all five, or rebuild the map locally as was done here.

---

## 2026-09-04 -- Kaggle, nbfinal -- JUDGE VALIDATED AGAINST A HUMAN. Objection 14 answered.

**Environment.** Repo at `6aee1df`, cloned public. No dataset attached and no
HF token needed: the labelled sheet is committed and the judge
(`Qwen/Qwen2.5-1.5B-Instruct`) is ungated. fp16 via `--no-4bit`. Whole run 96 s,
of which 25 s was the 3 GB model download.

**Command.**

```bash
!python scripts/score_human_labels.py human_labels_scored_100-v2.csv \
    --judge-model Qwen/Qwen2.5-1.5B-Instruct --no-4bit \
    --out judge_agreement.json
```

**Headline.** Human vs the fixed logit judge over 100 hand-labelled outputs
spanning all ten concepts:

| | pooled | within-concept |
| --- | --- | --- |
| human vs logit judge | +0.429 | **+0.326** |
| human vs lexicon | +0.461 | +0.184 |

The within-concept column is the one to report. The logit judge nearly doubles
the lexicon scorer there, which is the first direct evidence the primary
instrument tracks a human on the quantity controllability is built from.

**The positive control validates.** `sentiment` alpha = +0.756, human sd 0.247
against judge sd 0.304. Where behaviour actually moves under steering, the judge
follows the human closely. `sycophancy` +0.539.

**`topic_science` is clean, and that is the result that matters most.** Judge sd
0.020, the lowest of any concept and below the human's 0.047. The judge invents
no movement there, so the near-zero controllability that puts `topic_science` in
the danger zone on Mistral and Gemma is not a judge artifact. The single
confirmed dissociation in the study survives the check.

**`refusal` does not survive it.** The human scored all ten sampled outputs
identically at 0.000; the judge spread them over 0.159. Whatever the judge reads
there, a careful human does not see. Since noise adds apparent movement, the
measured controllability for `refusal` (0.033 to 0.047) is more likely an
over-estimate than an under-estimate and its CI wider than it should be, and
that CI is the only reason `refusal` misses P6. This does **not** license moving
`refusal` into the danger zone: that is a post-hoc argument from a limitation
and would need a re-measurement, not a re-reading. Same pattern on `honesty`,
`certainty` and `rudeness`.

**Method note kept deliberately.** Pooled alpha misled this study twice. The
first labelled sheet read +0.573 pooled and +0.022 within, because its sampling
(first 100 generations in curve order) left almost no within-concept variance:
56 of 100 rows were duplicates and 4 of 10 concepts including the positive
control were absent. `stratified_label_sample` fixed the sampling and
`score_human_labels.py` now prints both figures plus a per-concept variance
diagnosis, so the artifact cannot be read as a result again.

**Caveat on the sheet.** Intra-rater alpha was +1.000 across 10 deliberate
repeats separated by 38 rows or more. Clean, but perfect consistency cannot be
distinguished from the rater recognising the repeats, so report it with that
caveat or not at all.

**Next action.** Objection 14 is answered rather than outstanding, with the
`refusal` limitation stated. What remains before the abstract is writing, not
measurement: fold the judge-validation result into the paper's Limitations and
into the `refusal` boundary-case discussion. A validated `ClassifierScorer` as a
fourth rater and a larger labelled sample are upside, not blockers.

---

## 2026-09-14/15 -- Kaggle, T4 x2, validation runs (nb10)

**Environment.** Kaggle T4 x2, internet on, `HF_TOKEN` attached. Repo cloned
from the `iclr` remote. Notebook `notebooks/run_kaggle_validation.py`, three
stages sharing one loaded model: A ground truth, B re-judge, C k sensitivity.

### Session 1 (all three models, killed at the 12-hour wall)

Ran `run_all()` over Qwen, Mistral and Llama in one session. Got through two
models and died partway into Llama's stage A.

| | Qwen | Mistral | Llama |
|---|---|---|---|
| A ground truth | ok | ok | killed |
| B re-judge | **OOM** | **OOM** | not reached |
| C k sweep | ok | ok | not reached |

**What broke, and it was our bug.** `JUDGE_MODEL` was a 7B in fp16 and
`_judge_scorer` guarded only the *load*. On a 15GB T4 the 7B loads -- it
reported 14.28 GiB allocated of 14.56 -- and then dies on the first forward
pass with 14.81 MiB free. The load is not where a judge that barely fits fails.
Stage B was lost on both models.

**Second mistake:** stages ran A, B, C, so the expensive stage went first and
the third model produced nothing at all. Stage A is about five hours per model
at thirty evaluation prompts over a four-layer band; three models never fit in
twelve hours.

**Third:** session 1's stage A output for Qwen and Mistral was never
downloaded, and Kaggle wipes `/kaggle/working`. Those curves are gone. Only the
summary controllability printed in the log survives, which is not enough for
signed areas or intervals.

**Fixed in `6a2a0c2`:** judge defaults to 3B, an out-of-memory *during* the
stage drops to the smaller judge and retries once, stages run cheapest first
(C, B, A), and `run_all` defaults to one model.

### Session 2 (Llama, complete)

`run_all(["meta-llama/Llama-3.1-8B-Instruct"])`. All three stages ok. Output in
`validation_output/`.

**Stage A, and this is the result that matters.** With the judge removed the
two metrics stop disagreeing:

| concept | abs | signed | directional share |
|---|---|---|---|
| gt_french | 0.0047 | +0.0047 | 1.00 |
| gt_length | 0.0113 | +0.0112 | 0.99 |
| gt_uppercase | 0.0273 | +0.0265 | 0.97 |
| gt_digits | 0.0010 | -0.0003 | 0.29 |

Against a median directional share of 0.387 on the judge-scored concepts. The
divergence between absolute and signed area is a property of the judge, not of
steering.

No sign resolves (0 of 4), and the curves say why. The response is flat across
the grid and then moves at the largest usable coefficient:

```
gt_uppercase  0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.34
coefficient   -3.0 -2.0 -1.0 -0.5  0.0 +0.5 +1.0 +2.0 +3.0
```

A tenfold behavioral change integrating to an area of 0.027. `gt_french` is the
same shape, flat at 0.00 until +3 and then 0.06. The effect is real and sits at
the edge of the grid, so an integrated area over a mostly-flat sweep dilutes it.
That is a defect in the summary and in the grid range, not evidence that nothing
moved.

**Two design faults in the ground-truth concepts, recorded so they are not
rediscovered.** Three of four baselines sit at 0.00 to 0.03, so there is no
headroom downward and the negative arm cannot show anything. And the
coefficient grid stops exactly where the effects begin.

**Stage B.** Judge `Qwen/Qwen2.5-3B-Instruct` against the 1.5B used everywhere
else. Every sign preserved; magnitudes move a lot.

| concept | abs 1.5B -> 3B | signed 1.5B -> 3B | sign |
|---|---|---|---|
| topic_science | 0.011 -> **0.118** | +0.007 -> +0.021 | kept |
| refusal | 0.033 -> 0.097 | -0.016 -> -0.097 | kept |
| certainty | 0.049 -> 0.097 | -0.039 -> -0.014 | kept |
| sentiment | 0.101 -> 0.083 | +0.020 -> +0.083 | kept |

`topic_science` moves elevenfold and is the danger zone's only confirmed
occupant. 0.118 is well above the 0.05 threshold that put it there, so that
occupancy is judge-dependent. Nothing resolves under the new judge either.

**Stage C.** Output overlap rises steeply with k and its ranking is not stable.
On Llama the rank correlation between k=64 and k=2048 is +0.297; on Qwen it was
-0.137, an effective inversion. At k=2048 every concept projects similarly
(0.65 to 0.74) so the measure stops discriminating, and the partial correlation
attenuates from -0.457 at k=64 to -0.133 at k=2048. Stage C computes directions
at the mid-network layer rather than each concept's selected layer, so absolute
values are not identical to the paper's (mean absolute difference 0.016,
Spearman +0.601 at k=512); the claim it supports is about ranks.

### Next action

Run **Qwen** next, full session, `run_all(["Qwen/Qwen2.5-7B-Instruct"])`. It
needs stage B, which OOMed, and stage A, which was lost with session 1's
working directory. About six hours with the new stage order. Then Mistral on
the same basis. Download `validation_output` before the session expires.

---

## 2026-09-15/16 -- Kaggle, validation reruns (Qwen, Mistral) on the extended grid

**Environment.** Two accounts in parallel, commit `5c80f1f` or later, verified
by `preflight()` printing the commit and grid before loading anything. Stage A
on the 13-point extended grid; stage B on the 9-point default for comparability
with the 1.5B judge. Qwen and Mistral are ungated so neither session needed a
secret.

Both models: all three stages ok. Outputs consolidated into
`validation_output/`. Llama's stage A there is still the 9-point grid and is
flagged as such; its B and C are current.

### Stage A: the diagnostic is validated

Three ground-truth signs resolve, on two models, **all positive and all in the
intended direction**. Nothing resolves the wrong way anywhere.

| concept | model | signed | 95% CI |
|---|---|---|---|
| gt_uppercase | Mistral | +0.181 | [+0.147, +0.215] |
| gt_french | Mistral | +0.057 | [+0.040, +0.075] |
| gt_french | Qwen | +0.054 | [+0.033, +0.076] |

`gt_french` resolving on both models is the replication that matters: steering
makes the model speak French, fluently, and the signed metric recovers the known
direction with an interval excluding zero.

The extended grid earned its cost on Mistral. `gt_uppercase` was invisible on
the old grid and now runs 0.03 at baseline to 0.55 at +2, 0.77 at +3 and 0.75 at
+4, all under the fluency ceiling, breaking only at +5. That is a
twenty-five-fold behavioral change that the 9-point sweep never reached.

**Directional share on ground truth: median 0.983**, or 0.993 across the nine
points whose absolute area exceeds 0.005. Against 0.387 on the judge-scored
concepts. With the judge removed the two summaries coincide; the divergence the
paper is built around is the judge's.

Two concepts still fail to resolve and both reasons are legible rather than
mysterious. `gt_digits` moves only past the ceiling: on Qwen it reaches 0.21 at
+4 and 0.55 at +5, both broken on perplexity. `gt_length` has no consistent
response on any model.

### Stage B: a quarter of the signs flip

| concept | Qwen | Mistral | Llama |
|---|---|---|---|
| topic_science | 0.019 -> 0.260 | 0.024 -> 0.201 | 0.011 -> 0.118 |
| refusal | 0.047 -> 0.208, **flip** | 0.038 -> 0.056, **flip** | 0.033 -> 0.097 |
| certainty | 0.123 -> 0.153 | 0.113 -> 0.056, **flip** | 0.049 -> 0.097 |
| sentiment | 0.215 -> 0.345 | 0.163 -> 0.257 | 0.101 -> 0.083 |

Three of twelve signs flip when the judge changes from 1.5B to 3B. The earlier
reading that signs are judge-robust came from Llama alone, where none flipped,
and is withdrawn.

`topic_science` rises by eight to fourteen times on every model and lands at
0.118 to 0.260 against the 0.05 threshold that made it the danger zone's only
confirmed occupant. Under the larger judge it is four to five times clear of the
zone on all three models. The danger zone does not survive a change of
instrument.

`sentiment` resolves positive under both judges on Qwen and Mistral, so the
positive control holds where it held before.

### Next action

Gemma, full run, `run_all(["google/gemma-2-9b-it"])` on the account holding the
HF token. Its sentiment control is the reason it is withheld and that decision
rests on an unresolved sign, CI $[-0.107, +0.002]$ under the 1.5B judge. If the
3B judge resolves it positive the withholding was a judge artifact and the
40-point sample comes back. Then Llama stage A on the extended grid,
`stages="A"`, since its B and C are current.

---

## 2026-09-16 -- Kaggle, Llama stage A on the extended grid

`run_all(["meta-llama/Llama-3.1-8B-Instruct"], stages="A")` at commit
`5c80f1f`, which is before the gauntlet fix, so the six discarded interventions
were still being paid for. It finished anyway in 24904s (6.9h), inside the wall
clock. Output in `validation_output llama stage a/`, consolidated into
`validation_output/`, replacing the superseded 9-point files.

**All twelve ground-truth points are now on the same 13-point grid** across
Qwen, Mistral and Llama, with the same band and the same gauntlet behaviour, so
they sit in one table without a protocol caveat.

The extended grid changed Llama's numbers substantially:

| concept | 9-point | 13-point |
|---|---|---|
| gt_uppercase | 0.027 | 0.074 |
| gt_french | 0.005 | 0.040 |
| gt_length | 0.011 | 0.013 |
| gt_digits | 0.001 | 0.001 |

### gt_french replicates on all three models

| concept | Qwen | Mistral | Llama |
|---|---|---|---|
| gt_french | +0.054 * | +0.057 * | +0.040 * |
| gt_uppercase | +0.017 | +0.181 * | +0.073 * |
| gt_length | -0.015 | +0.003 | +0.012 |
| gt_digits | +0.008 | +0.001 | -0.000 |

`*` marks a signed area whose interval excludes zero. Five of twelve points
resolve, **every one of them positive**, and not a single point anywhere in the
study resolves in the wrong direction. `gt_french` resolves on all three models
and `gt_uppercase` on two of three.

Median directional share is 0.993 over the ten points whose absolute area
exceeds 0.005, and 0.978 over all twelve, against 0.387 on the judge-scored
concepts.

`gt_french` tripped the constant-readout guard on three sub-runs, which is the
guard behaving correctly: those layers produced no French at any coefficient
while others did, and the concept still finishes at 0.040. The warning text
called it a judge failure, which it is not; fixed in `9232d07`.

### Next action

Gemma, full run, on the second account. Nothing from the killed Gemma session
was kept, so it needs all three stages. With the gauntlet disabled in stage A
that is roughly six hours rather than the twelve-plus that session was heading
for.

---

## 2026-09-16 -- Kaggle, Gemma full validation run

`run_all(["google/gemma-2-9b-it"])` at commit `9232d07`, the first run with the
gauntlet disabled in stage A. All three stages ok in 18234s (5.1h), against the
twelve-plus hours the previous attempt was heading for. The reworded
constant-readout NOTE appeared as intended, describing a rule rather than a
judge.

**All sixteen ground-truth points across four models are now on the same
13-point grid.**

### The withholding is confirmed, not overturned

This run existed to test whether Gemma was withheld because of the judge.
It was not.

| | 1.5B judge | 3B judge |
|---|---|---|
| sentiment, absolute | 0.110 | 0.167 |
| sentiment, signed | $-0.039$ | $-0.056$ |
| 95% CI | $[-0.107, +0.002]$ | $[-0.198, +0.103]$ |

Unresolved under both judges, and negative under both. Gemma never demonstrates
directional sentiment steering regardless of instrument, so the preregistered
positive control withholds it on a reproducible basis rather than on one
borderline number. The paper's three-way withholding table stands and its middle
row remains the one we report.

### Ground truth says the same thing with no judge at all

| concept | Qwen | Mistral | Llama | Gemma |
|---|---|---|---|---|
| gt_french | +0.054 * | +0.057 * | +0.040 * | +0.001 |
| gt_uppercase | +0.017 | +0.181 * | +0.073 * | +0.009 |
| gt_length | -0.015 | +0.003 | +0.012 | -0.006 |
| gt_digits | +0.008 | +0.001 | -0.000 | +0.000 |

Nothing resolves on Gemma, and its median absolute area is 0.0040 against 0.0195
for the other three. With the judge removed entirely, Gemma is still the model
that does not move. That is independent of the control, of the judge, and of the
behavioral question, and it is the strongest evidence the study has that the
withholding tracks the model rather than the instrument.

Across all sixteen points, five resolve and every one is positive. No point
anywhere in this study resolves in the wrong direction.

### Stage B: the judge swap again

| concept | 1.5B -> 3B | sign |
|---|---|---|
| topic_science | 0.007 -> 0.181 | kept |
| refusal | 0.038 -> 0.139 | kept |
| certainty | 0.054 -> 0.153 | kept |
| sentiment | 0.110 -> 0.167 | kept |

`topic_science` rises twenty-six fold, the largest jump of the four models, and
lands at 0.181 against the 0.05 danger-zone threshold. The danger zone is now
dead on all four models.

All four Gemma signs hold. Across the sixteen re-judged points the flip count is
three: `refusal` on Qwen, `refusal` and `certainty` on Mistral.

### Next action

No further compute is required for the current claims. The paper still argues
the pre-validation story and is the blocking item.

---

## 2026-09-17 -- Kaggle, stage D first attempt: not a faithful replication

`run_all(["Qwen/Qwen2.5-7B-Instruct"], stages="D")`, which loads `gpt2-xl`.
Stage ok, output in `stage_d_published/`. The readout returned $0.000$ at every
coefficient.

**This run does not test the published result and is not reported as though it
did.** Inspecting the samples was what showed it: at $\alpha=+1$ the
continuation is ordinary coherent text, near-identical to the unsteered one, and
the intervention plainly did nothing rather than doing something that missed.
Three differences from activation addition as published account for it.

**Injection position.** This harness adds the vector at every token position,
including the ones being generated. Activation addition injects only at the
positions its contrast prompt occupied, so the vector shapes the start of the
continuation and then stops. Adding at every position is a different and much
stronger intervention, which is consistent with what the sweep showed: nothing
at the usable coefficients and degenerate repetition from $\pm 2$ outward, so
the fluency ceiling cut the usable range to $[-1, +1]$.

**Layer.** The pipeline steers at a layer chosen from the probe, here 14, with
the probe selecting 23. The published setting is layer 6.

**Coefficient scale.** Our $\alpha$ is in residual-RMS units on a unit-normalised
direction; theirs multiplies the raw activation difference. The two are not the
same quantity and our swept range does not bracket theirs.

So the preregistered comparison in `lbi/published.py` cannot be made from this
run. Neither of its two committed sentences applies, and writing "a published
steering result does not survive a directional reading" on this evidence would
be the exact overclaiming the paper exists to document.

**Fixed:** `SteeringSpec` now takes `positions`, limiting the intervention to a
number of leading token positions, defaulting to `None` for the existing
behaviour so every reported number is unaffected. Tests pin both forms. The
remaining two differences, layer and coefficient scale, are configuration rather
than code and are addressed before the next attempt.

### Next action

Re-run stage D with the published layer, position-limited injection, and a
coefficient range that brackets the published magnitude rather than our RMS
units. Verify `ACTADD_SETTING` against the paper first.

---

## 2026-09-17 -- Stage D, third attempt: blocked on a missing primitive

Layer 6, injection limited to the 3 positions the contrast prompt occupies, grid
widened to $\pm 20$. Stage ok; output in `stage_d_published/`. The readout is
$0.000$ at every coefficient.

The generations say why. At $\alpha=+15$ the continuation is near-identical to
the unsteered one and perplexity moves from $3.2$ to $3.4$, so the positive arm
does essentially nothing even at fifteen RMS units. The negative arm degenerates
into repetition from $-6$ outward. The intervention is reaching the model and it
is not carrying the behavior.

**What is still different from the published method.** Activation addition's
steering vector is position-wise: the two contrast prompts are padded to equal
length, $h(A) - h(B)$ is a $(\text{seq\_len}, d_\text{model})$ matrix, and each
row is added at its own position. This harness collapses that to one vector by
last-token pooling and adds the same vector at each of the first few positions.
That is a different intervention, not a different setting, and it is the most
likely reason the direction does not carry weddings.

**Blocked, and recorded as blocked.** Closing it needs `SteeringSpec` to accept
a matrix and `_apply` to broadcast per position, which touches the code path
every other number in the study depends on. Three sessions have gone into this
and the honest status is unchanged: the preregistered comparison cannot be made,
and neither of its two committed sentences applies.

### Next action

Item 3 in `PLAN_MAIN.md`: concepts 10 to 20+, ground-truth concepts 4 to 10. No
new primitives, and it is what makes the mechanism question testable.

## 2026-09-18: ten concepts at a single layer, three models

Qwen, Mistral and Llama each ran all five stages on commit `6c9987f`
(single layer, extended grid, 3900 generations for stage A). Sessions were
independent; stage B and stage C outputs are byte-identical to the earlier
run on all three models, so those stages are deterministic and the merge
overwrote nothing. Gemma is still outstanding.

The 30 ground-truth results are merged into `validation_output/results_groundtruth/`.
The 16 earlier banded results stay in `results_groundtruth_banded/` as the
protocol comparison. Session logs are kept under `validation_output/logs/`.

### What the re-sweep was for

Baselines now span 0.000 to 0.978, so baseline position varies independently
of concept identity. The headroom explanation for a signed area failing to
resolve does not survive:

  - headroom vs whether the sign resolves: rho +0.006, p 0.976.
  - room available to an absolute deviation, max(baseline, 1-baseline), vs
    absolute area: rho -0.576, p 0.001. More room, *less* measured effect,
    which is the reverse of what a ceiling artifact predicts.
  - `gt_lowercase` (baseline 0.968) and `gt_uppercase` (baseline 0.032) read
    the same surface property from opposite ends, so they are matched on room
    and differ only in which side it is on. Absolute area within each model:
    Llama 0.0057 vs 0.0044, Mistral 0.0253 vs 0.0199, Qwen 0.0032 vs 0.0032.
  - Kruskal on absolute area by concept H 15.47 p 0.079; by model H 0.36
    p 0.836.

Directional share median 1.000 (mean 0.804) against 0.387 on the
judge-scored concepts.

### The cost of dropping the band, stated plainly

4 of 30 signs resolve, against 5 of 16 banded. Per model: Mistral 3/10,
Qwen 1/10, Llama 0/10. `gt_french` resolved on three of four models under the
band and resolves only on Mistral at a single layer. This is expected, because
banded controllability is a maximum over four layers and single-layer
controllability is one of those four, so it reads lower by construction. Both
tables should be reported rather than whichever one reads better.

### Stage A caveat carried from the log

The P9 positive control was not written during stage A, because `sentiment`
is not a ground-truth concept. The control does exist from stage B for all
three models. Nothing is withheld on this basis.

### Stage D ran and is not valid

`D_published` reports ok on all three sessions and the readout returned 0.0
for all 130 generations at every coefficient, including ActAdd's own setting
of coefficient 1.0. That is not a result about ActAdd. It is the harness:

1. `lbi/steering.py` generates greedily (`temperature=0.0`). ActAdd's wedding
   demonstration is a sampling result on a base model. Greedy gpt2-xl loops,
   and the saved samples show exactly that: the unsteered completion repeats
   "I'm going to go to the bathroom" and coefficients +1 through +15 are
   byte-identical to it. A coefficient-1.0 nudge cannot move an argmax path.
2. `SteeringSpec` requires a unit-norm direction and applies
   `coeff * layer_RMS * direction`. ActAdd adds the raw difference vector,
   whose norm the log reports as 175.125, at coefficient 1.0. The two
   coefficient scales are unrelated, so the +-20 grid is not shown to contain
   their operating point.
3. gpt2-xl has no chat template, so the raw prompt form was used. That part
   is correct.

To make stage D a replication it needs a raw-vector mode, the RMS-equivalent
of coefficient 1.0 computed at runtime and forced into the grid, and sampling
for this stage only with the deviation stated. The positions primitive that
previously blocked the stage does work: the log confirms injection at the
first 3 positions. Item 2 stays open.

### Stage D fixed

Three changes, each pinned by tests in `tests/test_raw_units.py`.

**Coefficient units.** `SteeringSpec` gained `unit_mode`, defaulting to
`"rms"`, which is what every reported number uses and is unchanged. The new
`"raw_norm"` mode scales by an explicit `raw_scale` instead of the layer's RMS
norm, so with `raw_scale` set to the norm of the raw activation difference,
coefficient 1.0 adds exactly that difference. That is the published setting
itself rather than a guess at the conversion, and the grid is now multiples of
it: -4 to +4 with 1.0 on the grid. The old approach swept +-20 in RMS units
against a raw norm of 175, so it was never shown to contain their operating
point. `raw_norm` is refused for `clamp` and `ablate`, whose targets are
defined in RMS units and would otherwise be measured against the wrong ruler.

**Decoder.** `run_steering` gained `temperature` and `n_samples`, both
defaulting to the greedy single-draw path. Stage D now samples at temperature
1.0 with 5 draws per prompt, because the published demonstration is a sampling
result and greedy gpt2-xl loops. `n_samples > 1` under greedy decoding raises
rather than averaging the same draw five times into a falsely tight interval.

**A positive control gating the null.** The preregistration in
`lbi/published.py` now covers the outcome the first two attempts actually
produced, which was neither preregistered branch: a readout constant at zero.
Before any null is reported the sweep must reproduce the effect somewhere,
meaning the readout rises above baseline by more than that point's interval at
some usable coefficient. If it does not, the result is "not replicated here"
and implicates the harness, not the method. Stage D prints PASSED or FAILED
and records `positive_control_reproduced` and `behavior_at_published_coeff` in
the JSON. A flat zero is the one outcome that reads like a strong finding
while being the absence of one, which is why it now has a gate instead of a
footnote.

Cost: 13 coefficients x 10 prompts x 5 samples = 650 generations on gpt2-xl,
a few minutes. The setting in `ACTADD_SETTING` and the decoding in
`ACTADD_DECODING` are still recorded from our reading of the paper and still
need checking against the source before the number is reported.

## 2026-09-19: Gemma, and a claim that does not survive

Gemma ran all five stages on `ad429c1`, so it is the first session to exercise
the fixed stage D. Stage B and C came back byte-identical to the earlier run,
as they did for the other three. The ten ground-truth results are merged, so
`validation_output/results_groundtruth/` now holds 40 points across 4 models.

### The headroom result holds at four models

  - headroom vs whether the sign resolves: rho +0.006, p 0.970 (40 points).
  - room available to an absolute deviation vs absolute area: rho -0.521,
    p 0.001, still the reverse of what a ceiling artifact predicts.
  - Kruskal on absolute area by concept H 21.54 p 0.0104, by model H 0.33
    p 0.954. With the fourth model the concept effect reaches significance and
    the model effect is nowhere near it.

6 of 40 signs resolve, all positive. Directional share median 0.983 against
0.387 on the judge-scored concepts.

The designed pair behaves as intended on three models and not on Gemma:

    model          lowercase (base 0.97)   uppercase (base 0.03)
    Llama-3.1-8B            0.0057                 0.0044
    Mistral-7B              0.0253                 0.0199
    Qwen2.5-7B              0.0032                 0.0032
    Gemma-2-9b              0.0417                 0.0043

Gemma splits the pair by a factor of ten. That is not headroom either, since
the two are matched on room, and within Gemma the high-room arm of uppercase
gives +0.002 while the high-room arm of lowercase gives +0.086. It is a
concept-by-model interaction, and it should be shown rather than averaged away.

### The Gemma withholding claim was a protocol artifact

This is the one that matters, because it reverses a finding.

                      BANDED (4 concepts)      SINGLE LAYER (same 4)
    Gemma-2-9b        0.0040   resolved 0/4    0.0040   resolved 1/4
    Llama-3.1-8B      0.0262   resolved 2/4    0.0030   resolved 0/4
    Mistral-7B        0.0342   resolved 2/4    0.0132   resolved 1/4
    Qwen2.5-7B        0.0195   resolved 1/4    0.0021   resolved 0/4

Under the band Gemma's median sits five to eight times below the other three,
which is what "Gemma withholds" rested on, and it held under two judges and
with no judge. Restricted to the same four concepts at a single layer, Gemma is
not the lowest: its median is above Llama's and Qwen's, and it resolves as many
signs as Mistral. Across all ten concepts Kruskal by model gives p 0.954.

Banded controllability is a maximum over four layers. The gap was the other
three models benefiting from that maximum more than Gemma did, not Gemma
withholding. The claim does not survive and must not be reported. Gemma was
already withheld from the paper, so nothing published depends on it, but the
reason recorded for withholding it was wrong.

### Stage D, on the fixed harness

    raw difference norm: 175.125
    coeff 1.0 == the published setting (raw_norm units)
    decoding: temperature 1.0, 5 samples per prompt
    grid (multiples of their coefficient): [-4 ... 1.0 ... 4]
    pub_wedding  ctrl 0.0000  perplexity > 2.0x baseline at coeff -0.5
      baseline 0.0007, at their coeff 0.0000
      positive control FAILED

The units and the decoder are fixed and the control still fails, which is the
control doing its job. The remaining cause is the one that blocked this stage
originally, in `df1adba`, and adding `positions` did not resolve it:

`capture_cached` pools with `pooling="last"`, so the direction is the
difference at the final token, one vector. "Weddings" is three tokens and " "
is one; activation addition pads them, differences them position by position,
and injects row i at position i. Ours adds the same pooled vector at all three
positions. `positions` controls which positions receive a vector, never gives
each its own. Stage D needs a position-wise steering matrix, exactly as
`df1adba` said, and treating that blocker as resolved was an error.

Stage D is therefore still not a replication and nothing from it is
reportable. What is now true is that the two deviations that were maskable are
gone, and the remaining one is identified precisely rather than suspected.

### Packaging bug

The full-run notebooks zipped `results_groundtruth`, `results_rejudge` and
`results_ksweep` but not `results_published`, so four sessions ran stage D and
none returned its JSON; only the log survived. Fixed in
`scripts/make_kaggle_notebooks.py` and the notebooks are regenerated.

## 2026-09-19: the position-wise matrix

`SteeringSpec.direction` now accepts a `(P, d_model)` matrix whose row i is
applied at absolute position i, with every row unit-norm, and `raw_scale`
accepts one scale per row because each position of a raw activation difference
has its own norm. `lbi.extraction.capture_positionwise` returns per-position
activations instead of pooling, padding the shorter contrast prompt on the
right with its own token rather than a pad token, so every position holds a
real token. `lbi.published.actadd_direction` assembles the two into the
published contrast and drops trailing positions where the padded prompts agree
and the difference is zero. Stage D uses it.

This is what `df1adba` said was needed and what `positions` did not provide.

### A second bug, which the matrix work exposed

`_apply` decided which positions to touch from the column index of the tensor
it was handed. That is right for the prompt pass and wrong for every pass
after it: under the KV cache each generated token arrives as its own length-1
tensor whose column 0 is not position 0, so a "first three positions"
intervention was steering the prompt *and then every generated token*. The
hook now carries a per-layer count of positions seen and passes an offset, so
the intervention lands on absolute positions.

No number in the paper is affected. `positions=None` is the default, adds at
every position by design, and never consults the offset; the only caller that
passed a position limit was stage D, whose results were never reportable. The
default path is pinned by a test for exactly this reason.

Worth stating plainly: the earlier claim that position limiting "confines the
intervention to the prompt, which is what activation addition does" was in the
`_apply` docstring and was false for the whole of generation. It had a test,
`test_limited_touches_only_the_leading_positions`, which passed because it
only ever checked a single forward pass.

### Where stage D stands

Three deviations from the published method are now fixed: the coefficient
scale, the decoder, and the direction. The positive control from the
preregistration decides whether that is enough, and it has not been run on
the corrected direction yet. The outstanding item is unchanged and is not a
code problem: `ACTADD_SETTING` and `ACTADD_DECODING` are still recorded from
our reading of the paper rather than verified against it, and coefficient 1.0
now being exactly their setting makes getting that setting right matter more,
not less.

271 tests pass.

## 2026-09-19: stage E, thirty eval prompts

The judge-scored concepts ship six eval prompts each and the ground-truth
concepts ship thirty. The study's headline comparison is between them, so it
confounded the readout with the sample size: ground truth resolves 6 of 40
signs and the judge-scored concepts 4 of 12, but at thirty prompts against six.
A signed area whose interval is wider for want of prompts is not evidence about
judges.

`lbi/concepts.py` now carries twenty-four more prompts for each of the four
re-judged concepts, in the register each set was written in. The sentiment
prompts stay neutral descriptions rather than invitations to an opinion,
because a judge scored unsteered output at 0.967 on the latter and left no
headroom. The refusal prompts stay at the level of explaining how something
works. The certainty prompts all invite a forecast, which is what gives hedging
somewhere to go.

The original six are a strict prefix of the thirty, as the extended coefficient
grid is a strict superset of the default one, so the six-prompt numbers are
recoverable by subsetting and stage B's outputs stay on disk as the comparison.
Stage E writes to `results_prompts30`, uses the same judge and the same grid as
stage B, and raises rather than running if any concept still has six prompts,
because that would silently produce a duplicate of stage B under a new name.

Cost: 1080 generations against stage B's 216, so roughly five times, about
three hours.

Both outcomes are worth reporting and neither is the better one. If signs start
resolving at thirty prompts then the judge-scored null was underpowering, and
the paper has to say so and make a smaller claim. If they still do not resolve
while the intervals visibly tighten, the difference is the judge rather than
the sample, which is the claim the paper wants and cannot currently support.

`notebooks/kaggle/prompts30_qwen.ipynb` runs stage D then stage E in one
session. 291 tests pass.

## 2026-09-20: stage E ran, stage D still fails its control

### Stage D, on the corrected direction

    raw difference norm per position: 81.94, 111.06, 87.91
    baseline 0.00073   at their coefficient 0.00036
    controllability 0.00019   ceiling: no breakage in swept range
    positive control FAILED

All three harness deviations are now fixed and the readout is still flat. The
decoder fix is visibly working: the samples are varied rather than looping and
baseline perplexity is 13.8 against 3.2 under greedy decoding. The direction is
genuinely position-wise, with three distinct per-position norms. Nothing breaks
anywhere in the swept range, so the intervention is gentle rather than
destructive, unlike the RMS-unit version that broke at -0.5.

That leaves `ACTADD_SETTING` itself, which has never been checked against the
paper. Stage F is added to test the most likely way of mis-recording it.

### Stage E: thirty eval prompts

The stage ran and wrote every result. It then reported `failed` because the
summary loop called `.get` on `ConceptRun` objects, which are dataclasses and
not dicts. The data was already on disk; only the summary line was lost. Fixed.

    Qwen2.5-7B, same judge, same grid.

                     6 prompts (stage B)        30 prompts (stage E)
    certainty     -0.056 [-0.117, +0.008]   +0.108 [-0.056, +0.232]
    refusal       -0.042 [-0.148, +0.000]   -0.061 [-0.160, +0.106]
    sentiment     +0.262 [+0.106, +0.374] * +0.188 [+0.041, +0.290] *
    topic_science -0.115 [-0.298, +0.058]   +0.017 [-0.169, +0.193]

    resolved 1/4 -> 1/4
    mean CI width 0.2245 -> 0.2914, ratio 1.30; pure sampling would give 0.45
    narrower at thirty: 1 of 4
    signs flipped: certainty and topic_science

Five times the prompts and the intervals did not tighten. Two of four signs
flipped under a change that should only reduce noise. Taken at face value this
is the result the paper wants and could not previously support: the
judge-scored null is not underpowering, and it lines up with the existing
re-judge evidence, where a different judge flipped 3 of 16 signs. Two
independent perturbations, changing the judge and changing the prompt sample,
both move signs and neither tightens anything.

### Why that is not yet reportable

The expanded prompt set moved the baselines: sentiment 0.833 to 0.400,
topic_science 0.833 to 0.500, refusal 0.667 to 0.733, certainty unchanged. The
shift is in the right direction, since the six-prompt sentiment set was close
to saturating the judge and the concept's own docstring warns about exactly
that. But it means the two runs differ in their operating point as well as in
their prompt count, so this is not a clean comparison and the CI-width ratio
cannot carry the claim on its own. Four concepts on one model is thin for that
ratio in any case.

The clean version is to subset the thirty-prompt run to its first six, which
holds the run, the judge and the generations fixed and changes only the prompt
count. That was impossible offline: `run_steering` computes per-prompt scores
and `jsonable` dropped them, so nothing downstream could re-aggregate. Fixed by
adding `DosePoint.scores`, one score per generation in prompt order. Stage E
needs one re-run to produce them, which is what `round2_qwen.ipynb` does.

### Stage F: the contrast spelling diagnostic

Six spellings of the contrast, swept on a coarse wide grid, scored by the same
rule and judged by the same positive control. GPT-2 gives "Weddings",
" Weddings" and " weddings" different token ids, so a mis-recorded leading
space is a different direction.

This is deliberately not a search for a setting that works. If no spelling
reproduces the effect, the stage D null is robust to the most likely recording
error and that is what gets reported. If one does, it identifies something to
verify against the paper before anything is reported, and a variant selected
because it worked is not evidence of the published setting. The
preregistration governs stage D and is untouched.

291 tests pass.
