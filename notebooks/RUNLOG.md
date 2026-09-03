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
