# Context

Start here. This is the orientation document for anyone -- or any session --
picking the project up cold.

## What the study asks

If a concept is clearly encoded in a model's activations, does that mean we can
causally control the model through that representation?

The claim under test is that it does not: **readability and controllability can
dissociate.** Readability is held-out linear-probe AUROC. Controllability is the
area under a steering dose-response curve up to a fluency ceiling. Plot every
concept-model pair in that plane and the high-readability / low-controllability
corner is the danger zone.

A probe interrogates the *encoder*. Steering interrogates the *decoder*. Nothing
ties class separation to downstream causal sensitivity, which is why a gap
should be expected rather than merely hoped for.

## Where the project stands (2026-09-04)

| | |
| --- | --- |
| Code | Experiments 1-4 complete, 143 CPU tests, synthetic demo passes |
| Real model runs | **All four models complete** (Qwen2.5-7B, Mistral-7B, Llama-3.1-8B, Gemma-2-9b), 40 points, all controls pass. See RESULTS.md |
| Judges | Fixed logit judge (Qwen2.5-1.5B) + lexicon. **alpha 0.13 / 0.31 / 0.30 / 0.15 -- too low.** Human sheet unfilled |
| Concept set | 10 built, 8 clear the surface audit, 2 declared surface-confounded |
| Target | ICLR 2027 -- abstract **Sep 18 2026**, paper **Sep 25 2026** |

Data collection is effectively done. The remaining work is the abstract framing
and the human judge labels, not more runs. See "What is left" below.

The next action is always in [notebooks/RUNLOG.md](notebooks/RUNLOG.md). Read it
before proposing a run; append to it after one.

## The four documents and what each is for

| File | Role |
| --- | --- |
| `research_design_legible_but_immovable.md` | **Source of truth.** Section 12 logs every revision and why |
| `PLAN.md` | Preregistration (P1-P9), objection ledger, timeline, kill criteria |
| `README.md` | How to run it, and what the code enforces that is easy to get wrong |
| `notebooks/RUNLOG.md` | Append-only record of every session: commands, results, failures |

`PLAN.md` Part 2 carries an implementation-status table mapping each
preregistered point to the code enforcing it and the test pinning it. **A blank
enforcement cell is an unkept promise, not a detail** -- four points (P4, P6, P7,
P9) were once documented and silently unimplemented.

## Constraints that shape every decision

**No local GPU.** The development machine cannot run the pipeline. Runs happen
on Kaggle (T4 x2). Locally only `pytest tests/` (~60 s) and
`scripts/demo_synthetic.py` (~40 s) work, and between them they check every
metric path against planted ground truth.

**Kaggle Save Version re-runs every cell** under papermill. No cell can be
skipped in a committed notebook, so optional steps must degrade rather than
raise. `/kaggle/working` persists only when a version is saved.

**No training.** Directions come from cached activations three ways
(difference-of-means, probe weights, RepE reading vector). Nothing is fine-tuned.

**Sub-10B.** A stated limitation, in the abstract on purpose.

## Two honest caveats to carry into the paper

**The surface audit passes by construction.** Every non-confounded concept
scores exactly 0.500, the floor, because marker vocabulary is disjoint across
template families by build-time invariant. That is a check that the construction
succeeded, not independent evidence the probe reads semantics. What carries the
argument is the probe's own held-out-family AUROC, where it faces marker words
it has never seen.

**"Immovable" is earned, not assumed.** A concept that resists the default
intervention is *unsteerable under tested interventions*. Only one that survives
the six-intervention gauntlet is immovable, and even that is bounded by those
six rather than proving none exists. `gap_map.json` reports the two sets
separately.

## What the data says (nb7, 2026-09-04, four models, 40 points)

The pipeline works and the full study has run. The result is **not** the paper
that was planned, and the next decision is a framing decision rather than a
technical one. The 40-point map lives at `results nb7/combined_40point/`,
rebuilt locally from the nb6 and nb7 result files (the nb7 notebook aggregate
saw only its own two models; see the runlog).

**Two danger-zone occupants, both `topic_science`:** on Mistral and on Gemma,
each confirmed immovable against all six interventions. It is the least
controllable concept on all four models (0.019 / 0.024 / 0.011 / 0.007) and
survived the gauntlet on all four; Qwen and Llama place it just outside the zone
only because its readability CI dips below the 0.9 floor. But `topic_science` is
one of the two concepts declared surface-confounded before the run, so its
readability is the number the audit says not to trust. The claim has occupants
on two model families and can use none of them.

**The near-miss holds across all four.** `refusal` is not surface-confounded,
scores 0.047 / 0.038 / 0.033 / 0.038, and survived the gauntlet on Mistral and
Llama. It misses P6 only on the controllability CI upper bound (0.108 / 0.080 /
0.107 / 0.146, all above 0.05). This is a boundary case, not a chase: the CI
arithmetic says more eval prompts will not close it at feasible cost.

**H1 is now a moderate contradiction, not a null.** With 40 points partial rho
= -0.451, CI [-0.723, -0.041], which **excludes zero on the side opposite**
Section 2.5's prediction. More output overlap goes with *less* controllability,
not more. `topic_science` drives it: it is the highest-overlap and
least-controllable concept on all four models, the M1 spectator-feature picture
inverted. The honest arc is "the obvious geometric predictor points the wrong
way, and here is why," with a lexicality reading offered as the mechanism (see
Section 2.5's dated block). Both exploratory analyses stay null (p_BH 0.833) and
the ridge predictor is at LOO R^2 0.001.

**Two things blunt every number above.** Readability is saturated -- 31 of 40
points at exactly AUROC 1.000 -- so the primary x-axis has almost no variance
and the headline correlation is uninformative rather than null. And judge
agreement is 0.13 / 0.31 / 0.30 / 0.15, which leaves objection 14 unanswered.

## What is left

**No model needs re-running.** All four are complete, all results are on disk
and committed, and the 40-point map is correct. The H1 bug was a reporting bug
fixed in-place by re-aggregation, not a data bug, so nothing has to be
regenerated on a GPU.

Two things gate the paper, and neither is a run:

1. **Fill `human_labels.csv` (objection 14).** Judge agreement is poor on every
   model. This is the single weakest point and it needs ~100 hand labels, not
   GPU time. Pair it with a validated `ClassifierScorer` as the third rater.
2. **Write the abstract to the inverted-H1 arc.** Section 1 and Section 2.5 are
   already aligned to it; the abstract itself still needs writing before Sep 18.

Optional upside, not a blocker: a fifth model family would de-risk the H1 sign
and could dent the saturation problem. It is nice to have, not required, and it
does not change the code. If run, attach the four existing result datasets so
`--aggregate-only` covers all five at once -- the recurring failure has been the
nb6 result files not being attached at the path the copy cell expects, which
silently leaves the notebook aggregate on a subset. Rebuild locally if in doubt.

`load_model` selects `attn_implementation="eager"` automatically for Gemma-2
(verified in the code and in the outputs: Gemma generations are fluent and the
judge is not degenerate). That family soft-caps its attention logits and the
fused SDPA path ignores the cap, so without the guard the model would load,
generate fluent text, and be quietly wrong.

## Go / no-go

From `PLAN.md` Part 8, decided from data rather than hope:

- Positive control fails on two or more models → do not submit; the harness is
  wrong in a way that invalidates everything.
- Control works but no concept shows the dissociation -> pivot to the
  "predict steerability from geometry" framing in design Section 11.
- Dissociation holds -> write it.

**Where nb7 leaves this.** All four controls pass, so the first branch is
closed. The dissociation has two confirmed instances but both are the same
confounded concept, so the third branch is earned only with the confound stated.
The geometry pivot did not vanish; it inverted. H1 came back a moderate
contradiction (CI excludes zero, wrong sign), which is a stronger claim about
the field's working picture than a clean predictor: the standard first-order
account does not just fail to explain the gap, it predicts it backwards. What is
defensible today is a characterisation paper with that inversion as its geometric
result: detection does not predict control across ten concepts and four model
families, the obvious geometric account is inverted, and one non-confounded
concept (`refusal`) sits at the boundary on every model. That is the arc; the
abstract has to be written to it.

A model where steering is broken scores low controllability on *every* concept,
so a broken harness puts the whole model in the danger zone and is
indistinguishable from the finding. That is why the control is preregistered
(P9) and enforced in `aggregate`, not merely noted.
