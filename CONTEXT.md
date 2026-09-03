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

## Where the project stands (2026-09-03)

| | |
| --- | --- |
| Code | Experiments 1-4 complete, 140 CPU tests, synthetic demo passes |
| Real model runs | **Qwen2.5-7B and Mistral-7B complete**, 20 points, both controls pass. See RESULTS.md |
| Judges | Fixed logit judge (Qwen2.5-1.5B) + lexicon. **alpha 0.13 / 0.31 -- too low.** Human sheet unfilled |
| Concept set | 10 built, 8 clear the surface audit, 2 declared surface-confounded |
| Target | ICLR 2027 -- abstract **Sep 18 2026**, paper **Sep 25 2026** |

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

## What the data says (nb6, 2026-09-02)

The pipeline works and the study has run. The result is **not** the paper that
was planned, and the next decision is a framing decision rather than a
technical one.

**One danger-zone occupant:** `topic_science@Mistral`, confirmed immovable
against all six interventions. It replicates on Qwen (0.019 vs 0.024, gauntlet
survived on both). But `topic_science` is one of the two concepts declared
surface-confounded before the run, so its readability is the number the audit
says not to trust. The claim has an occupant and cannot use it.

**The near-miss is the one to watch.** `refusal` is not surface-confounded,
scores 0.047 / 0.038 across the two models, and survived the gauntlet on
Mistral. It misses P6 only on the CI: controllability upper bound 0.080 against
a 0.05 threshold. Tightening that interval -- more eval prompts, a third model
-- is the single highest-value experiment left.

**H1 is uninformative, not supported.** Partial rho -0.430, CI [-0.729, 0.070],
and the point estimate has the opposite sign to Section 2.5's prediction. Both
exploratory analyses are null (p_BH 0.907) and the ridge predictor has negative
LOO R^2. None of the three named mechanisms explains the gap here.

**Two things blunt every number above.** Readability is saturated -- sixteen of
twenty points at exactly AUROC 1.000 -- so the primary x-axis has almost no
variance and the headline correlation is uninformative rather than null. And
judge agreement is 0.13 / 0.31, which leaves objection 14 unanswered.

## Go / no-go

From `PLAN.md` Part 8, decided from data rather than hope:

- Positive control fails on two or more models → do not submit; the harness is
  wrong in a way that invalidates everything.
- Control works but no concept shows the dissociation -> pivot to the
  "predict steerability from geometry" framing in design Section 11.
- Dissociation holds -> write it.

**Where nb6 leaves this.** Both controls pass, so the first branch is closed.
The dissociation has exactly one confirmed instance and it is a confounded
concept, so the third branch is not yet earned. The geometry pivot is also
unavailable as stated, because H1 and both exploratory features came back null.
What remains defensible today is a characterisation paper: detection does not
predict control across ten concepts and two model families, the standard
first-order geometric account does not explain it either, and one
non-confounded concept (`refusal`) sits at the boundary. Decide this before
writing the abstract, not after.

A model where steering is broken scores low controllability on *every* concept,
so a broken harness puts the whole model in the danger zone and is
indistinguishable from the finding. That is why the control is preregistered
(P9) and enforced in `aggregate`, not merely noted.
