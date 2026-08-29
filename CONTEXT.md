# Context

Start here. This is the orientation document for anyone — or any session —
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

## Where the project stands (2026-08-27)

| | |
| --- | --- |
| Code | Complete for Experiments 1-4, 125 CPU tests, synthetic end-to-end demo passes |
| Real model runs | **None.** Every number in the repo is synthetic and validates plumbing only |
| Judges | Panel wired (LLM + lexicon + human sheet); `ClassifierScorer.model_map` deliberately empty |
| Concept set | 10 built, 8 clear the surface audit, 2 declared surface-confounded |
| Target | ICLR 2027 — abstract **Sep 18 2026**, paper **Sep 25 2026** |

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
enforcement cell is an unkept promise, not a detail** — four points (P4, P6, P7,
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

## Go / no-go

From `PLAN.md` Part 8, decided from data rather than hope:

- Positive control fails on two or more models → do not submit; the harness is
  wrong in a way that invalidates everything.
- Control works but no concept shows the dissociation → pivot to the
  "predict steerability from geometry" framing in design Section 11.
- Dissociation holds → write it.

A model where steering is broken scores low controllability on *every* concept,
so a broken harness puts the whole model in the danger zone and is
indistinguishable from the finding. That is why the control is preregistered
(P9) and enforced in `aggregate`, not merely noted.
