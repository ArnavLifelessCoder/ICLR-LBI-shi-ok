# Legible but Immovable

Code for the study in [research_design_legible_but_immovable.md](research_design_legible_but_immovable.md):
measuring how well a linear probe **detects** a concept (readability) against how
much steering along that concept's direction **changes behavior** (controllability),
across many concepts and models, plus a geometric predictor of the gap between them.

The design document is the source of truth. Section 12 there records the revisions
made during implementation and why, including the third pass that audited the
code against the preregistration and found four preregistered commitments
documented but not implemented.

**New here?** [CONTEXT.md](CONTEXT.md) is the orientation doc — what the study
asks, where it stands, the constraints, and the go/no-go criteria.
[RESULTS.md](RESULTS.md) is what has actually been measured (as of now: the
CPU-only surface audit, and nothing from a real model).
[notebooks/RUNLOG.md](notebooks/RUNLOG.md) is the per-session record and always
carries the next action.

## Install

```bash
pip install torch transformers accelerate bitsandbytes scikit-learn scipy matplotlib pytest
```

`bitsandbytes` is only needed for 4-bit loading on a GPU. The tests and the
synthetic demo run on CPU with just numpy, scikit-learn, scipy and matplotlib.

Two version notes, both learned the hard way:

* **NumPy 2 is fine, but torch must match it.** `np.trapz` was removed in NumPy
  2.0 (the code uses `np.trapezoid` with a fallback, so either works). A torch
  built against NumPy 1.x, however, cannot convert tensors to arrays at all —
  `.numpy()` raises `RuntimeError: Numpy is not available`, and
  `capture_activations` dies on its first batch. If you are on NumPy 2, use
  torch 2.4.1 or newer.
* **On Windows, import torch before scikit-learn.** They ship separate Intel
  OpenMP runtimes and whichever loads second fails with
  `WinError 1114 ... c10.dll`. `tests/conftest.py` handles this for the test
  suite; if you write your own entry point, import torch first. Linux and macOS
  are unaffected.

## Check the wiring before spending GPU quota

```bash
python -m pytest tests/ -q          # 114 tests, ~60s, no GPU, no downloads
python scripts/demo_synthetic.py    # full pipeline on planted ground truth, ~40s
```

The demo plants two concepts as readable-but-immovable and confirms the pipeline
recovers exactly those in the danger zone, and that the geometry predictor picks
up the planted signal. It also runs the surface-shortcut audit (R15) to verify
it catches a planted shortcut concept and passes the real concepts. If any
ground-truth check fails, something is miswired -- fix it before touching a T4.

## Surface-shortcut audit (Phase A gate)

Before extracting any activations, run the TF-IDF surface-shortcut audit on
every concept to catch lexical shortcuts:

```python
from lbi.concepts import audit_all_concepts
for r in audit_all_concepts():
    print(f"{r.concept:<14} {'PASS' if r.passed else 'FAIL'}  "
          f"surface_auroc={r.surface_auroc:.3f}  worst={r.worst_fold_auroc:.3f}  "
          f"{r.reason}")
```

The audit is **leave-one-family-out**: every template family takes a turn as the
test fold, and the reported number is the mean over folds, with the worst fold
reported alongside it. A concept passes only if that mean is below 0.65, or
falls at least 0.15 below the activation probe's AUROC. Concepts that fail are
rebuilt, not reported.

The statistic is **two-sided** — `max(auroc, 1 - auroc)` per fold. A classifier
that ranks a held-out family perfectly *backwards* has still found a lexical
contrast; only its polarity failed to transfer. Scored one-sided, two concepts
in the original set read AUROC 0.00 and 0.01 and were recorded as the cleanest
in the study.

### What the audit does and does not prove

Every concept that is not declared surface-confounded now scores exactly 0.500,
which is the floor rather than a good number, and the reason is worth stating
plainly in the methods section rather than letting a reviewer find it:

Concepts are built so that each family's positive/negative contrast is carried
by vocabulary that appears **nowhere else in the concept** — not in another
family's markers, not in another family's carrier, not in the topic strings.
Two builder-enforced invariants pin this (`check_marker_disjointness`,
`check_marker_length_match`), so a TF-IDF model trained on five families has no
feature that fires on the sixth, and its decision function is constant there.

So the audit is not an independent discovery instrument for these concepts. It
is a **check that the construction succeeded**, and it earns its place by
failing loudly when it has not: it caught duplicated template families in
`topic_science`, marker words shared across families in five concepts, and a
length imbalance between paired markers that separated families at AUROC 0.94
with no marker word transferring at all.

What still has to carry the argument is the probe's own held-out-family AUROC.
The probe faces marker words it has never seen, so a high score there is
evidence of something beyond vocabulary; the audit only establishes that the
vocabulary route is closed.

### The two declared exceptions

`verbosity` and `topic_science` set `surface_confounded=True` and **fail** the
audit. The flag does not convert a failure into a pass — `passed` stays False,
and `test_declared_confounded_concepts_still_fail_rather_than_being_excused`
pins that. It records that the failure was predicted in advance:

* **verbosity** is length, and length is a surface property by definition. The
  terse side of each pair is a strict prefix of the verbose side, so the two
  differ only by padding and not by content words, which is as close to a
  minimal pair as the concept allows. It still reads 1.000.
* **topic_science** is domain membership, which is carried by content
  vocabulary. Families are subject areas (biology, physics, geology, chemistry,
  astronomy, computing), the strongest split available. It reads 0.833.

Report both with the confound stated, or drop them. Do not report them as clean.

## Terminology: two different claims, kept apart

**"Unsteerable under tested interventions"** is a claim about what was tried.
**"Immovable"** is a claim about the concept. Only the first is earned by
failing to move something once, and the code will not let the second be
asserted without the gauntlet.

A concept that lands in the danger zone has cleared two raw thresholds with
CI exclusion (P6) under the default difference-of-means sweep. That is not
enough. `pipeline.confirm_immovable` then runs six interventions — single-layer
addition, multi-layer addition, clamping and directional ablation on the
difference-of-means direction, plus single-layer addition on the probe-weight
and RepE reading-vector directions — and only a concept that resists all six
earns the stronger word.

That distinction is structural, not editorial. `GapPoint.gauntlet_passed`
carries the gauntlet result into the map, and `gap_map.json` reports the two
sets separately:

```json
"confirmed_immovable": ["refusal@Qwen2.5-7B-Instruct"],
"unsteerable_under_tested_interventions": ["honesty@Qwen2.5-7B-Instruct"]
```

`gauntlet_passed=None` means the gauntlet never ran, which is deliberately not
the same as failing it — it runs only on concepts the default intervention
already failed to move, so most points legitimately carry `None`.

Even a confirmed pass is bounded by those six interventions. It is evidence
that the concept resists every standard method while an identical pipeline
moves the sentiment control cleanly. It is not a proof that no intervention
exists, and the paper should not claim one.

## Run it for real

One model per invocation; results are written per concept as they finish, so a
killed Colab session loses at most one concept.

```bash
python scripts/run_experiment.py --model Qwen/Qwen2.5-7B-Instruct
python scripts/run_experiment.py --model meta-llama/Llama-3.1-8B-Instruct
python scripts/run_experiment.py --model Qwen/Qwen2.5-1.5B-Instruct --no-4bit
python scripts/run_experiment.py --aggregate-only    # gap map + predictor
```

Activations cache to `cache/activations/` keyed by model, text set and pooling,
so re-running a concept after a crash re-reads rather than re-computes.

Both `cache/` and `results/` are gitignored: the first is large and
machine-specific, the second changes every run. Regenerate the demo output with
`python scripts/demo_synthetic.py`; real-run results belong in a release
artifact or the paper's supplement, not in the history.

Measured on Kaggle T4, 2026-08-29: the positive control alone -- one concept
through the full P4 band -- took **21.7 min** on Qwen2.5-7B in 4-bit, so a
ten-concept model is roughly **3.6 h**. An earlier estimate of 1.5 h was derived
for single-layer steering and missed that the band is four sweeps. Probe fitting
adds ~5 s per concept on CPU. See `notebooks/README.md` for the full table and
what to trim if the budget does not fit.

## Layout

| Module | Design doc section |
| --- | --- |
| `lbi/concepts.py` | Section 4 -- 10 concepts as minimal pairs, tagged by template family; **R15 surface-shortcut audit** and the two construction invariants it tests |
| `lbi/extraction.py` | Activation capture via forward hooks, with disk cache |
| `lbi/probes.py` | Experiment 1 -- readability, with selectivity controls; **R16 RepE reading vector** |
| `lbi/steering.py` | Experiment 2 -- controllability, all four robustness variants; **P4 layer band protocol** |
| `lbi/behavior.py` | Behavior judges and the fluency ceiling |
| `lbi/gapmap.py` | Experiment 3 -- the gap map, figures 1 and 2 |
| `lbi/geometry.py` | Experiment 4 -- **R13** one primary hypothesis (output overlap) + two exploratory analyses |
| `lbi/pipeline.py` | Orchestration, including the immovability gauntlet and **P9 positive control gate** |

## Three direction-derivation methods (R16)

The study derives concept directions three independent ways, so a reviewer
cannot dismiss a null as "you used the wrong direction":

1. **Difference-of-means** — the standard, robust choice (also exactly CAA).
2. **Probe weights** — the logistic regression weight vector, projected back to
   raw activation space.
3. **RepE reading vector** — the first principal component of paired activation
   differences (Zou et al.), computed from cached activations at zero extra cost.

## What the code enforces that is easy to get wrong

**Surface-shortcut audit runs before extraction (R15).** A TF-IDF classifier
on raw text must fail to separate positive from negative examples, or the concept
is reading vocabulary rather than activations. This is the single highest-leverage
gate in the project.

**Danger-zone thresholds are absolute, not normalized.** Both axes are min-max
normalized within model for the correlation and the plot, because raw AUROC and
raw dose-response AUC are not comparable across models. But normalization
guarantees *someone* sits at controllability 0.0, so applying the danger-zone
threshold to normalized values manufactures a finding out of pure scatter. Raw
thresholds (AUROC >= 0.9, dose-response AUC <= 0.05) are claims about the concept
itself and survive rescaling. `test_normalization_cannot_manufacture_a_danger_zone`
pins this.

**Danger-zone membership needs the CI to exclude the threshold, not just the
point estimate (P6).** `build_gap_map` requires the readability CI's lower
bound at or above 0.9 *and* the controllability CI's upper bound at or below
0.05. A concept measured at controllability 0.04 with an interval running to
0.30 has demonstrated scatter, not immovability. A NaN bound never qualifies.
`require_ci_exclusion=False` restores the weaker point-estimate rule for
comparison; it is not the preregistered one.

**The gap-map correlation CI clusters by concept (P7).** The unit of
generalization is the concept, not the concept-model pair — ten concepts across
five models is ten observations, not fifty. On a synthetic set with eight
concepts replicated five times, the i.i.d. bootstrap returns [0.34, 0.82] and
the cluster bootstrap [-0.12, 0.93]. The first excludes zero and the second
does not, so this is the difference between claiming a result and not.

**Controllability is best-over-band, and the band actually gets swept.**
`run_model` calls `run_steering_best_over_band`, so the answer to "you steered
at the wrong layer" is a number rather than a paragraph. This costs four
generation sweeps per concept and is budgeted for in PLAN.md's zero-cost
ledger. `--single-layer` skips it for smoke runs and prints a warning that the
resulting numbers are not reportable.

**A failed positive control withholds the model (P9).** `run_model` runs
sentiment first and warns immediately, then writes `<model>_control.json`;
`aggregate` reads those records back and drops every point from a model whose
control fell below the floor. This has to be enforced rather than noted,
because a model where steering is broken produces low controllability on
*every* concept — so a broken harness lands the whole model in the danger zone
and looks exactly like the paper's headline finding.

**The preregistered primary test needs the raw axes, not the gap.** H1 is a
partial Spearman of output overlap against controllability *controlling for
readability*, so `fit_gap_predictor` cannot compute it from the gap alone —
pass `controllabilities=` and `readabilities=`. Without them it returns a
`primary` report whose verdict says `primary test NOT RUN` rather than
substituting the ridge fit's R². The cluster bootstrap (P7) and the
BH-corrected exploratory analyses (P8) only run on that path too.

**A concept is only called immovable after the full gauntlet, and the gap map
knows whether it ran.** `confirm_immovable`'s verdict used to live only in the
per-concept JSON, so `build_gap_map` labelled points "readable-but-immovable"
from thresholds alone and the gauntlet's answer sat in a file nothing read. The
result now propagates through `ConceptRun.gauntlet` and `GapPoint.gauntlet_passed`
into the map. See **Terminology** above.

**Steering strength is in RMS units, measured once.** The coefficient is scaled by
the residual-stream RMS at the intervention layer, captured from the first
unsteered forward pass so the scale does not drift as steering pushes the norm
around. Without this, coefficients mean different things at different layers and
in different models, and the cross-model comparison is meaningless.

## Before the numbers go in the paper

`LexiconScorer` is the development judge -- transparent, offline, and good enough
to shake out the pipeline. It is not good enough for published behavior scores.
Swap in `ClassifierScorer` (a validated HuggingFace classifier per concept) and
`LLMJudgeScorer`, and report `judge_agreement` between them, which the design doc
requires so a reviewer cannot dismiss the behavior axis.

`judge_agreement` reports **Krippendorff's alpha** alongside Pearson and
Spearman, and alpha is the one to quote. Two judges offset by a constant
correlate at Pearson 1.0 and agree on nothing; the behaviour axis is read in
absolute terms against a fixed danger-zone threshold, so a systematic offset
matters and correlation cannot see it. On exactly that case alpha reads 0.32.
`krippendorff_alpha_interval` accepts NaN for unrated items, so the human
spot-check over 100 outputs goes in as a third rater without padding.

The judges are the largest piece of remaining work: `ClassifierScorer.model_map`
ships empty by design, so that no unvalidated default judge can silently end up
in the paper's numbers.

The `rudeness` concept ships as a courtesy-register proxy for toxicity so the repo
does not generate toxic text; substitute Civil Comments pairs for the paper run.
