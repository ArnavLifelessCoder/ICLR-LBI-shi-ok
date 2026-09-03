# Results

**Two models measured end to end. Both positive controls pass. The danger zone
holds exactly one point, and it is a concept declared surface-confounded before
the run, so the headline claim is not yet carried by a concept whose
readability can be trusted.**

Session-by-session detail is in [notebooks/RUNLOG.md](notebooks/RUNLOG.md).
Every number below comes from nb6 (2026-09-02): one Kaggle session, identical
code and identical judge for both models.

---

## Status

| Experiment | State |
| --- | --- |
| 1. Readability | Done, 2 models x 10 concepts |
| 2. Controllability | Done, 2 models x 10 concepts |
| 3. Gap map | Done, 20 points, 1 danger-zone occupant |
| 4. Geometry predictor | Done, **H1 uninformative** |
| 5. Safety concepts | Readable off Figure 1 |
| 6. Scale sweep | Not run |

Judge held fixed across both models: `Qwen/Qwen2.5-1.5B-Instruct`, fp16 on the
second GPU, scored from logits. Every record carries `judge_is_self=false`.

## Positive control (P9)

| Model | Control | Floor | Verdict |
| --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | 0.215 | 0.10 | PASS |
| Mistral-7B-Instruct-v0.3 | 0.163 | 0.10 | PASS |

Steering works on both. Neither model is withheld.

## The gap map

**20 concept-model points. Spearman rho = 0.119, CI [-0.233, 0.505].**

The interval includes zero: detection does not predict control on this data.
That is the direction the thesis predicts, but the honest reading is weaker
than it sounds, because the readability axis is saturated. See the caveats.

**Danger zone: one point.**

`topic_science@Mistral-7B-Instruct-v0.3` -- readability 1.000 [1.000, 1.000],
controllability 0.024 [0.013, 0.046], and it survived the full
six-intervention gauntlet. It is the only point in the study entitled to the
word *immovable*.

**And it is one of the two concepts declared surface-confounded.**
`topic_science` is domain membership, carried by content vocabulary, and it
fails the surface audit at 0.833 by design. Its readability is exactly the
number the audit says not to trust, so the single danger-zone occupant cannot
serve as the paper's headline example.

## Per-concept

| Model | Concept | Read | Selec | Control probe | Ctrl | Ctrl CI | Layer | Gauntlet |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen | certainty | 1.000 | 0.369 | 0.631 **flag** | 0.123 | [0.094, 0.220] | 13 | - |
| Qwen | factuality | 1.000 | 0.534 | 0.466 | 0.120 | [0.088, 0.177] | 13 | - |
| Qwen | formality | 0.992 | 0.598 | 0.394 | 0.034 | [0.021, 0.070] | 13 | moved |
| Qwen | honesty | 1.000 | 0.438 | 0.562 | 0.081 | [0.070, 0.176] | 20 | - |
| Qwen | refusal | 1.000 | 0.536 | 0.464 | 0.047 | [0.043, 0.108] | 13 | moved |
| Qwen | rudeness | 0.797 | 0.153 | 0.644 **flag** | 0.153 | [0.080, 0.259] | 15 | - |
| Qwen | sentiment | 1.000 | 0.435 | 0.565 | 0.215 | [0.115, 0.339] | 13 | - |
| Qwen | sycophancy | 1.000 | 0.385 | 0.615 **flag** | 0.118 | [0.083, 0.202] | 3 | - |
| Qwen | topic_science | 0.944 | 0.477 | 0.468 | 0.019 | [0.014, 0.029] | 13 | **survived** |
| Qwen | verbosity | 1.000 | 0.455 | 0.545 | 0.050 | [0.029, 0.088] | 13 | moved |
| Mistral | certainty | 1.000 | 0.420 | 0.580 | 0.113 | [0.069, 0.213] | 15 | - |
| Mistral | factuality | 1.000 | 0.611 | 0.389 | 0.032 | [0.017, 0.063] | 15 | moved |
| Mistral | formality | 1.000 | 0.454 | 0.546 | 0.099 | [0.081, 0.182] | 0 | - |
| Mistral | honesty | 1.000 | 0.624 | 0.376 | 0.107 | [0.056, 0.233] | 3 | - |
| Mistral | refusal | 1.000 | 0.512 | 0.488 | 0.038 | [0.027, 0.080] | 15 | **survived** |
| Mistral | rudeness | 1.000 | 0.488 | 0.512 | 0.211 | [0.117, 0.301] | 15 | - |
| Mistral | sentiment | 1.000 | 0.426 | 0.574 | 0.163 | [0.067, 0.273] | 15 | - |
| Mistral | sycophancy | 0.574 | 0.055 | 0.519 | 0.101 | [0.078, 0.189] | 2 | - |
| Mistral | topic_science | 1.000 | 0.530 | 0.470 | 0.024 | [0.013, 0.046] | 15 | **survived** |
| Mistral | verbosity | 1.000 | 0.399 | 0.601 **flag** | 0.054 | [0.027, 0.097] | 15 | - |

A dash means the gauntlet never ran, because the default intervention already
moved the concept. That is not the same as failing it.

## What replicates across the two models

**`topic_science` is the least controllable concept on both** -- 0.019 on Qwen,
0.024 on Mistral -- and survived the gauntlet on both. The effect is real and
not model-specific. It is also the concept whose readability is confounded, so
it cannot carry the claim.

**`refusal` is the interesting near-miss.** 0.047 on Qwen, 0.038 on Mistral,
survived the gauntlet on Mistral, and it is *not* surface-confounded. It misses
the danger zone only on P6's CI requirement: controllability CI upper bound
0.080 against a 0.05 threshold. More eval prompts or more models would tighten
that interval. This is the concept to watch.

**`rudeness` is the most controllable non-control concept** on both models
(0.153, 0.211), above the sentiment control on Mistral.

**Two readability collapses.** `sycophancy` reads 1.000 on Qwen and 0.574 on
Mistral; `rudeness` reads 0.797 on Qwen and 1.000 on Mistral. Whatever the
probe picks up for those two is not stable across model families.

## Primary test (H1) and exploratory

H1, output overlap, partial Spearman controlling for readability, cluster
bootstrap over 10 concepts:

```
partial rho = -0.430, CI [-0.729, 0.070]

UNINFORMATIVE: the CI includes zero, so no relationship can be claimed in
either direction. Per P7 this is a null result about H1, not weak support.
```

Note the sign. Section 2.5 predicts that *more* output overlap means *more*
controllability, and the point estimate is negative. With the interval spanning
zero this is not evidence of a reversal either, but it is not support.

**Exploratory, BH-corrected.** E1 participation ratio rho = -0.051, p_BH =
0.907. E2 low-variance PC alignment rho = 0.005, p_BH = 0.907. Both null.

**Ridge gap predictor.** LOO R^2 = -0.033, worse than predicting the mean.

None of the three named mechanisms explains the gap on this data.

## Judge agreement

| Model | Krippendorff alpha | Items |
| --- | --- | --- |
| Qwen2.5-7B | 0.132 | 3240 |
| Mistral-7B | 0.309 | 2916 |

Both poor. The panel is the logit judge (primary) against the lexicon scorer,
and the lexicon scorer is a development instrument that returns its no-hit
neutral 0.5 on much free text. This is objection 14 and it is **not yet
answered**: a validated `ClassifierScorer` and the hand-labelled third rater
are both outstanding. `human_labels.csv` was written by both runs and has not
been filled in.

## Caveats that belong in Methods, not in a rebuttal

**The readability axis is saturated.** Sixteen of twenty points sit at exactly
AUROC 1.000. A near-constant x cannot correlate with anything, and that, rather
than a clean null, is why the headline Spearman is uninformative. Selectivity
(AUROC minus the shuffled-label control) has the spread readability lacks and
is available as a declared secondary axis
(`build_gap_map(x_axis="selectivity")`). The preregistered axis is unchanged.

**The shuffled-label control is not at chance.** P3's flag fires on four of
twenty points: Qwen certainty 0.631, rudeness 0.644, sycophancy 0.615, and
Mistral verbosity 0.601. Simulated on isotropic noise of the same shape, a
12-shuffle control sits well inside 0.5 plus or minus 0.05, so these are real
structure rather than noise. Readability at those points is partly whatever the
control is picking up.

**Layer selection is a prior, not a measurement.** Validation AUROC saturates
across most of the network, so the preregistered mid-network tie-break, not
validation performance, chooses the layer. Three points landed at layer 0-3
(Mistral formality, Mistral sycophancy, Qwen sycophancy), where the tie-break
had least to work with.

**Sub-10B, two model families, templated stimuli.** Stated in the abstract.

## Surface-shortcut audit (CPU-only, no model)

Unchanged from concept construction: 8 of 10 pass at exactly 0.500, the floor.
`topic_science` (0.833) and `verbosity` (1.000) fail and are declared
surface-confounded in the builder. See [CONTEXT.md](CONTEXT.md) for why 0.500
is the floor rather than a good score.
