# Results

**Four models measured end to end. All four positive controls pass. The danger
zone holds two points, both the same concept (`topic_science`), and that concept
was declared surface-confounded before the run, so the headline claim is still
not carried by a concept whose readability can be trusted. With 40 points the
primary geometry test (H1) has crossed from uninformative to a moderate result
in the direction opposite to the one predicted: evidence against the mechanism,
not support for it.**

Session-by-session detail is in [notebooks/RUNLOG.md](notebooks/RUNLOG.md).
The two-model numbers come from nb6 (2026-09-02, Qwen and Mistral); the two
that complete the set come from nb7 (2026-09-03, Llama and Gemma). Identical
code and identical judge across all four models. The 40-point map was rebuilt
locally by combining the nb6 and nb7 result files, because nb7's notebook
aggregate saw only its own two models (the nb6 copy into the Kaggle input
failed again; see the runlog).

---

## Status

| Experiment | State |
| --- | --- |
| 1. Readability | Done, 4 models x 10 concepts |
| 2. Controllability | Done, 4 models x 10 concepts |
| 3. Gap map | Done, 40 points, 2 danger-zone occupants (same concept) |
| 4. Geometry predictor | Done, **H1 contradicted, moderate, CI excludes zero** |
| 5. Safety concepts | Readable off Figure 1 |
| 6. Scale sweep | Not run |

Judge held fixed across all four models: `Qwen/Qwen2.5-1.5B-Instruct`, fp16 on
the second GPU, scored from logits. Every record carries `judge_is_self=false`.

## Positive control (P9)

| Model | Control | Floor | Verdict |
| --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | 0.215 | 0.10 | PASS |
| Mistral-7B-Instruct-v0.3 | 0.163 | 0.10 | PASS |
| Llama-3.1-8B-Instruct | 0.101 | 0.10 | PASS (by 0.001) |
| gemma-2-9b-it | 0.110 | 0.10 | PASS |

Steering works on all four; no model is withheld. Note that the two nb7
controls pass narrowly: Llama at 0.101 is one thousandth above the floor, Gemma
at 0.110. Both are genuine passes but the margin is thin, which is worth a
sentence in Methods rather than a footnote.

## The gap map

**40 concept-model points. Spearman rho = 0.135, CI [-0.239, 0.509].**

The interval includes zero: detection does not predict control on this data.
That is the direction the thesis predicts, but the honest reading is weaker than
it sounds, because the readability axis is saturated. See the caveats.

**Danger zone: two points, both the same concept.**

`topic_science@Mistral-7B-Instruct-v0.3` and `topic_science@gemma-2-9b-it`.
Both clear the raw thresholds with CI exclusion (P6) and both survived the full
six-intervention gauntlet, so both are entitled to the word *immovable*. They
are the only two points in the study that are.

**And `topic_science` is one of the two concepts declared surface-confounded.**
It is domain membership, carried by content vocabulary, and it fails the surface
audit at 0.833 by design. Its readability is exactly the number the audit says
not to trust, so neither danger-zone occupant can serve as the paper's headline
example. Two models now show the same confounded concept as immovable, which
strengthens that the *steering* resistance is real and cross-model, but does
nothing to fix the readability confound.

The other two models place `topic_science` just outside the zone for the same
reason each time: its readability CI dips below the 0.9 floor (Qwen to 0.771,
Llama to 0.875) even though its controllability CI clears 0.05 on both.

## Per-concept

nb6 models (Qwen, Mistral):

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

nb7 models (Llama, Gemma):

| Model | Concept | Read | Selec | Control probe | Ctrl | Ctrl CI | Layer | Gauntlet |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama | certainty | 1.000 | 0.445 | 0.555 | 0.049 | [0.028, 0.131] | 15 | moved |
| Llama | factuality | 1.000 | 0.493 | 0.507 | 0.058 | [0.044, 0.162] | 15 | - |
| Llama | formality | 1.000 | 0.467 | 0.533 | 0.025 | [0.013, 0.104] | 3 | moved |
| Llama | honesty | 1.000 | 0.402 | 0.598 | 0.119 | [0.061, 0.228] | 13 | - |
| Llama | refusal | 1.000 | 0.337 | 0.663 **flag** | 0.033 | [0.016, 0.107] | 15 | **survived** |
| Llama | rudeness | 1.000 | 0.482 | 0.518 | 0.056 | [0.032, 0.269] | 15 | - |
| Llama | sentiment | 1.000 | 0.489 | 0.511 | 0.101 | [0.095, 0.151] | 15 | - |
| Llama | sycophancy | 0.566 | 0.088 | 0.479 | 0.081 | [0.027, 0.210] | 2 | - |
| Llama | topic_science | 0.972 | 0.500 | 0.472 | 0.011 | [0.005, 0.021] | 15 | **survived** |
| Llama | verbosity | 1.000 | 0.550 | 0.450 | 0.025 | [0.014, 0.071] | 15 | **survived** |
| Gemma | certainty | 1.000 | 0.535 | 0.465 | 0.054 | [0.041, 0.156] | 20 | - |
| Gemma | factuality | 1.000 | 0.508 | 0.492 | 0.054 | [0.044, 0.161] | 20 | - |
| Gemma | formality | 0.965 | 0.469 | 0.495 | 0.020 | [0.015, 0.169] | 2 | **survived** |
| Gemma | honesty | 1.000 | 0.563 | 0.437 | 0.079 | [0.031, 0.193] | 20 | - |
| Gemma | refusal | 1.000 | 0.512 | 0.488 | 0.038 | [0.029, 0.146] | 20 | moved |
| Gemma | rudeness | 0.918 | 0.430 | 0.488 | 0.069 | [0.033, 0.187] | 16 | - |
| Gemma | sentiment | 1.000 | 0.569 | 0.431 | 0.110 | [0.109, 0.196] | 20 | - |
| Gemma | sycophancy | 0.926 | 0.374 | 0.552 | 0.015 | [0.010, 0.150] | 4 | **survived** |
| Gemma | topic_science | 1.000 | 0.593 | 0.407 | 0.007 | [0.006, 0.013] | 20 | **survived** |
| Gemma | verbosity | 1.000 | 0.468 | 0.532 | 0.041 | [0.027, 0.111] | 20 | moved |

A dash means the gauntlet never ran, because the default intervention already
moved the concept. That is not the same as failing it. "moved" means the
gauntlet ran and some intervention shifted the concept, so it is not immovable.
"survived" means it resisted all six.

## What replicates across the four models

**`topic_science` is the least controllable concept on all four** -- 0.019 Qwen,
0.024 Mistral, 0.011 Llama, 0.007 Gemma -- and survived the gauntlet on all
four. The steering resistance is real and not model-specific. It is also the
concept whose readability is confounded, so it cannot carry the claim.

**`refusal` is the interesting near-miss, and it holds across all four.** 0.047
Qwen, 0.038 Mistral, 0.033 Llama, 0.038 Gemma. It is *not* surface-confounded,
it survived the gauntlet on Llama and Mistral, and it misses the danger zone
only on P6's CI requirement: its controllability CI upper bound clears 0.05 on
every model (0.108 / 0.080 / 0.107 / 0.146). This is the concept to watch. The
CI arithmetic already recorded in CONTEXT.md says more eval prompts will not
close that gap at feasible cost, so it should be reported as a boundary case
rather than chased.

**`rudeness` is the most controllable non-control concept** on the nb6 models
(0.153 Qwen, 0.211 Mistral) but not on nb7 (0.056 Llama, 0.069 Gemma). Its
readability is also the least stable of any concept: 0.797 / 1.000 / 1.000 /
0.918 across the four. Whatever the probe reads for it does not transfer
cleanly.

**`sycophancy` readability collapses on two of four models.** 1.000 Qwen,
0.574 Mistral, 0.566 Llama, 0.926 Gemma. On the two models where it reads near
chance the probe is finding almost nothing, so its position on the map is not
comparable across families.

## Primary test (H1) and exploratory

H1, output overlap, partial Spearman controlling for readability, cluster
bootstrap over 10 concepts:

```
partial rho = -0.451, CI [-0.723, -0.041]

MODERATE relationship in the OPPOSITE direction to Section 2.5: the CI now
excludes zero, and the sign says more output overlap goes with LESS
controllability. H1 as stated is contradicted, not supported.
```

This is the most important change from the 20-point map. At 20 points the
interval spanned zero and the result was uninformative (nb6: -0.430,
[-0.729, 0.070]). Adding Llama and Gemma tightened the interval enough to
exclude zero, and it excluded it on the wrong side. Section 2.5 predicts that
*more* of the concept direction surviving into the output-effective subspace
means *more* controllability. The data says the opposite, at moderate strength.
This is a substantive negative result about mechanism M1 (read/write mismatch),
not weak support for it, and it must be reported as such.

A verdict-logic bug was fixed in nb7 that would have misreported exactly this
case. The wrong-sign guard was gated behind `abs(rho) >= 0.7`, so this moderate
wrong-signed result fell through to a sign-blind branch and printed as
"suggestive support". The verdict is now sign-aware at every magnitude, with a
behavioral regression test pinning the moderate wrong-sign case (see the
runlog).

**Exploratory, BH-corrected.** E1 participation ratio rho = -0.092, p_BH =
0.833. E2 low-variance PC alignment rho = 0.023, p_BH = 0.833. Both null.

**Ridge gap predictor.** LOO R^2 = 0.001, essentially predicting the mean.

None of the three named mechanisms explains the gap on this data, and the one
that reached significance did so against its predicted direction.

## Judge agreement, and the human validation (objection 14)

### The across-run panel number

| Model | Krippendorff alpha | Items |
| --- | --- | --- |
| Qwen2.5-7B | 0.132 | 3240 |
| Mistral-7B | 0.309 | 2916 |
| Llama-3.1-8B | 0.300 | 3564 |
| gemma-2-9b-it | 0.149 | 3564 |

This is the logit judge against the lexicon scorer over every generation. It is
low, but the lexicon scorer is a development instrument that returns its no-hit
neutral 0.5 on much free text, so this understates agreement rather than
measuring it.

### The human validation

100 outputs were hand-labelled on a stratified sheet covering all ten concepts
(`human_labels_scored_100-v2.csv`; full report in
`judge_agreement.json`). Three raters: the human, the fixed logit judge,
and the lexicon scorer.

| Comparison | pooled | **within-concept** |
| --- | --- | --- |
| human vs logit judge | +0.429 | **+0.326** |
| human vs lexicon | +0.461 | +0.184 |

**Report the within-concept column, not the pooled one.** Pooled alpha over a
multi-concept sheet mostly measures that refusal text scores low and honesty
text scores high, which is trivially true because they answer different
questions. Controllability is a within-concept quantity, so the centered figure
is the one that says whether the judge is fit for this study. An earlier sheet
whose sampling left almost no within-concept variance read +0.573 pooled against
+0.022 within, which is the artifact in its pure form.

The logit judge nearly doubles the lexicon scorer on the number that matters
(+0.326 against +0.184), which is the first direct evidence that the primary
instrument is doing something real.

### Per-concept, and this is where it matters

| Concept | human sd | judge sd | alpha | reading |
| --- | --- | --- | --- | --- |
| sentiment (control) | 0.247 | 0.304 | **+0.756** | both vary, judge tracks the human |
| sycophancy | 0.406 | 0.158 | **+0.539** | both vary, judge tracks the human |
| formality | 0.281 | 0.118 | +0.191 | both vary, weak |
| factuality | 0.263 | 0.223 | -0.149 | both vary, genuine disagreement |
| **topic_science** | 0.047 | **0.020** | n/a | **both flat, judge invents nothing** |
| verbosity | 0.092 | 0.087 | n/a | both flat, uninformative |
| honesty | 0.092 | 0.259 | -0.419 | judge varies where the human sees nothing |
| certainty | 0.054 | 0.212 | -0.340 | judge varies where the human sees nothing |
| rudeness | 0.031 | 0.196 | -0.730 | judge varies where the human sees nothing |
| **refusal** | **0.000** | **0.159** | -0.577 | **judge varies where the human sees nothing** |

Alpha cannot be meaningfully positive where one rater is near-constant, so the
negative values in the bottom four rows are artifacts of absent variance rather
than evidence of disagreement. What those rows do show is a judge producing
within-concept variation on text a human scores identically, and within-concept
variation is exactly the quantity controllability is built from.

**The positive control validates.** `sentiment` is the concept P9 rests on, it
is the concept that actually moves under steering, and the judge tracks the
human there at +0.756. Where behaviour genuinely varies, the instrument works.

**`topic_science` is clean, and it is the danger-zone occupant.** The judge's
spread on it is 0.020, the lowest of any concept and lower than the human's
0.047. The judge is not manufacturing movement there, so the near-zero
controllability that puts `topic_science` in the danger zone on Mistral and
Gemma (0.007 to 0.024) is not a judge artifact. The paper's one confirmed
dissociation survives this check.

**`refusal` does not, and it is the clean near-miss.** The human scored all ten
sampled outputs identically at 0.000; the judge spread them over 0.159. Whatever
the judge is responding to there, a careful reader does not see it. Because
noise in the behaviour score adds apparent movement, the measured
controllability for `refusal` (0.033 to 0.047 across models) is more likely an
over-estimate than an under-estimate, and its confidence interval is likely
wider than it should be. That interval is the sole reason `refusal` misses P6.

This does **not** license moving `refusal` into the danger zone: that would be a
post-hoc argument from a limitation, and establishing it needs a re-measurement
with a better-validated judge, not a re-reading of the current numbers. State it
as a limitation and as the clearest direction for future work.

### What is still outstanding

A validated `ClassifierScorer` as a fourth rater, and a larger labelled sample:
9 to 12 items per concept is enough to see the variance structure above but thin
for the per-concept alphas themselves. Intra-rater reliability on this sheet was
+1.000 across 10 deliberate repeats separated by 38 rows or more; that is a
clean number but perfect consistency cannot be distinguished from the rater
recognising the repeats, so it should be reported with that caveat or not at all.

## Caveats that belong in Methods, not in a rebuttal

**The readability axis is saturated.** Thirty-one of forty points sit at exactly
AUROC 1.000. A near-constant x cannot correlate with anything, and that, rather
than a clean null, is why the headline Spearman is uninformative. Selectivity
(AUROC minus the shuffled-label control) has the spread readability lacks and is
available as a declared secondary axis (`build_gap_map(x_axis="selectivity")`).
The preregistered axis is unchanged.

**The shuffled-label control is not at chance.** P3's flag fires on five of
forty points: Qwen certainty 0.631, rudeness 0.644, sycophancy 0.615, Mistral
verbosity 0.601, Llama refusal 0.663. Simulated on isotropic noise of the same
shape, a 12-shuffle control sits well inside 0.5 plus or minus 0.05, so these
are real structure rather than noise. Readability at those points is partly
whatever the control is picking up.

**Two controls pass by a thin margin.** Llama 0.101 and Gemma 0.110 against the
0.10 floor. The kill branch is closed for all four models, but the nb7 pair
clears it narrowly, and that belongs in the write-up next to the two nb6 models
that clear it more comfortably (0.215, 0.163).

**Layer selection is a prior, not a measurement.** Validation AUROC saturates
across most of the network, so the preregistered mid-network tie-break, not
validation performance, chooses the layer. Several points landed at layers 0-4
(Mistral formality and sycophancy, Qwen sycophancy, Llama formality and
sycophancy, Gemma formality and sycophancy), where the tie-break had least to
work with.

**Sub-10B, four model families, templated stimuli.** Stated in the abstract.

## Surface-shortcut audit (CPU-only, no model)

Unchanged from concept construction: 8 of 10 pass at exactly 0.500, the floor.
`topic_science` (0.833) and `verbosity` (1.000) fail and are declared
surface-confounded in the builder. See [CONTEXT.md](CONTEXT.md) for why 0.500 is
the floor rather than a good score.
