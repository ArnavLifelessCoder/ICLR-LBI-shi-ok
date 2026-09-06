# Reframed title and abstract

Drafted against the two reviews and the three diagnostics in
`scripts/make_review_figures.py` and `scripts/make_review_tables.py`.

## Title

**Moved but Not Steered: Absolute-Deviation Metrics Overstate Control of
Concept Directions**

Alternatives, in order of preference:

2. Moved but Not Steered: What Steering Dose-Response Curves Measure
3. Disturbance Is Not Direction: Re-examining Controllability of Concept Directions
4. Readable, Perturbable, Not Steerable

The old title asserted the thing the saturated readability axis cannot
establish. Both reviewers said so independently, and they are right.

## Abstract

Activation steering is usually evaluated by adding a concept direction to the
residual stream and measuring how much behavior changes. We show that this
measurement, in its standard absolute-deviation form, does not distinguish
moving a model in the intended direction from disturbing it. For ten concepts
across four instruction-tuned models we sweep a steering coefficient to a
preregistered fluency ceiling and compute two quantities from the same curves:
the usual area under the absolute behavioral deviation, and its signed
analogue, which credits an intervention only when positive coefficients raise
the target behavior and negative ones lower it. The two are uncorrelated
(Spearman $-0.06$ over 40 concept-model points). Twenty of the forty points
have a negative signed area, meaning the intervention moved behavior against
the direction it was estimated from, and the median point attributes only 44%
of its measured effect to directional movement. The pattern is not judge noise:
it appears at the same rate in concepts where our human validation shows high
agreement and in those where it does not.

The consequence is that conclusions drawn from the absolute metric are
metric-dependent. We preregistered the hypothesis that a direction is steerable
to the extent it projects into the model's output-effective (unembedding)
subspace. Under the absolute metric that hypothesis is inverted (partial
Spearman $-0.45$ conditioning on readability), and the inversion is not an
artifact of the fluency ceiling: output overlap is unrelated to where the
ceiling falls (Spearman $0.012$), 37 of 40 sweeps reach the end of the grid
without breaking, and conditioning on the ceiling leaves the estimate at
$-0.452$. Leave-one-concept-out keeps the sign negative in all ten refits.
Under the signed metric the same test returns $+0.12$ with an interval
containing zero. The geometric result is therefore a property of how
controllability was measured rather than of the representations.

We also report what the readability axis can and cannot support. Held-out probe
AUROC saturates at 1.00 for 31 of 40 points, so the near-zero
readability-controllability correlation is uninformative rather than a null.
The claim the data support is the weaker one: high linear readability does not
guarantee that a direction is an effective control handle. We release the
stimuli, probes, steering harness, and the per-coefficient curves, and we
recommend that steering evaluations report a signed dose-response alongside any
absolute measure.

## What this changes in the body

| Section | Change |
|---|---|
| Title, abstract | Replace, as above. |
| 1 Introduction | Lead with the measurement claim. Keep the detection-versus-control motivation as the reason the metric was built. |
| 3.2 Controllability | Define both metrics. Absolute stays the preregistered primary; signed is introduced as the directional check. |
| 5.1 | Restate as "readability is saturated, so this correlation is uninformative". Delete the strong version. |
| 5.3 H1 | Add the ceiling control (Table `tab:ceiling`, Figure `fig6`) and LOCO (Table `tab:loco`, Figure `fig7`). State the sign is stable and the interval is not. Report that the result does not survive the signed metric. |
| New 5.4 | The directional result. Figure `fig8`, Table `tab:directional`. |
| 5.2 danger zone | Under a directional criterion 35 of 40 points qualify, which is a statement about the instrument, not the models. Report it as a bound on what the criterion can detect. |
| 6 Discussion | Drop "probe accuracy is not evidence that a representation affords control" in that form. Replace with the metric recommendation. |
| Limitations | Add: k is adaptive (4 to 6); only 2 generations cached per cell so a second judge needs a rerun; P9 floor crossed on interval logic by Mistral and Llama. |
| Appendix H | Add this as the latest instance of the same pattern: the bug flattered the hypothesis. |

## Open items that need a decision

- **P9.** By interval logic the sentiment floor is crossed by Mistral
  ($[0.067, 0.273]$) and Llama ($[0.095, 0.151]$). Either apply interval
  exclusion consistently and withhold, or state plainly that the floor is a
  point comparison and say why. Under the signed metric only Qwen ($+0.208$)
  and Mistral ($+0.163$) show real directional sentiment control.
- **Second judge.** Requires a rerun; only 720 generations are cached, two per
  curve point.
- **sycophancy.** Readability $0.57$ on Gemma and Llama is a failed probe.
  Excluding those four points moves the readability-controllability Spearman
  from $0.177$ to $0.230$. Report both.

## Sign-convention audit (run before trusting the directional result)

`certainty` has a negative dose response on all four models (Mistral $-0.73$,
Qwen $-0.67$, Llama $-0.86$, Gemma $-0.30$). A consistent wrong-way response
reproducing across four independent models would normally indicate an inverted
sign convention in the direction estimate or in the judge question, which would
make the signed metric score that concept backwards.

It does not survive the check. Only one of ten concepts is same-signed across
all four models. Under a null in which the sign of each model's response is
independent, the probability that a given concept comes out same-signed is
$2 \cdot 2^{-4} = 0.125$, so the expected count over ten concepts is $1.25$ and
the probability of seeing at least one is $0.74$. The observed count is one.
There is no evidence of a systematic polarity error, and the wrong-way movement
in Table~\ref{tab:directional} is not an artifact of the diagnostic.

The stronger reading of the same table is that within-concept monotonicity is
close to random across models for nine of ten concepts. The intervention is not
producing a reliable directional effect that merely fails to be large. It is
not producing a reliable directional effect at all.

## The positive control, directionally

Sentiment monotonicity is $+0.98$ (Mistral), $+0.79$ (Qwen), $+0.50$ (Llama),
$-0.68$ (Gemma). The control behaves as designed on three models and inverts on
Gemma, whose signed area is $-0.039$. Combined with the interval issue on the
P9 floor, the defensible statement is that the pipeline demonstrably steers
sentiment on Mistral and Qwen, weakly on Llama, and not directionally on Gemma.
That is narrower than the current claim that all four models pass.
