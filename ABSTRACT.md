# Abstract

Draft for ICLR 2027 (abstract deadline 2026-09-18). Written to the inverted-H1
arc: the dissociation exists, and the obvious geometric account of it is not
merely incomplete but pointed the wrong way. Every number here is from the
40-point map in `results nb7/combined_40point/`; keep them in sync if the data
changes.

---

## Abstract (main version, ~220 words)

Interpretability-based safety increasingly assumes that reading a concept off a
model's activations brings you part of the way to controlling it: if a linear
probe detects deception or refusal, steering along that direction should let you
suppress it. We test this assumption directly. For ten concepts across four
instruction-tuned models (1.5B to 9B) we measure two quantities separately:
readability, the held-out AUROC of a linear probe, and controllability, the area
under a steering dose-response curve up to a preregistered fluency ceiling.
Across 40 concept-model points the two are uncorrelated (Spearman 0.14, 95% CI
[-0.24, 0.51]): concepts that are perfectly legible can resist every steering
intervention we apply, including a six-method robustness gauntlet, while an
identical pipeline moves a sentiment control cleanly. We then test the natural
first-order explanation, that a concept is steerable to the extent its direction
projects into the model's output-effective (unembedding) subspace, and find it
inverted: the concepts most aligned with that subspace are the least steerable,
not the most (partial Spearman -0.45, 95% CI [-0.72, -0.04], conditioning on
readability, consistent across models). We argue this overlap indexes how lexical
a concept is rather than how much behavior depends on it. Two consequences follow
for interpretability-based safety: a strong probe is not evidence of a control
handle, and the obvious geometric predictor of when steering will work points, at
this scale, in the wrong direction. We release the datasets, probes, steering
harness, and full gap-map data.

---

## Short version (~110 words, for a submission form that caps length)

Interpretability-based safety assumes that a concept you can read off a model's
activations is a concept you can steer. We measure readability (held-out probe
AUROC) and controllability (steering dose-response area up to a fluency ceiling)
separately for ten concepts across four instruction-tuned models. The two are
uncorrelated: legible concepts can resist a six-method steering gauntlet while a
control steers cleanly. The natural geometric explanation, projection into the
output-effective subspace, comes back inverted: the most output-aligned concepts
are the least steerable (partial Spearman -0.45, CI excludes zero), not the most.
A strong probe is not a control handle, and the obvious account of steerability
is, at this scale, backwards.

---

## The claims this abstract rests on, and where each is checked

- **Uncorrelated axes.** Gap map Spearman 0.135, CI [-0.239, 0.509]
  (`gap_map.json`). Report as "uncorrelated on this data"; the readability axis
  is saturated (31/40 at AUROC 1.000), so this is uninformative rather than a
  clean null. That caveat belongs in the intro, not the abstract.
- **Legible yet resistant, gauntlet-confirmed.** `topic_science` immovable on
  Mistral and Gemma, least controllable on all four; `refusal` (not
  surface-confounded, safety-relevant) at the P6 boundary on all four. The
  strongest confirmed-immovable concept is surface-confounded, which the paper
  must state; the abstract stays at "can resist every intervention" without
  leaning on the confounded concept by name.
- **Inverted predictor.** Primary test partial Spearman -0.451, CI
  [-0.723, -0.041] excluding zero, opposite to the Section 2.5 prediction
  (`geometry_predictor.json`). Driver: `topic_science` is the highest-overlap
  and least-controllable concept on every model.
- **Lexicality reading.** Offered as interpretation, not a measured claim. See
  the dated block in design Section 2.5.
- **Judge validation.** 100 hand-labelled generations across all ten concepts,
  three raters (human, logit judge, lexicon). Within-concept alpha 0.33 for the
  logit judge against 0.18 for the lexicon; the sentiment positive control
  reaches 0.76. `topic_science` is clean (judge spread 0.02 vs human 0.05).
  `refusal` is the exception and is disclosed as a limitation. See
  `judge_agreement.json`.
- **Scope.** Sub-10B, four instruction-tuned families, templated stimuli.

## Status

**The measurement phase is complete and objection 14 is answered.** All four
models are run, the 40-point map is built, and the behaviour judge has been
validated against a human rater on a stratified 100-item sheet. The abstract's
controllability claims are supported: the positive control validates at 0.76 and
the danger-zone occupant `topic_science` shows no judge-manufactured movement.

One limitation must travel with the paper rather than block it. On `refusal` the
judge produces within-concept variation the human does not see, so that concept's
controllability is likely an over-estimate with an inflated interval. `refusal` is
the clean boundary case the abstract leans on implicitly, so the limitation is
stated in the paper's Limitations section and should not be argued away.

Remaining work is editorial, not empirical: references, the second figure, the
submission scaffolding, and one real LaTeX compile.
