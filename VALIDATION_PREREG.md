# Judge validation, round 3: preregistration

Written and committed before any label in this round exists.

## Why this round

The paper's limitation is that the judge is validated on one concept,
`sentiment`, where it resolves direction on all four models. Two readings fit
that: the judge limits what can be measured, or sentiment is simply easier to
steer. A second validated concept separates them.

There is also a gap in what exists. The earlier validation (about ten samples
per concept, `human_labels_scored_100-v2.csv`) was scored against the 1.5B logit
judge used in the main study. The matched-protocol comparison, which carries
the paper's headline, uses Qwen2.5-3B-Instruct. That judge has never been
checked against a human on anything, including sentiment. This round covers it.

## Design

- Texts: generations from the stage G runs (`validation_output/results_judge_matched`),
  the same runs and the same 3B judge the headline uses. Two generations per
  coefficient per model are stored with the judge's score for each, so no
  judge is re-run.
- Concepts: `sentiment` (re-validation under the 3B judge), `sycophancy`,
  `formality`, `rudeness`.
- 25 texts per concept, 100 in total, sampled deterministically and spread
  across models and across the negative, near-zero and positive coefficient
  range, so that the texts differ in the behavior being rated.
- Excluded: points past the fluency ceiling, empty generations, texts under five
  words, duplicates.
- Blind: the sheet shows only the concept's question and the text. Model,
  coefficient and judge score are in a separate key file that the labeller does
  not open.
- Scale: 0, 0.25, 0.5, 0.75 or 1, answering the concept's question.

## Criterion, fixed now

Agreement is Krippendorff's alpha (interval) within concept, human against the
3B judge, with a 95% bootstrap interval over texts.

A concept is **validated** if:

1. alpha is at least 0.667, Krippendorff's threshold for tentative conclusions, and
2. both raters vary: the standard deviation of scores is at least 0.11 for the
   human and for the judge. Below that, alpha is uninformative, which is the rule
   the existing appendix already applies.

Alpha at or above 0.8 is reported as strong. The point estimate decides; the
interval is reported beside it.

## What each outcome means

Matched-protocol resolution counts, already known: `sentiment` 4/4,
`formality` 2/4, `rudeness` 2/4, `sycophancy` 0/4.

- **`sentiment` fails to validate under the 3B judge.** The paper's statement
  that the judge resolves direction where it is validated loses its only
  example, and is withdrawn for the matched analysis.
- **`sycophancy` validates.** The judge measures it and it still resolves on no
  model. That is evidence that some judge-scored nulls reflect a concept that
  does not steer under these interventions, not the judge. The judge-limits
  account is then at most partial, and the paper says so.
- **`formality` or `rudeness` validates.** Resolution there is 2/4, between the
  two accounts. It adds a validated concept without deciding between them.
- **Nothing beyond `sentiment` validates.** The limitation stands as written,
  now with the 3B judge's sentiment agreement measured rather than borrowed.

Every outcome is reported, in the appendix and in the Limitations paragraph,
whichever occurs.
