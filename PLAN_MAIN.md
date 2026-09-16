# PLAN_MAIN.md

What it takes to get this paper to an 8, and in what order.

[PLAN.md](PLAN.md) was written under one hard constraint, that nothing in it
adds compute. That was the right call at the time and it is now retired: the
validation runs of 2026-09-14 to 09-16 spent four Kaggle sessions and changed
what the paper is about. PLAN.md's Part 1 reviewer model and Part 2
preregistration still hold and are not reopened. Scope and priority live here.

---

## Part 0. The scoring problem, stated plainly

Reviews of the pre-validation draft landed at 5 and at "marginally below
acceptance". Both reviewers agreed on the cause, and it was not the writing:
the headline claim rested on a metric nobody had checked, and the strongest
result was carried by a concept the paper itself declared invalid.

An 8 is not "objections fixed". At an 8 a reviewer argues for the paper in
discussion, which means the thing being reviewed is the contribution rather than
the caveats. That is the bar this document is written against.

---

## Part 1. Where this stands

**The experiments are ahead of the paper.** Everything below is measured and
sitting in `validation_output/`. None of it is in the PDF, which still argues
the pre-validation story.

| Result | Evidence |
|---|---|
| The directional metric works | 5 of 16 ground-truth points resolve, **every one positive**; `gt_french` on 3 of 4 models, `gt_uppercase` on 2 of 4. Nothing anywhere in the study resolves in the wrong direction. |
| The absolute/signed divergence is the judge's | Directional share 0.993 without a judge against 0.387 with one |
| Judge-scored magnitudes are not reproducible | 8x to 26x swings between a 1.5B and a 3B judge |
| Judge-scored signs are not fully reproducible | 3 of 16 re-judged points flip |
| The danger zone was an artifact | `topic_science` moves from 0.007-0.024 to 0.118-0.260 against a 0.05 threshold, clearing the zone on all four models |
| Withholding Gemma was correct | Unresolved and negative under both judges, and with no judge at all its median ground-truth area is 0.0040 against 0.0195 for the other three |
| Output overlap is not stable in k | Rank correlation between k=64 and k=2048 is -0.137 on Qwen and +0.297 on Llama; the partial correlation attenuates from -0.457 to -0.133 |

What the study still cannot support is any claim that steering is undirected.
Three of forty judge-scored points have a resolved sign, and the conditional
framing already in the draft is the correct one.

That set closes the gap the reviewers named. It does not by itself reach an 8,
because an 8 needs a contribution and this is still shaped as a warning.

---

## Part 2. The paper this becomes

**C1. A validated diagnostic.** Signed dose-response and the arm split, checked
against concepts whose intended direction is known by construction because the
readout is a rule over the generated string rather than a judgment. This is the
piece that was missing and it is what makes every negative statement elsewhere
safe: the instrument demonstrably detects real steering, so a null is about the
concept and not about the metric.

**C2. The standard summary is instrument-dependent.** Absolute dose-response
area is uncorrelated with direction, diverges from it only when a judge is
involved, and moves by up to 26x when the judge changes. A published
controllability number is therefore not comparable across papers.

**C3. What the check does to a result in the literature.** Missing, and it is
the difference between a warning and a standard.

**C4. When directional control is recoverable at all.** One concept's worth of
signal. See Part 4.

The two findings the paper currently leads with, the
readability-controllability dissociation and the inverted geometric predictor,
become worked examples inside C2. The danger zone and the
immovable/unsteerable terminology move to an appendix: both are computed on the
metric being disowned, and the zone's only occupant does not survive a change of
judge.

---

## Part 3. Work queue

Ordered by value per hour. Do not reorder without writing down why.

### 1. Rewrite the paper. No GPU.

Blocking everything else. Results in JSON while the PDF says something else are
worth nothing. Sections 5.1 and 5.2 shrink, the gauntlet moves to an appendix,
the ground-truth experiment gets the section it does not have, and the abstract
leads with C1.

Gate: no number appears in the PDF that is not reproduced by a script in
`scripts/`. On 2026-09-15 three figure titles advertised the 40-point sample on
30-point plots, which is why this is a gate and not a preference.

### 2. Re-score one published steering result. One session.

Take a concept and model from ActAdd or CAA, reproduce the published sweep on
this harness, report both summaries.

Both outcomes are publishable and the paper should be written so either drops
in:

- the effect shrinks or flips under the signed metric, which is the abstract's
  first sentence;
- the effect holds, which shows the check is applicable rather than a
  complaint, and the recommendation becomes "here is the check, and here is a
  case that passes it".

Pre-commit in writing to reporting whichever occurs, before the run.

### 3. Concepts 10 to 20+, ground-truth concepts 4 to 10. Two to three sessions.

The largest credibility win available. The Limitations already concede that
cluster inference over-rejects at ten clusters and that the primary interval is
indicative rather than exact; a reviewer will quote that back. Twenty concepts
is engineering on template-generated stimuli, not research.

Design constraint that matters for C4: choose ground-truth readouts whose
**baseline position varies independently of the concept**. The current four do
not, which is exactly why Part 4 is not yet a result.

### 4. Evaluation prompts 6 to 30 on the judge-scored concepts. Folded into 3.

Appendix B states that six prompts bound the resolution of every number
computed from them. The ground-truth concepts already use thirty. No reason to
keep the lower bound on concepts being re-run anyway.

### 5. Scale sweep. One to two sessions.

All four models are instruction-tuned and between 7B and 9B. `run_kaggle.py`
already lists Qwen 0.5B, 1.5B and 3B for this. Whether the absolute/signed
divergence grows or shrinks with scale is a question that will be asked and the
answer is cheap.

---

## Part 4. The mechanism question, and why it is not yet a result

Across all 56 measured points, headroom -- how far the unsteered baseline sits
from the nearer extreme of the readout -- correlates $-0.448$ with whether a
sign resolves. The direction is the opposite of the obvious guess: concepts
whose baseline sits at a **floor** resolve, mid-range ones do not.

| concept | baseline | resolves |
|---|---|---|
| `gt_french` | 0.00 | 3 of 4 models |
| `gt_uppercase` | 0.03 | 2 of 4 |
| `gt_digits` | 0.00 | 0 of 4 |
| `gt_length` | 0.55-0.69 | 0 of 4 |

It is interpretable. A behavior the model never produces by default can be
pushed into unambiguously; a mid-range behavior drifts both ways and the signed
area cancels.

At four ground-truth concepts it is also effectively n=1. "Mid-range never
resolves" is "`gt_length` never resolves", and headroom cannot be separated from
concept identity. Publishing it as a predictor would be the same overclaiming
the reviewers already caught, which is why item 3 carries a design constraint
instead of this section carrying a finding.

If it survives ten ground-truth concepts with baselines that vary
independently, the paper stops being purely negative and gains a second
contribution. That is the most plausible route to an 8 that does not depend on
somebody else's published numbers.

**Decision required before item 3 runs:** commit to writing this section
whatever it returns, including null. A validated tool plus an honest null is
strong. A predictor fitted to one concept is not.

---

## Part 5. Open decisions

**Deadline.** Unset. This determines whether items 3 to 5 are feasible or
whether the paper ships on 1 and 2. Everything above is written so that 1 and 2
alone produce a coherent submission.

**Whether Gemma stays withheld.** Yes, and it is now settled on evidence rather
than judgment: unresolved and negative under two judges, and the least movable
model with no judge in the loop at all. The three-way withholding table stays
and its middle row remains the reported one.

**Whether to keep the four-layer band.** Yes, for now. All sixteen ground-truth
points are swept with it, and controllability over a band is a maximum across
four layers, which is not comparable to a single layer. If item 3 re-sweeps
everything anyway, single-layer becomes available at a quarter of the cost;
that is the moment to switch, not before.

---

## Part 6. Declined, recorded so the decision is visible

- **A second and third human annotator.** The right fix for the judge and
  outside a solo budget. Limitations says so, and the ground-truth experiment is
  the substitute: it removes the judge rather than averaging it.
- **Models above 9B.** Free-tier T4s. The scale sweep goes downward instead.
- **A paid API judge.** Same reason, and it would break the released pipeline's
  reproducibility.
- **Re-running the original 40-point study on the extended grid.** The grid
  matters for concepts whose response starts past the default range, which is a
  ground-truth phenomenon. The judge-scored concepts move well inside it.
- **Reviving the gauntlet as a main-text contribution.** It is computed on the
  disowned metric and its verdicts rest on margins of a few thousandths. The
  terminology distinction survives in an appendix; the verdicts do not.
