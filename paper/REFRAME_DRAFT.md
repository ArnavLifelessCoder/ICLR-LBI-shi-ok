# Reframe: executed

This plan has been applied to `paper.tex`. Numbers below are the final ones, on
the 30-point sample with Gemma withheld. Figures come from
`scripts/make_review_figures.py`, tables from `scripts/make_review_tables.py`,
and the gap map from `scripts/rebuild_gapmap_withheld.py`.

## Title

**Moved but Not Steered: Absolute-Deviation Metrics Overstate Control of
Concept Directions**

The old title asserted the thing the saturated readability axis cannot
establish. Both reviewers said so independently.

## The three decisions

1. **Reframe** around the measurement result. Done.
2. **Withhold Gemma.** It clears the absolute positive-control floor ($0.110$
   against $0.10$) but fails directionally: sentiment monotonicity $-0.68$,
   signed area $-0.039$. Since the directional reading is the one the paper
   relies on, the preregistered rule withholds it. Main text is now three
   models, 30 points; Gemma survives only in Appendix C.
3. **Harsher framing.** Adopted, and the data supports it more strongly at 30
   points than at 40.

## Final numbers

| Quantity | 40 points | 30 points (final) |
|---|---|---|
| Readability at AUROC 1.00 | 31/40 | 24/30 |
| Readability vs controllability | $+0.177$ | $+0.143$ ($p=0.45$) |
| Gap-map Spearman (normalized) | $0.14$ [$-0.24$, $0.51$] | $0.12$ [$-0.28$, $0.58$] |
| H1, absolute metric | $-0.451$ [$-0.723$, $-0.041$] | $-0.450$ [$-0.748$, $+0.038$] |
| H1, signed metric | $+0.115$ | $+0.141$ [$-0.265$, $0.444$] |
| Spearman(absolute, signed) | $-0.058$ | $-0.009$ |
| Points moving the wrong way | 20/40 | 16/30 |
| Median directional share | $0.438$ | $0.387$ |
| Median monotonicity | $+0.033$ | $-0.039$ |
| Sign audit (same-signed concepts) | 1/10 vs 1.25 expected | 2/10 vs 2.50 expected |
| Ceiling: overlap vs max usable | $+0.012$ | $-0.096$ ($p=0.61$) |
| Sweeps reaching grid end | 37/40 | 27/30 |
| Danger zone, CI-confirmed | 2 | **1** (topic_science on Mistral) |
| Gauntlet entered / immovable | 17 / 9 | 12 / 6 |

**The consequence of withholding Gemma:** the H1 interval crosses zero. Applying
the preregistered control consistently removes the paper's second contribution
as a significance claim. Only the sign survives, negative in all ten
leave-one-out refits. This is stated plainly in Section 5.4.

## Sign-convention audit

`certainty` responds negatively on every model, which would normally suggest an
inverted polarity that makes the signed metric score it backwards. It does not
survive the check: 2 of 10 concepts are same-signed across all three models
against 2.5 expected under a random-sign null. No systematic polarity error, so
the wrong-way movement is real. The stronger reading is that within-concept
response direction is close to random, meaning the intervention is not producing
a reliable directional effect at all.

## Still open

- **Page limit.** Estimated 9.1 pages against a 9-page limit, and this estimate
  is not a substitute for compiling. Trims already applied: Limitations detail
  moved to Appendix H, the H1 figure demoted to Appendix G, related work and
  introduction compressed, the duplicated withholding paragraph removed.
- **Second judge and second annotator.** Cannot be done from cached artifacts:
  only 720 generations are stored, two per curve point. Requires a rerun.
- **sycophancy.** Readability $0.57$ on Llama is a failed probe that still sits
  in the primary correlation. Excluding it moves the readability-controllability
  Spearman from $0.177$ to $0.230$ on the 40-point sample.
