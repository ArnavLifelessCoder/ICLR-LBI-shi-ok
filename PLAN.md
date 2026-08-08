# PLAN.md

Execution plan for *Legible but Immovable*. The design document says what the study is. This says how it gets built, in what order, what has to be true before each step is allowed to proceed, and what a reviewer will do to each piece of it.

Written under one hard constraint: **nothing here adds compute.** Every addition is CPU-only, weights-only, reuses a cached activation, or is a change to how existing numbers are analyzed and reported. Where an obvious improvement would cost GPU hours, it is named and declined explicitly, with the reason, so the decision is visible rather than accidental.

---

## Part 0. How to use this document

Read Part 1 before writing any code, because it determines what "done" means. Freeze Part 2 before the first real model run and do not touch it afterward. Work through Part 3 in order and do not skip a gate. Parts 4 through 7 are written to be consulted, not read start to finish: the objection ledger during writing, the rebuttal playbook during the review cycle.

The organizing idea: on a solo timeline the paper is not lost by running out of ideas, it is lost by discovering in week 10 that something in week 2 was wrong. Every gate below exists because a specific failure is cheap to catch early and expensive to catch late.

---

## Part 1. The reviewer model

Four readers. Write for all four. Each one can sink the paper alone, and each is attacking something different.

### R1, the interpretability skeptic

Believes probing papers routinely mistake dataset artifacts for model internals. Their opening move on any probing result is "your probe learned the vocabulary." Their second move is "you picked the best layer on the test set."

What convinces them: selectivity controls, held-out template families, a surface-text baseline that fails where the probe succeeds, and layer selection made on a validation split. What does not convince them: high AUROC.

Their worst question, and it is a good one: *"Your danger zone is defined as high readability plus low controllability. If readability is inflated by lexical shortcuts on exactly the templated concepts, and those same concepts are hard to steer because the templates are unnatural, you have manufactured the entire finding."* This is the single most dangerous objection in the paper and it is answered by Part 3, Phase A, not by anything later.

### R2, the steering practitioner

Has implemented ActAdd, CAA, or RepE themselves. Knows steering is finicky. Their default explanation for any null steering result is that the implementation is wrong, the layer is wrong, or the coefficient range is wrong.

What convinces them: a positive control run through identical machinery on the same model at the same layer, a swept coefficient range that visibly reaches the fluency ceiling, more than one direction-derivation method, and the immovability gauntlet. What does not convince them: assurances.

Their worst question: *"Steering effects are known to be layer-sensitive. You chose the intervention layer by probe accuracy. Why would the layer where a concept is most readable be the layer where it is most steerable?"* This is a genuinely hard question and the honest answer is that it would not necessarily be, which is why the protocol below sweeps a layer band and reports the best-over-band controllability. See Part 2, item P4.

### R3, the statistics and reproducibility reviewer

Counts the data points. Notices that concept-model pairs are not independent observations. Asks about multiple comparisons, about power, and about how many things were tried before the reported analysis.

What convinces them: a preregistration block, a stated unit of generalization, cluster-level resampling, one primary test with everything else labeled exploratory, and an honest power statement. What does not convince them: bootstrap CIs computed over the wrong resampling unit.

Their worst question: *"You have roughly fifty points, but they come from twelve concepts across five models. Your effective sample size for a claim about concepts is twelve, not fifty. What is your power to detect the effect you are claiming?"* Part 2, item P7 answers this in advance and it is uncomfortable, which is why it goes in the paper rather than waiting for the rebuttal.

### R4, the area chair deciding venue fit

Does not read the appendix. Reads the abstract, the main figure, and the limitations. Wants to know whether the paper's claim is sized correctly for its evidence, and whether the contribution is conceptual or incremental.

What convinces them: a claim that is exactly as large as the evidence supports, a limitations section that names the scale ceiling before they do, and a framing that is about a phenomenon rather than a method. What does not convince them: a safety framing on sub-10B models, which is why Experiment 5 was demoted.

Their worst question: *"This is a characterization of models under 10B parameters. Why should I believe it says anything about the systems interpretability-based safety actually cares about?"* The honest answer is that it does not directly, and the paper should say so. Claims about small models, tested on small models, stated as being about small models, are defensible. The overreach is what gets rejected.

---

## Part 2. The preregistration block

Freeze this before the first real model run. Commit it to the repo with a timestamp so the freeze is verifiable. Everything decided here is a researcher degree of freedom that a reviewer would otherwise be right to suspect.

**P1. Concept inclusion.** A concept enters the study only if its contrastive set passes the surface-shortcut audit: TF-IDF plus logistic regression on raw text, evaluated on held-out template families, must reach AUROC below 0.65, or at least 0.15 below the activation probe's held-out-family AUROC on the same split. Concepts that fail are rebuilt once and re-audited. A concept that fails twice is dropped and reported as dropped, with its numbers in the appendix. The final concept count is whatever survives, floor of eight.

**P2. Probe layer selection.** Chosen on a validation split of template families that is disjoint from both the training families and the test families. Never on test AUROC. Readability is the test-split AUROC at the validation-selected layer, averaged over five seeds, with a bootstrap CI.

**P3. Selectivity control.** Every reported readability number is accompanied by a shuffled-label control probe at the same layer with the same capacity. Selectivity is the difference. A concept whose control probe exceeds 0.6 AUROC is flagged in the table.

**P4. Steering layer protocol, and why it is not symmetric with P2.** Readability is measured at one validation-selected layer. Controllability is measured as the best over a preregistered band of layers, specifically the validation-selected probe layer plus or minus 20 percent of model depth, sampled at four evenly spaced layers in that band.

This asymmetry is deliberate and must be stated in the paper, because it looks like a flaw until it is explained. The reason: the paper's claim is that a readable concept can be *unsteerable*, and that claim gets stronger, not weaker, the harder you try to steer. Giving controllability the more generous protocol biases the study against its own headline finding. A reviewer who notices the asymmetry and is not told why will assume the worse explanation, so put it in the methods section in one sentence: the protocol favors the null hypothesis.

Cost check: four layers instead of one is more forward passes. This is not new compute, it is the compute the original design already implied under Experiment 2's "try steering at multiple layers" fallback, now moved from a conditional fallback into the standard protocol and bounded so it cannot expand. The band is fixed at four layers and does not grow if results are disappointing.

**P5. Coefficient sweep.** Fixed grid in RMS units, identical for every concept and model, spanning zero up to the point where the fluency ceiling is crossed for the positive control. Preregistered as eight points on that grid. Not tuned per concept.

**P6. Danger zone membership.** Raw AUROC at or above 0.9 with raw dose-response AUC at or below 0.05, *and* the bootstrap CI on each must exclude the respective threshold. The CI requirement is new here relative to the design doc and it matters: without it, a concept lands in the danger zone by scatter, which is a milder version of the same error that revision R10 caught with normalization. Membership is claimed only after the full immovability gauntlet.

**P7. Primary statistical test, unit of generalization, and power.** The unit of generalization is the **concept**, not the concept-model pair. Points from the same concept across five models are not independent observations. The primary test is a partial Spearman correlation between output overlap and controllability, controlling for readability, with a **cluster bootstrap resampling concepts with replacement**, not points.

Power, stated honestly and in the paper: with ten to twelve concepts as the effective sample, this design detects a strong monotone relationship, roughly rho of 0.7 or above, with reasonable confidence. It does not have the power to establish a moderate one. Therefore the preregistered claim is binary: either output overlap shows a strong relationship to controllability or the study is uninformative about weaker ones. A moderate observed effect is reported as suggestive and not as confirmation. Writing this before seeing the data is what makes it credible afterward.

**P8. Exploratory analyses.** E1 (participation ratio) and E2 (low-variance PC alignment, conditioned on the fluency-limited subset) are labeled exploratory in the paper, get Benjamini-Hochberg correction across the exploratory family, and cannot be promoted to the primary claim regardless of how they turn out. If an exploratory analysis produces the paper's most interesting number, it goes in the discussion as a hypothesis for follow-up work, not in the abstract.

**P9. Positive control gate.** Sentiment must reach a dose-response AUC above a preregistered floor on every model. A model that fails the control has its controllability numbers withheld from the analysis entirely and reported as a pipeline failure for that model. This is written down now precisely so it cannot be relaxed later when a model fails and the deadline is close.

**P10. Behavior judges.** Two independent judges plus a human check, all free. Judge one is a validated HuggingFace classifier per concept, running on CPU. Judge two is a local instruct model already downloaded for the study, prompted as a scorer, running on the same T4. Judge three is you, hand-labeling 100 randomly sampled steered outputs stratified across concepts and strengths. Report Krippendorff alpha across all three. No paid API judge, which keeps cost at zero and also removes a reproducibility objection, since anyone can rerun a local judge and nobody can rerun your API credits.

---

## Part 3. Phased execution

Five phases. Each has a deliverable, a gate, and a branch for what happens if the gate fails. The gates are hard. A failed gate means stopping and fixing, not proceeding while making a note.

### Phase A, weeks 1 to 3. Datasets and the shortcut audit

This is CPU-only and it is the phase that determines whether anything downstream means anything. R1's worst question lives here.

Work: run the literature check from design Section 11 across two days. Build contrastive sets for twelve to fifteen candidate concepts as minimal pairs, matched on topic, length bucket, register and named entities. Tag every pair with its template family. Split families into train, validation and test, disjoint. Build the TF-IDF surface audit and run it on everything. Rebuild what fails. Assemble a small naturalistic evaluation set, real text rather than templates, for the four concepts most likely to carry the paper.

Deliverable: a frozen concept set with an audit table showing, per concept, the surface baseline AUROC and the pass or fail decision, published as an artifact.

**Gate A.** At least eight concepts pass the audit, and the surviving set spans the expected range of steerability, meaning it contains both concepts prior work says steer easily and concepts prior work suggests do not. A set that is all easy or all hard cannot show a gap.

Branch if Gate A fails: if fewer than eight pass, the templating approach is the problem, not the individual concepts. Switch the failing concepts from generated templates to sampled natural text with human-verified labels, accept a smaller and noisier dataset per concept, and re-audit. This costs time, not compute. If the range is too narrow, add sentiment and formality at the easy end, which are known-good and cheap.

Common failure to watch for: negation. Contrastive pairs built by inserting "not" are trivially separable by surface features and will fail the audit. Build the negative member as an independent generation with the same constraints, not as an edit of the positive.

### Phase B, weeks 4 to 5. The readability axis

Work: extract residual-stream activations at every layer for every concept and model, cache to disk as float16. Train per-layer probes with five seeds. Select layer on validation. Report test AUROC with selectivity control and bootstrap CI. Run the naturalistic transfer check on the four headline concepts.

Deliverable: the full readability table, plus a per-concept layer profile figure for the appendix.

**Gate B.** The four headline concepts retain at least 0.8 AUROC when transferred from templates to naturalistic text, and every reported probe beats its shuffled-label control by a clear margin.

Branch if Gate B fails: if naturalistic transfer collapses, the concept is a template artifact regardless of what the surface audit said, because the audit tests lexical shortcuts and this tests distributional ones. Demote that concept out of the headline set and say so. If transfer collapses for all four, the paper's scope narrows to templated stimuli and the title and abstract must say "in templated contrastive stimuli," which is a real result about a real limitation and is still publishable at a workshop.

Cost note: the naturalistic check reuses the same extraction code on a few hundred extra examples. It is a rounding error against the main extraction pass.

### Phase C, weeks 6 to 8. The controllability axis

The fiddliest phase and the one R2 is reading. Budget the most slack here.

Work: derive three directions per concept from cached activations, difference of means, probe weights, and the RepE reading vector as the first principal component of paired differences. Steer by activation addition over the preregistered layer band and coefficient grid. Score behavior with all three judges. Track perplexity and repetition for the fluency ceiling. Run the sentiment positive control on every model first, before anything else, because a broken pipeline discovered in week 8 costs the paper.

For any concept that looks flat, run the full immovability gauntlet: single-layer addition, multi-layer addition, clamping, directional ablation, probe-weight direction. Only concepts that resist all of them are called immovable.

Deliverable: the full controllability table with CIs, judge agreement numbers, and the gauntlet traces for every flat concept.

**Gate C.** The positive control steers cleanly on every model, judge agreement clears a preregistered floor, and at least one concept survives the full gauntlet as immovable, or the flat concepts are cleanly explained by the fluency ceiling instead.

Branch if Gate C fails: if the positive control fails on a model, that model is withheld per P9 and the study proceeds with the rest, reported honestly. If the control fails everywhere, stop and debug the harness, because nothing downstream is interpretable. If no concept is immovable and everything steers, the paper becomes the correlation story, "detection largely does predict control, quantified across concepts and models for the first time, with these residuals," which the design already anticipates in Section 7 and which is a fine paper. Reframe the title, do not force the finding.

Watch for: judges agreeing with each other while both being wrong in the same direction. A classifier and an LLM judge can share a bias, particularly toward scoring fluent text as more concept-positive. The human spot-check exists specifically to catch this and it is why P10 has three judges rather than two.

### Phase D, week 9. The gap map

Work: build the scatter. Normalize within model for the correlation and the plot, apply danger-zone thresholds to raw values with CI exclusion per P6. Compute the readability-controllability correlation with the cluster bootstrap over concepts. Write the one paragraph on where the safety-relevant concepts land.

Deliverable: figure 1, the paper's thesis in one image.

**Gate D.** The figure supports a single clear sentence. Either the axes come apart, or they do not, or they come apart only for a nameable subset. If you cannot write that sentence, the figure is not ready.

Branch: no branch needed. Every outcome here is a paper, which was the point of choosing a characterization.

### Phase E, weeks 10 to 11. Geometry, scale, and the timebox

Work, week 10: compute output overlap from the unembedding matrix and cached directions. Weights-only, no forward passes, should take days not weeks. Run the P7 primary test.

Work, week 11: exploratory E1 and E2 with BH correction. Experiment 6 scale re-run over the Qwen sizes, which is existing code over already-downloaded models. If H1 failed, the base-versus-instruct reshaping fallback on the Qwen pair, also already downloaded.

**Gate E.** The primary test is run once, reported whatever it says, and the timebox holds. No extension into week 12 to keep looking.

Branch if H1 fails: report the preregistered null. Because design Section 2.5 states the first-order account in advance, a null is the finding that the standard account does not explain the gap, which is substantive. Figure 3 becomes the base-versus-instruct analysis. Do not go feature hunting. The value of the preregistration is entirely destroyed by one unregistered search, and R3 will find it.

### Phase F, weeks 12 to 14. Writing and release

Work: three figures clean, artifacts packaged, the short demo video of the dissociation, limitations written before the abstract is finalized. Run the objection ledger in Part 4 as a checklist and confirm every row has a home in the paper.

---

## Part 4. The objection ledger

Every row is something a reviewer will say. The paper must have a place where it is already answered, before the rebuttal. If a row has no home, it is a hole.

| # | Objection | Who says it | Answered by | Lives in |
|---|---|---|---|---|
| 1 | The probe learned vocabulary, not the concept | R1 | TF-IDF surface audit with preregistered threshold | Methods, plus audit table artifact |
| 2 | You picked the best layer on test | R1 | Validation-split layer selection, P2 | Methods |
| 3 | Templates are not real language | R1 | Naturalistic transfer check on headline concepts | Results subsection |
| 4 | Your steering implementation is broken | R2 | Sentiment positive control, same panel as figure 2 | Figure 2, Methods |
| 5 | You steered at the wrong layer | R2 | Preregistered four-layer band, best-over-band, P4 | Methods, one sentence on the asymmetry |
| 6 | You used only one direction-derivation method | R2 | Three directions, diff-of-means, probe weights, RepE reading vector | Methods |
| 7 | Diff-of-means is not a real baseline | R2 | Statement that it is exactly CAA and ActAdd's one-pair form | Related work |
| 8 | Your coefficient range was too small | R2 | Fixed grid extending past the fluency ceiling, shown in the dose-response figures | Appendix figures |
| 9 | The null is just fluency breaking first | R2 | Ceiling-limited versus flat distinction reported per concept, ties to M3 | Results |
| 10 | Fifty points from twelve concepts is not fifty observations | R3 | Cluster bootstrap over concepts, P7 | Statistical analysis section |
| 11 | How many analyses did you run | R3 | Preregistration block with timestamp, one primary test, BH on exploratory | Methods, plus repo commit |
| 12 | You are underpowered | R3 | Stated power for rho 0.7, binary preregistered claim, P7 | Statistical analysis section |
| 13 | Danger-zone membership is scatter | R3 | Raw thresholds plus CI exclusion, P6, and revision R10's normalization argument | Methods |
| 14 | Judges are unvalidated | R3 | Three judges including human spot-check, Krippendorff alpha reported | Methods |
| 15 | Why should output overlap predict anything | R1, R4 | Section 2.5 first-order derivation, stated before the experiment | Introduction and theory section |
| 16 | This is sub-10B, so what | R4 | Scope stated in abstract, scale ceiling named in limitations before they raise it | Abstract, Limitations |
| 17 | The safety claim is unsupported | R4 | Experiment 5 demoted, claim sized to one paragraph and one discussion sentence | Discussion |
| 18 | Novelty over the two motivating papers | R4 | Neither maps detection against control across concepts, nor builds a predictor | Related work, first paragraph |

Rows 5, 10 and 12 are the three where the honest answer is partly unfavorable. Put those in the paper voluntarily. A limitation you name yourself is a sign of care. The same limitation found by a reviewer is a sign of carelessness, and it is the same limitation.

---

## Part 5. Paper skeleton, with what each section is doing

**Abstract.** One sentence of setup, one of method, the finding, the predictor result, the scope caveat. The scope caveat is in the abstract on purpose. R4 reads the abstract and the limitations and nothing else, and the caveat there is what stops them reading the paper as an overclaim.

**Introduction.** Open with the assumption being tested, that reading a concept implies a handle on it, stated as an assumption the field makes implicitly rather than as a strawman anyone defends explicitly. Then the two motivating papers. Then the contribution in three bullets: the two-axis measurement, the gap map, the predictor.

**Section 2, why a gap should be expected.** The theory from design Section 2.5. This section exists to make Experiment 4 a test rather than a search, and to make a null informative. It is also the section R4 uses to decide whether the paper is conceptual or incremental, so it carries more weight than its length suggests.

**Section 3, methods.** Definitions, the preregistration block reproduced or linked, the layer asymmetry explained in one sentence, the judges, the audit.

**Section 4, readability.** The table, the selectivity controls, the naturalistic transfer.

**Section 5, controllability.** The positive control first, then the main results, then the gauntlet for flat concepts. Ordering matters: showing the control works before showing things that do not is the difference between a finding and an excuse.

**Section 6, the gap map.** Figure 1 and the one sentence it supports. The safety-concept paragraph goes here.

**Section 7, geometry.** Primary test, then exploratory clearly fenced off.

**Section 8, limitations.** Scale ceiling. Templated stimuli. Cluster-level power. The output-overlap proxy standing in for a Jacobian the compute budget cannot afford. Written before the abstract is finalized, because writing limitations honestly sometimes reveals the abstract is overclaiming.

**Section 9, discussion.** What this means for interpretability-based safety, sized to one paragraph.

---

## Part 6. Figures

**Figure 1, the gap map.** Readability against controllability, normalized within model, every concept-model pair a point, concepts distinguished by marker and models by color. Danger zone shaded, membership decided on raw values per P6. Error bars on both axes for the danger-zone points at minimum. The single most important design choice here is that the shading is drawn at the raw-threshold positions after transformation, not at fixed positions in normalized space, and the caption says so.

**Figure 2, the dissociation.** Two dose-response curves on one set of axes, sentiment control rising, a danger-zone concept flat, both with CIs, both with the fluency ceiling marked as a vertical line. Identical machinery, stated in the caption. This is the figure that answers R2 without a word of text.

**Figure 3, the predictor.** Output overlap against the controllability residual after readability is partialled out. Points, fit line, CI band, the preregistered rho and its CI in the corner. If H1 failed, this becomes the base-versus-instruct reshaping comparison and the caption says plainly that the preregistered predictor did not hold.

All three figures must be readable in grayscale and legible at single-column width. Reviewers print papers.

---

## Part 7. Rebuttal playbook

Prepared answers for the reviews you will actually get.

**"Add a larger model."** You cannot, and saying so plainly is better than promising it. The response: the scale ceiling is a stated limitation, Experiment 6 characterizes the trend across 1.5B to 8B, and the phenomenon's existence does not depend on scale even if its magnitude does. Offer the trend, not a promise.

**"Add steering method X."** Point at the three direction-derivation methods and the gauntlet, then at the positive control. The argument is not that no better method exists, it is that the concept resists every standard method while an identical pipeline moves the control cleanly. If X is cheap and reuses cached activations, add it. If it needs training, decline explicitly and give the reason.

**"The effect is weak."** If it is, you preregistered that a moderate effect is reported as suggestive. Say that. A reviewer who sees you holding to a preregistered interpretation against your own interest will trust the rest of the paper more.

**"How do we know the concept sets are good."** The audit table is a released artifact. Point at it, and at the concepts you dropped, because dropped concepts are evidence the audit was real.

**"This is a workshop paper."** It might be. The ladder is BlackboxNLP, then SoLaR, then EMNLP Findings, then TMLR. TMLR is genuinely the best fit for a careful characterization with no benchmark chasing, and is worth considering ahead of Findings rather than below it.

---

## Part 8. Kill criteria

Stated in advance so that stopping is a decision rather than a defeat.

- Fewer than eight concepts survive the audit after one rebuild round, and the natural-text fallback also fails. Then the study is about datasets and not about representations, and it should be re-scoped or shelved.
- The positive control fails on every model after two weeks of debugging. Then the harness is wrong in a way that invalidates everything and no result should be published.
- The literature check finds the exact study already done, many concepts, several models, both axes, with a predictor. Then pivot to the narrower "predict steerability from geometry" slice per design Section 11, which stays open.

Anything short of these is not a kill, it is a branch, and every branch above lands on a paper.

---

## Part 9. The zero-cost ledger

Everything in this plan, confirmed against the compute constraint.

| Addition | Cost | Why it is free |
|---|---|---|
| Theory section | Writing only | No experiment |
| Surface-shortcut audit | CPU, seconds | TF-IDF on text, no model |
| Naturalistic transfer check | Negligible | A few hundred extra extractions on existing code |
| RepE reading vector | One PCA | Uses activations cached in Phase B |
| Positive control | Included | Sentiment is already in the concept set |
| Four-layer steering band | Reallocated | Was already the Experiment 2 fallback, now bounded and standard |
| Output overlap (H1) | Weights only | One matmul against the unembedding, no forward pass |
| Third judge | Free | You, hand-labeling 100 outputs |
| Local LLM judge | Included | Model already downloaded for the study |
| Cluster bootstrap | CPU | Resampling of existing numbers |
| Experiment 5 demotion | Negative | Removes a week |
| Experiment 4 trimmed to one primary feature | Negative | Fewer features, fewer tests |

Declined on cost grounds, recorded so the decision is visible: CCA steering, learned steering vectors, any model above 9B, Jacobian-based causal alignment measured directly, and a paid API judge.
