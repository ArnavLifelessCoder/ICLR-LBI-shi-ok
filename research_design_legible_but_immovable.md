# Research Design
## Legible but Immovable: When Reading a Concept Fails to Grant Control

A study of the gap between what a model's activations let you *detect* and what they let you *steer*, why some concepts are readable yet uncontrollable, and what that means for interpretability-based safety.

---

## 0. How to read this document

This is a full research plan. Every experiment has three parts: what you do, what you measure, and what to try if it does not work. The fallback lines matter as much as the main plan, because on a solo timeline the difference between a finished paper and an abandoned one is usually having a Plan B ready the day something breaks.

Constraints this design respects, set over the whole planning process: free Colab or Kaggle T4 (16GB) only, no training anywhere, 3-4 months solo, accept-regardless-of-result (it is a characterization, both outcomes are findings), low scoop risk, a genuine punch, and citation upside through fresh motivating work plus released artifacts.

Honest ceiling: the realistic ladder is strong workshop (BlackboxNLP, SoLaR) then EMNLP Findings then TMLR, with an outside main-track shot if the predictive-geometry result is clean. Free-T4 caps you below the 70B robustness that main-track reviewers want, so the design maximizes the floor and treats top-tier as upside.

---

## 1. The one-paragraph thesis

Interpretability-based safety quietly assumes that if you can read a concept off a model's activations, you have a handle on it: probe for deception, probe for harmful intent, and you are part way to controlling it. This paper shows that assumption is false in a measurable, predictable way. Across many concepts and models we measure two things separately: how well a linear probe *detects* a concept (readability) and how much steering along that concept's direction actually *changes behavior* (controllability). We show these come apart, that some concepts are perfectly legible yet nearly immovable, and that the gap is predictable from the concept's geometry before you ever try to steer. We end by showing safety-relevant concepts can land on the wrong side of this gap, which means a good probe can give false confidence about control.

---

## 2. Why this is open, not crowded

The two papers that motivate this are recent and they *open* the gap rather than closing it.

"Perfect Detection, Failed Control: The Geometry of Knowing vs. Steering in Language Models" reports the phenomenon on a narrow case (entity knowledge) and shows steering strength does not transfer across models, but it does not systematically map detection against control across many concepts, nor build a predictor of the gap. Its authors leave the full activation-trajectory comparison as explicit future work.

"There Is More to Refusal than a Single Direction" (Feb 2026) shows refusal is many geometrically distinct directions that all act as one shared control knob, which hints that readability and controllability decouple, but it studies only refusal.

Nobody has done the general study: many concepts, several models, detection vs control as the two axes, plus a geometric predictor of the gap. That is this paper. Cite both works as motivation and position your contribution as the systematic characterization they point toward.

Beyond the two motivating papers, the related-work section must position against the standard steering literature or reviewers will bounce it: ActAdd (Turner et al.), Contrastive Activation Addition (Rimsky et al.), Inference-Time Intervention (Li et al.), Representation Engineering (Zou et al.), the refusal-direction ablation paper (Arditi et al.), and the linear representation hypothesis (Park et al.). None of these measure detection and control as separate axes across many concepts, which is exactly the room this paper occupies — but they define the steering methods this paper uses, so they are baselines, not competitors.

Required first step before writing: run a fresh literature check with the queries in Section 11, because this area moves fast. Budget two days for it.

---

## 2.5 Why a gap should exist (the mechanism the experiments test)

The original draft measured readability, then controllability, then looked for geometry that connects them, without saying why any geometric quantity ought to matter. That is the first thing a reviewer at a selective venue will push on, and "we tested several features" is not an answer. The argument below costs nothing to include and turns Experiment 4 from a feature hunt into a hypothesis test.

**The asymmetry.** Probing and steering interrogate opposite halves of the network. A probe at layer L asks a question about the *encoder*: given everything the first L layers computed, is the concept written into the residual stream in a linearly separable way? Steering at layer L asks a question about the *decoder*: is the computation of layers L+1 through the output sensitive to movement along this particular direction? Nothing in the architecture ties these together.

Write it out. Let h be the layer-L residual stream, v the unit concept direction, and f the map from h to the behavior score. To first order, steering by strength a changes behavior by

    delta_behavior ~ a * ||grad f(h)|| * cos(v, grad f(h))

so controllability is governed by how much of v lies in the subspace the downstream computation is actually sensitive to. Readability, meanwhile, is governed by a Fisher-style ratio: the between-class separation along v over the within-class variance along v. A concept can have enormous between-class separation along a direction that the rest of the network barely reads. That is the entire phenomenon, stated in one line: **readability is a property of how the concept is written, controllability is a property of whether anything downstream reads it.**

**Three mechanisms that produce a gap.** These are not competing theories, they are the three distinct ways the alignment term above can be small, and each makes a different measurable prediction.

*M1, read/write mismatch.* The concept is written into a subspace that carries little influence on the output distribution. It is a spectator feature: fully present, causally inert. Prediction: controllability falls as the concept direction's overlap with the model's output-effective subspace falls. This is the primary hypothesis, because it is the most direct restatement of the asymmetry and because it is measurable from weights alone with no extra forward passes.

*M2, redundant encoding with downstream restoration.* If the concept is written by many components into many directions, as the refusal paper found, then perturbing one of them leaves the others intact and later layers re-derive the concept from the surviving copies. The intervention is undone before it reaches the output. Prediction: controllability falls as the effective dimensionality of the class difference rises. This is where participation ratio belongs, and this is the reason it belongs there, rather than because it was on a list of things to try.

*M3, off-manifold steering.* If the concept direction has small variance under the model's natural activation distribution, then moving along it far enough to change behavior also moves activations somewhere the model has never been, and fluency collapses first. Prediction: controllability, as measured up to the fluency ceiling, falls as the concept direction aligns with low-variance principal components of the residual stream. Note that this mechanism predicts a *ceiling-limited* zero rather than a flat dose-response curve, so Experiment 2 can already tell M3 apart from M1 and M2 by whether steering broke fluency or simply did nothing.

**What this buys.** Three things. It gives Experiment 4 a preregistered primary hypothesis instead of a regression over whatever features were available. It gives the danger zone a mechanistic story rather than a scatter plot. And it makes a null result informative: if none of M1, M2 or M3 predicts the gap, the honest conclusion is that the gap is not explained by the standard first-order picture of steering, which is a real statement about a real gap in the field's understanding rather than a shrug.

---

## 3. Core definitions (so the metrics are unambiguous)

Readability of a concept: the test AUROC of a linear probe trained to detect the concept from residual-stream activations at the best layer, evaluated on **held-out template families** (not just held-out examples), and reported alongside a control-task probe (shuffled labels) so selectivity is explicit. High readability means the concept is linearly present, not that the probe memorized surface cues.

Controllability of a concept: the effect of steering along the concept direction, with the steering coefficient **normalized in units of the residual-stream RMS norm at the intervention layer** (so strengths are comparable across layers and models). The headline number is the area under the dose-response curve (behavior score vs normalized strength) up to the fluency ceiling, which is more robust than a single-point effect size. The fluency ceiling is pre-registered: the largest strength at which steered-output perplexity under the unsteered model stays below 2x the unsteered baseline and no degenerate repetition appears.

The gap: a concept sits in the danger zone when readability is high and controllability is low. The whole paper is about locating, explaining, and predicting that zone.

---

## 4. Models and concepts

Models (all public, base or instruct as needed, all fit a T4 in 4-bit for inference):
- Llama-3.1-8B-Instruct
- Qwen2.5-7B-Instruct (and its 1.5B and 3B siblings for the scale axis, free because they are pretrained and public)
- Mistral-7B-Instruct-v0.3
- Gemma-2-9B-it if it fits, otherwise Gemma-2-2b-it (Gemma is worth including because the motivating paper found it behaves differently, so it stresses your conclusions)

Concepts (aim for 10 to 15, spanning easy to hard to steer, so the gap has room to appear):
- Sentiment (positive vs negative)
- Formality
- Toxicity
- Sycophancy
- Refusal
- Honesty vs deception
- Topic or domain (a couple of these)
- Language register or a factual-vs-fictional distinction
- One or two safety-relevant capabilities (framed carefully)

The point of a wide concept set is that the story needs both ends: concepts that are readable and controllable (the expected case) and concepts that are readable but not controllable (the finding). If every concept behaves the same, that itself is a clean result, see the fallback.

**Two concepts carry special roles.** Sentiment is the *positive control*: it is reliably readable and reliably steerable in every prior steering paper, so if it does not steer in this pipeline the pipeline is broken and no other result should be believed. This is stated in advance so it functions as a gate rather than a post-hoc excuse. Refusal is the *negative-space anchor*: the motivating refusal paper already characterizes it, so agreement there is evidence the measurements are calibrated against published work.

**Dataset rigor is the largest single risk in this project, and gets budget accordingly.** If the contrastive sets leak lexical shortcuts, the probe is reading vocabulary rather than the concept, readability is inflated for exactly the concepts most likely to be templated, and the danger zone becomes an artifact of dataset construction. This failure is silent: the probe looks great, the steering looks flat, and the paper's headline claim is wrong. Three requirements, none of which need a GPU:

- **Surface-shortcut audit, preregistered.** For every concept, fit a TF-IDF plus logistic regression classifier on the raw *text* of the contrastive pairs, with the same held-out-template-family split as the probe. A concept passes only if the surface baseline reaches AUROC below 0.65, or at least 0.15 below the activation probe's held-out-family AUROC. Concepts that fail get rebuilt, not reported. Run this before any activation is extracted, because it costs seconds and can save weeks.
- **Minimal pairs mean minimal.** Positive and negative members of a pair differ in the concept and nothing else: same topic, same length bucket, same register, same named entities. Length and topic are the two cues probes latch onto first.
- **A naturalistic held-out set for the headline concepts.** For at least the three or four concepts that end up carrying the paper, assemble a small evaluation set of real text rather than templates, and confirm readability survives the transfer. Templates alone are the standard reviewer objection to probing work, and answering it for a handful of concepts is far cheaper than doing it for all fifteen.

Concept count is a slider, not a fixed number. Fifteen shaky concepts are worth less than ten that pass the audit. If the audit kills concepts, drop to ten rather than patching them.

---

## 5. Experiments, with fallbacks for each

### Experiment 1: Build the readability axis (probing)
What you do: for each concept and model, construct contrastive datasets (positive vs negative examples of the concept). Extract residual-stream activations at every layer at the last token. Train a linear probe (logistic regression) per layer. Record the best-layer AUROC as the concept's readability.
What you measure: per-concept, per-model readability (AUROC), and which layer is best.
If it does not work:
- Probes weak everywhere: your contrastive sets are noisy. Rebuild with tighter minimal pairs (change only the concept, hold topic and length fixed). This is the single most common failure and the fix is almost always better pairs.
- Best layer unstable across seeds: average over several seeds and report a band, and use mean-pooled activations over the response instead of last-token if last-token is noisy.
- Concept not linearly readable at all: keep it in the set as a "low readability" point, it is still data. If most concepts are unreadable, widen to easier concepts (sentiment, formality are reliably readable) so the axis has spread.

### Experiment 2: Build the controllability axis (steering)
What you do: for each concept, form a steering direction (difference of means between positive and negative activations, which is the standard and robust choice). Steer by activation addition at the best layer, sweeping the coefficient. For each strength, measure the target behavior with an automatic metric (a classifier or an LLM judge scoring the output for the concept) and also track fluency (perplexity or a coherence check) so you can find the strength just before output breaks.
What you measure: steering effect size at the pre-breakage strength, per concept per model. This is controllability.
If it does not work:
- Steering does nothing at any strength for a concept: this is the interesting case, not a failure, it is a "readable but immovable" candidate. Confirm it is real by trying alternative interventions before declaring it (see next bullet).
- Confirming a true zero: try (a) steering at multiple layers, (b) all-layer steering rather than single-layer, (c) directional ablation and clamping (set the projection to a target value) instead of pure addition, (d) the probe-weight direction instead of difference-of-means. Only call a concept uncontrollable if it resists all of these. This robustness is exactly what makes the finding defensible.
- Steering always destroys fluency before it changes behavior: lower the strength granularity, and report the concept as low-controllability with the fluency ceiling as the reason, which is itself a finding about entanglement.
- Judge metric noisy: use two independent judges (a trained classifier plus an LLM judge) and report agreement, so a reviewer cannot dismiss the behavior scores.

**Closing the "maybe your steering method is just bad" objection, at zero extra compute.** This is the single most likely reviewer attack on a null-controllability finding, and it can be answered without buying more GPU hours:

- *Baseline coverage by construction.* Difference-of-means at a single layer with a swept coefficient is exactly Contrastive Activation Addition, and ActAdd is the same operation with a one-pair direction. Say so explicitly in the paper rather than implying they are untried alternatives. Add one more canonical direction, the Representation Engineering reading vector (first principal component of the paired activation differences, Zou et al.), which is computed from activations already cached for Experiment 1 and therefore costs one PCA rather than one forward pass. With difference-of-means, probe weights, and the RepE reading vector, the paper covers three of the four standard ways to derive a concept direction.
- *The positive control does the rest.* Every reported steering run includes the sentiment control on the same model, at the same layer band, through the same judges. A figure showing sentiment moving cleanly while a danger-zone concept stays flat, under identical machinery, is a far stronger rebuttal than any additional method would be. Preregister the control: if sentiment fails to steer on a given model, that model's controllability numbers are withheld, not explained away.
- *Report the ceiling, not just the effect.* For each flat concept, state whether the dose-response curve was flat up to the fluency ceiling (consistent with M1 or M2) or whether fluency broke before behavior moved (consistent with M3). This distinction costs nothing extra since both quantities are already measured, and it converts a null into a mechanistic observation.

What is deliberately *not* added: CCA-based steering, learned steering vectors, and anything requiring a training loop. All of them break the no-training constraint or the T4 budget, and the positive control makes them unnecessary.

### Experiment 3: The gap map (the core figure)
What you do: plot every concept-model pair as a point, readability on one axis, controllability on the other.
What you measure: the shape of the cloud. Concepts in the high-readability low-controllability corner are the danger zone. Compute the correlation between the two axes.
If it does not work:
- Everything correlates tightly (readability predicts controllability): that is still a publishable result, "detection largely does predict control, with these specific exceptions," and the exceptions become the focus. Reframe the paper around the residuals.
- No structure at all, pure scatter: check for a confound (are your two metrics on comparable scales across models, did you normalize steering strength). If the scatter survives careful normalization, the finding is "control is unpredictable from detection," which is a strong safety message.

### Experiment 4: Predict the gap from geometry (the punch)

The original version of this experiment listed four candidate geometric features and proposed to see which ones worked. That reads as fishing, and with roughly forty to sixty concept-model points, four features plus interactions is close to fitting noise. The structure below tests one preregistered hypothesis and labels everything else exploratory, which is both more honest and cheaper.

**Primary hypothesis, H1 (mechanism M1, read/write mismatch).** Controllability, conditioned on readability, is predicted by how much of the concept direction survives projection into the model's output-effective subspace.

Operationalization: take the unit concept direction v at the intervention layer, apply the final-layernorm scaling, and compute the norm of W_U v, normalized by the mean over random unit directions drawn in the same space. Call this the *output overlap*. It is computed from the unembedding matrix and one cached direction, so it needs no forward pass and no additional quota. Preregistered test: partial Spearman correlation between output overlap and controllability, controlling for readability, with a bootstrap CI. One test, one number, stated in advance.

Why this operationalization and not the Jacobian directly: the exact quantity in the first-order expansion is the alignment between v and the gradient of the behavior score, but estimating that under 4-bit quantization means backward passes the T4 budget does not have. Output overlap is the cheap, weights-only proxy for the same thing, and its limitations belong in the Limitations section rather than being hidden.

What you measure: whether output overlap predicts the residual of controllability after readability is accounted for. The win is that you can call a concept steerable or not from the weights, before spending a single generation on it.

**Exploratory analysis E1 (mechanism M2, redundancy).** Participation ratio of the class-difference covariance, that is, the effective dimensionality of how the concept is written. Prediction: higher effective dimensionality goes with lower controllability. Reported as exploratory with the correlation and CI, no claim of confirmation.

**Exploratory analysis E2 (mechanism M3, off-manifold).** Alignment of v with the low-variance principal components of the residual stream, tested only against the subset of concepts whose steering broke fluency before it moved behavior, since that is the population M3 is about. This is a conditional analysis on a small subset and must be labeled as such.

**Explicitly dropped:** residual-stream norm at the steering layer, which is no longer a free parameter because the controllability metric already normalizes strength in RMS units (see revision R2); and "how many distinct directions carry the concept" as a separate feature, which is what E1 measures and does not need counting twice.

If it does not work:
- H1 fails and neither exploratory analysis shows anything: report "the gap is real and is not explained by the standard first-order account of steering." Because Section 2.5 states that account in advance, this is a substantive negative result about the field's working model rather than an admission that the search came up empty. Then try the one non-obvious predictor kept in reserve: whether the concept direction lies in the subspace that instruction-tuning reshaped (compare base vs instruct activations for the Qwen pair, which is free since both are already downloaded), because a concept the model was tuned to suppress may be readable but deliberately hard to move.
- H1 holds on some model families and not others: report it as family-dependent, which connects directly to the motivating paper's cross-model non-transfer result and strengthens the citation link. Do not average it away.
- H1 holds but only weakly: report the effect size honestly with its CI. A modest but preregistered effect is worth more to reviewers than a large one found by search.

### Experiment 5 (demoted): the safety concepts are points on the map, not a separate study

This was originally a standalone experiment framed as the citation hook. It is now a reported subsection of Experiment 3 plus an appendix, for two reasons. First, it was never new machinery: deception, refusal and sycophancy are already in the concept set, so "locating them on the gap map" is reading three points off a figure that Experiment 3 produces anyway. Calling that an experiment inflates the plan without adding evidence. Second, and more importantly, a safety framing bolted onto a characterization paper weakens it. The main story is readability versus controllability as a general property of representations. If the paper's spine becomes a safety claim, reviewers evaluate it as a safety paper, and as a safety paper the evidence is thin: a handful of concepts on models under 10B.

What stays in the main text: one paragraph in the Experiment 3 results noting where the safety-relevant concepts land, and one sentence in the Discussion drawing the implication, which is that a strong probe is not evidence of a control handle and interpretability-based safety arguments should not treat it as one. That is the honest size of the claim and it is enough.

What moves to the appendix: the per-concept breakdown, the judge agreement numbers for the safety concepts, and the immovability gauntlet traces.

The time this frees goes to dataset construction (Section 4), which is where the actual risk to the paper lives.

### Experiment 6: Scale consistency (free robustness)
What you do: repeat Experiments 1 to 3 across Qwen sizes (1.5B, 3B, 7B).
What you measure: does the gap grow, shrink, or move with scale.
If it does not work: inconsistency across scale is itself a finding (the gap is scale-dependent). Report the trend rather than forcing a single conclusion.

---

## 6. The three figures that carry the paper
1. The gap map (Experiment 3): readability vs controllability scatter, danger zone highlighted. This is the thesis in one image.
2. The dissociation existence proof (Experiment 2): two dose-response curves on the same axes, the sentiment positive control rising cleanly and a danger-zone concept staying flat, both under identical steering machinery and the same judges. Putting the control in the same panel is what makes this figure an argument rather than an anecdote, and it answers the "your steering was bad" objection visually before anyone raises it. The screenshot.
3. The geometry predictor (Experiment 4): output overlap against the controllability residual, one preregistered feature, with the CI shown. Showing you can call controllability from the weights in advance. The punch.

If H1 fails, figure 3 becomes the base-vs-instruct reshaping analysis instead, which still explains part of the gap.

---

## 7. What could sink the whole paper, and the safety net
The one existential risk is that readability and controllability turn out to be the same thing (tight correlation, no gap). Two reasons that is survivable. First, the motivating paper already found a dissociation, so at least one exists to anchor the existence proof. Second, even a tight correlation is a publishable characterization ("detection predicts control across concepts, quantified for the first time"), it just changes the title's spin from "they come apart" to "they mostly agree, here are the exceptions." You cannot end up with nothing.

## 8. What you release (citation engine)
- The concept contrastive datasets and probes.
- The steering harness with the robustness variants (multi-layer, ablation, clamping).
- The gap-map data (every concept-model readability and controllability number).
- The geometry-feature extraction code.
Artifacts plus two fresh motivating papers is the realistic path to citations for a solo paper.

---

## 9. Timeline (3-4 months, one person, free T4)

Weeks 1-3: literature check (Section 11), then build contrastive datasets for 10-15 concepts and the probing pipeline on one model. This is a week longer than the original plan because dataset quality is the largest risk in the project and the extra week is CPU-only, so it costs no GPU quota. Gate, both parts required: every concept passes the surface-shortcut audit in Section 4, and probes train and separate for the easy concepts. Concepts that fail the audit are rebuilt or dropped here, not later.

Weeks 4-5: Experiment 1 across all models and concepts, plus the naturalistic held-out check for the three or four headline concepts. Gate: you have the full readability table and the headline concepts survive the template-to-natural-text transfer.

Weeks 6-8: Experiment 2, the steering axis, including the robustness variants for any zero-controllability concept and the sentiment positive control on every model. This is the most fiddly part, so it gets the most time. Gate: the positive control steers on every model, and you have the full controllability table with confirmed danger-zone concepts.

Week 9: Experiment 3, the gap map, including the one-paragraph read of where the safety concepts land. Gate: you can see (or rule out) the danger zone.

Weeks 10-11: Experiment 4. Week 10 is H1, which is weights-only and should take days rather than weeks. Week 11 is the two exploratory analyses and, if H1 failed, the base-vs-instruct reshaping fallback. Timeboxed hard: no extending into week 12 to keep searching for a predictor, because a preregistered null is a result and a found-by-search effect is not.

Also week 11: Experiment 6 (scale), which is a re-run of existing code over the Qwen sizes and can share the week.

Weeks 12-14: write, release artifacts, make the three figures clean, record a short demo of the dissociation (probe fires, steering does nothing). Pre-register both outcome framings, Limitations names the scale ceiling honestly.

Buffer is built in because free-T4 quota interruptions will cost days. Plan for them.

---

## 10. Why this is the right final choice
It uses exactly your existing toolkit (linear probes plus difference-of-means steering), so with limited intuition your effort goes into execution, not learning a new area. It needs no training, gives multi-scale robustness for free, cannot fail as a result, has a real demo and a safety hook for citations, and builds on two papers that are weeks old so no lab is ahead of you. The punch is in the framing and the guaranteed dissociation example, not in a lucky result, which is what makes it safe punch rather than a gamble.

---

## 11. Literature check to run first (day one)
Search these and read anything from the last six months before committing:
- "detection steering gap language models"
- "probing versus steering controllability concept directions"
- "readability controllability activation steering"
- "when does steering fail linear probe concept"
- "geometry of steering effectiveness residual stream"
- "concept direction detectable not steerable"
If a paper has already done the systematic detection-vs-control map across many concepts with a geometric predictor, pivot to the narrower "predict steerability from geometry" slice, which would still be open. If not, proceed as written.

---

## 12. Design revisions (2026-07-19 review pass)

Changes made to this document and the accompanying codebase, with reasons. Each is an improvement a reviewer or a failed run would otherwise have forced later.

**R1. Related work hardened (Section 2).** Added the standard steering baselines (ActAdd, CAA, ITI, RepE, Arditi et al., linear representation hypothesis). The original draft only cited the two motivating papers; a submission without the steering canon gets desk-rejected on positioning.

**R2. Controllability metric made precise (Section 3).** The steering coefficient is now defined in units of the layer's residual-stream RMS norm, and the headline controllability number is the area under the dose-response curve up to the fluency ceiling rather than a single-point effect size. Single-point effects are noisy and the "strength just below breakage" point was underspecified; dose-response AUC is stable and pre-registerable.

**R3. Fluency ceiling pre-registered (Section 3).** Breakage = steered-output perplexity above 2x unsteered baseline under the unsteered model, or degenerate repetition. Without a pre-registered threshold, the ceiling becomes a researcher degree of freedom that a reviewer can attack.

**R4. Probe selectivity controls (Section 3).** Readability is reported with a shuffled-label control probe (Hewitt & Liang-style selectivity), and probes are evaluated on held-out *template families*, not just held-out examples. Template-generated minimal pairs risk the probe reading lexical cues rather than the concept; the split-by-template-family design is the mitigation, and the codebase enforces it.

**R5. Data-leakage rule.** Steering evaluation prompts are disjoint from the pairs used to derive the direction. Enforced structurally in the codebase (each concept ships separate `pairs` and `eval_prompts`).

**R6. Engineering decision recorded.** Raw HuggingFace forward hooks on the decoder layers, not TransformerLens: TransformerLens rewrites weights and does not play well with 4-bit quantization on a T4, and this study only needs residual-stream capture and addition, which hooks do natively. Activations are cached to disk as float16 so Colab session death costs a re-mount, not a re-run.

**R7. Toxicity handled by proxy.** The generated contrastive set uses rude-vs-courteous register as the buildable proxy; swap in Civil Comments examples for the paper version. Keeps the repo shippable without generating actually toxic text.

**R8. Statistics.** Both axes get bootstrap confidence intervals; the gap-map correlation is Spearman with a bootstrap CI; probes are averaged over seeds. The gap claim lives or dies on whether the danger-zone points are outside the noise, so the CIs are not optional.

**R10. Danger-zone thresholds must be absolute, not normalized.** This one was caught by a test during implementation and it would have been a serious error in the paper. Experiment 3 normalizes both axes within model to remove the cross-model scale confound (Section 5 names that confound). But min-max normalization *guarantees* that some concept sits at normalized controllability 0.0 and some at normalized readability 1.0. Applying the danger-zone thresholds to normalized values therefore manufactures a danger-zone occupant out of pure scatter, and the paper's headline claim would be an artifact of the rescaling rather than a fact about any model. The fix: normalize for the correlation and the figure, but decide danger-zone membership on **raw** values (AUROC >= 0.9 with dose-response AUC <= 0.05), which are absolute statements about a concept. Anyone reproducing this work should check the same thing — the failure mode is silent and the resulting figure looks completely normal.

**R11. Immovability requires the full gauntlet before it is claimed.** Implemented as `pipeline.confirm_immovable`: single-layer addition, multi-layer addition, clamping, directional ablation, and probe-weight direction in place of difference-of-means. It runs only when the default sweep already looks flat, so the cost lands on the few concepts that matter. A concept is called immovable only if every variant fails to move behavior.

**R9. Codebase started.** `lbi/` package implements Experiments 1-4: contrastive dataset generation (10 concepts), activation extraction with caching, per-layer probing with controls, the steering harness with all four robustness variants from Experiment 2, behavior + fluency scoring, geometry features, and the gap-map aggregation. 71 tests run on CPU with no model downloads, and `scripts/demo_synthetic.py` exercises the whole pipeline against planted ground truth (two concepts planted as readable-but-immovable, checked to be exactly the two recovered) so wiring bugs surface before any T4 quota is spent. See README.md for the run order.

---

### Second review pass (external reader, 2026-08)

An outside reader rated the design well on novelty and technical depth but flagged four things: the theory was thin, Experiment 4 looked like feature fishing, the safety experiment weakened the spine, and datasets were the real risk while getting the least budget. All four are addressed below. None of the changes increase compute; two of them reduce it.

**R12. Theory section added (new Section 2.5).** The design now states, before any experiment, why readability and controllability should be expected to dissociate: a probe interrogates the encoder, steering interrogates the decoder, and the first-order expansion of the behavior change makes explicit that controllability depends on the alignment between the concept direction and the downstream sensitivity, which nothing ties to class separation. Three mechanisms (read/write mismatch, redundant encoding, off-manifold steering) are named, and each maps to a specific measurable prediction. Cost: writing time only. This is what converts Experiment 4 from a search into a test, and it is what makes a null result publishable rather than embarrassing.

**R13. Experiment 4 reduced to one preregistered hypothesis plus two exploratory analyses.** Output overlap with the unembedding-effective subspace is the single primary feature, tested by partial Spearman controlling for readability. Participation ratio and low-variance-PC alignment are demoted to labeled exploratory analyses, the latter conditioned on the fluency-limited subset only. Residual-stream norm is dropped outright, since R2's RMS normalization already absorbs it, and multi-directionality is dropped as a separate feature because participation ratio measures the same thing. With roughly forty to sixty concept-model points, four features was close to fitting noise. This change *reduces* compute and improves the statistics.

**R14. Experiment 5 demoted from a standalone experiment to a results subsection plus appendix.** It was never new machinery, only a read of three points off the Experiment 3 figure, and a safety framing invites reviewers to judge the paper as a safety paper, where evidence from sub-10B models is thin. The main claim survives at its honest size: a strong probe is not evidence of a control handle. The freed week goes to datasets.

**R15. Dataset rigor promoted to a first-class gate with a preregistered audit (Section 4).** A TF-IDF surface baseline is fit on the raw text of every contrastive set under the same held-out-template-family split, and a concept only proceeds if the surface baseline stays below 0.65 AUROC or falls at least 0.15 below the activation probe. Plus a naturalistic held-out set for the headline concepts. This is the highest-leverage change in this pass: lexical shortcuts are a silent failure that inflates readability, manufactures danger-zone occupants, and cannot be detected after the fact from the numbers alone. The audit runs on CPU in seconds.

**R16. Steering-method objection closed at zero extra compute (Experiment 2).** Three moves. State explicitly that difference-of-means with a swept coefficient *is* CAA and that ActAdd is its one-pair form, so the canon is covered rather than ignored. Add the RepE reading vector (first PC of paired activation differences) as a third direction-derivation method, computed from activations already cached for Experiment 1, so it costs a PCA and not a forward pass. And run the sentiment positive control on every model through the same machinery, preregistered such that a model failing the control has its numbers withheld. The figure showing the control rising while a danger-zone concept stays flat is a stronger rebuttal than any additional steering method would be. CCA steering and learned steering vectors are explicitly declined: they break the no-training constraint and the positive control makes them unnecessary.

**R17. Timeline reallocated, total length unchanged.** Dataset construction goes from two weeks to three, Experiment 1 shifts by a week, Experiment 4 is compressed because H1 is weights-only, and the week previously held by Experiment 5 is absorbed. Still 12-14 weeks, still the same GPU budget, with the extra time spent on the CPU-only work that most determines whether the numbers mean anything.

**Still open before the paper run:** no real model has been executed yet (no GPU on the development machine), so every number so far is synthetic and only validates the plumbing. The behavior judge is still the development lexicon scorer, which must be replaced by a validated classifier plus an LLM judge with reported agreement (Experiment 2's requirement) before any behavior number is quoted. Experiment 6 is a data-collection pass over the existing code rather than new machinery, and Experiment 5 is now a read of the Experiment 3 output, so neither needs further implementation.

The second review pass implies four code changes, all small: add the TF-IDF surface-shortcut audit to `lbi/concepts.py` as a gate that runs before extraction (R15); add the RepE reading-vector direction to `lbi/steering.py` alongside difference-of-means and probe weights (R16); mark sentiment as the positive control in `lbi/pipeline.py` so a failed control withholds rather than reports (R16); and trim `lbi/geometry.py` to the one primary feature plus two clearly separated exploratory ones, with the partial-Spearman-controlling-for-readability test as the preregistered analysis (R13). The synthetic demo should be extended to plant a surface-shortcut concept and confirm the audit catches it, in the same spirit as the ground-truth check that already exists.

---

### Third pass: code audited against the preregistration (2026-08-24)

The first two passes revised the design. This one checked the codebase against
it, concept set by concept set and P-number by P-number. The headline finding is
uncomfortable and worth stating plainly, because it is the failure mode a design
review cannot catch: **four preregistered commitments were written in PLAN.md and
not implemented.** The plan was sound. The code disagreed with it, silently, and
would have produced numbers that looked exactly like the preregistered analysis.

**R18. Four preregistered points implemented rather than only documented.**

- *P4, best-over-band controllability.* `run_steering_best_over_band` existed and
  nothing called it; `run_model` used single-layer steering. The answer to
  objection 5 ("you steered at the wrong layer") was a paragraph, not a number.
  Now wired, at four generation sweeps per concept -- already budgeted in Part 9's
  ledger. `--single-layer` remains for smoke runs and warns that its numbers are
  not reportable.
- *P6, CI exclusion for danger-zone membership.* Membership was decided on point
  estimates; the CI fields on `GapPoint` were dead. A concept at controllability
  0.04 with an interval reaching 0.30 qualified. Now both bounds must clear their
  thresholds, and a NaN bound never qualifies.
- *P7, cluster bootstrap over concepts.* The gap map's headline correlation CI
  resampled concept-model points independently. On eight concepts replicated
  across five models the i.i.d. interval is [0.34, 0.82] and the clustered one is
  [-0.12, 0.93]: the difference between claiming a result and not. This is
  objection 10, answered in prose and nowhere else.
- *P9, positive-control withholding.* `run_model` printed a warning that said the
  aggregate script enforced the withholding; the aggregate script never looked.
  This matters more than it reads: a model where steering is broken scores low
  controllability on *every* concept, so a broken harness puts the whole model in
  the danger zone and is indistinguishable from the paper's headline finding. The
  control now runs first, writes a record, and `aggregate` drops failing models.

Also in this pass: `fit_gap_predictor` was fabricating the primary test. Its
`PrimaryTestReport.partial_spearman` was the ridge model's leave-one-out
prediction correlation rather than the partial Spearman of output overlap against
controllability controlling for readability; the CI was hardcoded `(nan, nan)` so
the cluster bootstrap never ran; the verdict was the R-squared verdict; and the
exploratory list was empty, so P8's BH correction never ran. Nothing called the
correct `primary_test` and `exploratory_analysis`. Called now, and without the raw
axes the report says `primary test NOT RUN` instead of substituting the ridge fit.

**R19. The surface-shortcut audit was passing concepts it should have failed.**
R15 specified the audit; the implementation held out one RNG-seeded family and
scored one-sided AUROC. Made leave-one-family-out (every family takes a turn,
mean reported with the worst fold) and two-sided, `max(auroc, 1 - auroc)`: a
classifier that ranks a held-out family perfectly *backwards* has still found a
lexical contrast, and scored one-sided, `rudeness` and `factuality` read 0.00 and
0.01 and were recorded as the cleanest concepts in the set.

Under the corrected audit, **nine of ten concepts failed**. All were rebuilt as
strict minimal pairs on a shared carrier, with two invariants enforced at build
time: marker vocabulary must not appear anywhere else in the concept (other
families' markers, other families' carriers, or the topic strings), and a
family's two markers must match in token count. The length rule is not cosmetic --
TF-IDF vectors are L2-normalised, so a two-token difference separated families at
AUROC 0.94 with no marker word transferring at all. A capitalisation confound
went with it: casual-register templates were written lower case and formal ones
capitalised, aligning perfectly with the label on `formality`, invisible to the
audit because TfidfVectorizer lower-cases before counting.

`topic_science` had a genuine data bug: families were `block{i // 4}` over a list
repeated twice, so block2 and block3 were byte-identical to block0 and block1 and
every held-out family was already in training.

**R20. Two concepts declared surface-confounded rather than dressed up.**
`verbosity` is length, and length is a surface property by definition;
`topic_science` is domain membership, carried by content vocabulary. Neither can
pass a lexical audit, so both set `surface_confounded=True` and both still
**fail** -- the flag does not convert a failure into a pass, and a test pins that.
It records that the failure was predicted rather than discovered by a reviewer.

An honest caveat belongs in Methods: every non-confounded concept now scores
exactly 0.500, the floor. That is the audit passing *by construction*. It is a
check that the construction succeeded, not independent evidence the probe reads
semantics; what carries that argument is the probe's own held-out-family AUROC,
where the probe faces marker words it has never seen.

**R21. "Immovable" separated from "unsteerable under tested interventions."**
Prompted by an external reader's objection that a concept unresponsive to
difference-of-means steering has not been shown to be uncontrollable. R11 already
required the gauntlet, but the gauntlet's verdict never reached the gap map:
`build_gap_map` applied the two thresholds and printed "readable-but-immovable"
while `confirm_immovable`'s answer sat unread in the per-concept JSON. The result
now propagates via `GapPoint.gauntlet_passed`, and `gap_map.json` reports
`confirmed_immovable` and `unsteerable_under_tested_interventions` separately.
`None` means the gauntlet never ran, which is not the same as failing it. Even a
pass is bounded by the six interventions tested and is not a proof that none
exists; the paper should say so.

**R22. Measurement bugs that would have moved numbers.**

- The RepE reading vector mean-centred paired differences before PCA. The concept
  signal lives almost entirely in that mean, so centring deleted it: PC1 recovered
  a planted direction at cosine 0.31 instead of above 0.90. One of R16's three
  direction-derivation methods was returning noise.
- `bootstrap_curve_ci` integrated every coefficient while `dose_response_auc`
  excluded points past the fluency ceiling, so the interval and the point estimate
  were different estimands (0.025 with a CI of [0.1375, 0.1375]). The ceiling
  filter now lives inside the function.
- `capture_activations` and `perplexity` ran raw forward passes under left
  padding without position ids, so HuggingFace fell back to `arange` and every
  padded row had its RoPE phase shifted. Perplexity additionally scored the first
  real token from a pad position with no context, making shorter texts in a batch
  look less fluent -- and perplexity is what the fluency ceiling thresholds on.
- `LexiconScorer` matched with `str.count`, so "but" fired on "contribution" and
  "will" on ordinary future tense.
- The unembedding SVD was recomputed per concept: about 40 s and a 2.2 GB float32
  view at a 152k x 3584 head, ten times per model, next to a loaded 7B.

**R23. Krippendorff's alpha implemented.** Part 4's objection ledger answers
"judges are unvalidated" with "Krippendorff alpha reported", and no code computed
one. Added for interval data with NaN support, so the human spot-check enters as a
third rater. Alpha rather than correlation because two judges offset by a constant
correlate at Pearson 1.0 and agree on nothing, and the behaviour axis is read in
absolute terms against a fixed threshold.

**Still open after this pass.** No real model has been run; every number remains
synthetic and validates plumbing only. The behaviour judges are still the
development lexicon scorer -- `ClassifierScorer.model_map` ships empty by design so
no unvalidated judge can drift into the paper. The `rudeness` concept is still a
courtesy-register proxy; substituting Civil Comments will not satisfy the two
marker invariants, since natural text cannot, so it will need the same declared
exception the two surface-confounded concepts get, and that decision belongs in
Phase A rather than at write-up.
