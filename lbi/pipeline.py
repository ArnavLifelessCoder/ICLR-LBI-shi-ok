"""End-to-end run: Experiments 1-4 for one model.

Designed for an unreliable session: results are written per concept as they
finish and `run_model(resume=True)` skips any concept whose file already
exists, so a dead Kaggle or Colab runtime costs at most the concept that was
in flight. Activations are cached too, but generation is not and generation is
essentially the whole cost, so the resume check is what actually saves the
session.

Revisions applied:
  R16  RepE reading vector added to the gauntlet alongside diff-of-means and
       probe weights.
  P4   Controllability is best-over-band across the preregistered 4-layer band,
       not single-layer.
  P9   Sentiment is the positive control; if it fails to steer on a model,
       that model's controllability numbers are withheld.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from . import behavior as bh
from . import geometry as geo
from . import steering as st
from .concepts import Concept, all_concepts
from .extraction import LoadedModel, capture_cached
from .gapmap import GapPoint, build_gap_map
from .probes import (
    ProbeResult,
    default_splits,
    diff_of_means_direction,
    pair_texts,
    repe_reading_vector,
    train_probes,
)


DEFAULT_COEFFS = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]

# --- Positive control gate (P9 / R16) ---
POSITIVE_CONTROL = "sentiment"
CONTROL_FLOOR = 0.10  # minimum dose-response AUC for the control concept


@dataclass
class ConceptRun:
    probe: ProbeResult
    steering: st.SteeringResult
    features: geo.GeometryFeatures
    # Gauntlet result, or None when the concept moved under the default
    # intervention and the gauntlet was never run.
    gauntlet: dict | None = None

    @property
    def gauntlet_passed(self) -> bool | None:
        if self.gauntlet is None:
            return None
        return bool(self.gauntlet.get("immovable"))


@dataclass
class ControlStatus:
    """P9 status for one model, as written to `<model>_control.json`.

    `aggregate` reads these back and withholds a failing model's points, which
    is the step the preregistration calls for and the pipeline used only to
    print a warning about.
    """

    model: str
    controllability: float | None
    passed: bool

    @classmethod
    def from_record(cls, rec: dict) -> "ControlStatus":
        return cls(
            model=rec["model"],
            controllability=rec.get("controllability"),
            passed=bool(rec.get("passed")),
        )


def jsonable(obj):
    """Recursively convert dataclasses/numpy scalars into JSON-safe values.

    Non-finite floats become null rather than the bare `NaN`/`Infinity` tokens
    json.dump emits by default, which are not valid JSON and choke every reader
    outside Python. A missing CI has to survive the round trip to a plotting
    script or a reviewer's notebook.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "item") and getattr(obj, "shape", None) == ():
        obj = obj.item()  # 0-d numpy scalar
    if hasattr(obj, "tolist"):
        return jsonable(obj.tolist())  # numpy array
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, indent=2)


def run_probing(
    lm: LoadedModel, concept: Concept, cache_dir: str, seeds: tuple[int, ...] = (0, 1, 2)
) -> tuple[ProbeResult, np.ndarray, np.ndarray, list[str]]:
    """Experiment 1 for one concept. Returns (result, acts, labels, families)."""
    # P2: three-way split by family. The layer is chosen on validation and
    # readability is reported on test, which are different families and both
    # disjoint from training.
    val_fams, test_fams = default_splits(concept)
    texts, labels = pair_texts(concept.pairs)
    # Families repeat per pair member, so expand alongside the flattened texts.
    families = [p.family for p in concept.pairs for _ in (0, 1)]
    val_mask = np.array([f in val_fams for f in families])
    test_mask = np.array([f in test_fams for f in families])
    train_mask = ~(val_mask | test_mask)

    acts = capture_cached(lm, texts, cache_dir=cache_dir, tag=f"{concept.name}_pairs")
    result = train_probes(
        concept, acts, labels, train_mask, lm.name,
        seeds=seeds, held_out_families=sorted(test_fams),
        val_mask=val_mask,
    )
    return result, acts, labels, families


def run_steering(
    lm: LoadedModel,
    concept: Concept,
    direction: np.ndarray,
    layer: int,
    scorer: bh.Scorer,
    coeffs: list[float] | None = None,
    variant: str = "add",
    direction_source: str = "diff_of_means",
    max_new_tokens: int = 64,
) -> st.SteeringResult:
    """Experiment 2 for one concept: sweep the coefficient, find the ceiling."""
    coeffs = list(coeffs or DEFAULT_COEFFS)
    layers = (
        st.layer_band(layer, lm.n_layers) if variant == "add_all" else [layer]
    )
    prompts = concept.eval_prompts

    # Baseline: unsteered behavior and the perplexity the ceiling is relative to.
    base_out = st.generate(lm, prompts, spec=None, max_new_tokens=max_new_tokens)
    base_scores = scorer.score(base_out, concept.name)
    baseline = float(np.mean(base_scores))
    baseline_ppl = float(np.mean(bh.perplexity(lm, base_out)))

    curve: list[st.DosePoint] = []
    per_prompt: dict[float, list[float]] = {}

    for c in coeffs:
        if c == 0.0:
            outs, scores = base_out, base_scores
        else:
            spec = st.SteeringSpec(
                direction=direction,
                layers=layers,
                variant=variant,
                coeff=c,
                clamp_target=c if variant == "clamp" else None,
            )
            outs = st.generate(lm, prompts, spec=spec, max_new_tokens=max_new_tokens)
            scores = scorer.score(outs, concept.name)

        ppl = float(np.mean(bh.perplexity(lm, outs)))
        rep = float(np.mean([bh.repetition_score(o) for o in outs]))
        per_prompt[c] = scores
        curve.append(
            st.DosePoint(
                coeff=c,
                behavior=float(np.mean(scores)),
                behavior_ci=(
                    float(np.percentile(scores, 2.5)),
                    float(np.percentile(scores, 97.5)),
                ),
                perplexity=ppl,
                repetition=rep,
                broken=False,
                samples=outs[:2],
            )
        )

    # A judge that returns the same number for every text parses perfectly and
    # yields a dose-response AUC of exactly 0.000, which is indistinguishable
    # from a concept that cannot be steered. The parse-failure guard in
    # LLMJudgeScorer does not catch this: the output is readable, it is just
    # constant. Flag it here, where the whole sweep is visible.
    all_scores = [s for scores in per_prompt.values() for s in scores]
    if all_scores and len(set(all_scores)) == 1:
        print(
            f"WARNING [{concept.name}]: the judge returned {all_scores[0]} for "
            f"all {len(all_scores)} generations across every coefficient. "
            f"Controllability will be exactly 0.000 for a reason that has "
            f"nothing to do with steering. Inspect the samples in the result "
            f"file before believing this number."
        )

    max_usable, reason = st.find_ceiling(curve, baseline_ppl)
    st.mark_broken(curve, max_usable)
    controllability = st.dose_response_auc(curve, baseline)
    ci = st.bootstrap_curve_ci(per_prompt, baseline, max_usable=max_usable)

    return st.SteeringResult(
        concept=concept.name,
        model=lm.name,
        variant=variant,
        layers=layers,
        direction_source=direction_source,
        curve=curve,
        controllability=controllability,
        controllability_ci=ci,
        baseline_behavior=baseline,
        max_usable_coeff=max_usable,
        ceiling_reason=reason,
    )


def run_steering_best_over_band(
    lm: LoadedModel,
    concept: Concept,
    direction: np.ndarray,
    best_layer: int,
    scorer: bh.Scorer,
    coeffs: list[float] | None = None,
    direction_source: str = "diff_of_means",
    max_new_tokens: int = 64,
) -> st.SteeringResult:
    """P4: Controllability is the best over a preregistered 4-layer band.

    Runs single-layer steering at each layer in the band and returns the
    result with the highest dose-response AUC. The asymmetry with
    probe-layer selection (which uses one layer) is deliberate: it biases
    the study against its own headline finding.
    """
    band = st.layer_band(best_layer, lm.n_layers, width=4)
    best_result = None
    for layer in band:
        result = run_steering(
            lm, concept, direction, layer, scorer,
            coeffs=coeffs, direction_source=direction_source,
            max_new_tokens=max_new_tokens,
        )
        if best_result is None or result.controllability > best_result.controllability:
            best_result = result
    return best_result


def confirm_immovable(
    lm: LoadedModel,
    concept: Concept,
    dom_direction: np.ndarray,
    probe_direction: np.ndarray,
    repe_direction: np.ndarray,
    layer: int,
    scorer: bh.Scorer,
    threshold: float = 0.05,
) -> dict:
    """The robustness gauntlet from the design doc's Experiment 2 fallback.

    A concept is only called uncontrollable if it resists every variant and all
    three direction sources (R16: diff-of-means, probe weights, RepE reading
    vector). Returns each variant's controllability plus the verdict.
    """
    trials = [
        ("add", dom_direction, "diff_of_means"),
        ("add_all", dom_direction, "diff_of_means"),
        ("clamp", dom_direction, "diff_of_means"),
        ("ablate", dom_direction, "diff_of_means"),
        ("add", probe_direction, "probe_weights"),
        ("add", repe_direction, "repe_reading_vector"),
    ]
    results = {}
    for variant, direction, source in trials:
        r = run_steering(
            lm, concept, direction, layer, scorer,
            variant=variant, direction_source=source,
        )
        results[f"{variant}:{source}"] = r.controllability

    best = max(results.values())
    passed = bool(best < threshold)
    return {
        "per_variant": results,
        "best_controllability": best,
        "immovable": passed,
        "n_interventions_tested": len(trials),
        "verdict": (
            # "Immovable" is a claim about the concept and is only earned here,
            # after every variant and all three direction-derivation methods
            # have failed. Anything short of this gauntlet gets the weaker
            # label "unsteerable under tested interventions", which is a claim
            # about what was tried rather than about what is possible. Even
            # this verdict is bounded by the six interventions listed above --
            # it is not a proof that no intervention exists.
            f"immovable: resisted all {len(trials)} tested interventions "
            f"(4 variants x diff-of-means, plus probe weights and the RepE "
            f"reading vector); best controllability {best:.3f} < {threshold}"
            if passed
            else f"movable via {max(results, key=results.get)} "
                 f"(controllability {best:.3f})"
        ),
    }


def run_model(
    lm: LoadedModel,
    scorer: bh.Scorer,
    out_dir: str,
    cache_dir: str,
    concepts: list[Concept] | None = None,
    immovable_threshold: float = 0.05,
    best_over_band: bool = True,
    resume: bool = True,
) -> list[ConceptRun]:
    """Experiments 1, 2 and 4 for every concept on one model.

    Returns the list of ConceptRun results. The positive control runs first and
    warns immediately if it fails (P9); `aggregate` is what withholds the
    model's numbers, using the control record written to `out_dir`.

    `best_over_band=False` drops back to single-layer steering. It is four
    times cheaper and useful for a smoke run, but it is not the preregistered
    protocol and its numbers should not be reported.

    `resume=True` skips any concept whose result file already exists, which is
    what makes a killed session cost one concept instead of the sweep. Skipped
    concepts are not in the returned list -- their numbers are on disk, and
    `run_experiment.py --aggregate-only` reads every record from there, so the
    analysis is unaffected. Only the in-session summary is partial.
    """
    concepts = concepts or all_concepts()
    # P9: the positive control goes first. PLAN.md Phase C is explicit that a
    # broken harness discovered late costs the paper, and there is no reason to
    # spend a T4 session on nine concepts before finding out that steering does
    # not move the one concept everybody agrees is steerable.
    concepts = sorted(concepts, key=lambda c: c.name != POSITIVE_CONTROL)

    lm_head = None
    try:
        lm_head = lm.model.get_output_embeddings().weight.detach().float().cpu().numpy()
    except AttributeError:
        pass  # tied or absent head: output_overlap reports NaN

    runs: list[ConceptRun] = []
    skipped: list[str] = []
    for concept in concepts:
        result_path = os.path.join(
            out_dir, f"{_slug(lm.name)}_{concept.name}.json"
        )
        if resume and os.path.exists(result_path):
            # The module docstring has always promised this. It did not happen:
            # every concept was recomputed from scratch, so a session that died
            # on concept nine of ten redid all nine, and running the positive
            # control before the sweep meant paying for sentiment twice.
            # Activations are cached but generation is not, and generation is
            # essentially the whole cost.
            skipped.append(concept.name)
            print(f"  {concept.name}: already in {os.path.basename(result_path)}, "
                  f"skipping (pass resume=False, or delete the file, to redo)")
            continue

        probe, acts, labels, families = run_probing(lm, concept, cache_dir)
        layer = probe.best_layer
        acts_layer = acts[layer]
        dom = diff_of_means_direction(acts_layer, labels)
        repe = repe_reading_vector(acts_layer, labels)

        # P4: controllability is the best over the preregistered layer band,
        # not the single probe-selected layer. The asymmetry with probe-layer
        # selection is deliberate and biases the study against its own headline
        # finding -- which only holds if the band actually gets swept.
        steer = (
            run_steering_best_over_band(lm, concept, dom, layer, scorer)
            if best_over_band
            else run_steering(lm, concept, dom, layer, scorer)
        )

        # Only pay for the gauntlet when the concept looks immovable.
        gauntlet = None
        if steer.controllability < immovable_threshold:
            gauntlet = confirm_immovable(
                lm, concept, dom, probe.probe_direction(), repe, layer, scorer,
                threshold=immovable_threshold,
            )

        features = geo.compute_features(
            concept, lm.name, layer, acts_layer, labels, families,
            probe_direction=probe.probe_direction(), lm_head_weight=lm_head,
        )

        if concept.name == POSITIVE_CONTROL and steer.controllability < CONTROL_FLOOR:
            print(
                f"WARNING [P9]: positive control '{POSITIVE_CONTROL}' scored "
                f"{steer.controllability:.3f} < floor {CONTROL_FLOOR} on "
                f"{lm.name}. Steering is not working on this model; every "
                f"controllability number from it will be withheld at aggregate "
                f"time. Stop and debug rather than spending the session."
            )

        runs.append(
            ConceptRun(probe=probe, steering=steer, features=features,
                       gauntlet=gauntlet)
        )
        _write_json(
            os.path.join(out_dir, f"{_slug(lm.name)}_{concept.name}.json"),
            {
                "probe": probe.summary(),
                # Layer-by-layer AUROCs, so the layer choice is auditable.
                "probe_layers": probe.per_layer_curve(),
                "steering": steer.summary(),
                # `samples` and `repetition` are persisted because without them
                # a failed run cannot be diagnosed from its own output. The
                # first real run returned controllability exactly 0.000 and the
                # result file could not say whether the model had produced
                # sensible text that steering failed to move, or garbage, or
                # whether the judge had returned a constant -- the generations
                # existed in memory and were dropped on write.
                "curve": [
                    {"coeff": p.coeff, "behavior": p.behavior,
                     "behavior_ci_low": p.behavior_ci[0],
                     "behavior_ci_high": p.behavior_ci[1],
                     "perplexity": p.perplexity, "repetition": p.repetition,
                     "broken": p.broken, "samples": p.samples}
                    for p in steer.curve
                ],
                "geometry": features.to_dict(),
                "gauntlet": gauntlet,
                # Which judge produced the behaviour numbers. Recorded per
                # concept so a dataset accidentally scored by two different
                # judges is detectable afterwards instead of assumed away --
                # raw controllability is what danger-zone membership is
                # decided on, and it is only comparable across models when the
                # judge is held fixed.
                "judge_model": getattr(scorer, "judge_model_name", None),
                "judge_is_self": getattr(scorer, "judge_is_self", None),
            },
        )

    # --- Positive control gate (P9) ---
    # Only written when the control concept was actually part of this run. A
    # subset run like `--concepts refusal honesty` has not failed the control,
    # it has not measured it, and recording that as a failure would make
    # `aggregate` withhold the whole model on the strength of a missing number.
    control_runs = [r for r in runs if r.probe.concept == POSITIVE_CONTROL]
    if control_runs:
        ctrl_value = control_runs[0].steering.controllability
        _write_json(
            os.path.join(out_dir, f"{_slug(lm.name)}_control.json"),
            {
                "model": lm.name,
                "control_concept": POSITIVE_CONTROL,
                "controllability": ctrl_value,
                "floor": CONTROL_FLOOR,
                "passed": bool(ctrl_value >= CONTROL_FLOOR),
                "judge_model": getattr(scorer, "judge_model_name", None),
                "judge_is_self": getattr(scorer, "judge_is_self", None),
                "note": (
                    "P9: if this did not pass, the model's controllability "
                    "numbers are withheld, not explained away."
                ),
            },
        )
    elif os.path.exists(os.path.join(out_dir, f"{_slug(lm.name)}_control.json")):
        # The control ran in an earlier session (or in stage 1) and was skipped
        # here by resume. Leave that record alone rather than overwriting a real
        # measurement with an absence.
        print(f"NOTE [P9]: control record for {lm.name} already on disk, kept.")
    else:
        print(
            f"NOTE [P9]: '{POSITIVE_CONTROL}' was not in this run, so no "
            f"control record was written for {lm.name}. Its points will not be "
            f"withheld at aggregate time -- run the control before reporting."
        )

    if skipped:
        print(
            f"\nresumed: skipped {len(skipped)} concept(s) already on disk "
            f"({', '.join(skipped)}). Their numbers are in {out_dir} and "
            f"--aggregate-only reads them; only this summary is partial."
        )
    return runs


def to_gap_points(runs: list[ConceptRun], concepts: list[Concept] | None = None) -> list[GapPoint]:
    by_name = {c.name: c for c in (concepts or all_concepts())}
    return [
        GapPoint(
            concept=r.probe.concept,
            model=r.probe.model,
            readability=r.probe.readability,
            controllability=r.steering.controllability,
            readability_ci=r.probe.auroc_ci,
            controllability_ci=r.steering.controllability_ci,
            safety_relevant=by_name[r.probe.concept].safety_relevant,
            best_layer=r.probe.best_layer,
            selectivity=r.probe.selectivity,
            gauntlet_passed=r.gauntlet_passed,
        )
        for r in runs
    ]


def build_and_save_map(runs: list[ConceptRun], out_dir: str):
    """Experiment 3 from accumulated runs (pass runs from all models)."""
    from .gapmap import plot_gap_map

    gm = build_gap_map(to_gap_points(runs))
    gm.save(os.path.join(out_dir, "gap_map.json"))
    plot_gap_map(gm, os.path.join(out_dir, "fig1_gap_map.png"))
    return gm


def fit_predictor(runs: list[ConceptRun], out_dir: str):
    """Experiment 4 from accumulated runs."""
    points = to_gap_points(runs)
    from .gapmap import normalize_within_model

    normed = normalize_within_model(points)
    gaps = [p.gap for p in normed]
    report = geo.fit_gap_predictor(
        [r.features for r in runs],
        gaps,
        controllabilities=[p.controllability for p in normed],
        readabilities=[p.readability for p in normed],
    )
    _write_json(os.path.join(out_dir, "geometry_predictor.json"), report)
    return report


def _slug(name: str) -> str:
    return name.split("/")[-1].replace(".", "-")
