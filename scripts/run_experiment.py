"""Run Experiments 1-4 on real models.

Intended for a free Colab/Kaggle T4. One model per invocation, results written
per concept as they finish, so a killed session loses at most one concept.

    python scripts/run_experiment.py --model Qwen/Qwen2.5-7B-Instruct
    python scripts/run_experiment.py --model Qwen/Qwen2.5-1.5B-Instruct --no-4bit

Then, once every model has run:

    python scripts/run_experiment.py --aggregate-only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lbi.behavior import LexiconScorer
from lbi.concepts import all_concepts, get_concept
from lbi.extraction import load_model
from lbi.gapmap import GapPoint, build_gap_map, plot_gap_map
from lbi.geometry import GeometryFeatures, fit_gap_predictor
from lbi.pipeline import ControlStatus, build_and_save_map, jsonable, run_model


def aggregate(out_dir: str) -> int:
    """Rebuild the gap map and predictor from every per-concept result file."""
    by_name = {c.name: c for c in all_concepts()}
    points: list[GapPoint] = []
    features: list[GeometryFeatures] = []

    paths = sorted(glob.glob(os.path.join(out_dir, "*.json")))
    control_paths = [p for p in paths if os.path.basename(p).endswith("_control.json")]
    paths = [p for p in paths if p not in control_paths
             and os.path.basename(p) not in
             ("gap_map.json", "geometry_predictor.json")]
    if not paths:
        print(f"no per-concept results in {out_dir}; run a model first")
        return 1

    # P9: a model whose sentiment positive control failed has its
    # controllability numbers withheld, not explained away. Without this the
    # gap map silently absorbs points from a model on which steering demonstrably
    # does not work -- and those points land in the danger zone by construction,
    # because a broken harness looks exactly like an immovable concept.
    withheld: dict[str, float | None] = {}
    for path in control_paths:
        with open(path, encoding="utf-8") as f:
            status = ControlStatus.from_record(json.load(f))
        if not status.passed:
            withheld[status.model] = status.controllability

    n_withheld = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        probe, steer, geom = rec["probe"], rec["steering"], rec["geometry"]
        if probe["model"] in withheld:
            n_withheld += 1
            continue
        points.append(
            GapPoint(
                concept=probe["concept"],
                model=probe["model"],
                readability=probe["readability"],
                controllability=steer["controllability"],
                readability_ci=(probe["auroc_ci_low"], probe["auroc_ci_high"]),
                controllability_ci=(
                    steer["controllability_ci_low"], steer["controllability_ci_high"]
                ),
                safety_relevant=by_name[probe["concept"]].safety_relevant,
                best_layer=probe["best_layer"],
                selectivity=probe["selectivity"],
                # None when the gauntlet never ran, which is not the same as
                # failing it. Only points that survived it may be called
                # immovable rather than merely unsteerable under what was tried.
                gauntlet_passed=(
                    bool(rec["gauntlet"].get("immovable"))
                    if rec.get("gauntlet")
                    else None
                ),
            )
        )
        features.append(GeometryFeatures(**geom))

    for model, value in sorted(withheld.items()):
        shown = "n/a" if value is None else f"{value:.3f}"
        print(f"WITHHELD [P9]: {model} -- positive control scored {shown}, "
              f"below the floor; its points are excluded from the gap map.")
    if n_withheld:
        print(f"  {n_withheld} concept-model result(s) withheld in total.")

    if len(points) < 3:
        print(f"only {len(points)} usable point(s) after the P9 gate; "
              "need at least 3 for a gap map")
        return 1

    gm = build_gap_map(points)
    gm.save(os.path.join(out_dir, "gap_map.json"))
    plot_gap_map(gm, os.path.join(out_dir, "fig1_gap_map.png"))

    print(f"gap map over {len(points)} concept-model points")
    print(f"  Spearman rho = {gm.spearman:.3f} "
          f"[{gm.spearman_ci[0]:.3f}, {gm.spearman_ci[1]:.3f}]")
    print(f"  {gm.interpretation}")

    if len(points) >= 5:
        # Pass the raw axes, not just the gap: without them the preregistered
        # primary test and its cluster bootstrap cannot be computed at all.
        report = fit_gap_predictor(
            features,
            [p.gap for p in gm.points],
            controllabilities=[p.controllability for p in gm.points],
            readabilities=[p.readability for p in gm.points],
        )
        with open(os.path.join(out_dir, "geometry_predictor.json"), "w",
                  encoding="utf-8") as f:
            json.dump(jsonable(report), f, indent=2)
        prim = report.primary
        print(f"  primary test (H1, {prim.feature}): partial rho = "
              f"{prim.partial_spearman:.3f} "
              f"[{prim.partial_spearman_ci[0]:.3f}, {prim.partial_spearman_ci[1]:.3f}] "
              f"over {prim.n_concepts} concepts")
        print(f"    {prim.verdict}")
        for ex in report.exploratory:
            print(f"  {ex.label} ({ex.feature}): rho = {ex.partial_spearman:.3f}, "
                  f"p_BH = {ex.p_value_bh:.3f}")
        print(f"  geometry predictor: LOO R^2 = {report.r2_loo:.3f} -- {report.verdict}")
    else:
        print("  (need >= 5 points before fitting the geometry predictor)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HuggingFace model id")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--cache-dir", default="cache/activations")
    ap.add_argument("--concepts", nargs="*", help="subset of concept names")
    ap.add_argument("--no-4bit", action="store_true", help="load in fp16 instead")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument(
        "--single-layer",
        action="store_true",
        help="steer only at the probe layer instead of sweeping the P4 band. "
             "Four times cheaper, useful for a smoke run, NOT the preregistered "
             "protocol -- do not report numbers produced this way.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.aggregate_only:
        return aggregate(args.out_dir)
    if not args.model:
        ap.error("--model is required unless --aggregate-only is passed")

    concepts = (
        [get_concept(n) for n in args.concepts] if args.concepts else all_concepts()
    )

    print(f"loading {args.model} ({'fp16' if args.no_4bit else '4-bit'})...")
    lm = load_model(args.model, load_in_4bit=not args.no_4bit)
    print(f"  {lm.n_layers} layers, d_model={lm.d_model}, device={lm.device}")

    # LexiconScorer is the development judge. Swap in ClassifierScorer or
    # LLMJudgeScorer (and report agreement) before the numbers go in the paper.
    scorer = LexiconScorer()

    if args.single_layer:
        print("  NOTE: --single-layer is set; controllability is NOT "
              "best-over-band and these numbers are not reportable.")

    runs = run_model(
        lm, scorer, out_dir=args.out_dir, cache_dir=args.cache_dir,
        concepts=concepts, best_over_band=not args.single_layer,
    )
    for r in runs:
        print(
            f"  {r.probe.concept:<14} readability={r.probe.readability:.3f} "
            f"(layer {r.probe.best_layer}) "
            f"controllability={r.steering.controllability:.3f} "
            f"ceiling={r.steering.max_usable_coeff:g} ({r.steering.ceiling_reason})"
        )

    if len(runs) >= 3:
        gm = build_and_save_map(runs, args.out_dir)
        print(f"\n{gm.interpretation}")
    print(f"\nresults in {args.out_dir}; run --aggregate-only once all models are done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
