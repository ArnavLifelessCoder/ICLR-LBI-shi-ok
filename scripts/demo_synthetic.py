"""End-to-end dry run on synthetic activations -- no GPU, no model download.

Exercises Experiments 1, 3 and 4 against planted ground truth: some concepts are
readable *and* controllable, one is readable but immovable. If the pipeline is
wired correctly, the planted immovable concept lands in the danger zone and the
geometry predictor recovers the planted signal.

Also exercises the surface-shortcut audit (R15): a planted shortcut concept
must fail the audit, and the real concepts must pass.

Run this before burning T4 quota -- it catches wiring bugs in seconds.

    python scripts/demo_synthetic.py --out-dir results/demo
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lbi.concepts import (
    Concept,
    Pair,
    all_concepts,
    audit_all_concepts,
    surface_shortcut_audit,
)
from lbi.gapmap import GapPoint, build_gap_map, plot_dose_response, plot_gap_map
from lbi.geometry import GeometryFeatures, fit_gap_predictor
from lbi.probes import default_held_out, pair_texts, train_probes

# Concepts planted as readable-but-immovable. Everything else gets
# controllability proportional to its readability plus noise.
IMMOVABLE = {"refusal", "honesty"}
N_LAYERS = 8
D_MODEL = 64


def synthetic_acts(labels: np.ndarray, strength: float, seed: int):
    """Activations with the concept linearly present at one planted layer."""
    rng = np.random.default_rng(seed)
    acts = rng.normal(size=(N_LAYERS, len(labels), D_MODEL)).astype(np.float32)
    direction = rng.normal(size=D_MODEL)
    direction /= np.linalg.norm(direction)
    signal_layer = int(rng.integers(2, N_LAYERS - 1))
    signed = np.where(labels == 1, 1.0, -1.0)[:, None]
    acts[signal_layer] += strength * signed * direction
    return acts, signal_layer


def _check_surface_audit() -> bool:
    """R15: verify the surface-shortcut audit catches shortcuts and passes clean concepts."""
    print("\n--- Surface-shortcut audit (R15) ---")

    # 1. A planted shortcut concept that must FAIL.
    shortcut_pairs = []
    for i in range(20):
        shortcut_pairs.append(
            Pair(
                positive=f"The MAGIC_TOKEN is present in sentence {i} about cooking.",
                negative=f"There is nothing special in sentence {i} about cooking.",
                family=f"fam{i % 4}",
            )
        )
    shortcut_concept = Concept(
        name="planted_shortcut",
        description="Should fail the audit.",
        pairs=shortcut_pairs,
        eval_prompts=["Write something."],
        behavior_question="Score it.",
    )
    shortcut_result = surface_shortcut_audit(shortcut_concept)
    print(f"  planted shortcut: surface AUROC={shortcut_result.surface_auroc:.3f} "
          f"passed={shortcut_result.passed} ({shortcut_result.reason})")

    # 2. Real concepts must pass unless their builder declared them
    #    surface-confounded before the audit ran. An undeclared failure is a
    #    construction bug in the concept set, not a result.
    results = audit_all_concepts()
    n_passed = sum(1 for r in results if r.passed)
    undeclared = [r.concept for r in results if not r.passed and not r.surface_confounded]
    for r in results:
        status = "PASS" if r.passed else ("FAIL*" if r.surface_confounded else "FAIL")
        print(f"  {r.concept:<14} [{status}] surface AUROC={r.surface_auroc:.3f} "
              f"(worst fold {r.worst_fold_auroc:.3f}) -- {r.reason}")

    ok = not shortcut_result.passed and n_passed >= 8 and not undeclared
    status = "OK" if ok else "MISMATCH"
    print(f"\n  Audit check [{status}]: "
          f"shortcut={'caught' if not shortcut_result.passed else 'MISSED'}, "
          f"passed={n_passed}/{len(results)} (need >=8), "
          f"undeclared failures={undeclared or 'none'}")
    print("  FAIL* = declared surface-confounded in the builder; the failure was "
          "predicted, and the concept is reported with the confound stated.")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/demo")
    ap.add_argument("--models", nargs="+", default=["fake-8b", "fake-2b"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    concepts = all_concepts()

    points: list[GapPoint] = []
    features: list[GeometryFeatures] = []
    curves: dict[str, list[tuple[float, float]]] = {}

    for mi, model in enumerate(args.models):
        for ci, concept in enumerate(concepts):
            seed = args.seed + 1000 * mi + ci
            texts, labels = pair_texts(concept.pairs)
            families = [p.family for p in concept.pairs for _ in (0, 1)]
            held = default_held_out(concept, seed=seed)
            train_mask = np.array([f not in held for f in families])

            # Readability: everything is readable here; the study's question is
            # whether that readability buys control.
            strength = float(rng.uniform(1.5, 3.0))
            acts, planted_layer = synthetic_acts(labels, strength, seed)
            probe = train_probes(
                concept, acts, labels, train_mask, model,
                seeds=(seed,), held_out_families=sorted(held),
            )

            immovable = concept.name in IMMOVABLE
            controllability = (
                float(rng.uniform(0.0, 0.03))
                if immovable
                else float(np.clip(probe.readability - 0.35 + rng.normal(scale=0.05), 0.05, 1.0))
            )

            points.append(
                GapPoint(
                    concept=concept.name,
                    model=model,
                    readability=probe.readability,
                    controllability=controllability,
                    readability_ci=probe.auroc_ci,
                    controllability_ci=(controllability - 0.02, controllability + 0.02),
                    safety_relevant=concept.safety_relevant,
                    best_layer=probe.best_layer,
                    selectivity=probe.selectivity,
                )
            )

            # Plant the geometric signal the predictor should recover:
            # immovable concepts sit outside the unembedding subspace.
            features.append(
                GeometryFeatures(
                    concept=concept.name,
                    model=model,
                    layer=probe.best_layer,
                    output_overlap=(
                        float(rng.uniform(0.05, 0.2)) if immovable
                        else float(rng.uniform(0.6, 0.95))
                    ),
                    participation_ratio=float(rng.uniform(5, 40)),
                    low_variance_pc_alignment=(
                        float(rng.uniform(0.5, 0.9)) if immovable
                        else float(rng.uniform(0.05, 0.3))
                    ),
                    residual_norm=float(rng.uniform(2, 8)),
                    n_directions=float(rng.integers(3, 6) if immovable else rng.integers(1, 3)),
                    direction_coherence=float(rng.uniform(0, 1)),
                    probe_dom_cosine=float(rng.uniform(0.5, 1.0)),
                )
            )

            if mi == 0 and concept.name in ("sentiment", "refusal"):
                coeffs = np.linspace(-3, 3, 9)
                if immovable:
                    ys = [0.5 + rng.normal(scale=0.01) for _ in coeffs]
                else:
                    ys = list(np.clip(0.5 + 0.18 * coeffs, 0, 1))
                label = f"{concept.name} ({'immovable' if immovable else 'controllable'})"
                curves[label] = list(zip(coeffs.tolist(), ys))

            print(
                f"  {model:>9} / {concept.name:<14} "
                f"readability={probe.readability:.3f} (layer {probe.best_layer}, "
                f"planted {planted_layer}) selectivity={probe.selectivity:+.3f} "
                f"controllability={controllability:.3f}"
            )

    gm = build_gap_map(points)
    gm.save(os.path.join(args.out_dir, "gap_map.json"))
    f1 = plot_gap_map(gm, os.path.join(args.out_dir, "fig1_gap_map.png"))
    f2 = plot_dose_response(curves, os.path.join(args.out_dir, "fig2_dose_response.png"))

    gaps = [p.gap for p in gm.points]
    report = fit_gap_predictor(
        features,
        gaps,
        controllabilities=[p.controllability for p in gm.points],
        readabilities=[p.readability for p in gm.points],
    )

    print("\n--- Experiment 3: gap map ---")
    print(f"  Spearman rho = {gm.spearman:.3f} "
          f"[{gm.spearman_ci[0]:.3f}, {gm.spearman_ci[1]:.3f}]")
    print(f"  {gm.interpretation}")
    print("\n--- Experiment 4: geometry predictor ---")
    prim = report.primary
    print(f"  primary test (H1, {prim.feature}): partial rho = "
          f"{prim.partial_spearman:.3f} "
          f"[{prim.partial_spearman_ci[0]:.3f}, {prim.partial_spearman_ci[1]:.3f}] "
          f"over {prim.n_concepts} concepts")
    print(f"    {prim.verdict}")
    for ex in report.exploratory:
        print(f"  {ex.label} ({ex.feature}): rho = {ex.partial_spearman:.3f}, "
              f"p_BH = {ex.p_value_bh:.3f}")
    print(f"  leave-one-out R^2 = {report.r2_loo:.3f}  (n={report.n})")
    print(f"  {report.verdict}")
    top = sorted(report.coefficients.items(), key=lambda kv: -abs(kv[1]))[:3]
    print("  strongest features: " + ", ".join(f"{k} ({v:+.3f})" for k, v in top))
    print(f"\nFigures: {f1}\n         {f2}")

    # Ground-truth check: the planted concepts must be the ones flagged.
    found = {p.concept for p in gm.danger_zone}
    expected = {c for c in IMMOVABLE if any(p.concept == c for p in points)}
    gap_status = "OK" if found == expected else "MISMATCH"
    print(f"\nGround truth check [{gap_status}]: planted {sorted(expected)}, "
          f"detected {sorted(found)}")

    # Surface-shortcut audit check (R15).
    audit_ok = _check_surface_audit()

    overall = gap_status == "OK" and audit_ok
    print(f"\n{'=' * 50}")
    print(f"Overall: {'ALL CHECKS PASSED' if overall else 'SOME CHECKS FAILED'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
