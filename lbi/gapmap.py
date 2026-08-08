"""Experiment 3: the gap map -- the figure that carries the paper.

Every concept-model pair is a point: readability on x, controllability on y.
The danger zone is high readability with low controllability. Both axes are
min-max normalized *within model* for the correlation and the plot, because raw
AUROC and raw dose-response AUC are not on comparable scales across models --
the design doc names that confound explicitly as the thing to rule out before
believing a pure-scatter result.

Danger-zone membership, however, is decided on **raw** values, and this is not
a detail. Min-max normalization guarantees that some point sits at readability
1.0 and some point at controllability 0.0, so thresholds applied to normalized
values manufacture a danger-zone occupant out of pure scatter -- the paper's
headline claim would then be an artifact of the rescaling. Raw AUROC >= 0.9 and
raw dose-response AUC <= 0.05 are absolute statements about a concept ("the
probe reads it, steering does not move it") and survive the normalization.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class GapPoint:
    concept: str
    model: str
    readability: float          # raw best-layer AUROC
    controllability: float      # raw dose-response AUC
    readability_ci: tuple[float, float]
    controllability_ci: tuple[float, float]
    safety_relevant: bool
    best_layer: int
    selectivity: float
    # Filled by normalize_within_model; None until then.
    norm_readability: float | None = None
    norm_controllability: float | None = None

    @property
    def gap(self) -> float:
        """Readability minus controllability on the within-model scale.

        Falls back to raw values if normalization has not been applied.
        """
        r = self.readability if self.norm_readability is None else self.norm_readability
        c = (
            self.controllability
            if self.norm_controllability is None
            else self.norm_controllability
        )
        return r - c

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gap"] = self.gap
        return d


@dataclass
class GapMap:
    points: list[GapPoint]
    spearman: float
    spearman_ci: tuple[float, float]
    pearson: float
    danger_zone: list[GapPoint]
    interpretation: str

    def to_dict(self) -> dict:
        return {
            "points": [p.to_dict() for p in self.points],
            "spearman": self.spearman,
            "spearman_ci_low": self.spearman_ci[0],
            "spearman_ci_high": self.spearman_ci[1],
            "pearson": self.pearson,
            "danger_zone": [p.concept + "@" + p.model for p in self.danger_zone],
            "interpretation": self.interpretation,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def normalize_within_model(points: list[GapPoint]) -> list[GapPoint]:
    """Min-max both axes within each model, writing the `norm_*` fields.

    Raw values are left untouched: danger-zone membership is judged on them
    (see the module docstring for why normalized thresholds are unsafe).
    """
    out: list[GapPoint] = []
    for m in sorted({p.model for p in points}):
        group = [p for p in points if p.model == m]
        for axis in ("readability", "controllability"):
            vals = np.array([getattr(p, axis) for p in group], dtype=float)
            lo, hi = float(vals.min()), float(vals.max())
            span = hi - lo
            for p in group:
                v = getattr(p, axis)
                setattr(
                    p, f"norm_{axis}", 0.5 if span <= 0 else (v - lo) / span
                )
        out.extend(group)
    return out


def _bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    concepts: list[str] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap CI on Spearman rho, clustering by concept when told how (P7).

    The unit of generalization is the concept, not the concept-model pair:
    one concept measured across five models contributes five correlated points,
    and resampling them independently treats correlated observations as fresh
    ones, narrowing the interval. That is objection 10 in PLAN.md's ledger
    ("fifty points from twelve concepts is not fifty observations"), which the
    plan answers with a cluster bootstrap -- so the headline correlation on
    Figure 1 has to use one.

    `concepts=None` falls back to the i.i.d. bootstrap, which is only correct
    when every point comes from a distinct concept.
    """
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    vals = []

    if concepts is None:
        draw = lambda: rng.integers(0, len(x), len(x))  # noqa: E731
    else:
        arr = np.array(concepts)
        unique = sorted(set(concepts))
        members = {c: np.where(arr == c)[0] for c in unique}

        def draw():
            sampled = rng.choice(unique, size=len(unique), replace=True)
            return np.concatenate([members[c] for c in sampled])

    for _ in range(n_boot):
        idx = draw()
        if len(idx) < 3 or np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        rho = spearmanr(x[idx], y[idx])[0]
        if not np.isnan(rho):
            vals.append(rho)
    if len(vals) < 100:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def _ci_excludes(
    value: float, ci: tuple[float, float], threshold: float, direction: str
) -> bool:
    """Does the interval sit wholly on the qualifying side of the threshold?

    `direction="above"` wants the CI's lower bound at or above the threshold,
    `"below"` wants the upper bound at or below it. A missing (NaN) bound never
    qualifies: an unknown interval is not an excluding one.
    """
    lo, hi = ci
    if direction == "above":
        return value >= threshold and not np.isnan(lo) and lo >= threshold
    return value <= threshold and not np.isnan(hi) and hi <= threshold


def build_gap_map(
    points: list[GapPoint],
    readability_threshold: float = 0.9,
    controllability_threshold: float = 0.05,
    normalize: bool = True,
    require_ci_exclusion: bool = True,
) -> GapMap:
    """Assemble the map, locate the danger zone, and interpret the correlation.

    Thresholds are absolute (raw AUROC, raw dose-response AUC), deliberately:
    see the module docstring.

    P6 additionally requires the bootstrap CI on each axis to exclude its
    threshold, not merely the point estimate to clear it. Without that, a
    concept measured at controllability 0.04 with an interval running to 0.30
    counts as immovable on the strength of scatter -- the same error revision
    R10 caught with normalization, one step milder. `require_ci_exclusion=False`
    reproduces the point-estimate-only rule for comparison; it is not the
    preregistered one.
    """
    from scipy.stats import pearsonr, spearmanr

    if len(points) < 3:
        raise ValueError(f"need at least 3 points for a gap map, got {len(points)}")

    pts = list(points)
    if normalize:
        normalize_within_model(pts)
        x = np.array([p.norm_readability for p in pts], dtype=float)
        y = np.array([p.norm_controllability for p in pts], dtype=float)
    else:
        x = np.array([p.readability for p in pts], dtype=float)
        y = np.array([p.controllability for p in pts], dtype=float)

    rho = float(spearmanr(x, y)[0]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    r = float(pearsonr(x, y)[0]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    ci = _bootstrap_spearman_ci(x, y, concepts=[p.concept for p in pts])

    if require_ci_exclusion:
        danger = [
            p
            for p in pts
            if _ci_excludes(
                p.readability, p.readability_ci, readability_threshold, "above"
            )
            and _ci_excludes(
                p.controllability,
                p.controllability_ci,
                controllability_threshold,
                "below",
            )
        ]
    else:
        danger = [
            p
            for p in pts
            if p.readability >= readability_threshold
            and p.controllability <= controllability_threshold
        ]

    # The design doc pre-registers all three outcomes; say which one this is
    # rather than leaving the framing to be chosen after seeing the number.
    if not np.isnan(rho) and rho > 0.7 and not danger:
        interpretation = (
            "Detection largely predicts control (rho={:.2f}); no danger-zone "
            "points. Reframe around the residuals per Section 7.".format(rho)
        )
    elif danger:
        interpretation = (
            "Danger zone occupied by {} point(s): {}. Readable-but-immovable "
            "concepts exist; confirm each against all steering variants before "
            "claiming it.".format(
                len(danger), ", ".join(f"{p.concept}@{p.model}" for p in danger)
            )
        )
    elif not np.isnan(rho) and abs(rho) < 0.3:
        interpretation = (
            "No reliable relationship (rho={:.2f}) after within-model "
            "normalization: control is not predictable from detection.".format(rho)
        )
    else:
        interpretation = (
            "Partial relationship (rho={:.2f}) with no point clearing the "
            "danger-zone thresholds.".format(rho)
        )

    return GapMap(
        points=pts,
        spearman=rho,
        spearman_ci=ci,
        pearson=r,
        danger_zone=danger,
        interpretation=interpretation,
    )


def plot_gap_map(
    gm: GapMap,
    path: str,
    title: str = "Detection vs. control",
    label_all: bool = False,
) -> str:
    """Figure 1: the gap map.

    By default only the points that carry the argument are labelled -- danger
    zone and safety-relevant concepts -- because forty overlapping annotations
    are unreadable. `label_all=True` restores every label for the appendix
    version.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def coords(p: GapPoint) -> tuple[float, float]:
        return (
            p.readability if p.norm_readability is None else p.norm_readability,
            p.controllability
            if p.norm_controllability is None
            else p.norm_controllability,
        )

    fig, ax = plt.subplots(figsize=(8.5, 6))

    danger = {id(p) for p in gm.danger_zone}
    markers = ["o", "s", "^", "D", "v", "P"]
    models = sorted({p.model for p in gm.points})
    for i, m in enumerate(models):
        group = [p for p in gm.points if p.model == m]
        xs = [coords(p)[0] for p in group]
        ys = [coords(p)[1] for p in group]
        safety = [p.safety_relevant for p in group]
        ax.scatter(
            xs, ys,
            marker=markers[i % len(markers)],
            s=[90 if s else 55 for s in safety],
            edgecolors=["#d62728" if s else "none" for s in safety],
            linewidths=1.6,
            alpha=0.85,
            label=m.split("/")[-1],
            zorder=3,
        )
        # Alternate the offset so that near-coincident labels stay legible.
        for j, p in enumerate(group):
            if not (label_all or p.safety_relevant or id(p) in danger):
                continue
            dy = 6 if j % 2 == 0 else -11
            ax.annotate(
                p.concept, coords(p),
                fontsize=7, alpha=0.9 if id(p) in danger else 0.7,
                xytext=(5, dy), textcoords="offset points",
                color="#d62728" if id(p) in danger else "#333333",
            )

    # Danger-zone membership is decided on raw values, so it is circled per
    # point rather than drawn as a fixed region of the normalized axes.
    for p in gm.danger_zone:
        px, py = coords(p)
        ax.scatter(
            [px], [py], s=260, facecolors="none",
            edgecolors="#d62728", linewidths=1.8, zorder=2,
        )
    if gm.danger_zone:
        ax.scatter([], [], s=120, facecolors="none", edgecolors="#d62728",
                   linewidths=1.8, label="danger zone (raw AUROC high, AUC ~0)")

    ax.set_xlabel("Readability (probe AUROC, normalized within model)")
    ax.set_ylabel("Controllability (dose-response AUC, normalized within model)")
    ax.set_title(f"{title}\nSpearman rho = {gm.spearman:.2f} "
                 f"[{gm.spearman_ci[0]:.2f}, {gm.spearman_ci[1]:.2f}]")
    ax.set_xlim(-0.08, 1.12)
    ax.set_ylim(-0.08, 1.12)
    ax.grid(alpha=0.2, zorder=0)
    # Outside the axes: with points pinned to the corners by min-max
    # normalization, any in-axes legend position covers data.
    ax.legend(
        fontsize=8, title="model", loc="upper left",
        bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=False,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_dose_response(
    curves: dict[str, list[tuple[float, float]]],
    path: str,
    title: str = "Dose-response: probe fires, steering does not move behavior",
) -> str:
    """Figure 2: the dissociation existence proof.

    `curves` maps a label to [(coeff, behavior), ...]. Pass one readable-and-
    controllable concept and one readable-but-immovable one; the flat line next
    to the steep one is the screenshot the paper is built around.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, pts in curves.items():
        pts = sorted(pts)
        ax.plot(
            [c for c, _ in pts], [b for _, b in pts],
            marker="o", markersize=4, label=label,
        )
    ax.set_xlabel("Steering coefficient (residual-stream RMS units)")
    ax.set_ylabel("Behavior score")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
