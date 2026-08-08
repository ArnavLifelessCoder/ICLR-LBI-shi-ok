"""Experiment 1: the readability axis.

Per-layer logistic-regression probes on residual activations, with two controls
the design doc's revision R4 requires:

  * evaluation on held-out *template families*, so a probe that only works
    within a family (i.e. reads lexical cues) scores badly;
  * a shuffled-label control probe, giving selectivity = AUROC - control AUROC.

Readability of a concept = best-layer held-out AUROC, with a bootstrap CI (R8).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .concepts import Concept, Pair


@dataclass
class LayerProbeResult:
    layer: int
    auroc: float
    control_auroc: float
    selectivity: float
    direction: np.ndarray  # probe weight vector, unit norm

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("direction")
        return d


@dataclass
class ProbeResult:
    """Readability for one concept-model pair."""

    concept: str
    model: str
    best_layer: int
    readability: float          # best-layer AUROC
    control_auroc: float
    selectivity: float
    auroc_ci: tuple[float, float]
    per_layer: list[LayerProbeResult]
    held_out_families: list[str]
    n_train: int
    n_test: int

    def layer_curve(self) -> list[float]:
        return [r.auroc for r in self.per_layer]

    def probe_direction(self) -> np.ndarray:
        return self.per_layer[self.best_layer].direction

    def summary(self) -> dict:
        return {
            "concept": self.concept,
            "model": self.model,
            "best_layer": self.best_layer,
            "readability": self.readability,
            "control_auroc": self.control_auroc,
            "selectivity": self.selectivity,
            "auroc_ci_low": self.auroc_ci[0],
            "auroc_ci_high": self.auroc_ci[1],
            "held_out_families": ",".join(self.held_out_families),
            "n_train": self.n_train,
            "n_test": self.n_test,
        }


def pair_texts(pairs: list[Pair]) -> tuple[list[str], np.ndarray]:
    """Flatten pairs into (texts, labels) with label 1 = concept present."""
    texts, labels = [], []
    for p in pairs:
        texts.append(p.positive)
        labels.append(1)
        texts.append(p.negative)
        labels.append(0)
    return texts, np.array(labels)


def default_held_out(concept: Concept, n: int = 1, seed: int = 0) -> set[str]:
    """Pick `n` template families to hold out, deterministically."""
    fams = concept.families()
    if len(fams) <= n:
        raise ValueError(
            f"{concept.name} has {len(fams)} families; cannot hold out {n}"
        )
    rng = np.random.default_rng(seed)
    return set(rng.choice(fams, size=n, replace=False).tolist())


def _bootstrap_auroc_ci(
    y: np.ndarray, scores: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], scores[idx]))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def _fit_one_layer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    C: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Returns (auroc, unit-norm direction, test scores)."""
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=2000, C=C, random_state=seed)
    clf.fit(scaler.transform(X_train), y_train)
    scores = clf.decision_function(scaler.transform(X_test))
    auroc = float(roc_auc_score(y_test, scores))
    # Undo the scaling so the direction lives in raw activation space.
    w = clf.coef_[0] / np.maximum(scaler.scale_, 1e-8)
    w = w / (np.linalg.norm(w) + 1e-12)
    return auroc, w, scores


def train_probes(
    concept: Concept,
    acts: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    model_name: str,
    layers: list[int] | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
    C: float = 1.0,
    held_out_families: list[str] | None = None,
) -> ProbeResult:
    """Fit per-layer probes.

    acts: (n_layers, n_texts, d_model) as returned by extraction.capture_*.
    labels: (n_texts,) with 1 = concept present.
    train_mask: (n_texts,) bool, True for training rows (family split).
    """
    n_layers = acts.shape[0]
    layers = list(range(n_layers)) if layers is None else layers
    y_train, y_test = labels[train_mask], labels[~train_mask]
    if len(np.unique(y_test)) < 2:
        raise ValueError(f"{concept.name}: held-out split has one class only")

    per_layer: list[LayerProbeResult] = []
    best_scores: np.ndarray | None = None
    best_auroc = -1.0

    for li in layers:
        X = acts[li].astype(np.float32)
        X_train, X_test = X[train_mask], X[~train_mask]

        aurocs, dirs, score_sets = [], [], []
        control_aurocs = []
        for seed in seeds:
            auroc, w, scores = _fit_one_layer(X_train, y_train, X_test, y_test, seed, C)
            aurocs.append(auroc)
            dirs.append(w)
            score_sets.append(scores)

            rng = np.random.default_rng(seed + 9973)
            y_shuf = rng.permutation(y_train)
            ctrl_auroc, _, _ = _fit_one_layer(
                X_train, y_shuf, X_test, y_test, seed, C
            )
            control_aurocs.append(ctrl_auroc)

        mean_auroc = float(np.mean(aurocs))
        mean_ctrl = float(np.mean(control_aurocs))
        direction = np.mean(dirs, axis=0)
        direction = direction / (np.linalg.norm(direction) + 1e-12)

        per_layer.append(
            LayerProbeResult(
                layer=li,
                auroc=mean_auroc,
                control_auroc=mean_ctrl,
                selectivity=mean_auroc - mean_ctrl,
                direction=direction,
            )
        )
        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            best_scores = np.mean(score_sets, axis=0)

    best = max(per_layer, key=lambda r: r.auroc)
    ci = _bootstrap_auroc_ci(y_test, best_scores)

    return ProbeResult(
        concept=concept.name,
        model=model_name,
        best_layer=best.layer,
        readability=best.auroc,
        control_auroc=best.control_auroc,
        selectivity=best.selectivity,
        auroc_ci=ci,
        per_layer=per_layer,
        held_out_families=sorted(held_out_families or []),
        n_train=int(train_mask.sum()),
        n_test=int((~train_mask).sum()),
    )


def diff_of_means_direction(
    acts_layer: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Difference-of-means direction at one layer, unit norm.

    The design doc's default steering direction (Experiment 2): standard and
    more robust than probe weights for intervention.
    """
    pos = acts_layer[labels == 1].astype(np.float32).mean(axis=0)
    neg = acts_layer[labels == 0].astype(np.float32).mean(axis=0)
    d = pos - neg
    return d / (np.linalg.norm(d) + 1e-12)


def repe_reading_vector(
    acts_layer: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """RepE reading vector: first PC of paired activation differences (Zou et al.).

    Revision R16: the third direction-derivation method. Computed from
    activations already cached for Experiment 1, so it costs one PCA and no
    forward pass.

    Pairs are formed by aligning positive and negative examples in order of
    appearance (which is how pair_texts() flattens them: pos, neg, pos, neg...).
    The first principal component of those difference vectors is the reading
    vector.

    The differences are deliberately *not* mean-centred. The concept signal in
    (pos - neg) lives almost entirely in the mean of those differences, so
    subtracting it leaves only noise and PC1 recovers nothing. Zou et al. avoid
    this by randomising which member of each pair is subtracted, which makes the
    mean vanish by symmetry and puts the signal in the variance; taking the top
    right singular vector of the uncentred differences is the equivalent that
    does not depend on a random draw.
    """
    X = acts_layer.astype(np.float32)
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_acts = X[pos_mask]
    neg_acts = X[neg_mask]
    n = min(len(pos_acts), len(neg_acts))
    if n < 2:
        # Fall back to diff-of-means when there aren't enough pairs for PCA.
        return diff_of_means_direction(acts_layer, labels)
    diffs = pos_acts[:n] - neg_acts[:n]
    # Top right singular vector of the uncentred differences (see docstring).
    _, _, Vt = np.linalg.svd(diffs, full_matrices=False)
    pc1 = Vt[0]
    # Ensure consistent sign: align with diff-of-means so downstream code
    # doesn't flip polarity unexpectedly.
    dom = diff_of_means_direction(acts_layer, labels)
    if np.dot(pc1, dom) < 0:
        pc1 = -pc1
    return pc1 / (np.linalg.norm(pc1) + 1e-12)

