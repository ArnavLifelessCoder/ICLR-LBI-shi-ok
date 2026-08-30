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
    auroc: float              # test-split AUROC
    control_auroc: float
    selectivity: float
    direction: np.ndarray     # probe weight vector, unit norm
    val_auroc: float = float("nan")  # validation AUROC; what the layer is chosen on

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
    n_val: int = 0

    def layer_curve(self) -> list[float]:
        return [r.auroc for r in self.per_layer]

    def probe_direction(self) -> np.ndarray:
        by_layer = {r.layer: r for r in self.per_layer}
        return by_layer[self.best_layer].direction

    def per_layer_curve(self) -> list[dict]:
        """Layer-by-layer AUROCs, for the result file.

        Persisted because the layer choice has to be auditable: when the first
        real run put the probe at layer 1 of 28 there was no way to tell from
        the output whether that layer was a genuine peak or one of many tied at
        1.000.
        """
        return [
            {"layer": r.layer, "val_auroc": r.val_auroc, "test_auroc": r.auroc,
             "control_auroc": r.control_auroc}
            for r in self.per_layer
        ]

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
            "n_val": self.n_val,
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


def default_splits(
    concept: Concept, seed: int = 0
) -> tuple[set[str], set[str]]:
    """Return (validation_families, test_families), disjoint from each other.

    P2 requires the probe layer to be chosen on a validation split that is
    disjoint from both training and test families, and **never** on test AUROC.
    A single train/test split cannot honour that: picking the layer by test
    AUROC and then reporting that same number is selection on the test set, and
    it inflates readability by exactly the amount the selection buys.

    It also biases *which* layer is chosen. When several layers saturate at
    AUROC 1.000 the argmax returns the earliest of them, and on the first real
    run that put the sentiment probe at layer 1 of 28 -- which then anchored the
    steering band to layers 0-6, where an intervention large enough to move
    behaviour drives perplexity from 5 to 506.
    """
    fams = concept.families()
    if len(fams) < 3:
        raise ValueError(
            f"{concept.name} has {len(fams)} families; a train/validation/test "
            f"split by family needs at least 3"
        )
    rng = np.random.default_rng(seed)
    picked = rng.choice(fams, size=2, replace=False).tolist()
    return {picked[0]}, {picked[1]}


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
    val_mask: np.ndarray | None = None,
) -> ProbeResult:
    """Fit per-layer probes.

    acts: (n_layers, n_texts, d_model) as returned by extraction.capture_*.
    labels: (n_texts,) with 1 = concept present.
    train_mask: (n_texts,) bool, True for training rows (family split).
    val_mask: (n_texts,) bool, the validation families used to *choose* the
        layer. Test is everything in neither mask, and readability is the test
        AUROC at the validation-selected layer.

    Without `val_mask` the layer is chosen on the test split and reported from
    it, which P2 forbids in as many words. That path is kept only so old
    callers do not break silently, and it warns.
    """
    n_layers = acts.shape[0]
    layers = list(range(n_layers)) if layers is None else layers

    if val_mask is None:
        import warnings

        warnings.warn(
            "train_probes called without val_mask: the layer will be chosen on "
            "the test split and reported from it. P2 forbids this and it "
            "inflates readability. Pass val_mask from default_splits().",
            stacklevel=2,
        )
        # Legacy behaviour: selection and reporting share one split.
        val_mask = ~train_mask
        test_mask = ~train_mask
    else:
        test_mask = ~(train_mask | val_mask)
    y_train = labels[train_mask]
    y_val = labels[val_mask]
    y_test = labels[test_mask]
    if len(np.unique(y_test)) < 2:
        raise ValueError(f"{concept.name}: test split has one class only")
    if len(np.unique(y_val)) < 2:
        raise ValueError(f"{concept.name}: validation split has one class only")

    per_layer: list[LayerProbeResult] = []
    test_scores_by_layer: dict[int, np.ndarray] = {}

    for li in layers:
        X = acts[li].astype(np.float32)
        X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]

        val_aurocs, test_aurocs, dirs, test_score_sets = [], [], [], []
        control_aurocs = []
        for seed in seeds:
            val_auroc, w, _ = _fit_one_layer(X_train, y_train, X_val, y_val, seed, C)
            test_auroc, _, test_scores = _fit_one_layer(
                X_train, y_train, X_test, y_test, seed, C
            )
            val_aurocs.append(val_auroc)
            test_aurocs.append(test_auroc)
            dirs.append(w)
            test_score_sets.append(test_scores)

            rng = np.random.default_rng(seed + 9973)
            y_shuf = rng.permutation(y_train)
            ctrl_auroc, _, _ = _fit_one_layer(
                X_train, y_shuf, X_test, y_test, seed, C
            )
            control_aurocs.append(ctrl_auroc)

        mean_val = float(np.mean(val_aurocs))
        mean_test = float(np.mean(test_aurocs))
        mean_ctrl = float(np.mean(control_aurocs))
        direction = np.mean(dirs, axis=0)
        direction = direction / (np.linalg.norm(direction) + 1e-12)

        per_layer.append(
            LayerProbeResult(
                layer=li,
                auroc=mean_test,
                control_auroc=mean_ctrl,
                selectivity=mean_test - mean_ctrl,
                direction=direction,
                val_auroc=mean_val,
            )
        )
        test_scores_by_layer[li] = np.mean(test_score_sets, axis=0)

    # Selection on validation. Ties broken toward the middle of the network
    # rather than by list order: several layers saturating at 1.000 is common,
    # and argmax then silently returns the earliest, which is the worst place
    # to steer from.
    best_val = max(r.val_auroc for r in per_layer)
    tied = [r for r in per_layer if r.val_auroc >= best_val - 1e-9]
    mid = (n_layers - 1) / 2.0
    best = min(tied, key=lambda r: abs(r.layer - mid))
    ci = _bootstrap_auroc_ci(y_test, test_scores_by_layer[best.layer])

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
        n_test=int(test_mask.sum()),
        n_val=int(val_mask.sum()),
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

