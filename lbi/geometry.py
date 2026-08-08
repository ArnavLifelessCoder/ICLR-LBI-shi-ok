"""Experiment 4: predict the gap from geometry.

Revision R13 trimmed this module from a multi-feature regression to one
preregistered primary hypothesis plus two labeled exploratory analyses.

**Primary hypothesis H1 (mechanism M1, read/write mismatch):**
    output_overlap  -- how much of the concept direction survives projection
                       into the model's output-effective subspace (via the
                       unembedding matrix).  Tested by partial Spearman
                       controlling for readability, with cluster bootstrap
                       resampling concepts (P7).

**Exploratory E1 (mechanism M2, redundancy):**
    participation_ratio -- effective dimensionality of the class-difference
                           covariance.

**Exploratory E2 (mechanism M3, off-manifold):**
    low_variance_pc_alignment -- alignment of the concept direction with the
                                 low-variance principal components of the
                                 residual stream.  Conditioned on the fluency-
                                 limited subset only.

Dropped from the predictor (per R13):
    residual_norm   -- absorbed by R2's RMS normalization of steering strength
    n_directions    -- redundant with participation_ratio

Kept as appendix metadata:
    direction_coherence, probe_dom_cosine, tuning_shift, residual_norm,
    n_directions
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, asdict

import numpy as np

from .concepts import Concept, Pair
from .probes import diff_of_means_direction


@dataclass
class GeometryFeatures:
    concept: str
    model: str
    layer: int
    # --- Primary (H1) ---
    output_overlap: float
    # --- Exploratory (E1) ---
    participation_ratio: float
    # --- Exploratory (E2) ---
    low_variance_pc_alignment: float
    # --- Appendix metadata (not in the predictor) ---
    residual_norm: float
    n_directions: float
    direction_coherence: float
    probe_dom_cosine: float
    tuning_shift: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


# The unembedding basis depends only on the model, not the concept, but
# output_overlap is called once per concept. At Qwen2.5-7B's 152k x 3584 head
# each randomized SVD runs about 40 seconds and touches a 2.2 GB float32 view,
# so recomputing it for ten concepts cost roughly seven minutes of a T4 session
# and ten chances to run Colab out of host RAM next to the loaded model.
# Keyed by object identity and validated through a weakref, so a freed array
# whose address gets reused cannot serve a stale basis.
_BASIS_CACHE: "dict[tuple[int, int], tuple[weakref.ref, np.ndarray]]" = {}


def _top_unembedding_basis(W: np.ndarray, k: int, owner) -> np.ndarray:
    from sklearn.utils.extmath import randomized_svd

    key = (id(owner), k)
    hit = _BASIS_CACHE.get(key)
    if hit is not None and hit[0]() is owner:
        return hit[1]

    _, _, Vt = randomized_svd(W, n_components=k, random_state=0)
    if len(_BASIS_CACHE) > 4:  # at most a couple of models per process
        _BASIS_CACHE.clear()
    try:
        _BASIS_CACHE[key] = (weakref.ref(owner), Vt)
    except TypeError:
        pass  # not weak-referenceable; correctness is unaffected
    return Vt


def output_overlap(direction: np.ndarray, lm_head_weight: np.ndarray, top_k: int = 512) -> float:
    """Fraction of the direction's norm lying in the top unembedding subspace.

    Renamed from unembed_alignment for clarity (matches design doc Section 2.5
    terminology).  Uses the top-`top_k` right singular vectors of the
    unembedding matrix, which is far cheaper than the full vocabulary and
    captures the directions the output head actually reads.

    The basis is cached per unembedding matrix; see `_top_unembedding_basis`.
    """
    W = np.asarray(lm_head_weight, dtype=np.float32)  # (vocab, d_model)
    if W.shape[1] != direction.shape[0]:
        W = W.T
    k = min(top_k, min(W.shape) - 1)
    Vt = _top_unembedding_basis(W, k, lm_head_weight)
    d = direction / (np.linalg.norm(direction) + 1e-12)
    proj = Vt @ d
    return float(np.linalg.norm(proj))


def participation_ratio(acts: np.ndarray) -> float:
    """Effective dimensionality: (sum λ)^2 / sum λ^2 over the covariance spectrum."""
    X = np.asarray(acts, dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    # Eigenvalues of the covariance == squared singular values / (n-1).
    s = np.linalg.svd(X, compute_uv=False)
    lam = s.astype(np.float64) ** 2
    denom = float((lam**2).sum())
    if denom <= 0:
        return 0.0
    return float((lam.sum() ** 2) / denom)


def low_variance_pc_alignment(
    direction: np.ndarray,
    acts: np.ndarray,
    bottom_fraction: float = 0.25,
) -> float:
    """Alignment of the concept direction with low-variance PCs of the residual stream.

    Exploratory E2 (mechanism M3, off-manifold): if the concept direction
    aligns with directions the model's activation distribution has low variance
    along, then steering far enough to change behavior also moves activations
    somewhere the model has never been, and fluency collapses first.

    Returns the fraction of the direction's norm lying in the bottom
    `bottom_fraction` of PCs by variance.
    """
    X = np.asarray(acts, dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    # s is sorted descending; the low-variance PCs are at the tail.
    n_low = max(1, int(len(s) * bottom_fraction))
    low_pcs = Vt[-n_low:]  # (n_low, d_model)
    d = direction / (np.linalg.norm(direction) + 1e-12)
    proj = low_pcs @ d
    return float(np.linalg.norm(proj))


def residual_norm(acts: np.ndarray) -> float:
    """Mean RMS norm of the activations, matching steering's strength unit."""
    X = np.asarray(acts, dtype=np.float32)
    return float(np.sqrt((X**2).mean(axis=1)).mean())


def per_family_directions(
    acts_layer: np.ndarray, labels: np.ndarray, families: list[str]
) -> dict[str, np.ndarray]:
    """A difference-of-means direction per template family."""
    fams = sorted(set(families))
    out = {}
    arr_fams = np.array(families)
    for f in fams:
        mask = arr_fams == f
        if len(np.unique(labels[mask])) < 2:
            continue
        out[f] = diff_of_means_direction(acts_layer[mask], labels[mask])
    return out


def n_effective_directions(dirs: dict[str, np.ndarray], threshold: float = 0.9) -> float:
    """How many singular directions to explain `threshold` of the per-family set.

    One shared direction across families gives ~1; a concept carried by several
    geometrically distinct directions gives more.
    """
    if len(dirs) < 2:
        return float(len(dirs))
    M = np.stack(list(dirs.values()))
    s = np.linalg.svd(M, compute_uv=False)
    energy = np.cumsum(s**2) / (s**2).sum()
    return float(np.searchsorted(energy, threshold) + 1)


def direction_coherence(dirs: dict[str, np.ndarray]) -> float:
    """Mean pairwise cosine between per-family directions."""
    vals = list(dirs.values())
    if len(vals) < 2:
        return 1.0
    cos = []
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            cos.append(float(np.dot(vals[i], vals[j])))
    return float(np.mean(cos))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def compute_features(
    concept: Concept,
    model_name: str,
    layer: int,
    acts_layer: np.ndarray,
    labels: np.ndarray,
    families: list[str],
    probe_direction: np.ndarray,
    lm_head_weight: np.ndarray | None = None,
    base_direction: np.ndarray | None = None,
) -> GeometryFeatures:
    """All geometry features for one concept-model pair at the steering layer."""
    dom = diff_of_means_direction(acts_layer, labels)
    fam_dirs = per_family_directions(acts_layer, labels, families)

    return GeometryFeatures(
        concept=concept.name,
        model=model_name,
        layer=layer,
        output_overlap=(
            output_overlap(dom, lm_head_weight)
            if lm_head_weight is not None
            else float("nan")
        ),
        participation_ratio=participation_ratio(acts_layer),
        low_variance_pc_alignment=low_variance_pc_alignment(dom, acts_layer),
        residual_norm=residual_norm(acts_layer),
        n_directions=n_effective_directions(fam_dirs),
        direction_coherence=direction_coherence(fam_dirs),
        probe_dom_cosine=abs(cosine(probe_direction, dom)),
        tuning_shift=(
            cosine(dom, base_direction) if base_direction is not None else float("nan")
        ),
    )


# --------------------------------------------------------------------------
# The predictor (R13: one primary test + two exploratory)
# --------------------------------------------------------------------------

# Primary feature (H1):
PRIMARY_FEATURE = "output_overlap"

# Exploratory features:
EXPLORATORY_FEATURES = ["participation_ratio", "low_variance_pc_alignment"]

# All features that exist in the dataclass (for backward compat / appendix):
ALL_FEATURE_NAMES = [
    "output_overlap",
    "participation_ratio",
    "low_variance_pc_alignment",
    "residual_norm",
    "n_directions",
    "direction_coherence",
    "probe_dom_cosine",
    "tuning_shift",
]


@dataclass
class PrimaryTestReport:
    """Result of the preregistered primary test (H1)."""
    feature: str
    partial_spearman: float
    partial_spearman_ci: tuple[float, float]
    n_concepts: int
    n_points: int
    verdict: str


@dataclass
class ExploratoryReport:
    """Result of one exploratory analysis."""
    feature: str
    partial_spearman: float
    partial_spearman_ci: tuple[float, float]
    p_value_raw: float
    p_value_bh: float  # Benjamini-Hochberg corrected
    n: int
    label: str  # "exploratory E1" or "exploratory E2"


@dataclass
class PredictorReport:
    """Combined report for Experiment 4."""
    primary: PrimaryTestReport
    exploratory: list[ExploratoryReport]
    # Backward-compat fields for existing tests and the aggregator:
    features_used: list[str]
    r2_loo: float
    spearman: float
    coefficients: dict[str, float]
    n: int
    verdict: str


def _partial_spearman(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> float:
    """Partial Spearman correlation between x and y controlling for z.

    Regress ranks of x on ranks of z, regress ranks of y on ranks of z,
    correlate the residuals.
    """
    from scipy.stats import spearmanr, rankdata

    rx = rankdata(x)
    ry = rankdata(y)
    rz = rankdata(z)

    # Residualize x and y against z via OLS on ranks.
    def _resid(target, covariate):
        cov_mean = covariate - covariate.mean()
        target_mean = target - target.mean()
        if np.dot(cov_mean, cov_mean) < 1e-12:
            return target_mean
        beta = np.dot(target_mean, cov_mean) / np.dot(cov_mean, cov_mean)
        return target_mean - beta * cov_mean

    res_x = _resid(rx, rz)
    res_y = _resid(ry, rz)

    if np.std(res_x) < 1e-12 or np.std(res_y) < 1e-12:
        return float("nan")
    return float(spearmanr(res_x, res_y)[0])


def _cluster_bootstrap_partial_spearman(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    concepts: list[str],
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap CI on partial Spearman, resampling concepts with replacement (P7).

    The unit of generalization is the concept, not the concept-model pair.
    """
    rng = np.random.default_rng(seed)
    unique_concepts = sorted(set(concepts))
    concept_arr = np.array(concepts)
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_concepts, size=len(unique_concepts), replace=True)
        idx = np.concatenate([np.where(concept_arr == c)[0] for c in sampled])
        if len(idx) < 4 or np.std(x[idx]) < 1e-12 or np.std(y[idx]) < 1e-12:
            continue
        vals.append(_partial_spearman(x[idx], y[idx], z[idx]))
    # A resample can still come back NaN (residuals with no variance left after
    # controlling for z) despite the guard above, and np.percentile propagates a
    # single NaN to the whole interval -- which would silently wipe out the CI
    # on the study's one preregistered test. Drop them and report on the rest.
    vals = [v for v in vals if not np.isnan(v)]
    if len(vals) < 100:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def primary_test(
    features: list[GeometryFeatures],
    controllabilities: list[float],
    readabilities: list[float],
    concepts: list[str],
) -> PrimaryTestReport:
    """Preregistered test: partial Spearman of output_overlap vs controllability | readability.

    Cluster bootstrap resamples concepts (P7). Binary preregistered claim:
    either a strong relationship (rho >= 0.7) or uninformative about weaker ones.
    """
    x = np.array([f.output_overlap for f in features], dtype=float)
    y = np.array(controllabilities, dtype=float)
    z = np.array(readabilities, dtype=float)

    # Drop NaN output_overlap entries.
    mask = ~np.isnan(x)
    if mask.sum() < 5:
        return PrimaryTestReport(
            feature="output_overlap",
            partial_spearman=float("nan"),
            partial_spearman_ci=(float("nan"), float("nan")),
            n_concepts=len(set(c for c, m in zip(concepts, mask) if m)),
            n_points=int(mask.sum()),
            verdict="insufficient data for the primary test",
        )

    x, y, z = x[mask], y[mask], z[mask]
    c_masked = [c for c, m in zip(concepts, mask) if m]

    rho = _partial_spearman(x, y, z)
    ci = _cluster_bootstrap_partial_spearman(x, y, z, c_masked)
    n_concepts = len(set(c_masked))

    if not np.isnan(rho) and abs(rho) >= 0.7:
        verdict = (
            f"output overlap predicts controllability (partial rho={rho:.3f}); "
            "geometry explains the gap"
        )
    elif not np.isnan(rho) and abs(rho) >= 0.3:
        verdict = (
            f"suggestive but not strong (partial rho={rho:.3f}); "
            "report as suggestive, not confirmation"
        )
    else:
        verdict = (
            "the gap is real and not explained by the standard first-order "
            "account of steering (output overlap does not predict controllability)"
        )

    return PrimaryTestReport(
        feature="output_overlap",
        partial_spearman=rho,
        partial_spearman_ci=ci,
        n_concepts=n_concepts,
        n_points=int(mask.sum()),
        verdict=verdict,
    )


def exploratory_analysis(
    features: list[GeometryFeatures],
    controllabilities: list[float],
    readabilities: list[float],
    concepts: list[str],
    fluency_limited_mask: np.ndarray | None = None,
) -> list[ExploratoryReport]:
    """Exploratory E1 (participation_ratio) and E2 (low_variance_pc_alignment).

    E2 is conditioned on the fluency-limited subset only (per design Section
    2.5, mechanism M3). BH correction across the exploratory family (P8).
    """
    from scipy.stats import spearmanr

    results = []
    y_all = np.array(controllabilities, dtype=float)
    z_all = np.array(readabilities, dtype=float)

    for i, (feat_name, label) in enumerate([
        ("participation_ratio", "exploratory E1"),
        ("low_variance_pc_alignment", "exploratory E2"),
    ]):
        if feat_name == "low_variance_pc_alignment" and fluency_limited_mask is not None:
            mask = fluency_limited_mask
        else:
            mask = np.ones(len(features), dtype=bool)

        x = np.array([getattr(f, feat_name) for f in features], dtype=float)
        valid = mask & ~np.isnan(x)
        if valid.sum() < 5:
            results.append(ExploratoryReport(
                feature=feat_name, partial_spearman=float("nan"),
                partial_spearman_ci=(float("nan"), float("nan")),
                p_value_raw=float("nan"), p_value_bh=float("nan"),
                n=int(valid.sum()), label=label,
            ))
            continue

        xv, yv, zv = x[valid], y_all[valid], z_all[valid]
        cv = [c for c, m in zip(concepts, valid) if m]
        rho = _partial_spearman(xv, yv, zv)
        ci = _cluster_bootstrap_partial_spearman(xv, yv, zv, cv)

        # Two-sided p-value from the bootstrap (fraction of bootstrap rhos
        # crossing zero).
        rng = np.random.default_rng(42)
        n_boot = 2000
        unique_concepts = sorted(set(cv))
        concept_arr = np.array(cv)
        boot_vals = []
        for _ in range(n_boot):
            sampled = rng.choice(unique_concepts, size=len(unique_concepts), replace=True)
            idx = np.concatenate([np.where(concept_arr == c)[0] for c in sampled])
            if len(idx) < 4:
                continue
            boot_vals.append(_partial_spearman(xv[idx], yv[idx], zv[idx]))
        if boot_vals:
            boot_arr = np.array(boot_vals)
            p_raw = float(np.mean(np.sign(boot_arr) != np.sign(rho)) * 2)
            p_raw = min(p_raw, 1.0)
        else:
            p_raw = float("nan")

        results.append(ExploratoryReport(
            feature=feat_name, partial_spearman=rho,
            partial_spearman_ci=ci, p_value_raw=p_raw,
            p_value_bh=float("nan"),  # filled below
            n=int(valid.sum()), label=label,
        ))

    # Benjamini-Hochberg correction across the exploratory family.
    raw_ps = [r.p_value_raw for r in results if not np.isnan(r.p_value_raw)]
    if raw_ps:
        m = len(raw_ps)
        sorted_ps = sorted(enumerate(raw_ps), key=lambda x: x[1])
        bh = [0.0] * m
        for rank, (orig_idx, p) in enumerate(sorted_ps, 1):
            bh[orig_idx] = min(p * m / rank, 1.0)
        # Enforce monotonicity.
        for j in range(m - 2, -1, -1):
            bh[j] = min(bh[j], bh[j + 1]) if j + 1 < m else bh[j]
        pi = 0
        for r in results:
            if not np.isnan(r.p_value_raw):
                r.p_value_bh = bh[pi]
                pi += 1

    return results


def fit_gap_predictor(
    features: list[GeometryFeatures],
    gaps: list[float],
    feature_names: list[str] | None = None,
    controllabilities: list[float] | None = None,
    readabilities: list[float] | None = None,
    fluency_limited_mask: np.ndarray | None = None,
) -> PredictorReport:
    """Ridge gap-predictor, plus the preregistered primary and exploratory tests.

    Pass `controllabilities` and `readabilities` to get the real Experiment 4
    analysis: `primary_test` (H1, partial Spearman of output overlap against
    controllability controlling for readability, with the P7 cluster bootstrap
    CI) and `exploratory_analysis` (E1 and E2 under BH correction, P8).

    Without them only the descriptive ridge fit can be computed, and the
    returned `primary` says so rather than dressing the ridge R^2 up as the
    preregistered result. That is what this function used to do: it reported the
    correlation between the gap and the ridge model's leave-one-out predictions
    as `partial_spearman`, hardcoded the CI to (nan, nan) so the cluster
    bootstrap never ran, copied the R^2 verdict into the primary verdict, and
    returned an empty exploratory list -- so a pipeline that only called this
    function never ran the study's one preregistered hypothesis at all.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr

    # Extract readabilities and controllabilities from gaps and features.
    # Gaps are readability - controllability (on normalized scale), but for
    # the primary test we need raw values.  Since this is called from
    # multiple contexts, we use the gap itself and output_overlap.
    names = feature_names or [PRIMARY_FEATURE] + EXPLORATORY_FEATURES
    rows = []
    for f in features:
        d = f.to_dict()
        rows.append([d.get(n, float("nan")) for n in names])
    X = np.array(rows, dtype=float)
    y = np.array(gaps, dtype=float)

    # Drop all-NaN columns.
    keep = [i for i in range(X.shape[1]) if not np.all(np.isnan(X[:, i]))]
    X, names = X[:, keep], [names[i] for i in keep]
    # Impute remaining NaNs with the column mean.
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    if len(y) < 5:
        raise ValueError(f"need at least 5 points to fit a predictor, got {len(y)}")

    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    pred = cross_val_predict(pipe, X, y, cv=LeaveOneOut())
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rho = (
        float(spearmanr(y, pred)[0])
        if len(y) > 2 and np.std(y) > 0 and np.std(pred) > 0
        else float("nan")
    )

    pipe.fit(X, y)
    coefs = dict(zip(names, pipe[-1].coef_.tolist()))

    if r2 > 0.3:
        verdict = "geometry predicts the gap"
    elif r2 > 0.0:
        verdict = "weak geometric signal; report with caution"
    else:
        verdict = (
            "no geometric predictor: the gap is real and not explained by "
            "these standard quantities"
        )

    concepts = [f.concept for f in features]

    if controllabilities is not None and readabilities is not None:
        primary = primary_test(features, controllabilities, readabilities, concepts)
        exploratory = exploratory_analysis(
            features, controllabilities, readabilities, concepts,
            fluency_limited_mask=fluency_limited_mask,
        )
    else:
        primary = PrimaryTestReport(
            feature="output_overlap",
            partial_spearman=float("nan"),
            partial_spearman_ci=(float("nan"), float("nan")),
            n_concepts=len(set(concepts)),
            n_points=len(y),
            verdict=(
                "primary test NOT RUN: fit_gap_predictor was called without "
                "controllabilities and readabilities, so the partial "
                "correlation and its cluster bootstrap could not be computed. "
                f"Descriptive ridge fit only: LOO R^2={r2:.3f} ({verdict})."
            ),
        )
        exploratory = []

    return PredictorReport(
        primary=primary,
        exploratory=exploratory,
        features_used=names,
        r2_loo=float(r2),
        spearman=rho,
        coefficients=coefs,
        n=len(y),
        verdict=verdict,
    )

