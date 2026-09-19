"""Experiment 2: the controllability axis.

Steering strength is normalized in units of the layer's residual-stream RMS norm
(revision R2), so a coefficient means the same thing across layers and models.
Controllability is the area under the dose-response curve up to the fluency
ceiling, not a single-point effect.

All four robustness variants from the design doc's Experiment 2 fallback list
are here, because a "readable but immovable" claim is only defensible if the
concept resists every one of them:

  * ``add``      single-layer activation addition (default)
  * ``add_all``  addition across a band of layers
  * ``clamp``    set the projection onto the direction to a target value
  * ``ablate``   remove the direction from the residual stream entirely

and the direction itself can be difference-of-means or probe weights.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np


VARIANTS = ("add", "add_all", "clamp", "ablate")

# `np.trapz` was deprecated in NumPy 1.x and removed in 2.0, where it is
# spelled `np.trapezoid`. Both spellings are in the wild -- Colab pins whatever
# it pins -- and the dose-response AUC *is* the controllability axis, so this
# resolving to the wrong thing does not degrade a number, it raises
# AttributeError halfway through a GPU session.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class SteeringSpec:
    """One intervention configuration."""

    # Unit-norm. Shape (d_model,) for one vector applied at every position the
    # intervention covers, or (P, d_model) for a position-wise matrix whose row
    # i is applied at absolute position i, with each row unit-norm.
    #
    # The matrix form exists because activation addition is position-wise. Its
    # contrast prompts are tokenised, padded to a common length, and
    # differenced at each position, so the vector added at position 0 is not
    # the one added at position 2. Collapsing that to a single pooled vector
    # and adding it at every covered position is a different intervention, and
    # the replication that did so failed its positive control.
    direction: np.ndarray
    layers: list[int]
    variant: str = "add"
    coeff: float = 0.0          # in units of `unit_mode`; sign gives direction
    clamp_target: float | None = None  # for variant="clamp", in RMS units
    # What one unit of `coeff` means.
    #
    # "rms" scales by the layer's residual-stream RMS norm, measured from the
    # unsteered activations. Every number this study reports uses it, and it is
    # what makes a coefficient comparable across layers and models.
    #
    # "raw_norm" scales by `raw_scale` instead, so coeff=1.0 adds exactly
    # `raw_scale * direction` regardless of the layer's norm. It exists for
    # replication: activation addition adds a raw activation difference at
    # coefficient 1.0, and an RMS coefficient of 1.0 is a different
    # intervention by whatever ratio the two scales happen to stand in. With
    # `raw_scale` set to that difference's norm, coeff=1.0 is the published
    # setting exactly, and the grid reads in the published method's own units.
    unit_mode: str = "rms"
    # Scalar, or one scale per row when `direction` is a matrix: each position
    # of a raw activation difference has its own norm, and forcing them to a
    # common one would rescale the published intervention.
    raw_scale: float | np.ndarray | None = None
    # How many leading token positions to intervene on. None means every
    # position, which is this study's default and what all its reported
    # numbers use.
    #
    # Activation addition as published injects only at the positions its
    # contrast prompt occupied, so the vector shapes the start of the
    # continuation and then stops. Adding at every position, including the
    # tokens being generated, is a different and much stronger intervention:
    # replicating the published wedding demonstration that way produced no
    # wedding vocabulary at any usable coefficient and degenerated into
    # repetition beyond them. Faithfulness to a published method needs this
    # knob, so it exists and defaults to the old behaviour.
    positions: int | None = None

    @property
    def is_matrix(self) -> bool:
        return np.asarray(self.direction).ndim == 2

    @property
    def n_direction_rows(self) -> int:
        """Positions the direction itself covers; 1 for the vector form."""
        return int(np.asarray(self.direction).shape[0]) if self.is_matrix else 1

    def __post_init__(self):
        if self.positions is not None and self.positions < 1:
            raise ValueError(
                f"positions must be >= 1 or None, got {self.positions}"
            )
        arr = np.asarray(self.direction)
        if arr.ndim not in (1, 2):
            raise ValueError(
                f"direction must be 1-D or 2-D, got shape {arr.shape}"
            )
        if arr.ndim == 2:
            if arr.shape[0] < 1:
                raise ValueError("a direction matrix needs at least one row")
            # A matrix already says which positions it covers, so a separate
            # count could only agree or contradict.
            if self.positions is not None and self.positions != arr.shape[0]:
                raise ValueError(
                    "positions=%d contradicts a direction matrix with %d rows; "
                    "leave positions as None and let the matrix say"
                    % (self.positions, arr.shape[0])
                )
            if self.variant not in ("add", "add_all"):
                raise ValueError(
                    "a direction matrix only applies to variant 'add' or "
                    f"'add_all', got {self.variant!r}"
                )
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")
        if self.variant == "clamp" and self.clamp_target is None:
            raise ValueError("variant='clamp' requires clamp_target")
        if self.unit_mode not in ("rms", "raw_norm"):
            raise ValueError(
                f"unit_mode must be 'rms' or 'raw_norm', got {self.unit_mode!r}"
            )
        if self.unit_mode == "raw_norm":
            if self.raw_scale is None:
                raise ValueError("unit_mode='raw_norm' requires raw_scale")
            scale = np.asarray(self.raw_scale, dtype=float)
            if scale.ndim not in (0, 1):
                raise ValueError(
                    f"raw_scale must be a scalar or 1-D, got shape {scale.shape}"
                )
            if scale.ndim == 1 and scale.shape[0] != self.n_direction_rows:
                raise ValueError(
                    "raw_scale has %d entries but the direction has %d row(s)"
                    % (scale.shape[0], self.n_direction_rows)
                )
            if not np.all(scale > 0):
                raise ValueError(f"raw_scale must be > 0, got {self.raw_scale}")
            # clamp_target and the ablation projection are both defined in RMS
            # units, so a raw scale would silently mean something else there.
            if self.variant not in ("add", "add_all"):
                raise ValueError(
                    "unit_mode='raw_norm' only applies to variant 'add' or "
                    f"'add_all', got {self.variant!r}"
                )
        elif self.raw_scale is not None:
            raise ValueError("raw_scale is only meaningful with unit_mode='raw_norm'")
        norms = np.linalg.norm(arr, axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            if arr.ndim == 1:
                raise ValueError(
                    f"direction must be unit norm, got norm {float(norms):.4f}"
                )
            bad = int(np.argmax(np.abs(norms - 1.0)))
            raise ValueError(
                "every direction row must be unit norm; row %d has norm %.4f"
                % (bad, float(norms[bad]))
            )


def _rms(hidden) -> float:
    """Mean RMS norm per token position, used as the strength unit."""
    import torch

    with torch.no_grad():
        return float(hidden.float().pow(2).mean(dim=-1).sqrt().mean())


def _apply(hidden, spec: SteeringSpec, unit: float, offset: int = 0):
    """Apply the intervention to a (B, T, D) hidden-state tensor.

    `offset` is the absolute position of column 0 of `hidden` in the sequence
    so far. It matters because of the KV cache: the prompt arrives in one pass
    and each generated token then arrives as its own length-1 tensor, whose
    column 0 is not position 0. Deciding by column index alone would steer
    every generated token, which is the opposite of confining the intervention
    to the prompt. The hook counts positions and passes the offset here.

    A position-limited or matrix intervention therefore touches absolute
    positions `[offset, offset + T)` intersected with the covered range, and a
    pass that lies entirely past that range is returned untouched.
    """
    import torch

    arr = np.asarray(spec.direction)
    d = torch.tensor(arr, dtype=hidden.dtype, device=hidden.device)

    if spec.variant in ("add", "add_all"):
        if spec.unit_mode == "rms":
            scale = torch.tensor(unit, dtype=hidden.dtype, device=hidden.device)
        else:
            scale = torch.tensor(
                np.asarray(spec.raw_scale, dtype=float),
                dtype=hidden.dtype, device=hidden.device,
            )
        if spec.is_matrix:
            # (P, D), row i for absolute position i.
            delta = spec.coeff * (scale.reshape(-1, 1) if scale.ndim else scale) * d
            return _add_at_positions(hidden, delta, offset, delta.shape[0])
        delta = spec.coeff * scale * d
        if spec.positions is None:
            return hidden + delta
        return _add_at_positions(hidden, delta, offset, spec.positions)

    proj = (hidden.float() @ d.float()).unsqueeze(-1)  # (B, T, 1)
    if spec.variant == "ablate":
        new = (hidden.float() - proj * d.float()).to(hidden.dtype)
    else:
        # clamp: force the projection to a fixed value in RMS units
        target = spec.clamp_target * unit
        new = (hidden.float() + (target - proj) * d.float()).to(hidden.dtype)

    if spec.positions is None:
        return new
    lo, hi = _covered_span(hidden.shape[1], offset, spec.positions)
    if lo >= hi:
        return hidden
    out = hidden.clone()
    out[:, lo:hi, :] = new[:, lo:hi, :]
    return out


def _covered_span(seq_len: int, offset: int, n_positions: int) -> tuple[int, int]:
    """Columns of this tensor lying inside absolute positions [0, n_positions)."""
    lo = max(0, -offset)
    hi = min(seq_len, n_positions - offset)
    return lo, max(lo, hi)


def _add_at_positions(hidden, delta, offset: int, n_positions: int):
    """Add `delta` over the covered span; `delta` is (D,) or (P, D)."""
    lo, hi = _covered_span(hidden.shape[1], offset, n_positions)
    if lo >= hi:
        return hidden
    out = hidden.clone()
    if delta.dim() == 1:
        out[:, lo:hi, :] = out[:, lo:hi, :] + delta
    else:
        # Row i of delta belongs to absolute position i.
        out[:, lo:hi, :] = out[:, lo:hi, :] + delta[offset + lo : offset + hi]
    return out


@contextmanager
def steering_hooks(lm, spec: SteeringSpec):
    """Install forward hooks implementing `spec`; removes them on exit.

    The RMS unit is measured from the *unsteered* activations of the first
    forward pass through each layer, so the strength scale does not drift as
    steering pushes the norm around.

    Each layer also carries a running count of the positions it has seen, so a
    position-limited or matrix intervention lands on absolute positions rather
    than on column indices. Under the KV cache a generated token arrives as a
    length-1 tensor whose column 0 is not position 0, so without the count the
    intervention would follow generation instead of staying on the prompt. The
    count resets with the context, which is entered once per batch, and the
    default `positions=None` path never consults it.
    """
    handles = []
    units: dict[int, float] = {}
    seen: dict[int, int] = {}

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            if layer_idx not in units:
                units[layer_idx] = _rms(hidden)
            offset = seen.get(layer_idx, 0)
            seen[layer_idx] = offset + int(hidden.shape[1])
            new_hidden = _apply(hidden, spec, units[layer_idx], offset=offset)
            if is_tuple:
                return (new_hidden,) + tuple(output[1:])
            return new_hidden

        return hook

    try:
        for li in spec.layers:
            handles.append(lm.layer_module(li).register_forward_hook(make_hook(li)))
        yield
    finally:
        for h in handles:
            h.remove()


def layer_band(best_layer: int, n_layers: int, width: int = 4, fraction: float = 0.2) -> list[int]:
    """A band of layers around the best layer, for the `add_all` variant.

    P4 protocol: the band spans best_layer ± fraction * n_layers, sampled
    at `width` evenly spaced layers.  The asymmetry between probe-layer
    selection (one layer) and steering-layer selection (best over band) is
    deliberate: it biases the study against its own headline finding.
    """
    half = max(1, int(n_layers * fraction))
    lo = max(0, best_layer - half)
    hi = min(n_layers - 1, best_layer + half)
    n_available = hi - lo + 1
    if n_available <= width:
        return list(range(lo, hi + 1))
    # Sample `width` evenly spaced layers within [lo, hi].
    indices = np.linspace(lo, hi, width).round().astype(int).tolist()
    return sorted(set(indices))


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def generate(
    lm,
    prompts: list[str],
    spec: SteeringSpec | None = None,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    batch_size: int = 4,
    seed: int = 0,
    use_chat_template: bool | None = None,
) -> list[str]:
    """Generate continuations, optionally under a steering intervention.

    Greedy by default: steering effects are the signal, sampling noise is not.

    Every model in the study is an *instruct* model and every eval prompt is an
    instruction, so the tokenizer's chat template is applied when it has one.
    Feeding a raw instruction to an instruct model is base-model prompting: it
    continues the text instead of following it, and the behaviour being scored
    is then not the behaviour the concept is about. `use_chat_template=False`
    forces the raw form for a base model or for a comparison.
    """
    import torch

    torch.manual_seed(seed)
    outputs: list[str] = []

    tok = lm.tokenizer
    if use_chat_template is None:
        use_chat_template = bool(getattr(tok, "chat_template", None))
    if use_chat_template:
        prompts = [
            tok.apply_chat_template(
                # Some concepts (refusal) already ship prompts in "User: ..."
                # transcript form; strip it so the template does not wrap a
                # role marker inside another role marker.
                [{"role": "user", "content": p[len("User: "):]
                  if p.startswith("User: ") else p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = lm.tokenizer(batch, return_tensors="pt", padding=True).to(lm.device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": lm.tokenizer.pad_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        ctx = steering_hooks(lm, spec) if spec is not None else _null_context()
        with ctx, torch.no_grad():
            out = lm.model.generate(**enc, **gen_kwargs)

        for i in range(len(batch)):
            new_tokens = out[i, enc["input_ids"].shape[1] :]
            outputs.append(
                lm.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            )

    return outputs


@contextmanager
def _null_context():
    yield


# --------------------------------------------------------------------------
# Dose-response sweep
# --------------------------------------------------------------------------


@dataclass
class DosePoint:
    coeff: float
    behavior: float          # mean behavior score in [0, 1]
    behavior_ci: tuple[float, float]
    perplexity: float        # under the unsteered model
    repetition: float        # degenerate-repetition score in [0, 1]
    broken: bool             # past the fluency ceiling
    samples: list[str] = field(default_factory=list)
    # One score per generation, in prompt order, so the point can be
    # recomputed on a subset of the prompts without re-running the model.
    #
    # This is what makes a prompt-count comparison an offline analysis. Stage
    # E measures the judge-scored concepts at thirty prompts where stage B used
    # six, and the honest comparison holds everything else fixed by taking the
    # first six of the same thirty: same run, same judge, same generations.
    # Without these, that comparison needs another GPU session and the two
    # runs still differ in more than the prompt count.
    #
    # With `n_samples > 1` the prompt list is tiled, so this has
    # `n_prompts * n_samples` entries and prompt i owns every index congruent
    # to i modulo `n_prompts`.
    scores: list[float] = field(default_factory=list)


@dataclass
class SteeringResult:
    """Controllability for one concept-model-variant."""

    concept: str
    model: str
    variant: str
    layers: list[int]
    direction_source: str
    curve: list[DosePoint]
    controllability: float           # dose-response AUC up to the ceiling
    controllability_ci: tuple[float, float]
    baseline_behavior: float
    max_usable_coeff: float
    ceiling_reason: str
    # True when the judge returned one distinct score across the whole sweep.
    # Controllability is then exactly 0.000 for a reason unrelated to steering,
    # and the point must not be allowed into the danger zone.
    judge_degenerate: bool = False

    def summary(self) -> dict:
        return {
            "concept": self.concept,
            "model": self.model,
            "variant": self.variant,
            "layers": ",".join(str(l) for l in self.layers),
            "direction_source": self.direction_source,
            "controllability": self.controllability,
            "controllability_ci_low": self.controllability_ci[0],
            "controllability_ci_high": self.controllability_ci[1],
            "baseline_behavior": self.baseline_behavior,
            "max_usable_coeff": self.max_usable_coeff,
            "ceiling_reason": self.ceiling_reason,
            "judge_degenerate": self.judge_degenerate,
        }


def dose_response_auc(curve: list[DosePoint], baseline: float) -> float:
    """Normalized area between the behavior curve and its baseline.

    Trapezoid over usable (unbroken) points only, divided by the coefficient
    span so the number is a mean absolute behavior shift in [0, 1] and is
    comparable across concepts whose ceilings differ.
    """
    usable = [p for p in curve if not p.broken]
    if len(usable) < 2:
        return 0.0
    xs = np.array([p.coeff for p in usable], dtype=float)
    ys = np.abs(np.array([p.behavior for p in usable], dtype=float) - baseline)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    span = xs[-1] - xs[0]
    if span <= 0:
        return 0.0
    return float(_trapezoid(ys, xs) / span)


def find_ceiling(
    curve: list[DosePoint],
    baseline_ppl: float,
    ppl_ratio: float = 2.0,
    repetition_max: float = 0.5,
) -> tuple[float, str]:
    """Pre-registered fluency ceiling (revision R3).

    Breakage is perplexity above `ppl_ratio` x the unsteered baseline, or a
    repetition score above `repetition_max`. Returns the largest usable
    coefficient magnitude and the reason the ceiling was hit.
    """
    reason = "no breakage in swept range"
    usable = 0.0
    for p in sorted(curve, key=lambda q: abs(q.coeff)):
        if p.perplexity > ppl_ratio * baseline_ppl:
            reason = f"perplexity > {ppl_ratio}x baseline at coeff {p.coeff:g}"
            break
        if p.repetition > repetition_max:
            reason = f"degenerate repetition at coeff {p.coeff:g}"
            break
        usable = max(usable, abs(p.coeff))
    return usable, reason


def mark_broken(curve: list[DosePoint], max_usable: float) -> None:
    """Mark by magnitude. Prefer `mark_broken_by_fluency` -- see its docstring."""
    for p in curve:
        p.broken = abs(p.coeff) > max_usable


def mark_broken_by_fluency(
    curve: list[DosePoint],
    baseline_ppl: float,
    ppl_ratio: float = 2.0,
    repetition_max: float = 0.5,
) -> tuple[float, str]:
    """Mark each point broken on its own fluency, then close over each sign.

    `find_ceiling` + `mark_broken` decided breakage by |coeff| against a single
    scalar, and that lets a point that breached the gate be counted as usable.
    On the first passing control, -3.0 had perplexity 9.59 (under the 12.96
    threshold) so it set max_usable = 3.0; +3.0 then measured 14.57, breached,
    and stopped the walk -- but `|coeff| > 3.0` marked nothing broken, so the
    breaching point went into the dose-response AUC anyway.

    Marking per point fixes that. The per-sign closure keeps the ceiling
    meaning what it says: once fluency has gone on one side, everything further
    out on that side is out, even if its own perplexity happens to read lower.
    Steering is not symmetric, so the two sides get their own ceilings.

    Returns (largest usable |coeff|, reason) for reporting.
    """
    threshold = ppl_ratio * baseline_ppl
    for p in curve:
        p.broken = bool(p.perplexity > threshold or p.repetition > repetition_max)

    for sign in (1, -1):
        side = sorted(
            (p for p in curve if p.coeff * sign > 0), key=lambda q: abs(q.coeff)
        )
        seen_break = False
        for p in side:
            if seen_break:
                p.broken = True
            elif p.broken:
                seen_break = True

    usable = max((abs(p.coeff) for p in curve if not p.broken), default=0.0)
    broken = [p for p in curve if p.broken]
    if not broken:
        return usable, "no breakage in swept range"
    first = min(broken, key=lambda q: abs(q.coeff))
    if first.repetition > repetition_max:
        reason = f"degenerate repetition at coeff {first.coeff:g}"
    else:
        reason = f"perplexity > {ppl_ratio}x baseline at coeff {first.coeff:g}"
    return usable, reason


def bootstrap_curve_ci(
    per_prompt_scores: dict[float, list[float]],
    baseline: float,
    n_boot: int = 500,
    seed: int = 0,
    max_usable: float | None = None,
    usable_coeffs: "set[float] | list[float] | None" = None,
) -> tuple[float, float]:
    """CI on controllability by resampling prompts (revision R8).

    `per_prompt_scores` maps coefficient -> per-prompt behavior scores; every
    coefficient must cover the same prompts in the same order.

    `usable_coeffs` is the exact set of unbroken coefficients and is what to
    pass: breakage is decided per point and per sign, so "everything with
    |coeff| below some scalar" is not the same set. `max_usable` is the older
    magnitude form, kept for callers that only have the scalar.
    Passing it is strongly preferred over pre-filtering at the call site: the
    point estimate and its interval have to be the same estimand, and when they
    were not, a curve that is flat below the ceiling and jumps above it reported
    controllability 0.025 with a CI of [0.1375, 0.1375] -- an interval five
    times the estimate, excluding it, and built entirely out of the degenerate
    text the ceiling exists to discard.
    """
    if usable_coeffs is not None:
        keep = {round(float(c), 6) for c in usable_coeffs}
        per_prompt_scores = {
            c: v for c, v in per_prompt_scores.items() if round(float(c), 6) in keep
        }
    elif max_usable is not None:
        per_prompt_scores = {
            c: v for c, v in per_prompt_scores.items() if abs(c) <= max_usable
        }
    coeffs = sorted(per_prompt_scores)
    if len(coeffs) < 2:
        return (0.0, 0.0)
    n_prompts = len(per_prompt_scores[coeffs[0]])
    rng = np.random.default_rng(seed)
    xs = np.array(coeffs, dtype=float)
    span = xs[-1] - xs[0]
    if span <= 0:
        return (0.0, 0.0)

    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_prompts, n_prompts)
        ys = np.array(
            [
                abs(float(np.mean([per_prompt_scores[c][i] for i in idx])) - baseline)
                for c in coeffs
            ]
        )
        vals.append(float(_trapezoid(ys, xs) / span))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
