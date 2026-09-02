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

    direction: np.ndarray       # unit-norm, shape (d_model,)
    layers: list[int]
    variant: str = "add"
    coeff: float = 0.0          # in RMS units; sign gives push direction
    clamp_target: float | None = None  # for variant="clamp", in RMS units

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")
        if self.variant == "clamp" and self.clamp_target is None:
            raise ValueError("variant='clamp' requires clamp_target")
        n = float(np.linalg.norm(self.direction))
        if not np.isclose(n, 1.0, atol=1e-3):
            raise ValueError(f"direction must be unit norm, got norm {n:.4f}")


def _rms(hidden) -> float:
    """Mean RMS norm per token position, used as the strength unit."""
    import torch

    with torch.no_grad():
        return float(hidden.float().pow(2).mean(dim=-1).sqrt().mean())


def _apply(hidden, spec: SteeringSpec, unit: float):
    """Apply the intervention to a (B, T, D) hidden-state tensor."""
    import torch

    d = torch.tensor(
        spec.direction, dtype=hidden.dtype, device=hidden.device
    )

    if spec.variant in ("add", "add_all"):
        return hidden + spec.coeff * unit * d

    proj = (hidden.float() @ d.float()).unsqueeze(-1)  # (B, T, 1)
    if spec.variant == "ablate":
        return (hidden.float() - proj * d.float()).to(hidden.dtype)

    # clamp: force the projection to a fixed value in RMS units
    target = spec.clamp_target * unit
    return (hidden.float() + (target - proj) * d.float()).to(hidden.dtype)


@contextmanager
def steering_hooks(lm, spec: SteeringSpec):
    """Install forward hooks implementing `spec`; removes them on exit.

    The RMS unit is measured from the *unsteered* activations of the first
    forward pass through each layer, so the strength scale does not drift as
    steering pushes the norm around.
    """
    handles = []
    units: dict[int, float] = {}

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            if layer_idx not in units:
                units[layer_idx] = _rms(hidden)
            new_hidden = _apply(hidden, spec, units[layer_idx])
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
