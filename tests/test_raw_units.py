"""Coefficient units, and the decoder the replication needs.

This exists for the same reason `test_positions.py` does. The study's
coefficient is in residual-RMS units on a unit-normalised direction; activation
addition as published adds a raw activation difference at coefficient 1.0. The
two are different interventions by whatever ratio those scales stand in, which
for the wedding contrast at layer 6 of gpt2-xl is a factor of about 175. A
sweep of +-20 in RMS units is therefore not a wide bracket around the published
setting, it is a sweep not shown to contain it, and the run it produced was a
flat zero readout that looked like a finding.

The tests pin the RMS default, because every number the paper reports uses it,
and pin `raw_norm` to mean exactly what the published method does.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lbi.steering import SteeringSpec, _apply  # noqa: E402

D = 8


def _dir():
    v = np.zeros(D, dtype=np.float32)
    v[0] = 1.0
    return v


def _hidden(batch=2, seq=4):
    return torch.zeros(batch, seq, D)


# --------------------------------------------------------------------------
# The default is untouched
# --------------------------------------------------------------------------


def test_rms_is_the_default_and_still_scales_by_the_unit():
    """Every reported number depends on this, so it is pinned separately."""
    spec = SteeringSpec(direction=_dir(), layers=[0], coeff=2.0)
    assert spec.unit_mode == "rms"
    out = _apply(_hidden(), spec, unit=3.0)
    assert torch.allclose(out[:, :, 0], torch.full((2, 4), 6.0))


def test_raw_scale_is_rejected_without_the_mode_that_uses_it():
    """Otherwise it would sit in the spec looking effective and do nothing."""
    with pytest.raises(ValueError, match="only meaningful"):
        SteeringSpec(direction=_dir(), layers=[0], coeff=1.0, raw_scale=175.0)


# --------------------------------------------------------------------------
# raw_norm reproduces the published intervention exactly
# --------------------------------------------------------------------------


def test_raw_norm_ignores_the_rms_unit():
    """The whole point: the layer's norm must not enter the strength."""
    spec = SteeringSpec(direction=_dir(), layers=[0], coeff=1.0,
                        unit_mode="raw_norm", raw_scale=175.0)
    for unit in (1.0, 3.0, 100.0):
        out = _apply(_hidden(), spec, unit=unit)
        assert torch.allclose(out[:, :, 0], torch.full((2, 4), 175.0))


def test_coeff_one_adds_back_the_original_raw_vector():
    """coeff=1.0 in raw_norm units is the published setting, not an estimate.

    Decomposing a raw difference into a unit direction and its norm and then
    adding it at coefficient 1.0 has to return the vector we started with.
    """
    rng = np.random.default_rng(0)
    raw = rng.normal(size=D).astype(np.float32) * 12.0
    raw_norm = float(np.linalg.norm(raw))
    spec = SteeringSpec(direction=raw / raw_norm, layers=[0], coeff=1.0,
                        unit_mode="raw_norm", raw_scale=raw_norm)
    out = _apply(_hidden(batch=1, seq=1), spec, unit=7.0)
    assert np.allclose(out[0, 0].numpy(), raw, atol=1e-4)


def test_raw_norm_requires_a_scale():
    with pytest.raises(ValueError, match="requires raw_scale"):
        SteeringSpec(direction=_dir(), layers=[0], coeff=1.0,
                     unit_mode="raw_norm")


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_raw_scale_must_be_positive(bad):
    with pytest.raises(ValueError, match="raw_scale must be"):
        SteeringSpec(direction=_dir(), layers=[0], coeff=1.0,
                     unit_mode="raw_norm", raw_scale=bad)


def test_unknown_unit_mode_is_rejected():
    with pytest.raises(ValueError, match="unit_mode must be"):
        SteeringSpec(direction=_dir(), layers=[0], coeff=1.0,
                     unit_mode="raw")


@pytest.mark.parametrize("variant", ["clamp", "ablate"])
def test_raw_norm_is_refused_where_it_would_mean_something_else(variant):
    """clamp_target and the ablation projection are both defined in RMS units.

    Accepting a raw scale there would not raise, it would quietly measure the
    projection against the wrong ruler.
    """
    with pytest.raises(ValueError, match="only applies to variant"):
        SteeringSpec(direction=_dir(), layers=[0], coeff=1.0, variant=variant,
                     clamp_target=1.0, unit_mode="raw_norm", raw_scale=175.0)


def test_raw_norm_composes_with_position_limiting():
    """The replication needs both at once, so the combination is pinned."""
    spec = SteeringSpec(direction=_dir(), layers=[0], coeff=1.0,
                        unit_mode="raw_norm", raw_scale=175.0, positions=2)
    out = _apply(_hidden(), spec, unit=1.0)
    assert torch.allclose(out[:, :2, 0], torch.full((2, 2), 175.0))
    assert torch.allclose(out[:, 2:, 0], torch.zeros(2, 2))


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def test_the_published_grid_contains_their_coefficient():
    """A bracket that misses the point it brackets is the bug being fixed."""
    from lbi.published import ACTADD_SETTING, PUBLISHED_COEFF_MULTIPLES

    assert ACTADD_SETTING["coefficient"] == 1.0
    assert 1.0 in PUBLISHED_COEFF_MULTIPLES
    assert 0.0 in PUBLISHED_COEFF_MULTIPLES  # the baseline point
    assert PUBLISHED_COEFF_MULTIPLES == sorted(PUBLISHED_COEFF_MULTIPLES)


def test_published_decoding_samples():
    """Greedy here produced a null about the decoder; pin that it is not greedy."""
    from lbi.published import ACTADD_DECODING

    assert ACTADD_DECODING["temperature"] > 0
    assert ACTADD_DECODING["n_samples"] > 1


def test_many_samples_under_greedy_decoding_is_refused():
    """It would average n identical draws and report a falsely tight interval.

    The guard sits above anything that touches the model, so None gets there.
    """
    from lbi.pipeline import run_steering

    with pytest.raises(ValueError, match="repeat one draw"):
        run_steering(None, None, _dir(), 0, None, n_samples=5, temperature=0.0)

    with pytest.raises(ValueError, match="n_samples must be"):
        run_steering(None, None, _dir(), 0, None, n_samples=0, temperature=1.0)


def test_greedy_single_sample_remains_the_default_path():
    """Every reported number comes from it, so a change here must be deliberate."""
    import inspect

    from lbi.pipeline import run_steering

    sig = inspect.signature(run_steering)
    assert sig.parameters["temperature"].default == 0.0
    assert sig.parameters["n_samples"].default == 1
    assert sig.parameters["unit_mode"].default == "rms"
    assert sig.parameters["raw_scale"].default is None
