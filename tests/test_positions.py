"""Position-limited injection.

This exists because the study's default, adding at every position, is not what
activation addition does, and replicating a published result with the wrong
convention produced a null that said nothing about the published result. The
tests pin both behaviours: the default must be unchanged, because every number
the paper reports was produced with it, and the limited form must touch only
the positions it claims to.
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


def _hidden(batch=2, seq=5):
    return torch.zeros(batch, seq, D)


def test_default_touches_every_position():
    """Unchanged from before the knob existed: the reported numbers depend on it."""
    spec = SteeringSpec(direction=_dir(), layers=[0], variant="add", coeff=1.0)
    out = _apply(_hidden(), spec, unit=1.0)
    assert torch.allclose(out[:, :, 0], torch.ones(2, 5))


def test_limited_touches_only_the_leading_positions():
    spec = SteeringSpec(direction=_dir(), layers=[0], variant="add", coeff=1.0,
                        positions=2)
    out = _apply(_hidden(), spec, unit=1.0)
    assert torch.allclose(out[:, :2, 0], torch.ones(2, 2))
    assert torch.allclose(out[:, 2:, 0], torch.zeros(2, 3))


def test_limit_beyond_the_sequence_is_harmless():
    """During generation each new token arrives in a tensor of length one."""
    spec = SteeringSpec(direction=_dir(), layers=[0], variant="add", coeff=1.0,
                        positions=99)
    out = _apply(_hidden(batch=1, seq=1), spec, unit=1.0)
    assert torch.allclose(out[:, :, 0], torch.ones(1, 1))


def test_other_dimensions_are_untouched():
    spec = SteeringSpec(direction=_dir(), layers=[0], variant="add", coeff=1.0,
                        positions=2)
    out = _apply(_hidden(), spec, unit=1.0)
    assert torch.allclose(out[:, :, 1:], torch.zeros(2, 5, D - 1))


def test_ablate_respects_positions():
    spec = SteeringSpec(direction=_dir(), layers=[0], variant="ablate",
                        positions=2)
    h = torch.ones(1, 4, D)
    out = _apply(h, spec, unit=1.0)
    assert torch.allclose(out[:, :2, 0], torch.zeros(1, 2))   # projection removed
    assert torch.allclose(out[:, 2:, 0], torch.ones(1, 2))    # left alone


def test_clamp_respects_positions():
    spec = SteeringSpec(direction=_dir(), layers=[0], variant="clamp",
                        clamp_target=3.0, positions=1)
    h = torch.zeros(1, 3, D)
    out = _apply(h, spec, unit=1.0)
    assert pytest.approx(float(out[0, 0, 0]), abs=1e-5) == 3.0
    assert torch.allclose(out[:, 1:, 0], torch.zeros(1, 2))


def test_input_is_not_mutated_in_place():
    """A hook that edited the model's own tensor would corrupt the unsteered
    baseline the RMS unit is measured from."""
    spec = SteeringSpec(direction=_dir(), layers=[0], variant="add", coeff=1.0,
                        positions=2)
    h = _hidden()
    before = h.clone()
    _apply(h, spec, unit=1.0)
    assert torch.allclose(h, before)


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_a_nonsense_limit(bad):
    with pytest.raises(ValueError, match="positions"):
        SteeringSpec(direction=_dir(), layers=[0], variant="add", coeff=1.0,
                     positions=bad)
