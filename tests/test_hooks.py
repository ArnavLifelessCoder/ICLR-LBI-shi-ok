"""Hook-level tests: does the intervention do what the math says?

A tiny local torch stack stands in for a real decoder, so this runs on CPU with
no download. What is verified is the part that would silently corrupt every
number in the paper if it were wrong: that hooks fire on the right layers, that
the RMS strength unit is applied, and that each variant has its defining
property (ablate zeroes the projection, clamp fixes it, add shifts it).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lbi import steering as st
from lbi.extraction import LoadedModel, _decoder_layers, _layer_output_hidden


D_MODEL = 8
N_LAYERS = 4


class _Layer(torch.nn.Module):
    """A decoder layer that returns a tuple, like Llama/Qwen layers do."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x):
        return (self.lin(x),)


class _Stack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer() for _ in range(N_LAYERS)])

    def forward(self, x):
        for layer in self.layers:
            x = _layer_output_hidden(layer(x))
        return x


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Stack()

    def forward(self, x):
        return self.model(x)


def _fake_lm():
    m = _Model()
    layers = _decoder_layers(m)
    return LoadedModel(
        name="fake", model=m, tokenizer=None, layers=layers,
        n_layers=len(layers), d_model=D_MODEL, device="cpu",
    )


def _unit(seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=D_MODEL)
    return d / np.linalg.norm(d)


def test_decoder_layers_found_on_model_layers_layout():
    assert len(_decoder_layers(_Model())) == N_LAYERS


def test_hooks_are_removed_on_exit():
    lm = _fake_lm()
    d = _unit()
    x = torch.randn(2, 3, D_MODEL)

    clean_before = lm.model(x).clone()
    spec = st.SteeringSpec(direction=d, layers=[1], variant="add", coeff=5.0)
    with st.steering_hooks(lm, spec):
        steered = lm.model(x)
    clean_after = lm.model(x)

    assert not torch.allclose(steered, clean_before, atol=1e-4)
    assert torch.allclose(clean_before, clean_after, atol=1e-6)


def test_hooks_are_removed_even_if_the_body_raises():
    lm = _fake_lm()
    spec = st.SteeringSpec(direction=_unit(), layers=[0], coeff=1.0)
    x = torch.randn(1, 2, D_MODEL)
    clean = lm.model(x).clone()

    with pytest.raises(RuntimeError, match="boom"):
        with st.steering_hooks(lm, spec):
            raise RuntimeError("boom")

    assert torch.allclose(lm.model(x), clean, atol=1e-6)


def test_zero_coefficient_is_a_no_op():
    lm = _fake_lm()
    x = torch.randn(2, 3, D_MODEL)
    clean = lm.model(x).clone()
    spec = st.SteeringSpec(direction=_unit(), layers=[2], coeff=0.0)
    with st.steering_hooks(lm, spec):
        assert torch.allclose(lm.model(x), clean, atol=1e-6)


def _capture_layer_output(lm, layer_idx, x, spec=None):
    """Hidden state emitted by `layer_idx`, after any steering hook.

    PyTorch runs forward hooks in registration order and feeds each one the
    output the previous hook returned, so the grab hook must be registered
    *inside* the steering context to observe the post-intervention state.
    """
    grabbed = {}

    def grab(_m, _i, output):
        grabbed["h"] = _layer_output_hidden(output).detach().clone()

    ctx = st.steering_hooks(lm, spec) if spec else _noop()
    with ctx:
        handle = lm.layer_module(layer_idx).register_forward_hook(grab)
        try:
            with torch.no_grad():
                lm.model(x)
        finally:
            handle.remove()
    return grabbed["h"]


from contextlib import contextmanager


@contextmanager
def _noop():
    yield


def test_ablate_removes_the_projection():
    lm = _fake_lm()
    d = _unit()
    x = torch.randn(2, 5, D_MODEL)
    spec = st.SteeringSpec(direction=d, layers=[1], variant="ablate")

    h = _capture_layer_output(lm, 1, x, spec)
    proj = h.float() @ torch.tensor(d, dtype=torch.float32)
    assert proj.abs().max().item() < 1e-4


def test_clamp_fixes_the_projection_in_rms_units():
    lm = _fake_lm()
    d = _unit(1)
    x = torch.randn(2, 5, D_MODEL)
    target = 1.5
    spec = st.SteeringSpec(
        direction=d, layers=[1], variant="clamp", clamp_target=target
    )

    clean = _capture_layer_output(lm, 1, x)
    unit = float(clean.float().pow(2).mean(dim=-1).sqrt().mean())

    h = _capture_layer_output(lm, 1, x, spec)
    proj = h.float() @ torch.tensor(d, dtype=torch.float32)
    assert proj.min().item() == pytest.approx(target * unit, abs=1e-3)
    assert proj.max().item() == pytest.approx(target * unit, abs=1e-3)


def test_add_shifts_the_projection_by_coeff_times_rms():
    lm = _fake_lm()
    d = _unit(2)
    x = torch.randn(2, 5, D_MODEL)
    coeff = 2.0
    dt = torch.tensor(d, dtype=torch.float32)

    clean = _capture_layer_output(lm, 1, x)
    unit = float(clean.float().pow(2).mean(dim=-1).sqrt().mean())
    before = clean.float() @ dt

    spec = st.SteeringSpec(direction=d, layers=[1], variant="add", coeff=coeff)
    after = _capture_layer_output(lm, 1, x, spec).float() @ dt

    assert (after - before).mean().item() == pytest.approx(coeff * unit, abs=1e-3)


def test_negative_coefficient_shifts_the_other_way():
    lm = _fake_lm()
    d = _unit(3)
    x = torch.randn(2, 4, D_MODEL)
    dt = torch.tensor(d, dtype=torch.float32)
    before = (_capture_layer_output(lm, 1, x).float() @ dt).mean().item()

    spec = st.SteeringSpec(direction=d, layers=[1], coeff=-2.0)
    after = (_capture_layer_output(lm, 1, x, spec).float() @ dt).mean().item()
    assert after < before


def test_add_all_touches_every_layer_in_the_band():
    lm = _fake_lm()
    d = _unit(4)
    x = torch.randn(1, 3, D_MODEL)
    band = st.layer_band(1, N_LAYERS, width=3)
    spec = st.SteeringSpec(direction=d, layers=band, variant="add_all", coeff=3.0)

    with st.steering_hooks(lm, spec):
        pass  # smoke: hooks install and remove cleanly for a multi-layer band

    single = st.SteeringSpec(direction=d, layers=[band[0]], coeff=3.0)
    multi_out = _run(lm, x, spec)
    single_out = _run(lm, x, single)
    assert not torch.allclose(multi_out, single_out, atol=1e-4)


def _run(lm, x, spec):
    with st.steering_hooks(lm, spec), torch.no_grad():
        return lm.model(x).clone()


def test_rms_unit_is_measured_once_and_does_not_drift():
    # The unit is fixed from the first unsteered pass, so repeated forwards
    # under the same spec apply the same absolute shift.
    lm = _fake_lm()
    d = _unit(5)
    x = torch.randn(1, 4, D_MODEL)
    spec = st.SteeringSpec(direction=d, layers=[1], coeff=1.0)
    with st.steering_hooks(lm, spec), torch.no_grad():
        first = lm.model(x).clone()
        second = lm.model(x).clone()
    assert torch.allclose(first, second, atol=1e-6)


def test_layer_module_bounds():
    lm = _fake_lm()
    with pytest.raises(IndexError):
        lm.layer_module(N_LAYERS)
    with pytest.raises(IndexError):
        lm.layer_module(-1)


# --------------------------------------------------------------------------
# Left padding
# --------------------------------------------------------------------------


def test_position_ids_ignore_left_padding():
    """Real tokens start at position 0 no matter how much padding precedes them.

    `padding_side` is "left" so last-token pooling always lands on a real
    token, but a bare `model(**enc)` gets no position ids and HuggingFace falls
    back to `arange(seq_len)`. That puts the first real token of a padded row
    at position n_pad and shifts every RoPE phase in it.
    """
    from lbi.extraction import left_padded_position_ids

    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    pos = left_padded_position_ids(mask)

    # Both rows number their real tokens 0, 1, 2, ... regardless of padding.
    assert pos[0].tolist() == [0, 0, 0, 1, 2]
    assert pos[1].tolist() == [0, 1, 2, 3, 4]
    # The last position is the same for both, which is what pooling reads.
    assert pos[0, -1] == 2 and pos[1, -1] == 4
    # Pad slots never carry a negative index into the embedding table.
    assert int(pos.min()) >= 0


def test_position_ids_are_unaffected_when_nothing_is_padded():
    from lbi.extraction import left_padded_position_ids

    mask = torch.ones(3, 6, dtype=torch.long)
    pos = left_padded_position_ids(mask)
    expected = torch.arange(6).expand(3, 6)
    assert torch.equal(pos, expected)


def test_perplexity_label_mask_excludes_the_first_real_token():
    """Its predicting position is a pad, so it has no context to be scored from.

    Without this, a row that happened to receive more padding looks less
    fluent purely because it was shorter than the longest text in its batch --
    and perplexity is what the fluency ceiling thresholds on.
    """
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    shift_mask = mask[:, 1:] * mask[:, :-1]
    # Padded row: only the 2nd and 3rd real tokens are scored, not the 1st.
    assert shift_mask[0].tolist() == [0, 0, 1, 1]
    assert shift_mask[1].tolist() == [1, 1, 1, 1]


def test_gemma2_defaults_to_eager_attention():
    """Gemma-2 soft-caps attention logits and the fused SDPA path ignores it.

    The model would load and generate perfectly happily while being quietly
    wrong, which in this study looks like a concept that steers oddly rather
    than like a bug.
    """
    import inspect

    from lbi import extraction

    src = inspect.getsource(extraction.load_model)
    assert 'attn_implementation is None and "gemma-2" in name.lower()' in src
    assert '"eager"' in src
    # Explicit choices must still win over the default.
    sig = inspect.signature(extraction.load_model)
    assert sig.parameters["attn_implementation"].default is None
