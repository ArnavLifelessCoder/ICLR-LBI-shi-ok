"""Position-wise direction matrices, and absolute position under the KV cache.

Activation addition differences its two contrast prompts at each position and
adds row i at position i. Three attempts at replicating it used a pooled
last-token vector added at every covered position instead, which is a different
intervention, and the positive control is what eventually caught it.

The second thing here is a bug the matrix work exposed. `_apply` used to decide
which positions to touch from the column index of the incoming tensor. Under
the KV cache the prompt arrives in one pass and each generated token then
arrives as its own length-1 tensor whose column 0 is not position 0, so a
"first k positions" intervention was in fact steering every generated token as
well. Nothing the paper reports used that path, because `positions=None` is the
default and adds everywhere, but the replication did.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lbi.steering import SteeringSpec, _apply  # noqa: E402

D = 8


def _rows(p):
    """p unit-norm rows, each a distinct basis vector, so rows are told apart."""
    m = np.zeros((p, D), dtype=np.float32)
    for i in range(p):
        m[i, i] = 1.0
    return m


def _hidden(batch=2, seq=5):
    return torch.zeros(batch, seq, D)


# --------------------------------------------------------------------------
# The matrix lands row i at position i
# --------------------------------------------------------------------------


def test_each_row_lands_on_its_own_position():
    """The whole point of the matrix: position 0 must not receive row 2."""
    spec = SteeringSpec(direction=_rows(3), layers=[0], coeff=1.0)
    out = _apply(_hidden(), spec, unit=1.0)
    for i in range(3):
        assert out[0, i, i] == pytest.approx(1.0), f"position {i} missed row {i}"
        others = [j for j in range(D) if j != i]
        assert torch.allclose(out[0, i, others], torch.zeros(len(others)))


def test_positions_past_the_matrix_are_untouched():
    spec = SteeringSpec(direction=_rows(3), layers=[0], coeff=1.0)
    out = _apply(_hidden(seq=5), spec, unit=1.0)
    assert torch.allclose(out[:, 3:, :], torch.zeros(2, 2, D))


def test_a_pooled_vector_is_not_the_same_intervention():
    """Pin the difference the replication was blind to for three attempts."""
    mat = SteeringSpec(direction=_rows(3), layers=[0], coeff=1.0)
    pooled = SteeringSpec(direction=_rows(3)[-1], layers=[0], coeff=1.0,
                          positions=3)
    a = _apply(_hidden(), mat, unit=1.0)
    b = _apply(_hidden(), pooled, unit=1.0)
    assert not torch.allclose(a, b)


def test_per_position_scales_are_applied_per_position():
    """Each position of a raw difference has its own norm; a common one rescales."""
    scales = np.array([2.0, 5.0, 11.0])
    spec = SteeringSpec(direction=_rows(3), layers=[0], coeff=1.0,
                        unit_mode="raw_norm", raw_scale=scales)
    out = _apply(_hidden(), spec, unit=1.0)
    for i, s in enumerate(scales):
        assert out[0, i, i] == pytest.approx(s)


def test_matrix_rows_must_each_be_unit_norm():
    m = _rows(3)
    m[1] *= 4.0
    with pytest.raises(ValueError, match="row 1 has norm"):
        SteeringSpec(direction=m, layers=[0], coeff=1.0)


def test_scale_count_must_match_row_count():
    with pytest.raises(ValueError, match="2 entries but the direction has 3"):
        SteeringSpec(direction=_rows(3), layers=[0], coeff=1.0,
                     unit_mode="raw_norm", raw_scale=np.array([1.0, 2.0]))


def test_positions_may_not_contradict_the_matrix():
    with pytest.raises(ValueError, match="contradicts a direction matrix"):
        SteeringSpec(direction=_rows(3), layers=[0], coeff=1.0, positions=2)


@pytest.mark.parametrize("variant", ["clamp", "ablate"])
def test_matrix_refused_where_a_single_direction_is_assumed(variant):
    with pytest.raises(ValueError, match="only applies to variant"):
        SteeringSpec(direction=_rows(3), layers=[0], coeff=1.0,
                     variant=variant, clamp_target=1.0)


# --------------------------------------------------------------------------
# Absolute position, not column index
# --------------------------------------------------------------------------


def test_a_generated_token_is_not_steered_again():
    """The bug: under the KV cache every generated token has column index 0.

    The prompt pass covers positions 0..4; the next token is position 5,
    arriving as its own length-1 tensor. A "first 3 positions" intervention
    must not touch it.
    """
    spec = SteeringSpec(direction=_rows(1)[0], layers=[0], coeff=1.0,
                        positions=3)
    prompt = _apply(_hidden(seq=5), spec, unit=1.0, offset=0)
    assert torch.allclose(prompt[:, :3, 0], torch.ones(2, 3))
    assert torch.allclose(prompt[:, 3:, 0], torch.zeros(2, 2))

    nxt = _apply(_hidden(seq=1), spec, unit=1.0, offset=5)
    assert torch.allclose(nxt, torch.zeros(2, 1, D)), "generated token steered"


def test_matrix_also_respects_the_offset():
    spec = SteeringSpec(direction=_rows(3), layers=[0], coeff=1.0)
    nxt = _apply(_hidden(seq=1), spec, unit=1.0, offset=7)
    assert torch.allclose(nxt, torch.zeros(2, 1, D))


def test_a_pass_straddling_the_boundary_takes_the_right_rows():
    """A prompt split across passes must still get row i at position i."""
    spec = SteeringSpec(direction=_rows(4), layers=[0], coeff=1.0)
    out = _apply(_hidden(seq=3), spec, unit=1.0, offset=2)
    # Absolute positions 2, 3, 4: rows 2 and 3 apply, position 4 is past the end.
    assert out[0, 0, 2] == pytest.approx(1.0)
    assert out[0, 1, 3] == pytest.approx(1.0)
    assert torch.allclose(out[:, 2, :], torch.zeros(2, D))


def test_the_default_path_ignores_the_offset_entirely():
    """positions=None adds everywhere and produced every number in the paper."""
    spec = SteeringSpec(direction=_rows(1)[0], layers=[0], coeff=1.0)
    for offset in (0, 5, 999):
        out = _apply(_hidden(seq=4), spec, unit=1.0, offset=offset)
        assert torch.allclose(out[:, :, 0], torch.ones(2, 4))


def test_hook_counts_positions_across_passes():
    """The offset has to come from somewhere; pin that the hook tracks it."""
    import inspect

    from lbi import steering

    src = inspect.getsource(steering.steering_hooks)
    assert "seen" in src and "offset" in src


# --------------------------------------------------------------------------
# Through the hook, on the CPU stub, because this is what runs on Kaggle
# --------------------------------------------------------------------------


def _stub_lm(d_model=D, n_layers=3, vocab=32):
    """A tiny decoder with a tokenizer, enough for the stage D code path."""
    from lbi.extraction import LoadedModel, _decoder_layers, _layer_output_hidden

    class _Layer(torch.nn.Module):
        def forward(self, x):
            return (x,)          # identity, so activations are the embeddings

    class _Stack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([_Layer() for _ in range(n_layers)])
            self.embed = torch.nn.Embedding(vocab, d_model)

        def forward(self, x):
            for layer in self.layers:
                x = _layer_output_hidden(layer(x))
            return x

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Stack()

        def forward(self, input_ids=None, attention_mask=None, **kw):
            return self.model(self.model.embed(input_ids))

    class _Tok:
        """One token per character, so token counts are predictable."""

        def __call__(self, text, add_special_tokens=True, **kw):
            return {"input_ids": [ord(c) % vocab for c in text]}

    m = _Model()
    torch.manual_seed(0)
    torch.nn.init.normal_(m.model.embed.weight)
    return LoadedModel(
        name="stub", model=m, tokenizer=_Tok(), layers=_decoder_layers(m),
        n_layers=n_layers, d_model=d_model, device="cpu",
    )


def test_capture_positionwise_keeps_every_position():
    from lbi.extraction import capture_positionwise

    lm = _stub_lm()
    acts = capture_positionwise(lm, ["abc", "x"], layer=1)
    assert acts.shape == (2, 3, D), "shorter text must be padded, not pooled"
    # "x" padded with " ": positions 1 and 2 are the space embedding, and equal.
    assert np.allclose(acts[1, 1], acts[1, 2])
    # "abc" has three distinct tokens, so its positions differ.
    assert not np.allclose(acts[0, 0], acts[0, 1])


def test_capture_positionwise_rejects_a_layer_past_the_end():
    from lbi.extraction import capture_positionwise

    with pytest.raises(ValueError, match="has 3 layers"):
        capture_positionwise(_stub_lm(), ["a"], layer=3)


def test_the_hook_confines_the_matrix_to_the_prompt_across_passes():
    """End to end: the prompt pass is steered, the next token's pass is not."""
    from lbi import steering as st

    lm = _stub_lm()
    # raw_norm, as stage D uses: the scale must not depend on the activations,
    # which here are zeros and would give an RMS unit of zero.
    spec = st.SteeringSpec(direction=_rows(2), layers=[1], coeff=1.0,
                           unit_mode="raw_norm", raw_scale=np.array([1.0, 1.0]))
    prompt = torch.zeros(1, 4, D)
    nxt = torch.zeros(1, 1, D)

    with st.steering_hooks(lm, spec):
        out_prompt = lm.model.model(prompt)
        out_next = lm.model.model(nxt)

    # Layers are identity, so whatever comes out is what the hook added.
    assert out_prompt[0, 0, 0] == pytest.approx(1.0)
    assert out_prompt[0, 1, 1] == pytest.approx(1.0)
    assert torch.allclose(out_prompt[0, 2:], torch.zeros(2, D))
    assert torch.allclose(out_next, torch.zeros(1, 1, D)), (
        "the generated token was steered; the hook is not counting positions"
    )
