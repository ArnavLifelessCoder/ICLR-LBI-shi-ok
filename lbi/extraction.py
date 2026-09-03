"""Residual-stream activation capture.

Raw HuggingFace forward hooks on the decoder layers rather than TransformerLens
(revision R6): TransformerLens rewrites weights and is awkward under 4-bit
quantization on a T4, and this study only needs residual capture plus addition,
which hooks do natively.

Activations cache to disk as float16 (`.npz`) so a dead Colab session costs a
re-mount rather than a re-run.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# Model handles
# --------------------------------------------------------------------------


@dataclass
class LoadedModel:
    """A model plus the pieces the rest of the pipeline needs."""

    name: str
    model: object
    tokenizer: object
    layers: list  # decoder layer modules, index == layer number
    n_layers: int
    d_model: int
    device: str

    def layer_module(self, layer: int):
        if not 0 <= layer < self.n_layers:
            raise IndexError(f"layer {layer} out of range 0..{self.n_layers - 1}")
        return self.layers[layer]


def _decoder_layers(model):
    """Find the decoder layer list across Llama/Qwen/Mistral/Gemma layouts."""
    for path in ("model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    raise RuntimeError(
        f"could not locate decoder layers on {type(model).__name__}; "
        "add its layout to _decoder_layers"
    )


def load_model(
    name: str,
    load_in_4bit: bool = True,
    device: str | None = None,
    dtype: str = "float16",
    device_index: int = 0,
    attn_implementation: str | None = None,
) -> LoadedModel:
    """Load a model for inference. 4-bit by default so 7-9B fits a T4.

    `device_index` picks the GPU. It exists so the fixed judge can sit on a
    second card while the model under study occupies the first: a 7-9B in 4-bit
    needs about 6 GB and a 1.5B judge about 3 GB, so both fit two T4s
    comfortably and the judge never competes with the study for memory.

    `attn_implementation` defaults to eager for Gemma-2. That family soft-caps
    its attention logits, and the fused SDPA path silently ignores the cap: the
    model loads, generates, and is quietly wrong. A study whose entire signal is
    a behaviour difference under intervention cannot afford a model that is
    subtly miscomputed, and the failure would look like a concept that steers
    oddly rather than like a bug.
    """
    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = (
            f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"
        )
    torch_dtype = getattr(torch, dtype)

    if attn_implementation is None and "gemma-2" in name.lower():
        attn_implementation = "eager"

    kwargs = {"torch_dtype": torch_dtype}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if load_in_4bit and device.startswith("cuda"):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": device_index}
    else:
        kwargs["device_map"] = None

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # last-token capture needs left padding

    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if kwargs["device_map"] is None:
        model = model.to(device)
    model.eval()

    layers = _decoder_layers(model)
    return LoadedModel(
        name=name,
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        n_layers=len(layers),
        d_model=int(model.config.hidden_size),
        device=device,
    )


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------


def _layer_output_hidden(output):
    """Decoder layers return either a tensor or a tuple whose first item is it."""
    return output[0] if isinstance(output, tuple) else output


def left_padded_position_ids(attention_mask):
    """Position ids that ignore left padding.

    `padding_side` is "left" so the last position is always a real token, which
    is what last-token pooling needs. But a bare `model(**enc)` does not get
    position ids, and HuggingFace then falls back to `arange(seq_len)` -- so a
    sequence with three pad tokens has its first real token at position 3 and
    every RoPE phase in that row is shifted. `generate()` handles this; a raw
    forward pass does not. Counting only real tokens puts every sequence back
    at position 0 regardless of how much padding it received.
    """
    pos = attention_mask.long().cumsum(dim=-1) - 1
    return pos.masked_fill(attention_mask == 0, 0)


def capture_activations(
    lm: LoadedModel,
    texts: list[str],
    layers: list[int] | None = None,
    pooling: str = "last",
    batch_size: int = 8,
    max_length: int = 256,
) -> np.ndarray:
    """Residual-stream activations for `texts`.

    Returns float16 array of shape (n_layers_requested, n_texts, d_model).

    `pooling` is "last" (final non-pad token) or "mean" (mean over non-pad
    tokens). The design doc names mean-pooling as the fallback when last-token
    probes are unstable across seeds (Experiment 1).
    """
    import torch

    if pooling not in ("last", "mean"):
        raise ValueError(f"pooling must be 'last' or 'mean', got {pooling!r}")
    layers = list(range(lm.n_layers)) if layers is None else sorted(layers)

    out = np.zeros((len(layers), len(texts), lm.d_model), dtype=np.float16)

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = lm.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(lm.device)
        mask = enc["attention_mask"]

        captured: dict[int, "torch.Tensor"] = {}
        handles = []

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                captured[layer_idx] = _layer_output_hidden(output).detach()

            return hook

        try:
            for li in layers:
                handles.append(
                    lm.layer_module(li).register_forward_hook(make_hook(li))
                )
            with torch.no_grad():
                lm.model(**enc, position_ids=left_padded_position_ids(mask))
        finally:
            for h in handles:
                h.remove()

        for row, li in enumerate(layers):
            hidden = captured[li].float()  # (B, T, D)
            if pooling == "last":
                # Left padding means the last position is always real.
                pooled = hidden[:, -1, :]
            else:
                m = mask.unsqueeze(-1).float()
                pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            out[row, start : start + len(batch), :] = (
                pooled.cpu().numpy().astype(np.float16)
            )

    return out


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------


def _cache_key(model_name: str, texts: list[str], pooling: str, max_length: int) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode())
    h.update(pooling.encode())
    h.update(str(max_length).encode())
    for t in texts:
        h.update(t.encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def capture_cached(
    lm: LoadedModel,
    texts: list[str],
    cache_dir: str,
    tag: str,
    layers: list[int] | None = None,
    pooling: str = "last",
    batch_size: int = 8,
    max_length: int = 256,
) -> np.ndarray:
    """`capture_activations` with an on-disk cache keyed by model+texts+pooling."""
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(lm.name, texts, pooling, max_length)
    safe_tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tag)
    path = os.path.join(cache_dir, f"{safe_tag}_{key}.npz")

    requested = list(range(lm.n_layers)) if layers is None else sorted(layers)

    if os.path.exists(path):
        with np.load(path) as z:
            cached_layers = z["layers"].tolist()
            acts = z["acts"]
        if all(li in cached_layers for li in requested):
            idx = [cached_layers.index(li) for li in requested]
            return acts[idx]

    acts = capture_activations(
        lm, texts, layers=requested, pooling=pooling,
        batch_size=batch_size, max_length=max_length,
    )
    np.savez_compressed(
        path,
        acts=acts,
        layers=np.array(requested),
        meta=np.array(json.dumps({"model": lm.name, "pooling": pooling, "n": len(texts)})),
    )
    return acts
