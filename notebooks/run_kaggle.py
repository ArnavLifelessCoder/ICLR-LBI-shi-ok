"""Kaggle T4 x2 driver. Paste these as cells, or attach the repo and import.

Kaggle specifics that differ from a local run:
  * Internet is OFF by default and nothing works without it (Settings ->
    Internet on; needs phone verification).
  * /kaggle/working is the only persisted directory and only if you Save
    Version; /kaggle/temp does not survive the session.
  * Gated models (Llama, Gemma) need an HF token in Kaggle Secrets and the
    licence accepted on huggingface.co first.
  * Two T4s are visible. A 7-9B in 4-bit needs about 6 GB, so it fits on one
    and the study pins device 0; the second card is spare capacity, not a
    requirement.
"""

REPO = "https://github.com/ArnavLifelessCoder/legible-but-immovable.git"
WORK = "/kaggle/working"
OUT_DIR = f"{WORK}/results"
CACHE_DIR = f"{WORK}/cache/activations"

MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",           # start here, ungated
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct",   # gated: accept licence + HF token
    "google/gemma-2-9b-it",               # gated
    "Qwen/Qwen2.5-0.5B-Instruct",         # Experiment 6 scale sweep
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]
