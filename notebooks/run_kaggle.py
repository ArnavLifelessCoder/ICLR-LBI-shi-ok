"""Kaggle T4 x2 config. The stages live in `lbi/driver.py`.

Read `notebooks/README.md` for the cells and `notebooks/RUNLOG.md` for what has
already been run and what came back.

The one Kaggle behaviour that shapes everything here: **Save Version re-runs
every cell top to bottom under papermill**. There is no skipping a cell in a
committed notebook, so any cell that can raise will eventually kill a run.
Optional steps must degrade to a printed note instead of an exception --
`hf_login_if_available` below is the reason Version 1 failed.
"""

# Keep this repository PUBLIC, and always clone with GIT_TERMINAL_PROMPT=0.
# Cloning a private repo over HTTPS prompts for a username; under papermill
# nothing can answer and the notebook hangs until Kaggle's 12-hour limit kills
# it, producing no output and burning the weekly GPU quota.
REPO = "https://github.com/ArnavLifelessCoder/ICLR-LBI-shi-ok.git"
CLONE = f"GIT_TERMINAL_PROMPT=0 git clone -q {REPO}"
WORK = "/kaggle/working"
REPO_DIR = f"{WORK}/lbi-repo"
OUT_DIR = f"{WORK}/results"
CACHE_DIR = f"{WORK}/cache/activations"

# Ungated first. On the first real run a licence problem must be impossible, so
# that anything that breaks is known to be the pipeline.
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",           # ungated, start here
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct",   # gated: licence + HF_TOKEN
    "google/gemma-2-9b-it",               # gated
    "Qwen/Qwen2.5-0.5B-Instruct",         # Experiment 6 scale sweep
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]


def hf_login_if_available(secret_label: str = "HF_TOKEN") -> bool:
    """Log in to Hugging Face if the secret exists; otherwise say so and go on.

    Only the gated models (Llama, Gemma) need this. Raising when the secret is
    absent means a Qwen-only run cannot be saved as a version at all, which is
    exactly how the first Kaggle attempt died: every CPU check had passed and
    the notebook still came back as "failed to run".
    """
    try:
        from kaggle_secrets import UserSecretsClient
        from huggingface_hub import login
    except ImportError:
        print("NOTE: not on Kaggle, or huggingface_hub missing; skipping HF login.")
        return False

    try:
        token = UserSecretsClient().get_secret(secret_label)
    except Exception as exc:  # BackendError when the label does not exist
        print(f"NOTE: no Kaggle secret {secret_label!r} attached to this "
              f"notebook, so no HF login. Ungated models (Qwen, Mistral) are "
              f"unaffected; Llama and Gemma will 403.\n      ({exc})")
        return False

    login(token)
    print(f"HF login OK via secret {secret_label!r}.")
    return True
