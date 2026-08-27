"""Colab driver for the study. Paste cells into a notebook, or %run this file.

Written as a plain script so it lives in git as reviewable text rather than a
JSON blob with embedded outputs. `notebooks/README.md` explains the cell split.

The order below is not arbitrary. The positive control runs first and on its
own, because PLAN.md Phase C is explicit that a broken harness discovered late
costs the paper, and a T4 session is the scarcest thing in this project.

    STAGE 0  environment, ~3 min
    STAGE 1  positive control only, ~5 min      <- stop here if it fails
    STAGE 2  full sweep for one model, ~1.5 h
    STAGE 3  aggregate across models, CPU only
"""

# ---------------------------------------------------------------------------
# STAGE 0. Environment.
# ---------------------------------------------------------------------------
# On Windows, torch must be imported before scikit-learn or its DLLs fail to
# load; on Linux the order is harmless, so keep it uniform.
SETUP = r"""
!git clone https://github.com/ArnavLifelessCoder/legible-but-immovable.git
%cd legible-but-immovable
!pip -q install transformers accelerate bitsandbytes scikit-learn scipy matplotlib
!python -m pytest tests/ -q
!python scripts/demo_synthetic.py
"""
# Both must pass before spending a single GPU-minute. They need no GPU and no
# downloads, and between them they check every metric path against planted
# ground truth.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402  (before sklearn, see above)

from lbi import behavior as bh  # noqa: E402
from lbi.concepts import all_concepts, audit_all_concepts, get_concept  # noqa: E402
from lbi.extraction import load_model  # noqa: E402
from lbi.pipeline import CONTROL_FLOOR, POSITIVE_CONTROL, run_model  # noqa: E402


MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-9b-it",
    # Experiment 6, the scale sweep. Small ones are minutes each.
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

OUT_DIR = "/content/drive/MyDrive/lbi/results"
CACHE_DIR = "/content/drive/MyDrive/lbi/cache/activations"
# Point both at Drive so a dead runtime costs a re-mount, not a re-run. Mount
# with:  from google.colab import drive; drive.mount('/content/drive')


def preflight() -> bool:
    """CPU-only checks that must pass before any model is loaded."""
    print("=== surface-shortcut audit (Phase A gate) ===")
    ok = True
    for r in audit_all_concepts():
        tag = "PASS " if r.passed else ("FAIL*" if r.surface_confounded else "FAIL ")
        print(f"  {r.concept:<14} {tag} mean={r.surface_auroc:.3f} "
              f"worst={r.worst_fold_auroc:.3f}")
        if not r.passed and not r.surface_confounded:
            ok = False
    print("  FAIL* = declared surface-confounded; expected, reported not hidden.")
    if not ok:
        print("  UNDECLARED AUDIT FAILURE -- fix the concept set before running.")
    return ok


def build_judges(lm):
    """The judge panel. Never run the study on LexiconScorer alone.

    Two independent judges with reported agreement is a design-doc requirement,
    and the lexicon scorer is a development instrument. The LLM judge reuses the
    model already in memory, so it costs generation time and no extra download.
    """
    llm = bh.LLMJudgeScorer(
        generate_fn=bh.make_local_generate_fn(lm),
        behavior_questions={c.name: c.behavior_question for c in all_concepts()},
    )
    return bh.PanelScorer({"llm": llm, "lexicon": bh.LexiconScorer()}, primary="llm")


# ---------------------------------------------------------------------------
# STAGE 1. Positive control, alone, before anything else.
# ---------------------------------------------------------------------------


def stage1_control(model_name: str, load_in_4bit: bool = True) -> bool:
    """Run only the control concept. ~5 minutes. The go/no-go for this model."""
    lm = load_model(model_name, load_in_4bit=load_in_4bit)
    print(f"{lm.n_layers} layers, d_model={lm.d_model}, device={lm.device}")

    panel = build_judges(lm)
    runs = run_model(
        lm, panel, out_dir=OUT_DIR, cache_dir=CACHE_DIR,
        concepts=[get_concept(POSITIVE_CONTROL)],
    )
    value = runs[0].steering.controllability
    passed = value >= CONTROL_FLOOR

    print(f"\ncontrol controllability = {value:.3f} (floor {CONTROL_FLOOR})")
    print("PASS -- steering works on this model." if passed else
          "FAIL -- steering does not move sentiment. Do not run the sweep.\n"
          "  Check, in order: the chat template, the coefficient grid reaching\n"
          "  far enough in RMS units, and whether the fluency ceiling is being\n"
          "  hit before any behaviour change (ceiling_reason says which).")
    print(f"judge parse-failure rate: {panel.judges['llm'].failure_rate():.1%}")
    return passed


# ---------------------------------------------------------------------------
# STAGE 2. Full sweep for one model.
# ---------------------------------------------------------------------------


def stage2_full(model_name: str, load_in_4bit: bool = True):
    """All concepts for one model. ~1.5 h for a 7B on a T4."""
    lm = load_model(model_name, load_in_4bit=load_in_4bit)
    panel = build_judges(lm)
    runs = run_model(lm, panel, out_dir=OUT_DIR, cache_dir=CACHE_DIR)

    for r in runs:
        flag = ""
        if r.gauntlet_passed is True:
            flag = "  [IMMOVABLE: survived the gauntlet]"
        elif r.gauntlet_passed is False:
            flag = f"  [moved: {r.gauntlet['verdict']}]"
        print(f"  {r.probe.concept:<14} read={r.probe.readability:.3f} "
              f"ctrl={r.steering.controllability:.3f} "
              f"ceiling={r.steering.max_usable_coeff:g}{flag}")

    agreement = panel.agreement()
    print(f"\njudge agreement: alpha={agreement['krippendorff_alpha']:.3f} "
          f"over {agreement['n_items']} items")
    print(f"judge parse-failure rate: {panel.judges['llm'].failure_rate():.1%}")

    # The third rater is you. Dump a sample to hand-label; blanks are allowed.
    samples = [
        (r.probe.concept, s)
        for r in runs for p in r.steering.curve for s in p.samples
    ][:100]
    sheet = os.path.join(OUT_DIR, "human_labels.csv")
    bh.write_labeling_sheet(
        sheet, samples, {c.name: c.behavior_question for c in all_concepts()}
    )
    print(f"hand-label 100 outputs in {sheet}, then re-run agreement with it.")
    return runs


# ---------------------------------------------------------------------------
# STAGE 3. Aggregate. CPU only, run after every model is done.
# ---------------------------------------------------------------------------

AGGREGATE = r"""
!python scripts/run_experiment.py --aggregate-only --out-dir {OUT_DIR}
"""
# Reads every per-concept record, withholds any model whose control failed
# (P9), builds the gap map with CI-excluded danger-zone membership (P6) and a
# concept-clustered correlation CI (P7), then runs the preregistered primary
# test and the BH-corrected exploratory analyses.


if __name__ == "__main__":
    if not preflight():
        raise SystemExit(1)
    target = sys.argv[1] if len(sys.argv) > 1 else MODELS[0]
    if stage1_control(target):
        stage2_full(target)
