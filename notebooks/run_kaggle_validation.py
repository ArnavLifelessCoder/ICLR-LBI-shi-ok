"""Validation run: three experiments, one session per model.

Read `notebooks/README.md` for the cells. This file is the config and the
stages; `lbi/` holds the logic.

The three experiments answer three separate objections, and they share a loaded
model, which is the only reason they are in one notebook. Loading a 7-9B model
in 4-bit is a few minutes; doing it three times across three notebooks wastes
most of a free-tier session.

  A. Ground truth. Four concepts whose behavioral readout is computed from the
     generated string by rule, so no judge is in the loop. The intended
     direction is known by construction, which is what lets a signed area be
     scored correct or incorrect rather than merely reported. This is the
     experiment that separates "steering has no directional effect" from "the
     judge emits noise", because on these concepts there is no judge to blame.

  B. Re-judge. The four judge-scored concepts that carry the most weight,
     re-run against a judge of comparable scale to the models being judged. The
     existing numbers come from a 1.5B judge scoring 7-9B outputs. If the signs
     hold, the directional result is robust; if they flip, the directional
     metric is judge-sensitive and that is the finding.

  C. Sensitivity to k. Output overlap is the fraction of a concept direction
     lying in the top-k right singular subspace of the unembedding, with k=512
     fixed in advance and never varied. The sign of the primary test is a claim
     about that quantity, so it should not rest on one untested constant. This
     stage needs no generation at all: activations for the pairs, a difference
     of means, then the same projection at several k. It is minutes, not hours.

The Kaggle behaviour that shapes all of this: **Save Version re-runs every cell
top to bottom under papermill**. Nothing optional may raise. Every stage below
catches its own failures and prints, so one bad model cannot cost the session.
"""

from __future__ import annotations

import json
import os
import traceback

# Keep this repository PUBLIC, and always clone with GIT_TERMINAL_PROMPT=0.
REPO = "https://github.com/ArnavLifelessCoder/ICLR-LBI-shi-ok.git"
CLONE = f"GIT_TERMINAL_PROMPT=0 git clone -q {REPO}"
WORK = "/kaggle/working"
REPO_DIR = f"{WORK}/lbi-repo"
OUT_GT = f"{WORK}/results_groundtruth"
OUT_RJ = f"{WORK}/results_rejudge"
OUT_K = f"{WORK}/results_ksweep"
CACHE_DIR = f"{WORK}/cache/activations"

# The three retained models. Gemma is withheld from the paper's main analysis
# and is therefore not re-run here; adding it back would mean re-deciding the
# positive control, which is a paper decision rather than a compute one.
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct",   # gated: licence + HF_TOKEN
]

# Comparable scale to the models under study, against the 1.5B currently used.
# fp16 rather than 4-bit: the judge is the instrument, and quantisation noise in
# the instrument is the last thing this study needs. On a 15GB T4 a 7B in fp16
# is about 14GB and leaves very little headroom, so JUDGE_FALLBACK is tried if
# the load raises. Whichever one is used is recorded in every result file.
JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
JUDGE_FALLBACK = "Qwen/Qwen2.5-3B-Instruct"

# The judge-scored concepts worth the regeneration cost: the one the judge is
# validated on, the one with the largest apparent wrong-way effect, the
# safety-relevant one, and the danger-zone occupant.
REJUDGE_CONCEPTS = ["sentiment", "certainty", "refusal", "topic_science"]

K_VALUES = [64, 128, 512, 2048]


def hf_login_if_available(secret_label: str = "HF_TOKEN") -> bool:
    """Log in to Hugging Face if the secret exists; otherwise say so and go on."""
    try:
        from kaggle_secrets import UserSecretsClient
        from huggingface_hub import login
    except ImportError:
        print("NOTE: not on Kaggle, or huggingface_hub missing; skipping HF login.")
        return False
    try:
        token = UserSecretsClient().get_secret(secret_label)
    except Exception as exc:
        print(f"NOTE: no Kaggle secret {secret_label!r}; Llama will 403. ({exc})")
        return False
    login(token)
    print("HF login OK.")
    return True


def _judge_scorer(concepts, device_index: int = 1):
    """Load the larger judge, falling back rather than killing the session.

    Returns (scorer, judge_name, lm) so the caller can free the judge before
    loading the next target model.
    """
    from lbi import behavior as bh
    from lbi.driver import load_judge

    for name in (JUDGE_MODEL, JUDGE_FALLBACK):
        try:
            lm = load_judge(name, device_index=device_index)
        except Exception as exc:
            print(f"  judge {name} failed to load ({type(exc).__name__}: {exc})")
            continue
        scorer = bh.LLMJudgeScorer(
            generate_fn=bh.make_local_generate_fn(lm),
            behavior_questions={c.name: c.behavior_question for c in concepts},
        )
        print(f"  judge: {name}")
        return scorer, name, lm
    raise RuntimeError("no judge could be loaded")


def stage_a_groundtruth(lm, out_dir: str = OUT_GT) -> bool:
    """Four concepts, deterministic readouts, no judge anywhere in the loop."""
    from lbi.groundtruth import DeterministicScorer, ground_truth_concepts
    from lbi.pipeline import run_model

    print("\n--- A: ground truth (no judge) ---")
    concepts = ground_truth_concepts()
    runs = run_model(
        lm, DeterministicScorer(), out_dir=out_dir, cache_dir=CACHE_DIR,
        concepts=concepts, resume=True,
    )
    for r in runs:
        print(f"  {r.probe.concept:<14} read {r.probe.readability:.2f}  "
              f"ctrl {r.steering.controllability:.3f}")
    return True


def stage_b_rejudge(lm, out_dir: str = OUT_RJ) -> bool:
    """The four load-bearing concepts, re-run against a larger judge."""
    from lbi.concepts import all_concepts
    from lbi.pipeline import run_model

    print("\n--- B: re-judge with a larger judge ---")
    every = all_concepts()
    concepts = [c for c in every if c.name in REJUDGE_CONCEPTS]
    missing = set(REJUDGE_CONCEPTS) - {c.name for c in concepts}
    if missing:
        print(f"  WARNING: unknown concepts {sorted(missing)}")

    scorer, judge_name, judge_lm = _judge_scorer(every)
    try:
        runs = run_model(
            lm, scorer, out_dir=out_dir, cache_dir=CACHE_DIR,
            concepts=concepts, resume=True,
        )
        # The judge identity is the whole point of this stage, so stamp it onto
        # every record rather than relying on the notebook log.
        for fn in os.listdir(out_dir):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(out_dir, fn)
            try:
                rec = json.load(open(path, encoding="utf-8"))
            except Exception:
                continue
            if isinstance(rec, dict) and "judge_model" in rec:
                rec["judge_model"] = judge_name
                rec["judge_is_self"] = judge_name == lm.name
                json.dump(rec, open(path, "w", encoding="utf-8"), indent=2)
        for r in runs:
            print(f"  {r.probe.concept:<14} ctrl {r.steering.controllability:.3f}")
    finally:
        del judge_lm
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    return True


def stage_c_ksweep(lm, out_dir: str = OUT_K) -> bool:
    """Output overlap at several k. No generation, so this is cheap."""
    import numpy as np
    from lbi import geometry as geo
    from lbi.concepts import all_concepts
    from lbi.extraction import capture_cached
    from lbi.probes import diff_of_means_direction, pair_texts

    print("\n--- C: sensitivity of output overlap to k ---")
    head = lm.model.get_output_embeddings().weight.detach().float().cpu().numpy()
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for concept in all_concepts():
        try:
            texts, labels = pair_texts(concept.pairs)
            acts = capture_cached(lm, texts, cache_dir=CACHE_DIR,
                                  tag=f"{concept.name}_ksweep")
            # Mid-network is where most concepts select; the comparison across k
            # only needs a fixed layer, not the per-concept optimum.
            layer = lm.n_layers // 2
            dom = diff_of_means_direction(acts[layer], labels)
            row = {"concept": concept.name, "model": lm.name, "layer": layer}
            for k in K_VALUES:
                row[f"overlap_k{k}"] = geo.output_overlap(dom, head, top_k=k)
            rows.append(row)
            print("  %-14s " % concept.name
                  + "  ".join("k=%d %.3f" % (k, row[f"overlap_k{k}"]) for k in K_VALUES))
        except Exception as exc:
            print(f"  {concept.name}: FAILED ({type(exc).__name__}: {exc})")

    slug = lm.name.replace("/", "_").replace(".", "-")
    path = os.path.join(out_dir, f"{slug}_ksweep.json")
    json.dump(rows, open(path, "w", encoding="utf-8"), indent=2)
    print(f"  wrote {path}")
    return True


def run_all(models=None) -> dict:
    """Load each model once and run all three stages against it."""
    from lbi.extraction import load_model

    models = models or MODELS
    status: dict[str, dict] = {}
    for name in models:
        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)
        status[name] = {}
        try:
            lm = load_model(name, load_in_4bit=True, device_index=0)
        except Exception as exc:
            print(f"LOAD FAILED ({type(exc).__name__}: {exc})")
            status[name]["load"] = f"failed: {exc}"
            continue
        status[name]["load"] = "ok"

        for tag, fn in (("A_groundtruth", stage_a_groundtruth),
                        ("B_rejudge", stage_b_rejudge),
                        ("C_ksweep", stage_c_ksweep)):
            try:
                fn(lm)
                status[name][tag] = "ok"
            except Exception as exc:
                # One stage failing must not cost the other two, or the model.
                print(f"\n{tag} FAILED ({type(exc).__name__}: {exc})")
                traceback.print_exc()
                status[name][tag] = f"failed: {exc}"

        del lm
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    print("\n" + "=" * 70)
    print(json.dumps(status, indent=2))
    json.dump(status, open(f"{WORK}/validation_status.json", "w"), indent=2)
    return status
