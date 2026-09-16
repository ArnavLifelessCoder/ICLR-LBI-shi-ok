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
#
# Run ONE PER SESSION. Stage A costs about five hours per model at thirty
# evaluation prompts over a four-layer band, so three models do not fit inside
# Kaggle's twelve-hour limit: the first attempt got through two and was killed
# partway into the third. `run_all` defaults to the first entry for that reason;
# pass the model you want explicitly.
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct",   # gated: licence + HF_TOKEN
]

# Larger than the 1.5B used elsewhere, small enough to leave working memory.
#
# The first validation run set this to the 7B and lost stage B on both models.
# A 7B in fp16 loads on a 15GB T4 -- it reported 14.28 GiB allocated of 14.56 --
# and then died on the first forward pass with 14.81 MiB free. Guarding only the
# *load* was the mistake: the load is not where a judge that barely fits fails.
# 3B in fp16 is about 6GB and leaves room to generate.
#
# fp16 rather than 4-bit: the judge is the instrument, and quantisation noise in
# the instrument is the last thing this study needs.
JUDGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
JUDGE_FALLBACK = "Qwen/Qwen2.5-1.5B-Instruct"

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


def _judge_scorer(concepts, device_index: int = 1, preferred: str | None = None):
    """Load the larger judge, falling back rather than killing the session.

    Returns (scorer, judge_name, lm) so the caller can free the judge before
    loading the next target model.
    """
    from lbi import behavior as bh
    from lbi.driver import load_judge

    candidates = ([preferred] if preferred else [JUDGE_MODEL, JUDGE_FALLBACK])
    for name in candidates:
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
    """Four concepts, deterministic readouts, no judge anywhere in the loop.

    Swept on the extended coefficient grid. The first validation run found the
    response flat across the whole default grid and then moving at the last
    point: gt_uppercase sat at 0.03 through the sweep and reached 0.34 at
    alpha=+3. The default grid ends exactly where these concepts start to move,
    so it measures the run-up and clips the effect. The extended grid is a
    strict superset, so a default-grid number can still be recovered by
    subsetting.
    """
    from lbi.groundtruth import DeterministicScorer, ground_truth_concepts
    from lbi.pipeline import EXTENDED_COEFFS, run_model

    print("\n--- A: ground truth (no judge) ---")
    print("  grid: %s" % EXTENDED_COEFFS)
    concepts = ground_truth_concepts()
    runs = run_model(
        lm, DeterministicScorer(), out_dir=out_dir, cache_dir=CACHE_DIR,
        concepts=concepts, resume=True, coeffs=EXTENDED_COEFFS,
        # Single layer, not the four-layer band. The band exists so that
        # controllability is the best over a preregistered window, which makes
        # the ten study concepts comparable to each other and biases that study
        # against its own headline. These concepts are a separate experiment
        # about whether the metric can recover a known direction, and they are
        # not compared against those numbers, so the band buys nothing here and
        # costs four times the generations.
        #
        # It cost eight hours of a Gemma session before anyone noticed: four
        # concepts over thirteen coefficients and thirty prompts is 1560
        # generations, and the band made it 6240, on a 9B model that must run
        # eager attention because of its logit soft-cap.
        best_over_band=False,
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
        try:
            runs = run_model(
                lm, scorer, out_dir=out_dir, cache_dir=CACHE_DIR,
                concepts=concepts, resume=True,
            )
        except Exception as exc:
            # A judge that fits in memory but cannot generate is the failure the
            # first run hit, and it is not visible at load time. Drop to the
            # smaller judge and retry once rather than losing the stage.
            if "out of memory" not in str(exc).lower():
                raise
            print(f"  {judge_name} ran out of memory generating; "
                  f"retrying with {JUDGE_FALLBACK}")
            del judge_lm, scorer
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            scorer, judge_name, judge_lm = _judge_scorer(
                every, preferred=JUDGE_FALLBACK)
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


def preflight() -> dict:
    """Print what is actually about to run, and refuse a stale checkout.

    The Qwen validation session swept nine coefficients instead of thirteen
    because `git clone` into an existing directory fails silently and the cell
    swallowed it, so the session ran a checkout from before the grid was
    extended. Nothing in the run said so; it was only visible afterwards by
    counting points in the result files. Seven hours of GPU time produced a
    sweep nobody wanted.

    So the run now states its own version before spending anything, and raises
    if the code does not have the extension it is supposed to have.
    """
    import subprocess

    from lbi.pipeline import DEFAULT_COEFFS, EXTENDED_COEFFS

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        commit = subprocess.run(
            ["git", "-C", here, "log", "--oneline", "-1"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:
        commit = f"(unavailable: {exc})"

    info = {
        "commit": commit,
        "default_grid": DEFAULT_COEFFS,
        "extended_grid": EXTENDED_COEFFS,
        "judge": JUDGE_MODEL,
        "judge_fallback": JUDGE_FALLBACK,
        "stage_order": ["C_ksweep", "B_rejudge", "A_groundtruth"],
        "models_available": MODELS,
    }
    print("commit        :", info["commit"])
    print("stage A grid  : %d points, %s" % (len(EXTENDED_COEFFS), EXTENDED_COEFFS))
    print("stage B grid  : %d points (default, for comparability with the 1.5B judge)"
          % len(DEFAULT_COEFFS))
    print("judge         : %s (fallback %s)" % (JUDGE_MODEL, JUDGE_FALLBACK))
    print("stage order   : C, B, A (cheapest first)")

    if set(DEFAULT_COEFFS) > set(EXTENDED_COEFFS) or \
            max(EXTENDED_COEFFS) <= max(DEFAULT_COEFFS):
        raise RuntimeError(
            "stale checkout: EXTENDED_COEFFS does not extend DEFAULT_COEFFS. "
            "Delete /kaggle/working/lbi-repo and clone again."
        )
    print("preflight OK")
    return info


def run_all(models=None, stages=None) -> dict:
    """Load each model once and run the requested stages against it.

    Defaults to one model, because three do not fit in a session. Pass a list
    to choose which, and resume means a second session picks up where the
    first stopped rather than repeating it.

    `stages` selects a subset by letter, e.g. `stages="A"` or
    `stages=["C", "A"]`. Stages always execute cheapest first regardless of the
    order given. This exists because a change can invalidate one stage and not
    the others: extending the coefficient grid invalidated stage A while
    leaving B and C, which are already stored, perfectly good. Re-running all
    three to refresh one of them costs an hour of a twelve-hour session and
    overwrites results that were fine.
    """
    from lbi.extraction import load_model

    # Before anything expensive. A stale checkout costs a whole session and is
    # invisible until the results are counted afterwards.
    preflight()

    if models is None:
        models = MODELS[:1]
    elif isinstance(models, str):
        models = [models]

    # Cheapest first, always. Stage C is ten minutes and stage A is hours, and
    # in the first run the wall clock arrived during stage A on the third
    # model, which had therefore produced nothing at all. Ordering by cost
    # means a truncated session still returns its cheap results.
    ALL_STAGES = (("C_ksweep", stage_c_ksweep),
                  ("B_rejudge", stage_b_rejudge),
                  ("A_groundtruth", stage_a_groundtruth))
    if stages is None:
        selected = ALL_STAGES
    else:
        want = {s.strip().upper()[0] for s in
                ([stages] if isinstance(stages, str) else stages)}
        unknown = want - {"A", "B", "C"}
        if unknown:
            raise ValueError(f"unknown stages {sorted(unknown)}; use A, B, C")
        selected = tuple(t for t in ALL_STAGES if t[0][0] in want)
    print("stages        : %s" % ", ".join(t[0] for t in selected))

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

        for tag, fn in selected:
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
    try:
        os.makedirs(WORK, exist_ok=True)
        json.dump(status, open(f"{WORK}/validation_status.json", "w"), indent=2)
    except OSError as exc:
        # Off Kaggle there is no /kaggle/working. The status is already printed,
        # and failing here would throw away a completed run's summary.
        print(f"(could not write validation_status.json: {exc})")
    return status
