"""Turn a hand-filled human_labels.csv into the judge-agreement number.

This closes objection 14. Filling the sheet is necessary but not sufficient:
per-sample judge scores are NOT persisted in the result JSONs (only the mean
`behavior` per coefficient), so the judges have to be re-run on the exact texts
that were hand-labelled before human and judge can be compared at all.

The lexicon judge runs on CPU, so a useful human-vs-lexicon alpha is available
with no GPU. Pass --judge-model to add the fixed logit judge (the primary
instrument) and get the three-rater number that actually answers the objection.

    # CPU only, no downloads
    python scripts/score_human_labels.py "results nb7/results/human_labels.csv"

    # On Kaggle, with the primary judge (a few seconds for 100 short texts)
    python scripts/score_human_labels.py results/human_labels.csv \
        --judge-model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from lbi import behavior as bh
from lbi.concepts import all_concepts


def _read_rows(path: str) -> list[dict]:
    """Read the sheet preserving text, concept and the human score."""
    import csv

    rows = []
    # utf-8-sig, because Excel adds a BOM on save and that renames the first
    # column to "﻿index" for DictReader.
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get("score") or "").strip()
            try:
                human = float(raw)
            except ValueError:
                human = float("nan")
            if not math.isnan(human) and not 0.0 <= human <= 1.0:
                raise ValueError(
                    f"score {human} at index {row.get('index')} is outside [0, 1]"
                )
            rows.append(
                {
                    "index": row.get("index"),
                    "concept": row["concept"],
                    "text": row.get("text", ""),
                    "human": human,
                }
            )
    return rows


def _score_by_concept(scorer, rows: list[dict], label: str) -> list[float]:
    """Score every row, grouping by concept because scorers are per-concept."""
    out = [float("nan")] * len(rows)
    by_concept: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_concept.setdefault(r["concept"], []).append(i)
    for concept, idxs in by_concept.items():
        texts = [rows[i]["text"] for i in idxs]
        try:
            scores = scorer.score(texts, concept)
        except Exception as exc:  # a missing lexicon or judge should not abort
            print(f"  WARNING [{label}] {concept}: {type(exc).__name__}: {exc}")
            continue
        for i, s in zip(idxs, scores):
            out[i] = float(s)
    return out


def _within_concept_alpha(a, b, concepts):
    """Alpha after removing each concept's mean from both raters.

    Pooled alpha over a multi-concept sheet is mostly the between-concept
    signal: both raters know refusal text scores low and honesty text scores
    high, which is trivially true because they answer different questions. That
    inflates the number without saying anything about whether the judge tracks
    the human on individual items.

    Controllability is a within-concept quantity -- it is the behaviour score
    moving across steering strengths for one concept -- so the centered number
    is the one that says whether the judge is fit for this study. On the first
    labelled sheet pooled read +0.573 while this read +0.022.
    """
    from collections import defaultdict

    idx = defaultdict(list)
    for i, c in enumerate(concepts):
        if not (np.isnan(a[i]) or np.isnan(b[i])):
            idx[c].append(i)
    ac, bc = [], []
    for ii in idx.values():
        if len(ii) < 2:
            continue  # a single item carries no within-concept information
        am = float(np.mean([a[i] for i in ii]))
        bm = float(np.mean([b[i] for i in ii]))
        for i in ii:
            ac.append(a[i] - am)
            bc.append(b[i] - bm)
    if len(ac) < 2:
        return float("nan"), 0
    return bh.krippendorff_alpha_interval([ac, bc]), len(ac)


def _variance_diagnosis(a, b, concepts, name_a, name_b, flat=0.11):
    """Per-concept spread for both raters, with what a low spread implies.

    Alpha cannot be meaningfully positive when one rater is near-constant:
    there is nothing to covary with, so a negative value there is an artifact
    of no variance rather than evidence of disagreement. Worse, a judge that
    varies where a human sees no difference is manufacturing within-concept
    variation out of noise, and within-concept variation is exactly what
    controllability measures.
    """
    from collections import defaultdict

    idx = defaultdict(list)
    for i, c in enumerate(concepts):
        if not (np.isnan(a[i]) or np.isnan(b[i])):
            idx[c].append(i)

    out = []
    for c in sorted(idx):
        ii = idx[c]
        if len(ii) < 2:
            continue
        asd = float(np.std([a[i] for i in ii]))
        bsd = float(np.std([b[i] for i in ii]))
        alpha = bh.krippendorff_alpha_interval(
            [[a[i] for i in ii], [b[i] for i in ii]]
        )
        if asd < flat and bsd < flat:
            note = "both flat: texts do not differ here, alpha uninformative"
        elif asd < flat:
            note = name_a + " flat, " + name_b + " varies: " + name_b + " may be reading noise"
        elif bsd < flat:
            note = name_a + " varies, " + name_b + " flat: " + name_b + " blind here"
        else:
            note = "both vary: alpha is meaningful"
        out.append({
            "concept": c, "n": len(ii),
            "sd_" + name_a: asd, "sd_" + name_b: bsd,
            "alpha": alpha, "note": note,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet", help="path to a filled human_labels.csv")
    ap.add_argument("--judge-model", help="HF id of the fixed logit judge (needs GPU)")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--out", help="write the agreement report as JSON here")
    args = ap.parse_args()

    rows = _read_rows(args.sheet)
    n_filled = sum(1 for r in rows if not math.isnan(r["human"]))
    print(f"{args.sheet}: {len(rows)} rows, {n_filled} hand-labelled")
    if n_filled < 2:
        print(
            "Fewer than 2 labelled rows. Fill the `score` column with a number "
            "in [0, 1] answering that row's `question`, then re-run. Blanks are "
            "allowed and are treated as missing, so a partial sheet is fine."
        )
        return 1

    raters: dict[str, list[float]] = {"human": [r["human"] for r in rows]}

    print("scoring with the lexicon judge (CPU)...")
    raters["lexicon"] = _score_by_concept(bh.LexiconScorer(), rows, "lexicon")

    if args.judge_model:
        print(f"loading fixed logit judge {args.judge_model}...")
        from lbi.extraction import load_model

        judge_lm = load_model(args.judge_model, load_in_4bit=not args.no_4bit)
        questions = {c.name: c.behavior_question for c in all_concepts()}
        raters["logit_judge"] = _score_by_concept(
            bh.LogitJudgeScorer(judge_lm, questions), rows, "logit_judge"
        )
    else:
        print(
            "no --judge-model given, so the primary logit judge is not included. "
            "The human-vs-lexicon number below is a lower bound: the lexicon "
            "scorer is a development instrument that returns a neutral 0.5 on "
            "text with no keyword hits."
        )

    names = list(raters)
    report: dict = {"sheet": args.sheet, "n_rows": len(rows), "n_labelled": n_filled}

    print("\npairwise Krippendorff alpha (interval), on rows both raters scored:")
    pairwise = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = np.array(raters[names[i]]), np.array(raters[names[j]])
            mask = ~np.isnan(a) & ~np.isnan(b)
            if mask.sum() < 2:
                print(f"  {names[i]:>12} vs {names[j]:<12} n<2, skipped")
                continue
            alpha = bh.krippendorff_alpha_interval([a[mask].tolist(), b[mask].tolist()])
            pairwise[f"{names[i]}_vs_{names[j]}"] = {
                "alpha": alpha,
                "n": int(mask.sum()),
            }
            print(f"  {names[i]:>12} vs {names[j]:<12} alpha={alpha:+.3f}  n={mask.sum()}")
    report["pairwise"] = pairwise

    # The pooled figures above are not the ones that validate the judge.
    concepts = [r["concept"] for r in rows]
    print("\n" "pooled vs within-concept (the within number is the one that counts):")
    decomposition = {}
    for other in [n for n in names if n != "human"]:
        a, b = np.array(raters["human"]), np.array(raters[other])
        mask = ~np.isnan(a) & ~np.isnan(b)
        if mask.sum() < 2:
            continue
        pooled = bh.krippendorff_alpha_interval([a[mask].tolist(), b[mask].tolist()])
        within, n_w = _within_concept_alpha(a, b, concepts)
        decomposition["human_vs_" + other] = {
            "pooled": pooled, "pooled_n": int(mask.sum()),
            "within_concept": within, "within_n": n_w,
        }
        print("  human vs " + other + ":")
        print(f"     pooled          alpha={pooled:+.3f}  n={mask.sum()}")
        print(f"     within-concept  alpha={within:+.3f}  n={n_w}")
    report["decomposition"] = decomposition

    # Per-concept spread, which says where alpha can mean anything at all.
    diagnosis = {}
    for other in [n for n in names if n != "human"]:
        a, b = np.array(raters["human"]), np.array(raters[other])
        rowsd = _variance_diagnosis(a, b, concepts, "human", other)
        if not rowsd:
            continue
        diagnosis[other] = rowsd
        print("\n" "per-concept spread, human vs " + other + ":")
        print(f"  {'concept':<14} {'n':>3} {'human_sd':>9} {'judge_sd':>9} "
              f"{'alpha':>8}  diagnosis")
        for d in rowsd:
            print(f"  {d['concept']:<14} {d['n']:>3} {d['sd_human']:>9.3f} "
                  f"{d['sd_' + other]:>9.3f} {d['alpha']:>+8.3f}  {d['note']}")
    report["variance_diagnosis"] = diagnosis

    if len(names) >= 3:
        alpha_all = bh.krippendorff_alpha_interval([raters[n] for n in names])
        report["all_raters"] = {"raters": names, "alpha": alpha_all}
        print(f"\nall {len(names)} raters together: alpha={alpha_all:+.3f}")

    # Where human and the primary judge disagree most is the useful diagnostic.
    if "logit_judge" in raters:
        h, j = np.array(raters["human"]), np.array(raters["logit_judge"])
        mask = ~np.isnan(h) & ~np.isnan(j)
        if mask.any():
            diffs = np.abs(h - j)
            order = np.argsort(-np.where(mask, diffs, -np.inf))[:5]
            print("\nlargest human/judge disagreements (inspect these first):")
            for k in order:
                if not mask[k]:
                    continue
                print(
                    f"  idx {rows[k]['index']:>3} {rows[k]['concept']:<14} "
                    f"human={h[k]:.2f} judge={j[k]:.2f}  "
                    f"{rows[k]['text'][:70]!r}"
                )

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
