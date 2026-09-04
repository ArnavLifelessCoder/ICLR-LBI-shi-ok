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
    with open(path, encoding="utf-8", newline="") as f:
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
