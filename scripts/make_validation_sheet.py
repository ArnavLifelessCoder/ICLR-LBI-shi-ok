"""Build the round-3 judge-validation sheet from the stage G generations.

    python scripts/make_validation_sheet.py

Writes two files:

  human_labels_v3.csv              the sheet to fill in: index, concept,
                                   question, text, and an empty score column
  human_labels_v3_KEY_do_not_open.csv
                                   model, coefficient and the 3B judge's score
                                   for each index, kept apart so the labeller
                                   stays blind to condition and to the judge

The design and the pass criterion are fixed in VALIDATION_PREREG.md, which was
committed before this sheet existed.

Each stage G result stores two generations per coefficient per model, and the
judge's score for each of them sits at the same position in `scores`, because
`run_steering` scores the generations in order and keeps the first two as
samples. So the judge side of the comparison is already on disk and no model is
re-run.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lbi.concepts import all_concepts  # noqa: E402

CONCEPTS = ["sentiment", "sycophancy", "formality", "rudeness"]
PER_CONCEPT = 25
MIN_WORDS = 5
SEED = 0


def coeff_bin(c: float) -> str:
    """Negative, near zero, or positive: where along the sweep a text came from."""
    if c < -0.75:
        return "neg"
    if c > 0.75:
        return "pos"
    return "mid"


def collect(result_dir: str, concept: str) -> list[dict]:
    """Every usable stored generation for one concept, with its judge score."""
    items: list[dict] = []
    seen: set[str] = set()
    for path in sorted(glob.glob(os.path.join(result_dir, "*_%s.json" % concept))):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        if rec.get("probe", {}).get("concept") != concept:
            continue
        model = rec["probe"]["model"]
        for pt in rec.get("curve", []):
            if pt.get("broken"):
                continue
            samples = pt.get("samples") or []
            scores = pt.get("scores") or []
            for i, text in enumerate(samples):
                if i >= len(scores):
                    break
                t = (text or "").strip()
                if len(t.split()) < MIN_WORDS or t in seen:
                    continue
                seen.add(t)
                items.append({
                    "concept": concept, "model": model,
                    "coeff": float(pt["coeff"]), "text": t,
                    "judge": float(scores[i]),
                })
    return items


def spread_sample(items: list[dict], n: int, seed: int) -> list[dict]:
    """Round-robin over (model, coefficient bin) cells so the sample varies.

    A random draw would be dominated by texts near baseline, where the behavior
    barely differs between texts and agreement cannot be measured.
    """
    rng = random.Random(seed)
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for it in items:
        cells[(it["model"], coeff_bin(it["coeff"]))].append(it)
    for v in cells.values():
        rng.shuffle(v)
    order = sorted(cells)
    rng.shuffle(order)
    out: list[dict] = []
    while len(out) < n and any(cells[k] for k in order):
        for k in order:
            if cells[k] and len(out) < n:
                out.append(cells[k].pop())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="validation_output/results_judge_matched")
    ap.add_argument("-o", "--out", default="human_labels_v3.csv")
    ap.add_argument("--key", default="human_labels_v3_KEY_do_not_open.csv")
    ap.add_argument("--per-concept", type=int, default=PER_CONCEPT)
    args = ap.parse_args()

    questions = {c.name: c.behavior_question for c in all_concepts()}
    rows: list[dict] = []
    for concept in CONCEPTS:
        pool = collect(args.results, concept)
        picked = spread_sample(pool, args.per_concept, SEED)
        if len(picked) < args.per_concept:
            print("WARNING: %s has only %d usable texts" % (concept, len(picked)))
        # Shuffle within concept so sheet order carries no information about
        # model or coefficient.
        random.Random(SEED + 1).shuffle(picked)
        rows.extend(picked)
        bins = defaultdict(int)
        for it in picked:
            bins[coeff_bin(it["coeff"])] += 1
        print("%-11s pool %3d  picked %d  models %d  bins %s"
              % (concept, len(pool), len(picked),
                 len({it["model"] for it in picked}), dict(bins)))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "concept", "question", "text", "score"])
        for i, it in enumerate(rows):
            w.writerow([i, it["concept"], questions[it["concept"]], it["text"], ""])
    with open(args.key, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "concept", "model", "coeff", "judge"])
        for i, it in enumerate(rows):
            w.writerow([i, it["concept"], it["model"], it["coeff"], it["judge"]])
    print("wrote %s (%d rows) and %s" % (args.out, len(rows), args.key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
