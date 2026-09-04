"""Build a clean hand-labelling sheet from result JSONs already on disk.

The sheet the pipeline writes at the end of a run covers only that run's model.
This builds one from any set of result directories, so a validation sheet can
span every model and every concept without re-running anything on a GPU.

    python scripts/make_label_sheet.py -o human_labels_v2.csv \
        "nb-6 results/results" "results nb7/results"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lbi import behavior as bh
from lbi.concepts import all_concepts


def collect(dirs: list[str]) -> list[tuple[str, float, str]]:
    """Every stored generation as (concept, coeff, text)."""
    items: list[tuple[str, float, str]] = []
    skipped = 0
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            base = os.path.basename(path)
            if base in ("gap_map.json", "geometry_predictor.json") or base.endswith(
                "_control.json"
            ):
                continue
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
            concept = rec.get("probe", {}).get("concept")
            if not concept:
                skipped += 1
                continue
            for point in rec.get("curve", []):
                for text in point.get("samples") or []:
                    if text and text.strip():
                        items.append((concept, float(point["coeff"]), text))
    if skipped:
        print(f"  skipped {skipped} file(s) with no probe.concept")
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="result directories to draw from")
    ap.add_argument("-o", "--out", default="human_labels_v2.csv")
    ap.add_argument("-n", "--n", type=int, default=100)
    ap.add_argument("--n-repeat", type=int, default=10,
                    help="rows re-presented to measure intra-rater reliability")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items = collect(args.dirs)
    print(f"collected {len(items)} generations from {len(args.dirs)} director(ies)")
    uniq = {(c, t) for c, _, t in items}
    print(f"  {len(uniq)} unique (concept, text) pairs across "
          f"{len({c for c, _, _ in items})} concepts")

    samples = bh.stratified_label_sample(
        items, n=args.n, n_repeat=args.n_repeat, seed=args.seed
    )
    bh.write_labeling_sheet(
        args.out, samples, {c.name: c.behavior_question for c in all_concepts()}
    )

    counts = Counter(c for c, _ in samples)
    print(f"\nwrote {args.out}: {len(samples)} rows, "
          f"{len(set(samples))} distinct, "
          f"{len(samples) - len(set(samples))} deliberate repeats")
    print("per-concept rows:")
    for c in sorted(counts):
        print(f"  {c:<14} {counts[c]}")
    missing = {c.name for c in all_concepts()} - set(counts)
    if missing:
        print(f"\nWARNING: no rows for {sorted(missing)} -- "
              "those concepts have no stored generations in the given directories.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
