"""Rebuild the gap map with withheld models removed.

    python scripts/rebuild_gapmap_withheld.py

Gemma fails the preregistered positive control under the directional metric, so
it is withheld from the main analysis. The gap map, its correlation and its
danger zone all have to be recomputed on the retained points rather than
filtered after the fact, because within-model normalization and the danger-zone
CI test both depend on which points are in the sample.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lbi.gapmap import GapPoint, build_gap_map  # noqa: E402

WITHHELD = ("google/gemma-2-9b-it",)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src",
                    default="results nb7/combined_40point/gap_map.json")
    ap.add_argument("-o", "--out",
                    default="results nb7/combined_30point/gap_map.json")
    ap.add_argument("--withhold", nargs="*", default=list(WITHHELD))
    args = ap.parse_args()

    raw = json.load(open(args.src, encoding="utf-8"))
    kept = [p for p in raw["points"] if p["model"] not in set(args.withhold)]
    print("points: %d -> %d (withheld: %s)"
          % (len(raw["points"]), len(kept), ", ".join(args.withhold)))

    pts = [GapPoint(
        concept=p["concept"], model=p["model"],
        readability=p["readability"], controllability=p["controllability"],
        readability_ci=tuple(p["readability_ci"]),
        controllability_ci=tuple(p["controllability_ci"]),
        safety_relevant=p["safety_relevant"], best_layer=p["best_layer"],
        selectivity=p["selectivity"], gauntlet_passed=p.get("gauntlet_passed"),
    ) for p in kept]

    gm = build_gap_map(pts)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(gm.to_dict(), open(args.out, "w", encoding="utf-8"), indent=2)

    print("Spearman %.3f CI [%.3f, %.3f]"
          % (gm.spearman, gm.spearman_ci[0], gm.spearman_ci[1]))
    print("danger zone: %d | confirmed immovable: %d"
          % (len(gm.danger_zone), len(gm.confirmed_immovable)))
    for p in gm.confirmed_immovable:
        print("   confirmed:", p.concept, p.model)
    sat = sum(1 for p in pts if p.readability >= 0.999)
    print("readability at 1.00: %d/%d" % (sat, len(pts)))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
