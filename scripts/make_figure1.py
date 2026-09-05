"""Figure 1 for the paper, drawn from a saved gap_map.json.

The library's `plot_gap_map` labels every safety-relevant point, which is right
for inspecting a run and wrong for the paper: readability is saturated, so 31 of
40 points pile up at x = 1.0 and their labels land on top of each other. Here
only the danger-zone points are labelled, safety-relevant concepts are marked by
edge colour and explained in the caption, and the legend sits outside the axes.

    python scripts/make_figure1.py \
        --gap-map "results nb7/combined_40point/gap_map.json" \
        -o paper/fig1_gap_map.png
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

SHORT = {
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "gemma-2-9b-it": "Gemma-2-9b",
}
MARKERS = ["o", "s", "^", "D", "v", "P"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-map", default="results nb7/combined_40point/gap_map.json")
    ap.add_argument("-o", "--out", default="paper/fig1_gap_map.png")
    args = ap.parse_args()

    with open(args.gap_map, encoding="utf-8") as f:
        gm = json.load(f)
    pts = gm["points"]
    danger = set(gm.get("danger_zone", []))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    by_model = defaultdict(list)
    for p in pts:
        by_model[p["model"]].append(p)

    for i, model in enumerate(sorted(by_model)):
        group = by_model[model]
        xs = [p["norm_readability"] for p in group]
        ys = [p["norm_controllability"] for p in group]
        safety = [p["safety_relevant"] for p in group]
        ax.scatter(
            xs, ys,
            marker=MARKERS[i % len(MARKERS)],
            s=[74 if s else 46 for s in safety],
            edgecolors=["#c0392b" if s else "#333333" for s in safety],
            linewidths=[1.5 if s else 0.6 for s in safety],
            alpha=0.85, zorder=3,
            label=SHORT.get(model, model.split("/")[-1]),
        )

    # Only the danger-zone points carry a text label. Everything else is a
    # marker: with the readability axis saturated there is no room for more.
    for p in pts:
        key = p["concept"] + "@" + p["model"]
        if key not in danger:
            continue
        x, y = p["norm_readability"], p["norm_controllability"]
        ax.scatter([x], [y], s=230, facecolors="none", edgecolors="#c0392b",
                   linewidths=1.8, zorder=4)
        ax.annotate(
            p["concept"], (x, y), fontsize=9, color="#c0392b", zorder=5,
            xytext=(-14, 16), textcoords="offset points", ha="right",
            arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.8),
        )

    ax.scatter([], [], s=90, facecolors="none", edgecolors="#c0392b",
               linewidths=1.8, label="danger zone")
    ax.scatter([], [], s=74, facecolors="#999999", edgecolors="#c0392b",
               linewidths=1.5, label="safety-relevant")

    ax.set_xlabel("Readability (probe AUROC, normalized within model)")
    ax.set_ylabel("Controllability (dose-response area,\nnormalized within model)")
    ax.set_title(
        "Detection vs. control, 40 concept-model points\n"
        "Spearman rho = %.2f, 95%% CI [%.2f, %.2f]"
        % (gm["spearman"], gm["spearman_ci_low"], gm["spearman_ci_high"]),
        fontsize=10,
    )
    ax.set_xlim(-0.08, 1.12)
    ax.set_ylim(-0.10, 1.12)
    ax.grid(alpha=0.22, zorder=0)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, borderaxespad=0)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s (%d points, %d labelled)" % (args.out, len(pts), len(danger)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
