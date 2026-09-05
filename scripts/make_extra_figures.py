"""Three further figures: the H1 inversion, the selectivity axis, and the judge check.

    python scripts/make_extra_figures.py

fig3 goes in the main text. The paper's second headline result, that the
geometric predictor has the wrong sign, was reported only as a number; this
draws it. fig4 and fig5 go in the appendix, where there is no page limit.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DIRS = ["nb-6 results/results", "results nb7/results"]
SHORT = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "google/gemma-2-9b-it": "Gemma-2-9b",
}
MARKERS = {"Qwen2.5-7B": "o", "Mistral-7B": "D", "Llama-3.1-8B": "^", "Gemma-2-9b": "s"}
COLORS = {"Qwen2.5-7B": "#1f77b4", "Mistral-7B": "#d62728",
          "Llama-3.1-8B": "#2ca02c", "Gemma-2-9b": "#ff7f0e"}


def load():
    rows = []
    for d in DIRS:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            b = os.path.basename(f)
            if b in ("gap_map.json", "geometry_predictor.json") or b.endswith("_control.json"):
                continue
            r = json.load(open(f, encoding="utf-8"))
            p, s, g = r["probe"], r["steering"], r["geometry"]
            rows.append(dict(
                concept=p["concept"], model=SHORT.get(p["model"], p["model"]),
                read=p["readability"], sel=p["selectivity"],
                ctrl=s["controllability"], overlap=g.get("output_overlap"),
            ))
    return rows


def fig3_inversion(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for m in sorted({r["model"] for r in rows}):
        g = [r for r in rows if r["model"] == m and r["overlap"] is not None]
        ax.scatter([r["overlap"] for r in g], [r["ctrl"] for r in g],
                   marker=MARKERS.get(m, "o"), color=COLORS.get(m), s=48,
                   edgecolors="black", linewidths=0.5, alpha=0.85, label=m, zorder=3)

    # topic_science drives the effect; mark it on every model.
    ts = [r for r in rows if r["concept"] == "topic_science" and r["overlap"]]
    ax.scatter([r["overlap"] for r in ts], [r["ctrl"] for r in ts],
               s=210, facecolors="none", edgecolors="#c0392b", linewidths=1.6, zorder=4)
    hi = max(ts, key=lambda r: r["overlap"])
    ax.annotate("topic_science:\nhighest overlap,\nlowest controllability",
                (hi["overlap"], hi["ctrl"]), fontsize=8.5, color="#c0392b",
                xytext=(-6, 34), textcoords="offset points", ha="right",
                arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.8))

    # the control, for contrast
    sen = [r for r in rows if r["concept"] == "sentiment" and r["overlap"]]
    hs = max(sen, key=lambda r: r["ctrl"])
    ax.annotate("sentiment (control)", (hs["overlap"], hs["ctrl"]), fontsize=8.5,
                color="#333333", xytext=(14, -4), textcoords="offset points",
                ha="left", va="center")
    ax.margins(y=0.12)

    ax.set_xlabel("Output overlap (projection into the top unembedding subspace)")
    ax.set_ylabel("Controllability\n(dose-response area)")
    ax.set_title("H1 predicts this slope is positive. It is negative.\n"
                 "Partial Spearman $-0.45$, 95% CI $[-0.72, -0.04]$, conditioning on readability",
                 fontsize=9.5)
    ax.grid(alpha=0.22, zorder=0)
    ax.legend(fontsize=8, loc="upper right", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig4_selectivity(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, key, name in ((a1, "read", "Readability (probe AUROC)"),
                          (a2, "sel", "Selectivity (AUROC minus control probe)")):
        for m in sorted({r["model"] for r in rows}):
            g = [r for r in rows if r["model"] == m]
            ax.scatter([r[key] for r in g], [r["ctrl"] for r in g],
                       marker=MARKERS.get(m, "o"), color=COLORS.get(m), s=40,
                       edgecolors="black", linewidths=0.4, alpha=0.85, label=m, zorder=3)
        ax.set_xlabel(name)
        ax.grid(alpha=0.22, zorder=0)
    a1.set_ylabel("Controllability")
    a1.set_title("The preregistered axis is saturated:\n31 of 40 points sit at AUROC 1.0",
                 fontsize=9)
    a2.set_title("Selectivity spreads the same points out", fontsize=9)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig5_judge(path_json, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.load(open(path_json, encoding="utf-8"))
    rows = d["variance_diagnosis"]["logit_judge"]
    rows = sorted(rows, key=lambda r: r["sd_human"])
    y = range(len(rows))
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.barh([i - 0.19 for i in y], [r["sd_human"] for r in rows], height=0.38,
            color="#4c72b0", label="human annotator")
    ax.barh([i + 0.19 for i in y], [r["sd_logit_judge"] for r in rows], height=0.38,
            color="#dd8452", label="logit judge")
    ax.axvline(0.11, color="#c0392b", ls="--", lw=1,
               label="below this a rater is effectively flat")
    ax.set_yticks(list(y))
    ax.set_yticklabels([r["concept"] for r in rows], fontsize=8.5)
    ax.set_xlabel("Standard deviation of scores within the concept")
    ax.set_title("Where the judge varies and the human does not, the variation is\n"
                 "not tracking anything a careful reader can see", fontsize=9.5)
    ax.grid(alpha=0.22, axis="x")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="paper")
    args = ap.parse_args()
    rows = load()
    print("loaded %d concept-model points" % len(rows))
    fig3_inversion(rows, os.path.join(args.outdir, "fig3_h1_inversion.png"))
    fig4_selectivity(rows, os.path.join(args.outdir, "fig4_selectivity.png"))
    fig5_judge("judge_agreement.json", os.path.join(args.outdir, "fig5_judge_variance.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
