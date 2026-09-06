"""Three figures answering the review: the ceiling control, leave-one-concept-out,
and the directional metric.

    python scripts/make_review_figures.py

fig6 shows that the fluency ceiling does not explain the H1 sign, which is the
alternative explanation the metric invites. fig7 is the leave-one-concept-out
forest plot for the primary test. fig8 contrasts the absolute controllability
metric with its signed analogue, and is the reason the framing changed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

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


def signed_area(curve, baseline):
    """Directional analogue of the controllability AUC.

    Same trapezoid over the same usable points, but integrating
    sign(alpha) * (behavior - baseline) instead of its absolute value, so a
    concept scores positively only if positive coefficients push behavior up
    and negative coefficients push it down.
    """
    usable = [p for p in curve if not p.get("broken")]
    if len(usable) < 2:
        return float("nan")
    xs = np.array([p["coeff"] for p in usable], dtype=float)
    ys = np.array([p["behavior"] for p in usable], dtype=float) - baseline
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    span = xs[-1] - xs[0]
    if span <= 0:
        return float("nan")
    return float(np.trapezoid(np.sign(xs) * ys, xs) / span)


def load():
    rows = []
    for d in DIRS:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            b = os.path.basename(f)
            if b in ("gap_map.json", "geometry_predictor.json") or b.endswith("_control.json"):
                continue
            r = json.load(open(f, encoding="utf-8"))
            p, s, g = r["probe"], r.get("steering"), r.get("geometry")
            if not s or not g:
                continue
            curve = r.get("curve", [])
            rows.append(dict(
                concept=p["concept"], model=SHORT.get(p["model"], p["model"]),
                read=p["readability"], ctrl=s["controllability"],
                overlap=g.get("output_overlap"),
                maxc=s["max_usable_coeff"], reason=s["ceiling_reason"],
                n_usable=len([c for c in curve if not c.get("broken")]),
                n_grid=len(curve), curve=curve, baseline=s["baseline_behavior"],
                signed=signed_area(curve, s["baseline_behavior"]),
            ))
    return rows


def fig6_ceiling(rows, out):
    """The ceiling does not explain the inversion."""
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.9))

    ov = np.array([r["overlap"] for r in rows], dtype=float)
    mc = np.array([r["maxc"] for r in rows], dtype=float)
    rho = spearmanr(ov, mc)[0]

    # Jitter only the vertical axis: max usable coefficient takes two values.
    rng = np.random.default_rng(0)
    for r in rows:
        ax1.scatter(r["overlap"], r["maxc"] + rng.uniform(-0.08, 0.08),
                    marker=MARKERS[r["model"]], color=COLORS[r["model"]],
                    s=42, alpha=0.8, edgecolors="none")
    ax1.set_xlabel("output overlap")
    ax1.set_ylabel("max usable coefficient")
    ax1.set_yticks([2.0, 3.0])
    ax1.set_ylim(1.6, 3.4)
    ax1.set_title("(a) overlap does not truncate the sweep\n"
                  r"Spearman $\rho=%.3f$ ($p=%.2f$)" % (rho, spearmanr(ov, mc)[1]),
                  fontsize=10)
    ax1.grid(alpha=0.25, linestyle=":")
    ax1.text(0.03, 0.06, "37 of 40 points reach the\ngrid end without breaking",
             transform=ax1.transAxes, fontsize=8, style="italic", color="#444")

    # topic_science curves: flat to the end of the grid, not cut short.
    for r in sorted([r for r in rows if r["concept"] == "topic_science"],
                    key=lambda q: q["model"]):
        cur = sorted(r["curve"], key=lambda p: p["coeff"])
        xs = [p["coeff"] for p in cur]
        ys = [p["behavior"] for p in cur]
        ax2.plot(xs, ys, marker=MARKERS[r["model"]], color=COLORS[r["model"]],
                 label=r["model"], lw=1.5, ms=4.5, alpha=0.9)
        broken = [p for p in cur if p.get("broken")]
        if broken:
            ax2.scatter([p["coeff"] for p in broken], [p["behavior"] for p in broken],
                        marker="x", color="k", s=55, zorder=5, lw=1.4)
    ax2.set_xlabel(r"steering coefficient $\alpha$ (residual RMS units)")
    ax2.set_ylabel("judged behavior")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_title("(b) topic_science is flat, not truncated", fontsize=10)
    ax2.grid(alpha=0.25, linestyle=":")
    ax2.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    ax2.text(0.97, 0.30, "x marks a point excluded\nby the fluency ceiling",
             transform=ax2.transAxes, fontsize=7.5, ha="right",
             style="italic", color="#444")

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print("wrote", out)


def fig7_loco(rows, out):
    """Leave-one-concept-out forest plot for the primary test."""
    import matplotlib.pyplot as plt
    from lbi.geometry import _partial_spearman, _cluster_bootstrap_partial_spearman

    ov = np.array([r["overlap"] for r in rows], dtype=float)
    ct = np.array([r["ctrl"] for r in rows], dtype=float)
    rd = np.array([r["read"] for r in rows], dtype=float)
    cons = [r["concept"] for r in rows]

    entries = [("all 10 concepts",
                _partial_spearman(ov, ct, rd),
                _cluster_bootstrap_partial_spearman(ov, ct, rd, cons))]
    for c in sorted(set(cons)):
        m = np.array([x != c for x in cons])
        sub = [x for x in cons if x != c]
        entries.append(("drop " + c,
                        _partial_spearman(ov[m], ct[m], rd[m]),
                        _cluster_bootstrap_partial_spearman(ov[m], ct[m], rd[m], sub)))

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ys = np.arange(len(entries))[::-1]
    for y, (name, rho, ci) in zip(ys, entries):
        excl = ci[1] < 0 or ci[0] > 0
        col = "#b2182b" if excl else "#888888"
        lw, ms = (2.2, 7) if name.startswith("all") else (1.5, 5.5)
        ax.plot(ci, [y, y], color=col, lw=lw, solid_capstyle="round", alpha=0.85)
        ax.plot([rho], [y], "o", color=col, ms=ms, zorder=4)
    ax.axvline(0, color="k", lw=1, linestyle="--", alpha=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels([e[0] for e in entries], fontsize=9)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlabel(r"partial Spearman $\rho$ (output overlap vs controllability $\mid$ readability)")
    ax.set_title("Leave-one-concept-out: the sign is stable, the interval is not",
                 fontsize=10.5)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    n_excl = sum(1 for _, _, ci in entries[1:] if ci[1] < 0 or ci[0] > 0)
    # Leave a clear band under the last row for the caption line.
    ax.set_ylim(-1.6, len(entries) - 0.4)
    ax.text(0.02, 0.035,
            "every leave-one-out fit stays negative; the interval excludes zero "
            "in %d of 10\n(red = interval excludes zero)" % n_excl,
            transform=ax.transAxes, fontsize=8, style="italic", color="#444")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print("wrote", out)


def fig8_directional(rows, out):
    """The absolute metric is not measuring directional control."""
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    a = np.array([r["ctrl"] for r in rows], dtype=float)
    s = np.array([r["signed"] for r in rows], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))

    ax1.axhspan(-0.25, 0, color="#b2182b", alpha=0.07, zorder=0)
    lim = max(a.max(), abs(s).max()) * 1.12
    ax1.plot([0, lim], [0, lim], color="#666", lw=1, linestyle="--", zorder=1)
    for r in rows:
        ax1.scatter(r["ctrl"], r["signed"], marker=MARKERS[r["model"]],
                    color=COLORS[r["model"]], s=46, alpha=0.85,
                    edgecolors="none", zorder=3, label=r["model"])
    ax1.axhline(0, color="k", lw=1, zorder=2)
    ax1.set_xlim(0, lim)
    ax1.set_ylim(-0.13, lim)
    ax1.set_xlabel("controllability (absolute deviation)")
    ax1.set_ylabel("directional controllability (signed)")
    ax1.set_title(r"(a) the two axes are unrelated, Spearman $\rho=%.2f$"
                  % spearmanr(a, s)[0], fontsize=10)
    ax1.grid(alpha=0.25, linestyle=":")
    ax1.text(0.97, 0.05, "%d of 40 points move\nopposite to the direction"
             % int((s < 0).sum()), transform=ax1.transAxes, fontsize=8,
             ha="right", style="italic", color="#b2182b")
    ax1.text(0.35, 0.93, r"$y=x$: purely directional", transform=ax1.transAxes,
             fontsize=7.5, color="#666", rotation=0)
    handles, labels = ax1.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax1.legend(uniq.values(), uniq.keys(), fontsize=7.5, loc="upper left",
               framealpha=0.9)

    # Per-concept: how much of the measured effect is directional?
    concepts = sorted(set(r["concept"] for r in rows))
    frac = [np.nanmedian([abs(r["signed"]) / r["ctrl"] for r in rows
                          if r["concept"] == c and r["ctrl"] > 0]) for c in concepts]
    order = np.argsort(frac)
    ax2.barh([concepts[i] for i in order],
             [frac[i] for i in order], color="#4a7ba7", alpha=0.85)
    ax2.axvline(1.0, color="k", lw=1, linestyle="--")
    ax2.set_xlabel(r"median $|$signed$|$ / absolute")
    ax2.set_title("(b) share of the measured effect that is directional",
                  fontsize=10)
    ax2.set_xlim(0, 1.15)
    ax2.grid(axis="x", alpha=0.25, linestyle=":")
    ax2.tick_params(labelsize=8.5)
    ax2.text(1.02, 0.5, "purely\ndirectional", rotation=90, fontsize=7,
             va="center", color="#444")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="paper")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                         "text.usetex": False, "mathtext.default": "regular"})

    rows = load()
    print("loaded %d points, %d concepts" % (rows.__len__(),
                                             len(set(r["concept"] for r in rows))))
    fig6_ceiling(rows, os.path.join(args.outdir, "fig6_ceiling_control.png"))
    fig7_loco(rows, os.path.join(args.outdir, "fig7_loco.png"))
    fig8_directional(rows, os.path.join(args.outdir, "fig8_directional.png"))


if __name__ == "__main__":
    main()
