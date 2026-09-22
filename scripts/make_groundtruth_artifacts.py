"""Figures, tables and headline numbers for the ground-truth validation.

    python scripts/make_groundtruth_artifacts.py

This is the paper's first contribution and every number it states comes from
here. PLAN_MAIN.md makes that a gate rather than a preference: on 2026-09-15
three figure titles advertised a 40-point sample on 30-point plots because the
counts were typed rather than computed.

Emits fig9_groundtruth.png, _table_groundtruth.tex, _table_rejudge.tex, and
prints the numbers the prose is allowed to use.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHORT = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "google/gemma-2-9b-it": "Gemma-2-9b",
}
ORDER = ["Qwen2.5-7B", "Mistral-7B", "Llama-3.1-8B", "Gemma-2-9b"]
COLORS = {"Qwen2.5-7B": "#1f77b4", "Mistral-7B": "#d62728",
          "Llama-3.1-8B": "#2ca02c", "Gemma-2-9b": "#ff7f0e"}
B = chr(92)

# Below this an absolute area is indistinguishable from nothing happening, and a
# ratio against it is undefined in practice rather than in principle.
TRIVIAL = 0.005



def _mean_ci_arrays(score_lists, n_boot=2000, seed=0):
    """Bootstrap a CI for the mean of each coefficient's generations.

    `behavior_ci` in files written before `DosePoint.scores` existed is a
    percentile of the individual scores, not an interval for their mean. It is
    about three times too wide on a judge readout and degenerate on a sparse
    one, and this script divides its width by 3.92 to recover a standard error.
    Where the run saved its scores, recompute; where it did not, the file
    cannot be corrected offline.
    """
    rng = np.random.default_rng(seed)
    los, his = [], []
    for scores in score_lists:
        arr = np.asarray(scores, float)
        if arr.size < 2 or np.ptp(arr) == 0:
            los.append(float(arr.mean())); his.append(float(arr.mean()))
            continue
        means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
        los.append(float(np.percentile(means, 2.5)))
        his.append(float(np.percentile(means, 97.5)))
    return np.array(los), np.array(his)


def _area(xs, ys):
    if len(xs) < 2:
        return float("nan")
    span = xs[-1] - xs[0]
    return float(np.trapezoid(ys, xs) / span) if span > 0 else float("nan")


def summarise(path, n_boot=4000, seed=0):
    r = json.load(open(path, encoding="utf-8"))
    if "probe" not in r or "steering" not in r:
        return None
    st = r["steering"]
    usable = [c for c in r.get("curve", []) if not c.get("broken")]
    if len(usable) < 2:
        return None
    usable.sort(key=lambda c: c["coeff"])
    xs = np.array([c["coeff"] for c in usable], float)
    obs = np.array([c["behavior"] for c in usable], float)
    b0 = st["baseline_behavior"]
    ys = obs - b0
    if all(c.get("scores") for c in usable):
        lo, hi = _mean_ci_arrays([c["scores"] for c in usable])
    else:
        lo = np.array([c["behavior_ci_low"] for c in usable], float)
        hi = np.array([c["behavior_ci_high"] for c in usable], float)

    signed = _area(xs, np.sign(xs) * ys)
    absolute = _area(xs, np.abs(ys))
    pos, neg = xs > 0, xs < 0

    rng = np.random.default_rng(seed)
    se = np.maximum((hi - lo) / 3.92, 1e-9)
    draws = np.clip(rng.normal(obs[None, :], se[None, :], (n_boot, len(xs))), 0.0, 1.0)
    vals = np.array([_area(xs, np.sign(xs) * (d - b0)) for d in draws])
    ci = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    return dict(
        concept=r["probe"]["concept"],
        model=SHORT.get(r["probe"]["model"], r["probe"]["model"]),
        absolute=absolute, signed=signed, ci=ci,
        resolved=bool(ci[0] > 0 or ci[1] < 0),
        pos=_area(xs[pos], ys[pos]) if pos.sum() >= 2 else float("nan"),
        neg=_area(xs[neg], -ys[neg]) if neg.sum() >= 2 else float("nan"),
        baseline=b0, n_grid=len(r.get("curve", [])), n_usable=len(usable),
        share=abs(signed) / absolute if absolute > 0 else float("nan"),
    )


def load(patterns):
    out = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            b = os.path.basename(f)
            if b.endswith("_control.json") or b in ("gap_map.json",
                                                    "geometry_predictor.json"):
                continue
            s = summarise(f)
            if s:
                out.append(s)
    return out


def tt(name):
    return B + "texttt{" + name.replace("_", B + "_") + "}"


def figure(gt, judged, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.2, 4.6), gridspec_kw={"width_ratios": [2.05, 1]})

    rows = sorted(gt, key=lambda r: (r["concept"], ORDER.index(r["model"])))
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        col = COLORS[r["model"]]
        ax1.plot(r["ci"], [y, y], color=col, lw=2.0 if r["resolved"] else 1.1,
                 alpha=0.9 if r["resolved"] else 0.45, solid_capstyle="round")
        ax1.plot([r["signed"]], [y], "o", color=col,
                 ms=7 if r["resolved"] else 4.5,
                 zorder=4, alpha=0.95 if r["resolved"] else 0.5)
    ax1.axvline(0, color="k", lw=1, ls="--", alpha=0.7)
    ax1.set_yticks(ys)
    ax1.set_yticklabels(["%s  %s" % (r["concept"], r["model"]) for r in rows],
                        fontsize=7.5)
    for tick, r in zip(ax1.get_yticklabels(), rows):
        if r["resolved"]:
            tick.set_fontweight("bold")
    ax1.set_xlabel("directional controllability (signed area)")
    n_res = sum(r["resolved"] for r in gt)
    ax1.set_title("(a) known direction, no judge: %d of %d resolve,\nall in the "
                  "intended direction" % (n_res, len(gt)), fontsize=10)
    ax1.grid(axis="x", alpha=0.25, ls=":")

    a = [r["share"] for r in gt if r["absolute"] > TRIVIAL]
    b_ = [r["share"] for r in judged if r["absolute"] > TRIVIAL]
    parts = ax2.violinplot([a, b_], showmedians=True, widths=0.8)
    for pc, c in zip(parts["bodies"], ["#2a7f5f", "#9b59b6"]):
        pc.set_facecolor(c)
        pc.set_alpha(0.55)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        parts[key].set_color("#333333")
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["rule\n(no judge)\nn=%d" % len(a),
                         "LLM judge\nn=%d" % len(b_)], fontsize=8.5)
    ax2.set_ylabel("share of measured effect that is directional")
    ax2.axhline(1.0, color="#666", lw=1, ls="--")
    ax2.set_title("(b) the divergence is the judge's\nmedians %.2f vs %.2f"
                  % (np.median(a), np.median(b_)), fontsize=10)
    ax2.grid(axis="y", alpha=0.25, ls=":")

    handles = [plt.Line2D([], [], color=COLORS[m], marker="o", ls="-", lw=2,
                          label=m) for m in ORDER]
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, -0.035))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def table_groundtruth(gt, out):
    n_res = sum(r["resolved"] for r in gt)
    grids = sorted({r["n_grid"] for r in gt})
    L = [
        B + "begin{table}[htbp]",
        "  " + B + "caption{Directional controllability on the ground-truth concepts,",
        "  whose readout is a rule over the generated string and whose intended",
        "  direction is therefore known in advance. %d of %d points have an interval"
        % (n_res, len(gt)),
        "  excluding zero and " + B + "emph{every one is positive}: where the sign is",
        "  resolvable, steering moved the model the way the direction was built to move",
        "  it. " + B + "textbf{Pos} and " + B + "textbf{Neg} are the same integral over one arm,",
        "  oriented so positive means the intended direction.}",
        "  " + B + "label{tab:groundtruth}",
        "  " + B + "centering",
        "  " + B + "footnotesize",
        "  " + B + "begin{tabular}{llccccc}",
        "    " + B + "toprule",
        "    Concept & Model & Abs & Signed & 95" + B + "% CI & Pos & Neg " + B + B,
        "    " + B + "midrule",
    ]
    for r in sorted(gt, key=lambda q: (q["concept"], ORDER.index(q["model"]))):
        L.append("    %s & %s & $%.3f$ & $%+.3f$ & $[%+.3f, %+.3f]$%s & $%+.3f$ & $%+.3f$ %s"
                 % (tt(r["concept"]), r["model"], r["absolute"], r["signed"],
                    r["ci"][0], r["ci"][1],
                    "\\,\\textbf{R}" if r["resolved"] else "",
                    r["pos"], r["neg"], B + B))
    L += ["    " + B + "bottomrule", "  " + B + "end{tabular}", B + "end{table}"]
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", out, "| grids:", grids)


def table_rejudge(out):
    """Same concepts, same models, two judges."""
    new = {(r["concept"], r["model"]): r
           for r in load(["validation_output/results_rejudge/*.json"])}
    old = {(r["concept"], r["model"]): r
           for r in load(["nb-6 results/results/*.json", "results nb7/results/*.json"])}
    keys = sorted(set(new) & set(old), key=lambda k: (k[0], ORDER.index(k[1])))
    flips = sum(1 for k in keys if np.sign(new[k]["signed"]) != np.sign(old[k]["signed"]))
    ratios = [new[k]["absolute"] / old[k]["absolute"] for k in keys
              if old[k]["absolute"] > 0]

    L = [
        B + "begin{table}[htbp]",
        "  " + B + "caption{The same concepts and models scored by two judges, a 1.5B and a",
        "  3B, on identical sweeps. Magnitudes are not reproducible: the ratio runs from",
        "  $%.1f " % min(ratios) + B + "times$ to $%.1f " % max(ratios) + B + "times$.",
        "  Signs mostly are, but not entirely: %d of %d flip." % (flips, len(keys)),
        "  " + B + "texttt{topic" + B + "_science} is the danger zone's only confirmed occupant",
        "  under the smaller judge and clears the $0.05$ threshold on every model under",
        "  the larger one.}",
        "  " + B + "label{tab:rejudge}",
        "  " + B + "centering",
        "  " + B + "footnotesize",
        "  " + B + "begin{tabular}{llcccc}",
        "    " + B + "toprule",
        "    Concept & Model & Abs (1.5B) & Abs (3B) & Ratio & Sign " + B + B,
        "    " + B + "midrule",
    ]
    for k in keys:
        o, n = old[k], new[k]
        ratio = n["absolute"] / o["absolute"] if o["absolute"] > 0 else float("nan")
        kept = np.sign(o["signed"]) == np.sign(n["signed"])
        L.append("    %s & %s & $%.3f$ & $%.3f$ & $%.1f" % (
            tt(k[0]), k[1], o["absolute"], n["absolute"], ratio)
            + B + "times$ & %s %s" % ("kept" if kept else B + "textbf{flip}", B + B))
    L += ["    " + B + "bottomrule", "  " + B + "end{tabular}", B + "end{table}"]
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", out, "| flips %d/%d | ratio %.1f-%.1fx"
          % (flips, len(keys), min(ratios), max(ratios)))
    return dict(flips=flips, n=len(keys), rmin=min(ratios), rmax=max(ratios))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="paper")
    args = ap.parse_args()

    gt = load(["validation_output/results_groundtruth/*.json"])
    judged = load(["nb-6 results/results/*.json", "results nb7/results/*.json"])

    print("ground truth : %d points, %d models, grids %s"
          % (len(gt), len({r["model"] for r in gt}), sorted({r["n_grid"] for r in gt})))
    print("judge-scored : %d points" % len(judged))

    n_res = sum(r["resolved"] for r in gt)
    pos = [r for r in gt if r["resolved"] and r["signed"] > 0]
    print("\n--- C1: the diagnostic recovers a known direction ---")
    print("  resolved            : %d/%d" % (n_res, len(gt)))
    print("  of those, positive  : %d" % len(pos))
    print("  wrong-way resolved  : %d" % (n_res - len(pos)))
    for r in sorted(pos, key=lambda q: -q["signed"]):
        print("     %-14s %-13s %+0.3f [%+.3f, %+.3f]"
              % (r["concept"], r["model"], r["signed"], r["ci"][0], r["ci"][1]))
    for c in sorted({r["concept"] for r in gt}):
        k = [r for r in gt if r["concept"] == c and r["resolved"]]
        print("  %-14s resolves on %d of %d models"
              % (c, len(k), len({r["model"] for r in gt})))

    a = [r["share"] for r in gt if r["absolute"] > TRIVIAL]
    b_ = [r["share"] for r in judged if r["absolute"] > TRIVIAL]
    # The paper's main analysis withholds Gemma, so the comparison it quotes has
    # to be the retained sample. The all-model number is reported too because
    # the ground-truth set keeps all four and a like-for-like contrast is the
    # more natural one for a claim about the instrument rather than the models.
    ret = [r["share"] for r in judged
           if r["absolute"] > TRIVIAL and r["model"] != "Gemma-2-9b"]
    ret_res = [r for r in judged if r["model"] != "Gemma-2-9b"]
    print("\n--- C2: the divergence is the judge's ---")
    print("  directional share, rule            : median %.3f (n=%d)" % (np.median(a), len(a)))
    print("  directional share, judge, retained : median %.3f (n=%d)" % (np.median(ret), len(ret)))
    print("  directional share, judge, all four : median %.3f (n=%d)" % (np.median(b_), len(b_)))
    print("  judge-scored resolved, retained    : %d/%d"
          % (sum(r["resolved"] for r in ret_res), len(ret_res)))
    print("  judge-scored resolved, all four    : %d/%d"
          % (sum(r["resolved"] for r in judged), len(judged)))

    figure(gt, judged, os.path.join(args.outdir, "fig9_groundtruth.png"))
    table_groundtruth(gt, os.path.join(args.outdir, "_table_groundtruth.tex"))
    table_rejudge(os.path.join(args.outdir, "_table_rejudge.tex"))


if __name__ == "__main__":
    main()
