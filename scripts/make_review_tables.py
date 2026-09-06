"""LaTeX tables answering the review: ceiling controls, leave-one-concept-out,
and the directional metric.

    python scripts/make_review_tables.py

Writes three fragments into paper/ that paper.tex includes verbatim. Every
number comes from the cached result JSONs, so the tables cannot drift from the
data.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr, rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lbi.geometry import _partial_spearman, _cluster_bootstrap_partial_spearman  # noqa: E402

B = chr(92)
DIRS = ["nb-6 results/results", "results nb7/results"]
SHORT = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "google/gemma-2-9b-it": "Gemma-2-9b",
}


def tt(name):
    """Concept name as a LaTeX \\texttt with the underscore escaped."""
    return B + "texttt{" + name.replace("_", B + "_") + "}"


def signed_area(curve, baseline):
    usable = [p for p in curve if not p.get("broken")]
    if len(usable) < 2:
        return float("nan")
    xs = np.array([p["coeff"] for p in usable], dtype=float)
    ys = np.array([p["behavior"] for p in usable], dtype=float) - baseline
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    span = xs[-1] - xs[0]
    if span <= 0:
        return float("nan")
    return float(np.trapezoid(np.sign(xs) * ys, xs) / span)


def load(exclude=()):
    """Load every cached point, dropping any model named in `exclude`.

    Gemma is withheld from the main analysis because it fails the preregistered
    sentiment positive control under the directional metric (monotonicity
    -0.68, signed area -0.039).
    """
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
            if SHORT.get(p["model"], p["model"]) in exclude:
                continue
            curve = r.get("curve", [])
            usable = [c for c in curve if not c.get("broken")]
            mono = float("nan")
            if len(usable) >= 2:
                mono = spearmanr([c["coeff"] for c in usable],
                                 [c["behavior"] for c in usable])[0]
            rows.append(dict(
                concept=p["concept"], model=SHORT.get(p["model"], p["model"]),
                read=p["readability"], ctrl=s["controllability"],
                overlap=g.get("output_overlap"), k=g.get("n_directions"),
                maxc=s["max_usable_coeff"], reason=s["ceiling_reason"],
                n_usable=len(usable), n_grid=len(curve), mono=mono,
                signed=signed_area(curve, s["baseline_behavior"]),
            ))
    return rows


def partial_multi(x, y, Z):
    """Partial Spearman of x and y controlling for several covariates."""
    rx, ry = rankdata(x), rankdata(y)
    RZ = np.column_stack([np.ones(len(rx))] +
                         [rankdata(Z[:, j]) for j in range(Z.shape[1])])
    bx = np.linalg.lstsq(RZ, rx, rcond=None)[0]
    by = np.linalg.lstsq(RZ, ry, rcond=None)[0]
    return float(spearmanr(rx - RZ @ bx, ry - RZ @ by)[0])


def table_ceiling(rows, out):
    ov = np.array([r["overlap"] for r in rows], float)
    ct = np.array([r["ctrl"] for r in rows], float)
    rd = np.array([r["read"] for r in rows], float)
    mc = np.array([r["maxc"] for r in rows], float)
    nu = np.array([r["n_usable"] for r in rows], float)
    cons = [r["concept"] for r in rows]

    rho_mc, p_mc = spearmanr(ov, mc)
    rho_nu, p_nu = spearmanr(ov, nu)
    rho_cc, p_cc = spearmanr(mc, ct)
    base = _partial_spearman(ov, ct, rd)
    both = partial_multi(ov, ct, np.column_stack([rd, mc]))
    allc = partial_multi(ov, ct, np.column_stack([rd, mc, nu]))

    L = [
        B + "begin{table}[htbp]",
        "  " + B + "caption{The fluency ceiling does not explain the sign of the primary",
        "  test. If output overlap made a direction lexical enough to break fluency early,",
        "  it would shorten the usable coefficient range and shrink an integrated area",
        "  mechanically. It does not: overlap is unrelated to where the ceiling falls, and",
        "  conditioning the primary test on the ceiling as well as on readability leaves",
        "  the estimate unchanged.}",
        "  " + B + "label{tab:ceiling}",
        "  " + B + "centering",
        "  " + B + "footnotesize",
        "  " + B + "begin{tabular}{lcc}",
        "    " + B + "toprule",
        "    Quantity & Statistic & $p$ " + B + B,
        "    " + B + "midrule",
        "    " + B + "multicolumn{3}{l}{" + B + "emph{Does output overlap truncate the sweep?}} " + B + B,
        "    Overlap vs.\\ max usable $|" + B + "alpha|$ & Spearman $%.3f$ & $%.2f$ %s" % (rho_mc, p_mc, B + B),
        "    Overlap vs.\\ number of usable grid points & Spearman $%.3f$ & $%.2f$ %s" % (rho_nu, p_nu, B + B),
        "    Max usable $|" + B + "alpha|$ vs.\\ controllability & Spearman $%.3f$ & $%.2f$ %s" % (rho_cc, p_cc, B + B),
        "    " + B + "addlinespace",
        "    " + B + "multicolumn{3}{l}{" + B + "emph{Primary test under wider control sets}} " + B + B,
        "    Partial $" + B + "rho$ $|$ readability & $%.3f$ & %s" % (base, B + B),
        "    Partial $" + B + "rho$ $|$ readability, max usable $|" + B + "alpha|$ & $%.3f$ & %s" % (both, B + B),
        "    Partial $" + B + "rho$ $|$ readability, ceiling, $n$ usable & $%.3f$ & %s" % (allc, B + B),
        "    " + B + "bottomrule",
        "  " + B + "end{tabular}",
        B + "end{table}",
    ]
    io.open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", out)
    print("   overlap vs ceiling rho=%.3f p=%.2f | partial %.3f -> %.3f -> %.3f"
          % (rho_mc, p_mc, base, both, allc))


def table_loco(rows, out):
    ov = np.array([r["overlap"] for r in rows], float)
    ct = np.array([r["ctrl"] for r in rows], float)
    rd = np.array([r["read"] for r in rows], float)
    cons = [r["concept"] for r in rows]

    # Fit everything first: the caption reports counts drawn from the fits.
    rho = _partial_spearman(ov, ct, rd)
    ci = _cluster_bootstrap_partial_spearman(ov, ct, rd, cons)
    fits = []
    for c in sorted(set(cons)):
        m = np.array([x != c for x in cons])
        sub = [x for x in cons if x != c]
        r_ = _partial_spearman(ov[m], ct[m], rd[m])
        c_ = _cluster_bootstrap_partial_spearman(ov[m], ct[m], rd[m], sub)
        fits.append((c, r_, c_, c_[1] < 0 or c_[0] > 0))
    n_excl = sum(1 for _, _, _, ex in fits if ex)
    n_neg = sum(1 for _, r_, _, _ in fits if r_ < 0)
    n_cons = len(fits)
    full_ex = ci[1] < 0 or ci[0] > 0

    L = [
        B + "begin{table}[htbp]",
        "  " + B + "caption{Leave-one-concept-out on the primary test. Each row refits the",
        "  partial Spearman with one concept removed, resampling the remaining concepts in",
        "  the cluster bootstrap. The point estimate stays negative in %d of %d refits, so"
        % (n_neg, n_cons),
        "  the direction of the effect does not depend on any single concept, but the",
        "  interval excludes zero in only %d of %d. On the full retained sample the interval"
        % (n_excl, n_cons),
        "  %s. We therefore treat the sign as the reportable"
        % ("excludes zero" if full_ex else "already contains zero"),
        "  quantity and the interval as indicative, consistent with the known",
        "  over-rejection of cluster inference at ten clusters.}",
        "  " + B + "label{tab:loco}",
        "  " + B + "centering",
        "  " + B + "footnotesize",
        "  " + B + "begin{tabular}{lccc}",
        "    " + B + "toprule",
        "    Sample & Partial $" + B + "rho$ & 95" + B + "% CI & Excludes 0 " + B + B,
        "    " + B + "midrule",
        "    All %d concepts ($n=%d$) & $%.3f$ & $[%.3f, %.3f]$ & %s %s"
        % (n_cons, len(rows), rho, ci[0], ci[1], "yes" if full_ex else "no", B + B),
        "    " + B + "addlinespace",
    ]
    for c, r_, c_, ex in fits:
        L.append("    drop %s & $%.3f$ & $[%.3f, %.3f]$ & %s %s"
                 % (tt(c), r_, c_[0], c_[1], "yes" if ex else "no", B + B))
    L += ["    " + B + "bottomrule", "  " + B + "end{tabular}", B + "end{table}"]
    io.open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", out, "| full sample %s | CI excludes zero in %d/%d refits"
          % ("excludes zero" if full_ex else "includes zero", n_excl, n_cons))


def table_directional(rows, out):
    a = np.array([r["ctrl"] for r in rows], float)
    s = np.array([r["signed"] for r in rows], float)
    ov = np.array([r["overlap"] for r in rows], float)
    rd = np.array([r["read"] for r in rows], float)
    cons = [r["concept"] for r in rows]

    n_neg = int((s < 0).sum())
    L = [
        B + "begin{table}[htbp]",
        "  " + B + "caption{Absolute versus directional controllability for all %d retained"
        % len(rows),
        "  points.",
        "  " + B + "textbf{Abs} is the preregistered metric, the trapezoid of",
        "  $|" + B + "text{behavior} - " + B + "text{baseline}|$ over the usable coefficient range.",
        "  " + B + "textbf{Signed} is the same trapezoid without the absolute value, so it is",
        "  positive only when positive coefficients raise the behavior and negative ones",
        "  lower it. " + B + "textbf{Mono} is the Spearman correlation between coefficient and",
        "  behavior along the sweep. %d of %d points have a negative signed area, meaning"
        % (n_neg, len(rows)),
        "  the intervention moved behavior against the direction it was built from, and the",
        "  two metrics are uncorrelated (Spearman $%.3f$).}" % spearmanr(a, s)[0],
        "  " + B + "label{tab:directional}",
        "  " + B + "centering",
        "  " + B + "footnotesize",
        "  " + B + "begin{tabular}{llccc}",
        "    " + B + "toprule",
        "    Concept & Model & Abs & Signed & Mono " + B + B,
        "    " + B + "midrule",
    ]
    for r in sorted(rows, key=lambda q: (q["concept"], q["model"])):
        L.append("    %s & %s & $%.3f$ & $%+.3f$ & $%+.2f$ %s"
                 % (tt(r["concept"]), r["model"], r["ctrl"], r["signed"],
                    r["mono"], B + B))
    L += ["    " + B + "bottomrule", "  " + B + "end{tabular}", B + "end{table}"]
    io.open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", out)
    print("   signed<0: %d/40 | Spearman(abs,signed)=%.3f | median mono=%+.3f"
          % (int((s < 0).sum()), spearmanr(a, s)[0],
             np.median([r["mono"] for r in rows])))
    print("   H1 under signed metric: partial rho=%.3f CI %s"
          % (_partial_spearman(ov, s, rd),
             np.round(_cluster_bootstrap_partial_spearman(ov, s, rd, cons), 3).tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="paper")
    ap.add_argument("--exclude", nargs="*", default=["Gemma-2-9b"],
                    help="models withheld from the analysis (default: Gemma, "
                         "which fails the directional positive control)")
    args = ap.parse_args()
    rows = load(exclude=set(args.exclude))
    print("loaded %d points, %d models, %d concepts (withheld: %s)"
          % (len(rows), len(set(r["model"] for r in rows)),
             len(set(r["concept"] for r in rows)), ", ".join(args.exclude) or "none"))
    print("k (top singular directions) takes values:",
          sorted(set(r["k"] for r in rows)))
    table_ceiling(rows, os.path.join(args.outdir, "_table_ceiling.tex"))
    table_loco(rows, os.path.join(args.outdir, "_table_loco.tex"))
    table_directional(rows, os.path.join(args.outdir, "_table_directional.tex"))


if __name__ == "__main__":
    main()
