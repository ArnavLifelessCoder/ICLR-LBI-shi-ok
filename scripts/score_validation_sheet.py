"""Score the round-3 validation sheet against the criterion in VALIDATION_PREREG.md.

    python scripts/score_validation_sheet.py human_labels_v3.csv

Joins the filled sheet to the hidden key, computes within-concept Krippendorff
alpha (interval) between the human and the 3B judge with a bootstrap interval,
checks both raters vary, and applies the preregistered verdict. It then sets
each concept's verdict beside its matched-protocol resolution count, which is
the comparison the round exists for, and writes a LaTeX table for the appendix.

The criterion is not a parameter. It was fixed in VALIDATION_PREREG.md before
any label existed, and changing it here after seeing the labels would defeat
the point of having written it down.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lbi import behavior as bh  # noqa: E402

# Fixed in VALIDATION_PREREG.md. Do not tune.
ALPHA_VALID = 0.667
ALPHA_STRONG = 0.8
MIN_SD = 0.11
ALLOWED = {0.0, 0.25, 0.5, 0.75, 1.0}
N_BOOT = 2000


def read_labels(sheet: str, key: str) -> list[dict]:
    with open(key, encoding="utf-8") as f:
        k = {int(r["index"]): r for r in csv.DictReader(f)}
    rows: list[dict] = []
    bad: list[str] = []
    with open(sheet, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            raw = (r.get("score") or "").strip()
            if raw == "":
                continue
            try:
                v = float(raw)
            except ValueError:
                bad.append("%s:%r" % (r["index"], raw))
                continue
            if v not in ALLOWED:
                bad.append("%s:%r" % (r["index"], raw))
                continue
            i = int(r["index"])
            if i not in k or k[i]["concept"] != r["concept"]:
                raise ValueError("row %d does not match the key" % i)
            rows.append({"index": i, "concept": r["concept"], "human": v,
                         "judge": float(k[i]["judge"]), "model": k[i]["model"],
                         "coeff": float(k[i]["coeff"])})
    if bad:
        raise ValueError("scores must be one of 0, 0.25, 0.5, 0.75, 1; "
                         "fix these rows: " + ", ".join(bad[:10]))
    return rows


def alpha(h: list[float], j: list[float]) -> float:
    return float(bh.krippendorff_alpha_interval([list(h), list(j)]))


def boot_ci(h: np.ndarray, j: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(h)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        hh, jj = h[idx], j[idx]
        if np.ptp(hh) == 0 and np.ptp(jj) == 0:
            continue
        a = alpha(hh.tolist(), jj.tolist())
        if np.isfinite(a):
            vals.append(a)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def verdict(a: float, sd_h: float, sd_j: float) -> str:
    if sd_h < MIN_SD and sd_j < MIN_SD:
        return "both flat, uninformative"
    if sd_h < MIN_SD:
        return "human flat, uninformative"
    if sd_j < MIN_SD:
        return "judge flat, uninformative"
    if a >= ALPHA_STRONG:
        return "VALIDATED (strong)"
    if a >= ALPHA_VALID:
        return "VALIDATED"
    return "not validated"


def resolution_counts(result_dir: str) -> dict[str, tuple[int, int]]:
    """Matched-protocol resolution per concept, from the stage G files."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from analyse_validation import summarise
    out: dict[str, list[int]] = {}
    for f in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        if f.endswith("_control.json"):
            continue
        r = json.load(open(f, encoding="utf-8"))
        if "probe" not in r:
            continue
        s = summarise(r)
        if not s:
            continue
        k = out.setdefault(s["concept"], [0, 0])
        k[1] += 1
        if s["ci"][0] > 0 or s["ci"][1] < 0:
            k[0] += 1
    return {c: (v[0], v[1]) for c, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet")
    ap.add_argument("--key", default="human_labels_v3_KEY_do_not_open.csv")
    ap.add_argument("--results", default="validation_output/results_judge_matched")
    ap.add_argument("--tex", default="paper/_table_judge3b.tex")
    ap.add_argument("--json", default="validation_output/judge3b_validation.json")
    args = ap.parse_args()

    rows = read_labels(args.sheet, args.key)
    res = resolution_counts(args.results)
    concepts = sorted({r["concept"] for r in rows})

    print("criterion (VALIDATION_PREREG.md): alpha >= %.3f and both sd >= %.2f"
          % (ALPHA_VALID, MIN_SD))
    print()
    print("%-11s %3s %8s %8s %8s %18s %8s  %s"
          % ("concept", "n", "human sd", "judge sd", "alpha", "95% CI",
             "resolves", "verdict"))
    summary = {}
    for c in concepts:
        sub = [r for r in rows if r["concept"] == c]
        h = np.array([r["human"] for r in sub])
        j = np.array([r["judge"] for r in sub])
        a = alpha(h.tolist(), j.tolist())
        lo, hi = boot_ci(h, j)
        sdh, sdj = float(h.std()), float(j.std())
        v = verdict(a, sdh, sdj)
        k, n = res.get(c, (0, 0))
        summary[c] = dict(n=len(sub), human_sd=sdh, judge_sd=sdj, alpha=a,
                          ci=[lo, hi], verdict=v, resolved=k, of=n)
        print("%-11s %3d %8.3f %8.3f %+8.3f  [%+.3f, %+.3f] %5d/%-2d  %s"
              % (c, len(sub), sdh, sdj, a, lo, hi, k, n, v))

    print()
    validated = [c for c in concepts if summary[c]["verdict"].startswith("VALIDATED")]
    print("validated:", validated or "none")
    for c in validated:
        if c == "sentiment":
            continue
        k = summary[c]["resolved"]
        if k <= 1:
            print("  %s: validated and resolves on %d of 4 -> evidence that its "
                  "null reflects the concept, not the judge" % (c, k))
        elif k >= 3:
            print("  %s: validated and resolves on %d of 4 -> supports the "
                  "judge-limits account" % (c, k))
        else:
            print("  %s: validated and resolves on %d of 4 -> between the two "
                  "accounts" % (c, k))
    if "sentiment" not in validated:
        print("  sentiment does NOT validate under the 3B judge: the statement that "
              "the judge resolves direction where it is validated loses its "
              "only example and is withdrawn for the matched analysis")

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump({"criterion": {"alpha_valid": ALPHA_VALID, "min_sd": MIN_SD},
                   "concepts": summary}, f, indent=2)

    B = chr(92)
    lines = [
        B + "begin{table}[htbp]",
        "  " + B + "caption{Human against the 3B judge used in the matched "
        "comparison, on generations from those runs. Criterion fixed before "
        "labelling: alpha at least $0.667$ with both raters varying (sd at "
        "least $0.11$). Resolved is the matched-protocol count.}",
        "  " + B + "label{tab:judge3b}",
        "  " + B + "centering",
        "  " + B + "small",
        "  " + B + "begin{tabular}{lcccccl}",
        "    " + B + "toprule",
        "    Concept & $n$ & human sd & judge sd & $" + B + "alpha$ (95" + B
        + "% CI) & Resolved & Verdict " + B + B,
        "    " + B + "midrule",
    ]
    for c in concepts:
        s = summary[c]
        lines.append("    " + B + "texttt{%s} & %d & %.2f & %.2f & $%+.2f$ "
                     "$[%+.2f, %+.2f]$ & %d/%d & %s " % (
                         c.replace("_", B + "_"), s["n"], s["human_sd"],
                         s["judge_sd"], s["alpha"], s["ci"][0], s["ci"][1],
                         s["resolved"], s["of"], s["verdict"]) + B + B)
    lines += ["    " + B + "bottomrule", "  " + B + "end{tabular}",
              B + "end{table}"]
    with open(args.tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %s and %s" % (args.json, args.tex))
    return 0


if __name__ == "__main__":
    sys.exit(main())
