"""Analyse a validation run: ground truth, re-judge, and k sensitivity.

    python scripts/analyse_validation.py --dir validation_output

Stage A is the one that matters. On the ground-truth concepts the readout is
computed from the string by rule, so a signed area that fails to resolve cannot
be blamed on a judge. If signs resolve here and not on the judge-scored
concepts, the null elsewhere is the instrument. If they fail to resolve here
too, the diagnostic is underpowered at this prompt count, which is a different
finding and a smaller paper.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHORT = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "google/gemma-2-9b-it": "Gemma-2-9b",
}


def _area(xs, ys):
    if len(xs) < 2:
        return float("nan")
    span = xs[-1] - xs[0]
    return float(np.trapezoid(ys, xs) / span) if span > 0 else float("nan")



def _mean_ci_arrays(score_lists, n_boot=2000, seed=0):
    """Bootstrap CIs for the mean of each coefficient's generations."""
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


def summarise(rec):
    """Signed area, its propagated interval, and the two arms."""
    st = rec["steering"]
    usable = [c for c in rec.get("curve", []) if not c.get("broken")]
    if len(usable) < 2:
        return None
    usable.sort(key=lambda c: c["coeff"])
    xs = np.array([c["coeff"] for c in usable], float)
    b0 = st["baseline_behavior"]
    ys = np.array([c["behavior"] for c in usable], float) - b0
    # Prefer recomputing from the per-generation scores when the run saved
    # them. Files written before `DosePoint.scores` existed carry a
    # `behavior_ci` that is a percentile of the individual scores rather than
    # an interval for their mean, which is about sqrt(n) too wide on a judge
    # readout and degenerate on a sparse one. Those files cannot be corrected
    # offline, so they are used as they are and flagged.
    if all(c.get("scores") for c in usable):
        lo, hi = _mean_ci_arrays([c["scores"] for c in usable])
        ci_kind = "mean"
    else:
        lo = np.array([c["behavior_ci_low"] for c in usable], float)
        hi = np.array([c["behavior_ci_high"] for c in usable], float)
        ci_kind = "percentile (legacy; interval not comparable)"

    signed = _area(xs, np.sign(xs) * ys)
    pos, neg = xs > 0, xs < 0
    a_pos = _area(xs[pos], ys[pos]) if pos.sum() >= 2 else float("nan")
    a_neg = _area(xs[neg], -ys[neg]) if neg.sum() >= 2 else float("nan")
    mono = spearmanr(xs, ys)[0] if len(xs) > 2 else float("nan")

    rng = np.random.default_rng(0)
    se = np.maximum((hi - lo) / 3.92, 1e-9)
    draws = np.clip(rng.normal(np.array([c["behavior"] for c in usable], float)[None, :],
                               se[None, :], size=(4000, len(xs))), 0.0, 1.0)
    vals = np.array([_area(xs, np.sign(xs) * (d - b0)) for d in draws])
    ci = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    return dict(
        concept=rec["probe"]["concept"],
        model=SHORT.get(rec["probe"]["model"], rec["probe"]["model"]),
        read=rec["probe"]["readability"], abs_ctrl=st["controllability"],
        signed=signed, ci=ci, pos=a_pos, neg=a_neg, mono=mono,
        baseline=b0, n_usable=len(usable), n_grid=len(rec.get("curve", [])),
        ci_kind=ci_kind,
        judge=rec.get("judge_model"), degenerate=st.get("judge_degenerate"),
    )


def load_dir(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        b = os.path.basename(f)
        if b.endswith("_control.json") or b in ("gap_map.json", "geometry_predictor.json"):
            continue
        rec = json.load(open(f, encoding="utf-8"))
        if "probe" not in rec or "steering" not in rec:
            continue
        r = summarise(rec)
        if r:
            out.append(r)
    return out


def show(rows, title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    if not rows:
        print("  (nothing)")
        return
    print("%-14s %-13s %6s %7s %8s %-20s %7s %7s %6s"
          % ("concept", "model", "read", "abs", "signed", "signed 95% CI",
             "pos", "neg", "mono"))
    for r in sorted(rows, key=lambda q: (q["concept"], q["model"])):
        res = r["ci"][0] > 0 or r["ci"][1] < 0
        print("%-14s %-13s %6.2f %7.3f %+8.3f [%+.3f, %+.3f]%s %+7.3f %+7.3f %+6.2f"
              % (r["concept"], r["model"], r["read"], r["abs_ctrl"], r["signed"],
                 r["ci"][0], r["ci"][1], " *" if res else "  ",
                 r["pos"], r["neg"], r["mono"]))
    n = len(rows)
    res = [r for r in rows if r["ci"][0] > 0 or r["ci"][1] < 0]
    print("\n  resolved sign: %d/%d" % (len(res), n))
    for r in res:
        print("     %s %s  %+.3f" % (r["concept"], r["model"], r["signed"]))
    print("  baselines: " + ", ".join("%s %.2f" % (r["concept"], r["baseline"])
                                      for r in sorted(rows, key=lambda q: q["concept"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="validation_output")
    args = ap.parse_args()

    gt = load_dir(os.path.join(args.dir, "results_groundtruth"))
    rj = load_dir(os.path.join(args.dir, "results_rejudge"))

    show(gt, "STAGE A: ground truth (no judge; direction known by construction)")
    show(rj, "STAGE B: re-judge")

    # Stage B is only meaningful against the original judge's numbers.
    if rj:
        print("\n" + "=" * 78)
        print("STAGE B vs the original 1.5B judge")
        print("=" * 78)
        orig = {}
        for d in ["nb-6 results/results", "results nb7/results"]:
            for f in sorted(glob.glob(os.path.join(d, "*.json"))):
                b = os.path.basename(f)
                if b.endswith("_control.json") or b in ("gap_map.json", "geometry_predictor.json"):
                    continue
                rec = json.load(open(f, encoding="utf-8"))
                r = summarise(rec)
                if r:
                    orig[(r["concept"], r["model"])] = r
        print("%-14s %-13s %17s %17s %10s" % ("concept", "model", "abs (1.5B -> new)",
                                              "signed (1.5B -> new)", "sign kept"))
        for r in sorted(rj, key=lambda q: q["concept"]):
            o = orig.get((r["concept"], r["model"]))
            if not o:
                print("%-14s %-13s   (no original)" % (r["concept"], r["model"]))
                continue
            kept = np.sign(o["signed"]) == np.sign(r["signed"])
            print("%-14s %-13s   %.3f -> %.3f   %+.3f -> %+.3f   %s"
                  % (r["concept"], r["model"], o["abs_ctrl"], r["abs_ctrl"],
                     o["signed"], r["signed"], "yes" if kept else "FLIPPED"))
        print("\n  judge used: %s" % {r["judge"] for r in rj})

    # k sweep
    ks = glob.glob(os.path.join(args.dir, "results_ksweep", "*.json"))
    if ks:
        print("\n" + "=" * 78)
        print("STAGE C: output overlap vs k")
        print("=" * 78)
        for f in ks:
            rows = json.load(open(f, encoding="utf-8"))
            keys = sorted((k for k in rows[0] if k.startswith("overlap_k")),
                          key=lambda s: int(s.split("k")[-1]))
            print("\n  " + os.path.basename(f))
            for r in rows:
                print("    %-14s " % r["concept"]
                      + "  ".join("%s %.3f" % (k.replace("overlap_", ""), r[k]) for k in keys))
            print("\n    rank correlation between k columns:")
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    x = [r[keys[i]] for r in rows]
                    y = [r[keys[j]] for r in rows]
                    print("      %-6s vs %-6s  %+.3f"
                          % (keys[i].replace("overlap_", ""),
                             keys[j].replace("overlap_", ""), spearmanr(x, y)[0]))


if __name__ == "__main__":
    main()
