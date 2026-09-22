"""Uncertainty and arm-split for the directional metric.

    python scripts/signed_uncertainty.py

Two things the review asked for, both computable from the cached curves.

Intervals. Per-prompt behavior scores are not cached, but each coefficient
carries a prompt-level bootstrap interval. We propagate those into the signed
area by Monte Carlo: draw a behavior value per coefficient from a normal with
that interval's half-width as 1.96 standard errors, recompute the area, repeat.
The draws are independent across coefficients, while the real scores share a
prompt set and are positively correlated, and correlated errors partly cancel in
a difference from baseline. The intervals below are therefore conservative,
which is the safe direction for a claim that a sign is resolved.

Arm split. Signed area assumes a response that is monotone through the
baseline. Figure 3 shows the sentiment control is not: its baseline sits near an
extreme and both directions move behavior the same way. Reporting the positive
and negative arms separately distinguishes a concept that responds on one side
only from one that responds randomly.
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


def _area(xs, ys):
    """Trapezoid normalized by span, the same shape as the reported metric."""
    if len(xs) < 2:
        return float("nan")
    span = xs[-1] - xs[0]
    return float(np.trapezoid(ys, xs) / span) if span > 0 else float("nan")


def signed(xs, ys):
    """Directional area: sign(alpha) * (behavior - baseline)."""
    return _area(xs, np.sign(xs) * ys)


def arms(xs, ys):
    """Directional area over each arm separately, both oriented so that
    positive means the model moved the way the direction intended."""
    pos, neg = xs > 0, xs < 0
    a_pos = _area(xs[pos], ys[pos]) if pos.sum() >= 2 else float("nan")
    a_neg = _area(xs[neg], -ys[neg]) if neg.sum() >= 2 else float("nan")
    return a_pos, a_neg


def load(exclude=()):
    rows = []
    for d in DIRS:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            b = os.path.basename(f)
            if b in ("gap_map.json", "geometry_predictor.json") or b.endswith("_control.json"):
                continue
            r = json.load(open(f, encoding="utf-8"))
            p, st = r["probe"], r.get("steering")
            if not st:
                continue
            model = SHORT.get(p["model"], p["model"])
            if model in exclude:
                continue
            usable = [c for c in r.get("curve", []) if not c.get("broken")]
            if len(usable) < 2:
                continue
            usable.sort(key=lambda c: c["coeff"])
            rows.append(dict(
                concept=p["concept"], model=model,
                abs_ctrl=st["controllability"], baseline=st["baseline_behavior"],
                xs=np.array([c["coeff"] for c in usable], float),
                ys=np.array([c["behavior"] for c in usable], float),
                **dict(zip(("lo", "hi"),
                           _mean_ci_arrays([c["scores"] for c in usable])
                           if all(c.get("scores") for c in usable)
                           else (np.array([c["behavior_ci_low"] for c in usable], float),
                                 np.array([c["behavior_ci_high"] for c in usable], float)))),
            ))
    return rows



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


def bootstrap(row, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    xs, b0 = row["xs"], row["baseline"]
    se = np.maximum((row["hi"] - row["lo"]) / 3.92, 1e-9)
    draws = rng.normal(row["ys"][None, :], se[None, :], size=(n_boot, len(xs)))
    draws = np.clip(draws, 0.0, 1.0)  # the judge emits a probability
    vals = np.array([signed(xs, d - b0) for d in draws])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=["Gemma-2-9b"])
    ap.add_argument("-o", "--out", default="signed_uncertainty.json")
    args = ap.parse_args()

    rows = load(exclude=set(args.exclude))
    print("%d points, %d models\n" % (len(rows), len(set(r["model"] for r in rows))))

    out = []
    print("%-14s %-13s %8s %8s %-18s %8s %8s" %
          ("concept", "model", "abs", "signed", "signed 95% CI", "pos arm", "neg arm"))
    for r in sorted(rows, key=lambda q: (q["concept"], q["model"])):
        ys = r["ys"] - r["baseline"]
        sg = signed(r["xs"], ys)
        lo, hi = bootstrap(r)
        ap_, an_ = arms(r["xs"], ys)
        resolved = lo > 0 or hi < 0
        out.append(dict(concept=r["concept"], model=r["model"], abs_ctrl=r["abs_ctrl"],
                        signed=sg, signed_ci=[lo, hi], signed_pos=ap_, signed_neg=an_,
                        sign_resolved=bool(resolved)))
        print("%-14s %-13s %8.3f %+8.3f [%+.3f, %+.3f]%s %+8.3f %+8.3f"
              % (r["concept"], r["model"], r["abs_ctrl"], sg, lo, hi,
                 " *" if resolved else "  ", ap_, an_))

    n = len(out)
    res = [o for o in out if o["sign_resolved"]]
    neg = [o for o in out if o["signed"] < 0]
    negres = [o for o in res if o["signed"] < 0]
    print("\n" + "=" * 72)
    print("signed area negative            : %d/%d" % (len(neg), n))
    print("sign resolved (CI excludes 0)   : %d/%d" % (len(res), n))
    print("  of those, negative            : %d" % len(negres))
    print("  of those, positive            : %d" % (len(res) - len(negres)))
    print()
    both = [o for o in out if o["signed_pos"] > 0 and o["signed_neg"] > 0]
    one = [o for o in out if (o["signed_pos"] > 0) != (o["signed_neg"] > 0)]
    none_ = [o for o in out if o["signed_pos"] <= 0 and o["signed_neg"] <= 0]
    print("arms: both correct %d | one arm only %d | neither %d" % (len(both), len(one), len(none_)))
    print("\nsentiment (the validated concept):")
    for o in out:
        if o["concept"] == "sentiment":
            print("   %-13s signed %+0.3f CI [%+.3f, %+.3f] pos %+0.3f neg %+0.3f %s"
                  % (o["model"], o["signed"], o["signed_ci"][0], o["signed_ci"][1],
                     o["signed_pos"], o["signed_neg"],
                     "RESOLVED" if o["sign_resolved"] else "unresolved"))
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
