"""Figure 2: the dissociation, one model, three concepts.

The design document imagined this as "steep line versus flat line". The real
control curve is not a monotone ramp -- on every model the unsteered baseline
sits near a local extremum and steering in either direction moves behaviour the
same way -- so drawing it as a ramp would be drawing something that is not
there. What the figure shows instead is the quantity controllability actually
integrates: absolute deviation from the unsteered baseline as steering strength
grows. On that axis the contrast is unambiguous and honest.

    python scripts/make_figure2.py -o paper/fig2_dose_response.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Concepts to show, in legend order, with the story each one carries.
SERIES = [
    ("sentiment", "sentiment (positive control): steering moves it"),
    ("refusal", "refusal (boundary case): barely moves"),
    ("topic_science", "topic_science (immovable): does not move"),
]
COLORS = {"sentiment": "#1f77b4", "refusal": "#ff7f0e", "topic_science": "#d62728"}
MARKERS = {"sentiment": "o", "refusal": "s", "topic_science": "^"}


def load(results_dir: str, model_file_stem: str, concept: str):
    path = os.path.join(results_dir, f"{model_file_stem}_{concept}.json")
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    curve = sorted(rec["curve"], key=lambda p: p["coeff"])
    return rec["steering"], curve


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results nb7/results")
    ap.add_argument("--model-stem", default="gemma-2-9b-it",
                    help="filename stem, e.g. gemma-2-9b-it")
    ap.add_argument("--model-label", default="Gemma-2-9b-it")
    ap.add_argument("-o", "--out", default="paper/fig2_dose_response.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    for concept, label in SERIES:
        steer, curve = load(args.results_dir, args.model_stem, concept)
        base = steer["baseline_behavior"]
        xs = [p["coeff"] for p in curve]
        ys = [p["behavior"] for p in curve]
        dev = [abs(y - base) for y in ys]
        kw = dict(color=COLORS[concept], marker=MARKERS[concept], markersize=5, lw=1.8)

        ax1.plot(xs, ys, label=label.replace("): ", f", AUC {steer['controllability']:.3f}): "), **kw)
        ax1.axhline(base, color=COLORS[concept], ls=":", lw=1, alpha=0.55)
        ax2.plot(xs, dev, **kw)

    ax1.set_xlabel("steering coefficient (residual-stream RMS units)")
    ax1.set_ylabel("behaviour score")
    ax1.set_title("Raw dose-response\n(dotted line = unsteered baseline)", fontsize=10)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.25)

    ax2.set_xlabel("steering coefficient (residual-stream RMS units)")
    ax2.set_ylabel("|behaviour - baseline|")
    ax2.set_title("Absolute deviation from baseline\n(what controllability integrates)",
                  fontsize=10)
    ax2.grid(alpha=0.25)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8.5, loc="lower center",
               ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"A legible concept that does not move, on {args.model_label}", fontsize=12
    )
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"wrote {args.out}")

    for concept, _ in SERIES:
        steer, curve = load(args.results_dir, args.model_stem, concept)
        ys = [p["behavior"] for p in curve]
        print(f"  {concept:<14} AUC={steer['controllability']:.3f}  "
              f"baseline={steer['baseline_behavior']:.2f}  "
              f"range={min(ys):.2f}-{max(ys):.2f}  span={max(ys)-min(ys):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
