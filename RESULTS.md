# Results

**One real run attempted; its positive control failed, so it yields no
reportable numbers. Everything else below is synthetic.**

That sentence stays at the top until a real run lands. The synthetic figures
exist to prove the plumbing recovers a planted answer; they are not evidence
about any language model and must never be quoted as if they were.

Session-by-session detail is in [notebooks/RUNLOG.md](notebooks/RUNLOG.md).

---

## Status

| Experiment | What it produces | State |
| --- | --- | --- |
| 1. Readability | Held-out-family probe AUROC per concept-model | awaiting first run |
| 2. Controllability | Dose-response AUC to the fluency ceiling | awaiting first run |
| 3. Gap map | Figure 1, danger-zone membership | awaiting first run |
| 4. Geometry predictor | H1 partial Spearman + 2 exploratory | awaiting first run |
| 5. Safety concepts | Read off Figure 1 (demoted, R14) | awaiting first run |
| 6. Scale sweep | Qwen 0.5B / 1.5B / 3B / 7B | awaiting first run |

## Phase A gate: surface-shortcut audit (real, CPU-only)

This one **is** a real result — it needs no model. Leave-one-family-out,
two-sided `max(auroc, 1 - auroc)`, mean over folds.

| Concept | Mean | Worst fold | Verdict |
| --- | --- | --- | --- |
| sentiment | 0.500 | 0.500 | PASS |
| formality | 0.500 | 0.500 | PASS |
| rudeness | 0.500 | 0.500 | PASS |
| sycophancy | 0.500 | 0.500 | PASS |
| refusal | 0.500 | 0.500 | PASS |
| honesty | 0.500 | 0.500 | PASS |
| factuality | 0.500 | 0.500 | PASS |
| certainty | 0.500 | 0.500 | PASS |
| topic_science | 0.833 | 1.000 | FAIL, declared surface-confounded |
| verbosity | 1.000 | 1.000 | FAIL, declared surface-confounded |

8/10 pass, 0 undeclared failures. Reproduce with `python -c "from lbi.concepts
import audit_all_concepts; ..."` or read it off `scripts/demo_synthetic.py`.

**Read 0.500 as the floor, not as a good score.** Marker vocabulary is disjoint
across template families by build-time invariant, so a TF-IDF model trained on
five families has no feature that fires on the sixth and its decision function
is constant there. The audit is a check that the construction succeeded. It
earns its keep by having failed loudly when it had not: it caught duplicated
template families in `topic_science`, markers shared across families in five
concepts, a two-token length imbalance that separated families at AUROC 0.94
with no marker word transferring, and a capitalisation cue aligned with the
label on `formality`.

The two declared failures are not excused. `verbosity` is length and
`topic_science` is domain vocabulary; neither can pass a lexical audit, both
still report `passed=False`, and a test pins that the flag cannot convert a
failure into a pass.

## Synthetic end-to-end check

`scripts/demo_synthetic.py`, planted ground truth, ~40 s, no GPU.

- Ground truth: planted `['honesty', 'refusal']` as readable-but-immovable →
  detected exactly `['honesty', 'refusal']`.
- Gap map over 20 points: Spearman rho = 0.087 [-0.403, 0.508].
- Danger zone: 4 points, all reported as *unsteerable under tested
  interventions* — the demo does not run the gauntlet, so none is entitled to
  "immovable".
- Primary test (H1, output_overlap): partial rho = 0.525 [-0.127, 0.863] over
  10 concepts → "suggestive but not strong".
- Exploratory E1 rho = -0.050, E2 rho = -0.577, both p_BH = 0.164.

These numbers are properties of the planted generator, not of anything.

---

## Real runs

### Qwen/Qwen2.5-7B-Instruct — 2026-08-29, Kaggle T4, 4-bit — CONTROL FAILED

| | |
| --- | --- |
| Control (sentiment) controllability | **0.000** (floor 0.10) |
| P9 verdict | **FAIL** — model withheld |
| Fluency ceiling | no breakage in swept range |
| Judge parse-failure rate | 0.0% |
| Commit | `685d8d6` |
| Wall clock | 21.7 min for the control alone |

**No numbers from this model are reportable.** A model whose control does not
move scores low controllability on every concept, which is indistinguishable
from the finding, so P9 withholds it.

Exactly 0.000 with all nine coefficients usable means the behaviour score
equalled baseline at *every* coefficient. With a 0.0% parse-failure rate the
leading hypothesis is a judge returning one constant parseable number; the
second is that `st.generate` was feeding raw instructions to an instruct model
with no chat template, so the scored text was not the behaviour the concept is
about. Both are now fixed and neither is confirmed as the cause — see
[notebooks/RUNLOG.md](notebooks/RUNLOG.md).

This is the positive control working as designed. It caught a broken harness
before nine more concepts were generated on top of it.

Each entry, once there is one, records: model, control controllability and
whether it cleared the P9 floor, the fluency `ceiling_reason`, judge
parse-failure rate and Krippendorff alpha, per-concept readability and
controllability, gauntlet verdicts, and the commit the run was made from.

A model whose positive control falls below the floor has its numbers
**withheld**, not explained away.
