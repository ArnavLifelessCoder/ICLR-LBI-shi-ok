"""Run stages, importable from a notebook on any platform.

The notebook drivers in `notebooks/` are thin wrappers that supply paths and
credentials; everything they actually do lives here so it is covered by tests
and reviewable as ordinary code rather than as cell output.

Stage order is not arbitrary. The positive control runs first and alone, because
PLAN.md Phase C is explicit that a broken harness discovered late costs the
paper, and GPU session time is the scarcest thing in this project.
"""

from __future__ import annotations

import os

from . import behavior as bh
from .concepts import all_concepts, audit_all_concepts, get_concept
from .pipeline import CONTROL_FLOOR, POSITIVE_CONTROL, run_model


def preflight(verbose: bool = True) -> bool:
    """CPU-only gate that must pass before any model is loaded.

    Returns False only for an *undeclared* audit failure. `verbosity` and
    `topic_science` are declared surface-confounded and are expected to fail;
    that is recorded in the builder, not discovered here.
    """
    results = audit_all_concepts()
    undeclared = []
    for r in results:
        tag = "PASS " if r.passed else ("FAIL*" if r.surface_confounded else "FAIL ")
        if verbose:
            print(f"  {r.concept:<14} {tag} mean={r.surface_auroc:.3f} "
                  f"worst={r.worst_fold_auroc:.3f}")
        if not r.passed and not r.surface_confounded:
            undeclared.append(r.concept)
    if verbose:
        print("  FAIL* = declared surface-confounded; expected, reported not hidden.")
        if undeclared:
            print(f"  UNDECLARED AUDIT FAILURE: {undeclared} -- fix before running.")
    return not undeclared


def build_judges(lm, max_unparsed_fraction: float = 0.2) -> bh.PanelScorer:
    """The judge panel. Never run the study on LexiconScorer alone.

    Two independent judges with reported agreement is a design-doc requirement
    and the lexicon scorer is a development instrument. The LLM judge reuses the
    model already in memory, so it costs generation time and no extra download.
    """
    llm = bh.LLMJudgeScorer(
        generate_fn=bh.make_local_generate_fn(lm),
        behavior_questions={c.name: c.behavior_question for c in all_concepts()},
        max_unparsed_fraction=max_unparsed_fraction,
    )
    return bh.PanelScorer({"llm": llm, "lexicon": bh.LexiconScorer()}, primary="llm")


def stage1_control(lm, out_dir: str, cache_dir: str) -> bool:
    """Run only the positive control. The go/no-go for this model.

    A model where sentiment will not steer scores low controllability on every
    concept, which puts the whole model in the danger zone and looks exactly
    like the paper's finding. `aggregate` withholds it (P9), but there is no
    point generating the data in the first place.
    """
    panel = build_judges(lm)
    runs = run_model(
        lm, panel, out_dir=out_dir, cache_dir=cache_dir,
        concepts=[get_concept(POSITIVE_CONTROL)],
    )
    value = runs[0].steering.controllability
    passed = value >= CONTROL_FLOOR
    ceiling_reason = runs[0].steering.ceiling_reason

    print(f"\ncontrol controllability = {value:.3f} (floor {CONTROL_FLOOR})")
    print(f"fluency ceiling: {ceiling_reason}")
    print(f"judge parse-failure rate: {panel.judges['llm'].failure_rate():.1%}")
    if passed:
        print("PASS -- steering works on this model, run the sweep.")
    else:
        print(
            "FAIL -- steering does not move sentiment. Do NOT run the sweep.\n"
            "  Check in order:\n"
            "   1. is the chat template being applied (tokenizer.chat_template)\n"
            "   2. does the coefficient grid reach far enough in RMS units\n"
            "   3. is the fluency ceiling hit before any behaviour change --\n"
            f"      this run says: {ceiling_reason}"
        )
    return passed


def stage2_full(lm, out_dir: str, cache_dir: str, label_sample: int = 100):
    """All concepts for one model, then the agreement report and label sheet."""
    panel = build_judges(lm)
    runs = run_model(lm, panel, out_dir=out_dir, cache_dir=cache_dir)

    for r in runs:
        flag = ""
        if r.gauntlet_passed is True:
            flag = "  [IMMOVABLE: survived the gauntlet]"
        elif r.gauntlet_passed is False:
            flag = f"  [moved: {r.gauntlet['verdict']}]"
        print(f"  {r.probe.concept:<14} read={r.probe.readability:.3f} "
              f"ctrl={r.steering.controllability:.3f} "
              f"ceiling={r.steering.max_usable_coeff:g}{flag}")

    agreement = panel.agreement()
    print(f"\njudge agreement: alpha={agreement['krippendorff_alpha']:.3f} "
          f"over {agreement['n_items']} items")
    print(f"judge parse-failure rate: {panel.judges['llm'].failure_rate():.1%}")

    samples = [
        (r.probe.concept, s)
        for r in runs for p in r.steering.curve for s in p.samples
    ][:label_sample]
    sheet = os.path.join(out_dir, "human_labels.csv")
    bh.write_labeling_sheet(
        sheet, samples, {c.name: c.behavior_question for c in all_concepts()}
    )
    print(f"\nhand-label {len(samples)} outputs in {sheet} (blanks allowed).")
    return runs, agreement


def diagnose_steering(lm, concept_name: str = POSITIVE_CONTROL,
                      coeffs=(-3.0, 0.0, 3.0), n_prompts: int = 2):
    """Print raw generations and raw judge replies at a few coefficients.

    For when the control returns controllability 0.000 and the result file does
    not say why. Three things can produce that number and they need different
    fixes: the model emits sensible text that steering genuinely does not move;
    the model emits garbage; or the judge returns a constant. Looking at the
    text separates them in about two minutes instead of a 20-minute re-run.
    """
    from . import steering as st
    from .probes import diff_of_means_direction, pair_texts
    from .extraction import capture_cached

    concept = get_concept(concept_name)
    prompts = concept.eval_prompts[:n_prompts]

    texts, labels = pair_texts(concept.pairs)
    acts = capture_cached(lm, texts, cache_dir="/tmp/diag", tag=f"{concept_name}_diag")
    layer = lm.n_layers // 2
    direction = diff_of_means_direction(acts[layer], labels)

    judge = bh.LLMJudgeScorer(
        generate_fn=bh.make_local_generate_fn(lm),
        behavior_questions={c.name: c.behavior_question for c in all_concepts()},
        max_unparsed_fraction=1.0,  # diagnosing, not gating
    )
    raw_judge = bh.make_local_generate_fn(lm)
    lex = bh.LexiconScorer()

    print(f"concept={concept_name}  layer={layer}  d_model={lm.d_model}")
    print(f"chat_template present: {bool(getattr(lm.tokenizer, 'chat_template', None))}\n")

    for c in coeffs:
        spec = None if c == 0.0 else st.SteeringSpec(
            direction=direction, layers=[layer], variant="add", coeff=c
        )
        outs = st.generate(lm, prompts, spec=spec, max_new_tokens=64)
        llm_scores = judge.score(outs, concept_name)
        lex_scores = lex.score(outs, concept_name)
        question = concept.behavior_question
        replies = raw_judge([
            f"{question}\n\nText:\n{o}\n\nRespond with only the number."
            for o in outs
        ])
        print(f"--- coeff {c:+.1f} ---")
        for o, ls, xs, rep in zip(outs, llm_scores, lex_scores, replies):
            print(f"  llm={ls:.2f} lex={xs:.2f} | judge said {rep.strip()[:40]!r}")
            print(f"  text: {o[:160]!r}\n")
