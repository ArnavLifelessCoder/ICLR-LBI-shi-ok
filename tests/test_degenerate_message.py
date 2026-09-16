"""A constant score means opposite things depending on what produced it.

From an LLM judge it is an instrument fault and the number is worthless. From a
rule computed on the generated string it is a measurement: the behavior did not
occur at any coefficient, and the zero is real. The Gemma ground-truth run
reported the second as though it were the first, which reads as a broken run
rather than a result.
"""

from __future__ import annotations

import inspect

from lbi import pipeline


def _guard_source() -> str:
    src = inspect.getsource(pipeline.run_steering)
    start = src.index("judge_degenerate = ")
    return src[start: start + 2000]


def test_guard_distinguishes_a_rule_from_a_judge():
    src = _guard_source()
    assert "DeterministicScorer" in src, (
        "the guard does not branch on the scorer, so a rule-based readout is "
        "still reported as a judge failure"
    )


def test_rule_branch_does_not_call_itself_a_judge():
    """The whole defect was the word 'judge' on a run that had none."""
    src = _guard_source()
    det = src.index("deterministic = ")
    end = src.index("else:", det)
    rule_branch = src[det:end]
    assert "the judge returned" not in rule_branch
    assert "readout" in rule_branch


def test_rule_branch_does_not_claim_the_zero_is_meaningless():
    """For a rule, 0.000 is the answer, not a symptom."""
    src = _guard_source()
    det = src.index("deterministic = ")
    end = src.index("else:", det)
    rule_branch = src[det:end]
    assert "nothing to do with steering" not in rule_branch
    assert "legitimately" in rule_branch or "measurement" in rule_branch


def test_judge_branch_still_warns():
    """The original guard must survive for the case it was written for."""
    src = _guard_source()
    idx = src.index("else:")
    judge_branch = src[idx:]
    assert "WARNING" in judge_branch
    assert "the judge returned" in judge_branch
