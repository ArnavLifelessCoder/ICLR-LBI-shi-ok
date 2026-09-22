"""The thirty-prompt eval sets for the judge-scored concepts.

The study's headline comparison is that ground-truth concepts resolve their
signs and judge-scored ones mostly do not. The ground-truth concepts are
evaluated on thirty prompts and the judge-scored ones on six, so the comparison
confounded the readout with the sample size. These extend the judge-scored sets
to thirty.

What matters here is that the instrument is extended rather than swapped: the
original six must remain a strict prefix, so every number already reported is
recoverable by taking the first six. That is the same discipline EXTENDED_COEFFS
follows with DEFAULT_COEFFS, and it is the reason the old results can stay on
disk as the comparison instead of being invalidated.
"""

from __future__ import annotations

import pytest

from lbi.concepts import (
    _EXTRA_EVAL_PROMPTS,
    all_concepts,
    expanded_eval_prompts,
    with_expanded_eval_prompts,
)

# All ten judge-scored concepts now carry thirty, so stage G can measure
# them exactly as stage A measures the rule-scored ten and the headline
# comparison is forty points against forty rather than forty against
# sixteen. The four re-judged ones were expanded first, which is why an
# earlier version of this list had only them.
REJUDGE = sorted(_EXTRA_EVAL_PROMPTS)


def _by_name():
    return {c.name: c for c in all_concepts()}


@pytest.mark.parametrize("name", REJUDGE)
def test_every_rejudged_concept_reaches_thirty(name):
    """Stage E refuses to run below thirty, so a gap here is a silent no-op."""
    c = with_expanded_eval_prompts(_by_name()[name])
    assert len(c.eval_prompts) == 30


@pytest.mark.parametrize("name", REJUDGE)
def test_the_original_six_are_a_strict_prefix(name):
    """This is what keeps the six-prompt numbers recoverable by subsetting."""
    original = _by_name()[name]
    expanded = with_expanded_eval_prompts(original)
    assert expanded.eval_prompts[: len(original.eval_prompts)] == \
        original.eval_prompts


@pytest.mark.parametrize("name", REJUDGE)
def test_no_duplicate_prompts(name):
    """A repeated prompt is a narrower interval bought with nothing."""
    prompts = with_expanded_eval_prompts(_by_name()[name]).eval_prompts
    assert len(set(prompts)) == len(prompts)


@pytest.mark.parametrize("name", REJUDGE)
def test_eval_prompts_stay_disjoint_from_the_pairs(name):
    """Revision R5: the direction comes from pairs, the effect from prompts.

    Extending the prompt set is the obvious way to break that accidentally.
    """
    c = with_expanded_eval_prompts(_by_name()[name])
    pair_texts = {p.positive for p in c.pairs} | {p.negative for p in c.pairs}
    assert not (set(c.eval_prompts) & pair_texts)


def test_expansion_leaves_other_concepts_alone():
    """Mapping it over every concept must be safe, so unlisted ones pass through."""
    for c in all_concepts():
        if c.name in _EXTRA_EVAL_PROMPTS:
            continue
        assert with_expanded_eval_prompts(c) is c


def test_ground_truth_already_has_thirty():
    """The point of the exercise is to match them, so pin what is being matched."""
    from lbi.groundtruth import ground_truth_concepts

    for c in ground_truth_concepts():
        assert len(c.eval_prompts) == 30


def test_helper_does_not_mutate_the_concept_it_is_given():
    original = _by_name()["sentiment"]
    before = list(original.eval_prompts)
    with_expanded_eval_prompts(original)
    assert original.eval_prompts == before


def test_expanded_eval_prompts_is_a_pure_function_of_its_arguments():
    base = ["a", "b"]
    out = expanded_eval_prompts("sentiment", base)
    assert out[:2] == base
    assert base == ["a", "b"], "the caller's list must not be extended in place"
