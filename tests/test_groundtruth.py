"""The ground-truth concepts exist to check the directional metric, so their
readouts have to be right before anything is concluded from them.

The load-bearing property is separation: on each concept the readout must score
the positive member of every minimal pair above the negative member. If it does
not, the "known" direction is not known, and a signed area computed against it
would be checked against the wrong answer while looking perfectly healthy.
"""

from __future__ import annotations

import pytest

from lbi.groundtruth import (
    DeterministicScorer,
    READOUTS,
    ground_truth_concepts,
    score_digits,
    score_french,
    score_length,
    score_uppercase,
)

CONCEPTS = {c.name: c for c in ground_truth_concepts()}


def test_every_concept_has_a_readout():
    assert set(CONCEPTS) == set(READOUTS)


@pytest.mark.parametrize("name", sorted(READOUTS))
def test_readout_separates_every_pair(name):
    """Positive must outscore negative on all pairs, not merely on average."""
    fn = READOUTS[name]
    failures = [
        (p.family, p.positive, p.negative)
        for p in CONCEPTS[name].pairs
        if fn(p.positive) <= fn(p.negative)
    ]
    assert not failures, (
        "%s: %d of %d pairs are not separated by the readout; first: %r"
        % (name, len(failures), len(CONCEPTS[name].pairs), failures[0])
    )


@pytest.mark.parametrize("name", sorted(READOUTS))
def test_readout_is_bounded(name):
    fn = READOUTS[name]
    texts = [p.positive for p in CONCEPTS[name].pairs]
    texts += [p.negative for p in CONCEPTS[name].pairs]
    texts += ["", "   ", "!!!", "123", "ABC"]
    assert all(0.0 <= fn(t) <= 1.0 for t in texts)


@pytest.mark.parametrize("name", sorted(READOUTS))
def test_eval_prompts_are_disjoint_from_pairs(name):
    """The direction is estimated on pairs and measured on prompts. If a prompt
    subject also appears in a pair, the measurement is partly in-sample."""
    c = CONCEPTS[name]
    pair_text = " ".join(p.positive + " " + p.negative for p in c.pairs)
    for prompt in c.eval_prompts:
        # The shared template wording is fine; the subject must not be shared.
        subject = prompt.split(" ", 1)[1].rstrip("?.").strip()
        assert subject not in pair_text, (
            "%s: eval subject %r also appears in the pairs" % (name, subject)
        )


def test_empty_and_degenerate_text_does_not_crash():
    s = DeterministicScorer()
    for name in READOUTS:
        assert s.score(["", " ", "\n\n"], name) == [0.0, 0.0, 0.0]


def test_unknown_concept_raises_rather_than_defaulting():
    """A silent default would produce a flat dose-response, which is the exact
    pattern the danger-zone criterion selects for."""
    with pytest.raises(KeyError, match="deterministic readout"):
        DeterministicScorer().score(["anything"], "not_a_ground_truth_concept")


def test_length_saturates():
    assert score_length("word " * 500) == 1.0


def test_uppercase_ignores_non_letters():
    assert score_uppercase("!!! 123 ???") == 0.0
    assert score_uppercase("ABC") == 1.0
    assert score_uppercase("abc") == 0.0


def test_digits_ignores_whitespace():
    assert score_digits("1 2 3") == 1.0
    assert score_digits("abc") == 0.0


def test_french_does_not_fire_on_plain_english():
    """The marker set excludes words that are common in English, so ordinary
    English prose must score near zero or the readout would report a language
    shift that did not happen."""
    english = [
        "The quarterly budget review went well and the team was pleased.",
        "I think the new coffee shop downtown is a very good thing for us.",
        "They said that the summer reading list was already ready.",
    ]
    assert all(score_french(t) < 0.05 for t in english)


def test_french_fires_on_french():
    french = [
        "Je pense que ce est une tres bonne chose pour nous.",
        "Il est clair que ces choses ne peuvent pas continuer.",
    ]
    assert all(score_french(t) > 0.30 for t in french)


def test_scorer_matches_the_module_level_functions():
    s = DeterministicScorer()
    text = ["The answer is 42 and 17 more."]
    assert s.score(text, "gt_digits") == [score_digits(text[0])]
    assert s.score(text, "gt_length") == [score_length(text[0])]
    assert s.score(text, "gt_uppercase") == [score_uppercase(text[0])]
    assert s.score(text, "gt_french") == [score_french(text[0])]
