"""The published replication has to be checkable before it is run.

The readout decides what the experiment concludes about somebody else's result,
so its failure modes matter more than usual. It must fire on wedding text, stay
quiet on ordinary text, and be wrong in the direction that understates the
published effect rather than overstating it. A readout that over-fires would
report that the intervention worked when the model merely wrote about a party.
"""

from __future__ import annotations

import pytest

from lbi.groundtruth import READOUTS, DeterministicScorer
from lbi.published import (
    ACTADD_SETTING,
    PUBLISHED_READOUTS,
    actadd_wedding,
    published_concepts,
    score_wedding,
)

CONCEPT = actadd_wedding()


def test_available_through_the_scorer_when_passed_explicitly():
    scorer = DeterministicScorer(extra=PUBLISHED_READOUTS)
    assert scorer.score(["a wedding"], "pub_wedding") == [score_wedding("a wedding")]


def test_does_not_leak_into_the_shared_registry():
    """Registering on import would change what a plain DeterministicScorer
    accepts depending on whether this module happened to be imported."""
    assert "pub_wedding" not in READOUTS
    with pytest.raises(KeyError):
        DeterministicScorer().score(["a wedding"], "pub_wedding")


def test_extra_cannot_shadow_a_built_in_readout():
    with pytest.raises(ValueError, match="shadow"):
        DeterministicScorer(extra={"gt_french": score_wedding})


def test_fires_on_wedding_text():
    assert score_wedding("The bride and groom exchanged vows at the altar.") > 0.25


def test_silent_on_ordinary_text():
    """Over-firing would report a wedding effect that did not happen."""
    ordinary = [
        "I went up to my friend and said hello and we talked for a while.",
        "The party had a cake and everyone wore a nice dress and a ring.",
        "Yesterday afternoon I decided to walk to the shop and buy some bread.",
        "She put on a white dress for the ceremony at the town hall.",
    ]
    assert all(score_wedding(t) == 0.0 for t in ordinary)


def test_ambiguous_words_are_excluded_on_purpose():
    """ring, party, dress, cake and ceremony are wedding vocabulary in context
    and ordinary words out of it. The readout must not claim them."""
    for w in ("ring", "rings", "party", "dress", "cake", "ceremony", "reception"):
        assert score_wedding(w) == 0.0, w


def test_bounded_and_safe_on_degenerate_input():
    for t in ("", "   ", "!!!", "123"):
        assert score_wedding(t) == 0.0
    assert score_wedding("wedding") == 1.0


def test_eval_prompts_do_not_invite_the_behavior():
    """A prompt that mentions weddings would make the intervention look
    effective when the prompt was doing the work."""
    for p in CONCEPT.eval_prompts:
        assert score_wedding(p) == 0.0, p
        low = p.lower()
        for banned in ("wed", "marri", "bride", "groom", "engage", "romance"):
            assert banned not in low, (p, banned)


def test_direction_comes_from_the_published_contrast():
    """Estimating from our own corpus would test our method on their behavior
    rather than testing their intervention."""
    pos = {p.positive for p in CONCEPT.pairs}
    neg = {p.negative for p in CONCEPT.pairs}
    assert pos == {ACTADD_SETTING["positive_prompt"]}
    assert neg == {ACTADD_SETTING["negative_prompt"]}


def test_family_split_is_possible():
    """The harness splits on families; a single family would raise."""
    fams = CONCEPT.families()
    assert len(fams) >= 3
    train, test = CONCEPT.split_by_family({fams[0]})
    assert train and test


def test_setting_records_what_must_be_verified():
    for key in ("model", "positive_prompt", "negative_prompt", "layer",
                "coefficient", "source"):
        assert key in ACTADD_SETTING
    assert ACTADD_SETTING["source"] == "turner2023actadd"


def test_published_concepts_is_the_entry_point():
    assert [c.name for c in published_concepts()] == ["pub_wedding"]
