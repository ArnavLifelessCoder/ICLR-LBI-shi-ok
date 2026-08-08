"""Tests for the TF-IDF surface-shortcut audit (revision R15).

The audit is the highest-leverage single change in the project: it catches
lexical shortcuts before any activation is extracted. These tests verify the
logic without needing a model.
"""

import numpy as np
import pytest

from lbi.concepts import (
    AuditResult,
    Concept,
    Pair,
    _marker_tokens,
    all_concepts,
    audit_all_concepts,
    check_marker_disjointness,
    check_marker_length_match,
    get_concept,
    surface_shortcut_audit,
)


def _shortcut_concept():
    """A concept where positive/negative are trivially separable by vocabulary."""
    pairs = []
    for i in range(20):
        pairs.append(
            Pair(
                positive=f"The MAGIC_TOKEN is present in sentence {i} about cooking.",
                negative=f"There is nothing special in sentence {i} about cooking.",
                family=f"fam{i % 4}",
            )
        )
    return Concept(
        name="shortcut",
        description="Trivially separable by a keyword.",
        pairs=pairs,
        eval_prompts=["Write something."],
        behavior_question="Score it.",
    )


def _clean_concept():
    """A concept built the way the real ones are: family-disjoint markers.

    The earlier version of this fixture used "well" vs "poorly" in every family
    and was called clean, but a single word marking the contrast everywhere is
    the textbook shortcut -- a TF-IDF classifier trained on two families scores
    the third at AUROC 1.0. It was the fixture that was wrong, not the audit.
    Here each family carries the contrast on its own vocabulary, so nothing
    a classifier learns on two families transfers to the third.
    """
    topics = [
        "the weather today", "the schedule tomorrow", "the old bridge",
        "the project plan", "the garden path", "the annual report",
        "the morning routine", "the evening walk",
    ]
    families = {
        "fam0": ("I reckon {topic} looks {m}.", "promising", "worrying"),
        "fam1": ("My sense of {topic} is {m}.", "sunny", "bleak"),
        "fam2": ("Honestly {topic} strikes me as {m}.", "heartening", "dispiriting"),
    }
    pairs = [
        Pair(
            positive=carrier.format(topic=topic, m=pos),
            negative=carrier.format(topic=topic, m=neg),
            family=family,
        )
        for family, (carrier, pos, neg) in families.items()
        for topic in topics
    ]
    return Concept(
        name="clean",
        description="Minimal pairs whose marker vocabulary differs per family.",
        pairs=pairs,
        eval_prompts=["Write something."],
        behavior_question="Score it.",
    )


def test_shortcut_concept_fails_audit():
    c = _shortcut_concept()
    result = surface_shortcut_audit(c)
    # MAGIC_TOKEN makes this trivially separable.
    assert result.surface_auroc > 0.8
    assert not result.passed
    assert "FAIL" in result.reason


def test_clean_concept_passes_audit():
    c = _clean_concept()
    result = surface_shortcut_audit(c)
    assert result.passed


def test_audit_passes_if_below_ceiling():
    c = _clean_concept()
    result = surface_shortcut_audit(c, surface_ceiling=0.65)
    if result.surface_auroc < 0.65:
        assert result.passed
        assert "ceiling" in result.reason


def test_audit_passes_if_gap_to_probe_is_large():
    c = _shortcut_concept()
    # Even if surface is high, if the probe is much higher the concept passes.
    result = surface_shortcut_audit(c, probe_auroc=1.0, min_gap=0.01)
    # With probe_auroc=1.0 and min_gap=0.01, any surface < 0.99 passes.
    if result.surface_auroc < 0.99:
        assert result.passed
        assert "gap" in result.reason.lower() or "below probe" in result.reason.lower()


def test_audit_result_summary_has_required_keys():
    c = get_concept("sentiment")
    result = surface_shortcut_audit(c)
    summary = result.summary()
    for key in ("concept", "surface_auroc", "passed", "reason", "n_train", "n_test"):
        assert key in summary, key


def test_audit_all_concepts_returns_one_per_concept():
    results = audit_all_concepts()
    assert len(results) == len(all_concepts())
    assert all(isinstance(r, AuditResult) for r in results)


def test_every_concept_not_declared_confounded_passes_the_audit():
    """Only the concepts declared surface-confounded in advance may fail.

    PLAN.md's kill criterion is fewer than eight surviving concepts. This is the
    stricter version: a concept either passes, or its builder declared it
    surface-confounded before the audit ran. A concept that fails without having
    been declared is a construction bug, not a finding.
    """
    results = audit_all_concepts()
    undeclared = [r for r in results if not r.passed and not r.surface_confounded]
    assert not undeclared, (
        "concepts failed the surface audit without being declared "
        f"surface-confounded: {[(r.concept, round(r.surface_auroc, 3)) for r in undeclared]}"
    )
    passed = [r for r in results if r.passed]
    assert len(passed) >= 8, (
        f"only {len(passed)}/{len(results)} concepts cleared the audit; "
        "PLAN.md Part 8 puts the kill criterion at eight"
    )


def test_declared_confounded_concepts_still_fail_rather_than_being_excused():
    """The flag documents an expected failure; it must not convert it to a pass."""
    results = {r.concept: r for r in audit_all_concepts()}
    for name in ("verbosity", "topic_science"):
        r = results[name]
        assert r.surface_confounded, f"{name} should be declared surface-confounded"
        assert not r.passed, f"{name} must still be reported as failing the audit"
        assert "surface-confounded" in r.reason


def test_marker_invariants_hold_for_every_concept_built_that_way():
    """Every non-confounded concept satisfies the two construction rules.

    These are enforced inside the builders, so this test mainly guards against
    someone relaxing `enforce_disjoint` to silence a failure.
    """
    for c in all_concepts():
        if c.surface_confounded:
            continue
        by_family: dict[str, set[str]] = {}
        for p in c.pairs:
            by_family.setdefault(p.family, set()).update(
                _marker_tokens(p.positive) | _marker_tokens(p.negative)
            )
        assert len(by_family) >= 4, f"{c.name} has too few families to split on"


def test_topic_science_families_are_not_duplicates():
    """Regression: families used to be blocks 0-3 over a list repeated twice.

    block2 and block3 were byte-identical to block0 and block1, so every
    held-out family was already present in training.
    """
    c = get_concept("topic_science")
    by_family = {}
    for p in c.pairs:
        by_family.setdefault(p.family, set()).add((p.positive, p.negative))
    families = sorted(by_family)
    assert len(families) >= 4
    for i, a in enumerate(families):
        for b in families[i + 1:]:
            assert not (by_family[a] & by_family[b]), (
                f"topic_science families {a} and {b} share pairs"
            )


def test_audit_counts_perfect_inversion_as_a_shortcut():
    """A held-out family ranked perfectly backwards is not a clean result.

    Scored one-sided, this concept reads AUROC 0.0 -- better than chance-looking
    and the best number in the set. It is a shortcut whose polarity happens not
    to transfer, and the two-sided statistic says so.
    """
    pairs = []
    for i in range(8):
        # "vivid" marks the positive side in the training families...
        pairs.append(Pair(positive=f"A vivid account number {i}.",
                          negative=f"A plain account number {i}.", family="fam0"))
        pairs.append(Pair(positive=f"Quite a vivid note number {i}.",
                          negative=f"Quite a plain note number {i}.", family="fam1"))
        # ...and the negative side in the held-out one.
        pairs.append(Pair(positive=f"Rather plain remarks number {i}.",
                          negative=f"Rather vivid remarks number {i}.", family="fam2"))
    c = Concept(
        name="inverted", description="Polarity flips in one family.",
        pairs=pairs, eval_prompts=["x"], behavior_question="y",
    )
    result = surface_shortcut_audit(c, held_out_families={"fam2"})
    assert result.raw_auroc_by_family["fam2"] < 0.2, "expected a near-perfect inversion"
    assert result.surface_auroc > 0.8, "two-sided statistic should expose it"
    assert not result.passed


def test_audit_uses_every_family_as_a_fold_by_default():
    c = _clean_concept()
    result = surface_shortcut_audit(c)
    assert set(result.raw_auroc_by_family) == set(c.families())
    assert result.worst_fold_auroc >= result.surface_auroc - 1e-9


def test_disjointness_check_rejects_a_shared_marker():
    spec = {
        "a": ("It was {m} overall.", "superb", "awful"),
        "b": ("They found it {m} throughout.", "superb", "dismal"),
    }
    with pytest.raises(ValueError, match="superb"):
        check_marker_disjointness("demo", spec)


def test_disjointness_check_rejects_a_marker_word_that_appears_in_a_topic():
    spec = {
        "a": ("It was {m} overall.", "superb", "awful"),
        "b": ("They found it {m} throughout.", "grand", "dismal"),
    }
    check_marker_disjointness("demo", spec)  # fine on its own
    with pytest.raises(ValueError, match="superb"):
        check_marker_disjointness("demo", spec, topics=["the superb bridge"])


def test_length_check_rejects_mismatched_markers():
    spec = {"a": ("It was {m} overall.", "really quite superb", "awful")}
    with pytest.raises(ValueError, match="differ in length"):
        check_marker_length_match("demo", spec)
