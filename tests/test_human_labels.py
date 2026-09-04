"""Tests for the human-label agreement path (objection 14).

Filling human_labels.csv only helps if something turns it into a number. Per-
sample judge scores are not persisted in the result JSONs, so the judges must be
re-run on the labelled texts; these tests pin that path, including the partial
sheet and out-of-range guards that a hand-edited CSV will hit in practice.
"""

import csv
import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
)

from lbi.behavior import read_labeling_sheet, write_labeling_sheet
from score_human_labels import _read_rows, _score_by_concept
from lbi.behavior import LexiconScorer


def _sheet(tmp_path, rows):
    p = os.path.join(str(tmp_path), "human_labels.csv")
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "concept", "question", "text", "score"])
        for i, (concept, text, score) in enumerate(rows):
            w.writerow([i, concept, "q?", text, score])
    return p


def test_read_rows_treats_blanks_as_missing(tmp_path):
    import math

    p = _sheet(tmp_path, [("sentiment", "great stuff", "1.0"),
                          ("sentiment", "awful", ""),
                          ("sentiment", "fine", "0.5")])
    rows = _read_rows(p)
    assert rows[0]["human"] == 1.0
    assert math.isnan(rows[1]["human"])
    assert rows[2]["human"] == 0.5


def test_read_rows_rejects_out_of_range_score(tmp_path):
    p = _sheet(tmp_path, [("sentiment", "x", "1.5")])
    with pytest.raises(ValueError, match="outside"):
        _read_rows(p)


def test_read_rows_accepts_the_bounds(tmp_path):
    p = _sheet(tmp_path, [("sentiment", "a", "0.0"), ("sentiment", "b", "1.0")])
    rows = _read_rows(p)
    assert [r["human"] for r in rows] == [0.0, 1.0]


def test_scoring_groups_by_concept_and_preserves_row_order(tmp_path):
    """Scorers are per-concept; rows interleave concepts, so order must survive."""
    rows = _read_rows(
        _sheet(
            tmp_path,
            [
                ("sentiment", "This was excellent and I loved it, truly wonderful.", "1"),
                ("refusal", "I cannot help with that, sorry.", "1"),
                ("sentiment", "This was terrible and awful, a total waste.", "0"),
            ],
        )
    )
    scores = _score_by_concept(LexiconScorer(), rows, "lexicon")
    assert len(scores) == 3
    # Row 0 is clearly positive sentiment, row 2 clearly negative.
    assert scores[0] > scores[2]
    # Row 1 is a refusal, scored on the refusal lexicon, not the sentiment one.
    assert scores[1] > 0.5


def test_unknown_concept_warns_but_does_not_abort(tmp_path, capsys):
    """A hand-edited sheet can carry a bad concept name; that must not kill the run."""
    import math

    rows = _read_rows(_sheet(tmp_path, [("not_a_concept", "text", "0.5")]))
    scores = _score_by_concept(LexiconScorer(), rows, "lexicon")
    assert math.isnan(scores[0])
    assert "WARNING" in capsys.readouterr().out


def test_write_then_read_round_trips(tmp_path):
    """The sheet the pipeline writes must be the sheet the scorer can read."""
    p = os.path.join(str(tmp_path), "sheet.csv")
    write_labeling_sheet(
        p,
        [("sentiment", "text with, a comma"), ("refusal", "multi\nline text")],
        {"sentiment": "Is it positive?", "refusal": "Is it a refusal?"},
    )
    concepts, scores = read_labeling_sheet(p)
    assert concepts == ["sentiment", "refusal"]
    assert len(scores) == 2
    rows = _read_rows(p)
    assert rows[1]["text"] == "multi\nline text"


# --------------------------------------------------------------------------
# stratified_label_sample: the three failures the old "first n in curve order"
# sheet actually had (56% duplicates, 6/10 concepts, no positive control).
# --------------------------------------------------------------------------

from lbi.behavior import stratified_label_sample


def _items(n_concepts=10, n_coeffs=9, n_per=2, dup_rate=0):
    """(concept, coeff, text) triples shaped like real curve output."""
    items = []
    for c in range(n_concepts):
        for k in range(n_coeffs):
            coeff = -3.0 + k * 0.75
            for s in range(n_per):
                # dup_rate>0 makes neighbouring coefficients emit identical text,
                # which is exactly what real steering does at low strengths.
                tag = (k // dup_rate) if dup_rate else k
                items.append((f"c{c}", coeff, f"c{c}_text{tag}_{s}"))
    return items


def test_sample_is_deduplicated():
    out = stratified_label_sample(_items(dup_rate=3), n=100, n_repeat=0, seed=0)
    assert len(out) == len(set(out)), "no (concept, text) pair may repeat"


def test_sample_covers_every_concept():
    out = stratified_label_sample(_items(n_concepts=10), n=100, n_repeat=10, seed=0)
    assert len({c for c, _ in out}) == 10


def test_small_concept_still_gets_rows():
    """Round-robin: a concept with few generations must not be crowded out."""
    items = _items(n_concepts=3)
    items += [("rare", 0.0, "rare_text")]  # a single generation
    out = stratified_label_sample(items, n=40, n_repeat=0, seed=0)
    assert "rare" in {c for c, _ in out}


def test_sample_spans_the_coefficient_range():
    """Within-concept variance is what validates the judge, so both ends of the
    sweep must appear, not just whichever end came first in curve order."""
    items = [("c0", -3.0 + k * 0.75, f"t{k}") for k in range(9)]
    out = stratified_label_sample(items, n=4, n_repeat=0, seed=0)
    picked = {t for _, t in out}
    assert "t0" in picked and "t8" in picked


def test_repeats_are_included_for_intra_rater_reliability():
    out = stratified_label_sample(_items(), n=100, n_repeat=10, seed=0)
    assert len(out) == 100
    assert len(out) - len(set(out)) == 10


def test_repeats_are_not_adjacent():
    """A repeat sitting next to its original is obvious and stops measuring
    anything about rater consistency."""
    out = stratified_label_sample(_items(), n=100, n_repeat=10, seed=0)
    for i in range(len(out) - 1):
        assert out[i] != out[i + 1]


def test_zero_repeats_gives_all_distinct():
    out = stratified_label_sample(_items(), n=50, n_repeat=0, seed=0)
    assert len(out) == len(set(out)) == 50


def test_deterministic_for_a_seed():
    a = stratified_label_sample(_items(), n=60, n_repeat=6, seed=3)
    b = stratified_label_sample(_items(), n=60, n_repeat=6, seed=3)
    assert a == b


def test_handles_fewer_items_than_requested():
    out = stratified_label_sample(_items(n_concepts=2, n_coeffs=2, n_per=1),
                                  n=100, n_repeat=5, seed=0)
    assert 0 < len(out) <= 100


def test_empty_input():
    assert stratified_label_sample([], n=100) == []


def test_repeats_are_well_separated_not_merely_non_adjacent():
    """A repeat two rows after its original measures recall, not consistency."""
    from collections import defaultdict

    out = stratified_label_sample(_items(), n=100, n_repeat=10, seed=0)
    pos = defaultdict(list)
    for i, k in enumerate(out):
        pos[k].append(i)
    gaps = [p[1] - p[0] for p in pos.values() if len(p) > 1]
    assert gaps, "expected some repeats"
    assert min(gaps) >= 10, f"repeat too close to its original: min gap {min(gaps)}"
