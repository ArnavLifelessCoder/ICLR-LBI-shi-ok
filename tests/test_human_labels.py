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
