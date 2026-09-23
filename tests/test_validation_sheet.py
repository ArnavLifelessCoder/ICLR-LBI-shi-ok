"""Round-3 judge validation: the sheet is blind, and the scorer applies the
preregistered criterion correctly to labels whose answer is known.

The scorer is exercised with planted labels: a human who copies the judge must
validate, a constant human must be flagged as uninformative rather than scored,
and a human who ignores the text must fail. If any of these did otherwise, the
verdict on real labels would mean nothing.
"""

from __future__ import annotations

import csv
import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import make_validation_sheet as mk  # noqa: E402
import score_validation_sheet as sc  # noqa: E402

RESULTS = os.path.join(ROOT, "validation_output", "results_judge_matched")
needs_results = pytest.mark.skipif(
    not os.path.isdir(RESULTS), reason="stage G results not present")


def test_bins():
    assert mk.coeff_bin(-5.0) == "neg"
    assert mk.coeff_bin(0.0) == "mid"
    assert mk.coeff_bin(0.5) == "mid"
    assert mk.coeff_bin(2.0) == "pos"


def test_spread_sample_is_deterministic_and_balanced():
    items = [{"model": m, "coeff": c, "text": "%s %s" % (m, c)}
             for m in "abcd" for c in (-4.0, -2.0, 0.0, 0.5, 2.0, 4.0)]
    a = mk.spread_sample(list(items), 12, 0)
    b = mk.spread_sample(list(items), 12, 0)
    assert [x["text"] for x in a] == [x["text"] for x in b]
    assert len({x["model"] for x in a}) == 4
    assert len({mk.coeff_bin(x["coeff"]) for x in a}) == 3


@needs_results
def test_collect_drops_broken_empty_and_short_texts():
    for concept in mk.CONCEPTS:
        items = mk.collect(RESULTS, concept)
        assert items, concept
        texts = [it["text"] for it in items]
        assert len(texts) == len(set(texts)), "duplicates survived"
        assert all(len(t.split()) >= mk.MIN_WORDS for t in texts)
        assert all(0.0 <= it["judge"] <= 1.0 for it in items)


def _build(tmp_path):
    sheet = tmp_path / "sheet.csv"
    key = tmp_path / "key.csv"
    old = sys.argv
    sys.argv = ["x", "--results", RESULTS, "-o", str(sheet), "--key", str(key)]
    try:
        assert mk.main() == 0
    finally:
        sys.argv = old
    return sheet, key


@needs_results
def test_sheet_is_blind(tmp_path):
    sheet, key = _build(tmp_path)
    with open(sheet, encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == ["index", "concept", "question", "text", "score"]
    for leak in ("model", "coeff", "judge"):
        assert leak not in header
    with open(key, encoding="utf-8") as f:
        assert {"model", "coeff", "judge"} <= set(next(csv.reader(f)))


def _fill(sheet, key, out, rule):
    with open(key, encoding="utf-8") as f:
        judge = {int(r["index"]): float(r["judge"]) for r in csv.DictReader(f)}
    with open(sheet, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["score"] = rule(judge[int(r["index"])], int(r["index"]))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _snap(x):
    return min(sc.ALLOWED, key=lambda v: abs(v - x))


@needs_results
def test_a_human_who_agrees_with_the_judge_validates(tmp_path):
    sheet, key = _build(tmp_path)
    filled = tmp_path / "filled.csv"
    _fill(sheet, key, filled, lambda j, i: _snap(j))
    rows = sc.read_labels(str(filled), str(key))
    for c in ("sentiment", "sycophancy", "formality"):
        sub = [r for r in rows if r["concept"] == c]
        h = [r["human"] for r in sub]
        j = [r["judge"] for r in sub]
        a = sc.alpha(h, j)
        import numpy as np
        v = sc.verdict(a, float(np.std(h)), float(np.std(j)))
        assert v.startswith("VALIDATED"), (c, a, v)


@needs_results
def test_a_constant_human_is_uninformative_not_scored(tmp_path):
    sheet, key = _build(tmp_path)
    filled = tmp_path / "filled.csv"
    _fill(sheet, key, filled, lambda j, i: 0.5)
    rows = sc.read_labels(str(filled), str(key))
    import numpy as np
    sub = [r for r in rows if r["concept"] == "sentiment"]
    h = [r["human"] for r in sub]
    j = [r["judge"] for r in sub]
    v = sc.verdict(sc.alpha(h, j), float(np.std(h)), float(np.std(j)))
    assert "flat" in v and not v.startswith("VALIDATED")


@needs_results
def test_a_human_who_ignores_the_text_does_not_validate(tmp_path):
    sheet, key = _build(tmp_path)
    filled = tmp_path / "filled.csv"
    rng = random.Random(7)
    _fill(sheet, key, filled, lambda j, i: rng.choice(sorted(sc.ALLOWED)))
    rows = sc.read_labels(str(filled), str(key))
    import numpy as np
    passed = 0
    for c in mk.CONCEPTS:
        sub = [r for r in rows if r["concept"] == c]
        h = [r["human"] for r in sub]
        j = [r["judge"] for r in sub]
        if sc.verdict(sc.alpha(h, j), float(np.std(h)),
                      float(np.std(j))).startswith("VALIDATED"):
            passed += 1
    assert passed == 0


@needs_results
def test_off_scale_labels_are_refused(tmp_path):
    sheet, key = _build(tmp_path)
    filled = tmp_path / "filled.csv"
    _fill(sheet, key, filled, lambda j, i: 0.3 if i == 0 else 0.5)
    with pytest.raises(ValueError, match="fix these rows"):
        sc.read_labels(str(filled), str(key))


def test_the_criterion_is_the_preregistered_one():
    """Pinned so a later edit to the scorer cannot move the goalposts quietly."""
    assert sc.ALPHA_VALID == 0.667
    assert sc.MIN_SD == 0.11
    assert sc.ALLOWED == {0.0, 0.25, 0.5, 0.75, 1.0}
    with open(os.path.join(ROOT, "VALIDATION_PREREG.md"), encoding="utf-8") as f:
        prereg = f.read()
    assert "0.667" in prereg and "0.11" in prereg
