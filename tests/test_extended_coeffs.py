"""The extended grid has to stay a strict superset of the default one.

That property is what keeps the two comparable: a sweep run on the extended
grid, subset back to the default coefficients, must reproduce the default-grid
controllability exactly. If someone later "tidies" the extended grid by
resampling it at round numbers, the two stop being comparable and nothing warns
them.
"""

from __future__ import annotations

import numpy as np

from lbi.pipeline import DEFAULT_COEFFS, EXTENDED_COEFFS


def test_extended_is_a_superset_of_default():
    assert set(DEFAULT_COEFFS) <= set(EXTENDED_COEFFS)


def test_extended_actually_extends_the_range():
    """The point of the grid is to reach past where behavior starts moving."""
    assert max(EXTENDED_COEFFS) > max(DEFAULT_COEFFS)
    assert min(EXTENDED_COEFFS) < min(DEFAULT_COEFFS)


def test_both_grids_are_sorted_and_symmetric_about_zero():
    for grid in (DEFAULT_COEFFS, EXTENDED_COEFFS):
        assert grid == sorted(grid)
        assert 0.0 in grid
        assert sorted(abs(c) for c in grid if c > 0) == \
               sorted(abs(c) for c in grid if c < 0)


def test_subsetting_the_extended_grid_reproduces_the_default_area():
    """The comparability claim, checked on a synthetic dose response.

    A trapezoid over the default coefficients must be recoverable from a sweep
    run on the extended grid by dropping the extra points.
    """
    def area(xs, ys):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        return float(np.trapezoid(np.abs(ys), xs) / (xs[-1] - xs[0]))

    rng = np.random.default_rng(0)
    behavior = {c: float(rng.uniform(0, 1)) for c in EXTENDED_COEFFS}

    from_default = area(DEFAULT_COEFFS, [behavior[c] for c in DEFAULT_COEFFS])
    subset = [c for c in EXTENDED_COEFFS if c in set(DEFAULT_COEFFS)]
    from_extended = area(subset, [behavior[c] for c in subset])

    assert from_default == from_extended


def test_extended_grid_is_not_absurdly_expensive():
    """Stage A is about five hours per model on the default grid, and a Kaggle
    session is twelve. A grid that doubles the cost does not fit alongside the
    other stages."""
    assert len(EXTENDED_COEFFS) / len(DEFAULT_COEFFS) < 1.6
