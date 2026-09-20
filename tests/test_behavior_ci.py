"""`behavior_ci` must be an interval for the mean, not a spread of the scores.

This field is what every signed-area interval in the study is propagated from:
`analyse_validation`, `signed_uncertainty` and `make_groundtruth_artifacts` all
take its width and divide by 3.92 to recover a standard error. It used to be
`np.percentile(scores, 2.5)` and `97.5`, which is the spread of the individual
generations. The two differ by roughly sqrt(n) and the error ran both ways:

  - judge scores spread over [0, 1], so the stored interval was near [0, 1]
    whatever the mean, about three times too wide at thirty prompts, and signed
    areas looked less resolved than the data supports;
  - a sparse rule readout scores 0 for most generations, so both percentiles
    were 0 and the interval was degenerate, which is why the stage F positive
    control could not pass however large the effect.

A percentile interval also does not shrink as prompts are added, which is what
made the thirty-against-six prompt comparison unreadable.
"""

from __future__ import annotations

import numpy as np
import pytest

from lbi.pipeline import _mean_ci


def test_a_sparse_readouts_lower_bound_is_not_pinned_to_zero():
    """The stage F failure.

    The positive control asks whether the lower bound clears baseline. Under
    the percentile form that bound is exactly 0 whenever more than 2.5% of
    generations score 0, which for a rare-word readout is nearly always, so no
    effect of any size could pass. Only the lower bound matters here; the
    upper one moves once enough generations are non-zero.
    """
    scores = [0.0] * 47 + [0.02] * 3
    assert np.percentile(scores, 2.5) == 0.0          # the old behaviour
    lo, hi = _mean_ci(scores)
    assert hi > np.mean(scores) > 0
    assert hi > lo


def test_an_even_sparser_readout_was_wholly_degenerate():
    """Below one in forty non-zero, both percentiles collapse onto zero.

    Stage D hit this: several coefficients stored [0.00000, 0.00000].
    """
    scores = [0.0] * 49 + [0.02]
    assert np.percentile(scores, 2.5) == 0.0
    assert np.percentile(scores, 97.5) == 0.0
    lo, hi = _mean_ci(scores)
    assert hi > lo, "the bootstrap must still separate the bounds"


def test_a_judge_readout_is_not_the_whole_unit_interval():
    """The old form returned [0, 1] for any mean at all."""
    scores = [0.0, 1.0] * 15
    assert (np.percentile(scores, 2.5), np.percentile(scores, 97.5)) == (0.0, 1.0)
    lo, hi = _mean_ci(scores)
    assert 0.0 < lo < 0.5 < hi < 1.0


def test_the_interval_shrinks_as_generations_are_added():
    """The property the percentile form does not have, and the whole point."""
    rng = np.random.default_rng(0)
    widths = []
    for n in (10, 100, 1000):
        s = rng.binomial(1, 0.4, size=n).astype(float)
        lo, hi = _mean_ci(s)
        widths.append(hi - lo)
    assert widths[0] > widths[1] > widths[2]
    # Roughly 1/sqrt(n): a tenfold n should cut the width by about three.
    assert 2.0 < widths[0] / widths[1] < 5.0


def test_the_interval_covers_the_mean():
    rng = np.random.default_rng(1)
    s = rng.normal(0.3, 0.1, size=40)
    lo, hi = _mean_ci(s)
    assert lo <= s.mean() <= hi


def test_a_constant_readout_gives_a_point_interval():
    """Every generation scored the same, so there is nothing to be unsure of."""
    lo, hi = _mean_ci([0.25] * 20)
    assert lo == hi == pytest.approx(0.25)


def test_empty_scores_are_not_an_exception():
    lo, hi = _mean_ci([])
    assert np.isnan(lo) and np.isnan(hi)


def test_it_is_deterministic():
    s = list(np.random.default_rng(2).binomial(1, 0.5, 50).astype(float))
    assert _mean_ci(s) == _mean_ci(s)


def test_dose_points_carry_their_scores():
    """Without these the interval cannot be recomputed from a saved run."""
    from lbi import steering as st

    p = st.DosePoint(coeff=1.0, behavior=0.5, behavior_ci=(0.4, 0.6),
                     perplexity=1.0, repetition=0.0, broken=False,
                     scores=[0.0, 1.0])
    from lbi.pipeline import jsonable

    assert jsonable(p)["scores"] == [0.0, 1.0]


def test_run_model_writes_the_scores_too():
    """run_model spells its curve out field by field instead of using jsonable,
    so a new DosePoint field is dropped there while appearing everywhere else.
    That cost a three-hour stage E run."""
    import inspect

    from lbi import pipeline

    src = inspect.getsource(pipeline.run_model)
    assert '"scores": p.scores' in src
