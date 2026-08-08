"""Tests for the lbi pipeline.

No GPU and no model download: a synthetic activation generator stands in for
the model wherever real activations would be needed. The point is to verify the
metric logic, the split discipline and the aggregation -- the parts that would
silently produce a wrong number in the paper.
"""

import numpy as np
import pytest

from lbi import behavior as bh
from lbi import geometry as geo
from lbi import steering as st
from lbi.concepts import all_concepts, get_concept, safety_concepts
from lbi.gapmap import GapPoint, build_gap_map, normalize_within_model
from lbi.probes import (
    default_held_out,
    diff_of_means_direction,
    pair_texts,
    repe_reading_vector,
    train_probes,
)


# --------------------------------------------------------------------------
# Concepts
# --------------------------------------------------------------------------


def test_concept_set_shape():
    cs = all_concepts()
    assert len(cs) == 10
    assert len({c.name for c in cs}) == 10
    for c in cs:
        assert len(c.pairs) >= 16, c.name
        assert len(c.families()) >= 2, c.name
        assert len(c.eval_prompts) >= 4, c.name
        assert c.behavior_question.strip()


def test_pairs_differ_and_are_length_matched():
    for c in all_concepts():
        for p in c.pairs:
            assert p.positive != p.negative, c.name
        pos_len = np.mean([len(p.positive.split()) for p in c.pairs])
        neg_len = np.mean([len(p.negative.split()) for p in c.pairs])
        if c.name == "verbosity":
            continue  # length *is* the concept here
        # Minimal pairs should not be separable on length alone.
        assert abs(pos_len - neg_len) < 0.5 * max(pos_len, neg_len), c.name


def test_eval_prompts_disjoint_from_pairs():
    # Revision R5: the direction must not be derived from the eval set.
    for c in all_concepts():
        pair_text = {p.positive for p in c.pairs} | {p.negative for p in c.pairs}
        assert not (set(c.eval_prompts) & pair_text), c.name


def test_family_split_is_disjoint_and_nonempty():
    for c in all_concepts():
        held = default_held_out(c)
        train, test = c.split_by_family(held)
        assert train and test
        assert not ({p.family for p in train} & {p.family for p in test})


def test_split_by_family_rejects_unknown_family():
    c = get_concept("sentiment")
    with pytest.raises(ValueError, match="unknown families"):
        c.split_by_family({"nonexistent"})


def test_safety_subset():
    names = {c.name for c in safety_concepts()}
    assert {"refusal", "honesty", "sycophancy"} <= names


# --------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------


def _synthetic_acts(n_layers, n_texts, d_model, labels, signal_layer, strength, seed=0):
    """Activations where the concept is linearly present only at signal_layer."""
    rng = np.random.default_rng(seed)
    acts = rng.normal(size=(n_layers, n_texts, d_model)).astype(np.float16)
    direction = rng.normal(size=d_model)
    direction /= np.linalg.norm(direction)
    signed = np.where(labels == 1, 1.0, -1.0)[:, None]
    acts[signal_layer] = (
        acts[signal_layer].astype(np.float32) + strength * signed * direction
    ).astype(np.float16)
    return acts, direction


def test_probe_finds_the_planted_layer():
    c = get_concept("sentiment")
    _, labels = pair_texts(c.pairs)
    families = [p.family for p in c.pairs for _ in (0, 1)]
    held = default_held_out(c)
    train_mask = np.array([f not in held for f in families])

    acts, _ = _synthetic_acts(6, len(labels), 32, labels, signal_layer=3, strength=3.0)
    res = train_probes(c, acts, labels, train_mask, "synthetic", seeds=(0,))

    assert res.best_layer == 3
    assert res.readability > 0.9
    # Layers without signal should be near chance.
    assert res.per_layer[0].auroc < 0.75


def test_control_probe_is_at_chance_so_selectivity_is_real():
    c = get_concept("sentiment")
    _, labels = pair_texts(c.pairs)
    families = [p.family for p in c.pairs for _ in (0, 1)]
    held = default_held_out(c)
    train_mask = np.array([f not in held for f in families])

    acts, _ = _synthetic_acts(4, len(labels), 32, labels, signal_layer=2, strength=3.0)
    res = train_probes(c, acts, labels, train_mask, "synthetic", seeds=(0, 1))

    assert abs(res.control_auroc - 0.5) < 0.25
    assert res.selectivity > 0.2


def test_no_signal_gives_chance_readability():
    c = get_concept("formality")
    _, labels = pair_texts(c.pairs)
    families = [p.family for p in c.pairs for _ in (0, 1)]
    held = default_held_out(c)
    train_mask = np.array([f not in held for f in families])

    acts, _ = _synthetic_acts(3, len(labels), 32, labels, signal_layer=0, strength=0.0)
    res = train_probes(c, acts, labels, train_mask, "synthetic", seeds=(0,))
    assert res.readability < 0.8


def test_diff_of_means_recovers_planted_direction():
    labels = np.array([1, 1, 1, 0, 0, 0])
    rng = np.random.default_rng(0)
    d = rng.normal(size=16)
    d /= np.linalg.norm(d)
    acts = rng.normal(scale=0.05, size=(6, 16)) + np.where(labels == 1, 1, -1)[:, None] * d
    got = diff_of_means_direction(acts, labels)
    assert abs(float(np.dot(got, d))) > 0.95


def test_repe_reading_vector_recovers_planted_direction():
    labels = np.array([1, 1, 1, 0, 0, 0])
    rng = np.random.default_rng(0)
    d = rng.normal(size=16)
    d /= np.linalg.norm(d)
    acts = rng.normal(scale=0.05, size=(6, 16)) + np.where(labels == 1, 1, -1)[:, None] * d
    got = repe_reading_vector(acts, labels)
    # RepE PC1 should align closely with the planted direction.
    assert abs(float(np.dot(got, d))) > 0.90


def test_repe_aligns_with_diff_of_means():
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    rng = np.random.default_rng(42)
    d = rng.normal(size=32)
    d /= np.linalg.norm(d)
    acts = rng.normal(scale=0.1, size=(8, 32)) + np.where(labels == 1, 1, -1)[:, None] * d
    dom = diff_of_means_direction(acts, labels)
    repe = repe_reading_vector(acts, labels)
    # Should be positively aligned (sign-corrected in the implementation).
    assert float(np.dot(dom, repe)) > 0.8


def test_probe_result_ci_brackets_readability():
    c = get_concept("sentiment")
    _, labels = pair_texts(c.pairs)
    families = [p.family for p in c.pairs for _ in (0, 1)]
    held = default_held_out(c)
    train_mask = np.array([f not in held for f in families])
    acts, _ = _synthetic_acts(3, len(labels), 32, labels, signal_layer=1, strength=2.0)
    res = train_probes(c, acts, labels, train_mask, "synthetic", seeds=(0,))
    lo, hi = res.auroc_ci
    assert lo <= res.readability <= hi


# --------------------------------------------------------------------------
# Steering metrics
# --------------------------------------------------------------------------


def test_steering_spec_rejects_non_unit_direction():
    with pytest.raises(ValueError, match="unit norm"):
        st.SteeringSpec(direction=np.array([3.0, 4.0]), layers=[0])


def test_steering_spec_requires_clamp_target():
    d = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="clamp_target"):
        st.SteeringSpec(direction=d, layers=[0], variant="clamp")


def test_steering_spec_rejects_unknown_variant():
    d = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="variant must be"):
        st.SteeringSpec(direction=d, layers=[0], variant="teleport")


def _curve(pairs, ppl=10.0, rep=0.0):
    return [
        st.DosePoint(coeff=c, behavior=b, behavior_ci=(b, b),
                     perplexity=ppl, repetition=rep, broken=False)
        for c, b in pairs
    ]


def test_flat_curve_is_zero_controllability():
    curve = _curve([(-2, 0.5), (0, 0.5), (2, 0.5)])
    assert st.dose_response_auc(curve, baseline=0.5) == pytest.approx(0.0)


def test_steep_curve_beats_flat_curve():
    flat = _curve([(-2, 0.5), (0, 0.5), (2, 0.5)])
    steep = _curve([(-2, 0.0), (0, 0.5), (2, 1.0)])
    assert st.dose_response_auc(steep, 0.5) > st.dose_response_auc(flat, 0.5)


def test_controllability_is_direction_agnostic():
    # Suppressing a behavior counts as much as amplifying it.
    up = _curve([(0, 0.5), (2, 1.0)])
    down = _curve([(0, 0.5), (2, 0.0)])
    assert st.dose_response_auc(up, 0.5) == pytest.approx(st.dose_response_auc(down, 0.5))


def test_broken_points_excluded_from_controllability():
    curve = _curve([(0, 0.5), (1, 0.5), (2, 1.0)])
    curve[-1].broken = True
    assert st.dose_response_auc(curve, 0.5) == pytest.approx(0.0)


def test_ceiling_triggers_on_perplexity():
    curve = _curve([(0, 0.5), (1, 0.6)])
    curve.append(
        st.DosePoint(coeff=2, behavior=0.9, behavior_ci=(0.9, 0.9),
                     perplexity=100.0, repetition=0.0, broken=False)
    )
    usable, reason = st.find_ceiling(curve, baseline_ppl=10.0, ppl_ratio=2.0)
    assert usable == 1.0
    assert "perplexity" in reason


def test_ceiling_triggers_on_repetition():
    curve = _curve([(0, 0.5), (1, 0.6)])
    curve.append(
        st.DosePoint(coeff=2, behavior=0.9, behavior_ci=(0.9, 0.9),
                     perplexity=10.0, repetition=0.9, broken=False)
    )
    usable, reason = st.find_ceiling(curve, baseline_ppl=10.0)
    assert usable == 1.0
    assert "repetition" in reason


def test_ceiling_reports_when_nothing_breaks():
    curve = _curve([(0, 0.5), (1, 0.6), (2, 0.7)])
    usable, reason = st.find_ceiling(curve, baseline_ppl=10.0)
    assert usable == 2.0
    assert "no breakage" in reason


def test_mark_broken_is_symmetric_in_sign():
    curve = _curve([(-3, 0.1), (0, 0.5), (3, 0.9)])
    st.mark_broken(curve, max_usable=1.0)
    assert [p.broken for p in curve] == [True, False, True]


def test_bootstrap_ci_brackets_point_estimate():
    per_prompt = {0.0: [0.5] * 8, 1.0: [0.8] * 8, 2.0: [0.9] * 8}
    curve = _curve([(0.0, 0.5), (1.0, 0.8), (2.0, 0.9)])
    point = st.dose_response_auc(curve, 0.5)
    lo, hi = st.bootstrap_curve_ci(per_prompt, baseline=0.5)
    assert lo <= point <= hi + 1e-9


def test_bootstrap_ci_respects_the_fluency_ceiling():
    """The interval and the point estimate must be the same estimand.

    A curve that is flat below the ceiling and jumps above it used to report
    controllability 0.025 with a CI of [0.1375, 0.1375]: five times the
    estimate, excluding it, and computed entirely from text the ceiling had
    already ruled degenerate.
    """
    per_prompt = {0.0: [0.5] * 8, 1.0: [0.55] * 8, 2.0: [0.95] * 8}
    curve = _curve([(0.0, 0.5), (1.0, 0.55), (2.0, 0.95)])
    st.mark_broken(curve, max_usable=1.0)
    point = st.dose_response_auc(curve, 0.5)
    lo, hi = st.bootstrap_curve_ci(per_prompt, baseline=0.5, max_usable=1.0)
    assert lo <= point <= hi + 1e-9

    # Without the ceiling the interval describes a different quantity entirely.
    bad_lo, bad_hi = st.bootstrap_curve_ci(per_prompt, baseline=0.5)
    assert not (bad_lo <= point <= bad_hi + 1e-9)


def test_layer_band_stays_in_range():
    """P4: probe layer +/- 20% of depth, sampled at four evenly spaced layers.

    The band is clipped at the model boundary but stays evenly spaced inside
    what is left of it -- it is not re-centred inward, because that would
    silently steer at layers the preregistered band never covered.
    """
    assert st.layer_band(0, 32, width=4) == [0, 2, 4, 6]
    assert st.layer_band(31, 32, width=4) == [25, 27, 29, 31]
    assert len(st.layer_band(16, 32, width=4)) == 4
    for best in (0, 5, 16, 31):
        band = st.layer_band(best, 32, width=4)
        assert band == sorted(set(band))
        assert all(0 <= li < 32 for li in band)
        assert min(band) <= best <= max(band)


def test_layer_band_collapses_when_the_model_is_shallow():
    """A band wider than the model returns every available layer, not duplicates."""
    band = st.layer_band(2, 6, width=4)
    assert band == sorted(set(band))
    assert all(0 <= li < 6 for li in band)


# --------------------------------------------------------------------------
# Behavior scoring
# --------------------------------------------------------------------------


def test_lexicon_scorer_separates_clear_cases():
    s = bh.LexiconScorer()
    pos = s.score(["This was excellent and I loved it, truly wonderful."], "sentiment")[0]
    neg = s.score(["This was terrible and awful, a total waste."], "sentiment")[0]
    assert pos > 0.8 and neg < 0.2


def test_lexicon_scorer_neutral_on_no_hits():
    s = bh.LexiconScorer()
    assert s.score(["The table has four legs."], "sentiment")[0] == 0.5


def test_lexicon_refusal_direction():
    s = bh.LexiconScorer()
    ref = s.score(["I cannot help with that, sorry."], "refusal")[0]
    comp = s.score(["Sure, here is how you can do this. First, open the app."], "refusal")[0]
    assert ref > comp


def test_verbosity_uses_length():
    s = bh.LexiconScorer()
    short = s.score(["Yes."], "verbosity")[0]
    long = s.score(["word " * 100], "verbosity")[0]
    assert short < long == 1.0


def test_lexicon_scorer_rejects_unknown_concept():
    with pytest.raises(KeyError):
        bh.LexiconScorer().score(["text"], "not_a_concept")


def test_repetition_score():
    assert bh.repetition_score("the cat sat on the mat and then slept well") < 0.2
    assert bh.repetition_score(("spam " * 30).strip()) > 0.8
    assert bh.repetition_score("too short") == 0.0


def test_judge_agreement_perfect_and_anti():
    a = [0.1, 0.4, 0.6, 0.9]
    assert bh.judge_agreement(a, a).pearson == pytest.approx(1.0)
    assert bh.judge_agreement(a, a[::-1]).pearson == pytest.approx(-1.0)


def test_judge_agreement_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        bh.judge_agreement([0.1, 0.2], [0.1])


def test_krippendorff_alpha_catches_what_correlation_hides():
    """Two judges offset by a constant correlate perfectly and agree on nothing.

    The behaviour axis is read in absolute terms against a fixed danger-zone
    threshold, so a systematic offset between judges matters and Pearson cannot
    see it. PLAN.md row 14 promises alpha; this is why.
    """
    a = [0.1, 0.4, 0.9, 0.3, 0.7, 0.5]
    offset = [x + 0.4 for x in a]
    ag = bh.judge_agreement(a, offset)
    assert ag.pearson == pytest.approx(1.0)
    assert ag.krippendorff_alpha < 0.5


def test_krippendorff_alpha_bounds():
    a = [0.1, 0.4, 0.9, 0.3, 0.7, 0.5]
    assert bh.krippendorff_alpha_interval([a, a]) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    independent = list(rng.uniform(size=len(a)))
    assert abs(bh.krippendorff_alpha_interval([a, independent])) < 0.5
    assert np.isnan(bh.krippendorff_alpha_interval([[0.5] * 4, [0.5] * 4]))


def test_krippendorff_alpha_tolerates_missing_ratings():
    """A human spot-check covers 100 of N outputs, so most cells are empty."""
    nan = float("nan")
    a = [0.1, 0.4, 0.9, 0.3, 0.7]
    human = [0.1, nan, 0.9, nan, 0.7]
    alpha = bh.krippendorff_alpha_interval([a, human])
    assert alpha == pytest.approx(1.0)


def test_krippendorff_alpha_needs_two_raters():
    with pytest.raises(ValueError, match="2 raters"):
        bh.krippendorff_alpha_interval([[0.1, 0.2, 0.3]])


def test_lexicon_scorer_does_not_match_inside_words():
    """`str.count` made "but" fire on "contribution" and "will" on future tense."""
    s = bh.LexiconScorer()
    flattery = s.score(
        ["You are absolutely right, a brilliant contribution."], "sycophancy"
    )[0]
    assert flattery > 0.9, "substring match on 'but' used to drag this to 0.67"
    hedged = s.score(["I am unsure, but it will possibly happen."], "certainty")[0]
    assert hedged < 0.1, "substring match on 'will' used to lift this to 0.33"


def test_lexicon_entries_with_punctuation_still_match():
    s = bh.LexiconScorer()
    assert s.score(["Here is how: first, open the file."], "refusal")[0] < 0.1
    assert s.score(["I can't help with that, sorry."], "refusal")[0] > 0.9


def test_llm_judge_parses_and_falls_back():
    judge = bh.LLMJudgeScorer(
        generate_fn=lambda prompts: ["0.8", "garbage", "1.0"],
        behavior_questions={"sentiment": "Score it."},
    )
    assert judge.score(["a", "b", "c"], "sentiment") == [0.8, 0.5, 1.0]


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def test_participation_ratio_low_for_rank_one_cloud():
    rng = np.random.default_rng(0)
    d = rng.normal(size=32)
    acts = np.outer(rng.normal(size=200), d)
    assert geo.participation_ratio(acts) < 1.5


def test_participation_ratio_high_for_isotropic_cloud():
    rng = np.random.default_rng(0)
    acts = rng.normal(size=(400, 32))
    assert geo.participation_ratio(acts) > 20


def test_n_effective_directions():
    d = np.array([1.0, 0.0, 0.0])
    shared = {"a": d, "b": d, "c": d}
    assert geo.n_effective_directions(shared) == 1.0

    spread = {
        "a": np.array([1.0, 0.0, 0.0]),
        "b": np.array([0.0, 1.0, 0.0]),
        "c": np.array([0.0, 0.0, 1.0]),
    }
    assert geo.n_effective_directions(spread) >= 2.0


def test_direction_coherence_bounds():
    d = np.array([1.0, 0.0])
    assert geo.direction_coherence({"a": d, "b": d}) == pytest.approx(1.0)
    assert geo.direction_coherence({"a": d, "b": -d}) == pytest.approx(-1.0)
    assert geo.direction_coherence({"a": d}) == 1.0


def test_output_overlap_is_high_inside_head_rowspace():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(64, 16))
    inside = W[0] / np.linalg.norm(W[0])
    assert geo.output_overlap(inside, W, top_k=15) > 0.8


def test_output_overlap_caches_the_basis_per_matrix():
    """Same head reuses the SVD; a different head must not.

    The basis depends on the model, not the concept, but output_overlap is
    called once per concept. At a 152k x 3584 head that was about 40 seconds
    and a 2.2 GB float32 view per call, ten times per model.
    """
    rng = np.random.default_rng(0)
    W = rng.normal(size=(64, 16))
    a = geo.output_overlap(W[0] / np.linalg.norm(W[0]), W, top_k=15)
    b = geo.output_overlap(W[0] / np.linalg.norm(W[0]), W, top_k=15)
    assert a == b, "a cached basis must be deterministic"

    # A different matrix of the same shape must get its own basis.
    W2 = np.zeros((64, 16))
    W2[:, :8] = rng.normal(size=(64, 8))
    outside = np.zeros(16)
    outside[12] = 1.0
    assert geo.output_overlap(outside, W2, top_k=7) < 0.2

    # And the first matrix still reports what it did before.
    assert geo.output_overlap(W[0] / np.linalg.norm(W[0]), W, top_k=15) == a


def test_output_overlap_cache_survives_a_reused_address():
    """A freed array whose id() is recycled must not serve a stale basis."""
    rng = np.random.default_rng(1)
    inside_vec = None
    first = None
    for _ in range(30):
        W = rng.normal(size=(64, 16))
        vec = W[0] / np.linalg.norm(W[0])
        got = geo.output_overlap(vec, W, top_k=15)
        # Every fresh matrix must score its own first row highly; a stale
        # cached basis from a previous, freed matrix would not.
        assert got > 0.8, got
        if first is None:
            first, inside_vec = got, vec
    assert first is not None and inside_vec is not None


def test_output_overlap_low_outside_rowspace():
    rng = np.random.default_rng(0)
    # A head that only spans the first 8 coordinates cannot see the last 8.
    W = np.zeros((64, 16))
    W[:, :8] = rng.normal(size=(64, 8))
    outside = np.zeros(16)
    outside[12] = 1.0
    assert geo.output_overlap(outside, W, top_k=7) < 0.2


def test_low_variance_pc_alignment_high_for_low_var_direction():
    rng = np.random.default_rng(0)
    # Cloud with most variance in first 8 dims, little in last 8.
    acts = np.zeros((200, 16), dtype=np.float32)
    acts[:, :8] = rng.normal(scale=10.0, size=(200, 8))
    acts[:, 8:] = rng.normal(scale=0.01, size=(200, 8))
    # A direction in the low-variance subspace.
    d = np.zeros(16)
    d[12] = 1.0
    align = geo.low_variance_pc_alignment(d, acts, bottom_fraction=0.5)
    assert align > 0.8


def test_low_variance_pc_alignment_low_for_high_var_direction():
    rng = np.random.default_rng(0)
    acts = np.zeros((200, 16), dtype=np.float32)
    acts[:, :8] = rng.normal(scale=10.0, size=(200, 8))
    acts[:, 8:] = rng.normal(scale=0.01, size=(200, 8))
    # A direction in the high-variance subspace.
    d = np.zeros(16)
    d[0] = 1.0
    align = geo.low_variance_pc_alignment(d, acts, bottom_fraction=0.5)
    assert align < 0.2


def _features(n, gaps, seed=0):
    rng = np.random.default_rng(seed)
    feats = []
    for i in range(n):
        feats.append(
            geo.GeometryFeatures(
                concept=f"c{i}",
                model="m",
                layer=10,
                # output_overlap carries the signal; the rest is noise.
                output_overlap=1.0 - gaps[i] + rng.normal(scale=0.01),
                participation_ratio=float(rng.normal(20, 3)),
                low_variance_pc_alignment=float(rng.uniform(0, 0.5)),
                residual_norm=float(rng.normal(5, 1)),
                n_directions=float(rng.integers(1, 4)),
                direction_coherence=float(rng.uniform(0, 1)),
                probe_dom_cosine=float(rng.uniform(0.5, 1)),
            )
        )
    return feats


def test_predictor_finds_a_real_signal():
    gaps = list(np.linspace(0.0, 1.0, 20))
    report = geo.fit_gap_predictor(_features(20, gaps), gaps)
    assert report.r2_loo > 0.3
    assert "geometry predicts" in report.verdict


def test_predictor_reports_honest_failure_on_noise():
    rng = np.random.default_rng(1)
    gaps = list(rng.uniform(size=20))
    shuffled = list(rng.permutation(gaps))
    report = geo.fit_gap_predictor(_features(20, shuffled), gaps)
    assert report.r2_loo < 0.3
    assert "predict" in report.verdict


def test_predictor_needs_enough_points():
    gaps = [0.1, 0.2, 0.3]
    with pytest.raises(ValueError, match="at least 5"):
        geo.fit_gap_predictor(_features(3, gaps), gaps)


def test_predictor_drops_all_nan_feature():
    gaps = list(np.linspace(0, 1, 12))
    feats = _features(12, gaps)  # tuning_shift defaults to NaN throughout
    report = geo.fit_gap_predictor(feats, gaps)
    assert "tuning_shift" not in report.features_used


def test_predictor_says_so_when_the_primary_test_was_not_run():
    """Without the raw axes the preregistered test is impossible, not optional.

    The old behaviour was to report the ridge model's leave-one-out prediction
    correlation as `partial_spearman`, with a hardcoded (nan, nan) CI and the
    R^2 verdict copied across -- an unrun test presented as a run one.
    """
    gaps = list(np.linspace(0.0, 1.0, 12))
    report = geo.fit_gap_predictor(_features(12, gaps), gaps)
    assert "NOT RUN" in report.primary.verdict
    assert np.isnan(report.primary.partial_spearman)
    assert report.exploratory == []


def test_predictor_runs_the_real_primary_test_when_given_the_raw_axes():
    rng = np.random.default_rng(7)
    n = 14
    readability = list(rng.uniform(0.7, 0.99, size=n))
    controllability = list(rng.uniform(0.0, 0.8, size=n))
    gaps = [r - c for r, c in zip(readability, controllability)]
    report = geo.fit_gap_predictor(
        _features(n, gaps),
        gaps,
        controllabilities=controllability,
        readabilities=readability,
    )
    assert "NOT RUN" not in report.primary.verdict
    assert not np.isnan(report.primary.partial_spearman)
    lo, hi = report.primary.partial_spearman_ci
    assert not np.isnan(lo) and not np.isnan(hi), "P7 cluster bootstrap must run"
    assert lo <= hi
    assert report.exploratory, "P8 exploratory analyses must run"
    assert all(0.0 <= e.p_value_bh <= 1.0 for e in report.exploratory)


def test_cluster_bootstrap_ci_survives_a_degenerate_resample():
    """One NaN resample must not wipe out the interval.

    np.percentile propagates NaN, so a single degenerate concept resample used
    to turn the CI on the study's one preregistered test into (nan, nan).
    """
    rng = np.random.default_rng(3)
    n = 24
    z = rng.uniform(size=n)
    x = z + rng.normal(scale=0.2, size=n)
    y = x + rng.normal(scale=0.2, size=n)
    # Several concepts appear once, so plenty of resamples are degenerate.
    concepts = [f"c{i}" for i in range(n)]
    lo, hi = geo._cluster_bootstrap_partial_spearman(x, y, z, concepts, n_boot=500)
    assert not np.isnan(lo) and not np.isnan(hi)
    assert lo <= hi


# --------------------------------------------------------------------------
# Gap map
# --------------------------------------------------------------------------


def _point(concept, model, r, c, safety=False):
    return GapPoint(
        concept=concept, model=model, readability=r, controllability=c,
        readability_ci=(r - 0.02, r + 0.02),
        controllability_ci=(c - 0.02, c + 0.02),
        safety_relevant=safety, best_layer=12, selectivity=0.4,
    )


def _point_ci(concept, model, r, c, r_ci, c_ci, safety=False):
    return GapPoint(
        concept=concept, model=model, readability=r, controllability=c,
        readability_ci=r_ci, controllability_ci=c_ci,
        safety_relevant=safety, best_layer=12, selectivity=0.4,
    )


def test_danger_zone_requires_the_ci_to_exclude_the_threshold():
    """P6: clearing the threshold on the point estimate alone is not enough.

    The headline claim is that a concept is readable and immovable. A concept
    at controllability 0.04 whose interval runs to 0.30 has not shown that; it
    has shown scatter, which is the same error R10 caught with normalization.
    """
    tight = _point_ci("tight", "m", 0.97, 0.01, (0.94, 0.99), (0.00, 0.03))
    wide = _point_ci("wide", "m", 0.97, 0.04, (0.94, 0.99), (0.00, 0.30))
    borderline = _point_ci("edge", "m", 0.905, 0.01, (0.88, 0.93), (0.00, 0.02))
    filler = _point_ci("filler", "m", 0.60, 0.60, (0.55, 0.65), (0.55, 0.65))

    gm = build_gap_map([tight, wide, borderline, filler])
    assert [p.concept for p in gm.danger_zone] == ["tight"]

    # The point-estimate-only rule admits all three, which is the old behaviour.
    loose = build_gap_map(
        [tight, wide, borderline, filler], require_ci_exclusion=False
    )
    assert sorted(p.concept for p in loose.danger_zone) == ["edge", "tight", "wide"]


def test_danger_zone_rejects_a_point_with_an_unknown_ci():
    """A NaN bound is not an excluding one."""
    nan = float("nan")
    unknown = _point_ci("unknown", "m", 0.97, 0.01, (0.94, 0.99), (nan, nan))
    filler = _point_ci("filler", "m", 0.60, 0.60, (0.55, 0.65), (0.55, 0.65))
    other = _point_ci("other", "m", 0.70, 0.40, (0.65, 0.75), (0.35, 0.45))
    gm = build_gap_map([unknown, filler, other])
    assert gm.danger_zone == []


def test_gap_map_ci_clusters_by_concept():
    """P7: one concept across many models is not many independent points.

    Replicating each concept across models must not shrink the interval the
    way independent observations would.
    """
    from lbi.gapmap import _bootstrap_spearman_ci

    rng = np.random.default_rng(0)
    n_concepts = 8
    base_x = rng.uniform(size=n_concepts)
    base_y = base_x + rng.normal(scale=0.4, size=n_concepts)

    # Five near-identical copies of each concept, one per "model".
    concepts, xs, ys = [], [], []
    for i in range(n_concepts):
        for _ in range(5):
            concepts.append(f"c{i}")
            xs.append(base_x[i] + rng.normal(scale=0.01))
            ys.append(base_y[i] + rng.normal(scale=0.01))
    xs, ys = np.array(xs), np.array(ys)

    iid_lo, iid_hi = _bootstrap_spearman_ci(xs, ys, concepts=None)
    cl_lo, cl_hi = _bootstrap_spearman_ci(xs, ys, concepts=concepts)
    assert (cl_hi - cl_lo) > (iid_hi - iid_lo), (
        "clustering by concept must widen the interval, not narrow it: "
        f"iid={iid_hi - iid_lo:.3f} clustered={cl_hi - cl_lo:.3f}"
    )


def test_normalization_removes_cross_model_scale_confound():
    pts = [
        _point("a", "big", 0.90, 0.90), _point("b", "big", 0.70, 0.50),
        _point("c", "big", 0.50, 0.10),
        # Same shape, compressed scale: normalization should align them.
        _point("a", "small", 0.49, 0.29), _point("b", "small", 0.39, 0.19),
        _point("c", "small", 0.29, 0.09),
    ]
    normed = normalize_within_model(pts)
    for model in ("big", "small"):
        vals = [p.norm_readability for p in normed if p.model == model]
        assert min(vals) == pytest.approx(0.0) and max(vals) == pytest.approx(1.0)


def test_normalization_preserves_raw_values():
    # Raw values carry the danger-zone decision, so they must survive.
    pts = [_point("a", "m", 0.9, 0.8), _point("b", "m", 0.5, 0.2),
           _point("c", "m", 0.7, 0.5)]
    normalize_within_model(pts)
    assert pts[0].readability == pytest.approx(0.9)
    assert pts[1].controllability == pytest.approx(0.2)


def test_danger_zone_detected():
    pts = [
        _point("sentiment", "m", 0.95, 0.90),
        _point("formality", "m", 0.85, 0.60),
        _point("refusal", "m", 0.93, 0.02, safety=True),
        _point("topic", "m", 0.55, 0.30),
    ]
    gm = build_gap_map(pts)
    assert [p.concept for p in gm.danger_zone] == ["refusal"]
    assert "Danger zone" in gm.interpretation


def test_normalization_cannot_manufacture_a_danger_zone():
    # Every concept is genuinely steerable; min-max still pins someone to
    # norm-controllability 0.0, and that must not read as "immovable".
    pts = [
        _point("a", "m", 0.95, 0.60), _point("b", "m", 0.80, 0.45),
        _point("c", "m", 0.92, 0.40), _point("d", "m", 0.70, 0.55),
    ]
    gm = build_gap_map(pts)
    assert gm.danger_zone == []
    assert min(p.norm_controllability for p in gm.points) == pytest.approx(0.0)


def test_tight_correlation_gets_the_reframe_message():
    pts = [_point(f"c{i}", "m", v, v) for i, v in enumerate([0.5, 0.6, 0.7, 0.8, 0.9])]
    gm = build_gap_map(pts)
    assert gm.spearman > 0.9
    assert "Reframe" in gm.interpretation


def test_pure_scatter_gets_the_unpredictable_message():
    pts = [
        _point("a", "m", 0.20, 0.55), _point("b", "m", 0.40, 0.35),
        _point("c", "m", 0.60, 0.60), _point("d", "m", 0.55, 0.30),
        _point("e", "m", 0.35, 0.50),
    ]
    gm = build_gap_map(pts)
    assert gm.danger_zone == []
    assert "not predictable" in gm.interpretation or "No reliable" in gm.interpretation


def test_gap_uses_normalized_values_after_normalization():
    pts = [_point("a", "m", 0.9, 0.9), _point("b", "m", 0.9, 0.1),
           _point("c", "m", 0.5, 0.5)]
    gm = build_gap_map(pts)
    b = next(p for p in gm.points if p.concept == "b")
    assert b.gap == pytest.approx(b.norm_readability - b.norm_controllability)


def test_gap_map_needs_points():
    with pytest.raises(ValueError, match="at least 3"):
        build_gap_map([_point("a", "m", 0.5, 0.5)])


def test_gap_is_readability_minus_controllability():
    p = _point("a", "m", 0.9, 0.1)
    assert p.gap == pytest.approx(0.8)


def test_gap_map_serializes(tmp_path):
    pts = [
        _point("a", "m", 0.9, 0.9), _point("b", "m", 0.9, 0.05),
        _point("c", "m", 0.4, 0.4),
    ]
    gm = build_gap_map(pts)
    out = tmp_path / "gap_map.json"
    gm.save(str(out))
    assert out.exists()
    import json

    loaded = json.loads(out.read_text())
    assert len(loaded["points"]) == 3
    assert "interpretation" in loaded


def test_figures_render(tmp_path):
    from lbi.gapmap import plot_dose_response, plot_gap_map

    pts = [
        _point("a", "m1", 0.9, 0.8), _point("b", "m1", 0.9, 0.05, safety=True),
        _point("c", "m2", 0.5, 0.4), _point("d", "m2", 0.7, 0.6),
    ]
    gm = build_gap_map(pts)
    f1 = plot_gap_map(gm, str(tmp_path / "fig1.png"))
    f2 = plot_dose_response(
        {
            "sentiment (controllable)": [(-2, 0.1), (0, 0.5), (2, 0.9)],
            "refusal (immovable)": [(-2, 0.5), (0, 0.5), (2, 0.5)],
        },
        str(tmp_path / "fig2.png"),
    )
    import os

    assert os.path.getsize(f1) > 1000 and os.path.getsize(f2) > 1000
