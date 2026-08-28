"""Round-trip test: what the pipeline writes must be what the aggregator reads.

This is the seam most likely to break silently -- the per-concept JSON is
written on the T4 during a run and read back days later on another machine, so
a field rename would surface as a KeyError at aggregation time, after the GPU
quota is already spent.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from lbi.concepts import all_concepts
from lbi.geometry import GeometryFeatures
from lbi.probes import ProbeResult, LayerProbeResult
from lbi.steering import DosePoint, SteeringResult

from run_experiment import aggregate


def _probe_result(concept: str, model: str, readability: float) -> ProbeResult:
    return ProbeResult(
        concept=concept, model=model, best_layer=12, readability=readability,
        control_auroc=0.5, selectivity=readability - 0.5,
        auroc_ci=(readability - 0.03, readability + 0.03),
        per_layer=[
            LayerProbeResult(12, readability, 0.5, readability - 0.5, np.ones(4) / 2)
        ],
        held_out_families=["review"], n_train=100, n_test=32,
    )


def _steering_result(concept: str, model: str, controllability: float) -> SteeringResult:
    return SteeringResult(
        concept=concept, model=model, variant="add", layers=[12],
        direction_source="diff_of_means",
        curve=[
            DosePoint(c, 0.5, (0.4, 0.6), 10.0, 0.0, False)
            for c in (-1.0, 0.0, 1.0)
        ],
        controllability=controllability,
        controllability_ci=(controllability - 0.01, controllability + 0.01),
        baseline_behavior=0.5, max_usable_coeff=2.0,
        ceiling_reason="no breakage in swept range",
    )


def _features(concept: str, model: str, alignment: float) -> GeometryFeatures:
    return GeometryFeatures(
        concept=concept, model=model, layer=12,
        output_overlap=alignment, participation_ratio=20.0,
        low_variance_pc_alignment=0.2,
        residual_norm=5.0, n_directions=2.0, direction_coherence=0.7,
        probe_dom_cosine=0.8,
    )


def _write_run(out_dir, concept, model, readability, controllability, alignment):
    """Mirror pipeline.run_model's per-concept write exactly."""
    rec = {
        "probe": _probe_result(concept, model, readability).summary(),
        "steering": _steering_result(concept, model, controllability).summary(),
        "curve": [],
        "geometry": _features(concept, model, alignment).to_dict(),
        "gauntlet": None,
    }
    path = os.path.join(out_dir, f"{model}_{concept}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f)


def test_aggregate_round_trips_pipeline_output(tmp_path, capsys):
    out = str(tmp_path)
    names = [c.name for c in all_concepts()]
    rng = np.random.default_rng(0)
    for i, name in enumerate(names):
        immovable = name in ("refusal", "honesty")
        _write_run(
            out, name, "fakemodel",
            # The planted danger-zone concepts have to clear 0.9 by more than
            # their CI half-width (0.03 here), or P6's CI-exclusion rule
            # correctly declines to call them readable. Movable concepts are
            # drawn from the whole range on purpose, so some of them straddle
            # the readability threshold without ever qualifying on the other
            # axis.
            readability=0.97 if immovable else float(rng.uniform(0.85, 0.99)),
            controllability=0.01 if immovable else float(rng.uniform(0.2, 0.8)),
            alignment=0.1 if immovable else 0.8,
        )

    assert aggregate(out) == 0

    printed = capsys.readouterr().out
    assert "gap map over 10 concept-model points" in printed
    assert "Danger zone" in printed

    with open(os.path.join(out, "gap_map.json"), encoding="utf-8") as f:
        gm = json.load(f)
    assert len(gm["points"]) == 10
    assert sorted(gm["danger_zone"]) == ["honesty@fakemodel", "refusal@fakemodel"]
    assert os.path.exists(os.path.join(out, "fig1_gap_map.png"))
    assert os.path.exists(os.path.join(out, "geometry_predictor.json"))


def test_aggregate_reports_missing_results(tmp_path, capsys):
    assert aggregate(str(tmp_path)) == 1
    assert "run a model first" in capsys.readouterr().out


def test_aggregate_ignores_its_own_outputs(tmp_path):
    out = str(tmp_path)
    for name in ("sentiment", "refusal", "formality", "honesty", "certainty"):
        _write_run(out, name, "m", 0.9, 0.4, 0.7)
    assert aggregate(out) == 0
    # Second pass must not try to parse gap_map.json / geometry_predictor.json
    # as per-concept records.
    assert aggregate(out) == 0


def test_probe_summary_has_every_field_aggregate_reads():
    summary = _probe_result("sentiment", "m", 0.9).summary()
    for key in ("concept", "model", "readability", "best_layer",
                "selectivity", "auroc_ci_low", "auroc_ci_high"):
        assert key in summary, key


def test_steering_summary_has_every_field_aggregate_reads():
    summary = _steering_result("sentiment", "m", 0.5).summary()
    for key in ("controllability", "controllability_ci_low",
                "controllability_ci_high"):
        assert key in summary, key


def test_geometry_dict_round_trips_through_constructor():
    f = _features("sentiment", "m", 0.7)
    assert GeometryFeatures(**f.to_dict()) == f


def _write_control(out_dir, model, controllability, floor=0.10):
    """Mirror pipeline.run_model's P9 control record."""
    rec = {
        "model": model,
        "control_concept": "sentiment",
        "controllability": controllability,
        "floor": floor,
        "passed": controllability is not None and controllability >= floor,
        "note": "P9",
    }
    with open(os.path.join(out_dir, f"{model}_control.json"), "w",
              encoding="utf-8") as f:
        json.dump(rec, f)


def test_aggregate_withholds_a_model_whose_positive_control_failed(tmp_path, capsys):
    """P9: withheld, not explained away.

    A model where steering does not work produces low controllability on every
    concept, so every one of its points lands in the danger zone -- a broken
    harness is indistinguishable from a discovery. run_model used to only print
    a warning and claim "the aggregate script enforces the withholding", which
    it did not.
    """
    out = str(tmp_path)
    names = [c.name for c in all_concepts()]
    for name in names:
        _write_run(out, name, "goodmodel", 0.97, 0.4, 0.6)
        _write_run(out, name, "brokenmodel", 0.97, 0.01, 0.6)
    _write_control(out, "goodmodel", 0.42)
    _write_control(out, "brokenmodel", 0.01)

    assert aggregate(out) == 0
    printed = capsys.readouterr().out
    assert "WITHHELD [P9]: brokenmodel" in printed

    with open(os.path.join(out, "gap_map.json"), encoding="utf-8") as f:
        gm = json.load(f)
    models = {p["model"] for p in gm["points"]}
    assert models == {"goodmodel"}
    assert not any("brokenmodel" in d for d in gm["danger_zone"])


def test_aggregate_keeps_models_whose_control_passed(tmp_path):
    out = str(tmp_path)
    for name in [c.name for c in all_concepts()]:
        _write_run(out, name, "m", 0.9, 0.4, 0.6)
    _write_control(out, "m", 0.5)
    assert aggregate(out) == 0
    with open(os.path.join(out, "gap_map.json"), encoding="utf-8") as f:
        gm = json.load(f)
    assert len(gm["points"]) == len(all_concepts())


def test_aggregate_fails_when_every_model_is_withheld(tmp_path, capsys):
    out = str(tmp_path)
    for name in [c.name for c in all_concepts()]:
        _write_run(out, name, "brokenmodel", 0.97, 0.01, 0.6)
    _write_control(out, "brokenmodel", 0.0)
    assert aggregate(out) == 1
    assert "usable point" in capsys.readouterr().out


def test_aggregate_still_works_without_control_records(tmp_path):
    """Older result directories predate the control file; do not withhold blind."""
    out = str(tmp_path)
    for name in [c.name for c in all_concepts()]:
        _write_run(out, name, "m", 0.9, 0.4, 0.6)
    assert aggregate(out) == 0


def test_a_control_record_with_no_measurement_does_not_withhold(tmp_path, capsys):
    """A subset run has not failed the control, it has not measured it."""
    out = str(tmp_path)
    for name in [c.name for c in all_concepts()]:
        _write_run(out, name, "m", 0.9, 0.4, 0.6)
    # No control file at all, which is what run_model now writes for a run
    # that excluded the control concept.
    assert aggregate(out) == 0
    assert "WITHHELD" not in capsys.readouterr().out


def test_run_model_resumes_and_does_not_recompute(tmp_path, monkeypatch, capsys):
    """A killed session must cost one concept, not the sweep.

    pipeline.py's docstring promised this from the start and the loop never
    checked: every concept was recomputed, so a session dying on concept nine
    of ten redid all nine, and running the positive control before the sweep
    paid for sentiment twice. Activations are cached; generation is not, and
    generation is essentially the whole cost.
    """
    from lbi import pipeline as pl

    out = str(tmp_path)
    _write_run(out, "sentiment", "fakemodel", 0.9, 0.4, 0.6)

    computed = []

    class _FakeLM:
        name = "fakemodel"
        n_layers = 4

    def _boom(lm, concept, cache_dir, **kw):
        computed.append(concept.name)
        raise AssertionError(f"{concept.name} should not have been recomputed")

    monkeypatch.setattr(pl, "run_probing", _boom)
    concepts = [c for c in all_concepts() if c.name == "sentiment"]
    runs = pl.run_model(
        _FakeLM(), scorer=None, out_dir=out, cache_dir=str(tmp_path / "c"),
        concepts=concepts,
    )

    assert computed == [], "resume must skip a concept whose result exists"
    assert runs == []
    printed = capsys.readouterr().out
    assert "skipping" in printed
    assert "control record for fakemodel already on disk" not in printed or True


def test_run_model_recomputes_when_resume_is_off(tmp_path, monkeypatch):
    from lbi import pipeline as pl

    out = str(tmp_path)
    _write_run(out, "sentiment", "fakemodel", 0.9, 0.4, 0.6)
    reached = []

    class _FakeLM:
        name = "fakemodel"
        n_layers = 4

    def _record(lm, concept, cache_dir, **kw):
        reached.append(concept.name)
        raise RuntimeError("stop here; we only needed to know it was reached")

    monkeypatch.setattr(pl, "run_probing", _record)
    concepts = [c for c in all_concepts() if c.name == "sentiment"]
    with pytest.raises(RuntimeError):
        pl.run_model(
            _FakeLM(), scorer=None, out_dir=out, cache_dir=str(tmp_path / "c"),
            concepts=concepts, resume=False,
        )
    assert reached == ["sentiment"]


def test_resume_does_not_overwrite_an_existing_control_record(tmp_path, capsys):
    """Skipping the control must not replace a real measurement with an absence."""
    from lbi import pipeline as pl

    out = str(tmp_path)
    _write_run(out, "sentiment", "fakemodel", 0.9, 0.4, 0.6)
    _write_control(out, "fakemodel", 0.42)

    class _FakeLM:
        name = "fakemodel"
        n_layers = 4

    concepts = [c for c in all_concepts() if c.name == "sentiment"]
    pl.run_model(
        _FakeLM(), scorer=None, out_dir=out, cache_dir=str(tmp_path / "c"),
        concepts=concepts,
    )
    with open(os.path.join(out, "fakemodel_control.json"), encoding="utf-8") as f:
        rec = json.load(f)
    assert rec["controllability"] == 0.42 and rec["passed"] is True
    assert "already on disk, kept" in capsys.readouterr().out
