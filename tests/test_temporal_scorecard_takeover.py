"""Known examples for corrected scoring; thresholds remain pre-registered."""
import copy
import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "examples/CORDEX_ML/cordex_temporal_experiment.py"
_spec = importlib.util.spec_from_file_location("temporal_scorecard_v2", _path)
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)


def report(**overrides):
    metrics = dict(autocorr_lag1_error=-1.0, autocorr_lag2_error=-1.0,
                   tendency_rmse=1.0, acc3d_rmse=1.0, rmse=1.0, mae=1.0,
                   spatial_r_daily_mean=0.8, std_ratio=1.0, tendency_std_ratio=1.0)
    metrics["q0.99_bias"] = -1.0
    metrics.update(overrides)
    return {"variables": {"pr": {"paired": metrics}}}


def improved():
    return report(autocorr_lag1_error=-0.8, autocorr_lag2_error=0.8,
                  tendency_rmse=0.9, acc3d_rmse=0.9)


def test_signed_errors_use_magnitudes_and_dotted_quantile_is_scored():
    assert score.score_acceptance(report(), improved(), ["pr"])["accepted"]
    bad = improved()
    bad["variables"]["pr"]["paired"]["q0.99_bias"] = -2.0
    assert not score.score_acceptance(report(), bad, ["pr"])["pass_guardrail"]
    bad["variables"]["pr"]["paired"]["autocorr_lag1_error"] = -2.0
    assert not score.score_acceptance(report(), bad, ["pr"])["pass_primary"]


def test_missing_nan_and_zero_baselines_cannot_pass():
    for value in (None, float("nan"), 0.0, 1e-14):
        base = report(autocorr_lag1_error=value)
        assert not score.score_acceptance(base, improved(), ["pr"])["accepted"]
    assert not score.score_acceptance(report(), improved(), [])["accepted"]
    missing = copy.deepcopy(improved())
    missing["variables"]["pr"]["paired"].pop("q0.99_bias")
    assert not score.score_acceptance(report(), missing, ["pr"])["accepted"]


def test_missing_control_metrics_cannot_be_beaten():
    reports = {"candidate": improved(), "control": report(autocorr_lag1_error=None)}
    result = score.beats_controls(reports, "candidate", ("control", "missing"), ["pr"])
    assert result["control"]["beats"] is False
    assert result["missing"]["status"] == "unavailable"


def test_exact_absolute_guardrail_definition_is_preserved():
    candidate = improved()
    candidate["variables"]["pr"]["paired"]["std_ratio"] = 0.97
    assert not score.score_acceptance(report(), candidate, ["pr"])["pass_guardrail"]
    assert score.MUST_BEAT == ("spatial_ft", "time_only")
    assert score.EXTRA_MUST_BEAT["native_pair_pretext"] == ("native_pair_nohistory", "native_pair")


def test_spatial_control_budget_counts_decoder_optimizer_updates():
    import pytest
    score.assert_optimizer_budget({"global_step": 600, "trained_temporal_steps": 0}, 600, 1)
    with pytest.raises(RuntimeError):
        score.assert_optimizer_budget({"global_step": 0, "trained_temporal_steps": 600}, 600, 1)


def test_rescore_rejects_invalidated_results_and_unknown_plan(tmp_path):
    import json
    path = _path.with_name("cordex_temporal_rescore_v2.py")
    spec = importlib.util.spec_from_file_location("temporal_rescore_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    variants = ["baseline", "spatial_ft", "time_only", "native_pair_nohistory", "native_pair"]
    manifest = {"variants": {name: {"status": "completed"} for name in variants}}
    for name in variants:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "evaluation.json").write_text(json.dumps(improved() if name == "native_pair" else report()))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert not module.rescore(tmp_path)["complete"]
    assert module.rescore(tmp_path, variants)["variants"]["native_pair"]["scientific_acceptance"]
    manifest["variants"]["native_pair"]["status"] = "invalidated"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = module.rescore(tmp_path, variants)
    assert not result["complete"]
    assert not result["variants"]["native_pair"]["scientific_acceptance"]


def test_rescore_uses_recorded_recovery_evaluation_without_stale_fallback(tmp_path):
    import json
    spec = importlib.util.spec_from_file_location("recovery_rescore", _path.with_name("cordex_temporal_rescore_v2.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("baseline", "spatial_ft"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "evaluation.json").write_text(json.dumps(report()))
    recovered = tmp_path / "spatial_ft/evaluation_recovery.json"
    recovered.write_text(json.dumps(improved()))
    manifest = {"planned_variants": ["baseline", "spatial_ft", "time_only"],
                "variants": {"baseline": {"status": "completed"},
                             "spatial_ft": {"status": "completed", "evaluation": str(recovered)},
                             "time_only": {"status": "running"}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = module.rescore(tmp_path)
    assert result["variants"]["spatial_ft"]["accepted"] is True
    assert result["complete"] is False
    assert result["missing_variants"] == ["time_only"]
    recovered.unlink()
    result = module.rescore(tmp_path)
    assert "spatial_ft" in result["missing_variants"]
    assert "spatial_ft" not in result["variants"]
