"""Known examples for geographic regions and saved-array metrics."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "examples/CORDEX_ML/cordex_temporal_regional_report.py"
_SPEC = importlib.util.spec_from_file_location("regional_report", _SCRIPT)
REPORT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(REPORT)


def test_geographic_southeast_is_independent_of_array_latitude_order():
    latitude = np.array([20, 10, 0, -10])
    longitude = np.array([100, 110, 120, 130])
    lat, lon = REPORT._meshes(latitude, longitude, (4, 4))
    regions, definitions = REPORT.regions_for_grid(lat, lon, 1)
    expected = np.zeros((4, 4), dtype=bool)
    expected[2:, 2:] = True
    np.testing.assert_array_equal(regions["southeast"], expected)
    assert regions["boundary"].sum() == 12
    assert regions["interior"].sum() == 4
    assert definitions["southeast"]["latitude_bounds_degrees_north"] == [-10.0, 0.0]
    reversed_regions, _ = REPORT.regions_for_grid(lat[::-1], lon[::-1], 1)
    np.testing.assert_array_equal(reversed_regions["southeast"][::-1], expected)


def test_regional_bias_rmse_quantile_and_wet_threshold_known_example():
    target = np.array([[[0.0, 1.0], [2.0, 3.0]]])
    pred = target + 2
    mask = np.ones_like(pred, dtype=bool)
    result = REPORT.region_metrics(pred, target, mask, np.ones((2, 2), bool), wet_threshold=1)
    assert result["bias"] == result["rmse"] == 2
    assert result["q99_error"] == pytest.approx(2)
    assert result["wet_day"]["frequency_pred"] == 1
    assert result["wet_day"]["frequency_target"] == 0.5
    assert result["wet_day"]["comparison"] == ">"
    mask[0, 0, 0] = False
    pred[0, 1, 1] = np.nan
    result = REPORT.region_metrics(pred, target, mask, np.ones((2, 2), bool))
    assert result["n_valid_gridcell_days"] == 2
    assert result["bias"] == result["rmse"] == 2
    empty = REPORT.region_metrics(pred, target, np.zeros_like(mask), np.ones((2, 2), bool))
    assert empty["status"] == "no_valid_samples"


def test_recovered_manifest_selects_new_predictions_and_missing_path_fails(tmp_path):
    import json
    folder = tmp_path / "native_pair"
    folder.mkdir()
    old = folder / "predictions.npz"
    old.write_bytes(b"old archive retained")
    recovered = folder / "predictions_recovery.npz"
    recovered.write_bytes(b"recovered archive")
    config = folder / "resolved.yaml"
    config.write_text("data: {}", encoding="utf-8")
    manifest = {"variants": {"native_pair": {"predictions": str(recovered), "configuration": str(config)}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert REPORT.resolve_artifacts(tmp_path, "native_pair") == (recovered, config)
    recovered.unlink()
    with pytest.raises(FileNotFoundError, match="predictions artifact is missing"):
        REPORT.resolve_artifacts(tmp_path, "native_pair")
    assert old.read_bytes() == b"old archive retained"
