"""Execution-status regressions: a saved label must not imply a live run."""
import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "examples/CORDEX_ML/cordex_temporal_takeover_summary.py"
_SPEC = importlib.util.spec_from_file_location("takeover_summary", _SCRIPT)
REPORT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(REPORT)


def test_stale_running_owner_is_interrupted_even_with_waiting_controller():
    original = {"status": "running", "pid": 12}
    result = REPORT.reconcile_variant("spatial_ft", original, [{"pid": 13}])
    assert result["status"] == "interrupted"
    assert result["reported_manifest_status"] == "running"
    assert "continuation process" in result["execution_note"]
    assert original["status"] == "running"
    assert REPORT.reconcile_variant("spatial_ft", original, [{"pid": 12}])["status"] == "running"


def test_completed_requires_saved_artifacts(tmp_path):
    entry = {"status": "completed"}
    for key in ("checkpoint", "predictions", "evaluation"):
        path = tmp_path / key
        path.write_bytes(b"saved")
        entry[key] = str(path)
    assert REPORT.reconcile_variant("native_pair", entry, [])["status"] == "completed"
    Path(entry["evaluation"]).unlink()
    result = REPORT.reconcile_variant("native_pair", entry, [])
    assert result["status"] == "interrupted"
    assert "evaluation" in result["execution_note"]
