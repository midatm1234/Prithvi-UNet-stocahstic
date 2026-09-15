"""Report-only handoff tests; fixtures are not climate-performance evidence."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from examples.CORDEX_ML.utils.scientific_acceptance import write_report

RUN_ROOT = Path(__file__).resolve().parents[1] / "artifacts/refinement_validation/scientific_acceptance_20260909T230807Z"
SPEC = importlib.util.spec_from_file_location("execution_report_renderer", RUN_ROOT / "refresh_execution_report.py")
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")
    return path


@pytest.fixture
def root(tmp_path):
    prerequisite = renderer.read(RUN_ROOT / "acceptance_prerequisite/report/scientific_acceptance_results.json")
    put(tmp_path / "acceptance_prerequisite/report/scientific_acceptance_results.json", prerequisite)
    return tmp_path


def complete_family(root, group):
    # Use the actual canonical incomplete-result schema, populating controlled
    # presentation values only; this fixture never enters scientific selection.
    report = copy.deepcopy(renderer.read(root / "acceptance_prerequisite/report/scientific_acceptance_results.json"))
    report["rows"] = []
    report["per_training_seed_metrics"] = {}
    report["status"] = "FAIL" if group == "control" else "INCONCLUSIVE"
    report["bootstrap"]["confidence"] = 0.95
    for head, variables in report["combinations"].items():
        for variable, combination in variables.items():
            combination["status"] = report["status"]
            report["per_training_seed_metrics"][f"{head}/{variable}/pooled"] = {
                str(seed): {"phase1": {"full": {"daily_rmse": 2.0}},
                            "refined": {"full": {"daily_rmse": 1.9}}}
                for seed in (101, 202, 303)
            }
            for metric in ("daily_rmse", "mean_absolute_climatological_bias", "daily_mae", "crps"):
                report["rows"].append({
                    "head": head, "variable": variable, "stratum": "pooled", "region": "full",
                    "metric": metric, "reference_product": "phase1",
                    "key": f"{head}/{variable}/pooled/full/{metric}/phase1",
                    "reference": 2.0, "candidate": 1.9, "change": -0.05,
                    "lower": -0.06, "upper": -0.01, "limit": -0.02,
                    "status": report["status"], "point_pass": True,
                    "units": "mm/day" if variable == "pr" else "K",
                    "signed_bias_reference": -0.2, "signed_bias_candidate": 0.1,
                })
    directory = root / "results" / group
    write_report(report, directory / "acceptance_report")
    put(directory / "independent_training_seed_results.json",
        {"registered_training_seeds": ["101", "202", "303"],
         "per_seed_metrics": report["per_training_seed_metrics"]})
    put(directory / "evidence.json", {"contract_sha256": report["contract_sha256"]})
    artifact = directory / "acceptance_report/scientific_acceptance_results.json"
    return {"artifact": str(artifact), "artifact_sha256": renderer.digest(artifact),
            "bundle_files": {str(p.resolve()): renderer.digest(p) for p in directory.rglob("*") if p.is_file()},
            "status": report["status"]}


def test_incomplete_render_preserves_scope_and_uses_stage_and_latest_progress(root):
    output = root / "job"
    put(root / "workflow/development_queue_artifacts.json",
        {"jobs": [{"id": "existing_full_recipe/diffusion_unet", "output": str(output),
                   "head": "diffusion_unet", "stage": "screen"}]})
    put(root / "workflow/development_queue_artifacts_execution.json",
        {"status": "RUNNING", "jobs": {"existing_full_recipe/diffusion_unet": {"status": "RUNNING"}}})
    put(output / "execution_status.json", {"epoch": 5, "updates": 240})
    put(output / "learning_curve.json", [{"epoch": 10, "updates": 480}])
    put(root / "cache/phase1.manifest.json", {"complete": False})
    put(root / "workflow/finish_execution.json", {"status": "WAITING_FOR_EXISTING_DEVELOPMENT"})
    renderer.main(["--run-root", str(root)])
    summary = renderer.read(root / "scientific_acceptance_results.json")
    text = (root / "SCIENTIFIC_ACCEPTANCE.md").read_text(encoding="utf8")
    assert summary["status"] == "INCONCLUSIVE"
    assert summary["execution_snapshot"]["phase1_cache_complete"] is False
    assert summary["execution_snapshot"]["jobs"][0]["epoch"] == 10
    assert "| existing_full_recipe/diffusion_unet | screen | RUNNING | 10 | 480 |" in text
    assert "no authenticated completed family merge" in text
    assert "1981-2000 regeneration is UNEXECUTED" in text
    assert "paired selected-minus-control" in text
    assert "DEVELOPMENT_FINDINGS.md" not in text


def test_completed_both_families_all_eight_rows_seed_handoff_and_actual_maps(root):
    selected, control = complete_family(root, "selected"), complete_family(root, "control")
    limits = put(root / "matched/shared_plot_limits.json", {"pr": [0, 1]})
    maps = put(root / "matched/comparison.json", {"shared_plot_limits": str(limits)})
    put(root / "workflow/finish_execution.json", {
        "status": "SCIENTIFIC_EVALUATION_COMPLETED",
        "scientific_results": {"selected": selected, "control": control},
        "matched_selected_control_maps": {"artifact": str(maps), "artifact_sha256": renderer.digest(maps)},
    })
    (root / "DEVELOPMENT_FINDINGS.md").write_text("Registered subset only.", encoding="utf8")
    renderer.main(["--run-root", str(root)])
    summary = renderer.read(root / "scientific_acceptance_results.json")
    text = (root / "SCIENTIFIC_ACCEPTANCE.md").read_text(encoding="utf8")
    assert set(summary["scientific_family_results"]) == {"selected", "control"}
    assert summary["scientific_family_results"]["control"]["status"] == "FAIL"
    assert text.count("101, 202, 303") == 16
    assert text.count("| daily_rmse (") == 16
    assert text.count("[-6.000%, -1.000%]") == 64
    assert "metrics are then averaged equally" in text
    assert "independent_training_seed_results.json" in text
    assert "results/control/acceptance_report/scientific_acceptance_metrics.csv" in text
    assert "matched/comparison.json" in text
    assert "DEVELOPMENT_FINDINGS.md" in text
    assert summary["production_promotion"] is False


def test_completed_merge_visible_while_final_plotting_pending(root):
    selected = complete_family(root, "selected")
    put(root / "workflow/finish_execution.json",
        {"status": "PAUSED_RESOURCE", "operations": {"merge|selected": {
            "status": "COMPLETED", "receipt": selected}}})
    renderer.main(["--run-root", str(root)])
    summary = renderer.read(root / "scientific_acceptance_results.json")
    assert summary["execution_snapshot"]["acceptance_evidence_available"] is True
    assert summary["execution_snapshot"]["finish_controller"]["status"] == "PAUSED_RESOURCE"


@pytest.mark.parametrize("changed_product", ["report", "per_seed", "metrics"])
def test_changed_completion_handoff_is_rejected(root, changed_product):
    selected = complete_family(root, "selected")
    report = Path(selected["artifact"])
    products = {"report": report, "per_seed": report.parent.parent / "independent_training_seed_results.json",
                "metrics": report.with_name("scientific_acceptance_metrics.csv")}
    products[changed_product].write_text("changed", encoding="utf8")
    prerequisite = renderer.read(root / "acceptance_prerequisite/report/scientific_acceptance_results.json")
    with pytest.raises(ValueError, match="differs|missing or changed"):
        renderer.completed_families(root, {"scientific_results": {"selected": selected}}, prerequisite)


def test_missing_head_cannot_disappear_from_final_table(root):
    selected = complete_family(root, "selected")
    path = Path(selected["artifact"])
    altered = renderer.read(path)
    altered["combinations"].pop(next(iter(altered["combinations"])))
    put(path, altered)
    selected["artifact_sha256"] = renderer.digest(path)
    prerequisite = renderer.read(root / "acceptance_prerequisite/report/scientific_acceptance_results.json")
    with pytest.raises(ValueError, match="omits a registered"):
        renderer.completed_families(root, {"scientific_results": {"selected": selected}}, prerequisite)


def test_supplemental_attribution_receipt_reports_numeric_contrast_without_acceptance(root):
    directory = root / "actual_contrast"
    directory.mkdir()
    for filename in ("contrasts.csv", "per_seed_metrics.csv", "bootstrap_changes.npz"):
        (directory / filename).write_bytes(b"controlled presentation fixture")
    artifact = put(directory / "comparison.json", {
        "status": "COMPLETE", "acceptance_decisions_unchanged": True,
        "acceptance_families_combined": False, "production_promotion": False,
        "rows": [{"head": "diffusion_unet", "variable": "pr", "stratum": "pooled", "region": "full",
            "metric": "daily_rmse", "units": "mm/day", "reference": 2., "candidate": 1.9,
            "physical_difference": -.1, "relative_ci_lower": -.06, "relative_ci_upper": -.01,
            "diagnostic_contrast": "LOWER_ERROR"}]})
    receipt = {"artifact": str(artifact), "artifact_sha256": renderer.digest(artifact),
        "bundle_files": {str(p.resolve()): renderer.digest(p) for p in directory.iterdir()}}
    text = "\n".join(renderer.attribution_report_lines(root, {"fresh_control_attribution": receipt}))
    assert "Computation status: **COMPLETE**; this is not an acceptance status." in text
    assert "| 2 | 1.9 | -0.1 | [-6.000%, -1.000%] | LOWER_ERROR |" in text
    assert "proof of a causal effect" in text
    assert "per_seed_metrics.csv" in text
    (directory / "contrasts.csv").write_text("changed", encoding="utf8")
    with pytest.raises(ValueError, match="handoff is missing or changed"):
        renderer.attribution_report_lines(root, {"fresh_control_attribution": receipt})


def test_pending_supplemental_registration_never_becomes_a_result(root):
    put(root / "fresh_control_contrast_registration.json", {"prepared_fixture": True})
    text = "\n".join(renderer.attribution_report_lines(root, {}))
    assert "UNEXECUTED / INCONCLUSIVE" in text
    assert "fresh_control_contrast_registration.json" in text
    assert "Computation status: **COMPLETE**" not in text
