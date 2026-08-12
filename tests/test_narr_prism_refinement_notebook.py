"""Checks for the generic NARR refinement notebook and smoke harness."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "examples"
    / "NARR_PRISM"
    / "notebooks"
    / "NARR_PRISM_refinement.ipynb"
)
SMOKE = ROOT / "examples" / "NARR_PRISM" / "refinement_smoke.py"
CONFIGS = (
    "NARR_PRISM_diffusion_unet.yaml",
    "NARR_PRISM_diffusion_transformer.yaml",
    "NARR_PRISM_flow_matching_unet.yaml",
    "NARR_PRISM_flow_matching_transformer.yaml",
)


def test_notebook_has_one_parameter_cell_and_complete_workflow_sections():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    parameter_cells = [
        cell
        for cell in notebook.cells
        if "parameters" in cell.metadata.get("tags", [])
    ]
    assert len(parameter_cells) == 1
    parameters = parameter_cells[0].source
    for name in (
        "REFINEMENT_CONFIG",
        "PHASE1_CHECKPOINT",
        "REFINEMENT_CHECKPOINT",
        "RESUME_CHECKPOINT",
        "ENSEMBLE_SIZE",
        "SMOKE_TEST",
        "PREDICTION_SPLIT",
        "OUTPUT_DIR",
    ):
        assert name in parameters
    text = "\n".join(cell.source for cell in notebook.cells)
    for required in (
        "Configuration and checkpoint compatibility",
        "Residual target",
        "Phase 2 training",
        "Resume",
        "Stochastic ensemble",
        "Evaluation",
        "Spatial maps",
        "extreme",
        "Saving",
    ):
        assert required.lower() in text.lower()
    assert "pip install" not in text
    assert "NARR_PRISM_PREDICTION_SPLIT" in parameters
    assert "{'validation', 'inference'}" in parameters
    assert text.count("'--split', PREDICTION_SPLIT") == 2
    assert "DAILY_OUTPUT_ROOT / 'validation'" in text
    assert "'scientific_validation': not SMOKE_TEST" not in text
    assert "'scientific_validation': scientific_validation_completed" in text
    assert "scientific_validation_report.is_file()" in text


@pytest.mark.parametrize("config_name", CONFIGS)
def test_shared_smoke_harness_executes_each_real_yaml(config_name, tmp_path):
    config = ROOT / "examples" / "NARR_PRISM" / config_name
    result = subprocess.run(
        [
            sys.executable,
            str(SMOKE),
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path),
            "--ensemble-size",
            "2",
            "--seed",
            "17",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    metrics_files = list(tmp_path.glob("*_smoke_metrics.json"))
    assert len(metrics_files) == 1
    metrics = json.loads(metrics_files[0].read_text())
    assert metrics["mode"] == "synthetic_smoke"
    assert metrics["scientific_validation"] is False
    assert metrics["finite_loss"] is True
    assert metrics["finite_valid_output"] is True
    assert metrics["phase1_frozen"] is True
    assert metrics["gradient_norm"] > 0.0
    assert list(tmp_path.glob("*_smoke.nc"))
    assert list(tmp_path.glob("*_smoke_diagnostics.png"))


def test_notebook_executes_headlessly_in_explicit_smoke_mode(tmp_path):
    executed = tmp_path / "NARR_PRISM_refinement_executed.ipynb"
    output_dir = tmp_path / "notebook_artifacts"
    env = os.environ.copy()
    env.update(
        {
            "NARR_PRISM_SMOKE_TEST": "1",
            "NARR_PRISM_REFINEMENT_DEVICE": "cpu",
            "NARR_PRISM_ENSEMBLE_SIZE": "2",
            "NARR_PRISM_REFINEMENT_SEED": "103",
            "NARR_PRISM_PREDICTION_SPLIT": "validation",
            "NARR_PRISM_REFINEMENT_CONFIG": str(
                ROOT / "examples" / "NARR_PRISM" / CONFIGS[0]
            ),
            "NARR_PRISM_REFINEMENT_OUTPUT_DIR": str(output_dir),
            "NARR_PRISM_RUN_TRAINING": "0",
            "NARR_PRISM_RUN_INFERENCE": "0",
            "NARR_PRISM_RUN_EVALUATION": "0",
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(NOTEBOOK),
            "--output",
            str(executed),
            "--ExecutePreprocessor.timeout=180",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert executed.is_file()
    notebook = nbformat.read(executed, as_version=4)
    for cell in notebook.cells:
        assert not any(
            output.output_type == "error" for output in cell.get("outputs", [])
        )

    rendered_output = "\n".join(
        str(output.get("text", ""))
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if output.output_type == "stream"
    )
    assert "--split validation" in rendered_output
    assert str(
        output_dir
        / "daily_predictions"
        / "validation"
        / "narr_prism_California"
    ) in rendered_output

    final_stream = "".join(
        str(output.get("text", ""))
        for output in notebook.cells[-1].get("outputs", [])
        if output.output_type == "stream"
    )
    final_summary = json.loads(final_stream)
    assert final_summary["prediction_split"] == "validation"
    assert final_summary["scientific_validation"] is False
    assert final_summary["scientific_validation_report"] is None

    metrics = json.loads(
        (output_dir / "diffusion_unet_smoke_metrics.json").read_text()
    )
    notebook_metrics = json.loads(
        (output_dir / "notebook_metrics.json").read_text()
    )
    assert metrics["finite_loss"] is True
    assert metrics["gradient_norm"] > 0.0
    assert metrics["scientific_validation"] is False
    assert notebook_metrics["mode"] == "synthetic_smoke"
    assert notebook_metrics["scientific_validation"] is False
    assert (output_dir / "diffusion_unet_smoke.nc").is_file()
    assert (output_dir / "diffusion_unet_smoke_diagnostics.png").is_file()
    for variable in ("ppt", "tmax", "tmin"):
        assert (output_dir / f"{variable}_phase1_vs_refined.png").is_file()
