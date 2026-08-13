"""Checks for the generic NARR refinement notebook and smoke harness."""

from __future__ import annotations

import ast
import codecs
import importlib.util
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
    / "narr_prism_refinement.ipynb"
)
LEGACY_CASE_NOTEBOOK = NOTEBOOK.with_name("NARR_PRISM_refinement.ipynb")
NARR_PRISM_DIR = NOTEBOOK.parent.parent
ARTIFACT_HELPER = NARR_PRISM_DIR / "narr_prism_artifacts.py"
SMOKE = ROOT / "examples" / "NARR_PRISM" / "refinement_smoke.py"
CONFIGS = (
    "NARR_PRISM_diffusion_unet.yaml",
    "NARR_PRISM_diffusion_transformer.yaml",
    "NARR_PRISM_flow_matching_unet.yaml",
    "NARR_PRISM_flow_matching_transformer.yaml",
)


def _parameter_cell_source() -> str:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    parameter_cells = [
        cell
        for cell in notebook.cells
        if "parameters" in cell.metadata.get("tags", [])
    ]
    assert len(parameter_cells) == 1
    return parameter_cells[0].source


def _cell_source_containing(marker: str) -> str:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    matching = [
        cell.source for cell in notebook.cells if marker in cell.source
    ]
    assert len(matching) == 1, marker
    return matching[0]


def _notebook_function(name: str, globals_: dict | None = None):
    source = _cell_source_containing(f"def {name}(")
    tree = ast.parse(source)
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    assert len(definitions) == 1
    namespace = dict(globals_ or {})
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(module, str(NOTEBOOK), "exec"), namespace)
    return namespace[name]


def _execute_parameter_cell(monkeypatch, repo_root: Path) -> dict:
    import random

    import numpy as np
    import torch

    namespace = {
        "Path": Path,
        "REPO_ROOT": repo_root,
        "json": json,
        "np": np,
        "os": os,
        "random": random,
        "torch": torch,
    }
    exec(compile(_parameter_cell_source(), str(NOTEBOOK), "exec"), namespace)
    return namespace


def test_lowercase_notebook_is_the_single_canonical_filename():
    assert NOTEBOOK.is_file()
    assert not LEGACY_CASE_NOTEBOOK.exists()


def test_local_artifact_portal_is_ignored():
    ignore_lines = {
        line.strip()
        for line in (NARR_PRISM_DIR / ".gitignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "/artifacts" in ignore_lines


def _artifact_helper_module():
    spec = importlib.util.spec_from_file_location(
        "narr_prism_artifacts_under_test", ARTIFACT_HELPER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("name", "expected_target"),
    [
        ("experiments", "artifacts/experiments"),
        ("preprocessed", "artifacts/preprocessed"),
        ("scalars_with_H", "artifacts/scalars_with_H"),
    ],
)
def test_tracked_artifact_links_are_relative_and_route_through_portal(
    name, expected_target
):
    link = NARR_PRISM_DIR / name
    assert link.is_symlink()
    target = Path(os.readlink(link))
    assert target == Path(expected_target)
    assert not target.is_absolute()
    assert os.path.abspath(link.parent / target) != os.path.abspath(link)


def test_artifact_link_validator_rejects_repository_self_reference(tmp_path):
    module = _artifact_helper_module()
    for name, target in module.MANAGED_LINKS.items():
        (tmp_path / name).symlink_to(target)
    (tmp_path / "artifacts").symlink_to(tmp_path, target_is_directory=True)

    with pytest.raises(
        module.ArtifactLinkError, match="source directory itself"
    ):
        module.validate_artifact_links(tmp_path, require_targets=False)


def test_artifact_link_validator_rejects_source_ancestor(tmp_path):
    module = _artifact_helper_module()
    source = tmp_path / "checkout" / "examples" / "NARR_PRISM"
    source.mkdir(parents=True)
    for name, target in module.MANAGED_LINKS.items():
        (source / name).symlink_to(target)
    (source / "artifacts").symlink_to(tmp_path, target_is_directory=True)

    with pytest.raises(module.ArtifactLinkError, match="one of its ancestors"):
        module.validate_artifact_links(source, require_targets=False)


def test_artifact_link_validator_resolves_external_portal(tmp_path):
    module = _artifact_helper_module()
    source = tmp_path / "source"
    external = tmp_path / "external"
    source.mkdir()
    external.mkdir()
    for name, target in module.MANAGED_LINKS.items():
        (source / name).symlink_to(target)
        (external / name).mkdir()
    (source / "artifacts").symlink_to(external, target_is_directory=True)

    status = module.validate_artifact_links(source)
    assert status["artifact_root"] == str(external.resolve())
    assert all(Path(path).is_dir() for path in status["links"].values())


def test_default_output_avoids_experiments_symlink(monkeypatch, tmp_path):
    monkeypatch.delenv("NARR_PRISM_REFINEMENT_OUTPUT_DIR", raising=False)
    namespace = _execute_parameter_cell(monkeypatch, tmp_path)
    expected = (
        tmp_path
        / "examples"
        / "NARR_PRISM"
        / "refinement_outputs"
        / "NARR_PRISM_diffusion_transformer"
    )
    assert namespace["OUTPUT_PATH"] == expected
    assert expected.is_dir()
    assert "experiments" not in expected.relative_to(tmp_path).parts


def test_production_defaults_train_diffusion_transformer_residual_head(
    monkeypatch, tmp_path
):
    for name in (
        "NARR_PRISM_REFINEMENT_CONFIG",
        "NARR_PRISM_SMOKE_TEST",
        "NARR_PRISM_RUN_TRAINING",
        "NARR_PRISM_RUN_INFERENCE",
        "NARR_PRISM_RUN_EVALUATION",
        "NARR_PRISM_REFINEMENT_OUTPUT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)

    namespace = _execute_parameter_cell(monkeypatch, tmp_path)
    assert namespace["SMOKE_TEST"] is False
    assert namespace["RUN_TRAINING"] is True
    assert namespace["RUN_INFERENCE"] is False
    assert namespace["RUN_EVALUATION"] is False
    assert namespace["REFINEMENT_CONFIG"].endswith(
        "NARR_PRISM_diffusion_transformer.yaml"
    )

    compatibility = _cell_source_containing(
        "This notebook is configured for residual-refinement-head-only training"
    )
    assert "refinement.train_on_residual" in compatibility
    assert "refinement.freeze_phase1" in compatibility
    assert "not refinement.joint_finetuning" in compatibility
    assert "not refinement.trainable_phase1_patterns" in compatibility


def test_explicit_smoke_mode_disables_default_production_training(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("NARR_PRISM_SMOKE_TEST", "1")
    monkeypatch.delenv("NARR_PRISM_RUN_TRAINING", raising=False)
    monkeypatch.setenv(
        "NARR_PRISM_REFINEMENT_OUTPUT_DIR", str(tmp_path / "smoke-output")
    )
    namespace = _execute_parameter_cell(monkeypatch, tmp_path)
    assert namespace["SMOKE_TEST"] is True
    assert namespace["RUN_TRAINING"] is False


def test_explicit_training_override_is_honored(monkeypatch, tmp_path):
    monkeypatch.setenv("NARR_PRISM_SMOKE_TEST", "0")
    monkeypatch.setenv("NARR_PRISM_RUN_TRAINING", "0")
    monkeypatch.setenv(
        "NARR_PRISM_REFINEMENT_OUTPUT_DIR", str(tmp_path / "inspect-only")
    )
    namespace = _execute_parameter_cell(monkeypatch, tmp_path)
    assert namespace["SMOKE_TEST"] is False
    assert namespace["RUN_TRAINING"] is False


def test_explicit_self_referential_output_has_actionable_error(
    monkeypatch, tmp_path
):
    loop = tmp_path / "output-loop"
    loop.symlink_to(loop)
    monkeypatch.setenv("NARR_PRISM_REFINEMENT_OUTPUT_DIR", str(loop / "run"))
    with pytest.raises(RuntimeError) as error:
        _execute_parameter_cell(monkeypatch, tmp_path)
    message = str(error.value)
    assert "NARR_PRISM_REFINEMENT_OUTPUT_DIR" in message
    assert "must not point to itself" in message


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
        "REFINEMENT_CHECKPOINT_DIR",
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
    assert "examples/NARR_PRISM/refinement_outputs" in parameters
    assert "/ Path(REFINEMENT_CONFIG).stem" in parameters
    assert "NARR_PRISM_ENSEMBLE_SIZE', '10'" in parameters
    assert (
        "RUN_TRAINING = _env_flag('NARR_PRISM_RUN_TRAINING', not SMOKE_TEST)"
        in parameters
    )
    assert (
        "examples/NARR_PRISM/experiments/refinement_notebook" not in parameters
    )
    assert "'--checkpoint-dir', str(PHASE2_CHECKPOINT_DIR)" in text
    assert "REFINEMENT_CHECKPOINT_DIR or raw_config['checkpoint_dir']" in text
    assert "NARR_PRISM_REFINEMENT_CHECKPOINT_DIR" in text
    assert text.count("'--device', DEVICE") == 2
    assert "NARR_PRISM_PHASE1_CHECKPOINT" in text
    assert "must not point to itself" in text
    assert "validate_artifact_links" in text
    assert "artifact portal is valid" in text
    assert "cannot be reconstructed from notebook logs" not in text
    assert text.count("'--split', PREDICTION_SPLIT") == 2
    assert "DAILY_OUTPUT_ROOT / 'validation'" in text
    assert "'scientific_validation': not SMOKE_TEST" not in text
    assert "'scientific_validation': scientific_validation_completed" in text
    assert "scientific_validation_report.is_file()" in text
    assert "one live combined train-and-validation `tqdm` bar" in text
    assert "run_streaming_command(train_command, cwd=REPO_ROOT)" in text
    assert "subprocess.run(train_command" not in text


def test_streaming_command_forwards_carriage_returns_and_failures(
    tmp_path, capsys
):
    runner = _notebook_function(
        "run_streaming_command",
        {
            "codecs": codecs,
            "os": os,
            "subprocess": subprocess,
            "sys": sys,
        },
    )
    completed = runner(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.stderr.write('epoch 1: 1/2\\r'); sys.stderr.flush(); "
                "sys.stderr.write('epoch 1: 2/2\\n'); sys.stderr.flush()"
            ),
        ],
        cwd=tmp_path,
    )
    rendered = capsys.readouterr().out
    assert completed.returncode == 0
    assert "epoch 1: 1/2\r" in rendered
    assert "epoch 1: 2/2\n" in rendered

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        runner(
            [sys.executable, "-c", "raise SystemExit(7)"],
            cwd=tmp_path,
        )
    assert exc_info.value.returncode == 7


def test_production_subprocesses_release_eager_gpu_state_after_diagnostics():
    inspection_cell = _cell_source_containing(
        "def release_production_inspection_state()"
    )
    assert "production_inspection_diagnostics = {" in inspection_cell
    for assignment in (
        "model = None",
        "train_loader = None",
        "validation_loader = None",
        "baseline_batch = None",
        "baseline_output = None",
        "baseline_normalized = None",
        "phase1_features = None",
        "baseline_residual = None",
        "baseline_valid = None",
        "valid_residual = None",
    ):
        assert assignment in inspection_cell
    assert "gc.collect()" in inspection_cell
    assert "torch.cuda.empty_cache()" in inspection_cell

    command_invocations = (
        "run_streaming_command(train_command, cwd=REPO_ROOT)",
        (
            "subprocess.run(infer_command, cwd=REPO_ROOT, check=True)"
        ),
    )
    for invocation in command_invocations:
        command_cell = _cell_source_containing(
            invocation
        )
        release_offset = command_cell.index(
            "release_production_inspection_state()"
        )
        subprocess_offset = command_cell.index(invocation)
        assert release_offset < subprocess_offset


def test_combined_training_and_inference_defers_checkpoint_requirement():
    compatibility_cell = _cell_source_containing(
        "Standalone RUN_INFERENCE requires an existing REFINEMENT_CHECKPOINT"
    )
    assert "RUN_INFERENCE and not RUN_TRAINING" in compatibility_cell
    assert "REFINEMENT_PATH is None" in compatibility_cell

    training_cell = _cell_source_containing("TRAINED_CHECKPOINT_PRIORITY")
    assert (
        "TRAINED_CHECKPOINT_PRIORITY = ('best.ckpt', 'last.ckpt')"
        in training_cell
    )
    train_offset = training_cell.index(
        "run_streaming_command(train_command, cwd=REPO_ROOT)"
    )
    selection_offset = training_cell.index(
        "REFINEMENT_PATH = select_trained_refinement_checkpoint"
    )
    assert train_offset < selection_offset
    assert "if RUN_INFERENCE and REFINEMENT_PATH is None:" not in training_cell
    assert "Training completed; selected Phase-2 checkpoint" in training_cell

    inference_cell = _cell_source_containing(
        "subprocess.run(infer_command, cwd=REPO_ROOT, check=True)"
    )
    assert "'--refinement-checkpoint', str(REFINEMENT_PATH)" in inference_cell
    assert (
        "REFINEMENT_PATH is None or not REFINEMENT_PATH.is_file()"
        in inference_cell
    )


def test_new_training_checkpoint_selection_prefers_best_then_last(tmp_path):
    select_checkpoint = _notebook_function(
        "select_trained_refinement_checkpoint",
        {"TRAINED_CHECKPOINT_PRIORITY": ("best.ckpt", "last.ckpt")},
    )

    last = tmp_path / "last.ckpt"
    last.touch()
    assert select_checkpoint(tmp_path) == last

    best = tmp_path / "best.ckpt"
    best.touch()
    assert select_checkpoint(tmp_path) == best

    best.unlink()
    last.unlink()
    with pytest.raises(
        FileNotFoundError, match="neither best.ckpt nor last.ckpt"
    ):
        select_checkpoint(tmp_path)


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
    assert (
        str(
            output_dir
            / "daily_predictions"
            / "validation"
            / "narr_prism_California"
        )
        in rendered_output
    )

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
