"""Lightweight contract tests for the non-notebook CORDEX Phase-2 CLI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "CORDEX_ML"
    / "cordex_refinement_training.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("cordex_refinement_training_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_import_is_lightweight_and_exposes_required_options() -> None:
    module = _load_module()
    parser = module.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert {
        "--config",
        "--device",
        "--epochs",
        "--batch-size",
        "--gradient-accumulation-steps",
        "--learning-rate",
        "--limit-train",
        "--max-train-batches",
        "--limit-valid",
        "--max-val-batches",
        "--checkpoint-dir",
        "--resume",
        "--case-index",
        "--dataset-index",
        "--tiny-overfit",
        "--validation-fraction",
    } <= option_strings


def test_limited_loader_is_reiterable_and_bounded() -> None:
    module = _load_module()
    loader = module._LimitedLoader([{"value": 1}, {"value": 2}, {"value": 3}], 2)
    assert len(loader) == 2
    assert [item["value"] for item in loader] == [1, 2]
    assert [item["value"] for item in loader] == [1, 2]


def test_chronological_holdout_is_disjoint_and_complete() -> None:
    module = _load_module()
    train, validation = module._chronological_split_indices(20, 0.2)
    assert list(train) == list(range(16))
    assert list(validation) == list(range(16, 20))
    assert set(train).isdisjoint(validation)


def test_diagnostic_sample_identity_and_physical_skill_are_explicit() -> None:
    module = _load_module()
    batch = {
        "y": torch.zeros(2, 1, 2, 2),
        "__sample_dataset_index": torch.tensor([17, 18]),
        "__sample_timestamp": ["2001-01-02T00:00:00", "2001-01-03T00:00:00"],
        "__sample_predictor_path": ["predictor-a.nc", "predictor-b.nc"],
    }
    identity = module._diagnostic_sample_identity(batch, batch_size=2)
    assert identity == {
        "dataset_index": 17,
        "timestamp": "2001-01-02T00:00:00",
        "predictor_path": "predictor-a.nc",
        "identity_available": True,
        "loader_batch_index": 0,
    }

    truth = torch.tensor([0.0, 1.0, 2.0])
    phase1 = torch.tensor([1.0, 2.0, 3.0])
    refined = torch.tensor([0.0, 1.0, 2.0])
    phase1_skill = module._paired_skill_metrics(phase1, truth)
    refined_skill = module._paired_skill_metrics(refined, truth)
    delta = module._skill_delta(refined_skill, phase1_skill)
    assert phase1_skill["bias"] == pytest.approx(1.0)
    assert phase1_skill["rmse"] == pytest.approx(1.0)
    assert refined_skill["rmse"] == pytest.approx(0.0)
    assert delta["rmse"] == pytest.approx(-1.0)
    assert delta["absolute_bias"] == pytest.approx(-1.0)


def test_checkpoint_run_identity_rejects_stale_best() -> None:
    module = _load_module()
    reference = {
        "checkpoint_schema_version": 2,
        "checkpoint_kind": "refinement",
        "refinement_type": "diffusion_transformer",
        "refinement_contract_fingerprint": "a" * 64,
        "residual_normalizer_state_fingerprint": "b" * 64,
        "phase1_fingerprint": "c" * 64,
        "case_name": "case",
        "resolved_config": {"data": {"output_vars": ["pr"]}},
        "epoch": 5,
        "global_step": 50,
    }
    best = dict(reference, epoch=3, global_step=30)
    assert module._checkpoint_belongs_to_run(best, reference)
    assert not module._checkpoint_belongs_to_run(
        dict(best, phase1_fingerprint="d" * 64), reference
    )
    assert not module._checkpoint_belongs_to_run(
        dict(best, global_step=60), reference
    )


def test_diagnostic_report_keeps_immutable_checkpoint_companion(tmp_path) -> None:
    module = _load_module()
    convenience = tmp_path / "diagnostics.json"
    digest = "a" * 64
    immutable = module._write_diagnostic_report(
        {"value": 1},
        convenience_path=convenience,
        checkpoint_sha256=digest,
    )
    assert immutable.name == f"diagnostics-{digest}.json"
    assert json.loads(immutable.read_text(encoding="utf-8")) == {"value": 1}

    same_path = module._write_diagnostic_report(
        {"value": 2},
        convenience_path=convenience,
        checkpoint_sha256=digest,
    )
    assert same_path == immutable
    assert json.loads(immutable.read_text(encoding="utf-8")) == {"value": 1}
    assert json.loads(convenience.read_text(encoding="utf-8")) == {"value": 2}
