from __future__ import annotations

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1] / "examples" / "CORDEX_ML"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import cordex_training
from cordex_training import (
    _contiguous_holdout_indices,
    build_optimizer_scheduler,
    create_finetune_model,
    get_dataloaders,
    _validate_train_validation_source_paths,
    _validate_scalar_holdout_metadata,
)


def test_contiguous_holdout_is_disjoint_and_covers_every_sample_once():
    train = _contiguous_holdout_indices(14_600, 0.10, "train")
    validation = _contiguous_holdout_indices(14_600, 0.10, "validation")

    assert (train.start, train.stop) == (0, 13_140)
    assert (validation.start, validation.stop) == (13_140, 14_600)
    assert train.stop == validation.start
    assert len(train) + len(validation) == 14_600
    assert set(train).isdisjoint(validation)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_contiguous_holdout_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="validation_holdout_fraction"):
        _contiguous_holdout_indices(10, fraction, "train")


def test_contiguous_holdout_keeps_both_sides_nonempty_for_small_dataset():
    train = _contiguous_holdout_indices(2, 0.01, "train")
    validation = _contiguous_holdout_indices(2, 0.01, "validation")
    assert list(train) == [0]
    assert list(validation) == [1]


def test_contiguous_holdout_rejects_unknown_role():
    with pytest.raises(ValueError, match="Unknown holdout role"):
        _contiguous_holdout_indices(10, 0.2, "test")


def test_source_path_check_rejects_single_shared_path_in_different_lists():
    with pytest.raises(ValueError, match="partially overlap"):
        _validate_train_validation_source_paths(
            ["train_predictor.nc", "shared_predictor.nc"],
            ["train_target_a.nc", "train_target_b.nc"],
            ["validation_predictor.nc", "shared_predictor.nc"],
            ["validation_target_a.nc", "validation_target_b.nc"],
        )


def test_source_path_check_accepts_fully_disjoint_lists():
    sources_identical = _validate_train_validation_source_paths(
        ["train_predictor_a.nc", "train_predictor_b.nc"],
        ["train_target_a.nc", "train_target_b.nc"],
        ["validation_predictor_a.nc", "validation_predictor_b.nc"],
        ["validation_target_a.nc", "validation_target_b.nc"],
    )

    assert sources_identical is False


@pytest.mark.parametrize("validation_sources", ["identical", "empty"])
def test_disabled_validation_builds_full_training_loader_once_and_returns_none(
    monkeypatch, capsys, validation_sources
):
    training_predictors = ["train_predictor.nc"]
    training_targets = ["train_target.nc"]
    if validation_sources == "identical":
        validation_predictors = list(training_predictors)
        validation_targets = list(training_targets)
    else:
        validation_predictors = []
        validation_targets = []
    config = SimpleNamespace(
        validation_enabled=False,
        data=SimpleNamespace(
            target_size_lat=128,
            target_size_lon=128,
            training_predictor_paths=training_predictors,
            training_target_paths=training_targets,
            validation_predictor_paths=validation_predictors,
            validation_target_paths=validation_targets,
            validation_holdout_fraction=0.10,
            validation_holdout_strategy="contiguous_tail",
        ),
    )
    full_training_loader = object()
    calls = []

    def fake_build_dataloader(config_arg, predictor_paths, target_paths, **kwargs):
        calls.append((config_arg, predictor_paths, target_paths, kwargs))
        return full_training_loader

    monkeypatch.setattr(cordex_training, "build_dataloader", fake_build_dataloader)

    train_loader, validation_loader = get_dataloaders(config, use_gpu=False)

    assert train_loader is full_training_loader
    assert validation_loader is None
    assert len(calls) == 1
    assert calls[0][0] is config
    assert calls[0][1] is training_predictors
    assert calls[0][2] is training_targets
    assert calls[0][3]["holdout_fraction"] is None
    assert calls[0][3]["holdout_role"] is None
    assert "using 100%" in capsys.readouterr().out


def test_joint_residual_policy_keeps_unet_in_optimizer(monkeypatch):
    class JointResidualToy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.residual_diffusion_enabled = True
            self.unet_output_head = torch.nn.Linear(2, 2)
            self.diffusion_head = torch.nn.Linear(2, 2)
            self.fixed_scaler = torch.nn.Parameter(
                torch.ones(1), requires_grad=False
            )

    model = JointResidualToy()
    config = SimpleNamespace(
        data=SimpleNamespace(
            input_static_surface_vars=[],
            output_vars=["pr", "tasmax"],
        ),
        device_target="cpu",
        path_model_weights="initializer.pt",
        training={"freeze_deterministic_baseline": False},
        learning_rate=1.0e-4,
        gradient_accumulation_steps=1,
        num_epochs=1,
        limit_steps_train=1,
        min_lr=1.0e-6,
    )
    monkeypatch.setattr(
        cordex_training, "build_predictand_specs", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        cordex_training, "get_finetune_model_UNET", lambda config_arg: model
    )
    monkeypatch.setattr(
        cordex_training,
        "load_pretrained_weights",
        lambda model_arg, path_arg: (0, 0),
    )

    created = create_finetune_model(config, verbose=False)
    optimizer, _, _ = build_optimizer_scheduler(
        config, created, train_loader_length=1, use_gpu=False
    )
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert created.unet_output_head.weight.requires_grad
    assert id(created.unet_output_head.weight) in optimized_ids
    assert id(created.diffusion_head.weight) in optimized_ids
    assert id(created.fixed_scaler) not in optimized_ids

    before = created.unet_output_head.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    created.unet_output_head(torch.ones(2, 2)).square().mean().backward()
    optimizer.step()
    assert not torch.equal(created.unet_output_head.weight.detach(), before)


def _scalar_config(tmp_path: Path) -> SimpleNamespace:
    scalar_dir = tmp_path / "scalars"
    scalar_dir.mkdir()
    target_sigma = scalar_dir / "targets_std.npy"
    target_sigma.write_bytes(b"test scalar")
    return SimpleNamespace(
        model=SimpleNamespace(target_sigma=str(target_sigma)),
        data=SimpleNamespace(),
    )


def test_scalar_metadata_rejects_old_full_dataset_statistics(tmp_path):
    config = _scalar_config(tmp_path)
    metadata_path = Path(config.model.target_sigma).parent / "metadata.json"
    metadata_path.write_text(json.dumps({"num_samples": 14_600}), encoding="utf-8")

    with pytest.raises(ValueError, match="num_samples=14600"):
        _validate_scalar_holdout_metadata(
            config,
            source_sample_count=14_600,
            training_sample_count=13_140,
            holdout_fraction=0.10,
            holdout_strategy="contiguous_tail",
        )


def test_scalar_metadata_accepts_exact_training_partition_audit(tmp_path):
    config = _scalar_config(tmp_path)
    metadata = {
        "num_samples": 13_140,
        "source_num_samples": 14_600,
        "used_num_samples": 13_140,
        "sample_selection": {
            "policy": "leading_training_partition_excluding_contiguous_validation_tail",
            "source_sample_count": 14_600,
            "used_sample_count": 13_140,
            "used_index_start": 0,
            "used_index_stop_exclusive": 13_140,
            "excluded_index_start": 13_140,
            "excluded_index_stop_exclusive": 14_600,
            "config_training_validation_sources_identical": True,
            "validation_holdout_fraction": 0.10,
            "validation_holdout_strategy": "contiguous_tail",
        },
    }
    metadata_path = Path(config.model.target_sigma).parent / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    loaded = _validate_scalar_holdout_metadata(
        config,
        source_sample_count=14_600,
        training_sample_count=13_140,
        holdout_fraction=0.10,
        holdout_strategy="contiguous_tail",
    )
    assert loaded == metadata
