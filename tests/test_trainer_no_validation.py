from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from granitewxc.utils import trainer


def _config(checkpoint_dir: Path, *, num_epochs: int) -> SimpleNamespace:
    return SimpleNamespace(
        validation_enabled=False,
        # Top-level policy must take precedence over the optional nested form.
        training={"validation_enabled": True},
        checkpoint_dir=str(checkpoint_dir),
        path_experiment=str(checkpoint_dir.parent),
        num_epochs=num_epochs,
        limit_steps_train=1,
        limit_steps_valid=1,
        gradient_accumulation_steps=1,
        auto_resume_if_checkpoint_exists=False,
        save_epoch_checkpoints=True,
    )


def _objects():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    return model, optimizer, scheduler


def _metadata(*args, **kwargs):
    del args, kwargs
    return {
        "head_type": "deterministic",
        "decoder_type": None,
        "diffusion_head": False,
        "git_commit": None,
        "git_branch": None,
        "config_fingerprint_sha256": "test-fingerprint",
    }


def test_validation_disabled_skips_validation_and_never_selects_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    train_epochs: list[int] = []
    validation_calls: list[int] = []

    def fake_train_one_epoch(*, epoch, **kwargs):
        del kwargs
        train_epochs.append(epoch)
        return torch.tensor(float(epoch + 1)), {}

    def forbidden_validation(*args, **kwargs):
        del args, kwargs
        validation_calls.append(1)
        raise AssertionError("validate_one_epoch must not run")

    monkeypatch.setattr(trainer, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(trainer, "validate_one_epoch", forbidden_validation)
    monkeypatch.setattr(trainer, "_checkpoint_head_metadata", _metadata)

    config = _config(checkpoint_dir, num_epochs=1)
    model, optimizer, scheduler = _objects()
    train_history, val_history = trainer.train_model(
        config,
        model,
        train_dl=[],
        val_dl=None,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        local_rank=0,
        use_gpu=False,
        save_every=1,
        loss_func=None,
    )

    assert train_history == [1.0]
    assert val_history == []
    assert validation_calls == []
    assert not (checkpoint_dir / "best.ckpt").exists()
    assert (checkpoint_dir / "epoch_001.ckpt").is_file()
    last = torch.load(
        checkpoint_dir / "last.ckpt", map_location="cpu", weights_only=False
    )
    assert last["epoch"] == 0
    assert last["validation_enabled"] is False
    assert math.isinf(last["val_loss"]) and last["val_loss"] > 0
    assert last["val_loss_history"] == []
    assert "best_val_loss" not in last

    # Resume the no-validation run and retain empty validation/best state.
    resumed_config = _config(checkpoint_dir, num_epochs=2)
    resumed_config.resume_training = True
    resumed_config.resume_checkpoint_path = str(checkpoint_dir / "last.ckpt")
    resumed_model, resumed_optimizer, resumed_scheduler = _objects()
    resumed_train, resumed_val = trainer.train_model(
        resumed_config,
        resumed_model,
        train_dl=[],
        val_dl=None,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=None,
        local_rank=0,
        use_gpu=False,
        save_every=1,
        loss_func=None,
    )

    assert resumed_train == [1.0, 2.0]
    assert resumed_val == []
    assert train_epochs == [0, 1]
    assert validation_calls == []
    assert not (checkpoint_dir / "best.ckpt").exists()
    resumed_last = torch.load(
        checkpoint_dir / "last.ckpt", map_location="cpu", weights_only=False
    )
    assert resumed_last["epoch"] == 1
    assert resumed_last["validation_enabled"] is False
    assert math.isinf(resumed_last["val_loss"])
    assert resumed_last["val_loss_history"] == []
    assert "best_val_loss" not in resumed_last


def test_validation_enabled_requires_loader_and_defaults_true(tmp_path: Path) -> None:
    assert trainer._validation_enabled(SimpleNamespace()) is True
    assert (
        trainer._validation_enabled(
            SimpleNamespace(training=SimpleNamespace(validation_enabled=False))
        )
        is False
    )

    config = SimpleNamespace(validation_enabled=True)
    with pytest.raises(ValueError, match="Validation is enabled but val_dl is None"):
        trainer.train_model(
            config,
            model=None,
            train_dl=None,
            val_dl=None,
            optimizer=None,
            scheduler=None,
            scaler=None,
            local_rank=0,
            use_gpu=False,
            save_every=1,
            loss_func=None,
        )


@pytest.mark.parametrize("value", ["sometimes", object()])
def test_validation_policy_rejects_ambiguous_values(value) -> None:
    with pytest.raises(ValueError, match="validation_enabled"):
        trainer._validation_enabled(SimpleNamespace(validation_enabled=value))
