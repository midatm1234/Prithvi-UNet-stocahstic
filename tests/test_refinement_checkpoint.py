"""Checkpoint compatibility and resume tests for the two-phase model."""

from __future__ import annotations

import pytest
import torch

from granitewxc.refinement import build_two_phase_model
from granitewxc.refinement.checkpoint import (
    CHECKPOINT_KIND_COMBINED,
    CHECKPOINT_KIND_REFINEMENT,
    build_refinement_checkpoint,
    extract_model_state,
    load_phase1_state_dict,
    load_refinement_state_dict,
    migrate_phase1_state_dict,
    phase1_state_fingerprint,
    save_checkpoint_atomic,
    strip_wrapper_prefixes,
    validate_phase1_reference,
)
from granitewxc.refinement.training import RefinementTrainer
from refinement_fixtures import TinyPhase1, make_batch
from test_refinement_models import REFINERS, build


def legacy_checkpoint(phase1: TinyPhase1) -> dict:
    """A checkpoint in the historical deterministic layout (unprefixed keys)."""
    return {
        "model": {k: v.clone() for k, v in phase1.state_dict().items()},
        "optimizer": None,
        "epoch": 17,
        "loss": 2.6,
        "val_loss": 2.56,
    }


def _resume_config(model, *, scheduler_t_max: int = 10, warmup_steps: int = 0):
    """Resolved numerical contract used by the production Phase-2 entry point."""
    return {
        "refinement": model.refinement_config.to_dict(),
        "performance": model.performance_config.to_dict(),
        "training": {
            "contract_version": 1,
            "optimizer": "torch.optim.AdamW",
            "base_learning_rate": 1.0e-3,
            "min_learning_rate": 0.0,
            "scheduler": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "scheduler_t_max": scheduler_t_max,
            "warmup_steps": warmup_steps,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": None,
            "epochs": 2,
            "limit_steps_train": 0,
            "limit_steps_valid": 0,
            "effective_train_batches_per_epoch": 1,
            "optimizer_steps_per_epoch": 1,
            "total_optimizer_steps": 2,
        },
    }


# ---------------------------------------------------------------------------
# Key migration
# ---------------------------------------------------------------------------


def test_migration_only_adds_the_phase1_prefix():
    phase1 = TinyPhase1()
    original = phase1.state_dict()
    migrated, renames = migrate_phase1_state_dict(original)
    assert set(migrated) == {f"phase1.{k}" for k in original}
    assert renames == {k: f"phase1.{k}" for k in original}
    for key, value in original.items():
        assert torch.equal(migrated[f"phase1.{key}"], value)


def test_migration_is_idempotent():
    phase1 = TinyPhase1()
    once, _ = migrate_phase1_state_dict(phase1.state_dict())
    twice, renames = migrate_phase1_state_dict(once)
    assert set(once) == set(twice)
    assert renames == {}


def test_wrapper_prefixes_are_stripped():
    state = {"module._orig_mod.conv.weight": torch.zeros(1)}
    assert list(strip_wrapper_prefixes(state)) == ["conv.weight"]
    for nested in (
        "model.module.conv.weight",
        "network._orig_mod.conv.weight",
        "module.model._orig_mod.conv.weight",
    ):
        assert list(strip_wrapper_prefixes({nested: torch.zeros(1)})) == [
            "conv.weight"
        ]


def test_standard_state_dict_envelope_wins_over_model_hyperparameters():
    tensors = {"encoder.weight": torch.ones(1)}
    checkpoint = {
        "model": {"architecture": "not-a-state-dict"},
        "state_dict": tensors,
    }
    extracted = extract_model_state(checkpoint)
    assert set(extracted) == set(tensors)
    assert torch.equal(extracted["encoder.weight"], tensors["encoder.weight"])


def test_fingerprint_is_prefix_independent_and_value_sensitive():
    phase1 = TinyPhase1()
    plain = phase1.state_dict()
    prefixed = {f"module.{k}": v for k, v in plain.items()}
    assert phase1_state_fingerprint(plain) == phase1_state_fingerprint(prefixed)
    mutated = {k: v.clone() for k, v in plain.items()}
    mutated["head.bias"] = mutated["head.bias"] + 1.0
    assert phase1_state_fingerprint(mutated) != phase1_state_fingerprint(plain)


# ---------------------------------------------------------------------------
# Loading a deterministic checkpoint
# ---------------------------------------------------------------------------


def test_deterministic_checkpoint_loads_into_deterministic_wrapper():
    trained = TinyPhase1()
    with torch.no_grad():
        for p in trained.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    checkpoint = legacy_checkpoint(trained)

    wrapper = build_two_phase_model(TinyPhase1(), {"refinement": {"type": "none"}})
    report = load_phase1_state_dict(wrapper, checkpoint)
    assert report.missing == []
    assert report.unexpected == []
    assert report.shape_mismatched == []
    assert report.loaded == len(trained.state_dict())

    batch = make_batch()
    with torch.no_grad():
        expected = trained(dict(batch))
        actual, normalized, _ = wrapper.run_phase1(dict(batch))
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("refiner_type", REFINERS)
def test_deterministic_checkpoint_initializes_every_refiner(refiner_type):
    trained = TinyPhase1()
    checkpoint = legacy_checkpoint(trained)
    batch = make_batch()
    model = build(refiner_type, batch)

    report = load_phase1_state_dict(model, checkpoint)
    assert report.unexpected == []
    assert report.shape_mismatched == []
    # The only missing keys belong to the newly introduced Phase-2 module.
    assert report.missing, "expected the refiner keys to be reported as missing"
    assert all(k.startswith("refiner.") for k in report.missing)

    with torch.no_grad():
        expected = trained(dict(batch))
        actual, _, _ = model.run_phase1(dict(batch))
    assert torch.equal(actual, expected)


def test_unexplained_missing_key_is_rejected():
    trained = TinyPhase1()
    checkpoint = legacy_checkpoint(trained)
    del checkpoint["model"]["head.weight"]
    wrapper = build_two_phase_model(TinyPhase1(), {"refinement": {"type": "none"}})
    with pytest.raises(RuntimeError, match="missing Phase-1 key"):
        load_phase1_state_dict(wrapper, checkpoint)


def test_unexpected_key_is_rejected():
    trained = TinyPhase1()
    checkpoint = legacy_checkpoint(trained)
    checkpoint["model"]["mystery.weight"] = torch.zeros(3)
    wrapper = build_two_phase_model(TinyPhase1(), {"refinement": {"type": "none"}})
    with pytest.raises(RuntimeError, match="unexpected key"):
        load_phase1_state_dict(wrapper, checkpoint)


def test_shape_mismatch_is_never_silently_ignored():
    trained = TinyPhase1()
    checkpoint = legacy_checkpoint(trained)
    checkpoint["model"]["head.weight"] = torch.zeros(99, 8, 1, 1)
    wrapper = build_two_phase_model(TinyPhase1(), {"refinement": {"type": "none"}})
    with pytest.raises(RuntimeError, match="shape mismatch"):
        load_phase1_state_dict(wrapper, checkpoint)


# ---------------------------------------------------------------------------
# Refinement checkpoints
# ---------------------------------------------------------------------------


def test_refinement_checkpoint_excludes_phase1_weights():
    batch = make_batch()
    model = build("diffusion_unet", batch)
    payload = build_refinement_checkpoint(model, kind=CHECKPOINT_KIND_REFINEMENT)
    assert payload["checkpoint_kind"] == CHECKPOINT_KIND_REFINEMENT
    assert all(k.startswith("refiner.") for k in payload["model"])
    assert payload["model"], "the refinement checkpoint is empty"


def test_combined_checkpoint_is_portable():
    batch = make_batch()
    model = build("flow_matching_unet", batch)
    payload = build_refinement_checkpoint(model, kind=CHECKPOINT_KIND_COMBINED)
    assert any(k.startswith("phase1.") for k in payload["model"])
    assert any(k.startswith("refiner.") for k in payload["model"])

    fresh = build("flow_matching_unet", batch)
    fresh.load_state_dict(payload["model"], strict=True)


def test_loading_a_refinement_checkpoint_does_not_alter_phase1():
    batch = make_batch()
    source = build("diffusion_transformer", batch)
    with torch.no_grad():
        for p in source.refiner.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    payload = build_refinement_checkpoint(source, kind=CHECKPOINT_KIND_REFINEMENT)

    target = build("diffusion_transformer", batch)
    before = {k: v.clone() for k, v in target.state_dict().items() if k.startswith("phase1.")}
    report = load_refinement_state_dict(target, payload)
    assert report.missing == [] and report.unexpected == []
    after = {k: v for k, v in target.state_dict().items() if k.startswith("phase1.")}
    for key, value in before.items():
        assert torch.equal(value, after[key])
    for key, value in source.state_dict().items():
        if key.startswith("refiner."):
            assert torch.equal(value, target.state_dict()[key])


def test_refinement_checkpoint_records_phase1_identity():
    batch = make_batch()
    model = build("diffusion_unet", batch)
    phase1_state = {
        k[len("phase1.") :]: v for k, v in model.state_dict().items() if k.startswith("phase1.")
    }
    fingerprint = phase1_state_fingerprint(phase1_state)
    payload = build_refinement_checkpoint(model, phase1_fingerprint=fingerprint,
                                          phase1_checkpoint="/some/last.ckpt")
    assert validate_phase1_reference(payload, phase1_state) is True

    other = {k: v + 1.0 for k, v in phase1_state.items() if v.dtype.is_floating_point}
    other.update({k: v for k, v in phase1_state.items() if not v.dtype.is_floating_point})
    with pytest.raises(RuntimeError, match="Phase-1 identity mismatch"):
        validate_phase1_reference(payload, other)


def test_missing_phase1_fingerprint_is_rejected():
    with pytest.raises(RuntimeError, match="does not record a Phase-1 fingerprint"):
        validate_phase1_reference({}, {}, strict=True)


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("refiner_type", REFINERS)
def test_save_and_resume_restores_full_training_state(refiner_type, tmp_path):
    batch = make_batch(height=16, width=16)
    model = build(refiner_type, batch)
    params = [p for p in model.refiner.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    trainer = RefinementTrainer(
        model,
        optimizer,
        scheduler=scheduler,
        scaler=scaler,
        checkpoint_dir=str(tmp_path),
        phase1_checkpoint="/reference/last.ckpt",
        resolved_config=_resume_config(model),
        case_name="unit_test",
        seed=7,
        logger=lambda _msg: None,
    )
    trainer.fit([batch], [batch], num_epochs=2, save_every=1)

    assert trainer.state.epoch == 2
    assert trainer.state.global_step == 2
    saved_lr = optimizer.param_groups[0]["lr"]
    saved_weights = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("refiner.")}

    # Fresh objects, then resume. In a real workflow the same deterministic
    # Phase-1 checkpoint is reloaded first; do the same here.
    model2 = build(refiner_type, batch)
    model2.load_state_dict(
        {**model2.state_dict(),
         **{k: v for k, v in model.state_dict().items() if k.startswith("phase1.")}},
        strict=True,
    )
    params2 = [p for p in model2.refiner.parameters() if p.requires_grad]
    optimizer2 = torch.optim.AdamW(params2, lr=1e-3)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=10)
    scaler2 = torch.amp.GradScaler("cpu", enabled=False)
    trainer2 = RefinementTrainer(
        model2,
        optimizer2,
        scheduler=scheduler2,
        scaler=scaler2,
        checkpoint_dir=str(tmp_path),
        phase1_checkpoint="/reference/last.ckpt",
        resolved_config=_resume_config(model2),
        case_name="unit_test",
        logger=lambda _msg: None,
    )
    state = trainer2.resume(str(tmp_path / "last.ckpt"))

    assert state.epoch == 2
    assert state.global_step == 2
    assert optimizer2.param_groups[0]["lr"] == pytest.approx(saved_lr)
    for key, value in saved_weights.items():
        assert torch.equal(value, model2.state_dict()[key])
    assert trainer2.state.train_loss_history == trainer.state.train_loss_history


def test_resume_rejects_changed_warmup_or_cosine_horizon(tmp_path):
    """A checkpoint taken during warmup may not resume on a new LR trajectory."""
    batch = make_batch(height=16, width=16)
    model = build("diffusion_unet", batch)
    optimizer = torch.optim.AdamW(model.refiner.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    trainer = RefinementTrainer(
        model,
        optimizer,
        scheduler=scheduler,
        checkpoint_dir=str(tmp_path),
        resolved_config=_resume_config(
            model, scheduler_t_max=8, warmup_steps=4
        ),
        warmup_steps=4,
        logger=lambda _message: None,
    )
    trainer.train_one_epoch([batch], limit_steps=1)
    assert trainer.state.global_step == 1 < trainer.warmup_steps
    trainer.save()

    model2 = build("diffusion_unet", batch)
    model2.load_state_dict(
        {
            **model2.state_dict(),
            **{
                key: value
                for key, value in model.state_dict().items()
                if key.startswith("phase1.")
            },
        },
        strict=True,
    )
    optimizer2 = torch.optim.AdamW(model2.refiner.parameters(), lr=1.0e-3)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=9)
    trainer2 = RefinementTrainer(
        model2,
        optimizer2,
        scheduler=scheduler2,
        checkpoint_dir=str(tmp_path),
        resolved_config=_resume_config(
            model2, scheduler_t_max=9, warmup_steps=5
        ),
        warmup_steps=5,
        logger=lambda _message: None,
    )
    with pytest.raises(RuntimeError, match="training contract.*inexact resume"):
        trainer2.resume(str(tmp_path / "last.ckpt"))


def test_resume_rejects_saved_performance_or_precision_mismatch(tmp_path):
    batch = make_batch(height=16, width=16)
    model = build("flow_matching_unet", batch)
    trainer = RefinementTrainer(
        model,
        torch.optim.AdamW(model.refiner.parameters(), lr=1.0e-3),
        checkpoint_dir=str(tmp_path),
        resolved_config=_resume_config(model),
        logger=lambda _message: None,
    )
    path = trainer.save()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["resolved_config"]["performance"]["precision"]["mode"] = "bf16"
    save_checkpoint_atomic(payload, path, atomic=False)

    model2 = build("flow_matching_unet", batch)
    model2.load_state_dict(
        {
            **model2.state_dict(),
            **{
                key: value
                for key, value in model.state_dict().items()
                if key.startswith("phase1.")
            },
        },
        strict=True,
    )
    trainer2 = RefinementTrainer(
        model2,
        torch.optim.AdamW(model2.refiner.parameters(), lr=1.0e-3),
        checkpoint_dir=str(tmp_path),
        resolved_config=_resume_config(model2),
        logger=lambda _message: None,
    )
    with pytest.raises(RuntimeError, match="performance contract.*inexact resume"):
        trainer2.resume(path)


def test_schema2_resume_rejects_missing_training_contract(tmp_path):
    batch = make_batch(height=16, width=16)
    model = build("flow_matching_unet", batch)
    trainer = RefinementTrainer(
        model,
        torch.optim.AdamW(model.refiner.parameters(), lr=1.0e-3),
        checkpoint_dir=str(tmp_path),
        resolved_config={"refinement": model.refinement_config.to_dict()},
        logger=lambda _message: None,
    )
    path = trainer.save()

    model2 = build("flow_matching_unet", batch)
    model2.load_state_dict(
        {
            **model2.state_dict(),
            **{
                key: value
                for key, value in model.state_dict().items()
                if key.startswith("phase1.")
            },
        },
        strict=True,
    )
    trainer2 = RefinementTrainer(
        model2,
        torch.optim.AdamW(model2.refiner.parameters(), lr=1.0e-3),
        checkpoint_dir=str(tmp_path),
        logger=lambda _message: None,
    )
    with pytest.raises(RuntimeError, match="resolved_config.training"):
        trainer2.resume(path)


def test_schema1_resume_is_rejected_before_state_loading(tmp_path):
    batch = make_batch(height=16, width=16)
    model = build("flow_matching_unet", batch)
    trainer = RefinementTrainer(
        model,
        torch.optim.AdamW(model.refiner.parameters(), lr=1.0e-3),
        checkpoint_dir=str(tmp_path),
        resolved_config=_resume_config(model),
        logger=lambda _message: None,
    )
    path = trainer.save()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["checkpoint_schema_version"] = 1
    payload["refinement_semantics_version"] = "legacy-hurdle-latent-v1"
    save_checkpoint_atomic(payload, path, atomic=False)

    with pytest.raises(RuntimeError, match="cannot be resumed safely"):
        trainer.resume(path)


def test_resume_detects_a_different_phase1(tmp_path):
    batch = make_batch(height=16, width=16)
    model = build("diffusion_unet", batch)
    optimizer = torch.optim.AdamW(model.refiner.parameters(), lr=1e-3)
    trainer = RefinementTrainer(
        model,
        optimizer,
        checkpoint_dir=str(tmp_path),
        resolved_config=_resume_config(model),
        logger=lambda _m: None,
    )
    trainer.save()

    other = build("diffusion_unet", batch)
    with torch.no_grad():
        for p in other.phase1.parameters():
            p.add_(1.0)
    trainer2 = RefinementTrainer(
        other,
        torch.optim.AdamW(other.refiner.parameters(), lr=1e-3),
        checkpoint_dir=str(tmp_path),
        resolved_config=_resume_config(other),
        logger=lambda _m: None,
    )
    with pytest.raises(RuntimeError, match="Phase-1 identity mismatch"):
        trainer2.resume(str(tmp_path / "last.ckpt"))


def test_atomic_save_leaves_no_temporary_files(tmp_path):
    path = tmp_path / "ckpt.pt"
    save_checkpoint_atomic({"a": torch.zeros(2)}, path, atomic=True)
    assert path.exists()
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt.pt"]
