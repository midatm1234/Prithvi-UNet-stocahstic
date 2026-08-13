"""Performance-parity tests.

Every optimization in this branch must be *numerically transparent*: the
optimized path has to reproduce the reference path bitwise, or within a
documented tolerance for the dtype involved.  These tests are the acceptance
gate referred to in the benchmark report.
"""

from __future__ import annotations

import pytest
import torch

from granitewxc.refinement.backbones import SpatialResidualTransformer
from granitewxc.refinement.cache import CACHE_SCHEMA_VERSION, Phase1CacheKey, Phase1ConditioningCache
from granitewxc.refinement.config import Phase1CachePerfConfig
from granitewxc.refinement.schedules import DiffusionSchedule
from refinement_fixtures import make_batch
from test_refinement_models import REFINERS, build

#: fp32 elementwise tolerance for mathematically-equivalent kernel swaps.
FP32_ATOL = 1e-5
FP32_RTOL = 1e-5


# ---------------------------------------------------------------------------
# Ensemble generation: serial vs batched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("refiner_type", REFINERS)
def test_serial_and_batched_ensembles_are_bitwise_identical(refiner_type):
    batch = make_batch(height=16, width=20)
    model = build(refiner_type, batch)
    model.eval()
    reference = model.predict(batch, ensemble_size=4, seed=99, chunk_size=1)
    for chunk in (2, 3, 4):
        optimized = model.predict(batch, ensemble_size=4, seed=99, chunk_size=chunk)
        assert torch.equal(reference.member_residuals, optimized.member_residuals), chunk


@pytest.mark.parametrize("refiner_type", REFINERS)
def test_ensemble_member_order_is_stable_across_chunkings(refiner_type):
    batch = make_batch(height=16, width=16)
    model = build(refiner_type, batch)
    model.eval()
    a = model.predict(batch, ensemble_size=5, seed=3, chunk_size=1)
    b = model.predict(batch, ensemble_size=5, seed=3, chunk_size=5)
    for member in range(5):
        assert torch.equal(a.member_residuals[:, member], b.member_residuals[:, member])


# ---------------------------------------------------------------------------
# Attention kernels
# ---------------------------------------------------------------------------


def test_sdpa_and_math_attention_agree():
    torch.manual_seed(0)
    model = SpatialResidualTransformer(
        in_channels=2, cond_channels=3, out_channels=2,
        patch_size=(4, 4), embedding_dim=32, num_heads=4, num_blocks=3,
        zero_init_output=False,
    ).eval()
    with torch.no_grad():
        for block in model.blocks:
            block.ada_ln[1].weight.normal_(0, 0.05)
            block.ada_ln[1].bias.normal_(0, 0.5)
    x = torch.randn(2, 2, 16, 20)
    cond = torch.randn(2, 3, 16, 20)
    t = torch.tensor([5.0, 500.0])

    model.set_attention_implementation("math")
    with torch.no_grad():
        reference = model(x, cond, t)
    model.set_attention_implementation("sdpa")
    with torch.no_grad():
        optimized = model(x, cond, t)
    assert torch.allclose(reference, optimized, atol=FP32_ATOL, rtol=FP32_RTOL)
    assert float((reference - optimized).abs().max()) < FP32_ATOL


def test_gradient_checkpointing_matches_eager_gradients():
    torch.manual_seed(0)
    kwargs = dict(
        in_channels=2, cond_channels=2, out_channels=2, patch_size=(4, 4),
        embedding_dim=32, num_heads=4, num_blocks=2, zero_init_output=False,
    )
    eager = SpatialResidualTransformer(**kwargs, gradient_checkpointing=False)
    checkpointed = SpatialResidualTransformer(**kwargs, gradient_checkpointing=True)
    checkpointed.load_state_dict(eager.state_dict())
    eager.train()
    checkpointed.train()

    x = torch.randn(1, 2, 8, 12)
    cond = torch.randn(1, 2, 8, 12)
    t = torch.tensor([100.0])

    eager(x, cond, t).sum().backward()
    checkpointed(x, cond, t).sum().backward()
    for (name, a), (_, b) in zip(
        eager.named_parameters(), checkpointed.named_parameters(), strict=True
    ):
        assert torch.allclose(a.grad, b.grad, atol=1e-5, rtol=1e-5), name


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("schedule", ["cosine", "linear", "scaled_linear"])
def test_schedule_is_deterministic_and_monotone(schedule):
    a = DiffusionSchedule(num_train_timesteps=200, schedule=schedule)
    b = DiffusionSchedule(num_train_timesteps=200, schedule=schedule)
    assert torch.equal(a.alphas_cumprod, b.alphas_cumprod)
    assert bool((a.alphas_cumprod.diff() <= 0).all()), "alpha_bar must be non-increasing"
    assert float(a.alphas_cumprod[0]) <= 1.0
    assert float(a.alphas_cumprod[-1]) >= 0.0


@pytest.mark.parametrize("prediction_type", ["epsilon", "velocity", "sample"])
def test_prediction_type_conversions_are_consistent(prediction_type):
    schedule = DiffusionSchedule(num_train_timesteps=100, schedule="cosine")
    torch.manual_seed(0)
    clean = torch.randn(4, 3, 8, 8)
    noise = torch.randn(4, 3, 8, 8)
    # Very late timesteps have sqrt(alpha_bar) -> 0, so the epsilon -> x0
    # inversion is intrinsically ill-conditioned there. Stay in the range the
    # samplers actually rely on for the exactness check.
    t = torch.tensor([5, 20, 60, 90])
    noisy = schedule.add_noise(clean, noise, t)
    target = schedule.training_target(prediction_type, clean, noise, t)
    assert torch.allclose(
        schedule.to_clean(prediction_type, target, noisy, t), clean, atol=1e-3, rtol=1e-3
    )
    assert torch.allclose(
        schedule.to_epsilon(prediction_type, target, noisy, t), noise, atol=1e-3, rtol=1e-3
    )


def test_inference_timesteps_are_descending_and_in_range():
    schedule = DiffusionSchedule(num_train_timesteps=1000, schedule="cosine")
    steps = schedule.inference_timesteps(50, torch.device("cpu"))
    assert steps.shape == (50,)
    assert bool((steps.diff() < 0).all())
    assert int(steps.max()) < 1000 and int(steps.min()) >= 0
    with pytest.raises(ValueError, match="exceeds"):
        schedule.inference_timesteps(2000, torch.device("cpu"))


def test_schedule_buffers_are_not_persisted():
    """Deterministic schedule tensors must not bloat refinement checkpoints."""
    schedule = DiffusionSchedule(num_train_timesteps=1000)
    assert schedule.state_dict() == {}


# ---------------------------------------------------------------------------
# Online vs cached Phase-1 conditioning
# ---------------------------------------------------------------------------


def _cache_key(fingerprint: str = "abc123", **overrides) -> Phase1CacheKey:
    payload = dict(
        phase1_fingerprint=fingerprint,
        phase1_architecture="TinyPhase1",
        case_name="unit_test",
        split="training",
        predictors=["a", "b"],
        targets=["ppt", "tmax", "tmin"],
        levels=[500, 700, 850],
        normalization={"predictor_mode": "global"},
        preprocessing={"regrid": "linear"},
        spatial_domain={"lat": [32.5, 41.0], "lon": [-124.5, -115.9]},
        grid_shape=[16, 16],
    )
    payload.update(overrides)
    return Phase1CacheKey(**payload)


def test_cached_phase1_conditioning_matches_online(tmp_path):
    batch = make_batch(height=16, width=16)
    model = build("diffusion_unet", batch)
    model.eval()

    cfg = Phase1CachePerfConfig(enabled=True, path=str(tmp_path), validate_samples=1)
    cache = Phase1ConditioningCache(tmp_path, _cache_key(), cfg)
    cache.open_for_write()
    cache.assert_compatible()

    _, normalized, _ = model.run_phase1(batch)
    cache.write("sample_0", timestamp=20160101, deterministic_normalized=normalized)

    stats = cache.validate(model, [("sample_0", batch)])
    assert stats["checked"] == 1.0
    assert stats["max_abs"] == 0.0

    cached_batch = cache.inject(dict(batch), cache.read("sample_0"))
    model.train()
    generator_a = torch.Generator().manual_seed(11)
    generator_b = torch.Generator().manual_seed(11)
    online = model.training_step(batch, generator=generator_a)
    replayed = model.training_step(cached_batch, generator=generator_b)
    assert torch.allclose(online.losses["loss"], replayed.losses["loss"], atol=1e-6)
    assert torch.allclose(online.residual_target, replayed.residual_target, atol=1e-6)


def test_stale_cache_is_rejected(tmp_path):
    cfg = Phase1CachePerfConfig(enabled=True, path=str(tmp_path))
    Phase1ConditioningCache(tmp_path, _cache_key("first"), cfg).open_for_write()

    # A different Phase-1 checkpoint maps to a different directory entirely.
    other = Phase1ConditioningCache(tmp_path, _cache_key("second"), cfg)
    assert not other.exists()
    with pytest.raises(FileNotFoundError):
        other.assert_compatible()


def test_cache_rejects_mismatched_manifest(tmp_path):
    cfg = Phase1CachePerfConfig(enabled=True, path=str(tmp_path))
    key = _cache_key()
    cache = Phase1ConditioningCache(tmp_path, key, cfg)
    cache.open_for_write()

    payload = {k: v for k, v in key.to_dict().items() if k != "digest"}
    payload["grid_shape"] = [32, 32]
    forged = Phase1CacheKey(**payload)
    # Same directory, different expectations -> explicit rejection.
    conflicting = Phase1ConditioningCache(tmp_path, forged, cfg)
    conflicting.directory = cache.directory
    with pytest.raises(RuntimeError, match="incompatible"):
        conflicting.assert_compatible()


def test_cache_entry_records_schema_version(tmp_path):
    cfg = Phase1CachePerfConfig(enabled=True, path=str(tmp_path))
    key = _cache_key()
    assert key.schema_version == CACHE_SCHEMA_VERSION
    cache = Phase1ConditioningCache(tmp_path, key, cfg)
    cache.open_for_write()
    cache.write("s", timestamp=1, deterministic_normalized=torch.zeros(1, 3, 4, 4))
    assert cache.read("s")["cache_digest"] == key.digest()


def test_cached_conditioning_is_refused_when_phase1_is_trainable():
    batch = make_batch(height=16, width=16)
    model = build("diffusion_unet", batch, joint_finetuning=True)
    batch = dict(batch)
    batch["__phase1_normalized"] = torch.zeros_like(batch["y"])
    with pytest.raises(RuntimeError, match="Phase 1 is trainable"):
        model.training_step(batch)


def test_complete_daily_cache_skips_phase1_for_init_scan_and_training(monkeypatch):
    batch = make_batch(height=16, width=16, nan_fraction=0.1)
    reference = build("flow_matching_unet", batch)
    _, baseline, _ = reference.run_phase1(batch)
    residual, valid = reference.target_space.residual_target(batch["y"], baseline)
    cached = dict(batch)
    cached["__phase1_normalized"] = baseline.detach()
    cached["__residual_target_normalized"] = residual.detach()
    cached["__residual_valid_mask"] = valid.detach()

    model = build(
        "flow_matching_unet",
        residual_normalization={"enabled": True, "epsilon": 1.0e-6},
    )

    def unexpected_phase1(*_args, **_kwargs):
        raise AssertionError("complete cached batch unexpectedly ran Phase 1")

    monkeypatch.setattr(model, "run_phase1", unexpected_phase1)
    model.initialize_from_batch(cached)
    metadata = model.fit_residual_normalizer([cached])
    assert metadata["fitted"] is True
    output = model.training_step(
        cached, generator=torch.Generator().manual_seed(4)
    )
    assert torch.isfinite(output.losses["loss"])


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("refiner_type", ["diffusion_unet", "flow_matching_unet"])
def test_masked_loss_reduction_is_computed_in_fp32(refiner_type):
    from granitewxc.refinement.base import masked_loss

    prediction = torch.randn(2, 3, 16, 16, dtype=torch.bfloat16)
    target = torch.randn(2, 3, 16, 16, dtype=torch.bfloat16)
    mask = torch.rand(2, 3, 16, 16) > 0.3
    loss = masked_loss(prediction, target, mask, "mse")
    assert loss.dtype == torch.float32


def test_masked_loss_excludes_invalid_cells_from_the_denominator():
    from granitewxc.refinement.base import masked_loss

    prediction = torch.zeros(1, 1, 4, 4)
    target = torch.ones(1, 1, 4, 4)
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    mask[0, 0, 0, 0] = True
    assert float(masked_loss(prediction, target, mask, "mse")) == pytest.approx(1.0)
    assert float(masked_loss(prediction, target, None, "mse")) == pytest.approx(1.0)
    mask[0, 0, 1, 1] = True
    assert float(masked_loss(prediction, target, mask, "mse")) == pytest.approx(1.0)
