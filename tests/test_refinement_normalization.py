"""Physical residual normalization and reconstruction contract tests."""

from __future__ import annotations

import pytest
import torch

from granitewxc.refinement import build_two_phase_model
from granitewxc.refinement.config import ResidualNormalizationConfig
from granitewxc.refinement.normalization import ResidualNormalizer
from granitewxc.refinement.target_space import NormalizedTargetSpace
from refinement_fixtures import TinyPhase1, make_batch


def test_training_residual_normalization_round_trip():
    normalizer = ResidualNormalizer(3, ResidualNormalizationConfig())
    residual = torch.randn(5, 3, 9, 7) * torch.tensor([1.0, 4.0, 0.2]).view(1, 3, 1, 1)
    residual += torch.tensor([0.4, -2.0, 0.05]).view(1, 3, 1, 1)
    valid = torch.ones_like(residual, dtype=torch.bool)
    valid[0, :, :2, :3] = False
    normalizer.update(residual[:2], valid[:2])
    normalizer.update(residual[2:], valid[2:])
    normalizer.finalize()

    normalized = normalizer.normalize(residual, valid)
    restored = normalizer.denormalize(normalized)
    torch.testing.assert_close(restored[valid], residual[valid], atol=1e-6, rtol=1e-6)
    assert normalizer.is_fitted
    assert torch.all(normalizer.scale > 0)


def test_unfitted_statistics_are_never_used_for_inference():
    normalizer = ResidualNormalizer(2, ResidualNormalizationConfig())
    with pytest.raises(RuntimeError, match="training split"):
        normalizer.normalize(torch.zeros(1, 2, 3, 4))
    with pytest.raises(RuntimeError, match="training split"):
        normalizer.denormalize(torch.zeros(1, 2, 3, 4))


def test_physical_residual_contract_and_zero_identity():
    phase1 = TinyPhase1()
    space = NormalizedTargetSpace(phase1)
    batch = make_batch()
    base, _ = phase1(batch, return_pre_inverse=True)
    residual, valid = space.physical_residual_target(batch["y"], base)
    reconstructed = space.reconstruct_physical(base, residual)
    torch.testing.assert_close(reconstructed[valid], batch["y"][valid], atol=1e-6, rtol=1e-6)
    assert torch.equal(space.reconstruct_physical(base, torch.zeros_like(base)), base)


def test_identity_mode_has_no_persistent_statistics():
    cfg = ResidualNormalizationConfig(method="identity")
    normalizer = ResidualNormalizer(2, cfg)
    residual = torch.randn(1, 2, 3, 4)
    assert normalizer.state_dict() == {}
    assert torch.equal(normalizer.denormalize(normalizer.normalize(residual)), residual)


def test_ensemble_is_aggregated_after_physical_reconstruction_and_constraints(
    monkeypatch,
):
    """Averaging latent residuals first is not equivalent near physical bounds."""
    batch = make_batch(batch_size=1, height=6, width=7)
    config = {
        "refinement": {
            "enabled": True,
            "type": "diffusion_unet",
            "ensemble_size": 2,
            "residual_normalization": {"method": "standardize"},
            "unet": {
                "hidden_channels": 8,
                "num_levels": 2,
                "attention_heads": 4,
            },
            "diffusion": {"training_timesteps": 20, "inference_steps": 2},
        }
    }
    model = build_two_phase_model(TinyPhase1(), config)
    model.initialize_from_batch(batch)
    normalizer = model.residual_normalizer
    assert normalizer is not None
    normalizer.mean.copy_(
        torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64).view(1, 3, 1, 1)
    )
    normalizer.scale.copy_(
        torch.tensor([2.0, 4.0, 0.5], dtype=torch.float64).view(1, 3, 1, 1)
    )
    normalizer.fitted.fill_(True)

    raw_members = []
    for values in ((-100.0, 0.0, 2.0), (2.0, 2.0, -2.0)):
        raw_members.append(
            torch.tensor(values).view(1, 3, 1, 1).expand(1, 3, 6, 7).clone()
        )
    calls = iter(raw_members)

    def controlled_sample(conditioning, **kwargs):
        del conditioning, kwargs
        return next(calls)

    monkeypatch.setattr(model.refiner, "sample", controlled_sample)
    model.eval()
    out = model.predict(batch, ensemble_size=2, seed=9, chunk_size=1)

    base = out.deterministic
    expected_members = torch.stack(
        [
            model.target_space.reconstruct_physical(
                base,
                normalizer.denormalize(raw),
            )
            for raw in raw_members
        ],
        dim=1,
    )
    torch.testing.assert_close(out.members, expected_members)
    torch.testing.assert_close(
        out.member_residuals_physical,
        expected_members - base.unsqueeze(1),
    )
    expected_mean = base.float() + (
        expected_members - base.unsqueeze(1)
    ).float().mean(dim=1)
    torch.testing.assert_close(out.ensemble_mean, expected_mean)
    torch.testing.assert_close(out.refined, expected_mean)

    latent_first = model.target_space.reconstruct_physical(
        base,
        normalizer.denormalize(torch.stack(raw_members, dim=1).mean(dim=1)),
    )
    assert not torch.allclose(
        out.ensemble_mean[:, 0],
        latent_first[:, 0],
    ), "ensemble members were averaged before physical reconstruction"


def test_explicit_prefill_target_mask_controls_statistics_and_training_target():
    batch = make_batch(batch_size=2, height=6, width=7)
    explicit = torch.ones_like(batch["y"], dtype=torch.bool)
    explicit[0, 0, :2, :3] = False
    explicit[1, 1, 3:, 4:] = False
    batch["__target_valid_mask"] = explicit
    # CORDEX keeps a finite-filled target for legacy deterministic training;
    # these zeros must not become valid `0 - phase1` refinement targets.
    batch["y"] = torch.where(explicit, batch["y"], torch.zeros_like(batch["y"]))

    model = build_two_phase_model(
        TinyPhase1(),
        {
            "refinement": {
                "type": "flow_matching_unet",
                "residual_normalization": {"method": "standardize"},
                "flow_matching": {"integration_steps": 2},
                "unet": {"hidden_channels": 8, "num_levels": 2},
            }
        },
    )
    model.initialize_from_batch(batch)
    base, _, _ = model.run_phase1(batch)
    expected = batch["y"] - base

    model.reset_residual_statistics()
    model.update_residual_statistics(batch)
    normalizer = model.residual_normalizer
    assert normalizer is not None
    expected_count = explicit.sum(dim=(0, 2, 3)).reshape(1, 3, 1, 1)
    torch.testing.assert_close(normalizer.count, expected_count.to(normalizer.count))
    for channel in range(3):
        torch.testing.assert_close(
            normalizer.mean[0, channel, 0, 0],
            expected[:, channel][explicit[:, channel]].double().mean(),
        )
    model.finalize_residual_statistics()

    output = model.training_step(
        batch, generator=torch.Generator().manual_seed(31)
    )
    assert torch.equal(output.valid_mask, explicit)
    assert torch.count_nonzero(output.residual_target_physical[~explicit]) == 0
    torch.testing.assert_close(
        output.residual_target_physical[explicit], expected[explicit]
    )


def test_signed_log_transform_round_trip_and_leaves_other_channels_unchanged():
    """Signed-log compression on nonnegative channels only, exact inverse."""
    mask = torch.tensor([True, False, False])
    cfg = ResidualNormalizationConfig(signed_log_nonnegative_channels=True, signed_log_scale=1.0)
    signed_log = ResidualNormalizer(3, cfg, nonnegative_mask=mask)
    plain = ResidualNormalizer(3, ResidualNormalizationConfig())

    torch.manual_seed(0)
    residual = torch.randn(6, 3, 5, 5) * torch.tensor([12.0, 3.0, 0.5]).view(1, 3, 1, 1)
    valid = torch.ones_like(residual, dtype=torch.bool)

    signed_log.update(residual, valid)
    signed_log.finalize()
    plain.update(residual, valid)
    plain.finalize()

    normalized = signed_log.normalize(residual, valid)
    restored = signed_log.denormalize(normalized)
    torch.testing.assert_close(restored, residual, atol=1e-5, rtol=1e-5)

    # Untransformed channels (1, 2) must standardize identically either way.
    torch.testing.assert_close(signed_log.mean[:, 1:], plain.mean[:, 1:])
    torch.testing.assert_close(signed_log.scale[:, 1:], plain.scale[:, 1:])
    normalized_plain = plain.normalize(residual, valid)
    torch.testing.assert_close(normalized[:, 1:], normalized_plain[:, 1:])

    # The masked (precipitation-like) channel is fit in transformed space, so
    # its statistics differ from a plain standardizer over the raw residual.
    assert not torch.allclose(signed_log.mean[:, 0], plain.mean[:, 0])
    assert not torch.allclose(signed_log.scale[:, 0], plain.scale[:, 0])

    # A large-magnitude residual on the masked channel must compress relative
    # to a small one: this is the entire point of the transform.
    small = torch.full((1, 3, 1, 1), 0.2)
    large = torch.full((1, 3, 1, 1), 40.0)
    small_z = signed_log.normalize(small, torch.ones_like(small, dtype=torch.bool))
    large_z = signed_log.normalize(large, torch.ones_like(large, dtype=torch.bool))
    assert (large_z[:, 0] / small_z[:, 0]).abs() < (large / small)[:, 0].abs()


def test_signed_log_requires_standardize_method():
    with pytest.raises(Exception, match="standardize"):
        ResidualNormalizationConfig.from_mapping(
            {"method": "identity", "signed_log_nonnegative_channels": True}
        )


def test_signed_log_wired_through_two_phase_model_from_phase1_nonneg_mask():
    """The model must derive the transform mask from Phase 1, not the caller."""
    batch = make_batch(batch_size=2, height=6, width=7)
    model = build_two_phase_model(
        TinyPhase1(),
        {
            "refinement": {
                "type": "flow_matching_unet",
                "residual_normalization": {
                    "method": "standardize",
                    "signed_log_nonnegative_channels": True,
                    "signed_log_scale": 1.0,
                },
                "flow_matching": {"integration_steps": 2},
                "unet": {"hidden_channels": 8, "num_levels": 2},
            }
        },
    )
    model.initialize_from_batch(batch)
    normalizer = model.residual_normalizer
    assert normalizer is not None
    assert normalizer.signed_log_enabled
    torch.testing.assert_close(
        normalizer.signed_log_mask.reshape(-1),
        torch.tensor([True, False, False]),
    )

    model.reset_residual_statistics()
    model.update_residual_statistics(batch)
    model.finalize_residual_statistics()
    output = model.training_step(batch, generator=torch.Generator().manual_seed(7))
    assert torch.isfinite(output.losses["loss"])


def test_explicit_target_mask_shape_is_strict():
    batch = make_batch(batch_size=1, height=6, width=7)
    batch["__target_valid_mask"] = torch.ones(1, 1, 6, 7, dtype=torch.bool)
    model = build_two_phase_model(
        TinyPhase1(),
        {
            "refinement": {
                "type": "flow_matching_unet",
                "residual_normalization": {"method": "standardize"},
                "flow_matching": {"integration_steps": 2},
                "unet": {"hidden_channels": 8, "num_levels": 2},
            }
        },
    )
    model.initialize_from_batch(batch)
    with pytest.raises(ValueError, match="__target_valid_mask shape"):
        model.update_residual_statistics(batch)
