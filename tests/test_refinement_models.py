"""Model-level tests for the two-phase stochastic refinement.

Every test uses :class:`tests.refinement_fixtures.TinyPhase1`, which reproduces
the real Phase-1 output contract (physical + normalized outputs, per-predictand
scaling codes, non-negativity mask, opt-in feature capture) at a size that runs
in a couple of seconds on CPU.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from granitewxc.refinement import build_two_phase_model
from granitewxc.refinement.target_space import NormalizedTargetSpace
from refinement_fixtures import TinyPhase1, make_batch

REFINERS = [
    "diffusion_unet",
    "flow_matching_unet",
    "diffusion_transformer",
    "flow_matching_transformer",
]


def small_config(refiner_type: str, **overrides):
    refinement = {
        "enabled": refiner_type != "none",
        "type": refiner_type,
        "ensemble_size": 3,
        "unet": {"hidden_channels": 8, "num_levels": 2, "attention_heads": 4},
        "transformer": {
            "embedding_dim": 32,
            "num_heads": 4,
            "num_blocks": 2,
            "patch_size": 4,
        },
        "diffusion": {"training_timesteps": 50, "inference_steps": 3},
        "flow_matching": {"integration_steps": 3},
    }
    refinement.update(overrides)
    return {"refinement": refinement}


def build(refiner_type: str, batch=None, **overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model = build_two_phase_model(TinyPhase1(), small_config(refiner_type, **overrides))
    if batch is not None:
        model.initialize_from_batch(batch)
    return model


# ---------------------------------------------------------------------------
# Deterministic behaviour
# ---------------------------------------------------------------------------


def test_disabled_refinement_is_a_pass_through():
    batch = make_batch()
    phase1 = TinyPhase1()
    torch.manual_seed(0)
    reference = phase1(dict(batch))
    model = build_two_phase_model(phase1, {"refinement": {"type": "none"}})
    assert model.refiner is None
    assert model.phase1_frozen is False
    assert torch.equal(model(dict(batch)), reference)


def test_predict_without_refinement_returns_deterministic_only():
    batch = make_batch()
    model = build("none", batch)
    out = model.predict(batch, ensemble_size=5)
    assert out.refined is None and out.members is None
    assert out.deterministic.shape == batch["y"].shape


# ---------------------------------------------------------------------------
# Shapes, training and inference for every refiner
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("refiner_type", REFINERS)
def test_forward_shapes(refiner_type):
    batch = make_batch(height=22, width=30)
    model = build(refiner_type, batch)
    cond = model.build_conditioning(batch, model.run_phase1(batch)[1])
    assert cond.shape[-2:] == batch["y"].shape[-2:]
    residual = model.refiner.sample(cond)
    assert residual.shape == batch["y"].shape


@pytest.mark.parametrize("refiner_type", REFINERS)
def test_one_training_step(refiner_type):
    batch = make_batch(height=22, width=30, nan_fraction=0.15)
    model = build(refiner_type, batch)
    model.train()
    out = model.training_step(batch, generator=torch.Generator().manual_seed(0))
    loss = out.losses["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.refiner.parameters() if p.grad is not None]
    assert grads, "the refiner received no gradient"
    assert any(float(g.abs().sum()) > 0 for g in grads)
    assert all(p.grad is None for p in model.phase1.parameters())


@pytest.mark.parametrize("refiner_type", REFINERS)
def test_one_inference_step(refiner_type):
    batch = make_batch(height=22, width=30)
    model = build(refiner_type, batch)
    model.eval()
    out = model.predict(batch, ensemble_size=1, seed=0)
    assert out.members.shape == (batch["y"].shape[0], 1, *batch["y"].shape[1:])
    assert out.refined.shape == batch["y"].shape
    assert torch.isfinite(out.refined).all()


# ---------------------------------------------------------------------------
# Residual construction / normalization
# ---------------------------------------------------------------------------


def test_encode_decode_round_trip():
    phase1 = TinyPhase1()
    space = NormalizedTargetSpace(phase1)
    physical = torch.rand(2, 3, 8, 9) * 10
    normalized = space.encode(physical)
    assert torch.allclose(space.decode(normalized), physical, atol=1e-5)


def test_deterministic_normalized_decodes_to_deterministic_physical():
    """The Phase-1 normalized output is an exact pre-image of its physical output."""
    batch = make_batch()
    phase1 = TinyPhase1()
    physical, normalized = phase1(dict(batch), return_pre_inverse=True)
    space = NormalizedTargetSpace(phase1)
    assert torch.allclose(space.decode(normalized), physical, atol=1e-6)


def test_residual_target_and_reconstruction_are_consistent():
    batch = make_batch(nan_fraction=0.2)
    phase1 = TinyPhase1()
    space = NormalizedTargetSpace(phase1)
    _, normalized = phase1(dict(batch), return_pre_inverse=True)

    residual, valid = space.residual_target(batch["y"], normalized)
    assert valid.shape == batch["y"].shape
    assert torch.equal(valid, torch.isfinite(batch["y"]))
    # Invalid cells contribute a zero residual, never a NaN.
    assert torch.isfinite(residual).all()
    assert float(residual[~valid].detach().abs().max()) == 0.0

    refined_norm, refined_physical = space.reconstruct(normalized, residual)
    # On valid, non-negativity-unconstrained channels the reconstruction returns
    # the ground truth exactly.
    free = ~phase1.predictand_nonneg_enabled_mask
    sel = valid & free.view(1, -1, 1, 1)
    assert torch.allclose(refined_physical[sel], batch["y"][sel], atol=1e-4)


def test_residual_is_added_in_normalized_space_not_physical():
    """A normalized residual must never be added to a physical field."""
    phase1 = TinyPhase1()
    space = NormalizedTargetSpace(phase1)
    normalized = torch.zeros(1, 3, 4, 4)
    residual = torch.ones(1, 3, 4, 4)
    refined_norm, refined_physical = space.reconstruct(normalized, residual)
    assert torch.equal(refined_norm, residual)
    # channel 0 is divide_only with sigma=2 -> physical == 2 * normalized
    assert torch.allclose(refined_physical[:, 0], torch.full((1, 4, 4), 2.0))
    # channel 1 is zscore with mu=1.0, sigma=1.5
    assert torch.allclose(refined_physical[:, 1], torch.full((1, 4, 4), 1.5 + 1.0))


def test_physical_constraints_are_idempotent_and_nonnegative():
    phase1 = TinyPhase1()
    space = NormalizedTargetSpace(phase1)
    physical = torch.tensor([[[[-3.0]], [[-3.0]], [[-3.0]]]])
    once = space.apply_physical_constraints(physical)
    twice = space.apply_physical_constraints(once)
    assert torch.equal(once, twice)
    assert float(once[0, 0]) == 0.0   # precipitation-like channel is clamped
    assert float(once[0, 1]) == -3.0  # temperature-like channel is untouched


@pytest.mark.parametrize("refiner_type", REFINERS)
def test_masked_cells_stay_masked_and_output_is_finite_elsewhere(refiner_type):
    batch = make_batch(height=20, width=20, nan_fraction=0.25)
    model = build(refiner_type, batch)
    model.eval()
    out = model.predict(batch, ensemble_size=2, seed=3)
    invalid = ~torch.isfinite(batch["y"])
    assert torch.isnan(out.members[:, 0][invalid]).all()
    assert torch.isfinite(out.members[:, 0][~invalid]).all()


@pytest.mark.parametrize("refiner_type", REFINERS)
def test_precipitation_channel_is_never_negative(refiner_type):
    batch = make_batch(height=20, width=20)
    model = build(refiner_type, batch)
    model.eval()
    out = model.predict(batch, ensemble_size=2, seed=5)
    precip = out.members[:, :, 0]
    assert float(precip.min()) >= 0.0


# ---------------------------------------------------------------------------
# Rectangular domains and exact cropping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("refiner_type", REFINERS)
@pytest.mark.parametrize("hw", [(16, 24), (23, 31), (17, 17)])
def test_rectangular_and_non_divisible_domains(refiner_type, hw):
    height, width = hw
    batch = make_batch(height=height, width=width)
    model = build(refiner_type, batch)
    model.eval()
    out = model.predict(batch, ensemble_size=1, seed=0)
    assert out.members.shape[-2:] == (height, width)
    assert out.deterministic.shape[-2:] == (height, width)


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("refiner_type", REFINERS)
def test_ensemble_dimensions_and_statistics(refiner_type):
    batch = make_batch(height=16, width=20)
    model = build(refiner_type, batch)
    model.eval()
    n = 4
    out = model.predict(batch, ensemble_size=n, seed=11)
    b, c, h, w = batch["y"].shape
    assert out.members.shape == (b, n, c, h, w)
    assert out.member_residuals.shape == (b, n, c, h, w)
    assert out.ensemble_mean.shape == (b, c, h, w)
    assert out.ensemble_spread.shape == (b, c, h, w)
    assert torch.allclose(out.ensemble_mean, out.members.mean(dim=1), atol=1e-5)
    assert torch.allclose(out.ensemble_spread, out.members.std(dim=1, unbiased=True), atol=1e-4)


@pytest.mark.parametrize("refiner_type", REFINERS)
def test_ensemble_reproducibility(refiner_type):
    batch = make_batch(height=16, width=20)
    model = build(refiner_type, batch)
    model.eval()
    a = model.predict(batch, ensemble_size=3, seed=42)
    b = model.predict(batch, ensemble_size=3, seed=42)
    c = model.predict(batch, ensemble_size=3, seed=43)
    assert torch.equal(a.member_residuals, b.member_residuals)
    assert not torch.equal(a.member_residuals, c.member_residuals)


@pytest.mark.parametrize("refiner_type", REFINERS)
def test_members_are_distinct(refiner_type):
    batch = make_batch(height=16, width=20)
    model = build(refiner_type, batch)
    model.eval()
    out = model.predict(batch, ensemble_size=3, seed=7)
    assert not torch.equal(out.member_residuals[:, 0], out.member_residuals[:, 1])
    assert not torch.equal(out.member_residuals[:, 1], out.member_residuals[:, 2])


# ---------------------------------------------------------------------------
# Freezing / joint fine-tuning
# ---------------------------------------------------------------------------


def test_frozen_phase1_stays_in_eval_mode():
    batch = make_batch()
    model = build("diffusion_unet", batch)
    model.train()
    assert model.phase1_frozen is True
    assert model.phase1.training is False
    assert all(not p.requires_grad for p in model.phase1.parameters())


def test_joint_finetuning_keeps_phase1_trainable():
    batch = make_batch()
    model = build("diffusion_unet", batch, joint_finetuning=True)
    model.train()
    assert model.phase1_frozen is False
    assert model.phase1.training is True
    # Non-trainable scaler buffers stay frozen; every learnable weight is live.
    assert all(p.requires_grad for p in model.phase1.body.parameters())
    assert all(p.requires_grad for p in model.phase1.head.parameters())


def test_unused_phase1_features_are_not_computed():
    batch = make_batch()
    model = build("diffusion_unet", batch)
    _, _, features = model.run_phase1(batch)
    assert features == {}


def test_requested_phase1_features_are_returned():
    batch = make_batch()
    model = build(
        "diffusion_unet",
        batch=None,
        conditioning={
            "deterministic_output": True,
            "input_predictors": True,
            "prithvi_features": True,
            "unet_features": True,
            "static_fields": False,
            "masks": False,
        },
    )
    model.initialize_from_batch(batch)
    _, _, features = model.run_phase1(batch)
    assert set(features) == {"prithvi", "unet"}
