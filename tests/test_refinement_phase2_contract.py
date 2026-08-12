"""Focused regressions for the audited NARR--PRISM Phase-2 contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from granitewxc.refinement import build_two_phase_model
from granitewxc.refinement.backbones import patchify_2d, unpatchify_2d
from granitewxc.refinement.checkpoint import (
    build_refinement_checkpoint,
    load_phase1_state_dict,
    load_refinement_state_dict,
)
from granitewxc.refinement.config import resolve_refinement_config
from granitewxc.refinement.training import RefinementTrainer
from granitewxc.utils.config import get_config
from refinement_fixtures import TinyPhase1, make_batch


REFINERS = (
    "diffusion_unet",
    "diffusion_transformer",
    "flow_matching_unet",
    "flow_matching_transformer",
)


def _tiny_config(kind: str, **extra):
    refinement = {
        "type": kind,
        "seed": 9,
        "conditioning": {
            "deterministic_output": True,
            "input_predictors": True,
            "static_fields": False,
            "masks": False,
        },
        "unet": {
            "hidden_channels": 4,
            "num_levels": 1,
            "time_embedding_dim": 8,
            "bottleneck_attention": False,
        },
        "transformer": {
            "patch_size": 2,
            "embedding_dim": 16,
            "num_heads": 4,
            "num_blocks": 1,
            "mlp_ratio": 2,
        },
    }
    if kind.startswith("diffusion"):
        refinement["diffusion"] = {"training_timesteps": 12, "inference_steps": 2}
    else:
        refinement["flow_matching"] = {"integration_steps": 2, "solver": "euler"}
    refinement.update(extra)
    return {"refinement": refinement}


class ScaledHaloPhase1(TinyPhase1):
    n_input_timestamps = 1

    def __init__(self) -> None:
        super().__init__()
        # Make the expected arithmetic exactly representable; the production
        # model's configured epsilon is covered independently.
        self.input_scalers_epsilon = 0.0

    def _resolve_input_scalers(self, reference, scaler_offset=None):
        channels = reference.shape[1]
        mu = torch.full(
            (1, channels, 1, 1), 10.0, device=reference.device, dtype=reference.dtype
        )
        sigma = torch.full_like(mu, 2.0)
        return mu, sigma


class HurdleLikePhase1(TinyPhase1):
    """Returns an ungated precipitation latent and a gated physical field."""

    def forward(self, batch, return_pre_inverse=False, return_raw_output=False):
        physical, latent = super().forward(batch, return_pre_inverse=True)
        physical = physical.clone()
        physical[:, 0, ::2, ::2] = 0.0
        if return_pre_inverse:
            return physical, latent
        return physical


def test_halo_conditioning_is_normalized_and_uses_exact_output_crop():
    phase1 = ScaledHaloPhase1()
    config = _tiny_config(
        "flow_matching_unet",
        conditioning={
            "deterministic_output": False,
            "input_predictors": True,
            "static_fields": False,
            "masks": False,
        },
    )
    model = build_two_phase_model(phase1, config)
    marker = torch.arange(4 * 8 * 10, dtype=torch.float32).reshape(1, 4, 8, 10)
    batch = {
        "x": marker,
        "y": torch.ones(1, 3, 4, 4),
        "__output_crop": torch.tensor([[2, 3, 4, 4]]),
        "__input_scaler_offset": torch.tensor([[0, 0]]),
        "__output_scaler_offset": torch.tensor([[0, 0]]),
    }
    _, baseline, features = model.run_phase1(batch)
    conditioning = model.build_conditioning(batch, baseline, features)
    expected = ((marker - 10.0) / 2.0)[..., 2:6, 3:7]
    assert torch.equal(conditioning, expected)


def test_hurdle_zero_residual_reconstructs_exact_phase1_field():
    batch = make_batch(batch_size=1, height=8, width=8, with_static=False)
    model = build_two_phase_model(HurdleLikePhase1(), _tiny_config("flow_matching_unet"))
    physical, baseline, _ = model.run_phase1(batch)
    decoded = model.target_space.decode(baseline)
    assert torch.allclose(decoded, physical, atol=1e-6)
    _, zero_reconstruction = model.target_space.reconstruct(
        baseline, torch.zeros_like(baseline)
    )
    assert torch.allclose(zero_reconstruction, physical, atol=1e-6)
    assert torch.equal(zero_reconstruction[:, 0, ::2, ::2], torch.zeros_like(physical[:, 0, ::2, ::2]))


def test_absolute_target_mode_is_rejected_instead_of_added_as_residual():
    with pytest.raises(ValueError, match="train_on_residual=true"):
        resolve_refinement_config(
            {"refinement": {"type": "diffusion_unet", "train_on_residual": False}}
        )


def test_residual_normalization_fits_training_residuals_and_round_trips_checkpoint(tmp_path):
    batch_a = make_batch(
        batch_size=1,
        height=8,
        width=8,
        seed=1,
        with_static=False,
        nan_fraction=0.2,
    )
    batch_b = make_batch(batch_size=1, height=8, width=8, seed=2, with_static=False)
    config = _tiny_config(
        "flow_matching_unet",
        residual_normalization={"enabled": True, "epsilon": 1e-6},
    )
    model = build_two_phase_model(TinyPhase1(), config)
    model.initialize_from_batch(batch_a)
    with pytest.raises(RuntimeError, match="has not been fitted"):
        model.training_step(batch_a)
    _, baseline, features = model.run_phase1(batch_a)
    conditioning = model.build_conditioning(batch_a, baseline, features)
    with pytest.raises(RuntimeError, match="has not been fitted"):
        model.refiner.sample(conditioning)

    metadata = model.fit_residual_normalizer([batch_a, batch_b])
    assert metadata["enabled"] and metadata["fitted"]
    assert metadata["fit_split"] == "training"
    training_output = model.training_step(batch_a)
    assert torch.isfinite(training_output.losses["loss"])
    invalid = ~training_output.valid_mask
    assert torch.equal(
        training_output.diagnostics["clean_residual"][invalid],
        torch.zeros_like(
            training_output.diagnostics["clean_residual"][invalid]
        ),
    )
    assert torch.equal(
        training_output.diagnostics["source_residual"][invalid],
        torch.zeros_like(
            training_output.diagnostics["source_residual"][invalid]
        ),
    )
    seed = 37
    low_level = model.refiner.sample(
        conditioning,
        generator=torch.Generator().manual_seed(seed),
        valid_mask=training_output.valid_mask,
    )
    target_space = model.refiner.sample_target_space(
        conditioning,
        generator=torch.Generator().manual_seed(seed),
        valid_mask=training_output.valid_mask,
    )
    expected = model.refiner.denormalize_residual(low_level)
    expected = model.refiner.apply_valid_mask(
        expected, training_output.valid_mask
    )
    assert torch.equal(target_space, expected)
    assert torch.equal(
        target_space[invalid], torch.zeros_like(target_space[invalid])
    )

    payload = build_refinement_checkpoint(model, phase1_fingerprint="fixture")
    assert payload["residual_normalization"] == metadata
    clone = build_two_phase_model(TinyPhase1(), config)
    clone.initialize_from_batch(batch_a)
    load_refinement_state_dict(clone, payload)
    assert clone.refiner.residual_normalization_metadata() == metadata


def test_selected_phase1_unfreeze_is_explicit_and_rest_stays_frozen():
    batch = make_batch(batch_size=1, height=8, width=8, with_static=False)
    model = build_two_phase_model(
        TinyPhase1(),
        _tiny_config("flow_matching_unet", trainable_phase1_patterns=["head.*"]),
    )
    model.initialize_from_batch(batch)
    assert model.phase1_frozen is False
    assert model.phase1_partially_trainable
    assert all(param.requires_grad for param in model.phase1.head.parameters())
    assert all(not param.requires_grad for param in model.phase1.body.parameters())
    model.train()
    assert model.phase1.training is False


@pytest.mark.parametrize(
    ("kind", "state_name"),
    (
        ("diffusion_unet", "noised_residual"),
        ("flow_matching_unet", "interpolated_residual"),
    ),
)
def test_invalid_target_cells_are_zero_in_stochastic_process_state(
    kind, state_name
):
    batch = make_batch(
        batch_size=1,
        height=8,
        width=8,
        seed=11,
        with_static=False,
        nan_fraction=0.25,
    )
    model = build_two_phase_model(TinyPhase1(), _tiny_config(kind))
    model.initialize_from_batch(batch)
    output = model.training_step(
        batch, generator=torch.Generator().manual_seed(9)
    )
    invalid = ~output.valid_mask
    state = output.diagnostics[state_name]
    assert torch.equal(state[invalid], torch.zeros_like(state[invalid]))


def test_nondivisible_pad_patchify_unpatchify_crop_is_exact():
    tensor = torch.randn(2, 3, 7, 11)
    padded = torch.nn.functional.pad(tensor, (0, 1, 0, 1))
    tokens, grid_h, grid_w = patchify_2d(padded, 2, 3)
    reconstructed = unpatchify_2d(tokens, 3, grid_h, grid_w, 2, 3)
    assert torch.equal(reconstructed[..., :7, :11], tensor)


@pytest.mark.parametrize(
    "filename,expected",
    (
        ("NARR_PRISM_diffusion_unet.yaml", "diffusion_unet"),
        ("NARR_PRISM_diffusion_transformer.yaml", "diffusion_transformer"),
        ("NARR_PRISM_flow_matching_unet.yaml", "flow_matching_unet"),
        ("NARR_PRISM_flow_matching_transformer.yaml", "flow_matching_transformer"),
    ),
)
def test_real_narr_refinement_yamls_parse_with_explicit_variable_order(filename, expected):
    root = Path(__file__).resolve().parents[1]
    config = get_config(str(root / "examples" / "NARR_PRISM" / filename))
    resolved = resolve_refinement_config(config)
    assert resolved.type == expected
    assert list(config.data.output_vars) == ["ppt", "tmax", "tmin"]
    assert list(config.data.input_vars[:16]) == [
        "shum_500", "shum_700", "shum_850",
        "uwnd_500", "uwnd_700", "uwnd_850",
        "vwnd_500", "vwnd_700", "vwnd_850",
        "air_500", "air_700", "air_850",
        "hgt_500", "hgt_700", "hgt_850", "elev",
    ]


@pytest.mark.parametrize("kind", REFINERS)
def test_each_head_auxiliary_clean_loss_has_finite_backward(kind):
    batch = make_batch(batch_size=1, height=8, width=10, with_static=False)
    model = build_two_phase_model(
        TinyPhase1(),
        _tiny_config(
            kind,
            auxiliary_loss={
                "reconstruction_weight": 0.1,
                "bias_weight": 0.02,
                "gradient_weight": 0.03,
                "laplacian_weight": 0.01,
                "multiscale_weight": 0.01,
                "tail_weight": 0.01,
                "tail_threshold": 0.5,
                "variable_weights": [1.0, 1.0, 1.0],
            },
        ),
    )
    model.initialize_from_batch(batch)
    output = model.training_step(batch, generator=torch.Generator().manual_seed(4))
    assert "stochastic_objective" in output.losses
    assert "clean_gradient_loss" in output.losses
    assert "estimated_clean_residual" in output.diagnostics
    output.losses["loss"].backward()
    gradients = [p.grad for p in model.refiner.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_all_four_wrappers_load_same_model_prefixed_phase1_state():
    source = TinyPhase1()
    checkpoint = {"state_dict": {f"model.{key}": value.clone() for key, value in source.state_dict().items()}}
    batch = make_batch(batch_size=1, height=8, width=8, with_static=False)
    for kind in REFINERS:
        model = build_two_phase_model(TinyPhase1(), _tiny_config(kind))
        report = load_phase1_state_dict(model, checkpoint)
        assert report.loaded == len(source.state_dict())
        assert not report.unexpected
        model.initialize_from_batch(batch)
        assert all(not parameter.requires_grad for parameter in model.phase1.parameters())


def test_diffusion_oracle_denoiser_recovers_clean_residual():
    batch = make_batch(batch_size=1, height=6, width=8, with_static=False)
    model = build_two_phase_model(TinyPhase1(), _tiny_config("diffusion_unet"))
    model.initialize_from_batch(batch)
    _, baseline, features = model.run_phase1(batch)
    conditioning = model.build_conditioning(batch, baseline, features)
    expected = torch.linspace(-0.5, 0.5, 6 * 8).reshape(1, 1, 6, 8).repeat(1, 3, 1, 1)
    schedule = model.refiner.schedule

    class Oracle(torch.nn.Module):
        def forward(self, noisy, conditioning, timestep):
            indices = timestep.long()
            shape = (-1,) + (1,) * (noisy.ndim - 1)
            sqrt_alpha = schedule.sqrt_alphas_cumprod[indices].reshape(shape).to(noisy)
            sqrt_noise = schedule.sqrt_one_minus_alphas_cumprod[indices].reshape(shape).to(noisy)
            return (noisy - sqrt_alpha * expected.to(noisy)) / sqrt_noise.clamp(min=1e-8)

    model.refiner.net = Oracle()
    sampled = model.refiner.sample(
        conditioning, generator=torch.Generator().manual_seed(3), num_steps=4
    )
    assert torch.allclose(sampled, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("solver", ("euler", "midpoint", "heun"))
def test_flow_integrates_forward_from_source_to_target(solver):
    config = _tiny_config(
        "flow_matching_unet",
        flow_matching={
            "integration_steps": 5,
            "solver": solver,
            "stochastic_initialization": False,
        },
    )
    batch = make_batch(batch_size=1, height=6, width=8, with_static=False)
    model = build_two_phase_model(TinyPhase1(), config)
    model.initialize_from_batch(batch)
    _, baseline, features = model.run_phase1(batch)
    conditioning = model.build_conditioning(batch, baseline, features)

    class ConstantVelocity(torch.nn.Module):
        def forward(self, state, conditioning, timestep):
            return torch.full_like(state, 0.75)

    model.refiner.net = ConstantVelocity()
    endpoint = model.refiner.sample(conditioning)
    assert torch.allclose(endpoint, torch.full_like(endpoint, 0.75), atol=1e-6)


@pytest.mark.parametrize("kind", REFINERS)
def test_each_refinement_head_overfits_one_fixed_residual_example(kind):
    """A functional head must substantially reduce a fixed stochastic objective."""
    torch.manual_seed(0)
    batch = make_batch(batch_size=1, height=6, width=6, with_static=False, seed=4)
    model = build_two_phase_model(TinyPhase1(), _tiny_config(kind))
    model.initialize_from_batch(batch)
    optimizer = torch.optim.Adam(model.refiner.parameters(), lr=2.0e-2)
    losses = []
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        # Resetting this generator makes the diffusion timestep/noise or flow
        # path point identical: this is a true tiny-example capacity check, not
        # a claim about generalization.
        output = model.training_step(
            batch, generator=torch.Generator().manual_seed(123)
        )
        loss = output.losses["loss"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < 0.5 * losses[0], (kind, losses[0], losses[-1])


def test_partial_gradient_accumulation_is_flushed(tmp_path):
    batch = make_batch(batch_size=1, height=6, width=6, with_static=False)
    model = build_two_phase_model(TinyPhase1(), _tiny_config("flow_matching_unet"))
    model.initialize_from_batch(batch)
    before = {key: value.clone() for key, value in model.refiner.state_dict().items()}
    optimizer = torch.optim.AdamW(model.refiner.parameters(), lr=1.0e-3)
    trainer = RefinementTrainer(
        model,
        optimizer,
        checkpoint_dir=str(tmp_path),
        gradient_accumulation_steps=8,
        logger=lambda _message: None,
        seed=1,
    )
    trainer.train_one_epoch([batch], limit_steps=1)
    assert trainer.state.global_step == 1
    assert any(
        not torch.equal(before[key], value)
        for key, value in model.refiner.state_dict().items()
        if value.dtype.is_floating_point
    )
