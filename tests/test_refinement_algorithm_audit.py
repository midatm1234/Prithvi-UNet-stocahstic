"""Independent process/geometry regressions for all four refinement options.

These are mathematical and synthetic learning checks, not climate validation.
"""

from __future__ import annotations

import pytest
import torch

from granitewxc.refinement.diffusion import ddim_reverse_step
from granitewxc.refinement.flow_matching import integrate_flow
from granitewxc.refinement.schedules import DiffusionSchedule


def test_half_precision_terminal_ddim_does_not_round_alpha_to_one():
    schedule = DiffusionSchedule()
    clean = torch.ones(1, 1, 2, 2, dtype=torch.float16)
    timestep = torch.tensor([0])
    noisy = schedule.add_noise(clean, clean, timestep)
    previous, recovered, _ = ddim_reverse_step(
        schedule, "sample", clean, noisy, timestep, None, eta=1.0
    )
    assert bool(torch.isfinite(previous).all())
    torch.testing.assert_close(previous, recovered)


@pytest.mark.parametrize("solver", ("euler", "midpoint", "heun"))
def test_half_precision_flow_accumulates_small_steps_in_float32(solver):
    initial = torch.full((1, 1, 2, 2), 100.0, dtype=torch.float16)
    endpoint = integrate_flow(
        initial, lambda state, time: torch.ones_like(state), steps=100, solver=solver
    )
    torch.testing.assert_close(
        endpoint.float(), torch.full_like(endpoint.float(), 101.0), atol=5e-4, rtol=0
    )


from granitewxc.refinement.backbones import (
    ConditionalResidualUNet,
    SpatialResidualTransformer,
    _pooled_centers,
    _resize_from_centers,
    patchify_2d,
    unpatchify_2d,
)
from granitewxc.refinement.base import build_refiner
from granitewxc.refinement.config import resolve_refinement_config
from granitewxc.refinement.flow_matching import flow_interpolate, flow_to_clean

HEADS = (
    "diffusion_unet", "diffusion_transformer",
    "flow_matching_unet", "flow_matching_transformer",
)


@pytest.fixture(autouse=True)
def one_cpu_thread_for_small_fields():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config(head, alignment="coordinates", eta=0.0):
    raw = {"refinement": {
        "type": head, "loss": "mse",
        "residual_normalization": {"method": "identity"},
        "reconstruction_loss_weight": 0.25,
        "multiscale_loss_weight": 0.0,
        "gradient_loss_weight": 0.0,
        "mean_bias_loss_weight": 0.0,
        "diffusion": {
            "training_timesteps": 32, "inference_steps": 4,
            "prediction_type": "sample", "eta": eta,
        },
        "flow_matching": {
            "integration_steps": 4, "solver": "heun", "sigma_min": 0.05,
        },
        "unet": {
            "hidden_channels": 8, "num_levels": 2,
            "time_embedding_dim": 16, "bottleneck_attention": True,
            "spatial_alignment": alignment,
        },
        "transformer": {
            "embedding_dim": 16, "num_heads": 2, "num_blocks": 1,
            "patch_size": [3, 4], "mlp_ratio": 2.0,
            "spatial_alignment": alignment,
        },
    }}
    raw["refinement"].pop("flow_matching" if head.startswith("diffusion") else "diffusion")
    return resolve_refinement_config(raw)


@pytest.mark.parametrize("prediction_type", ("sample", "epsilon", "velocity"))
def test_all_diffusion_parameterizations_recover_signed_clean_field(prediction_type):
    schedule = DiffusionSchedule(32)
    gen = torch.Generator().manual_seed(9)
    clean = torch.randn(3, 3, 7, 9, generator=gen)
    noise = torch.randn(3, 3, 7, 9, generator=gen)
    time = torch.tensor([0, 11, 30])
    noisy = schedule.add_noise(clean, noise, time)
    target = schedule.training_target(prediction_type, clean, noise, time)
    recovered = schedule.to_clean(prediction_type, target, noisy, time)
    recovered_noise = schedule.to_epsilon(prediction_type, target, noisy, time)
    torch.testing.assert_close(recovered, clean, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(recovered_noise, noise, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("sigma", (0.0, 0.05, 0.5))
def test_regularized_flow_round_trip_at_all_path_times(sigma):
    gen = torch.Generator().manual_seed(31)
    source = torch.randn(5, 3, 7, 9, generator=gen)
    target = torch.randn(5, 3, 7, 9, generator=gen)
    times = torch.linspace(0, 1, 5)
    state, velocity = flow_interpolate(source, target, times, sigma)
    torch.testing.assert_close(flow_to_clean(state, velocity, times, sigma), target)


def test_patch_round_trip_preserves_rows_columns_and_channel_order():
    field = torch.arange(2 * 3 * 6 * 8).reshape(2, 3, 6, 8)
    tokens, rows, columns = patchify_2d(field, 3, 4)
    torch.testing.assert_close(
        unpatchify_2d(tokens, 3, rows, columns, 3, 4), field
    )


def test_transformer_interpolation_uses_actual_patch_centers():
    cy, cx = torch.tensor([0.0, 3.0, 6.0]), torch.tensor([0.0, 4.0, 8.0])
    tokens = (2 * cy[:, None] - cx[None, :]).reshape(1, 1, 3, 3)
    result = _resize_from_centers(tokens, cy, cx, (7, 9))
    expected = (
        2 * torch.arange(7)[:, None] - torch.arange(9)[None, :]
    ).reshape(1, 1, 7, 9).float()
    torch.testing.assert_close(result, expected, atol=2e-6, rtol=0)
    legacy = torch.nn.functional.interpolate(tokens, size=(9, 12), mode="bilinear", align_corners=False)[..., :7, :9]
    assert float((legacy - expected).abs().mean()) > 0.5


def test_odd_pool_interpolation_uses_shortened_trailing_cell_centers():
    field = (
        2 * torch.arange(7)[:, None] - torch.arange(9)[None, :]
    ).reshape(1, 1, 7, 9).float()
    pooled = torch.nn.functional.avg_pool2d(field, 2, ceil_mode=True)
    cy, cx = _pooled_centers(7, field.device), _pooled_centers(9, field.device)
    result = _resize_from_centers(pooled, cy, cx, (7, 9))
    torch.testing.assert_close(result[..., 1:, 1:], field[..., 1:, 1:], atol=2e-6, rtol=0)
    legacy = torch.nn.functional.interpolate(pooled, size=(7, 9), mode="bilinear", align_corners=False)
    assert float((legacy[..., 1:, 1:] - field[..., 1:, 1:]).abs().mean()) > 0.1


@pytest.mark.parametrize("head", HEADS)
def test_each_head_reaches_conditioning_time_attention_and_updates(head):
    torch.manual_seed(12)
    model = build_refiner(_config(head), residual_channels=3, cond_channels=4)
    assert isinstance(
        model.net, SpatialResidualTransformer if head.endswith("transformer") else ConditionalResidualUNet
    )
    target = torch.randn(2, 3, 9, 11)
    conditioning = torch.randn(2, 4, 9, 11)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    before = model.net.out_proj.weight.detach().clone()
    for step in range(7):
        optimizer.zero_grad(set_to_none=True)
        result = model.training_loss(
            target, conditioning, generator=torch.Generator().manual_seed(step + 20)
        )
        assert bool(torch.isfinite(result["loss"]))
        result["loss"].backward()
        assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        optimizer.step()
    assert not torch.equal(model.net.out_proj.weight, before)
    for pathway in ("time_embed", "cond_proj" if head.endswith("transformer") else "film"):
        gradients = [p.grad for n, p in model.net.named_parameters() if pathway in n and p.grad is not None]
        assert gradients and sum(float(g.abs().sum()) for g in gradients) > 0
    qkv_gradients = [p.grad for n, p in model.net.named_parameters() if "qkv" in n and p.grad is not None]
    assert sum(float(g.abs().sum()) for g in qkv_gradients) > 0

    model.eval()
    state = target[:1]
    time = torch.tensor([500.0])
    reference = model.net(state, conditioning[:1], time)
    assert float((reference - model.net(state, -conditioning[:1], time)).abs().mean().detach()) > 1e-6
    assert float((reference - model.net(state, conditioning[:1], time + 200)).abs().mean().detach()) > 1e-6
    other_sample = model.net(
        torch.cat((state, state * 9)), torch.cat((conditioning[:1], conditioning[1:] * 8)),
        time.expand(2),
    )
    torch.testing.assert_close(reference, other_sample[:1], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("head", HEADS)
def test_shared_source_and_noise_snapshots_do_not_change_sampling(head):
    torch.manual_seed(71)
    diffusion = head.startswith("diffusion")
    model = build_refiner(_config(head, eta=0.5 if diffusion else 0), residual_channels=3, cond_channels=4).eval()
    conditioning = torch.randn(2, 4, 9, 11)
    source = torch.randn(2, 3, 9, 11)
    noise = [torch.randn_like(source) for _ in range(3)]
    kwargs = {"initial_state": source}
    if diffusion:
        kwargs["step_noises"] = noise
    stages = []

    def callback(stage, index, time, state):
        stages.append((stage, index, time.clone(), state.clone()))
        # Snapshot mutation must not alter process state.
        state.fill_(9000)

    reference = model.sample(conditioning, **kwargs)
    traced = model.sample(conditioning, trajectory_callback=callback, **kwargs)
    torch.testing.assert_close(traced, reference)
    assert stages[0][0] == "initial"
    torch.testing.assert_close(stages[0][3], source)
    assert stages[-1][0] == "final_normalized_residual"
    torch.testing.assert_close(stages[-1][3], reference)
    assert all(not state.requires_grad for _, _, _, state in stages)


@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("alignment", ("legacy", "coordinates"))
def test_each_head_preserves_odd_domain_shape_and_constant_resize(head, alignment):
    torch.manual_seed(41)
    model = build_refiner(_config(head, alignment), residual_channels=3, cond_channels=4).eval()
    out = model.net(torch.ones(2, 3, 7, 11), torch.ones(2, 4, 7, 11), torch.tensor([0.0, 1000.0]))
    assert out.shape == (2, 3, 7, 11)
    assert bool(torch.isfinite(out).all())
    constant = torch.ones(2, 3, 3, 4)
    resized = _resize_from_centers(
        constant, torch.tensor([0, 3, 6]), torch.tensor([0, 3, 6, 9]), (7, 11)
    )
    torch.testing.assert_close(resized, torch.ones(2, 3, 7, 11))

