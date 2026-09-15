"""Focused flow-matching regressions for mixed precipitation/temperature heads."""

from __future__ import annotations

import warnings

import pytest
import torch

from granitewxc.refinement import build_two_phase_model
from granitewxc.refinement.base import build_refiner
from granitewxc.refinement.config import resolve_refinement_config
from granitewxc.refinement.flow_matching import flow_interpolate, flow_to_clean, integrate_flow
from refinement_fixtures import TinyPhase1, make_batch


@pytest.mark.parametrize("solver", ("euler", "midpoint", "heun"))
def test_oracle_regularized_flow_integrates_forward_to_clean_target(solver):
    source = torch.randn(2, 2, 5, 7, generator=torch.Generator().manual_seed(17))
    target = torch.randn(2, 2, 5, 7, generator=torch.Generator().manual_seed(19))
    sigma_min = 0.07
    _, oracle_velocity = flow_interpolate(
        source, target, torch.zeros(source.shape[0]), sigma_min
    )

    endpoint = integrate_flow(
        source,
        lambda state, time: oracle_velocity.to(state),
        steps=9,
        solver=solver,
    )
    clean = flow_to_clean(
        endpoint,
        oracle_velocity,
        torch.ones(source.shape[0]),
        sigma_min,
    )
    torch.testing.assert_close(clean, target, atol=2e-6, rtol=2e-6)


def test_opt_in_mean_path_loss_and_flow_states_are_explicit():
    config = resolve_refinement_config(
        {
            "refinement": {
                "type": "flow_matching_unet",
                "loss": "mse",
                "reconstruction_loss_weight": 0.0,
                "multiscale_loss_weight": 0.0,
                "gradient_loss_weight": 0.0,
                "mean_bias_loss_weight": 0.0,
                "residual_normalization": {"method": "identity"},
                "flow_matching": {
                    "integration_steps": 3,
                    "mean_path_loss_weight": 0.25,
                },
                "unet": {
                    "hidden_channels": 8,
                    "num_levels": 2,
                    "time_embedding_dim": 16,
                    "bottleneck_attention": False,
                },
            }
        }
    )
    model = build_refiner(config, residual_channels=2, cond_channels=3)
    target = torch.randn(2, 2, 8, 8, generator=torch.Generator().manual_seed(3))
    conditioning = torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(5))
    losses = model.training_loss(
        target,
        conditioning,
        generator=torch.Generator().manual_seed(7),
    )

    expected = losses["process_loss"] + 0.25 * losses["mean_path_loss"]
    torch.testing.assert_close(losses["loss"], expected)
    for name in (
        "source_state",
        "interpolated_state",
        "target_velocity",
        "zero_source_state",
        "zero_source_velocity_prediction",
        "zero_source_target_velocity",
    ):
        assert losses[name].shape == target.shape
        assert losses[name].requires_grad is False
    assert float(losses["mean_path_loss"].detach()) > 0.0
    losses["loss"].backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.parameters()
    )


def test_mean_preserving_nonnegative_ensemble_tracks_memberwise_jensen_shift():
    batch = make_batch(batch_size=1, out_channels=3, height=8, width=8)
    config = {
        "refinement": {
            "type": "flow_matching_unet",
            "ensemble_size": 4,
            "nonnegative_ensemble_strategy": "mean_preserving",
            "residual_normalization": {"method": "identity"},
            "flow_matching": {"integration_steps": 2},
            "unet": {"hidden_channels": 8, "num_levels": 2},
        }
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model = build_two_phase_model(TinyPhase1(), config)
    model.initialize_from_batch(batch)
    model.eval()
    base, _, _ = model.run_phase1(batch)

    desired_unbounded = torch.empty(4, 3, 8, 8)
    desired_unbounded[:, 0, :, :4] = torch.tensor([-1.0, 1.0, -1.0, 1.0]).view(
        4, 1, 1
    )
    desired_unbounded[:, 0, :, 4:] = torch.tensor([-1.0, 3.0, -1.0, 3.0]).view(
        4, 1, 1
    )
    desired_unbounded[:, 1] = torch.tensor([-1.0, 3.0, -1.0, 3.0]).view(4, 1, 1)
    desired_unbounded[:, 2] = base[0, 2]
    prescribed_corrections = desired_unbounded - base.expand(4, -1, -1, -1)

    def prescribed_sample(conditioning, **kwargs):
        del kwargs
        assert conditioning.shape[0] == 4
        return prescribed_corrections.to(conditioning)

    model.refiner.sample = prescribed_sample
    output = model.predict(batch, ensemble_size=4, seed=9)

    torch.testing.assert_close(output.members_unbounded[0], desired_unbounded)
    # At the dry half, symmetric unbounded members have mean zero but ordinary
    # independent clipping induces +0.5. At the wet half it changes 1.0 to 1.5.
    # Both shifts remain explicit in the diagnostic fields.
    expected_unbounded_mean = torch.zeros(1, 8, 8)
    expected_unbounded_mean[..., 4:] = 1.0
    expected_memberwise_mean = expected_unbounded_mean + 0.5
    torch.testing.assert_close(output.unbounded_ensemble_mean[:, 0], expected_unbounded_mean)
    torch.testing.assert_close(output.memberwise_clipped_mean[:, 0], expected_memberwise_mean)
    torch.testing.assert_close(
        output.memberwise_clipping_mean_shift[:, 0], torch.full((1, 8, 8), 0.5)
    )
    # The selected physical members are nonnegative and mean-preserving. Exact
    # zero mean necessarily has zero non-negative spread, while the positive-mean
    # half retains nonzero stochastic spread.
    assert bool((output.members[:, :, 0] >= 0.0).all())
    torch.testing.assert_close(
        output.ensemble_mean[:, 0], expected_unbounded_mean, atol=1e-6, rtol=0.0
    )
    assert float(output.ensemble_spread[:, 0, :, :4].max()) < 1.0e-6
    assert float(output.ensemble_spread[:, 0, :, 4:].mean()) > 0.0
    # Affine tas-like channels are not touched by precipitation constraints.
    torch.testing.assert_close(output.members[0, :, 1], desired_unbounded[:, 1])
    torch.testing.assert_close(
        output.memberwise_clipping_mean_shift[:, 1], torch.zeros(1, 8, 8)
    )
    torch.testing.assert_close(output.ensemble_mean, output.members.mean(dim=1))


def test_raw_flow_unet_learns_intermittent_pr_and_smooth_tas_without_a_gate():
    config = resolve_refinement_config(
        {
            "refinement": {
                "type": "flow_matching_unet",
                "loss": "mse",
                "reconstruction_loss_weight": 0.5,
                "multiscale_loss_weight": 0.1,
                "gradient_loss_weight": 0.05,
                "mean_bias_loss_weight": 0.05,
                "residual_normalization": {"method": "identity"},
                "flow_matching": {
                    "integration_steps": 12,
                    "solver": "heun",
                    "time_sampling": "uniform",
                    "mean_path_loss_weight": 0.25,
                },
                "unet": {
                    "hidden_channels": 8,
                    "num_levels": 2,
                    "time_embedding_dim": 16,
                    "bottleneck_attention": False,
                },
            }
        }
    )
    torch.manual_seed(4)
    model = build_refiner(config, residual_channels=2, cond_channels=4)
    rows = torch.linspace(-1.0, 1.0, 8).view(1, 1, 8, 1)
    cols = torch.linspace(-1.0, 1.0, 8).view(1, 1, 1, 8)
    precipitation = torch.relu(
        0.9 * torch.exp(-6.0 * ((rows + 0.2) ** 2 + (cols - 0.2) ** 2))
        + 0.3 * torch.sin(2.0 * torch.pi * cols) * torch.cos(torch.pi * rows)
        - 0.15
    )
    temperature = (
        0.7 * rows
        - 0.3 * cols
        + 0.2 * torch.cos(torch.pi * rows) * torch.sin(torch.pi * cols)
    )
    target = torch.cat((precipitation, temperature), dim=1)
    target = (target - target.mean(dim=(-2, -1), keepdim=True)) / target.std(
        dim=(-2, -1), keepdim=True
    )
    target = target.repeat(2, 1, 1, 1)
    conditioning = torch.cat(
        (target, rows.expand(2, 1, 8, 8), cols.expand(2, 1, 8, 8)), dim=1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-3)
    generator = torch.Generator().manual_seed(44)
    for _ in range(220):
        optimizer.zero_grad(set_to_none=True)
        losses = model.training_loss(target, conditioning, generator=generator)
        losses["loss"].backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        samples = [
            model.sample(conditioning[:1], generator=torch.Generator().manual_seed(seed))
            for seed in (71, 72)
        ]
    assert not hasattr(model, "correction_gate")
    for sample in samples:
        for channel, minimum_correlation in ((0, 0.70), (1, 0.82)):
            predicted = sample[0, channel].flatten()
            truth = target[0, channel].flatten()
            correlation = torch.corrcoef(torch.stack((predicted, truth)))[0, 1]
            rmse_ratio = (predicted - truth).square().mean().sqrt() / truth.square().mean().sqrt()
            assert float(correlation) > minimum_correlation
            assert float(rmse_ratio) < 0.85
