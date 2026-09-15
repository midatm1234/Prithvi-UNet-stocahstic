"""Regression tests for truth-referenced residual structure losses."""

from __future__ import annotations

import torch

from granitewxc.refinement.base import residual_structure_losses


def test_true_high_frequency_residual_is_not_penalized_as_roughness():
    """Exact fine-scale truth must score zero; these are not smoothing priors."""
    rows = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1)
    cols = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8)
    checkerboard = ((rows + cols).remainder(2.0) * 2.0 - 1.0) * 3.0
    target = checkerboard + 0.2 * rows + 0.1 * cols

    exact = residual_structure_losses(target.clone(), target, None)

    for name, value in exact.items():
        assert value.item() == 0.0, f"{name} penalized a perfectly matched residual"


def test_structure_losses_compare_spatial_pattern_and_channel_mean_to_truth():
    rows = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1)
    cols = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8)
    target = rows.square() + 0.5 * cols

    shifted = torch.roll(target, shifts=2, dims=-1)
    shifted_losses = residual_structure_losses(shifted, target, None)
    assert shifted_losses["multiscale_loss"].item() > 0.0
    assert shifted_losses["gradient_loss"].item() > 0.0
    # A spatial permutation retains the exact domain mean. The bias term must
    # therefore remain zero instead of acting as an indiscriminate roughness loss.
    assert shifted_losses["mean_bias_loss"].item() == 0.0

    biased_losses = residual_structure_losses(target + 2.0, target, None)
    assert biased_losses["mean_bias_loss"].item() > 0.0


def test_structure_losses_exclude_invalid_cells_at_every_scale():
    target = torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(4))
    prediction = target.clone()
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[..., :4, :4] = False
    prediction[..., :4, :4] = 1.0e8

    losses = residual_structure_losses(prediction, target, valid)

    for name, value in losses.items():
        assert value.item() == 0.0, f"{name} leaked invalid target cells"
