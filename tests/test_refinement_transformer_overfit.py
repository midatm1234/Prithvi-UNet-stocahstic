"""Tiny-subset overfit regressions for coherent Transformer residuals."""

from __future__ import annotations

import pytest
import torch

from granitewxc.refinement.base import build_refiner
from granitewxc.refinement.config import resolve_refinement_config


@pytest.mark.parametrize(
    "refiner_type",
    ("diffusion_transformer", "flow_matching_transformer"),
)
def test_transformer_overfits_one_coherent_residual(refiner_type):
    refinement = {
        "enabled": True,
        "type": refiner_type,
        "loss": "mse",
        "reconstruction_loss_weight": 1.0,
        "multiscale_loss_weight": 0.2,
        "gradient_loss_weight": 0.1,
        "mean_bias_loss_weight": 0.1,
        "residual_normalization": {"method": "identity"},
        "transformer": {
            "embedding_dim": 16,
            "num_heads": 4,
            "num_blocks": 1,
            "patch_size": 2,
            "dropout": 0.0,
        },
    }
    if refiner_type == "diffusion_transformer":
        refinement["diffusion"] = {
            "training_timesteps": 20,
            "inference_steps": 20,
            "prediction_type": "sample",
        }
    else:
        refinement["flow_matching"] = {
            "integration_steps": 20,
            "solver": "heun",
            "stochastic_initialization": True,
        }
    config = resolve_refinement_config({"refinement": refinement})

    torch.manual_seed(1)
    model = build_refiner(config, residual_channels=1, cond_channels=2)
    assert model is not None
    rows = torch.linspace(-1.0, 1.0, 8).view(1, 1, 8, 1)
    cols = torch.linspace(-1.0, 1.0, 8).view(1, 1, 1, 8)
    target = (
        0.5 * torch.sin(2.0 * torch.pi * cols)
        + 0.3 * torch.cos(torch.pi * rows)
        + 0.2 * rows * cols
    )
    conditioning = torch.cat((target, cols.expand_as(target)), dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-3)

    initial_loss = None
    training_generator = torch.Generator().manual_seed(123)
    # Stochastic flow matching must learn to transport more than one source
    # noise realization; it needs a few more tiny-example updates than direct
    # clean-sample diffusion prediction for a stable full-integrator check.
    updates = 300 if refiner_type == "diffusion_transformer" else 400
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        losses = model.training_loss(
            target,
            conditioning,
            generator=training_generator,
        )
        if initial_loss is None:
            initial_loss = float(losses["loss"].detach())
        losses["loss"].backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final = model.training_loss(
            target,
            conditioning,
            generator=torch.Generator().manual_seed(99),
        )
        # Exercise the complete sampler/integrator.  The raw model-space sample
        # must itself be coherent; checking only the gated correction would let
        # a near-zero gate hide an unstable or noisy stochastic process.
        raw_samples = [
            model.sample(conditioning, generator=torch.Generator().manual_seed(seed))
            for seed in (77, 78)
        ]

    target_rms = target.square().mean().sqrt()
    correlations = [
        torch.corrcoef(torch.stack((target.flatten(), sample.flatten())))[0, 1]
        for sample in raw_samples
    ]
    raw_rmses = [(sample - target).square().mean().sqrt() for sample in raw_samples]
    raw_stds = [sample.std() for sample in raw_samples]

    assert initial_loss is not None
    assert float(final["loss"]) < 0.25 * initial_loss
    assert all(bool(torch.isfinite(sample).all()) for sample in raw_samples)
    assert min(float(value) for value in correlations) > 0.85
    assert max(float(value) for value in raw_rmses) < 0.65 * float(target_rms)
    assert min(float(value) for value in raw_stds) > 0.25 * float(target.std())
    assert max(float(value) for value in raw_stds) < 2.5 * float(target.std())
    assert float(model.correction_gate.detach().abs().max()) > 0.05
