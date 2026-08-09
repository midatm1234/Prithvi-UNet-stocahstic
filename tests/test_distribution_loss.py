from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from granitewxc.models.loss import CompositePredictandLoss, build_loss_fn, rmse_loss


def _batch(pred: torch.Tensor, target: torch.Tensor):
    del pred
    return {"y": target}


def test_build_loss_fn_defaults_to_rmse_when_block_missing():
    config = SimpleNamespace(loss=None)
    loss_fn = build_loss_fn(config, output_vars=["pr", "tasmax"])
    assert loss_fn is rmse_loss


def test_composite_moment_loss_adds_terms():
    config = SimpleNamespace(
        loss={
            "base": "rmse",
            "predictands": {
                "tasmax": {
                    "distribution_loss": {
                        "enabled": True,
                        "method": "moment",
                        "weight": 0.1,
                        "use_mean": True,
                        "use_std": True,
                    }
                }
            },
        }
    )
    loss_fn = build_loss_fn(config, output_vars=["pr", "tasmax"])
    assert isinstance(loss_fn, CompositePredictandLoss)

    pred = torch.tensor([[[[0.0, 1.0]] , [[2.0, 3.0]]]], dtype=torch.float32)
    target = torch.tensor([[[[0.0, 0.5]], [[2.5, 2.5]]]], dtype=torch.float32)
    loss = loss_fn(pred, _batch(pred, target))

    assert torch.isfinite(loss)
    terms = loss_fn.get_last_terms()
    assert "base.rmse" in terms
    assert "tasmax.distribution.moment" in terms
    assert "tasmax.distribution.moment.weighted" in terms


def test_bernoulli_gamma_uses_physical_threshold_and_nested_parameters():
    config = SimpleNamespace(
        loss={},
        precip_model="bernoulli_gamma",
        model=SimpleNamespace(
            bernoulli_gamma={
                "wet_threshold": 0.2,
                "lambda_occurrence": 1.7,
                "lambda_positive_amount": 0.6,
                "min_mu": 2e-4,
                "min_phi": 3e-4,
            }
        ),
    )

    loss_fn = build_loss_fn(config, output_vars=["pr", "tasmax"])

    assert isinstance(loss_fn, CompositePredictandLoss)
    assert loss_fn._bg_loss is not None
    bg_cfg = loss_fn._bg_loss.cfg
    # batch['y'] is physical mm/day, so this must not be divided by a p95 scaler.
    assert bg_cfg.wet_threshold == 0.2
    assert bg_cfg.lambda_occurrence == 1.7
    assert bg_cfg.lambda_positive_amount == 0.6
    assert bg_cfg.min_mu == 2e-4
    assert bg_cfg.min_phi == 3e-4

    description = loss_fn.describe()["bernoulli_gamma"]
    assert description == {
        "enabled": True,
        "precip_index": 0,
        "wet_threshold": 0.2,
        "lambda_occurrence": 1.7,
        "lambda_positive_amount": 0.6,
        "min_mu": 2e-4,
        "min_phi": 3e-4,
    }


def test_residual_diffusion_keeps_bernoulli_gamma_supervision_in_joint_loss():
    config = SimpleNamespace(
        precip_model="bernoulli_gamma",
        model=SimpleNamespace(
            head_type="diffusion",
            diffusion={"residual_diffusion": True},
            bernoulli_gamma={
                "wet_threshold": 0.2,
                "lambda_occurrence": 1.5,
                "lambda_positive_amount": 0.75,
            },
        ),
        loss={
            "deterministic_weight": 1.0,
            "diffusion": {"weight": 0.4},
        },
    )
    loss_fn = build_loss_fn(config, output_vars=["pr", "tasmax"])
    assert type(loss_fn).__name__ == "JointResidualDiffusionLoss"

    baseline = torch.tensor(
        [[[[0.2, 0.5]], [[295.0, 296.0]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    diffusion_loss = torch.tensor(0.25, requires_grad=True)
    logit_wet = torch.zeros(1, 1, 1, 2, requires_grad=True)
    log_mu = torch.zeros(1, 1, 1, 2, requires_grad=True)
    log_phi = torch.zeros(1, 1, 1, 2, requires_grad=True)
    batch = {
        "y": torch.tensor([[[[0.0, 2.0]], [[294.0, 297.0]]]]),
        "__bernoulli_gamma_aux": {
            "logit_wet": logit_wet,
            "log_mu": log_mu,
            "log_phi": log_phi,
        },
    }
    total = loss_fn(
        {
            "baseline_prediction": baseline,
            "diffusion_loss": diffusion_loss,
        },
        batch,
    )
    total.backward()

    terms = loss_fn.get_last_terms()
    assert torch.isfinite(total)
    assert "deterministic.bg.occurrence_bce" in terms
    assert "deterministic.bg.positive_gamma_nll" in terms
    assert terms["diffusion.weighted"] == pytest.approx(0.1)
    assert baseline.grad is not None
    assert float(diffusion_loss.grad.item()) == pytest.approx(0.4)
    assert logit_wet.grad is not None
    assert log_mu.grad is not None
    assert log_phi.grad is not None
