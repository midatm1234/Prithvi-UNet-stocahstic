from __future__ import annotations

from types import SimpleNamespace

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


def test_nan_only_targets_have_zero_finite_gradients():
    pred = torch.randn((1, 2, 4, 4), dtype=torch.float32, requires_grad=True)
    target = torch.full_like(pred, float("nan"))

    loss = rmse_loss(pred, _batch(pred, target))
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert torch.count_nonzero(pred.grad) == 0

    pred = torch.randn((1, 2, 4, 4), dtype=torch.float32, requires_grad=True)
    config = SimpleNamespace(loss={"base": "rmse"})
    loss_fn = build_loss_fn(config, output_vars=["pr", "tasmax"])
    loss = loss_fn(pred, _batch(pred, target))
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert torch.count_nonzero(pred.grad) == 0
