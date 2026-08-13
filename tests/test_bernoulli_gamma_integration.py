from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from granitewxc.models.cordex_finetune_model import (
    ClimateDownscaleFinetuneUNETModel,
)
from granitewxc.models.finetune_model import PatchEmbed
from granitewxc.models.loss import CompositePredictandLoss, build_loss_fn
from granitewxc.models.diffusion_loss import score_matching_loss
from granitewxc.models.diffusion_sde import VPSDE


def _config(*, head_type: str = "deterministic", residual: bool = False):
    return SimpleNamespace(
        data=SimpleNamespace(
            n_input_timestamps=1,
            use_static=False,
            output_vars=["pr", "tasmax"],
        ),
        model=SimpleNamespace(
            embed_dim=4,
            encoder_decoder_scale_per_stage=[[2], [2]],
            residual="none",
            residual_connection=False,
            downscaling_embed_dim=2,
            encoder_decoder_conv_channels=2,
            decoder_upsampling_mode="bilinear",
            output_scaler_resize_mode="bilinear",
            output_scaler_align_corners=False,
            static_embedding_scale=1.0,
            static_skip_scale=1.0,
            static_dropout_p=0.0,
            head_type=head_type,
            diffusion={
                "residual_diffusion": residual,
                "num_scales": 8,
                "num_sampling_steps": 2,
                "base_channels": 8,
                "channel_multipliers": [1],
                "num_res_blocks": 1,
                "time_embed_dim": 16,
                "projected_cond_channels": 4,
                "dropout": 0.0,
            },
        ),
        backbone_use=False,
        mask_unit_size=[4, 4],
        backbone_gradient_checkpointing=False,
        finetune_w_static=False,
        predictands={},
        precip_model="bernoulli_gamma",
        loss={},
    )


def _model(config) -> ClimateDownscaleFinetuneUNETModel:
    return ClimateDownscaleFinetuneUNETModel(
        embedding=PatchEmbed((1, 1), channels=1, embed_dim=2),
        embedding_static=None,
        backbone=nn.Identity(),
        patch_size_px_backbone=(1, 1),
        input_scalers_mu=torch.zeros(1),
        input_scalers_sigma=torch.ones(1),
        input_scalers_epsilon=0.0,
        static_input_scalers_mu=torch.empty(0),
        static_input_scalers_sigma=torch.empty(0),
        static_input_scalers_epsilon=1e-6,
        output_scalers_mu=torch.zeros(2),
        output_scalers_sigma=torch.ones(2),
        static_output_scalers_mu=torch.empty(0),
        static_output_scalers_sigma=torch.empty(0),
        n_bins=2,
        scale=[2, 2],
        kernel_size=[3, 3],
        config=config,
    )


def _batch(*, include_nan: bool = True) -> dict[str, torch.Tensor]:
    target = torch.cat(
        (torch.rand(2, 1, 16, 16), torch.randn(2, 1, 16, 16)), dim=1
    )
    if include_nan:
        target[0, :, 0, 0] = float("nan")
    return {"x": torch.randn(2, 1, 16, 16), "y": target}


def test_bernoulli_gamma_model_loss_forward_backward_is_operational():
    config = _config()
    model = _model(config)
    batch = _batch(include_nan=False)

    prediction = model(batch)
    assert prediction.shape == batch["y"].shape
    assert torch.isfinite(prediction).all()
    assert torch.all(prediction[:, 0] >= 0)
    assert set(model.get_last_bernoulli_gamma_aux() or {}) >= {
        "logit_wet",
        "log_mu",
        "log_phi",
        "pr_expected",
    }

    loss_fn = build_loss_fn(config, ["pr", "tasmax"])
    assert isinstance(loss_fn, CompositePredictandLoss)
    loss = loss_fn(prediction, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and bool(torch.count_nonzero(parameter.grad).item())
        for parameter in model.precip_bg_head.parameters()
    )
    assert loss_fn.describe()["bernoulli_gamma"]["enabled"] is True


def test_bernoulli_gamma_can_be_the_residual_diffusion_baseline():
    config = _config(head_type="diffusion", residual=True)
    model = _model(config)
    assert model.diffusion_enabled
    assert model.diffusion_head is not None
    assert model.output_conv_block is not None
    assert model.precip_bg_head is not None

    batch = _batch()
    result = model(batch)
    assert isinstance(result, dict)
    assert torch.isfinite(result["diffusion_loss"])
    assert result["baseline_prediction"].shape == batch["y"].shape
    assert torch.all(result["baseline_prediction"][:, 0] >= 0)
    assert "__bernoulli_gamma_aux" in batch


def test_score_matching_masks_nan_before_noising_and_has_zero_invalid_gradient():
    target = torch.randn(1, 1, 3, 4)
    target[..., 1, 2] = float("nan")
    valid = torch.isfinite(target)
    score_field = nn.Parameter(torch.zeros_like(target))

    def score_fn(x, cond, t):
        del x, cond, t
        return score_field

    loss = score_matching_loss(
        VPSDE(beta_min=0.1, beta_max=1.0, N=8),
        score_fn,
        target,
        torch.zeros(1, 1, 3, 4),
        valid_mask=valid,
        generator=torch.Generator().manual_seed(17),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert score_field.grad is not None
    assert score_field.grad[..., 1, 2].item() == 0.0
    assert bool(torch.count_nonzero(score_field.grad[valid]).item())


def test_score_matching_rejects_batch_without_any_valid_target_cell():
    target = torch.full((1, 1, 2, 2), float("nan"))

    with pytest.raises(ValueError, match="no finite valid target cells"):
        score_matching_loss(
            VPSDE(beta_min=0.1, beta_max=1.0, N=8),
            lambda x, cond, t: torch.zeros_like(x),
            target,
            torch.zeros_like(target),
        )


def test_full_field_diffusion_rejects_separate_bernoulli_gamma_decoder():
    config = _config(head_type="diffusion", residual=False)
    try:
        _model(config)
    except ValueError as exc:
        assert "full-field diffusion" in str(exc)
    else:  # pragma: no cover - protects the incompatibility contract
        raise AssertionError("full-field diffusion accepted a separate BG decoder")
