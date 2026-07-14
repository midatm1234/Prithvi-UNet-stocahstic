"""Tests for the optional conditional diffusion decoder head.

These tests use small dummy tensors only (no data, no GPU, no Prithvi backbone),
so they run quickly and verify:

* the deterministic head still imports/builds,
* the score-based SDEs produce finite marginals,
* the diffusion training loss is finite and gradients flow,
* the diffusion sampler returns ``[B, output_channels, H, W]`` (even + odd sizes),
* config plumbing (head-type resolver, loss passthrough, config parsing),
* standardized encode/decode round-trips and precipitation non-negativity.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from granitewxc.decoders.diffusion_head import (
    DiffusionHead,
    DiffusionHeadConfig,
    build_diffusion_head,
)
from granitewxc.decoders.downscaling import ConvEncoderDecoder
from granitewxc.models.cordex_finetune_model import (
    ClimateECCCFinetuneWrapper,
    resolve_head_type,
)
from granitewxc.models.diffusion_loss import DiffusionLossPassthrough
from granitewxc.models.diffusion_sde import VESDE, VPSDE, build_sde, subVPSDE
from granitewxc.models.loss import build_loss_fn


def _small_head_config(**overrides) -> DiffusionHeadConfig:
    params = dict(
        sde="subvpsde",
        num_scales=50,
        base_channels=8,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        time_embed_dim=16,
        num_sampling_steps=3,
        sampling_method="pc",
        predictor="euler_maruyama",
        corrector="none",
    )
    params.update(overrides)
    return DiffusionHeadConfig(**params)


# ---------------------------------------------------------------------------
# SDE math
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,cls", [("vpsde", VPSDE), ("subvpsde", subVPSDE), ("vesde", VESDE)])
def test_build_sde_returns_expected_type_and_finite_marginals(name, cls):
    sde = build_sde({"sde": name}, num_scales=100)
    assert isinstance(sde, cls)
    assert sde.N == 100

    x = torch.randn(2, 2, 6, 6)
    t = torch.rand(2)
    mean, std = sde.marginal_prob(x, t)
    assert mean.shape == x.shape
    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()
    assert (std >= 0).all()

    prior = sde.prior_sampling((2, 2, 6, 6))
    assert prior.shape == (2, 2, 6, 6)
    assert torch.isfinite(prior).all()


def test_build_sde_rejects_unknown():
    with pytest.raises(ValueError):
        build_sde({"sde": "not-a-real-sde"})


# ---------------------------------------------------------------------------
# Deterministic head still available
# ---------------------------------------------------------------------------
def test_deterministic_conv_head_still_builds_and_runs():
    head = ConvEncoderDecoder(
        in_channels=4,
        channels=8,
        out_channels=2,
        kernel_size=[3],
        scale=[2],
        upsampling_mode="nearest",
    )
    out = head(torch.randn(1, 4, 5, 5))
    assert out.shape[0] == 1 and out.shape[1] == 2


# ---------------------------------------------------------------------------
# Diffusion training loss
# ---------------------------------------------------------------------------
def test_diffusion_training_loss_is_finite_and_backprops():
    torch.manual_seed(0)
    head = DiffusionHead(cond_channels=6, output_channels=2, head_config=_small_head_config())
    cond = torch.randn(2, 6, 8, 8)
    target = torch.randn(2, 2, 8, 8)

    loss = head.training_loss(cond, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()
    grads = [p.grad for p in head.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_diffusion_forward_alias_matches_training_loss():
    torch.manual_seed(0)
    head = DiffusionHead(cond_channels=4, output_channels=2, head_config=_small_head_config())
    cond = torch.randn(1, 4, 6, 6)
    target = torch.randn(1, 2, 6, 6)
    torch.manual_seed(1)
    a = head.training_loss(cond, target)
    torch.manual_seed(1)
    b = head(cond, target)
    assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# Diffusion sampler shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("h,w", [(8, 8), (7, 9)])
def test_diffusion_sampler_returns_target_shape(h, w):
    torch.manual_seed(0)
    head = DiffusionHead(cond_channels=5, output_channels=2, head_config=_small_head_config())
    cond = torch.randn(2, 5, 4, 4)  # low-res conditioning features
    with torch.no_grad():
        sample = head.sample(cond, h, w)
    assert sample.shape == (2, 2, h, w)
    assert torch.isfinite(sample).all()


def test_ode_sampler_returns_target_shape():
    torch.manual_seed(0)
    cfg = _small_head_config(sampling_method="ode", num_sampling_steps=3)
    head = DiffusionHead(cond_channels=4, output_channels=2, head_config=cfg)
    cond = torch.randn(1, 4, 4, 4)
    with torch.no_grad():
        sample = head.sample(cond, 8, 8)
    assert sample.shape == (1, 2, 8, 8)
    assert torch.isfinite(sample).all()


def test_build_diffusion_head_factory_reads_config():
    config = SimpleNamespace(
        model=SimpleNamespace(
            head_type="diffusion",
            diffusion={
                "sde": "vpsde",
                "num_scales": 200,
                "base_channels": 8,
                "channel_multipliers": [1, 2],
                "num_sampling_steps": 4,
            },
        )
    )
    head = build_diffusion_head(config, cond_channels=3, output_channels=2)
    assert isinstance(head, DiffusionHead)
    assert head.cond_channels == 3
    assert head.output_channels == 2
    assert isinstance(head.sde, VPSDE)
    assert head.sde.N == 200


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["diffusion", "diffusion_head", "sde", "score", "score_sde"])
def test_resolve_head_type_diffusion_aliases(value):
    config = SimpleNamespace(model=SimpleNamespace(head_type=value))
    assert resolve_head_type(config) == "diffusion"


def test_resolve_head_type_decoder_type_alias_and_default():
    assert resolve_head_type(SimpleNamespace(model=SimpleNamespace(decoder_type="diffusion"))) == "diffusion"
    assert resolve_head_type(SimpleNamespace(model=SimpleNamespace())) == "deterministic"
    assert resolve_head_type(SimpleNamespace()) == "deterministic"
    assert resolve_head_type(None) == "deterministic"


def test_build_loss_fn_returns_diffusion_passthrough():
    config = SimpleNamespace(model=SimpleNamespace(head_type="diffusion"))
    loss_fn = build_loss_fn(config, output_vars=["pr", "tasmax"])
    assert isinstance(loss_fn, DiffusionLossPassthrough)

    scalar = torch.tensor(1.25, requires_grad=True)
    assert loss_fn(scalar, {"y": torch.zeros(1, 2, 4, 4)}) is scalar
    assert "type" in loss_fn.describe()


# ---------------------------------------------------------------------------
# Standardized encode/decode and precipitation non-negativity
# ---------------------------------------------------------------------------
class _DecoderDouble(ClimateECCCFinetuneWrapper):
    """Minimal object exposing the buffers the encode/decode helpers need."""

    def __init__(self, sigma, mu, codes, nonneg):
        torch.nn.Module.__init__(self)
        self.output_scalers_sigma = sigma
        self.output_scalers_mu = mu
        self.register_buffer("predictand_scaling_method_codes", codes)
        self.register_buffer("predictand_nonneg_enabled_mask", nonneg)

    def _resolve_output_scalers(self, x, scaler_offset=None):
        return self.output_scalers_mu.to(x.dtype), self.output_scalers_sigma.to(x.dtype)


def _make_decoder(codes, nonneg, sigma, mu):
    return _DecoderDouble(
        sigma=torch.tensor(sigma).view(1, -1, 1, 1),
        mu=torch.tensor(mu).view(1, -1, 1, 1),
        codes=torch.tensor(codes, dtype=torch.int64),
        nonneg=torch.tensor(nonneg, dtype=torch.bool),
    )


def test_encode_decode_roundtrip_zscore_and_divide_only():
    # channel 0 = pr (divide_only, code 1), channel 1 = tasmax (zscore, code 0)
    dec = _make_decoder(codes=[1, 0], nonneg=[True, False], sigma=[10.0, 2.0], mu=[0.0, 5.0])
    y = torch.stack(
        [torch.full((1, 4, 4), 3.5), torch.full((1, 4, 4), 12.0)], dim=1
    )  # [B=1, C=2, H, W]
    std = dec._encode_targets_std(y)
    back = dec._decode_targets_std(std)
    assert torch.allclose(back, y, atol=1e-5)


def test_encode_decode_roundtrip_log1p():
    dec = _make_decoder(codes=[2], nonneg=[True], sigma=[1.5], mu=[0.3])
    y = torch.full((1, 1, 4, 4), 4.0)
    std = dec._encode_targets_std(y)
    back = dec._decode_targets_std(std)
    assert torch.allclose(back, y, atol=1e-5)


def test_decode_clamps_precipitation_non_negativity():
    dec = _make_decoder(codes=[1, 0], nonneg=[True, False], sigma=[10.0, 2.0], mu=[0.0, 5.0])
    # negative standardized values -> pr would decode negative and must be clamped
    std = torch.stack(
        [torch.full((1, 4, 4), -1.0), torch.full((1, 4, 4), -1.0)], dim=1
    )
    decoded = dec._decode_targets_std(std)
    assert float(decoded[:, 0, ...].min()) >= 0.0  # pr clamped
    assert torch.allclose(decoded[:, 1, ...], torch.full((1, 4, 4), 3.0))  # tasmax untouched


# ---------------------------------------------------------------------------
# Shared _diffusion_forward path (the exact glue both CORDEX models call).
#
# Training/validation: the trainer calls ``model(batch)`` -> scalar loss.
# Inference: the blending helper calls
# ``model(batch, return_pre_inverse=True, return_raw_output=True)`` -> 3-tuple.
# ---------------------------------------------------------------------------
def _diffusion_decoder_double():
    dec = _make_decoder(codes=[1, 0], nonneg=[True, False], sigma=[10.0, 2.0], mu=[0.0, 5.0])
    dec.diffusion_head = DiffusionHead(
        cond_channels=4, output_channels=2, head_config=_small_head_config()
    )
    dec._last_precip_hurdle_aux = "stale"  # must be reset to None by the forward
    return dec


def test_diffusion_forward_training_branch_returns_scalar_loss():
    torch.manual_seed(0)
    dec = _diffusion_decoder_double()
    cond = torch.randn(2, 4, 6, 6)
    batch = {"y": torch.rand(2, 2, 12, 12)}  # physical pr/tasmax targets
    out = dec._diffusion_forward(cond, batch)  # no flags -> training loss
    assert out.ndim == 0 and torch.isfinite(out)
    assert dec._last_precip_hurdle_aux is None  # diffusion mode never emits hurdle aux


def test_diffusion_forward_inference_branch_returns_decoded_tuple():
    torch.manual_seed(0)
    dec = _diffusion_decoder_double()
    cond = torch.randn(2, 4, 5, 5)
    batch = {"y": torch.rand(2, 2, 16, 16)}
    with torch.no_grad():
        result = dec._diffusion_forward(
            cond, batch, return_pre_inverse=True, return_raw_output=True
        )
    assert isinstance(result, tuple) and len(result) == 3
    x_out, x_pre_inverse, raw_out = result
    # matches the target grid the NetCDF writer expects: [B, 2, H_target, W_target]
    assert x_out.shape == (2, 2, 16, 16)
    assert x_pre_inverse.shape[-2:] == (16, 16)
    assert raw_out.shape[-2:] == (16, 16)
    assert torch.isfinite(x_out).all()
    # precipitation channel remains non-negative after decoding
    assert float(x_out[:, 0, ...].min()) >= 0.0

