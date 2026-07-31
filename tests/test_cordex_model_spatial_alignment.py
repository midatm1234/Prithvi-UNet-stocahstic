from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from PrithviWxC.model import PrithviWxCEncoderDecoder

from granitewxc.models.cordex_finetune_model import (
    ClimateDownscaleFinetuneUNETModel,
    _prepare_normalized_outputs,
    _resolve_spatial_scalers,
)
from granitewxc.models.finetune_model import PatchEmbed
from granitewxc.utils.prism_tiling import (
    TilePlan,
    WeightedTileStitcher,
    blend_window,
    extract_halo_context,
)


def _tiny_unet(
    decoder_skip_source: str | None = None,
    *,
    backbone_attention_scope: str | None = None,
    residual_connection: bool = True,
    backbone_residual_mode: str | None = None,
    input_mu: torch.Tensor | None = None,
    input_sigma: torch.Tensor | None = None,
    output_mu: torch.Tensor | None = None,
    output_sigma: torch.Tensor | None = None,
) -> ClimateDownscaleFinetuneUNETModel:
    model_config = SimpleNamespace(
        embed_dim=4,
        encoder_decoder_scale_per_stage=[[2], [2]],
        residual="none",
        residual_connection=residual_connection,
        downscaling_embed_dim=2,
        encoder_decoder_conv_channels=2,
        decoder_upsampling_mode="bilinear",
        output_scaler_resize_mode="bilinear",
        output_scaler_align_corners=False,
        static_embedding_scale=1.0,
        static_skip_scale=1.0,
        static_dropout_p=0.0,
    )
    if decoder_skip_source is not None:
        model_config.decoder_skip_source = decoder_skip_source
    if backbone_attention_scope is not None:
        model_config.backbone_attention_scope = backbone_attention_scope
    if backbone_residual_mode is not None:
        model_config.backbone_residual_mode = backbone_residual_mode

    config = SimpleNamespace(
        data=SimpleNamespace(
            n_input_timestamps=1,
            use_static=False,
            output_vars=["tmax"],
        ),
        model=model_config,
        backbone_use=False,
        mask_unit_size=[4, 4],
        backbone_gradient_checkpointing=False,
        finetune_w_static=False,
        predictands={},
        precip_model="single_head",
    )

    input_mu = torch.zeros(1) if input_mu is None else input_mu
    input_sigma = torch.ones_like(input_mu) if input_sigma is None else input_sigma
    output_mu = torch.zeros(1) if output_mu is None else output_mu
    output_sigma = torch.ones_like(output_mu) if output_sigma is None else output_sigma
    empty_static = torch.empty(0)

    model = ClimateDownscaleFinetuneUNETModel(
        embedding=PatchEmbed((1, 1), channels=1, embed_dim=2),
        embedding_static=None,
        backbone=nn.Identity(),
        patch_size_px_backbone=(1, 1),
        input_scalers_mu=input_mu,
        input_scalers_sigma=input_sigma,
        input_scalers_epsilon=0.0,
        static_input_scalers_mu=empty_static,
        static_input_scalers_sigma=empty_static,
        static_input_scalers_epsilon=1e-6,
        output_scalers_mu=output_mu,
        output_scalers_sigma=output_sigma,
        static_output_scalers_mu=empty_static,
        static_output_scalers_sigma=empty_static,
        n_bins=1,
        scale=[2, 2],
        kernel_size=[3, 3],
        config=config,
    )
    return model.eval()


def test_spatial_scalers_require_offsets_and_take_exact_batched_slices():
    mu = torch.arange(12 * 14, dtype=torch.float32).reshape(1, 1, 12, 14)
    sigma = mu + 1000.0
    reference = torch.empty(2, 1, 3, 4)
    offsets = torch.tensor([[1, 2], [6, 8]])

    resolved_mu, resolved_sigma = _resolve_spatial_scalers(
        "target", mu, sigma, reference, scaler_offset=offsets
    )

    assert torch.equal(resolved_mu[0], mu[0, :, 1:4, 2:6])
    assert torch.equal(resolved_mu[1], mu[0, :, 6:9, 8:12])
    assert torch.equal(resolved_sigma[0], sigma[0, :, 1:4, 2:6])
    assert torch.equal(resolved_sigma[1], sigma[0, :, 6:9, 8:12])

    with pytest.raises(ValueError, match="scaler_offset is required"):
        _resolve_spatial_scalers("target", mu, sigma, reference)
    with pytest.raises(ValueError, match="outside scaler grid"):
        _resolve_spatial_scalers(
            "target", mu, sigma, reference, scaler_offset=torch.tensor([[10, 0], [0, 0]])
        )
    with pytest.raises(ValueError, match="must have shape"):
        _resolve_spatial_scalers(
            "target", mu, sigma, reference, scaler_offset=torch.tensor([[1, 2, 3]])
        )


def test_channel_only_and_matching_spatial_scalers_remain_offset_free():
    reference = torch.empty(3, 2, 5, 7)
    global_mu = torch.tensor([1.0, 2.0]).reshape(1, 2, 1, 1)
    global_sigma = torch.tensor([3.0, 4.0]).reshape(1, 2, 1, 1)
    spatial_mu = torch.zeros(1, 2, 5, 7)
    spatial_sigma = torch.ones(1, 2, 5, 7)

    resolved_global = _resolve_spatial_scalers(
        "input", global_mu, global_sigma, reference
    )
    resolved_spatial = _resolve_spatial_scalers(
        "input", spatial_mu, spatial_sigma, reference
    )

    assert resolved_global[0] is global_mu
    assert resolved_global[1] is global_sigma
    assert resolved_spatial[0] is spatial_mu
    assert resolved_spatial[1] is spatial_sigma


def test_spatial_input_scalers_follow_reflect_padded_halo_at_boundaries():
    mu = torch.arange(4 * 5, dtype=torch.float32).reshape(1, 1, 4, 5)
    sigma = mu + 100.0
    reference = torch.empty(1, 1, 6, 7)

    actual_mu, actual_sigma = _resolve_spatial_scalers(
        "input", mu, sigma, reference, scaler_offset=(-1, -1)
    )
    expected_mu = F.pad(mu, (1, 1, 1, 1), mode="reflect")
    expected_sigma = F.pad(sigma, (1, 1, 1, 1), mode="reflect")

    assert torch.equal(actual_mu, expected_mu)
    assert torch.equal(actual_sigma, expected_sigma)


def test_normalized_outputs_are_cropped_then_resized_with_wet_logits():
    raw = torch.arange(2 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8)
    wet = raw + 500.0
    crops = torch.tensor([[1, 2, 4, 3], [2, 1, 4, 3]])

    actual_raw, actual_wet = _prepare_normalized_outputs(
        raw, wet, (2, 2), output_crop=crops
    )
    cropped_raw = torch.cat(
        (raw[0:1, :, 1:5, 2:5], raw[1:2, :, 2:6, 1:4]), dim=0
    )
    cropped_wet = torch.cat(
        (wet[0:1, :, 1:5, 2:5], wet[1:2, :, 2:6, 1:4]), dim=0
    )

    assert torch.equal(
        actual_raw,
        F.interpolate(cropped_raw, size=(2, 2), mode="bilinear", align_corners=False),
    )
    assert torch.equal(
        actual_wet,
        F.interpolate(cropped_wet, size=(2, 2), mode="bilinear", align_corners=False),
    )

    with pytest.raises(ValueError, match="outside raw output grid"):
        _prepare_normalized_outputs(raw, wet, (2, 2), output_crop=(7, 7, 2, 2))


def test_decoder_skip_source_is_schema_compatible_and_defaults_to_legacy():
    torch.manual_seed(11)
    default_model = _tiny_unet()
    explicit_legacy = _tiny_unet("legacy")
    dynamic_model = _tiny_unet("dynamic")

    assert default_model.decoder_skip_source == "legacy"
    assert default_model.backbone_attention_scope == "legacy_global"
    assert list(default_model.state_dict()) == list(explicit_legacy.state_dict())
    assert list(default_model.state_dict()) == list(dynamic_model.state_dict())
    explicit_legacy.load_state_dict(default_model.state_dict(), strict=True)
    dynamic_model.load_state_dict(default_model.state_dict(), strict=True)

    calls = {"legacy": 0, "dynamic": 0}

    def count_legacy(*_args):
        calls["legacy"] += 1

    def count_dynamic(*_args):
        calls["dynamic"] += 1

    legacy_handle = default_model.downsampling_layers[0].register_forward_hook(count_legacy)
    dynamic_handle = dynamic_model.downsampling_layers[0].register_forward_hook(count_dynamic)
    batch = {"x": torch.randn(1, 1, 16, 16), "y": torch.zeros(1, 1, 16, 16)}
    with torch.no_grad():
        default_out = default_model(dict(batch))
        explicit_out = explicit_legacy(dict(batch))
        dynamic_out = dynamic_model(dict(batch))
    legacy_handle.remove()
    dynamic_handle.remove()

    assert calls == {"legacy": 0, "dynamic": 1}
    assert torch.equal(default_out, explicit_out)
    assert float((default_out - dynamic_out).abs().max()) > 1e-7

    with pytest.raises(ValueError, match="decoder_skip_source"):
        _tiny_unet("unsupported")
    with pytest.raises(ValueError, match="backbone_attention_scope"):
        _tiny_unet(backbone_attention_scope="unsupported")


def test_configured_shallow_residual_reaches_post_backbone_convolution():
    """Regression test for a residual that was computed and then discarded."""
    torch.manual_seed(29)
    residual_model = _tiny_unet(
        "legacy",
        residual_connection=True,
        backbone_residual_mode="pre_conv_add",
    )
    plain_model = _tiny_unet(
        "legacy",
        residual_connection=False,
        backbone_residual_mode="pre_conv_add",
    )
    plain_model.load_state_dict(residual_model.state_dict(), strict=True)

    captured: dict[str, torch.Tensor] = {}
    residual_handle = residual_model.conv_after_backbone.register_forward_pre_hook(
        lambda _module, args: captured.__setitem__("residual", args[0].detach().clone())
    )
    plain_handle = plain_model.conv_after_backbone.register_forward_pre_hook(
        lambda _module, args: captured.__setitem__("plain", args[0].detach().clone())
    )
    batch = {"x": torch.randn(1, 1, 16, 16), "y": torch.zeros(1, 1, 16, 16)}
    with torch.no_grad():
        residual_model(dict(batch))
        plain_model(dict(batch))
    residual_handle.remove()
    plain_handle.remove()

    # This tiny fixture disables the backbone, so deep and shallow features are
    # identical and enabling the residual must exactly double the conv input.
    torch.testing.assert_close(captured["residual"], 2.0 * captured["plain"])


def test_default_backbone_residual_mode_exactly_preserves_legacy_outputs():
    """Contractless checkpoints retain the historical ignored-residual path."""
    torch.manual_seed(31)
    default_model = _tiny_unet("legacy", residual_connection=True)
    explicit_legacy = _tiny_unet(
        "legacy",
        residual_connection=True,
        backbone_residual_mode="legacy_ignored",
    )
    residual_disabled = _tiny_unet("legacy", residual_connection=False)
    explicit_legacy.load_state_dict(default_model.state_dict(), strict=True)
    residual_disabled.load_state_dict(default_model.state_dict(), strict=True)

    batch = {"x": torch.randn(1, 1, 16, 16), "y": torch.zeros(1, 1, 16, 16)}
    with torch.no_grad():
        default_output = default_model(dict(batch))
        explicit_output = explicit_legacy(dict(batch))
        disabled_output = residual_disabled(dict(batch))

    assert torch.equal(default_output, explicit_output)
    assert torch.equal(default_output, disabled_output)

    with pytest.raises(ValueError, match="backbone_residual_mode"):
        _tiny_unet("legacy", backbone_residual_mode="unsupported")


def test_model_uses_separate_input_and_output_offsets_before_decoding():
    grid = torch.arange(20 * 20, dtype=torch.float32).reshape(1, 20, 20)
    input_mu = grid / 10.0
    input_sigma = torch.ones_like(input_mu)
    output_mu = grid / 20.0
    output_sigma = 1.0 + grid / 400.0
    model = _tiny_unet(
        "legacy",
        input_mu=input_mu,
        input_sigma=input_sigma,
        output_mu=output_mu,
        output_sigma=output_sigma,
    )

    captured: dict[str, torch.Tensor] = {}

    def capture_normalized_input(_module, args):
        captured["normalized_input"] = args[0].detach().clone()

    def capture_uncropped_output(_module, _args, output):
        captured["uncropped_output"] = output.detach().clone()

    input_handle = model.embedding.register_forward_pre_hook(capture_normalized_input)
    output_handle = model.output_conv_block.register_forward_hook(capture_uncropped_output)

    input_offset = (1, 2)
    output_offset = (5, 6)
    x = input_mu[:, 1:17, 2:18].unsqueeze(0) + 3.0 * input_sigma[
        :, 1:17, 2:18
    ].unsqueeze(0)
    batch = {
        "x": x,
        "y": torch.zeros(1, 1, 8, 8),
        "__input_scaler_offset": input_offset,
        "__output_scaler_offset": output_offset,
        "__output_crop": (3, 4, 10, 10),
    }
    with torch.no_grad():
        decoded, raw = model(batch, return_raw_output=True)

    expected_raw = F.interpolate(
        captured["uncropped_output"][..., 3:13, 4:14],
        size=(8, 8),
        mode="bilinear",
        align_corners=False,
    )
    expected_decoded = (
        expected_raw * output_sigma[:, 5:13, 6:14].unsqueeze(0)
        + output_mu[:, 5:13, 6:14].unsqueeze(0)
    )

    assert torch.allclose(captured["normalized_input"], torch.full_like(x, 3.0))
    assert torch.equal(raw, expected_raw)
    assert torch.equal(decoded, expected_decoded)

    # The legacy metadata key remains a fallback for both scaler grids.
    legacy_offset = (2, 3)
    legacy_x = input_mu[:, 2:18, 3:19].unsqueeze(0) + 2.0
    with torch.no_grad():
        legacy_decoded, legacy_raw = model(
            {
                "x": legacy_x,
                "y": torch.zeros(1, 1, 8, 8),
                "__scaler_offset": legacy_offset,
                "__output_crop": (4, 4, 8, 8),
            },
            return_raw_output=True,
        )
    legacy_expected = (
        legacy_raw * output_sigma[:, 2:10, 3:11].unsqueeze(0)
        + output_mu[:, 2:10, 3:11].unsqueeze(0)
    )

    input_handle.remove()
    output_handle.remove()
    assert torch.allclose(
        captured["normalized_input"], torch.full_like(legacy_x, 2.0)
    )
    assert torch.equal(legacy_decoded, legacy_expected)


def test_dynamic_decoder_halo_tiling_matches_full_frame_model() -> None:
    """Exercise the model crop/decoder contract, not only the accumulator."""
    torch.manual_seed(19)
    model = _tiny_unet(
        "dynamic", backbone_attention_scope="windowed_local"
    )
    model.backbone = PrithviWxCEncoderDecoder(
        embed_dim=4,
        n_blocks=1,
        mlp_multiplier=2,
        n_heads=1,
        dropout=0.0,
        drop_path=0.0,
    )
    model.backbone_use = True
    model.eval()
    field = torch.randn(1, 1, 64, 64)
    plan = TilePlan.build(
        (64, 64), (32, 32), overlap=(16, 16), halo=(16, 16)
    )
    plan.assert_globally_aligned(4)

    full_context = extract_halo_context(
        field[0], (0, 0), plan.domain_shape, plan.halo
    ).unsqueeze(0)
    with torch.no_grad():
        full = model(
            {
                "x": full_context,
                "y": torch.zeros(1, 1, *plan.domain_shape),
                "__output_crop": torch.tensor(
                    [*plan.halo, *plan.domain_shape], dtype=torch.long
                ),
            }
        )[0].numpy()

    stitcher = WeightedTileStitcher(1, plan.domain_shape)
    window = blend_window(plan.core_shape, plan.overlap, mode="hann")
    for origin in plan.positions:
        context = extract_halo_context(
            field[0], origin, plan.core_shape, plan.halo
        ).unsqueeze(0)
        with torch.no_grad():
            tile = model(
                {
                    "x": context,
                    "y": torch.zeros(1, 1, *plan.core_shape),
                    "__output_crop": torch.tensor(
                        [*plan.halo, *plan.core_shape], dtype=torch.long
                    ),
                }
            )[0].numpy()
        stitcher.add(tile, origin, window)

    tiled = stitcher.finalize()
    np.testing.assert_allclose(tiled, full, rtol=0.0, atol=1.0e-9)
