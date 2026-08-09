from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from examples.CORDEX_ML.utils.inference_blending import (
    BoundaryMitigationConfig,
    infer_batch_with_boundary_mitigation,
)


class _LocalReplicateModel(torch.nn.Module):
    diffusion_enabled = False
    mask_unit_size_px_backbone = (1, 1)

    def forward(self, batch, return_pre_inverse=False, return_raw_output=False):
        del return_pre_inverse, return_raw_output
        x = batch["x"][:, :1]
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 0.0]],
            device=x.device,
            dtype=x.dtype,
        ).view(1, 1, 3, 3) / 8.0
        out = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel)
        return out, out, out


class _GridpointScalingModel(torch.nn.Module):
    diffusion_enabled = False
    mask_unit_size_px_backbone = (1, 1)

    def __init__(self, height: int, width: int):
        super().__init__()
        values = torch.linspace(1.0, 3.0, height * width).reshape(1, 1, height, width)
        self.register_buffer("scale", values)

    def forward(self, batch, return_pre_inverse=False, return_raw_output=False):
        del return_pre_inverse, return_raw_output
        y0, x0 = batch.get("__scaler_offset", (0, 0))
        height, width = batch["x"].shape[-2:]
        local_scale = self.scale[..., y0 : y0 + height, x0 : x0 + width]
        out = batch["x"][:, :1] / local_scale
        return out, out, out


def _batch(x: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"x": x, "y": torch.zeros_like(x[:, :1])}


def _cfg(*, origin=(0, 0)) -> BoundaryMitigationConfig:
    return BoundaryMitigationConfig(
        enabled=True,
        force_full_frame=False,
        tile_size=(16, 16),
        overlap=(4, 4),
        halo=(1, 1),
        tile_origin=origin,
        blend_window="hann",
    )


def test_halo_tiled_local_model_matches_full_frame_and_has_no_seams():
    torch.manual_seed(7)
    model = _LocalReplicateModel()
    batch = _batch(torch.randn(2, 1, 32, 32))

    full = model(batch, return_pre_inverse=True, return_raw_output=True)[0]
    tiled = infer_batch_with_boundary_mitigation(model, batch, _cfg())[0]

    torch.testing.assert_close(tiled, full, rtol=0.0, atol=2e-7)


def test_shifting_tile_origin_does_not_shift_output_pattern():
    torch.manual_seed(11)
    model = _LocalReplicateModel()
    batch = _batch(torch.randn(1, 1, 32, 32))

    origin_zero = infer_batch_with_boundary_mitigation(model, batch, _cfg(origin=(0, 0)))[0]
    origin_shifted = infer_batch_with_boundary_mitigation(model, batch, _cfg(origin=(3, 5)))[0]

    torch.testing.assert_close(origin_shifted, origin_zero, rtol=0.0, atol=5e-7)


def test_constant_field_stays_constant_through_halo_stitching():
    model = _LocalReplicateModel()
    batch = _batch(torch.full((1, 1, 32, 32), 4.25))

    tiled = infer_batch_with_boundary_mitigation(model, batch, _cfg())[0]

    torch.testing.assert_close(tiled, torch.full_like(tiled, 4.25), rtol=0.0, atol=1e-6)


def test_gridpoint_scaler_offsets_match_full_frame_at_boundaries():
    model = _GridpointScalingModel(32, 32)
    batch = _batch(torch.full((1, 1, 32, 32), 6.0))

    full = model(batch, return_pre_inverse=True, return_raw_output=True)[0]
    tiled = infer_batch_with_boundary_mitigation(model, batch, _cfg())[0]

    torch.testing.assert_close(tiled, full, rtol=0.0, atol=5e-7)


def test_independent_diffusion_tiling_is_rejected():
    model = _LocalReplicateModel()
    model.diffusion_enabled = True
    batch = _batch(torch.ones(1, 1, 32, 32))

    try:
        infer_batch_with_boundary_mitigation(model, batch, _cfg())
    except RuntimeError as exc:
        assert "different reverse-process noise field" in str(exc)
    else:
        raise AssertionError("independent diffusion tiling must fail")


class _ModelWrapper(torch.nn.Module):
    def __init__(self, wrapped: torch.nn.Module, attribute: str):
        super().__init__()
        setattr(self, attribute, wrapped)
        self.attribute = attribute

    def forward(self, *args, **kwargs):
        return getattr(self, self.attribute)(*args, **kwargs)


@pytest.mark.parametrize("attribute", ["module", "_orig_mod"])
def test_wrapped_diffusion_tiling_is_rejected(attribute):
    inner = _LocalReplicateModel()
    inner.diffusion_enabled = True
    wrapped = _ModelWrapper(inner, attribute)
    batch = _batch(torch.ones(1, 1, 32, 32))

    with pytest.raises(RuntimeError, match="different reverse-process noise field"):
        infer_batch_with_boundary_mitigation(wrapped, batch, _cfg())


def test_nested_wrappers_expose_inner_stride_for_deterministic_tiling():
    inner = _LocalReplicateModel()
    inner.mask_unit_size_px_backbone = (4, 8)
    wrapped = _ModelWrapper(_ModelWrapper(inner, "_orig_mod"), "module")
    batch = _batch(torch.ones(1, 1, 32, 32))

    output = infer_batch_with_boundary_mitigation(wrapped, batch, _cfg())[0]

    torch.testing.assert_close(output, torch.ones_like(output), rtol=0.0, atol=1e-6)
