"""Production-path regression tests for additive residual diffusion.

These tests intentionally compose the same helpers used by CORDEX inference:
ensemble dispatch, halo stitching, residual-stage reconstruction, target-space
transforms, and xarray output construction.  The spatial oracle is
deterministic because independent tile-level reverse-diffusion sampling is
explicitly rejected by the production boundary helper.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

from examples.CORDEX_ML.utils.diffusion_inference import (
    add_predictions_to_dataset,
    infer_batch_ensemble,
    residual_transformation_stages,
)
from examples.CORDEX_ML.utils.inference_blending import (
    BoundaryMitigationConfig,
    infer_batch_with_boundary_mitigation,
)
from granitewxc.models.cordex_finetune_model import (
    ClimateDownscaleFinetuneUNETModel,
)
from granitewxc.models.diffusion_sampling import get_ddim_sampler
from granitewxc.models.diffusion_sde import VPSDE


class _ProductionTransformModel(ClimateDownscaleFinetuneUNETModel):
    """Minimal module using the production CORDEX target transforms."""

    diffusion_enabled = False
    mask_unit_size_px_backbone = (1, 1)

    def __init__(self) -> None:
        # The real model constructor needs the full backbone.  These regression
        # tests only need its inherited production encode/decode/scaler methods.
        torch.nn.Module.__init__(self)
        self.register_buffer(
            "output_scalers_mu",
            torch.tensor([0.0, 280.0], dtype=torch.float32).view(1, 2, 1, 1),
        )
        self.register_buffer(
            "output_scalers_sigma",
            torch.tensor([5.0, 2.0], dtype=torch.float32).view(1, 2, 1, 1),
        )
        # pr uses divide-only; tasmax uses z-score.
        self.register_buffer(
            "predictand_scaling_method_codes",
            torch.tensor([1, 0], dtype=torch.int64),
        )
        self.register_buffer(
            "predictand_nonneg_enabled_mask",
            torch.tensor([True, False], dtype=torch.bool),
        )


class _LocalResidualOracle(_ProductionTransformModel):
    """Local baseline plus local residual, suitable for exact halo testing."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        smooth = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ) / 8.0
        detail = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        )
        self.register_buffer("smooth_kernel", smooth.view(1, 1, 3, 3).repeat(2, 1, 1, 1))
        self.register_buffer("detail_kernel", detail.view(1, 1, 3, 3).repeat(2, 1, 1, 1))
        self.register_buffer(
            "residual_channel_scale",
            torch.tensor([0.08, 0.01], dtype=torch.float32).view(1, 2, 1, 1),
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        return_pre_inverse: bool = False,
        return_raw_output: bool = False,
    ):
        del return_pre_inverse, return_raw_output
        physical_input = batch["x"][:, :2]
        padded = F.pad(physical_input, (1, 1, 1, 1), mode="replicate")
        baseline_physical = F.conv2d(padded, self.smooth_kernel, groups=2)
        raw_residual_std = (
            F.conv2d(padded, self.detail_kernel, groups=2)
            * self.residual_channel_scale
        )
        baseline_std = self._encode_targets_std(baseline_physical)
        full_std = baseline_std + self.alpha * raw_residual_std
        final_physical = self._decode_targets_std(full_std)
        return final_physical, full_std, raw_residual_std


def _physical_fields(height: int = 19, width: int = 21) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height),
        torch.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    pr = 3.0 + 0.4 * xx + 0.2 * yy + 0.1 * torch.sin(4.0 * xx * yy)
    tasmax = 290.0 + 2.0 * xx - 1.5 * yy + 0.3 * torch.cos(3.0 * xx)
    return torch.stack((pr, tasmax), dim=0).unsqueeze(0)


def test_alpha_zero_or_zero_sample_returns_exact_deterministic_baseline() -> None:
    model = _ProductionTransformModel()
    baseline_physical = _physical_fields(height=3, width=4)
    baseline_std = model._encode_targets_std(baseline_physical)
    truth = baseline_physical.clone()
    truth[:, 0] += 0.25
    truth[:, 1] -= 0.75

    for raw_residual, alpha in (
        (torch.randn_like(baseline_std), 0.0),
        (torch.zeros_like(baseline_std), 1.0),
    ):
        full_std = baseline_std + alpha * raw_residual
        final = model._decode_targets_std(full_std)
        stages = residual_transformation_stages(
            model=model,
            truth_physical=truth,
            final_physical=final,
            full_standardized=full_std,
            generated_residual_standardized=raw_residual,
            baseline_standardized=baseline_std,
            baseline_physical=model._decode_targets_std(baseline_std),
            residual_application_scale=alpha,
        )

        # The deployed residual baseline is the deterministic field after its
        # production encode/decode round trip.  It must be bitwise unchanged by
        # alpha=0 (or a zero sample); the original physical tensor can differ
        # by a few floating-point ULPs after z-score inversion.
        deployed_baseline = model._decode_targets_std(baseline_std)
        assert torch.equal(final, deployed_baseline)
        torch.testing.assert_close(final, baseline_physical, rtol=2e-7, atol=2e-5)
        assert torch.count_nonzero(stages["applied_normalized_residual"]) == 0
        assert torch.count_nonzero(stages["applied_denormalized_residual"]) == 0


def test_true_residual_reconstructs_target_and_has_opposite_unet_error_sign() -> None:
    model = _ProductionTransformModel()
    baseline_physical = torch.tensor(
        [
            [
                [[2.0, 4.0, 1.0], [3.0, 5.0, 2.5]],
                [[286.0, 289.0, 291.0], [293.0, 288.0, 295.0]],
            ]
        ],
        dtype=torch.float32,
    )
    target_physical = torch.tensor(
        [
            [
                [[0.0, 5.0, 2.0], [4.0, 3.0, 2.5]],
                [[288.0, 287.0, 292.0], [290.0, 291.0, 294.0]],
            ]
        ],
        dtype=torch.float32,
    )
    baseline_std = model._encode_targets_std(baseline_physical)
    target_std = model._encode_targets_std(target_physical)
    true_residual_std = target_std - baseline_std
    reconstructed_std = baseline_std + true_residual_std
    reconstructed_physical = model._decode_targets_std(reconstructed_std)

    stages = residual_transformation_stages(
        model=model,
        truth_physical=target_physical,
        final_physical=reconstructed_physical,
        full_standardized=reconstructed_std,
        generated_residual_standardized=true_residual_std,
        baseline_standardized=baseline_std,
        baseline_physical=model._decode_targets_std(baseline_std),
        residual_application_scale=1.0,
    )

    torch.testing.assert_close(reconstructed_physical, target_physical, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(
        stages["normalized_true_residual"], true_residual_std, rtol=0.0, atol=1e-6
    )
    torch.testing.assert_close(
        stages["physical_true_residual"],
        target_physical - baseline_physical,
        rtol=0.0,
        atol=1e-6,
    )

    # The residual target is target - U-Net, hence exactly the negative of the
    # conventional U-Net error (U-Net - target).  A sign inversion must worsen
    # this oracle correction rather than silently pass a reconstruction test.
    unet_error_std = baseline_std - target_std
    torch.testing.assert_close(true_residual_std, -unet_error_std, rtol=0.0, atol=0.0)
    wrong_sign_physical = model._decode_targets_std(baseline_std - true_residual_std)
    baseline_rmse = torch.mean((baseline_physical - target_physical).square()).sqrt()
    wrong_sign_rmse = torch.mean((wrong_sign_physical - target_physical).square()).sqrt()
    assert wrong_sign_rmse > baseline_rmse


def test_precipitation_constraint_is_applied_after_residual_addition() -> None:
    model = _ProductionTransformModel()
    baseline_physical = torch.tensor([[[[1.0, 2.0]], [[285.0, 286.0]]]])
    baseline_std = model._encode_targets_std(baseline_physical)
    residual_std = torch.tensor([[[[-2.0, -3.0]], [[-1.0, 1.0]]]])

    decoded = model._decode_targets_std(baseline_std + residual_std)

    assert torch.equal(decoded[:, 0], torch.zeros_like(decoded[:, 0]))
    torch.testing.assert_close(
        decoded[:, 1], torch.tensor([[[283.0, 288.0]]]), rtol=0.0, atol=1e-6
    )


def test_halo_stitched_oracle_runs_residual_audit_and_netcdf_roundtrip(tmp_path) -> None:
    model = _LocalResidualOracle(alpha=1.0)
    physical_input = _physical_fields()
    batch = {"x": physical_input, "y": physical_input.clone()}
    full_physical, full_std, full_raw = model(
        batch, return_pre_inverse=True, return_raw_output=True
    )
    cfg = BoundaryMitigationConfig(
        enabled=True,
        force_full_frame=False,
        tile_size=(10, 11),
        overlap=(4, 4),
        halo=(1, 1),
        tile_origin=(2, 3),
        blend_window="hann",
    )

    output, standardized, raw_residual = infer_batch_ensemble(
        model=model,
        batch=batch,
        infer_batch=infer_batch_with_boundary_mitigation,
        boundary_cfg=cfg,
        head_type="diffusion",
        ensemble_size=2,
        base_seed=17,
        device=torch.device("cpu"),
    )

    assert output.shape == (1, 2, 2, 19, 21)
    for member in range(2):
        # Overlap accumulation changes float32 summation order.  The tolerance
        # is below one float32 ULP for the ~290 K channel and remains a strict
        # absolute check for precipitation.
        torch.testing.assert_close(
            output[:, member], full_physical, rtol=3e-7, atol=1e-6
        )
        torch.testing.assert_close(standardized[:, member], full_std, rtol=0.0, atol=3e-6)
        torch.testing.assert_close(raw_residual[:, member], full_raw, rtol=0.0, atol=3e-6)

    stages = residual_transformation_stages(
        model=model,
        truth_physical=full_physical,
        final_physical=output,
        full_standardized=standardized,
        generated_residual_standardized=raw_residual,
        baseline_standardized=model._encode_targets_std(
            F.conv2d(
                F.pad(physical_input, (1, 1, 1, 1), mode="replicate"),
                model.smooth_kernel,
                groups=2,
            )
        ),
        residual_application_scale=1.0,
    )
    torch.testing.assert_close(
        stages["normalized_true_residual"], full_raw, rtol=0.0, atol=1e-5
    )
    assert float(output[:, :, 0].min()) >= 0.0

    coords = {
        "time": np.array(["2001-01-01"], dtype="datetime64[D]"),
        "lat": np.linspace(-34.0, -22.0, 19),
        "lon": np.linspace(21.0, 33.0, 21),
    }
    prediction_ds = xr.Dataset(coords=coords)
    prediction_ds = add_predictions_to_dataset(
        prediction_ds=prediction_ds,
        target_vars=["pr", "tasmax"],
        outputs_np=output.detach().cpu().numpy(),
        coords=coords,
        time_dim="time",
        lat_dim="lat",
        lon_dim="lon",
        target_attrs={"pr": {"units": "mm/day"}, "tasmax": {"units": "K"}},
        head_type="diffusion",
        ensemble_size=2,
        base_seed=17,
    )
    output_path = tmp_path / "residual_oracle.nc"
    prediction_ds.to_netcdf(output_path, engine="h5netcdf")

    with xr.open_dataset(output_path, engine="h5netcdf") as reopened:
        reopened.load()
        assert reopened["pr"].dims == ("time", "ensemble", "lat", "lon")
        assert reopened["tasmax"].dims == ("time", "ensemble", "lat", "lon")
        assert reopened.attrs["head_type"] == "diffusion"
        assert reopened.attrs["ensemble_size"] == 2
        np.testing.assert_allclose(
            reopened["pr"].values,
            output[:, :, 0].detach().cpu().numpy(),
            rtol=0.0,
            atol=1e-6,
        )


def test_vp_ddim_analytical_score_oracle_reconstructs_clean_residual() -> None:
    torch.manual_seed(29)
    sde = VPSDE(beta_min=0.1, beta_max=20.0, N=5)
    clean_residual = torch.randn(2, 2, 4, 5)
    forward_noise = torch.randn_like(clean_residual)
    terminal_time = torch.ones(clean_residual.shape[0])
    terminal_mean, terminal_std = sde.marginal_prob(clean_residual, terminal_time)
    terminal_sample = (
        terminal_mean + terminal_std[:, None, None, None] * forward_noise
    )
    sde.prior_sampling = lambda shape: terminal_sample.clone()
    sampler = get_ddim_sampler(
        sde,
        clean_residual.shape[1:],
        eta=0.0,
        eps=1e-3,
        device="cpu",
    )
    conditioning = torch.zeros(2, 3, 4, 5)

    def analytical_score(
        noisy: torch.Tensor, _conditioning: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        mean, std = sde.marginal_prob(clean_residual, time)
        return -(noisy - mean) / torch.square(std[:, None, None, None])

    reconstructed = sampler(analytical_score, conditioning)
    torch.testing.assert_close(
        reconstructed, clean_residual, rtol=7e-5, atol=7e-5
    )
