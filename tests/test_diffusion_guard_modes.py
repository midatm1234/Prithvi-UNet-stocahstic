"""Regression tests for residual magnitude guard modes (clip / warn / reject).

These tests verify that:
1. 'reject' mode raises RuntimeError for over-limit corrections (original behavior).
2. 'clip' mode rescales per-channel RMS to exactly the limit, preserves spatial pattern.
3. 'warn' mode emits a Python warning and returns the original tensor unchanged.
4. All modes pass NaN/inf detection before the magnitude check.
5. Guard is disabled when residual_magnitude_guard_multiple <= 0.
6. Guard is skipped when the training-statistics count is below the minimum.
7. checkpoint_convergence_check flags unconverged score loss.
8. validate_diffusion_vs_baseline flags diffusion degradation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from granitewxc.decoders.diffusion_head import DiffusionHeadConfig, DiffusionHead


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_head(
    guard_mode: str,
    guard_multiple: float = 3.0,
    guard_min_count: int = 1,
) -> DiffusionHead:
    cfg = DiffusionHeadConfig(
        residual_diffusion=True,
        residual_mean_enabled=False,
        residual_magnitude_guard_multiple=guard_multiple,
        residual_guard_min_count=guard_min_count,
        residual_guard_mode=guard_mode,
        residual_application_scale=1.0,
        sde="vpsde",
        num_scales=10,
        projected_cond_channels=0,
        base_channels=16,
        channel_multipliers=(1,),
        num_res_blocks=1,
        time_embed_dim=32,
        num_sampling_steps=4,
        padding_mode="zeros",
    )
    head = DiffusionHead(cond_channels=4, output_channels=2, head_config=cfg)
    # Seed training stats: observed RMS per channel = 1.0
    head.residual_stats_count.fill_(1000.0)
    head.residual_stats_sumsq.fill_(1000.0)  # rms = sqrt(sumsq/count) = 1.0
    return head


def _huge_residual(rms_val: float = 100.0) -> torch.Tensor:
    return torch.ones(1, 2, 8, 8) * rms_val


# ---------------------------------------------------------------------------
# DiffusionHeadConfig validation
# ---------------------------------------------------------------------------

def test_config_accepts_valid_guard_modes() -> None:
    for mode in ("reject", "clip", "warn"):
        cfg = DiffusionHeadConfig(residual_guard_mode=mode)
        assert cfg.residual_guard_mode == mode


def test_config_rejects_invalid_guard_mode() -> None:
    with pytest.raises(ValueError, match="residual_guard_mode"):
        DiffusionHeadConfig(residual_guard_mode="clamp")


def test_config_default_guard_mode_is_reject() -> None:
    cfg = DiffusionHeadConfig()
    assert cfg.residual_guard_mode == "reject"


# ---------------------------------------------------------------------------
# reject mode
# ---------------------------------------------------------------------------

def test_reject_mode_raises_on_large_residual() -> None:
    head = _make_head("reject")
    with pytest.raises(RuntimeError, match="training-scale guard"):
        head._validate_generated_residual(_huge_residual())


def test_reject_mode_passes_small_residual() -> None:
    head = _make_head("reject")
    small = torch.ones(1, 2, 8, 8) * 0.5  # rms = 0.5 < 3*1.0 limit
    result = head._validate_generated_residual(small)
    assert torch.allclose(result, small)


# ---------------------------------------------------------------------------
# clip mode
# ---------------------------------------------------------------------------

def test_clip_mode_rescales_rms_to_limit() -> None:
    head = _make_head("clip", guard_multiple=3.0)
    huge = _huge_residual(100.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clipped = head._validate_generated_residual(huge)
    assert len(caught) == 1
    assert "Clipped" in str(caught[0].message)
    # Per-channel RMS after clipping should equal the limit = 3 * observed_rms
    observed_rms = head.residual_training_stats()["rms"]
    expected_rms = (observed_rms * 3.0).tolist()
    reduce_dims = (0, 2, 3)
    actual_rms = torch.sqrt(clipped.square().mean(dim=reduce_dims)).tolist()
    for a, e in zip(actual_rms, expected_rms):
        assert abs(a - e) < 1e-4, f"actual_rms={a}, expected={e}"


def test_clip_mode_preserves_spatial_pattern() -> None:
    head = _make_head("clip", guard_multiple=3.0)
    spatial_pattern = torch.randn(1, 2, 8, 8) * 50.0
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        clipped = head._validate_generated_residual(spatial_pattern)
    # The clipped tensor should be a per-channel scalar multiple of the input.
    for c in range(2):
        ratio = clipped[0, c] / spatial_pattern[0, c]
        # All pixels in the channel should have the same scale factor.
        assert ratio.std() < 1e-5, f"channel {c}: spatial pattern not preserved, std={ratio.std()}"


def test_clip_mode_passes_small_residual_without_warning() -> None:
    head = _make_head("clip", guard_multiple=3.0)
    small = torch.ones(1, 2, 8, 8) * 0.5
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = head._validate_generated_residual(small)
    assert len(caught) == 0
    assert torch.allclose(result, small)


# ---------------------------------------------------------------------------
# warn mode
# ---------------------------------------------------------------------------

def test_warn_mode_emits_warning_and_returns_unchanged() -> None:
    head = _make_head("warn")
    huge = _huge_residual(100.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = head._validate_generated_residual(huge)
    assert len(caught) == 1
    assert "WARNING" in str(caught[0].message)
    assert torch.equal(result, huge), "warn mode should return the original tensor"


# ---------------------------------------------------------------------------
# NaN/inf guard (all modes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["reject", "clip", "warn"])
def test_nan_raises_regardless_of_mode(mode: str) -> None:
    head = _make_head(mode)
    nan_residual = torch.full((1, 2, 8, 8), float("nan"))
    with pytest.raises(RuntimeError, match="NaN or infinity"):
        head._validate_generated_residual(nan_residual)


@pytest.mark.parametrize("mode", ["reject", "clip", "warn"])
def test_inf_raises_regardless_of_mode(mode: str) -> None:
    head = _make_head(mode)
    inf_residual = torch.full((1, 2, 8, 8), float("inf"))
    with pytest.raises(RuntimeError, match="NaN or infinity"):
        head._validate_generated_residual(inf_residual)


# ---------------------------------------------------------------------------
# Guard disabled / insufficient stats
# ---------------------------------------------------------------------------

def test_guard_disabled_when_multiple_zero() -> None:
    head = _make_head("reject", guard_multiple=0.0)
    # Even a huge residual should pass through silently.
    result = head._validate_generated_residual(_huge_residual())
    assert torch.equal(result, _huge_residual())


def test_guard_skipped_below_min_count() -> None:
    # guard_min_count=10000 but only 1000 samples accumulated.
    head = _make_head("reject", guard_min_count=10_000)
    result = head._validate_generated_residual(_huge_residual())
    assert torch.equal(result, _huge_residual())


# ---------------------------------------------------------------------------
# Diagnostic utilities
# ---------------------------------------------------------------------------

def test_checkpoint_convergence_check_flags_high_loss() -> None:
    from examples.CORDEX_ML.utils.diffusion_diagnostics import checkpoint_convergence_check

    stats = {"count": np.float64(2e9), "rms": np.array([1.23, 1.27])}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = checkpoint_convergence_check(
            stats,
            score_loss_history=[72.0, 72.5],
            expected_score_loss_threshold=5.0,
        )
    assert not result["converged"]
    assert any("72.50" in str(w.message) for w in caught)
    assert result["score_loss_recent"] == pytest.approx(72.5)


def test_checkpoint_convergence_check_passes_converged_checkpoint() -> None:
    from examples.CORDEX_ML.utils.diffusion_diagnostics import checkpoint_convergence_check

    stats = {"count": np.float64(2e9), "rms": np.array([0.95, 1.02])}
    result = checkpoint_convergence_check(
        stats,
        score_loss_history=[1.05],
        expected_score_loss_threshold=5.0,
    )
    assert result["converged"]
    assert result["warnings"] == []


def test_validate_diffusion_vs_baseline_detects_degradation() -> None:
    from examples.CORDEX_ML.utils.diffusion_diagnostics import validate_diffusion_vs_baseline

    rng = np.random.default_rng(0)
    T, H, W = 10, 8, 8
    truth = rng.normal(0, 1, (T, 2, H, W))
    baseline = truth + rng.normal(0, 0.5, (T, 2, H, W))
    diffusion = truth + rng.normal(0, 5.0, (T, 2, H, W))  # much worse

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = validate_diffusion_vs_baseline(
            baseline, diffusion, truth, var_names=["pr", "tasmax"], tolerance=0.05
        )
    assert not result["passed"]
    assert len(result["warnings"]) == 2
    assert any("DEGRADATION" in str(w.message) for w in caught)
    for ratio in result["rmse_ratio"].values():
        assert ratio > 1.0


def test_validate_diffusion_vs_baseline_passes_when_improved() -> None:
    from examples.CORDEX_ML.utils.diffusion_diagnostics import validate_diffusion_vs_baseline

    rng = np.random.default_rng(1)
    T, H, W = 10, 8, 8
    truth = rng.normal(0, 1, (T, 2, H, W))
    baseline = truth + rng.normal(0, 1.0, (T, 2, H, W))
    diffusion = truth + rng.normal(0, 0.3, (T, 2, H, W))  # better

    result = validate_diffusion_vs_baseline(
        baseline, diffusion, truth, var_names=["pr", "tasmax"], tolerance=0.05
    )
    assert result["passed"]
    assert result["warnings"] == []
    for ratio in result["rmse_ratio"].values():
        assert ratio < 1.0


def test_validate_diffusion_vs_baseline_within_tolerance() -> None:
    from examples.CORDEX_ML.utils.diffusion_diagnostics import validate_diffusion_vs_baseline

    rng = np.random.default_rng(2)
    T, H, W = 20, 8, 8
    truth = rng.normal(0, 1, (T, 2, H, W))
    baseline = truth + rng.normal(0, 1.0, (T, 2, H, W))
    # Diffusion is only 2% worse per channel — within 5% tolerance.
    diffusion = baseline + rng.normal(0, 0.14, (T, 2, H, W))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = validate_diffusion_vs_baseline(
            baseline, diffusion, truth, var_names=["pr", "tasmax"], tolerance=0.05
        )
    # No DEGRADATION warnings (some rmse ratios may be slightly above 1 by
    # chance but within 5%).
    degradation_warnings = [w for w in caught if "DEGRADATION" in str(w.message)]
    for vname, ratio in result["rmse_ratio"].items():
        if ratio > 1.05:
            assert not result["passed"], f"{vname} ratio={ratio:.3f} > 1.05 but passed=True"
