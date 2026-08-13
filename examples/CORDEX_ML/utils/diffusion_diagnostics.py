"""Diagnostic utilities for evaluating diffusion-head ensemble outputs.

Metrics
-------
* RMSE, MAE, bias and spatial correlation (deterministic or ensemble mean)
* CRPS             (Continuous Ranked Probability Score)
* Spread-skill     (ensemble std vs. RMSE of ensemble mean)
* Wet-day frequency bias, Brier score for occurrence
* Transformation-stage and true/generated residual distribution summaries
* Per-member distribution plots

All metric functions operate on numpy arrays and are intentionally
framework-agnostic.  PyTorch tensors are accepted and silently converted.

Usage example::

    from utils.diffusion_diagnostics import (
        compute_deterministic_metrics,
        compute_crps,
        compute_spread_skill,
        compute_wetday_metrics,
    )

    # truth: [T, H, W]  ensemble: [T, E, H, W]
    det  = compute_deterministic_metrics(ensemble.mean(axis=1), truth)
    crps = compute_crps(ensemble, truth)
    ss   = compute_spread_skill(ensemble, truth)
    wd   = compute_wetday_metrics(ensemble, truth, wet_threshold=0.1)
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_np(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float64)


def distribution_stats(
    values: Any,
    mask: Any | None = None,
) -> dict[str, float]:
    """Return finite-value moments and audit percentiles for any field.

    The result deliberately includes the residual-audit percentiles rather than
    silently discarding the tails where a normalization/sign error first shows
    up. Non-finite values are excluded and their count is reported.
    """
    array = _to_np(values)
    if mask is not None:
        array = array[_to_np(mask).astype(bool)]
    else:
        array = array.ravel()
    finite = np.isfinite(array)
    clean = array[finite]
    if clean.size == 0:
        raise ValueError("distribution_stats received no finite values")
    percentiles = np.percentile(clean, [1.0, 5.0, 50.0, 95.0, 99.0])
    return {
        "count": int(clean.size),
        "nonfinite_count": int(array.size - clean.size),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "rms": float(np.sqrt(np.mean(np.square(clean)))),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p95": float(percentiles[3]),
        "p99": float(percentiles[4]),
    }


def compute_spatial_correlation(
    pred: Any,
    truth: Any,
    mask: Any | None = None,
) -> float:
    """Pearson correlation between prediction and truth climatology maps.

    Inputs may be ``[T,H,W]`` fields or already-aggregated ``[H,W]`` maps.
    """
    p = _to_np(pred)
    t = _to_np(truth)
    if p.shape != t.shape:
        raise ValueError(f"pred/truth shape mismatch: {p.shape} != {t.shape}")
    if p.ndim >= 3:
        p = p.mean(axis=tuple(range(p.ndim - 2)))
        t = t.mean(axis=tuple(range(t.ndim - 2)))
    p = p.ravel()
    t = t.ravel()
    valid = np.isfinite(p) & np.isfinite(t)
    if mask is not None:
        mask_array = _to_np(mask).astype(bool)
        while mask_array.ndim > 2:
            mask_array = np.any(mask_array, axis=0)
        valid &= mask_array.ravel()
    p = p[valid]
    t = t[valid]
    if p.size < 2 or np.std(p) <= 1e-12 or np.std(t) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


# ---------------------------------------------------------------------------
# Deterministic / ensemble-mean metrics
# ---------------------------------------------------------------------------

def compute_deterministic_metrics(
    pred: Any,
    truth: Any,
    mask: Any | None = None,
) -> dict[str, float]:
    """Compute RMSE, MAE and mean bias.

    Args:
        pred:  predicted field  ``[..., H, W]`` or flat array.
        truth: target field with the same shape.
        mask:  optional boolean array; True = valid pixel.

    Returns:
        Dict with keys ``rmse``, ``mae``, ``bias``.
    """
    p = _to_np(pred).ravel()
    t = _to_np(truth).ravel()
    if mask is not None:
        m = _to_np(mask).ravel().astype(bool)
        p, t = p[m], t[m]
    diff = p - t
    return {
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "bias": float(np.mean(diff)),
    }


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------

def compute_crps(
    ensemble: Any,
    truth: Any,
    mask: Any | None = None,
) -> dict[str, float]:
    """Energy-form CRPS averaged over all valid pixels.

    Uses the identity::

        CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|

    where X, X' are iid draws from the ensemble distribution.  The second
    term is approximated via pairwise member differences.

    Args:
        ensemble: ``[T, E, H, W]`` or ``[E, H, W]`` ensemble of predictions.
        truth:    ``[T, H, W]``   or ``[H, W]``     verification values.
        mask:     optional boolean array broadcast-compatible with truth.

    Returns:
        Dict with keys ``crps_mean``, ``crps_skill`` (vs. climatological CRPS).
    """
    ens = _to_np(ensemble)  # [..., E, H, W]
    obs = _to_np(truth)     # [..., H, W]

    # normalise to [..., E, H, W]
    if ens.ndim == obs.ndim:
        # assume leading dim is members
        ens = ens[np.newaxis]  # shouldn't happen, but guard

    # |X - y| term
    obs_exp = obs[..., np.newaxis, :, :]  # [..., 1, H, W]
    abs_err = np.abs(ens - obs_exp)       # [..., E, H, W]

    E = ens.shape[-3]
    # |X - X'| term (pairwise mean via E^2 approximation)
    spread_term = 0.0
    if E > 1:
        # mean over all pairs i≠j; equals var * 2 * (E-1)/E for sorted samples
        ens_flat = ens.reshape(*ens.shape[:-3], E, -1)  # [..., E, H*W]
        spread_sum = 0.0
        for i in range(E):
            for j in range(i + 1, E):
                spread_sum = spread_sum + np.abs(ens_flat[..., i, :] - ens_flat[..., j, :])
        # Empirical-distribution expectation over all E^2 ordered pairs,
        # including the zero self-pairs:
        # E|X-X'| = (2/E^2) * sum_{i<j}|x_i-x_j|.
        spread_term = (2.0 / (E * E)) * spread_sum
        spread_term = spread_term.reshape(*ens.shape[:-3], *ens.shape[-2:])

    crps_field = np.mean(abs_err, axis=-3) - 0.5 * (spread_term if E > 1 else 0.0)

    if mask is not None:
        m = _to_np(mask).astype(bool)
        crps_vals = crps_field[m]
        obs_vals = obs[m]
    else:
        crps_vals = crps_field.ravel()
        obs_vals = obs.ravel()

    crps_mean = float(np.mean(crps_vals))

    # Climatological CRPS: single-member prediction = observation mean
    clim_mean = float(np.mean(obs_vals))
    clim_crps = float(np.mean(np.abs(obs_vals - clim_mean)))
    crps_skill = 1.0 - crps_mean / clim_crps if clim_crps > 1e-12 else float("nan")

    return {"crps_mean": crps_mean, "crps_skill": crps_skill}


# ---------------------------------------------------------------------------
# Spread-skill
# ---------------------------------------------------------------------------

def compute_spread_skill(
    ensemble: Any,
    truth: Any,
    mask: Any | None = None,
) -> dict[str, float]:
    """Compute ensemble spread vs. RMSE of the ensemble mean.

    A well-calibrated ensemble has spread ≈ RMSE (ratio ≈ 1).  Spread > RMSE
    indicates over-dispersion; spread < RMSE indicates under-dispersion.

    Args:
        ensemble: ``[T, E, H, W]`` or ``[E, H, W]``.
        truth:    ``[T, H, W]``   or ``[H, W]``.
        mask:     optional boolean mask.

    Returns:
        Dict with ``spread_mean``, ``rmse_ensmean``, ``spread_skill_ratio``.
    """
    ens = _to_np(ensemble)
    obs = _to_np(truth)

    ens_mean = ens.mean(axis=-3)     # [..., H, W]
    ens_std  = ens.std(axis=-3, ddof=1)  # unbiased spread

    diff = ens_mean - obs
    if mask is not None:
        m = _to_np(mask).astype(bool)
        spread_mean = float(np.mean(ens_std[m]))
        rmse = float(np.sqrt(np.mean(diff[m] ** 2)))
    else:
        spread_mean = float(np.mean(ens_std))
        rmse = float(np.sqrt(np.mean(diff ** 2)))

    ratio = spread_mean / rmse if rmse > 1e-12 else float("nan")
    return {
        "spread_mean": spread_mean,
        "rmse_ensmean": rmse,
        "spread_skill_ratio": ratio,
    }


# ---------------------------------------------------------------------------
# Wet-day frequency and Brier score
# ---------------------------------------------------------------------------

def compute_wetday_metrics(
    ensemble: Any,
    truth: Any,
    wet_threshold: float = 0.1,
    mask: Any | None = None,
) -> dict[str, float]:
    """Wet-day frequency bias and Brier score for precipitation occurrence.

    Args:
        ensemble:      ``[T, E, H, W]`` precipitation ensemble in physical units.
        truth:         ``[T, H, W]``   observed precipitation.
        wet_threshold: mm/day threshold for wet classification.
        mask:          optional boolean mask.

    Returns:
        Dict with:
        * ``obs_wet_freq``   – fraction of wet obs pixels.
        * ``ens_wet_freq``   – fraction of ensemble members that are wet (mean over members).
        * ``wet_freq_bias``  – ens_wet_freq - obs_wet_freq.
        * ``brier_score``    – Brier score for occurrence probability.
        * ``brier_skill``    – vs. climatological baseline.
    """
    ens = _to_np(ensemble)  # [..., E, H, W]
    obs = _to_np(truth)     # [..., H, W]

    obs_wet = (obs > wet_threshold).astype(np.float64)
    ens_wet = (ens > wet_threshold).astype(np.float64)
    prob_wet = ens_wet.mean(axis=-3)  # [..., H, W]

    if mask is not None:
        m = _to_np(mask).astype(bool)
        obs_wet_flat = obs_wet[m]
        prob_wet_flat = prob_wet[m]
    else:
        obs_wet_flat = obs_wet.ravel()
        prob_wet_flat = prob_wet.ravel()

    obs_freq = float(np.mean(obs_wet_flat))
    ens_freq = float(np.mean(prob_wet_flat))
    brier = float(np.mean((prob_wet_flat - obs_wet_flat) ** 2))
    # Climatological baseline: always predict climate wet fraction
    brier_clim = float(obs_freq * (1.0 - obs_freq))
    brier_skill = 1.0 - brier / brier_clim if brier_clim > 1e-12 else float("nan")

    return {
        "obs_wet_freq": obs_freq,
        "ens_wet_freq": ens_freq,
        "wet_freq_bias": ens_freq - obs_freq,
        "brier_score": brier,
        "brier_skill": brier_skill,
    }


# ---------------------------------------------------------------------------
# Climatological bias map
# ---------------------------------------------------------------------------

def compute_climatological_bias(
    pred: Any,
    truth: Any,
) -> np.ndarray:
    """Compute the mean bias map (pred_mean - truth_mean) over all samples.

    Args:
        pred:  ``[T, H, W]`` predicted climatology samples (or ensemble mean).
        truth: ``[T, H, W]`` observed samples.

    Returns:
        ``[H, W]`` bias map.
    """
    p = _to_np(pred)
    t = _to_np(truth)
    return p.mean(axis=0) - t.mean(axis=0)


def residual_distribution_report(
    true_residual: Any,
    generated_residual: Any,
    full_target: Any,
    *,
    representative_points: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Compare true and generated residuals without changing either field.

    Arrays are expected in the same space (normally standardized target space,
    or both in physical units). The spatial climatologies remain arrays so a
    caller can save maps; all other entries are JSON-serializable scalars/lists.
    """
    true = _to_np(true_residual)
    generated = _to_np(generated_residual)
    target = _to_np(full_target)
    if true.shape != generated.shape or true.shape != target.shape:
        raise ValueError(
            "true_residual, generated_residual, and full_target must share a shape; "
            f"got {true.shape}, {generated.shape}, {target.shape}"
        )
    temporal_axes = tuple(range(max(0, true.ndim - 2)))
    true_std = float(np.nanstd(true))
    target_std = float(np.nanstd(target))
    generated_std = float(np.nanstd(generated))
    points: dict[str, Any] = {}
    for row, col in representative_points or []:
        if not (0 <= row < true.shape[-2] and 0 <= col < true.shape[-1]):
            raise IndexError(
                f"representative point {(row, col)} is outside spatial shape {true.shape[-2:]}"
            )
        points[f"{row},{col}"] = {
            "true_temporal_mean": float(np.nanmean(true[..., row, col])),
            "generated_temporal_mean": float(np.nanmean(generated[..., row, col])),
            "target_temporal_mean": float(np.nanmean(target[..., row, col])),
        }
    return {
        "true": distribution_stats(true),
        "generated": distribution_stats(generated),
        "full_target": distribution_stats(target),
        "true_residual_to_target_std_ratio": (
            true_std / target_std if target_std > 1e-12 else float("nan")
        ),
        "generated_to_true_residual_std_ratio": (
            generated_std / true_std if true_std > 1e-12 else float("nan")
        ),
        "true_spatial_climatology": (
            np.nanmean(true, axis=temporal_axes) if temporal_axes else true.copy()
        ),
        "generated_spatial_climatology": (
            np.nanmean(generated, axis=temporal_axes) if temporal_axes else generated.copy()
        ),
        "representative_points": points,
    }


def transformation_stage_report(
    *,
    baseline_physical: Any,
    truth_physical: Any,
    true_residual_normalized: Any,
    generated_residual_normalized: Any,
    final_physical: Any,
) -> dict[str, dict[str, float]]:
    """Summarize every residual transformation stage requested by the audit.

    Physical residuals are calculated by subtraction in physical space. This is
    intentional: decoding a residual by itself would incorrectly restore the
    absolute target mean (and is invalid for nonlinear target transforms).
    """
    baseline = _to_np(baseline_physical)
    truth = _to_np(truth_physical)
    final = _to_np(final_physical)
    if baseline.shape != truth.shape or baseline.shape != final.shape:
        raise ValueError(
            "physical baseline, truth, and final prediction must share a shape; "
            f"got {baseline.shape}, {truth.shape}, {final.shape}"
        )
    true_residual_physical = truth - baseline
    generated_residual_physical = final - baseline
    stages = {
        "physical_unet_prediction": baseline,
        "physical_ground_truth": truth,
        "physical_true_residual": true_residual_physical,
        "normalized_true_residual": true_residual_normalized,
        "predicted_normalized_residual": generated_residual_normalized,
        "denormalized_predicted_residual": generated_residual_physical,
        "final_physical_prediction": final,
    }
    return {name: distribution_stats(value) for name, value in stages.items()}


# ---------------------------------------------------------------------------
# Aggregated report
# ---------------------------------------------------------------------------

def checkpoint_convergence_check(
    residual_stats: dict[str, Any],
    score_loss_history: list[float] | None = None,
    *,
    expected_score_loss_threshold: float = 5.0,
    expected_rms_max: float = 3.0,
) -> dict[str, Any]:
    """Evaluate whether a diffusion-head checkpoint has converged.

    A well-converged epsilon-MSE score model should have loss ≈ 1.0 at
    convergence.  Training-residual RMS should be O(1) in standardized space.
    This function surfaces red flags that indicate the checkpoint is not yet
    suitable for deployment — the primary culprit behind catastrophic inference
    noise is running inference with an unconverged score model.

    Args:
        residual_stats: dict from ``DiffusionHead.residual_training_stats()``,
            keys ``count``, ``mean``, ``std``, ``rms``, ``min``, ``max``.
        score_loss_history: optional list of recent score-matching loss values
            (from ``train_loss_history`` in checkpoint metadata).
        expected_score_loss_threshold: warn if the most recent score loss
            exceeds this value.  At epsilon-MSE convergence, loss ≈ 1.0.
        expected_rms_max: warn if any training-residual RMS channel exceeds
            this threshold (normalized units).

    Returns:
        Dict with keys:
        * ``converged``: bool, False if any check fails.
        * ``warnings``: list of warning strings describing failed checks.
        * ``score_loss_recent``: float or None.
        * ``residual_rms``: per-channel RMS list.
        * ``residual_count``: total sample count seen during training.
    """
    import warnings as _warnings

    msgs: list[str] = []

    score_loss_recent: float | None = None
    if score_loss_history:
        score_loss_recent = float(score_loss_history[-1])
        if score_loss_recent > expected_score_loss_threshold:
            msgs.append(
                f"Score-matching loss={score_loss_recent:.2f} >> expected ~1.0 at "
                "convergence. The epsilon network is likely near its zero-initialized "
                "output. DDIM will amplify prior noise by 1/alpha_T (~150x for "
                "beta_max=20), producing catastrophic physical-space corrections. "
                "Do not use this checkpoint for residual correction until training "
                f"has reduced the score loss below {expected_score_loss_threshold:.1f}."
            )

    rms_list: list[float] = []
    count_val: float = 0.0
    if residual_stats:
        count_val = float(_to_np(residual_stats.get("count", 0.0)).ravel()[0])
        rms_arr = _to_np(residual_stats.get("rms", np.array([])))
        rms_list = rms_arr.ravel().tolist()
        for ch_idx, rms_val in enumerate(rms_list):
            if rms_val > expected_rms_max:
                msgs.append(
                    f"Training residual RMS channel={ch_idx}: {rms_val:.3f} > "
                    f"{expected_rms_max:.1f} (normalized units). Large training "
                    "residuals indicate the U-Net baseline is systematically biased "
                    "or the target scaler is mis-configured."
                )
        if count_val < 1e6:
            msgs.append(
                f"Training residual statistics cover only {count_val:.0f} samples. "
                "The guard thresholds may not yet be reliable; accumulate more "
                "training steps before deployment."
            )

    converged = len(msgs) == 0
    for msg in msgs:
        _warnings.warn("DIFFUSION CONVERGENCE CHECK: " + msg, stacklevel=2)

    return {
        "converged": converged,
        "warnings": msgs,
        "score_loss_recent": score_loss_recent,
        "residual_rms": rms_list,
        "residual_count": count_val,
    }


def validate_diffusion_vs_baseline(
    baseline_pred: Any,
    diffusion_pred: Any,
    truth: Any,
    *,
    var_names: list[str] | None = None,
    tolerance: float = 0.05,
    emit_warning: bool = True,
) -> dict[str, Any]:
    """Compare diffusion-corrected predictions against the U-Net deterministic baseline.

    The primary deployment criterion is that the diffusion head must not
    increase RMSE beyond the deterministic baseline by more than ``tolerance``.
    This function computes per-variable metrics for both prediction sets and
    returns a structured report.  It emits a prominent warning when the
    diffusion head degrades any variable.

    Args:
        baseline_pred: U-Net deterministic predictions ``[T, V, H, W]`` or
            ``[V, H, W]`` in physical units.
        diffusion_pred: diffusion-corrected predictions with the same shape.
        truth: ground-truth observations with the same shape.
        var_names: optional list of variable names (length = V); defaults to
            ``['var_0', 'var_1', ...]``.
        tolerance: fraction of baseline RMSE by which diffusion is allowed to
            be worse before triggering a warning.  E.g. 0.05 means diffusion
            RMSE must be ≤ 1.05 × baseline RMSE.
        emit_warning: if True, emit a Python warning when the check fails.

    Returns:
        Dict with keys:
        * ``passed``: bool, True iff all variables satisfy the criterion.
        * ``warnings``: list of per-variable warning strings.
        * ``baseline``: per-variable metric dicts.
        * ``diffusion``: per-variable metric dicts.
        * ``rmse_ratio``: diffusion_rmse / baseline_rmse per variable.
    """
    import warnings as _warnings

    bl = _to_np(baseline_pred)
    diff = _to_np(diffusion_pred)
    obs = _to_np(truth)

    if bl.shape != diff.shape or bl.shape != obs.shape:
        raise ValueError(
            f"baseline, diffusion, truth must share a shape; "
            f"got {bl.shape}, {diff.shape}, {obs.shape}"
        )

    # Support [T, V, H, W] or [V, H, W]; treat leading dims as time/batch.
    if bl.ndim == 3:
        bl = bl[np.newaxis]
        diff = diff[np.newaxis]
        obs = obs[np.newaxis]

    n_vars = bl.shape[1]
    if var_names is None:
        var_names = [f"var_{i}" for i in range(n_vars)]
    if len(var_names) != n_vars:
        raise ValueError(
            f"var_names length {len(var_names)} != n_vars {n_vars}"
        )

    baseline_metrics: dict[str, dict[str, float]] = {}
    diffusion_metrics: dict[str, dict[str, float]] = {}
    rmse_ratios: dict[str, float] = {}
    msgs: list[str] = []

    for vi, vname in enumerate(var_names):
        bl_v = bl[:, vi]
        di_v = diff[:, vi]
        ob_v = obs[:, vi]
        bm = compute_deterministic_metrics(bl_v, ob_v)
        dm = compute_deterministic_metrics(di_v, ob_v)
        bm["spatial_correlation"] = compute_spatial_correlation(bl_v, ob_v)
        dm["spatial_correlation"] = compute_spatial_correlation(di_v, ob_v)
        baseline_metrics[vname] = bm
        diffusion_metrics[vname] = dm
        bl_rmse = bm["rmse"]
        di_rmse = dm["rmse"]
        ratio = di_rmse / bl_rmse if bl_rmse > 1e-12 else float("nan")
        rmse_ratios[vname] = ratio
        threshold = 1.0 + tolerance
        if np.isfinite(ratio) and ratio > threshold:
            msgs.append(
                f"Variable '{vname}': diffusion RMSE={di_rmse:.4f} > "
                f"{threshold:.2f} × baseline RMSE={bl_rmse:.4f} "
                f"(ratio={ratio:.3f}, tolerance={tolerance:.0%}). "
                "The diffusion correction is DEGRADING this variable. "
                "Consider reducing residual_application_scale or increasing "
                "training until score_loss ≈ 1."
            )

    passed = len(msgs) == 0
    if emit_warning:
        for msg in msgs:
            _warnings.warn("DIFFUSION DEGRADATION WARNING: " + msg, stacklevel=2)

    return {
        "passed": passed,
        "warnings": msgs,
        "baseline": baseline_metrics,
        "diffusion": diffusion_metrics,
        "rmse_ratio": rmse_ratios,
    }


def full_diagnostics_report(
    ensemble: Any,
    truth: Any,
    var_name: str = "variable",
    wet_threshold: float = 0.1,
    is_precip: bool = False,
    mask: Any | None = None,
) -> dict[str, Any]:
    """Run all diagnostics and return a flat dict of metric values.

    Args:
        ensemble:      ``[T, E, H, W]`` ensemble predictions.
        truth:         ``[T, H, W]``    verification observations.
        var_name:      variable name (used to prefix keys).
        wet_threshold: precipitation wet-day threshold in physical units.
        is_precip:     if True, also compute wet-day metrics.
        mask:          optional boolean mask.

    Returns:
        Dict with all metric values prefixed by ``var_name``.
    """
    ens = _to_np(ensemble)
    obs = _to_np(truth)
    ens_mean = ens.mean(axis=-3)

    det = compute_deterministic_metrics(ens_mean, obs, mask=mask)
    det["spatial_correlation"] = compute_spatial_correlation(
        ens_mean, obs, mask=mask
    )
    crps = compute_crps(ens, obs, mask=mask)
    ss = compute_spread_skill(ens, obs, mask=mask)

    report: dict[str, Any] = {}
    prefix = f"{var_name}."
    for k, v in {**det, **crps, **ss}.items():
        report[prefix + k] = v

    if is_precip:
        wd = compute_wetday_metrics(ens, obs, wet_threshold=wet_threshold, mask=mask)
        for k, v in wd.items():
            report[prefix + k] = v

    return report
