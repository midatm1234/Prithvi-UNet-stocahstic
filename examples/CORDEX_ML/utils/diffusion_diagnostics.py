"""Diagnostic utilities for evaluating diffusion-head ensemble outputs.

Metrics
-------
* RMSE, MAE, bias  (deterministic or ensemble mean)
* CRPS             (Continuous Ranked Probability Score)
* Spread-skill     (ensemble std vs. RMSE of ensemble mean)
* Wet-day frequency bias, Brier score for occurrence
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
        # normalise: 2/(E*(E-1)) * sum_{i<j}
        spread_term = (2.0 / (E * (E - 1))) * spread_sum
        spread_term = spread_term.reshape(*ens.shape[:-3], *ens.shape[-2:])
        spread_term = np.mean(spread_term, axis=-3)  # average over spatial if needed

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


# ---------------------------------------------------------------------------
# Aggregated report
# ---------------------------------------------------------------------------

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
