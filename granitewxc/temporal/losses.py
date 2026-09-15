"""Temporal objectives for sequence-conditioned downscaling.

These sit *on top of* the existing per-frame spatial loss; they do not replace
it. The per-frame term remains the dominant weight, because the thing that must
not regress is spatial accuracy.

The central design rule: **every temporal term matches an observed statistic.**
None of them penalizes variability per se.

The distinction matters and is easy to get wrong. A term like ``|Δŷ|`` (penalize
the model's own day-to-day change) is a smoothness prior: its minimum is a
constant field, so it actively destroys fronts, heat-wave onsets and storm
timing while reporting a lower loss. The term implemented here is
``|Δŷ − Δy|`` -- match the *observed* change. Its minimum is the correct
evolution, and it is minimized to zero by a perfect forecast rather than by a
flat one. :func:`tendency_loss` asserts this by construction, and
``tests/test_temporal_losses.py::test_tendency_loss_does_not_prefer_constant``
pins it: a constant-output model must score *worse* than the truth.

Similar care elsewhere:

* Precipitation accumulation matches ``k``-day totals, which is a distinct
  statistic from the daily field and is what droughts and floods depend on.
* Occurrence matching uses the wet/dry *transition* probabilities, so the model
  is rewarded for correct spell structure rather than for producing more drizzle.
  Increasing light rain everywhere improves the marginal wet-day frequency but
  worsens ``P(dry|wet)``, so this term resists that failure mode.
* Lag autocorrelation is computed on deseasonalized anomalies and is gated by a
  minimum-sample count, because an autocorrelation estimated from a handful of
  frames is noise and optimizing it would inject noise into the gradient.

A note on what is deliberately *absent*: no unweighted full-spectrum FFT loss.
By Parseval's theorem, the squared error of the complex Fourier coefficients
equals the spatial squared error up to a constant factor, so such a term is the
existing MSE under another name and cannot be an independent "anti-blurring"
mechanism. A spectral term only adds information if it is wavenumber-weighted or
restricted to a band; the existing multiscale and spatial-gradient losses in
``granitewxc/models/loss.py`` already target that scale-selectivity directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from granitewxc.temporal.config import TemporalLossConfig

__all__ = [
    "TemporalLossTerms",
    "TemporalLossComputer",
    "tendency_loss",
    "accumulation_loss",
    "occurrence_persistence_loss",
    "lag_autocorrelation_loss",
    "tmax_tmin_consistency_loss",
]

_EPS = 1.0e-8


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean over valid entries; returns 0 when nothing is valid."""
    if mask is None:
        return values.mean()
    mask = mask.to(values.dtype)
    total = mask.sum()
    if float(total) <= 0.0:
        return values.sum() * 0.0
    return (values * mask).sum() / total.clamp_min(1.0)


def _channel_index(names: Sequence[str], wanted: str) -> int:
    lowered = [str(n).lower() for n in names]
    target = str(wanted).lower()
    if target in lowered:
        return lowered.index(target)
    return -1


# ---------------------------------------------------------------------------
# individual terms
# ---------------------------------------------------------------------------
def tendency_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    interval_ratio: torch.Tensor | None = None,
    channel_weights: torch.Tensor | None = None,
    normalize_by_interval: bool = True,
) -> torch.Tensor:
    """Match the observed day-to-day change.

    ``pred``/``target`` are ``[B, T, C, H, W]``. Returns
    ``mean_valid | (Δpred − Δtarget) / Δt |``.

    This is **not** a smoothness penalty. The quantity being driven to zero is
    the *error in the tendency*, so the loss is minimized by reproducing the real
    evolution -- including sharp frontal transitions -- and a constant field
    scores badly whenever the truth changes.

    A difference is valid only if *both* of its endpoints are valid, so a missing
    day removes the two differences that touch it rather than being read as a
    jump to zero.
    """
    if pred.shape[1] < 2:
        return pred.sum() * 0.0
    d_pred = pred[:, 1:] - pred[:, :-1]
    d_target = target[:, 1:] - target[:, :-1]

    if mask is None:
        pair_mask = None
    else:
        pair_mask = mask[:, 1:] & mask[:, :-1]

    error = (d_pred - d_target).abs()

    if normalize_by_interval and interval_ratio is not None:
        # interval_ratio[:, t] is dt into frame t; the difference t-1 -> t spans
        # that interval. A two-day step must not be charged as a one-day change.
        if interval_ratio.shape[1] != pred.shape[1]:
            raise ValueError(
                f"interval_ratio has {interval_ratio.shape[1]} frames but pred has "
                f"{pred.shape[1]}. Pass the ratio restricted to the EMITTED frames "
                "(SequenceOutput.interval_ratio), not the whole context window."
            )
        ratio = interval_ratio[:, 1:].clamp_min(_EPS)
        error = error / ratio.view(ratio.shape[0], ratio.shape[1], 1, 1, 1)

    if channel_weights is not None:
        error = error * channel_weights.view(1, 1, -1, 1, 1)
    return _masked_mean(error, pair_mask)


def accumulation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    windows: Sequence[int],
    channels: Sequence[int],
    reduce: str = "sum",
) -> torch.Tensor:
    """Match multi-day running totals (precipitation) or means (temperature).

    Multi-day accumulation is a different statistic from the daily field: a model
    can have a good daily RMSE and still misplace weekly totals by mistiming
    events. A window is scored only where every day inside it is valid.
    """
    if not windows or not channels:
        return pred.sum() * 0.0
    total = pred.sum() * 0.0
    counted = 0
    idx = torch.as_tensor(list(channels), device=pred.device, dtype=torch.long)
    p = pred.index_select(2, idx)
    t = target.index_select(2, idx)
    m = None if mask is None else mask.index_select(2, idx)

    for window in windows:
        w = int(window)
        if w < 2 or p.shape[1] < w:
            continue
        # Cumulative-sum trick over the time axis for an exact rolling window.
        def roll(x: torch.Tensor) -> torch.Tensor:
            c = torch.cumsum(x, dim=1)
            return c[:, w - 1 :] - torch.cat(
                [torch.zeros_like(c[:, :1]), c[:, : -w]], dim=1
            )

        p_acc = roll(p)
        t_acc = roll(t)
        if m is None:
            win_mask = None
        else:
            valid = roll(m.to(p.dtype))
            win_mask = valid >= (w - 1.0e-6)
        if reduce == "mean":
            p_acc, t_acc = p_acc / w, t_acc / w
        total = total + _masked_mean((p_acc - t_acc).abs(), win_mask)
        counted += 1
    return total / max(counted, 1)


def occurrence_persistence_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    channel: int,
    wet_threshold: float,
    softness: float = 0.5,
    transition_weight: float = 0.5,
) -> torch.Tensor:
    """Match wet/dry occurrence *and* wet/dry transition probabilities.

    Occurrence is made differentiable with a soft indicator
    ``p = sigmoid((y − thr) / softness)`` in physical units (mm/day), so the
    threshold means what it says.

    Two parts:

    ``marginal``  -- match ``P(wet)``.
    ``transition``-- match ``P(wet_t | wet_{t-1})`` and ``P(wet_t | dry_{t-1})``.

    The transition part is the one that matters here. Sprinkling extra drizzle
    raises ``P(wet)`` toward the truth while *destroying* the persistence
    structure -- it pushes ``P(wet|dry)`` far too high. Because both parts are
    scored, that shortcut increases the loss instead of hiding inside it.
    """
    p_pred = torch.sigmoid((pred[:, :, channel] - wet_threshold) / max(softness, _EPS))
    p_true = torch.sigmoid((target[:, :, channel] - wet_threshold) / max(softness, _EPS))
    m = None if mask is None else mask[:, :, channel]

    marginal = (_masked_mean(p_pred, m) - _masked_mean(p_true, m)).abs()

    if pred.shape[1] < 2:
        return marginal

    a_pred, b_pred = p_pred[:, :-1], p_pred[:, 1:]
    a_true, b_true = p_true[:, :-1], p_true[:, 1:]
    pair = None if m is None else (m[:, :-1] & m[:, 1:])

    def conditional(prev: torch.Tensor, nxt: torch.Tensor, wet_prev: bool) -> torch.Tensor:
        weight = prev if wet_prev else (1.0 - prev)
        if pair is not None:
            weight = weight * pair.to(weight.dtype)
        denom = weight.sum().clamp_min(_EPS)
        return (weight * nxt).sum() / denom

    trans = torch.zeros((), device=pred.device, dtype=pred.dtype)
    for wet_prev in (True, False):
        trans = trans + (
            conditional(a_pred, b_pred, wet_prev) - conditional(a_true, b_true, wet_prev)
        ).abs()
    return marginal + float(transition_weight) * trans


def lag_autocorrelation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    lags: Sequence[int],
    channel_weights: torch.Tensor | None = None,
    min_samples: int = 64,
) -> torch.Tensor:
    """Match lag-``k`` autocorrelation of temporal anomalies.

    Anomalies are formed by removing each grid cell's mean over the window,
    which removes the seasonal/climatological component that would otherwise
    dominate a short window's correlation.

    Returns exactly zero when fewer than ``min_samples`` valid pairs are
    available. An autocorrelation from a handful of frames is noise, and
    back-propagating that noise would be worse than omitting the term.
    """
    if pred.shape[1] < 2:
        return pred.sum() * 0.0

    p = pred - pred.mean(dim=1, keepdim=True)
    t = target - target.mean(dim=1, keepdim=True)

    total = pred.sum() * 0.0
    counted = 0
    for lag in lags:
        k = int(lag)
        if k < 1 or p.shape[1] <= k:
            continue
        pair = None if mask is None else (mask[:, k:] & mask[:, :-k])
        if pair is not None and int(pair.sum()) < int(min_samples):
            continue

        def corr(x: torch.Tensor) -> torch.Tensor:
            a, b = x[:, k:], x[:, :-k]
            if pair is None:
                num = (a * b).mean(dim=(0, 1))
                den = (a.pow(2).mean(dim=(0, 1)) * b.pow(2).mean(dim=(0, 1))).sqrt()
            else:
                w = pair.to(x.dtype)
                denom_w = w.sum(dim=(0, 1)).clamp_min(1.0)
                num = (a * b * w).sum(dim=(0, 1)) / denom_w
                den = (
                    (a.pow(2) * w).sum(dim=(0, 1))
                    / denom_w
                    * (b.pow(2) * w).sum(dim=(0, 1))
                    / denom_w
                ).sqrt()
            return num / den.clamp_min(_EPS)

        diff = (corr(p) - corr(t)).abs()  # [C, H, W]
        if channel_weights is not None:
            diff = diff * channel_weights.view(-1, 1, 1)
        total = total + diff.mean()
        counted += 1
    return total / max(counted, 1)


def tmax_tmin_consistency_loss(
    pred: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    tmax_channel: int,
    tmin_channel: int,
) -> torch.Tensor:
    """Penalize ``tmin > tmax`` violations.

    Applies only when both variables are configured (the NARR/PRISM case has
    ``ppt/tmax/tmin``; the SA case has ``pr/tasmax`` and this term is skipped).
    Only the violating side is penalized, so a physically consistent pair
    contributes exactly zero and the term cannot pull the two fields together.
    """
    if tmax_channel < 0 or tmin_channel < 0:
        return pred.sum() * 0.0
    violation = F.relu(pred[:, :, tmin_channel] - pred[:, :, tmax_channel])
    m = None
    if mask is not None:
        m = mask[:, :, tmax_channel] & mask[:, :, tmin_channel]
    return _masked_mean(violation, m)


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------
@dataclass
class TemporalLossTerms:
    """Individual temporal terms and their weighted total."""

    per_frame: torch.Tensor
    tendency: torch.Tensor
    accumulation: torch.Tensor
    occurrence: torch.Tensor
    lag_autocorr: torch.Tensor
    tmax_tmin: torch.Tensor
    total: torch.Tensor

    def to_log(self) -> dict[str, float]:
        return {
            "per_frame": float(self.per_frame.detach()),
            "tendency": float(self.tendency.detach()),
            "accumulation": float(self.accumulation.detach()),
            "occurrence": float(self.occurrence.detach()),
            "lag_autocorr": float(self.lag_autocorr.detach()),
            "tmax_tmin": float(self.tmax_tmin.detach()),
            "total": float(self.total.detach()),
        }


class TemporalLossComputer:
    """Assemble the temporal loss for a batch of sequences.

    ``variable_scales`` divides each variable's residual by a fixed
    characteristic magnitude before weighting, so precipitation in mm/day and
    temperature in kelvin contribute comparably and one variable cannot dominate
    purely because of its units. The scales come from the training-set target
    standard deviations (i.e. from the same scalers the model was fitted with),
    never from the current batch, so the objective does not drift between
    batches.
    """

    def __init__(
        self,
        cfg: TemporalLossConfig,
        *,
        output_vars: Sequence[str],
        variable_scales: Mapping[str, float] | None = None,
        device: torch.device | str = "cpu",
    ):
        self.cfg = cfg
        self.output_vars = [str(v) for v in output_vars]
        self.n_channels = len(self.output_vars)

        scales = []
        for name in self.output_vars:
            value = float((variable_scales or {}).get(name, 1.0))
            scales.append(value if value > 0 else 1.0)
        self.variable_scales = torch.tensor(scales, dtype=torch.float32, device=device)

        weights = []
        for name in self.output_vars:
            weights.append(float(cfg.predictand_weights.get(name, 1.0)))
        self.channel_weights = torch.tensor(weights, dtype=torch.float32, device=device)

        tend = []
        for name in self.output_vars:
            tend.append(float(cfg.tendency.predictands.get(name, 1.0)))
        self.tendency_weights = torch.tensor(tend, dtype=torch.float32, device=device)

        lag = []
        for name in self.output_vars:
            lag.append(float(cfg.lag_autocorr.predictands.get(name, 1.0)))
        self.lag_weights = torch.tensor(lag, dtype=torch.float32, device=device)

        self.accum_channels = [
            i
            for i, name in enumerate(self.output_vars)
            if name in {str(p) for p in cfg.accumulation.predictands}
        ]
        self.precip_channel = _channel_index(self.output_vars, cfg.occurrence.variable)
        self.tmax_channel = _channel_index(self.output_vars, "tmax")
        if self.tmax_channel < 0:
            self.tmax_channel = _channel_index(self.output_vars, "tasmax")
        self.tmin_channel = _channel_index(self.output_vars, "tmin")
        if self.tmin_channel < 0:
            self.tmin_channel = _channel_index(self.output_vars, "tasmin")

        if cfg.occurrence.enabled and self.precip_channel < 0:
            raise ValueError(
                f"temporal.losses.occurrence.enabled=true but variable "
                f"{cfg.occurrence.variable!r} is not among output_vars {self.output_vars}."
            )
        if cfg.accumulation.enabled and not self.accum_channels:
            raise ValueError(
                "temporal.losses.accumulation.enabled=true but none of "
                f"{list(cfg.accumulation.predictands)} is among output_vars {self.output_vars}."
            )

    def to(self, device: torch.device | str) -> "TemporalLossComputer":
        self.variable_scales = self.variable_scales.to(device)
        self.channel_weights = self.channel_weights.to(device)
        self.tendency_weights = self.tendency_weights.to(device)
        self.lag_weights = self.lag_weights.to(device)
        return self

    def _rescale(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.variable_scales.view(1, 1, -1, 1, 1).to(x.dtype)

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        interval_ratio: torch.Tensor | None = None,
        per_frame_loss: torch.Tensor | None = None,
    ) -> TemporalLossTerms:
        """Compute all terms. ``pred``/``target`` are ``[B, T, C, H, W]``, physical units."""
        if pred.shape != target.shape:
            raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} differ.")
        cfg = self.cfg
        device, dtype = pred.device, pred.dtype
        zero = torch.zeros((), device=device, dtype=dtype)

        # Scale-free copies for terms that mix variables.
        p_s, t_s = self._rescale(pred), self._rescale(target)

        if per_frame_loss is None:
            err = (p_s - t_s).abs() * self.channel_weights.view(1, 1, -1, 1, 1).to(dtype)
            per_frame = _masked_mean(err, mask)
        else:
            per_frame = per_frame_loss

        tendency = zero
        if cfg.tendency.enabled and cfg.tendency.weight > 0:
            tendency = tendency_loss(
                p_s,
                t_s,
                mask,
                interval_ratio=interval_ratio,
                channel_weights=self.tendency_weights.to(dtype),
                normalize_by_interval=cfg.tendency.normalize_by_interval,
            )

        accumulation = zero
        if cfg.accumulation.enabled and cfg.accumulation.weight > 0:
            accumulation = accumulation_loss(
                p_s,
                t_s,
                mask,
                windows=cfg.accumulation.windows,
                channels=self.accum_channels,
                reduce="sum",
            )

        occurrence = zero
        if cfg.occurrence.enabled and cfg.occurrence.weight > 0:
            # Occurrence works in *physical* units: the wet threshold is mm/day.
            occurrence = occurrence_persistence_loss(
                pred,
                target,
                mask,
                channel=self.precip_channel,
                wet_threshold=cfg.occurrence.wet_threshold,
                softness=cfg.occurrence.softness,
                transition_weight=cfg.occurrence.transition_weight,
            )

        lag_autocorr = zero
        if cfg.lag_autocorr.enabled and cfg.lag_autocorr.weight > 0:
            lag_autocorr = lag_autocorrelation_loss(
                p_s,
                t_s,
                mask,
                lags=cfg.lag_autocorr.lags,
                channel_weights=self.lag_weights.to(dtype),
                min_samples=cfg.lag_autocorr.min_samples,
            )

        tmax_tmin = zero
        if self.tmax_channel >= 0 and self.tmin_channel >= 0:
            tmax_tmin = tmax_tmin_consistency_loss(
                pred, mask, tmax_channel=self.tmax_channel, tmin_channel=self.tmin_channel
            )

        total = (
            float(cfg.per_frame_weight) * per_frame
            + float(cfg.tendency.weight) * tendency
            + float(cfg.accumulation.weight) * accumulation
            + float(cfg.occurrence.weight) * occurrence
            + float(cfg.lag_autocorr.weight) * lag_autocorr
            + tmax_tmin
        )
        return TemporalLossTerms(
            per_frame=per_frame,
            tendency=tendency,
            accumulation=accumulation,
            occurrence=occurrence,
            lag_autocorr=lag_autocorr,
            tmax_tmin=tmax_tmin,
            total=total,
        )
