"""Chunked, state-carrying inference over long continuous sequences.

The contract this module implements:

* **Every date is written exactly once.** Windows overlap during training; for
  inference the run is decomposed into contiguous chunks that tile each run
  without overlap, so no date is produced twice and evaluation cannot
  double-count.
* **State is carried across chunk boundaries**, so the chunked result equals the
  single-pass result exactly rather than approximately. The equality is asserted
  by ``tests/test_temporal_model.py::test_chunked_inference_matches_single_pass``
  and can be re-checked on real data with :func:`verify_chunk_consistency`.
* **State is reset at run boundaries**, because a run boundary is a genuine
  discontinuity (a missing date or a period concatenation) and carrying memory
  across it would assert continuity the data does not have.
* **No observed target ever enters the model.** Targets are read only to be
  written alongside the prediction for evaluation, and the loader keeps them in a
  separate tensor from ``x``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from granitewxc.temporal.calendar import time_key
from granitewxc.temporal.config import TemporalConfig
from granitewxc.temporal.model import TemporalSequenceModel
from granitewxc.temporal.sequence_dataset import (
    ContiguousRun,
    FrameRef,
    TemporalSequenceDataset,
    build_runs,
    collate_sequences,
    date_in_range,
)

__all__ = [
    "InferenceResult",
    "run_sequence_inference",
    "verify_chunk_consistency",
    "write_netcdf",
]


@dataclass
class InferenceResult:
    """Predictions for one continuous run."""

    run_id: int
    dates: list[tuple[int, int, int, int, int, int]]
    predictions: np.ndarray  # [T, C, H, W]
    targets: np.ndarray | None
    valid_mask: np.ndarray | None
    seam_indices: list[int] = field(default_factory=list)
    lat: np.ndarray | None = None
    lon: np.ndarray | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "n_dates": len(self.dates),
            "first": self.dates[0] if self.dates else None,
            "last": self.dates[-1] if self.dates else None,
            "shape": list(self.predictions.shape),
            "seam_indices": self.seam_indices,
        }


def _chunk_batch(
    dataset: TemporalSequenceDataset,
    source,
    frames: Sequence[FrameRef],
    *,
    lat_slice: slice,
    lon_slice: slice,
    is_run_start: bool,
    position_offset: int,
    context_length: int,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Load one chunk of consecutive frames as a batch of size 1.

    ``position_offset`` is the index of ``frames[0]`` within its contiguous run and
    ``context_length`` is the *configured* ``temporal.context_length`` -- not the
    chunk length. Both are passed explicitly so the time-feature channels depend
    only on the data and the config, never on how the run was chunked. Deriving
    either from the chunk makes chunked and single-pass inference disagree.
    """
    from granitewxc.temporal.calendar import build_time_features, elapsed_days

    xs, ys, masks = [], [], []
    static = None
    geometry = {}
    for frame in frames:
        loaded = source.load_frame(frame, lat_slice, lon_slice)
        for key in ("__scaler_offset", "__input_scaler_offset", "__output_scaler_offset", "__output_crop"):
            if key in loaded:
                if key in geometry and not torch.equal(geometry[key], loaded[key]):
                    raise ValueError(f"Inference dates must share one geometry: {key} changed.")
                geometry[key] = loaded[key]
        x = loaded["x"]
        if dataset.static_channels > 0:
            n = x.shape[0] - dataset.static_channels
            if static is None:
                static = x[n:].clone()
            x = x[:n]
        xs.append(x)
        ys.append(loaded["y"])
        masks.append(loaded["valid_mask"])

    stamps = [f.timestamp for f in frames]
    features, spec = build_time_features(
        stamps,
        cadence_days=dataset.cadence_days,
        calendar=dataset._calendar,
        context_length=int(context_length),
        include_hour_of_day=dataset.include_hour_of_day,
        include_lead_time=dataset.include_lead_time,
        lead_times_days=(
            [dataset.lead_time_days] * len(stamps) if dataset.include_lead_time else None
        ),
        position_offset=int(position_offset),
        mark_sequence_start=bool(is_run_start),
    )
    deltas = elapsed_days(stamps, dataset._calendar)
    ratio = np.ones(len(stamps), dtype=np.float32)
    if len(stamps) > 1:
        ratio[1:] = (deltas[1:] / dataset.cadence_days).astype(np.float32)
    ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)

    reset = np.zeros(len(stamps), dtype=bool)
    reset[0] = bool(is_run_start)

    batch: dict[str, torch.Tensor] = {
        "x": torch.stack(xs, 0).unsqueeze(0),
        "y": torch.stack(ys, 0).unsqueeze(0),
        "time_features": torch.from_numpy(features).unsqueeze(0),
        "interval_ratio": torch.from_numpy(ratio).unsqueeze(0),
        "reset": torch.from_numpy(reset).unsqueeze(0),
        "__target_valid_mask": torch.stack(masks, 0).unsqueeze(0),
    }
    batch.update({key: value.unsqueeze(0) for key, value in geometry.items()})
    if static is not None:
        batch["static_x"] = static.unsqueeze(0)
        batch["static_y"] = static.clone().unsqueeze(0)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def run_sequence_inference(
    runner: TemporalSequenceModel,
    config: Any,
    cfg: TemporalConfig,
    *,
    split: str = "test",
    date_start: str | None = None,
    date_end: str | None = None,
    device: torch.device | str = "cpu",
    chunk_length: int | None = None,
    carry_state: bool | None = None,
    return_targets: bool = True,
    limit_runs: int | None = None,
    limit_frames_per_run: int | None = None,
    verbose: bool = True,
    _source: Any = None,
    _spatial_crop: tuple[slice, slice] | None = None,
) -> list[InferenceResult]:
    """Run the temporal model over every contiguous run in a split.

    Chunks tile each run with no overlap, so each date is emitted once. Warm-up
    is unnecessary when state is carried (the default) because the state already
    encodes the history; when ``carry_state`` is False the configured
    ``chunk_warmup`` frames are re-run purely to rebuild state and are then
    discarded, which is the deployment policy for a cold start at a split
    boundary. For native finite-history models, requested dates select outputs;
    available history before ``date_start`` is retained within the declared split
    and contiguous run. Only a true split/gap/archive boundary uses duplicated
    current inputs, under the explicit configured cold-start time convention.
    """
    from granitewxc.temporal.sources import build_frame_source

    chunk = int(chunk_length or cfg.inference.chunk_length)
    carry = cfg.inference.carry_state_across_chunks if carry_state is None else bool(carry_state)
    warmup = int(cfg.inference.chunk_warmup)
    # Frames of real history each chunk must be handed in addition to the frames
    # it emits (0 for the stateful backends, which carry state instead).
    history_context = (
        cfg.native_pair.max_offset if cfg.backend == "native_pair" else 0
    )
    if history_context and history_context >= chunk:
        raise ValueError(
            f"temporal.inference.chunk_length ({chunk}) must exceed the deepest history "
            f"offset ({history_context}) or a chunk would emit no frame."
        )
    # Cold starts are legitimate here and only here: at the first date of a run
    # there is genuinely no earlier date, and refusing to predict it would change
    # the evaluated date set relative to the frame-independent baseline.
    previous_cold_start = getattr(runner, "_allow_cold_start", False)
    from granitewxc.temporal.training import resolve_crop_size, resolve_static_channels, _split_dates

    source = _source if _source is not None else build_frame_source(config, split)
    plan = getattr(source, "inference_tile_plan", None)
    if plan is not None and _spatial_crop is None:
        # Carry independent temporal state within each spatial tile, then blend
        # output cores on the canonical domain. No tile can substitute a date.
        from granitewxc.utils.prism_tiling import blend_window
        weights = blend_window(plan.core_shape, plan.overlap, mode=source.inference_blend_mode)
        accumulated = None
        totals = np.zeros(plan.domain_shape, dtype=np.float64)
        source._inference_active = True
        try:
            for top, left in plan.positions:
                height, width = plan.core_shape
                region = (slice(top, top + height), slice(left, left + width))
                tile_runs = run_sequence_inference(
                    runner, config, cfg, split=split, date_start=date_start, date_end=date_end,
                    device=device, chunk_length=chunk_length, carry_state=carry_state,
                    return_targets=return_targets, limit_runs=limit_runs,
                    limit_frames_per_run=limit_frames_per_run, verbose=False,
                    _source=source, _spatial_crop=region,
                )
                if accumulated is None:
                    accumulated = []
                    for result in tile_runs:
                        shape = (*result.predictions.shape[:-2], *plan.domain_shape)
                        accumulated.append({
                            "result": result, "prediction": np.zeros(shape, dtype=np.float64),
                            "target": np.zeros(shape, dtype=np.float64) if result.targets is not None else None,
                            "target_weight": np.zeros(shape, dtype=np.float64) if result.targets is not None else None,
                            "mask": np.zeros(shape, dtype=bool) if result.valid_mask is not None else None,
                        })
                if len(tile_runs) != len(accumulated):
                    raise ValueError("Spatial tiles returned different temporal runs.")
                for saved, result in zip(accumulated, tile_runs):
                    if saved["result"].dates != result.dates:
                        raise ValueError("Spatial tiles returned different emitted dates.")
                    saved["prediction"][..., region[0], region[1]] += result.predictions * weights
                    if saved["mask"] is not None:
                        saved["mask"][..., region[0], region[1]] |= result.valid_mask
                    if saved["target"] is not None:
                        valid = result.valid_mask if result.valid_mask is not None else np.isfinite(result.targets)
                        saved["target"][..., region[0], region[1]] += np.where(valid, result.targets, 0) * weights
                        saved["target_weight"][..., region[0], region[1]] += valid * weights
                totals[region] += weights
            if not np.all(totals > 0):
                raise ValueError("NARR inference tiles leave canonical-grid cells uncovered.")
            results = []
            for saved in accumulated or []:
                previous = saved["result"]
                targets = None
                if saved["target"] is not None:
                    targets = (saved["target"] / np.maximum(saved["target_weight"], 1e-30)).astype(np.float32)
                result = InferenceResult(
                    run_id=previous.run_id, dates=previous.dates,
                    predictions=(saved["prediction"] / totals).astype(np.float32),
                    targets=targets, valid_mask=saved["mask"], seam_indices=previous.seam_indices,
                    lat=np.asarray(source._dataset.fine_lat), lon=np.asarray(source._dataset.fine_lon),
                )
                results.append(result)
                if verbose:
                    print(f"[infer] tiled canonical domain: {len(plan.positions)} tiles; {json.dumps(result.describe())}")
            return results
        finally:
            source._inference_active = False
            runner._allow_cold_start = previous_cold_start
    static_channels = resolve_static_channels(config)
    crop = resolve_crop_size(config, source)
    context_start, context_end = date_start, date_end
    if history_context:
        # A requested output interval is not a new observed run. Retain the
        # within-split timeline so the first output can use existing history.
        # Declared split boundaries still reset history, even if the source
        # archive contains earlier dates from another split.
        split_start, split_end = _split_dates(config, split)
        context_start = split_start
        bounds = [str(value) for value in (date_end, split_end) if value is not None]
        context_end = min(bounds) if bounds else None
    dataset = TemporalSequenceDataset(
        source,
        # Run/window bookkeeping only. Time features use cfg.context_length, passed
        # explicitly below, so nothing observable depends on the chunk length.
        window_length=1 if history_context else int(cfg.context_length),
        stride=1,
        cadence_days=cfg.cadence_days,
        crop_size=crop,
        random_crop=False,
        static_channels=static_channels,
        date_start=context_start,
        date_end=context_end,
        include_hour_of_day=cfg.include_hour_of_day,
        include_lead_time=cfg.include_lead_time,
        lead_time_days=cfg.lead_time_days,
    )
    lat_slice, lon_slice = _spatial_crop if _spatial_crop is not None else dataset._select_crop(0)
    runner.eval()

    results: list[InferenceResult] = []
    runs = []
    for run in dataset.runs:
        selected = [i for i, frame in enumerate(run.frames)
                    if date_in_range(frame.timestamp, date_start, date_end)]
        if selected:
            runs.append((run, selected[0], selected[-1] + 1))
    if limit_runs is not None:
        runs = runs[: int(limit_runs)]
    if not runs:
        raise ValueError("No requested inference dates lie within the available declared split.")
    for run, emission_start, emission_stop in runs:
        frames = list(run.frames[:emission_stop])
        if limit_frames_per_run is not None:
            # This budget counts emitted dates, not the necessary prior context.
            frames = frames[:emission_start + int(limit_frames_per_run)]
        state = None
        preds: list[np.ndarray] = []
        targs: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        seams: list[int] = []
        emitted = 0

        position = emission_start
        first_chunk = True
        while position < len(frames):
            if carry or first_chunk:
                lo, warm = position, 0
            else:
                lo = max(position - warmup, 0)
                warm = position - lo
            if history_context:
                # A finite-history pathway carries no state, so nothing can be
                # "carried across" a chunk boundary: the earlier *dates
                # themselves* must be in the batch. Extend the chunk backwards by
                # the deepest history offset and do not emit those frames again.
                # This is what makes chunked inference identical to a single pass
                # here, and it is the direct analogue of carrying hidden state for
                # the recurrent backends.
                new_lo = max(position - history_context, 0)
                warm = position - new_lo
                lo = new_lo
            hi = min(position + chunk, len(frames))
            block = frames[lo:hi]
            batch = _chunk_batch(
                dataset,
                source,
                block,
                lat_slice=lat_slice,
                lon_slice=lon_slice,
                is_run_start=first_chunk and (not history_context or lo == 0),
                position_offset=lo,
                context_length=cfg.context_length,
                device=device,
            )
            # Absolute index of this block's frame 0 within the run, so the pair
            # builder can distinguish "date precedes the run" (a cold start) from
            # "date exists but was not supplied" (a chunking bug).
            batch["__position_offset"] = lo
            emit = tuple(range(warm, len(block)))
            runner._allow_cold_start = bool(history_context)
            try:
                out = runner(
                    batch,
                    initial_state=state if (carry and not first_chunk) else None,
                    emit_indices=emit,
                )
            finally:
                runner._allow_cold_start = previous_cold_start
            state = runner.adapter.detach_state(out.final_state) if carry else None

            preds.append(out.predictions[0].detach().float().cpu().numpy())
            if return_targets and out.target_frames is not None:
                targs.append(out.target_frames[0].detach().float().cpu().numpy())
            if out.valid_mask is not None:
                masks.append(out.valid_mask[0].detach().cpu().numpy())

            if not first_chunk:
                seams.append(emitted)
            emitted += len(emit)
            position = hi
            first_chunk = False

        dates = [time_key(f.timestamp) for f in frames[emission_start:]]
        coordinates = getattr(source, "spatial_coordinates", None)
        latitude, longitude = coordinates(lat_slice, lon_slice) if coordinates is not None else (None, None)
        result = InferenceResult(
            run_id=run.run_id,
            dates=dates,
            predictions=np.concatenate(preds, axis=0),
            targets=np.concatenate(targs, axis=0) if targs else None,
            valid_mask=np.concatenate(masks, axis=0) if masks else None,
            seam_indices=seams,
            lat=latitude, lon=longitude,
        )
        if result.predictions.shape[0] != len(dates):
            raise RuntimeError(
                f"Run {run.run_id}: emitted {result.predictions.shape[0]} frames for "
                f"{len(dates)} dates. Every date must be written exactly once."
            )
        if verbose:
            print(f"[infer] {json.dumps(result.describe(), default=str)}")
        results.append(result)
    runner._allow_cold_start = previous_cold_start
    return results


@torch.no_grad()
def verify_chunk_consistency(
    runner: TemporalSequenceModel,
    config: Any,
    cfg: TemporalConfig,
    *,
    split: str = "validation",
    date_start: str | None = None,
    date_end: str | None = None,
    n_frames: int = 24,
    chunk_lengths: Sequence[int] = (24, 8, 5),
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Check on *real data* that chunking does not change the answer.

    Runs the same continuous stretch at several chunk lengths, including one pass
    that fits it entirely in a single chunk, and reports the maximum absolute
    difference. Anything above floating-point noise means state is not being
    carried correctly.
    """
    reference: np.ndarray | None = None
    report: dict[str, Any] = {"n_frames": int(n_frames), "chunks": {}}
    for length in chunk_lengths:
        results = run_sequence_inference(
            runner,
            config,
            cfg,
            split=split,
            date_start=date_start,
            date_end=date_end,
            device=device,
            chunk_length=int(length),
            carry_state=True,
            return_targets=False,
            limit_runs=1,
            limit_frames_per_run=int(n_frames),
            verbose=False,
        )
        if not results:
            continue
        arr = results[0].predictions[:n_frames]
        if reference is None:
            reference = arr
            report["chunks"][str(length)] = {"max_abs_diff": 0.0, "reference": True}
            continue
        n = min(arr.shape[0], reference.shape[0])
        diff = float(np.nanmax(np.abs(arr[:n] - reference[:n])))
        report["chunks"][str(length)] = {"max_abs_diff": diff, "reference": False}
    diffs = [v["max_abs_diff"] for v in report["chunks"].values()]
    report["max_abs_diff_overall"] = float(max(diffs)) if diffs else float("nan")
    report["exact"] = bool(report.get("max_abs_diff_overall", 1.0) == 0.0)
    return report


def write_netcdf(
    result: InferenceResult,
    path: str | os.PathLike[str],
    *,
    output_vars: Sequence[str],
    units: Sequence[str] | None = None,
    calendar: str = "standard",
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    attrs: Mapping[str, Any] | None = None,
    compression_level: int = 4,
) -> str:
    """Write one run's predictions (and targets when present) to NetCDF."""
    import cftime
    import xarray as xr

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    times = [
        cftime.datetime(y, m, d, h, mi, s, calendar=calendar)
        for (y, m, d, h, mi, s) in result.dates
    ]
    t, c, hgt, wid = result.predictions.shape
    coords: dict[str, Any] = {"time": times}
    dims = ("time", "lat", "lon")
    lat = lat if lat is not None else result.lat
    lon = lon if lon is not None else result.lon
    if lat is not None and lon is not None and np.ndim(lat) == np.ndim(lon) == 2:
        if np.shape(lat) != (hgt, wid) or np.shape(lon) != (hgt, wid):
            raise ValueError("Two-dimensional physical coordinates must match the prediction grid.")
        dims = ("time", "y", "x")
        coords["lat"] = (("y", "x"), lat)
        coords["lon"] = (("y", "x"), lon)
    else:
        coords["lat"] = lat if lat is not None else np.arange(hgt, dtype=np.float32)
        coords["lon"] = lon if lon is not None else np.arange(wid, dtype=np.float32)

    data_vars: dict[str, Any] = {}
    for idx, name in enumerate(output_vars):
        unit = units[idx] if units and idx < len(units) else ""
        data_vars[str(name)] = (
            dims,
            result.predictions[:, idx],
            {"units": unit, "long_name": f"downscaled {name}"},
        )
        if result.targets is not None:
            data_vars[f"{name}_target"] = (
                dims,
                result.targets[:, idx],
                {"units": unit, "long_name": f"reference {name}"},
            )
        if result.valid_mask is not None:
            data_vars[f"{name}_valid"] = (
                dims,
                result.valid_mask[:, idx].astype(np.int8),
                {"long_name": f"finite-target mask for {name}", "flag_values": "0,1"},
            )

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs.update(
        {
            "title": "Prithvi-UNet temporal downscaling output",
            "run_id": int(result.run_id),
            "seam_indices": json.dumps(result.seam_indices),
            "note": (
                "Produced by a causal, same-day, sequence-conditioned model: output date t "
                "uses predictors at t and earlier only. Zero predictor-to-target lead time."
            ),
            **{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in (attrs or {}).items()},
        }
    )
    encoding = {
        name: {"zlib": True, "complevel": int(compression_level)} for name in ds.data_vars
    }
    ds.to_netcdf(target, encoding=encoding)
    ds.close()
    return str(target)
