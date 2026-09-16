"""Contiguous-sequence windowing over an existing frame dataset.

The rules this module enforces, and why each one matters on the real files in
this repository:

**Sort by actual time, never by index.** Windows are built from the time
coordinate values, so a file whose records are not monotonic cannot silently
produce a scrambled "sequence".

**Split before windowing.** Date ranges are applied to the frame list *first*,
and windows are then built only inside a split. A window can therefore never
straddle the train/validation boundary, so no validation target is ever seen as
training context and no training target leaks into validation history.

**Never treat separated dates as consecutive.** The frame list is cut into
contiguous runs wherever :func:`~granitewxc.temporal.calendar.find_discontinuities`
reports an off-cadence step. On the SA ACCESS-CM2 training file this produces 12
runs: the 1961-1980 block is split at each removed 29 February, and the
1980 -> 2080 concatenation is a hard break. Windows live strictly inside one run.

**Consistent crops within a sequence.** A single spatial crop is drawn per
window from a window-keyed, seeded generator and applied to every frame, so the
recurrent state describes one fixed piece of geography. A per-frame crop would
make the temporal state meaningless.

**Masks travel with the data.** The finite-target mask is returned per frame and
is what both the spatial and temporal losses consume, so a missing day cannot be
read as 0 mm of precipitation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from granitewxc.temporal.calendar import (
    CalendarSpec,
    TimeFeatureSpec,
    build_time_features,
    describe_time_axis,
    elapsed_days,
    find_discontinuities,
    resolve_calendar,
    time_key,
)

__all__ = [
    "FrameRef",
    "FrameSource",
    "ContiguousRun",
    "SequenceWindow",
    "TemporalSequenceDataset",
    "build_runs",
    "build_windows",
    "date_in_range",
]


# ---------------------------------------------------------------------------
# frame addressing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameRef:
    """One (predictor, target) pair at one timestamp."""

    file_index: int
    predictor_time_index: int
    target_time_index: int
    timestamp: Any

    @property
    def key(self) -> tuple[int, int, int, int, int, int]:
        return time_key(self.timestamp)


class FrameSource(Protocol):
    """Minimal interface a frame dataset must expose to be sequenced.

    Deliberately small so both the CORDEX and NARR/PRISM datasets can be adapted
    without changing their preprocessing: the sequence layer reuses whatever
    regridding, unit conversion and masking the frame dataset already does, which
    is what makes the two temporal backends and the frame-independent baseline
    comparable.
    """

    def frames(self) -> list[FrameRef]:
        """All available frames, in file order."""

    def calendar(self) -> CalendarSpec:
        """Calendar of the predictor time axis."""

    def fine_shape(self) -> tuple[int, int]:
        """Target-grid spatial shape ``(H, W)``."""

    def load_frame(
        self, ref: FrameRef, lat_slice: slice, lon_slice: slice
    ) -> dict[str, torch.Tensor]:
        """Load one frame. Must return ``x``, ``y`` and ``valid_mask``.

        ``x`` may include trailing static channels; the sequence dataset splits
        them out according to ``static_channels``.
        """


# ---------------------------------------------------------------------------
# runs and windows
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ContiguousRun:
    """A maximal stretch of frames with a strictly regular cadence."""

    frames: tuple[FrameRef, ...]
    run_id: int

    def __len__(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class SequenceWindow:
    """One training/inference window inside a single :class:`ContiguousRun`."""

    run_id: int
    frames: tuple[FrameRef, ...]
    #: Index of the window's first frame within its run (for provenance).
    start_in_run: int

    def __len__(self) -> int:
        return len(self.frames)


def date_in_range(stamp: Any, start: str | None, end: str | None) -> bool:
    """Inclusive date-range test on ``(y, m, d)``, calendar-agnostic.

    Comparing ``(y, m, d)`` tuples rather than converting to ``datetime`` avoids
    constructing an invalid date (for example 29 February in a no-leap file),
    which is a real failure mode for the SA files.
    """
    key = time_key(stamp)[:3]

    def parse(text: str) -> tuple[int, int, int]:
        parts = str(text).strip().split("-")
        if len(parts) != 3:
            raise ValueError(f"Date bound {text!r} must be YYYY-MM-DD.")
        return tuple(int(p) for p in parts)  # type: ignore[return-value]

    if start is not None and key < parse(start):
        return False
    if end is not None and key > parse(end):
        return False
    return True


def build_runs(
    frames: Sequence[FrameRef],
    *,
    cadence_days: float,
    calendar: CalendarSpec | str,
    tolerance_days: float = 1e-6,
) -> list[ContiguousRun]:
    """Split ``frames`` into maximal runs of strictly regular cadence.

    ``frames`` is first sorted by actual timestamp. Duplicated timestamps are an
    error rather than something to deduplicate silently: a duplicate means the
    caller's file list is wrong, and guessing which copy to keep would hide it.
    """
    if not frames:
        return []
    ordered = sorted(frames, key=lambda f: f.key)
    keys = [f.key for f in ordered]
    duplicates = {k for k in keys if keys.count(k) > 1} if len(keys) != len(set(keys)) else set()
    if duplicates:
        sample = sorted(duplicates)[:3]
        raise ValueError(
            f"{len(duplicates)} duplicate timestamp(s) in the frame list, e.g. {sample}. "
            "Overlapping input files would double-count dates; fix the file list."
        )

    stamps = [f.timestamp for f in ordered]
    breaks = find_discontinuities(
        stamps, cadence_days, calendar=calendar, tolerance_days=tolerance_days
    )
    runs: list[ContiguousRun] = []
    current: list[FrameRef] = []
    run_id = 0
    for idx, frame in enumerate(ordered):
        if breaks[idx] and current:
            runs.append(ContiguousRun(frames=tuple(current), run_id=run_id))
            run_id += 1
            current = []
        current.append(frame)
    if current:
        runs.append(ContiguousRun(frames=tuple(current), run_id=run_id))
    return runs


def build_windows(
    runs: Sequence[ContiguousRun],
    *,
    window_length: int,
    stride: int,
    drop_last_partial: bool = True,
) -> list[SequenceWindow]:
    """Enumerate windows of exactly ``window_length`` frames inside each run.

    Runs shorter than ``window_length`` yield nothing. That is intentional: a
    short run cannot supply the configured history, and padding it would feed the
    model fabricated context.
    """
    if window_length < 1:
        raise ValueError("window_length must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    windows: list[SequenceWindow] = []
    for run in runs:
        n = len(run)
        if n < window_length:
            continue
        last_start = n - window_length
        for start in range(0, last_start + 1, stride):
            windows.append(
                SequenceWindow(
                    run_id=run.run_id,
                    frames=run.frames[start : start + window_length],
                    start_in_run=start,
                )
            )
        if not drop_last_partial:
            tail = last_start % stride
            if tail:
                windows.append(
                    SequenceWindow(
                        run_id=run.run_id,
                        frames=run.frames[last_start : last_start + window_length],
                        start_in_run=last_start,
                    )
                )
    return windows


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
class TemporalSequenceDataset(Dataset):
    """Yield contiguous sequence windows from a :class:`FrameSource`.

    Returned sample (before collation)::

        x               [T, C_pred,   H, W]
        y               [T, C_target, H, W]
        static_x        [C_static, H, W]      (omitted when use_static is False)
        static_y        [C_static, H, W]
        time_features   [T, F]
        interval_ratio  [T]
        reset           [T]  bool
        __target_valid_mask [T, C_target, H, W]

    ``reset[0]`` is always True: every window begins a fresh state during
    training. During chunked inference the runner overrides this by carrying
    state explicitly.
    """

    def __init__(
        self,
        source: FrameSource,
        *,
        window_length: int,
        stride: int = 1,
        cadence_days: float = 1.0,
        crop_size: tuple[int, int] | None = None,
        random_crop: bool = False,
        seed: int = 1234,
        static_channels: int = 0,
        date_start: str | None = None,
        date_end: str | None = None,
        include_hour_of_day: bool | str = "auto",
        include_lead_time: bool = False,
        lead_time_days: float = 0.0,
        min_valid_target_fraction: float = 0.0,
        epoch: int = 0,
    ) -> None:
        self.source = source
        self.window_length = int(window_length)
        self.stride = int(stride)
        self.cadence_days = float(cadence_days)
        self.static_channels = int(static_channels)
        self.random_crop = bool(random_crop)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.include_hour_of_day = include_hour_of_day
        self.include_lead_time = bool(include_lead_time)
        self.lead_time_days = float(lead_time_days)
        self.min_valid_target_fraction = float(min_valid_target_fraction)
        self.date_start = date_start
        self.date_end = date_end

        self._calendar = source.calendar()
        self._fine_shape = tuple(int(v) for v in source.fine_shape())

        all_frames = source.frames()
        # --- split BEFORE windowing -------------------------------------
        self.frames = [
            f for f in all_frames if date_in_range(f.timestamp, date_start, date_end)
        ]
        if not self.frames:
            raise ValueError(
                f"No frames fall inside [{date_start}, {date_end}] for this source. "
                f"The source spans {time_key(all_frames[0].timestamp)[:3]} .. "
                f"{time_key(all_frames[-1].timestamp)[:3]}."
                if all_frames
                else "The frame source is empty."
            )

        self.runs = build_runs(
            self.frames, cadence_days=self.cadence_days, calendar=self._calendar
        )
        self.windows = build_windows(
            self.runs, window_length=self.window_length, stride=self.stride
        )
        if not self.windows:
            longest = max((len(r) for r in self.runs), default=0)
            raise ValueError(
                f"No window of {self.window_length} contiguous frames fits in "
                f"[{date_start}, {date_end}]: the longest contiguous run is {longest} "
                f"frame(s) across {len(self.runs)} run(s). Reduce "
                "temporal.context_length or widen the date range."
            )

        if crop_size is None:
            self.crop_size = self._fine_shape
        else:
            self.crop_size = (
                min(int(crop_size[0]), self._fine_shape[0]),
                min(int(crop_size[1]), self._fine_shape[1]),
            )
        self._full_frame = self.crop_size == self._fine_shape
        self.spatial_tiles = tuple(getattr(source, "spatial_tiles", ()) or ())
        self._tiles_per_window = max(1, len(self.spatial_tiles))

    # -- introspection ----------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Audit of what was actually built, for logs and for the run record."""
        stamps = [f.timestamp for f in self.frames]
        axis = describe_time_axis(stamps, self.cadence_days, calendar=self._calendar)
        run_lengths = [len(r) for r in self.runs]
        return {
            "date_range": [self.date_start, self.date_end],
            "n_frames": len(self.frames),
            "n_runs": len(self.runs),
            "run_length_min": min(run_lengths) if run_lengths else 0,
            "run_length_max": max(run_lengths) if run_lengths else 0,
            "n_windows": len(self.windows),
            "n_sequence_samples": len(self),
            "tiles_per_window": self._tiles_per_window,
            "window_length": self.window_length,
            "stride": self.stride,
            "crop_size": list(self.crop_size),
            "fine_shape": list(self._fine_shape),
            "time_axis": axis,
        }

    def emitted_dates(self, output_length: int) -> list[tuple[int, int, int]]:
        """Distinct ``(y, m, d)`` emitted across all windows.

        With ``stride < output_length`` windows overlap, so the same date is
        produced by more than one window. Evaluation must not count those
        repeats; this returns the de-duplicated set and
        :meth:`unique_emission_plan` gives one window per date.
        """
        seen: set[tuple[int, int, int]] = set()
        for window in self.windows:
            for frame in window.frames[len(window) - output_length :]:
                seen.add(time_key(frame.timestamp)[:3])
        return sorted(seen)

    def unique_emission_plan(self, output_length: int) -> list[tuple[int, tuple[int, ...]]]:
        """Assign each date to exactly one window.

        Returns ``[(sample_index, frame_offsets), ...]`` such that every date is
        emitted once per spatial tile. For untiled sources this is once for the
        domain. Tiled outputs must be spatially blended before temporal scoring;
        overlapping tiles are not independent date observations.
        """
        claimed: set[tuple[int, int, int]] = set()
        plan: list[tuple[int, tuple[int, ...]]] = []
        for w_idx, window in enumerate(self.windows):
            offsets: list[int] = []
            base = len(window) - output_length
            for local, frame in enumerate(window.frames[base:], start=base):
                key = time_key(frame.timestamp)[:3]
                if key in claimed:
                    continue
                claimed.add(key)
                offsets.append(local)
            if offsets:
                for tile_index in range(self._tiles_per_window):
                    plan.append((w_idx * self._tiles_per_window + tile_index, tuple(offsets)))
        return plan

    # -- torch Dataset ----------------------------------------------------
    def __len__(self) -> int:
        return len(self.windows) * self._tiles_per_window

    def set_epoch(self, epoch: int) -> None:
        """Re-key the crop generator so crops vary across epochs, reproducibly."""
        self.epoch = int(epoch)

    def _select_crop(self, window_index: int) -> tuple[slice, slice]:
        if self.spatial_tiles:
            return self.spatial_tiles[window_index % self._tiles_per_window]
        h, w = self.crop_size
        fh, fw = self._fine_shape
        if self._full_frame:
            return slice(None), slice(None)
        if not self.random_crop:
            lat0 = (fh - h) // 2
            lon0 = (fw - w) // 2
        else:
            # One crop per window, derived deterministically from
            # (seed, epoch, window_index): identical for every frame in the
            # window, reproducible across runs and across workers.
            rng = np.random.default_rng((self.seed, self.epoch, window_index))
            lat0 = int(rng.integers(0, fh - h + 1)) if fh > h else 0
            lon0 = int(rng.integers(0, fw - w + 1)) if fw > w else 0
        return slice(lat0, lat0 + h), slice(lon0, lon0 + w)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index // self._tiles_per_window]
        lat_slice, lon_slice = self._select_crop(index)

        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        static: torch.Tensor | None = None
        geometry: dict[str, torch.Tensor] = {}

        for frame in window.frames:
            loaded = self.source.load_frame(frame, lat_slice, lon_slice)
            for key in ("__scaler_offset", "__input_scaler_offset", "__output_scaler_offset", "__output_crop"):
                if key in loaded:
                    if key in geometry and not torch.equal(geometry[key], loaded[key]):
                        raise ValueError(f"Sequence dates must share one geometry: {key} changed.")
                    geometry[key] = loaded[key]
            x = loaded["x"]
            if self.static_channels > 0:
                dynamic = x[: x.shape[0] - self.static_channels]
                frame_static = x[x.shape[0] - self.static_channels :]
                if static is None:
                    static = frame_static.clone()
                x = dynamic
            xs.append(x)
            ys.append(loaded["y"])
            masks.append(loaded["valid_mask"])

        stamps = [f.timestamp for f in window.frames]
        features, spec = build_time_features(
            stamps,
            cadence_days=self.cadence_days,
            calendar=self._calendar,
            context_length=self.window_length,
            include_hour_of_day=self.include_hour_of_day,
            include_lead_time=self.include_lead_time,
            lead_times_days=(
                [self.lead_time_days] * len(stamps) if self.include_lead_time else None
            ),
        )
        self.time_feature_spec = spec

        deltas = elapsed_days(stamps, self._calendar)
        ratio = np.ones(len(stamps), dtype=np.float32)
        if len(stamps) > 1:
            ratio[1:] = (deltas[1:] / self.cadence_days).astype(np.float32)
        ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)

        # Within a run every step is regular by construction, so the only reset
        # is the window start. Assert that invariant rather than trusting it.
        reset = np.zeros(len(stamps), dtype=bool)
        reset[0] = True
        irregular = np.abs(ratio[1:] - 1.0) > 1e-6
        if irregular.any():
            raise AssertionError(
                f"Window {index} (run {window.run_id}) contains an irregular step "
                f"{ratio[1:][irregular][:3]}; build_runs should have split it."
            )

        sample: dict[str, Any] = {
            "x": torch.stack(xs, dim=0),
            "y": torch.stack(ys, dim=0),
            "time_features": torch.from_numpy(features),
            "interval_ratio": torch.from_numpy(ratio),
            "reset": torch.from_numpy(reset),
            "__target_valid_mask": torch.stack(masks, dim=0),
            "__window_index": int(index),
            "__run_id": int(window.run_id),
            "__timestamps": [
                f"{k[0]:04d}-{k[1]:02d}-{k[2]:02d}T{k[3]:02d}:{k[4]:02d}:{k[5]:02d}"
                for k in (time_key(s) for s in stamps)
            ],
            "__crop": [
                0 if lat_slice.start is None else int(lat_slice.start),
                0 if lon_slice.start is None else int(lon_slice.start),
                int(self.crop_size[0]),
                int(self.crop_size[1]),
            ],
        }
        sample.update(geometry)
        if static is not None:
            sample["static_x"] = static
            sample["static_y"] = static.clone()
        return sample


def collate_sequences(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate sequence samples, keeping provenance lists as Python objects."""
    out: dict[str, Any] = {}
    tensor_keys = [
        k for k, v in batch[0].items() if torch.is_tensor(v)
    ]
    for key in tensor_keys:
        out[key] = torch.stack([sample[key] for sample in batch], dim=0)
    for key in ("__timestamps", "__crop", "__window_index", "__run_id"):
        if key in batch[0]:
            out[key] = [sample[key] for sample in batch]
    return out
