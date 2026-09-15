"""Frame-source adapters over the repository's existing frame datasets.

These adapters exist so the temporal layer reuses the *existing* preprocessing
(regridding, precipitation unit conversion, orography handling, masking) instead
of reimplementing it. That is what makes the comparison in section 9 of the work
plan meaningful: the frame-independent baseline, the ConvGRU model and the Mamba
model all consume byte-identical fields, differing only in how time is handled.

Two sources are provided:

:class:`CordexFrameSource`
    Wraps :class:`examples.CORDEX_ML.cordex_dataset.CordexDownscaleDataset`.
    Handles the SA ACCESS-CM2 case, including the predictor/target calendar
    mismatch in the held-out test files (predictors omit 29 February, targets
    keep it), which is resolved by the underlying dataset's exact timestamp join.

:class:`NarrPrismFrameSource`
    Wraps the NARR/PRISM preprocessed daily products used by
    ``examples/NARR_PRISM``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from granitewxc.temporal.calendar import CalendarSpec, resolve_calendar, time_key
from granitewxc.temporal.sequence_dataset import FrameRef

__all__ = [
    "CordexFrameSource",
    "NarrPrismFrameSource",
    "load_module_from_path",
    "build_frame_source",
]


def load_module_from_path(name: str, path: str | os.PathLike[str]):
    """Import a module from an explicit file path.

    The example workflows live outside the installed package (``examples/...``)
    and are normally run as scripts. Importing them by path keeps the temporal
    entry points able to reuse them without requiring the examples tree to be a
    package or mutating ``sys.path`` permanently.
    """
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Cannot import {name}: {resolved} does not exist.")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(resolved))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build an import spec for {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# CORDEX
# ---------------------------------------------------------------------------
class CordexFrameSource:
    """Frame source backed by :class:`CordexDownscaleDataset`.

    The underlying dataset already:

    * regrids coarse predictors onto the fine target grid (the SA training file
      is pre-regridded to 128x128 so this is near-identity; the held-out test
      files are native 16x16 and are genuinely interpolated),
    * converts precipitation to ``mm/day`` exactly once, rejecting unknown units,
    * joins predictor and target timestamps *by date*, refusing positional
      indexing, and
    * appends the static orography as the trailing channel of ``x``.

    None of that is duplicated here.
    """

    def __init__(
        self,
        predictor_paths: Sequence[str],
        target_paths: Sequence[str],
        *,
        orography_path: str | None,
        predictor_variables: Sequence[str],
        target_variables: Sequence[str],
        use_static: bool = True,
        regrid_method: str = "bilinear",
        allow_time_mismatch: bool = True,
        dataset_module_path: str | os.PathLike[str] | None = None,
    ) -> None:
        module_path = dataset_module_path or (
            Path(__file__).resolve().parents[2]
            / "examples"
            / "CORDEX_ML"
            / "cordex_dataset.py"
        )
        module = load_module_from_path("_temporal_cordex_dataset", module_path)
        self._dataset = module.CordexDownscaleDataset(
            predictor_files=list(predictor_paths),
            target_files=list(target_paths),
            orography_file=orography_path,
            predictor_variables=list(predictor_variables),
            target_variables=list(target_variables),
            regrid_method=regrid_method,
            crop_size=None,
            random_crop=False,
            use_static=use_static,
            # Held-out SA test files pair a 365-day predictor axis with a
            # Gregorian target axis. The underlying dataset resolves this by
            # exact timestamp join; positional indexing stays forbidden.
            allow_time_mismatch=allow_time_mismatch,
        )
        self.use_static = bool(use_static)
        self.target_units = list(getattr(self._dataset, "target_units", []))
        self._calendar = self._resolve_calendar(predictor_paths[0])

    @staticmethod
    def _resolve_calendar(path: str) -> CalendarSpec:
        import xarray as xr

        coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        with xr.open_dataset(path, decode_times=coder) as ds:
            for name in ("time", "Time", "valid_time"):
                if name in ds.coords:
                    declared = ds[name].encoding.get("calendar") or ds[name].attrs.get("calendar")
                    if declared:
                        return resolve_calendar(declared)
                    return resolve_calendar(ds[name].values)
        raise ValueError(f"No time coordinate found in {path}")

    # -- FrameSource ------------------------------------------------------
    def frames(self) -> list[FrameRef]:
        ds = self._dataset
        refs: list[FrameRef] = []
        import xarray as xr

        coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        for file_idx, path in enumerate(ds.predictor_paths):
            with xr.open_dataset(path, decode_times=coder) as handle:
                stamps = list(handle[ds.time_dim].values)
            mapped = ds._target_time_indices[file_idx]
            for t_idx, stamp in enumerate(stamps):
                refs.append(
                    FrameRef(
                        file_index=file_idx,
                        predictor_time_index=t_idx,
                        target_time_index=int(mapped[t_idx]),
                        timestamp=stamp,
                    )
                )
        return refs

    def calendar(self) -> CalendarSpec:
        return self._calendar

    def fine_shape(self) -> tuple[int, int]:
        return tuple(int(v) for v in self._dataset.fine_shape)  # type: ignore[return-value]

    def load_frame(self, ref: FrameRef, lat_slice: slice, lon_slice: slice) -> dict[str, torch.Tensor]:
        ds = self._dataset
        x = ds._load_predictors(
            ds.predictor_paths[ref.file_index], ref.predictor_time_index, lat_slice, lon_slice
        )
        y = ds._load_targets(
            ds.target_paths[ref.file_index], ref.target_time_index, lat_slice, lon_slice
        )
        valid = torch.isfinite(y)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return {"x": x, "y": y, "valid_mask": valid}

    @property
    def static_channels(self) -> int:
        return 1 if self.use_static else 0


# ---------------------------------------------------------------------------
# NARR / PRISM
# ---------------------------------------------------------------------------
class NarrPrismFrameSource:
    """Frame source over the NARR/PRISM preprocessed daily products.

    The NARR/PRISM workflow reads per-day preprocessed ``.npz``/NetCDF products
    written under ``preprocessed/<case_name>/<mode>/`` by
    ``examples/NARR_PRISM/preproc_narr_prism.py``. This adapter enumerates those
    products, reads their dates from the file index, and returns frames with the
    same channel ordering the frame trainer uses.

    The California subdomain uses a true Gregorian daily calendar, so unlike the
    SA case there is no leap-day gap; discontinuities correspond to genuinely
    missing days in the archive.
    """

    def __init__(
        self,
        preprocessed_dir: str | os.PathLike[str],
        *,
        case_name: str,
        mode: str,
        target_variables: Sequence[str],
        static_channels: int = 0,
        calendar: str = "standard",
    ) -> None:
        self.root = Path(preprocessed_dir) / case_name / mode
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"NARR/PRISM preprocessed directory not found: {self.root}. "
                "Run examples/NARR_PRISM/preproc_narr_prism.py for this case first "
                "(the products are generated locally and are not versioned)."
            )
        self.target_variables = list(target_variables)
        self._static_channels = int(static_channels)
        self._calendar = resolve_calendar(calendar)
        self._files = sorted(self.root.glob("*.npz"))
        if not self._files:
            raise FileNotFoundError(
                f"No preprocessed .npz day files under {self.root}."
            )
        self._shape: tuple[int, int] | None = None

    @staticmethod
    def _date_from_name(path: Path) -> tuple[int, int, int]:
        stem = path.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if len(digits) < 8:
            raise ValueError(f"Cannot parse a YYYYMMDD date from {path.name!r}.")
        token = digits[:8]
        return int(token[:4]), int(token[4:6]), int(token[6:8])

    def frames(self) -> list[FrameRef]:
        import cftime

        refs: list[FrameRef] = []
        for idx, path in enumerate(self._files):
            year, month, day = self._date_from_name(path)
            stamp = cftime.datetime(
                year, month, day, calendar=self._calendar.name
            )
            refs.append(
                FrameRef(
                    file_index=idx,
                    predictor_time_index=0,
                    target_time_index=0,
                    timestamp=stamp,
                )
            )
        return refs

    def calendar(self) -> CalendarSpec:
        return self._calendar

    def fine_shape(self) -> tuple[int, int]:
        if self._shape is None:
            with np.load(self._files[0]) as data:
                key = "y" if "y" in data else self.target_variables[0]
                arr = data[key]
            self._shape = (int(arr.shape[-2]), int(arr.shape[-1]))
        return self._shape

    def load_frame(self, ref: FrameRef, lat_slice: slice, lon_slice: slice) -> dict[str, torch.Tensor]:
        path = self._files[ref.file_index]
        with np.load(path) as data:
            x = np.asarray(data["x"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.float32)
        x_t = torch.from_numpy(x[..., lat_slice, lon_slice])
        y_t = torch.from_numpy(y[..., lat_slice, lon_slice])
        valid = torch.isfinite(y_t)
        y_t = torch.nan_to_num(y_t, nan=0.0, posinf=0.0, neginf=0.0)
        return {"x": x_t, "y": y_t, "valid_mask": valid}

    @property
    def static_channels(self) -> int:
        return self._static_channels


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def build_frame_source(config: Any, split: str):
    """Build the frame source for ``config.data.type`` and ``split``.

    ``split`` is one of ``train``/``validation``/``test`` and selects the
    configured path lists. Splits are *file/date* level here; the date-range
    narrowing happens in :class:`~granitewxc.temporal.sequence_dataset.TemporalSequenceDataset`.
    """
    data = config.data
    dtype = str(getattr(data, "type", "")).lower()

    if dtype == "cordex":
        key_map = {
            "train": ("training_predictor_paths", "training_target_paths"),
            "validation": ("validation_predictor_paths", "validation_target_paths"),
            "test": ("test_predictor_paths", "test_target_paths"),
        }
        pred_key, targ_key = key_map[split]
        predictors = list(getattr(data, pred_key, []) or [])
        targets = list(getattr(data, targ_key, []) or [])
        if not predictors or not targets:
            raise ValueError(
                f"config.data.{pred_key} / {targ_key} is empty; the {split} split has no files."
            )
        levels = [
            str(level)[:-2] if str(level).endswith(".0") else str(level)
            for level in data.input_levels
        ]
        predictor_vars = [f"{var}_{lvl}" for var in data.input_vars for lvl in levels]
        use_static = bool(getattr(data, "use_static", True))
        return CordexFrameSource(
            predictors,
            targets,
            orography_path=str(data.static_path) if use_static else None,
            predictor_variables=predictor_vars,
            target_variables=list(data.output_vars),
            use_static=use_static,
            regrid_method=str(getattr(data, "regrid_method", "bilinear")),
        )

    if dtype in {"narr_prism", "merra_prism"}:
        mode_map = {"train": "train", "validation": "validation", "test": "inference"}
        return NarrPrismFrameSource(
            getattr(data, "preprocessed_dir"),
            case_name=str(getattr(config, "case_name")),
            mode=mode_map[split],
            target_variables=list(data.output_vars),
            static_channels=int(getattr(config.model, "num_static_channels", 0)),
            calendar=str(getattr(data, "calendar", "standard")),
        )

    raise ValueError(
        f"config.data.type={dtype!r} has no temporal frame source. Supported: "
        "cordex, narr_prism, merra_prism."
    )
