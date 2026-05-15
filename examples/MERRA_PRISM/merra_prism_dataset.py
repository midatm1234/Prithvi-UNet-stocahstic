"""PyTorch Dataset for MERRA2-to-PRISM downscaling.

Reads daily MERRA2 predictors and daily PRISM targets, aligns by date,
and returns (predictor, target) tensor pairs normalised by pre-computed
scalars.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import xarray as xr
except ImportError:
    xr = None

from merra_prism_utils import (
    align_dates,
    discover_all_prism_targets,
    discover_merra2_files,
    load_yaml,
    parse_date_range_from_config,
    resolve_path,
    validate_dates_exist,
    validate_target_variables,
)


PathLike = Union[str, os.PathLike[str]]

LAT_CANDIDATES = ("lat", "latitude", "y")
LON_CANDIDATES = ("lon", "longitude", "x")


def _infer_coord_name(ds: Any, candidates: Sequence[str]) -> str:
    """Find the first matching coordinate name."""
    for name in candidates:
        if name in ds.coords or name in ds.data_vars:
            return name
    raise ValueError(f"Unable to find coordinate among {candidates} in dataset")


class MerraPrismDataset(Dataset):
    """Pair daily MERRA2 predictors with daily PRISM targets.

    Parameters
    ----------
    config_path : str or Path
        Path to the MERRA_PRISM.yaml configuration file.
    mode : str
        ``"training"`` or ``"inference"`` — selects the date range from the
        YAML ``dates`` block.
    predictor_variables : list[str] | None
        Override predictor variable names (default reads from YAML).
    target_variables : list[str] | None
        Override target variable names (default reads from YAML).
    scalars_dir : str | Path | None
        Override scalar directory (default reads from YAML ``data.scalar_dir``).
    dtype : torch.dtype
        Tensor dtype for returned samples.
    """

    def __init__(
        self,
        config_path: PathLike,
        mode: str = "training",
        predictor_variables: Optional[Sequence[str]] = None,
        target_variables: Optional[Sequence[str]] = None,
        scalars_dir: Optional[PathLike] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if xr is None:
            raise ImportError("xarray is required for MerraPrismDataset")

        self.cfg = load_yaml(config_path)
        self.mode = mode
        self.dtype = dtype

        data_cfg = self.cfg.get("data", {})
        self.predictor_dir = resolve_path(data_cfg["predictor_dir"])
        self.target_dir = resolve_path(data_cfg["target_dir"])

        self.predictor_vars: List[str] = list(
            predictor_variables or data_cfg.get("predictor_variables", [])
        )
        self.target_vars: List[str] = list(
            target_variables or data_cfg.get("target_variables", [])
        )
        if not self.predictor_vars:
            raise ValueError("No predictor variables specified")
        if not self.target_vars:
            raise ValueError("No target variables specified")

        # Validate target directories exist
        validate_target_variables(self.target_dir, self.target_vars)

        # Parse dates from YAML
        self.start_date, self.end_date = parse_date_range_from_config(self.cfg, mode)

        # Discover files
        merra_files = discover_merra2_files(self.predictor_dir, self.start_date, self.end_date)
        prism_files = discover_all_prism_targets(
            self.target_dir, self.target_vars, self.start_date, self.end_date
        )

        # Build date -> path maps
        self._predictor_map: Dict[Any, Path] = {d: p for d, p in merra_files}
        self._target_maps: Dict[str, Dict[Any, Path]] = {}
        for var, file_list in prism_files.items():
            self._target_maps[var] = {d: p for d, p in file_list}

        # Align dates across predictors and all targets
        target_date_lists = {
            var: list(dm.keys()) for var, dm in self._target_maps.items()
        }
        self._dates = align_dates(list(self._predictor_map.keys()), target_date_lists)

        # Validate requested dates exist
        validate_dates_exist(self._dates, list(self._predictor_map.keys()), "predictor")
        for var in self.target_vars:
            validate_dates_exist(
                self._dates, list(self._target_maps[var].keys()), f"target ({var})"
            )

        # Load scalars if available
        scalar_dir_raw = scalars_dir or data_cfg.get("scalar_dir", "")
        self._scalars = self._load_scalars(scalar_dir_raw)

    # ------------------------------------------------------------------
    # Scalar loading
    # ------------------------------------------------------------------
    def _load_scalars(self, scalar_dir: PathLike) -> Dict[str, np.ndarray]:
        """Load pre-computed normalisation scalars if they exist."""
        scalars: Dict[str, np.ndarray] = {}
        if not scalar_dir:
            return scalars

        sd = resolve_path(scalar_dir)
        names = ["inputs_mean", "inputs_std", "targets_mean", "targets_std"]
        for name in names:
            path = sd / f"{name}.npy"
            if path.exists():
                scalars[name] = np.load(str(path))
        return scalars

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._dates)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        sample_date = self._dates[index]

        # Load predictors
        x = self._load_predictor(sample_date)

        # Load targets
        y = self._load_targets(sample_date)

        # Apply normalisation
        x = self._normalize_input(x)
        y = self._normalize_target(y)

        return {"x": x, "y": y, "date": str(sample_date)}

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------
    def _load_predictor(self, sample_date: Any) -> torch.Tensor:
        """Load MERRA2 predictor variables for a single date."""
        path = self._predictor_map[sample_date]
        arrays: List[np.ndarray] = []
        with xr.open_dataset(str(path)) as ds:
            for var in self.predictor_vars:
                if var not in ds.data_vars:
                    raise ValueError(
                        f"MERRA2 file {path} does not contain variable '{var}'"
                    )
                da = ds[var]
                # Collapse time dim if present
                if "time" in da.dims:
                    da = da.isel(time=0, drop=True)
                arr = da.values.astype(np.float32)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                if arr.ndim == 1:
                    warnings.warn(f"Predictor '{var}' is 1-D; expanding to 2-D")
                    arr = arr[np.newaxis, :]
                arrays.append(arr)
        stacked = np.stack(arrays, axis=0)
        return torch.from_numpy(stacked).to(self.dtype)

    def _load_targets(self, sample_date: Any) -> torch.Tensor:
        """Load all PRISM target variables for a single date."""
        arrays: List[np.ndarray] = []
        for var in self.target_vars:
            path = self._target_maps[var][sample_date]
            with xr.open_dataset(str(path)) as ds:
                # Identify the data variable
                data_vars = list(ds.data_vars)
                if not data_vars:
                    raise ValueError(f"No data variables in {path}")
                # Use the first (usually only) data variable
                da = ds[data_vars[0]]
                if "time" in da.dims:
                    da = da.isel(time=0, drop=True)
                arr = da.values.astype(np.float32)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                arrays.append(arr)

        stacked = np.stack(arrays, axis=0)
        return torch.from_numpy(stacked).to(self.dtype)

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._scalars.get("inputs_mean")
        std = self._scalars.get("inputs_std")
        if mean is None or std is None:
            return x
        mean_t = torch.from_numpy(mean).to(self.dtype)
        std_t = torch.from_numpy(std).to(self.dtype)
        # Reshape for broadcasting: (C,) → (C, 1, 1) or keep spatial dims
        while mean_t.ndim < x.ndim:
            mean_t = mean_t.unsqueeze(-1)
            std_t = std_t.unsqueeze(-1)
        std_t = torch.clamp(std_t, min=1e-6)
        return (x - mean_t) / std_t

    def _normalize_target(self, y: torch.Tensor) -> torch.Tensor:
        mean = self._scalars.get("targets_mean")
        std = self._scalars.get("targets_std")
        if mean is None or std is None:
            return y
        mean_t = torch.from_numpy(mean).to(self.dtype)
        std_t = torch.from_numpy(std).to(self.dtype)
        while mean_t.ndim < y.ndim:
            mean_t = mean_t.unsqueeze(-1)
            std_t = std_t.unsqueeze(-1)
        std_t = torch.clamp(std_t, min=1e-6)
        return (y - mean_t) / std_t

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    @property
    def dates(self) -> List[Any]:
        return list(self._dates)

    @property
    def num_predictor_channels(self) -> int:
        return len(self.predictor_vars)

    @property
    def num_target_channels(self) -> int:
        return len(self.target_vars)
