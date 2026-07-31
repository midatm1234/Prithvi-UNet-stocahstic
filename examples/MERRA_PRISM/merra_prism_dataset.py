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
    expand_predictor_variables,
    load_elevation,
    load_yaml,
    parse_date_range_from_config,
    resolve_path,
    validate_dates_exist,
    validate_target_variables,
)

from granitewxc.utils import normalization as norm
from granitewxc.utils import prism_grid as prism_grid_contract
from granitewxc.utils import prism_preprocessed
from granitewxc.utils.prism_tiling import TilePlan, halo_crop_slices, pad_spatial_context


PathLike = Union[str, os.PathLike[str]]

LAT_CANDIDATES = ("lat", "latitude", "y")
LON_CANDIDATES = ("lon", "longitude", "x")


def _infer_coord_name(ds: Any, candidates: Sequence[str]) -> str:
    """Find the first matching coordinate name."""
    for name in candidates:
        if name in ds.coords or name in ds.data_vars:
            return name
    raise ValueError(f"Unable to find coordinate among {candidates} in dataset")


def _find_numeric_datavar(ds: Any, path: Any) -> str:
    """Return the first data variable with a numeric dtype, skipping metadata vars like 'crs'."""
    for v in ds.data_vars:
        if np.issubdtype(ds[v].dtype, np.number):
            return v
    raise ValueError(f"No numeric data variable found in {path}")


def _tile_origins(size: int, tile_size: int, stride: int) -> List[int]:
    """Return deterministic origins that cover an axis with fixed-size tiles."""
    tile_size = min(tile_size, size)
    stride = max(1, stride)
    origins = list(range(0, max(size - tile_size + 1, 1), stride))
    final_origin = size - tile_size
    if not origins or origins[-1] != final_origin:
        origins.append(final_origin)
    return sorted(set(int(origin) for origin in origins))


def _coord_subset_slice(
    coords: np.ndarray,
    lower: Optional[float],
    upper: Optional[float],
    axis_name: str,
) -> slice:
    """Return a contiguous index slice for inclusive coordinate bounds."""
    if lower is None and upper is None:
        return slice(None)

    lo = float(np.nanmin(coords) if lower is None else lower)
    hi = float(np.nanmax(coords) if upper is None else upper)
    if hi < lo:
        lo, hi = hi, lo

    coord_min = float(np.nanmin(coords))
    coord_max = float(np.nanmax(coords))
    if hi < coord_min or lo > coord_max:
        raise ValueError(
            f"spatial_subset {axis_name} bounds [{lo}, {hi}] do not overlap "
            f"grid range [{coord_min}, {coord_max}]"
        )

    diffs = np.diff(np.asarray(coords, dtype=np.float64))
    finite_diffs = np.abs(diffs[np.isfinite(diffs) & (diffs != 0)])
    tol = 0.0 if finite_diffs.size == 0 else float(np.nanmedian(finite_diffs)) * 0.51

    mask = (coords >= lo - tol) & (coords <= hi + tol)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(
            f"spatial_subset {axis_name} bounds [{lo}, {hi}] selected no grid cells"
        )
    return slice(int(idx[0]), int(idx[-1]) + 1)


class MerraPrismDataset(Dataset):
    """Pair daily MERRA2 predictors with daily PRISM targets.

    Parameters
    ----------
    config_path : str or Path
        Path to the MERRA_PRISM.yaml configuration file.
    mode : str
        ``"training"`` or ``"inference"`` — selects the date range from the
        YAML ``dates`` block.
    predictor_variables : dict[str, list] | None
        Override predictor variable config (default reads from YAML).
        Format: ``{"QV": [500, 700, 850], "U": [500, 700, 850], ...}``.
    target_variables : list[str] | None
        Override target variable names (default reads from YAML).
    scalars_dir : str | Path | None
        Override scalar directory (default: case-scoped
        ``<preprocessed_dir>/<case_name>/scalars``, resolved via
        ``normalization.resolve_scalar_dir``).
    dtype : torch.dtype
        Tensor dtype for returned samples.
    """

    def __init__(
        self,
        config_path: PathLike,
        mode: str = "training",
        predictor_variables: Optional[Dict[str, List]] = None,
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
        self.use_preprocessed = bool(data_cfg.get("use_preprocessed", False))
        self._case_preprocess_dir = norm.case_preprocess_dir(self.cfg)
        self._preprocessed_dir = self._case_preprocess_dir / mode
        if self.use_preprocessed and scalars_dir is not None:
            selected = resolve_path(scalars_dir).resolve()
            configured = norm.resolve_scalar_dir(
                self.cfg, for_writing=False
            ).resolve()
            if selected != configured:
                raise ValueError(
                    "Strict preprocessed MERRA datasets do not permit a scalar "
                    f"override: selected={selected}, case contract={configured}"
                )

        self.predictor_vars: List[Tuple[str, float]] = expand_predictor_variables(
            predictor_variables or data_cfg.get("predictor_variables", {})
        )
        self.target_vars: List[str] = list(
            target_variables or data_cfg.get("target_variables", [])
        )
        if not self.predictor_vars:
            raise ValueError("No predictor variables specified")
        if not self.target_vars:
            raise ValueError("No target variables specified")

        # Parse dates from YAML
        self.start_date, self.end_date = parse_date_range_from_config(self.cfg, mode)
        self._preprocessed_predictor_names = [
            prism_preprocessed.predictor_name(var, level)
            for var, level in self.predictor_vars
        ]
        self._preprocessed_target_names = [
            prism_preprocessed.target_name(var) for var in self.target_vars
        ]
        preprocess_cfg = self.cfg.get("preprocess", {}) or {}
        if mode == "training":
            self._preprocessed_include_targets = bool(
                preprocess_cfg.get("save_train_targets", True)
            )
        elif mode == "validation":
            self._preprocessed_include_targets = bool(
                preprocess_cfg.get("save_val_targets", True)
            )
        elif mode == "inference":
            self._preprocessed_include_targets = bool(
                preprocess_cfg.get("save_inference_targets", False)
                or preprocess_cfg.get(
                    "inference_include_observed_targets_for_eval", False
                )
            )
        else:
            raise ValueError(f"Unsupported dataset mode: {mode}")
        if mode in {"training", "validation"} and not self._preprocessed_include_targets:
            raise ValueError(
                f"MERRA {mode} requires embedded target_* variables; enable the "
                f"corresponding preprocess save_{'train' if mode == 'training' else 'val'}_targets option"
            )
        self._preprocessed_include_elevation = bool(
            data_cfg.get("static_elevation_file")
        )
        self._preprocessed_required_variables = [
            *self._preprocessed_predictor_names,
        ]
        if self._preprocessed_include_targets:
            self._preprocessed_required_variables.extend(
                self._preprocessed_target_names
            )
        if self._preprocessed_include_elevation:
            self._preprocessed_required_variables.append("static_elevation")

        self._preprocessed_map: Dict[Any, Path] = {}
        self._preprocessed_signature: Optional[str] = None
        self._predictor_preprocessing_signature: Optional[str] = None
        self._training_source_artifact_signatures: Dict[str, str] = {}
        self._training_source_artifact_split_signature: Optional[str] = None
        self._preprocessed_config_fields: Dict[str, Any] = {}
        self._predictor_map: Dict[Any, Path] = {}
        self._target_maps: Dict[str, Dict[Any, Path]] = {}
        if self.use_preprocessed:
            self._dates = prism_preprocessed.inclusive_daily_dates(
                self.start_date, self.end_date
            )
            self._preprocessed_map = prism_preprocessed.require_daily_products(
                self._preprocessed_dir, "merra_prism", self._dates
            )
            if mode == "training":
                (
                    self._training_source_artifact_signatures,
                    self._training_source_artifact_split_signature,
                ) = norm.training_source_artifact_contract(
                    self.cfg,
                    expected_dates=self._dates,
                    role="MERRA training dataset",
                )
            print(
                f"[dataset] strict preprocessed source: {self._preprocessed_dir} "
                f"({len(self._dates)} required daily products; raw fallback disabled)"
            )
        else:
            self.predictor_dir = resolve_path(data_cfg["predictor_dir"])
            self.target_dir = resolve_path(data_cfg["target_dir"])
            validate_target_variables(self.target_dir, self.target_vars)
            merra_files = discover_merra2_files(
                self.predictor_dir, self.start_date, self.end_date
            )
            prism_files = discover_all_prism_targets(
                self.target_dir, self.target_vars, self.start_date, self.end_date
            )
            self._predictor_map = {d: p for d, p in merra_files}
            for var, file_list in prism_files.items():
                self._target_maps[var] = {d: p for d, p in file_list}
            target_date_lists = {
                var: list(dm.keys()) for var, dm in self._target_maps.items()
            }
            self._dates = align_dates(
                list(self._predictor_map.keys()), target_date_lists
            )
            validate_dates_exist(
                self._dates, list(self._predictor_map.keys()), "predictor"
            )
            for var in self.target_vars:
                validate_dates_exist(
                    self._dates,
                    list(self._target_maps[var].keys()),
                    f"target ({var})",
                )

        # ------------------------------------------------------------------
        # Fine (PRISM) target grid. Predictors are regridded onto THIS grid so
        # that predictor and target are co-registered (same approach as the
        # working CORDEX_ML dataset, which regrids coarse GCM fields onto the
        # fine target grid before cropping).
        # ------------------------------------------------------------------
        self._elevation: Optional[np.ndarray] = None
        if self.use_preprocessed:
            canonical_grid = prism_grid_contract.load_canonical_grid(
                self._case_preprocess_dir, required=True
            )
            assert canonical_grid is not None
            self._canonical_grid = canonical_grid
            self._full_fine_lat = canonical_grid.lat
            self._full_fine_lon = canonical_grid.lon
            self._domain_lat_slice = slice(None)
            self._domain_lon_slice = slice(None)
            subset_enabled = False
            first_date = self._dates[0]
            first_product = self._preprocessed_map[first_date]
            self._preprocessed_config_fields = (
                prism_preprocessed.preprocessing_config_fields(
                    data_type="merra_prism",
                    predictor_variables=self.predictor_vars,
                    target_variables=self.target_vars,
                    include_targets=self._preprocessed_include_targets,
                    regrid_method=str(data_cfg.get("regrid_method", "bilinear")),
                    canonical_grid_fingerprint=canonical_grid.fingerprint,
                    static_elevation_required=self._preprocessed_include_elevation,
                    static_elevation_variable=data_cfg.get("static_elevation_var"),
                )
            )
            with xr.open_dataset(str(first_product)) as ds:
                lat_name, lon_name = prism_preprocessed.validate_daily_product(
                    ds,
                    first_product,
                    canonical_grid,
                    mode=self.mode,
                    sample_date=first_date,
                    required_variables=self._preprocessed_required_variables,
                    expected_preprocessing_fields=self._preprocessed_config_fields,
                    expected_source_artifact_signature=(
                        self._training_source_artifact_signatures.get(
                            str(first_date)
                        )
                    ),
                )
                self._preprocessed_signature = str(
                    ds.attrs[prism_preprocessed.PREPROCESSING_SIGNATURE_ATTR]
                )
                self._predictor_preprocessing_signature = str(
                    ds.attrs[
                        prism_preprocessed.PREDICTOR_PREPROCESSING_SIGNATURE_ATTR
                    ]
                )
                if self._preprocessed_include_elevation:
                    self._elevation = prism_preprocessed.read_spatial_variable(
                        ds, "static_elevation", lat_name, lon_name
                    )
            norm.assert_preprocessing_signature_matches(
                self.cfg,
                self._predictor_preprocessing_signature,
                role=f"MERRA {mode} dataset",
            )
        else:
            self._full_fine_lat, self._full_fine_lon = self._load_fine_grid()
            spatial_subset = data_cfg.get("spatial_subset", {}) or {}
            subset_enabled = bool(spatial_subset.get("enabled", False))
            if subset_enabled:
                self._domain_lat_slice = _coord_subset_slice(
                    self._full_fine_lat,
                    spatial_subset.get("lat_min"),
                    spatial_subset.get("lat_max"),
                    "latitude",
                )
                self._domain_lon_slice = _coord_subset_slice(
                    self._full_fine_lon,
                    spatial_subset.get("lon_min"),
                    spatial_subset.get("lon_max"),
                    "longitude",
                )
            else:
                self._domain_lat_slice = slice(None)
                self._domain_lon_slice = slice(None)
            candidate_lat = self._full_fine_lat[self._domain_lat_slice]
            candidate_lon = self._full_fine_lon[self._domain_lon_slice]
            first_grid_source = self._target_maps[self.target_vars[0]][self._dates[0]]
            self._canonical_grid = prism_grid_contract.ensure_canonical_grid(
                self._case_preprocess_dir,
                candidate_lat,
                candidate_lon,
                source=str(first_grid_source),
                context="MERRA dataset target grid",
            )
        self.fine_lat = self._canonical_grid.lat
        self.fine_lon = self._canonical_grid.lon
        self.fine_shape: Tuple[int, int] = (len(self.fine_lat), len(self.fine_lon))
        print(
            f"[dataset] canonical PRISM grid fingerprint="
            f"{self._canonical_grid.fingerprint[:12]}"
        )
        if subset_enabled:
            print(
                f"[dataset] spatial subset: lat {self.fine_lat[0]:.4f}.."
                f"{self.fine_lat[-1]:.4f}, lon {self.fine_lon[0]:.4f}.."
                f"{self.fine_lon[-1]:.4f}, shape={self.fine_shape}"
            )

        # Crop size (tile) used for training. Inference crops/tiles explicitly.
        crop_lat = int(data_cfg.get("train_crop_size_lat", 256))
        crop_lon = int(data_cfg.get("train_crop_size_lon", 256))
        self.crop_size: Tuple[int, int] = (
            min(crop_lat, self.fine_shape[0]),
            min(crop_lon, self.fine_shape[1]),
        )
        self.training_spatial_sampling = str(
            data_cfg.get("training_spatial_sampling", "random")
        ).lower()
        self.training_tile_stride: Tuple[int, int] = (
            int(data_cfg.get("training_tile_stride_lat", self.crop_size[0])),
            int(data_cfg.get("training_tile_stride_lon", self.crop_size[1])),
        )
        self.training_halo: Tuple[int, int] = (
            int(data_cfg.get("training_halo_lat", 0))
            if mode in {"training", "validation"}
            else 0,
            int(data_cfg.get("training_halo_lon", 0))
            if mode in {"training", "validation"}
            else 0,
        )
        if min(self.training_halo) < 0:
            raise ValueError(f"training halo must be non-negative, got {self.training_halo}")
        model_cfg = self.cfg.get("model", {}) or {}
        if mode in {"training", "validation"} and str(
            model_cfg.get("backbone_attention_scope", "legacy_global")
        ).lower() == "windowed_local":
            overlap = tuple(
                core - stride
                for core, stride in zip(self.crop_size, self.training_tile_stride)
            )
            training_plan = TilePlan.build(
                self.fine_shape,
                self.crop_size,
                overlap=overlap,
                halo=self.training_halo,
            )
            training_plan.assert_globally_aligned(
                self.cfg.get("mask_unit_size", [16, 16])
            )
        self._tile_slices: Optional[List[Tuple[slice, slice]]] = None
        self._rng = np.random.default_rng(0)
        self.min_valid_target_fraction = float(
            data_cfg.get("min_valid_target_fraction", 1.0e-4)
        )
        self.max_crop_retries = max(1, int(data_cfg.get("max_crop_retries", 32)))
        self.skip_empty_target_tiles = bool(
            data_cfg.get("skip_empty_target_tiles", True)
        )
        self._target_valid_mask: Optional[np.ndarray] = None

        if mode in {"training", "validation"} and self.training_spatial_sampling == "tiled":
            lat_origins = _tile_origins(
                self.fine_shape[0], self.crop_size[0], self.training_tile_stride[0]
            )
            lon_origins = _tile_origins(
                self.fine_shape[1], self.crop_size[1], self.training_tile_stride[1]
            )
            candidate_tiles = [
                (
                    slice(lat0, lat0 + self.crop_size[0]),
                    slice(lon0, lon0 + self.crop_size[1]),
                )
                for lat0 in lat_origins
                for lon0 in lon_origins
            ]
            if self.skip_empty_target_tiles and self.min_valid_target_fraction > 0.0:
                self._tile_slices = [
                    tile
                    for tile in candidate_tiles
                    if self.target_valid_fraction(*tile) >= self.min_valid_target_fraction
                ]
            else:
                self._tile_slices = candidate_tiles
            if not self._tile_slices:
                raise ValueError(
                    "No deterministic training tiles remain after target-mask filtering"
                )
            print(
                f"[dataset] deterministic training tiles: "
                f"{len(lat_origins)}x{len(lon_origins)}={len(candidate_tiles)} "
                f"candidate, {len(self._tile_slices)} kept, tile={self.crop_size}, "
                f"stride={self.training_tile_stride}, halo={self.training_halo}"
            )

        # Random crops for ordinary training; tiled training enumerates fixed windows.
        self.random_crop = mode == "training" and self._tile_slices is None

        # NOTE: the dataset deliberately does NOT z-score x or y. The model
        # normalizes inputs and denormalizes outputs internally via its
        # input/output scalers, and the loss is computed in physical units.
        #
        # For missing-data handling we DO load the precomputed input MEANS: NaN
        # predictor cells (pressure levels below the surface over high terrain)
        # are filled with their channel mean so that, after the model's internal
        # z-score, they become the neutral value 0. A separate binary validity
        # mask channel tells the model which cells were originally missing.
        self._scalars: Dict[str, np.ndarray] = {}
        # Resolve the scaler directory to the case-scoped location so the NaN-fill
        # means loaded here are byte-for-byte the same per-channel scalers the
        # model uses for its internal z-score (single source of truth).
        if scalars_dir is not None:
            scalar_dir = str(resolve_path(scalars_dir).resolve())
        else:
            # Case-scoped resolution only. Never fall back to a shared/flat
            # scalar_dir: that historically mixed another case's stale
            # per-gridpoint scalers into this run. A missing case_name raises a
            # clear error here (a config bug); merely-absent scalers resolve to
            # the canonical (empty) dir and are tolerated for the first scalar
            # pass below.
            scalar_dir = str(norm.resolve_scalar_dir(self.cfg, for_writing=False))
        self._fill_means: Optional[np.ndarray] = None
        if scalar_dir:
            self._scalars = self._load_scalars(scalar_dir)
            im = self._scalars.get("inputs_mean")
            if im is not None:
                norm.assert_channel_only("inputs_mean", im)
                # inputs_mean is [data_means(N), mask_means(N)] -> take the data
                # half for neutral NaN-fill (mask means are 0).
                n_data = len(self.predictor_vars) + 1  # + elevation
                self._fill_means = np.asarray(im[:n_data], dtype=np.float32)

        # Load static elevation ON THE PRISM GRID (appended as the last input
        # channel) so it is co-registered with the regridded predictors.
        elev_file = data_cfg.get("static_elevation_file", None)
        elev_var = data_cfg.get("static_elevation_var", None)
        if elev_file and not self.use_preprocessed:
            elev_path = resolve_path(elev_file)
            if elev_path.exists():
                full_elevation = load_elevation(
                    elev_path,
                    var_name=elev_var or None,
                    target_lat=self._full_fine_lat,
                    target_lon=self._full_fine_lon,
                )
                self._elevation = full_elevation[
                    self._domain_lat_slice, self._domain_lon_slice
                ]
                print(
                    f"[dataset] loaded static elevation {self._elevation.shape} "
                    f"(on PRISM grid) from {elev_path.name}"
                )
            else:
                raise FileNotFoundError(
                    f"Configured static_elevation_file was not found: {elev_path}"
                )

    # ------------------------------------------------------------------
    # Fine (target) grid helpers
    # ------------------------------------------------------------------
    def _load_fine_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the PRISM (fine) target lat/lon coordinate arrays."""
        first_var = self.target_vars[0]
        first_date = self._dates[0]
        path = self._target_maps[first_var][first_date]
        with xr.open_dataset(str(path)) as ds:
            lat_name = _infer_coord_name(ds, LAT_CANDIDATES)
            lon_name = _infer_coord_name(ds, LON_CANDIDATES)
            return (
                ds[lat_name].values.astype(np.float64),
                ds[lon_name].values.astype(np.float64),
            )

    def _load_validated_target_array(self, ds: Any, path: Path) -> np.ndarray:
        """Load one target on, and only on, this case's canonical PRISM grid."""
        dvar = _find_numeric_datavar(ds, path)
        da = ds[dvar]
        if "time" in da.dims:
            da = da.isel(time=0, drop=True)
        lat_name = _infer_coord_name(ds, LAT_CANDIDATES)
        lon_name = _infer_coord_name(ds, LON_CANDIDATES)
        observed_lat = np.asarray(ds[lat_name].values, dtype=np.float64)[
            self._domain_lat_slice
        ]
        observed_lon = np.asarray(ds[lon_name].values, dtype=np.float64)[
            self._domain_lon_slice
        ]
        prism_grid_contract.assert_grid_matches(
            self._canonical_grid,
            observed_lat,
            observed_lon,
            context=f"PRISM target {path}",
        )
        if lat_name not in da.dims or lon_name not in da.dims:
            raise ValueError(
                f"PRISM target {path} variable {dvar!r} must use coordinate "
                f"dimensions ({lat_name!r}, {lon_name!r}); got {da.dims}"
            )
        da = da.transpose(lat_name, lon_name)
        arr = np.asarray(da.values, dtype=np.float32)
        arr[np.isinf(arr)] = np.nan
        return arr

    def _validate_preprocessed_product(
        self, ds: Any, path: Path, sample_date: Any
    ) -> Tuple[str, str]:
        return prism_preprocessed.validate_daily_product(
            ds,
            path,
            self._canonical_grid,
            mode=self.mode,
            sample_date=sample_date,
            required_variables=self._preprocessed_required_variables,
            expected_preprocessing_fields=self._preprocessed_config_fields,
            expected_preprocessing_signature=self._preprocessed_signature,
            expected_source_artifact_signature=(
                self._training_source_artifact_signatures.get(str(sample_date))
            ),
        )

    def _select_crop(self, tile_index: Optional[int] = None) -> Tuple[slice, slice]:
        """Pick a (lat_slice, lon_slice) crop window on the fine grid."""
        if self._tile_slices is not None:
            if tile_index is None:
                raise ValueError("tile_index is required for tiled training")
            return self._tile_slices[tile_index]

        ch, cw = self.crop_size
        if self.crop_size == self.fine_shape:
            return slice(None), slice(None)
        if self.random_crop:
            lat0 = int(self._rng.integers(0, self.fine_shape[0] - ch + 1))
            lon0 = int(self._rng.integers(0, self.fine_shape[1] - cw + 1))
        else:
            lat0 = (self.fine_shape[0] - ch) // 2
            lon0 = (self.fine_shape[1] - cw) // 2
        return slice(lat0, lat0 + ch), slice(lon0, lon0 + cw)

    def _load_target_valid_mask(self) -> np.ndarray:
        """Return cells where every configured target variable is finite."""
        if self._target_valid_mask is not None:
            return self._target_valid_mask

        first_date = self._dates[0]
        targets = self._load_targets(first_date).detach().cpu().numpy()
        self._target_valid_mask = np.isfinite(targets).all(axis=0)
        return self._target_valid_mask

    def target_valid_fraction(self, lat_slice: slice, lon_slice: slice) -> float:
        """Fraction of cells with finite PRISM targets for a local subdomain tile."""
        mask = self._load_target_valid_mask()
        return float(mask[lat_slice, lon_slice].mean())

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
        if self._tile_slices is not None:
            return len(self._dates) * len(self._tile_slices)
        return len(self._dates)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        if self._tile_slices is not None:
            date_index = index // len(self._tile_slices)
            tile_index = index % len(self._tile_slices)
        else:
            date_index = index
            tile_index = None

        sample_date = self._dates[date_index]
        lat_slice, lon_slice = self._select_crop(tile_index)
        if self.use_preprocessed and not self._preprocessed_include_targets:
            lat0, lon0 = self._slice_start(lat_slice), self._slice_start(lon_slice)
            lat1 = self.fine_shape[0] if lat_slice.stop is None else int(lat_slice.stop)
            lon1 = self.fine_shape[1] if lon_slice.stop is None else int(lon_slice.stop)
            y = torch.zeros(
                (len(self.target_vars), lat1 - lat0, lon1 - lon0),
                dtype=self.dtype,
            )
        else:
            y = self._load_targets(sample_date, lat_slice, lon_slice)

        # Random PRISM crops can land entirely over missing ocean/outside-CONUS
        # pixels. The loss masks those safely, but such samples provide no
        # learning signal and used to trigger zero-RMSE edge cases. During
        # training, retry a few crop windows until at least a tiny target
        # fraction is finite.
        if self.random_crop and self.min_valid_target_fraction > 0.0:
            valid_fraction = float(torch.isfinite(y).to(torch.float32).mean().item())
            attempts = 1
            while (
                valid_fraction < self.min_valid_target_fraction
                and attempts < self.max_crop_retries
            ):
                lat_slice, lon_slice = self._select_crop()
                y = self._load_targets(sample_date, lat_slice, lon_slice)
                valid_fraction = float(torch.isfinite(y).to(torch.float32).mean().item())
                attempts += 1

        # Raw, physical-unit predictors (regridded onto the PRISM crop) and
        # targets. NO normalization here -- the model normalizes inputs and
        # denormalizes outputs internally, and the loss is in physical units.
        output_y0 = self._slice_start(lat_slice)
        output_x0 = self._slice_start(lon_slice)
        output_h, output_w = int(y.shape[-2]), int(y.shape[-1])
        (input_lat_slice, input_lon_slice), halo_padding = halo_crop_slices(
            self.fine_shape,
            (output_y0, output_x0),
            (output_h, output_w),
            self.training_halo,
        )
        x = self._load_predictor(sample_date, input_lat_slice, input_lon_slice)
        x = pad_spatial_context(x, halo_padding, pad_mode="reflect")
        expected_input_shape = (
            output_h + 2 * self.training_halo[0],
            output_w + 2 * self.training_halo[1],
        )
        if tuple(x.shape[-2:]) != expected_input_shape:
            raise ValueError(
                f"halo predictor crop has shape {tuple(x.shape[-2:])}, expected "
                f"{expected_input_shape} for output core {(output_h, output_w)}"
            )

        sample = {
            "x": x,
            "y": y,
            "date": str(sample_date),
            # Legacy hook for per-gridpoint (spatial) scalers: the model uses it
            # only when scalers are [C,H,W] to crop the matching tile. With the
            # default per-channel (global) scalers the model broadcasts [C,1,1]
            # and ignores this offset, so MERRA and NARR share one code path.
            "__scaler_offset": torch.tensor(
                [output_y0, output_x0],
                dtype=torch.long,
            ),
            "__input_scaler_offset": torch.tensor(
                [output_y0 - self.training_halo[0], output_x0 - self.training_halo[1]],
                dtype=torch.long,
            ),
            "__output_scaler_offset": torch.tensor(
                [output_y0, output_x0], dtype=torch.long
            ),
            "__output_crop": torch.tensor(
                [*self.training_halo, output_h, output_w], dtype=torch.long
            ),
        }
        if tile_index is not None:
            sample["tile_index"] = tile_index
        return sample

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------
    def _fine_coords(
        self, lat_slice: slice, lon_slice: slice
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.fine_lat[lat_slice], self.fine_lon[lon_slice]

    @staticmethod
    def _slice_start(s: slice) -> int:
        return 0 if s.start is None else int(s.start)

    def _absolute_slices(self, lat_slice: slice, lon_slice: slice) -> Tuple[slice, slice]:
        """Translate local subdomain slices to absolute PRISM-grid slices."""
        domain_lat0 = self._slice_start(self._domain_lat_slice)
        domain_lon0 = self._slice_start(self._domain_lon_slice)
        local_lat0 = self._slice_start(lat_slice)
        local_lon0 = self._slice_start(lon_slice)
        local_lat1 = len(self.fine_lat) if lat_slice.stop is None else int(lat_slice.stop)
        local_lon1 = len(self.fine_lon) if lon_slice.stop is None else int(lon_slice.stop)
        return (
            slice(domain_lat0 + local_lat0, domain_lat0 + local_lat1),
            slice(domain_lon0 + local_lon0, domain_lon0 + local_lon1),
        )

    def _check_predictor_alignment(
        self,
        stacked: np.ndarray,
        crop_lat: np.ndarray,
        crop_lon: np.ndarray,
    ) -> None:
        """Sanity check: predictors must be on the PRISM grid BEFORE tiling.

        Each continuous predictor was bilinearly interpolated onto the exact
        PRISM crop coordinates, so the regridded stack must match the crop grid
        cell-for-cell. If it does not, the low-resolution predictors would be
        spatially misaligned with the high-resolution target and produce
        low-resolution block artifacts in the downscaled output.
        """
        exp_hw = (len(crop_lat), len(crop_lon))
        if stacked.shape[-2:] != exp_hw:
            raise ValueError(
                f"Predictor regrid misaligned with PRISM grid: got "
                f"{tuple(stacked.shape[-2:])}, expected {exp_hw} (the PRISM crop)."
            )
        if not getattr(self, "_alignment_logged", False):
            source = (
                "strict preprocessed product"
                if self.use_preprocessed
                else "on-the-fly coarse-to-fine regridding"
            )
            print(
                f"[dataset] predictor->PRISM alignment OK: {source} is on the "
                f"canonical target grid {exp_hw} before tiling."
            )
            self._alignment_logged = True

    def _load_predictor(
        self,
        sample_date: Any,
        lat_slice: slice = slice(None),
        lon_slice: slice = slice(None),
    ) -> torch.Tensor:
        """Load MERRA2 predictors for a date, regridded onto the PRISM crop.

        Each predictor is bilinearly interpolated from the coarse MERRA2 grid
        onto the exact PRISM crop coordinates (``lat_slice``/``lon_slice`` of the
        fine grid). This co-registers the coarse predictors with the fine
        target, exactly like the CORDEX_ML dataset's regridder step. The static
        elevation field (already on the PRISM grid) is appended as the last
        channel so the full tensor is ``[dynamic_vars..., elevation]``.
        Returned values are RAW physical units.
        """
        crop_lat, crop_lon = self._fine_coords(lat_slice, lon_slice)
        arrays: List[np.ndarray] = []
        if self.use_preprocessed:
            path = self._preprocessed_map[sample_date]
            with xr.open_dataset(str(path)) as ds:
                lat_name, lon_name = self._validate_preprocessed_product(
                    ds, path, sample_date
                )
                arrays.extend(
                    prism_preprocessed.read_spatial_variable(
                        ds,
                        name,
                        lat_name,
                        lon_name,
                        lat_slice=lat_slice,
                        lon_slice=lon_slice,
                    )
                    for name in self._preprocessed_predictor_names
                )
                if self._preprocessed_include_elevation:
                    arrays.append(
                        prism_preprocessed.read_spatial_variable(
                            ds,
                            "static_elevation",
                            lat_name,
                            lon_name,
                            lat_slice=lat_slice,
                            lon_slice=lon_slice,
                        )
                    )
            stacked = np.stack(arrays, axis=0)
        else:
            path = self._predictor_map[sample_date]
            with xr.open_dataset(str(path)) as ds:
                lat_name = _infer_coord_name(ds, LAT_CANDIDATES)
                lon_name = _infer_coord_name(ds, LON_CANDIDATES)
                for var, level in self.predictor_vars:
                    if var not in ds.data_vars:
                        raise ValueError(
                            f"MERRA2 file {path} does not contain variable '{var}'"
                        )
                    da = ds[var]
                    if "time" in da.dims:
                        da = da.isel(time=0, drop=True)
                    if "lev" in da.dims:
                        da = da.sel(lev=level, drop=True)
                    da = da.interp(
                        {lat_name: crop_lat, lon_name: crop_lon}, method="linear"
                    )
                    arr = np.asarray(da.values, dtype=np.float32)
                    arr[np.isinf(arr)] = np.nan
                    arrays.append(arr)
            stacked = np.stack(arrays, axis=0)

        self._check_predictor_alignment(stacked, crop_lat, crop_lon)

        # Append elevation crop as the final static channel (always valid).
        if self._elevation is not None and not self.use_preprocessed:
            elev = self._elevation[lat_slice, lon_slice].astype(np.float32)
            stacked = np.concatenate([stacked, elev[np.newaxis]], axis=0)

        # ------------------------------------------------------------------
        # Missing-data handling (NOT zero-filling):
        #   * validity mask = 1 where the cell is finite, 0 where it was NaN
        #     (below-surface / missing);
        #   * fill the NaN data cells with the channel MEAN so that after the
        #     model's internal z-score they become the neutral value 0 (we do
        #     NOT inject physical zeros, which the model would read as real
        #     cold/zero values);
        #   * append the mask as extra channels so the model can learn the
        #     spatial pattern of missing data over high terrain.
        # Channel layout returned: [data_0..N-1, mask_0..N-1].
        # ------------------------------------------------------------------
        n_ch = stacked.shape[0]
        valid = np.isfinite(stacked)
        mask = valid.astype(np.float32)

        if self._fill_means is not None and len(self._fill_means) >= n_ch:
            fill = np.asarray(self._fill_means[:n_ch], dtype=np.float32)[:, None, None]
        else:
            # Fallback before scalars exist (e.g. first scalar pass): per-tile
            # channel nanmean, then 0 for all-NaN channels.
            with np.errstate(all="ignore"):
                fill = np.nanmean(
                    np.where(valid, stacked, np.nan), axis=(1, 2), keepdims=True
                )
            fill = np.nan_to_num(fill, nan=0.0)

        filled = np.where(valid, stacked, fill).astype(np.float32)
        out = np.concatenate([filled, mask], axis=0)
        return torch.from_numpy(out).to(self.dtype)

    def _load_predictor_day(
        self,
        sample_date: Any,
    ) -> torch.Tensor:
        """Load one full day of predictors regridded onto the PRISM grid.

        Inference uses many overlapping tiles from the same day. Calling
        ``_load_predictor`` for every tile reopens the same NetCDF file and
        repeats xarray interpolation hundreds or thousands of times. This
        helper performs the same interpolation and missing-data fill once on
        the full PRISM grid, so tile inference can slice from the cached tensor
        without changing the predictor values.
        """
        return self._load_predictor(sample_date, slice(None), slice(None))

    def _load_targets(
        self,
        sample_date: Any,
        lat_slice: slice = slice(None),
        lon_slice: slice = slice(None),
    ) -> torch.Tensor:
        """Load all PRISM target variables (raw physical) for a date/crop.

        NaN target cells (PRISM is missing over ocean / outside CONUS, ~44% of
        the grid) are PRESERVED so the loss can mask them. They are NOT replaced
        with zeros (which would be confused with real ppt=0 dry cells and with
        real cold temperatures).
        """
        arrays: List[np.ndarray] = []
        if self.use_preprocessed:
            if not self._preprocessed_include_targets:
                raise RuntimeError(
                    "This MERRA inference split was intentionally preprocessed "
                    "without observed target_* variables"
                )
            path = self._preprocessed_map[sample_date]
            with xr.open_dataset(str(path)) as ds:
                lat_name, lon_name = self._validate_preprocessed_product(
                    ds, path, sample_date
                )
                arrays.extend(
                    prism_preprocessed.read_spatial_variable(
                        ds,
                        name,
                        lat_name,
                        lon_name,
                        lat_slice=lat_slice,
                        lon_slice=lon_slice,
                    )
                    for name in self._preprocessed_target_names
                )
        else:
            for var in self.target_vars:
                path = self._target_maps[var][sample_date]
                with xr.open_dataset(str(path)) as ds:
                    arr = self._load_validated_target_array(ds, path)
                    abs_lat_slice, abs_lon_slice = self._absolute_slices(
                        lat_slice, lon_slice
                    )
                    arr = arr[abs_lat_slice, abs_lon_slice]
                    arrays.append(arr)

        stacked = np.stack(arrays, axis=0)
        return torch.from_numpy(stacked).to(self.dtype)

    # Metadata helpers
    # ------------------------------------------------------------------
    @property
    def dates(self) -> List[Any]:
        return list(self._dates)

    @property
    def has_observed_targets(self) -> bool:
        return not self.use_preprocessed or self._preprocessed_include_targets

    @property
    def predictor_preprocessing_signature(self) -> Optional[str]:
        return self._predictor_preprocessing_signature

    @property
    def training_source_artifact_split_signature(self) -> Optional[str]:
        return self._training_source_artifact_split_signature

    @property
    def num_predictor_channels(self) -> int:
        """Total input channels = 2 × (dynamic MERRA2 vars + elevation):
        N physical data channels followed by N binary validity-mask channels."""
        n_data = len(self.predictor_vars) + (1 if self._elevation is not None else 0)
        return 2 * n_data

    @property
    def num_target_channels(self) -> int:
        return len(self.target_vars)

    @property
    def has_elevation(self) -> bool:
        return self._elevation is not None
