#!/usr/bin/env python3
"""Evaluate Phase-1 and named NARR--PRISM refinement daily products.

This evaluator is intentionally independent of model construction and
training.  It streams exact-date NetCDF files, requires the persisted canonical
PRISM grid, and compares every method on the same finite target/grid cells.
Scalar and map moments are exact.  Distribution/tail metrics use a
deterministic bounded reservoir when the full evaluation contains more values
than ``reservoir_size``; the JSON report records whether those metrics were
exact or sampled.

Example::

    mamba run -n Prithvi python examples/NARR_PRISM/evaluate_refinement.py \
      --config examples/NARR_PRISM/NARR_PRISM_diffusion_unet.yaml \
      --phase1-dir examples/NARR_PRISM/experiments/inference_output \
      --method diffusion=.../refinement_diffusion_unet \
      --method flow=.../refinement_flow_matching_unet \
      --output-dir examples/NARR_PRISM/experiments/refinement_evaluation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from granitewxc.refinement.evaluation import (  # noqa: E402
    deterministic_metrics,
    ensemble_metrics,
    precipitation_metrics,
    temperature_metrics,
)
from granitewxc.utils.normalization import (  # noqa: E402
    TARGET_VALID_MASK_CRITERION,
    case_preprocess_dir,
)
from granitewxc.utils.prism_evaluation import (  # noqa: E402
    exact_axis_slice,
    validate_evaluation_coordinates,
)
from granitewxc.utils.prism_grid import (  # noqa: E402
    CanonicalPrismGrid,
    load_canonical_grid,
)

VARIABLES = ("ppt", "tmax", "tmin")
DATE_RE = re.compile(r"(\d{8})")
MEMBER_RE_TEMPLATE = r"^{variable}_member_(\d+)$"
LAT_NAMES = ("lat", "latitude", "y")
LON_NAMES = ("lon", "longitude", "x")
UNITS = {"ppt": "mm/day", "tmax": "degC", "tmin": "degC"}
TARGET_VALID_MASK_CONTENT_SHA256_ATTR = "target_valid_mask_content_sha256"
TARGET_VALID_MASK_SHA256_ATTR = "target_valid_mask_sha256"
TARGET_VALID_MASK_CRITERION_ATTR = "target_valid_mask_criterion"
TARGET_VALID_MASK_SOURCE_SPLIT_ATTR = "target_valid_mask_source_split"
TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR = (
    "target_valid_mask_training_source_artifact_split_signature"
)
TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR = (
    "target_valid_mask_grid_fingerprint"
)
SEASONS = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}
EVALUATION_SPLITS = ("validation", "inference")


def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return (
        path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    )


def _target_valid_mask_content_sha256(mask: np.ndarray) -> str:
    """Hash canonical boolean mask cells independently of NetCDF encoding."""
    values = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    digest = hashlib.sha256()
    digest.update(b"granitewxc-target-valid-mask-cells-v1\0")
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = _resolve_path(path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: configuration must be a mapping")
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"{config_path}: data section is required")
    target_order = list(data.get("target_variables", ()))
    output_order = list(data.get("output_vars", ()))
    if target_order != list(VARIABLES) or output_order != list(VARIABLES):
        raise ValueError(
            "NARR--PRISM refinement evaluation requires explicit identical "
            f"target/output order {list(VARIABLES)}; got "
            f"targets={target_order}, "
            f"outputs={output_order}"
        )
    case_name = str(config.get("case_name", "")).strip()
    if not case_name:
        raise ValueError(f"{config_path}: case_name is required")
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError(
            f"{config_path}: evaluation.truth_units is required so raw "
            "PRISM files without unit attributes are never interpreted "
            "implicitly"
        )
    truth_units = evaluation.get("truth_units")
    if not isinstance(truth_units, Mapping):
        raise ValueError(
            f"{config_path}: evaluation.truth_units must be a mapping with "
            f"the exact physical-unit contract {UNITS}"
        )
    configured_units = {
        str(variable): str(unit) for variable, unit in truth_units.items()
    }
    if configured_units != UNITS:
        raise ValueError(
            f"{config_path}: evaluation.truth_units must exactly equal "
            f"{UNITS}; got {configured_units}"
        )
    return config_path, config


def _parse_date(value: str) -> date:
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _daily_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError(f"end date {end} precedes start date {start}")
    return [
        start + timedelta(days=index)
        for index in range((end - start).days + 1)
    ]


def _evaluation_date_contract(
    config: Mapping[str, Any],
    *,
    split: str,
    start: str | None,
    end: str | None,
) -> tuple[date, date, date, date]:
    """Resolve an exact configured split and optional explicit subrange.

    Supplying either boundary opts into a subset evaluation, but both are
    required so a typo cannot silently change only half the scientific period.
    The explicit range must remain fully inside the configured split.
    """
    split = str(split).strip().lower()
    if split not in EVALUATION_SPLITS:
        raise ValueError(
            f"split must be one of {EVALUATION_SPLITS}, got {split!r}"
        )
    split_dates = config.get("dates", {}).get(split, {})
    if not isinstance(split_dates, Mapping):
        raise ValueError(f"dates.{split} must be a mapping")
    configured_start_raw = split_dates.get("start")
    configured_end_raw = split_dates.get("end")
    if not configured_start_raw or not configured_end_raw:
        raise ValueError(
            f"dates.{split}.start and dates.{split}.end are required"
        )
    configured_start = _parse_date(str(configured_start_raw))
    configured_end = _parse_date(str(configured_end_raw))
    _daily_range(configured_start, configured_end)

    if (start is None) != (end is None):
        raise ValueError(
            "--start and --end must be supplied together for an intentional "
            "evaluation subset"
        )
    selected_start = configured_start if start is None else _parse_date(start)
    selected_end = configured_end if end is None else _parse_date(end)
    _daily_range(selected_start, selected_end)
    if selected_start < configured_start or selected_end > configured_end:
        raise ValueError(
            f"requested evaluation range {selected_start}..{selected_end} is "
            f"outside configured dates.{split} "
            f"{configured_start}..{configured_end}"
        )
    return configured_start, configured_end, selected_start, selected_end


def _dated_files(directory: Path) -> dict[date, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"daily-file directory not found: {directory}")
    files: dict[date, Path] = {}
    for path in sorted(directory.glob("*.nc*")):
        if not path.is_file():
            continue
        match = DATE_RE.search(path.name)
        if match is None:
            continue
        try:
            sample_date = datetime.strptime(match.group(1), "%Y%m%d").date()
        except ValueError:
            continue
        if sample_date in files:
            raise ValueError(
                f"duplicate daily files for {sample_date}: "
                f"{files[sample_date]} and {path}"
            )
        files[sample_date] = path
    return files


def _case_directory(raw: str | Path, case_name: str) -> Path:
    root = _resolve_path(raw)
    if root.is_dir() and _dated_files(root):
        return root
    candidate = root / case_name
    if candidate.is_dir() and _dated_files(candidate):
        return candidate
    raise FileNotFoundError(
        f"no dated NetCDF files found in {root} or case directory {candidate}"
    )


def _truth_files(
    target_root: Path,
    variable: str,
    dates: Sequence[date],
) -> dict[date, Path]:
    required = set(dates)
    files: dict[date, Path] = {}
    for year in sorted({value.year for value in dates}):
        directory = target_root / variable / str(year)
        if not directory.is_dir():
            continue
        for sample_date, path in _dated_files(directory).items():
            if sample_date in required:
                files[sample_date] = path
    return files


def _require_dates(
    files: Mapping[date, Path],
    dates: Sequence[date],
    *,
    context: str,
) -> None:
    missing = [value for value in dates if value not in files]
    if missing:
        preview = ", ".join(str(value) for value in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"{context}: missing {len(missing)} required daily files: "
            f"{preview}{suffix}"
        )


def _coord_name(
    dataset: Any, candidates: Sequence[str], *, context: str
) -> str:
    lookup = {str(name).casefold(): str(name) for name in dataset.variables}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    raise ValueError(f"{context}: no coordinate among {tuple(candidates)}")


def _attribute_text(
    dataset: Any,
    name: str,
    *,
    context: str,
    required: bool = True,
) -> str | None:
    if name not in dataset.attrs:
        if required:
            raise ValueError(
                f"{context}: required attribute {name!r} is absent"
            )
        return None
    raw = dataset.attrs[name]
    value = (
        raw.decode("utf-8", errors="strict")
        if isinstance(raw, bytes)
        else str(raw)
    ).strip()
    if not value and required:
        raise ValueError(f"{context}: required attribute {name!r} is empty")
    return value or None


def _dataset_date(dataset: Any, *, context: str) -> date:
    if "time" not in dataset.variables:
        raise ValueError(f"{context}: daily product has no time coordinate")
    values = np.asarray(dataset["time"].values).reshape(-1)
    if values.size != 1:
        raise ValueError(
            f"{context}: daily product must contain exactly one time; "
            f"got {values.size}"
        )
    value = values[0]
    try:
        if np.issubdtype(np.asarray(value).dtype, np.datetime64):
            token = str(np.datetime64(value, "D"))
        else:
            token = str(value)[:10]
        return _parse_date(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context}: cannot decode daily time value {value!r}"
        ) from exc


def _validate_product_date(
    dataset: Any,
    expected: date,
    *,
    context: str,
    expected_split: str,
    configured_split_start: date,
    configured_split_end: date,
) -> dict[str, str]:
    observed = _dataset_date(dataset, context=context)
    if observed != expected:
        raise ValueError(
            f"{context}: time encodes {observed}, expected {expected}"
        )
    inference_date = _attribute_text(
        dataset, "inference_date", context=context, required=False
    )
    if inference_date is not None and inference_date != expected.isoformat():
        raise ValueError(
            f"{context}: inference_date={inference_date!r}, expected "
            f"{expected.isoformat()!r}"
        )
    for name in ("inference_start", "inference_end"):
        stored = _attribute_text(
            dataset, name, context=context, required=False
        )
        if stored is not None and stored != expected.isoformat():
            raise ValueError(
                f"{context}: {name}={stored!r}, expected "
                f"{expected.isoformat()!r}"
            )
    if inference_date is None and all(
        name not in dataset.attrs
        for name in ("inference_start", "inference_end")
    ):
        raise ValueError(
            f"{context}: daily product lacks inference_date or "
            "inference_start/inference_end provenance"
        )
    stored_split = _attribute_text(
        dataset, "dataset_split", context=context, required=True
    )
    stored_split_start = _attribute_text(
        dataset, "split_start", context=context, required=True
    )
    stored_split_end = _attribute_text(
        dataset, "split_end", context=context, required=True
    )
    expected_bounds = (
        configured_split_start.isoformat(),
        configured_split_end.isoformat(),
    )
    if stored_split != expected_split:
        raise ValueError(
            f"{context}: dataset_split={stored_split!r}, expected "
            f"{expected_split!r}"
        )
    if (stored_split_start, stored_split_end) != expected_bounds:
        raise ValueError(
            f"{context}: product split bounds "
            f"{stored_split_start!r}..{stored_split_end!r} do not match "
            f"configured dates.{expected_split} "
            f"{expected_bounds[0]!r}..{expected_bounds[1]!r}"
        )
    return {
        "dataset_split": stored_split,
        "split_start": stored_split_start,
        "split_end": stored_split_end,
    }


def _validate_grid(
    dataset: Any,
    grid: CanonicalPrismGrid,
    *,
    context: str,
) -> tuple[str, str]:
    fingerprint = _attribute_text(
        dataset, "prism_grid_fingerprint", context=context
    )
    if fingerprint != grid.fingerprint:
        raise ValueError(
            f"{context}: prism_grid_fingerprint={fingerprint!r} does not "
            "match "
            f"canonical fingerprint {grid.fingerprint!r}"
        )
    lat_name = _coord_name(dataset, LAT_NAMES, context=context)
    lon_name = _coord_name(dataset, LON_NAMES, context=context)
    validate_evaluation_coordinates(
        grid.lat,
        dataset[lat_name].values,
        name="latitude",
        context=context,
    )
    validate_evaluation_coordinates(
        grid.lon,
        dataset[lon_name].values,
        name="longitude",
        context=context,
    )
    return lat_name, lon_name


def _extract_field(
    dataset: Any,
    name: str,
    lat_name: str,
    lon_name: str,
    *,
    context: str,
) -> np.ndarray:
    if name not in dataset.data_vars:
        raise ValueError(
            f"{context}: required variable {name!r} is absent; "
            f"available={list(map(str, dataset.data_vars))}"
        )
    data = dataset[name]
    for dimension in list(data.dims):
        if dimension not in (lat_name, lon_name):
            if int(data.sizes[dimension]) != 1:
                raise ValueError(
                    f"{context}: {name!r} has non-singleton extra dimension "
                    f"{dimension}={data.sizes[dimension]}"
                )
            data = data.isel({dimension: 0}, drop=True)
    values = np.asarray(
        data.transpose(lat_name, lon_name).values, dtype=np.float64
    )
    values[~np.isfinite(values)] = np.nan
    return values


def _normalize_unit(value: Any) -> str:
    text = str(value or "").strip().casefold()
    return re.sub(r"[\s_]+", "", text).replace("°", "deg")


def _validate_units(
    dataset: Any,
    name: str,
    variable: str,
    *,
    configured_unit: str | None = None,
    context: str,
) -> tuple[str, str | None]:
    """Validate source metadata or apply the explicit YAML unit contract.

    Historical PRISM files in this workflow do not consistently carry a
    ``units`` attribute.  Missing metadata is accepted only because
    ``evaluation.truth_units`` was validated against :data:`UNITS` when the
    configuration was loaded.  A present attribute always takes precedence
    and must describe the same physical units.
    """

    if configured_unit is not None and configured_unit != UNITS[variable]:
        raise ValueError(
            f"{context}: configured truth unit for {variable!r} must be "
            f"{UNITS[variable]!r}, got {configured_unit!r}"
        )
    raw_unit = dataset[name].attrs.get("units")
    if raw_unit is None or not str(raw_unit).strip():
        if configured_unit is None:
            raise ValueError(
                f"{context}: {name!r} has no units attribute; only raw "
                "PRISM truth may use the explicit evaluation.truth_units "
                "contract"
            )
        return "configured_assumption", None
    unit = _normalize_unit(raw_unit)
    accepted = {
        "ppt": {"mm/day", "mmday-1", "mmd-1", "mm"},
        "tmax": {"degc", "c", "celsius", "degreecelsius", "degreescelsius"},
        "tmin": {"degc", "c", "celsius", "degreecelsius", "degreescelsius"},
    }[variable]
    if unit not in accepted:
        raise ValueError(
            f"{context}: {name!r} units {raw_unit!r} "
            f"are incompatible with physical {UNITS[variable]}"
        )
    return "source_attribute", str(raw_unit)


def _parse_variable_order(dataset: Any, *, context: str) -> list[str]:
    raw = _attribute_text(dataset, "variable_order", context=context)
    assert raw is not None
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{context}: variable_order is not valid JSON"
        ) from exc
    if values != list(VARIABLES):
        raise ValueError(
            f"{context}: variable_order must be {list(VARIABLES)}, "
            f"got {values!r}"
        )
    return list(values)


def _extract_members(
    dataset: Any,
    variable: str,
    lat_name: str,
    lon_name: str,
    *,
    context: str,
) -> np.ndarray | None:
    packed_name = f"{variable}_members"
    if packed_name in dataset.data_vars:
        data = dataset[packed_name]
        member_dims = [
            dimension
            for dimension in data.dims
            if dimension not in (lat_name, lon_name, "time")
        ]
        if len(member_dims) != 1:
            raise ValueError(
                f"{context}: {packed_name!r} must have one member dimension"
            )
        if "time" in data.dims:
            if int(data.sizes["time"]) != 1:
                raise ValueError(
                    f"{context}: {packed_name!r} time must be singleton"
                )
            data = data.isel(time=0, drop=True)
        member_dim = member_dims[0]
        values = np.asarray(
            data.transpose(member_dim, lat_name, lon_name).values,
            dtype=np.float64,
        )
        values[~np.isfinite(values)] = np.nan
        return values

    pattern = re.compile(
        MEMBER_RE_TEMPLATE.format(variable=re.escape(variable))
    )
    numbered: list[tuple[int, str]] = []
    for name in dataset.data_vars:
        match = pattern.match(str(name))
        if match:
            numbered.append((int(match.group(1)), str(name)))
    if not numbered:
        return None
    numbered.sort()
    indices = [index for index, _ in numbered]
    if indices != list(range(len(indices))):
        raise ValueError(
            f"{context}: {variable} member indices must be contiguous from "
            "zero; "
            f"got {indices}"
        )
    members = [
        _extract_field(dataset, name, lat_name, lon_name, context=context)
        for _, name in numbered
    ]
    for _, name in numbered:
        _validate_units(dataset, name, variable, context=context)
    return np.stack(members, axis=0)


def _read_product_valid_mask(  # noqa: C901
    dataset: Any,
    lat_name: str,
    lon_name: str,
    grid: CanonicalPrismGrid,
    *,
    context: str,
) -> tuple[np.ndarray, dict[str, str]]:
    """Read and authenticate one product's static training-support mask."""
    name = "prism_valid_mask"
    if name not in dataset.data_vars:
        raise ValueError(
            f"{context}: required variable {name!r} is absent; daily products "
            "must persist their training-derived output support"
        )
    data = dataset[name]
    if set(data.dims) != {lat_name, lon_name} or len(data.dims) != 2:
        raise ValueError(
            f"{context}: {name!r} must have exactly ({lat_name!r}, "
            f"{lon_name!r}) dimensions, got {data.dims}"
        )
    values = np.asarray(data.transpose(lat_name, lon_name).values)
    if values.shape != grid.shape:
        raise ValueError(
            f"{context}: {name!r} shape {values.shape} != canonical "
            f"{grid.shape}"
        )
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"{context}: {name!r} contains missing values")
    if not bool(np.isin(values, (0, 1, False, True)).all()):
        unique = np.unique(values)
        raise ValueError(
            f"{context}: {name!r} must be binary 0/1, got {unique[:8]}"
        )
    mask = np.asarray(values, dtype=bool)
    if not bool(mask.any()):
        raise ValueError(f"{context}: {name!r} contains no valid cells")

    attribute_names = (
        TARGET_VALID_MASK_SHA256_ATTR,
        TARGET_VALID_MASK_CONTENT_SHA256_ATTR,
        TARGET_VALID_MASK_CRITERION_ATTR,
        TARGET_VALID_MASK_SOURCE_SPLIT_ATTR,
        TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR,
        TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR,
    )
    provenance = {
        attribute: str(
            _attribute_text(dataset, attribute, context=context, required=True)
        )
        for attribute in attribute_names
    }
    for attribute in (
        TARGET_VALID_MASK_SHA256_ATTR,
        TARGET_VALID_MASK_CONTENT_SHA256_ATTR,
    ):
        if re.fullmatch(r"[0-9a-f]{64}", provenance[attribute]) is None:
            raise ValueError(
                f"{context}: {attribute} must be a lowercase SHA-256 digest"
            )
    computed = _target_valid_mask_content_sha256(mask)
    if provenance[TARGET_VALID_MASK_CONTENT_SHA256_ATTR] != computed:
        raise ValueError(
            f"{context}: {name!r} content digest mismatch: stored="
            f"{provenance[TARGET_VALID_MASK_CONTENT_SHA256_ATTR]}, "
            f"computed={computed}"
        )
    if (
        provenance[TARGET_VALID_MASK_CRITERION_ATTR]
        != TARGET_VALID_MASK_CRITERION
    ):
        raise ValueError(
            f"{context}: unsupported target-valid-mask criterion "
            f"{provenance[TARGET_VALID_MASK_CRITERION_ATTR]!r}"
        )
    if provenance[TARGET_VALID_MASK_SOURCE_SPLIT_ATTR] != "training":
        raise ValueError(
            f"{context}: target-valid-mask source split must be 'training'"
        )
    source_signature = provenance[
        TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR
    ]
    if (
        source_signature != "not-applicable:raw-training-source-mode"
        and re.fullmatch(r"[0-9a-f]{64}", source_signature) is None
    ):
        raise ValueError(
            f"{context}: {TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR} "
            "must be a SHA-256 digest or the explicit raw-source sentinel"
        )
    if provenance[TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR] != grid.fingerprint:
        raise ValueError(
            f"{context}: target-valid-mask grid fingerprint does not match "
            "the canonical evaluation grid"
        )
    return mask, provenance


@dataclass
class DailyProduct:
    fields: dict[str, np.ndarray]
    members: dict[str, np.ndarray]
    phase1_fields: dict[str, np.ndarray]
    provenance: dict[str, Any]
    valid_mask: np.ndarray
    mask_provenance: dict[str, str]


def _read_product_day(  # noqa: C901
    path: Path,
    grid: CanonicalPrismGrid,
    *,
    expected_date: date,
    case_name: str,
    role: str,
    expected_split: str,
    configured_split_start: date,
    configured_split_end: date,
) -> DailyProduct:
    import xarray as xr

    context = str(path)
    with xr.open_dataset(
        path, decode_times=True, mask_and_scale=True
    ) as dataset:
        stored_case = _attribute_text(dataset, "case_name", context=context)
        if stored_case != case_name:
            raise ValueError(
                f"{context}: case_name={stored_case!r} does not match "
                f"{case_name!r}"
            )
        split_provenance = _validate_product_date(
            dataset,
            expected_date,
            context=context,
            expected_split=expected_split,
            configured_split_start=configured_split_start,
            configured_split_end=configured_split_end,
        )
        lat_name, lon_name = _validate_grid(dataset, grid, context=context)
        product_valid_mask, mask_provenance = _read_product_valid_mask(
            dataset,
            lat_name,
            lon_name,
            grid,
            context=context,
        )

        fields: dict[str, np.ndarray] = {}
        members: dict[str, np.ndarray] = {}
        phase1_fields: dict[str, np.ndarray] = {}
        if role == "phase1":
            checkpoint = _attribute_text(
                dataset, "checkpoint", context=context, required=False
            ) or _attribute_text(dataset, "phase1_checkpoint", context=context)
            for variable in VARIABLES:
                name = (
                    f"{variable}_phase1"
                    if f"{variable}_phase1" in dataset.data_vars
                    else variable
                )
                _validate_units(dataset, name, variable, context=context)
                fields[variable] = _extract_field(
                    dataset, name, lat_name, lon_name, context=context
                )
            provenance = {
                "case_name": stored_case,
                "phase1_checkpoint": checkpoint,
                "phase1_fingerprint": _attribute_text(
                    dataset,
                    "phase1_fingerprint",
                    context=context,
                    required=False,
                ),
                "prism_grid_fingerprint": grid.fingerprint,
                **split_provenance,
                "target_valid_mask": mask_provenance,
            }
        elif role == "refinement":
            _parse_variable_order(dataset, context=context)
            if "ensemble_size" not in dataset.attrs:
                raise ValueError(
                    f"{context}: required ensemble_size is absent"
                )
            expected_ensemble_size = int(dataset.attrs["ensemble_size"])
            if expected_ensemble_size < 1:
                raise ValueError(
                    f"{context}: ensemble_size must be positive, got "
                    f"{expected_ensemble_size}"
                )
            if "base_seed" not in dataset.attrs:
                raise ValueError(f"{context}: required base_seed is absent")
            for variable in VARIABLES:
                candidates = (
                    variable,
                    f"{variable}_refined",
                    f"{variable}_ensemble_mean",
                )
                present = [
                    name for name in candidates if name in dataset.data_vars
                ]
                if not present:
                    raise ValueError(
                        f"{context}: no refined mean variable for {variable}; "
                        f"expected one of {candidates}"
                    )
                name = present[0]
                _validate_units(dataset, name, variable, context=context)
                mean = _extract_field(
                    dataset, name, lat_name, lon_name, context=context
                )
                member_values = _extract_members(
                    dataset,
                    variable,
                    lat_name,
                    lon_name,
                    context=context,
                )
                if member_values is None:
                    raise ValueError(
                        f"{context}: no stochastic members stored for "
                        f"{variable}"
                    )
                count = np.isfinite(member_values).sum(axis=0)
                member_mean = np.full(mean.shape, np.nan, dtype=np.float64)
                valid = count > 0
                member_mean[valid] = (
                    np.nansum(member_values, axis=0)[valid] / count[valid]
                )
                if not np.allclose(
                    mean,
                    member_mean,
                    rtol=2.0e-5,
                    atol=2.0e-5,
                    equal_nan=True,
                ):
                    raise ValueError(
                        f"{context}: {variable} refined mean does not equal "
                        "the stored ensemble-member mean"
                    )
                if expected_ensemble_size != member_values.shape[0]:
                    raise ValueError(
                        f"{context}: ensemble_size={expected_ensemble_size} "
                        f"but {member_values.shape[0]} {variable} members "
                        "were stored"
                    )
                members[variable] = member_values
                phase1_name = f"{variable}_phase1"
                if phase1_name not in dataset.data_vars:
                    raise ValueError(
                        f"{context}: required embedded baseline "
                        f"{phase1_name!r} is absent"
                    )
                _validate_units(
                    dataset, phase1_name, variable, context=context
                )
                phase1_fields[variable] = _extract_field(
                    dataset,
                    phase1_name,
                    lat_name,
                    lon_name,
                    context=context,
                )
                fields[variable] = mean
            provenance = {
                "case_name": stored_case,
                "phase1_checkpoint": _attribute_text(
                    dataset, "phase1_checkpoint", context=context
                ),
                "phase2_checkpoint": _attribute_text(
                    dataset, "phase2_checkpoint", context=context
                ),
                "phase1_fingerprint": _attribute_text(
                    dataset, "phase1_fingerprint", context=context
                ),
                "phase2_fingerprint": _attribute_text(
                    dataset, "phase2_fingerprint", context=context
                ),
                "refinement_type": _attribute_text(
                    dataset, "refinement_type", context=context
                ),
                "config_path": _attribute_text(
                    dataset, "config_path", context=context
                ),
                "config_fingerprint": _attribute_text(
                    dataset, "config_fingerprint", context=context
                ),
                "ensemble_size": expected_ensemble_size,
                "base_seed": int(dataset.attrs["base_seed"]),
                "prism_grid_fingerprint": grid.fingerprint,
                "variable_order": list(VARIABLES),
                **split_provenance,
                "target_valid_mask": mask_provenance,
            }
        else:
            raise ValueError(f"unknown product role {role!r}")
    for variable, values in fields.items():
        if values.shape != grid.shape:
            raise ValueError(
                f"{path}: {variable} shape {values.shape} != canonical "
                f"{grid.shape}"
            )
    contract_arrays = {
        **{f"prediction {name}": values for name, values in fields.items()},
        **{
            f"embedded Phase-1 {name}": values
            for name, values in phase1_fields.items()
        },
        **{
            f"ensemble members {name}": values
            for name, values in members.items()
        },
    }
    for label, values in contract_arrays.items():
        expected_valid = (
            np.broadcast_to(product_valid_mask, values.shape)
            if values.ndim > 2
            else product_valid_mask
        )
        observed_valid = np.isfinite(values)
        if not np.array_equal(observed_valid, expected_valid):
            outside = int(np.count_nonzero(observed_valid & ~expected_valid))
            missing = int(np.count_nonzero(expected_valid & ~observed_valid))
            raise ValueError(
                f"{path}: {label} finite-value support does not equal "
                "prism_valid_mask "
                f"(finite outside={outside}, missing inside={missing})"
            )
    return DailyProduct(
        fields=fields,
        members=members,
        phase1_fields=phase1_fields,
        provenance=provenance,
        valid_mask=product_valid_mask,
        mask_provenance=mask_provenance,
    )


def _read_truth_day(
    path: Path,
    variable: str,
    grid: CanonicalPrismGrid,
    *,
    configured_unit: str,
    expected_date: date,
    unit_audit: dict[str, Any] | None = None,
) -> np.ndarray:
    import xarray as xr

    context = str(path)
    with xr.open_dataset(
        path, decode_times=True, mask_and_scale=True
    ) as dataset:
        _validate_truth_date(dataset, expected_date, context=context)
        lat_name = _coord_name(dataset, LAT_NAMES, context=context)
        lon_name = _coord_name(dataset, LON_NAMES, context=context)
        lat_slice = exact_axis_slice(
            dataset[lat_name].values,
            grid.lat,
            name="latitude",
            context=context,
        )
        lon_slice = exact_axis_slice(
            dataset[lon_name].values,
            grid.lon,
            name="longitude",
            context=context,
        )
        candidates = [
            str(name)
            for name, data in dataset.data_vars.items()
            if lat_name in data.dims and lon_name in data.dims
        ]
        exact = [
            name
            for name in candidates
            if name.casefold() == variable.casefold()
        ]
        if len(exact) == 1:
            name = exact[0]
        elif len(candidates) == 1:
            name = candidates[0]
        else:
            raise ValueError(
                f"{context}: cannot identify unique PRISM field for "
                f"{variable}; "
                f"candidates={candidates}"
            )
        unit_source, observed_unit = _validate_units(
            dataset,
            name,
            variable,
            configured_unit=configured_unit,
            context=context,
        )
        if unit_audit is not None:
            unit_audit[f"{unit_source}_files"] += 1
            if observed_unit is not None:
                unit_audit["observed_source_units"].add(observed_unit)
        data = dataset[name]
        for dimension in list(data.dims):
            if dimension not in (lat_name, lon_name):
                if int(data.sizes[dimension]) != 1:
                    raise ValueError(
                        f"{context}: truth {name!r} has non-singleton "
                        f"dimension {dimension}"
                    )
                data = data.isel({dimension: 0}, drop=True)
        values = np.asarray(
            data.transpose(lat_name, lon_name)
            .isel({lat_name: lat_slice, lon_name: lon_slice})
            .values,
            dtype=np.float64,
        )
    if values.shape != grid.shape:
        raise ValueError(
            f"{context}: truth shape {values.shape} != {grid.shape}"
        )
    values[~np.isfinite(values)] = np.nan
    return values


def _validate_truth_date(
    dataset: Any, expected: date, *, context: str
) -> None:
    """Validate an embedded truth date when one exists.

    Some historical single-day PRISM rasters have only spatial dimensions; in
    that case the exact date is necessarily supplied by the validated file
    inventory. If a time coordinate/attribute exists, it must agree.
    """
    if "time" in dataset.variables:
        observed = _dataset_date(dataset, context=context)
        if observed != expected:
            raise ValueError(
                f"{context}: truth time encodes {observed}, expected "
                f"{expected}"
            )
    for attribute in ("date", "time_coverage_start"):
        raw = dataset.attrs.get(attribute)
        if raw is None:
            continue
        try:
            observed = _parse_date(str(raw)[:10])
        except ValueError as exc:
            raise ValueError(
                f"{context}: cannot decode truth {attribute}={raw!r}"
            ) from exc
        if observed != expected:
            raise ValueError(
                f"{context}: truth {attribute} encodes {observed}, expected "
                f"{expected}"
            )


class ProvenanceTracker:
    """Require one immutable provenance signature for every daily product."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.signature: dict[str, Any] | None = None

    def update(self, provenance: Mapping[str, Any], *, path: Path) -> None:
        observed = dict(provenance)
        if self.signature is None:
            self.signature = observed
        elif observed != self.signature:
            differing = sorted(
                key
                for key in set(observed) | set(self.signature)
                if observed.get(key) != self.signature.get(key)
            )
            details = "; ".join(
                f"{key}: first={self.signature.get(key)!r}, "
                f"observed={observed.get(key)!r}"
                for key in differing
            )
            raise ValueError(
                f"{path}: {self.label} mixes daily provenance ({details})"
            )

    def finalize(self) -> dict[str, Any]:
        if self.signature is None:
            raise ValueError(f"no {self.label} products were read")
        return dict(self.signature)


class TargetValidMaskTracker:
    """Require one support mask across every product and method."""

    def __init__(self) -> None:
        self.mask: np.ndarray | None = None
        self.provenance: dict[str, str] | None = None
        self.path: Path | None = None

    def update(self, product: DailyProduct, *, path: Path) -> None:
        if self.mask is None:
            self.mask = product.valid_mask.copy()
            self.provenance = dict(product.mask_provenance)
            self.path = path
            return
        assert self.provenance is not None
        if not np.array_equal(product.valid_mask, self.mask):
            differing = int(np.count_nonzero(product.valid_mask != self.mask))
            raise ValueError(
                f"target-valid-mask cells differ across products: {path} "
                "versus "
                f"{self.path} ({differing} differing cells)"
            )
        if product.mask_provenance != self.provenance:
            differing_fields = sorted(
                key
                for key in set(product.mask_provenance) | set(self.provenance)
                if product.mask_provenance.get(key) != self.provenance.get(key)
            )
            raise ValueError(
                "target-valid-mask provenance differs across products at "
                f"{path}; "
                f"differing fields={differing_fields}"
            )

    def finalize(self) -> dict[str, str]:
        if self.provenance is None:
            raise ValueError("no target-valid-mask products were read")
        return dict(self.provenance)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return (
        float(numerator / denominator) if denominator != 0.0 else float("nan")
    )


def _correlation_from_sums(
    count: np.ndarray | float,
    sum_prediction: np.ndarray | float,
    sum_target: np.ndarray | float,
    sum_prediction_sq: np.ndarray | float,
    sum_target_sq: np.ndarray | float,
    sum_product: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count_array = np.asarray(count, dtype=np.float64)
    safe_count = np.maximum(count_array, 1.0)
    covariance = np.asarray(sum_product) - (
        np.asarray(sum_prediction) * np.asarray(sum_target) / safe_count
    )
    prediction_variance = np.maximum(
        np.asarray(sum_prediction_sq)
        - np.square(np.asarray(sum_prediction)) / safe_count,
        0.0,
    )
    target_variance = np.maximum(
        np.asarray(sum_target_sq)
        - np.square(np.asarray(sum_target)) / safe_count,
        0.0,
    )
    denominator = np.sqrt(prediction_variance * target_variance)
    correlation = np.full(
        np.broadcast_shapes(count_array.shape, denominator.shape), np.nan
    )
    usable = (count_array >= 2.0) & (denominator > 0.0)
    correlation[usable] = covariance[usable] / denominator[usable]
    return (
        np.clip(correlation, -1.0, 1.0),
        prediction_variance,
        target_variance,
    )


class ScalarAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.prediction_sum = 0.0
        self.target_sum = 0.0
        self.prediction_sq_sum = 0.0
        self.target_sq_sum = 0.0
        self.product_sum = 0.0
        self.absolute_error_sum = 0.0
        self.squared_error_sum = 0.0

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        mask = np.asarray(valid, dtype=bool)
        mask &= np.isfinite(prediction) & np.isfinite(target)
        if not bool(mask.any()):
            return
        pred = np.asarray(prediction, dtype=np.float64)[mask]
        truth = np.asarray(target, dtype=np.float64)[mask]
        error = pred - truth
        self.count += int(pred.size)
        self.prediction_sum += float(np.sum(pred, dtype=np.float64))
        self.target_sum += float(np.sum(truth, dtype=np.float64))
        self.prediction_sq_sum += float(
            np.sum(np.square(pred), dtype=np.float64)
        )
        self.target_sq_sum += float(np.sum(np.square(truth), dtype=np.float64))
        self.product_sum += float(np.sum(pred * truth, dtype=np.float64))
        self.absolute_error_sum += float(
            np.sum(np.abs(error), dtype=np.float64)
        )
        self.squared_error_sum += float(
            np.sum(np.square(error), dtype=np.float64)
        )

    def finalize(self) -> dict[str, float | int]:
        if self.count <= 0:
            raise ValueError("no finite values were accumulated")
        count = float(self.count)
        prediction_mean = self.prediction_sum / count
        target_mean = self.target_sum / count
        bias = prediction_mean - target_mean
        mse = self.squared_error_sum / count
        _, prediction_variance_sum, target_variance_sum = (
            _correlation_from_sums(
                float(self.count),
                self.prediction_sum,
                self.target_sum,
                self.prediction_sq_sum,
                self.target_sq_sum,
                self.product_sum,
            )
        )
        covariance_sum = self.product_sum - (
            self.prediction_sum * self.target_sum / count
        )
        denominator = math.sqrt(
            float(prediction_variance_sum) * float(target_variance_sum)
        )
        correlation = _safe_ratio(covariance_sum, denominator)
        return {
            "valid_count": self.count,
            "prediction_mean": prediction_mean,
            "target_mean": target_mean,
            "mean_bias": bias,
            "absolute_bias": abs(bias),
            "mae": self.absolute_error_sum / count,
            "rmse": math.sqrt(max(mse, 0.0)),
            "centered_rmse": math.sqrt(max(mse - bias * bias, 0.0)),
            "pearson_correlation": correlation,
            "standard_deviation_ratio": _safe_ratio(
                math.sqrt(float(prediction_variance_sum) / count),
                math.sqrt(float(target_variance_sum) / count),
            ),
        }


class GradientAccumulator:
    def __init__(self) -> None:
        self.scalar = ScalarAccumulator()
        self.dot_sum = 0.0
        self.prediction_norm_sq = 0.0
        self.target_norm_sq = 0.0

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        for axis in (-2, -1):
            if prediction.shape[axis] < 2:
                continue
            left = [slice(None)] * prediction.ndim
            right = [slice(None)] * prediction.ndim
            left[axis] = slice(None, -1)
            right[axis] = slice(1, None)
            adjacent = valid[tuple(left)] & valid[tuple(right)]
            if not bool(adjacent.any()):
                continue
            pred_gradient = np.diff(prediction, axis=axis)
            target_gradient = np.diff(target, axis=axis)
            self.scalar.update(pred_gradient, target_gradient, adjacent)
            pred_values = pred_gradient[adjacent]
            target_values = target_gradient[adjacent]
            self.dot_sum += float(np.sum(pred_values * target_values))
            self.prediction_norm_sq += float(np.sum(np.square(pred_values)))
            self.target_norm_sq += float(np.sum(np.square(target_values)))

    def finalize(self) -> dict[str, float | int]:
        base = self.scalar.finalize()
        pred_norm = math.sqrt(max(self.prediction_norm_sq, 0.0))
        target_norm = math.sqrt(max(self.target_norm_sq, 0.0))
        gradient_count = int(base["valid_count"])
        prediction_rms = _safe_ratio(
            pred_norm, math.sqrt(float(gradient_count))
        )
        target_rms = _safe_ratio(
            target_norm, math.sqrt(float(gradient_count))
        )
        roughness_ratio = _safe_ratio(prediction_rms, target_rms)
        if pred_norm == 0.0 and target_norm == 0.0:
            agreement = 1.0
        elif pred_norm == 0.0 or target_norm == 0.0:
            agreement = 0.0
        else:
            agreement = float(
                np.clip(self.dot_sum / (pred_norm * target_norm), -1.0, 1.0)
            )
        return {
            "gradient_valid_count": int(base["valid_count"]),
            "gradient_mae": float(base["mae"]),
            "gradient_rmse": float(base["rmse"]),
            "gradient_agreement": agreement,
            "gradient_correlation": float(base["pearson_correlation"]),
            "gradient_standard_deviation_ratio": float(
                base["standard_deviation_ratio"]
            ),
            "prediction_gradient_rms": prediction_rms,
            "target_gradient_rms": target_rms,
            "spatial_roughness_ratio": roughness_ratio,
            "spatial_smoothing_index": 1.0 - roughness_ratio,
        }


class PairReservoir:
    """Deterministic uniform priority reservoir for paired values."""

    def __init__(self, size: int, seed: int) -> None:
        self.size = int(size)
        if self.size < 1:
            raise ValueError("reservoir_size must be at least one")
        self.rng = np.random.default_rng(int(seed))
        self.keys = np.empty(0, dtype=np.float64)
        self.prediction = np.empty(0, dtype=np.float64)
        self.target = np.empty(0, dtype=np.float64)
        self.total_seen = 0

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        mask = valid & np.isfinite(prediction) & np.isfinite(target)
        if not bool(mask.any()):
            return
        pred = np.asarray(prediction, dtype=np.float64)[mask]
        truth = np.asarray(target, dtype=np.float64)[mask]
        self.total_seen += int(pred.size)
        keys = self.rng.random(pred.size)
        keep = min(self.size, pred.size)
        if keep < pred.size:
            selected = np.argpartition(keys, -keep)[-keep:]
            keys = keys[selected]
            pred = pred[selected]
            truth = truth[selected]
        keys = np.concatenate((self.keys, keys))
        pred = np.concatenate((self.prediction, pred))
        truth = np.concatenate((self.target, truth))
        if keys.size > self.size:
            selected = np.argpartition(keys, -self.size)[-self.size :]
            keys = keys[selected]
            pred = pred[selected]
            truth = truth[selected]
        self.keys = keys
        self.prediction = pred
        self.target = truth

    def metadata(self) -> dict[str, Any]:
        return {
            "population_size": self.total_seen,
            "sample_size": int(self.prediction.size),
            "exact": self.total_seen <= self.size,
            "method": "all finite pairs"
            if self.total_seen <= self.size
            else "deterministic uniform priority reservoir",
        }


class TruthMapAccumulator:
    def __init__(self, shape: Sequence[int]) -> None:
        self.count = np.zeros(tuple(shape), dtype=np.uint32)
        self.sum = np.zeros(tuple(shape), dtype=np.float64)
        self.square_sum = np.zeros(tuple(shape), dtype=np.float64)

    def update(self, target: np.ndarray, valid: np.ndarray) -> None:
        self.count[valid] += 1
        self.sum[valid] += target[valid]
        self.square_sum[valid] += np.square(target[valid])


@dataclass
class FinalizedField:
    deterministic: dict[str, float | int]
    specialized: dict[str, float | int]
    sampling: dict[str, Any]
    prediction_climatology: np.ndarray
    target_climatology: np.ndarray
    bias: np.ndarray
    rmse: np.ndarray
    reservoir_prediction: np.ndarray
    reservoir_target: np.ndarray


class MethodFieldAccumulator:
    def __init__(
        self,
        shape: Sequence[int],
        *,
        reservoir_size: int,
        seed: int,
    ) -> None:
        shape = tuple(int(value) for value in shape)
        self.prediction_sum = np.zeros(shape, dtype=np.float64)
        self.prediction_sq_sum = np.zeros(shape, dtype=np.float64)
        self.product_sum = np.zeros(shape, dtype=np.float64)
        self.squared_error_sum = np.zeros(shape, dtype=np.float64)
        self.scalar = ScalarAccumulator()
        self.gradient = GradientAccumulator()
        self.reservoir = PairReservoir(reservoir_size, seed)
        self.spatial_correlation_sum = 0.0
        self.spatial_correlation_count = 0
        self.days = 0

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        self.prediction_sum[valid] += prediction[valid]
        self.prediction_sq_sum[valid] += np.square(prediction[valid])
        self.product_sum[valid] += prediction[valid] * target[valid]
        error = prediction[valid] - target[valid]
        self.squared_error_sum[valid] += np.square(error)
        self.scalar.update(prediction, target, valid)
        self.gradient.update(prediction, target, valid)
        self.reservoir.update(prediction, target, valid)
        daily = deterministic_metrics(
            prediction,
            target,
            mask=valid,
            quantiles=(),
            sample_axis=None,
            spatial_axes=(),
        )
        correlation = float(daily["pearson_correlation"])
        if np.isfinite(correlation):
            self.spatial_correlation_sum += correlation
            self.spatial_correlation_count += 1
        self.days += 1

    def finalize(
        self,
        truth: TruthMapAccumulator,
        *,
        variable: str,
        wet_day_threshold: float = 1.0,
        temperature_lower_thresholds: Mapping[str, float] | None = None,
        temperature_upper_thresholds: Mapping[str, float] | None = None,
    ) -> FinalizedField:
        valid = truth.count > 0
        if not bool(valid.any()):
            raise ValueError(f"no paired values accumulated for {variable}")
        denominator = np.maximum(truth.count.astype(np.float64), 1.0)
        prediction_climatology = np.full(truth.count.shape, np.nan)
        target_climatology = np.full(truth.count.shape, np.nan)
        rmse = np.full(truth.count.shape, np.nan)
        prediction_climatology[valid] = (
            self.prediction_sum[valid] / denominator[valid]
        )
        target_climatology[valid] = truth.sum[valid] / denominator[valid]
        rmse[valid] = np.sqrt(
            self.squared_error_sum[valid] / denominator[valid]
        )
        bias = prediction_climatology - target_climatology

        temporal, _, _ = _correlation_from_sums(
            truth.count.astype(np.float64),
            self.prediction_sum,
            truth.sum,
            self.prediction_sq_sum,
            truth.square_sum,
            self.product_sum,
        )
        usable_temporal = np.isfinite(temporal)
        temporal_correlation = (
            float(np.mean(temporal[usable_temporal]))
            if bool(usable_temporal.any())
            else float("nan")
        )
        climatology_correlation = float(
            deterministic_metrics(
                prediction_climatology,
                target_climatology,
                mask=valid,
                quantiles=(),
                sample_axis=None,
                spatial_axes=(),
            )["pearson_correlation"]
        )

        metrics = self.scalar.finalize()
        metrics.update(self.gradient.finalize())
        metrics.update(
            {
                "days": self.days,
                "temporal_correlation": temporal_correlation,
                "spatial_pattern_correlation": _safe_ratio(
                    self.spatial_correlation_sum,
                    float(self.spatial_correlation_count),
                ),
                "spatial_pattern_correlation_days": (
                    self.spatial_correlation_count
                ),
                "climatological_spatial_correlation": climatology_correlation,
            }
        )
        sample_metrics = deterministic_metrics(
            self.reservoir.prediction,
            self.reservoir.target,
            quantiles=(0.01, 0.05, 0.5, 0.9, 0.95, 0.99, 0.999),
            sample_axis=None,
            spatial_axes=(),
        )
        for key, value in sample_metrics.items():
            if key.startswith(
                (
                    "prediction_q",
                    "target_q",
                    "quantile_error_q",
                    "distribution_",
                )
            ):
                metrics[key] = value
        if variable == "ppt":
            specialized = precipitation_metrics(
                self.reservoir.prediction,
                self.reservoir.target,
                wet_day_threshold=wet_day_threshold,
            )
        else:
            specialized = temperature_metrics(
                self.reservoir.prediction,
                self.reservoir.target,
                lower_thresholds=temperature_lower_thresholds,
                upper_thresholds=temperature_upper_thresholds,
            )
        return FinalizedField(
            deterministic=metrics,
            specialized=specialized,
            sampling=self.reservoir.metadata(),
            prediction_climatology=prediction_climatology,
            target_climatology=target_climatology,
            bias=bias,
            rmse=rmse,
            reservoir_prediction=self.reservoir.prediction.copy(),
            reservoir_target=self.reservoir.target.copy(),
        )


class EnsembleAccumulator:
    def __init__(self) -> None:
        self.ensemble_size: int | None = None
        self.valid_count = 0
        self.spread_valid_count = 0
        self.minimum_members: int | None = None
        self.maximum_members: int | None = None
        self.weighted: dict[str, float] = {}
        self.weight: dict[str, int] = {}
        self.squared: dict[str, float] = {}

    def update(self, metrics: Mapping[str, float | int]) -> None:
        size = int(metrics["ensemble_size"])
        if self.ensemble_size is None:
            self.ensemble_size = size
        elif size != self.ensemble_size:
            raise ValueError(
                f"ensemble size changed from {self.ensemble_size} to {size}"
            )
        valid_count = int(metrics["valid_count"])
        spread_count = int(metrics["spread_valid_count"])
        self.valid_count += valid_count
        self.spread_valid_count += spread_count
        minimum = int(metrics["minimum_available_members"])
        maximum = int(metrics["maximum_available_members"])
        self.minimum_members = (
            minimum
            if self.minimum_members is None
            else min(self.minimum_members, minimum)
        )
        self.maximum_members = (
            maximum
            if self.maximum_members is None
            else max(self.maximum_members, maximum)
        )
        for key, raw_value in metrics.items():
            if key in {
                "valid_count",
                "spread_valid_count",
                "ensemble_size",
                "minimum_available_members",
                "maximum_available_members",
                "spread_skill_ratio",
                "prediction_interval_mean_absolute_coverage_error",
            }:
                continue
            if key.startswith(
                "prediction_interval_absolute_coverage_error_"
            ):
                # Recompute absolute errors after aggregating signed coverage
                # errors; averaging daily absolute errors would overstate the
                # reliability error when daily deviations cancel.
                continue
            value = float(raw_value)
            if not np.isfinite(value):
                continue
            item_weight = (
                spread_count
                if key in {"mean_ensemble_spread", "rms_ensemble_spread"}
                else valid_count
            )
            if key in {"ensemble_mean_rmse", "rms_ensemble_spread"}:
                self.squared[key] = self.squared.get(key, 0.0) + (
                    value * value * item_weight
                )
                self.weight[key] = self.weight.get(key, 0) + item_weight
            else:
                self.weighted[key] = self.weighted.get(key, 0.0) + (
                    value * item_weight
                )
                self.weight[key] = self.weight.get(key, 0) + item_weight

    def finalize(self) -> dict[str, float | int]:
        if self.ensemble_size is None or self.valid_count <= 0:
            raise ValueError("no ensemble metrics were accumulated")
        result: dict[str, float | int] = {
            "ensemble_size": self.ensemble_size,
            "valid_count": self.valid_count,
            "spread_valid_count": self.spread_valid_count,
            "minimum_available_members": int(self.minimum_members or 0),
            "maximum_available_members": int(self.maximum_members or 0),
        }
        for key, total in self.weighted.items():
            result[key] = total / self.weight[key]
        for key, total in self.squared.items():
            result[key] = math.sqrt(max(total / self.weight[key], 0.0))
        result["spread_skill_ratio"] = _safe_ratio(
            float(result.get("rms_ensemble_spread", float("nan"))),
            float(result.get("ensemble_mean_rmse", float("nan"))),
        )
        coverage_errors = {
            key: float(value)
            for key, value in result.items()
            if key.startswith("prediction_interval_coverage_error_")
        }
        for key, value in coverage_errors.items():
            suffix = key.removeprefix(
                "prediction_interval_coverage_error_"
            )
            result[
                f"prediction_interval_absolute_coverage_error_{suffix}"
            ] = abs(value)
        result["prediction_interval_mean_absolute_coverage_error"] = (
            float(np.mean(np.abs(list(coverage_errors.values()))))
            if coverage_errors
            else float("nan")
        )
        return result


class OrderingAccumulator:
    def __init__(self) -> None:
        self.valid_count = 0
        self.prediction_count = 0
        self.prediction_excess_sum = 0.0
        self.prediction_maximum = 0.0
        self.target_count = 0
        self.target_excess_sum = 0.0
        self.target_maximum = 0.0

    def update(
        self,
        predicted_tasmin: np.ndarray,
        predicted_tasmax: np.ndarray,
        target_tasmin: np.ndarray,
        target_tasmax: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        if not bool(valid.any()):
            return
        prediction_excess = predicted_tasmin[valid] - predicted_tasmax[valid]
        target_excess = target_tasmin[valid] - target_tasmax[valid]
        prediction_violation = prediction_excess > 0.0
        target_violation = target_excess > 0.0
        self.valid_count += int(prediction_excess.size)
        self.prediction_count += int(np.count_nonzero(prediction_violation))
        self.target_count += int(np.count_nonzero(target_violation))
        if bool(prediction_violation.any()):
            values = prediction_excess[prediction_violation]
            self.prediction_excess_sum += float(np.sum(values))
            self.prediction_maximum = max(
                self.prediction_maximum, float(np.max(values))
            )
        if bool(target_violation.any()):
            values = target_excess[target_violation]
            self.target_excess_sum += float(np.sum(values))
            self.target_maximum = max(
                self.target_maximum, float(np.max(values))
            )

    def finalize(self) -> dict[str, float | int]:
        if self.valid_count <= 0:
            raise ValueError("no paired tasmin/tasmax values were accumulated")
        prediction_rate = self.prediction_count / self.valid_count
        target_rate = self.target_count / self.valid_count
        return {
            "valid_count": self.valid_count,
            "tasmin_gt_tasmax_count": self.prediction_count,
            "tasmin_gt_tasmax_violation_rate": prediction_rate,
            "tasmin_gt_tasmax_mean_excess": _safe_ratio(
                self.prediction_excess_sum, float(self.prediction_count)
            )
            if self.prediction_count
            else 0.0,
            "tasmin_gt_tasmax_maximum_excess": self.prediction_maximum,
            "target_tasmin_gt_tasmax_count": self.target_count,
            "target_tasmin_gt_tasmax_violation_rate": target_rate,
            "target_tasmin_gt_tasmax_mean_excess": _safe_ratio(
                self.target_excess_sum, float(self.target_count)
            )
            if self.target_count
            else 0.0,
            "target_tasmin_gt_tasmax_maximum_excess": self.target_maximum,
            "tasmin_gt_tasmax_violation_rate_bias": prediction_rate
            - target_rate,
        }


def _stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256(
        "\x1f".join((str(base_seed), *map(str, parts))).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _load_static_array(
    path: str | Path,
    grid: CanonicalPrismGrid,
    *,
    variable: str | None,
    context: str,
) -> tuple[np.ndarray, str]:
    source = _resolve_path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{context} file not found: {source}")
    suffix = source.suffix.casefold()
    if suffix == ".npy":
        values = np.load(source, allow_pickle=False)
        alignment = "shape-only index alignment (.npy has no coordinates)"
    elif suffix == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            if variable is not None and variable in payload.files:
                values = payload[variable]
            elif len(payload.files) == 1:
                values = payload[payload.files[0]]
            else:
                raise ValueError(
                    f"{source}: specify a variable; arrays={payload.files}"
                )
        alignment = "shape-only index alignment (.npz has no coordinates)"
    else:
        import xarray as xr

        with xr.open_dataset(source, decode_times=False) as dataset:
            lat_name, lon_name = _validate_grid(
                dataset, grid, context=str(source)
            )
            candidates = [
                str(name)
                for name, data in dataset.data_vars.items()
                if lat_name in data.dims and lon_name in data.dims
            ]
            name = variable
            if name is None:
                if len(candidates) != 1:
                    raise ValueError(
                        f"{source}: static variable is ambiguous; "
                        f"candidates={candidates}"
                    )
                name = candidates[0]
            values = _extract_field(
                dataset, name, lat_name, lon_name, context=str(source)
            )
        alignment = "exact canonical coordinates and fingerprint"
    values = np.asarray(values)
    values = np.squeeze(values)
    if values.shape != grid.shape:
        raise ValueError(
            f"{source}: static shape {values.shape} != {grid.shape}"
        )
    return values, alignment


def _elevation_masks(
    elevation: np.ndarray,
    edges: Sequence[float],
) -> dict[str, np.ndarray]:
    boundaries = tuple(float(value) for value in edges)
    if not boundaries or any(not np.isfinite(value) for value in boundaries):
        raise ValueError("elevation bins must contain finite boundaries")
    if any(
        right <= left
        for left, right in zip(boundaries, boundaries[1:], strict=False)
    ):
        raise ValueError(
            "elevation bin boundaries must be strictly increasing"
        )
    lower = (-math.inf, *boundaries)
    upper = (*boundaries, math.inf)
    masks: dict[str, np.ndarray] = {}
    finite = np.isfinite(elevation)
    for low, high in zip(lower, upper, strict=True):
        if not np.isfinite(low):
            label = f"below_{high:g}"
        elif not np.isfinite(high):
            label = f"{low:g}_and_above"
        else:
            label = f"{low:g}_to_{high:g}"
        masks[label] = finite & (elevation >= low) & (elevation < high)
    return masks


def _parse_named_paths(
    values: Sequence[str], *, option: str
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"{option} must use NAME=PATH, got {raw!r}")
        name, path = raw.split("=", 1)
        name = name.strip()
        if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"invalid {option} name {name!r}")
        if name in result:
            raise ValueError(f"duplicate {option} name {name!r}")
        result[name] = _resolve_path(path.strip())
    return result


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(item) for item in value]
    if isinstance(value, np.ndarray):
        return _finite_or_none(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({str(key) for row in rows for key in row})
    leading = [
        name
        for name in ("method", "variable", "scope", "label")
        if name in fieldnames
    ]
    fieldnames = leading + [name for name in fieldnames if name not in leading]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_finite_or_none(row))


def _flatten_metrics(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def _sampled_target_magnitude_regimes(
    target: np.ndarray,
    *,
    variable: str,
    wet_day_threshold: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Define target-relative magnitude regimes on one paired reservoir.

    Quantile thresholds cannot be known during the first streaming pass.  We
    therefore derive and evaluate these regimes on the same deterministic
    paired reservoir used for distribution metrics.  Wet/dry precipitation
    strata are accumulated exactly elsewhere because their physical threshold
    is known before streaming begins.
    """

    values = np.asarray(target, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size == 0
        or not bool(np.isfinite(values).all())
    ):
        raise ValueError(
            f"{variable}: target magnitude regimes require a nonempty finite "
            "one-dimensional paired reservoir"
        )
    if variable == "ppt":
        empirical_q90 = float(np.quantile(values, 0.9))
        threshold = max(float(wet_day_threshold), empirical_q90)
        return (
            {
                "moderate_wet": (values >= wet_day_threshold)
                & (values < threshold),
                "extreme_q90": values >= threshold,
            },
            {
                "empirical_target_q90": empirical_q90,
                "effective_extreme_threshold": threshold,
                "wet_day_threshold": float(wet_day_threshold),
                "units": UNITS[variable],
            },
        )

    lower = float(np.quantile(values, 0.05))
    upper = float(np.quantile(values, 0.95))
    return (
        {
            "cold_extreme_q05": values <= lower,
            "moderate_q05_q95": (values > lower) & (values < upper),
            "warm_extreme_q95": values >= upper,
        },
        {
            "empirical_target_q05": lower,
            "empirical_target_q95": upper,
            "units": UNITS[variable],
        },
    )


def _plot_maps(  # noqa: C901
    path: Path,
    *,
    variable: str,
    lat: np.ndarray,
    lon: np.ndarray,
    methods: Sequence[str],
    fields: Mapping[str, FinalizedField],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = fields["phase1"]
    predictions = [
        baseline.target_climatology,
        *[fields[name].prediction_climatology for name in methods],
    ]
    finite_mean = np.concatenate(
        [values[np.isfinite(values)] for values in predictions]
    )
    mean_min, mean_max = np.quantile(finite_mean, (0.01, 0.99))
    bias_values = np.concatenate(
        [
            np.abs(fields[name].bias[np.isfinite(fields[name].bias)])
            for name in methods
        ]
    )
    bias_limit = max(float(np.quantile(bias_values, 0.99)), 1.0e-8)
    rmse_values = np.concatenate(
        [fields[name].rmse[np.isfinite(fields[name].rmse)] for name in methods]
    )
    rmse_max = max(float(np.quantile(rmse_values, 0.99)), 1.0e-8)
    improvements = []
    for name in methods[1:]:
        improvements.extend(
            [
                baseline.rmse - fields[name].rmse,
                np.abs(baseline.bias) - np.abs(fields[name].bias),
            ]
        )
    if improvements:
        finite_improvement = np.concatenate(
            [np.abs(value[np.isfinite(value)]) for value in improvements]
        )
        improvement_limit = max(
            float(np.quantile(finite_improvement, 0.99)), 1.0e-8
        )
    else:
        improvement_limit = 1.0

    columns = ["PRISM", *methods]
    fig, axes = plt.subplots(
        5,
        len(columns),
        figsize=(4.2 * len(columns), 16),
        constrained_layout=True,
        squeeze=False,
    )
    origin = "lower" if lat[-1] > lat[0] else "upper"
    extent = [float(lon[0]), float(lon[-1]), float(lat[0]), float(lat[-1])]
    unit = UNITS[variable]

    def draw(row: int, column: int, values: np.ndarray, **kwargs: Any) -> None:
        image = axes[row, column].imshow(
            values,
            origin=origin,
            extent=extent,
            interpolation="nearest",
            aspect="auto",
            **kwargs,
        )
        fig.colorbar(image, ax=axes[row, column], shrink=0.72)

    draw(
        0,
        0,
        baseline.target_climatology,
        cmap="viridis" if variable == "ppt" else "coolwarm",
        vmin=mean_min,
        vmax=mean_max,
    )
    for column, name in enumerate(methods, start=1):
        draw(
            0,
            column,
            fields[name].prediction_climatology,
            cmap="viridis" if variable == "ppt" else "coolwarm",
            vmin=mean_min,
            vmax=mean_max,
        )
        draw(
            1,
            column,
            fields[name].bias,
            cmap="RdBu_r",
            vmin=-bias_limit,
            vmax=bias_limit,
        )
        draw(
            2,
            column,
            fields[name].rmse,
            cmap="magma",
            vmin=0.0,
            vmax=rmse_max,
        )
        if name != "phase1":
            draw(
                3,
                column,
                baseline.rmse - fields[name].rmse,
                cmap="RdBu",
                vmin=-improvement_limit,
                vmax=improvement_limit,
            )
            draw(
                4,
                column,
                np.abs(baseline.bias) - np.abs(fields[name].bias),
                cmap="RdBu",
                vmin=-improvement_limit,
                vmax=improvement_limit,
            )
    for row in range(1, 5):
        axes[row, 0].axis("off")
    axes[3, 1].axis("off")
    axes[4, 1].axis("off")
    row_titles = (
        f"Mean ({unit})",
        f"Prediction - PRISM ({unit})",
        f"Temporal RMSE ({unit})",
        f"RMSE improvement ({unit})",
        f"Absolute-bias improvement ({unit})",
    )
    for column, title in enumerate(columns):
        axes[0, column].set_title(title)
    for row, title in enumerate(row_titles):
        axes[row, 0].set_ylabel(title)
    for row in range(5):
        for column in range(1, len(columns)):
            axes[row, column].set_xlabel("Longitude")
            axes[row, column].set_ylabel("Latitude")
    fig.suptitle(f"{variable}: Phase-1 and residual-refinement comparison")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_distributions(
    path: Path,
    *,
    variable: str,
    methods: Sequence[str],
    fields: Mapping[str, FinalizedField],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    truth = fields["phase1"].reservoir_target
    finite = [truth, *[fields[name].reservoir_prediction for name in methods]]
    lower, upper = np.quantile(np.concatenate(finite), (0.002, 0.998))
    bins = np.linspace(lower, upper, 80)
    axes[0].hist(
        truth, bins=bins, density=True, histtype="step", lw=2, label="PRISM"
    )
    for name in methods:
        axes[0].hist(
            fields[name].reservoir_prediction,
            bins=bins,
            density=True,
            histtype="step",
            lw=1.5,
            label=name,
        )
    axes[0].set_title("Physical-value distribution")
    axes[0].set_xlabel(UNITS[variable])
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)

    probabilities = np.array(
        [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999]
    )
    axes[1].plot(
        probabilities,
        np.quantile(truth, probabilities),
        marker="o",
        lw=2,
        label="PRISM",
    )
    for name in methods:
        axes[1].plot(
            probabilities,
            np.quantile(fields[name].reservoir_prediction, probabilities),
            marker=".",
            label=name,
        )
    axes[1].set_xscale("logit")
    axes[1].set_title("Central and extreme quantiles")
    axes[1].set_xlabel("Quantile probability")
    axes[1].set_ylabel(UNITS[variable])
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"{variable} distribution and extremes")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_evaluation(  # noqa: C901
    *,
    config_path: str | Path,
    phase1_dir: str | Path,
    method_dirs: Mapping[str, str | Path],
    output_dir: str | Path,
    canonical_grid_dir: str | Path | None = None,
    truth_dir: str | Path | None = None,
    split: str = "inference",
    start: str | None = None,
    end: str | None = None,
    reservoir_size: int = 200_000,
    seed: int = 0,
    wet_day_threshold: float = 1.0,
    elevation_file: str | Path | None = None,
    elevation_variable: str | None = None,
    elevation_bins: Sequence[float] | None = None,
    region_masks: Mapping[str, str | Path] | None = None,
    temperature_lower_thresholds: Mapping[str, float] | None = None,
    temperature_upper_thresholds: Mapping[str, float] | None = None,
    make_plots: bool = True,
    progress_every: int = 100,
) -> dict[str, Any]:
    """Run daily-file evaluation and return its JSON-ready report."""

    resolved_config, config = _load_config(config_path)
    case_name = str(config["case_name"])
    configured_truth_units = {
        variable: str(config["evaluation"]["truth_units"][variable])
        for variable in VARIABLES
    }
    (
        configured_split_start,
        configured_split_end,
        first_date,
        last_date,
    ) = _evaluation_date_contract(
        config,
        split=split,
        start=start,
        end=end,
    )
    split = str(split).strip().lower()
    dates = _daily_range(first_date, last_date)
    wet_day_threshold = float(wet_day_threshold)
    if not np.isfinite(wet_day_threshold) or wet_day_threshold < 0.0:
        raise ValueError("wet_day_threshold must be finite and nonnegative")
    if not method_dirs:
        raise ValueError(
            "at least one named refinement method directory is required"
        )
    if "phase1" in method_dirs or "PRISM" in method_dirs:
        raise ValueError("method names 'phase1' and 'PRISM' are reserved")

    grid_directory = (
        _resolve_path(canonical_grid_dir)
        if canonical_grid_dir is not None
        else case_preprocess_dir(config)
    )
    grid = load_canonical_grid(grid_directory, required=True)
    assert grid is not None
    target_root = _resolve_path(
        truth_dir
        if truth_dir is not None
        else str(config["data"]["target_dir"])
    )
    baseline_directory = _case_directory(phase1_dir, case_name)
    baseline_files = _dated_files(baseline_directory)
    _require_dates(baseline_files, dates, context="Phase-1 baseline")
    method_directories = {
        str(name): _case_directory(path, case_name)
        for name, path in method_dirs.items()
    }
    method_files = {
        name: _dated_files(directory)
        for name, directory in method_directories.items()
    }
    for name, files in method_files.items():
        _require_dates(files, dates, context=f"refinement method {name!r}")
    truth_files = {
        variable: _truth_files(target_root, variable, dates)
        for variable in VARIABLES
    }
    for variable, files in truth_files.items():
        _require_dates(files, dates, context=f"PRISM truth {variable}")

    static_masks: dict[str, dict[str, np.ndarray]] = {}
    static_metadata: dict[str, Any] = {}
    if elevation_file is not None:
        if not elevation_bins:
            raise ValueError("elevation_file requires explicit elevation_bins")
        elevation, alignment = _load_static_array(
            elevation_file,
            grid,
            variable=elevation_variable,
            context="elevation",
        )
        static_masks["elevation"] = _elevation_masks(elevation, elevation_bins)
        static_metadata["elevation"] = {
            "path": str(_resolve_path(elevation_file)),
            "variable": elevation_variable,
            "bin_boundaries": list(map(float, elevation_bins)),
            "alignment": alignment,
        }
    if region_masks:
        static_masks["region"] = {}
        static_metadata["regions"] = {}
        for name, source in region_masks.items():
            values, alignment = _load_static_array(
                source,
                grid,
                variable=None,
                context=f"region {name}",
            )
            static_masks["region"][str(name)] = np.isfinite(values) & (
                values != 0
            )
            static_metadata["regions"][str(name)] = {
                "path": str(_resolve_path(source)),
                "alignment": alignment,
            }

    methods = ["phase1", *method_directories]
    truth_accumulators = {
        variable: TruthMapAccumulator(grid.shape) for variable in VARIABLES
    }
    truth_unit_audit: dict[str, dict[str, Any]] = {
        variable: {
            "configured_unit": configured_truth_units[variable],
            "source_attribute_files": 0,
            "configured_assumption_files": 0,
            "observed_source_units": set(),
        }
        for variable in VARIABLES
    }
    field_accumulators = {
        (method, variable): MethodFieldAccumulator(
            grid.shape,
            reservoir_size=reservoir_size,
            # Every method sees the same finite target/prediction mask below.
            # Reuse one priority stream per variable so bounded distribution
            # reservoirs retain the same cells for genuinely paired method
            # comparisons.  Including ``method`` here would confound tail and
            # distribution metrics once the population exceeds the cap.
            seed=_stable_seed(seed, variable),
        )
        for method in methods
        for variable in VARIABLES
    }
    ensemble_accumulators: dict[tuple[str, str], EnsembleAccumulator] = {}
    stratum_scalars: dict[tuple[str, str, str, str], ScalarAccumulator] = {}
    stratum_ensembles: dict[
        tuple[str, str, str, str], EnsembleAccumulator
    ] = {}
    ordering = {method: OrderingAccumulator() for method in methods}
    baseline_provenance = ProvenanceTracker("Phase-1 baseline")
    method_provenance = {
        name: ProvenanceTracker(f"refinement method {name!r}")
        for name in method_directories
    }
    target_valid_mask_tracker = TargetValidMaskTracker()

    for day_index, sample_date in enumerate(dates, start=1):
        baseline = _read_product_day(
            baseline_files[sample_date],
            grid,
            expected_date=sample_date,
            case_name=case_name,
            role="phase1",
            expected_split=split,
            configured_split_start=configured_split_start,
            configured_split_end=configured_split_end,
        )
        baseline_provenance.update(
            baseline.provenance, path=baseline_files[sample_date]
        )
        target_valid_mask_tracker.update(
            baseline, path=baseline_files[sample_date]
        )
        daily_methods = {
            "phase1": baseline,
            **{
                name: _read_product_day(
                    method_files[name][sample_date],
                    grid,
                    expected_date=sample_date,
                    case_name=case_name,
                    role="refinement",
                    expected_split=split,
                    configured_split_start=configured_split_start,
                    configured_split_end=configured_split_end,
                )
                for name in method_directories
            },
        }
        for name in method_directories:
            target_valid_mask_tracker.update(
                daily_methods[name], path=method_files[name][sample_date]
            )
            method_provenance[name].update(
                daily_methods[name].provenance,
                path=method_files[name][sample_date],
            )
            for variable in VARIABLES:
                if not np.allclose(
                    daily_methods[name].phase1_fields[variable],
                    baseline.fields[variable],
                    rtol=2.0e-5,
                    atol=2.0e-5,
                    equal_nan=True,
                ):
                    raise ValueError(
                        f"{method_files[name][sample_date]}: embedded "
                        f"{variable} Phase-1 baseline does not match the "
                        "independently evaluated Phase-1 daily product"
                    )
        truth = {
            variable: _read_truth_day(
                truth_files[variable][sample_date],
                variable,
                grid,
                configured_unit=configured_truth_units[variable],
                expected_date=sample_date,
                unit_audit=truth_unit_audit[variable],
            )
            for variable in VARIABLES
        }
        common_masks: dict[str, np.ndarray] = {}
        for variable in VARIABLES:
            common = np.isfinite(truth[variable])
            for method in methods:
                common &= np.isfinite(daily_methods[method].fields[variable])
            if not bool(common.any()):
                raise ValueError(
                    f"{sample_date} {variable}: no finite cells shared by "
                    "all methods"
                )
            common_masks[variable] = common
            truth_accumulators[variable].update(truth[variable], common)
            month_label = f"{sample_date.month:02d}"
            season_label = SEASONS[sample_date.month]
            for method in methods:
                prediction = daily_methods[method].fields[variable]
                field_accumulators[(method, variable)].update(
                    prediction, truth[variable], common
                )
                for scope, label, stratum_mask in (
                    ("month", month_label, common),
                    ("season", season_label, common),
                ):
                    key = (method, variable, scope, label)
                    stratum_scalars.setdefault(
                        key, ScalarAccumulator()
                    ).update(prediction, truth[variable], stratum_mask)
                for scope, label_masks in static_masks.items():
                    for label, spatial_mask in label_masks.items():
                        key = (method, variable, scope, label)
                        stratum_scalars.setdefault(
                            key, ScalarAccumulator()
                        ).update(
                            prediction,
                            truth[variable],
                            common & spatial_mask,
                        )
                occurrence_masks: dict[str, np.ndarray] = {}
                if variable == "ppt":
                    occurrence_masks = {
                        "target_dry": common
                        & (truth[variable] < wet_day_threshold),
                        "target_wet": common
                        & (truth[variable] >= wet_day_threshold),
                    }
                    for label, occurrence_mask in occurrence_masks.items():
                        if not bool(occurrence_mask.any()):
                            continue
                        key = (
                            method,
                            variable,
                            "target_precipitation_occurrence",
                            label,
                        )
                        stratum_scalars.setdefault(
                            key, ScalarAccumulator()
                        ).update(
                            prediction,
                            truth[variable],
                            occurrence_mask,
                        )

                members = daily_methods[method].members.get(variable)
                if members is None:
                    continue
                daily_ensemble = ensemble_metrics(
                    members,
                    truth[variable],
                    mask=common,
                    coverage_levels=(0.5, 0.8, 0.9, 0.95),
                )
                ensemble_accumulators.setdefault(
                    (method, variable), EnsembleAccumulator()
                ).update(daily_ensemble)
                for scope, label, stratum_mask in (
                    ("month", month_label, common),
                    ("season", season_label, common),
                ):
                    key = (method, variable, scope, label)
                    stratum_ensembles.setdefault(
                        key, EnsembleAccumulator()
                    ).update(
                        ensemble_metrics(
                            members,
                            truth[variable],
                            mask=stratum_mask,
                            coverage_levels=(0.5, 0.8, 0.9, 0.95),
                        )
                    )
                for scope, label_masks in static_masks.items():
                    for label, spatial_mask in label_masks.items():
                        mask = common & spatial_mask
                        if not bool(mask.any()):
                            continue
                        key = (method, variable, scope, label)
                        stratum_ensembles.setdefault(
                            key, EnsembleAccumulator()
                        ).update(
                            ensemble_metrics(
                                members,
                                truth[variable],
                                mask=mask,
                                coverage_levels=(0.5, 0.8, 0.9, 0.95),
                            )
                        )
                for label, occurrence_mask in occurrence_masks.items():
                    if not bool(occurrence_mask.any()):
                        continue
                    key = (
                        method,
                        variable,
                        "target_precipitation_occurrence",
                        label,
                    )
                    stratum_ensembles.setdefault(
                        key, EnsembleAccumulator()
                    ).update(
                        ensemble_metrics(
                            members,
                            truth[variable],
                            mask=occurrence_mask,
                            coverage_levels=(0.5, 0.8, 0.9, 0.95),
                        )
                    )

        ordering_mask = common_masks["tmin"] & common_masks["tmax"]
        for method in methods:
            ordering[method].update(
                daily_methods[method].fields["tmin"],
                daily_methods[method].fields["tmax"],
                truth["tmin"],
                truth["tmax"],
                ordering_mask,
            )
        if progress_every > 0 and (
            day_index % progress_every == 0 or day_index == len(dates)
        ):
            print(
                f"[refinement-evaluation] {day_index}/{len(dates)} "
                f"({sample_date})",
                flush=True,
            )

    baseline_signature = baseline_provenance.finalize()
    method_signatures = {
        name: tracker.finalize() for name, tracker in method_provenance.items()
    }
    target_valid_mask_signature = target_valid_mask_tracker.finalize()
    baseline_checkpoint = str(baseline_signature["phase1_checkpoint"])
    baseline_resolved = str(Path(baseline_checkpoint).expanduser().resolve())
    for name, signature in method_signatures.items():
        method_phase1 = str(signature["phase1_checkpoint"])
        method_resolved = str(Path(method_phase1).expanduser().resolve())
        if method_resolved != baseline_resolved:
            raise ValueError(
                f"method {name!r} Phase-1 checkpoint {method_phase1!r} "
                "does not "
                f"match evaluated baseline checkpoint {baseline_checkpoint!r}"
            )

    finalized: dict[str, dict[str, FinalizedField]] = {
        method: {} for method in methods
    }
    for method in methods:
        for variable in VARIABLES:
            finalized[method][variable] = field_accumulators[
                (method, variable)
            ].finalize(
                truth_accumulators[variable],
                variable=variable,
                wet_day_threshold=wet_day_threshold,
                temperature_lower_thresholds=temperature_lower_thresholds,
                temperature_upper_thresholds=temperature_upper_thresholds,
            )

    strata_json: dict[str, dict[str, dict[str, Any]]] = {}
    csv_rows: list[dict[str, Any]] = []
    variables_json: dict[str, Any] = {}
    for method in methods:
        variables_json[method] = {}
        for variable in VARIABLES:
            field = finalized[method][variable]
            ensemble = (
                ensemble_accumulators[(method, variable)].finalize()
                if (method, variable) in ensemble_accumulators
                else None
            )
            variables_json[method][variable] = {
                "deterministic": field.deterministic,
                "specialized": field.specialized,
                "ensemble": ensemble,
                "distribution_sampling": field.sampling,
            }
            row: dict[str, Any] = {
                "method": method,
                "variable": variable,
                "scope": "overall",
                "label": "all",
                **field.deterministic,
                **_flatten_metrics("specialized_", field.specialized),
            }
            if ensemble is not None:
                row.update(_flatten_metrics("ensemble_", ensemble))
            csv_rows.append(row)
        variables_json[method]["temperature_ordering"] = ordering[
            method
        ].finalize()
        csv_rows.append(
            {
                "method": method,
                "variable": "tmin,tmax",
                "scope": "overall",
                "label": "temperature_ordering",
                **ordering[method].finalize(),
            }
        )

    for key in sorted(stratum_scalars):
        method, variable, scope, label = key
        metrics = stratum_scalars[key].finalize()
        ensemble = (
            stratum_ensembles[key].finalize()
            if key in stratum_ensembles
            else None
        )
        strata_json.setdefault(scope, {}).setdefault(label, {}).setdefault(
            method, {}
        )[variable] = {
            "deterministic": metrics,
            "ensemble": ensemble,
        }
        row = {
            "method": method,
            "variable": variable,
            "scope": scope,
            "label": label,
            **metrics,
        }
        if ensemble is not None:
            row.update(_flatten_metrics("ensemble_", ensemble))
        csv_rows.append(row)

    target_regime_definitions: dict[str, Any] = {
        "target_precipitation_occurrence": {
            "variable": "ppt",
            "dry": f"target < {wet_day_threshold:g} mm/day",
            "wet": f"target >= {wet_day_threshold:g} mm/day",
            "sampling": "exact over all shared finite daily cells",
        },
        "target_magnitude": {},
    }
    for variable in VARIABLES:
        reference = finalized["phase1"][variable]
        regime_masks, definition = _sampled_target_magnitude_regimes(
            reference.reservoir_target,
            variable=variable,
            wet_day_threshold=wet_day_threshold,
        )
        target_regime_definitions["target_magnitude"][variable] = {
            **definition,
            "threshold_source": (
                "paired deterministic distribution reservoir; exact only "
                "when distribution_sampling.exact is true"
            ),
            "distribution_sampling": reference.sampling,
        }
        for method in methods:
            field = finalized[method][variable]
            if not np.array_equal(
                field.reservoir_target,
                reference.reservoir_target,
                equal_nan=True,
            ):
                raise RuntimeError(
                    f"{variable}: target distribution reservoir is not "
                    f"paired between phase1 and {method}"
                )
            for label, regime_mask in regime_masks.items():
                if not bool(regime_mask.any()):
                    continue
                accumulator = ScalarAccumulator()
                accumulator.update(
                    field.reservoir_prediction,
                    field.reservoir_target,
                    regime_mask,
                )
                metrics = accumulator.finalize()
                strata_json.setdefault("target_magnitude", {}).setdefault(
                    label, {}
                ).setdefault(method, {})[variable] = {
                    "deterministic": metrics,
                    "ensemble": None,
                    "distribution_sampling": field.sampling,
                }
                csv_rows.append(
                    {
                        "method": method,
                        "variable": variable,
                        "scope": "target_magnitude",
                        "label": label,
                        **metrics,
                        **_flatten_metrics("sampling_", field.sampling),
                    }
                )

    destination = _resolve_path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{case_name}_refinement_{first_date:%Y%m%d}_{last_date:%Y%m%d}"
    json_path = destination / f"{stem}_metrics.json"
    csv_path = destination / f"{stem}_metrics.csv"
    plot_paths: list[str] = []
    plot_status = "disabled"
    if make_plots:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            plot_status = "matplotlib unavailable"
        else:
            plot_status = "generated"
            for variable in VARIABLES:
                variable_fields = {
                    method: finalized[method][variable] for method in methods
                }
                map_path = destination / f"{stem}_{variable}_maps.png"
                distribution_path = (
                    destination
                    / f"{stem}_{variable}_distribution_extremes.png"
                )
                _plot_maps(
                    map_path,
                    variable=variable,
                    lat=grid.lat,
                    lon=grid.lon,
                    methods=methods,
                    fields=variable_fields,
                )
                _plot_distributions(
                    distribution_path,
                    variable=variable,
                    methods=methods,
                    fields=variable_fields,
                )
                plot_paths.extend((str(map_path), str(distribution_path)))

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "case_name": case_name,
        "config": str(resolved_config),
        "config_sha256": hashlib.sha256(
            resolved_config.read_bytes()
        ).hexdigest(),
        "variable_order": list(VARIABLES),
        "dataset_split": split,
        "date_range": {
            "split": split,
            "configured_start": configured_split_start.isoformat(),
            "configured_end": configured_split_end.isoformat(),
            "start": first_date.isoformat(),
            "end": last_date.isoformat(),
            "inclusive_day_count": len(dates),
            "date_hash": hashlib.sha256(
                "\n".join(map(str, dates)).encode("utf-8")
            ).hexdigest(),
        },
        "canonical_grid": {
            "directory": str(grid_directory),
            **grid.manifest_entry(),
        },
        "alignment": (
            "exact canonical float64 coordinates, exact inclusive daily "
            "dates, "
            "shared finite target/method mask; no interpolation"
        ),
        "physical_units": dict(UNITS),
        "directories": {
            "truth": str(target_root),
            "phase1": str(baseline_directory),
            "methods": {
                name: str(directory)
                for name, directory in method_directories.items()
            },
        },
        "provenance": {
            "phase1": baseline_signature,
            "methods": method_signatures,
            "target_valid_mask": target_valid_mask_signature,
            "truth_units": {
                "policy": (
                    "validate every present source units attribute; when "
                    "absent, use only the exact evaluation.truth_units YAML "
                    "contract"
                ),
                "configured": dict(configured_truth_units),
                "per_variable": {
                    variable: {
                        **{
                            key: value
                            for key, value in audit.items()
                            if key != "observed_source_units"
                        },
                        "observed_source_units": sorted(
                            audit["observed_source_units"]
                        ),
                        "used_configured_assumption": bool(
                            audit["configured_assumption_files"]
                        ),
                    }
                    for variable, audit in truth_unit_audit.items()
                },
            },
        },
        "static_strata": static_metadata,
        "temperature_thresholds": {
            "lower": dict(temperature_lower_thresholds or {}),
            "upper": dict(temperature_upper_thresholds or {}),
        },
        "wet_day_threshold_mm_day": wet_day_threshold,
        "target_regime_definitions": target_regime_definitions,
        "metrics": variables_json,
        "strata": strata_json,
        "plots": {"status": plot_status, "paths": plot_paths},
        "metric_notes": {
            "error_sign": "prediction - PRISM",
            "improvement_sign": (
                "positive RMSE/bias improvement means lower error than Phase-1"
            ),
            "scalar_moments": "exact over every shared finite pair",
            "maps": "exact paired-day climatology, bias, and temporal RMSE",
            "distribution_metrics": (
                "exact when population <= reservoir_size, otherwise "
                "deterministic uniform priority sample; see each "
                "distribution_sampling record"
            ),
            "spatial_smoothing": (
                "spatial_roughness_ratio is prediction/PRISM RMS of valid "
                "adjacent-grid differences; spatial_smoothing_index is "
                "1 - that ratio (positive indicates less small-scale "
                "variation). This mask-aware proxy avoids artificial FFT "
                "power from zero-filled coastlines and is not an isotropic "
                "physical-wavenumber spectrum."
            ),
            "target_regimes": (
                "precipitation wet/dry errors are exact; target-magnitude "
                "regimes use paired reservoir quantiles and inherit the "
                "reported distribution-sampling exact/sampled status"
            ),
            "ensemble_crps": (
                "exact empirical ensemble CRPS per retained daily cell"
            ),
            "ensemble_reliability": (
                "central prediction-interval coverage error is observed "
                "minus nominal coverage; mean absolute coverage error "
                "summarizes the requested levels (zero is ideal)"
            ),
        },
    }
    _write_csv(csv_path, csv_rows)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            _finite_or_none(report),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    report["outputs"] = {"json": str(json_path), "csv": str(csv_path)}
    print(f"[refinement-evaluation] JSON: {json_path}")
    print(f"[refinement-evaluation] CSV: {csv_path}")
    for path in plot_paths:
        print(f"[refinement-evaluation] plot: {path}")
    return report


def _parse_float_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(
            "expected comma-separated numeric values"
        )
    return [float(part) for part in parts]


def _parse_named_floats(
    values: Sequence[str], *, option: str
) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                f"{option} values must use NAME=VALUE, got {value!r}"
            )
        name, raw_number = value.split("=", 1)
        name = name.strip()
        if not name or name in parsed:
            raise argparse.ArgumentTypeError(
                f"{option} threshold names must be nonempty and unique"
            )
        number = float(raw_number)
        if not np.isfinite(number):
            raise argparse.ArgumentTypeError(
                f"{option} threshold {name!r} must be finite"
            )
        parsed[name] = number
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream exact-grid PRISM, Phase-1, and named Phase-2 daily "
            "products"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase1-dir", required=True)
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help=(
            "named refinement daily-file directory; repeat for multiple "
            "methods"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--canonical-grid-dir", default=None)
    parser.add_argument("--truth-dir", default=None)
    parser.add_argument(
        "--split",
        choices=EVALUATION_SPLITS,
        default="inference",
        help="configured YAML split to evaluate",
    )
    parser.add_argument(
        "--start",
        default=None,
        help=(
            "intentional subset start inside the selected split; requires "
            "--end"
        ),
    )
    parser.add_argument(
        "--end",
        default=None,
        help=(
            "intentional subset end inside the selected split; requires "
            "--start"
        ),
    )
    parser.add_argument("--reservoir-size", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--wet-day-threshold",
        type=float,
        default=1.0,
        metavar="MM_PER_DAY",
        help="target precipitation threshold used for wet/dry strata",
    )
    parser.add_argument("--elevation-file", default=None)
    parser.add_argument("--elevation-variable", default=None)
    parser.add_argument(
        "--elevation-bins",
        default=None,
        help="comma-separated physical elevation bin boundaries",
    )
    parser.add_argument(
        "--region-mask",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="named boolean/numeric grid mask; repeat for multiple regions",
    )
    parser.add_argument(
        "--temperature-lower-threshold",
        action="append",
        default=[],
        metavar="NAME=DEGC",
        help=(
            "named cold threshold in degrees C; repeat for multiple "
            "threshold-exceedance diagnostics"
        ),
    )
    parser.add_argument(
        "--temperature-upper-threshold",
        action="append",
        default=[],
        metavar="NAME=DEGC",
        help=(
            "named warm threshold in degrees C; repeat for multiple "
            "threshold-exceedance diagnostics"
        ),
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    methods = _parse_named_paths(args.method, option="--method")
    regions = _parse_named_paths(args.region_mask, option="--region-mask")
    lower_thresholds = _parse_named_floats(
        args.temperature_lower_threshold,
        option="--temperature-lower-threshold",
    )
    upper_thresholds = _parse_named_floats(
        args.temperature_upper_threshold,
        option="--temperature-upper-threshold",
    )
    run_evaluation(
        config_path=args.config,
        phase1_dir=args.phase1_dir,
        method_dirs=methods,
        output_dir=args.output_dir,
        canonical_grid_dir=args.canonical_grid_dir,
        truth_dir=args.truth_dir,
        split=args.split,
        start=args.start,
        end=args.end,
        reservoir_size=args.reservoir_size,
        seed=args.seed,
        wet_day_threshold=args.wet_day_threshold,
        elevation_file=args.elevation_file,
        elevation_variable=args.elevation_variable,
        elevation_bins=_parse_float_list(args.elevation_bins),
        region_masks=regions,
        temperature_lower_thresholds=lower_thresholds,
        temperature_upper_thresholds=upper_thresholds,
        make_plots=not args.no_plots,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
