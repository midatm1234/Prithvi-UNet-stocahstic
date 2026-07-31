"""Strict readers for case-scoped daily ``*_PRISM`` preprocessing products."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from granitewxc.utils.prism_grid import CanonicalPrismGrid, assert_grid_matches


LAT_CANDIDATES = ("lat", "latitude", "y")
LON_CANDIDATES = ("lon", "longitude", "x")
PREPROCESSING_SCHEMA_VERSION = 1
PREPROCESSING_CONTRACT_ATTR = "preprocessing_contract"
PREPROCESSING_SIGNATURE_ATTR = "preprocessing_signature"
PREDICTOR_PREPROCESSING_SIGNATURE_ATTR = "predictor_preprocessing_signature"
SOURCE_ARTIFACT_SIGNATURE_ATTR = "source_artifact_signature"


def preprocessing_config_fields(
    *,
    data_type: str,
    predictor_variables: Sequence[Tuple[str, float]],
    target_variables: Sequence[str],
    include_targets: bool,
    regrid_method: str,
    canonical_grid_fingerprint: str,
    static_elevation_required: bool,
    static_elevation_variable: str | None,
) -> Dict[str, Any]:
    """Return config-derived fields that every daily product must match."""
    return {
        "schema_version": PREPROCESSING_SCHEMA_VERSION,
        "data_type": str(data_type).strip().lower(),
        "ordered_predictors": [
            [str(variable), float(level)]
            for variable, level in predictor_variables
        ],
        "target_variables": [str(value) for value in target_variables],
        "include_targets": bool(include_targets),
        "regrid_method": str(regrid_method).strip().lower(),
        "canonical_grid_fingerprint": str(canonical_grid_fingerprint),
        "static_elevation_required": bool(static_elevation_required),
        "static_elevation_variable": (
            None
            if not static_elevation_required
            else str(static_elevation_variable or "__auto_numeric__")
        ),
    }


def build_preprocessing_contract(
    *,
    data_type: str,
    predictor_variables: Sequence[Tuple[str, float]],
    target_variables: Sequence[str],
    include_targets: bool,
    regrid_method: str,
    canonical_grid_fingerprint: str,
    static_elevation_sha256: str | None,
    static_elevation_variable: str | None,
    source_grid_fingerprint: str,
    algorithm: str,
) -> Dict[str, Any]:
    """Build the complete, content-addressed daily-product provenance."""
    contract = preprocessing_config_fields(
        data_type=data_type,
        predictor_variables=predictor_variables,
        target_variables=target_variables,
        include_targets=include_targets,
        regrid_method=regrid_method,
        canonical_grid_fingerprint=canonical_grid_fingerprint,
        static_elevation_required=static_elevation_sha256 is not None,
        static_elevation_variable=static_elevation_variable,
    )
    contract.update(
        {
            "algorithm": str(algorithm),
            "source_grid_fingerprint": str(source_grid_fingerprint),
            "static_elevation_sha256": (
                None
                if static_elevation_sha256 is None
                else str(static_elevation_sha256)
            ),
            "missing_value_policy": "finite-linear-stencil-or-nan-v1",
        }
    )
    return contract


def preprocessing_signature(contract: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(contract), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"granitewxc-prism-preprocessing-v1\0")
    digest.update(payload)
    return digest.hexdigest()


def predictor_preprocessing_signature(contract: Mapping[str, Any]) -> str:
    """Hash only fields that determine the model's predictor tensor."""
    keys = (
        "schema_version",
        "data_type",
        "ordered_predictors",
        "regrid_method",
        "canonical_grid_fingerprint",
        "static_elevation_required",
        "static_elevation_variable",
        "algorithm",
        "source_grid_fingerprint",
        "static_elevation_sha256",
        "missing_value_policy",
    )
    predictor_contract = {key: contract.get(key) for key in keys}
    return preprocessing_signature(predictor_contract)


def source_artifact_signature(sources: Mapping[str, Path | str]) -> str:
    """Fingerprint the identity/version of every raw file behind one day.

    Device/inode catches atomic replacement, while size and nanosecond mtime
    catch ordinary in-place updates without rereading hundreds of megabytes of
    PRISM payload solely to validate a preprocessing cache hit.
    """
    entries: list[Dict[str, Any]] = []
    for label, raw_path in sorted(sources.items()):
        path = Path(raw_path).expanduser().resolve()
        stat = path.stat()
        entries.append(
            {
                "label": str(label),
                "path": str(path),
                "device": int(stat.st_dev),
                "inode": int(stat.st_ino),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return preprocessing_signature({"source_artifacts": entries})


def split_source_artifact_signature(
    daily_signatures: Mapping[str, str],
) -> str:
    """Bind an ordered date split to every daily raw-artifact identity."""
    entries = [
        [str(day), str(signature)]
        for day, signature in sorted(daily_signatures.items())
    ]
    return preprocessing_signature({"daily_source_artifacts": entries})


def preprocessing_attrs(contract: Mapping[str, Any]) -> Dict[str, str | int]:
    """Serialize a preprocessing contract into NetCDF-safe attributes."""
    normalized = dict(contract)
    return {
        "preprocessing_schema_version": PREPROCESSING_SCHEMA_VERSION,
        PREPROCESSING_CONTRACT_ATTR: json.dumps(
            normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
        PREPROCESSING_SIGNATURE_ATTR: preprocessing_signature(normalized),
        PREDICTOR_PREPROCESSING_SIGNATURE_ATTR: (
            predictor_preprocessing_signature(normalized)
        ),
    }


def read_preprocessing_contract(ds: Any, path: Path | str) -> Tuple[Dict[str, Any], str]:
    product_path = Path(path)
    schema = ds.attrs.get("preprocessing_schema_version")
    if int(schema or -1) != PREPROCESSING_SCHEMA_VERSION:
        raise ValueError(
            f"Preprocessed product {product_path} has preprocessing schema "
            f"{schema!r}, expected {PREPROCESSING_SCHEMA_VERSION}; regenerate it"
        )
    raw_contract = ds.attrs.get(PREPROCESSING_CONTRACT_ATTR)
    raw_signature = ds.attrs.get(PREPROCESSING_SIGNATURE_ATTR)
    raw_predictor_signature = ds.attrs.get(
        PREDICTOR_PREPROCESSING_SIGNATURE_ATTR
    )
    if (
        raw_contract is None
        or raw_signature is None
        or raw_predictor_signature is None
    ):
        raise ValueError(
            f"Preprocessed product {product_path} lacks its preprocessing "
            "contract/signature; regenerate it"
        )
    try:
        contract = json.loads(_text_attr(raw_contract))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Preprocessed product {product_path} has an invalid preprocessing contract"
        ) from exc
    if not isinstance(contract, dict):
        raise ValueError(
            f"Preprocessed product {product_path} preprocessing contract must be a mapping"
        )
    signature = _text_attr(raw_signature)
    computed = preprocessing_signature(contract)
    if signature != computed:
        raise ValueError(
            f"Preprocessed product {product_path} preprocessing signature is invalid: "
            f"stored={signature}, computed={computed}"
        )
    expected_predictor_signature = predictor_preprocessing_signature(contract)
    if _text_attr(raw_predictor_signature) != expected_predictor_signature:
        raise ValueError(
            f"Preprocessed product {product_path} predictor preprocessing "
            "signature is invalid"
        )
    return contract, signature


def inclusive_daily_dates(start: Any, end: Any) -> List[Any]:
    """Return every configured calendar date, preserving the input date type."""
    if end < start:
        raise ValueError(f"date range end {end} precedes start {start}")
    dates: List[Any] = []
    current = start
    while current <= end:
        dates.append(current)
        current = current + timedelta(days=1)
    return dates


def require_daily_products(
    directory: Path | str,
    prefix: str,
    dates: Iterable[Any],
) -> Dict[Any, Path]:
    """Resolve one deterministic product per date and fail on any missing date."""
    root = Path(directory)
    requested = list(dates)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Required preprocessed directory does not exist: {root}"
        )
    products = {
        sample_date: root / f"{prefix}_{sample_date:%Y%m%d}.nc"
        for sample_date in requested
    }
    missing = [path for path in products.values() if not path.is_file()]
    if missing:
        shown = ", ".join(path.name for path in missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... (+{len(missing) - 5} more)"
        raise FileNotFoundError(
            f"Missing {len(missing)}/{len(requested)} required preprocessed daily "
            f"products under {root}: {shown}{suffix}. Raw-data fallback is disabled."
        )
    return products


def predictor_name(variable: str, level: float) -> str:
    return f"predictor_{variable}_{int(level)}"


def target_name(variable: str) -> str:
    return f"target_{variable}"


def _text_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _coord_name(ds: Any, candidates: Sequence[str], *, path: Path) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.data_vars:
            return name
    raise ValueError(f"Preprocessed product {path} lacks coordinates {candidates}")


def validate_daily_product(
    ds: Any,
    path: Path | str,
    canonical_grid: CanonicalPrismGrid,
    *,
    mode: str,
    sample_date: Any,
    required_variables: Sequence[str],
    expected_preprocessing_contract: Mapping[str, Any] | None = None,
    expected_preprocessing_fields: Mapping[str, Any] | None = None,
    expected_preprocessing_signature: str | None = None,
    expected_source_artifact_signature: str | None = None,
) -> Tuple[str, str]:
    """Validate provenance, exact coordinates, and required spatial variables."""
    product_path = Path(path)
    source_signature = ds.attrs.get(SOURCE_ARTIFACT_SIGNATURE_ATTR)
    if source_signature is None:
        raise ValueError(
            f"Preprocessed product {product_path} lacks source_artifact_signature; "
            "regenerate it"
        )
    source_signature = _text_attr(source_signature)
    if (
        expected_source_artifact_signature is not None
        and source_signature != str(expected_source_artifact_signature)
    ):
        raise ValueError(
            f"Preprocessed product {product_path} was built from stale raw "
            f"artifacts: product={source_signature}, "
            f"current={expected_source_artifact_signature}"
        )
    preprocessing_contract, product_signature = read_preprocessing_contract(
        ds, product_path
    )
    if (
        expected_preprocessing_contract is not None
        and preprocessing_contract != dict(expected_preprocessing_contract)
    ):
        raise ValueError(
            f"Preprocessed product {product_path} was built with a different "
            f"preprocessing contract: product={preprocessing_contract}, "
            f"expected={dict(expected_preprocessing_contract)}"
        )
    for key, expected in (expected_preprocessing_fields or {}).items():
        observed = preprocessing_contract.get(key)
        if observed != expected:
            raise ValueError(
                f"Preprocessed product {product_path} preprocessing field {key!r} "
                f"is {observed!r}, expected {expected!r}; regenerate it"
            )
    if (
        expected_preprocessing_signature is not None
        and product_signature != str(expected_preprocessing_signature)
    ):
        raise ValueError(
            f"Preprocessed product {product_path} signature {product_signature} "
            f"does not match split signature {expected_preprocessing_signature}"
        )
    fingerprint = ds.attrs.get("prism_grid_fingerprint")
    if fingerprint is None:
        raise ValueError(
            f"Preprocessed product {product_path} lacks prism_grid_fingerprint; "
            "regenerate it with the canonical-grid preprocessor"
        )
    fingerprint = _text_attr(fingerprint)
    if fingerprint != canonical_grid.fingerprint:
        raise ValueError(
            f"Preprocessed product {product_path} grid fingerprint {fingerprint} "
            f"does not match canonical {canonical_grid.fingerprint}"
        )

    product_mode = ds.attrs.get("mode")
    if product_mode is None or _text_attr(product_mode) != str(mode):
        raise ValueError(
            f"Preprocessed product {product_path} mode={product_mode!r}, "
            f"expected {mode!r}"
        )
    product_date = ds.attrs.get("date")
    if product_date is None or _text_attr(product_date) != str(sample_date):
        raise ValueError(
            f"Preprocessed product {product_path} date={product_date!r}, "
            f"expected {str(sample_date)!r}"
        )

    lat_name = _coord_name(ds, LAT_CANDIDATES, path=product_path)
    lon_name = _coord_name(ds, LON_CANDIDATES, path=product_path)
    assert_grid_matches(
        canonical_grid,
        np.asarray(ds[lat_name].values, dtype=np.float64),
        np.asarray(ds[lon_name].values, dtype=np.float64),
        context=f"preprocessed product {product_path}",
    )

    missing = [name for name in required_variables if name not in ds.data_vars]
    if missing:
        raise ValueError(
            f"Preprocessed product {product_path} lacks required variables: {missing}"
        )
    for name in required_variables:
        dims = tuple(ds[name].dims)
        if len(dims) != 2 or set(dims) != {lat_name, lon_name}:
            raise ValueError(
                f"Preprocessed variable {name!r} in {product_path} must have only "
                f"({lat_name!r}, {lon_name!r}) dimensions; got {dims}"
            )
    return lat_name, lon_name


def read_spatial_variable(
    ds: Any,
    name: str,
    lat_name: str,
    lon_name: str,
    *,
    lat_slice: slice = slice(None),
    lon_slice: slice = slice(None),
) -> np.ndarray:
    """Read one already-validated daily field as ``float32[lat, lon]``."""
    da = ds[name].transpose(lat_name, lon_name)
    values = np.array(
        da.values[lat_slice, lon_slice], dtype=np.float32, copy=True
    )
    values[np.isinf(values)] = np.nan
    return values
