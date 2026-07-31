"""Canonical, case-scoped PRISM latitude/longitude grid contract.

The NARR-PRISM and MERRA-PRISM workflows must use the exact same native
PRISM coordinates in preprocessing, scalar computation, training, and
evaluation.  Shape checks alone cannot detect a shifted or reversed grid, so
this module persists the selected coordinates as float64 arrays and records a
stable SHA-256 fingerprint of their values and ordering.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

try:
    import fcntl
except ImportError:  # pragma: no cover - PRISM workflows run on POSIX hosts
    fcntl = None


GRID_ARRAYS_NAME = "prism_grid.npz"
GRID_METADATA_NAME = "prism_grid.json"
GRID_SCHEMA_VERSION = 1
GRID_LOCK_NAME = ".prism_grid.lock"


@contextmanager
def _initialization_lock(case_dir: Path | str):
    """Serialize first-writer grid publication across preprocessing shards."""
    root = Path(case_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / GRID_LOCK_NAME
    with open(lock_path, "a+b") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _axis_hash(name: str, values: np.ndarray) -> str:
    """Hash an axis independent of host byte order and array memory layout."""
    canonical = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(b"granitewxc-prism-axis-v1\0")
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(struct.pack("<Q", canonical.size))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _grid_fingerprint(lat_hash: str, lon_hash: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"granitewxc-prism-grid-v1\0")
    digest.update(bytes.fromhex(lat_hash))
    digest.update(bytes.fromhex(lon_hash))
    return digest.hexdigest()


def _validate_axis(name: str, values: Any, *, context: str) -> Tuple[np.ndarray, str]:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(
            f"{context}: PRISM {name} coordinates must be one-dimensional; "
            f"got shape {raw.shape}"
        )
    if raw.size < 2:
        raise ValueError(
            f"{context}: PRISM {name} coordinates must contain at least two cells; "
            f"got {raw.size}"
        )
    try:
        axis = np.array(raw, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: PRISM {name} coordinates are not numeric") from exc
    if not np.isfinite(axis).all():
        bad = int((~np.isfinite(axis)).sum())
        raise ValueError(
            f"{context}: PRISM {name} coordinates contain {bad} non-finite values"
        )

    delta = np.diff(axis)
    if np.all(delta > 0.0):
        order = "ascending"
    elif np.all(delta < 0.0):
        order = "descending"
    else:
        duplicate = int((delta == 0.0).sum())
        raise ValueError(
            f"{context}: PRISM {name} coordinates must be strictly monotonic; "
            f"found {duplicate} duplicate and "
            f"{int(delta.size - duplicate)} non-uniform-direction steps"
        )
    axis.setflags(write=False)
    return axis, order


def _axis_metadata(values: np.ndarray, order: str, digest: str) -> Dict[str, Any]:
    spacing = np.abs(np.diff(values))
    return {
        "size": int(values.size),
        "sha256": digest,
        "order": order,
        "first": float(values[0]),
        "last": float(values[-1]),
        "min": float(values.min()),
        "max": float(values.max()),
        "spacing_min": float(spacing.min()),
        "spacing_max": float(spacing.max()),
        "spacing_median": float(np.median(spacing)),
    }


@dataclass(frozen=True)
class CanonicalPrismGrid:
    """Validated native PRISM axes and their persisted provenance metadata."""

    lat: np.ndarray
    lon: np.ndarray
    metadata: Mapping[str, Any]

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.lat.size), int(self.lon.size)

    @property
    def fingerprint(self) -> str:
        return str(self.metadata["fingerprint"])

    def manifest_entry(self) -> Dict[str, Any]:
        """Return the stable subset suitable for another artifact's manifest."""
        return {
            "schema_version": GRID_SCHEMA_VERSION,
            "dtype": "float64",
            "shape": [int(self.lat.size), int(self.lon.size)],
            "fingerprint": self.fingerprint,
            "lat_sha256": str(self.metadata["lat"]["sha256"]),
            "lon_sha256": str(self.metadata["lon"]["sha256"]),
            "lat_order": str(self.metadata["lat"]["order"]),
            "lon_order": str(self.metadata["lon"]["order"]),
        }


def validate_prism_grid(lat: Any, lon: Any, *, context: str) -> CanonicalPrismGrid:
    """Validate axes and build an in-memory canonical grid description."""
    lat64, lat_order = _validate_axis("latitude", lat, context=context)
    lon64, lon_order = _validate_axis("longitude", lon, context=context)
    lat_hash = _axis_hash("lat", lat64)
    lon_hash = _axis_hash("lon", lon64)
    metadata: Dict[str, Any] = {
        "schema_version": GRID_SCHEMA_VERSION,
        "dtype": "float64",
        "shape": [int(lat64.size), int(lon64.size)],
        "fingerprint": _grid_fingerprint(lat_hash, lon_hash),
        "lat": _axis_metadata(lat64, lat_order, lat_hash),
        "lon": _axis_metadata(lon64, lon_order, lon_hash),
    }
    return CanonicalPrismGrid(lat=lat64, lon=lon64, metadata=metadata)


def grid_contract_paths(case_dir: Path | str) -> Tuple[Path, Path]:
    root = Path(case_dir)
    return root / GRID_ARRAYS_NAME, root / GRID_METADATA_NAME


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_npz_atomic(path: Path, lat: np.ndarray, lon: np.ndarray) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as handle:
            np.savez(
                handle,
                lat=np.asarray(lat, dtype="<f8"),
                lon=np.asarray(lon, dtype="<f8"),
            )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_canonical_grid(
    case_dir: Path | str,
    lat: Any,
    lon: Any,
    *,
    source: Optional[str] = None,
    overwrite: bool = False,
) -> CanonicalPrismGrid:
    """Persist a validated grid contract under a case preprocessing directory."""
    grid = validate_prism_grid(lat, lon, context="canonical PRISM grid")
    arrays_path, metadata_path = grid_contract_paths(case_dir)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and (arrays_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"PRISM grid contract already exists under {arrays_path.parent}; "
            "use ensure_canonical_grid to validate it"
        )

    metadata = dict(grid.metadata)
    metadata["created_at"] = (
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    if source:
        metadata["source"] = str(source)
    _write_npz_atomic(arrays_path, grid.lat, grid.lon)
    _write_json_atomic(metadata_path, metadata)
    persisted = load_canonical_grid(case_dir, required=True)
    assert persisted is not None
    return persisted


def load_canonical_grid(
    case_dir: Path | str, *, required: bool = True
) -> Optional[CanonicalPrismGrid]:
    """Load a grid contract and verify its dtype, metadata, and coordinate hashes."""
    arrays_path, metadata_path = grid_contract_paths(case_dir)
    exists = (arrays_path.exists(), metadata_path.exists())
    if not any(exists):
        if required:
            raise FileNotFoundError(
                f"No PRISM grid contract found under {arrays_path.parent}"
            )
        return None
    if not all(exists):
        raise ValueError(
            f"Incomplete PRISM grid contract under {arrays_path.parent}: "
            f"{arrays_path.name} exists={exists[0]}, "
            f"{metadata_path.name} exists={exists[1]}"
        )

    with np.load(arrays_path, allow_pickle=False) as payload:
        if set(payload.files) != {"lat", "lon"}:
            raise ValueError(
                f"{arrays_path}: expected exactly lat/lon arrays, found {payload.files}"
            )
        stored_lat = payload["lat"]
        stored_lon = payload["lon"]
        if (
            stored_lat.dtype != np.dtype("float64")
            or stored_lon.dtype != np.dtype("float64")
        ):
            raise ValueError(
                f"{arrays_path}: canonical PRISM coordinates must be float64; "
                f"got lat={stored_lat.dtype}, lon={stored_lon.dtype}"
            )
        computed = validate_prism_grid(
            stored_lat, stored_lon, context=str(arrays_path)
        )

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema_version") != GRID_SCHEMA_VERSION:
        raise ValueError(
            f"{metadata_path}: unsupported PRISM grid schema_version "
            f"{metadata.get('schema_version')!r}"
        )
    expected_metadata = dict(computed.metadata)
    observed_metadata = {
        key: metadata.get(key) for key in expected_metadata
    }
    if observed_metadata != expected_metadata:
        raise ValueError(
            f"{metadata_path}: PRISM grid metadata does not match the persisted "
            f"coordinates; expected {expected_metadata}, found {observed_metadata}"
        )
    return CanonicalPrismGrid(lat=computed.lat, lon=computed.lon, metadata=metadata)


def _axis_mismatch_message(
    name: str, expected: np.ndarray, observed: np.ndarray, *, context: str
) -> str:
    if expected.shape != observed.shape:
        return (
            f"{context}: PRISM {name} coordinate shape {observed.shape} does "
            f"not match canonical shape {expected.shape}"
        )
    unequal = np.flatnonzero(expected != observed)
    first = int(unequal[0]) if unequal.size else -1
    max_abs = float(np.max(np.abs(expected - observed))) if unequal.size else 0.0
    return (
        f"{context}: PRISM {name} coordinates do not exactly match the "
        "canonical grid; "
        f"first mismatch at index {first}: canonical={expected[first]!r}, "
        f"observed={observed[first]!r}, max_abs_diff={max_abs:.17g}"
    )


def assert_grid_matches(
    canonical: CanonicalPrismGrid,
    lat: Any,
    lon: Any,
    *,
    context: str,
) -> None:
    """Require exact float64 coordinate values, ordering, and domain shape."""
    observed = validate_prism_grid(lat, lon, context=context)
    if not np.array_equal(canonical.lat, observed.lat):
        raise ValueError(
            _axis_mismatch_message(
                "latitude", canonical.lat, observed.lat, context=context
            )
        )
    if not np.array_equal(canonical.lon, observed.lon):
        raise ValueError(
            _axis_mismatch_message(
                "longitude", canonical.lon, observed.lon, context=context
            )
        )


def ensure_canonical_grid(
    case_dir: Path | str,
    lat: Any,
    lon: Any,
    *,
    source: Optional[str] = None,
    context: str = "PRISM grid",
) -> CanonicalPrismGrid:
    """Create the per-case contract once, then require exact matches thereafter."""
    # Every date shard reaches this point independently.  Recheck inside the
    # inter-process lock so only one shard publishes the two-file contract and
    # every follower validates the completed artifact.
    with _initialization_lock(case_dir):
        existing = load_canonical_grid(case_dir, required=False)
        if existing is None:
            return save_canonical_grid(case_dir, lat, lon, source=source)
        assert_grid_matches(existing, lat, lon, context=context)
        return existing
