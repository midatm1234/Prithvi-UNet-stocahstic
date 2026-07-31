from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from granitewxc.utils import normalization as norm
from granitewxc.utils.prism_grid import (
    GRID_ARRAYS_NAME,
    GRID_METADATA_NAME,
    assert_grid_matches,
    ensure_canonical_grid,
    load_canonical_grid,
    validate_prism_grid,
)


def _axes() -> tuple[np.ndarray, np.ndarray]:
    # Descending latitude matches the native PRISM files used by the examples.
    lat = np.array([41.0, 40.5, 40.0], dtype=np.float64)
    lon = np.array([-124.5, -124.0, -123.5, -123.0], dtype=np.float64)
    return lat, lon


def test_grid_contract_round_trip_is_float64_and_stable(tmp_path: Path) -> None:
    lat, lon = _axes()
    case_dir = tmp_path / "case"

    created = ensure_canonical_grid(
        case_dir,
        lat.astype(np.float32),
        lon.astype(np.float32),
        source="first_prism.nc",
    )
    loaded = load_canonical_grid(case_dir)

    assert loaded is not None
    assert loaded.lat.dtype == np.float64
    assert loaded.lon.dtype == np.float64
    np.testing.assert_array_equal(loaded.lat, lat)
    np.testing.assert_array_equal(loaded.lon, lon)
    assert loaded.fingerprint == created.fingerprint
    assert loaded.metadata["lat"]["order"] == "descending"
    assert loaded.metadata["lon"]["order"] == "ascending"
    assert (case_dir / GRID_ARRAYS_NAME).is_file()
    assert (case_dir / GRID_METADATA_NAME).is_file()

    # Re-entering from another pipeline stage validates rather than overwrites.
    same = ensure_canonical_grid(case_dir, lat, lon, context="scalar grid")
    assert same.fingerprint == created.fingerprint


def test_concurrent_first_writers_publish_one_complete_contract(tmp_path: Path) -> None:
    lat, lon = _axes()
    case_dir = tmp_path / "shared_case"

    def initialize(worker: int) -> str:
        return ensure_canonical_grid(
            case_dir,
            lat,
            lon,
            source=f"worker-{worker}.nc",
            context=f"worker {worker}",
        ).fingerprint

    with ThreadPoolExecutor(max_workers=8) as pool:
        fingerprints = list(pool.map(initialize, range(16)))

    assert len(set(fingerprints)) == 1
    loaded = load_canonical_grid(case_dir)
    assert loaded is not None
    np.testing.assert_array_equal(loaded.lat, lat)
    np.testing.assert_array_equal(loaded.lon, lon)


@pytest.mark.parametrize(
    "lat, lon, message",
    [
        ([1.0, 1.0], [10.0, 11.0], "strictly monotonic"),
        ([1.0, 0.0, 0.5], [10.0, 11.0], "strictly monotonic"),
        ([1.0, np.nan], [10.0, 11.0], "non-finite"),
        ([[1.0, 0.0]], [10.0, 11.0], "one-dimensional"),
    ],
)
def test_grid_validation_rejects_invalid_axes(lat, lon, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_prism_grid(lat, lon, context="test grid")


def test_exact_grid_validation_rejects_shift_and_reversal(tmp_path: Path) -> None:
    lat, lon = _axes()
    canonical = ensure_canonical_grid(tmp_path / "case", lat, lon)

    shifted_lon = lon.copy()
    shifted_lon[2] += 1.0e-12
    with pytest.raises(ValueError, match=r"first mismatch at index 2"):
        assert_grid_matches(
            canonical, lat, shifted_lon, context="shifted target file"
        )

    with pytest.raises(ValueError, match="do not exactly match"):
        assert_grid_matches(
            canonical, lat[::-1], lon, context="reversed target file"
        )


def test_grid_loader_detects_metadata_tampering(tmp_path: Path) -> None:
    lat, lon = _axes()
    case_dir = tmp_path / "case"
    ensure_canonical_grid(case_dir, lat, lon)

    metadata_path = case_dir / GRID_METADATA_NAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["lon"]["sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata does not match"):
        load_canonical_grid(case_dir)


def test_scalar_manifest_is_bound_to_canonical_grid(tmp_path: Path) -> None:
    cfg = {
        "case_name": "example_case",
        "data": {
            "preprocessed_dir": str(tmp_path),
            "target_variables": ["tmax"],
        },
        "normalization": {"predictor_mode": "global"},
        "predictands": {"tmax": {"normalization": {"mode": "global"}}},
    }
    case_dir = norm.case_preprocess_dir(cfg)
    lat, lon = _axes()
    grid = ensure_canonical_grid(case_dir, lat, lon)
    scalar_dir = case_dir / "scalars"
    scalar_dir.mkdir(parents=True)
    for name in norm.SCALAR_NAMES:
        np.save(scalar_dir / f"{name}.npy", np.array([1.0], dtype=np.float32))

    manifest_path = norm.write_manifest(
        scalar_dir,
        case_name="example_case",
        predictor_mode="global",
        cfg=cfg,
        train_date_range=["2000-01-01", "2000-12-31"],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["prism_grid"] == grid.manifest_entry()

    summary = norm.log_scalar_summary(cfg, "test", logger=lambda _: None)
    assert summary["prism_grid"]["fingerprint"] == grid.fingerprint

    manifest["prism_grid"]["fingerprint"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="PRISM grid fingerprint mismatch"):
        norm.log_scalar_summary(cfg, "test", logger=lambda _: None)
