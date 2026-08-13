"""Focused tests for date-local physical-target I/O in cached Phase 2."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import MethodType

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "examples" / "NARR_PRISM" / "narr_prism_dataset.py"


def _load_dataset_module():
    spec = importlib.util.spec_from_file_location(
        "_test_narr_prism_cached_target_io", DATASET_PATH
    )
    assert spec is not None and spec.loader is not None
    sys.path.insert(0, str(DATASET_PATH.parent))
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(DATASET_PATH.parent))


def _stub_dataset(*, cache_target_days: bool = True):
    module = _load_dataset_module()
    dataset = object.__new__(module.NarrPrismDataset)
    dataset.cache_target_days = cache_target_days
    dataset._cached_target_date = None
    dataset._cached_target_day = None
    dataset.target_vars = ["ppt", "tmax", "tmin"]
    dataset.fine_shape = (6, 7)
    full_days = {
        "2000-01-01": torch.arange(
            3 * 6 * 7, dtype=torch.float32
        ).reshape(3, 6, 7),
        "2000-01-02": torch.arange(
            3 * 6 * 7, dtype=torch.float32
        ).reshape(3, 6, 7)
        + 1000.0,
    }
    # Preserve a representative missing PRISM cell for exact NaN parity.
    full_days["2000-01-01"][0, 2, 3] = torch.nan
    calls: list[tuple[str, slice, slice]] = []

    def fake_load_targets(
        self,
        sample_date,
        lat_slice=slice(None),
        lon_slice=slice(None),
    ):
        calls.append((str(sample_date), lat_slice, lon_slice))
        return full_days[str(sample_date)][..., lat_slice, lon_slice].clone()

    dataset._load_targets = MethodType(fake_load_targets, dataset)
    return dataset, full_days, calls


def test_cached_target_day_loads_once_and_crops_exactly():
    dataset, full_days, calls = _stub_dataset()

    for tile_index in range(22):
        y0 = tile_index % 3
        x0 = tile_index % 4
        crop = dataset._load_target_crop(
            "2000-01-01", slice(y0, y0 + 3), slice(x0, x0 + 3)
        )
        torch.testing.assert_close(
            crop,
            full_days["2000-01-01"][..., y0 : y0 + 3, x0 : x0 + 3],
            equal_nan=True,
            rtol=0.0,
            atol=0.0,
        )
    assert [call[0] for call in calls] == ["2000-01-01"]

    next_day = dataset._load_target_crop(
        "2000-01-02", slice(2, 6), slice(1, 7)
    )
    torch.testing.assert_close(
        next_day,
        full_days["2000-01-02"][..., 2:6, 1:7],
        rtol=0.0,
        atol=0.0,
    )
    assert [call[0] for call in calls] == ["2000-01-01", "2000-01-02"]


def test_disabled_target_cache_preserves_direct_crop_io():
    dataset, full_days, calls = _stub_dataset(cache_target_days=False)
    crop = dataset._load_target_crop(
        "2000-01-01", slice(1, 4), slice(3, 7)
    )
    torch.testing.assert_close(
        crop,
        full_days["2000-01-01"][..., 1:4, 3:7],
        equal_nan=True,
        rtol=0.0,
        atol=0.0,
    )
    assert len(calls) == 1
    assert calls[0][0] == "2000-01-01"
    assert calls[0][1:] == (slice(1, 4), slice(3, 7))
