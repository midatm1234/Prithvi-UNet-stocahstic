from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPO_ROOT / "examples" / "CORDEX_ML"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.nearest_fill import repair_invalid_by_nearest_xr  # noqa: E402
from utils.postprocess_outputs import enforce_pr_nonnegative_xr  # noqa: E402


def test_repair_invalid_by_nearest_fills_nans_xr():
    lat = xr.DataArray([0.0, 1.0], dims=("lat",), name="lat")
    lon = xr.DataArray([10.0, 11.0, 12.0], dims=("lon",), name="lon")
    data = np.array([[[1.0, np.nan, 2.0], [np.inf, 3.0, -9999.0]]], dtype=np.float32)
    ds = xr.Dataset(
        {
            "u_850": xr.DataArray(
                data,
                dims=("time", "lat", "lon"),
                coords={"time": [0], "lat": lat, "lon": lon},
                attrs={"_FillValue": -9999.0},
            )
        }
    )

    repaired = repair_invalid_by_nearest_xr(ds, var_names=["u_850"], use_scipy=False)
    values = repaired["u_850"].values

    assert np.isfinite(values).all()
    assert not np.any(values == -9999.0)


def test_enforce_pr_nonnegative_clamps_xr():
    ds = xr.Dataset(
        {
            "pr": xr.DataArray(np.array([[[-1.5, 0.5], [2.0, -0.2]]], dtype=np.float32), dims=("time", "lat", "lon")),
            "tasmax": xr.DataArray(np.array([[[280.0, 281.0], [282.0, 283.0]]], dtype=np.float32), dims=("time", "lat", "lon")),
        }
    )

    clamped = enforce_pr_nonnegative_xr(ds)
    assert float(clamped["pr"].min()) >= 0.0
    assert np.allclose(clamped["tasmax"].values, ds["tasmax"].values)

