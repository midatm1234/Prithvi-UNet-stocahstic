from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPO_ROOT / "examples" / "CORDEX_ML"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from types import SimpleNamespace

from granitewxc.utils.predictands import build_predictand_specs  # noqa: E402
from utils.nearest_fill import repair_invalid_by_nearest_xr  # noqa: E402
from utils.predictand_runtime import assert_nonnegative_outputs  # noqa: E402


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


def test_nonnegative_assertion_for_predictands():
    config = SimpleNamespace(
        data=SimpleNamespace(output_vars=["pr", "tasmax"]),
        predictands={},
    )
    specs = {spec.name: spec for spec in build_predictand_specs(config, ["pr", "tasmax"])}

    positive_outputs = {
        "pr": np.array([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32),
        "tasmax": np.array([[[280.0, 281.0], [282.0, 283.0]]], dtype=np.float32),
    }
    assert_nonnegative_outputs(positive_outputs, specs, eps=1e-8)

    negative_outputs = dict(positive_outputs)
    negative_outputs["pr"] = np.array([[[-1e-2, 0.0], [1.0, 2.0]]], dtype=np.float32)
    try:
        assert_nonnegative_outputs(negative_outputs, specs, eps=1e-8)
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected assertion failure for negative pr outputs.")
