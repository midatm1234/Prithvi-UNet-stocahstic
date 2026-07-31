from datetime import date

import numpy as np
import pytest
import xarray as xr

from examples.evaluate_prism_inference import _read_inference_day
from granitewxc.utils.prism_grid import validate_prism_grid


def _write_inference_file(path, *, variable="tmax", case_name="case-a"):
    lat = np.array([32.0, 32.25, 32.5], dtype=np.float64)
    lon = np.array([-120.0, -119.75, -119.5, -119.25], dtype=np.float64)
    grid = validate_prism_grid(lat, lon, context="test evaluator grid")
    sample_date = date(2020, 1, 2)
    epoch_day = (sample_date - date(1970, 1, 1)).days
    dataset = xr.Dataset(
        {
            variable: (
                ("time", "lat", "lon"),
                np.ones((1, lat.size, lon.size), dtype=np.float32),
            )
        },
        coords={"time": [epoch_day], "lat": lat, "lon": lon},
        attrs={
            "case_name": case_name,
            "checkpoint": "/checkpoints/corrected.ckpt",
            "inference_start": sample_date.isoformat(),
            "inference_end": sample_date.isoformat(),
            "prism_grid_fingerprint": grid.fingerprint,
        },
    )
    dataset["time"].attrs["units"] = "days since 1970-01-01"
    dataset.to_netcdf(path)
    return grid, sample_date


def test_inference_reader_validates_named_variable_and_provenance(tmp_path):
    path = tmp_path / "case-a_inference_20200102.nc"
    grid, sample_date = _write_inference_file(path)

    values, legacy, provenance = _read_inference_day(
        path,
        "tmax",
        grid,
        case_name="case-a",
        expected_date=sample_date,
        allow_legacy_float32=False,
    )
    assert values.shape == grid.shape
    assert not legacy
    assert provenance["checkpoint"] == "/checkpoints/corrected.ckpt"

    with pytest.raises(ValueError, match="requested inference variable 'tmin' is absent"):
        _read_inference_day(
            path,
            "tmin",
            grid,
            case_name="case-a",
            expected_date=sample_date,
            allow_legacy_float32=False,
        )


def test_inference_reader_rejects_wrong_case_and_grid_fingerprint(tmp_path):
    path = tmp_path / "case-a_inference_20200102.nc"
    grid, sample_date = _write_inference_file(path)

    with pytest.raises(ValueError, match="does not match configured case"):
        _read_inference_day(
            path,
            "tmax",
            grid,
            case_name="case-b",
            expected_date=sample_date,
            allow_legacy_float32=False,
        )

    with xr.open_dataset(path, decode_times=False) as source:
        changed = source.load()
    changed.attrs["prism_grid_fingerprint"] = "wrong"
    changed.to_netcdf(path, mode="w")
    with pytest.raises(ValueError, match="does not match canonical fingerprint"):
        _read_inference_day(
            path,
            "tmax",
            grid,
            case_name="case-a",
            expected_date=sample_date,
            allow_legacy_float32=False,
        )
