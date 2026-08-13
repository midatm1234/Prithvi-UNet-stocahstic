from __future__ import annotations

from pathlib import Path
import sys

import cftime
import numpy as np
import pytest
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPO_ROOT / "examples" / "CORDEX_ML"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from cordex_dataset import CordexDownscaleDataset, _exact_time_index  # noqa: E402


def _gregorian_days(start: str, stop: str) -> np.ndarray:
    return np.arange(
        np.datetime64(start),
        np.datetime64(stop),
        np.timedelta64(1, "D"),
    )


def test_exact_time_index_maps_7300_noleap_days_into_7305_gregorian_days():
    predictor_times = xr.date_range(
        "1981-01-01",
        "2000-12-31",
        freq="D",
        calendar="noleap",
        use_cftime=True,
    ).values
    target_times = _gregorian_days("1981-01-01", "2001-01-01")

    target_index = _exact_time_index(predictor_times, target_times)

    assert len(predictor_times) == 7300
    assert len(target_times) == 7305
    assert target_index.shape == (7300,)
    assert target_index[0] == 0
    assert target_times[target_index[-1]] == np.datetime64("2000-12-31")

    march_1984 = next(
        index
        for index, value in enumerate(predictor_times)
        if (value.year, value.month, value.day) == (1984, 3, 1)
    )
    assert target_times[target_index[march_1984]] == np.datetime64("1984-03-01")
    assert target_index[march_1984] == march_1984 + 1


def test_exact_time_index_rejects_missing_predictor_date():
    predictor_times = np.asarray(
        [np.datetime64("2000-02-28"), np.datetime64("2000-03-01")]
    )
    target_times = np.asarray(
        [np.datetime64("2000-02-28"), np.datetime64("2000-02-29")]
    )

    with pytest.raises(ValueError, match="1 missing date.*2000-03-01"):
        _exact_time_index(
            predictor_times,
            target_times,
            predictor_path="predictor.nc",
            target_path="target.nc",
        )


@pytest.mark.parametrize("duplicate_role", ["predictor", "target"])
def test_exact_time_index_rejects_duplicate_dates(duplicate_role: str):
    unique = np.asarray(
        [np.datetime64("2000-02-28"), np.datetime64("2000-03-01")]
    )
    duplicate = np.asarray(
        [np.datetime64("2000-02-28"), np.datetime64("2000-02-28")]
    )
    predictor_times = duplicate if duplicate_role == "predictor" else unique
    target_times = duplicate if duplicate_role == "target" else unique

    with pytest.raises(ValueError, match=rf"(?i){duplicate_role}.*duplicate"):
        _exact_time_index(predictor_times, target_times)


def test_dataset_uses_mapped_target_index_instead_of_integer_position(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    predictor_path = tmp_path / "predictor.nc"
    target_path = tmp_path / "target.nc"

    predictor_times = np.asarray(
        [
            cftime.DatetimeNoLeap(2000, 2, 28),
            cftime.DatetimeNoLeap(2000, 3, 1),
        ],
        dtype=object,
    )
    target_times = np.asarray(
        [
            np.datetime64("2000-02-28"),
            np.datetime64("2000-02-29"),
            np.datetime64("2000-03-01"),
        ]
    )
    coordinates = {"lat": [0.0], "lon": [10.0]}
    xr.Dataset(
        {
            "x": (
                ("time", "lat", "lon"),
                np.asarray([1.0, 2.0], dtype=np.float32).reshape(2, 1, 1),
            )
        },
        coords={"time": predictor_times, **coordinates},
    ).to_netcdf(predictor_path)
    xr.Dataset(
        {
            "y": (
                ("time", "lat", "lon"),
                np.asarray([10.0, 999.0, 20.0], dtype=np.float32).reshape(3, 1, 1),
            )
        },
        coords={"time": target_times, **coordinates},
    ).to_netcdf(target_path)

    monkeypatch.setattr(
        CordexDownscaleDataset,
        "_build_regridder",
        lambda self, grid_in, grid_out, method: (lambda data: data),
    )
    dataset = CordexDownscaleDataset(
        predictor_files=[predictor_path],
        target_files=[target_path],
        orography_file=None,
        predictor_variables=["x"],
        target_variables=["y"],
        use_static=False,
        random_crop=False,
        allow_time_mismatch=True,
    )

    assert len(dataset) == 2
    assert dataset[0]["y"].item() == pytest.approx(10.0)
    assert dataset[1]["y"].item() == pytest.approx(20.0)
    assert dataset._target_time_indices[0].tolist() == [0, 2]


def test_exact_alignment_requires_a_time_coordinate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    predictor_path = tmp_path / "predictor_without_time_coordinate.nc"
    target_path = tmp_path / "target.nc"
    coordinates = {"lat": [0.0], "lon": [10.0]}
    xr.Dataset(
        {"x": (("time", "lat", "lon"), np.ones((1, 1, 1), dtype=np.float32))},
        coords=coordinates,
    ).to_netcdf(predictor_path)
    xr.Dataset(
        {"y": (("time", "lat", "lon"), np.ones((1, 1, 1), dtype=np.float32))},
        coords={"time": [np.datetime64("2000-01-01")], **coordinates},
    ).to_netcdf(target_path)

    monkeypatch.setattr(
        CordexDownscaleDataset,
        "_build_regridder",
        lambda self, grid_in, grid_out, method: (lambda data: data),
    )
    with pytest.raises(ValueError, match="Predictor file.*missing time coordinate"):
        CordexDownscaleDataset(
            predictor_files=[predictor_path],
            target_files=[target_path],
            orography_file=None,
            predictor_variables=["x"],
            target_variables=["y"],
            use_static=False,
            allow_time_mismatch=True,
        )


def test_dataset_rejects_nonfinite_targets_instead_of_zero_filling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    predictor_path = tmp_path / "predictor.nc"
    target_path = tmp_path / "target_with_missing_value.nc"
    coordinates = {
        "time": [np.datetime64("2000-01-01")],
        "lat": [0.0],
        "lon": [10.0, 11.0],
    }
    xr.Dataset(
        {
            "x": (
                ("time", "lat", "lon"),
                np.ones((1, 1, 2), dtype=np.float32),
            )
        },
        coords=coordinates,
    ).to_netcdf(predictor_path)
    xr.Dataset(
        {
            "y": (
                ("time", "lat", "lon"),
                np.asarray([[[1.0, np.nan]]], dtype=np.float32),
            )
        },
        coords=coordinates,
    ).to_netcdf(target_path)

    monkeypatch.setattr(
        CordexDownscaleDataset,
        "_build_regridder",
        lambda self, grid_in, grid_out, method: (lambda data: data),
    )
    dataset = CordexDownscaleDataset(
        predictor_files=[predictor_path],
        target_files=[target_path],
        orography_file=None,
        predictor_variables=["x"],
        target_variables=["y"],
        use_static=False,
        random_crop=False,
    )

    with pytest.raises(ValueError, match="masked targets are not supported"):
        _ = dataset[0]
