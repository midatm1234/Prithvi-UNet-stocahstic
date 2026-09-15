"""CORDEX dataset tests for exact pairing and explicit channel/grid layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from examples.CORDEX_ML.cordex_dataset import CordexDownscaleDataset


def _write_pair(tmp_path, *, include_march_first: bool = True):
    lat = np.asarray([-29.0, -30.0], dtype=np.float32)
    lon = np.asarray([24.0, 25.0, 26.0], dtype=np.float32)
    predictor_times = np.asarray(
        ["2000-02-28", "2000-03-01"], dtype="datetime64[ns]"
    )
    target_dates = ["2000-02-28", "2000-02-29"]
    if include_march_first:
        target_dates.append("2000-03-01")
    target_times = np.asarray(target_dates, dtype="datetime64[ns]")

    spatial = np.arange(6, dtype=np.float32).reshape(2, 3)
    first = np.stack((10.0 + spatial, 30.0 + spatial))
    second = np.stack((110.0 + spatial, 130.0 + spatial))
    predictors = xr.Dataset(
        {
            # Deliberately store the first requested channel lon-major.
            "first": (("time", "lon", "lat"), first.transpose(0, 2, 1)),
            "second": (("time", "lat", "lon"), second),
        },
        coords={"time": predictor_times, "lat": lat, "lon": lon},
    )

    target_count = len(target_dates)
    pr_canonical = np.stack(
        [100.0 * (index + 1) + spatial for index in range(target_count)]
    )
    tas_canonical = np.stack(
        [1000.0 * (index + 1) + spatial for index in range(target_count)]
    )
    targets = xr.Dataset(
        {
            # The two target variables intentionally use opposite dimension
            # orders; the returned tensor contract must still be [C,H,W].
            "pr": (
                ("time", "lon", "lat"),
                pr_canonical.transpose(0, 2, 1),
                {"units": "mm/day"},
            ),
            "tas": (
                ("time", "lat", "lon"),
                tas_canonical,
                {"units": "K"},
            ),
        },
        coords={"time": target_times, "lat": lat, "lon": lon},
    )

    predictor_path = tmp_path / "predictors.nc"
    target_path = tmp_path / "targets.nc"
    predictors.to_netcdf(predictor_path, engine="h5netcdf")
    targets.to_netcdf(target_path, engine="h5netcdf")
    return predictor_path, target_path, spatial


def _identity_regridder(self, grid_in, grid_out, method):
    del self, grid_in, grid_out, method
    return lambda data: data


def test_dataset_exact_timestamp_join_channel_order_and_chw_layout(
    tmp_path,
    monkeypatch,
):
    predictor_path, target_path, spatial = _write_pair(tmp_path)
    monkeypatch.setattr(
        CordexDownscaleDataset,
        "_build_regridder",
        _identity_regridder,
    )
    dataset = CordexDownscaleDataset(
        [predictor_path],
        [target_path],
        None,
        predictor_variables=("first", "second"),
        target_variables=("pr", "tas"),
        use_static=False,
        random_crop=False,
        allow_time_mismatch=True,
    )
    assert dataset.target_vars == ["pr", "tas"]
    assert dataset.target_units == ["mm/day", "K"]

    # Predictor index 1 is 1 March and must join target index 2, skipping the
    # Gregorian leap day rather than using positional index 1.
    sample = dataset[1]
    assert sample["x"].shape == (2, 2, 3)
    assert sample["y"].shape == (2, 2, 3)
    np.testing.assert_array_equal(sample["x"][0].numpy(), 30.0 + spatial)
    np.testing.assert_array_equal(sample["x"][1].numpy(), 130.0 + spatial)
    np.testing.assert_array_equal(sample["y"][0].numpy(), 300.0 + spatial)
    np.testing.assert_array_equal(sample["y"][1].numpy(), 3000.0 + spatial)
    assert sample["__sample_dataset_index"] == 1
    assert sample["__sample_file_index"] == 0
    assert sample["__sample_predictor_time_index"] == 1
    assert sample["__sample_target_time_index"] == 2
    assert sample["__sample_timestamp"] == "2000-03-01T00:00:00"
    assert sample["__sample_predictor_path"] == str(predictor_path)
    assert sample["__sample_target_path"] == str(target_path)


def test_dataset_refuses_positional_pairing_when_exact_date_is_missing(
    tmp_path,
    monkeypatch,
):
    predictor_path, target_path, _ = _write_pair(
        tmp_path,
        include_march_first=False,
    )
    monkeypatch.setattr(
        CordexDownscaleDataset,
        "_build_regridder",
        _identity_regridder,
    )
    with pytest.raises(ValueError, match="missing from the target"):
        CordexDownscaleDataset(
            [predictor_path],
            [target_path],
            None,
            predictor_variables=("first", "second"),
            target_variables=("pr", "tas"),
            use_static=False,
            random_crop=False,
            allow_time_mismatch=True,
        )


def test_precipitation_units_are_explicit_and_converted_exactly_once():
    flux = xr.DataArray(
        np.asarray([1.0 / 86_400.0]),
        name="pr",
        attrs={"units": "kg m-2 s-1"},
    )
    daily = CordexDownscaleDataset._convert_target_units(
        flux,
        variable_name="pr",
    )
    daily_again = CordexDownscaleDataset._convert_target_units(
        daily,
        variable_name="pr",
    )
    np.testing.assert_allclose(daily.values, [1.0])
    np.testing.assert_allclose(daily_again.values, daily.values)
    assert daily.attrs["units"] == "mm/day"

    tasmax = xr.DataArray([300.0], name="tasmax", attrs={"units": "K"})
    assert CordexDownscaleDataset._convert_target_units(
        tasmax,
        variable_name="tasmax",
    ) is tasmax

    for units in ("", "mm/month"):
        unknown = xr.DataArray([1.0], name="pr", attrs={"units": units})
        with pytest.raises(ValueError, match="precipitation.*units|Precipitation.*units"):
            CordexDownscaleDataset._convert_target_units(
                unknown,
                variable_name="pr",
            )


def test_target_validity_mask_precedes_legacy_finite_fill(tmp_path, monkeypatch):
    predictor_path, target_path, _ = _write_pair(tmp_path)
    with xr.open_dataset(target_path, engine="h5netcdf") as opened:
        targets = opened.load()
    targets["pr"].values[0, 0, 0] = np.nan
    targets["tas"].values[0, 0, 1] = np.inf
    targets.to_netcdf(target_path, engine="h5netcdf", mode="w")
    monkeypatch.setattr(
        CordexDownscaleDataset,
        "_build_regridder",
        _identity_regridder,
    )
    dataset = CordexDownscaleDataset(
        [predictor_path],
        [target_path],
        None,
        predictor_variables=("first", "second"),
        target_variables=("pr", "tas"),
        use_static=False,
        random_crop=False,
        allow_time_mismatch=True,
    )

    sample = dataset[0]
    assert sample["y"].isfinite().all()
    assert sample["__target_valid_mask"].dtype == torch.bool
    assert int(sample["__target_valid_mask"].numel()) == 12
    assert int(sample["__target_valid_mask"].sum()) == 10

    # The training wrapper must not discard the private mask before Phase-2
    # receives the batch.
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "examples" / "CORDEX_ML")
    )
    from cordex_training import CordexWrappedDataset

    wrapped = CordexWrappedDataset(dataset)[0]
    torch.testing.assert_close(
        wrapped["__target_valid_mask"],
        sample["__target_valid_mask"],
    )


def test_full_domain_disables_offset_augmentation(tmp_path, monkeypatch):
    predictor_path, target_path, _ = _write_pair(tmp_path)
    monkeypatch.setattr(
        CordexDownscaleDataset,
        "_build_regridder",
        _identity_regridder,
    )
    dataset = CordexDownscaleDataset(
        [predictor_path],
        [target_path],
        None,
        predictor_variables=("first", "second"),
        target_variables=("pr", "tas"),
        crop_size=(2, 3),
        random_crop=True,
        random_crop_offset=(1, 1),
        use_static=False,
        allow_time_mismatch=True,
    )
    assert dataset.random_crop is False

    def unexpected_offset(*args, **kwargs):
        del args, kwargs
        raise AssertionError("full-domain sample must not be spatially shifted")

    monkeypatch.setattr(dataset, "_apply_random_spatial_offset", unexpected_offset)
    sample = dataset[0]
    assert sample["y"].shape == (2, 2, 3)
    assert sample["__target_valid_mask"].all()
