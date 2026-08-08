"""NetCDF output tests for two-phase predictions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

xr = pytest.importorskip("xarray")

from granitewxc.refinement.io import build_refined_dataset, write_refined_netcdf


def make_outputs(times=3, members=4, variables=("ppt", "tmax"), lat=5, lon=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    n_var = len(variables)
    deterministic = torch.rand(times, n_var, lat, lon, generator=g)
    member_fields = torch.rand(times, members, n_var, lat, lon, generator=g)
    return {
        "variables": list(variables),
        "coords": {
            "time": np.array(
                ["2016-01-01", "2016-01-02", "2016-01-03"][:times], dtype="datetime64[ns]"
            ),
            "lat": np.linspace(32.5, 41.0, lat),
            "lon": np.linspace(-124.5, -115.9, lon),
        },
        "deterministic": deterministic,
        "residual": torch.randn(times, n_var, lat, lon, generator=g),
        "members": member_fields,
        "ensemble_mean": member_fields.mean(dim=1),
        "ensemble_spread": member_fields.std(dim=1, unbiased=True),
        "refined": member_fields.mean(dim=1),
        "truth": torch.rand(times, n_var, lat, lon, generator=g),
        "units": {"ppt": "mm", "tmax": "degC"},
        "attrs": {"case_name": "unit_test", "calendar": "standard"},
    }


def test_dataset_contains_every_product():
    outputs = make_outputs()
    dataset = build_refined_dataset(**outputs)
    for name in ("ppt", "tmax"):
        for suffix in ("", "_residual", "_refined", "_members", "_ensemble_mean",
                       "_ensemble_spread", "_truth"):
            assert f"{name}{suffix}" in dataset.data_vars


def test_dimensions_units_and_coordinates_are_preserved():
    outputs = make_outputs()
    dataset = build_refined_dataset(**outputs)
    assert dataset["ppt"].dims == ("time", "lat", "lon")
    assert dataset["ppt_members"].dims == ("time", "member", "lat", "lon")
    assert dataset.sizes["member"] == 4
    assert dataset["ppt"].attrs["units"] == "mm"
    assert dataset["tmax"].attrs["units"] == "degC"
    assert dataset.attrs["case_name"] == "unit_test"
    np.testing.assert_allclose(dataset["lat"].values, outputs["coords"]["lat"])
    np.testing.assert_allclose(dataset["lon"].values, outputs["coords"]["lon"])
    np.testing.assert_array_equal(dataset["time"].values, outputs["coords"]["time"])


def test_latitude_and_longitude_orientation_is_untouched():
    outputs = make_outputs()
    outputs["coords"]["lat"] = outputs["coords"]["lat"][::-1]  # descending latitudes
    dataset = build_refined_dataset(**outputs)
    assert dataset["lat"].values[0] > dataset["lat"].values[-1]
    assert float(dataset["lon"].min()) < 0.0, "longitudes must stay in the -180..180 convention"


def test_member_order_is_preserved():
    outputs = make_outputs()
    dataset = build_refined_dataset(**outputs)
    expected = outputs["members"][:, :, 0].numpy()
    np.testing.assert_allclose(dataset["ppt_members"].values, expected)
    np.testing.assert_array_equal(dataset["member"].values, np.arange(4))


def test_masked_cells_survive_the_round_trip(tmp_path):
    outputs = make_outputs()
    outputs["deterministic"][0, 0, 0, 0] = float("nan")
    outputs["members"][0, :, 0, 1, 1] = float("nan")
    dataset = build_refined_dataset(**outputs)
    path = write_refined_netcdf(dataset, tmp_path / "out.nc")
    with xr.open_dataset(path) as loaded:
        assert np.isnan(loaded["ppt"].values[0, 0, 0])
        assert np.isnan(loaded["ppt_members"].values[0, :, 1, 1]).all()


def test_values_are_bitwise_preserved_through_compression(tmp_path):
    outputs = make_outputs()
    dataset = build_refined_dataset(**outputs)
    path = write_refined_netcdf(
        dataset, tmp_path / "out.nc", compression=True, compression_level=4,
        chunk_sizes={"time": 1, "member": 1},
    )
    with xr.open_dataset(path) as loaded:
        for name in dataset.data_vars:
            np.testing.assert_array_equal(loaded[name].values, dataset[name].values)


def test_uncompressed_and_compressed_outputs_are_identical(tmp_path):
    dataset = build_refined_dataset(**make_outputs())
    plain = write_refined_netcdf(dataset, tmp_path / "plain.nc", compression=False)
    packed = write_refined_netcdf(dataset, tmp_path / "packed.nc", compression=True)
    with xr.open_dataset(plain) as a, xr.open_dataset(packed) as b:
        for name in dataset.data_vars:
            # Compression is lossless: identical values, identical ordering.
            np.testing.assert_array_equal(a[name].values, b[name].values)
            assert a[name].dims == b[name].dims


def test_atomic_write_leaves_no_temporary_files(tmp_path):
    dataset = build_refined_dataset(**make_outputs())
    write_refined_netcdf(dataset, tmp_path / "out.nc", atomic=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.nc"]


def test_channel_count_mismatch_is_rejected():
    outputs = make_outputs()
    outputs["variables"] = ["ppt"]
    with pytest.raises(ValueError, match="channels"):
        build_refined_dataset(**outputs)
