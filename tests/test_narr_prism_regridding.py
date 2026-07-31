from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from scipy.interpolate import LinearNDInterpolator

from examples.MERRA_PRISM import merra_prism_utils
from examples.NARR_PRISM import narr_prism_utils


SAMPLE_DATE = date(2001, 1, 2)


def _write_narr_channel(
    path: Path,
    values: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> None:
    dataset = xr.Dataset(
        {
            "air": (
                ("time", "level", "y", "x"),
                np.asarray(values, dtype=np.float32)[None, None, ...],
            )
        },
        coords={
            "time": [np.datetime64(SAMPLE_DATE)],
            "level": [850.0],
            "y": np.arange(values.shape[0]),
            "x": np.arange(values.shape[1]),
            "lat": (("y", "x"), np.asarray(lat, dtype=np.float64)),
            "lon": (("y", "x"), np.asarray(lon, dtype=np.float64)),
        },
    )
    dataset.to_netcdf(path)


def test_narr_constant_regridding_preserves_nan_outside_linear_support(
    tmp_path: Path,
) -> None:
    source_lat = np.array([[0.0, 0.0], [1.0, 1.0]])
    source_lon = np.array([[0.0, 1.0], [0.0, 1.0]])
    source = np.full((2, 2), 7.25, dtype=np.float32)
    source_path = tmp_path / "air.200101.nc"
    _write_narr_channel(source_path, source, source_lat, source_lon)

    target_lat = np.array([-0.25, 0.25, 0.75, 1.25])
    target_lon = np.array([-0.25, 0.25, 0.75, 1.25])
    result = narr_prism_utils.interpolate_narr_to_grid(
        {"air": source_path},
        "air",
        850.0,
        SAMPLE_DATE,
        target_lat,
        target_lon,
    )

    np.testing.assert_allclose(result[1:3, 1:3], 7.25)
    outside_support = np.ones(result.shape, dtype=bool)
    outside_support[1:3, 1:3] = False
    assert np.isnan(result[outside_support]).all()


def test_narr_linear_regridding_reproduces_a_linear_field(
    tmp_path: Path,
) -> None:
    source_lat_1d = np.array([0.0, 0.5, 1.0])
    source_lon_1d = np.array([10.0, 10.5, 11.0])
    source_lon, source_lat = np.meshgrid(source_lon_1d, source_lat_1d)
    source = 2.0 * source_lon - 3.0 * source_lat + 5.0
    source_path = tmp_path / "air.200101.nc"
    _write_narr_channel(source_path, source, source_lat, source_lon)

    target_lat = np.linspace(0.1, 0.9, 5)
    target_lon = np.linspace(10.1, 10.9, 6)
    result = narr_prism_utils.interpolate_narr_to_grid(
        source_path,
        "air",
        850.0,
        SAMPLE_DATE,
        target_lat,
        target_lon,
    )
    target_lon_2d, target_lat_2d = np.meshgrid(target_lon, target_lat)
    expected = 2.0 * target_lon_2d - 3.0 * target_lat_2d + 5.0

    np.testing.assert_allclose(result, expected, rtol=0.0, atol=2.0e-6)


def test_cached_barycentric_geometry_matches_scipy_and_supports_crops() -> None:
    source_y = np.linspace(0.0, 3.0, 5)
    source_x = np.linspace(10.0, 14.0, 6)
    source_lon, source_lat = np.meshgrid(source_x, source_y)
    source_lon = source_lon + 0.07 * source_lat
    source_lat = source_lat + 0.04 * np.sin(source_lon)
    values = np.sin(source_lon / 2.0) + 0.3 * source_lat**2

    target_lat = np.linspace(-0.2, 3.2, 21)
    target_lon = np.linspace(9.8, 14.4, 23)
    target_lon_2d, target_lat_2d = np.meshgrid(target_lon, target_lat)
    target_points = np.column_stack(
        [target_lon_2d.ravel(), target_lat_2d.ravel()]
    )
    reference = LinearNDInterpolator(
        np.column_stack([source_lon.ravel(), source_lat.ravel()]),
        values.ravel(),
        fill_value=np.nan,
    )(target_points).reshape(target_lat.size, target_lon.size)

    regridder = narr_prism_utils.NARRBarycentricRegridder.from_grids(
        source_lat, source_lon, target_lat, target_lon
    )
    actual = regridder.apply(values)
    np.testing.assert_allclose(actual, reference, rtol=2.0e-6, atol=2.0e-6)
    assert np.array_equal(np.isnan(actual), np.isnan(reference))

    crop = (slice(3, 17), slice(5, 20))
    cropped = regridder.apply(values, target_slices=crop)
    np.testing.assert_allclose(
        cropped, reference[crop], rtol=2.0e-6, atol=2.0e-6
    )


def test_cached_barycentric_geometry_propagates_nan_source_vertices() -> None:
    source_axis = np.array([0.0, 1.0, 2.0])
    source_lon, source_lat = np.meshgrid(source_axis, source_axis)
    target_axis = np.linspace(0.1, 1.9, 10)
    regridder = narr_prism_utils.NARRBarycentricRegridder.from_grids(
        source_lat, source_lon, target_axis, target_axis
    )

    values = 2.0 * source_lon - source_lat
    chosen_vertex = int(regridder.vertices[0, 0])
    values.ravel()[chosen_vertex] = np.nan
    result = regridder.apply(values)

    impacted = (regridder.vertices == chosen_vertex).any(axis=1).reshape(
        regridder.target_shape
    )
    assert impacted.any()
    assert (~impacted).any()
    assert np.isnan(result[impacted]).all()
    assert np.isfinite(result[~impacted]).all()


def test_barycentric_geometry_cache_round_trip(tmp_path: Path) -> None:
    source_axis = np.array([0.0, 1.0, 2.0])
    source_lon, source_lat = np.meshgrid(source_axis, source_axis)
    target_axis = np.linspace(-0.25, 2.25, 11)
    cache_path = tmp_path / "weights.npz"

    first = narr_prism_utils.NARRBarycentricRegridder.from_grids(
        source_lat,
        source_lon,
        target_axis,
        target_axis,
        cache_path=cache_path,
    )
    second = narr_prism_utils.NARRBarycentricRegridder.from_grids(
        source_lat,
        source_lon,
        target_axis,
        target_axis,
        cache_path=cache_path,
    )

    assert cache_path.is_file()
    assert not first.loaded_from_cache
    assert second.loaded_from_cache
    np.testing.assert_array_equal(second.vertices, first.vertices)
    np.testing.assert_array_equal(second.weights, first.weights)
    constant = np.full(source_lat.shape, 4.5, dtype=np.float32)
    result = second.apply(constant)
    inside = second.vertices[:, 0].reshape(second.target_shape) >= 0
    np.testing.assert_allclose(result[inside], 4.5)
    assert np.isnan(result[~inside]).all()


@pytest.mark.parametrize(
    "loader",
    [narr_prism_utils.load_elevation, merra_prism_utils.load_elevation],
    ids=["narr", "merra"],
)
def test_elevation_regridding_has_exact_lat_lon_alignment(
    loader,
    tmp_path: Path,
) -> None:
    source_lat = np.array([0.0, 1.0, 2.0])
    source_lon = np.array([10.0, 11.0, 12.0, 13.0])
    plane = 2.0 * source_lat[:, None] + 3.0 * source_lon[None, :] + 5.0
    elevation_path = tmp_path / "elevation.nc"
    xr.Dataset(
        {
            # Store longitude first to verify that the loader returns the
            # canonical (latitude, longitude) order, not the on-disk order.
            "elevation": (("lon", "lat"), plane.T.astype(np.float32))
        },
        coords={"lat": source_lat, "lon": source_lon},
    ).to_netcdf(elevation_path)

    target_lat = np.array([0.25, 1.5])
    target_lon = np.array([10.5, 12.25, 12.75])
    result = loader(
        elevation_path,
        var_name="elevation",
        target_lat=target_lat,
        target_lon=target_lon,
    )
    expected = 2.0 * target_lat[:, None] + 3.0 * target_lon[None, :] + 5.0

    assert result.shape == (target_lat.size, target_lon.size)
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=2.0e-6)


@pytest.mark.parametrize(
    "loader",
    [narr_prism_utils.load_elevation, merra_prism_utils.load_elevation],
    ids=["narr", "merra"],
)
def test_elevation_regridding_rejects_curvilinear_coordinates(
    loader,
    tmp_path: Path,
) -> None:
    elevation_path = tmp_path / "curvilinear_elevation.nc"
    xr.Dataset(
        {
            "elevation": (
                ("y", "x"),
                np.arange(4, dtype=np.float32).reshape(2, 2),
            )
        },
        coords={
            "lat": (("y", "x"), [[0.0, 0.0], [1.0, 1.0]]),
            "lon": (("y", "x"), [[10.0, 11.0], [10.1, 11.1]]),
        },
    ).to_netcdf(elevation_path)

    with pytest.raises(ValueError, match="must be one-dimensional"):
        loader(
            elevation_path,
            var_name="elevation",
            target_lat=np.array([0.25, 0.75]),
            target_lon=np.array([10.25, 10.75]),
        )
