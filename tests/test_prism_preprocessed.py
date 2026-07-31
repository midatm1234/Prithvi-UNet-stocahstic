from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from granitewxc.utils.prism_grid import validate_prism_grid
from granitewxc.utils.prism_preprocessed import (
    build_preprocessing_contract,
    inclusive_daily_dates,
    preprocessing_attrs,
    read_spatial_variable,
    require_daily_products,
    split_source_artifact_signature,
    validate_daily_product,
)


def _grid():
    return validate_prism_grid(
        np.array([41.0, 40.5], dtype=np.float64),
        np.array([-124.5, -124.0, -123.5], dtype=np.float64),
        context="test",
    )


def _product(grid, *, fingerprint=None, mode="training", day=date(2000, 1, 1)):
    shape = grid.shape
    contract = build_preprocessing_contract(
        data_type="narr_prism",
        predictor_variables=[("T", 850)],
        target_variables=["tmax"],
        include_targets=True,
        regrid_method="bilinear",
        canonical_grid_fingerprint=grid.fingerprint,
        static_elevation_sha256="a" * 64,
        static_elevation_variable="elevation",
        source_grid_fingerprint="b" * 64,
        algorithm="test-linear-v1",
    )
    return xr.Dataset(
        {
            "predictor_T_850": (("lat", "lon"), np.ones(shape, dtype=np.float32)),
            "target_tmax": (("lat", "lon"), np.full(shape, 2.0, dtype=np.float32)),
            "static_elevation": (
                ("lat", "lon"),
                np.full(shape, 3.0, dtype=np.float32),
            ),
        },
        coords={"lat": grid.lat, "lon": grid.lon},
        attrs={
            "prism_grid_fingerprint": fingerprint or grid.fingerprint,
            "mode": mode,
            "date": str(day),
            **preprocessing_attrs(contract),
            "source_artifact_signature": "d" * 64,
        },
    )


def test_require_daily_products_rejects_any_missing_date(tmp_path: Path) -> None:
    dates = inclusive_daily_dates(date(2000, 1, 1), date(2000, 1, 3))
    (tmp_path / "narr_prism_20000101.nc").touch()
    (tmp_path / "narr_prism_20000103.nc").touch()

    with pytest.raises(FileNotFoundError, match="Raw-data fallback is disabled"):
        require_daily_products(tmp_path, "narr_prism", dates)

    (tmp_path / "narr_prism_20000102.nc").touch()
    products = require_daily_products(tmp_path, "narr_prism", dates)
    assert list(products) == dates


def test_daily_product_validation_and_spatial_read() -> None:
    grid = _grid()
    ds = _product(grid)
    required = ["predictor_T_850", "target_tmax", "static_elevation"]

    lat_name, lon_name = validate_daily_product(
        ds,
        "narr_prism_20000101.nc",
        grid,
        mode="training",
        sample_date=date(2000, 1, 1),
        required_variables=required,
    )
    values = read_spatial_variable(
        ds,
        "target_tmax",
        lat_name,
        lon_name,
        lat_slice=slice(0, 1),
        lon_slice=slice(1, 3),
    )
    assert values.dtype == np.float32
    np.testing.assert_array_equal(values, np.full((1, 2), 2.0, dtype=np.float32))


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda ds: ds.attrs.pop("prism_grid_fingerprint"), "lacks prism_grid"),
        (
            lambda ds: ds.attrs.pop("preprocessing_signature"),
            "lacks its preprocessing",
        ),
        (
            lambda ds: ds.attrs.__setitem__("prism_grid_fingerprint", "bad"),
            "does not match canonical",
        ),
        (lambda ds: ds.attrs.__setitem__("mode", "inference"), "expected 'training'"),
        (lambda ds: ds.attrs.__setitem__("date", "2000-01-02"), "expected '2000-01-01'"),
        (lambda ds: ds.__delitem__("target_tmax"), "lacks required variables"),
    ],
)
def test_daily_product_validation_is_strict(mutation, message: str) -> None:
    grid = _grid()
    ds = _product(grid)
    mutation(ds)
    with pytest.raises(ValueError, match=message):
        validate_daily_product(
            ds,
            "bad.nc",
            grid,
            mode="training",
            sample_date=date(2000, 1, 1),
            required_variables=["predictor_T_850", "target_tmax"],
        )


def test_daily_product_rejects_shifted_coordinates() -> None:
    grid = _grid()
    ds = _product(grid).assign_coords(lon=grid.lon + 1.0e-8)
    with pytest.raises(ValueError, match="do not exactly match"):
        validate_daily_product(
            ds,
            "shifted.nc",
            grid,
            mode="training",
            sample_date=date(2000, 1, 1),
            required_variables=["predictor_T_850", "target_tmax"],
        )


def test_daily_product_rejects_stale_config_and_mixed_split_signature() -> None:
    grid = _grid()
    ds = _product(grid)
    expected_fields = {
        "ordered_predictors": [["T", 700.0]],
    }
    with pytest.raises(ValueError, match="ordered_predictors"):
        validate_daily_product(
            ds,
            "stale.nc",
            grid,
            mode="training",
            sample_date=date(2000, 1, 1),
            required_variables=["predictor_T_850", "target_tmax"],
            expected_preprocessing_fields=expected_fields,
        )

    with pytest.raises(ValueError, match="does not match split signature"):
        validate_daily_product(
            ds,
            "mixed.nc",
            grid,
            mode="training",
            sample_date=date(2000, 1, 1),
            required_variables=["predictor_T_850", "target_tmax"],
            expected_preprocessing_signature="c" * 64,
        )

    with pytest.raises(ValueError, match="stale raw artifacts"):
        validate_daily_product(
            ds,
            "stale-source.nc",
            grid,
            mode="training",
            sample_date=date(2000, 1, 1),
            required_variables=["predictor_T_850", "target_tmax"],
            expected_source_artifact_signature="e" * 64,
        )


def test_split_source_artifact_signature_binds_dates_and_daily_sources() -> None:
    first = {
        "2000-01-02": "b" * 64,
        "2000-01-01": "a" * 64,
    }
    reordered = dict(reversed(list(first.items())))
    assert split_source_artifact_signature(first) == (
        split_source_artifact_signature(reordered)
    )
    assert split_source_artifact_signature(first) != (
        split_source_artifact_signature(
            {**first, "2000-01-02": "c" * 64}
        )
    )
    assert split_source_artifact_signature(first) != (
        split_source_artifact_signature({"2000-01-01": "a" * 64})
    )
