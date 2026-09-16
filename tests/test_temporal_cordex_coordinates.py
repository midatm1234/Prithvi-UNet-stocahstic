"""SA/CORDEX physical coordinates survive inference and NetCDF export."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from granitewxc.temporal.config import parse_temporal_config
from granitewxc.temporal.inference import InferenceResult, run_sequence_inference, write_netcdf
from granitewxc.temporal.sources import CordexFrameSource, load_module_from_path
from tests.test_cordex_dataset_contract import _write_pair, _identity_regridder


@pytest.mark.parametrize("crop", [(2, 3), (1, 2)])
def test_cordex_inference_exports_physical_coordinates_for_exact_crop(tmp_path, monkeypatch, crop):
    predictor, target, _ = _write_pair(tmp_path)
    module = load_module_from_path("_temporal_cordex_dataset",
        Path(__file__).resolve().parents[1] / "examples/CORDEX_ML/cordex_dataset.py")
    monkeypatch.setattr(module.CordexDownscaleDataset, "_build_regridder", _identity_regridder)
    source = CordexFrameSource([str(predictor)], [str(target)], orography_path=None,
        predictor_variables=["first", "second"], target_variables=["pr", "tas"], use_static=False)
    monkeypatch.setattr("granitewxc.temporal.sources.build_frame_source", lambda *args: source)
    config = SimpleNamespace(data=SimpleNamespace(type="cordex", use_static=False,
        train_crop_size_lat=crop[0], train_crop_size_lon=crop[1]),
        model=SimpleNamespace(num_static_channels=0))
    cfg = parse_temporal_config({"enabled": True, "backend": "recurrent",
        "context_length": 1, "warmup_length": 0, "output_length": 1,
        "init_from_spatial_checkpoint": "unused.pt"})

    class GeometryRunner:
        adapter = SimpleNamespace(detach_state=lambda state: state)
        _allow_cold_start = False

        def eval(self):
            return self

        def __call__(self, batch, *, initial_state=None, emit_indices):
            return SimpleNamespace(predictions=batch["x"][:, emit_indices], final_state=None,
                target_frames=batch["y"][:, emit_indices], valid_mask=batch["__target_valid_mask"][:, emit_indices])

    results = run_sequence_inference(GeometryRunner(), config, cfg, verbose=False, chunk_length=2)
    assert len(results) == 2  # omitted leap day remains a real discontinuity
    expected_lat = np.asarray([-29.0, -30.0])[:crop[0]]
    expected_lon = np.asarray([24.0, 25.0, 26.0])[:crop[1]]
    for index, result in enumerate(results):
        np.testing.assert_array_equal(result.lat, expected_lat)
        np.testing.assert_array_equal(result.lon, expected_lon)
        path = tmp_path / f"physical_{index}.nc"
        write_netcdf(result, path, output_vars=["pr", "tas"], units=source.target_units)
        with xr.open_dataset(path) as written:
            np.testing.assert_array_equal(written.lat.values, expected_lat)
            np.testing.assert_array_equal(written.lon.values, expected_lon)
            assert written.pr.attrs["units"] == "mm/day"
            assert written.tas.attrs["units"] == "K"
            assert written.pr.shape[-2:] == crop


def test_netcdf_preserves_two_dimensional_physical_coordinates(tmp_path):
    lat = np.array([[-30.0, -29.9, -29.8], [-31.0, -30.9, -30.8]])
    lon = np.array([[24.0, 25.0, 26.0], [24.1, 25.1, 26.1]])
    result = InferenceResult(run_id=0, dates=[(2000, 1, 1, 0, 0, 0)],
        predictions=np.ones((1, 1, 2, 3)), targets=None, valid_mask=None, lat=lat, lon=lon)
    path = tmp_path / "curvilinear.nc"
    write_netcdf(result, path, output_vars=["tas"], units=["K"])
    with xr.open_dataset(path) as written:
        assert written.tas.dims == ("time", "y", "x")
        assert written.lat.dims == ("y", "x")
        np.testing.assert_array_equal(written.lat.values, lat)
        np.testing.assert_array_equal(written.lon.values, lon)
