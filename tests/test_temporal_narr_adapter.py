"""Strict synthetic NetCDF fixtures for the actual NARR temporal adapter."""
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from granitewxc.temporal.sources import NarrPrismDatasetFrameSource
from granitewxc.temporal.sequence_dataset import TemporalSequenceDataset, collate_sequences
from granitewxc.utils import normalization as norm
from granitewxc.utils.prism_grid import save_canonical_grid
from granitewxc.utils.prism_preprocessed import build_preprocessing_contract, preprocessing_attrs


def _write_case(tmp_path):
    cfg = {
        "case_name": "narr_prism_California",
        "data": {
            "type": "narr_prism", "use_preprocessed": True,
            "preprocessed_dir": str(tmp_path / "preprocessed"),
            "predictor_variables": {"shum": [850], "air": [850]},
            "target_variables": ["ppt", "tmax", "tmin"],
            "output_vars": ["ppt", "tmax", "tmin"],
            "input_vars": ["shum_850", "air_850", "elev", "mask_shum_850", "mask_air_850", "mask_elev"],
            "input_levels": [1], "static_elevation_file": "unused-in-preprocessed.nc",
            "static_elevation_var": None,
            "train_crop_size_lat": 4, "train_crop_size_lon": 4,
            "training_spatial_sampling": "tiled",
            "training_tile_stride_lat": 2, "training_tile_stride_lon": 2,
            "training_halo_lat": 1, "training_halo_lon": 1,
            "skip_empty_target_tiles": False,
        },
        "dates": {"validation": {"start": "2000-01-01", "end": "2000-01-05"}},
        "model": {"num_static_channels": 0},
        "normalization": {"predictor_mode": "gridpoint", "target_method": "per_variable"},
    }
    cfg["predictands"] = {v: {"normalization": {"mode": "gridpoint"}} for v in ("ppt", "tmax", "tmin")}
    case_dir = norm.case_preprocess_dir(cfg)
    grid = save_canonical_grid(case_dir, np.linspace(40, 39, 6), np.linspace(-124, -123, 6))
    contract = build_preprocessing_contract(
        data_type="narr_prism", predictor_variables=[("shum", 850), ("air", 850)],
        target_variables=["ppt", "tmax", "tmin"], include_targets=True,
        regrid_method="bilinear", canonical_grid_fingerprint=grid.fingerprint,
        static_elevation_sha256="a" * 64, static_elevation_variable=None,
        source_grid_fingerprint="b" * 64, algorithm="synthetic-test-v1",
    )
    attrs = preprocessing_attrs(contract)
    folder = case_dir / "validation"
    folder.mkdir()
    field = np.arange(36, dtype=np.float32).reshape(6, 6)
    for i in range(5):
        day = date(2000, 1, 1) + timedelta(days=i)
        atmospheric = field + i
        atmospheric = atmospheric.copy()
        atmospheric[0, 0] = np.nan
        target = field.copy()
        target[1, 1] = np.nan
        ds = xr.Dataset({
            "predictor_shum_850": (("lat", "lon"), atmospheric),
            "predictor_air_850": (("lat", "lon"), field + 270 + i),
            "static_elevation": (("lat", "lon"), field + 100),
            "target_ppt": (("lat", "lon"), target),
            "target_tmax": (("lat", "lon"), field + 25),
            "target_tmin": (("lat", "lon"), field + 15),
        }, coords={"lat": grid.lat, "lon": grid.lon}, attrs={
            **attrs, "date": str(day), "mode": "validation",
            "prism_grid_fingerprint": grid.fingerprint, "source_artifact_signature": "d" * 64,
        })
        ds.to_netcdf(folder / f"narr_prism_{day:%Y%m%d}.nc")
    scalars = case_dir / "scalars"
    scalars.mkdir()
    means = np.zeros((6, 6, 6), dtype=np.float32)
    means[0] = field + 7
    means[1] = 275
    means[2] = field + 100
    for key, values in {
        "inputs_mean": means, "inputs_std": np.ones_like(means),
        "targets_mean": np.zeros((3, 6, 6), dtype=np.float32),
        "targets_std": np.ones((3, 6, 6), dtype=np.float32),
    }.items():
        np.save(scalars / f"{key}.npy", values)
    norm.write_manifest(scalars, case_name=cfg["case_name"], predictor_mode="gridpoint", cfg=cfg,
        extra={"predictor_preprocessing_signature": attrs["predictor_preprocessing_signature"]})
    return cfg


def test_narr_netcdf_sequence_matches_spatial_loader_with_tile_halos(tmp_path):
    cfg = _write_case(tmp_path)
    source = NarrPrismDatasetFrameSource(cfg, "validation")
    sequence = TemporalSequenceDataset(source, window_length=3, crop_size=(4, 4))
    assert len(source.frames()) == 5
    assert len(sequence.spatial_tiles) == 4
    assert len(sequence) == len(sequence.windows) * 4
    plan = sequence.unique_emission_plan(output_length=2)
    assert len(plan) == 12
    assert {index % 4 for index, _ in plan} == {0, 1, 2, 3}
    for tile in (0, 3):
        sample = sequence[tile]
        assert sample["x"].shape == (3, 6, 6, 6)
        assert sample["y"].shape == (3, 3, 4, 4)
        assert tuple(sample["__output_crop"].tolist()) == (1, 1, 4, 4)
        for t in range(3):
            spatial = source._dataset[t * 4 + tile]
            assert torch.equal(sample["x"][t], spatial["x"])
            torch.testing.assert_close(sample["y"][t], torch.nan_to_num(spatial["y"]))
            assert torch.equal(sample["__target_valid_mask"][t], torch.isfinite(spatial["y"]))
        collated = collate_sequences([sample])
        assert collated["__input_scaler_offset"].shape == (1, 2)
    assert tuple(sequence[0]["__input_scaler_offset"].tolist()) == (-1, -1)
    assert tuple(sequence[3]["__input_scaler_offset"].tolist()) == (1, 1)
    # The missing atmospheric value is mean-filled, with its own mask zero.
    loaded = source.load_frame(source.frames()[0], slice(0, 4), slice(0, 4))
    assert loaded["x"][0, 1, 1] == 7
    assert loaded["x"][3, 1, 1] == 0


def test_narr_source_rejects_missing_daily_netcdf_instead_of_using_npz(tmp_path):
    cfg = _write_case(tmp_path)
    day = norm.case_preprocess_dir(cfg) / "validation" / "narr_prism_20000103.nc"
    day.rename(day.with_suffix(".missing"))
    with pytest.raises(FileNotFoundError, match="Raw-data fallback is disabled"):
        NarrPrismDatasetFrameSource(cfg, "validation")


@pytest.mark.parametrize("chunk", [2, 4])
def test_narr_chunked_inference_blends_every_canonical_cell(tmp_path, monkeypatch, chunk):
    from types import SimpleNamespace
    from granitewxc.temporal.config import parse_temporal_config
    from granitewxc.temporal.inference import run_sequence_inference

    raw = _write_case(tmp_path)
    raw["inference"] = {"inference_tile_size": [4, 4], "inference_overlap": [2, 2],
                        "inference_halo": [1, 1], "inference_blend_window": "hann"}
    source = NarrPrismDatasetFrameSource(raw, "validation")
    monkeypatch.setattr("granitewxc.temporal.sources.build_frame_source", lambda *args: source)
    config = SimpleNamespace(data=SimpleNamespace(**raw["data"]), model=SimpleNamespace(**raw["model"]))
    cfg = parse_temporal_config({"enabled": True, "backend": "native_pair",
        "context_length": 3, "warmup_length": 1, "output_length": 2,
        "native_pair": {"history_offsets": [1]}, "init_from_spatial_checkpoint": "unused.pt"})

    class GeometryRunner:
        """A deterministic geometry probe, not a trained scientific model."""
        adapter = SimpleNamespace(detach_state=lambda state: state)
        _allow_cold_start = False

        def eval(self):
            return self

        def __call__(self, batch, *, initial_state=None, emit_indices):
            top, left, height, width = batch["__output_crop"][0].tolist()
            prediction = batch["x"][:, emit_indices, :3, top:top + height, left:left + width]
            return SimpleNamespace(predictions=prediction, final_state=None,
                target_frames=batch["y"][:, emit_indices],
                valid_mask=batch["__target_valid_mask"][:, emit_indices])

    results = run_sequence_inference(GeometryRunner(), config, cfg, split="validation",
                                    chunk_length=chunk, verbose=False)
    assert len(results) == 1
    result = results[0]
    partial = run_sequence_inference(GeometryRunner(), config, cfg, split="validation",
        date_start="2000-01-03", date_end="2000-01-05", chunk_length=chunk, verbose=False)[0]
    assert partial.dates == result.dates[2:]
    np.testing.assert_array_equal(partial.predictions, result.predictions[2:])
    assert result.predictions.shape == (5, 3, 6, 6)
    assert len(result.dates) == 5
    field = np.arange(36, dtype=np.float32).reshape(6, 6)
    for t in range(5):
        expected = field + t
        expected = expected.copy()
        expected[0, 0] = 7  # persisted scaler mean at the invalid atmospheric cell
        np.testing.assert_allclose(result.predictions[t, 0], expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.predictions[t, 1], field + 270 + t)
        np.testing.assert_allclose(result.predictions[t, 2], field + 100)
        assert not result.valid_mask[t, 0, 1, 1]
        assert result.valid_mask[t, 1:].all()
    assert source._inference_active is False
    np.testing.assert_array_equal(result.lat, source._dataset.fine_lat)
    np.testing.assert_array_equal(result.lon, source._dataset.fine_lon)
    from granitewxc.temporal.inference import write_netcdf
    output = tmp_path / f"mosaic_{chunk}.nc"
    write_netcdf(result, output, output_vars=raw["data"]["output_vars"])
    with xr.open_dataset(output) as written:
        np.testing.assert_array_equal(written.lat.values, result.lat)
        np.testing.assert_array_equal(written.lon.values, result.lon)
