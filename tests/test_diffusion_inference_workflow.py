from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import xarray as xr

from examples.CORDEX_ML.utils.diffusion_inference import (
    add_predictions_to_dataset,
    infer_batch_ensemble,
    infer_head_type,
    resolve_ensemble_size,
    variable_array_map,
)


class _FakeModel(torch.nn.Module):
    diffusion_enabled = True


def _fake_infer(*, model, batch, cfg):
    del model, cfg
    b, _, h, w = batch["x"].shape
    sample = torch.randn(b, 2, h, w, device=batch["x"].device)
    return sample, sample + 1.0, sample + 2.0


def test_diffusion_checkpoint_metadata_wins_over_config_default():
    checkpoint = {"metadata": {"head_type": "diffusion"}}
    config = SimpleNamespace(model=SimpleNamespace())
    assert infer_head_type(checkpoint=checkpoint, config=config, model=None) == "diffusion"


def test_diffusion_checkpoint_state_keys_trigger_detection():
    checkpoint = {"model": {"diffusion_head.net.0.weight": torch.zeros(1)}}
    config = SimpleNamespace(model=SimpleNamespace())
    assert infer_head_type(checkpoint=checkpoint, config=config, model=None) == "diffusion"


def test_diffusion_ensemble_size_is_at_least_ten():
    config = SimpleNamespace(inference=SimpleNamespace(ensemble_size=3))
    assert resolve_ensemble_size(config, head_type="diffusion") == 10
    assert resolve_ensemble_size(config, head_type="deterministic") == 3


def test_diffusion_ensemble_uses_independent_seeded_samples():
    batch = {"x": torch.zeros(1, 1, 3, 4)}
    out, pre_inverse, raw = infer_batch_ensemble(
        model=_FakeModel(),
        batch=batch,
        infer_batch=_fake_infer,
        boundary_cfg=None,
        head_type="diffusion",
        ensemble_size=10,
        base_seed=123,
        device=torch.device("cpu"),
    )
    assert out.shape == (1, 10, 2, 3, 4)
    assert pre_inverse.shape == out.shape
    assert raw.shape == out.shape
    assert not torch.allclose(out[:, 0], out[:, 1])

    out2, _, _ = infer_batch_ensemble(
        model=_FakeModel(),
        batch=batch,
        infer_batch=_fake_infer,
        boundary_cfg=None,
        head_type="diffusion",
        ensemble_size=10,
        base_seed=123,
        device=torch.device("cpu"),
    )
    assert torch.allclose(out, out2)


def test_deterministic_path_keeps_previous_shape():
    batch = {"x": torch.zeros(2, 1, 3, 4)}
    out, _, _ = infer_batch_ensemble(
        model=torch.nn.Module(),
        batch=batch,
        infer_batch=_fake_infer,
        boundary_cfg=None,
        head_type="deterministic",
        ensemble_size=10,
        base_seed=123,
        device=torch.device("cpu"),
    )
    assert out.shape == (2, 2, 3, 4)


def test_diffusion_netcdf_dataset_has_ensemble_dimension():
    outputs = np.zeros((5, 10, 2, 3, 4), dtype=np.float32)
    coords = {
        "time": np.arange(5),
        "lat": np.arange(3),
        "lon": np.arange(4),
    }
    ds = xr.Dataset(coords=coords)
    ds = add_predictions_to_dataset(
        prediction_ds=ds,
        target_vars=["pr", "tasmax"],
        outputs_np=outputs,
        coords=coords,
        time_dim="time",
        lat_dim="lat",
        lon_dim="lon",
        target_attrs={"pr": {"units": "mm"}, "tasmax": {"units": "K"}},
        head_type="diffusion",
        ensemble_size=10,
        base_seed=42,
    )
    assert ds.sizes["ensemble"] == 10
    assert ds["pr"].dims == ("time", "ensemble", "lat", "lon")
    assert ds.attrs["head_type"] == "diffusion"
    assert ds.attrs["ensemble_generation"] == "diffusion_sampling"


def test_variable_array_map_accepts_ensemble_and_deterministic_arrays():
    deterministic = np.zeros((5, 2, 3, 4), dtype=np.float32)
    ensemble = np.zeros((5, 10, 2, 3, 4), dtype=np.float32)
    assert variable_array_map(["pr", "tasmax"], deterministic)["pr"].shape == (5, 3, 4)
    assert variable_array_map(["pr", "tasmax"], ensemble)["pr"].shape == (5, 10, 3, 4)
