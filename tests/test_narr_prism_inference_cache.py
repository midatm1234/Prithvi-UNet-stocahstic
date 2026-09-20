"""Scientific parity and invalidation tests for FP32 inference conditioning."""

import copy
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

NARR_DIR = Path(__file__).resolve().parents[1] / "examples" / "NARR_PRISM"
sys.path.insert(0, str(NARR_DIR))

import narr_prism_inference_cache as cache_module
import narr_prism_refinement_inference as inference
from granitewxc.utils.prism_grid import validate_prism_grid
from granitewxc.utils.prism_tiling import TilePlan
from test_refinement_models import build


@pytest.fixture
def setup_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(
        cache_module, "build_prism_checkpoint_contract",
        lambda config: {"normalization": "test-scaler-sha256"},
    )
    torch.manual_seed(71)
    plan = TilePlan.build((12, 12), (8, 8), overlap=(4, 4), halo=(2, 2))
    mask = np.ones((12, 12), dtype=bool)
    mask[0, 0] = False
    predictors = torch.randn(4, 12, 12)
    model = build("diffusion_unet").eval()
    lat = 32.55 + np.arange(12) / 120
    lon = -124.45 + np.arange(12) / 120
    cfg = {
        "case_name": "test_case",
        "data": {"target_variables": ["ppt", "tmax", "tmin"]},
        "model": {"refinement": {"type": "diffusion_unet"}},
        "dates": {"inference": {"start": "2016-01-01", "end": "2016-01-02"}},
    }
    kwargs = dict(
        cfg=cfg, config=object(), phase1_fingerprint="test-weights-sha256",
        plan=plan, positions=plan.positions, lat=lat, lon=lon,
        valid_mask=mask, mask_provenance={"mask": "signed-mask"},
        split="inference", batch_size=2, pad_multiple=1, blend_mode="hann",
    )
    contract = cache_module.inference_cache_contract(**kwargs)
    cache = cache_module.FP32Phase1InferenceCache(tmp_path / "cache", contract)

    def batch_factory(start, stop):
        return inference._make_tile_batch(
            predictors, plan.positions[start:stop], plan, n_targets=3,
            pad_multiple=1, device=torch.device("cpu"), domain_valid_mask=mask,
        )

    return SimpleNamespace(
        cache=cache, contract=contract, contract_kwargs=kwargs,
        batch_factory=batch_factory, model=model, predictors=predictors,
        plan=plan, mask=mask, cfg=cfg, lat=lat, lon=lon,
    )


def get_day(setup):
    return setup.cache.load_or_build(
        "2016-01-01", setup.predictors, setup.model, setup.batch_factory, "cpu",
    )


@pytest.mark.parametrize("refiner_type", [
    "diffusion_unet", "flow_matching_unet", "diffusion_transformer",
])
def test_fp32_cache_matches_live_refinement_and_skips_phase1(
    setup_cache, monkeypatch, refiner_type,
):
    s = setup_cache
    s.model = build(refiner_type).eval()
    calls = []
    original = s.model.run_phase1

    def checked(batch):
        assert not torch.is_autocast_enabled("cpu")
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert torch.get_float32_matmul_precision() == "highest"
        calls.append(1)
        return original(batch)

    monkeypatch.setattr(s.model, "run_phase1", checked)
    # Explicitly test that an enclosing mixed-precision context is overridden.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        payload = get_day(s)
    assert len(calls) == 3  # Two build batches plus one live validation batch.
    assert payload["normalized"].dtype == torch.float32

    batch = s.batch_factory(0, 2)
    with cache_module.strict_phase1_fp32("cpu"):
        s.model.initialize_from_batch(batch)
        expected = inference._predict_tile_ensemble(
            s.model, batch, positions=s.plan.positions[:2],
            domain_shape=s.plan.domain_shape, sample_date="2016-01-01",
            ensemble_size=2, base_seed=123,
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("Validated cached inference must bypass Phase 1")

    monkeypatch.setattr(s.model, "run_phase1", forbidden)
    payload = get_day(s)
    s.cache.attach(batch, payload, 0, 2, "cpu")
    with torch.inference_mode():
        s.model.initialize_from_batch(batch)
        actual = inference._predict_tile_ensemble(
            s.model, batch, positions=s.plan.positions[:2],
            domain_shape=s.plan.domain_shape, sample_date="2016-01-01",
            ensemble_size=2, base_seed=123,
        )
    assert torch.equal(expected[0], actual[0])
    for expected_member, actual_member in zip(expected[1], actual[1]):
        torch.testing.assert_close(actual_member, expected_member, rtol=0, atol=1e-6)


@pytest.mark.parametrize("change", ["checksum", "dtype", "shape", "date", "predictors"])
def test_invalid_cache_is_rejected(setup_cache, change):
    s = setup_cache
    payload = get_day(s)
    path = next(s.cache.directory.glob("*.pt"))
    if change == "checksum":
        payload["normalized"][0, 0, 0, 0] += 1
    elif change == "dtype":
        payload["normalized"] = payload["normalized"].bfloat16()
    elif change == "shape":
        payload["physical"] = payload["physical"][:1]
    elif change == "date":
        payload["date"] = "2016-01-02"
    else:
        s.predictors[0, 0, 0] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Phase-1"):
        get_day(s)


def test_new_reader_checks_live_fp32_parity(setup_cache, monkeypatch):
    s = setup_cache
    get_day(s)
    s.cache = cache_module.FP32Phase1InferenceCache(s.cache.directory.parent, s.contract)
    original = s.model.run_phase1

    def changed(batch):
        physical, normalized, features = original(batch)
        return physical + 0.01, normalized, features

    monkeypatch.setattr(s.model, "run_phase1", changed)
    with pytest.raises(ValueError, match="failed live FP32 parity"):
        get_day(s)


def test_interrupted_write_does_not_publish_partial_cache(setup_cache, monkeypatch):
    s = setup_cache

    def interrupted(payload, path):
        path.write_bytes(b"partial")
        raise OSError("write interrupted")

    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(OSError, match="write interrupted"):
        get_day(s)
    assert not list(s.cache.directory.glob("*.pt"))
    assert not list(s.cache.directory.glob("*.tmp"))


@pytest.mark.parametrize("field", ["phase1_fingerprint", "lat", "valid_mask", "positions"])
def test_contract_invalidates_scientific_changes(setup_cache, field):
    s = setup_cache
    kwargs = copy.deepcopy(s.contract_kwargs)
    if field == "phase1_fingerprint":
        kwargs[field] = "different-checkpoint"
    elif field == "lat":
        kwargs[field][0] += 0.001
    elif field == "valid_mask":
        kwargs[field][1, 1] = False
    else:
        kwargs[field] = kwargs[field][::-1]
    updated = cache_module.inference_cache_contract(**kwargs)
    other = cache_module.FP32Phase1InferenceCache(s.cache.directory.parent, updated)
    assert other.directory != s.cache.directory


def test_cache_is_shared_across_refinement_heads_but_not_scalers(setup_cache, monkeypatch):
    s = setup_cache
    kwargs = copy.deepcopy(s.contract_kwargs)
    kwargs["cfg"]["model"]["refinement"]["type"] = "flow_matching_unet"
    assert cache_module.inference_cache_contract(**kwargs) == s.contract
    monkeypatch.setattr(
        cache_module, "build_prism_checkpoint_contract",
        lambda config: {"normalization": "different-scaler-sha256"},
    )
    assert cache_module.inference_cache_contract(**kwargs) != s.contract


def test_strict_precision_policy_restores_state_after_exception(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    before = (
        torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic, torch.are_deterministic_algorithms_enabled(),
    )
    with pytest.raises(RuntimeError, match="probe"):
        with cache_module.strict_phase1_fp32("cpu"):
            raise RuntimeError("probe")
    after = (
        torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic, torch.are_deterministic_algorithms_enabled(),
    )
    assert after == before


def test_daily_inference_reuses_cache_and_preserves_grid_mask_outputs(
    setup_cache, tmp_path, monkeypatch,
):
    s = setup_cache
    dates = [date(2016, 1, 1), date(2016, 1, 2)]

    class Dataset:
        def __init__(self, config_path, mode, load_observed_targets):
            assert mode == "inference"
            assert load_observed_targets is False
            self.dates = dates
            self.fine_lat, self.fine_lon = s.lat, s.lon
            self.fine_shape = s.plan.domain_shape

        def _load_predictor_day(self, sample_date):
            return s.predictors + dates.index(sample_date)

    provenance = {
        inference.TARGET_VALID_MASK_SHA256_ATTR: "a" * 64,
        inference.TARGET_VALID_MASK_CONTENT_SHA256_ATTR: inference._target_valid_mask_content_sha256(s.mask),
        inference.TARGET_VALID_MASK_CRITERION_ATTR: inference.normalization_contract.TARGET_VALID_MASK_CRITERION,
        inference.TARGET_VALID_MASK_SOURCE_SPLIT_ATTR: "training",
        inference.TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR: "b" * 64,
        inference.TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR: validate_prism_grid(
            s.lat, s.lon, context="inference-cache-test",
        ).fingerprint,
    }
    monkeypatch.setattr(inference, "NarrPrismDataset", Dataset)
    monkeypatch.setattr(inference, "training_scaler_valid_mask", lambda *args: s.mask)
    monkeypatch.setattr(inference, "_target_valid_mask_provenance", lambda *args: provenance)
    monkeypatch.setattr(inference, "_inference_plan", lambda *args: s.plan)
    monkeypatch.setattr(inference, "_pad_multiple_from_config", lambda *args: 1)
    monkeypatch.setattr(inference, "_load_phase2", lambda *args: ("phase2-digest", {}))
    config = SimpleNamespace(data=SimpleNamespace(output_vars=["ppt", "tmax", "tmin"]))
    calls = []
    original = s.model.run_phase1

    def count(batch):
        calls.append(1)
        return original(batch)

    monkeypatch.setattr(s.model, "run_phase1", count)
    arguments = dict(
        config_path="unused.yaml", cfg=s.cfg, config=config, model=s.model,
        phase1_checkpoint="phase1.ckpt", phase1_fingerprint="test-weights-sha256",
        refinement_checkpoint="phase2.ckpt", device=torch.device("cpu"),
        ensemble_size=2, base_seed=123, batch_size=2, show_progress=False,
        phase1_cache_dir=str(tmp_path / "shared"),
    )
    first = inference.run_refined_inference(**arguments, output_dir=str(tmp_path / "first"))
    assert len(calls) == 5
    calls.clear()
    second = inference.run_refined_inference(**arguments, output_dir=str(tmp_path / "second"))
    assert len(calls) == 1  # One parity sample, no day/tile recomputation.
    for filename in first.glob("*.nc"):
        with xr.open_dataset(filename) as a, xr.open_dataset(second / filename.name) as b:
            for variable in a.data_vars:
                np.testing.assert_array_equal(a[variable].values, b[variable].values)
            np.testing.assert_array_equal(b.lat, s.lat)
            np.testing.assert_array_equal(b.lon, s.lon)
            np.testing.assert_array_equal(b.prism_valid_mask, s.mask)
            assert b.attrs["phase1_compute_dtype"] == "float32"
            assert b.attrs["phase1_tf32"] == "disabled"
            assert b.attrs["phase1_cache_digest"] == a.attrs["phase1_cache_digest"]
            assert np.isnan(b.ppt.values[0, 0, 0])
    calls.clear()
    inference.run_refined_inference(
        **arguments, output_dir=str(tmp_path / "second"), resume_existing=True,
    )
    assert calls == []  # Valid completed products need no model execution.

    # A product from the previous uncached path must not silently resume.
    path = next(second.glob("*.nc"))
    with xr.open_dataset(path) as ds:
        legacy = ds.load()
    legacy.attrs.pop("phase1_cache_digest")
    legacy.to_netcdf(path, engine="h5netcdf")
    inference.run_refined_inference(
        **arguments, output_dir=str(tmp_path / "second"), resume_existing=True,
    )
    assert len(calls) == 1  # Cached tiles plus the single live validation batch.
