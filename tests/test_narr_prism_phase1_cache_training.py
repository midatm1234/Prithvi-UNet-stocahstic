"""Focused contracts for reusable NARR--PRISM Phase-1 cache training."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

NARR_DIR = Path(__file__).resolve().parents[1] / "examples" / "NARR_PRISM"
if str(NARR_DIR) not in sys.path:
    sys.path.insert(0, str(NARR_DIR))

import narr_prism_phase1_cache as phase1_cache
import narr_prism_refinement as refinement_cli


def _manifest(tmp_path: Path) -> dict:
    return {
        "_manifest_path": str(tmp_path / "manifest.json"),
        "contract_digest": "test-contract",
        "contract": {
            "cache_geometry": {"domain_shape": [32, 32]},
            "target_variables": ["pr", "tasmax", "tasmin"],
        },
        "residual_normalization": {
            "enabled": True,
            "fitted": True,
            "mean": [0.1, -0.2, 0.3],
            "std": [1.1, 1.2, 1.3],
            "count": [100, 101, 102],
            "epsilon": 1.0e-6,
            "target_variables": ["pr", "tasmax", "tasmin"],
            "space": phase1_cache.TARGET_SPACE_NAME,
            "residual_definition": phase1_cache.RESIDUAL_DEFINITION,
            "fit_split": "training",
        },
    }


def test_daily_reader_opens_once_for_all_same_day_tiles(tmp_path, monkeypatch):
    opens = []
    arrays = {
        "deterministic_normalized": np.zeros((3, 32, 32), dtype=np.float32),
        "residual_target_normalized": np.ones((3, 32, 32), dtype=np.float32),
        "residual_valid_mask": np.ones((3, 32, 32), dtype=np.uint8),
        "deterministic_physical": np.full((3, 32, 32), 2.0, dtype=np.float32),
    }

    class FakeDataset:
        def __init__(self, path, _mode):
            opens.append(str(path))
            self.variables = arrays

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(phase1_cache, "NetCDFDataset", FakeDataset)
    reader = phase1_cache.Phase1ResidualCacheReader(
        _manifest(tmp_path), validate_daily=False
    )
    for tile in range(22):
        crop = reader.load_crop(
            "training", "2000-01-01", tile % 16, tile % 16, 8, 8
        )
        assert set(crop) == {
            "__phase1_normalized",
            "__residual_target_normalized",
            "__residual_valid_mask",
        }
        assert crop["__phase1_normalized"].shape == (3, 8, 8)
        assert crop["__residual_valid_mask"].dtype == torch.bool
    assert len(opens) == 1

    reader.load_crop("training", "2000-01-02", 0, 0, 8, 8)
    assert len(opens) == 2


def test_cache_statistics_install_fresh_and_match_resume(tmp_path):
    manifest = _manifest(tmp_path)

    class FakeRefiner:
        def __init__(self):
            self.metadata = {
                "enabled": True,
                "fitted": False,
                "mean": [0.0] * 3,
                "std": [1.0] * 3,
                "count": [0] * 3,
            }

        def set_residual_normalization(self, mean, std, count):
            self.metadata = {
                "enabled": True,
                "fitted": True,
                "mean": torch.as_tensor(mean).tolist(),
                "std": torch.as_tensor(std).tolist(),
                "count": torch.as_tensor(count).tolist(),
            }

        def residual_normalization_metadata(self):
            return self.metadata

    model = SimpleNamespace(
        target_space=SimpleNamespace(num_channels=3),
        refinement_config=SimpleNamespace(
            residual_normalization=SimpleNamespace(
                enabled=True, epsilon=1.0e-6
            )
        ),
        refiner=FakeRefiner(),
    )
    metadata = refinement_cli._validated_cache_statistics(model, manifest)
    refinement_cli._install_cache_statistics(model, metadata)
    refinement_cli._assert_cache_statistics_match_resume(
        model, metadata, tolerance=1.0e-6
    )

    model.refiner.metadata["mean"][0] += 0.01
    with pytest.raises(RuntimeError, match="residual mean differs"):
        refinement_cli._assert_cache_statistics_match_resume(
            model, metadata, tolerance=1.0e-6
        )


def test_cache_statistics_reject_wrong_residual_sign(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["residual_normalization"]["residual_definition"] = (
        "encode(Phase1)-encode(PRISM)"
    )
    model = SimpleNamespace(
        target_space=SimpleNamespace(num_channels=3),
        refinement_config=SimpleNamespace(
            residual_normalization=SimpleNamespace(
                enabled=True, epsilon=1.0e-6
            )
        ),
    )
    with pytest.raises(RuntimeError, match="residual sign"):
        refinement_cli._validated_cache_statistics(model, manifest)


def test_cache_statistics_reject_epsilon_mismatch(tmp_path):
    manifest = _manifest(tmp_path)
    model = SimpleNamespace(
        target_space=SimpleNamespace(num_channels=3),
        refinement_config=SimpleNamespace(
            residual_normalization=SimpleNamespace(
                enabled=True, epsilon=1.0e-5
            )
        ),
    )
    with pytest.raises(RuntimeError, match="epsilon does not match"):
        refinement_cli._validated_cache_statistics(model, manifest)


def test_cache_loader_fails_closed_with_builder_command(tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        raise FileNotFoundError("manifest absent")

    monkeypatch.setattr(
        phase1_cache.Phase1ResidualCacheReader, "from_path", fail
    )
    performance = SimpleNamespace(
        phase1_cache=SimpleNamespace(enabled=True, path=str(tmp_path))
    )
    with pytest.raises(
        RuntimeError, match="missing, incomplete.*phase1_cache.py build"
    ):
        refinement_cli._load_phase1_cache(
            config=SimpleNamespace(),
            raw_config={"_config_path": "refinement.yaml"},
            phase1_checkpoint="phase1.ckpt",
            phase1_fingerprint="state-sha256",
            performance=performance,
        )


class _ParityModel:
    def __init__(self, baseline):
        self.baseline = baseline
        self.training = True
        self.target_space = SimpleNamespace(
            residual_target=self._residual_target
        )

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def run_phase1(self, _batch):
        physical = self.baseline
        return physical, self.baseline, {}

    @staticmethod
    def _residual_target(target, baseline, scaler_offset=None):
        del scaler_offset
        valid = torch.isfinite(target)
        residual = torch.where(
            valid, target - baseline, torch.zeros_like(target)
        )
        return residual, valid


def _parity_batch(cached_offset=0.0):
    baseline = torch.full((1, 3, 4, 4), 0.25)
    target = torch.full_like(baseline, 0.75)
    return baseline, {
        "x": torch.zeros(1, 1, 4, 4),
        "y": target,
        "__phase1_normalized": baseline + cached_offset,
        "__residual_target_normalized": target - baseline,
        "__residual_valid_mask": torch.ones_like(target, dtype=torch.bool),
    }


def test_live_cache_parity_passes_and_restores_training_mode():
    baseline, batch = _parity_batch()
    model = _ParityModel(baseline)
    stats = refinement_cli._validate_phase1_cache_parity(
        model, [batch], device=torch.device("cpu"), samples=1, tolerance=1.0e-6
    )
    assert stats["samples"] == 1
    assert stats["baseline_max_abs"] == 0.0
    assert model.training is True


def test_live_cache_parity_rejects_stale_baseline():
    baseline, batch = _parity_batch(cached_offset=0.01)
    model = _ParityModel(baseline)
    with pytest.raises(RuntimeError, match="parity validation failed"):
        refinement_cli._validate_phase1_cache_parity(
            model,
            [batch],
            device=torch.device("cpu"),
            samples=1,
            tolerance=1.0e-6,
        )


def test_phase1_cache_runtime_honors_yaml_tf32_false():
    prior_matmul_precision = torch.get_float32_matmul_precision()
    prior_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    prior_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    performance = SimpleNamespace(
        phase1_cache=SimpleNamespace(enabled=True),
        precision=SimpleNamespace(mode="fp32", allow_tf32=False),
    )
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        policy = refinement_cli._configure_phase1_cache_runtime(
            performance, role="test"
        )
        assert policy["mode"] == "fp32"
        assert policy["allow_tf32"] is False
        assert policy["float32_matmul_precision"] == "highest"
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    finally:
        torch.set_float32_matmul_precision(prior_matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = prior_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prior_cudnn_tf32


def test_phase1_cache_runtime_rejects_incompatible_precision():
    performance = SimpleNamespace(
        phase1_cache=SimpleNamespace(enabled=True),
        precision=SimpleNamespace(mode="fp32", allow_tf32=True),
    )
    with pytest.raises(RuntimeError, match="strict FP32 with TF32 disabled"):
        refinement_cli._configure_phase1_cache_runtime(
            performance, role="test"
        )
