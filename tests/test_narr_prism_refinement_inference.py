"""Focused guards and output-contract tests for daily refined inference."""

from __future__ import annotations

import random
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

NARR_DIR = Path(__file__).resolve().parents[1] / "examples" / "NARR_PRISM"
if str(NARR_DIR) not in sys.path:
    sys.path.insert(0, str(NARR_DIR))

import narr_prism_refinement as refinement_cli
from compute_scalars_narr_prism import _derive_joint_target_valid_mask
from narr_prism_dataset import NarrPrismDataset
from narr_prism_inference import (
    TARGET_VALID_MASK_CONTENT_SHA256_ATTR,
    TARGET_VALID_MASK_CRITERION_ATTR,
    TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR,
    TARGET_VALID_MASK_SHA256_ATTR,
    TARGET_VALID_MASK_SOURCE_SPLIT_ATTR,
    TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR,
    _target_valid_mask_content_sha256,
)
from narr_prism_refinement import (
    _refinement_checkpoint,
    _require_checkpoint_path,
    _seed_training,
)
from narr_prism_refinement_inference import (
    CoordinateAlignedNoiseSource,
    _validate_dataset_dates,
    build_daily_refined_dataset,
    derive_tile_seed,
    training_scaler_valid_mask,
    validate_refinement_checkpoint_metadata,
)

from granitewxc.refinement.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    REFINEMENT_SEMANTICS_VERSION,
)
from granitewxc.refinement.config import resolve_refinement_config
from granitewxc.utils import normalization
from granitewxc.utils.prism_grid import validate_prism_grid


def test_active_command_checkpoint_guard_rejects_missing_artifact(
    tmp_path,
) -> None:
    with pytest.raises(FileNotFoundError, match="Phase-1.*required"):
        _require_checkpoint_path(None, label="Phase-1 deterministic")
    with pytest.raises(FileNotFoundError, match="not a regular file"):
        _require_checkpoint_path(
            str(tmp_path / "missing.ckpt"), label="Phase-2 refinement"
        )


def test_checkpoint_guard_accepts_file_and_yaml_phase2_fallback(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "phase2.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    config = SimpleNamespace(
        model=SimpleNamespace(refinement={"checkpoint": str(checkpoint)})
    )

    assert _refinement_checkpoint(config, None) == str(checkpoint)
    assert _require_checkpoint_path(
        str(checkpoint), label="Phase-2 refinement"
    ) == str(checkpoint.resolve())


def test_checkpoint_guard_reports_recursive_symlink(tmp_path) -> None:
    recursive = tmp_path / "loop.ckpt"
    recursive.symlink_to(recursive)
    with pytest.raises(FileNotFoundError, match="not accessible"):
        _require_checkpoint_path(str(recursive), label="Phase-1 deterministic")


def test_phase2_training_seed_covers_python_numpy_and_torch() -> None:
    def sample(seed: int) -> tuple[float, float, torch.Tensor]:
        _seed_training(seed)
        return random.random(), float(np.random.random()), torch.rand(4)

    first = sample(173)
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    second = sample(173)

    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


def test_training_command_seeds_before_model_construction(monkeypatch) -> None:
    class ModelConstructionReached(Exception):
        pass

    samples = []

    def stop_at_model_construction(*_args, **_kwargs):
        samples.append(
            (random.random(), float(np.random.random()), torch.rand(4))
        )
        raise ModelConstructionReached

    refinement = SimpleNamespace(is_active=True, seed=2718)
    monkeypatch.setattr(refinement_cli, "get_config", lambda _path: object())
    monkeypatch.setattr(
        refinement_cli, "resolve_refinement_config", lambda _config: refinement
    )
    monkeypatch.setattr(
        refinement_cli, "_phase1_checkpoint", lambda *_args: "/phase1.ckpt"
    )
    monkeypatch.setattr(
        refinement_cli, "_require_checkpoint_path", lambda path, **_kwargs: path
    )
    monkeypatch.setattr(
        refinement_cli, "build_model", stop_at_model_construction
    )
    args = SimpleNamespace(
        config="fixture.yaml", device="cpu", phase1_checkpoint=None
    )

    for perturbation in (11, 29):
        random.seed(perturbation)
        np.random.seed(perturbation)
        torch.manual_seed(perturbation)
        with pytest.raises(ModelConstructionReached):
            refinement_cli.cmd_train(args)

    assert samples[0][:2] == samples[1][:2]
    assert torch.equal(samples[0][2], samples[1][2])


class _RefinementConfig:
    train_on_residual = True

    def __init__(self):
        self._resolved = resolve_refinement_config(
            {
                "refinement": {
                    "type": "diffusion_unet",
                    "diffusion": {
                        "prediction_type": "epsilon",
                        "inference_steps": 8,
                    },
                }
            }
        )

    def to_dict(self):
        return self._resolved.to_dict()


def test_phase2_semantic_guard_precedes_state_loading() -> None:
    model = SimpleNamespace(refinement_config=_RefinementConfig())
    # Schema 2 carries the complete resolved refinement contract.
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "refinement_semantics_version": REFINEMENT_SEMANTICS_VERSION,
        "checkpoint_kind": "refinement",
        "resolved_config": {
            "refinement": model.refinement_config.to_dict()
        },
    }
    validate_refinement_checkpoint_metadata(payload, model)

    incompatible = {
        **payload,
        "resolved_config": {
            "refinement": {
                **payload["resolved_config"]["refinement"],
                "type": "flow_matching_unet",
            }
        },
    }
    with pytest.raises(ValueError, match="configuration mismatch"):
        validate_refinement_checkpoint_metadata(incompatible, model)
    with pytest.raises(ValueError, match="no resolved_config"):
        validate_refinement_checkpoint_metadata(
            {
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "refinement_semantics_version": REFINEMENT_SEMANTICS_VERSION,
                "checkpoint_kind": "refinement",
            },
            model,
        )
    with pytest.raises(
        ValueError, match="missing refinement.train_on_residual"
    ):
        validate_refinement_checkpoint_metadata(
            {
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "refinement_semantics_version": REFINEMENT_SEMANTICS_VERSION,
                "checkpoint_kind": "refinement",
                "resolved_config": {"refinement": {"type": "diffusion_unet"}},
            },
            model,
        )
    with pytest.raises(ValueError, match="Combined Phase-1/Phase-2"):
        validate_refinement_checkpoint_metadata(
            {
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "refinement_semantics_version": REFINEMENT_SEMANTICS_VERSION,
                "checkpoint_kind": "combined",
                "resolved_config": {
                    "refinement": model.refinement_config.to_dict()
                },
            },
            model,
        )


def test_phase2_semantic_guard_ignores_only_mutable_fields() -> None:
    model = SimpleNamespace(refinement_config=_RefinementConfig())
    full = model.refinement_config.to_dict()
    full.update({"checkpoint": "/old/path.ckpt", "ensemble_size": 100})
    validate_refinement_checkpoint_metadata(
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "refinement_semantics_version": REFINEMENT_SEMANTICS_VERSION,
            "checkpoint_kind": "refinement",
            "resolved_config": {"refinement": full},
        },
        model,
    )
    invalid = {**full, "diffusion": {"prediction_type": "sample"}}
    with pytest.raises(ValueError, match="diffusion.prediction_type"):
        validate_refinement_checkpoint_metadata(
            {
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "refinement_semantics_version": REFINEMENT_SEMANTICS_VERSION,
                "checkpoint_kind": "refinement",
                "resolved_config": {"refinement": invalid},
            },
            model,
        )


def test_partial_schema2_metadata_cannot_hide_current_nondefault_change():
    class ChangedConfig(_RefinementConfig):
        def __init__(self):
            self._resolved = resolve_refinement_config(
                {
                    "refinement": {
                        "type": "diffusion_unet",
                        "correction_scale": 0.0,
                        "diffusion": {
                            "prediction_type": "epsilon",
                            "inference_steps": 8,
                        },
                    }
                }
            )

    legacy = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "refinement_semantics_version": REFINEMENT_SEMANTICS_VERSION,
        "checkpoint_kind": "refinement",
        "resolved_config": {
            "refinement": {
                "type": "diffusion_unet",
                "train_on_residual": True,
                "diffusion": {
                    "prediction_type": "epsilon",
                    "inference_steps": 8,
                },
            }
        },
    }
    with pytest.raises(ValueError, match="correction_scale"):
        validate_refinement_checkpoint_metadata(
            legacy, SimpleNamespace(refinement_config=ChangedConfig())
        )


def test_schema1_phase2_checkpoint_is_rejected_before_state_loading() -> None:
    payload = {
        "checkpoint_schema_version": 1,
        "refinement_semantics_version": "legacy-hurdle-amount-latent-v1",
        "checkpoint_kind": "refinement",
        "resolved_config": {
            "refinement": _RefinementConfig().to_dict()
        },
    }
    with pytest.raises(ValueError, match="incompatible residual semantics"):
        validate_refinement_checkpoint_metadata(
            payload,
            SimpleNamespace(refinement_config=_RefinementConfig()),
        )


def test_date_tile_member_seed_is_stable_and_distinct() -> None:
    seed = derive_tile_seed(1234, "2016-01-01", (0, 0), 0)
    assert seed == derive_tile_seed(1234, "2016-01-01", (0, 0), 0)
    assert seed != derive_tile_seed(1234, "2016-01-02", (0, 0), 0)
    assert seed != derive_tile_seed(1234, "2016-01-01", (32, 0), 0)
    assert seed != derive_tile_seed(1234, "2016-01-01", (0, 0), 1)


def test_inference_dates_must_exactly_match_inclusive_yaml_range() -> None:
    cfg = {
        "dates": {"inference": {"start": "2016-02-28", "end": "2016-03-01"}}
    }
    expected = [date(2016, 2, 28), date(2016, 2, 29), date(2016, 3, 1)]
    assert _validate_dataset_dates(cfg, expected) == expected
    with pytest.raises(ValueError, match="do not exactly match"):
        _validate_dataset_dates(cfg, [expected[0], expected[2]])


def _mask_config(tmp_path):
    return SimpleNamespace(
        case_name="mask_case",
        data=SimpleNamespace(
            type="narr_prism",
            preprocessed_dir=str(tmp_path / "preprocessed"),
            scalar_dir=str(tmp_path / "legacy"),
            output_vars=["ppt", "tmax", "tmin"],
            target_variables=["ppt", "tmax", "tmin"],
        ),
        model=SimpleNamespace(),
        dates={
            "training": {"start": "1996-01-01", "end": "2013-12-31"}
        },
        normalization=SimpleNamespace(predictor_mode="global"),
        predictands={},
    )


def _write_mask_contract(config, expected):
    scalar_dir = normalization.resolve_scalar_dir(config, for_writing=True)
    scalar_dir.mkdir(parents=True, exist_ok=True)
    np.save(scalar_dir / "inputs_mean.npy", np.zeros(1, dtype=np.float32))
    np.save(scalar_dir / "inputs_std.npy", np.ones(1, dtype=np.float32))
    np.save(scalar_dir / "targets_mean.npy", np.zeros(3, dtype=np.float32))
    np.save(scalar_dir / "targets_std.npy", np.ones(3, dtype=np.float32))
    mask_path = normalization.target_valid_mask_path(scalar_dir)
    np.save(mask_path, expected)
    entry = normalization.build_target_valid_mask_manifest_entry(
        mask_path, cfg=config
    )
    normalization.write_manifest(
        scalar_dir,
        case_name=config.case_name,
        predictor_mode="global",
        cfg=config,
        train_date_range=["1996-01-01", "2013-12-31"],
        extra={normalization.TARGET_VALID_MASK_MANIFEST_KEY: entry},
    )
    return mask_path


def test_production_mask_comes_from_signed_training_artifact_not_scalers(
    tmp_path,
) -> None:
    expected = np.ones((3, 4), dtype=bool)
    expected[0, 0] = False
    config = _mask_config(tmp_path)
    _write_mask_contract(config, expected)
    mask = training_scaler_valid_mask(config, (3, 4))
    np.testing.assert_array_equal(mask, expected)
    with pytest.raises(ValueError, match="requested output grid"):
        training_scaler_valid_mask(config, (4, 4))


def test_production_mask_rejects_legacy_scaler_only_contract(tmp_path) -> None:
    config = _mask_config(tmp_path)
    with pytest.raises(FileNotFoundError, match="training-derived target support"):
        training_scaler_valid_mask(config, (3, 4))


def test_joint_support_intersects_all_training_target_channels() -> None:
    counts = np.ones((3, 2, 3), dtype=np.int64)
    counts[0, 0, 0] = 0
    counts[1, 0, 1] = 0
    counts[2, 1, 2] = 0
    observed = _derive_joint_target_valid_mask(
        counts, expected_grid_shape=(2, 3)
    )
    expected = np.array(
        [[False, False, True], [True, True, False]], dtype=bool
    )
    np.testing.assert_array_equal(observed, expected)


def test_predictor_only_inference_sample_never_loads_prism_targets() -> None:
    """The production dataset path must remain usable without observations."""
    dataset = object.__new__(NarrPrismDataset)
    dataset._tile_slices = None
    dataset._dates = [date(2016, 1, 1)]
    dataset.target_vars = ["ppt", "tmax", "tmin"]
    dataset._load_observed_targets = False
    dataset.fine_shape = (3, 4)
    dataset.crop_size = dataset.fine_shape
    dataset.random_crop = False
    dataset.min_valid_target_fraction = 0.0
    dataset.training_halo = (0, 0)
    dataset.dtype = torch.float32

    def forbidden_target_load(*_args, **_kwargs):
        raise AssertionError("PRISM targets must not be loaded for inference")

    dataset._load_targets = forbidden_target_load
    dataset._load_predictor = lambda *_args, **_kwargs: torch.ones(
        2, 3, 4, dtype=torch.float32
    )

    sample = dataset[0]
    assert sample["y"].shape == (3, 3, 4)
    assert torch.count_nonzero(sample["y"]) == 0
    assert sample["date"] == "2016-01-01"


def _noise(origin, *, sample_date="2016-01-01", member=0):
    source = CoordinateAlignedNoiseSource(
        base_seed=1234,
        sample_date=sample_date,
        member=member,
        origins=[origin],
        domain_shape=(6, 6),
    )
    return source.randn((1, 2, 4, 4), torch.device("cpu"), torch.float32)


def test_stochastic_noise_is_coordinate_aligned_across_tile_overlaps() -> None:
    top_left = _noise((0, 0))
    bottom_right = _noise((2, 2))

    torch.testing.assert_close(
        top_left[..., 2:, 2:], bottom_right[..., :2, :2:]
    )
    assert not torch.equal(top_left, _noise((0, 0), sample_date="2016-01-02"))
    assert not torch.equal(top_left, _noise((0, 0), member=1))


def test_daily_output_uses_refined_primary_names_and_provenance() -> None:
    variables = ["ppt", "tmax", "tmin"]
    lat = np.array([41.0, 40.5, 40.0], dtype=np.float64)
    lon = np.array([-124.5, -124.0, -123.5, -123.0], dtype=np.float64)
    phase1 = np.zeros((3, 3, 4), dtype=np.float32)
    members = np.stack([phase1 + 1.0, phase1 + 3.0])
    valid = np.ones((3, 4), dtype=bool)
    valid[0, 0] = False
    attrs = {
        "phase1_fingerprint": "phase1-sha256",
        "phase2_fingerprint": "phase2-sha256",
        "checkpoint": "/checkpoints/phase2.ckpt",
        "phase1_checkpoint": "/checkpoints/phase1.ckpt",
        "phase2_checkpoint": "/checkpoints/phase2.ckpt",
        "config_path": "/configs/refinement.yaml",
        "config_fingerprint": "config-sha256",
        "variable_order": '["ppt", "tmax", "tmin"]',
        TARGET_VALID_MASK_SHA256_ATTR: "a" * 64,
        TARGET_VALID_MASK_CONTENT_SHA256_ATTR: (
            _target_valid_mask_content_sha256(valid)
        ),
        TARGET_VALID_MASK_CRITERION_ATTR: (
            normalization.TARGET_VALID_MASK_CRITERION
        ),
        TARGET_VALID_MASK_SOURCE_SPLIT_ATTR: "training",
        TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR: "b" * 64,
        TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR: validate_prism_grid(
            lat, lon, context="refined-output-test"
        ).fingerprint,
    }

    dataset = build_daily_refined_dataset(
        variables=variables,
        sample_date="2016-02-29",
        lat=lat,
        lon=lon,
        valid_mask=valid,
        phase1=phase1,
        members=members,
        attrs=attrs,
    )

    expected = {"prism_valid_mask"}
    for variable in variables:
        expected.update(
            {
                variable,
                f"{variable}_phase1",
                f"{variable}_member_000",
                f"{variable}_member_001",
                f"{variable}_ensemble_spread",
            }
        )
    assert set(dataset.data_vars) == expected
    np.testing.assert_allclose(dataset["ppt"].values[:, 1:, :], 2.0)
    assert np.isnan(dataset["ppt"].values[0, 0, 0])
    assert dataset["ppt"].attrs["units"] == "mm/day"
    assert dataset["tmax"].attrs["units"] == "degC"
    assert dataset.attrs["phase1_fingerprint"] == "phase1-sha256"
    assert dataset.attrs["phase2_fingerprint"] == "phase2-sha256"
    assert dataset.attrs["checkpoint"] == "/checkpoints/phase2.ckpt"
    assert dataset.attrs["phase1_checkpoint"] == "/checkpoints/phase1.ckpt"
    assert dataset.attrs["phase2_checkpoint"] == "/checkpoints/phase2.ckpt"
    assert dataset.attrs["config_fingerprint"] == "config-sha256"
    assert dataset.attrs[TARGET_VALID_MASK_SHA256_ATTR] == "a" * 64
    assert dataset.attrs[TARGET_VALID_MASK_CONTENT_SHA256_ATTR] == (
        _target_valid_mask_content_sha256(valid)
    )
    assert dataset.attrs["prism_grid_fingerprint"]
    np.testing.assert_array_equal(dataset.lat.values, lat)
    np.testing.assert_array_equal(dataset.lon.values, lon)
    assert str(dataset.time.values[0])[:10] == "2016-02-29"
