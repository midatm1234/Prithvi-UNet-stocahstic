from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from granitewxc.utils import normalization
from granitewxc.utils.prism_preprocessed import split_source_artifact_signature


def _config(
    case_name: str,
    preprocessed_dir: Path,
    legacy_scalar_dir: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        case_name=case_name,
        data=SimpleNamespace(
            preprocessed_dir=str(preprocessed_dir),
            scalar_dir=str(legacy_scalar_dir),
            output_vars=["tmax", "tmin"],
        ),
        model=SimpleNamespace(),
        normalization=SimpleNamespace(predictor_mode="global"),
        predictands={
            "tmax": {"normalization": {"mode": "gridpoint"}},
            "tmin": {"normalization": {"mode": "gridpoint"}},
        },
    )


def _write_scalars(
    directory: Path,
    *,
    spatial_targets: bool = False,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(
        directory / "inputs_mean.npy",
        np.array([10.0, -4.0], dtype=np.float32),
    )
    np.save(
        directory / "inputs_std.npy",
        np.array([2.0, 0.5], dtype=np.float32),
    )
    if spatial_targets:
        target_mean = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10.0
        target_std = np.linspace(0.5, 2.0, 24, dtype=np.float32).reshape(
            2, 3, 4
        )
    else:
        target_mean = np.array([1.0, -2.0], dtype=np.float32)
        target_std = np.array([3.0, 4.0], dtype=np.float32)
    np.save(directory / "targets_mean.npy", target_mean)
    np.save(directory / "targets_std.npy", target_std)


def test_case_scoped_scaler_resolution_never_uses_flat_or_other_case(
    tmp_path: Path,
) -> None:
    preprocessed = tmp_path / "preprocessed"
    legacy = tmp_path / "legacy_scalars"
    case_a = _config("case_a", preprocessed, legacy)
    case_b = _config("case_b", preprocessed, legacy)

    # A historical flat scalar directory must never leak into either case.
    _write_scalars(legacy)
    canonical_a = normalization.resolve_scalar_dir(case_a, for_writing=True)
    canonical_b = normalization.resolve_scalar_dir(case_b, for_writing=True)
    assert normalization.resolve_scalar_dir(case_a) == canonical_a
    assert normalization.resolve_scalar_dir(case_b) == canonical_b

    # Likewise, canonical scalars for case A cannot satisfy case B.
    _write_scalars(canonical_a)
    assert normalization.resolve_scalar_dir(case_a) == canonical_a
    assert normalization.resolve_scalar_dir(case_b) == canonical_b
    with pytest.raises(FileNotFoundError, match="case_b"):
        normalization.assert_scalars_available(case_b, role="test")

    # A legacy fallback is accepted only when its directory is explicitly
    # scoped to the active case name.
    legacy_b = legacy / "case_b"
    _write_scalars(legacy_b)
    assert normalization.resolve_scalar_dir(case_b) == legacy_b


def test_saved_scalers_wire_once_and_round_trip_channel_and_spatial_data(
    tmp_path: Path,
) -> None:
    config = _config(
        "round_trip",
        tmp_path / "preprocessed",
        tmp_path / "legacy",
    )
    scalar_dir = normalization.resolve_scalar_dir(config, for_writing=True)
    _write_scalars(scalar_dir, spatial_targets=True)

    assert (
        normalization.assert_scalars_available(config, role="test")
        == scalar_dir
    )
    assert normalization.apply_scalar_paths(config) == scalar_dir
    assert Path(config.model.input_mu) == scalar_dir / "inputs_mean.npy"
    assert Path(config.model.input_sigma) == scalar_dir / "inputs_std.npy"
    assert Path(config.model.target_mu) == scalar_dir / "targets_mean.npy"
    assert Path(config.model.target_sigma) == scalar_dir / "targets_std.npy"

    input_mean = normalization.assert_valid_predictor_scaler(
        "inputs_mean", np.load(config.model.input_mu), config
    )[:, None, None]
    input_std = normalization.assert_valid_predictor_scaler(
        "inputs_std", np.load(config.model.input_sigma), config
    )[:, None, None]
    target_mean = normalization.assert_valid_target_scaler_for_config(
        "targets_mean", np.load(config.model.target_mu), config
    )
    target_std = normalization.assert_valid_target_scaler_for_config(
        "targets_std", np.load(config.model.target_sigma), config
    )

    inputs = np.linspace(-8.0, 13.0, 24, dtype=np.float32).reshape(2, 3, 4)
    normalized_inputs = (inputs - input_mean) / input_std
    reconstructed_inputs = normalized_inputs * input_std + input_mean
    np.testing.assert_allclose(
        reconstructed_inputs,
        inputs,
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    targets = np.linspace(-5.0, 30.0, 24, dtype=np.float32).reshape(2, 3, 4)
    normalized_targets = (targets - target_mean) / target_std
    reconstructed_targets = normalized_targets * target_std + target_mean
    np.testing.assert_allclose(
        reconstructed_targets,
        targets,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_manifest_case_name_must_match_active_config(tmp_path: Path) -> None:
    config = _config("active_case", tmp_path / "preprocessed", tmp_path / "legacy")
    scalar_dir = normalization.resolve_scalar_dir(config, for_writing=True)
    _write_scalars(scalar_dir, spatial_targets=True)
    normalization.write_manifest(
        scalar_dir,
        case_name="different_case",
        predictor_mode="global",
        cfg=config,
    )

    with pytest.raises(ValueError, match="case mismatch"):
        normalization.log_scalar_summary(config, "test", logger=lambda *_: None)


def test_corrected_prism_run_requires_scalar_manifest(tmp_path: Path) -> None:
    config = _config("strict_case", tmp_path / "preprocessed", tmp_path / "legacy")
    config.data.type = "narr_prism"
    config.data.use_preprocessed = True
    scalar_dir = normalization.resolve_scalar_dir(config, for_writing=True)
    _write_scalars(scalar_dir, spatial_targets=True)

    with pytest.raises(FileNotFoundError, match="require.*normalization_manifest"):
        normalization.log_scalar_summary(config, "test", logger=lambda *_: None)


def test_corrected_prism_manifest_requires_canonical_grid_binding(
    tmp_path: Path,
) -> None:
    config = _config("strict_case", tmp_path / "preprocessed", tmp_path / "legacy")
    config.data.type = "merra_prism"
    config.data.use_preprocessed = True
    scalar_dir = normalization.resolve_scalar_dir(config, for_writing=True)
    _write_scalars(scalar_dir, spatial_targets=True)
    normalization.write_manifest(
        scalar_dir,
        case_name="strict_case",
        predictor_mode="global",
    )

    with pytest.raises(ValueError, match="not bound to a canonical PRISM grid"):
        normalization.log_scalar_summary(config, "test", logger=lambda *_: None)


def test_manifest_rejects_stale_dates_channel_order_and_scaling(tmp_path: Path) -> None:
    config = _config("semantic_case", tmp_path / "preprocessed", tmp_path / "legacy")
    config.data.input_vars = ["air_850", "elev"]
    config.data.target_variables = ["tmax", "tmin"]
    config.dates = {
        "training": {"start": "2000-01-01", "end": "2001-12-31"}
    }
    config.predictands["tmax"]["scaling"] = {"scale_stat": "mean"}
    config.predictands["tmin"]["scaling"] = {"scale_stat": "mean"}
    scalar_dir = normalization.resolve_scalar_dir(config, for_writing=True)
    _write_scalars(scalar_dir, spatial_targets=True)
    normalization.write_manifest(
        scalar_dir,
        case_name=config.case_name,
        predictor_mode="global",
        cfg=config,
        train_date_range=["2000-01-01", "2001-12-31"],
    )
    normalization.log_scalar_summary(config, "test", logger=lambda *_: None)

    config.dates["training"]["end"] = "2002-12-31"
    with pytest.raises(ValueError, match="training date range mismatch"):
        normalization.log_scalar_summary(config, "test", logger=lambda *_: None)
    config.dates["training"]["end"] = "2001-12-31"

    config.data.input_vars = ["elev", "air_850"]
    with pytest.raises(ValueError, match="config contract mismatch"):
        normalization.log_scalar_summary(config, "test", logger=lambda *_: None)
    config.data.input_vars = ["air_850", "elev"]

    config.predictands["tmax"]["scaling"]["scale_stat"] = "p95"
    with pytest.raises(ValueError, match="config contract mismatch"):
        normalization.log_scalar_summary(config, "test", logger=lambda *_: None)


def test_training_source_artifact_contract_requires_exact_dates_and_aggregate(
    tmp_path: Path,
) -> None:
    config = _config("source_case", tmp_path / "preprocessed", tmp_path / "legacy")
    config.data.type = "narr_prism"
    config.data.use_preprocessed = True
    config.dates = {
        "training": {"start": "2000-01-01", "end": "2000-01-02"}
    }
    scalar_dir = normalization.resolve_scalar_dir(config, for_writing=True)
    _write_scalars(scalar_dir, spatial_targets=True)
    daily = {"2000-01-01": "a" * 64, "2000-01-02": "b" * 64}
    aggregate = split_source_artifact_signature(daily)
    normalization.write_manifest(
        scalar_dir,
        case_name=config.case_name,
        predictor_mode="global",
        cfg=config,
        train_date_range=["2000-01-01", "2000-01-02"],
        extra={
            "predictor_preprocessing_signature": "c" * 64,
            normalization.TRAINING_SOURCE_ARTIFACT_SIGNATURES_KEY: daily,
            normalization.TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY: aggregate,
        },
    )

    observed_daily, observed_aggregate = (
        normalization.training_source_artifact_contract(
            config,
            expected_dates=["2000-01-01", "2000-01-02"],
            role="test",
        )
    )
    assert observed_daily == daily
    assert observed_aggregate == aggregate

    with pytest.raises(ValueError, match="do not exactly match"):
        normalization.training_source_artifact_contract(
            config,
            expected_dates=["2000-01-01", "2000-01-02", "2000-01-03"],
            role="test",
        )

    normalization.write_manifest(
        scalar_dir,
        case_name=config.case_name,
        predictor_mode="global",
        cfg=config,
        train_date_range=["2000-01-01", "2000-01-02"],
        extra={
            "predictor_preprocessing_signature": "c" * 64,
            normalization.TRAINING_SOURCE_ARTIFACT_SIGNATURES_KEY: daily,
            normalization.TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY: "d" * 64,
        },
    )
    with pytest.raises(ValueError, match="split signature is invalid"):
        normalization.training_source_artifact_contract(
            config,
            expected_dates=["2000-01-01", "2000-01-02"],
            role="test",
        )
