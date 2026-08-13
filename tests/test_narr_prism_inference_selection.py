from __future__ import annotations

import json
import hashlib
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from examples.NARR_PRISM.narr_prism_inference import (
    TARGET_VALID_MASK_CONTENT_SHA256_ATTR,
    TARGET_VALID_MASK_CRITERION_ATTR,
    TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR,
    TARGET_VALID_MASK_SHA256_ATTR,
    TARGET_VALID_MASK_SOURCE_SPLIT_ATTR,
    TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR,
    _open_streaming_output,
    _select_inference_dates,
    _split_output_root,
    _target_valid_mask_content_sha256,
    _tensor_to_float32_numpy,
    _validate_dataset_dates,
    _validate_prediction_split,
)
from examples.NARR_PRISM.narr_prism_output_audit import (
    MANIFEST_NAME,
    audit_outputs,
    expected_output_dates,
    write_output_manifest,
)
from granitewxc.utils.normalization import TARGET_VALID_MASK_CRITERION
from granitewxc.utils.prism_grid import validate_prism_grid


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "examples" / "NARR_PRISM" / "notebooks"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tensor_to_float32_numpy_supports_inference_dtypes(dtype) -> None:
    tensor = torch.tensor([1.25, -2.5], dtype=dtype, requires_grad=True)

    converted = _tensor_to_float32_numpy(tensor)

    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted, np.array([1.25, -2.5], dtype=np.float32))


def test_selected_inference_dates_preserve_requested_order_and_native_values() -> None:
    start = date(2020, 1, 1)
    available = [start + timedelta(days=offset) for offset in range(5)]

    selected = _select_inference_dates(
        available,
        [np.datetime64("2020-01-05"), "2020-01-02"],
    )

    assert selected == [available[4], available[1]]
    assert _select_inference_dates(available, None) == available


@pytest.mark.parametrize(
    "selected, message",
    [
        ([], "at least one"),
        (["2020-01-02", "2020-01-02"], "duplicates"),
        (["2020-02-01"], "outside"),
        (["not-a-date"], "invalid ISO date"),
    ],
)
def test_selected_inference_dates_reject_invalid_requests(selected, message) -> None:
    available = [date(2020, 1, 1), date(2020, 1, 2)]
    with pytest.raises(ValueError, match=message):
        _select_inference_dates(available, selected)


def test_prediction_split_dates_are_exact_inclusive_and_leap_safe() -> None:
    cfg = {
        "dates": {
            "validation": {
                "start": "2015-02-28",
                "end": "2015-03-01",
            },
            "inference": {
                "start": "2016-02-28",
                "end": "2016-03-01",
            },
        }
    }
    validation = [date(2015, 2, 28), date(2015, 3, 1)]
    inference = [
        date(2016, 2, 28),
        date(2016, 2, 29),
        date(2016, 3, 1),
    ]
    assert _validate_dataset_dates(cfg, validation, "validation") == validation
    assert _validate_dataset_dates(cfg, inference) == inference
    with pytest.raises(ValueError, match=r"dates\.validation"):
        _validate_dataset_dates(cfg, validation[:-1], "validation")
    with pytest.raises(ValueError, match="split must be one of"):
        _validate_prediction_split("training")


def test_validation_output_root_isolated_without_changing_inference(
    tmp_path: Path,
) -> None:
    assert _split_output_root(tmp_path, "inference") == tmp_path
    assert _split_output_root(tmp_path, "validation") == tmp_path / "validation"
    assert (
        _split_output_root(tmp_path / "validation", "validation")
        == tmp_path / "validation"
    )


def test_daily_deterministic_output_persists_support_mask_provenance(
    tmp_path: Path,
) -> None:
    lat = np.array([40.0, 40.5], dtype=np.float64)
    lon = np.array([-121.0, -120.5, -120.0], dtype=np.float64)
    mask = np.array([[True, False, True], [True, True, False]])
    provenance = {
        TARGET_VALID_MASK_SHA256_ATTR: "a" * 64,
        TARGET_VALID_MASK_CONTENT_SHA256_ATTR: (
            _target_valid_mask_content_sha256(mask)
        ),
        TARGET_VALID_MASK_CRITERION_ATTR: TARGET_VALID_MASK_CRITERION,
        TARGET_VALID_MASK_SOURCE_SPLIT_ATTR: "training",
        TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR: "b" * 64,
        TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR: validate_prism_grid(
            lat, lon, context="deterministic-output-test"
        ).fingerprint,
    }
    output = tmp_path / "daily.nc"
    handle = _open_streaming_output(
        output,
        ["ppt"],
        lat,
        lon,
        "/checkpoints/phase1.ckpt",
        [date(2020, 1, 1)],
        "case",
        mask,
        provenance,
        "validation",
    )
    handle.variables["time"][:] = np.array([18262], dtype=np.int32)
    handle.variables["ppt"][0:1, :, :] = np.zeros(
        (1, *mask.shape), dtype=np.float32
    )
    handle.flush()
    handle.close()

    with xr.open_dataset(output) as dataset:
        np.testing.assert_array_equal(
            dataset["prism_valid_mask"].values.astype(bool), mask
        )
        for name, value in provenance.items():
            assert dataset.attrs[name] == value
        assert dataset.attrs["dataset_split"] == "validation"
        assert dataset.attrs["split_start"] == "2020-01-01"
        assert dataset.attrs["split_end"] == "2020-01-01"


def _write_auditable_daily_output(
    path: Path,
    *,
    sample_date: date,
    configured_dates: list[date],
    checkpoint: Path,
    invalid_fill: float = float("nan"),
) -> None:
    lat = np.array([40.0, 40.5], dtype=np.float64)
    lon = np.array([-121.0, -120.5, -120.0], dtype=np.float64)
    mask = np.array([[True, False, True], [True, True, False]])
    provenance = {
        TARGET_VALID_MASK_SHA256_ATTR: "a" * 64,
        TARGET_VALID_MASK_CONTENT_SHA256_ATTR: (
            _target_valid_mask_content_sha256(mask)
        ),
        TARGET_VALID_MASK_CRITERION_ATTR: TARGET_VALID_MASK_CRITERION,
        TARGET_VALID_MASK_SOURCE_SPLIT_ATTR: "training",
        TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR: "b" * 64,
        TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR: validate_prism_grid(
            lat, lon, context="output-audit-test"
        ).fingerprint,
    }
    handle = _open_streaming_output(
        path,
        ["ppt", "tmax", "tmin"],
        lat,
        lon,
        str(checkpoint),
        [sample_date],
        "audit_case",
        mask,
        provenance,
        "inference",
        configured_dates,
    )
    handle.variables["time"][:] = np.array(
        [(sample_date - date(1970, 1, 1)).days], dtype=np.int32
    )
    values = {"ppt": 1.0, "tmax": 20.0, "tmin": 10.0}
    for variable, value in values.items():
        output = np.full((1, 2, 3), value, dtype=np.float32)
        output[:, ~mask] = invalid_fill
        handle.variables[variable][:] = output
    handle.close()


def _output_audit_fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[date]]:
    output_root = tmp_path / "outputs"
    output_path = output_root / "audit_case"
    output_path.mkdir(parents=True)
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"test checkpoint")
    config_path = tmp_path / "audit.yaml"
    config_path.write_text(
        "\n".join(
            [
                "case_name: audit_case",
                "data:",
                "  target_variables: [ppt, tmax, tmin]",
                "dates:",
                "  inference:",
                "    start: '2020-02-28'",
                "    end: '2020-03-01'",
                "inference:",
                f"  output_dir: {output_root}",
                f"resume_checkpoint_path: {checkpoint}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dates = [date(2020, 2, 28), date(2020, 2, 29), date(2020, 3, 1)]
    return config_path, output_path, checkpoint, dates


def test_output_audit_reports_inclusive_missing_extra_and_duplicate_dates(
    tmp_path: Path,
) -> None:
    config_path, output_path, checkpoint, dates = _output_audit_fixture(tmp_path)
    for sample_date in (dates[0], dates[-1]):
        _write_auditable_daily_output(
            output_path / f"audit_case_inference_{sample_date:%Y%m%d}.nc",
            sample_date=sample_date,
            configured_dates=dates,
            checkpoint=checkpoint,
        )
    duplicate = output_path / "audit_case_inference_20200228_copy.nc"
    duplicate.write_bytes(
        (output_path / "audit_case_inference_20200228.nc").read_bytes()
    )

    report = audit_outputs(config_path, validation_level="endpoints")

    assert expected_output_dates(
        {"dates": {"inference": {"start": "2020-02-28", "end": "2020-03-01"}}}
    ) == ["2020-02-28", "2020-02-29", "2020-03-01"]
    assert report["status"] == "incomplete"
    assert report["inventory"]["expected_count"] == 3
    assert report["inventory"]["missing_files"] == [
        "audit_case_inference_20200229.nc"
    ]
    assert report["inventory"]["extra_files"] == [duplicate.name]
    assert report["inventory"]["duplicate_dates"] == {
        "2020-02-28": [
            "audit_case_inference_20200228.nc",
            duplicate.name,
        ]
    }
    assert all(item["valid"] for item in report["validated_files"])
    with pytest.raises(ValueError, match="refusing"):
        write_output_manifest(report)
    assert not (output_path / MANIFEST_NAME).exists()


def test_complete_output_audit_validates_contract_and_writes_atomic_manifest(
    tmp_path: Path,
) -> None:
    config_path, output_path, checkpoint, dates = _output_audit_fixture(tmp_path)
    for sample_date in dates:
        _write_auditable_daily_output(
            output_path / f"audit_case_inference_{sample_date:%Y%m%d}.nc",
            sample_date=sample_date,
            configured_dates=dates,
            checkpoint=checkpoint,
        )

    report = audit_outputs(config_path, validation_level="all")
    manifest_path = write_output_manifest(report)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert report["complete"] is True
    assert report["status"] == "complete"
    assert report["configured_checkpoint_sha256"] == hashlib.sha256(
        b"test checkpoint"
    ).hexdigest()
    assert len(report["validated_files"]) == 3
    assert all(item["valid"] for item in report["validated_files"])
    assert manifest_path == output_path / MANIFEST_NAME
    assert manifest["schema_version"] == 1
    assert len(manifest["files"]) == 3
    assert len(manifest["inventory_sha256"]) == 64
    assert not list(output_path.glob(f".{MANIFEST_NAME}.tmp-*"))


def test_endpoint_validation_cannot_publish_complete_manifest(
    tmp_path: Path,
) -> None:
    config_path, output_path, checkpoint, dates = _output_audit_fixture(tmp_path)
    for sample_date in dates:
        _write_auditable_daily_output(
            output_path / f"audit_case_inference_{sample_date:%Y%m%d}.nc",
            sample_date=sample_date,
            configured_dates=dates,
            checkpoint=checkpoint,
        )

    report = audit_outputs(config_path, validation_level="endpoints")

    assert report["inventory"]["complete"] is True
    assert report["status"] == "partially_validated"
    assert report["complete"] is False
    assert len(report["validated_files"]) == 2
    with pytest.raises(ValueError, match="all-file audit"):
        write_output_manifest(report)


def test_complete_inventory_without_validation_cannot_publish_manifest(
    tmp_path: Path,
) -> None:
    config_path, output_path, checkpoint, dates = _output_audit_fixture(tmp_path)
    for sample_date in dates:
        _write_auditable_daily_output(
            output_path / f"audit_case_inference_{sample_date:%Y%m%d}.nc",
            sample_date=sample_date,
            configured_dates=dates,
            checkpoint=checkpoint,
        )

    report = audit_outputs(config_path, validation_level="none")

    assert report["inventory"]["complete"] is True
    assert report["validated_files"] == []
    assert report["status"] == "unvalidated"
    assert report["complete"] is False
    with pytest.raises(ValueError, match="refusing"):
        write_output_manifest(report)


def test_output_audit_rejects_finite_values_outside_support_mask(
    tmp_path: Path,
) -> None:
    config_path, output_path, checkpoint, dates = _output_audit_fixture(tmp_path)
    for index, sample_date in enumerate(dates):
        _write_auditable_daily_output(
            output_path / f"audit_case_inference_{sample_date:%Y%m%d}.nc",
            sample_date=sample_date,
            configured_dates=dates,
            checkpoint=checkpoint,
            invalid_fill=0.0 if index == 0 else float("nan"),
        )

    report = audit_outputs(config_path, validation_level="all")

    assert report["status"] == "invalid"
    assert report["complete"] is False
    assert any(
        "finite values on invalid mask cells" in error
        for error in report["validated_files"][0]["errors"]
    )


def test_output_audit_rejects_swapped_spatial_dimension_order(
    tmp_path: Path,
) -> None:
    config_path, output_path, checkpoint, dates = _output_audit_fixture(tmp_path)
    paths: list[Path] = []
    for sample_date in dates:
        path = output_path / f"audit_case_inference_{sample_date:%Y%m%d}.nc"
        _write_auditable_daily_output(
            path,
            sample_date=sample_date,
            configured_dates=dates,
            checkpoint=checkpoint,
        )
        paths.append(path)

    with xr.open_dataset(paths[0]) as source:
        source.load()
        swapped = xr.Dataset(
            {
                "prism_valid_mask": (
                    ("lon", "lat"),
                    source["prism_valid_mask"].values.T,
                    dict(source["prism_valid_mask"].attrs),
                ),
                **{
                    variable: (
                        ("time", "lon", "lat"),
                        source[variable].values.transpose(0, 2, 1),
                        dict(source[variable].attrs),
                    )
                    for variable in ("ppt", "tmax", "tmin")
                },
            },
            coords={
                "time": source["time"].values,
                "lat": source["lat"].values,
                "lon": source["lon"].values,
            },
            attrs=dict(source.attrs),
        )
    temporary = paths[0].with_suffix(".swapped.nc")
    swapped.to_netcdf(temporary, engine="h5netcdf")
    os.replace(temporary, paths[0])

    report = audit_outputs(config_path, validation_level="all")

    assert report["status"] == "invalid"
    assert any(
        "dimensions" in error
        for error in report["validated_files"][0]["errors"]
    )


def test_output_audit_rejects_cross_day_mask_provenance_drift(
    tmp_path: Path,
) -> None:
    config_path, output_path, checkpoint, dates = _output_audit_fixture(tmp_path)
    paths: list[Path] = []
    for sample_date in dates:
        path = output_path / f"audit_case_inference_{sample_date:%Y%m%d}.nc"
        _write_auditable_daily_output(
            path,
            sample_date=sample_date,
            configured_dates=dates,
            checkpoint=checkpoint,
        )
        paths.append(path)

    with xr.open_dataset(paths[1]) as source:
        changed = source.load()
    changed.attrs[TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR] = "c" * 64
    temporary = paths[1].with_suffix(".changed.nc")
    changed.to_netcdf(temporary, engine="h5netcdf")
    os.replace(temporary, paths[1])

    report = audit_outputs(config_path, validation_level="all")

    assert report["status"] == "invalid"
    assert any(
        "differs from first validated output" in error
        for error in report["validated_files"][1]["errors"]
    )


def _notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_main_inference_notebook_has_no_spatial_mean_section() -> None:
    notebook = _notebook(NOTEBOOK_DIR / "narr_prism_inference.ipynb")
    cell_ids = {cell.get("id") for cell in notebook["cells"]}
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "timeseries-section" not in cell_ids
    assert "plot-timeseries" not in cell_ids
    assert "spatial_mean" not in source


def test_random_ten_notebook_is_cpu_only_and_plots_all_targets() -> None:
    notebook = _notebook(NOTEBOOK_DIR / "narr_prism_inference_random10_cpu.ipynb")
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "SAMPLE_COUNT = 10" in source
    assert 'device = torch.device("cpu")' in source
    assert "selected_dates=selected_dates" in source
    assert 'PLOT_VARIABLES = ["ppt", "tmax", "tmin"]' in source
    assert 'f"target_{variable}"' in source
    assert "Inference − PRISM" in source
    assert "spatial_mean" not in source
