"""Synthetic daily-file integration tests for refinement evaluation."""

from __future__ import annotations

import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.NARR_PRISM.evaluate_refinement import (  # noqa: E402
    TARGET_VALID_MASK_CONTENT_SHA256_ATTR,
    TARGET_VALID_MASK_CRITERION_ATTR,
    TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR,
    TARGET_VALID_MASK_SHA256_ATTR,
    TARGET_VALID_MASK_SOURCE_SPLIT_ATTR,
    TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR,
    _target_valid_mask_content_sha256,
    main,
    parse_args,
    run_evaluation,
)
from granitewxc.utils.normalization import (  # noqa: E402
    TARGET_VALID_MASK_CRITERION,
)
from granitewxc.utils.prism_grid import save_canonical_grid  # noqa: E402

VARIABLES = ("ppt", "tmax", "tmin")
UNITS = {"ppt": "mm/day", "tmax": "degC", "tmin": "degC"}


def _synthetic_case(tmp_path: Path) -> dict[str, Path | str]:
    case_name = "synthetic_narr_prism"
    lat = np.array([32.0, 32.25, 32.5], dtype=np.float64)
    lon = np.array([-120.0, -119.75, -119.5, -119.25], dtype=np.float64)
    grid_dir = tmp_path / "grid"
    grid = save_canonical_grid(grid_dir, lat, lon, source="synthetic")
    truth_root = tmp_path / "truth"
    phase1_dir = tmp_path / "phase1"
    refined_dir = tmp_path / "refined"
    phase1_dir.mkdir()
    refined_dir.mkdir()

    support_mask = np.ones((lat.size, lon.size), dtype=bool)
    mask_attrs = {
        TARGET_VALID_MASK_SHA256_ATTR: "a" * 64,
        TARGET_VALID_MASK_CONTENT_SHA256_ATTR: (
            _target_valid_mask_content_sha256(support_mask)
        ),
        TARGET_VALID_MASK_CRITERION_ATTR: TARGET_VALID_MASK_CRITERION,
        TARGET_VALID_MASK_SOURCE_SPLIT_ATTR: "training",
        TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR: "b" * 64,
        TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR: grid.fingerprint,
    }

    start = date(2020, 1, 1)
    for day_index in range(2):
        sample_date = start + timedelta(days=day_index)
        token = sample_date.strftime("%Y%m%d")
        y, x = np.mgrid[: lat.size, : lon.size]
        truths = {
            # Span the default 1 mm/day occurrence threshold so the
            # integration test exercises exact dry and wet target strata.
            "ppt": 0.2 + day_index + 0.4 * y + 0.2 * x,
            "tmax": 20.0 + day_index + 0.6 * y + 0.1 * x,
            "tmin": 10.0 + day_index + 0.3 * y + 0.1 * x,
        }
        for variable, values in truths.items():
            directory = truth_root / variable / str(sample_date.year)
            directory.mkdir(parents=True, exist_ok=True)
            dataset = xr.Dataset(
                {
                    variable: xr.DataArray(
                        values.astype(np.float32),
                        dims=("lat", "lon"),
                        attrs={"units": UNITS[variable]},
                    )
                },
                coords={
                    "time": np.array([np.datetime64(sample_date, "ns")]),
                    "lat": lat,
                    "lon": lon,
                },
            )
            dataset.to_netcdf(directory / f"{variable}_{token}.nc")

        phase1 = xr.Dataset(
            {
                variable: xr.DataArray(
                    (values + 2.0)[None].astype(np.float32),
                    dims=("time", "lat", "lon"),
                    attrs={"units": UNITS[variable]},
                )
                for variable, values in truths.items()
            },
            coords={
                "time": np.array([np.datetime64(sample_date, "ns")]),
                "lat": lat,
                "lon": lon,
            },
            attrs={
                "case_name": case_name,
                "checkpoint": "/checkpoints/last.ckpt",
                "inference_start": sample_date.isoformat(),
                "inference_end": sample_date.isoformat(),
                "dataset_split": "inference",
                "split_start": start.isoformat(),
                "split_end": (start + timedelta(days=1)).isoformat(),
                "prism_grid_fingerprint": grid.fingerprint,
                **mask_attrs,
            },
        )
        phase1["prism_valid_mask"] = xr.DataArray(
            support_mask.astype(np.uint8), dims=("lat", "lon")
        )
        phase1.to_netcdf(phase1_dir / f"{case_name}_inference_{token}.nc")

        data_vars: dict[str, xr.DataArray] = {}
        for variable, values in truths.items():
            first_member = values + 0.5
            second_member = values + 1.5
            data_vars[variable] = xr.DataArray(
                (values + 1.0)[None].astype(np.float32),
                dims=("time", "lat", "lon"),
                attrs={"units": UNITS[variable]},
            )
            data_vars[f"{variable}_phase1"] = xr.DataArray(
                (values + 2.0)[None].astype(np.float32),
                dims=("time", "lat", "lon"),
                attrs={"units": UNITS[variable]},
            )
            for member_index, member_values in enumerate(
                (first_member, second_member)
            ):
                data_vars[f"{variable}_member_{member_index:03d}"] = (
                    xr.DataArray(
                        member_values[None].astype(np.float32),
                        dims=("time", "lat", "lon"),
                        attrs={"units": UNITS[variable]},
                    )
                )
        data_vars["prism_valid_mask"] = xr.DataArray(
            support_mask.astype(np.uint8), dims=("lat", "lon")
        )
        refined = xr.Dataset(
            data_vars,
            coords={
                "time": np.array([np.datetime64(sample_date, "ns")]),
                "lat": lat,
                "lon": lon,
            },
            attrs={
                "case_name": case_name,
                "refinement_type": "diffusion_unet",
                "ensemble_size": 2,
                "base_seed": 17,
                "phase1_checkpoint": "/checkpoints/last.ckpt",
                "phase2_checkpoint": "/checkpoints/diffusion.ckpt",
                "phase1_fingerprint": "phase1-fingerprint",
                "phase2_fingerprint": "phase2-fingerprint",
                "config_path": "/configs/diffusion.yaml",
                "config_fingerprint": "config-fingerprint",
                "inference_date": sample_date.isoformat(),
                "dataset_split": "inference",
                "split_start": start.isoformat(),
                "split_end": (start + timedelta(days=1)).isoformat(),
                "variable_order": json.dumps(list(VARIABLES)),
                "prism_grid_fingerprint": grid.fingerprint,
                **mask_attrs,
            },
        )
        refined.to_netcdf(
            refined_dir / f"{case_name}_diffusion_refined_{token}.nc"
        )

    elevation = np.array(
        [[100.0, 300.0, 600.0, 900.0]] * lat.size, dtype=np.float32
    )
    elevation_path = tmp_path / "elevation.npy"
    np.save(elevation_path, elevation)
    region = np.zeros((lat.size, lon.size), dtype=np.uint8)
    region[:, :2] = 1
    region_path = tmp_path / "coast.npy"
    np.save(region_path, region)

    config = {
        "case_name": case_name,
        "data": {
            "target_dir": str(truth_root),
            "preprocessed_dir": str(tmp_path / "preprocessed"),
            "target_variables": list(VARIABLES),
            "output_vars": list(VARIABLES),
        },
        "dates": {
            "inference": {
                "start": start.isoformat(),
                "end": (start + timedelta(days=1)).isoformat(),
            }
        },
        "evaluation": {"truth_units": dict(UNITS)},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return {
        "case_name": case_name,
        "config": config_path,
        "grid": grid_dir,
        "truth": truth_root,
        "phase1": phase1_dir,
        "refined": refined_dir,
        "elevation": elevation_path,
        "region": region_path,
    }


def test_cli_streams_daily_files_and_writes_metrics_strata_and_plots(tmp_path):
    paths = _synthetic_case(tmp_path)
    output = tmp_path / "evaluation"
    main(
        [
            "--config",
            str(paths["config"]),
            "--phase1-dir",
            str(paths["phase1"]),
            "--method",
            f"diffusion={paths['refined']}",
            "--output-dir",
            str(output),
            "--canonical-grid-dir",
            str(paths["grid"]),
            "--truth-dir",
            str(paths["truth"]),
            "--reservoir-size",
            "1000",
            "--elevation-file",
            str(paths["elevation"]),
            "--elevation-bins",
            "250,750",
            "--region-mask",
            f"coast={paths['region']}",
            "--temperature-lower-threshold",
            "cold=11",
            "--temperature-upper-threshold",
            "hot=21",
            "--progress-every",
            "0",
        ]
    )

    stem = f"{paths['case_name']}_refinement_20200101_20200102"
    json_path = output / f"{stem}_metrics.json"
    csv_path = output / f"{stem}_metrics.csv"
    assert json_path.is_file()
    assert csv_path.is_file()
    for variable in VARIABLES:
        assert (output / f"{stem}_{variable}_maps.png").is_file()
        assert (
            output / f"{stem}_{variable}_distribution_extremes.png"
        ).is_file()

    with open(json_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["variable_order"] == list(VARIABLES)
    assert len(report["config_sha256"]) == 64
    assert report["date_range"]["inclusive_day_count"] == 2
    assert report["dataset_split"] == "inference"
    assert report["date_range"]["split"] == "inference"
    assert report["date_range"]["configured_start"] == "2020-01-01"
    assert report["date_range"]["configured_end"] == "2020-01-02"
    assert report["metrics"]["phase1"]["ppt"]["deterministic"][
        "rmse"
    ] == pytest.approx(2.0)
    assert report["metrics"]["diffusion"]["ppt"]["deterministic"][
        "rmse"
    ] == pytest.approx(1.0)
    assert report["metrics"]["diffusion"]["ppt"]["deterministic"][
        "distribution_wasserstein_1"
    ] == pytest.approx(1.0)
    assert report["metrics"]["diffusion"]["ppt"]["deterministic"][
        "spatial_roughness_ratio"
    ] == pytest.approx(1.0)
    assert report["metrics"]["diffusion"]["ppt"]["deterministic"][
        "spatial_smoothing_index"
    ] == pytest.approx(0.0, abs=1.0e-8)
    assert report["metrics"]["diffusion"]["ppt"]["ensemble"][
        "empirical_crps"
    ] == pytest.approx(0.75)
    assert report["metrics"]["diffusion"]["ppt"]["ensemble"][
        "prediction_interval_mean_absolute_coverage_error"
    ] > 0.0
    assert report["metrics"]["diffusion"]["ppt"]["distribution_sampling"][
        "exact"
    ]
    assert report["metrics"]["diffusion"]["temperature_ordering"][
        "tasmin_gt_tasmax_violation_rate"
    ] == pytest.approx(0.0)
    assert report["metrics"]["diffusion"]["tmax"]["specialized"][
        "upper_threshold_hot"
    ] == pytest.approx(21.0)
    assert report["metrics"]["diffusion"]["tmin"]["specialized"][
        "lower_threshold_cold"
    ] == pytest.approx(11.0)
    assert report["temperature_thresholds"] == {
        "lower": {"cold": 11.0},
        "upper": {"hot": 21.0},
    }
    assert "01" in report["strata"]["month"]
    assert "DJF" in report["strata"]["season"]
    assert "coast" in report["strata"]["region"]
    assert set(report["strata"]["elevation"]) == {
        "below_250",
        "250_to_750",
        "750_and_above",
    }
    assert set(report["strata"]["target_precipitation_occurrence"]) == {
        "target_dry",
        "target_wet",
    }
    assert "moderate_wet" in report["strata"]["target_magnitude"]
    assert "extreme_q90" in report["strata"]["target_magnitude"]
    assert "moderate_q05_q95" in report["strata"]["target_magnitude"]
    assert report["target_regime_definitions"]["target_magnitude"]["ppt"][
        "distribution_sampling"
    ]["exact"]
    assert (
        report["provenance"]["methods"]["diffusion"]["phase2_fingerprint"]
        == "phase2-fingerprint"
    )
    assert report["provenance"]["phase1"]["dataset_split"] == "inference"
    assert (
        report["provenance"]["methods"]["diffusion"]["dataset_split"]
        == "inference"
    )

    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert any(
        row["method"] == "diffusion"
        and row["variable"] == "ppt"
        and row["scope"] == "overall"
        for row in rows
    )
    assert any(
        row["scope"] == "season" and row["label"] == "DJF" for row in rows
    )
    assert any(
        row["scope"] == "target_precipitation_occurrence"
        and row["label"] == "target_dry"
        for row in rows
    )
    assert any(
        row["scope"] == "target_magnitude"
        and row["label"] == "extreme_q90"
        for row in rows
    )


def test_evaluator_cli_split_defaults_to_inference_and_accepts_validation() -> None:
    required = [
        "--config",
        "config.yaml",
        "--phase1-dir",
        "phase1",
        "--method",
        "flow=refined",
        "--output-dir",
        "metrics",
    ]
    assert parse_args(required).split == "inference"
    assert parse_args([*required, "--split", "validation"]).split == (
        "validation"
    )


def test_evaluator_rejects_shifted_refinement_coordinates(tmp_path):
    paths = _synthetic_case(tmp_path)
    refined_file = sorted(Path(paths["refined"]).glob("*.nc"))[0]
    with xr.open_dataset(refined_file) as source:
        changed = source.load()
    changed = changed.assign_coords(lon=changed["lon"].values + 0.01)
    changed.to_netcdf(refined_file, mode="w")

    with pytest.raises(ValueError, match="coordinates do not exactly match"):
        run_evaluation(
            config_path=paths["config"],
            phase1_dir=paths["phase1"],
            method_dirs={"diffusion": paths["refined"]},
            output_dir=tmp_path / "out",
            canonical_grid_dir=paths["grid"],
            truth_dir=paths["truth"],
            make_plots=False,
            progress_every=0,
        )


def test_evaluator_allows_only_explicit_subsets_inside_selected_split(
    tmp_path,
) -> None:
    paths = _synthetic_case(tmp_path)
    report = run_evaluation(
        config_path=paths["config"],
        phase1_dir=paths["phase1"],
        method_dirs={"diffusion": paths["refined"]},
        output_dir=tmp_path / "subset",
        canonical_grid_dir=paths["grid"],
        truth_dir=paths["truth"],
        start="2020-01-02",
        end="2020-01-02",
        make_plots=False,
        progress_every=0,
    )
    assert report["date_range"]["start"] == "2020-01-02"
    assert report["date_range"]["end"] == "2020-01-02"
    assert report["date_range"]["configured_start"] == "2020-01-01"
    assert report["date_range"]["configured_end"] == "2020-01-02"

    with pytest.raises(ValueError, match="must be supplied together"):
        run_evaluation(
            config_path=paths["config"],
            phase1_dir=paths["phase1"],
            method_dirs={"diffusion": paths["refined"]},
            output_dir=tmp_path / "half-bound",
            canonical_grid_dir=paths["grid"],
            truth_dir=paths["truth"],
            start="2020-01-02",
            make_plots=False,
            progress_every=0,
        )
    with pytest.raises(ValueError, match="outside configured dates.inference"):
        run_evaluation(
            config_path=paths["config"],
            phase1_dir=paths["phase1"],
            method_dirs={"diffusion": paths["refined"]},
            output_dir=tmp_path / "outside",
            canonical_grid_dir=paths["grid"],
            truth_dir=paths["truth"],
            start="2019-12-31",
            end="2020-01-01",
            make_plots=False,
            progress_every=0,
        )


@pytest.mark.parametrize("role", ["phase1", "refinement"])
def test_evaluator_rejects_cross_split_product_provenance(
    tmp_path, role
) -> None:
    paths = _synthetic_case(tmp_path)
    directory = Path(paths["phase1"] if role == "phase1" else paths["refined"])
    product = sorted(directory.glob("*.nc"))[0]
    with xr.open_dataset(product) as source:
        changed = source.load()
    changed.attrs["dataset_split"] = "validation"
    changed.to_netcdf(product, mode="w")

    with pytest.raises(ValueError, match="dataset_split='validation'"):
        run_evaluation(
            config_path=paths["config"],
            phase1_dir=paths["phase1"],
            method_dirs={"diffusion": paths["refined"]},
            output_dir=tmp_path / "cross-split",
            canonical_grid_dir=paths["grid"],
            truth_dir=paths["truth"],
            make_plots=False,
            progress_every=0,
        )


def test_evaluator_rejects_product_bounds_not_matching_yaml(tmp_path) -> None:
    paths = _synthetic_case(tmp_path)
    product = sorted(Path(paths["refined"]).glob("*.nc"))[0]
    with xr.open_dataset(product) as source:
        changed = source.load()
    changed.attrs["split_end"] = "2020-01-01"
    changed.to_netcdf(product, mode="w")

    with pytest.raises(ValueError, match="product split bounds"):
        run_evaluation(
            config_path=paths["config"],
            phase1_dir=paths["phase1"],
            method_dirs={"diffusion": paths["refined"]},
            output_dir=tmp_path / "wrong-bounds",
            canonical_grid_dir=paths["grid"],
            truth_dir=paths["truth"],
            make_plots=False,
            progress_every=0,
        )


def test_evaluator_validation_split_uses_validation_yaml_and_provenance(
    tmp_path,
) -> None:
    paths = _synthetic_case(tmp_path)
    with open(paths["config"], "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["dates"]["validation"] = {
        "start": "2020-01-01",
        "end": "2020-01-02",
    }
    config["dates"]["inference"] = {
        "start": "2019-01-01",
        "end": "2019-01-02",
    }
    with open(paths["config"], "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    for directory in (Path(paths["phase1"]), Path(paths["refined"])):
        for product in directory.glob("*.nc"):
            with xr.open_dataset(product) as source:
                changed = source.load()
            changed.attrs["dataset_split"] = "validation"
            changed.to_netcdf(product, mode="w")

    report = run_evaluation(
        config_path=paths["config"],
        phase1_dir=paths["phase1"],
        method_dirs={"diffusion": paths["refined"]},
        output_dir=tmp_path / "validation-evaluation",
        canonical_grid_dir=paths["grid"],
        truth_dir=paths["truth"],
        split="validation",
        make_plots=False,
        progress_every=0,
    )
    assert report["dataset_split"] == "validation"
    assert report["date_range"]["split"] == "validation"
    assert report["date_range"]["inclusive_day_count"] == 2


def test_bounded_distribution_reservoir_retains_identical_truth_cells(
    tmp_path,
):
    """Method comparisons must remain paired when the reservoir is sampled."""
    paths = _synthetic_case(tmp_path)
    report = run_evaluation(
        config_path=paths["config"],
        phase1_dir=paths["phase1"],
        method_dirs={"diffusion": paths["refined"]},
        output_dir=tmp_path / "out",
        canonical_grid_dir=paths["grid"],
        truth_dir=paths["truth"],
        reservoir_size=3,
        make_plots=False,
        progress_every=0,
    )

    for variable in VARIABLES:
        baseline = report["metrics"]["phase1"][variable]["deterministic"]
        refined = report["metrics"]["diffusion"][variable]["deterministic"]
        assert not report["metrics"]["phase1"][variable][
            "distribution_sampling"
        ]["exact"]
        target_quantiles = [
            key for key in baseline if key.startswith("target_q")
        ]
        assert target_quantiles
        for key in target_quantiles:
            assert refined[key] == pytest.approx(baseline[key])


def test_evaluator_records_configured_units_when_source_metadata_is_absent(
    tmp_path,
):
    paths = _synthetic_case(tmp_path)
    for path in Path(paths["truth"]).glob("*/*/*.nc"):
        with xr.open_dataset(path) as source:
            changed = source.load()
        variable = path.parent.parent.name
        changed[variable].attrs.pop("units", None)
        changed.to_netcdf(path, mode="w")

    report = run_evaluation(
        config_path=paths["config"],
        phase1_dir=paths["phase1"],
        method_dirs={"diffusion": paths["refined"]},
        output_dir=tmp_path / "out",
        canonical_grid_dir=paths["grid"],
        truth_dir=paths["truth"],
        make_plots=False,
        progress_every=0,
    )

    audit = report["provenance"]["truth_units"]
    assert audit["configured"] == UNITS
    for variable in VARIABLES:
        observed = audit["per_variable"][variable]
        assert observed["configured_unit"] == UNITS[variable]
        assert observed["source_attribute_files"] == 0
        assert observed["configured_assumption_files"] == 2
        assert observed["observed_source_units"] == []
        assert observed["used_configured_assumption"]


def test_evaluator_rejects_invalid_configured_truth_units(tmp_path):
    paths = _synthetic_case(tmp_path)
    with open(paths["config"], "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["evaluation"]["truth_units"]["ppt"] = "kg m-2 s-1"
    with open(paths["config"], "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    with pytest.raises(ValueError, match=r"truth_units must exactly equal"):
        run_evaluation(
            config_path=paths["config"],
            phase1_dir=paths["phase1"],
            method_dirs={"diffusion": paths["refined"]},
            output_dir=tmp_path / "out",
            canonical_grid_dir=paths["grid"],
            truth_dir=paths["truth"],
            make_plots=False,
            progress_every=0,
        )


def test_evaluator_rejects_mixed_phase2_provenance(tmp_path):
    paths = _synthetic_case(tmp_path)
    refined_file = sorted(Path(paths["refined"]).glob("*.nc"))[1]
    with xr.open_dataset(refined_file) as source:
        changed = source.load()
    changed.attrs["phase2_fingerprint"] = "different-phase2"
    changed.to_netcdf(refined_file, mode="w")

    with pytest.raises(ValueError, match="mixes daily provenance"):
        run_evaluation(
            config_path=paths["config"],
            phase1_dir=paths["phase1"],
            method_dirs={"diffusion": paths["refined"]},
            output_dir=tmp_path / "out",
            canonical_grid_dir=paths["grid"],
            truth_dir=paths["truth"],
            make_plots=False,
            progress_every=0,
        )


def test_evaluator_rejects_different_internally_valid_support_mask(tmp_path):
    paths = _synthetic_case(tmp_path)
    refined_file = sorted(Path(paths["refined"]).glob("*.nc"))[1]
    with xr.open_dataset(refined_file) as source:
        changed = source.load()
    changed_mask = np.asarray(changed["prism_valid_mask"].values, dtype=bool)
    changed_mask[0, 0] = False
    changed["prism_valid_mask"].values[:] = changed_mask.astype(np.uint8)
    for name in changed.data_vars:
        if name != "prism_valid_mask":
            changed[name].values[..., 0, 0] = np.nan
    changed.attrs[TARGET_VALID_MASK_CONTENT_SHA256_ATTR] = (
        _target_valid_mask_content_sha256(changed_mask)
    )
    changed.attrs[TARGET_VALID_MASK_SHA256_ATTR] = "c" * 64
    changed.to_netcdf(refined_file, mode="w")

    with pytest.raises(ValueError, match="target-valid-mask cells differ"):
        run_evaluation(
            config_path=paths["config"],
            phase1_dir=paths["phase1"],
            method_dirs={"diffusion": paths["refined"]},
            output_dir=tmp_path / "out",
            canonical_grid_dir=paths["grid"],
            truth_dir=paths["truth"],
            make_plots=False,
            progress_every=0,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("config", "config_fingerprint"),
        ("truth_date", "truth time encodes"),
        ("truth_units", "units.*incompatible"),
    ),
)
def test_evaluator_rejects_mixed_config_or_invalid_truth(
    tmp_path, mutation, match
):
    paths = _synthetic_case(tmp_path)
    if mutation == "config":
        path = sorted(Path(paths["refined"]).glob("*.nc"))[1]
        with xr.open_dataset(path) as source:
            changed = source.load()
        changed.attrs["config_fingerprint"] = "different-config"
    else:
        path = sorted((Path(paths["truth"]) / "ppt" / "2020").glob("*.nc"))[0]
        with xr.open_dataset(path) as source:
            changed = source.load()
        if mutation == "truth_date":
            changed = changed.assign_coords(
                time=np.array([np.datetime64("2019-12-31", "ns")])
            )
        else:
            changed["ppt"].attrs["units"] = "kg m-2 s-1"
    changed.to_netcdf(path, mode="w")

    with pytest.raises(ValueError, match=match):
        run_evaluation(
            config_path=paths["config"],
            phase1_dir=paths["phase1"],
            method_dirs={"diffusion": paths["refined"]},
            output_dir=tmp_path / "out",
            canonical_grid_dir=paths["grid"],
            truth_dir=paths["truth"],
            make_plots=False,
            progress_every=0,
        )
