"""Inventory and validate deterministic NARR--PRISM daily inference outputs.

This command is deliberately model- and dataset-free: it resolves the output
location and inclusive date contract from the training YAML, inventories daily
NetCDF files, and validates their self-described output contract.  It is safe
to run while inference is in progress.  A manifest is published atomically
only for a complete, valid output set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from narr_prism_utils import (  # noqa: E402
    case_output_dir,
    date_range,
    get_case_name,
    load_yaml,
    parse_date_range_from_config,
    resolve_path,
)

from granitewxc.utils.prism_grid import validate_prism_grid  # noqa: E402

MANIFEST_NAME = "inference_output_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
PREDICTION_SPLITS = ("validation", "inference")
VARIABLE_UNITS = {"ppt": "mm/day", "tmax": "degC", "tmin": "degC"}
MASK_VARIABLE = "prism_valid_mask"
MASK_CONTENT_ATTR = "target_valid_mask_content_sha256"
REQUIRED_PROVENANCE_ATTRS = (
    "target_valid_mask_sha256",
    MASK_CONTENT_ATTR,
    "target_valid_mask_criterion",
    "target_valid_mask_source_split",
    "target_valid_mask_training_source_artifact_split_signature",
    "target_valid_mask_grid_fingerprint",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mask_content_sha256(mask: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    digest = hashlib.sha256()
    digest.update(b"granitewxc-target-valid-mask-cells-v1\0")
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _split_output_root(output_root: str | Path, split: str) -> Path:
    split = str(split).strip().lower()
    if split not in PREDICTION_SPLITS:
        raise ValueError(
            f"split must be one of {PREDICTION_SPLITS}, got {split!r}"
        )
    path = resolve_path(output_root)
    if split == "inference" or path.name == split or path.parent.name == split:
        return path
    return path / split


def resolve_output_path(
    cfg: Mapping[str, Any],
    *,
    output_dir: Optional[str | Path] = None,
    split: str = "inference",
) -> Path:
    """Resolve the exact case directory used by deterministic inference."""
    configured = (cfg.get("inference", {}) or {}).get(
        "output_dir", "./examples/NARR_PRISM/experiments/inference_output"
    )
    root = _split_output_root(
        output_dir if output_dir is not None else configured, split
    )
    return case_output_dir(root, get_case_name(cfg))


def expected_output_dates(
    cfg: Mapping[str, Any], split: str = "inference"
) -> list[str]:
    """Return the exact inclusive configured dates, including leap days."""
    split = str(split).strip().lower()
    if split not in PREDICTION_SPLITS:
        raise ValueError(
            f"split must be one of {PREDICTION_SPLITS}, got {split!r}"
        )
    start, end = parse_date_range_from_config(dict(cfg), split)
    return [value.isoformat() for value in date_range(start, end)]


def _configured_checkpoint(
    cfg: Mapping[str, Any], explicit: Optional[str | Path]
) -> Optional[Path]:
    candidates: list[str | Path] = []
    if explicit:
        return resolve_path(explicit).resolve(strict=False)
    inference_checkpoint = (cfg.get("inference", {}) or {}).get(
        "checkpoint_path"
    )
    if inference_checkpoint:
        candidates.append(inference_checkpoint)
    if cfg.get("resume_checkpoint_path"):
        candidates.append(cfg["resume_checkpoint_path"])
    case_name = get_case_name(cfg)
    if cfg.get("checkpoint_dir"):
        root = case_output_dir(resolve_path(cfg["checkpoint_dir"]), case_name)
        candidates.extend((root / "last.ckpt", root / "best.ckpt"))
    if cfg.get("path_experiment"):
        root = case_output_dir(
            resolve_path(cfg["path_experiment"]) / "checkpoints", case_name
        )
        candidates.extend((root / "last.ckpt", root / "best.ckpt"))
    for candidate in candidates:
        path = resolve_path(candidate)
        try:
            if path.is_file():
                return path.resolve()
        except OSError:
            continue
    return (
        resolve_path(candidates[0]).resolve(strict=False)
        if candidates
        else None
    )


def _daily_name(case_name: str, iso_date: str) -> str:
    return f"{case_name}_inference_{iso_date.replace('-', '')}.nc"


def _inventory(
    output_path: Path, case_name: str, expected_dates: Sequence[str]
) -> Dict[str, Any]:
    expected_names = {
        _daily_name(case_name, value): value for value in expected_dates
    }
    actual = sorted(
        path for path in output_path.glob("*.nc") if path.is_file()
    )
    actual_names = {path.name for path in actual}
    date_pattern = re.compile(
        rf"^{re.escape(case_name)}_inference_(\d{{8}})(.*?)\.nc$"
    )
    by_date: Dict[str, list[str]] = {}
    malformed: list[str] = []
    for path in actual:
        match = date_pattern.fullmatch(path.name)
        if match is None:
            malformed.append(path.name)
            continue
        token = match.group(1)
        iso_date = f"{token[:4]}-{token[4:6]}-{token[6:8]}"
        try:
            datetime.strptime(iso_date, "%Y-%m-%d")
        except ValueError:
            malformed.append(path.name)
            continue
        by_date.setdefault(iso_date, []).append(path.name)

    missing = [name for name in expected_names if name not in actual_names]
    extra = [
        name for name in sorted(actual_names) if name not in expected_names
    ]
    duplicates = {
        value: names
        for value, names in sorted(by_date.items())
        if len(names) > 1
    }
    return {
        "expected_count": len(expected_dates),
        "daily_file_count": len(actual),
        "missing_files": missing,
        "extra_files": extra,
        "duplicate_dates": duplicates,
        "malformed_files": malformed,
        "complete": not missing
        and not extra
        and not duplicates
        and not malformed,
    }


def _attribute_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _same_path(left: str | Path, right: Path) -> bool:
    try:
        return Path(left).expanduser().resolve(strict=False) == right.resolve(
            strict=False
        )
    except (OSError, RuntimeError):
        return str(left) == str(right)


def _validate_daily_file(  # noqa: C901
    path: Path,
    *,
    expected_date: str,
    expected_variables: Sequence[str],
    case_name: str,
    split: str,
    split_start: str,
    split_end: str,
    checkpoint: Optional[Path],
    reference_grid_fingerprint: Optional[str],
    reference_mask_digest: Optional[str],
    reference_provenance: Optional[Mapping[str, str]],
) -> Dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    result: Dict[str, Any] = {"file": path.name, "date": expected_date}
    try:
        with xr.open_dataset(path) as dataset:
            if "lat" not in dataset.coords or "lon" not in dataset.coords:
                errors.append("missing lat/lon coordinates")
                return {
                    **result,
                    "valid": False,
                    "errors": errors,
                    "warnings": warnings,
                }
            if dataset["lat"].dims != ("lat",):
                errors.append(
                    f"lat dimensions {dataset['lat'].dims!r} != ('lat',)"
                )
            if dataset["lon"].dims != ("lon",):
                errors.append(
                    f"lon dimensions {dataset['lon'].dims!r} != ('lon',)"
                )
            grid = validate_prism_grid(
                dataset["lat"].values,
                dataset["lon"].values,
                context=str(path),
            )
            result["grid_fingerprint"] = grid.fingerprint
            result["grid_shape"] = list(grid.shape)
            if (
                reference_grid_fingerprint
                and grid.fingerprint != reference_grid_fingerprint
            ):
                errors.append(
                    "coordinate grid differs from the first validated output"
                )
            attr_grid = _attribute_text(
                dataset.attrs.get("prism_grid_fingerprint", "")
            )
            if attr_grid != grid.fingerprint:
                errors.append(
                    "prism_grid_fingerprint does not match coordinates"
                )

            observed_variables = [
                name for name in dataset.data_vars if name != MASK_VARIABLE
            ]
            if observed_variables != list(expected_variables):
                expected_variable_list = list(expected_variables)
                errors.append(
                    "variable order "
                    f"{observed_variables!r} != {expected_variable_list!r}"
                )
            if MASK_VARIABLE not in dataset.data_vars:
                errors.append(f"missing {MASK_VARIABLE}")
            else:
                if dataset[MASK_VARIABLE].dims != ("lat", "lon"):
                    errors.append(
                        f"{MASK_VARIABLE} dimensions "
                        f"{dataset[MASK_VARIABLE].dims!r} != ('lat', 'lon')"
                    )
                mask = np.asarray(dataset[MASK_VARIABLE].values)
                if mask.shape != grid.shape:
                    errors.append(
                        f"mask shape {mask.shape} != grid shape {grid.shape}"
                    )
                if not np.isin(mask, (0, 1, False, True)).all():
                    errors.append("mask contains values other than 0/1")
                mask_bool = mask.astype(bool)
                mask_digest = _mask_content_sha256(mask_bool)
                result["mask_content_sha256"] = mask_digest
                if (
                    reference_mask_digest
                    and mask_digest != reference_mask_digest
                ):
                    errors.append(
                        "mask cells differ from the first validated output"
                    )
                if (
                    _attribute_text(dataset.attrs.get(MASK_CONTENT_ATTR, ""))
                    != mask_digest
                ):
                    errors.append(
                        f"{MASK_CONTENT_ATTR} does not match mask cells"
                    )

                for variable in expected_variables:
                    if variable not in dataset:
                        continue
                    if dataset[variable].dims != ("time", "lat", "lon"):
                        errors.append(
                            f"{variable} dimensions {dataset[variable].dims!r} "
                            "!= ('time', 'lat', 'lon')"
                        )
                    values = np.asarray(dataset[variable].values)
                    if values.shape != (1, *grid.shape):
                        expected_shape = (1, *grid.shape)
                        errors.append(
                            f"{variable} shape {values.shape} != "
                            f"{expected_shape}"
                        )
                        continue
                    if not np.isfinite(values[0][mask_bool]).all():
                        errors.append(
                            f"{variable} has non-finite values on valid cells"
                        )
                    if np.isfinite(values[0][~mask_bool]).any():
                        errors.append(
                            f"{variable} has finite values on invalid mask "
                            "cells"
                        )
                    if variable == "ppt" and np.any(
                        values[0][mask_bool] < -1.0e-6
                    ):
                        errors.append("ppt has negative values on valid cells")
                    expected_unit = VARIABLE_UNITS.get(variable)
                    if (
                        expected_unit
                        and _attribute_text(
                            dataset[variable].attrs.get("units", "")
                        )
                        != expected_unit
                    ):
                        errors.append(
                            f"{variable} units are not configured physical "
                            f"units {expected_unit!r}"
                        )

            if dataset.sizes.get("time") != 1:
                errors.append(f"time size {dataset.sizes.get('time')} != 1")
            elif "time" not in dataset.coords:
                errors.append("missing time coordinate")
            else:
                if dataset["time"].dims != ("time",):
                    errors.append(
                        f"time dimensions {dataset['time'].dims!r} != ('time',)"
                    )
                observed_date = np.datetime_as_string(
                    np.asarray(dataset["time"].values).reshape(-1)[0], unit="D"
                )
                if observed_date != expected_date:
                    errors.append(
                        f"time {observed_date} != filename date "
                        f"{expected_date}"
                    )

            expected_attrs = {
                "case_name": case_name,
                "dataset_split": split,
                "split_start": split_start,
                "split_end": split_end,
            }
            for name, expected in expected_attrs.items():
                observed = _attribute_text(dataset.attrs.get(name, ""))
                if observed != expected:
                    errors.append(
                        f"attribute {name}={observed!r} != {expected!r}"
                    )
            for name in REQUIRED_PROVENANCE_ATTRS:
                if not _attribute_text(dataset.attrs.get(name, "")).strip():
                    errors.append(f"missing provenance attribute {name}")
            result["provenance"] = {
                name: _attribute_text(dataset.attrs.get(name, ""))
                for name in (
                    "case_name",
                    "dataset_split",
                    "split_start",
                    "split_end",
                    "prism_grid_fingerprint",
                    *REQUIRED_PROVENANCE_ATTRS,
                )
            }
            if reference_provenance is not None:
                for name, expected in reference_provenance.items():
                    observed = result["provenance"].get(name, "")
                    if observed != expected:
                        errors.append(
                            f"provenance attribute {name}={observed!r} "
                            f"differs from first validated output {expected!r}"
                        )
            if (
                _attribute_text(
                    dataset.attrs.get("target_valid_mask_source_split", "")
                )
                != "training"
            ):
                errors.append("target-valid mask is not training-derived")
            if (
                _attribute_text(
                    dataset.attrs.get("target_valid_mask_grid_fingerprint", "")
                )
                != grid.fingerprint
            ):
                errors.append(
                    "target-valid-mask grid fingerprint does not match "
                    "coordinates"
                )
            checkpoint_attr = _attribute_text(
                dataset.attrs.get("checkpoint", "")
            )
            result["checkpoint"] = checkpoint_attr
            if not checkpoint_attr:
                errors.append("missing checkpoint attribute")
            elif checkpoint is not None and not _same_path(
                checkpoint_attr, checkpoint
            ):
                errors.append(
                    f"checkpoint {checkpoint_attr!r} != configured "
                    f"{str(checkpoint)!r}"
                )
    except Exception as exc:
        errors.append(f"cannot validate NetCDF: {type(exc).__name__}: {exc}")
    result.update(valid=not errors, errors=errors, warnings=warnings)
    return result


def _validation_dates(expected_dates: Sequence[str], level: str) -> list[str]:
    if level == "none" or not expected_dates:
        return []
    if level == "all":
        return list(expected_dates)
    return list(dict.fromkeys((expected_dates[0], expected_dates[-1])))


def audit_outputs(
    config_path: str | Path,
    *,
    output_dir: Optional[str | Path] = None,
    checkpoint_path: Optional[str | Path] = None,
    split: str = "inference",
    validation_level: str = "endpoints",
) -> Dict[str, Any]:
    """Return a JSON-serializable run-level output audit."""
    if validation_level not in {"none", "endpoints", "all"}:
        raise ValueError("validation_level must be none, endpoints, or all")
    config_path = Path(config_path).expanduser().resolve()
    cfg = load_yaml(config_path)
    case_name = get_case_name(cfg)
    expected_dates = expected_output_dates(cfg, split)
    output_path = resolve_output_path(cfg, output_dir=output_dir, split=split)
    checkpoint = _configured_checkpoint(cfg, checkpoint_path)
    checkpoint_exists = bool(checkpoint and checkpoint.is_file())
    checkpoint_sha256 = (
        _sha256_file(checkpoint)
        if checkpoint_exists and validation_level != "none"
        else None
    )
    inventory = _inventory(output_path, case_name, expected_dates)

    validations: list[Dict[str, Any]] = []
    reference_grid: Optional[str] = None
    reference_mask: Optional[str] = None
    reference_provenance: Optional[Dict[str, str]] = None
    for iso_date in _validation_dates(expected_dates, validation_level):
        path = output_path / _daily_name(case_name, iso_date)
        if not path.is_file():
            validations.append(
                {
                    "file": path.name,
                    "date": iso_date,
                    "valid": False,
                    "errors": ["expected daily file is missing"],
                    "warnings": [],
                }
            )
            continue
        validation = _validate_daily_file(
            path,
            expected_date=iso_date,
            expected_variables=list(
                (cfg.get("data", {}) or {}).get("target_variables", [])
            ),
            case_name=case_name,
            split=split,
            split_start=expected_dates[0],
            split_end=expected_dates[-1],
            checkpoint=checkpoint,
            reference_grid_fingerprint=reference_grid,
            reference_mask_digest=reference_mask,
            reference_provenance=reference_provenance,
        )
        validations.append(validation)
        if validation.get("valid"):
            reference_grid = reference_grid or validation.get(
                "grid_fingerprint"
            )
            reference_mask = reference_mask or validation.get(
                "mask_content_sha256"
            )
            reference_provenance = reference_provenance or validation.get(
                "provenance"
            )

    full_validation = (
        validation_level == "all"
        and len(validations) == len(expected_dates)
    )
    validation_ok = bool(
        full_validation
        and validations
        and all(item["valid"] for item in validations)
    )
    complete = bool(
        inventory["complete"] and checkpoint_exists and validation_ok
    )
    status = (
        "complete"
        if complete
        else "incomplete"
        if not inventory["complete"]
        else "missing_checkpoint"
        if not checkpoint_exists
        else "unvalidated"
        if validation_level == "none"
        else "partially_validated"
        if validation_level != "all"
        else "invalid"
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "audited_at": _utc_now(),
        "status": status,
        "complete": complete,
        "config": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "case_name": case_name,
        "split": split,
        "split_start": expected_dates[0],
        "split_end": expected_dates[-1],
        "output_path": str(output_path),
        "configured_checkpoint": str(checkpoint) if checkpoint else None,
        "configured_checkpoint_exists": checkpoint_exists,
        "configured_checkpoint_sha256": checkpoint_sha256,
        "target_variables": list(
            (cfg.get("data", {}) or {}).get("target_variables", [])
        ),
        "validation_level": validation_level,
        "inventory": inventory,
        "validated_files": validations,
    }


def _manifest_file_records(
    output_path: Path, case_name: str, dates: Iterable[str]
) -> tuple[list[Dict[str, Any]], str]:
    records: list[Dict[str, Any]] = []
    digest = hashlib.sha256()
    digest.update(b"granitewxc-narr-prism-output-inventory-v1\0")
    for iso_date in dates:
        path = output_path / _daily_name(case_name, iso_date)
        stat = path.stat()
        record = {
            "date": iso_date,
            "name": path.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        records.append(record)
        digest.update(
            f"{record['date']}\0{record['name']}\0{record['size']}\0{record['mtime_ns']}\n".encode(
                "utf-8"
            )
        )
    return records, digest.hexdigest()


def write_output_manifest(
    report: Mapping[str, Any], destination: Optional[str | Path] = None
) -> Path:
    """Atomically publish a manifest, rejecting incomplete/invalid audits."""
    expected_count = int(report.get("inventory", {}).get("expected_count", 0))
    validated_count = len(report.get("validated_files", ()))
    if (
        not report.get("complete")
        or report.get("validation_level") != "all"
        or validated_count != expected_count
        or not report.get("configured_checkpoint_sha256")
    ):
        raise ValueError(
            "refusing to write a manifest without a complete all-file audit "
            "and an authenticated checkpoint"
        )
    output_path = Path(str(report["output_path"]))
    destination_path = (
        Path(destination).expanduser().resolve()
        if destination is not None
        else output_path / MANIFEST_NAME
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    records, signature = _manifest_file_records(
        output_path,
        str(report["case_name"]),
        expected_output_dates(
            load_yaml(str(report["config"])), str(report["split"])
        ),
    )
    payload = dict(report)
    payload["manifest_written_at"] = _utc_now()
    payload["files"] = records
    payload["inventory_sha256"] = signature
    temporary = destination_path.with_name(
        f".{destination_path.name}.tmp-{os.getpid()}"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit deterministic NARR-PRISM daily inference outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True, help="NARR-PRISM YAML config"
    )
    parser.add_argument(
        "--output-dir", default=None, help="Override YAML output root"
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Expected checkpoint path"
    )
    parser.add_argument(
        "--split", choices=PREDICTION_SPLITS, default="inference"
    )
    parser.add_argument(
        "--validation-level",
        choices=("none", "endpoints", "all"),
        default="endpoints",
        help=(
            "NetCDF content validation scope; inventory always checks every "
            "date"
        ),
    )
    parser.add_argument(
        "--write-manifest",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "atomically write a manifest after a successful audit; omit "
            f"PATH for <output>/{MANIFEST_NAME}"
        ),
    )
    parser.add_argument(
        "--json-out", default=None, help="Optional atomic JSON report path"
    )
    return parser.parse_args(argv)


def _write_report_atomic(
    report: Mapping[str, Any], destination: str | Path
) -> Path:
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = audit_outputs(
        args.config,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        split=args.split,
        validation_level=args.validation_level,
    )
    if args.json_out is not None:
        _write_report_atomic(report, args.json_out)
    if args.write_manifest is not None:
        if not report["complete"]:
            print(json.dumps(report, indent=2, sort_keys=True))
            print(
                "[output-audit] refusing manifest: output set is incomplete "
                "or invalid",
                file=sys.stderr,
            )
            return 2
        destination = args.write_manifest or None
        path = write_output_manifest(report, destination)
        report = dict(report)
        report["manifest_path"] = str(path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
