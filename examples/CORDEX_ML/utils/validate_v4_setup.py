#!/usr/bin/env python3
"""Lightweight validation utility for CORDEX v4 configuration and artifacts.

This script is intentionally robust to minimal environments. It always runs:
- Python syntax checks
- YAML parsing checks
- v4 YAML backup presence checks
- runs_v4 copy policy checks (no prediction artifacts)
- precipitation log1p round-trip checks

Optional checks that require installed dependencies:
- checkpoint readability via torch.load
"""

from __future__ import annotations

import argparse
import json
import math
import os
import py_compile
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
CORDEX_ROOT = REPO_ROOT / "examples" / "CORDEX_ML"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate CORDEX v4 YAML/runtime setup",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-json",
        default=str(CORDEX_ROOT / "evaluations" / "v4_validation_summary.json"),
        help="Where to write validation summary JSON.",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=20,
        help="Maximum number of checkpoints to probe for readability when torch is available.",
    )
    return parser.parse_args()


def _status(ok: bool, details: Any) -> dict[str, Any]:
    return {"ok": bool(ok), "details": details}


def _collect_python_files() -> list[Path]:
    return sorted(CORDEX_ROOT.rglob("*.py"))


def _collect_yaml_files() -> list[Path]:
    return sorted(CORDEX_ROOT.rglob("*.yaml"))


def _check_py_compile() -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    files = _collect_python_files()
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append({"file": str(path), "error": str(exc)})
    return _status(
        ok=(len(failures) == 0),
        details={
            "checked_files": len(files),
            "failed_files": failures,
        },
    )


def _check_yaml_parse() -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    files = _collect_yaml_files()
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                yaml.safe_load(handle)
        except Exception as exc:  # noqa: BLE001
            failures.append({"file": str(path), "error": str(exc)})
    return _status(
        ok=(len(failures) == 0),
        details={
            "checked_files": len(files),
            "failed_files": failures,
        },
    )


def _is_v4_yaml(path: Path) -> bool:
    return path.stem.endswith("_v4") or path.name.endswith(".v4.yaml")


def _expected_v4_yaml(path: Path) -> Path:
    return path.with_name(f"{path.stem}_v4{path.suffix}")


def _check_v4_yaml_backups() -> dict[str, Any]:
    yaml_files = _collect_yaml_files()
    base_files = [p for p in yaml_files if not _is_v4_yaml(p)]
    missing: list[str] = []
    for base in base_files:
        expected = _expected_v4_yaml(base)
        if not expected.exists():
            missing.append(str(expected))
    return _status(
        ok=(len(missing) == 0),
        details={
            "base_yaml_files": len(base_files),
            "missing_v4_backups": missing,
        },
    )


def _is_prediction_artifact(path: Path) -> bool:
    low = path.name.lower()
    banned_tokens = (
        "prediction",
        "predictions",
        "inference_output",
        ".png",
        ".pdf",
    )
    if any(tok in low for tok in banned_tokens):
        return True
    # Keep strict for common netcdf prediction outputs.
    if low.endswith(".nc") and "checkpoint" not in low:
        return True
    return False


def _check_runs_v4_policy() -> dict[str, Any]:
    runs_v4_dirs = sorted(CORDEX_ROOT.rglob("runs_v4"))
    if not runs_v4_dirs:
        return _status(False, {"error": "No runs_v4 directories found."})

    disallowed: list[str] = []
    total_files = 0
    for runs_v4 in runs_v4_dirs:
        for file_path in runs_v4.rglob("*"):
            if file_path.is_file():
                total_files += 1
                if _is_prediction_artifact(file_path):
                    disallowed.append(str(file_path))

    return _status(
        ok=(len(disallowed) == 0),
        details={
            "runs_v4_dirs": [str(p) for p in runs_v4_dirs],
            "checked_files": total_files,
            "disallowed_prediction_like_files": disallowed[:200],
            "disallowed_count": len(disallowed),
        },
    )


def _check_pr_roundtrip() -> dict[str, Any]:
    # Pure-Python scalar checks.
    values = [0.0, 1e-12, 1e-8, 1e-4, 0.1, 1.0, 10.0, 100.0]
    means = [0.0, 0.2, 1.5]
    stds = [1e-6, 0.1, 1.0, 3.0]
    max_abs_err = 0.0
    min_inverse_value = float("inf")

    for v in values:
        for mu in means:
            for sigma in stds:
                tx = math.log1p(max(v, 0.0))
                norm = (tx - mu) / sigma
                inv_tx = norm * sigma + mu
                inv = math.expm1(inv_tx)
                if inv < 0.0 and inv > -1e-7:
                    inv = 0.0
                max_abs_err = max(max_abs_err, abs(inv - v))
                min_inverse_value = min(min_inverse_value, inv)

    ok = (max_abs_err < 1e-10) and (min_inverse_value >= -1e-10)
    return _status(
        ok=ok,
        details={
            "max_abs_roundtrip_error": max_abs_err,
            "min_inverse_value": min_inverse_value,
            "checked_values": values,
        },
    )


def _check_checkpoint_readability(max_checkpoints: int) -> dict[str, Any]:
    checkpoints = sorted(
        p for p in CORDEX_ROOT.rglob("*.ckpt") if "runs_v4" in p.parts
    )
    if not checkpoints:
        return _status(False, {"error": "No .ckpt files found under runs_v4."})

    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return _status(
            False,
            {
                "skipped": True,
                "reason": f"torch unavailable: {exc}",
                "checkpoint_count": len(checkpoints),
            },
        )

    checked = 0
    failures: list[dict[str, str]] = []
    for ckpt in checkpoints[: max(1, max_checkpoints)]:
        try:
            torch.load(str(ckpt), map_location="cpu", weights_only=False)
            checked += 1
        except Exception as exc:  # noqa: BLE001
            failures.append({"file": str(ckpt), "error": str(exc)})

    return _status(
        ok=(len(failures) == 0),
        details={
            "total_checkpoints_found": len(checkpoints),
            "checkpoints_probed": checked + len(failures),
            "failures": failures,
        },
    )


def main() -> None:
    args = _parse_args()
    os.chdir(REPO_ROOT)

    summary = {
        "repo_root": str(REPO_ROOT),
        "python_executable": sys.executable,
        "checks": {
            "py_compile": _check_py_compile(),
            "yaml_parse": _check_yaml_parse(),
            "v4_yaml_backups": _check_v4_yaml_backups(),
            "runs_v4_policy": _check_runs_v4_policy(),
            "pr_roundtrip": _check_pr_roundtrip(),
            "checkpoint_readability": _check_checkpoint_readability(args.max_checkpoints),
        },
    }

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved validation summary: {output_json}")
    for name, result in summary["checks"].items():
        status = "PASS" if result.get("ok") else "FAIL/SKIP"
        print(f"[{status}] {name}")


if __name__ == "__main__":
    main()
