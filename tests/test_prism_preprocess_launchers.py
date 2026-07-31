from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = (
    (
        REPO_ROOT / "examples" / "NARR_PRISM" / "run_preprocess.sh",
        "preproc_narr_prism.py",
        "compute_scalars_narr_prism.py",
    ),
    (
        REPO_ROOT / "examples" / "MERRA_PRISM" / "run_preprocess.sh",
        "preproc_merra_prism.py",
        "compute_scalars_merra_prism.py",
    ),
)


def _stub_python(tmp_path: Path) -> Path:
    stub = tmp_path / "python-stub"
    stub.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
  printf '%s\n' "${HAS_VALIDATION:-1}"
  exit 0
fi
printf '%s\n' "$*" >> "$CALL_LOG"
if [[ "${FAIL_TRAIN_SHARD:-}" == "0" && "$*" == *"--mode training"* ]]; then
  if [[ "$*" == *"--date-shard-index 0"* ]]; then
    exit 7
  fi
  exec sleep 30
fi
""",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


@pytest.mark.parametrize("launcher,preprocessor,scalar_script", LAUNCHERS)
def test_launcher_recomputes_scalars_between_training_and_other_splits(
    tmp_path: Path,
    launcher: Path,
    preprocessor: str,
    scalar_script: str,
) -> None:
    # Pass a caller-relative path while the launcher later changes to REPO_ROOT.
    config = tmp_path / "case.yaml"
    config.write_text("case_name: launcher-test\n", encoding="utf-8")
    log = tmp_path / "calls.log"
    env = {
        **os.environ,
        "PYTHON_BIN": str(_stub_python(tmp_path)),
        "CALL_LOG": str(log),
        "HAS_VALIDATION": "1",
    }
    result = subprocess.run(
        [
            "bash",
            str(launcher),
            "--config",
            config.name,
            "--shards",
            "1",
            "--overwrite",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 4
    assert preprocessor in calls[0] and "--mode training" in calls[0]
    assert scalar_script in calls[1]
    assert preprocessor in calls[2] and "--mode validation" in calls[2]
    assert preprocessor in calls[3] and "--mode inference" in calls[3]
    assert "--overwrite" in calls[0]
    assert "--overwrite" not in calls[1]
    assert "Scalars            : always recomputed" in result.stdout


@pytest.mark.parametrize("launcher,preprocessor,scalar_script", LAUNCHERS)
def test_failed_training_shard_aborts_before_scalar_recomputation(
    tmp_path: Path,
    launcher: Path,
    preprocessor: str,
    scalar_script: str,
) -> None:
    config = tmp_path / "case.yaml"
    config.write_text("case_name: launcher-test\n", encoding="utf-8")
    log = tmp_path / "calls.log"
    env = {
        **os.environ,
        "PYTHON_BIN": str(_stub_python(tmp_path)),
        "CALL_LOG": str(log),
        "HAS_VALIDATION": "0",
        "FAIL_TRAIN_SHARD": "0",
    }
    result = subprocess.run(
        ["bash", str(launcher), config.name, "--shards", "2"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode != 0
    calls = log.read_text(encoding="utf-8").splitlines()
    # The failing shard can be reaped before its sibling reaches the logging
    # statement, so the exact number of observed training calls is scheduler-
    # dependent.  What matters is that no later stage is started.
    assert 1 <= len(calls) <= 2
    assert all(preprocessor in call and "--mode training" in call for call in calls)
    assert all(scalar_script not in call for call in calls)
    assert "scalars were not recomputed" in result.stderr


@pytest.mark.parametrize("launcher,_,__", LAUNCHERS)
def test_launcher_rejects_partial_validation_range(
    tmp_path: Path, launcher: Path, _: str, __: str
) -> None:
    config = tmp_path / "partial.yaml"
    config.write_text(
        "dates:\n  validation:\n    start: '2014-01-01'\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(launcher), "--config", str(config)],
        env={**os.environ, "PYTHON_BIN": sys.executable},
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "dates.validation must define both start and end" in result.stderr


@pytest.mark.parametrize("launcher,_,__", LAUNCHERS)
def test_launcher_rejects_zero_shards(
    tmp_path: Path, launcher: Path, _: str, __: str
) -> None:
    result = subprocess.run(
        ["bash", str(launcher), "--shards", "0"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "must be a positive integer" in result.stderr
