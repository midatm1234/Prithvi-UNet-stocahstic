"""Shared parameter helpers for the ALPS CORDEX fine-tune notebooks."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple


def _coerce_path(value: Any) -> Path:
    if isinstance(value, Path):
        return value.expanduser()
    return Path(str(value)).expanduser()


def _coerce_sequence(values: Sequence[Any] | None) -> list[Path]:
    if values is None:
        return []
    return [_coerce_path(item) for item in values]


def _extract_config_version(path_like: Any) -> int | None:
    if not path_like:
        return None
    stem = Path(str(path_like)).stem
    if "_v" not in stem:
        return None
    suffix = stem.rsplit("_v", 1)[-1]
    if suffix.isdigit():
        return int(suffix)
    return None


def _run_config_version(run_dir: Path) -> int | None:
    manifest_path = _coerce_path(run_dir) / "run_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception:
        return None
    return _extract_config_version(manifest.get("base_config") or manifest.get("config_snapshot"))


def _validate_run_version(run_dir: Path, requested_config_path: Path) -> None:
    requested_version = _extract_config_version(requested_config_path)
    if requested_version is None:
        return
    run_version = _run_config_version(run_dir)
    if run_version is None:
        return
    if run_version != requested_version:
        raise RuntimeError(
            "Config/run version mismatch: "
            f"requested config {requested_config_path} (v{requested_version}) but "
            f"selected run '{_coerce_path(run_dir).name}' is v{run_version}. "
            "Choose a matching run or config version."
        )


@dataclass
class UserParams:
    """User-editable parameters shared across ALPS_ CORDEX notebooks."""

    repo_root: Path
    project_dir: Path
    runs_root: Path
    config_path: Path
    run_name: str | None = None
    inference_run_name: str | None = None
    inference_output_root: Path | None = None
    inference_predictor_root: Path | None = None
    inference_target_root: Path | None = None
    checkpoint_path: Path | None = None
    preferred_checkpoint: str = "best"
    train_predictor_paths: Sequence[Path] | None = None
    train_target_paths: Sequence[Path] | None = None
    val_predictor_paths: Sequence[Path] | None = None
    val_target_paths: Sequence[Path] | None = None
    test_predictor_paths: Sequence[Path] | None = None
    test_target_paths: Sequence[Path] | None = None
    use_static: bool | None = None
    device_target: str = "cuda"
    batch_size: int | None = None
    num_workers: int | None = None
    env_run_name_var: str = "ALPS_RUN_NAME"
    env_inference_run_var: str = "ALPS_FINETUNE_RUN"
    notes: str | None = None

    def summary(self) -> str:
        """Return a human-readable summary of the resolved parameters."""

        def _count(values: Sequence[Any] | None) -> int:
            return len(values or [])

        lines = [
            f"Repo root: {self.repo_root}",
            f"Project dir: {self.project_dir}",
            f"Runs root: {self.runs_root}",
            f"Config: {self.config_path}",
            f"Run name override: {self.run_name or '<auto>'}",
            f"Inference run override: {self.inference_run_name or '<latest>'}",
            f"Preferred checkpoint: {self.preferred_checkpoint}",
            f"Use static inputs: {self.use_static if self.use_static is not None else '<config>'}",
            f"Device target: {self.device_target}",
            f"Batch size override: {self.batch_size if self.batch_size is not None else '<config>'}",
            f"Dataloader workers override: {self.num_workers if self.num_workers is not None else '<config>'}",
            f"Train predictors: {_count(self.train_predictor_paths)} file(s)",
            f"Validation predictors: {_count(self.val_predictor_paths)} file(s)",
            f"Test predictors: {_count(self.test_predictor_paths)} file(s)",
            f"Checkpoint path override: {self.checkpoint_path or '<auto>'}",
        ]
        if self.inference_output_root:
            lines.append(f"Inference output root: {self.inference_output_root}")
        if self.inference_predictor_root:
            lines.append(f"Inference predictor root: {self.inference_predictor_root}")
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)


def validate_paths(params: UserParams, *, require_inference: bool = False) -> None:
    """Verify that critical filesystem locations exist."""

    repo_root = _coerce_path(params.repo_root)
    project_dir = _coerce_path(params.project_dir)
    runs_root = _coerce_path(params.runs_root)
    config_path = _coerce_path(params.config_path)

    for path, desc, must_exist in (
        (repo_root, "repo_root", True),
        (project_dir, "project_dir", True),
        (config_path, "config_path", True),
    ):
        if must_exist and not path.exists():
            raise FileNotFoundError(f"{desc} does not exist: {path}")

    runs_root.mkdir(parents=True, exist_ok=True)

    def _check_sequence(name: str, values: Sequence[Path] | None) -> None:
        for path in _coerce_sequence(values):
            if not path.exists():
                raise FileNotFoundError(f"{name} entry does not exist: {path}")

    _check_sequence("train_predictor_paths", params.train_predictor_paths)
    _check_sequence("train_target_paths", params.train_target_paths)
    _check_sequence("val_predictor_paths", params.val_predictor_paths)
    _check_sequence("val_target_paths", params.val_target_paths)
    _check_sequence("test_predictor_paths", params.test_predictor_paths)
    _check_sequence("test_target_paths", params.test_target_paths)

    if require_inference:
        if params.inference_predictor_root and not _coerce_path(
            params.inference_predictor_root
        ).exists():
            raise FileNotFoundError(
                f"inference_predictor_root does not exist: {params.inference_predictor_root}"
            )


def resolve_run_dir(
    params: UserParams, config: Any, *, timestamp: datetime | None = None
) -> Tuple[str, Path]:
    """Determine a run name/root for a new fine-tune run."""

    runs_root = _coerce_path(params.runs_root)
    env_override = os.environ.get(params.env_run_name_var or "")
    run_name = env_override or params.run_name
    if not run_name:
        job_id = getattr(config, "job_id", "nz_finetune")
        run_name = str(job_id)
        append_ts = bool(getattr(config, "append_timestamp_to_run_name", False))
        env_append = os.environ.get("CORDEX_APPEND_RUN_TIMESTAMP", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if append_ts or env_append:
            stamp = (timestamp or datetime.utcnow()).strftime("%Y%m%d-%H%M%S")
            run_name = f"{run_name}_{stamp}"
    run_dir = runs_root / run_name
    return run_name, run_dir


def resolve_existing_run_dir(params: UserParams) -> Tuple[str, Path]:
    """Locate an existing fine-tune run directory for inference."""

    runs_root = _coerce_path(params.runs_root)
    requested_config_path = _coerce_path(params.config_path)
    env_override = os.environ.get(params.env_inference_run_var or "")
    run_name = env_override or params.inference_run_name
    if run_name:
        run_dir = runs_root / run_name
        manifest = run_dir / "run_manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"Requested run '{run_name}' not found or missing manifest under {runs_root}"
            )
        _validate_run_version(run_dir, requested_config_path)
        return run_name, run_dir

    candidates: list[Path] = []
    for path in runs_root.iterdir():
        if not path.is_dir():
            continue
        if not (path / "run_manifest.json").exists():
            continue
        candidates.append(path)
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No fine-tune runs with manifests found under {runs_root}")

    requested_version = _extract_config_version(requested_config_path)
    if requested_version is not None:
        matching = [path for path in candidates if _run_config_version(path) == requested_version]
        if not matching:
            available_versions = sorted(
                {
                    version
                    for version in (_run_config_version(path) for path in candidates)
                    if version is not None
                }
            )
            raise FileNotFoundError(
                f"No fine-tune runs with config version v{requested_version} found under {runs_root}. "
                f"Available versions: {available_versions or '<unknown>'}"
            )
        candidates = matching

    run_dir = candidates[0]
    _validate_run_version(run_dir, requested_config_path)
    return run_dir.name, run_dir


def resolve_checkpoint(params: UserParams, run_dir: Path) -> Path:
    """Resolve which checkpoint file to load for inference."""

    if params.checkpoint_path:
        checkpoint = _coerce_path(params.checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Explicit checkpoint not found: {checkpoint}")
        return checkpoint

    checkpoint_dir = _coerce_path(run_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    preference = (params.preferred_checkpoint or "best").lower()
    best_ckpt = checkpoint_dir / "best.ckpt"
    last_ckpt = checkpoint_dir / "last.ckpt"
    def _epoch_index(path: Path) -> int:
        token = path.stem[len("epoch_") :]
        return int(token) if token.isdigit() else -1

    epoch_ckpts = sorted(checkpoint_dir.glob("epoch_*.ckpt"), key=_epoch_index)
    latest_epoch_ckpt = epoch_ckpts[-1] if epoch_ckpts else None

    ordered_candidates: list[Path] = []
    if preference == "last":
        if latest_epoch_ckpt is not None:
            ordered_candidates.append(latest_epoch_ckpt)
        ordered_candidates.extend([last_ckpt, best_ckpt])
    else:
        ordered_candidates.append(best_ckpt)
        if latest_epoch_ckpt is not None:
            ordered_candidates.append(latest_epoch_ckpt)
        ordered_candidates.append(last_ckpt)

    for candidate in ordered_candidates:
        if candidate.exists():
            return candidate

    remaining = sorted(
        checkpoint_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True
    )
    if remaining:
        return remaining[0]
    raise FileNotFoundError(f"No checkpoints found under {checkpoint_dir}")


def export_params(
    params: UserParams, destination: Path, *, extra: Mapping[str, Any] | None = None
) -> Path:
    """Write the resolved parameter set to JSON for traceability."""

    destination = _coerce_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def _serialize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [_serialize(item) for item in value]
        return value

    payload = {key: _serialize(value) for key, value in asdict(params).items()}
    if extra:
        payload.update(extra)

    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return destination
