from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

import yaml


@dataclass
class RunPaths:
    """Container describing the directory layout for a single fine-tune run."""

    run_dir: Path
    checkpoints: Path
    scalars: Path
    preproc: Path
    config_dir: Path
    manifest_path: Path


def _rebase_manifest_paths(payload: Any, old_root: Path, new_root: Path) -> Any:
    """Return payload with paths rooted at old_root rewritten under new_root."""

    if isinstance(payload, str):
        path = Path(payload).expanduser()
        if not path.is_absolute():
            return payload
        try:
            relative = path.relative_to(old_root)
        except ValueError:
            return payload
        return str((new_root / relative).resolve(strict=False))
    if isinstance(payload, dict):
        return {key: _rebase_manifest_paths(value, old_root, new_root) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_rebase_manifest_paths(value, old_root, new_root) for value in payload]
    return payload


def _infer_old_root_from_config(data: Any, actual_run_dir: Path) -> Path | None:
    """Best-effort detection of the previous run root recorded in the config."""

    if not isinstance(data, dict):
        return None

    candidate_keys = ("path_experiment", "checkpoint_dir", "scalar_dir", "preproc_dir")
    for key in candidate_keys:
        value = data.get(key)
        if not isinstance(value, str):
            continue
        path = Path(value).expanduser()
        if path != actual_run_dir:
            return path
    return None


def _rebase_config_snapshot(snapshot_path: Path, old_root: Path, new_root: Path) -> None:
    """Rewrite the YAML config snapshot if it still references the old run root."""

    snapshot_path = snapshot_path.expanduser()
    if not snapshot_path.exists():
        return

    with open(snapshot_path, "r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle)

    config_data = config_data or {}
    rebase_from = old_root
    if rebase_from.resolve(strict=False) == new_root.resolve(strict=False):
        inferred = _infer_old_root_from_config(config_data, new_root.resolve(strict=False))
        if inferred:
            rebase_from = inferred

    updated = _rebase_manifest_paths(config_data, rebase_from, new_root)
    if updated != config_data:
        with open(snapshot_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(updated, handle, sort_keys=False)


def prepare_run_paths(root: Path, run_name: str) -> RunPaths:
    """Create the directory structure for a run and return the corresponding paths."""

    run_dir = root / run_name
    checkpoints = run_dir / "checkpoints"
    scalars = run_dir / "scalars"
    preproc = run_dir / "preproc"
    config_dir = run_dir / "config"
    manifest_path = run_dir / "run_manifest.json"

    for directory in (run_dir, checkpoints, scalars, preproc, config_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_dir=run_dir,
        checkpoints=checkpoints,
        scalars=scalars,
        preproc=preproc,
        config_dir=config_dir,
        manifest_path=manifest_path,
    )


def _copy_if_needed(src_path: Optional[str], destination_dir: Path) -> Optional[str]:
    if not src_path:
        return None
    src = Path(src_path).expanduser()
    if not src.exists():
        return None
    destination_dir.mkdir(parents=True, exist_ok=True)
    dest = destination_dir / src.name
    try:
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
    except FileNotFoundError:
        shutil.copy2(src, dest)
    return str(dest.resolve())


def copy_scalars_to_run(config: Any, scalar_dir: Path) -> Dict[str, Any]:
    """Copy model/data scalar files into the run directory and re-point the config."""

    mapping: Dict[str, Any] = {}
    model_attr_names = ("input_mu", "input_sigma", "target_mu", "target_sigma")
    for attr_name in model_attr_names:
        value = getattr(config.model, attr_name, None)
        new_path = _copy_if_needed(value, scalar_dir)
        if new_path:
            setattr(config.model, attr_name, new_path)
            mapping[f"model.{attr_name}"] = new_path

    scalers: Optional[Mapping[str, Any]] = getattr(config.data, "scalers", None)
    if isinstance(scalers, MutableMapping):
        new_scalers: Dict[str, str] = {}
        for key, path_str in scalers.items():
            new_path = _copy_if_needed(path_str, scalar_dir)
            if new_path:
                new_scalers[key] = new_path
        if new_scalers:
            config.data.scalers = new_scalers
            mapping["data.scalers"] = new_scalers

    return mapping


def snapshot_resolved_config(config: Any, path: Path) -> Path:
    """Persist the fully-resolved ExperimentConfig to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
    return path


def write_manifest(
    run_paths: RunPaths,
    *,
    run_name: str,
    base_config: Optional[str],
    resolved_config: Path,
    scalars: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist a JSON manifest listing the critical run artifacts."""

    manifest = {
        "run_name": run_name,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "run_dir": str(run_paths.run_dir),
        "checkpoint_dir": str(run_paths.checkpoints),
        "scalar_dir": str(run_paths.scalars),
        "preproc_dir": str(run_paths.preproc),
        "config_snapshot": str(resolved_config),
        "base_config": base_config,
        "scalars": scalars,
    }
    if extra:
        manifest.update(extra)

    with open(run_paths.manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest


def load_run_manifest(run_dir: Path) -> Dict[str, Any]:
    """Load the manifest JSON for a completed/ongoing run."""

    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    recorded = Path(manifest.get("run_dir") or run_dir).expanduser()
    actual_run_dir = run_dir.expanduser().resolve()
    if not recorded.is_absolute():
        recorded_run_dir = actual_run_dir
    else:
        recorded_run_dir = recorded.resolve(strict=False)

    need_rebase = recorded_run_dir != actual_run_dir
    if need_rebase:
        manifest = _rebase_manifest_paths(manifest, recorded_run_dir, actual_run_dir)
        manifest["run_dir"] = str(actual_run_dir)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
    snapshot = Path(manifest["config_snapshot"])
    _rebase_config_snapshot(snapshot, recorded_run_dir if need_rebase else actual_run_dir, actual_run_dir)

    return manifest


def update_manifest(run_dir: Path, **updates: Any) -> Dict[str, Any]:
    """Merge additional keys into the manifest."""

    manifest = load_run_manifest(run_dir)
    manifest.update(updates)
    manifest_path = Path(manifest["run_dir"]) / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def resolve_checkpoint_path(manifest: Mapping[str, Any], preference: str = "best") -> Path:
    """Determine the checkpoint path to load for inference."""

    checkpoint_dir = Path(manifest["checkpoint_dir"])
    candidates = {
        "best": checkpoint_dir / "best.ckpt",
        "last": checkpoint_dir / "last.ckpt",
    }
    preferred = candidates.get(preference, candidates["best"])
    if preferred.exists():
        return preferred

    for path in candidates.values():
        if path.exists():
            return path
    raise FileNotFoundError(f"No checkpoint available under {checkpoint_dir}")


def assert_no_eccc_reference(*paths: Path) -> None:
    """Fail loudly if any artifact path still references the ECCC baseline assets."""

    for path in paths:
        text = str(path)
        if "eccc" in text.lower():
            raise RuntimeError(
                f"Path {path} still references the ECCC example assets. "
                "Ensure the NZ fine-tune artifacts are being used."
            )
