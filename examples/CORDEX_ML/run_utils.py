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
        return json.load(handle)


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
