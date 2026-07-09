"""Shared normalization-scalar utilities for the ``*_PRISM`` downscaling examples.

This module defines a *single* normalization contract that is reused, without
branch-specific hacks, by the ``MERRA_PRISM`` and ``NARR_PRISM`` preprocessing,
training, validation, testing, and inference code paths:

* **Predictor** scalers are always **per channel** -- shape ``[n_variables]`` (a
  channel-only ``[C, 1, 1]`` is also tolerated).  Per-gridpoint ``[C, lat, lon]``
  or per-tile ``[C, tile_y, tile_x]`` predictor scalers are rejected, because the
  predictors are *coarse* fields and a per-gridpoint scaler bakes a
  low-resolution spatial field into the normalization -> blocky downscaled output.
* **Target** scalers may be either per channel ``[C]`` *or* per gridpoint
  ``[C, lat, lon]``.  Targets are native high-resolution PRISM, so a per-cell
  climatology is a genuinely high-resolution field: using it lets the model
  predict standardised *anomalies* while the fine-scale climatology is restored
  deterministically (the classic BCSD / MOS downscaling paradigm).  This is the
  only place where a spatial scaler is allowed, and only for targets.
* Mean/std are computed from the *training split only*, ignoring NaNs (that part
  lives in the ``compute_scalars_*`` scripts; this module enforces the shapes and
  records provenance).
* The *same* saved files are reused everywhere.  They are resolved under a
  per-``case_name`` directory (``<preprocessed_dir>/<case_name>/scalars``) and a
  manifest (sha256 + shapes) lets training and inference assert byte-for-byte
  identity.  Scalers are **never** read from a shared/flat directory that is not
  scoped to the active ``case_name``.

The helpers accept either a raw YAML ``dict`` (used by the ``compute_scalars_*``
scripts) or a parsed ``ExperimentConfig``-like object (used by training and
inference), so a single code path serves every stage.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

SCALAR_NAMES = ("inputs_mean", "inputs_std", "targets_mean", "targets_std")
MANIFEST_NAME = "normalization_manifest.json"

# Repo root = .../granitewxc/utils/normalization.py -> parents[2]. Relative scaler
# directories are resolved against this (NOT the CWD) so preprocessing, training,
# and inference agree on the same absolute path regardless of where they run from.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_repo_relative(raw: str) -> Path:
    p = Path(os.path.expanduser(str(raw)))
    if p.is_absolute():
        return p
    return (_REPO_ROOT / p).resolve()


# ---------------------------------------------------------------------------
# Config accessors (work for both raw dict and ExperimentConfig-like objects)
# ---------------------------------------------------------------------------

def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def get_case_name(cfg: Any) -> str:
    case_name = _get(cfg, "case_name")
    if not case_name:
        raise ValueError("case_name must be set in the YAML config")
    return str(case_name)


def _scalar_dir_base(cfg: Any) -> Optional[str]:
    data = _get(cfg, "data", {}) or {}
    base = _get(data, "scalar_dir")
    return str(base) if base else None


def _preprocessed_base(cfg: Any) -> Path:
    """Root under which every case-scoped preprocessing artifact lives.

    Read from ``data.preprocessed_dir`` (default ``./preprocessed``) and resolved
    repo-relative so preprocessing, training, and inference agree regardless of
    the working directory.
    """
    data = _get(cfg, "data", {}) or {}
    base = _get(data, "preprocessed_dir", "./preprocessed") or "./preprocessed"
    return _resolve_repo_relative(str(base))


def case_preprocess_dir(cfg: Any) -> Path:
    """Canonical case-scoped preprocessing root ``<preprocessed_dir>/<case_name>``.

    All preprocessing outputs for a given YAML/case (normalized predictors,
    normalized targets, scalars, cached/tiled artifacts) are isolated here so
    reruns for a different case never overwrite another case's outputs.
    """
    base = _preprocessed_base(cfg)
    case_name = get_case_name(cfg)
    return base if base.name == case_name else base / case_name


def _has_all_scalars(directory: Path) -> bool:
    return all((directory / f"{name}.npy").exists() for name in SCALAR_NAMES)


# ---------------------------------------------------------------------------
# Directory resolution (single source of truth for train / val / test / infer)
# ---------------------------------------------------------------------------

def resolve_scalar_dir(cfg: Any, *, for_writing: bool = False) -> Path:
    """Resolve the case-scoped scalar directory for the active ``case_name``.

    The canonical location is ``<preprocessed_dir>/<case_name>/scalars`` so every
    normalization artifact is isolated per case and reruns never clobber another
    case's scalers.

    * When ``for_writing`` is ``True`` (``compute_scalars_*``) the canonical
      directory is always returned.
    * When reading (training / inference) the canonical directory is preferred.
      If it does not yet contain scalers, we fall back **only** to the
      *same-case* legacy location ``<scalar_dir>/<case_name>`` (previously
      computed runs). We NEVER fall back to a shared/flat directory that is not
      scoped to this ``case_name`` -- doing so would silently mix scalers across
      cases (the historical source of per-gridpoint shape errors).
    """
    canonical = case_preprocess_dir(cfg) / "scalars"
    if for_writing:
        return canonical
    if _has_all_scalars(canonical):
        return canonical

    # Same-case-only legacy fallback: <scalar_dir>/<case_name>. This is safe
    # because it is still scoped to THIS case; it can never return another
    # case's (or a shared flat) directory.
    legacy_raw = _scalar_dir_base(cfg)
    if legacy_raw:
        case_name = get_case_name(cfg)
        legacy_base = _resolve_repo_relative(legacy_raw)
        legacy_case = (
            legacy_base if legacy_base.name == case_name else legacy_base / case_name
        )
        if legacy_case != canonical and _has_all_scalars(legacy_case):
            print(
                f"[normalization] using legacy case-scoped scalars at {legacy_case}; "
                f"recompute to migrate them under {canonical}"
            )
            return legacy_case

    # No scalers found anywhere case-scoped. Return the canonical (missing)
    # directory so downstream code raises a clear, case-specific error instead
    # of silently reading another case's scalers.
    return canonical


def assert_scalars_available(cfg: Any, *, role: str = "") -> Path:
    """Return the resolved scalar dir, or raise a clear error if incomplete.

    Use at stages that strictly require precomputed scalers (fine-tuning,
    inference). The dataset itself stays tolerant of missing scalers so the very
    first ``compute_scalars_*`` pass can still construct it.
    """
    directory = resolve_scalar_dir(cfg, for_writing=False)
    if not _has_all_scalars(directory):
        case_name = get_case_name(cfg)
        missing = [
            f"{name}.npy"
            for name in SCALAR_NAMES
            if not (directory / f"{name}.npy").exists()
        ]
        raise FileNotFoundError(
            f"Normalization scalers are missing for case '{case_name}' "
            f"(role={role or 'train/inference'}): expected all of "
            f"{list(SCALAR_NAMES)} under {directory}, missing {missing}. "
            f"Run compute_scalars for THIS YAML/case first, e.g. "
            f"`python compute_scalars_*_prism.py --config <the-same-YAML>`. "
            f"Per-case scalers are never shared across cases."
        )
    return directory


def log_case_context(cfg: Any, role: str, *, logger=print) -> Dict[str, str]:
    """Print (and return) the active case_name and case-scoped directories.

    Emits the standardized lines requested for every entry point, e.g.::

        [inference] Using case_name: narr_prism_LA_county
        [inference] Using preprocessing directory: <repo>/.../preprocessed/narr_prism_LA_county
        [inference] Using scalar directory: <repo>/.../preprocessed/narr_prism_LA_county/scalars
    """
    case_name = get_case_name(cfg)
    preprocess_dir = case_preprocess_dir(cfg)
    scalar_dir = resolve_scalar_dir(cfg, for_writing=False)
    logger(f"[{role}] Using case_name: {case_name}")
    logger(f"[{role}] Using preprocessing directory: {preprocess_dir}")
    logger(f"[{role}] Using scalar directory: {scalar_dir}")
    return {
        "case_name": case_name,
        "preprocess_dir": str(preprocess_dir),
        "scalar_dir": str(scalar_dir),
    }


def scalar_paths(directory: Path | str) -> Dict[str, Path]:
    directory = Path(directory)
    return {name: directory / f"{name}.npy" for name in SCALAR_NAMES}


# ---------------------------------------------------------------------------
# Shape enforcement -- reject per-gridpoint / per-tile scalers
# ---------------------------------------------------------------------------

def is_channel_only(arr: np.ndarray) -> bool:
    """True when *arr* is a per-channel scaler ``[C]`` or ``[C, 1, 1]``."""
    a = np.asarray(arr)
    if a.ndim <= 1:
        return True
    # tolerate trailing singleton spatial dims, e.g. (C, 1, 1)
    return int(np.prod(a.shape[1:])) == 1


def assert_channel_only(name: str, arr: np.ndarray) -> np.ndarray:
    """Raise if *arr* is per-gridpoint/per-tile instead of per-channel."""
    a = np.asarray(arr)
    if not is_channel_only(a):
        raise ValueError(
            f"Normalization array '{name}' has shape {a.shape}, which is "
            f"per-gridpoint/per-tile. Scalers must be per-channel "
            f"[n_variables] (or [C,1,1]). Set normalization.predictor_mode: "
            f"global and recompute scalers."
        )
    return a


# Predictors are coarse -> their scalers must never be spatial. This is a strict
# alias for readability at predictor call sites.
def assert_predictor_channel_only(name: str, arr: np.ndarray) -> np.ndarray:
    return assert_channel_only(name, arr)


def is_spatial(arr: np.ndarray) -> bool:
    """True when *arr* is a per-gridpoint scaler ``[C, H, W]`` with ``H*W > 1``."""
    a = np.asarray(arr)
    return a.ndim == 3 and int(np.prod(a.shape[1:])) > 1


def scaler_kind(arr: np.ndarray) -> str:
    """Return ``"channel"`` for ``[C]``/``[C,1,1]`` or ``"spatial"`` for ``[C,H,W]``."""
    return "spatial" if is_spatial(arr) else "channel"


def assert_valid_target_scaler(name: str, arr: np.ndarray) -> np.ndarray:
    """Allow target scalers to be per-channel ``[C]`` OR per-gridpoint ``[C,H,W]``.

    Targets are native-resolution PRISM, so a per-cell climatology is a legitimate
    high-resolution field (anomaly-prediction). Reject only degenerate shapes
    (``ndim > 3``, or a ``[C, H, 1]`` / ``[C, 1, W]`` half-spatial map that would
    stretch a 1-D profile across the grid).
    """
    a = np.asarray(arr)
    if is_channel_only(a):
        return a
    if a.ndim == 3 and a.shape[1] > 1 and a.shape[2] > 1:
        return a
    raise ValueError(
        f"Target normalization array '{name}' has shape {a.shape}. Target scalers "
        f"must be per-channel [C] / [C,1,1] or per-gridpoint [C,H,W] (with H,W>1). "
        f"For per-gridpoint targets set predictands.<var>.normalization.mode: "
        f"gridpoint and data.scalar_stride: 1, then recompute scalers."
    )


def assert_target_grid_matches(
    name: str, arr: np.ndarray, n_lat: int, n_lon: int
) -> np.ndarray:
    """For a *spatial* target scaler, assert its grid equals the PRISM domain grid.

    Per-gridpoint target scalers are sliced per tile/crop by the model using the
    crop origin, so the scaler grid MUST be the exact ``(n_lat, n_lon)`` domain
    the dataset/inference iterate over. Channel-only scalers are grid-agnostic and
    pass through unchecked.
    """
    a = np.asarray(arr)
    if not is_spatial(a):
        return a
    if tuple(a.shape[-2:]) != (int(n_lat), int(n_lon)):
        raise ValueError(
            f"Spatial target scaler '{name}' grid {tuple(a.shape[-2:])} does not "
            f"match the PRISM domain grid ({int(n_lat)}, {int(n_lon)}). The "
            f"per-gridpoint climatology must be computed on the SAME spatial_subset "
            f"used for training/inference (data.scalar_stride: 1). Recompute scalers."
        )
    return a


def targets_are_spatial(directory: Path | str) -> bool:
    """Return True if the saved ``targets_mean.npy`` is a per-gridpoint ``[C,H,W]``."""
    path = Path(directory) / "targets_mean.npy"
    if not path.exists():
        return False
    return is_spatial(np.load(path, mmap_mode="r"))


# ---------------------------------------------------------------------------
# Manifest (provenance + integrity check shared by train and inference)
# ---------------------------------------------------------------------------

def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    directory: Path | str,
    *,
    case_name: str,
    predictor_mode: str = "global",
    train_date_range: Optional[Iterable[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``normalization_manifest.json`` next to the scaler ``.npy`` files."""
    directory = Path(directory)
    entries: Dict[str, Any] = {}
    target_mode = "channel"
    for name, path in scalar_paths(directory).items():
        arr = np.load(path)
        if name.startswith("inputs"):
            assert_predictor_channel_only(name, arr)
            kind = "channel"
        else:
            assert_valid_target_scaler(name, arr)
            kind = scaler_kind(arr)
            if kind == "spatial":
                target_mode = "spatial"
        entries[name] = {
            "shape": list(np.asarray(arr).shape),
            "sha256": sha256_file(path),
            "n_channels": int(np.asarray(arr).shape[0]) if arr.ndim else 0,
            "kind": kind,
        }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "case_name": case_name,
        "predictor_mode": predictor_mode,
        "target_mode": target_mode,
        "train_date_range": list(train_date_range) if train_date_range else None,
        "scalers": entries,
    }
    if extra:
        manifest.update(extra)
    manifest_path = directory / MANIFEST_NAME
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest_path


def load_manifest(directory: Path | str) -> Optional[Dict[str, Any]]:
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Wiring helpers used by training / inference entry points
# ---------------------------------------------------------------------------

def apply_scalar_paths(config: Any) -> Path:
    """Point ``config.model.*`` and ``config.data.scalar_dir`` at the resolved
    case-scoped scaler directory so the dataset (NaN-fill means) and the model
    (z-score) load byte-for-byte identical files.

    Returns the resolved directory.
    """
    directory = resolve_scalar_dir(config, for_writing=False)
    paths = scalar_paths(directory)
    model = _get(config, "model")
    if model is not None:
        setattr(model, "input_mu", str(paths["inputs_mean"]))
        setattr(model, "input_sigma", str(paths["inputs_std"]))
        setattr(model, "target_mu", str(paths["targets_mean"]))
        setattr(model, "target_sigma", str(paths["targets_std"]))
    data = _get(config, "data")
    if data is not None:
        setattr(data, "scalar_dir", str(directory))
    return directory


def log_scalar_summary(
    config: Any,
    role: str,
    *,
    logger: Any = print,
) -> Dict[str, Any]:
    """Load the resolved scalers, assert per-channel shapes, and log path + sha.

    ``role`` is a free-form label ("training" / "inference" / ...). The returned
    dict (path, per-file sha/shape) lets callers cross-check that every stage
    consumed the same file.
    """
    directory = resolve_scalar_dir(config, for_writing=False)
    logger(f"[normalization:{role}] scalar_dir = {directory}")
    summary: Dict[str, Any] = {"role": role, "scalar_dir": str(directory), "scalers": {}}
    for name, path in scalar_paths(directory).items():
        if not path.exists():
            logger(f"[normalization:{role}] MISSING {path}")
            continue
        arr = np.load(path)
        if name.startswith("inputs"):
            assert_predictor_channel_only(name, arr)
            kind, note = "channel", "per-channel OK"
        else:
            assert_valid_target_scaler(name, arr)
            kind = scaler_kind(arr)
            note = "per-channel OK" if kind == "channel" else "per-gridpoint (spatial) OK"
        sha = sha256_file(path)
        summary["scalers"][name] = {
            "shape": list(np.asarray(arr).shape),
            "sha256": sha,
            "kind": kind,
        }
        logger(
            f"[normalization:{role}] {name}: shape={tuple(np.asarray(arr).shape)} "
            f"({note}) sha256={sha[:12]}"
        )
    manifest = load_manifest(directory)
    if manifest is not None:
        logger(
            f"[normalization:{role}] manifest case_name={manifest.get('case_name')} "
            f"predictor_mode={manifest.get('predictor_mode')} "
            f"target_mode={manifest.get('target_mode', 'channel')} "
            f"created_at={manifest.get('created_at')}"
        )
        # Cross-check the on-disk files against the manifest sha values.
        for name, entry in (manifest.get("scalers") or {}).items():
            got = summary["scalers"].get(name, {}).get("sha256")
            want = entry.get("sha256")
            if got and want and got != want:
                raise ValueError(
                    f"[normalization:{role}] sha256 mismatch for {name}: "
                    f"on-disk {got[:12]} != manifest {want[:12]}. The scaler "
                    f"files changed after the manifest was written; recompute."
                )
    else:
        logger(
            f"[normalization:{role}] WARNING no {MANIFEST_NAME} found in {directory}; "
            f"cannot verify train/inference used identical scalers."
        )
    return summary


# ---------------------------------------------------------------------------
# Seam / block-artifact metric for inference sanity checks
# ---------------------------------------------------------------------------

def seam_gradient_ratio(field: np.ndarray, origins: Iterable[int], axis: int) -> float:
    """Ratio of the mean absolute gradient *at* tile seams to the interior mean.

    A value near 1.0 means seams are indistinguishable from the interior (good
    blending); large values (>~1.5) indicate visible low-resolution block edges.
    NaNs are ignored. ``origins`` are the tile start indices along ``axis``; the
    seam lines are the interior origins (all but the first).
    """
    f = np.asarray(field, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"seam_gradient_ratio expects a 2-D field, got {f.shape}")
    grad = np.abs(np.diff(f, axis=axis))
    n = f.shape[axis]
    seam_lines = sorted({int(o) for o in origins if 0 < int(o) < n})
    if not seam_lines:
        return float("nan")
    if axis == 0:
        seam_grad = np.concatenate([grad[max(0, o - 1):o + 1, :].ravel() for o in seam_lines])
    else:
        seam_grad = np.concatenate([grad[:, max(0, o - 1):o + 1].ravel() for o in seam_lines])
    seam_mean = np.nanmean(seam_grad)
    interior_mean = np.nanmean(grad)
    if not np.isfinite(interior_mean) or interior_mean == 0:
        return float("nan")
    return float(seam_mean / interior_mean)
