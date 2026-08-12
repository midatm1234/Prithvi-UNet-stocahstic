"""Shared normalization-scalar utilities for the ``*_PRISM`` downscaling examples.

This module defines a *single* normalization contract that is reused, without
branch-specific hacks, by the ``MERRA_PRISM`` and ``NARR_PRISM`` preprocessing,
training, validation, testing, and inference code paths:

* **Predictor** scaler shape is configured by
  ``normalization.predictor_mode``: ``global`` expects per-channel scalers and
  ``gridpoint`` expects per-gridpoint scalers.
* **Target** scaler shape is configured by
  ``predictands.<var>.normalization.mode``.  If any target uses ``gridpoint``,
  target scalers are spatial ``[C, lat, lon]``; otherwise they are per-channel.
* Mean/std are computed from the *training split only*, ignoring NaNs (that part
  lives in the ``compute_scalars_*`` scripts; this module enforces the shapes and
  binds them to every signed daily training source artifact).
* The *same* saved files are reused everywhere.  They are resolved under a
  per-``case_name`` directory (``<preprocessed_dir>/<case_name>/scalars``) and a
  manifest (sha256 + shapes + canonical PRISM grid fingerprint) lets training
  and inference assert byte-for-byte and coordinate-grid identity.  Scalers are
  **never** read from a shared/flat directory that is not scoped to the active
  ``case_name``.

The helpers accept either a raw YAML ``dict`` (used by the ``compute_scalars_*``
scripts) or a parsed ``ExperimentConfig``-like object (used by training and
inference), so a single code path serves every stage.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date as calendar_date
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from granitewxc.utils.predictands import (
    PHYSICALLY_NONNEGATIVE_VARS,
    canonicalize_precip_model,
    canonicalize_scaling_method,
)

SCALAR_NAMES = ("inputs_mean", "inputs_std", "targets_mean", "targets_std")
MANIFEST_NAME = "normalization_manifest.json"
MANIFEST_SCHEMA_VERSION = 3
TARGET_VALID_MASK_NAME = "target_valid_mask"
TARGET_VALID_MASK_FILENAME = f"{TARGET_VALID_MASK_NAME}.npy"
TARGET_VALID_MASK_MANIFEST_KEY = TARGET_VALID_MASK_NAME
TARGET_VALID_MASK_CRITERION = (
    "all configured target variables have at least one finite training-period "
    "observation at the grid cell"
)
TRAINING_SOURCE_ARTIFACT_SIGNATURES_KEY = (
    "training_source_artifact_signatures"
)
TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY = (
    "training_source_artifact_split_signature"
)

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


def _primitive(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _primitive(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "to_dict"):
        return _primitive(value.to_dict())
    if hasattr(value, "__dict__"):
        return _primitive(vars(value))
    return str(value)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    return [_primitive(value)]


def _configured_training_range(cfg: Any) -> Optional[list[str]]:
    dates = _get(cfg, "dates", {}) or {}
    training = _get(dates, "training", {}) or {}
    start = _get(training, "start")
    end = _get(training, "end")
    if start is None or end is None:
        return None
    return [str(start), str(end)]


def _configured_training_dates(cfg: Any) -> Optional[list[str]]:
    configured = _configured_training_range(cfg)
    if configured is None:
        return None
    try:
        start = calendar_date.fromisoformat(configured[0])
        end = calendar_date.fromisoformat(configured[1])
    except ValueError:
        return None
    if end < start:
        return None
    return [
        str(start + timedelta(days=offset))
        for offset in range((end - start).days + 1)
    ]


def _predictand_semantics(cfg: Any, output_vars: list[str]) -> list[Dict[str, Any]]:
    predictands = _get(cfg, "predictands", {}) or {}
    result: list[Dict[str, Any]] = []
    for name in output_vars:
        raw = _get(predictands, name, {}) or {}
        raw_normalization = _get(raw, "normalization", {}) or {}
        raw_scaling = _get(raw, "scaling", {}) or {}
        active = raw_normalization or raw_scaling
        allow_negative = bool(
            _get(
                raw,
                "allow_negative_value",
                _get(raw_normalization, "allow_negative_value", False),
            )
        )
        default_nonnegative = (
            str(name).lower() in PHYSICALLY_NONNEGATIVE_VARS
            and not allow_negative
        )
        raw_nonnegative = _get(raw, "nonnegativity", {}) or {}
        nonnegative_enabled = bool(
            _get(raw_nonnegative, "enabled", default_nonnegative)
        )
        nonnegative_method = str(
            _get(
                raw_nonnegative,
                "method",
                "softplus" if nonnegative_enabled else "none",
            )
        ).lower()
        if not nonnegative_enabled:
            nonnegative_method = "none"
        default_method = "divide_only" if default_nonnegative else "zscore"
        method = canonicalize_scaling_method(
            _get(active, "method", default_method)
        )
        default_stat = "p95" if method == "divide_only" else "mean"
        fixed_scale = _get(
            active,
            "fixed_scale",
            _get(raw_scaling, "fixed_scale", None),
        )
        result.append(
            {
                "name": str(name),
                "allow_negative_value": allow_negative,
                "nonnegativity": {
                    "enabled": nonnegative_enabled,
                    "method": nonnegative_method,
                },
                "scaling": {
                    "method": method,
                    "mode": str(
                        _get(active, "mode", _get(raw_scaling, "mode", "global"))
                    ).lower(),
                    "eps_std": float(
                        _get(
                            active,
                            "eps_std",
                            _get(raw_scaling, "eps_std", 1.0e-6),
                        )
                    ),
                    "scale_stat": str(
                        _get(
                            active,
                            "scale_stat",
                            _get(raw_scaling, "scale_stat", default_stat),
                        )
                    ).lower(),
                    "fixed_scale": (
                        None if fixed_scale is None else float(fixed_scale)
                    ),
                },
            }
        )
    return result


def normalization_config_contract(cfg: Any) -> Dict[str, Any]:
    """Return scalar-defining configuration with explicit channel order."""
    data = _get(cfg, "data", {}) or {}
    normalization_cfg = _get(cfg, "normalization", {}) or {}
    output_vars = [
        str(value)
        for value in _as_list(
            _get(data, "output_vars", _get(data, "target_variables", []))
        )
    ]
    predictor_variables = _get(data, "predictor_variables", {}) or {}
    ordered_predictors: list[list[Any]] = []
    if isinstance(predictor_variables, Mapping):
        for variable, levels in predictor_variables.items():
            for level in _as_list(levels):
                ordered_predictors.append([str(variable), float(level)])
    return {
        "data_type": str(_get(data, "type", "")).lower(),
        "input_vars": [str(value) for value in _as_list(_get(data, "input_vars", []))],
        "ordered_predictors": ordered_predictors,
        "output_vars": output_vars,
        "target_variables": [
            str(value)
            for value in _as_list(_get(data, "target_variables", output_vars))
        ],
        "n_input_timestamps": int(_get(data, "n_input_timestamps", 1)),
        "scalar_stride": int(_get(data, "scalar_stride", 1)),
        "predictor_method": str(
            _get(normalization_cfg, "predictor_method", "standardize")
        ).lower(),
        "predictor_mode": predictor_mode(cfg),
        "target_method": str(
            _get(normalization_cfg, "target_method", "per_variable")
        ).lower(),
        "quantile_histogram_maximum": float(
            _get(normalization_cfg, "quantile_histogram_maximum", 512.0)
        ),
        "quantile_histogram_bins": int(
            _get(normalization_cfg, "quantile_histogram_bins", 65536)
        ),
        "predictands": _predictand_semantics(cfg, output_vars),
        "precip_model": canonicalize_precip_model(
            _get(cfg, "precip_model", _get(cfg, "precip_head_type", "single_head"))
        ),
    }


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


def target_valid_mask_path(directory_or_config: Path | str | Any) -> Path:
    """Return the case-scoped training-target support-mask path.

    A path/string denotes a scalar directory directly.  Any other value is
    treated as a config and resolved through the same case-scoped scalar
    directory contract as the normalization arrays.
    """
    if isinstance(directory_or_config, (str, os.PathLike, Path)):
        directory = Path(directory_or_config)
    else:
        directory = resolve_scalar_dir(directory_or_config, for_writing=False)
    return directory / TARGET_VALID_MASK_FILENAME


def build_target_valid_mask_manifest_entry(
    path: Path | str,
    *,
    cfg: Any,
    training_source_artifact_split_signature: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the signed manifest entry for a training-derived support mask.

    This helper intentionally does not derive the mask.  The NARR scalar pass
    derives it while iterating the configured training targets, then calls this
    function only after persisting the boolean array.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Target-valid-mask artifact does not exist: {path}")
    mask = np.load(path, allow_pickle=False)
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError(
            f"{path} must contain a two-dimensional boolean array, got "
            f"dtype={mask.dtype}, shape={mask.shape}"
        )
    data = _get(cfg, "data", {}) or {}
    variables = [
        str(value)
        for value in _as_list(
            _get(data, "output_vars", _get(data, "target_variables", []))
        )
    ]
    if not variables:
        raise ValueError(
            "Cannot bind target_valid_mask without an explicit configured output order"
        )
    train_range = _configured_training_range(cfg)
    if train_range is None:
        raise ValueError(
            "Cannot bind target_valid_mask without dates.training.start/end"
        )

    from granitewxc.utils.prism_grid import load_canonical_grid

    canonical_grid = load_canonical_grid(case_preprocess_dir(cfg), required=False)
    if canonical_grid is not None and tuple(mask.shape) != tuple(canonical_grid.shape):
        raise ValueError(
            f"Target-valid-mask grid {mask.shape} does not match the canonical "
            f"PRISM grid {canonical_grid.shape}"
        )
    return {
        "filename": TARGET_VALID_MASK_FILENAME,
        "shape": list(mask.shape),
        "dtype": "bool",
        "sha256": sha256_file(path),
        "valid_cells": int(mask.sum()),
        "total_cells": int(mask.size),
        "source_split": "training",
        "train_date_range": train_range,
        "target_variables": variables,
        "criterion": TARGET_VALID_MASK_CRITERION,
        "grid_fingerprint": (
            canonical_grid.fingerprint if canonical_grid is not None else None
        ),
        TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY: (
            None
            if training_source_artifact_split_signature is None
            else str(training_source_artifact_split_signature)
        ),
    }


def load_target_valid_mask(
    cfg: Any,
    *,
    role: str,
    expected_shape: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Load and authenticate the static, training-derived joint target mask.

    Neutral target scaler fills are finite by design and therefore cannot
    distinguish ocean/outside-domain cells.  Production consumers must use
    this separately persisted artifact and must fail when a legacy scalar set
    does not provide it.
    """
    directory = resolve_scalar_dir(cfg, for_writing=False)
    path = directory / TARGET_VALID_MASK_FILENAME
    manifest_path = directory / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"[{role}] required training-derived target support mask is missing: "
            f"{path}. Re-run compute_scalars_narr_prism.py with the same YAML; "
            "output-scaler finiteness is not a valid land/support mask."
        )
    manifest = load_manifest(directory)
    if manifest is None:
        raise FileNotFoundError(
            f"[{role}] cannot authenticate {path} without {manifest_path}"
        )
    entry = manifest.get(TARGET_VALID_MASK_MANIFEST_KEY)
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"[{role}] {manifest_path} does not bind {TARGET_VALID_MASK_FILENAME}; "
            "recompute NARR-PRISM scalars from the configured training split"
        )

    mask = np.load(path, allow_pickle=False)
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError(
            f"[{role}] {path} must be a two-dimensional boolean mask, got "
            f"dtype={mask.dtype}, shape={mask.shape}"
        )
    expected_entry_shape = tuple(int(value) for value in entry.get("shape", []))
    if expected_entry_shape != tuple(mask.shape):
        raise ValueError(
            f"[{role}] target-valid-mask shape mismatch: manifest="
            f"{expected_entry_shape}, file={mask.shape}"
        )
    if entry.get("dtype") != "bool":
        raise ValueError(
            f"[{role}] target-valid-mask manifest dtype must be 'bool', got "
            f"{entry.get('dtype')!r}"
        )
    observed_sha = sha256_file(path)
    expected_sha = entry.get("sha256")
    if expected_sha != observed_sha:
        raise ValueError(
            f"[{role}] target-valid-mask sha256 mismatch: file="
            f"{observed_sha[:12]}, manifest={str(expected_sha)[:12]}"
        )
    if entry.get("source_split") != "training":
        raise ValueError(
            f"[{role}] target-valid-mask source_split must be 'training'"
        )
    if entry.get("criterion") != TARGET_VALID_MASK_CRITERION:
        raise ValueError(
            f"[{role}] unsupported target-valid-mask derivation criterion "
            f"{entry.get('criterion')!r}"
        )
    configured_range = _configured_training_range(cfg)
    if configured_range is None or entry.get("train_date_range") != configured_range:
        raise ValueError(
            f"[{role}] target-valid-mask training range mismatch: manifest="
            f"{entry.get('train_date_range')!r}, config={configured_range!r}"
        )
    data = _get(cfg, "data", {}) or {}
    configured_variables = [
        str(value)
        for value in _as_list(
            _get(data, "output_vars", _get(data, "target_variables", []))
        )
    ]
    if entry.get("target_variables") != configured_variables:
        raise ValueError(
            f"[{role}] target-valid-mask variable-order mismatch: manifest="
            f"{entry.get('target_variables')!r}, config={configured_variables!r}"
        )
    source_signature = manifest.get(
        TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY
    )
    if entry.get(TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY) != source_signature:
        raise ValueError(
            f"[{role}] target-valid-mask training-source signature does not "
            "match the scalar manifest"
        )

    from granitewxc.utils.prism_grid import load_canonical_grid

    canonical_grid = load_canonical_grid(case_preprocess_dir(cfg), required=False)
    if canonical_grid is not None:
        if tuple(mask.shape) != tuple(canonical_grid.shape):
            raise ValueError(
                f"[{role}] target-valid-mask grid {mask.shape} does not match "
                f"canonical PRISM grid {canonical_grid.shape}"
            )
        if entry.get("grid_fingerprint") != canonical_grid.fingerprint:
            raise ValueError(
                f"[{role}] target-valid-mask canonical-grid fingerprint mismatch"
            )
    if expected_shape is not None:
        shape = tuple(int(value) for value in expected_shape)
        if tuple(mask.shape) != shape:
            raise ValueError(
                f"[{role}] target-valid-mask grid {mask.shape} does not match "
                f"requested output grid {shape}"
            )
    if int(entry.get("valid_cells", -1)) != int(mask.sum()) or int(
        entry.get("total_cells", -1)
    ) != int(mask.size):
        raise ValueError(
            f"[{role}] target-valid-mask cell counts disagree with the signed file"
        )
    if not bool(mask.any()):
        raise ValueError(f"[{role}] target-valid-mask contains no valid output cells")
    return np.asarray(mask, dtype=bool)


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


def predictor_mode(cfg: Any) -> str:
    """Return the configured predictor normalization mode."""
    normalization = _get(cfg, "normalization", {}) or {}
    mode = str(_get(normalization, "predictor_mode", "global")).lower()
    if mode not in {"global", "gridpoint"}:
        raise ValueError(
            "normalization.predictor_mode must be either 'global' or 'gridpoint'"
        )
    return mode


def target_mode(cfg: Any) -> str:
    """Return ``spatial`` if any configured target uses gridpoint normalization.

    ``get_config`` initially preserves the YAML shape, where the mode lives at
    ``predictands.<var>.normalization.mode``.  Model construction canonicalizes
    predictand specs in place and stores the same value at ``scaling.mode``.
    Accept both representations so validation remains stable throughout a
    notebook session.
    """
    data = _get(cfg, "data", {}) or {}
    target_vars = (
        _get(data, "target_variables", None)
        or _get(data, "output_vars", None)
        or []
    )
    predictands = _get(cfg, "predictands", {}) or {}
    for var in list(target_vars):
        var_cfg = _get(predictands, str(var), {}) or {}
        normalization = _get(var_cfg, "normalization", {}) or {}
        scaling = _get(var_cfg, "scaling", {}) or {}
        mode = _get(normalization, "mode", _get(scaling, "mode", "global"))
        if str(mode).lower() == "gridpoint":
            return "spatial"
    return "channel"


def assert_valid_predictor_scaler(name: str, arr: np.ndarray, cfg: Any) -> np.ndarray:
    """Validate predictor scaler shape against ``normalization.predictor_mode``."""
    a = np.asarray(arr)
    mode = predictor_mode(cfg)
    if mode == "global":
        return assert_predictor_channel_only(name, a)
    if is_spatial(a):
        return a
    raise ValueError(
        f"Predictor normalization array '{name}' has shape {a.shape}, but "
        "normalization.predictor_mode is 'gridpoint'. Expected per-gridpoint "
        "[C,H,W] scalers computed with data.scalar_stride: 1."
    )


def assert_valid_target_scaler_for_config(
    name: str, arr: np.ndarray, cfg: Any
) -> np.ndarray:
    """Validate target scaler shape against predictand normalization modes."""
    a = assert_valid_target_scaler(name, arr)
    mode = target_mode(cfg)
    if mode == "channel" and not is_channel_only(a):
        raise ValueError(
            f"Target normalization array '{name}' has shape {a.shape}, but all "
            "predictands use normalization.mode: global. Expected per-channel "
            "[C] or [C,1,1] scalers."
        )
    if mode == "spatial" and not is_spatial(a):
        raise ValueError(
            f"Target normalization array '{name}' has shape {a.shape}, but at "
            "least one predictand uses normalization.mode: gridpoint. Expected "
            "per-gridpoint [C,H,W] scalers computed with data.scalar_stride: 1."
        )
    return a


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
    cfg: Optional[Any] = None,
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
            if cfg is not None:
                assert_valid_predictor_scaler(name, arr, cfg)
            else:
                assert_predictor_channel_only(name, arr)
            kind = scaler_kind(arr)
        else:
            if cfg is not None:
                assert_valid_target_scaler_for_config(name, arr, cfg)
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
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "case_name": case_name,
        "predictor_mode": predictor_mode,
        "target_mode": target_mode,
        "train_date_range": list(train_date_range) if train_date_range else None,
        "scalers": entries,
    }
    if cfg is not None:
        manifest["config_contract"] = normalization_config_contract(cfg)
        # Import lazily so the generic grid utility remains independent of the
        # normalization/config resolver.  Scalar manifests then bind the arrays
        # to exact PRISM coordinate values, not merely an (H, W) shape.
        from granitewxc.utils.prism_grid import load_canonical_grid

        prism_grid = load_canonical_grid(case_preprocess_dir(cfg), required=False)
        if prism_grid is not None:
            manifest["prism_grid"] = prism_grid.manifest_entry()
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


def _validated_training_source_artifact_contract(
    manifest: Mapping[str, Any],
    *,
    expected_dates: Optional[Iterable[Any]] = None,
    role: str,
) -> tuple[Dict[str, str], str]:
    """Validate the raw-artifact set that produced training normalization.

    The predictor preprocessing signature intentionally describes processing
    semantics shared by training/validation/inference.  It cannot also encode
    training-day input files, because those differ by split.  This companion
    contract binds the scalar arrays to every signed daily training product so
    replacing/rebuilding one source day cannot silently retain stale scalars.
    """
    raw_daily = manifest.get(TRAINING_SOURCE_ARTIFACT_SIGNATURES_KEY)
    raw_split = manifest.get(TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY)
    if not isinstance(raw_daily, Mapping) or not raw_daily or raw_split is None:
        raise ValueError(
            f"[{role}] scalar manifest lacks the signed training source-artifact "
            "set; rerun preprocessing and recompute scalars"
        )

    daily = {str(day): str(signature) for day, signature in raw_daily.items()}
    invalid = [
        day
        for day, signature in daily.items()
        if len(signature) != 64
        or any(
            character not in "0123456789abcdef"
            for character in signature.lower()
        )
    ]
    if invalid:
        raise ValueError(
            f"[{role}] scalar manifest contains invalid training source-artifact "
            f"signatures for dates {invalid[:5]}"
        )

    if expected_dates is not None:
        expected = [str(value) for value in expected_dates]
        expected_set = set(expected)
        observed_set = set(daily)
        if len(expected_set) != len(expected) or observed_set != expected_set:
            missing = sorted(expected_set - observed_set)
            extra = sorted(observed_set - expected_set)
            raise ValueError(
                f"[{role}] scalar manifest training source-artifact dates do not "
                f"exactly match the configured split: missing={missing[:5]}, "
                f"extra={extra[:5]}"
            )

    from granitewxc.utils.prism_preprocessed import (
        split_source_artifact_signature,
    )

    computed = split_source_artifact_signature(daily)
    split_signature = str(raw_split)
    if split_signature != computed:
        raise ValueError(
            f"[{role}] scalar manifest training source-artifact split signature "
            f"is invalid: stored={split_signature}, computed={computed}"
        )
    return daily, split_signature


def training_source_artifact_contract(
    cfg: Any,
    *,
    expected_dates: Optional[Iterable[Any]] = None,
    role: str,
) -> tuple[Dict[str, str], str]:
    """Load and validate the scalar manifest's exact training-source set."""
    manifest_path = resolve_scalar_dir(cfg, for_writing=False) / MANIFEST_NAME
    manifest = load_manifest(manifest_path.parent)
    if manifest is None:
        raise FileNotFoundError(
            f"[{role}] cannot validate training sources without {manifest_path}"
        )
    return _validated_training_source_artifact_contract(
        manifest,
        expected_dates=expected_dates,
        role=role,
    )


def assert_preprocessing_signature_matches(
    cfg: Any,
    observed_signature: Optional[str],
    *,
    role: str,
) -> None:
    """Bind every consumed split to the predictors used for scalar fitting."""
    data = _get(cfg, "data", {}) or {}
    corrected_prism = (
        str(_get(data, "type", "")).lower() in {"narr_prism", "merra_prism"}
        and bool(_get(data, "use_preprocessed", False))
    )
    if not corrected_prism:
        return
    manifest_path = resolve_scalar_dir(cfg, for_writing=False) / MANIFEST_NAME
    manifest = load_manifest(manifest_path.parent)
    if manifest is None:
        raise FileNotFoundError(
            f"[{role}] cannot bind preprocessed predictors without {manifest_path}"
        )
    expected = manifest.get("predictor_preprocessing_signature")
    if expected is None:
        raise ValueError(
            f"[{role}] {manifest_path} lacks predictor_preprocessing_signature; "
            "recompute scalars from the signed training products"
        )
    if observed_signature is None or str(observed_signature) != str(expected):
        raise ValueError(
            f"[{role}] predictor preprocessing signature mismatch: product="
            f"{observed_signature!r}, training-scalars={expected!r}. Re-run "
            "preprocessing and scalar computation with the same config/source grid."
        )


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
            assert_valid_predictor_scaler(name, arr, config)
            kind = scaler_kind(arr)
            note = "per-gridpoint OK" if kind == "spatial" else "per-channel OK"
        else:
            assert_valid_target_scaler_for_config(name, arr, config)
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
        active_case = get_case_name(config)
        manifest_case = manifest.get("case_name")
        if manifest_case != active_case:
            raise ValueError(
                f"[normalization:{role}] case mismatch: active config is "
                f"{active_case!r}, but {directory / MANIFEST_NAME} declares "
                f"{manifest_case!r}. Recompute scalars for the active case; "
                "normalization artifacts may not be shared across cases."
            )
        configured_predictor_mode = predictor_mode(config)
        if manifest.get("predictor_mode") != configured_predictor_mode:
            raise ValueError(
                f"[normalization:{role}] predictor_mode mismatch: manifest="
                f"{manifest.get('predictor_mode')!r}, config="
                f"{configured_predictor_mode!r}. Recompute scalars."
            )
        configured_target_mode = target_mode(config)
        if manifest.get("target_mode", "channel") != configured_target_mode:
            raise ValueError(
                f"[normalization:{role}] target_mode mismatch: manifest="
                f"{manifest.get('target_mode')!r}, config="
                f"{configured_target_mode!r}. Recompute scalars."
            )
        configured_training_range = _configured_training_range(config)
        if (
            configured_training_range is not None
            and manifest.get("train_date_range") != configured_training_range
        ):
            raise ValueError(
                f"[normalization:{role}] training date range mismatch: manifest="
                f"{manifest.get('train_date_range')!r}, config="
                f"{configured_training_range!r}. Recompute scalars from the "
                "current training split."
            )
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
        manifest_grid = manifest.get("prism_grid")
        data = _get(config, "data", {}) or {}
        corrected_prism = (
            str(_get(data, "type", "")).lower()
            in {"narr_prism", "merra_prism"}
            and bool(_get(data, "use_preprocessed", False))
        )
        if corrected_prism and manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"[normalization:{role}] scalar manifest schema "
                f"{manifest.get('schema_version')!r} predates the semantic "
                "normalization contract; recompute scalars."
            )
        if corrected_prism and manifest_grid is None:
            raise ValueError(
                f"[normalization:{role}] {directory / MANIFEST_NAME} is not "
                "bound to a canonical PRISM grid. Recompute the scalars after "
                "canonical preprocessing; spatial climatologies cannot be "
                "validated by shape alone."
            )
        if manifest_grid is not None:
            from granitewxc.utils.prism_grid import load_canonical_grid

            canonical_grid = load_canonical_grid(
                case_preprocess_dir(config), required=True
            )
            assert canonical_grid is not None
            current_grid = canonical_grid.manifest_entry()
            if manifest_grid != current_grid:
                raise ValueError(
                    f"[normalization:{role}] PRISM grid fingerprint mismatch: "
                    f"manifest={manifest_grid.get('fingerprint')} "
                    f"canonical={current_grid.get('fingerprint')}. Recompute "
                    "scalars on the active case's canonical PRISM grid."
                )
            summary["prism_grid"] = current_grid
            logger(
                f"[normalization:{role}] PRISM grid: "
                f"shape={tuple(current_grid['shape'])} "
                f"fingerprint={current_grid['fingerprint'][:12]}"
            )
        if corrected_prism and not manifest.get(
            "predictor_preprocessing_signature"
        ):
            raise ValueError(
                f"[normalization:{role}] scalar manifest is not bound to the "
                "signed training predictors; recompute scalars from the strict "
                "preprocessed training split."
            )
        if corrected_prism:
            daily_sources, source_split_signature = (
                _validated_training_source_artifact_contract(
                    manifest,
                    expected_dates=_configured_training_dates(config),
                    role=f"normalization:{role}",
                )
            )
            summary[TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY] = (
                source_split_signature
            )
            logger(
                f"[normalization:{role}] training source artifacts: "
                f"{len(daily_sources)} dates "
                f"split_sha256={source_split_signature[:12]}"
            )
        observed_contract = manifest.get("config_contract")
        expected_contract = normalization_config_contract(config)
        if corrected_prism and observed_contract is None:
            raise ValueError(
                f"[normalization:{role}] scalar manifest lacks config_contract; "
                "recompute scalars so channel order and predictand scaling can be verified."
            )
        if observed_contract is not None and observed_contract != expected_contract:
            mismatch_keys = [
                key
                for key in sorted(set(observed_contract) | set(expected_contract))
                if observed_contract.get(key) != expected_contract.get(key)
            ]
            raise ValueError(
                f"[normalization:{role}] scalar config contract mismatch in "
                f"{mismatch_keys}: manifest={observed_contract!r}, "
                f"config={expected_contract!r}. Recompute scalars."
            )
    else:
        data = _get(config, "data", {}) or {}
        data_type = str(_get(data, "type", "")).lower()
        if data_type in {"narr_prism", "merra_prism"} and bool(
            _get(data, "use_preprocessed", False)
        ):
            raise FileNotFoundError(
                f"[normalization:{role}] corrected {data_type} runs require "
                f"{directory / MANIFEST_NAME} so the active case, training "
                "dates, scaler hashes, and canonical PRISM grid can be "
                "verified. Recompute scalars from the strict training products."
            )
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
