"""Optional precomputed Phase-1 conditioning cache.

Refinement-only training re-runs a *frozen* Phase 1 on every epoch even though
its output can never change.  When enabled, this cache stores the deterministic
normalized output (and optionally the requested Prithvi/UNet feature maps) once
and replays them, removing the Prithvi forward pass from the Phase-2 training
step entirely.

Correctness rules
-----------------
* The cache is keyed by a fingerprint covering the Phase-1 checkpoint identity,
  the Phase-1 architecture, the case name, the dataset split, the predictor and
  target definitions, pressure levels, normalization and preprocessing
  configuration, the spatial domain, the grid resolution, and a cache-schema
  version.  Any change invalidates the cache.
* A stale or incompatible cache is **rejected**, never silently reused.
* Cached entries are validated against a live Phase-1 pass before use.
* The cache is unusable during joint Phase-1/Phase-2 fine-tuning; that
  combination is rejected by the configuration validator.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch

from granitewxc.refinement.config import Phase1CachePerfConfig, config_fingerprint

__all__ = ["CACHE_SCHEMA_VERSION", "Phase1CacheKey", "Phase1ConditioningCache"]

#: Bump whenever the on-disk layout or the semantics of a cached tensor change.
CACHE_SCHEMA_VERSION = 1

_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class Phase1CacheKey:
    """Everything that must match for a cached Phase-1 output to be reusable."""

    phase1_fingerprint: str
    phase1_architecture: str
    case_name: str
    split: str
    predictors: Any
    targets: Any
    levels: Any
    normalization: Any
    preprocessing: Any
    spatial_domain: Any
    grid_shape: Any
    schema_version: int = CACHE_SCHEMA_VERSION

    def digest(self) -> str:
        return config_fingerprint(
            {
                "phase1_fingerprint": self.phase1_fingerprint,
                "phase1_architecture": self.phase1_architecture,
                "case_name": self.case_name,
                "split": self.split,
                "predictors": self.predictors,
                "targets": self.targets,
                "levels": self.levels,
                "normalization": self.normalization,
                "preprocessing": self.preprocessing,
                "spatial_domain": self.spatial_domain,
                "grid_shape": self.grid_shape,
                "schema_version": self.schema_version,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase1_fingerprint": self.phase1_fingerprint,
            "phase1_architecture": self.phase1_architecture,
            "case_name": self.case_name,
            "split": self.split,
            "predictors": self.predictors,
            "targets": self.targets,
            "levels": self.levels,
            "normalization": self.normalization,
            "preprocessing": self.preprocessing,
            "spatial_domain": self.spatial_domain,
            "grid_shape": self.grid_shape,
            "schema_version": self.schema_version,
            "digest": self.digest(),
        }


class Phase1ConditioningCache:
    """File-backed cache of frozen Phase-1 conditioning tensors.

    One ``.pt`` file per sample keeps memory bounded (no unbounded worker
    caches) and lets DataLoader workers read entries independently.
    """

    def __init__(self, root: str | os.PathLike, key: Phase1CacheKey, config: Phase1CachePerfConfig) -> None:
        self.root = os.fspath(root)
        self.key = key
        self.config = config
        self.directory = os.path.join(self.root, key.digest()[:32])

    # -- lifecycle -------------------------------------------------------
    @property
    def manifest_path(self) -> str:
        return os.path.join(self.directory, _MANIFEST_NAME)

    def exists(self) -> bool:
        return os.path.isfile(self.manifest_path)

    def open_for_write(self) -> None:
        os.makedirs(self.directory, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as handle:
            json.dump(self.key.to_dict(), handle, indent=2, sort_keys=True, default=str)

    def assert_compatible(self) -> None:
        """Reject a cache that was written for a different configuration."""
        if not self.exists():
            raise FileNotFoundError(
                f"No Phase-1 conditioning cache at {self.directory!r}. Generate it first."
            )
        with open(self.manifest_path, encoding="utf-8") as handle:
            stored = json.load(handle)
        expected = self.key.to_dict()
        if stored.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Phase-1 cache schema version {stored.get('schema_version')!r} does not "
                f"match the current version {CACHE_SCHEMA_VERSION}. Regenerate the cache."
            )
        mismatches = [
            field
            for field in expected
            if field != "digest" and stored.get(field) != expected[field]
        ]
        if mismatches:
            raise RuntimeError(
                "Phase-1 conditioning cache is incompatible with the current run; "
                f"mismatching field(s): {mismatches}. Regenerate the cache instead of "
                "reusing it."
            )

    # -- entries ---------------------------------------------------------
    def _entry_path(self, sample_id: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(sample_id))
        return os.path.join(self.directory, f"{safe}.pt")

    def write(
        self,
        sample_id: str,
        *,
        timestamp: Any,
        deterministic_normalized: torch.Tensor | None = None,
        features: Mapping[str, torch.Tensor] | None = None,
        masks: torch.Tensor | None = None,
        coordinates: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "sample_id": str(sample_id),
            "timestamp": timestamp,
            "coordinates": dict(coordinates or {}),
            "metadata": dict(metadata or {}),
            "cache_digest": self.key.digest(),
        }
        if self.config.include_deterministic_output:
            if deterministic_normalized is None:
                raise ValueError(
                    "include_deterministic_output is enabled but no deterministic "
                    "tensor was supplied."
                )
            payload["deterministic_normalized"] = deterministic_normalized.detach().cpu()
        if masks is not None:
            payload["masks"] = masks.detach().cpu()
        features = features or {}
        if self.config.include_prithvi_features:
            if "prithvi" not in features:
                raise ValueError("include_prithvi_features is enabled but no 'prithvi' feature was supplied.")
            payload["feature_prithvi"] = features["prithvi"].detach().cpu()
        if self.config.include_unet_features:
            if "unet" not in features:
                raise ValueError("include_unet_features is enabled but no 'unet' feature was supplied.")
            payload["feature_unet"] = features["unet"].detach().cpu()

        path = self._entry_path(sample_id)
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return path

    def read(self, sample_id: str) -> dict[str, Any]:
        path = self._entry_path(sample_id)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No cached Phase-1 conditioning for sample {sample_id!r}.")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("cache_digest") != self.key.digest():
            raise RuntimeError(
                f"Cached entry {sample_id!r} was written for a different configuration "
                "digest; refusing to use it."
            )
        return payload

    def inject(self, batch: dict[str, torch.Tensor], payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Attach cached tensors to a batch under the wrapper's private keys."""
        if "deterministic_normalized" in payload:
            batch["__phase1_normalized"] = payload["deterministic_normalized"]
        if "feature_prithvi" in payload:
            batch["__phase1_feature_prithvi"] = payload["feature_prithvi"]
        if "feature_unet" in payload:
            batch["__phase1_feature_unet"] = payload["feature_unet"]
        return batch

    # -- validation ------------------------------------------------------
    def validate(
        self,
        model,
        samples: Iterable[tuple[str, Mapping[str, torch.Tensor]]],
    ) -> dict[str, float]:
        """Compare cached conditioning with a live frozen Phase-1 pass.

        Returns the worst absolute/relative deviation observed. Raises when the
        deviation exceeds ``performance.phase1_cache.validate_tolerance``.
        """
        tol = float(self.config.validate_tolerance)
        max_abs = 0.0
        max_rel = 0.0
        checked = 0
        for sample_id, batch in samples:
            if checked >= max(0, int(self.config.validate_samples)):
                break
            payload = self.read(sample_id)
            cached = payload.get("deterministic_normalized")
            if cached is None:
                continue
            with torch.no_grad():
                _, live, _ = model.run_phase1(batch)
            live = live.detach().cpu().float()
            cached = cached.float()
            if cached.shape != live.shape:
                raise RuntimeError(
                    f"Cached Phase-1 output for {sample_id!r} has shape {tuple(cached.shape)} "
                    f"but the live pass produced {tuple(live.shape)}."
                )
            diff = (cached - live).abs()
            denom = live.abs().clamp(min=1e-8)
            max_abs = max(max_abs, float(diff.max()))
            max_rel = max(max_rel, float((diff / denom).max()))
            checked += 1

        if checked and max_abs > tol:
            raise RuntimeError(
                f"Phase-1 conditioning cache validation failed: max |cached - live| = "
                f"{max_abs:.3e} exceeds tolerance {tol:.3e}. The cache is stale."
            )
        return {"checked": float(checked), "max_abs": max_abs, "max_rel": max_rel}
