"""Validated FP32 Phase-1 tile conditioning for predictor-only inference.

This cache stores the physical and encoded outputs of each *unblended* tile.
It is separate from both the older BF16 daily products and the training
residual cache, and never requires observed PRISM targets.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from granitewxc.refinement.config import config_fingerprint
from granitewxc.utils.prism_checkpoint import build_prism_checkpoint_contract
from narr_prism_phase1_cache import configure_phase1_cache_numerics


def array_digest(value) -> str:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


@contextmanager
def strict_phase1_fp32(device):
    """Disable autocast/TF32 for Phase 1 and restore the caller's policy."""
    matmul = torch.get_float32_matmul_precision()
    cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    benchmark = torch.backends.cudnn.benchmark
    deterministic = torch.backends.cudnn.deterministic
    algorithms = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        configure_phase1_cache_numerics()
        with torch.inference_mode(), torch.autocast(
            device_type=torch.device(device).type, enabled=False
        ):
            yield
    finally:
        torch.set_float32_matmul_precision(matmul)
        torch.backends.cuda.matmul.allow_tf32 = cuda_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic
        torch.use_deterministic_algorithms(algorithms, warn_only=warn_only)


def inference_cache_contract(
    *, cfg, config, phase1_fingerprint, plan, positions, lat, lon,
    valid_mask, mask_provenance, split, batch_size, pad_multiple, blend_mode,
):
    """Bind weights, data/scalers, coordinates, masks and exact tile ordering."""
    pipeline = build_prism_checkpoint_contract(config)
    if pipeline is None or not phase1_fingerprint:
        raise ValueError("FP32 inference cache requires a PRISM checkpoint contract")
    return {
        "schema": "narr_prism_fp32_tile_conditioning_v1",
        "phase1_fingerprint": phase1_fingerprint,
        "pipeline": pipeline,
        "phase1_model": {
            key: value for key, value in cfg.get("model", {}).items()
            if key not in {"phase1", "refinement"}
        },
        "target_variables": list(cfg["data"]["target_variables"]),
        "split": split,
        "split_dates": cfg["dates"][split],
        "lat_sha256": array_digest(lat),
        "lon_sha256": array_digest(lon),
        "mask_sha256": array_digest(valid_mask),
        "mask_provenance": mask_provenance,
        "domain_shape": list(plan.domain_shape),
        "core_shape": list(plan.core_shape),
        "overlap": list(plan.overlap),
        "halo": list(plan.halo),
        "positions": [list(position) for position in positions],
        "pad_multiple": pad_multiple,
        "blend_mode": blend_mode,
        "phase1_batch_size": batch_size,
        "precision": {
            "dtype": "float32", "autocast": False, "allow_tf32": False,
            "deterministic_algorithms": True,
        },
        "baseline_semantics": "unblended_tile_physical_then_encoded",
        "torch_version": str(torch.__version__),
    }


class FP32Phase1InferenceCache:
    """Build missing days atomically; reject damaged or incompatible entries."""

    def __init__(self, root, contract):
        self.contract = contract
        self.digest = config_fingerprint(contract)
        self.directory = Path(root).expanduser().resolve() / self.digest
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shape = (
            len(contract["positions"]), len(contract["target_variables"]),
            *contract["core_shape"],
        )
        self.live_validated = False
        # Multiple refinement heads may start against the same cache at once.
        with (self.directory / "manifest.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.directory / "manifest.json"
            expected = json.loads(json.dumps(contract))
            if path.exists():
                if json.loads(path.read_text()) != expected:
                    raise ValueError(f"Incompatible Phase-1 cache manifest: {path}")
            else:
                temporary = path.with_suffix(f".{os.getpid()}.tmp")
                temporary.write_text(json.dumps(expected, indent=2) + "\n")
                temporary.replace(path)

    def _validate(self, payload, sample_date, predictors_digest):
        if not isinstance(payload, dict):
            raise ValueError("Invalid Phase-1 cache payload")
        for key, expected in {
            "contract_digest": self.digest,
            "date": str(sample_date)[:10],
            "predictors_sha256": predictors_digest,
        }.items():
            if payload.get(key) != expected:
                raise ValueError(f"Phase-1 cache {key} mismatch for {sample_date}")
        for name in ("physical", "normalized"):
            tensor = payload.get(name)
            if (
                not torch.is_tensor(tensor) or tensor.dtype != torch.float32
                or tuple(tensor.shape) != self.shape
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError(f"Invalid FP32 Phase-1 {name} for {sample_date}")
            if array_digest(tensor) != payload.get(f"{name}_sha256"):
                raise ValueError(f"Phase-1 {name} checksum mismatch for {sample_date}")
        return payload

    def load_or_build(self, sample_date, predictors, model, batch_factory, device):
        """Return CPU tiles, validating one live batch on first use per run."""
        if not model.phase1_frozen or model._requested_features:
            raise ValueError(
                "FP32 tile caching requires frozen Phase 1 without feature conditioning"
            )
        if any(
            value.is_floating_point() and value.dtype != torch.float32
            for value in list(model.phase1.parameters()) + list(model.phase1.buffers())
        ):
            raise ValueError("Phase-1 weights and floating buffers must be FP32")
        model.phase1.eval()
        predictors_digest = array_digest(predictors)
        token = str(sample_date)[:10].replace("-", "")
        path = self.directory / f"phase1_{token}.pt"
        batch_size = self.contract["phase1_batch_size"]
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=True)
                payload = self._validate(payload, sample_date, predictors_digest)
            else:
                physical, normalized = [], []
                with strict_phase1_fp32(device):
                    for start in range(0, self.shape[0], batch_size):
                        batch = batch_factory(start, start + batch_size)
                        output, encoded, _ = model.run_phase1(batch)
                        if output.dtype != torch.float32 or encoded.dtype != torch.float32:
                            raise ValueError("Phase-1 forward did not produce FP32 conditioning")
                        physical.append(output.detach().cpu())
                        normalized.append(encoded.detach().cpu())
                payload = {
                    "contract_digest": self.digest,
                    "date": str(sample_date)[:10],
                    "predictors_sha256": predictors_digest,
                    "physical": torch.cat(physical),
                    "normalized": torch.cat(normalized),
                }
                for name in ("physical", "normalized"):
                    payload[f"{name}_sha256"] = array_digest(payload[name])
                self._validate(payload, sample_date, predictors_digest)
                temporary = path.with_suffix(f".{os.getpid()}.tmp")
                try:
                    torch.save(payload, temporary)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                # Verify the serialized tensors too, before refinement sees them.
                payload = self._validate(
                    torch.load(path, map_location="cpu", weights_only=True),
                    sample_date, predictors_digest,
                )
                print(f"[phase1-cache] wrote validated FP32 tiles: {path}", flush=True)

        if not self.live_validated:
            with strict_phase1_fp32(device):
                batch = batch_factory(0, batch_size)
                physical, normalized, _ = model.run_phase1(batch)
                count = physical.shape[0]
                for name, live in (("physical", physical), ("normalized", normalized)):
                    if not torch.allclose(
                        payload[name][:count], live.detach().cpu(),
                        rtol=0, atol=1.0e-5,
                    ):
                        raise ValueError(f"Phase-1 cache {name} failed live FP32 parity")
            self.live_validated = True
            print("[phase1-cache] live FP32 parity validated", flush=True)
        return payload

    @staticmethod
    def attach(batch, payload, start, stop, device):
        # Keep the original physical values for output, avoiding a second
        # encode/decode rounding step. Phase 2 consumes the stored encoding.
        batch["__phase1_normalized"] = payload["normalized"][start:stop].to(device)
        batch["__phase1_physical"] = payload["physical"][start:stop].to(device)
        return batch
