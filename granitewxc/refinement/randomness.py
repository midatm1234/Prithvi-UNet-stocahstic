"""Versioned, sample-identified stochastic fields for paired inference.

A field is keyed by experiment seed, a caller-owned stable sample identifier,
member index, and innovation index (initial source = 0, later draws = 1, 2, ...).
Batch positions, batch sizes and execution order never enter the key. Fields
are generated on CPU in float32, then copied to the requested device/dtype.
This guarantees the same random fields on a fixed PyTorch RNG implementation;
it alone does not guarantee batch-invariant neural-network arithmetic.
"""
from __future__ import annotations

import hashlib
import json
from typing import Protocol, Sequence

import torch

STREAM_VERSION = "refinement.sample_member_innovation.v1"


class NoiseSource(Protocol):
    def randn(self, shape: tuple[int, ...], device, dtype) -> torch.Tensor: ...


def stable_seed(
    experiment_seed: int, sample_id: str | int, member_index: int, innovation_index: int,
) -> int:
    """Return a process-independent seed; Python's randomized hash is not used."""
    if not isinstance(sample_id, (str, int)) or isinstance(sample_id, bool) or not str(sample_id):
        raise ValueError("sample_id must be a nonempty string or integer.")
    if member_index < 0 or innovation_index < 0:
        raise ValueError("Member and innovation indices must be nonnegative.")
    key = json.dumps(
        [STREAM_VERSION, int(experiment_seed), str(sample_id), int(member_index), int(innovation_index)],
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") & ((1 << 63) - 1)


def stable_randn(
    shape: tuple[int, ...], *, experiment_seed: int, sample_id: str | int,
    member_index: int, innovation_index: int = 0, device="cpu", dtype=torch.float32,
) -> torch.Tensor:
    """Draw one complete domain field (usually CHW, without a batch dimension)."""
    generator = torch.Generator(device="cpu").manual_seed(
        stable_seed(experiment_seed, sample_id, member_index, innovation_index)
    )
    return torch.randn(tuple(shape), generator=generator, dtype=torch.float32).to(
        device=device, dtype=dtype
    )


class StableNoiseSource:
    """Identified fields in sample-major/member-minor flattened batch order.

    Use a fresh instance for each sampler invocation. If an explicit initial
    state bypasses the sampler's first random draw, set start_innovation=1 for
    subsequent innovations. Changing the integration grid changes the meaning
    of corresponding innovation indices; this is paired randomness, not an
    assertion of a common continuous Brownian path on different grids.
    """
    def __init__(
        self, experiment_seed: int, sample_ids: Sequence[str | int],
        member_indices: Sequence[int], *, start_innovation: int = 0,
    ) -> None:
        self.experiment_seed = int(experiment_seed)
        self.sample_ids = tuple(sample_ids)
        self.member_indices = tuple(int(m) for m in member_indices)
        self.innovation_index = int(start_innovation)
        if not self.sample_ids or not self.member_indices:
            raise ValueError("Stable noise requires at least one sample and member.")
        for sample_id in self.sample_ids:
            for member in self.member_indices:
                stable_seed(self.experiment_seed, sample_id, member, self.innovation_index)

    def randn(self, shape: tuple[int, ...], device, dtype) -> torch.Tensor:
        expected = len(self.sample_ids) * len(self.member_indices)
        if len(shape) < 1 or shape[0] != expected:
            raise ValueError(f"StableNoiseSource expected leading dimension {expected}, got {shape}.")
        fields = [
            stable_randn(
                tuple(shape[1:]), experiment_seed=self.experiment_seed,
                sample_id=sample_id, member_index=member,
                innovation_index=self.innovation_index, device=device, dtype=dtype,
            )
            for sample_id in self.sample_ids for member in self.member_indices
        ]
        self.innovation_index += 1
        return torch.stack(fields, dim=0)
