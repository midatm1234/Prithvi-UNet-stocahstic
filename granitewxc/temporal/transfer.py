"""Bounded regional pretrained-transfer branch, separate from native_pair.

Reuses a complete local/global transformer pair at its original width. A newly
learned projection maps actual reduced-variable, normalized history/current fields
into its latent space; these are not represented as native MERRA-2 inputs. The
remaining encoder blocks, decoder, native patch embedding, and MERRA-2 scalers are
explicitly excluded. Frozen computations retain autograd for the input adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from granitewxc.models.model import audited_pretrained_load, extract_pretrained_state

SCHEMA = "granitewxc.regional_pretrained_transfer.v3"
LEGACY_SCHEMAS = {"granitewxc.regional_pretrained_transfer.v1",
                  "granitewxc.regional_pretrained_transfer.v2"}
INITIALIZATIONS = ("pretrained", "random_control")
HISTORY_MODES = ("observed", "duplicate_current")
FRAME_METADATA = ("static_x", "static_y", "__output_crop", "__input_scaler_offset",
                  "__output_scaler_offset", "__scaler_offset")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenLocalGlobalPair(nn.Module):
    """Exact first two pretrained Transformer modules, no width reduction."""
    def __init__(self, width=2560, heads=16, multiplier=4):
        super().__init__()
        from PrithviWxC.model import Transformer
        self.width, self.heads, self.multiplier = width, heads, multiplier
        self.transformers = nn.ModuleList([
            Transformer(width, multiplier, heads, 0.0, 0.0) for _ in range(2)
        ])
        self.input_time_embedding = nn.Linear(1, width // 4)
        self.lead_time_embedding = nn.Linear(1, width // 4)
        self.requires_grad_(False)

    def forward(self, tokens, hours):
        # Upstream SampleSpec stores (current - historical) / hour, positive.
        # Negative constructor offsets in examples are not the tensor convention.
        it = self.input_time_embedding(hours.reshape(-1, 1, 1, 1))
        lt = self.lead_time_embedding(torch.zeros_like(hours).reshape(-1, 1, 1, 1))
        enc = torch.cat((it.cos(), lt.cos(), it.sin(), lt.sin()), -1)
        x = self.transformers[0]((tokens + enc, None))
        x = self.transformers[1]((x.transpose(1, 2), None))
        return x.transpose(1, 2)


def load_pretrained_pair(spec):
    """Require pinned SHA256 and complete selected-component loading."""
    path = Path(spec["path"])
    for key in ("source", "revision", "sha256"):
        if not spec.get(key):
            raise ValueError(f"foundation.{key} is required; provenance cannot be inferred")
    actual_hash = sha256_file(path)
    if actual_hash != spec["sha256"]:
        raise ValueError(f"Foundation SHA256 mismatch for {path}: {actual_hash}")
    state = extract_pretrained_state(torch.load(path, weights_only=False,
                                                map_location="cpu", mmap=True))
    selected = {}
    for i in range(2):
        prefix = f"encoder.lgl_block.transformers.{i}."
        selected.update({f"transformers.{i}." + k[len(prefix):]: v
                         for k, v in state.items() if k.startswith(prefix)})
    selected.update({k: v for k, v in state.items() if k.startswith(
        ("input_time_embedding.", "lead_time_embedding."))})
    pair = FrozenLocalGlobalPair()
    report = audited_pretrained_load(pair, selected, backbone_prefix="transformers.",
                                     require_backbone=True, strict=True)
    report.update({"source": dict(spec), "verified_sha256": actual_hash,
                   "source_state_tensors": len(state),
                   "source_non_scaler_numel": sum(v.numel() for k, v in state.items() if "scalers" not in k),
                   "selected_non_scaler_numel": sum(v.numel() for v in selected.values()),
                   "selected_encoder_transformer_indices": [0, 1],
                   "full_foundation_transfer": False,
                   "excluded": ["encoder transformers 2..24", "decoder", "patch embeddings",
                                "MERRA-2 scalers", "mask token", "output head"]})
    pair.requires_grad_(False)
    return pair, report


class RegionalPretrainedBranch(nn.Module):
    """History/current interaction precedes the reused attention representation."""
    def __init__(self, pair, *, initialization, indices, input_mu, input_sigma, latent_channels,
                 width=2560, token_grid=(8, 8), local_grid=(2, 2),
                 mask_indices=None, gate_init=1e-3):
        super().__init__()
        if initialization not in INITIALIZATIONS:
            raise ValueError("Explicit initialization must be pretrained or random_control")
        self.initialization = initialization
        self.pair = pair.requires_grad_(False)
        self.indices = tuple(int(i) for i in indices)
        self.mask_indices = tuple(mask_indices or ())
        if not self.indices or len(set(self.indices)) != len(self.indices):
            raise ValueError("Atmospheric channel indices must be nonempty and unique")
        if self.mask_indices and len(self.mask_indices) != len(self.indices):
            raise ValueError("One validity-mask channel is required per atmospheric channel")
        self.grid, self.local = tuple(token_grid), tuple(local_grid)
        if any(g <= 0 or l <= 0 or g % l for g, l in zip(self.grid, self.local)):
            raise ValueError("Token grid must be divisible by the local grid")
        self.width = width
        # Immutable downstream per-variable normalization, not fabricated MERRA-2 state.
        mu = input_mu.detach().clone()[:, self.indices]
        sigma = input_sigma.detach().clone()[:, self.indices]
        if not torch.isfinite(mu).all() or not torch.isfinite(sigma).all() or (sigma <= 0).any():
            raise ValueError("Transfer normalization must be finite with positive sigma")
        self.register_buffer("input_scalers_mu", mu)
        self.register_buffer("input_scalers_sigma", sigma)
        # Values and coverage for both dates, plus two relative regional coordinates.
        self.input_projection = nn.Conv2d(4 * len(self.indices) + 2, width, 1)
        self.input_norm = nn.GroupNorm(1, width)
        self.output_norm = nn.GroupNorm(1, width)
        self.latent_projection = nn.Conv2d(width, latent_channels, 1)
        nn.init.normal_(self.latent_projection.weight, std=1e-3)
        nn.init.zeros_(self.latent_projection.bias)
        self.gate = nn.Parameter(torch.full((1, latent_channels, 1, 1), gate_init))

    def forward(self, states, hours):
        if states.ndim != 5 or states.shape[1] != 2:
            raise ValueError("Transfer requires actual historical/current states [B,2,C,H,W]")
        if (hours <= 0).any() or not torch.isfinite(hours).all():
            raise ValueError("Historical interval must be finite and strictly positive")
        x = states[:, :, self.indices]
        valid = torch.isfinite(x)
        if self.mask_indices:
            valid = valid & (states[:, :, self.mask_indices] > 0.5)
        mu, sigma = self.input_scalers_mu, self.input_scalers_sigma
        if mu.shape[-2:] != (1, 1) and mu.shape[-2:] != x.shape[-2:]:
            raise ValueError("Transfer scaler geometry differs from input crop; migrate explicitly")
        x = torch.where(valid, (x - mu[:, None]) / (sigma[:, None] + 1e-6), 0)
        b, _, c, h, w = x.shape
        coverage = F.adaptive_avg_pool2d(valid.float().flatten(1, 2), self.grid)
        values = F.adaptive_avg_pool2d(x.flatten(1, 2), self.grid) / coverage.clamp_min(1e-6)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, self.grid[0], device=x.device),
                                torch.linspace(-1, 1, self.grid[1], device=x.device), indexing="ij")
        coords = torch.stack((yy, xx))[None].expand(b, -1, -1, -1)
        z = self.input_norm(self.input_projection(torch.cat((values, coverage, coords), 1)))
        lh, lw = self.local
        gh, gw = self.grid[0] // lh, self.grid[1] // lw
        tokens = z.reshape(b, self.width, gh, lh, gw, lw).permute(0, 2, 4, 3, 5, 1)
        tokens = tokens.reshape(b, gh * gw, lh * lw, self.width)
        # Deliberately no detach/no_grad: upstream adapter needs these derivatives.
        tokens = self.pair(tokens, hours.to(dtype=z.dtype, device=z.device))
        z = tokens.reshape(b, gh, gw, lh, lw, self.width).permute(0, 5, 1, 3, 2, 4)
        z = z.reshape(b, self.width, *self.grid)
        return self.gate * self.latent_projection(self.output_norm(z))


class _LatentInjection(nn.Module):
    def forward(self, latent, state, *_args, **_kwargs):
        if state is None:
            raise ValueError("Pretrained branch features were not supplied")
        delta = F.interpolate(state, size=latent.shape[-2:], mode="bilinear", align_corners=False)
        return delta, state


class RegionalTransferModel(nn.Module):
    """Preserve a frozen, original one-timestamp Phase-1 model as reference."""
    def __init__(self, base, branch, *, output_shape, output_channels):
        super().__init__()
        if base.temporal_adapter is not None:
            raise ValueError("Transfer requires the original Phase-1 model, without another adapter")
        self.base = base.requires_grad_(False)
        self.branch = branch
        self.output_shape = tuple(output_shape)
        self.output_channels = output_channels
        self.base.temporal_adapter = _LatentInjection()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        self.branch.pair.eval()
        return self

    def forward(self, batch, *, history_mode="observed", use_branch=True):
        states = batch["x"][:, -2:]
        if states.shape[1] != 2:
            raise ValueError("Two real dates are required; no implicit cold-start duplication")
        stamps = batch.get("__timestamps")
        if stamps is None:
            raise ValueError("Actual timestamps are required to reject discontinuous history")
        hours = states.new_tensor([(datetime.fromisoformat(t[-1]) - datetime.fromisoformat(t[-2])).total_seconds() / 3600 for t in stamps])
        expected = float(batch.get("expected_interval_hours", 24.0))
        if not torch.allclose(hours, torch.full_like(hours, expected), rtol=0, atol=1e-6):
            raise ValueError("Transfer history crosses a cadence discontinuity")
        if history_mode == "duplicate_current":
            states = states[:, -1:].expand(-1, 2, -1, -1, -1)
        elif history_mode != "observed":
            raise ValueError(f"Unknown history mode {history_mode}")
        delta = self.branch(states, hours) if use_branch else None
        # y supplies geometry to the inherited decoder. Never pass verification values.
        frame = {"x": states[:, -1], "y": states.new_zeros((states.shape[0], self.output_channels, *self.output_shape))}
        frame.update({k: batch[k] for k in FRAME_METADATA if k in batch})
        self.base._temporal_ctx = {"state": delta} if use_branch else None
        try:
            return self.base(frame)
        finally:
            self.base._temporal_ctx = None


def _tensor_digest(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()



def active_representation_contract(branch):
    """Identify the actual frozen computation, including random-control weights.

    The foundation file hash identifies an available asset, not necessarily the
    active representation. Tensor names, shapes, dtypes, and values are hashed.
    Architecture and conditioning semantics must also match for exact restore.
    """
    if branch.initialization not in INITIALIZATIONS:
        raise ValueError("Unknown active representation initialization")
    digest = hashlib.sha256()
    for name, tensor in sorted(branch.pair.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        metadata = json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                              separators=(",", ":")).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {
        "initialization": branch.initialization,
        "pair_state_sha256": digest.hexdigest(),
        "digest_format": "sorted_state_names_shapes_dtypes_bytes_v1",
        "architecture": {"width": branch.pair.width, "heads": branch.pair.heads,
                         "multiplier": branch.pair.multiplier},
        "conditioning": "current_minus_history_hours_positive;main_lead_zero",
    }


def save_transfer_adapters(branch, checkpoint, *, config, foundation_sha256,
                           phase1_digest, optimizer, optimizer_updates):
    """Save small trainable state, binding it to the exact immutable pair."""
    representation = active_representation_contract(branch)
    if config.get("initialization", "pretrained") != representation["initialization"]:
        raise ValueError("Transfer config and active representation initialization differ")
    torch.save({
        "schema": SCHEMA, "config": config, "optimizer_updates": optimizer_updates,
        "active_representation": representation,
        "branch_adapters": {k: v.cpu() for k, v in branch.state_dict().items()
                            if not k.startswith("pair.")},
        "optimizer": optimizer.state_dict(), "torch_rng_state": torch.get_rng_state(),
        "phase1_digest": phase1_digest, "foundation_sha256": foundation_sha256,
        "data_order_resume": "audit only; exact multi-window resume is not implemented",
    }, checkpoint)
    return representation


def restore_transfer_adapters(branch, checkpoint, *, foundation_sha256,
                              phase1_digest, optimizer=None):
    """Restore adapters only against the exact saved frozen representation.

    Legacy v1/v2 checkpoints lack verifiable active pair identity and require a
    separately validated migration; changing only their schema is insufficient.
    Optimizer state is optional. Exact data-order resume is not implemented.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") in LEGACY_SCHEMAS:
        raise ValueError("Legacy transfer checkpoint lacks verified active representation "
                         "identity; explicit validated migration is required")
    if payload.get("schema") != SCHEMA:
        raise ValueError("Transfer checkpoint schema mismatch")
    if payload.get("foundation_sha256") != foundation_sha256:
        raise ValueError("Transfer checkpoint foundation source differs")
    if payload.get("phase1_digest") != phase1_digest:
        raise ValueError("Transfer checkpoint Phase-1 reference differs")
    representation = payload.get("active_representation")
    if not isinstance(representation, dict):
        raise ValueError("Transfer checkpoint lacks active representation identity")
    current_representation = active_representation_contract(branch)
    if representation.get("initialization") != current_representation["initialization"]:
        raise ValueError("Transfer active representation initialization differs")
    if representation != current_representation:
        raise ValueError("Transfer active representation state or computation differs")
    contract = payload.get("config", {})
    if contract.get("initialization", "pretrained") != representation["initialization"]:
        raise ValueError("Transfer checkpoint config and representation initialization differ")
    for key, expected in (("atmospheric_indices", list(branch.indices)),
                          ("mask_indices", list(branch.mask_indices)),
                          ("token_grid", list(branch.grid)),
                          ("local_grid", list(branch.local))):
        default = [] if key == "mask_indices" else None
        if contract.get(key, default) != expected:
            raise ValueError(f"Transfer semantic/geometry contract mismatch: {key}")
    incoming = payload["branch_adapters"]
    expected = {k: v for k, v in branch.state_dict().items() if not k.startswith("pair.")}
    if set(incoming) != set(expected):
        raise ValueError("Transfer adapter checkpoint keys differ")
    for key, tensor in incoming.items():
        if tensor.shape != expected[key].shape:
            raise ValueError(f"Transfer adapter shape mismatch: {key}")
        if "scalers" in key and not torch.equal(tensor.cpu(), expected[key].cpu()):
            raise ValueError(f"Immutable transfer normalization mismatch: {key}")
    branch.load_state_dict(incoming, strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def resolve_transfer_config(config_path, *, initialization=None, history_mode=None):
    """Resolve optional controls over one case YAML, retaining its other settings."""
    import yaml
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("Transfer configuration must be a YAML mapping")
    for key, override, choices, default in (
        ("initialization", initialization, INITIALIZATIONS, "pretrained"),
        ("history_mode", history_mode, HISTORY_MODES, "observed"),
    ):
        value = spec.get(key, default) if override is None else override
        if value not in choices:
            raise ValueError(f"{key} must be one of {', '.join(choices)}")
        spec[key] = value
    return spec


def run_update_audit(config_path, output, device="cpu", updates=1, *,
                     initialization=None, history_mode=None):
    """Bounded real-data audit; config/checkpoint records include effective controls."""
    if updates < 1 or updates > 10:
        raise ValueError("Transfer engineering audit is bounded to 1..10 optimizer updates")
    spec = resolve_transfer_config(config_path, initialization=initialization,
                                   history_mode=history_mode)
    from granitewxc.temporal.entrypoints import _load
    from granitewxc.temporal.checkpoint import initialize_from_spatial_checkpoint, contract_from_model
    from granitewxc.temporal.training import build_sequence_dataloaders, seed_everything
    from granitewxc.models.model import get_finetune_model_UNET
    from granitewxc.models.loss import build_loss_fn
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    seed_everything(int(spec.get("seed", 173)))
    torch.set_num_threads(min(8, torch.get_num_threads()))
    config, cfg = _load(spec["base_config"])
    config.data.n_input_timestamps = 1
    config.device_target = device
    loaders = build_sequence_dataloaders(config, cfg, splits=("train",), batch_size=1)
    base = get_finetune_model_UNET(config)
    phase1 = spec.get("phase1_checkpoint", cfg.init_from_spatial_checkpoint)
    migration = initialize_from_spatial_checkpoint(
        base, phase1, expected_contract=contract_from_model(base, list(config.data.output_vars)))
    pair, provenance = load_pretrained_pair(spec["foundation"])
    if spec["initialization"] == "random_control":
        # Same dimensions, trainable capacity, time maps, and geometry; only source initialization differs.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(spec.get("seed", 173)) + 7)
            pair = FrozenLocalGlobalPair()
        provenance["control"] = "random initialization, not pretrained transfer"
    provenance["active_initialization"] = spec["initialization"]
    provenance["active_pretrained_numel"] = (sum(p.numel() for p in pair.parameters())
        if provenance["active_initialization"] == "pretrained" else 0)
    branch = RegionalPretrainedBranch(pair, initialization=provenance["active_initialization"],
        indices=spec["atmospheric_indices"],
        mask_indices=spec.get("mask_indices"), input_mu=base.input_scalers_mu,
        input_sigma=base.input_scalers_sigma, latent_channels=base.embed_dim_backbone,
        token_grid=spec.get("token_grid", [8, 8]), local_grid=spec.get("local_grid", [2, 2]))
    output_shape = loaders["train"].dataset.crop_size
    model = RegionalTransferModel(base, branch, output_shape=output_shape,
                                  output_channels=len(config.data.output_vars)).to(device)
    loss_fn = build_loss_fn(config, list(config.data.output_vars))
    optimizer = torch.optim.AdamW([p for p in branch.parameters() if p.requires_grad],
                                 lr=float(spec.get("learning_rate", 1e-4)), weight_decay=0)
    frozen_before = _tensor_digest(base)
    pair_before = _tensor_digest(pair)
    before = {n: p.detach().clone() for n, p in branch.named_parameters() if p.requires_grad}
    adapter_initial_digest = hashlib.sha256(b"".join(
        before[n].cpu().contiguous().numpy().tobytes() for n in sorted(before))).hexdigest()
    records = []
    first_batch = None
    t0 = time.monotonic()
    model.train()
    for index, batch in enumerate(loaders["train"]):
        if index >= updates:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        first_batch = batch if first_batch is None else first_batch
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch, history_mode=spec["history_mode"])
        valid = batch["__target_valid_mask"][:, -1]
        target = torch.where(valid, batch["y"][:, -1], float("nan"))
        loss_batch = {"y": target, **{k: batch[k] for k in FRAME_METADATA if k in batch}}
        loss = loss_fn(prediction, loss_batch)
        if isinstance(loss, tuple):
            loss = loss[0]
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite real-data transfer loss")
        loss.backward()
        gradients = {n: float(p.grad.abs().max()) if p.grad is not None else None
                     for n, p in branch.named_parameters() if p.requires_grad}
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in branch.parameters()):
            raise ValueError("Nonfinite transfer gradient")
        optimizer.step()
        records.append({"optimizer_step": index + 1, "loss": float(loss.detach()),
                        "timestamps": batch["__timestamps"],
                        "used_input_timestamps": [t[-2:] for t in batch["__timestamps"]],
                        "target_timestamps": [t[-1] for t in batch["__timestamps"]],
                        "gradient_absmax": gradients})
    if not records:
        raise RuntimeError("No real optimizer update executed")
    model.eval()
    with torch.no_grad():
        observed = model(first_batch, history_mode="observed")
        predicted = model(first_batch, history_mode=spec["history_mode"])
        reference = model(first_batch, use_branch=False)
        duplicate = model(first_batch, history_mode="duplicate_current")
        perturbed = dict(first_batch, y=torch.randn_like(first_batch["y"]) * 1000)
        target_only = model(perturbed, history_mode=spec["history_mode"])
    changed = {n: float((p.detach() - before[n]).abs().max())
               for n, p in branch.named_parameters() if n in before}
    report = {"schema": SCHEMA, "status": "actually_optimized_real_data",
        "scientifically_evaluated": False, "accepted": False,
        "config": spec, "adapter_initial_digest": adapter_initial_digest,
        "optimizer_updates": len(records), "target_samples": len(records),
        "backbone_evaluations_training": {"phase1": len(records), "pretrained_pair": len(records)},
        "seconds": time.monotonic() - t0, "phase1_migration": migration.to_dict(),
        "provenance": provenance, "updates": records, "parameter_change_absmax": changed,
        "phase1_unchanged": frozen_before == _tensor_digest(base),
        "pretrained_pair_unchanged": pair_before == _tensor_digest(pair),
        "history_prediction_max_abs": float((observed - duplicate).abs().max()),
        "phase1_prediction_max_abs": float((predicted - reference).abs().max()),
        "target_perturbation_max_abs": float((predicted - target_only).abs().max()),
        "per_variable_prediction_changes": {name: {
            "phase1_max_abs": float((predicted[:, i] - reference[:, i]).abs().max()),
            "history_max_abs": float((observed[:, i] - duplicate[:, i]).abs().max())}
            for i, name in enumerate(config.data.output_vars)},
        "trainable_numel": sum(p.numel() for p in branch.parameters() if p.requires_grad),
        "frozen_representation_numel": sum(p.numel() for p in pair.parameters()),
        "frozen_pretrained_numel": provenance["active_pretrained_numel"],
    }
    # Reference large immutable checkpoints; bind adapters to their exact active pair.
    report["active_representation"] = save_transfer_adapters(
        branch, out / "adapters.pt", config=spec, optimizer=optimizer,
        optimizer_updates=len(records), phase1_digest=frozen_before,
        foundation_sha256=provenance["verified_sha256"])
    (out / "update_audit.json").write_text(json.dumps(report, indent=2, default=str))
    torch.save({"predictions": predicted.cpu(), "phase1_reference": reference.cpu(),
                "duplicate_current": duplicate.cpu(), "timestamps": first_batch["__timestamps"]},
               out / "first_window_predictions.pt")
    print(json.dumps({k: report[k] for k in ("status", "optimizer_updates", "seconds", "phase1_unchanged", "pretrained_pair_unchanged", "history_prediction_max_abs", "target_perturbation_max_abs")}, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--initialization", choices=INITIALIZATIONS,
                        help="Override the YAML initialization; omitted uses its value")
    parser.add_argument("--history-mode", choices=HISTORY_MODES,
                        help="Override the YAML history mode; omitted uses its value")
    args = parser.parse_args(argv)
    run_update_audit(args.config, args.output, args.device, args.updates,
                     initialization=args.initialization, history_mode=args.history_mode)


if __name__ == "__main__":
    main()
