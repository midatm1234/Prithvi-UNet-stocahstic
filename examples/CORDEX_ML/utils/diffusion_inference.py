"""Helpers for diffusion-head checkpoint detection and ensemble inference."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Callable

import numpy as np
import torch
import xarray as xr


DIFFUSION_HEAD_ALIASES = {"diffusion", "diffusion_head", "sde", "score", "score_sde"}
_ENSEMBLE_GENERATORS_ATTR = "_diffusion_ensemble_generators"


class PersistentEnsembleGenerators:
    """Independent, reproducible RNG streams for diffusion ensemble members.

    The current samplers can draw the initial prior on CPU and later noise on
    the inference device.  Each member therefore owns both streams. Activating
    a member temporarily installs its saved states as the global torch/NumPy
    states (which keeps compatibility with model code that does not accept an
    explicit ``generator``), then stores the advanced states after inference.
    The caller's ambient RNG state is restored on exit.
    """

    def __init__(
        self,
        *,
        ensemble_size: int,
        base_seed: int,
        device: torch.device,
    ) -> None:
        self.ensemble_size = int(ensemble_size)
        self.base_seed = int(base_seed)
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA ensemble generators requested but CUDA is unavailable.")
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())

        self._cpu_generators: list[torch.Generator] = []
        self._device_generators: list[torch.Generator] = []
        self._numpy_generators: list[np.random.RandomState] = []
        for member_idx in range(self.ensemble_size):
            seed = self.base_seed + member_idx
            self._cpu_generators.append(torch.Generator(device="cpu").manual_seed(seed))
            if self.device.type == "cuda":
                self._device_generators.append(
                    torch.Generator(device=self.device).manual_seed(seed)
                )
            self._numpy_generators.append(
                np.random.RandomState(seed % (2**32 - 1))
            )

    def matches(
        self,
        *,
        ensemble_size: int,
        base_seed: int,
        device: torch.device,
    ) -> bool:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None and torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        return (
            self.ensemble_size == int(ensemble_size)
            and self.base_seed == int(base_seed)
            and self.device == device
        )

    @contextmanager
    def activate(self, member_idx: int):
        """Activate one member's RNG stream and save its advanced state."""
        if not 0 <= int(member_idx) < self.ensemble_size:
            raise IndexError(
                f"member_idx must be in [0, {self.ensemble_size}), got {member_idx}."
            )
        member_idx = int(member_idx)

        outer_cpu_state = torch.random.get_rng_state()
        outer_numpy_state = np.random.get_state()
        outer_device_state = None
        if self.device.type == "cuda":
            outer_device_state = torch.cuda.get_rng_state(self.device)

        torch.random.set_rng_state(self._cpu_generators[member_idx].get_state())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(
                self._device_generators[member_idx].get_state(), self.device
            )
        np.random.set_state(self._numpy_generators[member_idx].get_state())

        try:
            yield
        finally:
            self._cpu_generators[member_idx].set_state(torch.random.get_rng_state())
            if self.device.type == "cuda":
                self._device_generators[member_idx].set_state(
                    torch.cuda.get_rng_state(self.device)
                )
            self._numpy_generators[member_idx].set_state(np.random.get_state())

            torch.random.set_rng_state(outer_cpu_state)
            if outer_device_state is not None:
                torch.cuda.set_rng_state(outer_device_state, self.device)
            np.random.set_state(outer_numpy_state)


def _resolve_ensemble_generators(
    *,
    model: torch.nn.Module,
    ensemble_size: int,
    base_seed: int,
    device: torch.device,
) -> PersistentEnsembleGenerators:
    """Reuse a model-local generator pool so streams advance across batches."""
    generators = getattr(model, _ENSEMBLE_GENERATORS_ATTR, None)
    if not isinstance(generators, PersistentEnsembleGenerators) or not generators.matches(
        ensemble_size=ensemble_size,
        base_seed=base_seed,
        device=device,
    ):
        generators = PersistentEnsembleGenerators(
            ensemble_size=ensemble_size,
            base_seed=base_seed,
            device=device,
        )
        setattr(model, _ENSEMBLE_GENERATORS_ATTR, generators)
    return generators


def reset_ensemble_generators(model: torch.nn.Module) -> bool:
    """Reset model-local diffusion RNG streams before a new scenario run.

    Diffusion member streams deliberately persist across batches within one
    scenario so consecutive dates do not replay the same noise.  They must not
    persist across independent scenarios, however, because run ordering would
    then change the output for a fixed scenario.  Clear the cache on the model
    and on common wrappers so the next batch restarts from ``base_seed``.

    Returns ``True`` when at least one cached pool was removed.
    """

    removed = False
    pending: list[torch.nn.Module] = [model]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        if hasattr(current, _ENSEMBLE_GENERATORS_ATTR):
            delattr(current, _ENSEMBLE_GENERATORS_ATTR)
            removed = True
        for attribute in ("module", "_orig_mod"):
            candidate = getattr(current, attribute, None)
            if isinstance(candidate, torch.nn.Module):
                pending.append(candidate)
    return removed


def _get_nested(obj: Any, *keys: str) -> Any:
    current = obj
    for key in keys:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def canonical_head_type(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in DIFFUSION_HEAD_ALIASES:
        return "diffusion"
    if text in {"deterministic", "conv", "convolutional", "unet", "standard"}:
        return "deterministic"
    return text or None


def checkpoint_head_type(checkpoint: Any) -> str | None:
    """Return a checkpoint-declared head type when present."""
    if not isinstance(checkpoint, dict):
        return None

    candidates = [
        checkpoint.get("head_type"),
        checkpoint.get("decoder_type"),
        checkpoint.get("diffusion_head"),
        _get_nested(checkpoint, "model_config", "head_type"),
        _get_nested(checkpoint, "model_config", "decoder_type"),
        _get_nested(checkpoint, "model", "head_type"),
        _get_nested(checkpoint, "config", "model", "head_type"),
        _get_nested(checkpoint, "config", "model", "decoder_type"),
        _get_nested(checkpoint, "metadata", "head_type"),
        _get_nested(checkpoint, "metadata", "decoder_type"),
        _get_nested(checkpoint, "metadata", "diffusion_head"),
    ]
    for value in candidates:
        if isinstance(value, bool):
            if value:
                return "diffusion"
            continue
        head_type = canonical_head_type(value)
        if head_type:
            return head_type
    state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
    if isinstance(state_dict, dict) and any("diffusion_head" in str(key) for key in state_dict.keys()):
        return "diffusion"
    return None


def config_head_type(config: Any) -> str:
    head_type = canonical_head_type(_get_nested(config, "model", "head_type"))
    if head_type:
        return head_type
    head_type = canonical_head_type(_get_nested(config, "model", "decoder_type"))
    if head_type:
        return head_type
    return "deterministic"


def model_head_type(model: Any) -> str | None:
    diffusion_enabled = getattr(model, "diffusion_enabled", None)
    if diffusion_enabled is not None:
        return "diffusion" if bool(diffusion_enabled) else "deterministic"
    return canonical_head_type(getattr(model, "head_type", None))


def infer_head_type(checkpoint: Any = None, config: Any = None, model: Any = None) -> str:
    """Detect model head type, preferring explicit checkpoint metadata."""
    return checkpoint_head_type(checkpoint) or config_head_type(config) or model_head_type(model) or "deterministic"


def resolve_ensemble_size(
    config: Any = None,
    *,
    requested: int | None = None,
    head_type: str = "deterministic",
) -> int:
    """Resolve ensemble size from an explicit request or the loaded config.

    Both deterministic and diffusion paths default to one member when the YAML
    omits ``inference.ensemble_size``. No diffusion-specific size is imposed.
    """
    value = requested
    if value is None:
        value = _get_nested(config, "inference", "ensemble_size")
    if value is None:
        value = 1
    value = int(value)
    if value < 1:
        raise ValueError(f"ensemble_size must be at least 1, got {value}")
    return value


def resolve_base_seed(config: Any = None, *, default: int = 42) -> int:
    value = _get_nested(config, "inference", "base_seed")
    if value is None:
        value = _get_nested(config, "inference", "ensemble_seed")
    return int(default if value is None else value)


def seed_everything(seed: int, device: torch.device | None = None) -> None:
    torch.manual_seed(int(seed))
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))


def call_model_with_optional_raw(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model_out = model(batch, return_pre_inverse=True, return_raw_output=True)
    if isinstance(model_out, tuple):
        out = model_out[0]
        if len(model_out) >= 3:
            pre_inverse = model_out[1]
            raw = model_out[2]
        elif len(model_out) == 2:
            pre_inverse = model_out[1]
            raw = model_out[1]
        else:
            pre_inverse = out
            raw = out
    else:
        out = model_out
        pre_inverse = model_out
        raw = model_out
    return out, pre_inverse, raw


def infer_batch_ensemble(
    *,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    infer_batch: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    boundary_cfg: Any,
    head_type: str,
    ensemble_size: int,
    base_seed: int,
    device: torch.device,
    autocast_context: Callable[[], Any] | None = None,
    force_float32: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one deterministic pass or multiple independent diffusion samples.

    Deterministic outputs keep shape ``[B, V, H, W]``. Diffusion ensemble outputs
    use shape ``[B, E, V, H, W]``. Each diffusion member owns a persistent RNG
    stream cached on ``model``: repeated batch calls advance the stream instead
    of replaying the same spatial noise. Call ``reset_ensemble_generators`` at
    each scenario boundary to reproduce that scenario independently of run
    ordering. Exact stochastic replay currently also requires the same batch
    partitioning because one sampler call consumes one batched RNG sequence.
    """
    is_diffusion = canonical_head_type(head_type) == "diffusion"
    n_members = resolve_ensemble_size(head_type=head_type, requested=ensemble_size) if is_diffusion else 1
    autocast_context = autocast_context or nullcontext
    ensemble_generators = (
        _resolve_ensemble_generators(
            model=model,
            ensemble_size=n_members,
            base_seed=base_seed,
            device=device,
        )
        if is_diffusion
        else None
    )

    outs: list[torch.Tensor] = []
    pre_inverse_outs: list[torch.Tensor] = []
    raw_outs: list[torch.Tensor] = []

    for member_idx in range(n_members):
        rng_context = (
            ensemble_generators.activate(member_idx)
            if ensemble_generators is not None
            else nullcontext()
        )
        with rng_context:
            with autocast_context():
                out, pre_inverse, raw = infer_batch(
                    model=model, batch=batch, cfg=boundary_cfg
                )
        if force_float32:
            out = out.float()
            pre_inverse = pre_inverse.float()
            raw = raw.float()
        outs.append(out)
        pre_inverse_outs.append(pre_inverse)
        raw_outs.append(raw)
        if not is_diffusion:
            break

    if is_diffusion:
        return (
            torch.stack(outs, dim=1),
            torch.stack(pre_inverse_outs, dim=1),
            torch.stack(raw_outs, dim=1),
        )
    return outs[0], pre_inverse_outs[0], raw_outs[0]


def residual_transformation_stages(
    *,
    model: torch.nn.Module,
    truth_physical: torch.Tensor,
    final_physical: torch.Tensor,
    full_standardized: torch.Tensor,
    generated_residual_standardized: torch.Tensor,
    baseline_standardized: torch.Tensor,
    baseline_physical: torch.Tensor | None = None,
    residual_application_scale: float | None = None,
) -> dict[str, torch.Tensor]:
    """Reconstruct and validate every additive-residual inference stage.

    The residual model returns the complete standardized prediction and the
    raw generated standardized correction separately. The deployed full field
    is ``baseline + alpha * raw_residual``. The baseline must be supplied
    directly by the model; recovering it as ``full - residual`` in float32 is
    numerically fragile when terms nearly cancel. Decoding that baseline
    (rather than decoding a residual by itself) is valid for all configured
    target transforms and final precipitation constraints.

    ``final_physical`` may be ``[B,C,H,W]`` or the diffusion-ensemble shape
    ``[B,E,C,H,W]``. Truth and deterministic baseline stages intentionally do
    not acquire a redundant ensemble dimension; generated stages retain it.
    """
    if truth_physical.ndim != 4:
        raise ValueError(
            "truth_physical must have shape [B,C,H,W], got "
            f"{tuple(truth_physical.shape)}"
        )
    if final_physical.shape != full_standardized.shape:
        raise ValueError(
            "final physical and standardized predictions must share a shape; "
            f"got {tuple(final_physical.shape)} and {tuple(full_standardized.shape)}"
        )
    if full_standardized.shape != generated_residual_standardized.shape:
        raise ValueError(
            "full and residual standardized predictions must share a shape; "
            f"got {tuple(full_standardized.shape)} and "
            f"{tuple(generated_residual_standardized.shape)}"
        )
    if final_physical.ndim not in (4, 5):
        raise ValueError(
            "residual predictions must have shape [B,C,H,W] or [B,E,C,H,W], "
            f"got {tuple(final_physical.shape)}"
        )
    if baseline_standardized.ndim != 4:
        raise ValueError(
            "baseline_standardized must have shape [B,C,H,W], got "
            f"{tuple(baseline_standardized.shape)}"
        )
    if tuple(baseline_standardized.shape) != tuple(truth_physical.shape):
        raise ValueError(
            "baseline/truth dimensions are incompatible: "
            f"{tuple(baseline_standardized.shape)} vs {tuple(truth_physical.shape)}"
        )

    # DDP and torch.compile wrap the methods/config that own target transforms.
    transform_model: Any = model
    for attribute in ("module", "_orig_mod"):
        candidate = getattr(transform_model, attribute, None)
        if candidate is not None:
            transform_model = candidate
    if residual_application_scale is None:
        head = getattr(transform_model, "diffusion_head", None)
        cfg = getattr(head, "cfg", None)
        residual_application_scale = float(
            getattr(cfg, "residual_application_scale", 1.0)
        )
    application_scale = float(residual_application_scale)
    if not np.isfinite(application_scale) or not 0.0 <= application_scale <= 1.0:
        raise ValueError(
            "residual_application_scale must be finite and in [0, 1], got "
            f"{application_scale!r}."
        )

    applied_residual_standardized = (
        generated_residual_standardized * application_scale
    )
    if final_physical.ndim == 5:
        if final_physical.shape[0] != truth_physical.shape[0] or tuple(
            final_physical.shape[2:]
        ) != tuple(truth_physical.shape[1:]):
            raise ValueError(
                "ensemble prediction/truth dimensions are incompatible: "
                f"{tuple(final_physical.shape)} vs {tuple(truth_physical.shape)}"
            )
        baseline_for_final = baseline_standardized[:, None]
    else:
        if tuple(final_physical.shape) != tuple(truth_physical.shape):
            raise ValueError(
                "prediction/truth dimensions are incompatible: "
                f"{tuple(final_physical.shape)} vs {tuple(truth_physical.shape)}"
            )
        baseline_for_final = baseline_standardized

    expected_full_standardized = (
        baseline_for_final + applied_residual_standardized
    )
    try:
        torch.testing.assert_close(
            full_standardized,
            expected_full_standardized,
            rtol=2e-5,
            atol=2e-6,
        )
    except AssertionError as exc:
        max_error = float(
            (full_standardized - expected_full_standardized).abs().max().item()
        )
        raise RuntimeError(
            "The full standardized prediction is incompatible with the direct "
            "deterministic U-Net baseline and applied diffusion residual "
            f"(max standardized difference={max_error:.6g}); the baseline changed "
            "across ensemble members or the residual/full outputs are inconsistent."
        ) from exc

    encode = getattr(transform_model, "_encode_targets_std", None)
    decode = getattr(transform_model, "_decode_targets_std", None)
    if not callable(encode) or not callable(decode):
        raise TypeError(
            "Residual stage diagnostics require model._encode_targets_std and "
            "model._decode_targets_std."
        )

    truth_standardized = encode(truth_physical)
    if baseline_physical is None:
        baseline_physical = decode(baseline_standardized)
    elif tuple(baseline_physical.shape) != tuple(truth_physical.shape):
        raise ValueError(
            "physical baseline/truth dimensions are incompatible: "
            f"{tuple(baseline_physical.shape)} vs {tuple(truth_physical.shape)}"
        )
    true_residual_physical = truth_physical - baseline_physical
    true_residual_standardized = truth_standardized - baseline_standardized
    baseline_physical_for_final = (
        baseline_physical[:, None]
        if final_physical.ndim == 5
        else baseline_physical
    )
    applied_residual_physical = final_physical - baseline_physical_for_final

    # A nonlinear target transform cannot decode a residual in isolation. Decode
    # the counterfactual baseline + raw residual, then subtract the same baseline.
    raw_full_standardized = baseline_for_final + generated_residual_standardized
    if raw_full_standardized.ndim == 5:
        batch_size, ensemble_size = raw_full_standardized.shape[:2]
        raw_full_physical = decode(raw_full_standardized.flatten(0, 1)).unflatten(
            0, (batch_size, ensemble_size)
        )
    else:
        raw_full_physical = decode(raw_full_standardized)
    raw_residual_physical = raw_full_physical - baseline_physical_for_final

    stages = {
        "physical_unet_prediction": baseline_physical,
        "physical_ground_truth": truth_physical,
        "physical_true_residual": true_residual_physical,
        "normalized_true_residual": true_residual_standardized,
        "raw_predicted_normalized_residual": generated_residual_standardized,
        "predicted_normalized_residual": generated_residual_standardized,
        "applied_normalized_residual": applied_residual_standardized,
        "raw_denormalized_predicted_residual": raw_residual_physical,
        "denormalized_predicted_residual": raw_residual_physical,
        "applied_denormalized_residual": applied_residual_physical,
        "final_physical_prediction": final_physical,
    }
    for name, value in stages.items():
        if not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(
                f"Residual transformation stage {name!r} contains NaN or infinity."
            )

    # Keep the additive bookkeeping checks as hard failures. These expressions
    # subtract and then re-add float32 values; near-zero precipitation can
    # therefore lose a few ULPs when larger baseline/residual terms cancel.
    # PyTorch's standard float32 absolute tolerance is 1e-5. The independent
    # ensemble-baseline consistency check above intentionally remains tighter.
    torch.testing.assert_close(
        truth_physical,
        baseline_physical + true_residual_physical,
        rtol=2e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        final_physical,
        baseline_physical_for_final + applied_residual_physical,
        rtol=2e-5,
        atol=1e-5,
    )
    return stages


def variable_array_map(var_names: list[str], values: np.ndarray) -> dict[str, np.ndarray]:
    """Map ``[T,V,H,W]`` or ``[T,E,V,H,W]`` arrays to per-variable arrays."""
    arr = np.asarray(values, dtype=np.float32)
    out: dict[str, np.ndarray] = {}
    for var_idx, var_name in enumerate(var_names):
        if arr.ndim == 5:
            out[var_name] = arr[:, :, var_idx]
        else:
            out[var_name] = arr[:, var_idx]
    return out


def add_predictions_to_dataset(
    *,
    prediction_ds: xr.Dataset,
    target_vars: list[str],
    outputs_np: np.ndarray,
    coords: dict[str, Any],
    time_dim: str,
    lat_dim: str,
    lon_dim: str,
    target_attrs: dict[str, dict[str, Any]],
    head_type: str,
    ensemble_size: int,
    base_seed: int,
) -> xr.Dataset:
    """Populate prediction variables, adding ensemble dim only for diffusion."""
    is_diffusion = canonical_head_type(head_type) == "diffusion"
    if is_diffusion:
        prediction_ds = prediction_ds.assign_coords(ensemble=np.arange(int(ensemble_size), dtype=np.int32))
        dims = (time_dim, "ensemble", lat_dim, lon_dim)
    else:
        dims = (time_dim, lat_dim, lon_dim)

    for var_idx, name in enumerate(target_vars):
        values = outputs_np[:, :, var_idx] if is_diffusion else outputs_np[:, var_idx]
        prediction_ds[name] = xr.DataArray(
            values,
            dims=dims,
            coords={dim: prediction_ds.coords[dim] for dim in dims},
            attrs=target_attrs.get(name, {}),
        ).astype(np.float32)

    if is_diffusion:
        prediction_ds.attrs.update(
            {
                "head_type": "diffusion",
                "ensemble_size": int(ensemble_size),
                "ensemble_generation": "diffusion_sampling",
                "base_seed": int(base_seed),
            }
        )
    else:
        prediction_ds.attrs.setdefault("head_type", "deterministic")
    return prediction_ds


def as_ensemble_mean(values: np.ndarray, *, ensemble_axis: int = 1) -> np.ndarray:
    """Default evaluation policy for diffusion outputs: evaluate ensemble mean."""
    arr = np.asarray(values)
    axis = int(ensemble_axis)
    if axis < 0:
        axis += arr.ndim
    if axis < 0 or axis >= arr.ndim:
        return arr
    if arr.shape[axis] < 1:
        raise ValueError("Cannot compute an ensemble mean over an empty dimension.")
    # Mean even a singleton ensemble so downstream evaluation always receives
    # the same [T,V,H,W] contract for E=1 and E>1 diffusion output.
    return arr.mean(axis=axis)
