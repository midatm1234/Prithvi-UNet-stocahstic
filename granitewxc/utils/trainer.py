from collections.abc import Callable, Mapping
from typing import Dict

import math
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm as auto_tqdm
from tqdm.std import tqdm as terminal_tqdm
from time import time

from granitewxc.utils.distributed import is_main_process
from granitewxc.utils.checkpoint_metadata import build_checkpoint_metadata

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as TorchFSDP,
        FullOptimStateDictConfig,
        FullStateDictConfig,
        StateDictType,
    )
except Exception:
    TorchFSDP = None
    FullOptimStateDictConfig = None
    FullStateDictConfig = None
    StateDictType = None



def _to_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().float().mean().item()
    return value


def _default_checkpoint_dir(config) -> str:
    """Resolve the case-scoped checkpoint directory with a legacy fallback."""

    path_checkpoints = None
    if bool(getattr(config, "derive_output_paths", False)):
        try:
            path_checkpoints = getattr(config, "path_checkpoints", None)
        except (AttributeError, ValueError):
            # Lightweight test configurations may not define a case name.
            path_checkpoints = None
    if path_checkpoints:
        return str(path_checkpoints)
    return os.path.join(config.path_experiment, "weights")


def _loader_step_count(loader: DataLoader, limit_steps: int = 0) -> int:
    if 0 < limit_steps:
        return min(int(limit_steps), len(loader))
    return len(loader)


def _make_epoch_pbar(desc: str, total: int, colour: str):
    if not is_main_process():
        return None
    # Headless notebook runners normally make ``tqdm.auto`` choose a Jupyter
    # display widget. Papermill stores those display updates in the executed
    # notebook instead of emitting a useful live terminal progress bar. The
    # tmux launcher opts into the plain-text renderer through this environment
    # variable; interactive notebooks retain their normal automatic renderer.
    tqdm_cls = (
        terminal_tqdm
        if os.environ.get("GRANITEWXC_TQDM_MODE", "").strip().lower()
        in {"terminal", "text"}
        else auto_tqdm
    )
    return tqdm_cls(
        total=total,
        unit="batch",
        colour=colour,
        desc=desc,
        dynamic_ncols=True,
        leave=True,
        position=0,
    )


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device | int):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _find_latest_epoch_checkpoint(checkpoint_dir: str | os.PathLike | None) -> str | None:
    if checkpoint_dir is None:
        return None
    checkpoint_dir = str(checkpoint_dir)
    if not os.path.isdir(checkpoint_dir):
        return None

    candidates: list[tuple[int, str]] = []
    for name in os.listdir(checkpoint_dir):
        if not (name.startswith("epoch_") and name.endswith(".ckpt")):
            continue
        epoch_str = name[len("epoch_") : -len(".ckpt")]
        if not epoch_str.isdigit():
            continue
        candidates.append((int(epoch_str), os.path.join(checkpoint_dir, name)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _resolve_resume_probe_checkpoint(checkpoint_dir: str | os.PathLike | None) -> str | None:
    latest_epoch = _find_latest_epoch_checkpoint(checkpoint_dir)
    if latest_epoch is not None:
        return latest_epoch
    if checkpoint_dir is None:
        return None
    checkpoint_dir = str(checkpoint_dir)
    for leaf in ("last.ckpt", "best.ckpt"):
        candidate = os.path.join(checkpoint_dir, leaf)
        if os.path.exists(candidate):
            return candidate
    return None


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _auto_resume_enabled(config) -> bool:
    top_level = _coerce_bool(getattr(config, "auto_resume_if_checkpoint_exists", None))
    if top_level is not None:
        return top_level

    training_cfg = getattr(config, "training", None)
    if isinstance(training_cfg, dict):
        nested = _coerce_bool(training_cfg.get("auto_resume_if_checkpoint_exists"))
    else:
        nested = _coerce_bool(
            getattr(training_cfg, "auto_resume_if_checkpoint_exists", None) if training_cfg is not None else None
        )
    if nested is not None:
        return nested

    env = _coerce_bool(os.environ.get("CORDEX_AUTO_RESUME_IF_CHECKPOINT", "1"))
    if env is not None:
        return env
    return True


def _coerce_mapping(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _validation_enabled(config) -> bool:
    """Resolve validation policy, defaulting to historical enabled behavior."""

    top_level_raw = getattr(config, "validation_enabled", None)
    top_level = _coerce_bool(top_level_raw)
    if top_level is not None:
        return top_level
    if top_level_raw is not None:
        raise ValueError(
            "config.validation_enabled must be a boolean-like value, got "
            f"{top_level_raw!r}."
        )
    training_cfg = getattr(config, "training", None)
    nested_raw = (
        training_cfg.get("validation_enabled")
        if isinstance(training_cfg, dict)
        else getattr(training_cfg, "validation_enabled", None)
        if training_cfg is not None
        else None
    )
    nested = _coerce_bool(nested_raw)
    if nested is not None:
        return nested
    if nested_raw is not None:
        raise ValueError(
            "config.training.validation_enabled must be a boolean-like value, "
            f"got {nested_raw!r}."
        )
    return True


def _resolve_correction_validation(config, loss_func, validation_enabled: bool):
    validation_cfg = _coerce_mapping(getattr(config, "validation", None))
    raw_cfg = _coerce_mapping(validation_cfg.get("residual_correction"))
    enabled_raw = raw_cfg.get("enabled", False)
    enabled = _coerce_bool(enabled_raw)
    if enabled is None:
        raise ValueError(
            "validation.residual_correction.enabled must be boolean-like, got "
            f"{enabled_raw!r}."
        )
    if not enabled:
        return None
    if not validation_enabled:
        raise ValueError(
            "Residual-correction qualification requires validation_enabled=true."
        )
    evaluator = getattr(loss_func, "evaluate_configured_prediction", None)
    if not callable(evaluator):
        raise TypeError(
            "Residual-correction validation requires a loss exposing "
            "evaluate_configured_prediction()."
        )
    ensemble_size = int(raw_cfg.get("ensemble_size", 3))
    if ensemble_size < 2:
        raise ValueError(
            "validation.residual_correction.ensemble_size must be at least 2."
        )
    application_scale = float(raw_cfg.get("application_scale", 1.0))
    if not math.isfinite(application_scale) or application_scale != 1.0:
        raise ValueError(
            "A full-correction qualification requires "
            "validation.residual_correction.application_scale=1.0."
        )
    minimum_relative_improvement = float(
        getattr(loss_func, "minimum_relative_improvement", 0.0)
    )
    require_terms = _coerce_bool(
        raw_cfg.get("require_each_configured_term_non_degradation", False)
    )
    if require_terms is None:
        raise ValueError(
            "validation.residual_correction."
            "require_each_configured_term_non_degradation must be boolean-like."
        )
    model_cfg = _coerce_mapping(getattr(config, "model", None))
    diffusion_cfg = _coerce_mapping(model_cfg.get("diffusion"))
    return {
        "ensemble_size": ensemble_size,
        "base_seed": int(raw_cfg.get("base_seed", 42)),
        "application_scale": application_scale,
        "minimum_relative_improvement": minimum_relative_improvement,
        "require_each_configured_term_non_degradation": require_terms,
        "term_tolerance": float(raw_cfg.get("term_tolerance", 0.0)),
        "limit_steps": int(raw_cfg.get("limit_steps", 0) or 0),
        "sampling_method": str(diffusion_cfg.get("sampling_method", "unknown")),
        "num_sampling_steps": int(diffusion_cfg.get("num_sampling_steps", 0) or 0),
        "eta": float(diffusion_cfg.get("eta", 0.0) or 0.0),
    }


def _checkpoint_head_metadata(
    config,
    model,
    *,
    epoch: int | None = None,
    global_step: int | None = None,
) -> dict[str, object]:
    return build_checkpoint_metadata(
        config, model, epoch=epoch, global_step=global_step
    )


def _save_epoch_checkpoints_enabled(config) -> bool:
    value = _coerce_bool(getattr(config, "save_epoch_checkpoints", None))
    return True if value is None else value


def _iter_wrapped_modules(model: torch.nn.Module):
    """Yield a model and common wrapper modules (DDP/FSDP) from outermost to innermost."""
    seen: set[int] = set()
    current = model
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        next_module = getattr(current, "module", None)
        if next_module is None:
            next_module = getattr(current, "_fsdp_wrapped_module", None)
        current = next_module


def _is_fsdp_wrapped_model(model: torch.nn.Module) -> bool:
    if TorchFSDP is None:
        return False
    return any(isinstance(candidate, TorchFSDP) for candidate in _iter_wrapped_modules(model))


def _checkpoint_requires_all_ranks(model: torch.nn.Module) -> bool:
    if not (dist.is_available() and dist.is_initialized()):
        return False
    return _is_fsdp_wrapped_model(model)


def _assert_finite_scalar(value: torch.Tensor, name: str, local_rank: int) -> None:
    """Raise on every distributed rank if any rank sees a non-finite scalar."""
    is_bad = torch.zeros(1, device=value.device, dtype=torch.int32)
    if not torch.isfinite(value.detach()).all():
        is_bad.fill_(1)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(is_bad, op=dist.ReduceOp.SUM)
    if int(is_bad.item()) > 0:
        local_value = _to_scalar(value)
        raise FloatingPointError(
            f"Non-finite {name} detected on rank {local_rank}: {local_value}. "
            "Stopping before optimizer/checkpoint state is corrupted."
        )


def _collect_checkpoint_states(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[dict, dict | None]:
    """
    Gather model/optimizer states for checkpointing.

    For FSDP + distributed, all ranks must participate in state-dict collection.
    We use FULL_STATE_DICT with rank0_only to avoid materializing full states
    on non-zero ranks.
    """
    if (
        TorchFSDP is not None
        and StateDictType is not None
        and FullStateDictConfig is not None
        and _checkpoint_requires_all_ranks(model)
    ):
        fsdp_root = None
        for candidate in _iter_wrapped_modules(model):
            if isinstance(candidate, TorchFSDP):
                fsdp_root = candidate
                break

        if fsdp_root is not None:
            full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            try:
                if FullOptimStateDictConfig is not None:
                    full_optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with TorchFSDP.state_dict_type(
                        fsdp_root,
                        StateDictType.FULL_STATE_DICT,
                        full_cfg,
                        full_optim_cfg,
                    ):
                        model_state = model.state_dict()
                        if optimizer is not None:
                            try:
                                optimizer_state = TorchFSDP.optim_state_dict(fsdp_root, optimizer)
                            except Exception:
                                optimizer_state = optimizer.state_dict()
                        else:
                            optimizer_state = None
                else:
                    with TorchFSDP.state_dict_type(
                        fsdp_root,
                        StateDictType.FULL_STATE_DICT,
                        full_cfg,
                    ):
                        model_state = model.state_dict()
                        optimizer_state = optimizer.state_dict() if optimizer is not None else None
                return model_state, optimizer_state
            except Exception as exc:
                if is_main_process():
                    print(
                        "[checkpoint] warning: FSDP full-state collection failed; "
                        f"falling back to generic state_dict ({exc})"
                    )

    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    return model_state, optimizer_state


def _fsdp_root_module(model: torch.nn.Module):
    if TorchFSDP is None:
        return None
    for candidate in _iter_wrapped_modules(model):
        if isinstance(candidate, TorchFSDP):
            return candidate
    return None


def _restore_checkpoint_states(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    model_state: dict,
    optimizer_state: dict | None,
) -> None:
    """
    Load model/optimizer states saved by :func:`_collect_checkpoint_states`.

    Checkpoints for FSDP models are stored as FULL_STATE_DICT (unsharded).
    The optimizer state in particular must be re-sharded to match the local
    flat parameters via ``FSDP.optim_state_dict_to_load`` before it can be
    loaded; a plain ``optimizer.load_state_dict`` would install full-size
    momentum buffers that mismatch the sharded gradients at ``optimizer.step``.
    """
    fsdp_root = None
    if (
        TorchFSDP is not None
        and StateDictType is not None
        and FullStateDictConfig is not None
        and _checkpoint_requires_all_ranks(model)
    ):
        fsdp_root = _fsdp_root_module(model)

    if fsdp_root is not None:
        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        if FullOptimStateDictConfig is not None:
            full_optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
            state_dict_ctx = TorchFSDP.state_dict_type(
                fsdp_root, StateDictType.FULL_STATE_DICT, full_cfg, full_optim_cfg,
            )
        else:
            state_dict_ctx = TorchFSDP.state_dict_type(
                fsdp_root, StateDictType.FULL_STATE_DICT, full_cfg,
            )
        with state_dict_ctx:
            model.load_state_dict(model_state, strict=True)
            if optimizer is not None and optimizer_state is not None:
                sharded_optimizer_state = TorchFSDP.optim_state_dict_to_load(
                    model, optimizer, optimizer_state,
                )
                optimizer.load_state_dict(sharded_optimizer_state)
        return

    model.load_state_dict(model_state, strict=True)
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)


def _inject_precip_hurdle_aux(batch: Dict[str, torch.Tensor], model: torch.nn.Module) -> None:
    """
    Ensure hurdle aux tensors are present on the caller batch, even when wrappers
    (e.g. DDP/FSDP) forward with copied input containers.
    """
    aux = None
    for candidate in _iter_wrapped_modules(model):
        getter = getattr(candidate, "get_last_precip_hurdle_aux", None)
        if callable(getter):
            aux = getter()
            break

    if aux is not None:
        batch["__precip_hurdle_aux"] = aux
    else:
        batch.pop("__precip_hurdle_aux", None)


def _inject_bernoulli_gamma_aux(
    batch: Dict[str, torch.Tensor], model: torch.nn.Module
) -> None:
    """Publish BG parameter tensors through DDP/FSDP wrapper boundaries."""

    aux = None
    for candidate in _iter_wrapped_modules(model):
        getter = getattr(candidate, "get_last_bernoulli_gamma_aux", None)
        if callable(getter):
            aux = getter()
            break
        value = getattr(candidate, "_last_bg_aux", None)
        if value is not None:
            aux = value
            break
    if aux is not None:
        batch["__bernoulli_gamma_aux"] = aux
    else:
        batch.pop("__bernoulli_gamma_aux", None)


def _last_diffusion_baseline(model: torch.nn.Module) -> torch.Tensor:
    for candidate in _iter_wrapped_modules(model):
        getter = getattr(candidate, "get_last_diffusion_baseline", None)
        if callable(getter):
            pair = getter()
            if pair is not None:
                physical, _ = pair
                if torch.is_tensor(physical):
                    return physical
    raise RuntimeError(
        "Diffusion sampling did not publish its paired deterministic baseline."
    )


def _sampled_correction_batch(
    *,
    batch: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    loss_func,
    correction_validation: Mapping[str, object],
    batch_index: int,
    gpu: bool,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, dict[str, float]]:
    """Score the paired baseline and actual stochastic ensemble mean."""

    ensemble_size = int(correction_validation["ensemble_size"])
    base_seed = int(correction_validation["base_seed"])
    application_scale = float(correction_validation["application_scale"])
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    input_tensor = batch.get("x")
    if not torch.is_tensor(input_tensor):
        raise TypeError("Correction validation requires tensor batch['x'].")
    device = input_tensor.device

    ensemble_sum: torch.Tensor | None = None
    baseline_physical: torch.Tensor | None = None
    for member_index in range(ensemble_size):
        seed = (
            base_seed
            + rank * 1_000_000_000
            + batch_index * ensemble_size
            + member_index
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        if gpu:
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
            with autocast(device_type="cuda", dtype=dtype):
                sampled = model(
                    batch,
                    return_pre_inverse=True,
                    diffusion_generator=generator,
                    diffusion_application_scale=application_scale,
                )
        else:
            sampled = model(
                batch,
                return_pre_inverse=True,
                diffusion_generator=generator,
                diffusion_application_scale=application_scale,
            )
        physical_sample = sampled[0] if isinstance(sampled, tuple) else sampled
        if not torch.is_tensor(physical_sample):
            raise TypeError(
                "Diffusion sampling must return a physical prediction tensor."
            )
        physical_sample = physical_sample.float()
        ensemble_sum = (
            physical_sample
            if ensemble_sum is None
            else ensemble_sum + physical_sample
        )
        if baseline_physical is None:
            baseline_physical = _last_diffusion_baseline(model).float()

    if ensemble_sum is None or baseline_physical is None:
        raise RuntimeError("Correction validation produced no ensemble members.")
    ensemble_mean = ensemble_sum / float(ensemble_size)
    evaluate = loss_func.evaluate_configured_prediction
    baseline_loss, baseline_terms = evaluate(baseline_physical, batch)
    ensemble_loss, ensemble_terms = evaluate(ensemble_mean, batch)
    return baseline_loss, baseline_terms, ensemble_loss, ensemble_terms


def batch_step(
    batch: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    loss_func: Callable,
    gpu: bool,
    local_rank: int,
):
    if gpu:
        # Pinned DataLoader batches can overlap their host-to-device copy with
        # GPU work.  ``non_blocking`` is harmless for non-pinned tensors.
        batch = {k: v.to(local_rank, non_blocking=True) for k, v in batch.items()}
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with autocast(device_type="cuda", dtype=dtype):
            prediction = model(batch)
            _inject_precip_hurdle_aux(batch, model)
            _inject_bernoulli_gamma_aux(batch, model)
            loss = loss_func(prediction, batch)
    else:
        prediction = model(batch)
        _inject_precip_hurdle_aux(batch, model)
        _inject_bernoulli_gamma_aux(batch, model)
        loss = loss_func(prediction, batch)

    return loss


def validate_one_epoch(
    model: torch.nn.Module,
    local_rank: int,
    validation_loader: DataLoader,
    loss_func,
    epoch: int,
    gpu: bool,
    limit_steps: int = 0,
    correction_validation: Mapping[str, object] | None = None,
):
    model.eval()
    ddp_loss = torch.zeros(2)
    inner_pbar = None
    benchmark_data = np.zeros(2)
    benchmark_forward = np.zeros(2)
    benchmark_total = np.zeros(2)
    benchmark_samples = 0
    correction_batches = 0
    correction_count = torch.zeros((), dtype=torch.float64)
    correction_batch_count = torch.zeros((), dtype=torch.float64)
    correction_sums: dict[str, torch.Tensor] = {}

    if gpu:
        ddp_loss = ddp_loss.to(local_rank)
        correction_count = correction_count.to(local_rank)
        correction_batch_count = correction_batch_count.to(local_rank)
    
    sampler = validation_loader.sampler
    if hasattr(sampler, 'set_epoch'):
        sampler.set_epoch(epoch)
    elif dist.is_available() and dist.is_initialized():
        print('WARNING: Not calling set_epoch.')

    max_steps = _loader_step_count(validation_loader, limit_steps)
    inner_pbar = _make_epoch_pbar(
        desc=f"Validation {epoch + 1}",
        total=max_steps,
        colour="green",
    )

    # Inference mode further reduces autograd metadata/memory compared to no_grad.
    with torch.inference_mode():
        benchmark_timer, benchmark_timer_total = time(), time()
        for i, batch in enumerate(validation_loader):
            if 0 < limit_steps <= i:
                break

            benchmark_data[0] += time() - benchmark_timer
            benchmark_data[1] += 1

            benchmark_timer = time()
            loss = batch_step(
                batch, model, loss_func, gpu, local_rank
            )
            benchmark_forward[0] += time() - benchmark_timer
            benchmark_forward[1] += 1

            ddp_loss[0] += loss.item()  # sum up batch loss
            ddp_loss[1] += 1

            correction_limit = (
                int(correction_validation.get("limit_steps", 0) or 0)
                if correction_validation is not None
                else 0
            )
            run_correction = correction_validation is not None and (
                correction_limit <= 0 or correction_batches < correction_limit
            )
            if run_correction:
                (
                    baseline_configured_loss,
                    baseline_terms,
                    ensemble_configured_loss,
                    ensemble_terms,
                ) = _sampled_correction_batch(
                    batch=batch,
                    model=model,
                    loss_func=loss_func,
                    correction_validation=correction_validation,
                    batch_index=i,
                    gpu=gpu,
                )
                batch_size = int(batch["y"].shape[0])
                correction_count += batch_size
                correction_batch_count += 1
                paired_values = {
                    "baseline.total": baseline_configured_loss,
                    "ensemble_mean.total": ensemble_configured_loss,
                }
                paired_values.update(
                    {
                        f"baseline.term.{name}": value
                        for name, value in baseline_terms.items()
                    }
                )
                paired_values.update(
                    {
                        f"ensemble_mean.term.{name}": value
                        for name, value in ensemble_terms.items()
                    }
                )
                for name, value in paired_values.items():
                    scalar = (
                        value.detach().to(
                            device=correction_count.device, dtype=torch.float64
                        )
                        if isinstance(value, torch.Tensor)
                        else torch.tensor(
                            float(value),
                            device=correction_count.device,
                            dtype=torch.float64,
                        )
                    )
                    if scalar.numel() != 1:
                        raise ValueError(
                            f"Correction validation term {name!r} is not scalar."
                        )
                    correction_sums.setdefault(
                        name, torch.zeros_like(correction_count)
                    ).add_(scalar.reshape(()) * batch_size)
                correction_batches += 1

            if inner_pbar is not None:
                inner_pbar.update(1)
                inner_pbar.set_postfix(loss=loss.item())

            # model.module.swap_masking()

            # Batch size; computed in this way for compatibility between Hiera and Swin branches
            benchmark_samples += len(next(iter(batch.values())))
            benchmark_total[0] += time() - benchmark_timer_total
            benchmark_total[1] += 1
            benchmark_timer, benchmark_timer_total = time(), time()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        if correction_validation is not None:
            dist.all_reduce(correction_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(correction_batch_count, op=dist.ReduceOp.SUM)
            for name in sorted(correction_sums):
                dist.all_reduce(correction_sums[name], op=dist.ReduceOp.SUM)
    val_loss = ddp_loss[0] / max(ddp_loss[1], 1)

    if inner_pbar is not None:
        inner_pbar.close()

    metrics = {
        'val.benchmark.data': benchmark_data[0] / benchmark_data[1],
        'val.benchmark.forward': benchmark_forward[0] / benchmark_forward[1],
        'val.benchmark.total': benchmark_total[0] / benchmark_total[1],
        'val.benchmark.samples': benchmark_samples,
    }

    if correction_validation is not None:
        paired_sample_count = float(correction_count.detach().cpu().item())
        if paired_sample_count <= 0.0:
            raise RuntimeError(
                "Residual-correction validation was enabled but evaluated no samples."
            )
        averaged = {
            name: float((value / correction_count).detach().cpu().item())
            for name, value in correction_sums.items()
        }
        baseline_total = averaged["baseline.total"]
        ensemble_total = averaged["ensemble_mean.total"]
        minimum_relative_improvement = float(
            correction_validation["minimum_relative_improvement"]
        )
        finite = math.isfinite(baseline_total) and math.isfinite(ensemble_total)
        total_qualified = finite and ensemble_total <= baseline_total * (
            1.0 - minimum_relative_improvement
        )
        term_qualified = True
        failed_terms: list[str] = []
        if bool(
            correction_validation[
                "require_each_configured_term_non_degradation"
            ]
        ):
            tolerance = float(correction_validation["term_tolerance"])
            baseline_prefix = "baseline.term."
            for key, baseline_value in averaged.items():
                if not key.startswith(baseline_prefix):
                    continue
                term_name = key[len(baseline_prefix) :]
                ensemble_key = f"ensemble_mean.term.{term_name}"
                if ensemble_key not in averaged:
                    failed_terms.append(term_name)
                    term_qualified = False
                    continue
                ensemble_value = averaged[ensemble_key]
                if (
                    not math.isfinite(baseline_value)
                    or not math.isfinite(ensemble_value)
                    or ensemble_value > baseline_value + tolerance
                ):
                    failed_terms.append(term_name)
                    term_qualified = False
        qualified = total_qualified and term_qualified
        relative_improvement = (
            (baseline_total - ensemble_total) / max(abs(baseline_total), 1e-12)
        )
        metrics.update(
            {f"val.correction.{name}": value for name, value in averaged.items()}
        )
        metrics.update(
            {
                "val.correction.relative_improvement": relative_improvement,
                "val.correction.minimum_relative_improvement": minimum_relative_improvement,
                "val.correction.total_qualified": float(total_qualified),
                "val.correction.terms_qualified": float(term_qualified),
                "val.correction.qualified": float(qualified),
                "val.correction.failed_terms": failed_terms,
                "val.correction.ensemble_size": int(
                    correction_validation["ensemble_size"]
                ),
                "val.correction.base_seed": int(
                    correction_validation["base_seed"]
                ),
                "val.correction.application_scale": float(
                    correction_validation["application_scale"]
                ),
                "val.correction.sample_count": int(paired_sample_count),
                "val.correction.batch_count": int(
                    correction_batch_count.detach().cpu().item()
                ),
                "val.correction.sampling_method": str(
                    correction_validation["sampling_method"]
                ),
                "val.correction.num_sampling_steps": int(
                    correction_validation["num_sampling_steps"]
                ),
                "val.correction.eta": float(correction_validation["eta"]),
            }
        )

    return val_loss, metrics


def train_one_epoch(
    model: torch.nn.Module,
    local_rank: int,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_func,
    epoch: int,
    scaler: None,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    gpu: bool = False,
    limit_steps: int = 0,
    **kwargs,
):
    '''
    Regarding ShardedGradScaler, see the example in the documentation: https://github.com/pytorch/pytorch/blob/e5841bb8d5aa1f413cdec1c904ed9b68b91ea356/torch/distributed/fsdp/sharded_grad_scaler.py#L50
    '''
    model.train()
    ddp_loss = torch.zeros(2)
    node_count = torch.ones(1)
    inner_pbar = None
    benchmark_data = np.zeros(2)
    benchmark_forward = np.zeros(2)
    benchmark_backward = np.zeros(2)
    benchmark_optimizer = np.zeros(2)
    benchmark_total = np.zeros(2)
    benchmark_samples = 0

    if gpu:
        ddp_loss = ddp_loss.to(local_rank)
        node_count = node_count.to(local_rank)

    sampler = train_loader.sampler
    if hasattr(sampler, 'set_epoch'):
        sampler.set_epoch(epoch)
    elif dist.is_available() and dist.is_initialized():
        print('WARNING: Not calling set_epoch.')

    if is_main_process():
        if kwargs.get('num_epochs') is not None:
            num_epochs = kwargs['num_epochs']
        else:
            num_epochs = 0

        inner_pbar = _make_epoch_pbar(
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            total=_loader_step_count(train_loader, limit_steps),
            colour="blue",
        )

    benchmark_timer, benchmark_timer_total = time(), time()

    benchmark_batch_mean = np.zeros(2)
    accumulation_steps = max(1, int(kwargs.get("gradient_accumulation_steps", 1)))
    max_grad_norm = float(kwargs.get("max_grad_norm", 0.0) or 0.0)
    optimizer_steps = 0
    max_steps = _loader_step_count(train_loader, limit_steps)
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(train_loader):
        if 0 < limit_steps <= i:
            break

        benchmark_batch_mean[0] += batch['x'].mean()
        benchmark_batch_mean[1] += 1

        benchmark_data[0] += time() - benchmark_timer
        benchmark_data[1] += 1

        benchmark_timer = time()

        loss = batch_step(
            batch, model, loss_func, gpu, local_rank
        )
        _assert_finite_scalar(loss, "training loss", local_rank)
        
        benchmark_forward[0] += time() - benchmark_timer
        benchmark_forward[1] += 1

        benchmark_timer = time()

        loss_for_backward = loss / accumulation_steps
        if scaler is None:
            loss_for_backward.backward()
        else:
            benchmark_timer = time()
            scaler.scale(loss_for_backward).backward()

        benchmark_backward[0] += time() - benchmark_timer
        benchmark_backward[1] += 1


        benchmark_timer = time()

        should_step = ((i + 1) % accumulation_steps == 0) or ((i + 1) >= max_steps)
        if should_step:
            grad_norm = None
            if scaler is not None:
                scaler.unscale_(optimizer)
            if max_grad_norm > 0.0:
                grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
                _assert_finite_scalar(grad_norm, "gradient norm", local_rank)
            if scaler is None:
                optimizer.step()
                did_step = True
            else:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                did_step = scaler.get_scale() >= scale_before
            optimizer.zero_grad(set_to_none=True)
            if did_step:
                scheduler.step()
                optimizer_steps += 1

        benchmark_optimizer[0] += time() - benchmark_timer
        benchmark_optimizer[1] += 1

        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1

        if inner_pbar is not None:
            postfix = {"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}
            if hasattr(loss_func, "get_last_terms"):
                last_terms = loss_func.get_last_terms()
                for name, value in list(last_terms.items())[:2]:
                    postfix[name] = value
            inner_pbar.set_postfix(**postfix)
            inner_pbar.update(1)

        # model.module.swap_masking()

        # Batch size; computed in this way for compatibility between Hiera and Swin branches
        benchmark_samples += len(next(iter(batch.values())))
        benchmark_total[0] += time() - benchmark_timer_total
        benchmark_total[1] += 1
        benchmark_timer, benchmark_timer_total = time(), time()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(node_count, op=dist.ReduceOp.SUM)
    train_loss = ddp_loss[0] / max(ddp_loss[1], 1)

    if inner_pbar is not None:
        inner_pbar.close()

    metrics = {
        'train.benchmark.data': benchmark_data[0] / benchmark_data[1],
        'train.benchmark.forward': benchmark_forward[0] / benchmark_forward[1],
        'train.benchmark.backward': benchmark_backward[0] / benchmark_backward[1],
        'train.benchmark.optimizer': benchmark_optimizer[0] / benchmark_optimizer[1],
        'train.benchmark.total': benchmark_total[0] / benchmark_total[1],
        'train.benchmark.samples': benchmark_samples,
        'train.num_gpus': node_count,
        'train.benchmark.data.batch_mean': benchmark_batch_mean[0] / benchmark_batch_mean[1],
        'train.gradient_accumulation_steps': accumulation_steps,
        'train.max_grad_norm': max_grad_norm,
        'train.optimizer_steps': optimizer_steps,
    }

    metrics = metrics 

    return train_loss, metrics


def save_checkpoint(
    config: dict,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    train_loss: float,
    curr_val_loss: float,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    train_loss_history: list[float] | None = None,
    val_loss_history: list[float] | None = None,
    best_val_loss: float | None = None,
    best_correction_loss: float | None = None,
    correction_report: Mapping[str, object] | None = None,
    checkpoint_dir: str | None = None,
    is_best: bool = False,
    save_epoch_checkpoint: bool = True,
    save_last_checkpoint: bool = True,
    write_to_disk: bool = True,
):
    collectives_required = _checkpoint_requires_all_ranks(model)
    if not write_to_disk and not collectives_required:
        return

    model_state, optimizer_state = _collect_checkpoint_states(model, optimizer)
    if not write_to_disk:
        return

    checkpoint_dir = checkpoint_dir or getattr(config, "checkpoint_dir", None)
    if checkpoint_dir is None:
        checkpoint_dir = _default_checkpoint_dir(config)

    os.makedirs(checkpoint_dir, exist_ok=True)

    sche_dict = None
    if scheduler is not None:
        sche_state = scheduler.state_dict()
        sche_dict = {k: v for k, v in sche_state.items() if k != "anneal_func"}  # fix OneCycleLR serialization bug

    scheduler_global_step = (
        int(getattr(scheduler, "last_epoch", 0)) if scheduler is not None else None
    )
    metadata = _checkpoint_head_metadata(
        config,
        model,
        epoch=epoch,
        global_step=scheduler_global_step,
    )
    state_dict = {
        "model": model_state,
        "optimizer": optimizer_state,
        "epoch": epoch,
        "loss": train_loss,
        "val_loss": curr_val_loss,
        "validation_enabled": _validation_enabled(config),
        "metadata": metadata,
    }
    # PRISM models have coordinate-sensitive scalers, crop offsets, and decoder
    # skip semantics that are not encoded by tensor shapes. Persist their
    # contract so resume/inference can reject silently incompatible checkpoints.
    from granitewxc.utils.prism_checkpoint import (
        CONTRACT_KEY,
        build_prism_checkpoint_contract,
    )

    prism_contract = build_prism_checkpoint_contract(config)
    if prism_contract is not None:
        state_dict[CONTRACT_KEY] = prism_contract
    if sche_dict is not None:
        state_dict["scheduler"] = sche_dict
    if scaler is not None:
        state_dict["scaler"] = scaler.state_dict()
    if train_loss_history is not None:
        state_dict["train_loss_history"] = list(train_loss_history)
    if val_loss_history is not None:
        state_dict["val_loss_history"] = list(val_loss_history)
    if best_val_loss is not None:
        state_dict["best_val_loss"] = best_val_loss
    if best_correction_loss is not None:
        state_dict["best_correction_loss"] = best_correction_loss
    if correction_report is not None:
        state_dict["correction_qualification"] = dict(correction_report)

    if save_epoch_checkpoint:
        epoch_checkpoint = os.path.join(checkpoint_dir, f"epoch_{epoch + 1:03d}.ckpt")
        torch.save(state_dict, epoch_checkpoint)
        print(f"--> saved {epoch_checkpoint}")
    if is_best:
        best_checkpoint = os.path.join(checkpoint_dir, "best.ckpt")
        torch.save(state_dict, best_checkpoint)
        print(f"--> saved {best_checkpoint}")
    if save_last_checkpoint:
        last_checkpoint = os.path.join(checkpoint_dir, "last.ckpt")
        torch.save(state_dict, last_checkpoint)
        print(f"--> saved {last_checkpoint}")
    

def train_model(config, model, train_dl, val_dl, optimizer, scheduler, scaler, local_rank, use_gpu, save_every, loss_func):
    train_loss = []
    val_loss = []
    best_val_loss = None
    best_correction_loss = None
    latest_correction_report: dict[str, object] | None = None
    validation_enabled = _validation_enabled(config)
    if validation_enabled and val_dl is None:
        raise ValueError(
            "Validation is enabled but val_dl is None. Provide a validation "
            "DataLoader or set config.validation_enabled=false explicitly."
        )
    correction_validation = _resolve_correction_validation(
        config, loss_func, validation_enabled
    )
    checkpoint_dir = getattr(config, "checkpoint_dir", None)
    start_epoch = 0
    resume_requested = bool(
        getattr(config, "resume_training", False)
        or getattr(config, "resume_from_last_checkpoint", False)
        or getattr(config, "resume_checkpoint_path", None)
    )
    if not resume_requested and _auto_resume_enabled(config):
        probe_checkpoint_dir = checkpoint_dir
        if probe_checkpoint_dir is None:
            probe_checkpoint_dir = _default_checkpoint_dir(config)
        auto_resume_checkpoint = _resolve_resume_probe_checkpoint(probe_checkpoint_dir)
        if auto_resume_checkpoint is not None:
            setattr(config, "resume_training", True)
            setattr(config, "resume_from_last_checkpoint", False)
            setattr(config, "resume_checkpoint_path", str(auto_resume_checkpoint))
            resume_requested = True
            if is_main_process():
                print(f"[resume] auto-resume enabled; found checkpoint {auto_resume_checkpoint}")
    if resume_requested:
        if checkpoint_dir is None:
            checkpoint_dir = _default_checkpoint_dir(config)
        resume_checkpoint = getattr(config, "resume_checkpoint_path", None)
        if resume_checkpoint:
            resume_checkpoint = str(resume_checkpoint)
        else:
            resume_checkpoint = _find_latest_epoch_checkpoint(checkpoint_dir)
            if resume_checkpoint is None:
                legacy_last = os.path.join(checkpoint_dir, "last.ckpt")
                legacy_best = os.path.join(checkpoint_dir, "best.ckpt")
                if os.path.exists(legacy_last):
                    resume_checkpoint = legacy_last
                elif os.path.exists(legacy_best):
                    resume_checkpoint = legacy_best
                else:
                    resume_checkpoint = os.path.join(checkpoint_dir, "epoch_001.ckpt")

        if not os.path.exists(resume_checkpoint):
            raise FileNotFoundError(
                f"Resume requested but checkpoint was not found: {resume_checkpoint}"
            )

        map_location: torch.device | str
        if use_gpu and torch.cuda.is_available():
            map_location = torch.device(f"cuda:{local_rank}")
        else:
            map_location = torch.device("cpu")
        checkpoint = torch.load(resume_checkpoint, map_location=map_location, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Expected a dictionary checkpoint when resuming, got {type(checkpoint)!r}"
            )

        from granitewxc.utils.prism_checkpoint import validate_prism_checkpoint_contract

        validate_prism_checkpoint_contract(config, checkpoint, role="resume")

        model_state = checkpoint.get("model", checkpoint)
        if not isinstance(model_state, dict):
            raise ValueError(
                "Checkpoint does not contain a valid model state dictionary under key 'model'."
            )

        optimizer_state = checkpoint.get("optimizer")
        try:
            _restore_checkpoint_states(model, optimizer, model_state, optimizer_state)
            if optimizer_state is not None and use_gpu and torch.cuda.is_available():
                _move_optimizer_state_to_device(optimizer, local_rank)
        except Exception as exc:
            if is_main_process():
                print(
                    "[resume] warning: FSDP-aware restore failed; falling back to "
                    f"plain state-dict load ({exc})"
                )
            model.load_state_dict(model_state, strict=True)
            if optimizer_state is not None:
                try:
                    optimizer.load_state_dict(optimizer_state)
                    if use_gpu and torch.cuda.is_available():
                        _move_optimizer_state_to_device(optimizer, local_rank)
                except Exception as opt_exc:
                    if is_main_process():
                        print(
                            "[resume] warning: failed to restore optimizer state; "
                            f"continuing with initialized optimizer ({opt_exc})"
                        )

        scheduler_state = checkpoint.get("scheduler")
        if scheduler is not None and scheduler_state is not None:
            try:
                scheduler.load_state_dict(scheduler_state)
            except Exception as exc:
                if is_main_process():
                    print(
                        "[resume] warning: failed to restore scheduler state; "
                        f"continuing with initialized scheduler ({exc})"
                    )

        scaler_state = checkpoint.get("scaler")
        if scaler is not None and scaler_state is not None:
            try:
                scaler.load_state_dict(scaler_state)
            except Exception as exc:
                if is_main_process():
                    print(
                        "[resume] warning: failed to restore scaler state; "
                        f"continuing with initialized scaler ({exc})"
                    )

        checkpoint_epoch = int(checkpoint.get("epoch", -1))
        start_epoch = checkpoint_epoch + 1

        saved_train_history = checkpoint.get("train_loss_history")
        saved_val_history = checkpoint.get("val_loss_history")
        if isinstance(saved_train_history, list):
            train_loss = [float(_to_scalar(item)) for item in saved_train_history]
        if validation_enabled and isinstance(saved_val_history, list):
            val_loss = [float(_to_scalar(item)) for item in saved_val_history]

        if validation_enabled:
            restored_best = checkpoint.get("best_val_loss", checkpoint.get("val_loss"))
            if restored_best is not None:
                best_val_loss = float(_to_scalar(restored_best))
        restored_best_correction = checkpoint.get("best_correction_loss")
        if restored_best_correction is not None:
            best_correction_loss = float(_to_scalar(restored_best_correction))
        restored_correction_report = checkpoint.get("correction_qualification")
        if isinstance(restored_correction_report, Mapping):
            latest_correction_report = dict(restored_correction_report)

        if is_main_process():
            print(f"[resume] loaded checkpoint: {resume_checkpoint}")
            print(
                f"[resume] checkpoint epoch={checkpoint_epoch + 1} -> restarting at epoch {start_epoch + 1}"
            )

    if checkpoint_dir is None:
        checkpoint_dir = _default_checkpoint_dir(config)

    # Ensure a resumable checkpoint exists from the beginning of training.
    # In distributed mode this decision must be synchronized across ranks;
    # otherwise rank 0 can create/checkpoint while another rank skips and
    # collectives (FSDP full-state gather) will deadlock.
    should_initialize_resume_checkpoint = False
    if start_epoch == 0:
        if dist.is_available() and dist.is_initialized():
            if is_main_process():
                should_initialize_resume_checkpoint = (
                    _resolve_resume_probe_checkpoint(checkpoint_dir) is None
                )
            decision = torch.tensor(
                [1 if should_initialize_resume_checkpoint else 0],
                device=(torch.device(f"cuda:{local_rank}") if use_gpu and torch.cuda.is_available() else torch.device("cpu")),
                dtype=torch.int32,
            )
            dist.broadcast(decision, src=0)
            should_initialize_resume_checkpoint = bool(int(decision.item()))
        else:
            should_initialize_resume_checkpoint = (
                _resolve_resume_probe_checkpoint(checkpoint_dir) is None
            )

    if should_initialize_resume_checkpoint:
        write_checkpoint = (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)
        must_participate = write_checkpoint or _checkpoint_requires_all_ranks(model)
        if must_participate:
            save_checkpoint(
                config=config,
                scheduler=scheduler,
                epoch=-1,
                model=model,
                optimizer=optimizer,
                train_loss=float("inf"),
                curr_val_loss=float("inf"),
                scaler=scaler,
                train_loss_history=train_loss,
                val_loss_history=val_loss,
                best_val_loss=best_val_loss,
                best_correction_loss=best_correction_loss,
                correction_report=latest_correction_report,
                checkpoint_dir=checkpoint_dir,
                is_best=False,
                save_epoch_checkpoint=False,
                save_last_checkpoint=True,
                write_to_disk=write_checkpoint,
            )
            if write_checkpoint:
                print(f"[checkpoint] initialized resumable checkpoint at {os.path.join(checkpoint_dir, 'last.ckpt')}")

    if start_epoch >= config.num_epochs:
        if is_main_process():
            print(
                f"[resume] checkpoint already reached num_epochs={config.num_epochs}; "
                "no additional training steps are required."
            )
        return train_loss, val_loss

    for epoch in range(start_epoch, config.num_epochs):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if is_main_process():
            print(f"Learning rate: {scheduler.get_last_lr()[0]}")
            print(f"Rank {local_rank} starting epoch {epoch + 1}...")

        curr_train_loss_scalar = None
        curr_val_loss_scalar = None
        try:
            curr_train_loss, _ = train_one_epoch(
                model=model,
                local_rank=local_rank,
                train_loader=train_dl,
                optimizer=optimizer,
                loss_func=loss_func,
                epoch=epoch,
                scaler=scaler,
                scheduler=scheduler,
                gpu=use_gpu,
                limit_steps=config.limit_steps_train,
                num_epochs=config.num_epochs,
                gradient_accumulation_steps=int(getattr(config, "gradient_accumulation_steps", 1)),
                max_grad_norm=float(getattr(config, "max_grad_norm", 0.0) or 0.0),
            )

            curr_train_loss_scalar = float(_to_scalar(curr_train_loss))
            if validation_enabled:
                if use_gpu and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                curr_val_loss, val_metrics = validate_one_epoch(
                    model=model,
                    local_rank=local_rank,
                    validation_loader=val_dl,
                    loss_func=loss_func,
                    epoch=epoch,
                    gpu=use_gpu,
                    limit_steps=config.limit_steps_valid,
                    correction_validation=correction_validation,
                )
                curr_val_loss_scalar = float(_to_scalar(curr_val_loss))
                if correction_validation is not None:
                    correction_metrics = {
                        key: value
                        for key, value in val_metrics.items()
                        if key.startswith("val.correction.")
                    }
                    latest_correction_report = {
                        "epoch": epoch,
                        "qualified": bool(
                            correction_metrics.get("val.correction.qualified", 0.0)
                        ),
                        "selection_metric": (
                            "yaml_configured_physical_ensemble_mean_loss"
                        ),
                        "metrics": correction_metrics,
                    }
                if use_gpu and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                curr_val_loss_scalar = float("inf")
        except Exception as exc:
            write_checkpoint = (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)
            must_participate = write_checkpoint or _checkpoint_requires_all_ranks(model)
            fallback_train = (
                float(curr_train_loss_scalar)
                if curr_train_loss_scalar is not None
                else (float(train_loss[-1]) if train_loss else float("inf"))
            )
            fallback_val = (
                float("inf")
                if not validation_enabled
                else float(curr_val_loss_scalar)
                if curr_val_loss_scalar is not None
                else (float(val_loss[-1]) if val_loss else float("inf"))
            )
            finite_failure = (
                not isinstance(exc, FloatingPointError)
                and np.isfinite(fallback_train)
                and np.isfinite(fallback_val)
            )
            if must_participate and finite_failure:
                try:
                    save_checkpoint(
                        config=config,
                        scheduler=scheduler,
                        epoch=epoch - 1,
                        model=model,
                        optimizer=optimizer,
                        train_loss=fallback_train,
                        curr_val_loss=fallback_val,
                        scaler=scaler,
                        train_loss_history=train_loss,
                        val_loss_history=val_loss,
                        best_val_loss=best_val_loss,
                        best_correction_loss=best_correction_loss,
                        correction_report=latest_correction_report,
                        checkpoint_dir=checkpoint_dir,
                        is_best=False,
                        save_epoch_checkpoint=False,
                        save_last_checkpoint=True,
                        write_to_disk=write_checkpoint,
                    )
                    if write_checkpoint:
                        print(
                            "[checkpoint] wrote emergency last.ckpt after failure "
                            f"in epoch {epoch + 1}: {exc}"
                        )
                except Exception as save_exc:
                    if write_checkpoint:
                        print(f"[checkpoint] failed to write emergency checkpoint: {save_exc}")
            elif write_checkpoint:
                print(
                    "[checkpoint] skipped emergency last.ckpt because the failure "
                    f"may involve non-finite state: {exc}"
                )
            raise

        train_loss.append(curr_train_loss_scalar)
        if validation_enabled:
            val_loss.append(curr_val_loss_scalar)
            joint_is_best = (
                best_val_loss is None or curr_val_loss_scalar < best_val_loss
            )
            if joint_is_best:
                best_val_loss = curr_val_loss_scalar
            if correction_validation is not None:
                if latest_correction_report is None:
                    raise RuntimeError(
                        "Correction validation did not return a qualification report."
                    )
                correction_metrics = latest_correction_report["metrics"]
                qualified = bool(
                    correction_metrics["val.correction.qualified"]
                )
                correction_loss = float(
                    correction_metrics["val.correction.ensemble_mean.total"]
                )
                is_best = qualified and (
                    best_correction_loss is None
                    or correction_loss < best_correction_loss
                )
                if is_best:
                    best_correction_loss = correction_loss
            else:
                is_best = joint_is_best
        else:
            is_best = False

        save_every_n = max(1, int(save_every or 1))
        save_epoch_checkpoint = _save_epoch_checkpoints_enabled(config) and (
            ((epoch + 1) % save_every_n == 0)
            or ((epoch + 1) == config.num_epochs)
        )
        # Always persist a resumable last.ckpt after each completed epoch.
        save_now = True
        write_checkpoint = (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)
        must_participate = write_checkpoint or _checkpoint_requires_all_ranks(model)
        if save_now and must_participate:
            save_checkpoint(
                config=config,
                scheduler=scheduler,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                train_loss=curr_train_loss_scalar,
                curr_val_loss=curr_val_loss_scalar,
                scaler=scaler,
                train_loss_history=train_loss,
                val_loss_history=val_loss,
                best_val_loss=best_val_loss,
                best_correction_loss=best_correction_loss,
                correction_report=latest_correction_report,
                checkpoint_dir=checkpoint_dir,
                is_best=is_best,
                save_epoch_checkpoint=save_epoch_checkpoint,
                save_last_checkpoint=True,
                write_to_disk=write_checkpoint,
            )

    return train_loss, val_loss
