from collections.abc import Callable
from typing import Dict

import os
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time

from granitewxc.utils.distributed import is_main_process

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


def batch_step(
    batch: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    loss_func: Callable,
    gpu: bool,
    local_rank: int,
):
    if gpu:
        batch = {k: v.to(local_rank) for k, v in batch.items()}
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with autocast(device_type="cuda", dtype=dtype):
            prediction = model(batch)
            _inject_precip_hurdle_aux(batch, model)
            loss = loss_func(prediction, batch)
    else:
        prediction = model(batch)
        _inject_precip_hurdle_aux(batch, model)
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
):
    model.eval()
    ddp_loss = torch.zeros(2)
    inner_pbar = None
    benchmark_data = np.zeros(2)
    benchmark_forward = np.zeros(2)
    benchmark_total = np.zeros(2)
    benchmark_samples = 0

    if gpu:
        ddp_loss = ddp_loss.to(local_rank)
    
    sampler = validation_loader.sampler
    if hasattr(sampler, 'set_epoch'):
        sampler.set_epoch(epoch)
    elif dist.is_available() and dist.is_initialized():
        print('WARNING: Not calling set_epoch.')

    if is_main_process():
        inner_pbar = tqdm(
            range(min(limit_steps, len(validation_loader))),
            unit='batch',
            colour="green",
            desc="Validation Epoch",
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

            if is_main_process():
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
    val_loss = ddp_loss[0] / max(ddp_loss[1], 1)

    if is_main_process():
        inner_pbar.close()

    metrics = {
        'val.benchmark.data': benchmark_data[0] / benchmark_data[1],
        'val.benchmark.forward': benchmark_forward[0] / benchmark_forward[1],
        'val.benchmark.total': benchmark_total[0] / benchmark_total[1],
        'val.benchmark.samples': benchmark_samples,
    }

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

        inner_pbar = tqdm(
            range(min(limit_steps, len(train_loader))),
            unit="batch",
            colour="blue",
            desc=f"Training Epoch {epoch+1}/{num_epochs}",
        )

    benchmark_timer, benchmark_timer_total = time(), time()

    benchmark_batch_mean = np.zeros(2)
    accumulation_steps = max(1, int(kwargs.get("gradient_accumulation_steps", 1)))
    max_grad_norm = float(kwargs.get("max_grad_norm", 0.0) or 0.0)
    optimizer_steps = 0
    max_steps = min(limit_steps, len(train_loader)) if 0 < limit_steps else len(train_loader)
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

        if is_main_process():
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

    if is_main_process():
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
        checkpoint_dir = os.path.join(config.path_experiment, "weights")

    os.makedirs(checkpoint_dir, exist_ok=True)

    sche_dict = None
    if scheduler is not None:
        sche_state = scheduler.state_dict()
        sche_dict = {k: v for k, v in sche_state.items() if k != "anneal_func"}  # fix OneCycleLR serialization bug

    state_dict = {
        "model": model_state,
        "optimizer": optimizer_state,
        "epoch": epoch,
        "loss": train_loss,
        "val_loss": curr_val_loss,
    }
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
            probe_checkpoint_dir = os.path.join(config.path_experiment, "weights")
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
            checkpoint_dir = os.path.join(config.path_experiment, "weights")
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

        model_state = checkpoint.get("model", checkpoint)
        if not isinstance(model_state, dict):
            raise ValueError(
                "Checkpoint does not contain a valid model state dictionary under key 'model'."
            )
        model.load_state_dict(model_state, strict=True)

        optimizer_state = checkpoint.get("optimizer")
        if optimizer_state is not None:
            try:
                optimizer.load_state_dict(optimizer_state)
                if use_gpu and torch.cuda.is_available():
                    _move_optimizer_state_to_device(optimizer, local_rank)
            except Exception as exc:
                if is_main_process():
                    print(
                        "[resume] warning: failed to restore optimizer state; "
                        f"continuing with initialized optimizer ({exc})"
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
        if isinstance(saved_val_history, list):
            val_loss = [float(_to_scalar(item)) for item in saved_val_history]

        restored_best = checkpoint.get("best_val_loss", checkpoint.get("val_loss"))
        if restored_best is not None:
            best_val_loss = float(_to_scalar(restored_best))

        if is_main_process():
            print(f"[resume] loaded checkpoint: {resume_checkpoint}")
            print(
                f"[resume] checkpoint epoch={checkpoint_epoch + 1} -> restarting at epoch {start_epoch + 1}"
            )

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(config.path_experiment, "weights")

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

            # Free cached training allocations before validation to reduce fragmentation/OOM risk.
            if use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()

            curr_train_loss_scalar = float(_to_scalar(curr_train_loss))
            curr_val_loss, _ = validate_one_epoch(
                model=model,
                local_rank=local_rank,
                validation_loader=val_dl,
                loss_func=loss_func,
                epoch=epoch,
                gpu=use_gpu,
                limit_steps=config.limit_steps_valid,
            )
            curr_val_loss_scalar = float(_to_scalar(curr_val_loss))

            if use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            write_checkpoint = (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)
            must_participate = write_checkpoint or _checkpoint_requires_all_ranks(model)
            fallback_train = (
                float(curr_train_loss_scalar)
                if curr_train_loss_scalar is not None
                else (float(train_loss[-1]) if train_loss else float("inf"))
            )
            fallback_val = (
                float(curr_val_loss_scalar)
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
        val_loss.append(curr_val_loss_scalar)

        is_best = best_val_loss is None or curr_val_loss_scalar < best_val_loss
        if is_best:
            best_val_loss = curr_val_loss_scalar

        save_every_n = max(1, int(save_every or 1))
        save_epoch_checkpoint = ((epoch + 1) % save_every_n == 0) or ((epoch + 1) == config.num_epochs)
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
                checkpoint_dir=checkpoint_dir,
                is_best=is_best,
                save_epoch_checkpoint=save_epoch_checkpoint,
                save_last_checkpoint=True,
                write_to_disk=write_checkpoint,
            )

    return train_loss, val_loss
