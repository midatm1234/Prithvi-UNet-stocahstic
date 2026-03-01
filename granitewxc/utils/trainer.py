from collections.abc import Callable
from typing import Dict

import os
import numpy as np
import torch
import torch.distributed as dist
from torch import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time

from granitewxc.utils.distributed import is_main_process



def batch_step(
    batch: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    loss_func: Callable,
    gpu: bool,
    local_rank: int,
):
    if gpu:
        batch = {k: v.to(local_rank) for k, v in batch.items()}

    dtype = torch.bfloat16 if gpu and torch.cuda.is_bf16_supported() else torch.float16
    with autocast(device_type='cuda', dtype=dtype):
        prediction = model(batch)
        # Directly pass batch['y'] to the loss function without pre-processing
        # Ensure that the masked_rmse_loss function handles the tensor correctly
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
    else:
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
    else:
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

    for i, batch in enumerate(train_loader):
        if 0 < limit_steps <= i:
            break

        benchmark_batch_mean[0] += batch['x'].mean()
        benchmark_batch_mean[1] += 1

        benchmark_data[0] += time() - benchmark_timer
        benchmark_data[1] += 1

        benchmark_timer = time()

        optimizer.zero_grad(set_to_none=True)

        loss = batch_step(
            batch, model, loss_func, gpu, local_rank
        )
        
        benchmark_forward[0] += time() - benchmark_timer
        benchmark_forward[1] += 1

        benchmark_timer = time()

        if scaler is None:       
            loss.backward()
        else:
            benchmark_timer = time()
            scaler.scale(loss).backward()

        benchmark_backward[0] += time() - benchmark_timer
        benchmark_backward[1] += 1


        benchmark_timer = time()

        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()

        benchmark_optimizer[0] += time() - benchmark_timer
        benchmark_optimizer[1] += 1

        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1

        if is_main_process():
            inner_pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
            inner_pbar.update(1)


        scheduler.step()

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
        'train.benchmark.data.batch_mean': benchmark_batch_mean[0] / benchmark_batch_mean[1]
    }

    metrics = metrics 

    return train_loss, metrics


def save_checkpoint(
    config: dict,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loss: float,
    curr_val_loss: float,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    checkpoint_dir: str | None = None,
    is_best: bool = False,
):
    checkpoint_dir = checkpoint_dir or getattr(config, "checkpoint_dir", None)
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(config.path_experiment, "weights")

    os.makedirs(checkpoint_dir, exist_ok=True)

    sche_dict = None
    if scheduler is not None:
        sche_state = scheduler.state_dict()
        sche_dict = {k: v for k, v in sche_state.items() if k != "anneal_func"}  # fix OneCycleLR serialization bug

    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": train_loss,
        "val_loss": curr_val_loss,
    }
    if sche_dict is not None:
        state_dict["scheduler"] = sche_dict

    last_checkpoint = os.path.join(checkpoint_dir, "last.ckpt")
    torch.save(state_dict, last_checkpoint)
    if is_best:
        best_checkpoint = os.path.join(checkpoint_dir, "best.ckpt")
        torch.save(state_dict, best_checkpoint)
        print(f"--> saved {best_checkpoint}")
    else:
        print(f"--> saved {last_checkpoint}")
    

def train_model(config, model, train_dl, val_dl, optimizer, scheduler, scaler, local_rank, use_gpu, save_every, loss_func):
    train_loss = []
    val_loss = []
    best_val_loss = None
    checkpoint_dir = getattr(config, "checkpoint_dir", None)

    for epoch in range(config.num_epochs):  
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        print(f"Learning rate: {scheduler.get_last_lr()[0]}")
        print(f"Rank {local_rank} starting epoch {epoch + 1}...")

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
        )

        # Free cached training allocations before validation to reduce fragmentation/OOM risk.
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()

        curr_val_loss, _ = validate_one_epoch(
            model=model,
            local_rank=local_rank,
            validation_loader=val_dl,
            loss_func=loss_func,
            epoch=epoch,
            gpu=use_gpu,
            limit_steps=config.limit_steps_valid,
        )

        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_loss.append(curr_train_loss.tolist())
        val_loss.append(curr_val_loss.tolist())

        is_best = best_val_loss is None or curr_val_loss < best_val_loss
        if is_best:
            best_val_loss = curr_val_loss

        save_now = (
            ((epoch + 1) % max(1, save_every) == 0)
            or is_best
            or (epoch + 1 == config.num_epochs)
        )
        should_save = (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)
        if save_now and should_save:
            save_checkpoint(
                config=config,
                scheduler=scheduler,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                train_loss=train_loss,
                curr_val_loss=curr_val_loss,
                checkpoint_dir=checkpoint_dir,
                is_best=is_best,
            )

    return train_loss, val_loss
