"""Retrain a versioned regional refiner on frozen Phase-1 cache fields.

Uses the normal CORDEX chronological 90/10 split (1961-1980+2080-2095
fitting, 2096-2099 validation). Optional sample counts are screening runs and
are recorded as such. Checkpoints remain loadable by the normal workflow.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from granitewxc.models.model import get_finetune_model_UNET
from granitewxc.utils.config import ExperimentConfig
from granitewxc.refinement import build_two_phase_model, build_refiner, resolve_refinement_config
from granitewxc.refinement.checkpoint import (_build_scientific_contract, load_phase1_state_dict, phase1_state_fingerprint,
                                             build_refinement_checkpoint, save_checkpoint_atomic,
                                             load_refinement_state_dict)
from granitewxc.refinement.boundary import boundary_regions, physical_metrics
from granitewxc.refinement.config import ResidualNormalizationConfig


class CachedDomains(Dataset):
    def __init__(self,path,indices):
        self.path=path;self.indices=list(indices);self.file=None
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        if self.file is None:self.file=h5py.File(self.path,'r')
        i=self.indices[index]
        return {key:torch.from_numpy(value[i].copy()) for key,value in self.file['fields'].items()}


def choose(indices,count):
    indices=np.asarray(indices)
    return indices if count is None else indices[np.unique(np.linspace(0,len(indices)-1,min(count,len(indices)),dtype=int))]


def evaluate(wrapper,dataset,device,members,steps,diffusion_solver=None):
    wrapper.refiner.eval()
    means=[];truth=[];bases=[];masks=[];member_fields=[]
    for index,batch in enumerate(tqdm(DataLoader(dataset,batch_size=1,shuffle=False), desc="Physical ensemble validation", mininterval=2.)):
        with torch.inference_mode():
            # Frozen Phase-1 is on CPU and used only for scaler/conditioning logic.
            cond=wrapper.build_conditioning(batch,batch['__phase1_normalized']).to(device)
            base=batch['__phase1_physical'].to(device)
            z=wrapper.refiner.sample(cond.expand(members,-1,-1,-1),
                    generator=torch.Generator(device=device).manual_seed(83107+index),num_steps=steps,
                    **({"solver":diffusion_solver} if diffusion_solver is not None and wrapper.refinement_config.is_diffusion else {}))
            residual=wrapper.residual_normalizer.denormalize(z)
            residual=wrapper._effective_physical_residual(residual)
            selected=wrapper.target_space.constrain_physical_ensemble((base+residual).unsqueeze(0),
                    strategy=wrapper.refinement_config.nonnegative_ensemble_strategy)
        physical=selected['members'].cpu().numpy()
        member_fields.append(physical)
        means.append(physical.mean(1,dtype=np.float64));truth.append(batch['y'].numpy())
        bases.append(batch['__phase1_physical'].numpy());masks.append(batch['__target_valid_mask'].numpy())
    arrays={'mean':np.concatenate(means),'members':np.concatenate(member_fields),'truth':np.concatenate(truth),
            'baseline':np.concatenate(bases),'valid':np.concatenate(masks)}
    with h5py.File(dataset.path,'r') as cache:
        _,regions=boundary_regions(arrays['valid'].any((0,1)),cache['latitude'][:],cache['longitude'][:])
    metrics={name:{product:{region:physical_metrics(arrays[product][:,c],arrays['truth'][:,c],arrays['valid'][:,c],
                    region=mask,precipitation=c==0) for region,mask in regions.items() if mask.any()}
                 for product in ('baseline','mean')} for c,name in enumerate(('pr','tasmax'))}
    return metrics,arrays


def main(args):
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=bool(args.resume))
    if any((output / name).exists() for name in ('validation_members.npz', 'validation_metrics.json')):
        raise FileExistsError('Evaluation outputs already exist; choose a fresh --output directory when resuming a completed run')
    if args.deterministic:
        # Set before CUDA initialization. Required for reproducible cuBLAS GEMMs.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
    raw=yaml.safe_load(Path(args.config).read_text());cfg=resolve_refinement_config(raw)
    if args.accumulation_steps < 1 or (args.cosine_epochs is not None and args.cosine_epochs < 1):
        raise ValueError('Accumulation and cosine epoch counts must be positive')
    if args.cosine_epochs is not None and args.epochs > args.cosine_epochs:
        raise ValueError('Epoch count cannot exceed the fixed cosine horizon')
    if args.diffusion_solver is not None and not cfg.is_diffusion:
        raise ValueError('--diffusion-solver applies only to diffusion heads')
    cache_manifest=json.loads(Path(args.cache).with_suffix('.manifest.json').read_text())
    print('Verifying frozen cache integrity',flush=True)
    with open(args.cache,'rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest!=cache_manifest['cache_sha256']:raise ValueError('Cache SHA256 mismatch')
    phase1_path=Path(raw['model']['phase1']['checkpoint'])
    experiment=ExperimentConfig.from_dict(raw)
    phase1=get_finetune_model_UNET(experiment)
    wrapper=build_two_phase_model(phase1,experiment)
    p1=torch.load(phase1_path,map_location='cpu',mmap=True,weights_only=False)
    load_phase1_state_dict(wrapper,p1);del p1
    fingerprint=phase1_state_fingerprint(wrapper.phase1.state_dict())
    if fingerprint!=cache_manifest['phase1_state_fingerprint']:raise ValueError('Cache Phase-1 identity differs')
    with h5py.File(args.cache,'r') as f:
        dates=f['dates'].asstr()[:];n=len(dates);split=n-round(n*.1)
        if not all(int(d[:4])>=2096 for d in dates[split:]):raise ValueError('Expected untouched 2096-2099 chronological validation block')
    train_indices=choose(np.arange(split),args.train_samples)
    validation_indices=choose(np.arange(split,n),args.validation_samples)
    train=CachedDomains(args.cache,train_indices);validation=CachedDomains(args.cache,validation_indices)
    sample=next(iter(DataLoader(train,batch_size=1)))
    cond=wrapper.build_conditioning(sample,sample['__phase1_normalized'])
    wrapper.initialize_refiner(cond.shape[1])
    teacher=None
    conditioning_transfer=None
    if args.normalize_conditioning_transfer and (not args.initialize or args.epochs==0 or args.temperature_teacher_weight):
        raise ValueError("Conditioning transfer requires retraining from an initialization, with teacher weight zero")
    if args.initialize:
        initial=torch.load(args.initialize,map_location='cpu',mmap=True,weights_only=False)
        old_cfg=resolve_refinement_config(initial['resolved_config'])
        if args.epochs==0 and not args.resume and _build_scientific_contract(old_cfg)!=_build_scientific_contract(cfg):
            raise ValueError('Evaluation cannot reinterpret weights under a different scientific configuration')
        if args.epochs>0 and float(initial['model']['residual_normalizer.count'].max())>split*sample['y'].shape[-2]*sample['y'].shape[-1]:
            raise ValueError('Initialization statistics include current validation dates; retrain from scratch')
        if old_cfg.residual_normalization!=cfg.residual_normalization and not (args.normalizer_checkpoint or args.normalizer_state):
            raise ValueError('Target transformation transfer requires explicit training statistics')
        if args.normalize_conditioning_transfer:
            if old_cfg.conditioning.normalize_predictors or replace(old_cfg.conditioning, normalize_predictors=True)!=cfg.conditioning:
                raise ValueError('Only explicit raw-to-normalized predictor transfer is supported')
        elif old_cfg.conditioning!=cfg.conditioning:
            raise ValueError('Conditioning semantics differ without an explicit affine transfer')
        if initial['phase1_fingerprint']!=fingerprint:raise ValueError('Initialization Phase-1 mismatch')
        wrapper.refiner.load_state_dict({k[8:]:v for k,v in initial['model'].items() if k.startswith('refiner.')},strict=True)
        if args.normalize_conditioning_transfer:
            from granitewxc.refinement.conditioning_transfer import frozen_conditioning_affine, fold_conditioning_affine
            mean,scale=frozen_conditioning_affine(wrapper,sample)
            fold_conditioning_affine(wrapper.refiner,mean,scale)
            conditioning_transfer={'method':'global_affine_input_convolution_v1','mean':mean.tolist(),'scale':scale.tolist()}
        if not args.normalizer_state:
            wrapper.residual_normalizer.load_state_dict({k[len('residual_normalizer.'):]:v for k,v in initial['model'].items() if k.startswith('residual_normalizer.')},strict=True)
        elif args.temperature_teacher_weight:
            raise ValueError('Disable the teacher when transferring to a new precipitation transformation')
        if args.temperature_teacher_weight:
            teacher=build_refiner(old_cfg,residual_channels=2,cond_channels=cond.shape[1]).to(args.device).eval()
            teacher.load_state_dict({k[8:]:v for k,v in initial['model'].items() if k.startswith('refiner.')},strict=True)
            teacher.requires_grad_(False)
        del initial
    elif not (args.normalizer_checkpoint or args.normalizer_state):
        # Statistics use fitting dates only; no validation values are read.
        for batch in tqdm(DataLoader(train,batch_size=32),desc='Training residual statistics'):
            residual=batch['y']-batch['__phase1_physical']
            wrapper.residual_normalizer.update(residual,batch['__target_valid_mask'])
        wrapper.residual_normalizer.finalize()
    if args.normalizer_checkpoint:
        statistics=torch.load(args.normalizer_checkpoint,map_location='cpu',weights_only=False,mmap=True)
        stats_cfg=resolve_refinement_config(statistics['resolved_config'])
        if stats_cfg.residual_normalization!=cfg.residual_normalization:raise ValueError('Statistics transformation differs')
        if statistics['phase1_fingerprint']!=fingerprint:raise ValueError('Statistics Phase-1 differs')
        stats_state={k[len('residual_normalizer.'):]:v for k,v in statistics['model'].items() if k.startswith('residual_normalizer.')}
        if args.initialize:
            for key in ('mean','scale'):
                if not torch.allclose(wrapper.residual_normalizer.state_dict()[key][:,1],stats_state[key][:,1]):raise ValueError('Tmax statistics would change during transfer')
        wrapper.residual_normalizer.load_state_dict(stats_state,strict=True)
        del statistics
    if args.normalizer_state:
        if args.normalizer_checkpoint:raise ValueError('Specify only one normalization source')
        statistics=torch.load(args.normalizer_state,map_location='cpu',weights_only=False)
        if statistics['schema']!='training_residual_statistics_v1':raise ValueError('Unsupported statistics artifact')
        if ResidualNormalizationConfig.from_mapping(statistics['normalization_config'])!=cfg.residual_normalization:raise ValueError('Statistics transformation mismatch')
        if statistics['phase1_fingerprint']!=fingerprint or statistics['cache_sha256']!=digest:raise ValueError('Statistics cache identity mismatch')
        if statistics['train_dates']!=dates[:split].tolist():raise ValueError('Statistics fitting dates differ')
        wrapper.residual_normalizer.load_state_dict(statistics['state_dict'],strict=True)
        del statistics
    wrapper.refiner.to(args.device);wrapper.residual_normalizer.to(args.device)
    parameters=[{'params':[p for name,p in wrapper.refiner.named_parameters() if name!='correction_gate']}]
    if hasattr(wrapper.refiner,'correction_gate'):
        parameters.append({'params':[wrapper.refiner.correction_gate],'lr':args.gate_learning_rate})
    optimizer=torch.optim.AdamW(parameters,lr=args.learning_rate,weight_decay=0.)
    updates_per_epoch=math.ceil(math.ceil(len(train)/args.batch_size)/args.accumulation_steps)
    scheduler=None if args.cosine_epochs is None else torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,T_max=args.cosine_epochs*updates_per_epoch,eta_min=0.)
    generator=torch.Generator(device=args.device).manual_seed(args.seed+91)
    sampler_generator=torch.Generator().manual_seed(args.seed+31)
    start_epoch=0;global_step=0
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        load_refinement_state_dict(wrapper,saved)
        previous=saved['boundary_retraining_provenance']
        for key in ('initialize','seed','train_samples','validation_samples','batch_size','temperature_teacher_weight','normalizer_checkpoint','normalizer_state','gate_learning_rate','deterministic','precision','normalize_conditioning_transfer','accumulation_steps','cosine_epochs','learning_rate'):
            if previous['args'].get(key,{'gate_learning_rate':0.001,'deterministic':False,'precision':'fp32','normalize_conditioning_transfer':False,'accumulation_steps':1}.get(key))!=getattr(args,key):raise ValueError(f'Resume protocol mismatch: {key}')
        optimizer.load_state_dict(saved['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(saved['scheduler'])
        start_epoch=saved['epoch']+1;global_step=saved['global_step']
        generator.set_state(saved['process_generator'].cpu());sampler_generator.set_state(saved['sampler_generator'].cpu())
        torch.set_rng_state(saved['rng_state']['cpu'].cpu())
        if saved['rng_state']['cuda'] is not None:torch.cuda.set_rng_state_all(saved['rng_state']['cuda'])
    manifest={'config':str(Path(args.config).resolve()),'initialization':args.initialize,'seed':args.seed,
              'scope':'screening' if args.train_samples or args.validation_samples else 'full_chronological_split',
              'train_dates':dates[train_indices].tolist(),'validation_dates':dates[validation_indices].tolist(),
              'normalizer':wrapper.residual_normalizer.metadata(),'phase1_fingerprint':fingerprint,
              'temperature_teacher_weight':args.temperature_teacher_weight,'cache_sha256':digest,'args':vars(args),
              'conditioning_transfer':conditioning_transfer,
              'cuda_precision':{'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}}
    sources = sorted((ROOT/'granitewxc/refinement').glob('*.py')) + [Path(__file__)]
    manifest['source_sha256'] = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'resolved.yaml').write_text(yaml.safe_dump(raw,sort_keys=False))
    loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,generator=sampler_generator,num_workers=0)
    for epoch in range(start_epoch,args.epochs):
        wrapper.refiner.train();running={};started=time.time()
        bar=tqdm(loader,desc=f'{cfg.type} epoch {epoch+1}/{args.epochs}',mininterval=2.)
        for index,batch in enumerate(bar):
            condition=wrapper.build_conditioning(batch,batch['__phase1_normalized']).to(args.device)
            residual=(batch['y']-batch['__phase1_physical']).to(args.device)
            valid=batch['__target_valid_mask'].to(args.device)
            target=wrapper.residual_normalizer.normalize(residual,valid)
            zero=wrapper.residual_normalizer.normalize(torch.zeros_like(residual))
            if index % args.accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(args.device).type,dtype=torch.bfloat16,enabled=args.precision=='bf16'):
                result=wrapper.refiner.training_loss(target,condition,valid,generator=generator,zero_residual=zero)
                result=wrapper.add_clean_boundary_loss(result,residual,valid)
                loss=result['loss']
                if teacher is not None:
                    state=result['interpolated_state'] if cfg.is_flow_matching else result['intermediate_state']
                    process_time=result['flow_time'] if cfg.is_flow_matching else result['timesteps']
                    with torch.no_grad():reference=teacher.predict_process(state,condition,process_time)
                    distill=(result['prediction'][:,1]-reference[:,1]).square().mean()
                    loss=loss+args.temperature_teacher_weight*distill
                    result['temperature_preservation_loss']=distill
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            group_start=(index//args.accumulation_steps)*args.accumulation_steps
            group_size=min(args.accumulation_steps,len(loader)-group_start)
            (loss/group_size).backward()
            if (index+1)%args.accumulation_steps==0 or index+1==len(loader):
                torch.nn.utils.clip_grad_norm_(wrapper.refiner.parameters(),1.)
                optimizer.step();global_step+=1
                if scheduler is not None:scheduler.step()
            for key,value in result.items():
                if key.endswith('loss') and value.ndim==0:running[key]=running.get(key,0.)+float(value.detach())
            bar.set_postfix(loss=float(loss.detach()),updates=global_step,refresh=False)
        record={'epoch':epoch,'global_step':global_step,'seconds':time.time()-started,
                'losses':{k:v/len(loader) for k,v in running.items()},
                'learning_rates':[group['lr'] for group in optimizer.param_groups]}
        checkpoint=build_refinement_checkpoint(wrapper,phase1_checkpoint=str(phase1_path),phase1_fingerprint=fingerprint,
                     resolved_config=raw,epoch=epoch,global_step=global_step,optimizer=optimizer,scheduler=scheduler,
                     extra={'process_generator':generator.get_state(),'sampler_generator':sampler_generator.get_state(),
                            'boundary_retraining_provenance':manifest,'record':record})
        save_checkpoint_atomic(checkpoint,output/'last.ckpt')
        if args.keep_epoch_checkpoints:
            epoch_path=output/f'epoch_{epoch+1:04d}.ckpt'
            if epoch_path.exists():
                raise FileExistsError(f'Refusing to overwrite retained checkpoint: {epoch_path}')
            save_checkpoint_atomic(checkpoint,epoch_path)
        with (output/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    metrics,arrays=evaluate(wrapper,validation,args.device,args.members,args.steps,args.diffusion_solver)
    np.savez_compressed(output/'validation_members.npz',**arrays,dates=dates[validation_indices].astype(str))
    (output/'validation_metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    print(json.dumps({v:{p:{r:metrics[v][p][r] for r in ('outer_1','interior')} for p in metrics[v]} for v in metrics},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--initialize');p.add_argument('--resume');p.add_argument('--normalizer-checkpoint');p.add_argument('--normalizer-state');p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--train-samples',type=int);p.add_argument('--validation-samples',type=int)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--accumulation-steps',type=int,default=1)
    p.add_argument('--cosine-epochs',type=int,help='Fixed cosine-decay horizon, identical on initial and resumed runs')
    p.add_argument('--temperature-teacher-weight',type=float,default=1.)
    p.add_argument('--gate-learning-rate',type=float,default=0.001)
    p.add_argument('--normalize-conditioning-transfer', action='store_true')
    p.add_argument('--deterministic', action='store_true', help='Require deterministic CUDA operations for exact training replay')
    p.add_argument('--keep-epoch-checkpoints', action='store_true', help='Retain each completed epoch for validation-based model selection')
    p.add_argument('--precision',choices=('fp32','bf16'),default='fp32')
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=103)
    p.add_argument('--members',type=int,default=4);p.add_argument('--steps',type=int,default=50)
    p.add_argument('--diffusion-solver',choices=('ddim','heun'),help='Explicit evaluation-only numerical solver override')
    main(p.parse_args())
