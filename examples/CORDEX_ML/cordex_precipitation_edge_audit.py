"""Trace an existing refinement checkpoint on cached real regional fields.

No fitting or target-conditioned sampling occurs. Outputs are immutable and
the cache's split and checkpoint provenance remain attached to each audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from granitewxc.refinement import build_refiner, resolve_refinement_config, ResidualNormalizer
from granitewxc.refinement.boundary import boundary_regions, physical_metrics
from granitewxc.refinement.target_space import NormalizedTargetSpace
from granitewxc.refinement.config import resolve_performance_config
from granitewxc.refinement.precision import configure_refinement_precision


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    cfg = resolve_refinement_config(payload['resolved_config'])
    precision = configure_refinement_precision(resolve_performance_config(payload['resolved_config']))
    manifest = json.loads(Path(args.cache).with_suffix('.manifest.json').read_text())
    if manifest['identity']['conditioning'] != cfg.conditioning.to_dict():
        raise ValueError('Prepared cache conditioning differs from checkpoint training configuration')
    if digest(args.cache) != manifest['sha256']:
        raise ValueError('Prepared cache hash differs from its manifest')
    if payload['phase1_fingerprint'] != manifest['identity']['phase1_fingerprint']:
        raise ValueError('Checkpoint and cache use different Phase-1 weights')
    normalizer = ResidualNormalizer(2, cfg.residual_normalization, nonnegative_mask=torch.tensor([True, False]))
    normalizer.load_state_dict({k.removeprefix('residual_normalizer.'): v for k,v in payload['model'].items()
                                if k.startswith('residual_normalizer.')}, strict=True)
    normalizer.to(args.device)
    refiner = build_refiner(cfg, residual_channels=2, cond_channels=manifest['cond_channels'])
    refiner.load_state_dict({k.removeprefix('refiner.'): v for k,v in payload['model'].items()
                             if k.startswith('refiner.')}, strict=True)
    refiner.to(args.device).eval()
    constraint = NormalizedTargetSpace(SimpleNamespace(output_scalers_sigma=torch.ones(1,2,1,1),
                                                       predictand_nonneg_enabled_mask=torch.tensor([True,False])))
    report = {'checkpoint': str(Path(args.checkpoint).resolve()), 'checkpoint_sha256': digest(args.checkpoint),
              'checkpoint_epoch': payload.get('epoch'), 'cache': str(Path(args.cache).resolve()),
              'split': manifest['identity']['split'], 'normalizer': normalizer.metadata(),
              'conditioning': cfg.conditioning.to_dict(), 'head': cfg.type, 'members': args.members,
              'steps': args.steps, 'note': 'diagnosis only; old checkpoint training split may overlap cache validation',
              'convolutions': [{'name':n,'kernel':list(m.kernel_size),'padding':list(m.padding),
                               'mode':m.padding_mode} for n,m in refiner.named_modules() if isinstance(m,torch.nn.Conv2d)]}
    stages, network_outputs, trajectory, trajectory_times = {}, {}, {}, {}
    report['cuda_precision'] = precision
    report['residual_normalization_config'] = cfg.residual_normalization.to_dict()
    report['nonnegative_ensemble_strategy'] = cfg.nonnegative_ensemble_strategy
    report['process_preconditioning_channels']=list(cfg.process_preconditioning_channels)
    report['unit_correction_gate_channels']=list(cfg.unit_correction_gate_channels)
    with h5py.File(args.cache, 'r') as f:
        available=np.flatnonzero(np.array([int(d[:4])>=args.start_year for d in f['dates'].asstr()[:]]))
        if not len(available):raise ValueError('No dates in selected period')
        indices=available[np.unique(np.linspace(0,len(available)-1,min(args.dates,len(available)),dtype=int))]
        latitude, longitude = f['latitude'][:], f['longitude'][:]
        dates = f['dates'].asstr()[indices]
        for ordinal, index in enumerate(indices):
            batch = {key:torch.from_numpy(f[key][index:index+1]).to(args.device)
                     for key in ('conditioning','baseline','truth','valid')}
            base, truth, valid = batch['baseline'], batch['truth'], batch['valid']
            residual = truth-base
            normalized_target = normalizer.normalize(residual,valid)
            values = {'phase1_physical':base, 'target_physical':truth, 'target_physical_residual':residual,
                      'target_transformed_residual':normalizer._forward_transform(residual),
                      'target_normalized_residual':normalized_target, 'conditioning':batch['conditioning'],
                      'valid':valid, 'residual_roundtrip_error':normalizer.denormalize(normalized_target)-residual}
            calls = [0]
            process_calls=[0]
            original_predict=refiner.predict_process
            def capture_process(state,conditioning,time):
                prediction=original_predict(state,conditioning,time)
                k=process_calls[0];process_calls[0]+=1
                if ordinal==0 and k in (0,1,10,24,50,74,90,99,100):
                    for label,value in [('state',state),('time',time),('prediction',prediction)]:
                        network_outputs[f'process_{label}_{k:03d}']=value.detach().float().cpu().numpy()
                return prediction
            refiner.predict_process=capture_process
            def network_hook(module, inputs, output):
                k = calls[0]; calls[0] += 1
                if ordinal == 0 and k in (0,1,10,24,50,74,90,99,100):
                    network_outputs[f'output_{k:03d}'] = output.detach().float().cpu().numpy()
                    network_outputs[f'input_{k:03d}'] = inputs[0].detach().float().cpu().numpy()
                    network_outputs[f'time_{k:03d}'] = inputs[2].detach().float().cpu().numpy()
            def callback(stage,step,time,state):
                if ordinal == 0:
                    trajectory[f'{stage}_{step:03d}'] = state.detach().float().cpu().numpy()
                    trajectory_times[f'{stage}_{step:03d}'] = float(time.detach().cpu())
            hook = refiner.net.register_forward_hook(network_hook)
            with torch.inference_mode():
                cond = batch['conditioning'].expand(args.members,-1,-1,-1)
                z = refiner.sample(cond, generator=torch.Generator(device=args.device).manual_seed(1234+ordinal),
                                   num_steps=args.steps, trajectory_callback=callback)
                transformed = z*normalizer.scale.to(z)+normalizer.mean.to(z)
                correction = normalizer.denormalize(z)
                if hasattr(refiner,'apply_correction_gate'):correction=refiner.apply_correction_gate(correction)
                unbounded = (base+correction).unsqueeze(0)
                selected = constraint.constrain_physical_ensemble(unbounded,strategy=cfg.nonnegative_ensemble_strategy)
            hook.remove()
            refiner.predict_process=original_predict
            values.update(final_normalized_residual=z.unsqueeze(0), denormalized_transformed_residual=transformed.unsqueeze(0),
                          physical_residual=correction.unsqueeze(0), unbounded_members=unbounded,
                          physical_members=selected['members'], physical_mean=selected['members'].mean(1),
                          unbounded_mean=selected['unbounded_mean'], clipping_shift=selected['memberwise_clipping_mean_shift'])
            for key,value in values.items():
                stages.setdefault(key,[]).append(value.detach().float().cpu().numpy())
            print(f"{cfg.type}: traced {ordinal+1}/{len(indices)} dates ({dates[ordinal]})",flush=True)
    stages={key:np.concatenate(value) for key,value in stages.items()}
    np.savez_compressed(out/'stages.npz',**stages,dates=dates,latitude=latitude,longitude=longitude)
    np.savez_compressed(out/'trajectory.npz',**trajectory)
    (out/'trajectory_times.json').write_text(json.dumps(trajectory_times,indent=2))
    np.savez_compressed(out/'network_outputs.npz',**network_outputs)
    distance, regions = boundary_regions(stages['valid'].astype(bool).any((0,1)),latitude,longitude)
    results={}
    for channel,name in enumerate(('pr','tasmax')):
        results[name]={}
        for product in ('phase1_physical','unbounded_mean','physical_mean'):
            results[name][product]={region:physical_metrics(stages[product][:,channel],stages['target_physical'][:,channel],
                  stages['valid'][:,channel].astype(bool),region=mask,precipitation=name=='pr') for region,mask in regions.items()}
    report['metrics']=results
    report['roundtrip_by_region']={name:{region:float(np.max(np.abs(stages['residual_roundtrip_error'][:,c][:,mask][stages['valid'][:,c][:,mask].astype(bool)]))) for region,mask in regions.items() if mask.any()} for c,name in enumerate(('pr','tasmax'))}
    np.savez_compressed(out/'normalization_fields.npz',**{key:np.broadcast_to(getattr(normalizer,key).detach().cpu().numpy(),(1,2,len(latitude),len(longitude))) for key in ('mean','scale','count')})
    report['roundtrip_max_absolute_error'] = float(np.max(np.abs(stages['residual_roundtrip_error'][stages['valid'].astype(bool)])))
    report['stage_bands']={}
    for key in ('target_normalized_residual','final_normalized_residual','denormalized_transformed_residual','physical_residual','clipping_shift'):
        a=stages[key]
        if a.ndim==5: a=a.mean(1)
        report['stage_bands'][key]={name:{region:{'mean':float(a[:,c,mask].mean()),'std':float(a[:,c,mask].std())}
            for region,mask in regions.items() if mask.any()} for c,name in enumerate(('pr','tasmax'))}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({name:{product:{region:results[name][product][region] for region in ('outer_1','interior')}
                             for product in results[name]} for name in results},indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--cache',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--dates',type=int,default=8)
    parser.add_argument('--start-year',type=int,default=0)
    parser.add_argument('--members',type=int,default=4)
    parser.add_argument('--steps',type=int,default=50)
    run(parser.parse_args())
