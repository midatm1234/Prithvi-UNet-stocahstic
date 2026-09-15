"""Create an immutable full-development Phase-1 cache, with exact year blocks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import h5py
import numpy as np
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True); p.add_argument('--output-dir',required=True)
    p.add_argument('--batch-size',type=int,default=16); p.add_argument('--device',default='cuda:0')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv)
    from cordex_training import build_dataloader
    from granitewxc.utils.config import get_config
    from granitewxc.models.model import get_finetune_model_UNET
    from granitewxc.refinement.two_phase import build_two_phase_model
    from granitewxc.refinement.checkpoint import load_phase1_state_dict, phase1_state_fingerprint
    from granitewxc.refinement.scientific_data import CACHE_VERSION, canonical_sha256, file_sha256, registered_splits
    out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    final=out/'phase1.h5'; partial=out/'phase1.partial.h5'
    if final.exists(): raise FileExistsError('Completed cache is immutable; select a new directory.')
    if partial.exists() and not args.resume: raise FileExistsError('Use --resume for the existing partial cache.')
    raw=yaml.safe_load(Path(args.config).read_text(encoding='utf8'))
    config=get_config(args.config); config.batch_size=args.batch_size
    config.dl_num_workers=0; config.dl_pin_memory=False
    model=build_two_phase_model(get_finetune_model_UNET(config),config).to(args.device).eval()
    phase1=Path(config.model.phase1['checkpoint'])
    if not phase1.is_absolute(): phase1=(ROOT/phase1).resolve()
    payload=torch.load(phase1,map_location='cpu',weights_only=False)
    report=load_phase1_state_dict(model,payload)
    del payload
    print('strict Phase-1 restoration:',report.summary(),flush=True)
    assert not model.phase1.training and not any(q.requires_grad for q in model.phase1.parameters())
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    source_paths=list(config.data.training_predictor_paths)+list(config.data.training_target_paths)
    input_files=[dict(path=str(Path(v).resolve()),size=Path(v).stat().st_size,sha256=file_sha256(v)) for v in source_paths]
    scalers={key: {'path':str(value),'sha256':file_sha256(value)} for key,value in raw['data']['scalers'].items()}
    manifest={'version':CACHE_VERSION,'complete':False,'source_case':'CORDEX-SA-ACCESS-CM2-historical-future-training',
              'config':raw,'config_sha256':canonical_sha256(raw),'phase1_checkpoint':str(phase1),
              'phase1_checkpoint_sha256':file_sha256(phase1),
              'phase1_state_fingerprint':phase1_state_fingerprint(model.phase1.state_dict()),
              'source_files':input_files,'scalers':scalers,'variables':list(config.data.output_vars),
              'sampling':'No refinement sampling or fitted targets used to construct Phase-1 predictions.',
              'precision':'FP32, TF32 disabled','phase1_frozen':True,'phase1_eval':True}
    loader=build_dataloader(config,config.data.training_predictor_paths,config.data.training_target_paths,
        shuffle=False,use_gpu=True,distributed=False,rank=0,world_size=1,crop_size=(128,128),random_crop=False,
        random_crop_offset=(0,0))
    n=len(loader.dataset); dates=[]; timer=time.perf_counter()
    with xr.open_dataset(config.data.training_target_paths[0]) as target:
        latitude=target.lat.values; longitude=target.lon.values
        manifest['units']={v:target[v].attrs.get('units','') for v in manifest['variables']}
        manifest['calendar']=target.time.encoding.get('calendar',target.time.attrs.get('calendar','standard'))
    mode='r+' if partial.exists() else 'w'
    with h5py.File(partial,mode) as cache:
        if mode=='w':
            cache.attrs['identity']=canonical_sha256(manifest); cache.attrs['completed_samples']=0
            cache.create_group('fields'); cache.create_dataset('latitude',data=latitude); cache.create_dataset('longitude',data=longitude)
            cache.create_dataset('dates',(n,),dtype=h5py.string_dtype('utf-8'))
        elif cache.attrs['identity'] != canonical_sha256(manifest):
            raise ValueError('Partial cache source/checkpoint/config changed.')
        completed=int(cache.attrs['completed_samples'])
        offset=0
        for batch in loader:
            b=len(batch['y']); stop=offset+b
            dates.extend(list(batch['__sample_timestamp']))
            if stop<=completed: offset=stop; continue
            if offset<completed: raise ValueError('Resume batch boundary changed; use original batch size.')
            moved={k:v.to(args.device) if torch.is_tensor(v) else v for k,v in batch.items()}
            with torch.inference_mode(): base,normalized,_=model._prepare(moved)
            fields={k:v for k,v in batch.items() if torch.is_tensor(v) and not k.startswith('__sample_')}
            fields['__phase1_physical']=base.cpu(); fields['__phase1_normalized']=normalized.cpu()
            for key,value in fields.items():
                array=value.detach().cpu().numpy()
                if key not in cache['fields']:
                    cache['fields'].create_dataset(key,shape=(n,*array.shape[1:]),dtype=array.dtype,
                        chunks=(1,*array.shape[1:]),compression='lzf',shuffle=True)
                cache['fields'][key][offset:stop]=array
            cache['dates'][offset:stop]=list(batch['__sample_timestamp'])
            cache.attrs['completed_samples']=stop; cache.flush(); offset=stop
            if offset % (args.batch_size*10)==0 or offset==n:
                print(f'cache {offset}/{n}, elapsed={time.perf_counter()-timer:.1f}s',flush=True)
        if offset!=n: raise RuntimeError('Incomplete cache.')
    manifest['splits']=registered_splits(dates)
    manifest['elapsed_seconds']=time.perf_counter()-timer; manifest['complete']=True
    partial.rename(final)
    manifest['cache_sha256']=file_sha256(final)
    (out/'phase1.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    (out/'split_manifest.json').write_text(json.dumps(manifest['splits'],indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in manifest.items() if k in ('cache_sha256','elapsed_seconds','complete','units')},indent=2),flush=True)

if __name__=='__main__': main()
