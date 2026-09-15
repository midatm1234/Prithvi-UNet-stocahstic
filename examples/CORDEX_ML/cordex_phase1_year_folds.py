
"""Whole-year Phase-1 cross-validation with fold-fitted scalers and fresh weights.

This explicit driver uses original CORDEX files through whole-year Subsets;
it does not create new scientific observations or materialize duplicate NetCDFs.
Its configs must be run through this driver, not the unfiltered normal loader.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import xarray as xr
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).resolve().parent)]
from granitewxc.utils.config import get_config
from granitewxc.utils.predictands import build_predictand_specs

PLAN_VERSION="cordex.phase1.whole_year_folds.v1"


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def year_folds(dates, fold_count=5):
    """Balanced contiguous blocks within each contiguous source-year interval."""
    if fold_count < 3:
        raise ValueError("At least three folds are needed for separate fit/inner/outer roles.")
    if len(dates)!=len(set(dates)):
        raise ValueError("Duplicate timestamps cannot define disjoint folds.")
    years=sorted({int(date[:4]) for date in dates})
    if any(2041<=year<=2060 for year in years):
        raise ValueError("The quarantined 2041-2060 period is excluded from this route.")
    intervals=[]
    for year in years:
        if not intervals or year!=intervals[-1][-1]+1:
            intervals.append([])
        intervals[-1].append(year)
    if any(len(interval)<fold_count for interval in intervals):
        raise ValueError("Each contiguous source-year interval must span at least fold_count years.")
    partitions=[[] for _ in range(fold_count)]
    for interval in intervals:
        for index,block in enumerate(np.array_split(interval,fold_count)):
            partitions[index].extend(int(value) for value in block)
    array=np.asarray([int(date[:4]) for date in dates])
    folds=[]
    for index in range(fold_count):
        outer=set(partitions[index])
        inner=set(partitions[(index+1)%fold_count])
        fit=set(years)-outer-inner
        folds.append({
            "fold":index,"fit_years":sorted(fit),"inner_validation_years":sorted(inner),"outer_years":sorted(outer),
            "fit_indices":np.flatnonzero(np.isin(array,list(fit))).tolist(),
            "inner_validation_indices":np.flatnonzero(np.isin(array,list(inner))).tolist(),
            "outer_indices":np.flatnonzero(np.isin(array,list(outer))).tolist(),
        })
    return folds


def source_header_identity(raw):
    """Only coordinates/metadata are read; no physical target values are loaded."""
    data=raw["data"]
    predictors=list(data["training_predictor_paths"])
    targets=list(data["training_target_paths"])
    if len(predictors)!=len(targets) or not predictors:
        raise ValueError("Training source lists must be nonempty and paired.")
    dates=[]
    sources=[]
    from cordex_dataset import CordexDownscaleDataset
    for predictor,target in zip(predictors,targets):
        pair_dates=[]
        identities=[]
        for path in (predictor,target):
            resolved=Path(path).resolve()
            if any(token in str(resolved) for token in ("2041-2060","2041_2060")):
                raise ValueError("Quarantined target/source path is excluded.")
            with xr.open_dataset(resolved) as ds:
                times=np.asarray(ds["time"].values)
                keys=[CordexDownscaleDataset._format_time_key(CordexDownscaleDataset._time_key(value)) for value in times]
                if any(2041<=int(key[:4])<=2060 for key in keys):
                    raise ValueError("Quarantined years are excluded.")
                header={
                    "path":str(resolved),"size_bytes":resolved.stat().st_size,"mtime_ns":resolved.stat().st_mtime_ns,
                    "sizes":dict(ds.sizes),"variables":{name:{"dims":list(value.dims),"shape":list(value.shape),"dtype":str(value.dtype)}
                                                       for name,value in ds.variables.items()},
                    "date_sha256":digest_json(keys),
                    "coordinate_sha256":{axis:hashlib.sha256(np.asarray(ds[axis].values,dtype="<f8").tobytes()).hexdigest() for axis in ("lat","lon")},
                }
            identities.append(header)
            pair_dates.append(keys)
        if pair_dates[0]!=pair_dates[1]:
            raise ValueError("Predictor and target timestamps must exactly match for fold registration.")
        dates.extend(pair_dates[0])
        sources.append({"predictor":identities[0],"target":identities[1]})
    return dates,sources


def prepare(base_config,output_root,fold_count=5,epochs=100,seed=6149):
    raw=yaml.safe_load(Path(base_config).read_text(encoding="utf-8"))
    dates,sources=source_header_identity(raw)
    folds=year_folds(dates,fold_count)
    output=Path(output_root).resolve()
    output.mkdir(parents=True,exist_ok=False)
    plan={"version":PLAN_VERSION,"source_config":str(Path(base_config).resolve()),"dates":dates,"sources":sources,"folds":folds,
          "policy":"Fit scalers and all Phase-1 weights on fitting years only; inner years select checkpoints; outer years are excluded from both.",
          "initialization":"fresh random parameters; no pretrained backbone or exposed current checkpoint loaded",
          "quarantine":"2041-2060 target values/files excluded","source_authentication":"Header/coordinates/path/size/mtime. Physical file payloads were not read during registration."}
    plan["sha256"]=digest_json(plan)
    manifest=output/"fold_manifest.json"
    manifest.write_text(json.dumps(plan,indent=2))
    config_paths=[]
    for fold in folds:
        index=fold["fold"]
        folder=output/f"fold_{index:02d}"
        folder.mkdir()
        cfg=copy.deepcopy(raw)
        cfg["case_name"]=f"SA_T2_ACCESS-CM2_phase1_fresh_year_fold_{index:02d}"
        cfg["job_id"]=cfg["case_name"]
        cfg["model"].pop("phase1",None)
        cfg["model"]["refinement"]={"enabled":False,"type":"none"}
        cfg.pop("refinement_cases",None)
        cfg["path_model_weights"]=None
        cfg["backbone_use"]=True
        cfg["backbone_freeze"]=False
        cfg["auto_resume_if_checkpoint_exists"]=False
        cfg["resume_training"]=False
        cfg["resume_from_last_checkpoint"]=False
        cfg["resume_checkpoint_path"]=None
        cfg["load_model"]=None
        cfg["path_experiment"]=str(folder/"training")
        cfg["checkpoint_dir"]=str(folder/"checkpoints")
        cfg["num_epochs"]=int(epochs)
        cfg["limit_steps_train"]=0
        cfg["limit_steps_valid"]=0
        cfg["batch_size"]=2
        cfg["per_device_batch_size"]=2
        cfg["gradient_accumulation_steps"]=16
        cfg["dl_num_workers"]=0
        cfg["dl_prefetch_size"]=0
        cfg["device_target"]="cuda"
        cfg["distributed_strategy"]="none"
        cfg["training"]={"num_gpus":1,"distributed":"none","gradient_accumulation_steps":16,"auto_resume_if_checkpoint_exists":False}
        cfg["data"]["test_predictor_paths"]=[]
        cfg["data"]["test_target_paths"]=[]
        # Both lists point to the registered sources, but this driver's Subsets
        # apply the frozen fit/inner indices before any values are consumed.
        cfg["data"]["validation_predictor_paths"]=cfg["data"]["training_predictor_paths"]
        cfg["data"]["validation_target_paths"]=cfg["data"]["training_target_paths"]
        scalar_folder=folder/"scalers"
        scaler_keys=("inputs_mean","inputs_std","targets_mean","targets_std")
        paths={name:str(scalar_folder/(name+".npy")) for name in scaler_keys}
        cfg["data"]["scalers"]=paths
        cfg["model"].update(input_mu=paths["inputs_mean"],input_sigma=paths["inputs_std"],
                            target_mu=paths["targets_mean"],target_sigma=paths["targets_std"])
        cfg["phase1_fold"]={
            "manifest":str(manifest),"manifest_sha256":plan["sha256"],"fold":index,
            "initialization":"fresh_random_no_checkpoint","seed":int(seed)+index,
            "scalers_dir":str(scalar_folder),"fit_years":fold["fit_years"],
            "inner_validation_years":fold["inner_validation_years"],"outer_years":fold["outer_years"],
            "required_entry_point":"examples/CORDEX_ML/cordex_phase1_year_folds.py",
        }
        config_path=folder/"phase1.yaml"
        config_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        config_paths.append(config_path)
    return config_paths


def load_registered_fold(config_path):
    raw=yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    contract=raw["phase1_fold"]
    manifest=json.loads(Path(contract["manifest"]).read_text())
    unsigned={key:value for key,value in manifest.items() if key!="sha256"}
    if digest_json(unsigned)!=manifest["sha256"] or manifest["sha256"]!=contract["manifest_sha256"]:
        raise ValueError("Fold manifest changed.")
    if contract["initialization"]!="fresh_random_no_checkpoint" or raw.get("path_model_weights") is not None or raw.get("load_model"):
        raise ValueError("This route permits fresh random Phase 1 only; no checkpoint warm start.")
    if any(raw.get(key) for key in ("auto_resume_if_checkpoint_exists","resume_training","resume_from_last_checkpoint","resume_checkpoint_path")):
        raise ValueError("Resume is disabled for independent fresh folds.")
    dates,sources=source_header_identity(raw)
    if dates!=manifest["dates"] or sources!=manifest["sources"]:
        raise ValueError("Registered source headers, timestamps, or file identity changed.")
    fold=manifest["folds"][int(contract["fold"])]
    for key in ("fit_years","inner_validation_years","outer_years"):
        if contract[key]!=fold[key]:
            raise ValueError("Fold year membership changed.")
    config=get_config(str(config_path))
    return raw,config,manifest,fold


class FiniteRegrid:
    """Reject missing predictor cells before the legacy loader can fill zeros."""
    def __init__(self,wrapped):
        self.wrapped=wrapped
    def __call__(self,field):
        result=self.wrapped(field)
        values=np.asarray(result.values if isinstance(result,xr.DataArray) else result)
        if not np.isfinite(values).all():
            raise ValueError("Fold route requires a masked-scaler extension for missing regridded predictors; refusing zero-filled statistics.")
        return result


class CheckedSamples(Dataset):
    def __init__(self,base,indices,*,tensor_only=False):
        self.base=base
        self.indices=tuple(int(i) for i in indices)
        self.tensor_only=tensor_only
    def __len__(self):
        return len(self.indices)
    def __getitem__(self,index):
        sample=self.base[self.indices[index]]
        valid=sample.get("__target_valid_mask")
        if valid is not None and not bool(valid.all()):
            raise ValueError("Native Phase-1 scalar/loss routines do not exclude invalid targets; refusing this fold before finite-filling changes statistics.")
        if self.tensor_only:
            return {key:value for key,value in sample.items() if torch.is_tensor(value)}
        return sample


def build_base(config):
    from cordex_training import _build_base_dataset
    base=_build_base_dataset(
        config,list(config.data.training_predictor_paths),list(config.data.training_target_paths),
        crop_size=(int(config.data.target_size_lat),int(config.data.target_size_lon)),
        random_crop=False,random_crop_offset=(0,0),
    )
    # The regional static field is time-independent. Require all its original
    # cells finite too, before accepting the legacy loader's sanitation.
    if base.use_static:
        with xr.open_dataset(base.orography_path) as ds:
            var=base.orography_var if base.orography_var in ds.data_vars else next(iter(ds.data_vars))
            if not np.isfinite(np.asarray(ds[var].values)).all():
                raise ValueError("Static input contains missing values; fold scalar fitting requires an explicit mask policy.")
    base.regridder=FiniteRegrid(base.regridder)
    return base


def fit_scalers(config_path):
    raw,config,manifest,fold=load_registered_fold(config_path)
    folder=Path(raw["phase1_fold"]["scalers_dir"])
    folder.mkdir(parents=True,exist_ok=False)
    from compute_scalars_cordex import compute_scalars
    base=build_base(config)
    selected=CheckedSamples(base,fold["fit_indices"])
    variables=list(config.data.output_vars)
    specs=build_predictand_specs(config,output_vars=variables)
    stats=compute_scalars(selected,progress_interval=100,target_specs=specs,target_vars=variables,target_scale_sample_size=2048)
    hashes={}
    for name in ("inputs_mean","inputs_std","targets_mean","targets_std"):
        path=folder/(name+".npy")
        np.save(path,stats[name])
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    metadata={
        "fold_manifest_sha256":manifest["sha256"],"fold":fold["fold"],"fit_years":fold["fit_years"],
        "fit_indices_sha256":digest_json(fold["fit_indices"]),"sample_count":len(selected),
        "predictands":{spec.name:spec.to_dict() for spec in specs},"files_sha256":hashes,
        "scalar_implementation":"compute_scalars_cordex.compute_scalars, selected fitting indices only",
        "quantile_sampling":"Native fixed-seed42; 2048 spatial values per fitting date, including valid dry zeros.",
        "missing_policy":"Reject missing predictor/static/target values instead of silently treating finite-filled zeros as observations.",
    }
    (folder/"fold_scaler_metadata.json").write_text(json.dumps(metadata,indent=2))
    return metadata


def assert_scalers(config,manifest,fold):
    folder=Path(config.phase1_fold["scalers_dir"])
    meta=json.loads((folder/"fold_scaler_metadata.json").read_text())
    if meta["fold_manifest_sha256"]!=manifest["sha256"] or meta["fold"]!=fold["fold"] or meta["fit_years"]!=fold["fit_years"]:
        raise ValueError("Scalers belong to another fold.")
    if meta["fit_indices_sha256"]!=digest_json(fold["fit_indices"]):
        raise ValueError("Scalar fitting indices differ from this fold.")
    bindings={"inputs_mean":"input_mu","inputs_std":"input_sigma","targets_mean":"target_mu","targets_std":"target_sigma"}
    configured_data=config.data.scalers
    for name,attribute in bindings.items():
        expected_path=(folder/(name+".npy")).resolve()
        if Path(getattr(config.model,attribute)).resolve()!=expected_path or Path(configured_data[name]).resolve()!=expected_path:
            raise ValueError("Model scaler paths must point only to this fold: "+name)
    for name,expected in meta["files_sha256"].items():
        if hashlib.sha256((folder/(name+".npy")).read_bytes()).hexdigest()!=expected:
            raise ValueError("Fold scalar file changed: "+name)
    return meta


def fresh_phase1_model(config):
    """Call the native factory directly; never load any checkpoint weights."""
    if config.phase1_fold["initialization"]!="fresh_random_no_checkpoint" or config.path_model_weights is not None:
        raise ValueError("Fresh fold initialization cannot load weights.")
    from granitewxc.models.model import get_finetune_model_UNET
    return get_finetune_model_UNET(config)


def train(config_path,device_text="cuda:0"):
    raw,config,manifest,fold=load_registered_fold(config_path)
    scalar_meta=assert_scalers(config,manifest,fold)
    checkpoint_dir=Path(config.checkpoint_dir)
    if checkpoint_dir.exists():
        raise FileExistsError("Fresh fold training refuses an existing checkpoint directory.")
    seed=int(config.phase1_fold["seed"])
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    from cordex_training import CordexWrappedDataset, build_loss_fn, build_optimizer_scheduler
    from granitewxc.utils.trainer import train_model
    base=build_base(config)
    wrapped=CordexWrappedDataset(base)
    training=CheckedSamples(wrapped,fold["fit_indices"],tensor_only=True)
    validation=CheckedSamples(wrapped,fold["inner_validation_indices"],tensor_only=True)
    device=torch.device(device_text)
    use_gpu=device.type=="cuda"
    common=dict(batch_size=config.batch_size,num_workers=0,pin_memory=use_gpu)
    train_loader=DataLoader(training,shuffle=True,generator=torch.Generator().manual_seed(seed),**common)
    valid_loader=DataLoader(validation,shuffle=False,**common)
    # Native scheduler min(loader_length,limit_steps_train) interprets zero
    # differently from native full-epoch iteration. Supply exact full lengths.
    config.limit_steps_train=len(train_loader)
    config.limit_steps_valid=len(valid_loader)
    model=fresh_phase1_model(config).to(device)
    if use_gpu:
        torch.cuda.set_device(device)
    optimizer,scaler,scheduler=build_optimizer_scheduler(config,model,len(train_loader),use_gpu=use_gpu)
    loss=build_loss_fn(config,list(config.data.output_vars))
    Path(config.path_experiment).mkdir(parents=True,exist_ok=False)
    run_contract={"initialization":"fresh random; no checkpoint loaded","seed":seed,"fold":fold,
                  "manifest_sha256":manifest["sha256"],"scalar_metadata":scalar_meta,
                  "precision":"Native trainer CUDA autocast (bf16 if supported, otherwise fp16); CPU path float32.",
                  "outer_loader_created":False}
    (Path(config.path_experiment)/"fold_training_contract.json").write_text(json.dumps(run_contract,indent=2))
    train_model(config,model,train_loader,valid_loader,optimizer,scheduler,scaler,
                local_rank=device.index or 0,use_gpu=use_gpu,save_every=5,loss_func=loss)
    for leaf in ("best.ckpt","last.ckpt"):
        checkpoint=checkpoint_dir/leaf
        if not checkpoint.exists():
            raise RuntimeError("Native training did not produce "+leaf)
        hasher=hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b""):
                hasher.update(chunk)
        (checkpoint_dir/(leaf+".fold.json")).write_text(json.dumps({
            **run_contract,"checkpoint_sha256":hasher.hexdigest(),
            "resolved_config":config.to_dict(),"best_selected_on":"inner_validation_years only",
        },indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    prepare_parser=sub.add_parser("prepare",help="Read date coordinates only and write frozen whole-year fold configs.")
    prepare_parser.add_argument("--base-config",required=True)
    prepare_parser.add_argument("--output-root",required=True)
    prepare_parser.add_argument("--fold-count",type=int,default=5)
    prepare_parser.add_argument("--epochs",type=int,default=100)
    prepare_parser.add_argument("--seed",type=int,default=6149)
    for name in ("check","fit-scalers","train"):
        item=sub.add_parser(name)
        item.add_argument("--fold-config",required=True)
        if name=="train":
            item.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    if args.command=="prepare":
        for path in prepare(args.base_config,args.output_root,args.fold_count,args.epochs,args.seed):
            print(path)
    elif args.command=="check":
        raw,config,manifest,fold=load_registered_fold(args.fold_config)
        print(json.dumps({key:value for key,value in fold.items() if not key.endswith("_indices")},indent=2))
        print("fit / inner validation / outer samples:",*(len(fold[key]) for key in ("fit_indices","inner_validation_indices","outer_indices")))
        print("Checked headers only. No physical values, scalers or model weights loaded.")
    elif args.command=="fit-scalers":
        print(json.dumps(fit_scalers(args.fold_config),indent=2))
    else:
        train(args.fold_config,args.device)


if __name__=="__main__":
    main()

