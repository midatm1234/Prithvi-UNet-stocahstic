"""Whole-year fold registration and native scalar/trainer plumbing."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch
from torch import nn
import xarray as xr
import yaml

from examples.CORDEX_ML import cordex_phase1_year_folds as folds
from refinement_fixtures import TinyPhase1


def dates():
    return [f"{year}-07-01T12:00:00" for year in list(range(1961,1971))+list(range(2080,2090))]


def test_whole_year_roles_are_disjoint_and_every_date_is_outer_once():
    sample_dates=[f"{year}-{month:02d}-01T12:00:00" for year in list(range(1961,1981))+list(range(2080,2100)) for month in (1,4,7,10)]
    plan=folds.year_folds(sample_dates)
    visited=[]
    for fold in plan:
        groups=[set(fold[key]) for key in ("fit_indices","inner_validation_indices","outer_indices")]
        assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
        assert set.union(*groups)==set(range(len(sample_dates)))
        assert len(fold["fit_years"])==24
        assert len(fold["inner_validation_years"])==len(fold["outer_years"])==8
        for role in ("fit","inner_validation","outer"):
            assert all(int(sample_dates[i][:4]) in fold[role+"_years"] for i in fold[role+"_indices"])
        visited.extend(fold["outer_indices"])
    assert sorted(visited)==list(range(len(sample_dates)))


def test_quarantined_years_and_duplicate_dates_fail():
    with pytest.raises(ValueError,match="quarantined"):
        folds.year_folds(dates()+["2041-01-01T12:00:00"])
    with pytest.raises(ValueError,match="Duplicate"):
        folds.year_folds(dates()+dates()[:1])


def prepared(tmp_path):
    root=Path(__file__).resolve().parents[1]
    original=root/"examples/CORDEX_ML/refinement_audit_configs/SA_T2_ACCESS-CM2_static_flow_matching_unet_audit_legacy_schema2_fixed.yaml"
    cfg=yaml.safe_load(original.read_text())
    values=np.zeros((len(dates()),2,3),np.float32)
    coords={"time":np.array(dates(),dtype="datetime64[ns]"),"lat":[-30.,-29.],"lon":[22.,23.,24.]}
    for filename,var in (("predictor.nc","u_850"),("target.nc","pr")):
        xr.Dataset({var:(("time","lat","lon"),values)},coords=coords).to_netcdf(tmp_path/filename)
    cfg["data"]["training_predictor_paths"]=[str(tmp_path/"predictor.nc")]
    cfg["data"]["training_target_paths"]=[str(tmp_path/"target.nc")]
    base=tmp_path/"base.yaml"
    base.write_text(yaml.safe_dump(cfg))
    configs=folds.prepare(base,tmp_path/"folds",epochs=1)
    return configs[0]


class SyntheticSource:
    use_static=True
    def __init__(self,fold):
        self.fold=fold
        self.visited=[]
    def __len__(self):
        return len(dates())
    def __getitem__(self,index):
        self.visited.append(index)
        level=float(index+1) if index in self.fold["fit_indices"] else 1000. if index in self.fold["inner_validation_indices"] else 1e9
        x=torch.full((16,2,3),level)
        y=torch.stack([torch.full((2,3),level),torch.full((2,3),280+level)])
        return {"x":x,"y":y,"__target_valid_mask":torch.ones_like(y,dtype=torch.bool),
                "__sample_timestamp":dates()[index]}


def test_scalers_fit_only_registered_fit_years_and_reject_other_fold_paths(tmp_path,monkeypatch):
    config_path=prepared(tmp_path)
    raw,config,manifest,fold=folds.load_registered_fold(config_path)
    source=SyntheticSource(fold)
    monkeypatch.setattr(folds,"build_base",lambda config:source)
    result=folds.fit_scalers(config_path)
    assert source.visited==fold["fit_indices"]
    folder=Path(raw["phase1_fold"]["scalers_dir"])
    expected=np.mean([i+1 for i in fold["fit_indices"]])
    np.testing.assert_allclose(np.load(folder/"inputs_mean.npy"),expected)
    np.testing.assert_allclose(np.load(folder/"targets_mean.npy")[1],280+expected)
    assert result["sample_count"]==len(fold["fit_indices"])
    folds.assert_scalers(config,manifest,fold)
    config.model.input_mu="exposed_old_scalars.npy"
    with pytest.raises(ValueError,match="only to this fold"):
        folds.assert_scalers(config,manifest,fold)
    with pytest.raises(FileExistsError):
        folds.fit_scalers(config_path)


def test_missing_targets_and_predictors_fail_before_native_zero_filled_statistics():
    class Missing:
        def __getitem__(self,index):
            return {"x":torch.ones(2,2,3),"y":torch.zeros(2,2,3),"__target_valid_mask":torch.zeros(2,2,3,dtype=torch.bool)}
    with pytest.raises(ValueError,match="invalid targets"):
        folds.CheckedSamples(Missing(),[0])[0]
    with pytest.raises(ValueError,match="missing regridded"):
        folds.FiniteRegrid(lambda x:np.array([[float("nan")]]))(None)


def test_fresh_factory_never_calls_pretrained_loader(monkeypatch):
    import granitewxc.models.model as factory
    import cordex_training
    called=[]
    monkeypatch.setattr(factory,"get_finetune_model_UNET",lambda config:called.append(config) or nn.Linear(2,2))
    monkeypatch.setattr(cordex_training,"load_pretrained_weights",lambda *a,**k:pytest.fail("Exposed checkpoint loading was called"))
    config=SimpleNamespace(phase1_fold={"initialization":"fresh_random_no_checkpoint"},path_model_weights=None)
    assert isinstance(folds.fresh_phase1_model(config),nn.Linear)
    assert called==[config]
    config.path_model_weights="current_exposed_checkpoint.ckpt"
    with pytest.raises(ValueError,match="cannot load"):
        folds.fresh_phase1_model(config)


def test_native_training_smoke_uses_inner_only_and_records_bound_fold_checkpoint(tmp_path,monkeypatch):
    import cordex_training
    config_path=prepared(tmp_path)
    raw,config,manifest,fold=folds.load_registered_fold(config_path)
    source=SyntheticSource(fold)
    monkeypatch.setattr(folds,"build_base",lambda config:source)
    folds.fit_scalers(config_path)
    source.visited.clear()
    monkeypatch.setattr(folds,"fresh_phase1_model",lambda config:TinyPhase1(
        in_channels=15,out_channels=2,scaling_codes=(1,0),nonneg=(True,False),hidden=2))
    monkeypatch.setattr(cordex_training,"build_loss_fn",lambda *a:lambda prediction,batch:((prediction-batch["y"])**2).mean())
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        folds.train(config_path,"cpu")
    finally:
        torch.set_num_threads(previous)
    assert set(source.visited)==set(fold["fit_indices"]+fold["inner_validation_indices"])
    assert not set(source.visited)&set(fold["outer_indices"])
    sidecar=Path(raw["checkpoint_dir"])/"best.ckpt.fold.json"
    saved=json.loads(sidecar.read_text())
    assert saved["outer_loader_created"] is False
    assert saved["initialization"]=="fresh random; no checkpoint loaded"
    assert saved["manifest_sha256"]==manifest["sha256"]
    assert len(saved["checkpoint_sha256"])==64
    with pytest.raises(FileExistsError,match="existing checkpoint"):
        folds.train(config_path,"cpu")
