"""Bounded orchestration regression: reuse valid spatial updates and reject gaps."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PATH=Path(__file__).resolve().parents[1]/"examples/CORDEX_ML/cordex_temporal_continue_bounded.py"
spec=importlib.util.spec_from_file_location("bounded_recovery_under_test",PATH)
recovery=importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def checkpoint(count=600,temporal=0,complete=True):
    return {"global_step":count,"temporal":{"trained_temporal_steps":temporal},
        "training_state":{"schema":"temporal_training_v2","epoch_complete":complete,
            "protocol":{"max_steps_per_epoch":600,"max_val_steps":60,
                        "gradient_accumulation_steps":1,"batch_size":1,"num_workers":0}}}


def test_spatial_budget_uses_global_step_and_rejects_partial_epoch():
    payload=checkpoint()
    module=SimpleNamespace(torch=SimpleNamespace(load=lambda *a,**k:payload))
    _,count,trained=recovery.checkpoint_progress(module,Path("unused"),"spatial_ft")
    assert count==600 and trained==0
    payload["global_step"]=599
    with pytest.raises(ValueError,match="undershot"):
        recovery.checkpoint_progress(module,Path("unused"),"spatial_ft")
    payload["training_state"]["epoch_complete"]=False
    assert recovery.checkpoint_progress(module,Path("unused"),"spatial_ft")[1]==599
    payload["global_step"]=601
    with pytest.raises(ValueError,match="exceeds"):
        recovery.checkpoint_progress(module,Path("unused"),"spatial_ft")


def test_process_creation_identity_mismatch_never_waits(monkeypatch):
    closed=[]
    kernel=SimpleNamespace(CloseHandle=lambda h:closed.append(h),
                WaitForSingleObject=lambda *a:pytest.fail("wrong process must not be waited on"))
    monkeypatch.setattr(recovery,"process_handle",lambda pid:(kernel,123,900))
    with pytest.raises(RuntimeError,match="creation identity"):
        recovery.wait_for_original(1,901,1,lambda elapsed:None)
    assert closed==[123]


def test_prediction_contract_rejects_dates_and_variables(tmp_path):
    path=tmp_path/"predictions.npz"
    args={"pred":np.ones((2,2,1,1)),"target":np.ones((2,2,1,1)),
          "mask":np.ones((2,2,1,1)),"dates":np.array([[1981,1,1],[1981,1,2]]),
          "seams":np.array([]),"output_vars":np.array(["pr","tasmax"])}
    np.savez(path,**args)
    module=SimpleNamespace(np=np)
    contract=recovery.prediction_contract(module,path)
    assert contract["shape"]==[2,2,1,1]
    args["dates"][1]=args["dates"][0]
    np.savez(path,**args)
    with pytest.raises(ValueError,match="duplicated"):
        recovery.prediction_contract(module,path)


def test_complete_score_requires_all_controls_and_exact_budgets():
    manifest={"variants":{name:{"status":"completed","actual_optimizer_steps":600}
                          for name in recovery.VARIANTS}}
    reports={name:{} for name in recovery.VARIANTS}
    module=SimpleNamespace(SCORER_VERSION="2",CANDIDATES=("native_pair","native_pair_pretext"),
        MUST_BEAT=("spatial_ft","time_only"),EXTRA_MUST_BEAT={"native_pair":("native_pair_nohistory",)},
        score_acceptance=lambda *a:{"accepted":True},
        beats_controls=lambda reports,name,controls,variables:{c:{"beats":True} for c in controls})
    assert recovery.score_complete(module,manifest,reports)["variants"]["native_pair"]["scientific_acceptance"]
    manifest["variants"]["spatial_ft"]["actual_optimizer_steps"]=0
    with pytest.raises(ValueError,match="incomplete or under-budget"):
        recovery.score_complete(module,manifest,reports)
    manifest["variants"]["spatial_ft"]["actual_optimizer_steps"]=600
    reports.pop("native_pair_nohistory")
    with pytest.raises(ValueError,match="incomplete or under-budget"):
        recovery.score_complete(module,manifest,reports)


def test_controller_reuses_spatial_600_and_trains_only_remaining(monkeypatch,tmp_path):
    root=tmp_path/"run";root.mkdir()
    snapshot=tmp_path/"snapshot";snapshot.mkdir()
    raw={name:{"variant":name,"temporal":{}} for name in recovery.VARIANTS}
    entries={name:{"status":"not started"} for name in recovery.VARIANTS}
    entries["baseline"]["status"]="completed"
    entries["spatial_ft"]["status"]="failed"
    for name in recovery.VARIANTS:
        (root/name).mkdir()
        (root/name/"resolved.yaml").write_text("placeholder")
    (root/"baseline/predictions.npz").write_bytes(b"existing baseline")
    (root/"baseline/evaluation.json").write_text("{}")
    entries["baseline"].update(predictions=str(root/"baseline/predictions.npz"),
                               evaluation=str(root/"baseline/evaluation.json"))
    (root/"spatial_ft/checkpoints").mkdir()
    (root/"spatial_ft/checkpoints/last.ckpt").write_bytes(b"valid trained spatial")
    (root/"manifest.json").write_text(json.dumps({"variants":entries,"device":"cpu"}))
    (root/"execution.json").write_text(json.dumps({"pid":1,"status":"failed"}))
    trained=[];inferred=[];counts={"spatial_ft":600}
    def train(config,cfg,**kwargs):
        name=config.variant;trained.append(name);counts[name]=600
        directory=Path(kwargs["output_dir"]);directory.mkdir(exist_ok=True)
        (directory/"last.ckpt").write_bytes(b"new checkpoint")
        assert kwargs["max_steps_per_epoch"]==600 and kwargs["max_val_steps"]==60 and kwargs["num_epochs"]==1
        return {"global_step":600,"trained_temporal_steps":600,"history":[]}
    module=SimpleNamespace(ExperimentConfig=SimpleNamespace(from_dict=lambda raw:SimpleNamespace(**raw)),
        parse_temporal_config=lambda data:SimpleNamespace(),train_temporal_model=train,
        torch=SimpleNamespace(device=lambda value:SimpleNamespace(type=value)),
        SCORER_VERSION="2",CANDIDATES=("native_pair","native_pair_pretext"),
        MUST_BEAT=("spatial_ft","time_only"),EXTRA_MUST_BEAT={"native_pair":("native_pair_nohistory",)},
        score_acceptance=lambda *a:{"accepted":True},
        beats_controls=lambda reports,name,controls,variables:{c:{"beats":True} for c in controls})
    def progress(module,path,name,cfg=None):
        count=counts[name]
        payload={"training_state":{"epoch_complete":True,"counters":{"history":[]}}}
        return payload,count,0 if name=="spatial_ft" else count
    def infer(module,config,cfg,root,name,device):
        inferred.append(name)
        prediction=root/name/"predictions_recovery.npz";prediction.write_bytes(b"new prediction")
        evaluation=root/name/"evaluation_recovery.json";evaluation.write_text("{}")
        return {},{"predictions":str(prediction),"evaluation":str(evaluation)}
    monkeypatch.setattr(recovery,"verify_snapshot",lambda path:{"checked_files":1})
    monkeypatch.setattr(recovery,"validate_manifest",lambda *args:None)
    monkeypatch.setattr(recovery,"load_frozen_experiment",lambda path:module)
    monkeypatch.setattr(recovery,"validate_config",lambda module,name,root:json.loads(json.dumps(raw[name])))
    monkeypatch.setattr(recovery,"inspect_process",lambda pid:{"exists":False})
    monkeypatch.setattr(recovery,"checkpoint_progress",progress)
    monkeypatch.setattr(recovery,"infer_and_evaluate",infer)
    monkeypatch.setattr(recovery,"prediction_contract",lambda *args:{"identical":"dates-targets-masks"})
    monkeypatch.setattr(recovery.runpy,"run_path",lambda *a,**k:None)
    monkeypatch.setattr(recovery.sys,"argv",[str(PATH),"--snapshot",str(snapshot),"--out",str(root)])
    assert recovery.main()==0
    assert trained==list(recovery.VARIANTS[2:])
    assert inferred==list(recovery.VARIANTS[1:])
    saved=json.loads((root/"manifest.json").read_text())
    assert saved["variants"]["spatial_ft"]["actual_optimizer_steps"]==600
    assert saved["variants"]["spatial_ft"]["trained_temporal_steps"]==0
    assert all(info["status"]=="completed" for info in saved["variants"].values())
    assert (root/"baseline/predictions.npz").read_bytes()==b"existing baseline"
    assert not (root/"bounded_recovery.lock").exists()
