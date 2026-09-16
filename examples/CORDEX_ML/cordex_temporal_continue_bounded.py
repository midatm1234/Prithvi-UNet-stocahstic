"""Continue the frozen six-variant experiment after its original process exits.

Only orchestration is corrected: successful optimizer updates use global_step,
while trained_temporal_steps remains zero for the frozen-temporal spatial control.
The model, trainer, losses, configuration and evaluation code come from the
recorded source snapshot. No process is stopped and no completed update repeats.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

VARIANTS = ("baseline", "spatial_ft", "time_only", "native_pair", "native_pair_nohistory", "native_pair_pretext")
STEPS, VAL_STEPS, EPOCHS = 600, 60, 1
TEST_PERIOD = ("1981-01-01", "1983-12-31")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value):
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def process_handle(pid: int):
    """Open the exact Windows process, retaining identity across PID reuse."""
    if os.name != "nt":
        raise RuntimeError("Waiting on a PID currently requires the Windows execution host")
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    handle = kernel.OpenProcess(0x00100000 | 0x1000, False, int(pid))
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: PID no longer exists.
            return kernel, None, None
        raise OSError(error, f"Cannot inspect PID {pid}")
    times = [wintypes.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(handle, *(ctypes.byref(item) for item in times)):
        error = ctypes.get_last_error()
        kernel.CloseHandle(handle)
        raise OSError(error, f"Cannot read creation time of PID {pid}")
    created = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    return kernel, handle, created


def inspect_process(pid: int):
    kernel, handle, created = process_handle(pid)
    if handle is not None:
        kernel.CloseHandle(handle)
    return {"pid": pid, "creation_filetime": created, "exists": handle is not None}


def wait_for_original(pid: int, expected_creation: int, timeout: float, on_wait):
    kernel, handle, created = process_handle(pid)
    if handle is None:
        return {"pid": pid, "creation_filetime": expected_creation, "already_exited": True}
    try:
        if created != expected_creation:
            raise RuntimeError(f"PID {pid} creation identity differs; refusing to wait on an unrelated process")
        started = time.monotonic()
        while True:
            result = kernel.WaitForSingleObject(handle, 30000)
            if result == 0:
                return {"pid": pid, "creation_filetime": created, "already_exited": False}
            if result != 258:
                raise OSError(ctypes.get_last_error(), "WaitForSingleObject failed")
            elapsed = time.monotonic() - started
            on_wait(elapsed)
            if elapsed >= timeout:
                raise TimeoutError("Original experiment has not exited within the bounded wait")
    finally:
        kernel.CloseHandle(handle)


def verify_snapshot(snapshot: Path):
    recorded = json.loads((snapshot / "source_manifest.json").read_text(encoding="utf-8"))
    checked = 0
    for relative, expected in recorded.items():
        path = (snapshot / relative).resolve()
        if not path.is_relative_to(snapshot):
            raise ValueError(f"Source manifest escapes its snapshot: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Frozen source digest changed: {relative}")
        checked += 1
    if checked == 0:
        raise ValueError("Source snapshot manifest is empty")
    return {"manifest_sha256": hashlib.sha256((snapshot / "source_manifest.json").read_bytes()).hexdigest(),
            "checked_files": checked}


def validate_manifest(manifest, snapshot: Path):
    if tuple(manifest.get("planned_variants", ())) != VARIANTS:
        raise ValueError("Recovery requires the exact recorded six-variant order")
    for key, expected in (("optimizer_updates_per_epoch",STEPS), ("validation_windows",VAL_STEPS), ("epochs",EPOCHS)):
        if manifest.get(key) != expected:
            raise ValueError(f"Recorded {key} differs from bounded protocol {expected}")
    if tuple(manifest.get("test_period", ())) != TEST_PERIOD:
        raise ValueError("Recorded held-out period differs from 1981-01-01 through 1983-12-31")
    if Path(manifest["source_root"]).resolve() != snapshot:
        raise ValueError("Experiment source_root differs from the supplied frozen snapshot")
    script = snapshot / "examples/CORDEX_ML/cordex_temporal_experiment.py"
    if hashlib.sha256(script.read_bytes()).hexdigest() != manifest.get("source_sha256"):
        raise ValueError("Experiment driver digest does not match the frozen source")


def load_frozen_experiment(snapshot: Path):
    if any(name == "granitewxc" or name.startswith("granitewxc.") for name in sys.modules):
        raise RuntimeError("Recovery must start in a fresh interpreter before importing granitewxc")
    sys.path.insert(0, str(snapshot))
    script = snapshot / "examples/CORDEX_ML/cordex_temporal_experiment.py"
    spec = importlib.util.spec_from_file_location("frozen_bounded_experiment", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import granitewxc.temporal.training as trainer
    if not Path(trainer.__file__).resolve().is_relative_to(snapshot):
        raise RuntimeError("Trainer did not import from the frozen source snapshot")
    return module


def expected_config(module, name: str, root: Path):
    patch = module.variant_overrides()[name]
    raw = module.yaml.safe_load((Path(module._REPO_ROOT) / patch["_source"]).read_text(encoding="utf-8"))
    raw = module._deep_merge(raw, {key: value for key,value in patch.items() if not key.startswith("_")})
    directory = root / name
    raw.update(path_experiment=str(directory), checkpoint_dir=str(directory/"checkpoints"),
               run_dir=str(directory/"runs"), case_name=f"sa_temporal_experiment_{name}",
               job_id=f"sa_temporal_experiment_{name}", gradient_accumulation_steps=1)
    raw["temporal"]["inference"]["output_dir"] = str(directory/"inference")
    raw.setdefault("training", {})["gradient_accumulation_steps"] = 1
    return raw


def validate_config(module, name: str, root: Path):
    expected = expected_config(module, name, root)
    saved = root / name / "resolved.yaml"
    if saved.exists():
        actual = module.yaml.safe_load(saved.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"{name}: resolved configuration differs from frozen variant contract")
    return expected


def checkpoint_progress(module, path: Path, name: str, cfg=None):
    payload = module.torch.load(path, map_location="cpu", weights_only=False)
    if cfg is not None:
        stored = dict((payload.get("temporal") or {}).get("config") or {})
        expected = cfg.to_dict()
        for key in ("init_from_spatial_checkpoint", "resume_from_temporal_checkpoint"):
            stored.pop(key, None)
            expected.pop(key, None)
        if stored != expected:
            raise ValueError(f"{name}: checkpoint temporal configuration differs from its frozen variant")
        if (payload.get("temporal") or {}).get("architecture_version") != cfg.architecture_version:
            raise ValueError(f"{name}: checkpoint architecture version differs")
        if (payload.get("temporal") or {}).get("output_vars") != ["pr", "tasmax"]:
            raise ValueError(f"{name}: checkpoint output-variable contract differs")
    count = int(payload.get("global_step", -1))
    trained = int((payload.get("temporal") or {}).get("trained_temporal_steps", -1))
    state = payload.get("training_state") or {}
    if not 0 <= count <= STEPS or state.get("schema") != "temporal_training_v2":
        raise ValueError(f"{name}: checkpoint lacks compatible full training state or exceeds bounded update budget")
    if trained != (0 if name == "spatial_ft" else count):
        raise ValueError(f"{name}: temporal update counter does not match its freeze policy")
    protocol = state.get("protocol", {})
    if any(protocol.get(key) != value for key,value in {
        "max_steps_per_epoch":STEPS, "max_val_steps":VAL_STEPS,
        "gradient_accumulation_steps":1, "batch_size":1, "num_workers":0}.items()):
        raise ValueError(f"{name}: checkpoint training protocol differs from the bounded run")
    if count < STEPS and state.get("epoch_complete"):
        raise ValueError(f"{name}: completed epoch undershot its budget; refusing to repeat data as an exact resume")
    return payload, count, trained


def prediction_contract(module, path: Path):
    np = module.np
    with np.load(path, allow_pickle=False) as saved:
        required = {"pred", "target", "mask", "dates", "output_vars", "seams"}
        if not required.issubset(saved.files):
            raise ValueError(f"Prediction archive incomplete: {path}")
        prediction = saved["pred"]
        target = saved["target"]
        mask = saved["mask"]
        dates = saved["dates"]
        if prediction.shape != target.shape or mask.shape != target.shape or len(dates) != len(prediction):
            raise ValueError(f"Prediction/target/date geometry differs: {path}")
        if len(dates) == 0 or not np.isfinite(prediction).all():
            raise ValueError(f"Predictions are empty or nonfinite: {path}")
        if len({tuple(date) for date in dates}) != len(dates):
            raise ValueError(f"Scored dates are duplicated: {path}")
        if list(saved["output_vars"]) != ["pr", "tasmax"]:
            raise ValueError(f"Output-variable order differs: {path}")
        for date in dates:
            stamp = f"{int(date[0]):04d}-{int(date[1]):02d}-{int(date[2]):02d}"
            if not TEST_PERIOD[0] <= stamp <= TEST_PERIOD[1]:
                raise ValueError(f"Date outside bounded test period: {stamp}")
        digest = lambda array: hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        return {"shape": list(prediction.shape), "dates_sha256":digest(dates),
                "targets_sha256":digest(target), "mask_sha256":digest(mask),
                "output_vars":list(saved["output_vars"])}


def infer_and_evaluate(module, config, cfg, root: Path, name: str, device):
    np = module.np
    probe = module.build_sequence_dataloaders(config,cfg,splits=("train",),batch_size=1,verbose=False)
    dim,names = module.resolve_time_feature_dim(probe["train"])
    del probe
    if device.type == "cuda":
        module.torch.cuda.reset_peak_memory_stats(device)
    runner,migration,_ = module.build_temporal_model(config,cfg,time_feature_dim=dim,
                          time_feature_names=names,device=device,verbose=True)
    started = time.monotonic()
    results = module.run_sequence_inference(runner,config,cfg,split="test",
                    date_start=TEST_PERIOD[0],date_end=TEST_PERIOD[1],device=device,verbose=True)
    seconds = time.monotonic()-started
    peak_memory = int(module.torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    predictions = np.concatenate([result.predictions for result in results])
    targets = np.concatenate([result.targets for result in results])
    masks = np.concatenate([result.valid_mask for result in results])
    dates = [date for result in results for date in result.dates]
    seams,offset = [],0
    for result in results:
        seams.extend(int(value)+offset for value in result.seam_indices)
        offset += len(result.predictions)
    variables = [str(value) for value in config.data.output_vars]
    report = module.evaluate_predictions(predictions,targets,output_vars=variables,
        dates=[tuple(int(value) for value in date) for date in dates],mask=masks.astype(bool),
        event_aligned=True,wet_threshold=cfg.evaluation.wet_threshold,lags=cfg.evaluation.autocorr_lags,
        accumulation_windows=cfg.evaluation.accumulation_windows,
        boundary_width=cfg.evaluation.boundary_width,seam_indices=seams)
    directory=root/name
    # New recovery artifacts never replace partial originals.
    prediction_path=directory/"predictions_recovery.npz"
    evaluation_path=directory/"evaluation_recovery.json"
    if prediction_path.exists() or evaluation_path.exists():
        suffix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        prediction_path=directory/f"predictions_recovery_{suffix}.npz"
        evaluation_path=directory/f"evaluation_recovery_{suffix}.json"
    np.savez_compressed(prediction_path,pred=predictions.astype(np.float32),
        target=targets.astype(np.float32),mask=masks.astype(np.int8),dates=np.array(dates),
        seams=np.array(seams,dtype=np.int64),output_vars=np.array(variables))
    atomic_json(evaluation_path,report)
    del runner, results
    module.torch.cuda.empty_cache()
    return report,{"predictions":str(prediction_path),"evaluation":str(evaluation_path),
                   "n_dates":len(dates),"inference_seconds":seconds,"inference_peak_memory_allocated_bytes":peak_memory,
                   "migration":migration.to_dict()}


def score_complete(module, manifest, reports):
    complete = all(manifest["variants"].get(name,{}).get("status") == "completed" and name in reports
                   for name in VARIANTS)
    complete = complete and all(manifest["variants"][name].get("actual_optimizer_steps") == STEPS
                               for name in VARIANTS if name != "baseline")
    if not complete:
        raise ValueError("Cannot score an incomplete or under-budget bounded experiment")
    variables=["pr","tasmax"]
    card={"scorer_version":module.SCORER_VERSION,"complete":True,"variables":variables,
          "expected_variants":list(VARIANTS),"variants":{},
          "recovery_note":"Driver counter assertion corrected; frozen numerical training protocol unchanged"}
    for name in VARIANTS[1:]:
        entry=module.score_acceptance(reports["baseline"],reports[name],variables)
        if name in module.CANDIDATES:
            controls=tuple(module.MUST_BEAT)+tuple(module.EXTRA_MUST_BEAT.get(name,()))
            entry["versus_controls"]=module.beats_controls(reports,name,controls,variables)
            entry["must_beat"]=list(controls)
            entry["must_beat_missing"]=[control for control in controls if control not in reports]
            entry["beats_all_controls"]=all(value.get("beats") is True for value in entry["versus_controls"].values())
            entry["scientific_acceptance"]=bool(entry["accepted"] and entry["beats_all_controls"] and not entry["must_beat_missing"])
        else:
            entry["scientific_acceptance"]=None
        card["variants"][name]=entry
    return card


def render_report(manifest, scorecard, reports):
    lines=["# Recovered bounded temporal experiment", "",
        "Frozen model, trainer, loss and configuration sources were retained. Only the driver counter assertion was corrected.", "",
        "| Variant | Optimizer updates | Temporal updates | Verdict |", "|---|---:|---:|---|"]
    for name in VARIANTS:
        info=manifest["variants"][name]
        accepted=scorecard["variants"].get(name,{}).get("scientific_acceptance")
        verdict={True:"accepted",False:"not accepted",None:"control"}[accepted]
        lines.append(f"| {name} | {info['actual_optimizer_steps']} | {info['trained_temporal_steps']} | {verdict} |")
    for variable in scorecard["variables"]:
        lines += ["",f"## {variable}","", "| Variant | RMSE | Tendency RMSE | Lag-1 error | Lag-2 error | 3-day accumulation RMSE |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name in VARIANTS:
            values=[module_value(reports[name],variable,key) for key in
                    ("rmse","tendency_rmse","autocorr_lag1_error","autocorr_lag2_error","acc3d_rmse")]
            lines.append("| "+name+" | "+" | ".join("missing" if value is None else f"{value:.6g}" for value in values)+" |")
    lines += ["", "Full spatial/extreme guardrail values and individual checks are retained in the machine-readable evaluation and scorecard files.",
              "Genuine foundation-pretrained transfer remains separate and unverified by this architectural experiment."]
    return "\n".join(lines)+"\n"


def module_value(report,variable,key):
    return report.get("variables",{}).get(variable,{}).get("paired",{}).get(key)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot")
    parser.add_argument("--out")
    parser.add_argument("--inspect-pid",type=int)
    parser.add_argument("--wait-pid",type=int)
    parser.add_argument("--wait-created-filetime",type=int)
    parser.add_argument("--max-wait-seconds",type=float,default=43200)
    parser.add_argument("--validate-only",action="store_true")
    args=parser.parse_args()
    if args.inspect_pid is not None:
        print(json.dumps(inspect_process(args.inspect_pid),indent=2)); return 0
    if not args.snapshot or not args.out:
        parser.error("--snapshot and --out are required")
    if (args.wait_pid is None) != (args.wait_created_filetime is None):
        parser.error("--wait-pid and --wait-created-filetime must be supplied together")
    if args.max_wait_seconds <= 0:
        parser.error("--max-wait-seconds must be positive")
    snapshot,root=Path(args.snapshot).resolve(),Path(args.out).resolve()
    evidence=verify_snapshot(snapshot)
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest,snapshot)
    module=load_frozen_experiment(snapshot)
    for name in VARIANTS:
        validate_config(module,name,root)
    if args.validate_only:
        print(json.dumps({"valid":True,"snapshot":str(snapshot),"experiment":str(root),**evidence},indent=2)); return 0
    original=json.loads((root/"execution.json").read_text(encoding="utf-8"))
    original_pid=int(original["pid"])
    if args.wait_pid is not None and args.wait_pid != original_pid:
        raise ValueError("Wait PID differs from the original experiment execution record")
    if args.wait_pid is None and inspect_process(original_pid)["exists"]:
        raise RuntimeError("Original PID still exists: supply its exact identity to wait safely")
    lock_path=root/"bounded_recovery.lock"
    descriptor=os.open(lock_path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,json.dumps({"pid":os.getpid(),"created_utc":utc_now()}).encode()); os.close(descriptor)
    token=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record_path=root/f"recovery_execution_{token}.json"
    record={"status":"waiting" if args.wait_pid else "validating", "pid":os.getpid(),
        "started_utc":utc_now(),"snapshot":str(snapshot),"source_verification":evidence,
        "original_execution":str(root/"execution.json"),"driver":str(Path(__file__).resolve()),
        "driver_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reused":[],"resumed":[],"newly_trained":[],
        "correction":"Check global_step for optimizer budgets; spatial_ft legitimately has zero temporal updates"}
    current=None
    try:
        atomic_json(record_path,record)
        if args.wait_pid is not None:
            def heartbeat(elapsed):
                record.update(wait_seconds=round(elapsed,1),updated_utc=utc_now())
                atomic_json(record_path,record)
                print(f"[recovery] waiting for original PID {args.wait_pid}: {elapsed:.0f}s",flush=True)
            record["waited_process"]=wait_for_original(args.wait_pid,args.wait_created_filetime,args.max_wait_seconds,heartbeat)
        # Re-read only after the original writer has exited.
        manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
        validate_manifest(manifest,snapshot)
        verify_snapshot(snapshot)
        atomic_json(root/f"manifest_before_recovery_{token}.json",manifest)
        manifest.setdefault("recovery_records",[]).append(str(record_path))
        record.update(status="running",original_exit_observed_utc=utc_now())
        atomic_json(record_path,record)
        device=module.torch.device(manifest["device"])
        reports={}
        baseline_contract=None
        for name in VARIANTS:
            current=name
            raw=validate_config(module,name,root)
            info=manifest["variants"][name]
            directory=root/name
            config_path=directory/"resolved.yaml"
            if not config_path.exists():
                module.build_variant_config(name,module.variant_overrides()[name],root)
            config=module.ExperimentConfig.from_dict(raw)
            cfg=module.parse_temporal_config(config.temporal)
            checkpoint=None
            if name != "baseline":
                checkpoint=directory/"checkpoints/last.ckpt"
                if checkpoint.is_file():
                    payload,count,trained=checkpoint_progress(module,checkpoint,name,cfg)
                else:
                    payload,count,trained={},0,0
                if count < STEPS or (payload and not payload["training_state"]["epoch_complete"]):
                    if payload:
                        raw["temporal"]["init_from_spatial_checkpoint"]=None
                        raw["temporal"]["resume_from_temporal_checkpoint"]=str(checkpoint)
                        config=module.ExperimentConfig.from_dict(raw)
                        cfg=module.parse_temporal_config(config.temporal)
                        record["resumed"].append({"variant":name,"from_optimizer_step":count})
                    else:
                        record["newly_trained"].append(name)
                    info.update(status="running",phase="training",pid=os.getpid(),configuration=str(config_path),
                                actual_optimizer_steps=count,trained_temporal_steps=trained,trained=True)
                    atomic_json(root/"manifest.json",manifest); atomic_json(record_path,record)
                    started=time.monotonic()
                    if device.type == "cuda":
                        module.torch.cuda.reset_peak_memory_stats(device)
                    summary=module.train_temporal_model(config,cfg,device=device,output_dir=str(directory/"checkpoints"),
                         max_steps_per_epoch=STEPS,max_val_steps=VAL_STEPS,num_epochs=EPOCHS,
                         batch_size=1,num_workers=0,verbose=True)
                    info["train_seconds"]=float(info.get("train_seconds",0))+time.monotonic()-started
                    info["recovery_training_peak_memory_allocated_bytes"]=(
                        int(module.torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None)
                    info["training_summary"]=summary
                    info["history"]=summary["history"]
                    if int(summary["global_step"]) != STEPS:
                        raise RuntimeError(f"{name}: optimizer updates did not reach exactly {STEPS}")
                    payload,count,trained=checkpoint_progress(module,checkpoint,name,cfg)
                else:
                    record["reused"].append({"variant":name,"optimizer_steps":count,"checkpoint":str(checkpoint)})
                    info.setdefault("history",payload["training_state"]["counters"].get("history",[]))
                counters=payload["training_state"]["counters"]
                info["training_accounting"]={key:counters.get(key) for key in
                    ("global_step","trained_temporal_steps","microbatches","backward_passes",
                     "accumulation_cycles","target_samples","skipped_nonfinite_updates","skipped_amp_overflow_updates")}
                info["backbone_evaluations"]=sum(epoch.get("train",{}).get("backbone_evaluations",0)
                                                for epoch in counters.get("history",[]))
                info.update(actual_optimizer_steps=count,trained_temporal_steps=trained,trained=True,
                            checkpoint=str(checkpoint),configuration=str(config_path))
                # The single bounded epoch selects its final checkpoint. If a
                # best checkpoint exists, require the identical complete budget.
                best=directory/"checkpoints/best.ckpt"
                if best.exists():
                    _,best_steps,_=checkpoint_progress(module,best,name,cfg)
                    if best_steps != STEPS:
                        raise ValueError(f"{name}: best checkpoint is not from the complete bounded epoch")
                    checkpoint=best
                # Strict model contract validation occurs in the frozen builder.
                raw=validate_config(module,name,root)
                raw["temporal"]["init_from_spatial_checkpoint"]=None
                raw["temporal"]["resume_from_temporal_checkpoint"]=str(checkpoint)
                config=module.ExperimentConfig.from_dict(raw)
                cfg=module.parse_temporal_config(config.temporal)
                info["checkpoint"]=str(checkpoint)
                del payload
            else:
                info.update(actual_optimizer_steps=0,trained_temporal_steps=0,trained=False)
            pred_path=Path(info.get("predictions",directory/"predictions.npz"))
            eval_path=Path(info.get("evaluation",directory/"evaluation.json"))
            if info.get("status") == "completed" and pred_path.is_file() and eval_path.is_file():
                reports[name]=json.loads(eval_path.read_text(encoding="utf-8"))
                record["reused"].append({"variant":name,"predictions":str(pred_path)})
            else:
                info.update(status="running",phase="inference",pid=os.getpid())
                atomic_json(root/"manifest.json",manifest)
                reports[name],inferred=infer_and_evaluate(module,config,cfg,root,name,device)
                info.update(inferred)
                pred_path=Path(info["predictions"])
            contract=prediction_contract(module,pred_path)
            if name == "baseline":
                baseline_contract=contract
            elif contract != baseline_contract:
                raise ValueError(f"{name}: scored dates, targets, masks or output geometry differ from baseline")
            info.update(status="completed",phase="completed",prediction_contract=contract,
                        completed_utc=utc_now(),actual_optimizer_steps=0 if name == "baseline" else STEPS)
            atomic_json(root/"manifest.json",manifest); atomic_json(record_path,record)
        current=None
        scorecard=score_complete(module,manifest,reports)
        score_path=root/f"scorecard_recovery_{token}.json"
        atomic_json(score_path,scorecard)
        manifest["scorecard_path"]=str(score_path)
        atomic_json(root/"manifest.json",manifest)
        (root/f"report_recovery_{token}.md").write_text(render_report(manifest,scorecard,reports),encoding="utf-8")
        # Plotting script requires canonical filenames; use a separate report-view
        # directory with hard links, preserving all inherited files in place.
        view=root/f"report_view_recovery_{token}"
        view.mkdir()
        atomic_json(view/"scorecard.json",scorecard); atomic_json(view/"manifest.json",manifest)
        for name in VARIANTS:
            (view/name).mkdir()
            info=manifest["variants"][name]
            os.link(Path(info["predictions"]),view/name/"predictions.npz")
            os.link(Path(info["evaluation"]),view/name/"evaluation.json")
        script=snapshot/"examples/CORDEX_ML/cordex_temporal_native_pair_plots.py"
        previous_argv=sys.argv
        try:
            sys.argv=[str(script),"--experiment",str(view),"--out",str(view/"figures")]
            try:
                runpy.run_path(str(script),run_name="__main__")
            except SystemExit as exc:
                if exc.code not in (0,None):
                    raise
        finally:
            sys.argv=previous_argv
        record.update(status="completed",exit_code=0,scorecard=str(score_path),report_view=str(view))
        return 0
    except BaseException as exc:
        record.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",
                      exit_code=1,error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
        if current is not None and manifest["variants"].get(current,{}).get("status") == "running":
            manifest["variants"][current].update(status=record["status"],error=record["error"])
            atomic_json(root/"manifest.json",manifest)
        traceback.print_exc()
        return 1
    finally:
        record["finished_utc"]=utc_now()
        atomic_json(record_path,record)
        lock_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
