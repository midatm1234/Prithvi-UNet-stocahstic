"""Read-only inherited experiment and real-data contracts audit (CPU)."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import os
import subprocess
import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
import xarray as xr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/temporal_takeover/20260915T113310")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from granitewxc.temporal.entrypoints import _load
    from granitewxc.temporal.sources import build_frame_source
    from granitewxc.temporal.training import build_sequence_dataloaders
    from granitewxc.temporal.calendar import time_key
    config_path = "examples/CORDEX_ML/SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_prithvi_native_pair.yaml"
    config, cfg = _load(config_path)
    run = Path("examples/CORDEX_ML/runs_temporal/experiment_native_pair")
    manifest = json.loads((run / "manifest.json").read_text())
    processes = []
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time", "cwd"]):
            try:
                if "python" in (proc.info["name"] or "").lower():
                    processes.append(proc.info)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
    except ImportError:
        pass
    relevant = [p for p in processes if any("cordex_temporal_experiment.py" in a for a in p["cmdline"] or [])]
    inventory = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                 "inherited_root": str(run), "active_experiment_processes": relevant,
                 "variants": {}}
    for name in ("baseline", "spatial_ft", "time_only", "native_pair", "native_pair_nohistory", "native_pair_pretext"):
        directory = run / name
        saved = manifest["variants"].get(name, {})
        checkpoint = directory / "checkpoints" / "last.ckpt"
        pred, evaluation, resolved = [directory / p for p in ("predictions.npz", "evaluation.json", "resolved.yaml")]
        status = "completed" if pred.exists() and evaluation.exists() else "interrupted" if resolved.exists() else "not started"
        inventory["variants"][name] = {
            "status": status, "configuration": str(resolved) if resolved.exists() else None,
            "checkpoint": saved.get("checkpoint") if name == "baseline" else str(checkpoint) if checkpoint.exists() else None,
            "actual_optimizer_steps": 0 if name == "baseline" or status == "not started" else None,
            "predictions": str(pred) if pred.exists() else None,
            "evaluation": str(evaluation) if evaluation.exists() else None,
            "evidence": [str(p) for p in directory.rglob("*") if p.is_file()] if directory.exists() else [],
            "recovery": "preserve completed reference" if name == "baseline" else
                        "corrected fresh run required; no compatible saved training state",
            "scientific_status": "reference only" if name == "baseline" else "incomplete",
        }
    smoke = Path("artifacts/pretext_smoke/checkpoints/last.ckpt")
    if smoke.exists():
        payload = torch.load(smoke, map_location="cpu", weights_only=False, mmap=True)
        log = json.loads(smoke.with_name("training_log.jsonl").read_text().splitlines()[-1])
        inventory["inherited_pretext_smoke"] = {
            "status": "invalidated", "reason": "3 microbatches < 16 accumulation steps; remainder dropped",
            "microbatches": log["train"]["n_batches"], "backward_passes": log["train"]["n_batches"],
            "optimizer_updates": log["global_step"], "mixed_precision": "FP32; no overflow scaler used",
            "checkpoint_keys": list(payload), "checkpoint": str(smoke),
        }
        state = payload.get("model", payload.get("model_state_dict", payload.get("state_dict", {})))
        baseline = torch.load(cfg.init_from_spatial_checkpoint, map_location="cpu", weights_only=False, mmap=True)
        baseline_state = baseline.get("model", baseline.get("model_state_dict", baseline.get("state_dict", {})))
        same = [k for k in state if k in baseline_state and hasattr(state[k], "shape") and state[k].shape == baseline_state[k].shape]
        changed = [k for k in same if not torch.equal(state[k], baseline_state[k])]
        inventory["inherited_pretext_smoke"].update(comparable_spatial_tensors=len(same), changed_spatial_tensors=changed)
        del payload, baseline
    (out / "inventory.json").write_text(json.dumps(inventory, indent=2, default=str))

    audit = {"config": config_path, "interpreter": sys.executable, "packages": {}, "SA": {}, "NARR": {}}
    for name in ("torch", "numpy", "xarray", "pytest", "PrithviWxC", "psutil", "transformers"):
        try:
            audit["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            audit["packages"][name] = "not installed (not required by this workflow)"
    audit["cuda"] = {"available": torch.cuda.is_available(), "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    for split in ("train", "test"):
        source = build_frame_source(config, split)
        ref = source.frames()[0]
        sample = source.load_frame(ref, slice(0, 128), slice(0, 128))
        data = source._dataset
        path = data.target_paths[ref.file_index]
        with xr.open_dataset(path) as raw:
            metadata = {v: dict(raw[v].attrs) for v in config.data.output_vars}
            target = raw["pr"].isel({data.time_dim: ref.target_time_index})
            values = np.asarray(target.values, dtype=np.float32)
            compact = str(target.attrs.get("units", "")).lower().replace(" ", "").replace("**", "")
            factor = 86400.0 if compact in {"kgm-2s-1", "kgm^-2s^-1", "mm/s", "mms-1"} else 1.0
            actual = sample["y"][0].numpy()
            assert np.allclose(actual, values * factor, rtol=2e-6, atol=1e-6, equal_nan=True), (split, compact, factor)
            converted = data._convert_target_units(target, variable_name="pr")
            twice = data._convert_target_units(converted, variable_name="pr")
            assert np.array_equal(converted.values, twice.values, equal_nan=True)
        with xr.open_dataset(data.predictor_paths[0]) as raw:
            predictor_meta = {v: dict(raw[v].attrs) for v in data.predictor_vars}
        audit["SA"][split] = {"target_file": str(path), "target_metadata": metadata,
             "decoded_target_units": source.target_units, "conversion_factor": factor,
             "conversion_matches_file": True, "second_conversion_identity": True,
             "predictor_metadata": predictor_meta, "predictor_channels_with_static": sample["x"].shape[0],
             "timestamp": time_key(ref.timestamp), "target_time_index": ref.target_time_index}
    loaders = build_sequence_dataloaders(config, cfg, splits=("train", "validation", "test"), verbose=False)
    audit["SA"]["split_windows"] = {k: v.dataset.describe() for k, v in loaders.items()}
    audit["SA"]["split_limitations"] = {
        "temporal_splits": getattr(config, "temporal_splits", None),
        "phase1_validation_overlap": "Historical Phase1 notebook/config used 1961-1980 plus 2080-2099; temporal validation 1977-1980 is not independently held out from Phase1",
        "scalers": "Inherited Phase1 scalers retained unchanged; their earlier overlap prevents clean validation independence",
        "test": "1981-1983 disjoint from listed Phase1 downscaling dates; foundation pretraining overlap depends on unresolved historical provenance",
    }
    narr_path = "examples/NARR_PRISM/NARR_PRISM_subdomain_temporal_prithvi_native_pair_pretext.yaml"
    narr, narr_cfg = _load(narr_path)
    raw = yaml.safe_load(Path(narr_path).read_text())
    audit["NARR"] = {"case_name": narr.case_name, "input_vars": raw["data"]["input_vars"],
                     "input_channels": len(raw["data"]["input_vars"]), "output_vars": list(narr.data.output_vars),
                     "static_channels": narr.model.num_static_channels, "precip_model": narr.precip_model}
    for label, path in {"checkpoint": narr_cfg.init_from_spatial_checkpoint,
                        "preprocessed": narr.data.preprocessed_dir,
                        "predictors": narr.data.predictor_dir, "targets": narr.data.target_dir}.items():
        p = Path(path)
        audit["NARR"][label] = {"path": str(path), "exists": p.exists(), "is_directory": p.is_dir(),
                                "symlink_stub": p.read_text() if p.is_file() and p.stat().st_size < 500 else None}
    (out / "data_contracts.json").write_text(json.dumps(audit, indent=2, default=str))
    print(json.dumps({"inventory": str(out / "inventory.json"), "data_contracts": str(out / "data_contracts.json")}, indent=2))


if __name__ == "__main__":
    main()
