"""Instrument the real trainer on real SA data; require actual adapter updates.

This is an engineering check, not a scientific acceptance experiment. It uses
the selected case loss/freeze policy, one update, then a stateful resume update.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from granitewxc.utils.config import ExperimentConfig
from granitewxc.temporal.config import parse_temporal_config
from granitewxc.temporal.model import TemporalSequenceModel
from granitewxc.temporal.training import (
    train_temporal_model, build_sequence_dataloaders, build_temporal_model,
    resolve_time_feature_dim,
)
from granitewxc.temporal.native_pair import compute_pretext_losses
from granitewxc.temporal.inference import verify_chunk_consistency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", default=["native_pair", "native_pair_pretext"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    evidence = {"scientific_status": "engineering optimizer proof only", "variants": {}}
    for variant in args.variants:
        path = ROOT / f"examples/CORDEX_ML/SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_prithvi_{variant}.yaml"
        # filenames use native_pair rather than an additional prithvi_native prefix
        raw = yaml.safe_load(path.read_text())
        raw["gradient_accumulation_steps"] = 1
        raw["training"]["gradient_accumulation_steps"] = 1
        config = ExperimentConfig.from_dict(raw)
        cfg = parse_temporal_config(config.temporal)
        run = out / variant
        run.mkdir()
        (run / "resolved.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
        loader = build_sequence_dataloaders(config, cfg, splits=("validation",), num_workers=0, verbose=False)["validation"]
        td, tn = resolve_time_feature_dim(loader)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in next(iter(loader)).items()}
        item = {"config": str(path), "updates": [], "auxiliary_gradient_checks": {}}
        cache = {}

        def before(model, optimizer, state):
            history_name = "temporal_adapter.history_projection.weight"
            named = dict(model.named_parameters())
            historical = named[history_name]
            memberships = [(g.get("name"), g["lr"]) for g in optimizer.param_groups
                           for p in g["params"] if p is historical]
            assert historical.requires_grad and len(memberships) == 1
            assert memberships[0][1] == cfg.freeze.lr_temporal
            immutable = {n: p.detach().cpu().clone() for n, p in named.items() if not p.requires_grad}
            normalizers = [n for n in named if "scalers_" in n]
            assert normalizers and all(not named[n].requires_grad for n in normalizers)
            optimizer_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
            assert all(id(named[n]) not in optimizer_ids for n in normalizers)
            cache.update(immutable=immutable, history=historical.detach().clone())
            item["history_optimizer_membership"] = memberships
            item["immutable_tensors"] = list(immutable)
            item["normalizers"] = normalizers
            item.setdefault("optimizer_state_at_start", []).append(len(optimizer.state))
            runner = TemporalSequenceModel(model, cfg, adapter=model.temporal_adapter)
            model.eval()
            with torch.no_grad():
                cache["initial_prediction"] = runner(batch).predictions.detach().cpu()
            if state.global_step == 0 and cfg.native_pair.pretext.any_enabled:
                model.train()
                terms, forwards = compute_pretext_losses(model, model.temporal_adapter, batch,
                    batch["x"].shape[1] - 1, cfg, generator=torch.Generator(device=device).manual_seed(999))
                lead = model.temporal_adapter.time_conditioning.lead_time_embedding.weight
                for name, loss in terms.items():
                    grads = torch.autograd.grad(loss, [historical, lead], retain_graph=True, allow_unused=True)
                    values = [0.0 if g is None else float(g.detach().abs().sum()) for g in grads]
                    assert values[0] > 0, (name, values)
                    if "transition" in name:
                        assert values[1] > 0, values
                    item["auxiliary_gradient_checks"][name] = {
                        "weighted_loss": float(loss.detach()), "history_gradient_l1": values[0],
                        "lead_slope_gradient_l1": values[1], "backbone_evaluations": forwards,
                        "updated_shared_component": history_name, "backbone_weights_frozen": True}
                del terms, loss, grads
            model.zero_grad(set_to_none=True)
            model.train()

        def stepped(model, optimizer, state):
            named = dict(model.named_parameters())
            history = named["temporal_adapter.history_projection.weight"]
            changed = float((history.detach() - cache["history"]).abs().max())
            assert changed > 0
            assert history.grad is not None and float(history.grad.abs().sum()) > 0
            drift = [n for n, original in cache["immutable"].items() if not torch.equal(original, named[n].detach().cpu())]
            assert not drift, drift
            runner = TemporalSequenceModel(model, cfg, adapter=model.temporal_adapter)
            model.eval()
            perturbed = dict(batch)
            perturbed["x"] = batch["x"].clone()
            perturbed["x"][:, -2] = batch["x"][:, -3]
            with torch.no_grad():
                pred = runner(batch).predictions
                alt = runner(perturbed).predictions
                history_effect = float((pred[:, -1] - alt[:, -1]).abs().max())
            assert history_effect > 0, "Trained history must affect predictions without amplified gates"
            item["updates"].append({"global_step": state.global_step,
                "history_max_parameter_delta": changed, "history_prediction_max_delta": history_effect,
                "history_gradient_l1": float(history.grad.abs().sum()), "immutable_drift": drift,
                "optimizer_history_step": int(optimizer.state[history]["step"].item()),
                "auxiliary_head_gradients": {n: float(p.grad.abs().sum()) if p.grad is not None else None
                    for n, p in named.items() if "_head." in n and n.startswith("temporal_adapter.")}})
            cache["trained_prediction"] = pred.detach().cpu()
            model.train()

        started = time.time()
        summary = train_temporal_model(config, cfg, device=device, output_dir=run / "checkpoints",
            max_steps_per_epoch=1, max_val_steps=1, num_epochs=1, batch_size=1, num_workers=0,
            before_training_callback=before, optimizer_step_callback=stepped)
        assert summary["trained_temporal_steps"] == 1
        resume_cfg = replace(cfg, init_from_spatial_checkpoint=None,
                             resume_from_temporal_checkpoint=str(run / "checkpoints/last.ckpt"))
        resume_summary = train_temporal_model(config, resume_cfg, device=device,
            output_dir=run / "resumed", max_steps_per_epoch=1, max_val_steps=1, num_epochs=2,
            batch_size=1, num_workers=0, before_training_callback=before, optimizer_step_callback=stepped)
        assert resume_summary["trained_temporal_steps"] == 2
        assert item["optimizer_state_at_start"][1] > 0
        cfg_infer = replace(cfg, init_from_spatial_checkpoint=None,
                            resume_from_temporal_checkpoint=str(run / "resumed/last.ckpt"))
        runner, migration, _ = build_temporal_model(config, cfg_infer, time_feature_dim=td,
            time_feature_names=tn, device=device, verbose=False)
        runner.eval()
        with torch.no_grad():
            loaded = runner(batch).predictions.detach().cpu()
            no_targets = dict(batch)
            no_targets.pop("y")
            if "__target_valid_mask" in no_targets:
                no_targets.pop("__target_valid_mask")
            independent = runner(no_targets).predictions.detach().cpu()
        torch.testing.assert_close(loaded, cache["trained_prediction"], rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(independent, loaded, rtol=0, atol=0)
        item.update(training_summary=summary, resume_summary=resume_summary,
            save_load_max_difference=float((loaded - cache["trained_prediction"]).abs().max()),
            target_free_inference=True, seconds=time.time() - started,
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
        item["chunk_consistency"] = verify_chunk_consistency(runner, config, cfg_infer,
            split="validation", date_start=None, date_end=None, n_frames=8,
            chunk_lengths=(8, 3), device=device)
        evidence["variants"][variant] = item
        (out / "optimizer_proof.json").write_text(json.dumps(evidence, indent=2, default=str))
        print(json.dumps({variant: item["updates"], "seconds": item["seconds"]}, indent=2), flush=True)
        del runner, batch, cache
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
