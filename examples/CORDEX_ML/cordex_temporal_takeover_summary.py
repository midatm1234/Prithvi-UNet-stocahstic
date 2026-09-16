"""Mechanically summarize saved evidence and current bounded-run process state."""
from __future__ import annotations
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def reconcile_variant(name, entry, active):
    """A stale running label is evidence of interruption when its process exited."""
    result = dict(entry)
    result["reported_manifest_status"] = entry.get("status", "not started")
    result["status"] = result["reported_manifest_status"]
    if result["status"] == "running" and entry.get("pid") not in {p["pid"] for p in active}:
        result["status"] = "interrupted"
        result["execution_note"] = "Manifest's owning process is no longer active."
        if active:
            result["execution_note"] += " A continuation process is active; consult its recovery record."
    if result["status"] == "completed":
        required = ("predictions", "evaluation") if name == "baseline" else ("checkpoint", "predictions", "evaluation")
        missing = [key for key in required if not entry.get(key) or not Path(entry[key]).is_file()]
        if missing:
            result["status"] = "interrupted"
            result["execution_note"] = "Completed label lacks required artifacts: " + ", ".join(missing)
    return result


def comparison_lines(status, card):
    lines = ["", "## Saved comparison results", "",
             "Only completed variants with saved evaluations appear below; missing variants remain unscored."]
    for variable in ("pr", "tasmax"):
        units = "mm/day" if variable == "pr" else "K"
        lines += ["", f"### {variable} ({units})", "",
                  "| variant | RMSE | tendency RMSE | lag-1 error | lag-2 error | 3-day accumulation RMSE |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, entry in status["variants"].items():
            if entry["status"] != "completed" or not entry.get("evaluation"):
                continue
            details = read(Path(entry["evaluation"])).get("variables", {}).get(variable, {}).get("paired", {})
            values = [details.get(key) for key in ("rmse", "tendency_rmse", "autocorr_lag1_error", "autocorr_lag2_error", "acc3d_rmse")]
            lines.append("| " + name + " | " + " | ".join("missing" if v is None else f"{v:.6g}" for v in values) + " |")
    if status["scientific_status"] != "incomplete":
        lines += ["", "### Complete comparison verdicts", ""]
        for name in ("native_pair", "native_pair_pretext"):
            entry = card.get("variants", {}).get(name, {})
            verdict = "accepted" if entry.get("scientific_acceptance") is True else "not accepted"
            lines.append(f"- **{name}: {verdict}.**")
            for variable, detail in entry.get("variables", {}).items():
                failed = [label for group in ("primary", "guardrail") for label, result in detail.get(group, {}).items()
                          if result.get("status") != "pass"]
                if failed:
                    lines.append(f"  - {variable}: " + "; ".join(failed) + ".")
            if not entry.get("beats_all_controls"):
                lines.append("  - Required control comparisons did not all pass; see the versioned scorecard.")
    return lines


def main():
    evidence = ROOT / "artifacts/temporal_takeover/20260915T113310"
    run = ROOT / "examples/CORDEX_ML/runs_temporal/experiment_native_pair_v2_20260915"
    manifest = read(run / "manifest.json")
    proof = read(ROOT / "artifacts/temporal_takeover/optimizer_proof_v1/optimizer_proof.json")
    inventory = read(evidence / "inventory.json")
    import psutil
    active = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            argv = process.info["cmdline"] or []
            if "python" in (process.info["name"] or "").lower() and any(
                "cordex_temporal_bounded_launch.py" in value or "cordex_temporal_continue_bounded.py" in value or "continue_bounded_driver_v1.py" in value
                for value in argv
            ):
                active.append(process.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    status = {"as_of_utc": datetime.now(timezone.utc).isoformat(),
              "scientific_status": "incomplete", "active_processes": active,
              "source_snapshot": "artifacts/temporal_takeover/bounded_source_v2_20260915",
              "experiment": str(run), "variants": {},
              "original_execution": read(run / "execution.json"),
              "recovery_executions": [read(path) for path in sorted(run.glob("recovery_execution_*.json"))],
              "final_report_execution": read(ROOT / "artifacts/temporal_takeover/bounded_final_reports_v1_20260915/execution.json")}
    for name, original_entry in manifest.get("variants", {}).items():
        entry = reconcile_variant(name, original_entry, active)
        checkpoint = run / name / "checkpoints/last.ckpt"
        item = {"status": entry.get("status", "not started"),
                "configuration": entry.get("configuration"),
                "actual_optimizer_steps": entry.get("actual_optimizer_steps"),
                "trained_temporal_steps": entry.get("trained_temporal_steps"),
                "checkpoint": entry.get("checkpoint"),
                "predictions": entry.get("predictions"), "evaluation": entry.get("evaluation"),
                "phase": entry.get("phase"), "pid": entry.get("pid"),
                "reported_manifest_status": entry["reported_manifest_status"],
                "execution_note": entry.get("execution_note"),
                "training_accounting": entry.get("training_accounting"),
                "backbone_evaluations": entry.get("backbone_evaluations"),
                "train_seconds": entry.get("train_seconds")}
        if name == "baseline":
            item["actual_optimizer_steps"] = 0
        if checkpoint.is_file():
            # Read metadata promptly and release mmap before the writer's next atomic save.
            import torch
            try:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
                item["last_saved_optimizer_steps"] = payload.get("global_step")
                item["last_saved_temporal_steps"] = (payload.get("temporal") or {}).get("trained_temporal_steps")
                item["checkpoint"] = str(checkpoint)
                item["checkpoint_counter_scope"] = "durable lower bound while training is active"
                del payload
                gc.collect()
            except (OSError, RuntimeError) as exc:
                item["checkpoint_read_note"] = str(exc)
        status["variants"][name] = item
    card_path = Path(manifest["scorecard_path"]) if manifest.get("scorecard_path") else run / "scorecard.json"
    card = read(card_path)
    if card.get("complete") and len(status["variants"]) == 6 and all(
        value["status"] == "completed" for value in status["variants"].values()
    ):
        status["scientific_status"] = "accepted" if any(
            value.get("scientific_acceptance") is True for value in card.get("variants", {}).values()
        ) else "not accepted"
        status["scorecard"] = str(card_path)
    latest = ROOT / "artifacts/temporal_takeover/latest_status.json"
    latest.write_text(json.dumps(status, indent=2), encoding="utf-8")
    lines = ["# Temporal takeover results", "", f"Status captured: {status['as_of_utc']}", "",
             f"**Scientific status: {status['scientific_status']}.** The bounded comparison is separate",
             "from optimizer engineering checks and the pretrained-transfer branch.", "",
             "## Execution and preservation", "",
             "Branch `Prithvi-UNet_temporal_model`, HEAD `2f237147f24f29b3146701b6d4ccb1b1fcbf1c21`.",
             "Inherited changes/output inventory preserved under `artifacts/temporal_takeover/20260915T113310`.",
             "No old experiment output was deleted or overwritten; no push was made.", "",
             "| variant | inherited status | current status | saved optimizer updates |",
             "|---|---|---|---|"]
    for name, entry in status["variants"].items():
        old = inventory.get("variants", {}).get(name, {})
        steps = entry.get("last_saved_optimizer_steps", entry.get("actual_optimizer_steps"))
        lines.append(f"| {name} | {old.get('status', 'unknown')} | {entry['status']} | {steps if steps is not None else 'not yet saved'} |")
    lines += ["", "Active process IDs: " + ", ".join(str(p["pid"]) for p in active) + ".",
              "Current process commands, paths, and checkpoint counters: `artifacts/temporal_takeover/latest_status.json`.",
              "A running checkpoint counter is a durable lower bound, not an estimate of live updates.", "",
              "The frozen initial driver checks the wrong counter for spatial_ft. Its valid checkpoint",
              "is recovered by the separately versioned continuation driver using the same frozen model",
              "sources and original 600-update, 60-validation-window, one-epoch protocol.", "",
              "## Actual learning through the real trainer", "",
              "| variant | actual updates including resume | max history weight change, first step | save/load max difference | chunk max difference |",
              "|---|---|---|---|---|"]
    for name, entry in proof.get("variants", {}).items():
        lines.append(f"| {name} | {entry['updates'][-1]['global_step']} | {entry['updates'][0]['history_max_parameter_delta']:.7g} | {entry['save_load_max_difference']} | {entry['chunk_consistency']['max_abs_diff_overall']} |")
    baseline = read(run / "baseline/evaluation.json")
    lines += ["", "### Saved SA reference metrics (1981–1983)", "",
              "| variable | units | RMSE | tendency RMSE | lag-1 error | 3-day accumulation RMSE |",
              "|---|---|---|---|---|---|"]
    for variable, details in baseline.get("variables", {}).items():
        paired = details.get("paired", {})
        values = [paired.get(key) for key in ("rmse", "tendency_rmse", "autocorr_lag1_error", "acc3d_rmse")]
        units = "mm/day" if variable == "pr" else "K"
        lines.append("| " + variable + " | " + units + " | " + " | ".join("missing" if value is None else f"{value:.6g}" for value in values) + " |")
    lines += ["", "The full 1,095-date baseline archive matched the inherited predictions, targets,",
              "masks, dates, variables and seams exactly; max prediction difference was zero."]
    lines += ["", "Historical parameters occur exactly once at LR 2e-4. Frozen encoder/backbone and",
              "all eight physical normalization tensors stayed unchanged. History affects predictions",
              "after optimization without manual amplification. Both auxiliary losses reach the shared",
              "historical projection; the positive-lead transition exercises the lead-time slope.",
              "Auxiliary heads are optimized and checkpointed. Frozen transformer weights do not update.",
              "Zero differences above are measurements on this host; portable tests also use explicit tolerances.",
              "Evidence: `artifacts/temporal_takeover/optimizer_proof_v1/optimizer_proof.json`.", "",
              "## Physical units, splits and NARR", "",
              "SA precipitation matches file values converted exactly once to mm/day; applying the",
              "converter twice is an identity after the first conversion. Temperature stays Kelvin.",
              "Timestamp joins and discontinuity-aware windows preserve actual daily means and gaps.",
              "Phase-1 saw temporal validation years 1977–1980 and the inherited scalers include those",
              "years. Validation is not independent from Phase-1; shared exposure does not remove its effect.",
              "NARR has 32 inputs, ppt/tmax/tmin outputs, zero separate static channels and a hurdle head.",
              "Actual NARR assets remain absent; strict NetCDF/tile/mask tests use synthetic fixtures.",
              "See `data_contracts.json` and `docs/temporal_native_correctness_takeover.md`.", "",
              "## Provenance and separate transfer", "",
              "Current 2560-to-1024 loading matches no backbone learned parameters. Historical Phase-1",
              "foundation provenance remains unresolved; a related notebook records loading a smaller",
              "checkpoint. Weight histograms do not settle provenance or whether time weights learned.",
              "The separate transfer path loads the official rollout checkpoint's first local/global",
              "transformer pair and time maps, with new regional input/latent adapters. It explicitly",
              "adapts reduced daily predictors and regional token geometry. It does not claim full-state",
              "MERRA-2 compatibility or full-model transfer. Negative-time v1 audits are invalidated;",
              "use corrected v2 evidence in `docs/temporal_pretrained_transfer_takeover.md`.", "",
              "## Scores, figures and runnable workflow", "",
              "Original ACCEPTANCE thresholds, MUST_BEAT and extra no-history/pretext controls remain",
              "unchanged. Corrected scoring handles dotted metric names, missing/NaN metrics, near-zero",
              "baselines, missing controls and invalidated/partial experiments. Archived ConvGRU/Mamba",
              "verdicts remain not accepted. Versioned additional scorecards are under the takeover evidence.",
              "Eleven baseline figures: `artifacts/temporal_takeover/20260915T113310/inherited_baseline_figures`.",
              "Geographic full/boundary/interior/southeast metrics and mean-bias maps: `regional_baseline/`.",
              "These figures describe the saved baseline, not a completed native-pair comparison.", "",
              "Selected workflow: `examples/CORDEX_ML/notebooks/Temporal_selected_training_v2.ipynb`.",
              "CLI: `train --resume CHECKPOINT`, repeated `--set dotted.path=YAML_value`, then `infer`/`evaluate`.",
              "Four optional extended YAMLs exist for both cases and both native variants; none launched.",
              "Staged unfreezing retains existing Adam moments and never unfreezes normalization.",
              "All four stochastic-refinement interfaces remain; their scientific recalibration is separate.", "",
              "## Continuation", "",
              "Consult `docs/temporal_codex_handoff.md` before acting. Do not launch duplicate GPU work.",
              "Refresh this report with `python examples/CORDEX_ML/cordex_temporal_takeover_summary.py`.",
              "Final scientific comparison, figures and verdict require all planned variant artifacts.",
              "A separate CPU-only report process waits for the bounded controller, then generates",
              "regional maps/metrics for all six variants under",
              "`artifacts/temporal_takeover/bounded_final_reports_v1_20260915/`. Its execution.json",
              "records completion or the precise blocker; it never starts training."]
    lines += comparison_lines(status, card)
    (ROOT / "docs/temporal_takeover_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(latest)


if __name__ == "__main__":
    main()
