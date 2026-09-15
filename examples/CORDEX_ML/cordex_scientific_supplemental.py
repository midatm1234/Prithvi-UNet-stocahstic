#!/usr/bin/env python
"""Register isolated conditioning/gate controls and retain all outcomes.

The original plan, queue and execution receipts are never mutated. Supplemental
jobs use the existing native/direct screen interface and separately change
predictor normalization or Transformer gate calibration. Combined evidence is a frozen selection input, not
a new execution queue and not evidence of scientific acceptance.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.CORDEX_ML import cordex_scientific_workflow as workflow
from granitewxc.refinement.scientific_data import canonical_sha256, file_sha256

VERSION = "registered_conditioning_and_gate_controls_v1"
COMBINED_VERSION = "immutable_combined_development_evidence_v1"
TERMINAL = {"COMPLETED_REGISTERED_STOP", "CONDITION_NOT_TRIGGERED", "FAIL_NUMERICAL_EXECUTION"}


def _queue(path):
    value = workflow.read(path)
    if canonical_sha256({k: v for k, v in value.items() if k != "sha256"}) != value.get("sha256"):
        raise ValueError("Development queue hash mismatch.")
    ids = [job["id"] for job in value["jobs"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate registered job IDs.")
    return value


def _parent(original_path):
    original_path = Path(original_path).resolve()
    original = _queue(original_path)
    if original.get("supplemental_registration") or original.get("combined_evidence_version"):
        raise ValueError("Expected the preserved original development queue.")
    root = Path(original["run_root"]).resolve()
    plan_path = root / "registered_experiment_plan.json"
    plan = workflow.read(plan_path)
    digest = hashlib.sha256(plan_path.read_text(encoding="utf8").encode()).hexdigest()
    if (digest != plan_path.with_suffix(".sha256").read_text().strip()
            or digest != original["registered_plan_sha256"]
            or plan["acceptance_contract_sha256"] != original["acceptance_contract_sha256"]):
        raise ValueError("Frozen parent plan or acceptance hash mismatch.")
    return original_path, original, root, plan


def _definition(original_path):
    original_path, original, root, plan = _parent(original_path)
    artifact_root = Path(original.get("artifact_root", root)).resolve()
    jobs, comparisons = [], []
    for head in workflow.HEADS:
        matches = [job for job in original["jobs"] if job["id"] == "native/" + head]
        if len(matches) != 1:
            raise ValueError("Original native control must cover each of the four heads exactly once.")
        source = matches[0]
        if (source["head"] != head or source["recipe"] != "native"
                or source["formulation"] != "direct" or source["stage"] != "screen"
                or source["seed"] != plan["screening_seed"]
                or source.get("normalize_predictors", False) is not False
                or source.get("condition") or source.get("mean_checkpoint")):
            raise ValueError("Original native/direct control settings differ from the registered comparison.")
        job = copy.deepcopy(source)
        job.update(id="normalized_native/" + head,
                   output=str(artifact_root / "workflow" / "runs" / "normalized_native" / head),
                   normalize_predictors=True)
        jobs.append(job)
        config_path = (ROOT / source["config"]).resolve()
        comparisons.append({"factor": "predictor_conditioning_normalization", "head": head, "original_job": source["id"],
                            "original_job_sha256": canonical_sha256(source),
                            "supplemental_job": job["id"],
                            "configuration_sha256": file_sha256(config_path)})
    for head in (name for name in workflow.HEADS if name.endswith("transformer")):
        matches = [job for job in original["jobs"] if job["id"] == "native_production_gate/" + head]
        if len(matches) != 1:
            raise ValueError("Both Transformer native production-gate controls are required.")
        source = matches[0]
        if (source["head"] != head or source["recipe"] != "native_production_gate"
                or source["formulation"] != "direct" or source["stage"] != "screen"
                or source["seed"] != plan["screening_seed"]
                or source.get("normalize_predictors", False) is not False or source.get("condition")):
            raise ValueError("Original Transformer gate control differs from the registered comparison.")
        job = copy.deepcopy(source)
        job.update(id="gate_calibration_loss/" + head, recipe="gate_calibration_loss",
                   output=str(artifact_root / "workflow" / "runs" / "gate_calibration_loss" / head))
        jobs.append(job)
        comparisons.append({"factor": "isolated_gate_calibration_loss", "head": head,
                            "original_job": source["id"], "original_job_sha256": canonical_sha256(source),
                            "supplemental_job": job["id"], "gate_policy": "production_trainable",
                            "coefficient": 0.25, "all_other_auxiliary_coefficients": 0.0,
                            "process_loss_coefficient": 1.0, "normalize_predictors": False,
                            "configuration_sha256": file_sha256((ROOT / source["config"]).resolve())})
    for comparison in comparisons:
        source = next(job for job in original["jobs"] if job["id"] == comparison["original_job"])
        config = yaml.safe_load((ROOT / source["config"]).read_text(encoding="utf8"))["model"]["refinement"]
        if config.get("conditioning", {}).get("normalize_predictors", False) is not False:
            raise ValueError("Original native configuration does not use the registered raw predictor control.")
        if comparison["factor"] == "isolated_gate_calibration_loss" and config["reconstruction_loss_weight"] != .25:
            raise ValueError("The isolated Transformer gate coefficient differs from the registered 0.25.")
    registration = {
        "version": VERSION,
        "parent_experiment_plan_sha256": original["registered_plan_sha256"],
        "acceptance_contract_sha256": original["acceptance_contract_sha256"],
        "original_queue": str(original_path), "original_queue_sha256": original["sha256"],
        "registered_before_supplemental_training": True,
        "rationale": "Mean and mean-remainder experiments automatically normalize predictors while original native/direct controls do not; the difference confounds attribution to the mean component. These four paired native/direct controls isolate predictor conditioning normalization.",
        "gate_rationale": "The original queue compares trainable production gates and the full recipe, but omits the already implemented isolated gate-calibration objective. Two Transformer controls isolate this single coefficient against native_production_gate without combining it with predictor normalization or other auxiliary terms.",
        "heads": list(workflow.HEADS), "comparisons": comparisons,
        "changed_setting": {"conditioning.normalize_predictors": {"original": False, "supplemental": True}},
        "independent_gate_change": {"gate_calibration_loss": {"original": 0.0, "supplemental": 0.25}},
        "job_counts": {"normalized_native": 4, "isolated_transformer_gate_calibration": 2, "total": 6},
        "factorial_combinations": False,
        "training_seed": plan["screening_seed"], "fresh_weights_and_statistics": True,
        "unchanged": ["parent plan and acceptance criteria", "screen fitting and selection dates",
                      "stopping rules", "native process objective and its coefficient", "gates relative to each paired control",
                      "architecture and alignment", "sampling configuration", "frozen Phase-1 weights and scalers"],
        "execution_order": "These six fixed supplemental controls may execute concurrently with the preserved original development queue. All original and supplemental applicable jobs must finish before full-recipe selection.",
        "concurrent_original_development_permitted": True,
        "selection_rule": "Retain all original and supplemental outcomes, including unfavorable and numerical failures; use the existing provisional full-recipe selection rule only after every applicable job has finished.",
        "full_existing_recipe_controls_unchanged": True,
        "production_promotion": False,
    }
    registration["sha256"] = canonical_sha256(registration)
    registration_path = root / "conditioning_control_registration.json"
    queue = {
        "version": workflow.VERSION, "run_root": str(root), "artifact_root": str(artifact_root),
        "registered_plan_sha256": original["registered_plan_sha256"],
        "acceptance_contract_sha256": original["acceptance_contract_sha256"],
        "supplemental_registration": str(registration_path),
        "supplemental_registration_sha256": registration["sha256"],
        "original_queue": str(original_path), "original_queue_sha256": original["sha256"],
        "jobs": jobs, "head_order": list(workflow.HEADS), "production_promotion": False,
        "execution_entry_point": "cordex_scientific_supplemental.run_supplemental",
    }
    queue["sha256"] = canonical_sha256(queue)
    return root / "workflow" / "supplemental_queue.json", queue, registration_path, registration


def _unchanged(path, expected):
    if Path(path).exists() and workflow.read(path) != expected:
        raise ValueError("Refusing to overwrite a different immutable artifact: " + str(path))


def register_conditioning_controls(original_queue_path):
    """Return (path, queue); registration is legal only before these jobs start."""
    path, queue, registration_path, registration = _definition(original_queue_path)
    _unchanged(registration_path, registration)
    _unchanged(path, queue)
    if not registration_path.exists():
        execution_path = path.with_name(path.stem + "_execution.json")
        if execution_path.exists() or any(Path(job["output"]).exists() and any(Path(job["output"]).iterdir())
                                          for job in queue["jobs"]):
            raise ValueError("Conditioning controls must be registered before any supplemental training starts.")
    for destination, value in ((registration_path, registration), (path, queue)):
        if not destination.exists():
            workflow.write(destination, value)
    return path, queue


def _supplemental(path):
    path = Path(path).resolve()
    queue = _queue(path)
    expected_path, expected, registration_path, registration = _definition(queue["original_queue"])
    if (path != expected_path.resolve() or queue != expected
            or not registration_path.is_file() or workflow.read(registration_path) != registration):
        raise ValueError("Supplemental registration, hash or exact six-job/four-head coverage mismatch.")
    return queue


def _completed(path, queue, *, supplemental=False):
    state_path = Path(path).with_name(Path(path).stem + "_execution.json")
    if not state_path.is_file():
        raise ValueError("Development evidence is unfinished: no execution receipt.")
    state = workflow.read(state_path)
    if state.get("queue_sha256") != queue["sha256"]:
        raise ValueError("Execution receipt queue hash mismatch.")
    expected_ids = {job["id"] for job in queue["jobs"]}
    if set(state.get("jobs", {})) != expected_ids:
        raise ValueError("Execution receipt job coverage differs from its complete registered queue.")
    allowed = TERMINAL - {"CONDITION_NOT_TRIGGERED"} if supplemental else TERMINAL
    if (state.get("status") != "REGISTERED_QUEUE_COMPLETED"
            or any(value.get("status") not in allowed for value in state["jobs"].values())):
        raise ValueError("Development evidence is unfinished; all applicable outcomes must be retained.")
    return state_path, state


def run_supplemental(supplemental_queue_path):
    """Run fixed independent controls; only combined selection waits for both queues."""
    _supplemental(supplemental_queue_path)
    return workflow.run_queue(supplemental_queue_path, only_stage="screen")


def combine_development_evidence(original_queue_path, supplemental_queue_path):
    """Return an immutable combined queue path for prepare_full, never training."""
    original_path, original, root, _ = _parent(original_queue_path)
    supplemental_path = Path(supplemental_queue_path).resolve()
    supplemental = _supplemental(supplemental_path)
    if (Path(supplemental["original_queue"]).resolve() != original_path
            or supplemental["original_queue_sha256"] != original["sha256"]):
        raise ValueError("Supplemental queue identifies a different original development queue.")
    original_state_path, original_state = _completed(original_path, original)
    supplemental_state_path, supplemental_state = _completed(supplemental_path, supplemental, supplemental=True)
    jobs = copy.deepcopy(original["jobs"] + supplemental["jobs"])
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Combined job IDs overlap.")
    sources = [{"queue": str(path), "queue_sha256": data["sha256"], "queue_file_sha256": file_sha256(path),
                "execution": str(state_path), "execution_sha256": file_sha256(state_path)}
               for path, data, state_path in ((original_path, original, original_state_path),
                                              (supplemental_path, supplemental, supplemental_state_path))]
    combined = {"version": workflow.VERSION, "combined_evidence_version": COMBINED_VERSION,
                "run_root": str(root), "artifact_root": original.get("artifact_root", str(root)),
                "registered_plan_sha256": original["registered_plan_sha256"],
                "acceptance_contract_sha256": original["acceptance_contract_sha256"],
                "supplemental_registration_sha256": supplemental["supplemental_registration_sha256"],
                "jobs": jobs, "head_order": list(workflow.HEADS), "source_evidence": sources,
                "execution_allowed": False, "production_promotion": False,
                "all_outcomes_retained": True}
    combined["sha256"] = canonical_sha256(combined)
    execution = {"queue_sha256": combined["sha256"], "status": "REGISTERED_QUEUE_COMPLETED",
                 "jobs": copy.deepcopy({**original_state["jobs"], **supplemental_state["jobs"]}),
                 "source_evidence": sources, "combined_evidence_only": True, "production_promotion": False}
    path = root / "workflow" / "combined_development_queue.json"
    state_path = path.with_name(path.stem + "_execution.json")
    # Check both files before writing either. A interrupted first creation may
    # safely finish the missing matching file; existing bytes are never changed.
    _unchanged(path, combined)
    _unchanged(state_path, execution)
    for destination, value in ((path, combined), (state_path, execution)):
        if not destination.exists():
            workflow.write(destination, value)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    register = sub.add_parser("register"); register.add_argument("--original-queue", required=True)
    run = sub.add_parser("run"); run.add_argument("--supplemental-queue", required=True)
    combine = sub.add_parser("combine")
    combine.add_argument("--original-queue", required=True)
    combine.add_argument("--supplemental-queue", required=True)
    args = parser.parse_args(argv)
    if args.operation == "register":
        print(register_conditioning_controls(args.original_queue)[0]); return 0
    if args.operation == "run":
        return run_supplemental(args.supplemental_queue)
    print(combine_development_evidence(args.original_queue, args.supplemental_queue)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
