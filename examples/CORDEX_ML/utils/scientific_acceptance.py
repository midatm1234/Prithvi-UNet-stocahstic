#!/usr/bin/env python
"""Freeze, collect, evaluate and select under one scientific acceptance contract."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

from granitewxc.refinement.acceptance import (
    AcceptanceError, ScientificSelector, canonical_hash, collect_year_statistics,
    evaluate_scientific_acceptance, freeze_acceptance_contract, load_contract,
    merge_year_statistics, paired_resampling_plan, registered_years,
    save_year_statistics,
)
from examples.CORDEX_ML.utils.evaluate_refinement_outputs import (
    _sha256_file, _timestamp_keys, _timestamp_text, _validate_same_spatial_grid,
    align_exact_timestamps, convert_dataarray_units, validate_baseline_output,
    validate_refinement_output, validate_target,
)


def collect_files(args):
    contract = load_contract(args.contract)
    metadata = json.loads(Path(args.metadata_json).read_text())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        prediction = stack.enter_context(xr.open_dataset(args.prediction))
        baseline = stack.enter_context(xr.open_dataset(args.baseline))
        target = stack.enter_context(xr.open_dataset(args.target))
        selection = validate_refinement_output(
            prediction, variables=contract["variables"],
            expected_refinement_type=args.head,
            expected_checkpoint=args.checkpoint)
        validate_baseline_output(
            baseline, variables=contract["variables"],
            refinement_case=selection.refinement_case,
            phase1_checkpoint=selection.phase1_checkpoint,
            phase1_fingerprint=selection.phase1_fingerprint)
        validate_target(target, variables=contract["variables"])
        _validate_same_spatial_grid(prediction, baseline, "Phase-1 baseline")
        _validate_same_spatial_grid(prediction, target, "ground-truth target")
        aligned = align_exact_timestamps(prediction, baseline, target,
                                         require_all_prediction_times=True)
        keys = _timestamp_keys(aligned.prediction, "prediction")
        date_ids = [_timestamp_text(key)[:10] for key in keys]
        years = np.array([key[0] for key in keys])
        if not set(years).issubset(registered_years(contract)):
            raise AcceptanceError("Acceptance collection refuses development, fitting or quarantined years.")
        control = None
        if args.control:
            control = stack.enter_context(xr.open_dataset(args.control))
            _validate_same_spatial_grid(prediction, control, "deterministic control")
            if _timestamp_keys(control, "control") != keys:
                raise AcceptanceError("Control must contain identical date identifiers and ordering.")
        if metadata.get("mean_component_used") and control is None:
            raise AcceptanceError("Mean-component refinement requires its frozen deterministic control.")
        lat, lon = aligned.prediction.lat.values, aligned.prediction.lon.values
        common_metadata = {
            **metadata, "authenticated": True, "date_ids": date_ids,
            "training_seed": args.training_seed,
            "data_scope": args.data_scope,
            "contract_sha256": canonical_hash(contract),
            "phase1_sha256": _sha256_file(Path(selection.phase1_checkpoint)),
            "baseline_sha256": _sha256_file(Path(args.baseline)),
            "target_sha256": _sha256_file(Path(args.target)),
            "dates_sha256": hashlib.sha256(json.dumps(date_ids).encode()).hexdigest(),
            "coordinates_sha256": canonical_hash({"lat": lat.tolist(), "lon": lon.tolist()}),
            "config_sha256": _sha256_file(Path(args.case_config)),
            "normalization_sha256": selection.residual_normalizer_state_fingerprint,
            "checkpoint_sha256": selection.refinement_checkpoint_sha256,
            "prediction_sha256": _sha256_file(Path(args.prediction)),
        }
        entries = {}
        for variable in contract["variables"]:
            units = contract["units"][variable]
            members, member_conversion = convert_dataarray_units(
                aligned.prediction[selection.variables[variable]], units, variable=variable)
            base, baseline_conversion = convert_dataarray_units(
                aligned.baseline[variable], units, variable=variable)
            truth, target_conversion = convert_dataarray_units(
                aligned.target[variable], units, variable=variable)
            deterministic = None
            if control is not None:
                deterministic, _ = convert_dataarray_units(control[variable], units, variable=variable)
            accumulated = None
            mask_hash = hashlib.sha256()
            for start in range(0, len(years), args.chunk_time):
                stop = min(start + args.chunk_time, len(years))
                sl = {"time": slice(start, stop)}
                ens = members.isel(**sl).transpose("time", selection.member_dim, "lat", "lon").values
                b = base.isel(**sl).transpose("time", "lat", "lon").values
                t = truth.isel(**sl).transpose("time", "lat", "lon").values
                c = None if deterministic is None else deterministic.isel(**sl).transpose("time", "lat", "lon").values
                mask_hash.update((np.isfinite(b) & np.isfinite(t)).tobytes())
                chunk = collect_year_statistics(ens, b, t, years[start:stop], lat, lon,
                                                deterministic_control=c)
                accumulated = merge_year_statistics(accumulated, chunk)
            accumulated["metadata"].update({
                **common_metadata, "units": units, "mask_sha256": mask_hash.hexdigest(),
                "raw_members_saved": f"{variable}_members_unbounded" in aligned.prediction,
                "unit_conversions": {"members": member_conversion,
                                     "phase1": baseline_conversion, "target": target_conversion},
            })
            path = output / f"{args.head}_seed{args.training_seed}_{variable}_year_statistics.npz"
            if path.exists():
                raise AcceptanceError(f"Refusing to overwrite existing evidence: {path}")
            entries[variable] = {"path": str(path.resolve()),
                                 "sha256": save_year_statistics(path, accumulated)}
        manifest = {"contract_sha256": canonical_hash(contract),
                    "scope": "component_validation",
                    "records": {args.head: {str(args.training_seed): entries}},
                    "auxiliary_metrics": {},
                    "limitations": ["Extreme bootstrap and required diagnostic artifacts must be supplied before acceptance."]}
        destination = output / "evidence.json"
        if destination.exists():
            raise AcceptanceError(f"Refusing to overwrite {destination}")
        destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


def write_report(report, output_dir):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / "scientific_acceptance_results.json"
    if result_path.exists():
        raise AcceptanceError(f"Refusing to overwrite acceptance results: {result_path}")
    result_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    rows = report["rows"]
    columns = sorted(set().union(*(row.keys() for row in rows))) if rows else ["head", "variable", "status"]
    with (root / "scientific_acceptance_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Scientific acceptance", "",
        f"Component-validation outcome: **{report['status']}**.",
        "End-to-end acceptance: **INCONCLUSIVE**. Production promotion: **false**.", "",
        f"Contract SHA-256: {report['contract_sha256']}.", "",
        "| Head | Variable | Status | Point guardrails |",
        "|---|---|---|---|",
    ]
    for head, variables in report["combinations"].items():
        for variable, entry in variables.items():
            lines.append(f"| {head} | {variable} | {entry['status']} | {entry['point_guardrails_pass']} |")
    lines.extend(["", "Prerequisites preventing acceptance:", ""])
    lines.extend(f"- {reason}" for reason in report["prerequisite_failures"])
    lines.extend([
        "", "Intervals resample paired complete years and independent training seeds.",
        "Spatial fields and each issued ensemble stay intact. Inference is conditional",
        "on the saved ensemble seed sets; projected members are never resampled as independent raw members.",
        "Point-eligible validation candidates remain separate from production acceptance.",
    ])
    (root / "SCIENTIFIC_ACCEPTANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default="examples/CORDEX_ML/scientific_acceptance.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--output-dir", required=True)
    collect = sub.add_parser("collect")
    for name in ("prediction", "baseline", "target", "checkpoint", "head",
                 "case-config", "metadata-json", "output-dir"):
        collect.add_argument(f"--{name}", required=True)
    collect.add_argument("--training-seed", type=int, required=True)
    collect.add_argument("--chunk-time", type=int, default=16)
    collect.add_argument("--data-scope", choices=["screening", "component_validation"], required=True)
    collect.add_argument("--control")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--evidence", required=True)
    evaluate.add_argument("--output-dir", required=True)
    plan = sub.add_parser("resampling-plan")
    plan.add_argument("--output", required=True)
    select = sub.add_parser("select")
    select.add_argument("--head", required=True)
    select.add_argument("--checkpoint", required=True)
    select.add_argument("--evidence", required=True)
    select.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    if args.command == "freeze":
        print(json.dumps(freeze_acceptance_contract(args.contract, args.output_dir), indent=2))
    elif args.command == "collect":
        print(json.dumps(collect_files(args), indent=2))
    elif args.command == "evaluate":
        evidence = json.loads(Path(args.evidence).read_text())
        report = evaluate_scientific_acceptance(evidence, contract)
        write_report(report, args.output_dir)
        print(json.dumps({"status": report["status"], "production_promotion": False}))
    elif args.command == "resampling-plan":
        payload = paired_resampling_plan(contract)
        output = Path(args.output)
        if output.exists():
            raise AcceptanceError("Refusing to overwrite a registered resampling plan.")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **payload)
        print(payload["sha256"])
    else:
        evidence = json.loads(Path(args.evidence).read_text())
        result = ScientificSelector(args.output_dir, contract, args.head).consider(args.checkpoint, evidence)
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
