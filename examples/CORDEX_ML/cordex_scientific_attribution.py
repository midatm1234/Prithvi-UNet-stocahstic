#!/usr/bin/env python
"""Supplemental paired contrasts against freshly fitted stochastic controls.

This reports the selected development procedure versus a fresh existing recipe.
It does not modify the frozen acceptance family, select checkpoints, rerun a
network, or establish an independent causal effect of one individual loss term.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from granitewxc.refinement.acceptance import (
    AcceptanceError, _date_completeness, _product_metrics, _validate_statistics,
    acceptance_regions, bootstrap_comparison_changes, canonical_hash,
    load_contract, load_year_statistics, paired_resampling_plan, registered_years,
    simultaneous_max_statistic_intervals,
)
from granitewxc.refinement.scientific_data import file_sha256
from granitewxc.refinement.config import config_fingerprint, resolve_refinement_config
from granitewxc.refinement.experiments import experiment_recipes

VERSION = "paired_selected_fresh_stochastic_control_v1"
FAMILY = "supplemental_selected_vs_fresh_stochastic_control"
CORE = ("daily_rmse", "daily_mae", "mean_absolute_climatological_bias")
REFERENCE = "fresh_stochastic_control"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def protocol(contract, parent_plan_sha256, registered_at_utc):
    plan = paired_resampling_plan(contract)
    result = {
        "version": VERSION, "parent_experiment_plan_sha256": parent_plan_sha256,
        "acceptance_contract_sha256": canonical_hash(contract),
        "registered_at_utc": registered_at_utc, "registered_before_full_fitting": True,
        "simultaneous_family": FAMILY, "heads": contract["heads"],
        "variables": contract["variables"], "training_seeds": contract["training_seeds"],
        "regions": contract["regions"], "pooled_region_metrics": list(CORE),
        "full_domain_additional_metric": "crps",
        "climate_regimes": "pooled: all mandatory regions; historical and future: full domain for all four metrics",
        "physical_product": "postprocessed physical members used by the original acceptance collector",
        "reference": "same head and training seed, freshly fitted existing_full_recipe stochastic ensemble",
        "point_reduction": "compute each seed's physical metric, then arithmetic mean across three paired training seeds",
        "contrast": "(selected_metric - fresh_control_metric) / max(abs(fresh_control_metric), fixed_variable_scale)",
        "physical_values": "selected and fresh-control metric values and their signed difference in physical units",
        "near_zero_reference": contract["near_zero_reference"],
        "resampling_plan_sha256": plan["sha256"], "bootstrap": contract["bootstrap"],
        "interval_family": "all listed head-variable-region-stratum relative contrasts together, separate from acceptance",
        "interpretation": "negative relative contrast means lower error; simultaneous upper bound below zero supports lower error conditional on the saved checkpoints and finite member seeds",
        "member_resampling": "forbidden; each issued ensemble and spatial field stays intact",
        "sampler_pairing": "Issued inference settings, nonnegative postprocessing strategy, stream/member mapping and dates must match; training-only flow mean-path/time-sampling fields may differ. Different sampler interventions require a separately registered contrast.",
        "acceptance_decisions_unchanged": True, "production_promotion": False,
        "limitations": ["Development-selected recipes and checkpoints reuse component validation; selection uncertainty is not removed by this bootstrap.",
                        "The comparison isolates the selected procedure from additional fitting of the existing recipe; it cannot attribute a multi-setting procedure to one individual mechanism.",
                        "Eight year blocks and three training seeds limit precision; intervals condition on finite saved sampler seeds.",
                        "Existing Phase-1 exposure prevents independent end-to-end acceptance; 1981-2000 regeneration is unexecuted and 2041-2060 targets remain quarantined."],
    }
    result["sha256"] = canonical_hash(result)
    return result


def register_contrast(run_root):
    root = Path(run_root).resolve()
    parent_path = root / "registered_experiment_plan.json"
    parent = read(parent_path)
    parent_hash = hashlib.sha256(parent_path.read_text(encoding="utf8").encode()).hexdigest()
    if parent_hash != parent_path.with_suffix(".sha256").read_text().strip():
        raise AcceptanceError("Frozen parent plan hash changed.")
    contract = load_contract(root / "frozen_scientific_acceptance_v2.json")
    if parent["acceptance_contract_sha256"] != canonical_hash(contract):
        raise AcceptanceError("Parent acceptance contract link changed.")
    path = root / "fresh_control_contrast_registration.json"
    stamp = read(path)["registered_at_utc"] if path.exists() else datetime.now(timezone.utc).isoformat()
    value = protocol(contract, parent_hash, stamp)
    if path.exists():
        if read(path) != value:
            raise AcceptanceError("Refusing to overwrite a different frozen supplemental contrast protocol.")
        return path, value
    original = root / "workflow" / "development_queue_artifacts.json"
    artifact_root = Path(read(original).get("artifact_root", root)) if original.exists() else root
    for base in {root, artifact_root}:
        for folder in (base / "workflow" / "runs").glob("full_*"):
            if any(item.name in ("last.ckpt", "run_contract.json", "execution_status.json") for item in folder.rglob("*")):
                raise AcceptanceError("Supplemental contrast must be registered before full fitting starts.")
    write(path, value)
    return path, value


def _check(path, digest, checked):
    path = Path(path).resolve(); key = str(path)
    if not digest or not path.is_file():
        raise AcceptanceError("Missing referenced artifact: " + key)
    if key not in checked:
        checked[key] = file_sha256(path)
    if checked[key] != digest:
        raise AcceptanceError("Referenced artifact hash changed: " + key)
    return path


def _pairs(value, checked):
    if isinstance(value, dict):
        for a, b in (("path", "sha256"), ("artifact", "artifact_sha256")):
            if a in value and b in value:
                _check(value[a], value[b], checked)
        for child in value.values():
            _pairs(child, checked)
    elif isinstance(value, list):
        for child in value:
            _pairs(child, checked)


def _same_collected_statistics(merged, source):
    for name in ("years", "lat", "lon"):
        if not np.array_equal(merged[name], source[name]):
            raise AcceptanceError("Merged coordinates/years differ from authenticated collected statistics.")
    if set(merged["products"]) != set(source["products"]):
        raise AcceptanceError("Merged physical products differ from authenticated collection.")
    for product, fields in source["products"].items():
        if set(merged["products"][product]) != set(fields):
            raise AcceptanceError("Merged statistic fields differ from authenticated collection.")
        for name, values in fields.items():
            if not np.array_equal(values, merged["products"][product][name]):
                raise AcceptanceError("Merged numerical statistics differ from the authenticated processed physical product.")
    metadata = []
    for record in (merged, source):
        item = copy.deepcopy(record["metadata"])
        item.get("diagnostics", {}).pop("independently_trained_seed_results", None)
        metadata.append(item)
    if metadata[0] != metadata[1]:
        raise AcceptanceError("Merged statistics metadata changed outside the declared seed-diagnostic update.")


def _inference_settings(authentication):
    fields = ("effective_sampling_steps", "training_timesteps", "prediction_type", "schedule", "beta_start", "beta_end",
              "cosine_s", "clip_sample", "clip_sample_range", "eta", "solver", "source_distribution",
              "stochastic_initialization", "sigma_min")
    settings = authentication.get("sampling_settings")
    if not isinstance(settings, dict) or "effective_sampling_steps" not in settings:
        raise AcceptanceError("Resolved issued sampler settings are missing.")
    return {name: settings[name] for name in fields if name in settings}


def load_runner_contract(path, expected_sha256):
    """Runner identities use config_fingerprint, not compact acceptance JSON."""
    payload = read(path)
    run = payload["contract"]
    if config_fingerprint(run) != payload.get("sha256") or payload["sha256"] != expected_sha256:
        raise AcceptanceError("Runner contract fingerprint differs from its persisted checkpoint identity.")
    return run


def _load_family(path, label, contract, registration, checked, full_jobs):
    path = Path(path).resolve()
    family = read(path)
    if family.get("contract_sha256") != canonical_hash(contract) or family.get("scope") != "component_validation":
        raise AcceptanceError("Family contract or component scope mismatch.")
    marker = read(path.parent / "completion.json")
    if (marker.get("complete") is not True or marker.get("operation") != "merge"
            or marker.get("contract_sha256") != canonical_hash(contract)):
        raise AcceptanceError("Family merge is not authenticated complete.")
    _check(path.parent / "acceptance_report" / "scientific_acceptance_results.json", marker["artifact_sha256"], checked)
    expected = {(head, str(seed)) for head in contract["heads"] for seed in contract["training_seeds"]}
    sources, source_records, incomplete = {}, {}, []
    for reference in family.get("input_evidence", []):
        source_path = _check(reference["path"], reference["sha256"], checked)
        source = read(source_path)
        if source.get("contract_sha256") != canonical_hash(contract):
            raise AcceptanceError("Source evidence contract mismatch.")
        identities = [(h, str(s)) for h, seeds in source["records"].items() for s in seeds]
        if len(identities) != 1 or identities[0] not in expected or identities[0] in sources:
            raise AcceptanceError("Repeated, unknown or mixed source head/seed identity.")
        head, seed = identities[0]
        _pairs(source, checked)
        authentication = read(source["authentication"]["artifact"])
        if (authentication.get("authenticated") is not True or authentication.get("head") != head
                or str(authentication.get("training_seed")) != seed):
            raise AcceptanceError("Source authentication identifies different fields or head/seed.")
        if not authentication.get("full_training_verified"):
            incomplete.append(f"{label}/{head}/{seed}: full fitting not verified")
        for a, b in (("prediction_path", "prediction_sha256"), ("cache_path", "cache_sha256"),
                     ("checkpoint_path", "checkpoint_sha256")):
            _check(authentication[a], authentication[b], checked)
        _pairs(authentication.get("completed_run_authentication", {}), checked)
        run_path = Path(authentication["checkpoint_path"]).parent / "run_contract.json"
        run = load_runner_contract(run_path, authentication.get("run_contract_sha256"))
        checked[str(run_path.resolve())] = file_sha256(run_path)
        if (run.get("plan_sha256") != registration["parent_experiment_plan_sha256"]
                or run.get("acceptance_contract_sha256") != canonical_hash(contract)
                or run.get("head") != head or str(run.get("seed")) != seed):
            raise AcceptanceError("Full run identity or parent plan differs from its authenticated source.")
        planned = full_jobs.get((label, head, seed))
        if (planned is None or Path(planned["output"]).resolve() != Path(authentication["checkpoint_path"]).parent.resolve()
                or planned["recipe"] != run.get("recipe", {}).get("name")):
            raise AcceptanceError("Family checkpoint is not the selected output of its registered full job.")
        for name in ("cache_sha256", "phase1_sha256"):
            source_key = "source_cache_sha256" if name == "cache_sha256" else "source_phase1_sha256"
            if run.get(source_key) != authentication.get(name):
                raise AcceptanceError("Authenticated full run and physical source identity differ.")
        if run.get("stage") != "full":
            incomplete.append(f"{label}/{head}/{seed}: source is not a full-stage fit")
        if label == "control" and (run.get("recipe", {}).get("name") != "existing_full_recipe" or run.get("formulation") != "direct"):
            raise AcceptanceError("Fresh control must be the full existing recipe stochastic direct model.")
        if label == "control":
            original_config = resolve_refinement_config(run["source_config"])
            expected_recipe = experiment_recipes(original_config, contract["variables"])["existing_full_recipe"].to_dict()
            if (canonical_hash(original_config.to_dict()) != canonical_hash(run["refinement_config"])
                    or canonical_hash(expected_recipe) != canonical_hash(run["recipe"])):
                raise AcceptanceError("Fresh existing-recipe control contains an unregistered configuration or recipe override.")
        authentication = dict(authentication, procedure_configuration={"recipe": run["recipe"],
            "formulation": run.get("formulation"), "refinement_config": run.get("refinement_config"),
            "source_config": run.get("source_config"), "target_contract": authentication.get("target_contract"),
            "normalization_sha256": authentication.get("normalization_sha256")})
        sources[head, seed] = authentication
        source_records[head, seed] = source["records"][head][seed]
    records = {}
    for head, seeds in family.get("records", {}).items():
        for seed, variables in seeds.items():
            seed = str(seed)
            if (head, seed) not in expected:
                raise AcceptanceError("Unregistered head/seed in merged statistics.")
            if (head, seed) not in sources:
                raise AcceptanceError("Merged statistics have no matching authenticated source.")
            for variable, reference in variables.items():
                if variable not in contract["variables"]:
                    raise AcceptanceError("Unregistered target variable.")
                record = load_year_statistics(_check(reference["path"], reference["sha256"], checked), reference["sha256"])
                source_reference = source_records[head, seed].get(variable)
                if source_reference is None:
                    raise AcceptanceError("Merged target lacks its authenticated collected statistics.")
                collected = load_year_statistics(_check(source_reference["path"], source_reference["sha256"], checked), source_reference["sha256"])
                _same_collected_statistics(record, collected)
                metadata, authentication = record["metadata"], sources[head, seed]
                if (metadata.get("authenticated") is not True or str(metadata.get("training_seed")) != seed
                        or metadata.get("checkpoint_sha256") != authentication["checkpoint_sha256"]
                        or metadata.get("date_ids") != authentication["date_ids"]):
                    raise AcceptanceError("Statistics are not bound to their authenticated source.")
                field_hashes = authentication["field_sha256"][variable]
                for metadata_key, source_key in (("baseline_sha256", "baseline"), ("target_sha256", "target"), ("mask_sha256", "mask")):
                    if metadata.get(metadata_key) != field_hashes.get(source_key):
                        raise AcceptanceError("Year statistics baseline/target/mask hashes differ from source authentication.")
                for name in ("ensemble_size", "phase1_sha256", "dates_sha256", "coordinates_sha256", "calendar", "fit_years"):
                    if metadata.get(name) != authentication.get(name):
                        raise AcceptanceError("Year statistics source metadata differs: " + name)
                if (metadata.get("data_scope") != "component_validation"
                        or not np.array_equal(record["years"], registered_years(contract))
                        or not _date_completeness(metadata, registered_years(contract))):
                    incomplete.append(f"{label}/{head}/{seed}/{variable}: complete validation years unavailable")
                if metadata.get("ensemble_size", 0) < contract["sampling"]["minimum_members"]:
                    incomplete.append(f"{label}/{head}/{seed}/{variable}: insufficient physical members")
                records.setdefault(head, {}).setdefault(seed, {})[variable] = record
    for head, seed in sorted(expected):
        for variable in contract["variables"]:
            if variable not in records.get(head, {}).get(seed, {}):
                incomplete.append(f"{label}/{head}/{seed}/{variable}: missing case")
    _validate_statistics(records, contract)
    return records, sources, incomplete


def paired_records(selected, control, selected_auth, control_auth, contract):
    result = {}
    identity_fields = ("phase1_sha256", "baseline_sha256", "target_sha256", "mask_sha256",
                       "dates_sha256", "coordinates_sha256", "date_ids", "calendar", "units",
                       "ensemble_size", "sampling_seed_ids", "fit_years")
    for head in contract["heads"]:
        for label, sources in (("selected", selected_auth), ("control", control_auth)):
            hashes = [sources[head, str(seed)]["checkpoint_sha256"] for seed in contract["training_seeds"]]
            if len(set(hashes)) != len(hashes):
                raise AcceptanceError(f"{label}/{head}: independently trained seeds reuse checkpoint bytes.")
        for seed in map(str, contract["training_seeds"]):
            left, right = selected_auth[head, seed], control_auth[head, seed]
            if Path(left["checkpoint_path"]).parent.resolve() == Path(right["checkpoint_path"]).parent.resolve():
                raise AcceptanceError("Selected and fresh control reuse the same fitted run.")
            if _inference_settings(left) != _inference_settings(right):
                raise AcceptanceError("Selected/control issued sampler settings differ from this matched-sampler protocol.")
            for name in ("cache_sha256", "phase1_sha256", "dates_sha256", "coordinates_sha256", "field_sha256",
                         "date_ids", "source_timestamps", "sample_ids", "calendar", "variables", "units",
                         "ensemble_size", "sampling_seed", "canonical_member_batch", "random_stream_version", "nonnegative_strategy", "fit_years"):
                if name not in left or left.get(name) != right.get(name):
                    raise AcceptanceError("Authenticated selected/control pairing mismatch: " + name)
            for variable in contract["variables"]:
                a, b = selected[head][seed][variable], control[head][seed][variable]
                for name in identity_fields:
                    if name not in a["metadata"] or a["metadata"].get(name) != b["metadata"].get(name):
                        raise AcceptanceError("Selected/control pairing mismatch: " + name)
                if a["metadata"]["units"] != contract["units"][variable]:
                    raise AcceptanceError("Paired physical units differ from the contract.")
                for name in ("years", "lat", "lon"):
                    if not np.array_equal(a[name], b[name]):
                        raise AcceptanceError("Selected/control coordinate or year ordering mismatch.")
                for name in a["products"]["phase1"]:
                    if not np.array_equal(a["products"]["phase1"][name], b["products"]["phase1"][name]):
                        raise AcceptanceError("Selected/control baseline or target sufficient statistics differ.")
                # 'control' in existing statistics is a deterministic mean.
                # The fresh stochastic reference retains its own refined CRPS.
                result.setdefault(head, {}).setdefault(seed, {})[variable] = {
                    **a, "products": {"phase1": a["products"]["phase1"], "refined": a["products"]["refined"],
                                      REFERENCE: b["products"]["refined"]}}
    _validate_statistics(result, contract)
    return result


def point_contrasts(records, contract):
    rows, individual = [], []
    seeds = list(map(str, contract["training_seeds"]))
    for head in contract["heads"]:
        for variable in contract["variables"]:
            prototype = records[head][seeds[0]][variable]
            years = prototype["years"]
            masks = acceptance_regions(prototype["lat"], prototype["lon"], contract)
            strata = {"pooled": np.ones(len(years))}
            strata.update({name: ((years >= bounds[0]) & (years <= bounds[1])).astype(float)
                           for name, bounds in contract["splits"]["component_validation"].items()})
            for stratum, weights in strata.items():
                scores = {seed: {name: _product_metrics(records[head][seed][variable]["products"][name], weights, masks)
                                 for name in ("refined", REFERENCE)} for seed in seeds}
                regions = masks if stratum == "pooled" else {"full": masks["full"]}
                for region in regions:
                    if region == "southeast_precipitation_box" and variable != "pr":
                        continue
                    for metric in CORE + (("crps",) if region == "full" else ()):
                        key = f"{head}/{variable}/{stratum}/{region}/{metric}/{REFERENCE}"
                        pairs = [(scores[seed]["refined"][region][metric], scores[seed][REFERENCE][region][metric]) for seed in seeds]
                        if any(value is None or not np.isfinite(value) for pair in pairs for value in pair):
                            raise AcceptanceError("Nonfinite or unsupported mandatory paired metric: " + key)
                        candidate, reference = np.mean(pairs, axis=0)
                        delta = candidate - reference
                        change = delta / max(abs(reference), contract["near_zero_reference"][variable])
                        rows.append({"key": key, "head": head, "variable": variable, "stratum": stratum,
                                     "region": region, "metric": metric, "reference_product": REFERENCE,
                                     "candidate": float(candidate), "reference": float(reference),
                                     "physical_difference": float(delta), "change": float(change),
                                     "units": contract["units"][variable]})
                        for seed, (candidate_seed, control_seed) in zip(seeds, pairs):
                            individual.append({"key": key, "training_seed": int(seed), "head": head,
                                               "variable": variable, "stratum": stratum, "region": region, "metric": metric,
                                               "selected_metric": candidate_seed, "fresh_control_metric": control_seed,
                                               "physical_difference": candidate_seed - control_seed,
                                               "units": contract["units"][variable]})
    return rows, individual


def _csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def compare_selected_fresh_control(selected_evidence, control_evidence, contract, registration, output_dir, *, batch_size=32):
    contract = load_contract(contract) if not isinstance(contract, dict) else contract
    registration_path = Path(registration).resolve()
    registration = read(registration_path)
    if protocol(contract, registration["parent_experiment_plan_sha256"], registration["registered_at_utc"]) != registration:
        raise AcceptanceError("Frozen supplemental contrast registration changed.")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise AcceptanceError("Supplemental output must be new or empty; preserve previous artifacts.")
    checked = {}
    full_path = registration_path.parent / "workflow" / "full_queue.json"
    full = read(full_path)
    if (canonical_hash({k: v for k, v in full.items() if k != "sha256"}) != full.get("sha256")
            or full.get("registered_plan_sha256") != registration["parent_experiment_plan_sha256"]
            or full.get("acceptance_contract_sha256") != canonical_hash(contract)):
        raise AcceptanceError("Registered full queue or parent links changed.")
    full_jobs = {(job["comparison_group"], job["head"], str(job["seed"])): job
                 for job in full["jobs"] if job["head"] in contract["heads"]}
    expected_jobs = {(label, head, str(seed)) for label in ("selected", "control")
                     for head in contract["heads"] for seed in contract["training_seeds"]}
    if set(full_jobs) != expected_jobs or sum(job["head"] in contract["heads"] for job in full["jobs"]) != len(expected_jobs):
        raise AcceptanceError("Full registration does not cover both complete paired head/seed families.")
    checked[str(full_path.resolve())] = file_sha256(full_path)
    a, a_auth, missing_a = _load_family(selected_evidence, "selected", contract, registration, checked, full_jobs)
    b, b_auth, missing_b = _load_family(control_evidence, "control", contract, registration, checked, full_jobs)
    report = {"version": VERSION, "contract_sha256": canonical_hash(contract),
              "registration_sha256": registration["sha256"], "simultaneous_family": FAMILY,
              "scope": "component_validation_diagnostic_conditional_on_selected_procedure",
              "acceptance_decisions_unchanged": True, "acceptance_families_combined": False,
              "production_promotion": False, "physical_product": registration["physical_product"],
              "limitations": registration["limitations"], "missing_cases": missing_a + missing_b,
              "status": "INCONCLUSIVE" if missing_a or missing_b else "COMPLETE", "rows_count": 0,
              "input_evidence": [{"comparison_group": group, "path": str(Path(path).resolve()), "sha256": file_sha256(path)}
                                 for group, path in (("selected", selected_evidence), ("control", control_evidence))]}
    output.mkdir(parents=True, exist_ok=True)
    products = {}
    if report["status"] == "COMPLETE":
        records = paired_records(a, b, a_auth, b_auth, contract)
        rows, per_seed = point_contrasts(records, contract)
        plan = paired_resampling_plan(contract)
        if plan["sha256"] != registration["resampling_plan_sha256"]:
            raise AcceptanceError("Paired year/seed draw identity changed.")
        changes = bootstrap_comparison_changes(records, contract, rows, plan, batch_size=batch_size)
        intervals = simultaneous_max_statistic_intervals([row["change"] for row in rows], changes, contract["bootstrap"]["confidence"])
        for index, row in enumerate(rows):
            row.update(relative_ci_lower=float(intervals["lower"][index]), relative_ci_upper=float(intervals["upper"][index]),
                       bootstrap_standard_error=float(intervals["standard_error"][index]),
                       degenerate_conditional_interval=bool(intervals["degenerate"][index]))
            row["diagnostic_contrast"] = ("LOWER_ERROR" if row["relative_ci_upper"] < 0 else
                                          "HIGHER_ERROR" if row["relative_ci_lower"] > 0 else "UNRESOLVED")
        _csv(output / "contrasts.csv", rows); _csv(output / "per_seed_metrics.csv", per_seed)
        np.savez_compressed(output / "bootstrap_changes.npz", changes=changes,
                            row_keys=np.asarray([row["key"] for row in rows]), year_weights=plan["year_weights"],
                            seed_indices=plan["seed_indices"], years=plan["years"], plan_sha256=np.asarray(plan["sha256"]))
        report["paired_procedures"] = [{"head": head, "training_seed": int(seed),
            "selected_checkpoint": a_auth[head, seed]["checkpoint_path"],
            "fresh_control_checkpoint": b_auth[head, seed]["checkpoint_path"],
            "selected": a_auth[head, seed]["procedure_configuration"],
            "fresh_control": b_auth[head, seed]["procedure_configuration"],
            "issued_sampler": _inference_settings(a_auth[head, seed]),
            "recorded_selected_sampling_settings": a_auth[head, seed]["sampling_settings"],
            "recorded_control_sampling_settings": b_auth[head, seed]["sampling_settings"]}
            for head in contract["heads"] for seed in map(str, contract["training_seeds"])]
        report.update(rows_count=len(rows), rows=rows, confidence=contract["bootstrap"]["confidence"],
                      resampling_plan_sha256=plan["sha256"], bootstrap_replicates=len(changes),
                      simultaneous_critical_value=intervals["critical_value"],
                      interval_quantity="relative contrast; physical metric values and differences are point estimates")
        products = {name: {"path": str(output / name), "sha256": file_sha256(output / name)}
                    for name in ("contrasts.csv", "per_seed_metrics.csv", "bootstrap_changes.npz")}
    write(output / "comparison.json", report)
    write(output / "evidence.json", {"version": VERSION, "contract_sha256": canonical_hash(contract),
        "input_evidence": report["input_evidence"], "registration": {"path": str(registration_path), "sha256": file_sha256(registration_path)},
        "products": products, "authenticated_input_files": [{"path": path, "sha256": digest} for path, digest in checked.items()],
        "scope": report["scope"], "acceptance_decisions_unchanged": True})
    write(output / "completion.json", {"version": VERSION, "operation": "attribution", "complete": True,
        "contract_sha256": canonical_hash(contract), "artifact_sha256": file_sha256(output / "comparison.json")})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    register = sub.add_parser("register"); register.add_argument("--run-root", required=True)
    compare = sub.add_parser("compare")
    for name in ("selected", "control", "contract", "registration", "output"):
        compare.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    if args.operation == "register":
        print(register_contrast(args.run_root)[0]); return 0
    result = compare_selected_fresh_control(args.selected, args.control, args.contract, args.registration, args.output)
    print(json.dumps({"status": result["status"], "rows_count": result["rows_count"], "production_promotion": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
