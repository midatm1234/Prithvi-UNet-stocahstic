"""Authenticate experimental HDF5 products and build scientific evidence.

No fitting, sampling, normalization fitting, or quarantined target access.
Screening checkpoints remain provisional when sampled over full validation.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from granitewxc.refinement.acceptance import (
    AcceptanceError, acceptance_regions, canonical_hash, collect_year_statistics,
    comparison_rows, evaluate_scientific_acceptance, load_contract,
    load_year_statistics, merge_year_statistics, paired_resampling_plan,
    registered_years, save_year_statistics, _product_metrics, _date_completeness,
)
from granitewxc.refinement.config import config_fingerprint
from granitewxc.refinement.diagnostics import empirical_crps, negative_value_summary
from granitewxc.refinement.mean_correction import tensor_fingerprint
from granitewxc.refinement.scientific_data import CACHE_VERSION, canonical_sha256, file_sha256
from granitewxc.refinement.scientific_extremes import YearExtremeStatistics
from examples.CORDEX_ML import cordex_scientific_experiments as runner
from examples.CORDEX_ML.utils.evaluate_refinement_outputs import (
    evaluate_variable, write_evaluation_outputs, _quantile_metrics,
)
from examples.CORDEX_ML.utils.scientific_acceptance import write_report
EVIDENCE_VERSION = "authenticated_scientific_hdf5_evidence_v1"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def authenticate_cache(path):
    path = Path(path).resolve()
    manifest_path = path.with_suffix(".manifest.json")
    manifest = _json(manifest_path)
    if manifest.get("version") != CACHE_VERSION or manifest.get("complete") is not True:
        raise AcceptanceError("Cache is incomplete or has an unsupported schema.")
    if file_sha256(path) != manifest.get("cache_sha256"):
        raise AcceptanceError("Immutable Phase-1 cache bytes changed.")
    split = manifest["splits"]
    if canonical_sha256({k: v for k, v in split.items() if k != "sha256"}) != split.get("sha256"):
        raise AcceptanceError("Cache split manifest changed.")
    if manifest.get("phase1_frozen") is not True or manifest.get("phase1_eval") is not True:
        raise AcceptanceError("Cache does not authenticate frozen evaluation-mode Phase 1.")
    if file_sha256(manifest["phase1_checkpoint"]) != manifest.get("phase1_checkpoint_sha256"):
        raise AcceptanceError("Phase-1 checkpoint bytes differ from the cache reference.")
    with h5py.File(path, "r") as source:
        dates = source["dates"].asstr()[:].tolist()
        if dates != split["dates"] or len(dates) != len(set(dates)):
            raise AcceptanceError("Cache dates differ from registered unique identifiers.")
        if int(source.attrs.get("completed_samples", -1)) != len(dates):
            raise AcceptanceError("Incomplete Phase-1 cache coverage.")
    return {"path": str(path), "manifest": manifest,
            "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
            "manifest_sha256": file_sha256(manifest_path)}


def _take(dataset, indices):
    order = np.argsort(indices)
    return np.asarray(dataset[np.asarray(indices)[order]])[np.argsort(order)]


def _same(actual, expected, label):
    if actual.shape != expected.shape or not np.array_equal(actual, expected, equal_nan=True):
        raise AcceptanceError(f"Prediction {label} differs from the immutable Phase-1 cache.")


def full_training_limitations(run_contract, payload, plan, source, contract):
    """Keep provisional checkpoint ranking separate from full acceptance dates."""
    fit_dates = [source["splits"]["dates"][i] for i in source["splits"]["fit_indices"]]
    validation_dates = [source["splits"]["dates"][i] for i in source["splits"]["validation_indices"]]
    reasons = []
    for split_name, selected in (("component_fit", fit_dates), ("component_validation", validation_dates)):
        expected_years = registered_years(contract, split_name)
        civil_dates = [date[:10] for date in selected]
        coverage = {"date_ids": civil_dates, "calendar": source["calendar"],
                    "observed_dates_per_year": {str(year): sum(date.startswith(f"{year}-") for date in civil_dates)
                                                for year in expected_years}}
        if not _date_completeness(coverage, expected_years):
            reasons.append(f"Immutable cache lacks complete registered {split_name} years.")
    if run_contract.get("stage") != "full":
        reasons.append("Checkpoint was trained in a screening/overfit stage.")
    fit_hash = canonical_sha256(fit_dates)
    if (run_contract.get("fit_selection_sha256") != fit_hash
            or payload["provenance"].get("training_selection_fingerprint") != fit_hash):
        reasons.append("Checkpoint/statistics did not use every registered fitting date.")
    selection_days = plan.get("validation", {}).get("selection_day_of_month")
    selection_dates = (validation_dates if selection_days is None else
                       [date for date in validation_dates if int(date[8:10]) in selection_days])
    if run_contract.get("validation_selection_sha256") != canonical_sha256(selection_dates):
        reasons.append("Checkpoint ranking did not use the registered validation-selection dates.")
    return reasons


def authenticate_completed_run(checkpoint_path, payload, plan, contract):
    """Bind an earlier selected model to the completed registered training run."""
    directory = Path(checkpoint_path).resolve().parent
    execution_path = directory / "execution_status.json"
    selection_path = directory / "scientific_selection.json"
    last_path = directory / "last.ckpt"
    missing = [str(path) for path in (execution_path, selection_path, last_path) if not path.is_file()]
    if missing:
        return {"status": "incomplete", "reason": "Completed-run receipt files unavailable.",
                "missing": missing, "completed_run_epoch": None, "completed_run_updates": None}
    execution, selection = _json(execution_path), _json(selection_path)
    if (selection.get("contract_sha256") != canonical_hash(contract)
            or selection.get("head") != payload["run_contract"]["head"]):
        raise AcceptanceError("Completed-run selector identifies a different contract or architecture.")
    selected = selection.get("best_eligible") or selection.get("nearest_provisional")
    if (not selected or Path(selected.get("selected_checkpoint", "")).resolve() != Path(checkpoint_path).resolve()
            or selected.get("checkpoint_sha256") != file_sha256(checkpoint_path)):
        raise AcceptanceError("Prediction checkpoint is not the authenticated selected candidate of its run.")
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    if (last.get("kind") != runner.RUNNER_VERSION
            or last.get("run_contract_sha256") != payload["run_contract_sha256"]
            or config_fingerprint(last.get("run_contract")) != payload["run_contract_sha256"]):
        raise AcceptanceError("Completed-run final checkpoint belongs to another experiment.")
    if canonical_sha256(last.get("progress")) != canonical_sha256(execution):
        raise AcceptanceError("Completed execution receipt differs from final checkpoint progress.")
    stopping = plan["stopping"]["full"]
    epoch = int(execution.get("epoch", 0))
    status = str(execution.get("status", ""))
    conditions = (
        epoch >= stopping["minimum_epochs"]
        and epoch >= int(payload["progress"].get("epoch", 0))
        and int(execution.get("cursor", -1)) == 0
        and execution.get("pending_assessment") is False
        and (status == "STOPPED_REGISTERED_PATIENCE_PROVISIONAL"
             and int(execution.get("patience", -1)) >= stopping["patience_assessments"]
             or status == "REACHED_REGISTERED_MAXIMUM_PROVISIONAL"
             and epoch >= stopping["maximum_epochs"]))
    return {"status": "complete" if conditions else "incomplete",
            "reason": "Selected checkpoint retained after completed registered stopping."
                      if conditions else "Registered stopping rule has not completed.",
            "selected_checkpoint_epoch": int(payload["progress"].get("epoch", 0)),
            "completed_run_epoch": epoch, "completed_run_updates": int(execution.get("updates", 0)),
            "execution_status": status,
            "artifacts": {name: {"path": str(path), "sha256": file_sha256(path)}
                          for name, path in (("execution", execution_path), ("selection", selection_path), ("last_checkpoint", last_path))}}


def authenticate_prediction(prediction_path, cache_path, contract, *,
                            chunk_time=16, cache_authentication=None):
    """Check hashes and actual paired values; never infer independence from labels."""
    cache_auth = cache_authentication or authenticate_cache(cache_path)
    source_path = Path(cache_path).resolve()
    if (str(source_path) != cache_auth["path"] or source_path.stat().st_size != cache_auth["size"]
            or source_path.stat().st_mtime_ns != cache_auth["mtime_ns"]):
        raise AcceptanceError("Cache changed within this collection invocation.")
    source = cache_auth["manifest"]
    path = Path(prediction_path).resolve()
    manifest = _json(path.with_suffix(".manifest.json"))
    if manifest.get("version") != runner.RUNNER_VERSION or manifest.get("complete") is not True:
        raise AcceptanceError("Incomplete or unsupported prediction artifact.")
    if file_sha256(path) != manifest.get("sha256"):
        raise AcceptanceError("Prediction HDF5 bytes differ from its manifest.")
    authentication = manifest.get("authentication") or {}
    checkpoint_path = Path(authentication.get("checkpoint_path", ""))
    if not checkpoint_path.is_file() or file_sha256(checkpoint_path) != authentication.get("checkpoint_sha256"):
        raise AcceptanceError("Prediction checkpoint is missing or its hash changed.")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("kind") != runner.RUNNER_VERSION:
        raise AcceptanceError("Unsupported experimental checkpoint semantics.")
    run_contract = payload["run_contract"]
    run_hash = config_fingerprint(run_contract)
    if run_hash != payload.get("run_contract_sha256") or run_hash != authentication.get("run_contract_sha256"):
        raise AcceptanceError("Checkpoint/run-contract fingerprint mismatch.")
    digest = canonical_hash(contract)
    if authentication.get("acceptance_contract_sha256") != digest or run_contract.get("acceptance_contract_sha256") != digest:
        raise AcceptanceError("A different scientific acceptance contract was used.")
    if authentication.get("source_cache_sha256") != source["cache_sha256"] or run_contract.get("source_cache_sha256") != source["cache_sha256"]:
        raise AcceptanceError("Prediction/checkpoint identifies a different cache.")
    if authentication.get("phase1_checkpoint_sha256") != source["phase1_checkpoint_sha256"] or run_contract.get("source_phase1_sha256") != source["phase1_checkpoint_sha256"]:
        raise AcceptanceError("Prediction/checkpoint identifies a different Phase 1.")
    plan, plan_hash, checkpoint_acceptance = runner._load_registered_plan(payload["plan_path"], payload["contract_path"])
    if plan_hash != run_contract.get("plan_sha256") or plan_hash != authentication.get("plan_sha256"):
        raise AcceptanceError("Registered experiment plan changed.")
    if canonical_hash(checkpoint_acceptance) != digest:
        raise AcceptanceError("Checkpoint's frozen acceptance contract changed.")
    model, target = runner.restore_candidate(payload, "cpu")
    if tensor_fingerprint(model) != manifest.get("candidate_state_fingerprint"):
        raise AcceptanceError("Prediction identifies different candidate weights.")
    formulation = payload["formulation"]
    if formulation == "deterministic_mean":
        raise AcceptanceError("Deterministic control is not a stochastic acceptance candidate.")
    target_contract = runner.target_contract_for(formulation, target)
    if target_contract["normalizer_fingerprint"] != manifest.get("target_normalizer_fingerprint"):
        raise AcceptanceError("Prediction identifies different fitted statistics.")
    mean_used = formulation in ("mean_remainder", "crossfit_remainder")
    if manifest.get("mean_component_used") is not mean_used:
        raise AcceptanceError("Prediction mean-component semantics mismatch.")
    mean_fingerprint = getattr(target, "mean_fingerprint", None)
    if mean_used and manifest.get("deterministic_control_sha256") != mean_fingerprint:
        raise AcceptanceError("Frozen mean-control fingerprint mismatch.")
    del model
    variables = list(payload["variables"])
    if variables != manifest.get("variables") or variables != list(run_contract["recipe"]["variables"]):
        raise AcceptanceError("Prediction/checkpoint target order mismatch.")
    if not set(variables).issubset(contract["variables"]):
        raise AcceptanceError("Unregistered variable.")
    for variable in variables:
        if source["units"].get(variable) != contract["units"][variable]:
            raise AcceptanceError(f"Cache units for {variable} are incompatible; do not double-convert physical fields.")
    fit_dates = [source["splits"]["dates"][i] for i in source["splits"]["fit_indices"]]
    validation_dates = [source["splits"]["dates"][i] for i in source["splits"]["validation_indices"]]
    full_reasons = full_training_limitations(run_contract, payload, plan, source, contract)
    run_completion = authenticate_completed_run(checkpoint_path, payload, plan, contract)
    if run_completion["status"] != "complete":
        full_reasons.append(run_completion["reason"])
    if manifest.get("source_code_fingerprints") != run_contract.get("source_code_fingerprints"):
        raise AcceptanceError("Prediction and checkpoint implementation fingerprints differ.")
    if not isinstance(manifest.get("sampling_settings"), dict):
        raise AcceptanceError("Resolved sampling settings were not recorded.")
    with h5py.File(path, "r") as prediction, h5py.File(source_path, "r") as cache:
        dates = prediction["dates"].asstr()[:].tolist()
        ids = prediction["sample_ids"].asstr()[:].tolist()
        if len(dates) != len(set(dates)) or len(ids) != len(set(ids)) or len(ids) != len(dates):
            raise AcceptanceError("Prediction sample IDs must be unique and aligned.")
        if not bool(prediction.attrs.get("complete")) or int(prediction.attrs.get("completed_dates", -1)) != len(dates):
            raise AcceptanceError("Prediction date coverage is incomplete.")
        if prediction.attrs.get("prediction_identity") != manifest.get("prediction_identity"):
            raise AcceptanceError("Prediction file/manifest identity mismatch.")
        if canonical_sha256(dates) != manifest.get("dates_sha256"):
            raise AcceptanceError("Prediction date fingerprint mismatch.")
        if json.loads(prediction.attrs["variables"]) != variables:
            raise AcceptanceError("HDF5 target ordering mismatch.")
        seed = int(prediction.attrs["training_seed"])
        if seed != run_contract.get("seed") or seed not in contract["training_seeds"]:
            raise AcceptanceError("Training-seed provenance mismatch.")
        head = run_contract.get("head")
        if head not in contract["heads"]:
            raise AcceptanceError("Unregistered stochastic architecture.")
        member_count = int(prediction.attrs["ensemble_size"])
        if member_count != manifest.get("members") or member_count < 2:
            raise AcceptanceError("Inconsistent physical ensemble size.")
        if prediction.attrs["formulation"] != formulation:
            raise AcceptanceError("HDF5 target formulation mismatch.")
        scope = str(prediction.attrs["data_scope"])
        if scope != manifest.get("scope"):
            raise AcceptanceError("Prediction scope mismatch.")
        if dates != validation_dates or scope != "component_validation":
            full_reasons.append("Predictions are a subset/diagnostic rather than full validation.")
        lookup = {date: i for i, date in enumerate(source["splits"]["dates"])}
        if any(date not in lookup for date in dates):
            raise AcceptanceError("Prediction date outside authenticated development cache.")
        if any(int(date[:4]) not in registered_years(contract) for date in dates):
            raise AcceptanceError("Fitting, historical-development or quarantined targets cannot enter acceptance.")
        if ids != [f"{source['source_case']}|{date}" for date in dates]:
            raise AcceptanceError("Stable sampling IDs differ from immutable cache IDs.")
        _same(prediction["latitude"][:], cache["latitude"][:], "latitude")
        _same(prediction["longitude"][:], cache["longitude"][:], "longitude")
        channels = [source["variables"].index(variable) for variable in variables]
        hashes = {variable: {name: hashlib.sha256() for name in ("baseline", "target", "mask")} for variable in variables}
        mean_roundoff = {"raw": 0., "processed": 0.}
        corrections = {variable: False for variable in variables}
        for start in range(0, len(dates), chunk_time):
            stop = min(start + chunk_time, len(dates))
            indices = [lookup[date] for date in dates[start:stop]]
            base = _take(cache["fields/__phase1_physical"], indices)[:, channels]
            truth = _take(cache["fields/y"], indices)[:, channels]
            valid = np.isfinite(base) & np.isfinite(truth)
            if "__target_valid_mask" in cache["fields"]:
                mask = _take(cache["fields/__target_valid_mask"], indices)
                mask = np.broadcast_to(mask, (len(indices), len(source["variables"]), *base.shape[-2:]))
                valid &= mask[:, channels].astype(bool)
            _same(prediction["baseline"][start:stop], base, "baseline values")
            _same(prediction["truth"][start:stop], truth, "target values")
            _same(prediction["valid"][start:stop], valid, "mask values")
            for label in ("raw", "processed"):
                members = prediction[f"{label}_members"][start:stop]
                if members.shape != (stop - start, member_count, len(variables), *base.shape[-2:]):
                    raise AcceptanceError("Physical member dimensions mismatch.")
                if np.any(valid[:, None] & ~np.isfinite(members)):
                    raise AcceptanceError("Nonfinite physical member on valid support.")
                mean = members.mean(axis=1, dtype=np.float64)
                delta = np.abs(prediction[f"{label}_ensemble_mean"][start:stop].astype(np.float64) - mean)
                tolerance = 4 * member_count * np.finfo(members.dtype).eps * np.maximum(np.abs(mean), 1.)
                if np.any(valid & (delta > tolerance)):
                    raise AcceptanceError("Stored mean conflicts with physical-member reconstruction.")
                mean_roundoff[label] = max(mean_roundoff[label], float(delta[valid].max()) if valid.any() else 0.)
                if label == "raw":
                    for channel, variable in enumerate(variables):
                        corrections[variable] |= bool(np.any(valid[:, channel] & (mean[:, channel] != base[:, channel])))
            for channel, variable in enumerate(variables):
                for name, values in (("baseline", base[:, channel]), ("target", truth[:, channel]), ("mask", valid[:, channel])):
                    hashes[variable][name].update(np.ascontiguousarray(values).tobytes())
        report = {
            "version": EVIDENCE_VERSION, "authenticated": True,
            "prediction_path": str(path), "prediction_sha256": manifest["sha256"],
            "cache_path": str(source_path), "cache_sha256": source["cache_sha256"],
            "cache_manifest_sha256": cache_auth["manifest_sha256"],
            "phase1_sha256": source["phase1_checkpoint_sha256"],
            "checkpoint_path": str(checkpoint_path.resolve()), "checkpoint_sha256": authentication["checkpoint_sha256"],
            "run_contract_sha256": run_hash, "config_sha256": config_fingerprint(run_contract["source_config"]),
            "normalization_sha256": target_contract["normalizer_fingerprint"], "target_contract": target_contract,
            "head": head, "training_seed": seed, "variables": variables,
            "formulation": formulation, "recipe_sha256": config_fingerprint(run_contract["recipe"]),
            "refinement_config_sha256": config_fingerprint(run_contract["refinement_config"]),
            "source_code_fingerprints": run_contract.get("source_code_fingerprints"),
            "units": {variable: source["units"][variable] for variable in variables},
            "calendar": source["calendar"], "date_ids": [date[:10] for date in dates],
            "source_timestamps": dates, "sample_ids": ids,
            "dates_sha256": canonical_sha256([date[:10] for date in dates]),
            "coordinates_sha256": canonical_sha256({"lat": prediction["latitude"][:].tolist(), "lon": prediction["longitude"][:].tolist()}),
            "field_sha256": {variable: {name: digest.hexdigest() for name, digest in fields.items()} for variable, fields in hashes.items()},
            "ensemble_size": member_count, "sampling_seed": int(prediction.attrs["sampling_seed"]),
            "canonical_member_batch": int(prediction.attrs["canonical_member_batch"]),
            "random_stream_version": str(prediction.attrs["random_stream_version"]),
            "nonnegative_strategy": str(prediction.attrs["nonnegative_strategy"]),
            "sampling_settings": manifest.get("sampling_settings"),
            "mean_component_used": mean_used, "deterministic_control_sha256": mean_fingerprint,
            "effective_sampled_correction": corrections, "full_training_verified": not full_reasons,
            "full_scope_limitations": full_reasons,
            "selected_checkpoint_epoch": int(payload["progress"].get("epoch", 0)),
            "completed_training_epochs": run_completion["completed_run_epoch"],
            "completed_run_updates": run_completion["completed_run_updates"],
            "completed_run_authentication": run_completion,
            "physical_member_mean_max_roundoff": mean_roundoff,
            "fit_years": registered_years(contract, "component_fit") if not full_reasons else [],
            "provenance_note": "Cache/forecast values compared exactly; means checked within finite-precision summation bounds. Phase-1 end-to-end independence remains unverified.",
        }
    if mean_used:
        report["mean_control_authentication"] = authenticate_mean_control(
            path, manifest, target, formulation, variables, source, chunk_time=chunk_time)
    del target
    return report


def authenticate_mean_control(path, manifest, target, formulation, variables, source, *, chunk_time=16):
    """Replay only the frozen deterministic component on authenticated conditioning.

    Cross-device floating point convolution can differ by roundoff. The explicit
    tolerance is confined to this replay; cached truth, baseline and masks above
    must still compare bitwise, and no scientific metric uses a tolerance.
    """
    prepared_path = Path(manifest.get("prepared_cache_path") or "")
    if not prepared_path.is_file() or file_sha256(prepared_path) != manifest.get("prepared_cache_sha256"):
        raise AcceptanceError("Frozen mean replay requires its authenticated prepared conditioning cache.")
    prepared_manifest = _json(prepared_path.with_suffix(".manifest.json"))
    if (prepared_manifest.get("complete") is not True
            or prepared_manifest.get("sha256") != manifest["prepared_cache_sha256"]
            or prepared_manifest.get("identity", {}).get("source_cache_sha256") != source["cache_sha256"]):
        raise AcceptanceError("Prepared conditioning provenance mismatch for frozen mean replay.")
    target.eval()
    maximum = {"unbounded": 0., "physical": 0.}
    with h5py.File(path, "r") as prediction, h5py.File(prepared_path, "r") as prepared:
        prepared_dates = prepared["dates"].asstr()[:].tolist()
        lookup = {date: i for i, date in enumerate(prepared_dates)}
        dates = prediction["dates"].asstr()[:].tolist()
        if any(date not in lookup for date in dates) or len(lookup) != len(prepared_dates):
            raise AcceptanceError("Mean replay conditioning dates missing or duplicated.")
        if "mean_control_unbounded" not in prediction:
            raise AcceptanceError("Signed and postprocessed deterministic controls must both be retained.")
        nonnegative = [prepared_manifest["nonnegative"][prepared_manifest["variables"].index(v)] for v in variables]
        # Match runner's one-domain control execution; CPU may differ from GPU only by roundoff.
        for row, date in enumerate(dates):
            conditioning = torch.from_numpy(prepared["conditioning"][lookup[date]][None])
            baseline = torch.from_numpy(prediction["baseline"][row:row+1])
            with torch.inference_mode():
                raw = (target.inference_mean(baseline, conditioning) if formulation == "crossfit_remainder"
                       else baseline.float() + target.mean_corrector(conditioning))
                physical = runner.physical_postprocess(raw[:, None], nonnegative,
                                                       str(prediction.attrs["nonnegative_strategy"]))["members"][:, 0]
            valid = prediction["valid"][row:row+1].astype(bool)
            for label, dataset, value in (
                    ("unbounded", "mean_control_unbounded", raw),
                    ("physical", "mean_control", physical)):
                expected = value.cpu().numpy()
                actual = prediction[dataset][row:row+1]
                delta = np.abs(actual.astype(np.float64) - expected)
                tolerance = 1e-5 + 1e-6 * np.abs(expected)
                if np.any(valid & (~np.isfinite(actual) | (delta > tolerance))):
                    raise AcceptanceError(f"Frozen deterministic {label} control differs from replay.")
                if valid.any():
                    maximum[label] = max(maximum[label], float(delta[valid].max()))
    return {"status": "complete", "prepared_cache_path": str(prepared_path.resolve()),
            "prepared_cache_sha256": manifest["prepared_cache_sha256"],
            "maximum_absolute_roundoff": maximum, "replay_absolute_tolerance": 1e-5,
            "replay_relative_tolerance": 1e-6,
            "phase1_forward_calls": 0, "control": "physical one-member postprocessed deterministic point"}


def exact_distribution_diagnostics(raw, processed, baseline, target, valid, masks):
    """Exact physical-member and mean quantiles; no spatial subsampling."""
    levels = [0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
    result = {"quantile_levels": levels, "scope": "all authenticated valid observations",
              "physical_member_order": "member inverse transformation precedes all averaging",
              "members": {"raw": [], "processed": []}, "regional_ensemble_mean": {}}
    target_values, baseline_values = target[valid], baseline[valid]
    result["target_quantiles"] = np.quantile(target_values, levels).tolist()
    result["phase1_quantiles"] = np.quantile(baseline_values, levels).tolist()
    for label, members in (("raw", raw), ("processed", processed)):
        for member in range(members.shape[1]):
            values = members[:, member][valid]
            result["members"][label].append({
                "member": member, "quantiles": np.quantile(values, levels).tolist(),
                "negative_values": negative_value_summary(values)})
    means = {"raw": raw.mean(axis=1, dtype=np.float64),
             "processed": processed.mean(axis=1, dtype=np.float64)}
    for name, mask in masks.items():
        support = valid & mask[None]
        if not support.any():
            result["regional_ensemble_mean"][name] = {"status": "undefined_no_valid_cells"}
            continue
        result["regional_ensemble_mean"][name] = {
            label: _quantile_metrics(mean[support], baseline[support], target[support])
            for label, mean in means.items()
        }
    return result


def probabilistic_constraint_diagnostics(raw, processed, baseline, target, valid,
                                       masks, *, variable, threshold, chunk_time=16):
    """Raw/post variance, CRPS, wet Brier/reliability and negative impacts."""
    labels = ("raw", "processed")
    fields = ("crps", "variance", "negative_count", "negative_deficit",
              "member_change_count", "absolute_member_change", "mean_change")
    totals = {name: {"valid_count": 0, **{f"{label}_{field}": 0. for label in labels for field in fields},
                    "wet": {}} for name in masks}
    thresholds = list(dict.fromkeys((float(threshold), 1.0))) if variable == "pr" else []
    for item in totals.values():
        for wet_threshold in thresholds:
            item["wet"][str(wet_threshold)] = {
                label: {"brier_sum": 0., "member_wet_count": 0, "member_wet_sum": 0.,
                        "reliability_count": np.zeros(10, dtype=np.int64),
                        "reliability_probability_sum": np.zeros(10),
                        "reliability_event_sum": np.zeros(10)}
                for label in labels}
    member_count = raw.shape[1]
    for start in range(0, len(target), chunk_time):
        sl = slice(start, min(start + chunk_time, len(target)))
        truth, support = target[sl], valid[sl]
        for label, source in (("raw", raw), ("processed", processed)):
            members = source[sl].astype(np.float64)
            before = raw[sl].astype(np.float64)
            shift = members - before
            negative = members < 0
            values = {
                "crps": empirical_crps(members, truth),
                "variance": np.var(members, axis=1, ddof=1),
                "negative_count": negative.sum(axis=1),
                "negative_deficit": np.where(negative, -members, 0).sum(axis=1),
                "member_change_count": (shift != 0).sum(axis=1),
                "absolute_member_change": np.abs(shift).sum(axis=1),
                "mean_change": shift.mean(axis=1),
            }
            wet_values = {}
            for wet_threshold in thresholds:
                wet = members >= wet_threshold
                probability = wet.mean(axis=1)
                event = truth >= wet_threshold
                wet_values[str(wet_threshold)] = {
                    "probability": probability, "event": event,
                    "bins": np.minimum((probability * 10).astype(np.int64), 9),
                    "brier": np.square(probability - event),
                    "member_count": wet.sum(axis=1),
                    "member_sum": np.where(wet, members, 0).sum(axis=1),
                }
            for name, mask in masks.items():
                selection = support & mask[None]
                count = int(selection.sum())
                if label == "raw":
                    totals[name]["valid_count"] += count
                for field, value in values.items():
                    totals[name][f"{label}_{field}"] += float(value[selection].sum())
                for wet_threshold, wet in wet_values.items():
                    state = totals[name]["wet"][wet_threshold][label]
                    state["brier_sum"] += float(wet["brier"][selection].sum())
                    state["member_wet_count"] += int(wet["member_count"][selection].sum())
                    state["member_wet_sum"] += float(wet["member_sum"][selection].sum())
                    bins = wet["bins"][selection]
                    state["reliability_count"] += np.bincount(bins, minlength=10)
                    state["reliability_probability_sum"] += np.bincount(bins, weights=wet["probability"][selection], minlength=10)
                    state["reliability_event_sum"] += np.bincount(bins, weights=wet["event"][selection], minlength=10)
    result = {}
    for name, state in totals.items():
        count = state["valid_count"]
        member_n = count * member_count
        item = {"valid_cell_days": count, "physical_member_observations": member_n}
        for label in labels:
            negatives = state[f"{label}_negative_count"]
            item[label] = {
                "crps": state[f"{label}_crps"] / count if count else None,
                "mean_member_sample_variance": state[f"{label}_variance"] / count if count else None,
                "negative_fraction": negatives / member_n if member_n else None,
                "mean_negative_magnitude": state[f"{label}_negative_deficit"] / negatives if negatives else 0. if member_n else None,
                "mean_negative_deficit": state[f"{label}_negative_deficit"] / member_n if member_n else None,
                "fraction_changed_from_raw": state[f"{label}_member_change_count"] / member_n if member_n else None,
                "mean_absolute_change_from_raw": state[f"{label}_absolute_member_change"] / member_n if member_n else None,
                "signed_mean_change_from_raw": state[f"{label}_mean_change"] / count if count else None,
            }
        item["wet_day"] = {}
        for wet_threshold, stages in state["wet"].items():
            entry = {"threshold_mm_day": float(wet_threshold),
                     "role": "configured_primary" if float(wet_threshold) == threshold else "1mm_sensitivity"}
            for label, stage in stages.items():
                bin_count = stage["reliability_count"]
                entry[label] = {
                    "brier_score": stage["brier_sum"] / count if count else None,
                    "pooled_member_wet_frequency": stage["member_wet_count"] / member_n if member_n else None,
                    "pooled_member_wet_intensity": stage["member_wet_sum"] / stage["member_wet_count"] if stage["member_wet_count"] else None,
                    "reliability": [
                        {"lower": index / 10, "upper": (index + 1) / 10,
                         "count": int(bin_count[index]),
                         "mean_probability": float(stage["reliability_probability_sum"][index] / bin_count[index]) if bin_count[index] else None,
                         "observed_frequency": float(stage["reliability_event_sum"][index] / bin_count[index]) if bin_count[index] else None}
                        for index in range(10)],
                }
            item["wet_day"][wet_threshold] = entry
        result[name] = item
    return {"regions": result, "variance_definition": "sample variance across issued physical members (ddof=1)",
            "wet_event": "precipitation >= threshold; valid zeros retained",
            "projection_member_coupling": "whole issued ensembles retained; no member-composition bootstrap",
            "signed_raw_corrections": "raw_members - immutable Phase-1 baseline"}


def extreme_year_draws(prediction, baseline, target, years, valid, plan):
    """Exact per-seed metric values; merge training-seed draws only afterward."""
    statistics = YearExtremeStatistics(prediction, baseline, target, years, valid)
    point = statistics.evaluate()
    result = {"years": statistics.years.tolist(),
              "per_seed_point": {metric: {product: float(point[metric][product][0])
                                          for product in ("candidate", "reference")}
                                 for metric in ("p99_absolute_error", "observed_p99_event_rmse")}}
    if np.array_equal(statistics.years, plan["years"]):
        # Repeated year vectors also produce identical exact quantiles.
        unique, inverse = np.unique(plan["year_weights"], axis=0, return_inverse=True)
        values = statistics.evaluate(unique)
        result["year_draw_values"] = {
            metric: {product: values[metric][product][inverse].tolist()
                     for product in ("candidate", "reference")}
            for metric in ("p99_absolute_error", "observed_p99_event_rmse")}
        result["resampling_plan_sha256"] = plan["sha256"]
    else:
        result["year_draw_values"] = None
        result["limitation"] = "Missing registered year blocks; no complete-family extreme interval."
    return result


def _time_coordinate(timestamps, calendar):
    import cftime
    import re
    calendar_type = {
        "noleap": cftime.DatetimeNoLeap, "365_day": cftime.DatetimeNoLeap,
        "360_day": cftime.Datetime360Day, "all_leap": cftime.DatetimeAllLeap,
        "366_day": cftime.DatetimeAllLeap, "julian": cftime.DatetimeJulian,
        "standard": cftime.DatetimeGregorian, "gregorian": cftime.DatetimeGregorian,
        "proleptic_gregorian": cftime.DatetimeProlepticGregorian,
    }.get(calendar)
    if calendar_type is None:
        raise AcceptanceError(f"Unsupported authenticated calendar {calendar}.")
    values = []
    for stamp in timestamps:
        pieces = re.split("[-T: ]", stamp)
        year, month, day = map(int, pieces[:3])
        hour, minute = (int(pieces[3]) if len(pieces) > 3 else 0,
                        int(pieces[4]) if len(pieces) > 4 else 0)
        seconds = float(pieces[5]) if len(pieces) > 5 else 0.
        values.append(calendar_type(year, month, day, hour, minute,
                                    int(seconds), int(round((seconds % 1) * 1e6))))
    return np.asarray(values, dtype=object)


def _diagnostic_reference(path, *, status="complete", **details):
    return {"status": status, "artifact": str(Path(path).resolve()),
            "artifact_sha256": file_sha256(path), **details}


def nested_member_diagnostics(paths, primary_authentication, cache_path, contract,
                              cache_authentication, output):
    if not paths:
        return {"status": "missing", "reason": "No independently issued nested 10/20/50 diagnostic ensembles supplied."}
    reports = [authenticate_prediction(path, cache_path, contract,
                                       cache_authentication=cache_authentication) for path in paths]
    counts = [report["ensemble_size"] for report in reports]
    expected = contract["sampling"]["diagnostic_nested_member_counts"]
    if sorted(counts) != sorted(expected) or len(counts) != len(set(counts)):
        raise AcceptanceError("Nested diagnostic files must contain exactly 10, 20 and 50 members.")
    reports.sort(key=lambda report: report["ensemble_size"])
    first = reports[0]
    expected_dates = runner.diagnostic_dates([
        cache_authentication["manifest"]["splits"]["dates"][index]
        for index in cache_authentication["manifest"]["splits"]["validation_indices"]])
    for report in reports:
        for name in ("checkpoint_sha256", "run_contract_sha256", "sampling_seed",
                     "canonical_member_batch", "random_stream_version", "variables"):
            if report[name] != primary_authentication[name]:
                raise AcceptanceError(f"Nested diagnostic {name} differs from the selected candidate.")
        if report["sampling_settings"] != primary_authentication["sampling_settings"]:
            raise AcceptanceError("Nested ensembles used different sampling settings.")
        if report["source_timestamps"] != expected_dates:
            raise AcceptanceError("Nested ensemble dates are not the registered representative diagnostic set.")
        if report["source_timestamps"] != first["source_timestamps"]:
            raise AcceptanceError("Nested diagnostic dates are not paired.")
    with h5py.File(reports[-1]["prediction_path"], "r") as largest:
        for report in reports[:-1]:
            with h5py.File(report["prediction_path"], "r") as smaller:
                for index in range(len(expected_dates)):
                    if not np.array_equal(smaller["raw_members"][index],
                                          largest["raw_members"][index, :report["ensemble_size"]]):
                        raise AcceptanceError("Nested raw sampler member prefixes do not share exact initial/member identities.")
    # Bind the sensitivity ensembles to the actual full acceptance members,
    # not only to each other. Whole-ensemble constraints can change prefixes
    # when issued sizes differ, so physical processed equality is checked only
    # at equal member count.
    with h5py.File(primary_authentication["prediction_path"], "r") as primary:
        primary_lookup = {date: index for index, date in enumerate(primary["dates"].asstr()[:])}
        for report in reports:
            with h5py.File(report["prediction_path"], "r") as diagnostic:
                for row, date in enumerate(expected_dates):
                    if date not in primary_lookup:
                        raise AcceptanceError("Nested date missing from the primary accepted ensemble.")
                    index = primary_lookup[date]
                    count = min(primary_authentication["ensemble_size"], report["ensemble_size"])
                    if not np.array_equal(primary["raw_members"][index, :count],
                                          diagnostic["raw_members"][row, :count], equal_nan=True):
                        raise AcceptanceError("Primary and nested raw member prefixes differ.")
                    if primary_authentication["ensemble_size"] == report["ensemble_size"]:
                        if not np.array_equal(primary["processed_members"][index],
                                              diagnostic["processed_members"][row], equal_nan=True):
                            raise AcceptanceError("Equal-size primary and nested processed members differ.")
    result = {"status": "complete", "raw_prefix_equality": True,
              "processed_prefix_equality_required": False,
              "primary_raw_prefix_equality": True, "equal_size_primary_processed_equality": True,
              "reason": "Mean-preserving projections may couple the full issued member set.",
              "source_dates": expected_dates, "ensembles": []}
    for report in reports:
        with h5py.File(report["prediction_path"], "r") as f:
            entry = {"member_count": report["ensemble_size"], "prediction_sha256": report["prediction_sha256"],
                     "variables": {}}
            for channel, variable in enumerate(report["variables"]):
                raw = f["raw_members"][:, :, channel]
                post = f["processed_members"][:, :, channel]
                baseline, truth = f["baseline"][:, channel], f["truth"][:, channel]
                valid = f["valid"][:, channel].astype(bool)
                masks = acceptance_regions(f["latitude"][:], f["longitude"][:], contract)
                stats = collect_year_statistics(
                    post, baseline, truth, [int(date[:4]) for date in report["date_ids"]],
                    f["latitude"][:], f["longitude"][:], valid_mask=valid)
                point = {name: _product_metrics(fields, np.ones(len(stats["years"])), masks)
                         for name, fields in stats["products"].items()}
                impacts = probabilistic_constraint_diagnostics(
                    raw, post, baseline, truth, valid, masks, variable=variable,
                    threshold=contract["diagnostics"]["configured_wet_day_threshold_mm_day"])
                entry["variables"][variable] = {"point_metrics": point, "probabilistic_raw_post": impacts}
            result["ensembles"].append(entry)
    write_json(output, result)
    return _diagnostic_reference(output, raw_prefix_equality=True)


def collect_prediction_evidence(prediction_path, cache_path, contract, output_dir, *,
                                chunk_time=16, nested_paths=(), make_plots=True):
    """Authenticate first, then evaluate one variable at a time in physical units."""
    contract = load_contract(contract) if not isinstance(contract, dict) else contract
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise AcceptanceError("Evidence output directory must be new or empty.")
    output.mkdir(parents=True, exist_ok=True)
    cache_auth = authenticate_cache(cache_path)
    authentication = authenticate_prediction(prediction_path, cache_path, contract,
                                             chunk_time=chunk_time, cache_authentication=cache_auth)
    write_json(output / "authentication.json", authentication)
    nested = nested_member_diagnostics(nested_paths, authentication, cache_path, contract,
                                       cache_auth, output / "nested_member_diagnostics.json")
    plan = paired_resampling_plan(contract)
    records, auxiliary, extreme_files, map_files = {}, {}, {}, {}
    with h5py.File(prediction_path, "r") as f:
        lat, lon = f["latitude"][:], f["longitude"][:]
        years = np.array([int(date[:4]) for date in authentication["date_ids"]])
        timestamps = _time_coordinate(authentication["source_timestamps"], authentication["calendar"])
        masks = acceptance_regions(lat, lon, contract)
        for channel, variable in enumerate(authentication["variables"]):
            print(f"Evaluating {variable}: authenticated physical members, full configured support", flush=True)
            raw = f["raw_members"][:, :, channel]
            members = f["processed_members"][:, :, channel]
            baseline, target = f["baseline"][:, channel], f["truth"][:, channel]
            valid = f["valid"][:, channel].astype(bool)
            control = f["mean_control"][:, channel] if authentication["mean_component_used"] else None
            if np.any(valid & (~np.isfinite(target) | ~np.isfinite(baseline))):
                raise AcceptanceError("Stored support includes invalid target/baseline cells.")
            coords = {"time": timestamps, "member": np.arange(members.shape[1]), "lat": lat, "lon": lon}
            member_array = xr.DataArray(members, dims=("time", "member", "lat", "lon"),
                                        coords=coords, attrs={"units": contract["units"][variable]})
            raw_array = xr.DataArray(raw, dims=member_array.dims, coords=coords, attrs=member_array.attrs)
            scalar_coords = {name: coords[name] for name in ("time", "lat", "lon")}
            base_array = xr.DataArray(np.where(valid, baseline, np.nan), dims=("time", "lat", "lon"),
                                      coords=scalar_coords, attrs=member_array.attrs)
            target_array = xr.DataArray(np.where(valid, target, np.nan), dims=base_array.dims,
                                        coords=scalar_coords, attrs=member_array.attrs)
            metrics, maps = evaluate_variable(
                member_array, base_array, target_array, member_dim="member", variable=variable,
                chunk_time=chunk_time, wet_day_threshold=contract["diagnostics"]["configured_wet_day_threshold_mm_day"],
                unbounded_members=raw_array)
            statistics, raw_statistics = None, None
            for start in range(0, len(years), chunk_time):
                sl = slice(start, min(start + chunk_time, len(years)))
                kwargs = {"valid_mask": valid[sl],
                          "deterministic_control": None if control is None else control[sl]}
                statistics = merge_year_statistics(statistics, collect_year_statistics(
                    members[sl], baseline[sl], target[sl], years[sl], lat, lon, **kwargs))
                raw_statistics = merge_year_statistics(raw_statistics, collect_year_statistics(
                    raw[sl], baseline[sl], target[sl], years[sl], lat, lon, **kwargs))
            detailed_regions = {}
            for regime, bounds in {"pooled": [int(years.min()), int(years.max())],
                                   **contract["splits"]["component_validation"]}.items():
                weights = ((statistics["years"] >= bounds[0]) & (statistics["years"] <= bounds[1])).astype(float)
                detailed_regions[regime] = {
                    name: _product_metrics(fields, weights, masks)
                    for name, fields in statistics["products"].items()}
                detailed_regions[regime]["raw"] = _product_metrics(
                    raw_statistics["products"]["refined"], weights, masks)
            metrics["registered_regions_and_regimes"] = detailed_regions
            print(f"{variable}: exact physical-member quantiles and raw/post diagnostics", flush=True)
            distributions = exact_distribution_diagnostics(raw, members, baseline, target, valid, masks)
            impacts = probabilistic_constraint_diagnostics(
                raw, members, baseline, target, valid, masks, variable=variable,
                threshold=contract["diagnostics"]["configured_wet_day_threshold_mm_day"],
                chunk_time=chunk_time)
            metrics["exact_physical_distributions"] = distributions
            metrics["raw_postprocessed_probability_and_constraints"] = impacts
            mean = members.mean(axis=1, dtype=np.float64)
            extremes = extreme_year_draws(mean, baseline, target, years, valid, plan)
            extreme_path = output / f"{variable}_exact_extreme_year_draws.json"
            write_json(extreme_path, extremes)
            extreme_files[variable] = _diagnostic_reference(extreme_path)
            auxiliary[variable] = {
                metric: {"per_seed": {str(authentication["training_seed"]): value}}
                for metric, value in extremes["per_seed_point"].items()}
            summary = {
                "evaluation_contract": EVIDENCE_VERSION,
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "refinement_type": authentication["head"],
                "checkpoint_sha256": authentication["checkpoint_sha256"],
                "scope": "component_validation" if authentication["full_training_verified"] else "provisional_screening",
                "variables": {variable: metrics},
            }
            products = write_evaluation_outputs(summary, {variable: maps}, lat=lat, lon=lon,
                                                 output_dir=output / variable, make_plots=make_plots)
            diagnostic_path = products["summary_json"]
            common = _diagnostic_reference(diagnostic_path)
            implemented = {
                "seasonal_metrics", "regime_metrics", "signed_bias", "climatological_rmse",
                "pattern_agreement", "wet_day_frequency_intensity", "wet_day_brier_reliability",
                "member_quantiles_extremes", "p99_error_and_observed_p99_event_rmse",
                "ensemble_spread_error", "finite_ensemble_interval_coverage",
                "member_spatial_dependence", "raw_negative_and_constraint_effects"}
            diagnostics = {
                name: dict(common) if name in implemented else
                {"status": "missing", "reason": "No implemented diagnostic provider."}
                for name in contract["diagnostics"]["required"]}
            if variable != "pr":
                for name in ("wet_day_frequency_intensity", "wet_day_brier_reliability"):
                    diagnostics[name] = {**common, "status": "not_applicable",
                                         "reason": "Temperature has no precipitation wet-day event."}
            if any(value["time_count"] == 0 for value in metrics["seasonal"].values()):
                diagnostics["seasonal_metrics"]["status"] = "incomplete"
            if any(not np.any((years >= start) & (years <= end))
                   for start, end in contract["splits"]["component_validation"].values()):
                diagnostics["regime_metrics"]["status"] = "incomplete"
            diagnostics["nested_10_20_50_member_sensitivity"] = nested
            diagnostics["independently_trained_seed_results"] = {
                "status": "pending_merge", "reason": "Only one independently trained checkpoint is collected here."}
            diagnostics["p99_error_and_observed_p99_event_rmse"] = extreme_files[variable]
            field_hashes = authentication["field_sha256"][variable]
            metadata = {
                **{key: authentication[key] for key in (
                    "phase1_sha256", "dates_sha256", "coordinates_sha256", "config_sha256",
                    "normalization_sha256", "checkpoint_sha256", "date_ids", "calendar",
                    "training_seed", "mean_component_used", "deterministic_control_sha256")},
                "baseline_sha256": field_hashes["baseline"], "target_sha256": field_hashes["target"],
                "mask_sha256": field_hashes["mask"], "units": contract["units"][variable],
                "authenticated": True,
                "data_scope": "component_validation" if authentication["full_training_verified"] else "screening",
                "fit_years": authentication["fit_years"],
                "sampling_seed_ids": [
                    f"{authentication['random_stream_version']}|experiment={authentication['sampling_seed']}|member={index}"
                    for index in range(authentication["ensemble_size"])],
                "raw_members_saved": True,
                "effective_sampled_correction": authentication["effective_sampled_correction"][variable],
                "edge_fallback_or_suppression": False, "diagnostics": diagnostics,
                "authentication_artifact": str((output / "authentication.json").resolve()),
                "full_scope_limitations": authentication["full_scope_limitations"],
                "sampling_settings": authentication["sampling_settings"],
            }
            statistics["metadata"].update(metadata)
            raw_statistics["metadata"].update(metadata)
            stat_path = output / f"{variable}_year_statistics.npz"
            records[variable] = {"path": str(stat_path), "sha256": save_year_statistics(stat_path, statistics)}
            save_year_statistics(output / f"{variable}_raw_year_statistics.npz", raw_statistics)
            map_files[variable] = products
            del raw, members, baseline, target, valid, control, mean
            del member_array, raw_array, base_array, target_array, statistics, raw_statistics
            gc.collect()
    evidence = {
        "version": EVIDENCE_VERSION, "contract_sha256": canonical_hash(contract),
        "scope": "component_validation", "production_accepted": False,
        "records": {authentication["head"]: {str(authentication["training_seed"]): records}},
        "auxiliary_metrics": {authentication["head"]: auxiliary},
        "exact_extreme_year_draws": extreme_files, "diagnostic_outputs": map_files,
        "diagnostic_output_sha256": {variable: {name: file_sha256(path) for name, path in products.items()}
                                     for variable, products in map_files.items()},
        "authentication": _diagnostic_reference(output / "authentication.json"),
        "limitations": authentication["full_scope_limitations"],
    }
    write_json(output / "evidence.json", evidence)
    write_json(output / "completion.json", {
        "version": EVIDENCE_VERSION, "complete": True, "operation": "collect",
        "contract_sha256": canonical_hash(contract),
        "artifact_sha256": file_sha256(output / "evidence.json"), "plots_enabled": make_plots})
    return evidence


def merge_extreme_seed_draws(per_seed, registered_seeds, metric, plan, near_zero_scale):
    """Average paired trained-seed metrics before taking a relative change."""
    values = {}
    for product in ("reference", "candidate"):
        by_seed = np.asarray([
            per_seed[seed]["year_draw_values"][metric][product] for seed in registered_seeds])
        if by_seed.shape != (len(registered_seeds), len(plan["year_weights"])):
            raise AcceptanceError("Exact extreme-draw dimensions differ from the frozen plan.")
        if not np.isfinite(by_seed).all():
            raise AcceptanceError("Nonfinite exact extreme metric in a year/seed draw.")
        values[product] = np.take_along_axis(by_seed, plan["seed_indices"].T, axis=0).mean(axis=0)
    return ((values["candidate"] - values["reference"])
            / np.maximum(np.abs(values["reference"]), near_zero_scale)).tolist()


def merge_prediction_evidence(evidence_paths, contract, output_dir, *, make_plots=True):
    """Merge independent training seeds without pooling their ensemble members."""
    contract = load_contract(contract) if not isinstance(contract, dict) else contract
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise AcceptanceError("Merged-evidence output must be new or empty.")
    output.mkdir(parents=True, exist_ok=True)
    combined = {"version": EVIDENCE_VERSION, "contract_sha256": canonical_hash(contract),
                "scope": "component_validation", "records": {}, "auxiliary_metrics": {},
                "input_evidence": []}
    records, extremes, sources, authentications = {}, {}, {}, {}
    for path in evidence_paths:
        evidence = _json(path)
        if evidence.get("version") != EVIDENCE_VERSION or evidence.get("contract_sha256") != canonical_hash(contract):
            raise AcceptanceError("Evidence manifest version/frozen contract mismatch.")
        auth_reference = evidence["authentication"]
        if file_sha256(auth_reference["artifact"]) != auth_reference["artifact_sha256"]:
            raise AcceptanceError("Authentication report changed.")
        authentication = _json(auth_reference["artifact"])
        for variable, products in evidence.get("diagnostic_outputs", {}).items():
            for name, product_path in products.items():
                expected_hash = evidence.get("diagnostic_output_sha256", {}).get(variable, {}).get(name)
                if not expected_hash or file_sha256(product_path) != expected_hash:
                    raise AcceptanceError("A retained diagnostic map, summary or table changed after collection.")
        if file_sha256(authentication["prediction_path"]) != authentication["prediction_sha256"]:
            raise AcceptanceError("Retained physical ensemble bytes changed after collection.")
        for head, seeds in evidence["records"].items():
            for seed, variables in seeds.items():
                if seed in records.get(head, {}):
                    raise AcceptanceError("Duplicate head/seed evidence; recipes cannot be silently replaced.")
                if authentication["head"] != head or str(authentication["training_seed"]) != seed:
                    raise AcceptanceError("Evidence head/seed differs from authenticated checkpoint.")
                records.setdefault(head, {})[seed] = {}
                extremes.setdefault(head, {})[seed] = {}
                sources.setdefault(head, {})[seed] = evidence
                authentications.setdefault(head, {})[seed] = authentication
                for variable, entry in variables.items():
                    records[head][seed][variable] = load_year_statistics(entry["path"], entry["sha256"])
                    reference = evidence["exact_extreme_year_draws"][variable]
                    if file_sha256(reference["artifact"]) != reference["artifact_sha256"]:
                        raise AcceptanceError("Exact extreme-draw artifact changed.")
                    extremes[head][seed][variable] = _json(reference["artifact"])
        combined["input_evidence"].append({"path": str(Path(path).resolve()), "sha256": file_sha256(path)})
    plan = paired_resampling_plan(contract)
    registered_seeds = [str(seed) for seed in contract["training_seeds"]]
    point_rows, individual = comparison_rows(records, contract)
    seed_report_path = output / "independent_training_seed_results.json"
    write_json(seed_report_path, {
        "definition": "Metrics per independently trained checkpoint; ensembles are never pooled across trained models.",
        "registered_training_seeds": registered_seeds,
        "records": {head: {seed: {key: report.get(key) for key in (
            "checkpoint_sha256", "run_contract_sha256", "completed_training_epochs",
            "full_training_verified", "full_scope_limitations")} for seed, report in seeds.items()}
                    for head, seeds in authentications.items()},
        "per_seed_metrics": individual, "aggregate_point_rows": point_rows,
    })
    for head, seeds in records.items():
        complete = set(seeds) == set(registered_seeds)
        if complete:
            prototype = authentications[head][registered_seeds[0]]
            for seed in registered_seeds[1:]:
                current = authentications[head][seed]
                for name in ("ensemble_size", "sampling_seed", "sampling_settings",
                             "recipe_sha256", "refinement_config_sha256", "formulation"):
                    if current.get(name) != prototype.get(name):
                        raise AcceptanceError(f"Independent seeds for {head} used different registered {name}.")
            if len({authentications[head][seed]["checkpoint_sha256"] for seed in registered_seeds}) != len(registered_seeds):
                raise AcceptanceError("Independent training seeds cannot reuse the same checkpoint bytes.")
        combined["records"][head] = {}
        combined["auxiliary_metrics"][head] = {}
        for seed, variables in seeds.items():
            combined["records"][head][seed] = {}
            for variable, statistics in variables.items():
                statistics = copy.deepcopy(statistics)
                statistics["metadata"]["diagnostics"]["independently_trained_seed_results"] = _diagnostic_reference(
                    seed_report_path, status="complete" if complete else "incomplete",
                    available_training_seeds=sorted(seeds), expected_training_seeds=registered_seeds)
                destination = output / "year_statistics" / head / seed / f"{variable}.npz"
                combined["records"][head][seed][variable] = {
                    "path": str(destination), "sha256": save_year_statistics(destination, statistics)}
        for variable in contract["variables"]:
            available = [seed for seed in registered_seeds if variable in extremes[head].get(seed, {})]
            if not available:
                continue
            combined["auxiliary_metrics"][head][variable] = {}
            for metric in ("p99_absolute_error", "observed_p99_event_rmse"):
                item = {"per_seed": {seed: extremes[head][seed][variable]["per_seed_point"][metric] for seed in available}}
                all_draws = complete and len(available) == len(registered_seeds) and all(
                    extremes[head][seed][variable].get("resampling_plan_sha256") == plan["sha256"]
                    and extremes[head][seed][variable].get("year_draw_values") is not None
                    for seed in available)
                if all_draws:
                    item["bootstrap_changes"] = merge_extreme_seed_draws(
                        {seed: extremes[head][seed][variable] for seed in registered_seeds},
                        registered_seeds, metric, plan, contract["near_zero_reference"][variable])
                    item["resampling_plan_sha256"] = plan["sha256"]
                combined["auxiliary_metrics"][head][variable][metric] = item
    write_json(output / "evidence.json", combined)
    report = evaluate_scientific_acceptance(combined, contract)
    write_report(report, output / "acceptance_report")
    if make_plots and sources:
        from examples.CORDEX_ML.utils.evaluate_refinement_outputs import shared_map_limits
        loaded = []
        for head, seeds in sources.items():
            for seed, evidence in seeds.items():
                maps, summaries, coords = {}, {}, None
                for variable, paths in evidence["diagnostic_outputs"].items():
                    with np.load(paths[f"{variable}_numerical_maps"], allow_pickle=False) as archive:
                        coords = archive["lat"], archive["lon"]
                        maps[variable] = {key: archive[key] for key in archive.files if key not in ("lat", "lon")}
                    summaries[variable] = _json(paths["summary_json"])["variables"][variable]
                loaded.append((head, seed, maps, summaries, coords))
        limits = shared_map_limits([item[2] for item in loaded])
        write_json(output / "shared_plot_limits.json", limits)
        for head, seed, maps, summaries, coords in loaded:
            write_evaluation_outputs({"refinement_type": head, "variables": summaries},
                maps, lat=coords[0], lon=coords[1], output_dir=output / "comparable_maps" / head / seed,
                make_plots=True, plot_limits=limits)
    write_json(output / "completion.json", {
        "version": EVIDENCE_VERSION, "complete": True, "operation": "merge",
        "contract_sha256": canonical_hash(contract),
        "artifact_sha256": file_sha256(output / "acceptance_report" / "scientific_acceptance_results.json"),
        "plots_enabled": make_plots})
    return report


def compare_family_maps(selected_evidence, control_evidence, contract, output_dir):
    """Render common scales across both families without combining their gates."""
    from examples.CORDEX_ML.utils.evaluate_refinement_outputs import shared_map_limits
    contract = load_contract(contract) if not isinstance(contract, dict) else contract
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise AcceptanceError("Matched-map output directory must be new or empty.")
    output.mkdir(parents=True, exist_ok=True)
    expected = {(head, str(seed)) for head in contract["heads"] for seed in contract["training_seeds"]}
    collected, inputs = [], []
    for group, path in (("selected", selected_evidence), ("control", control_evidence)):
        path = Path(path).resolve()
        family = _json(path)
        if family.get("contract_sha256") != canonical_hash(contract):
            raise AcceptanceError("Matched-map families identify different scientific contracts.")
        inputs.append({"comparison_group": group, "path": str(path), "sha256": file_sha256(path)})
        found = set()
        for reference in family["input_evidence"]:
            if file_sha256(reference["path"]) != reference["sha256"]:
                raise AcceptanceError("Matched-map source evidence changed.")
            evidence = _json(reference["path"])
            identities = [(head, seed) for head, seeds in evidence["records"].items() for seed in seeds]
            if len(identities) != 1 or identities[0] in found or identities[0] not in expected:
                raise AcceptanceError("Matched maps need exactly one source for every registered head/seed.")
            head, seed = identities[0]; found.add((head, seed))
            maps, summaries, coordinates = {}, {}, None
            for variable in contract["variables"]:
                products = evidence["diagnostic_outputs"][variable]
                for name, product_path in products.items():
                    if file_sha256(product_path) != evidence["diagnostic_output_sha256"][variable][name]:
                        raise AcceptanceError("Matched-map diagnostic file changed.")
                with np.load(products[f"{variable}_numerical_maps"], allow_pickle=False) as archive:
                    coordinate = (archive["lat"].copy(), archive["lon"].copy())
                    if coordinates is not None and any(not np.array_equal(a, b) for a, b in zip(coordinates, coordinate)):
                        raise AcceptanceError("Matched-map variable coordinates differ.")
                    coordinates = coordinate
                    maps[variable] = {key: archive[key] for key in archive.files if key not in ("lat", "lon")}
                summaries[variable] = _json(products["summary_json"])["variables"][variable]
            collected.append((group, head, seed, maps, summaries, coordinates))
        if found != expected:
            raise AcceptanceError("Matched-map comparison is missing registered head/seed sources.")
    prototype = collected[0][-1]
    if any(any(not np.array_equal(a, b) for a, b in zip(item[-1], prototype)) for item in collected):
        raise AcceptanceError("Matched-map families use different coordinate grids.")
    limits = shared_map_limits([item[3] for item in collected])
    write_json(output/"shared_plot_limits.json", limits)
    for group, head, seed, maps, summaries, coordinates in collected:
        write_evaluation_outputs(
            {"refinement_type": f"{group}: {head}", "comparison_group": group,
             "training_seed": int(seed), "scope": "matched_component_validation_maps_only",
             "variables": summaries}, maps, lat=coordinates[0], lon=coordinates[1],
            output_dir=output/group/head/seed, make_plots=True, plot_limits=limits)
    write_json(output/"evidence.json", {"contract_sha256": canonical_hash(contract),
        "input_evidence": inputs, "scope": "plot_comparison_only_no_combined_acceptance_family"})
    report = {"contract_sha256": canonical_hash(contract), "comparison_groups": ["selected", "control"],
              "source_count": len(collected), "shared_plot_limits": str(output/"shared_plot_limits.json"),
              "acceptance_families_combined": False, "production_promotion": False,
              "scope": "registered component validation; 1981-2000 regeneration unexecuted; midcentury quarantined"}
    write_json(output/"comparison.json", report)
    write_json(output/"completion.json", {"version": EVIDENCE_VERSION, "operation": "compare",
        "complete": True, "contract_sha256": canonical_hash(contract),
        "artifact_sha256": file_sha256(output/"comparison.json")})
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--prediction", required=True)
    collect.add_argument("--cache", required=True)
    collect.add_argument("--contract", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--nested", nargs="*", default=[])
    collect.add_argument("--chunk-time", type=int, default=16)
    collect.add_argument("--no-plots", action="store_true")
    merge = sub.add_parser("merge")
    merge.add_argument("--evidence", nargs="+", required=True)
    merge.add_argument("--contract", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--no-plots", action="store_true")
    compare = sub.add_parser("compare-families")
    compare.add_argument("--selected", required=True)
    compare.add_argument("--control", required=True)
    compare.add_argument("--contract", required=True)
    compare.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.operation == "collect":
        evidence = collect_prediction_evidence(args.prediction, args.cache, args.contract, args.output,
            chunk_time=args.chunk_time, nested_paths=args.nested, make_plots=not args.no_plots)
        print(json.dumps({"evidence": str(Path(args.output).resolve() / "evidence.json"),
                          "head_count": len(evidence["records"]), "production_accepted": False}))
    elif args.operation == "merge":
        report = merge_prediction_evidence(args.evidence, args.contract, args.output, make_plots=not args.no_plots)
        print(json.dumps({"status": report["status"], "production_promotion": False}))
    else:
        report = compare_family_maps(args.selected, args.control, args.contract, args.output)
        print(json.dumps(report))


if __name__ == "__main__":
    main()
