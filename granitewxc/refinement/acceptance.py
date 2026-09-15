"""Frozen scientific acceptance for paired CORDEX refinement experiments.

Inputs are physical fields. Whole years and complete ensemble sets stay intact;
pixels and projected members are never treated as independent trials.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
import yaml

from .diagnostics import boundary_region_masks, empirical_crps
from .metrics import (
    ErrorAccumulator, map_difference_metrics, error_metrics_from_sums,
    mean_absolute_climatological_bias_from_sums,
)


class AcceptanceError(ValueError):
    """Evidence is incompatible with the registered scientific comparison."""


def canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_contract(path: str | Path) -> dict:
    path = Path(path)
    payload = (json.loads(path.read_text()) if path.suffix == ".json"
               else yaml.safe_load(path.read_text()))
    if "contract" in payload:
        if canonical_hash(payload["contract"]) != payload.get("sha256"):
            raise AcceptanceError("Frozen acceptance contract hash mismatch.")
        payload = payload["contract"]
    if payload.get("schema_version") != 1:
        raise AcceptanceError("Unsupported acceptance schema.")
    if not payload.get("heads") or not payload.get("variables"):
        raise AcceptanceError("Contract requires heads and variables.")
    return payload


def freeze_acceptance_contract(config_path, artifact_dir) -> dict:
    contract = load_contract(config_path)
    digest = canonical_hash(contract)
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    suffix = f"_v{contract['protocol_revision']}" if contract.get("protocol_revision", 1) > 1 else ""
    path = root / f"frozen_scientific_acceptance{suffix}.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("sha256") != digest:
            raise AcceptanceError("Refusing to change the frozen contract.")
        load_contract(path)
        return previous
    payload = {
        "contract": contract, "sha256": digest,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(config_path),
        "source_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (root / f"scientific_acceptance{suffix}.sha256").write_text(digest + "\n")
    return payload


def registered_years(contract, split="component_validation") -> list[int]:
    return sorted(year for start, end in contract["splits"][split].values()
                  for year in range(int(start), int(end) + 1))


def acceptance_regions(lat, lon, contract) -> dict:
    masks, _ = boundary_region_masks(lat, lon, contract["regions"]["widths"])
    expected = ["full"] + [
        f"{prefix}_{width}"
        for width in contract["regions"]["widths"]
        for prefix in contract["regions"]["required_prefixes"]
    ]
    missing = set(expected).difference(masks)
    if missing:
        raise AcceptanceError(f"Domain cannot support registered regions: {sorted(missing)}")
    regions = {name: masks[name] for name in expected}
    box = contract["regions"]["southeast_precipitation_box"]
    lat, lon = np.asarray(lat), np.asarray(lon)
    regions["southeast_precipitation_box"] = (
        (lat[:, None] >= box["latitude_min"])
        & (lat[:, None] <= box["latitude_max"])
        & (lon[None, :] >= box["longitude_min"])
        & (lon[None, :] <= box["longitude_max"])
    )
    return regions


_SUM_FIELDS = (
    "count", "error_sum", "absolute_error_sum", "squared_error_sum",
    "prediction_sum", "target_sum", "crps_sum", "spread_sum", "identical_sum",
)


def collect_year_statistics(
    members, baseline, target, years, lat, lon, *,
    valid_mask=None, deterministic_control=None, metadata=None,
) -> dict:
    """Collect exact year/gridpoint sums from reconstructed physical members.

    Members are [time,member,lat,lon], other fields [time,lat,lon]. Caller
    authenticates dates, units, calendar, checkpoints and masks. Candidate
    NaNs cannot remove valid baseline/target cells; zero rainfall stays valid.
    """
    ensemble = np.asarray(members, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    phase1 = np.asarray(baseline, dtype=np.float64)
    year = np.asarray(years, dtype=np.int64)
    if (ensemble.ndim != 4 or ensemble.shape[0] != truth.shape[0]
            or ensemble.shape[2:] != truth.shape[1:]
            or truth.shape != phase1.shape or year.shape != (truth.shape[0],)
            or truth.shape[1:] != (len(lat), len(lon))):
        raise AcceptanceError("Expected matched [T,M,H,W] and [T,H,W] fields.")
    if ensemble.shape[1] < 2:
        raise AcceptanceError("A stochastic refinement needs at least two members.")
    valid = np.isfinite(truth) & np.isfinite(phase1)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape not in (truth.shape, truth.shape[1:]):
            raise AcceptanceError("Mask dimensions do not match target.")
        valid &= np.broadcast_to(mask, truth.shape)
    if not valid.any():
        raise AcceptanceError("No valid evaluation cells.")
    if np.any(valid & ~np.isfinite(ensemble).all(axis=1)):
        raise AcceptanceError("Nonfinite refinement members on valid baseline support.")
    products = {"phase1": phase1, "refined": ensemble.mean(axis=1)}
    if deterministic_control is not None:
        control = np.asarray(deterministic_control, dtype=np.float64)
        if control.shape != truth.shape or np.any(valid & ~np.isfinite(control)):
            raise AcceptanceError("Invalid deterministic control or mismatched support.")
        products["control"] = control
    unique_years = np.unique(year)
    score = empirical_crps(ensemble, truth)
    spread = np.std(ensemble, axis=1, ddof=1)
    identical = np.ptp(ensemble, axis=1) == 0
    result = {"years": unique_years, "lat": np.asarray(lat), "lon": np.asarray(lon),
              "metadata": dict(metadata or {}), "products": {}}
    result["metadata"].update({
        "ensemble_size": int(ensemble.shape[1]),
        "physical_members_before_averaging": True,
        "collector_version": 1, "valid_cell_days": int(valid.sum()),
        "observed_dates_per_year": {
            str(y): int(np.count_nonzero(year == y)) for y in unique_years
        },
    })
    for name, prediction in products.items():
        error = prediction - truth
        fields = {
            "count": valid.astype(np.float64), "error_sum": error,
            "absolute_error_sum": np.abs(error), "squared_error_sum": error**2,
            "prediction_sum": prediction, "target_sum": truth,
            "crps_sum": score if name == "refined" else np.abs(error),
            "spread_sum": spread if name == "refined" else np.zeros_like(truth),
            "identical_sum": identical if name == "refined" else np.ones_like(truth),
        }
        result["products"][name] = {
            key: np.stack([np.where(valid[year == y], value[year == y], 0).sum(axis=0)
                           for y in unique_years])
            for key, value in fields.items()
        }
    return result


def save_year_statistics(path, statistics) -> str:
    """NPZ has no object arrays; JSON metadata and its file hash are explicit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {key: np.asarray(statistics[key]) for key in ("years", "lat", "lon")}
    flat["metadata_json"] = np.asarray(json.dumps(statistics["metadata"], sort_keys=True))
    for product, fields in statistics["products"].items():
        flat.update({f"{product}__{name}": values for name, values in fields.items()})
    with path.open("wb") as handle:
        np.savez_compressed(handle, **flat)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_year_statistics(path, expected_sha256=None) -> dict:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise AcceptanceError("Year-statistics file hash mismatch.")
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in ("years", "lat", "lon")}
        result["metadata"] = json.loads(str(archive["metadata_json"]))
        result["products"] = {}
        for key in archive.files:
            if "__" in key:
                product, name = key.split("__", 1)
                result["products"].setdefault(product, {})[name] = archive[key].copy()
    result["statistics_file_sha256"] = digest
    return result


def _resolve_records(evidence) -> dict:
    result = {}
    for head, seeds in evidence.get("records", {}).items():
        result[head] = {}
        for seed, variables in seeds.items():
            result[head][str(seed)] = {}
            for variable, record in variables.items():
                if isinstance(record, str):
                    record = load_year_statistics(record)
                elif "path" in record:
                    record = load_year_statistics(record["path"], record.get("sha256"))
                result[head][str(seed)][variable] = record
    return result


def _product_metrics(fields, weights, masks) -> dict:
    summed = {name: np.tensordot(weights, values, axes=(0, 0))
              for name, values in fields.items()}
    count = summed["count"]
    valid = count > 0
    predicted = np.divide(summed["prediction_sum"], count,
                          out=np.full_like(count, np.nan), where=valid)
    observed = np.divide(summed["target_sum"], count,
                         out=np.full_like(count, np.nan), where=valid)
    result = {}
    for region, mask in masks.items():
        accumulator = ErrorAccumulator()
        accumulator.count = float(count[mask].sum())
        for name in ("error_sum", "absolute_error_sum", "squared_error_sum",
                     "prediction_sum", "target_sum"):
            setattr(accumulator, name, float(summed[name][mask].sum()))
        daily = accumulator.result()
        climatology = map_difference_metrics(np.where(mask, predicted, np.nan),
                                            np.where(mask, observed, np.nan))
        n = accumulator.count
        result[region] = {
            "daily_rmse": daily["rmse"], "daily_mae": daily["mae"],
            "signed_bias": daily["mean_bias"],
            "mean_absolute_climatological_bias": climatology["mean_absolute_bias"],
            "climatological_rmse": climatology["climatological_rmse"],
            "pattern_correlation": climatology["pattern_correlation"],
            "crps": float(summed["crps_sum"][mask].sum() / n) if n else None,
            "spread": float(summed["spread_sum"][mask].sum() / n) if n else None,
            "identical_fraction": float(summed["identical_sum"][mask].sum() / n) if n else None,
        }
    return result


def _safe_number(value):
    return value is not None and np.isfinite(value)


def _mean(values):
    return float(np.mean(values)) if values and all(_safe_number(v) for v in values) else None


def _gate_specs(contract, variable, regions):
    gates = contract["gates"]
    specs = [
        ("full", "daily_rmse", gates["full_daily_rmse_relative_change_max"], "phase1"),
        ("full", "mean_absolute_climatological_bias",
         gates["full_mean_absolute_climatological_bias_relative_change_max"], "phase1"),
        ("full", "daily_mae", gates["full_daily_mae_relative_change_max"], "phase1"),
        ("full", "crps", gates["full_crps_relative_change_max"], "phase1"),
        ("full", "spread_to_phase1_mae", -gates["minimum_spread_to_phase1_mae"], "phase1"),
        ("full", "identical_fraction", gates["maximum_fraction_identical_members"], "phase1"),
    ]
    for region in regions:
        if region == "full" or (region == "southeast_precipitation_box" and variable != "pr"):
            continue
        for metric in ("daily_rmse", "mean_absolute_climatological_bias"):
            specs.append((region, metric, gates[f"regional_{metric}_relative_change_max"], "phase1"))
    return specs


def comparison_rows(records, contract, *, year_weights=None, seed_indices=None):
    """Canonical metrics averaged equally across independent training seeds."""
    rows, individual = [], {}
    for head in contract["heads"]:
        available = records.get(head, {})
        seeds = [str(seed) for seed in contract["training_seeds"] if str(seed) in available]
        if not seeds:
            continue
        if seed_indices is not None:
            seeds = [seeds[index] for index in seed_indices]
        for variable in contract["variables"]:
            stats = [available[seed][variable] for seed in seeds if variable in available[seed]]
            if len(stats) != len(seeds):
                continue
            prototype = stats[0]
            masks = acceptance_regions(prototype["lat"], prototype["lon"], contract)
            years = np.asarray(prototype["years"])
            weights = (np.ones(len(years)) if year_weights is None else np.asarray(year_weights))
            strata = {"pooled": weights}
            strata.update({
                regime: weights * ((years >= bounds[0]) & (years <= bounds[1]))
                for regime, bounds in contract["splits"]["component_validation"].items()
            })
            for stratum, stratum_weights in strata.items():
                if not stratum_weights.any():
                    continue
                per_seed = [{
                    name: _product_metrics(fields, stratum_weights, masks)
                    for name, fields in record["products"].items()
                } for record in stats]
                individual[f"{head}/{variable}/{stratum}"] = {
                    seed: scores for seed, scores in zip(seeds, per_seed)
                }
                if stratum == "pooled":
                    specs = _gate_specs(contract, variable, masks)
                    if any(record["metadata"].get("mean_component_used") for record in stats):
                        specs.append(("full", "crps", 0.0, "control"))
                else:
                    specs = [
                        ("full", metric,
                         contract["gates"][f"regime_{metric}_relative_change_max"], "phase1")
                        for metric in ("daily_rmse", "mean_absolute_climatological_bias")
                    ]
                for region, metric, limit, reference in specs:
                    floor = float(contract["near_zero_reference"][variable])
                    if metric == "spread_to_phase1_mae":
                        refined = _mean([score["refined"][region]["spread"] for score in per_seed])
                        base = _mean([score["phase1"][region]["daily_mae"] for score in per_seed])
                        change = -refined / max(abs(base), floor) if _safe_number(refined) and _safe_number(base) else None
                    elif metric == "identical_fraction":
                        refined = _mean([score["refined"][region][metric] for score in per_seed])
                        base, change = 0.0, refined
                    else:
                        refined = _mean([score["refined"][region][metric] for score in per_seed])
                        base = _mean([score.get(reference, {}).get(region, {}).get(metric) for score in per_seed])
                        change = ((refined - base) / max(abs(base), floor)
                                  if _safe_number(refined) and _safe_number(base) else None)
                    key = f"{head}/{variable}/{stratum}/{region}/{metric}/{reference}"
                    rows.append({
                        "key": key, "head": head, "variable": variable,
                        "stratum": stratum, "region": region, "metric": metric,
                        "reference_product": reference, "reference": base,
                        "candidate": refined, "change": change, "limit": float(limit),
                        "point_pass": bool(_safe_number(change) and change < limit),
                        "units": contract["units"][variable],
                        "signed_bias_reference": _mean([score.get(reference, {}).get(region, {}).get("signed_bias") for score in per_seed]),
                        "signed_bias_candidate": _mean([score["refined"][region]["signed_bias"] for score in per_seed]),
                    })
    return rows, individual


def paired_resampling_plan(contract, years=None):
    """One plan shared by every head/variable/product and auxiliary metric."""
    years = np.asarray(registered_years(contract) if years is None else years)
    rng = np.random.default_rng(contract["bootstrap"]["seed"])
    repeats = int(contract["bootstrap"]["replicates"])
    weights = np.zeros((repeats, len(years)), dtype=np.int64)
    for start, end in contract["splits"]["component_validation"].values():
        positions = np.flatnonzero((years >= start) & (years <= end))
        if not positions.size:
            raise AcceptanceError("Bootstrap needs both registered climate regimes.")
        draws = rng.choice(positions, size=(repeats, len(positions)), replace=True)
        for index, draw in enumerate(draws):
            weights[index] += np.bincount(draw, minlength=len(years))
    seed_draws = rng.integers(0, len(contract["training_seeds"]),
                             size=(repeats, len(contract["training_seeds"])))
    digest = hashlib.sha256()
    digest.update(canonical_hash(contract).encode())
    digest.update(years.astype("<i8").tobytes())
    digest.update(weights.astype("<i8").tobytes())
    digest.update(seed_draws.astype("<i8").tobytes())
    return {"year_weights": weights, "seed_indices": seed_draws,
            "years": years, "sha256": digest.hexdigest()}


def simultaneous_max_statistic_intervals(point, bootstrap, confidence):
    """Two-sided centered standardized maximum across the registered family.

    A replicate is one paired hierarchical year/seed resample. Empirical
    members remain an intact issued ensemble, including coupled projections.
    These intervals are conditional on the finite saved sampler seed sets.
    """
    point = np.asarray(point, dtype=np.float64)
    draws = np.asarray(bootstrap, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[1:] != point.shape or draws.shape[0] < 19:
        raise AcceptanceError("Bootstrap must be [replicate, complete gate family].")
    if not np.isfinite(draws).all() or not np.isfinite(point).all():
        raise AcceptanceError("Nonfinite mandatory bootstrap metric.")
    standard_error = draws.std(axis=0, ddof=1)
    active = standard_error > np.finfo(np.float64).eps
    if np.any(~active & (np.max(np.abs(draws - point), axis=0) > 1e-12)):
        raise AcceptanceError("Degenerate bootstrap differs from the observed estimate.")
    normalized = np.zeros_like(draws)
    np.divide(np.abs(draws - point), standard_error,
              out=normalized, where=active[None, :])
    critical = float(np.quantile(normalized.max(axis=1), confidence, method="higher"))
    half_width = critical * standard_error
    return {
        "lower": point - half_width, "upper": point + half_width,
        "standard_error": standard_error, "critical_value": critical,
        "degenerate": ~active,
    }


def classify_interval(lower, upper, limit):
    """Strict noninferiority; wide intervals cannot be interpreted as a pass."""
    if not all(_safe_number(value) for value in (lower, upper, limit)):
        return "INCONCLUSIVE"
    if (upper <= limit and upper < 0) if limit < 0 else (upper < limit):
        return "PASS"
    if lower > limit:
        return "FAIL"
    return "INCONCLUSIVE"


def _date_completeness(metadata, years):
    import calendar
    date_ids = metadata.get("date_ids", [])
    if len(date_ids) != len(set(date_ids)) or not date_ids:
        return False
    calendar_name = metadata.get("calendar")
    if calendar_name not in ("noleap", "365_day", "standard", "gregorian",
                              "proleptic_gregorian", "360_day"):
        return False
    expected = set()
    for year in years:
        for month in range(1, 13):
            days = (30 if calendar_name == "360_day" else
                    calendar.monthrange(int(year), month)[1])
            if calendar_name in ("noleap", "365_day") and month == 2:
                days = 28
            expected.update(f"{int(year):04d}-{month:02d}-{day:02d}"
                            for day in range(1, days + 1))
    counts = {str(year): sum(date.startswith(f"{year}-") for date in date_ids)
              for year in years}
    return (set(date_ids) == expected
            and counts == metadata.get("observed_dates_per_year"))


def evidence_prerequisites(evidence, records, contract):
    """Missing authentication, full years or family members forbid promotion."""
    reasons = []
    if evidence.get("contract_sha256") != canonical_hash(contract):
        reasons.append("Evidence does not identify the frozen contract hash.")
    if evidence.get("scope") != "component_validation":
        reasons.append("Only registered component validation evidence is implemented; no final-test certification.")
    expected_years = registered_years(contract)
    pairing = {}
    ensemble_pairing = None
    checked_artifacts = {}
    required_hashes = (
        "phase1_sha256", "baseline_sha256", "target_sha256", "dates_sha256",
        "mask_sha256", "coordinates_sha256", "config_sha256",
        "normalization_sha256", "checkpoint_sha256",
    )
    for head in contract["heads"]:
        for seed in contract["training_seeds"]:
            for variable in contract["variables"]:
                label = f"{head}/{seed}/{variable}"
                record = records.get(head, {}).get(str(seed), {}).get(variable)
                if record is None:
                    reasons.append(f"{label}: missing registered head/seed/variable.")
                    continue
                metadata = record.get("metadata", {})
                if list(record["years"]) != expected_years:
                    reasons.append(f"{label}: incomplete or unregistered validation years.")
                if not _date_completeness(metadata, record["years"]):
                    reasons.append(f"{label}: incomplete calendar years or unauthenticated date identifiers.")
                if metadata.get("authenticated") is not True:
                    reasons.append(f"{label}: input provenance is not authenticated.")
                for key in required_hashes:
                    value = str(metadata.get(key, ""))
                    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                        reasons.append(f"{label}: missing or invalid {key}.")
                if metadata.get("ensemble_size", 0) < contract["sampling"]["minimum_members"]:
                    reasons.append(f"{label}: fewer than registered physical members.")
                if not metadata.get("physical_members_before_averaging"):
                    reasons.append(f"{label}: physical member reconstruction is unverified.")
                if metadata.get("training_seed") != seed:
                    reasons.append(f"{label}: training-seed provenance mismatch.")
                if metadata.get("units") != contract["units"][variable]:
                    reasons.append(f"{label}: physical units mismatch.")
                if metadata.get("edge_fallback_or_suppression") is not False:
                    reasons.append(f"{label}: no authenticated absence of edge suppression/fallback.")
                if metadata.get("effective_sampled_correction") is not True:
                    reasons.append(f"{label}: effective sampled corrections not demonstrated.")
                if not metadata.get("sampling_seed_ids"):
                    reasons.append(f"{label}: missing sampler seed identities.")
                current_ensemble = (metadata.get("ensemble_size"), tuple(metadata.get("sampling_seed_ids", [])))
                if ensemble_pairing is None:
                    ensemble_pairing = current_ensemble
                elif current_ensemble != ensemble_pairing:
                    reasons.append(f"{label}: physical ensemble size or sampler seed identities differ across the paired family.")
                if metadata.get("data_scope") != "component_validation":
                    reasons.append(f"{label}: metrics are not full registered component validation.")
                if metadata.get("fit_years") != registered_years(contract, "component_fit"):
                    reasons.append(f"{label}: registered fitting years not authenticated.")
                if metadata.get("raw_members_saved") is not True:
                    reasons.append(f"{label}: raw physical members unavailable.")
                if metadata.get("mean_component_used") and "control" not in record["products"]:
                    reasons.append(f"{label}: missing deterministic mean control.")
                pair = tuple(metadata.get(key) for key in required_hashes[:6])
                if variable in pairing and pair != pairing[variable]:
                    reasons.append(f"{label}: baseline, target, dates, mask or coordinates are not paired.")
                pairing.setdefault(variable, pair)
                checks = metadata.get("diagnostics", {})
                for name in contract["diagnostics"]["required"]:
                    check = checks.get(name, {})
                    applicable_statuses = {"complete"}
                    if variable != "pr" and name in {"wet_day_frequency_intensity", "wet_day_brier_reliability"}:
                        applicable_statuses.add("not_applicable")
                    if (not isinstance(check, dict) or check.get("status") not in applicable_statuses
                            or not check.get("artifact_sha256") or not check.get("artifact")):
                        reasons.append(f"{label}: required diagnostic {name} unavailable.")
                        continue
                    artifact = Path(check["artifact"])
                    expected_digest = check["artifact_sha256"]
                    artifact_key = str(artifact.resolve()), expected_digest
                    if artifact_key not in checked_artifacts:
                        checked_artifacts[artifact_key] = (
                            artifact.is_file()
                            and hashlib.sha256(artifact.read_bytes()).hexdigest() == expected_digest
                        )
                    if not checked_artifacts[artifact_key]:
                        reasons.append(f"{label}: diagnostic {name} artifact missing or hash mismatch.")
    return reasons


def _validate_statistics(records, contract):
    paired_fields = {}
    for head, seeds in records.items():
        if head not in contract["heads"]:
            raise AcceptanceError(f"Unregistered head {head}.")
        for seed, variables in seeds.items():
            if seed not in [str(s) for s in contract["training_seeds"]]:
                raise AcceptanceError(f"Unregistered training seed {seed}.")
            for variable, record in variables.items():
                if variable not in contract["variables"]:
                    raise AcceptanceError(f"Unregistered variable {variable}.")
                shape = (len(record["years"]), len(record["lat"]), len(record["lon"]))
                if not np.array_equal(np.unique(record["years"]), record["years"]):
                    raise AcceptanceError("Year statistics must be unique and sorted.")
                if not {"phase1", "refined"}.issubset(record["products"]):
                    raise AcceptanceError("Both Phase-1 and refined statistics are mandatory.")
                phase1 = record["products"]["phase1"]
                for product, fields in record["products"].items():
                    if set(fields) != set(_SUM_FIELDS):
                        raise AcceptanceError(f"{product}: missing or unexpected sufficient-statistic fields.")
                    if any(np.shape(value) != shape or not np.isfinite(value).all()
                           for value in fields.values()):
                        raise AcceptanceError("Nonfinite or misaligned year statistics.")
                    if np.any(fields["count"] < 0) or np.any(fields["count"] != np.floor(fields["count"])):
                        raise AcceptanceError("Valid-cell counts must be nonnegative integers.")
                    for name in ("absolute_error_sum", "squared_error_sum", "spread_sum", "identical_sum"):
                        if np.any(fields[name] < 0):
                            raise AcceptanceError(f"Negative nonnegative statistic {name}.")
                    if np.any(fields["identical_sum"] > fields["count"]):
                        raise AcceptanceError("Identical-member counts exceed valid support.")
                    for name in ("count", "target_sum"):
                        if not np.array_equal(fields[name], phase1[name]):
                            raise AcceptanceError("Products changed the matched evaluation support or target.")
                identity = tuple(record["years"])
                key = variable, identity
                reference = paired_fields.get(key)
                if reference is not None:
                    for name in _SUM_FIELDS:
                        if not np.array_equal(reference["products"]["phase1"][name], phase1[name]):
                            raise AcceptanceError("Phase-1 arrays differ between head/seed comparisons.")
                    if not np.array_equal(reference["lat"], record["lat"]) or not np.array_equal(reference["lon"], record["lon"]):
                        raise AcceptanceError("Coordinates differ between paired comparisons.")
                else:
                    paired_fields[key] = record


def _auxiliary_rows(evidence, records, contract, plan=None):
    rows, missing, bootstrap = [], [], []
    for head in contract["heads"]:
        if head not in records:
            continue
        for variable in contract["variables"]:
            metrics = evidence.get("auxiliary_metrics", {}).get(head, {}).get(variable, {})
            for metric, key in (
                ("p99_absolute_error", "maximum_p99_absolute_error_relative_change"),
                ("observed_p99_event_rmse", "maximum_observed_p99_event_rmse_relative_change"),
            ):
                item = metrics.get(metric, {})
                per_seed = item.get("per_seed", {})
                seeds = list(records[head])
                references = [per_seed.get(seed, {}).get("reference") for seed in seeds]
                candidates = [per_seed.get(seed, {}).get("candidate") for seed in seeds]
                reference, candidate = _mean(references), _mean(candidates)
                floor = contract["near_zero_reference"][variable]
                change = ((candidate - reference) / max(abs(reference), floor)
                          if _safe_number(candidate) and _safe_number(reference) else None)
                label = f"{head}/{variable}/pooled/full/{metric}/phase1"
                limit = contract["gates"][key]
                rows.append({
                    "key": label, "head": head, "variable": variable,
                    "stratum": "pooled", "region": "full", "metric": metric,
                    "reference_product": "phase1", "reference": reference,
                    "candidate": candidate, "change": change, "limit": limit,
                    "point_pass": bool(_safe_number(change) and change < limit),
                    "units": contract["units"][variable],
                })
                if change is None:
                    missing.append(f"{label}: exact extreme metrics unavailable.")
                draws = item.get("bootstrap_changes")
                if (plan is not None and
                    (item.get("resampling_plan_sha256") != plan["sha256"]
                     or np.shape(draws) != (contract["bootstrap"]["replicates"],))):
                    missing.append(f"{label}: matching whole-year/seed extreme bootstrap unavailable.")
                    draws = None
                bootstrap.append(draws)
    return rows, missing, bootstrap


def evaluate_scientific_acceptance(evidence, contract):
    """Return explicit per-head/variable acceptance and production blockers.

    Evidence records map head -> string training seed -> variable -> statistics
    or {path, sha256}. Auxiliary extreme metrics must be exact per-seed metrics
    plus bootstrap changes using paired_resampling_plan. Missing data never pass.
    """
    if not isinstance(contract, Mapping):
        contract = load_contract(contract)
    records = _resolve_records(evidence)
    _validate_statistics(records, contract)
    rows, individual = comparison_rows(records, contract)
    reasons = evidence_prerequisites(evidence, records, contract)
    plan = paired_resampling_plan(contract)
    auxiliary, aux_missing, aux_bootstrap = _auxiliary_rows(evidence, records, contract, plan)
    reasons.extend(aux_missing)
    all_rows = rows + auxiliary
    for row in all_rows:
        if not _safe_number(row["change"]):
            reasons.append(f"{row['key']}: mandatory point metric missing.")
        row.update({"status": "INCONCLUSIVE", "lower": None, "upper": None})
    bootstrap_report = {
        "computed": False, "plan_sha256": plan["sha256"],
        "registered_replicates": contract["bootstrap"]["replicates"],
        "confidence": contract["bootstrap"]["confidence"],
        "member_sets_resampled": False,
        "whole_fields_and_ensemble_sets_retained": True,
        "independent_training_seed_results_reported_separately": True,
        "sampling_uncertainty_scope": "conditional_on_saved_whole_ensemble_seed_sets",
        "family_scope": contract["bootstrap"]["family"],
    }
    if not reasons:
        draws = bootstrap_comparison_changes(records, contract, rows, plan)
        if auxiliary:
            draws = np.column_stack((draws, np.asarray(aux_bootstrap).T))
        intervals = simultaneous_max_statistic_intervals(
            [row["change"] for row in all_rows], draws,
            contract["bootstrap"]["confidence"])
        for index, row in enumerate(all_rows):
            row.update({
                "lower": float(intervals["lower"][index]),
                "upper": float(intervals["upper"][index]),
                "bootstrap_standard_error": float(intervals["standard_error"][index]),
                "degenerate_conditional_interval": bool(intervals["degenerate"][index]),
            })
            row["status"] = classify_interval(row["lower"], row["upper"], row["limit"])
        bootstrap_report.update({
            "computed": True, "critical_value": intervals["critical_value"],
            "mandatory_family_size": len(all_rows),
            "degenerate_interval_count": int(intervals["degenerate"].sum()),
        })
    combinations = {}
    for head in contract["heads"]:
        combinations[head] = {}
        for variable in contract["variables"]:
            subset = [row for row in all_rows if row["head"] == head and row["variable"] == variable]
            statuses = [row["status"] for row in subset]
            status = ("FAIL" if "FAIL" in statuses else
                      "PASS" if statuses and all(s == "PASS" for s in statuses) and not reasons
                      else "INCONCLUSIVE")
            combinations[head][variable] = {
                "status": status, "point_guardrails_pass": bool(subset and all(row["point_pass"] for row in subset)),
                "point_failures": [row["key"] for row in subset if not row["point_pass"]],
                "failed_intervals": [row["key"] for row in subset if row["status"] == "FAIL"],
            }
    statuses = [entry["status"] for variables in combinations.values() for entry in variables.values()]
    component_status = ("FAIL" if "FAIL" in statuses else "PASS" if all(s == "PASS" for s in statuses)
                        else "INCONCLUSIVE")
    return {
        "schema_version": 1, "contract_sha256": canonical_hash(contract),
        "scope": "component_validation", "status": component_status,
        "combinations": combinations, "prerequisite_failures": sorted(set(reasons)),
        "rows": all_rows, "per_training_seed_metrics": individual,
        "bootstrap": bootstrap_report,
        "end_to_end": {
            "status": "INCONCLUSIVE",
            "reason": "No independently untouched Phase-1/refinement/normalization/selection test is certified by this component evaluator.",
        },
        "production_promotion": False,
        "production_blockers": [
            "Component-validation candidate is not end-to-end acceptance.",
            *([] if component_status == "PASS" else ["Registered component acceptance has not passed."]),
        ],
    }


def candidate_point_decision(evidence, contract, head):
    """Checkpoint selection uses all variable guardrails, not joint process loss."""
    records = _resolve_records(evidence)
    _validate_statistics(records, contract)
    rows, _ = comparison_rows({head: records.get(head, {})}, contract)
    auxiliary, _, _ = _auxiliary_rows(evidence, {head: records.get(head, {})}, contract)
    rows += auxiliary
    rows = [row for row in rows if row["head"] == head]
    reasons = [row["key"] for row in rows if not row["point_pass"]]
    if evidence.get("contract_sha256") != canonical_hash(contract):
        reasons.append("Candidate evidence does not identify the frozen contract hash.")
    if evidence.get("scope") != "component_validation":
        reasons.append("Checkpoint selection is restricted to registered component validation.")
    for variable in contract["variables"]:
        if not any(row["variable"] == variable for row in rows):
            reasons.append(f"{head}/{variable}: missing variable.")
    for seed, variables in records.get(head, {}).items():
        for variable in contract["variables"]:
            record = variables.get(variable)
            if record is None:
                reasons.append(f"{head}/{seed}/{variable}: missing variable.")
                continue
            metadata = record["metadata"]
            if metadata.get("authenticated") is not True:
                reasons.append(f"{head}/{seed}/{variable}: candidate inputs are unauthenticated.")
            if metadata.get("training_seed") != int(seed) or metadata.get("units") != contract["units"][variable]:
                reasons.append(f"{head}/{seed}/{variable}: candidate seed or physical units mismatch.")
            if (list(record["years"]) != registered_years(contract)
                    or not _date_completeness(metadata, record["years"])
                    or metadata.get("data_scope") != "component_validation"):
                reasons.append(f"{head}/{seed}/{variable}: subset screening is provisional, not full-year selection.")
            if metadata.get("effective_sampled_correction") is not True:
                reasons.append(f"{head}/{seed}/{variable}: sampled correction unverified.")
            if metadata.get("edge_fallback_or_suppression") is not False:
                reasons.append(f"{head}/{seed}/{variable}: fallback/suppression unverified.")
    margins = [float(row["change"] - row["limit"]) for row in rows if _safe_number(row["change"])]
    missing = sum(not _safe_number(row["change"]) for row in rows)
    score = [int(missing), max(margins) if margins else 1e30,
             float(np.mean(margins)) if margins else 1e30]
    return {
        "scientifically_eligible_candidate": bool(rows and not reasons),
        "production_accepted": False, "acceptance_status": "NOT_EVALUATED",
        "score": score, "rejection_reasons": sorted(set(reasons)), "rows": rows,
    }


class ScientificSelector:
    """Preserve last, best eligible, and explicitly rejected provisional checkpoints."""

    def __init__(self, output_dir, contract, head):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.contract = load_contract(contract) if not isinstance(contract, Mapping) else dict(contract)
        if head not in self.contract["heads"]:
            raise AcceptanceError(f"Unregistered selector head {head}.")
        self.head = head
        self.state_path = self.output_dir / "scientific_selection.json"
        self.state = {"contract_sha256": canonical_hash(self.contract), "head": head,
                      "best_eligible": None, "nearest_provisional": None, "decisions": []}
        if self.state_path.exists():
            existing = json.loads(self.state_path.read_text())
            if existing["contract_sha256"] != self.state["contract_sha256"] or existing["head"] != head:
                raise AcceptanceError("Selector state belongs to a different head or frozen contract.")
            self.state = existing
        else:
            self.state_path.write_text(json.dumps(self.state, indent=2, allow_nan=False), encoding="utf-8")

    def consider(self, checkpoint_path, evidence, *, training_progress=None):
        source = Path(checkpoint_path)
        if not source.is_file():
            raise AcceptanceError(f"Checkpoint not found: {source}")
        decision = candidate_point_decision(evidence, self.contract, self.head)
        decision.update({
            "checkpoint": str(source.resolve()),
            "checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "training_progress": training_progress,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
        last = self.output_dir / "last.ckpt"
        if source.resolve() != last.resolve():
            shutil.copy2(source, last)
        eligible = decision["scientifically_eligible_candidate"]
        key = "best_eligible" if eligible else "nearest_provisional"
        current = self.state[key]
        selected = current is None or tuple(decision["score"]) < tuple(current["score"])
        decision["selected"] = selected
        if selected:
            name = (self.contract["selection"]["best_candidate_name"] if eligible
                    else self.contract["selection"]["provisional_name"])
            destination = self.output_dir / name
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            decision["selected_checkpoint"] = str(destination.resolve())
            self.state[key] = {k: v for k, v in decision.items() if k != "rows"}
        self.state["decisions"].append(decision)
        self.state_path.write_text(json.dumps(self.state, indent=2, allow_nan=False), encoding="utf-8")
        return decision


def merge_year_statistics(first, second):
    """Combine chunks additively; never average chunk metrics or climatologies."""
    if first is None:
        return second
    for name in ("lat", "lon"):
        if not np.array_equal(first[name], second[name]):
            raise AcceptanceError("Cannot merge different spatial coordinates.")
    if set(first["products"]) != set(second["products"]):
        raise AcceptanceError("Cannot merge different physical products.")
    if first["metadata"]["ensemble_size"] != second["metadata"]["ensemble_size"]:
        raise AcceptanceError("Cannot merge different ensemble sizes.")
    years = np.union1d(first["years"], second["years"])
    result = {"years": years, "lat": first["lat"], "lon": first["lon"],
              "metadata": dict(first["metadata"]), "products": {}}
    for product in first["products"]:
        result["products"][product] = {}
        for name in _SUM_FIELDS:
            values = np.zeros((len(years), len(first["lat"]), len(first["lon"])), dtype=np.float64)
            for source in (first, second):
                values[np.searchsorted(years, source["years"])] += source["products"][product][name]
            result["products"][product][name] = values
    counts = dict(first["metadata"]["observed_dates_per_year"])
    for year, count in second["metadata"]["observed_dates_per_year"].items():
        counts[year] = counts.get(year, 0) + count
    first_dates = first["metadata"].get("date_ids")
    second_dates = second["metadata"].get("date_ids")
    if first_dates is not None or second_dates is not None:
        if first_dates is None or second_dates is None:
            raise AcceptanceError("Every merged chunk must provide date identifiers if one does.")
        dates = list(first_dates) + list(second_dates)
        if len(dates) != len(set(dates)):
            raise AcceptanceError("Repeated dates in merged year statistics.")
        result["metadata"]["date_ids"] = sorted(dates)
    result["metadata"]["observed_dates_per_year"] = counts
    result["metadata"]["valid_cell_days"] += second["metadata"]["valid_cell_days"]
    return result


def _mask_rectangles(mask):
    """Exact rectangle decomposition of a boolean mask, with no interpolation."""
    rectangles, active = [], {}
    for row in range(mask.shape[0] + 1):
        present = set()
        if row < mask.shape[0]:
            changes = np.flatnonzero(np.diff(np.pad(mask[row].astype(np.int8), (1, 1))))
            present = {(int(a), int(b)) for a, b in zip(changes[::2], changes[1::2])}
        for span in list(active):
            if span not in present:
                rectangles.append((active.pop(span), row, span[0], span[1]))
        for span in present:
            active.setdefault(span, row)
    return rectangles


def _rectangular_region_sums(values, rectangles):
    """Summed-area reductions retain all cells, including outermost boundaries."""
    values = np.asarray(values, dtype=np.float64)
    integral = np.pad(values.cumsum(axis=-2).cumsum(axis=-1),
                      [(0, 0)] * (values.ndim - 2) + [(1, 0), (1, 0)])
    result = []
    for region in rectangles:
        total = np.zeros(values.shape[:-2], dtype=np.float64)
        for top, bottom, left, right in region:
            total += (integral[..., bottom, right] - integral[..., top, right]
                      - integral[..., bottom, left] + integral[..., top, left])
        result.append(total)
    return np.stack(result, axis=-1)


def _bootstrap_product_profiles(fields, year_weights, masks, years, contract, batch_size):
    """Vectorized execution of the canonical reducers; no alternative metrics."""
    names = list(masks)
    rectangles = [_mask_rectangles(masks[name]) for name in names]
    scalar_fields = ("count", "error_sum", "absolute_error_sum", "squared_error_sum",
                     "prediction_sum", "target_sum", "crps_sum", "spread_sum", "identical_sum")
    annual = np.stack([_rectangular_region_sums(fields[name], rectangles)
                       for name in scalar_fields], axis=1)  # [year,field,region]
    height, width = fields["count"].shape[-2:]
    map_fields = np.stack([fields[name] for name in ("count", "prediction_sum", "target_sum")], axis=1)
    map_matrix = map_fields.reshape(len(years), -1)
    metric_names = ("daily_rmse", "daily_mae", "mean_absolute_climatological_bias",
                    "crps", "spread", "identical_fraction")
    strata = {"pooled": np.ones(len(years), dtype=bool)}
    strata.update({name: (years >= bounds[0]) & (years <= bounds[1])
                   for name, bounds in contract["splits"]["component_validation"].items()})
    profiles = {}
    for stratum, year_mask in strata.items():
        region_indices = list(range(len(names))) if stratum == "pooled" else [names.index("full")]
        region_names = [names[index] for index in region_indices]
        region_rectangles = [rectangles[index] for index in region_indices]
        scalar_matrix = annual[:, :, region_indices].reshape(len(years), -1)
        # Multinomial whole-year bootstrap repeats many identical weight
        # vectors (at most 35 per four-year regime). Reuse identical maps
        # exactly, then restore the original paired replicate ordering.
        unique_weights, inverse = np.unique(year_weights * year_mask, axis=0, return_inverse=True)
        profile = {metric: np.full((len(unique_weights), len(region_names)), np.nan)
                   for metric in metric_names}
        profile["region_indices"] = {name: index for index, name in enumerate(region_names)}
        for start in range(0, len(unique_weights), batch_size):
            stop = min(start + batch_size, len(unique_weights))
            weights = unique_weights[start:stop]
            sums = (weights @ scalar_matrix).reshape(stop - start, len(scalar_fields), len(region_names))
            count = sums[:, 0]
            denominator = np.where(count > 0, count, 1)
            reduced = error_metrics_from_sums(denominator, *[sums[:, index] for index in range(1, 6)])
            profile["daily_rmse"][start:stop] = np.where(count > 0, reduced["rmse"], np.nan)
            profile["daily_mae"][start:stop] = np.where(count > 0, reduced["mae"], np.nan)
            for metric, field_index in (("crps", 6), ("spread", 7), ("identical_fraction", 8)):
                profile[metric][start:stop] = np.where(count > 0, sums[:, field_index] / denominator, np.nan)
            maps = (weights @ map_matrix).reshape(stop - start, 3, height, width)
            valid = maps[:, 0] > 0
            map_count = np.where(valid, maps[:, 0], 1)
            difference = maps[:, 1] / map_count - maps[:, 2] / map_count
            absolute = _rectangular_region_sums(np.where(valid, np.abs(difference), 0), region_rectangles)
            support = _rectangular_region_sums(valid, region_rectangles)
            profile["mean_absolute_climatological_bias"][start:stop] = np.where(
                support > 0, mean_absolute_climatological_bias_from_sums(absolute, np.where(support > 0, support, 1)), np.nan)
        for metric in metric_names:
            profile[metric] = profile[metric][inverse]
        profiles[stratum] = profile
    return profiles


def bootstrap_comparison_changes(records, contract, rows, plan, *, batch_size=32):
    """All core comparison draws without recomputing full Python metric trees.

    Regional sums use exact rectangle decompositions and summed-area tables.
    Climatology is still formed from each resampled whole-year map, followed by
    the same canonical equal-gridpoint reduction. Floating summation order can
    differ by roundoff; no spatial cells or stochastic members are discarded.
    """
    profiles = {}
    variables = contract["variables"]
    seeds = [str(seed) for seed in contract["training_seeds"]]
    weights = plan["year_weights"]
    for variable in variables:
        first = records[contract["heads"][0]][seeds[0]][variable]
        masks = acceptance_regions(first["lat"], first["lon"], contract)
        profiles[(variable, "phase1")] = _bootstrap_product_profiles(
            first["products"]["phase1"], weights, masks, first["years"], contract, batch_size)
        for head in contract["heads"]:
            for seed in seeds:
                record = records[head][seed][variable]
                for product in record["products"]:
                    if product == "phase1":
                        continue
                    profiles[(variable, head, seed, product)] = _bootstrap_product_profiles(
                        record["products"][product], weights, masks, record["years"],
                        contract, batch_size)
    changes = np.empty((len(weights), len(rows)), dtype=np.float64)
    draws = plan["seed_indices"].T
    for index, row in enumerate(rows):
        variable, head, stratum, region = (row[name] for name in ("variable", "head", "stratum", "region"))
        metric = row["metric"]
        candidate_metric = "spread" if metric == "spread_to_phase1_mae" else metric
        reference_metric = "daily_mae" if metric == "spread_to_phase1_mae" else metric
        values = []
        for seed in seeds:
            profile = profiles[(variable, head, seed, "refined")][stratum]
            values.append(profile[candidate_metric][:, profile["region_indices"][region]])
        candidate = np.take_along_axis(np.asarray(values), draws, axis=0).mean(axis=0)
        if metric == "identical_fraction":
            change = candidate
        else:
            values = []
            for seed in seeds:
                key = ((variable, "phase1") if row["reference_product"] == "phase1"
                       else (variable, head, seed, row["reference_product"]))
                profile = profiles[key][stratum]
                values.append(profile[reference_metric][:, profile["region_indices"][region]])
            reference = np.take_along_axis(np.asarray(values), draws, axis=0).mean(axis=0)
            scale = np.maximum(np.abs(reference), contract["near_zero_reference"][variable])
            change = -candidate / scale if metric == "spread_to_phase1_mae" else (candidate - reference) / scale
        changes[:, index] = change
    return changes
