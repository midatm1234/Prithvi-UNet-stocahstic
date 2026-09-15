"""Physical, paired supplemental contrasts; no climate model is trained here."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.CORDEX_ML import cordex_scientific_attribution as A
from granitewxc.refinement.acceptance import collect_year_statistics, save_year_statistics

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts/refinement_validation/scientific_acceptance_20260909T230807Z"


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    root = tmp_path_factory.mktemp("paired_attribution")
    contract = copy.deepcopy(A.load_contract(ARTIFACT / "frozen_scientific_acceptance_v2.json"))
    contract["regions"]["widths"] = [1]
    contract["bootstrap"]["replicates"] = 19
    contract["sampling"]["minimum_members"] = 3
    parent = {"acceptance_contract_sha256": A.canonical_hash(contract)}
    text = json.dumps(parent, indent=2) + "\n"
    A.write(root / "registered_experiment_plan.json", parent)
    (root / "registered_experiment_plan.sha256").write_text(hashlib.sha256(text.encode()).hexdigest())
    A.write(root / "frozen_scientific_acceptance_v2.json", {"contract": contract, "sha256": A.canonical_hash(contract)})
    registration_path, registration = A.register_contrast(root)
    years = np.repeat(A.registered_years(contract), 365)
    dates = [f"{year}-{month:02d}-{day:02d}" for year in A.registered_years(contract)
             for month, days in enumerate([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], 1)
             for day in range(1, days + 1)]
    lat, lon = np.linspace(-34, -30, 5), np.linspace(28, 32, 5)
    valid = np.ones((len(years), 5, 5), bool); valid[:, 2, 2] = False
    grid = np.linspace(0, 1, 25).reshape(1, 5, 5)
    year_signal = np.repeat(np.linspace(-.6, .6, 8), 365)[:, None, None]
    truth = np.broadcast_to(1 + grid + year_signal, valid.shape).copy(); truth[:, 0, 0] = 0.
    baseline = truth + 1.7
    arrays, annual = {}, {}
    for label in ("selected", "control"):
        for index, seed in enumerate(contract["training_seeds"]):
            mean = truth + (.24 if label == "selected" else .9) + index * .08 + year_signal * (.6 if label == "selected" else -.3)
            members = mean[:, None] + np.asarray([-.2, 0, .2] if label == "selected" else [-2., 0, 2.])[None, :, None, None]
            for variable in contract["variables"]:
                shift = 0 if variable == "pr" else 280.
                physical = np.maximum(members, 0) if variable == "pr" else members + shift
                target, base = truth + shift, baseline + shift
                arrays[label, str(seed), variable] = physical, target, base
                annual[label, str(seed), variable] = collect_year_statistics(
                    physical, base, target, years, lat, lon, valid_mask=valid,
                    deterministic_control=target + .01 if label == "selected" else None)
    cache = root / "cache_fixture.bin"; cache.write_bytes(b"synthetic common source")
    jobs, families = [], {}
    for label in ("selected", "control"):
        merged = {"contract_sha256": A.canonical_hash(contract), "scope": "component_validation", "records": {}, "input_evidence": []}
        for head in contract["heads"]:
            for seed in map(str, contract["training_seeds"]):
                directory = root / "workflow" / "runs" / ("full_" + label) / head / seed
                directory.mkdir(parents=True)
                checkpoint = directory / "candidate.ckpt"; checkpoint.write_bytes(f"{label}:{head}:{seed}".encode())
                prediction = directory / "prediction_fixture.bin"; prediction.write_bytes(f"issued-physical:{label}:{head}:{seed}".encode())
                source_config = {"model": {"refinement": {"enabled": True, "type": head}}}
                config = A.resolve_refinement_config(source_config)
                recipe_name = "native" if label == "selected" else "existing_full_recipe"
                recipe = A.experiment_recipes(config, contract["variables"])[recipe_name].to_dict()
                refined_config = config.to_dict()
                if label == "selected":
                    refined_config["conditioning"]["normalize_predictors"] = True
                    refined_config["transformer" if head.endswith("transformer") else "unet"]["spatial_alignment"] = "coordinates"
                formulation = "mean_remainder" if label == "selected" else "direct"
                run = {"stage": "full", "head": head, "seed": int(seed), "recipe": recipe, "formulation": formulation,
                       "source_config": source_config, "refinement_config": refined_config,
                       "plan_sha256": registration["parent_experiment_plan_sha256"],
                       "acceptance_contract_sha256": A.canonical_hash(contract),
                       "source_cache_sha256": A.file_sha256(cache), "source_phase1_sha256": "phase1"}
                A.write(directory / "run_contract.json", {"contract": run, "sha256": A.config_fingerprint(run)})
                jobs.append({"head": head, "seed": int(seed), "recipe": recipe_name, "formulation": formulation,
                             "comparison_group": label, "output": str(directory)})
                auth = {"authenticated": True, "head": head, "training_seed": int(seed), "full_training_verified": True,
                        "prediction_path": str(prediction), "prediction_sha256": A.file_sha256(prediction),
                        "checkpoint_path": str(checkpoint), "checkpoint_sha256": A.file_sha256(checkpoint),
                        "cache_path": str(cache), "cache_sha256": A.file_sha256(cache), "phase1_sha256": "phase1",
                        "run_contract_sha256": A.config_fingerprint(run), "date_ids": dates,
                        "dates_sha256": A.canonical_hash(dates), "coordinates_sha256": "coordinates",
                        "variables": contract["variables"], "units": contract["units"], "calendar": "noleap",
                        "source_timestamps": [date + "T12:00:00" for date in dates],
                        "sample_ids": ["case|" + date + "T12:00:00" for date in dates],
                        "ensemble_size": 3, "sampling_seed": 1234, "random_stream_version": "fixture-stream",
                        "canonical_member_batch": 8, "nonnegative_strategy": "memberwise",
                        "fit_years": A.registered_years(contract, "component_fit"),
                        "sampling_settings": {"effective_sampling_steps": 50, "solver": "euler", "sigma_min": .0001,
                                              "mean_path_loss_weight": 0 if label == "selected" else .5,
                                              "time_sampling": "uniform" if label == "selected" else "logit_normal"},
                        "field_sha256": {variable: {"baseline": "same-base-" + variable,
                            "target": "same-target-" + variable, "mask": "same-mask"} for variable in contract["variables"]},
                        "completed_run_authentication": {"status": "complete"},
                        "normalization_sha256": label + "-own-training-statistics"}
                auth_path = directory / "authentication.json"; A.write(auth_path, auth)
                source_records, merged_records = {}, {}
                for variable in contract["variables"]:
                    record = copy.deepcopy(annual[label, seed, variable])
                    record["metadata"].update({name: auth[name] for name in (
                        "training_seed", "checkpoint_sha256", "date_ids", "phase1_sha256", "dates_sha256",
                        "coordinates_sha256", "calendar", "fit_years")})
                    record["metadata"].update(authenticated=True, data_scope="component_validation",
                        baseline_sha256=auth["field_sha256"][variable]["baseline"],
                        target_sha256=auth["field_sha256"][variable]["target"], mask_sha256="same-mask",
                        units=contract["units"][variable], sampling_seed_ids=["member-0", "member-1", "member-2"],
                        mean_component_used=label == "selected",
                        diagnostics={"independently_trained_seed_results": {"status": "pending_merge"}})
                    stat_path = directory / (variable + "_processed_year_statistics.npz")
                    source_records[variable] = {"path": str(stat_path), "sha256": save_year_statistics(stat_path, record)}
                    combined = copy.deepcopy(record)
                    combined["metadata"]["diagnostics"]["independently_trained_seed_results"] = {"status": "complete"}
                    merged_path = root / label / head / seed / (variable + ".npz")
                    merged_path.parent.mkdir(parents=True, exist_ok=True)
                    merged_records[variable] = {"path": str(merged_path), "sha256": save_year_statistics(merged_path, combined)}
                evidence = {"contract_sha256": A.canonical_hash(contract), "records": {head: {seed: source_records}},
                            "authentication": {"artifact": str(auth_path), "artifact_sha256": A.file_sha256(auth_path)}}
                evidence_path = directory / "evidence.json"; A.write(evidence_path, evidence)
                merged["records"].setdefault(head, {})[seed] = merged_records
                merged["input_evidence"].append({"path": str(evidence_path), "sha256": A.file_sha256(evidence_path)})
        family_path = root / label / "evidence.json"; A.write(family_path, merged)
        report_path = family_path.parent / "acceptance_report" / "scientific_acceptance_results.json"
        A.write(report_path, {"status": "INCONCLUSIVE", "production_promotion": False})
        A.write(family_path.parent / "completion.json", {"complete": True, "operation": "merge",
            "contract_sha256": A.canonical_hash(contract), "artifact_sha256": A.file_sha256(report_path)})
        families[label] = family_path
    full = {"registered_plan_sha256": registration["parent_experiment_plan_sha256"],
            "acceptance_contract_sha256": A.canonical_hash(contract), "jobs": jobs}
    full["sha256"] = A.canonical_hash(full); A.write(root / "workflow" / "full_queue.json", full)
    return SimpleNamespace(root=root, contract=contract, registration=registration_path, families=families,
                           arrays=arrays, years=years, valid=valid, lat=lat, lon=lon)


def run(source, output):
    return A.compare_selected_fresh_control(source.families["selected"], source.families["control"],
        source.contract, source.registration, output, batch_size=8)


def test_normal_cli_keeps_all_eight_pairs_and_acceptance_unchanged(source, tmp_path):
    result_before = {label: (path.parent / "acceptance_report/scientific_acceptance_results.json").read_bytes()
                     for label, path in source.families.items()}
    assert A.main(["compare", "--selected", str(source.families["selected"]), "--control", str(source.families["control"]),
        "--contract", str(source.root / "frozen_scientific_acceptance_v2.json"), "--registration", str(source.registration), "--output", str(tmp_path)]) == 0
    report = A.read(tmp_path / "comparison.json")
    assert report["status"] == "COMPLETE" and report["rows_count"] == 348
    assert report["acceptance_decisions_unchanged"] is True and report["production_promotion"] is False
    assert {(row["head"], row["variable"]) for row in report["rows"]} == {(h, v) for h in source.contract["heads"] for v in source.contract["variables"]}
    assert all(row["reference_product"] == "fresh_stochastic_control" for row in report["rows"])
    assert all(row["region"] == "full" for row in report["rows"] if row["metric"] == "crps")
    assert {row["region"] for row in report["rows"] if row["stratum"] == "pooled"} == set(A.acceptance_regions(source.lat, source.lon, source.contract))
    assert len(report["paired_procedures"]) == 12
    assert all(item["selected"]["refinement_config"] != item["fresh_control"]["refinement_config"] for item in report["paired_procedures"])
    assert A.read(tmp_path / "completion.json")["operation"] == "attribution"
    for label, path in source.families.items():
        assert (path.parent / "acceptance_report/scientific_acceptance_results.json").read_bytes() == result_before[label]
    with pytest.raises(A.AcceptanceError, match="new or empty"):
        run(source, tmp_path)


def test_bootstrap_matches_literal_whole_year_and_seed_pairs_with_intact_ensembles(source, tmp_path):
    report = run(source, tmp_path)
    archive = np.load(tmp_path / "bootstrap_changes.npz", allow_pickle=False)
    plan = A.paired_resampling_plan(source.contract)
    assert np.array_equal(archive["year_weights"], plan["year_weights"])
    assert np.array_equal(archive["seed_indices"], plan["seed_indices"])
    observed = {}
    for metric in A.CORE + ("crps",):
        row_index = next(i for i, row in enumerate(report["rows"]) if row["head"] == "diffusion_unet"
                         and row["variable"] == "pr" and row["stratum"] == "pooled" and row["region"] == "full" and row["metric"] == metric)
        for replicate in range(3):
            ids = np.concatenate([np.tile(np.flatnonzero(source.years == year), weight)
                                  for year, weight in zip(plan["years"], plan["year_weights"][replicate]) if weight])
            values = {}
            for label in ("selected", "control"):
                scores = []
                for seed_index in plan["seed_indices"][replicate]:
                    seed = str(source.contract["training_seeds"][seed_index])
                    members, target, _ = source.arrays[label, seed, "pr"]
                    members, target, valid = members[ids], target[ids], source.valid[ids]
                    mean = members.mean(axis=1); error = mean - target
                    if metric == "daily_rmse":
                        value = np.sqrt(np.mean(error[valid] ** 2))
                    elif metric == "daily_mae":
                        value = np.mean(np.abs(error[valid]))
                    elif metric == "mean_absolute_climatological_bias":
                        value = np.mean(np.abs(error.mean(axis=0)[valid.any(axis=0)]))
                    else:
                        first = np.abs(members - target[:, None]).mean(axis=1)
                        pairwise = np.abs(members[:, :, None] - members[:, None, :]).mean(axis=(1, 2)) / 2
                        value = np.mean((first - pairwise)[valid])
                    scores.append(value)
                values[label] = np.mean(scores)
            expected = (values["selected"] - values["control"]) / max(abs(values["control"]), .001)
            assert archive["changes"][replicate, row_index] == pytest.approx(expected, abs=2e-12)
        observed[metric] = report["rows"][row_index]
    # The full stochastic control's CRPS differs from its point-distribution MAE.
    assert abs(observed["crps"]["reference"] - observed["daily_mae"]["reference"]) > .01


def test_missing_seed_case_is_explicit_inconclusive_without_smaller_family(source, tmp_path):
    family = A.read(source.families["selected"])
    family["records"]["diffusion_unet"]["101"].pop("pr")
    directory = tmp_path / "partial_family"; directory.mkdir()
    A.write(directory / "evidence.json", family)
    report_source = source.families["selected"].parent / "acceptance_report/scientific_acceptance_results.json"
    A.write(directory / "acceptance_report/scientific_acceptance_results.json", A.read(report_source))
    A.write(directory / "completion.json", A.read(source.families["selected"].parent / "completion.json"))
    result = A.compare_selected_fresh_control(directory / "evidence.json", source.families["control"], source.contract,
                                            source.registration, tmp_path / "output")
    assert result["status"] == "INCONCLUSIVE" and result["rows_count"] == 0 and result["missing_cases"]
    assert not (tmp_path / "output/bootstrap_changes.npz").exists()


def test_raw_statistics_cannot_replace_authenticated_processed_product(source, tmp_path):
    family = A.read(source.families["selected"])
    original = family["records"]["diffusion_unet"]["101"]["pr"]
    raw = A.load_year_statistics(original["path"], original["sha256"])
    raw["products"]["refined"]["error_sum"] += 1
    replacement = tmp_path / "pr_raw_year_statistics.npz"
    family["records"]["diffusion_unet"]["101"]["pr"] = {"path": str(replacement), "sha256": save_year_statistics(replacement, raw)}
    A.write(tmp_path / "evidence.json", family)
    A.write(tmp_path / "acceptance_report/scientific_acceptance_results.json", A.read(source.families["selected"].parent / "acceptance_report/scientific_acceptance_results.json"))
    A.write(tmp_path / "completion.json", A.read(source.families["selected"].parent / "completion.json"))
    with pytest.raises(A.AcceptanceError, match="authenticated processed physical product"):
        A.compare_selected_fresh_control(tmp_path / "evidence.json", source.families["control"], source.contract,
                                         source.registration, tmp_path / "output")


@pytest.mark.parametrize("field", ["sample_ids", "canonical_member_batch", "nonnegative_strategy", "mask_sha256", "date_ids"])
def test_pairing_rejects_member_date_mask_and_projection_drift(source, field):
    contract, registration = source.contract, A.read(source.registration)
    full = A.read(source.root / "workflow/full_queue.json")
    jobs = {(j["comparison_group"], j["head"], str(j["seed"])): j for j in full["jobs"]}
    left, la, _ = A._load_family(source.families["selected"], "selected", contract, registration, {}, jobs)
    right, ra, _ = A._load_family(source.families["control"], "control", contract, registration, {}, jobs)
    if field == "mask_sha256":
        right["diffusion_unet"]["101"]["pr"]["metadata"][field] = "different"
    else:
        ra["diffusion_unet", "101"][field] = "different"
    with pytest.raises(A.AcceptanceError, match="pairing mismatch"):
        A.paired_records(left, right, la, ra, contract)


def test_frozen_registration_is_idempotent_and_refuses_change_or_late_creation(source, tmp_path):
    before = source.registration.read_bytes()
    assert A.register_contrast(source.root)[0] == source.registration
    assert source.registration.read_bytes() == before
    parent = A.read(source.root / "registered_experiment_plan.json")
    A.write(tmp_path / "registered_experiment_plan.json", parent)
    (tmp_path / "registered_experiment_plan.sha256").write_bytes((source.root / "registered_experiment_plan.sha256").read_bytes())
    A.write(tmp_path / "frozen_scientific_acceptance_v2.json", {"contract": source.contract, "sha256": A.canonical_hash(source.contract)})
    checkpoint = tmp_path / "workflow/runs/full_selected/head/seed/last.ckpt"
    checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b"already fitting")
    with pytest.raises(A.AcceptanceError, match="before full fitting"):
        A.register_contrast(tmp_path)
    altered = A.read(source.registration); altered["member_resampling"] = "incorrect"
    A.write(tmp_path / "bad_registration.json", altered)
    with pytest.raises(A.AcceptanceError, match="registration changed"):
        A.compare_selected_fresh_control(source.families["selected"], source.families["control"], source.contract,
                                        tmp_path / "bad_registration.json", tmp_path / "output")


def test_reference_floor_and_zero_variability_are_explicit():
    intervals = A.simultaneous_max_statistic_intervals([0.], np.zeros((19, 1)), .95)
    assert intervals["degenerate"][0] and intervals["lower"][0] == intervals["upper"][0] == 0
    # Inference settings ignore training-only flow objectives/time distributions.
    common = {"effective_sampling_steps": 50, "solver": "heun", "sigma_min": .0001}
    assert A._inference_settings({"sampling_settings": {**common, "mean_path_loss_weight": .5}}) == A._inference_settings({"sampling_settings": {**common, "time_sampling": "uniform"}})




def test_runner_fingerprint_convention_is_not_compact_acceptance_hash(source, tmp_path):
    run_path = source.root / "workflow/runs/full_selected/diffusion_unet/101/run_contract.json"
    payload = A.read(run_path)
    assert payload["sha256"] == A.config_fingerprint(payload["contract"])
    assert payload["sha256"] != A.canonical_hash(payload["contract"])
    assert A.load_runner_contract(run_path, payload["sha256"]) == payload["contract"]
    wrong = dict(payload, sha256=A.canonical_hash(payload["contract"]))
    A.write(tmp_path / "wrong_convention.json", wrong)
    with pytest.raises(A.AcceptanceError, match="Runner contract fingerprint"):
        A.load_runner_contract(tmp_path / "wrong_convention.json", wrong["sha256"])
