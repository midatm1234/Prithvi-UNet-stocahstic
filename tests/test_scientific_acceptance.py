"""Acceptance is a scientific contract; these tests exercise its mechanics only."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from granitewxc.refinement.acceptance import (
    AcceptanceError, ScientificSelector, acceptance_regions, canonical_hash,
    candidate_point_decision, classify_interval, collect_year_statistics,
    evaluate_scientific_acceptance, freeze_acceptance_contract, load_contract,
    load_year_statistics, merge_year_statistics, paired_resampling_plan,
    registered_years, save_year_statistics,
    simultaneous_max_statistic_intervals, _product_metrics,
)
from granitewxc.refinement.metrics import ErrorAccumulator, map_difference_metrics


@pytest.fixture
def contract(tmp_path):
    result = load_contract("examples/CORDEX_ML/scientific_acceptance.yaml")
    result = copy.deepcopy(result)
    result["contract_name"] = "synthetic_test_protocol_not_CORDEX_acceptance"
    result["regions"]["widths"] = [1]
    result["bootstrap"]["replicates"] = 19
    artifact = tmp_path / "synthetic_diagnostics.json"
    artifact.write_text('{"synthetic_only":true}')
    result["_test_artifact"] = {"path": str(artifact), "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
    return result


def dates_for_years(years):
    import calendar
    return [f"{year}-{month:02d}-{day:02d}"
            for year in years for month in range(1, 13)
            for day in range(1, (28 if month == 2 else calendar.monthrange(year, month)[1]) + 1)]


def synthetic_record(contract, variable="pr", offset=0.2, seed=101):
    years = np.array(registered_years(contract))
    lat, lon = np.array([-34., -32., -30., -28.]), np.array([29., 30., 31., 32.])
    pattern = np.arange(16).reshape(4, 4) * 0.01
    truth = np.broadcast_to(pattern + (3 if variable == "pr" else 300), (len(years), 4, 4)).copy()
    baseline = truth + 1
    members = truth[:, None] + offset + np.linspace(-0.5, 0.5, 20)[None, :, None, None]
    record = collect_year_statistics(members, baseline, truth, years, lat, lon)
    # A complete synthetic noleap year consists of 365 identical copies.
    for fields in record["products"].values():
        for key in fields:
            fields[key] *= 365
    digest = "a" * 64
    record["metadata"].update({
        "observed_dates_per_year": {str(y): 365 for y in years},
        "date_ids": dates_for_years(years), "calendar": "noleap",
        "authenticated": True, "training_seed": seed, "units": contract["units"][variable],
        "edge_fallback_or_suppression": False, "effective_sampled_correction": True,
        "sampling_seed_ids": list(range(20)), "data_scope": "component_validation",
        "fit_years": registered_years(contract, "component_fit"),
        "raw_members_saved": True,
        **{key: digest for key in (
            "phase1_sha256", "baseline_sha256", "target_sha256", "dates_sha256",
            "mask_sha256", "coordinates_sha256", "config_sha256",
            "normalization_sha256", "checkpoint_sha256")},
        "diagnostics": {
            name: {"status": "complete", "artifact": contract["_test_artifact"]["path"], "artifact_sha256": contract["_test_artifact"]["sha256"]}
            for name in contract["diagnostics"]["required"]
        },
    })
    return record


def complete_evidence(contract):
    plan = paired_resampling_plan(contract)
    records, auxiliary = {}, {}
    for head in contract["heads"]:
        records[head] = {
            str(seed): {var: synthetic_record(contract, var, seed=seed) for var in contract["variables"]}
            for seed in contract["training_seeds"]}
        auxiliary[head] = {
            var: {
                name: {
                    "per_seed": {str(seed): {"reference": 1., "candidate": 0.1}
                                 for seed in contract["training_seeds"]},
                    "bootstrap_changes": [-0.9] * contract["bootstrap"]["replicates"],
                    "resampling_plan_sha256": plan["sha256"],
                } for name in ("p99_absolute_error", "observed_p99_event_rmse")
            } for var in contract["variables"]}
    return {"contract_sha256": canonical_hash(contract), "scope": "component_validation",
            "records": records, "auxiliary_metrics": auxiliary}


def test_repository_frozen_contract_unchanged():
    root = Path("artifacts/refinement_validation/scientific_acceptance_20260909T230807Z")
    contract = load_contract("examples/CORDEX_ML/scientific_acceptance.yaml")
    if not root.exists():
        pytest.skip("Run-local immutable protocol snapshots are not distributed with the package.")
    assert canonical_hash(contract) == "f649faeb83c7a5722cd975e5df8cf6effa6145d3d1822f4a410bb9c9a7245b30"
    assert load_contract(root / "frozen_scientific_acceptance_v2.json") == contract
    assert json.loads((root / "frozen_scientific_acceptance.json").read_text())["sha256"] == "a5747578cf4b7f2eaba89aa5501b5a25d18b9571614f136b5cf7368857772695"
    assert "correction_gate_must_equal_one" not in contract["gates"]


def test_freeze_refuses_mutation(tmp_path):
    source = tmp_path / "contract.json"
    source.write_text(json.dumps({"schema_version": 1, "heads": ["x"], "variables": ["y"]}))
    old = freeze_acceptance_contract(source, tmp_path / "frozen")
    assert freeze_acceptance_contract(source, tmp_path / "frozen") == old
    source.write_text(json.dumps({"schema_version": 1, "heads": ["x"], "variables": ["z"]}))
    with pytest.raises(AcceptanceError, match="Refusing to change"):
        freeze_acceptance_contract(source, tmp_path / "frozen")


def test_shared_canonical_metric_implementations():
    from examples.CORDEX_ML.utils.evaluate_refinement_outputs import (
        _ErrorAccumulator, _map_difference_metrics,
    )
    assert _ErrorAccumulator is ErrorAccumulator
    assert _map_difference_metrics is map_difference_metrics


def test_exact_sufficient_statistics_zero_and_signed_corrections(contract):
    truth = np.zeros((3, 4, 4))
    baseline = np.ones_like(truth)
    members = np.stack((truth - 0.2, truth + 0.4), axis=1)
    record = collect_year_statistics(members, baseline, truth, [1977, 1977, 2096],
                                    [-34, -32, -30, -28], [29, 30, 31, 32])
    fields = record["products"]["refined"]
    assert fields["count"].sum() == 48
    assert fields["prediction_sum"].sum() == pytest.approx(4.8)
    assert np.all(record["products"]["refined"]["error_sum"] < record["products"]["phase1"]["error_sum"])
    scores = _product_metrics(fields, np.ones(2), {"full": np.ones((4, 4), bool)})["full"]
    accumulator = ErrorAccumulator()
    accumulator.update(members.mean(axis=1), truth)
    assert scores["daily_rmse"] == pytest.approx(accumulator.result()["rmse"])
    assert scores["mean_absolute_climatological_bias"] == pytest.approx(0.1)


def test_missing_members_cannot_shrink_support():
    truth = np.zeros((2, 4, 4))
    members = np.zeros((2, 2, 4, 4))
    members[0, 1, 0, 0] = np.nan
    with pytest.raises(AcceptanceError, match="Nonfinite refinement"):
        collect_year_statistics(members, truth, truth, [1977, 2096], range(4), range(4))
    mask = np.ones_like(truth, dtype=bool)
    mask[0, 0, 0] = False
    result = collect_year_statistics(members, truth, truth, [1977, 2096], range(4), range(4), valid_mask=mask)
    assert result["products"]["refined"]["count"].sum() == 31


def test_chunk_merge_and_npz_round_trip(contract, tmp_path):
    years = np.array([1977, 1977, 1978, 2096])
    truth = np.arange(64).reshape(4, 4, 4) / 10
    members = np.stack((truth - 1, truth + 0.5), axis=1)
    kwargs = {"lat": [-34, -32, -30, -28], "lon": [29, 30, 31, 32]}
    full = collect_year_statistics(members, truth + 2, truth, years, **kwargs)
    first = collect_year_statistics(members[:1], truth[:1] + 2, truth[:1], years[:1], **kwargs)
    second = collect_year_statistics(members[1:], truth[1:] + 2, truth[1:], years[1:], **kwargs)
    merged = merge_year_statistics(first, second)
    for product in full["products"]:
        for field in full["products"][product]:
            np.testing.assert_allclose(merged["products"][product][field], full["products"][product][field])
    path = tmp_path / "years.npz"
    digest = save_year_statistics(path, merged)
    loaded = load_year_statistics(path, digest)
    np.testing.assert_array_equal(loaded["years"], years[[0, 2, 3]])
    with pytest.raises(AcceptanceError, match="hash mismatch"):
        load_year_statistics(path, "b" * 64)


def test_geographic_regions_and_unbounded_southeast(contract):
    masks = acceptance_regions([-38, -34, -30, -26], [29, 30, 31, 32], contract)
    assert masks["north_1"][-1].all()
    assert masks["south_1"][0].all()
    assert masks["southeast_precipitation_box"][0].all()
    flipped = acceptance_regions([-26, -30, -34, -38], [32, 31, 30, 29], contract)
    np.testing.assert_array_equal(flipped["southeast_precipitation_box"][::-1, ::-1],
                                  masks["southeast_precipitation_box"])


def test_paired_whole_year_plan_is_stratified_and_reproducible(contract):
    first = paired_resampling_plan(contract)
    second = paired_resampling_plan(contract)
    assert first["sha256"] == second["sha256"]
    np.testing.assert_array_equal(first["year_weights"], second["year_weights"])
    assert np.all(first["year_weights"][:, :4].sum(axis=1) == 4)
    assert np.all(first["year_weights"][:, 4:].sum(axis=1) == 4)
    assert first["seed_indices"].shape == (19, 3)
    assert set(np.unique(first["seed_indices"])) == {0, 1, 2}


def test_simultaneous_intervals_do_not_treat_duplicated_pixels_as_samples():
    rng = np.random.default_rng(4)
    draws = rng.normal(size=(1999, 2))
    one = simultaneous_max_statistic_intervals([0.], draws[:, :1], 0.95)
    two = simultaneous_max_statistic_intervals([0., 0.], draws, 0.95)
    repeated = simultaneous_max_statistic_intervals(np.zeros(100), np.repeat(draws[:, :1], 100, axis=1), 0.95)
    assert two["critical_value"] >= one["critical_value"]
    assert repeated["critical_value"] == pytest.approx(one["critical_value"])


@pytest.mark.parametrize("lower,upper,limit,expected", [
    (-0.10, -0.03, -0.02, "PASS"),
    (-0.03, 0.02, 0.01, "INCONCLUSIVE"),
    (0.02, 0.04, 0.01, "FAIL"),
    (-0.01, 0.01, 0.01, "INCONCLUSIVE"),
    (-0.01, 0.0, 0.0, "INCONCLUSIVE"),
    (-0.06, -0.05, -0.05, "PASS"),
])
def test_strict_intervals(lower, upper, limit, expected):
    assert classify_interval(lower, upper, limit) == expected


def test_empty_evidence_cannot_pass_any_combination(contract):
    report = evaluate_scientific_acceptance({}, contract)
    assert report["status"] == "INCONCLUSIVE"
    assert report["production_promotion"] is False
    assert len(report["combinations"]) == 4
    assert all(entry["status"] == "INCONCLUSIVE" for head in report["combinations"].values() for entry in head.values())


def test_complete_component_can_pass_but_cannot_promote_production(contract):
    evidence = complete_evidence(contract)
    report = evaluate_scientific_acceptance(evidence, contract)
    assert not report["prerequisite_failures"]
    assert report["bootstrap"]["computed"]
    assert report["status"] == "PASS"
    assert report["production_promotion"] is False
    assert report["end_to_end"]["status"] == "INCONCLUSIVE"
    assert report["bootstrap"]["member_sets_resampled"] is False


def test_missing_head_or_seed_forbids_smaller_family_promotion(contract):
    evidence = complete_evidence(contract)
    evidence["records"].pop(contract["heads"][-1])
    report = evaluate_scientific_acceptance(evidence, contract)
    assert report["status"] == "INCONCLUSIVE"
    assert not report["bootstrap"]["computed"]
    assert any("missing registered" in reason for reason in report["prerequisite_failures"])


def test_changed_phase1_array_rejected_even_if_hash_claims_match(contract):
    evidence = complete_evidence(contract)
    evidence["records"][contract["heads"][1]]["101"]["pr"]["products"]["phase1"]["prediction_sum"][0, 0, 0] += 1
    with pytest.raises(AcceptanceError, match="Phase-1 arrays differ"):
        evaluate_scientific_acceptance(evidence, contract)


def test_screening_stays_provisional_and_bad_precip_cannot_hide(contract, tmp_path):
    evidence = complete_evidence(contract)
    head = contract["heads"][0]
    for seed in evidence["records"][head]:
        record = synthetic_record(contract, "pr", offset=2, seed=int(seed))
        record["metadata"]["data_scope"] = "screening"
        evidence["records"][head][seed]["pr"] = record
    decision = candidate_point_decision(evidence, contract, head)
    assert not decision["scientifically_eligible_candidate"]
    assert any("/pr/" in reason for reason in decision["rejection_reasons"])
    source = tmp_path / "source.ckpt"
    source.write_bytes(b"checkpoint-v1")
    selector = ScientificSelector(tmp_path / "selection", contract, head)
    saved = selector.consider(source, evidence)
    assert saved["selected"]
    assert (tmp_path / "selection/last.ckpt").read_bytes() == b"checkpoint-v1"
    assert (tmp_path / "selection/nearest_provisional_candidate.ckpt").exists()
    assert not (tmp_path / "selection/best_scientific_candidate.ckpt").exists()
    selector.consider(tmp_path / "selection/last.ckpt", evidence)


def test_scientific_candidate_distinct_from_acceptance_and_last(contract, tmp_path):
    evidence = complete_evidence(contract)
    head = contract["heads"][0]
    checkpoint = tmp_path / "source.ckpt"
    checkpoint.write_bytes(b"eligible")
    selector = ScientificSelector(tmp_path / "selection", contract, head)
    decision = selector.consider(checkpoint, evidence)
    assert decision["scientifically_eligible_candidate"]
    assert not decision["production_accepted"]
    assert decision["acceptance_status"] == "NOT_EVALUATED"
    assert (tmp_path / "selection/best_scientific_candidate.ckpt").read_bytes() == b"eligible"
    for variables in evidence["records"][head].values():
        variables["pr"]["metadata"]["data_scope"] = "screening"
    checkpoint.write_bytes(b"later-provisional")
    selector.consider(checkpoint, evidence)
    assert (tmp_path / "selection/last.ckpt").read_bytes() == b"later-provisional"
    assert (tmp_path / "selection/best_scientific_candidate.ckpt").read_bytes() == b"eligible"


def test_selector_rejects_contract_drift(contract, tmp_path):
    head = contract["heads"][0]
    ScientificSelector(tmp_path, contract, head)
    changed = copy.deepcopy(contract)
    changed["near_zero_reference"]["pr"] = 10
    with pytest.raises(AcceptanceError, match="different head or frozen contract"):
        ScientificSelector(tmp_path, changed, head)


def test_selector_rejects_missing_authentication_and_wrong_contract(contract):
    evidence = complete_evidence(contract)
    head = contract["heads"][0]
    evidence["contract_sha256"] = "b" * 64
    evidence["records"][head]["101"]["pr"]["metadata"]["authenticated"] = False
    result = candidate_point_decision(evidence, contract, head)
    assert not result["scientifically_eligible_candidate"]
    assert any("contract hash" in reason for reason in result["rejection_reasons"])
    assert any("unauthenticated" in reason for reason in result["rejection_reasons"])


def test_duplicate_chunk_dates_are_rejected(contract):
    record = synthetic_record(contract)
    with pytest.raises(AcceptanceError, match="Repeated dates"):
        merge_year_statistics(record, record)


def test_missing_or_replaced_diagnostic_cannot_pass(contract):
    evidence = complete_evidence(contract)
    Path(contract["_test_artifact"]["path"]).write_text("replaced")
    result = evaluate_scientific_acceptance(evidence, contract)
    assert result["status"] == "INCONCLUSIVE"
    assert any("artifact missing or hash mismatch" in reason for reason in result["prerequisite_failures"])


def test_collapsed_ensemble_rejected_by_point_guardrail(contract):
    evidence = complete_evidence(contract)
    head = contract["heads"][0]
    for variables in evidence["records"][head].values():
        for record in variables.values():
            fields = record["products"]["refined"]
            fields["spread_sum"].fill(0)
            fields["identical_sum"] = fields["count"].copy()
    decision = candidate_point_decision(evidence, contract, head)
    assert not decision["scientifically_eligible_candidate"]
    assert any("spread_to_phase1_mae" in reason for reason in decision["rejection_reasons"])
    assert any("identical_fraction" in reason for reason in decision["rejection_reasons"])


def test_mean_component_requires_deterministic_crps_added_value(contract):
    evidence = complete_evidence(contract)
    head = contract["heads"][0]
    for variables in evidence["records"][head].values():
        for record in variables.values():
            record["metadata"]["mean_component_used"] = True
    decision = candidate_point_decision(evidence, contract, head)
    assert not decision["scientifically_eligible_candidate"]
    assert any("/crps/control" in reason for reason in decision["rejection_reasons"])


def test_rectangle_sums_include_mask_holes_and_odd_edges():
    from granitewxc.refinement.acceptance import _mask_rectangles, _rectangular_region_sums
    rng = np.random.default_rng(5)
    fields = rng.normal(size=(3, 7, 9))
    masks = [rng.random((7, 9)) > 0.3, np.ones((7, 9), bool),
             np.zeros((7, 9), bool)]
    actual = _rectangular_region_sums(fields, [_mask_rectangles(mask) for mask in masks])
    expected = np.stack([fields[:, mask].sum(axis=1) for mask in masks], axis=-1)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_accelerated_bootstrap_matches_canonical_reference(contract):
    from granitewxc.refinement.acceptance import (
        bootstrap_comparison_changes, comparison_rows,
    )
    evidence = complete_evidence(contract)
    records = evidence["records"]
    # Nonuniform yearly signed corrections make resampled climatologies vary.
    for head in records:
        for seed, variables in records[head].items():
            for variable, record in variables.items():
                factors = np.linspace(0.6, 1.4, len(record["years"]))[:, None, None]
                old_error = record["products"]["refined"]["error_sum"].copy()
                fields = record["products"]["refined"]
                fields["error_sum"] *= factors
                fields["absolute_error_sum"] *= factors
                fields["squared_error_sum"] *= factors**2
                fields["prediction_sum"] += fields["error_sum"] - old_error
    rows, _ = comparison_rows(records, contract)
    plan = paired_resampling_plan(contract)
    fast = bootstrap_comparison_changes(records, contract, rows, plan, batch_size=7)
    reference = []
    for weights, seeds in zip(plan["year_weights"], plan["seed_indices"]):
        result, _ = comparison_rows(records, contract, year_weights=weights, seed_indices=seeds)
        assert [row["key"] for row in result] == [row["key"] for row in rows]
        reference.append([row["change"] for row in result])
    np.testing.assert_allclose(fast, reference, rtol=1e-10, atol=1e-10)

def test_variable_applicability_and_family_ensemble_pairing(contract):
    evidence = complete_evidence(contract)
    for seeds in evidence["records"].values():
        for variables in seeds.values():
            for name in ("wet_day_frequency_intensity", "wet_day_brier_reliability"):
                variables["tasmax"]["metadata"]["diagnostics"][name]["status"] = "not_applicable"
    assert evaluate_scientific_acceptance(evidence, contract)["status"] == "PASS"
    first = evidence["records"][contract["heads"][0]][str(contract["training_seeds"][0])]["pr"]
    first["metadata"]["ensemble_size"] += 1
    result = evaluate_scientific_acceptance(evidence, contract)
    assert result["status"] == "INCONCLUSIVE"
    assert any("ensemble size or sampler seed" in item for item in result["prerequisite_failures"])
