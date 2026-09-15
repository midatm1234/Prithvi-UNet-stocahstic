"""Supplemental controls preserve original evidence and change one factor each."""
import copy
from pathlib import Path

import pytest

from examples.CORDEX_ML import cordex_scientific_supplemental as S
from examples.CORDEX_ML.cordex_scientific_full_plan import prepare_full
from granitewxc.refinement.scientific_data import canonical_sha256, file_sha256
from test_scientific_workflow import queue


def completed(path, data, *, numerical=()):
    states = {job["id"]: {"status": "FAIL_NUMERICAL_EXECUTION" if job["id"] in numerical
                         else "COMPLETED_REGISTERED_STOP", "effective_job": copy.deepcopy(job)}
              for job in data["jobs"]}
    state_path = path.with_name(path.stem + "_execution.json")
    S.workflow.write(state_path, {"queue_sha256": data["sha256"],
                                 "status": "REGISTERED_QUEUE_COMPLETED", "jobs": states})
    return state_path


def selection(job, score):
    folder = Path(job["output"]); folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "nearest_provisional_candidate.ckpt"
    checkpoint.write_bytes(job["id"].encode())
    point = {"score": [0, score, .02], "selected_checkpoint": str(checkpoint),
             "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint)}
    S.workflow.write(folder / "scientific_selection.json", {"nearest_provisional": point})


def test_exact_six_independent_controls_are_registered_before_training(queue):
    original_path, original = queue
    before = original_path.read_bytes()
    path, supplement = S.register_conditioning_controls(original_path)
    assert len(supplement["jobs"]) == 6
    normalized, gates = supplement["jobs"][:4], supplement["jobs"][4:]
    assert [job["head"] for job in normalized] == list(S.workflow.HEADS)
    assert [job["head"] for job in gates] == ["diffusion_transformer", "flow_matching_transformer"]
    for job in supplement["jobs"]:
        label = "native" if job in normalized else "native_production_gate"
        source = next(item for item in original["jobs"] if item["id"] == label + "/" + job["head"])
        changed = {key for key in source.keys() | job.keys() if source.get(key) != job.get(key)}
        assert changed == ({"id", "output", "normalize_predictors"} if job in normalized else {"id", "output", "recipe"})
        assert job["seed"] == 101 and job["stage"] == "screen" and job["formulation"] == "direct"
        args = S.workflow.train_arguments(job, supplement)
        assert "--resume" not in args and "--mean-checkpoint" not in args
        assert ("--normalize-predictors" in args) == (job in normalized)
    registration = S.workflow.read(supplement["supplemental_registration"])
    assert registration["registered_before_supplemental_training"] is True
    assert registration["job_counts"]["total"] == 6 and registration["factorial_combinations"] is False
    assert registration["original_queue_sha256"] == original["sha256"]
    assert original_path.read_bytes() == before
    assert S.register_conditioning_controls(original_path) == (path, supplement)
    assert not path.with_name(path.stem + "_execution.json").exists()


def test_execution_is_independent_but_full_selection_waits_for_both_queues(queue, monkeypatch):
    original_path, original = queue
    path, supplement = S.register_conditioning_controls(original_path)
    calls = []
    monkeypatch.setattr(S.workflow, "run_queue", lambda *args, **kwargs: calls.append((args, kwargs)) or 75)
    assert S.run_supplemental(path) == 75
    assert calls == [((path,), {"only_stage": "screen"})]
    completed(path, supplement)
    with pytest.raises(ValueError, match="unfinished"):
        S.combine_development_evidence(original_path, path)
    state_path = completed(original_path, original)
    state = S.workflow.read(state_path); state["jobs"][original["jobs"][-1]["id"]]["status"] = "PENDING_DEPENDENCY"
    S.workflow.write(state_path, state)
    assert S.run_supplemental(path) == 75
    with pytest.raises(ValueError, match="unfinished"):
        S.combine_development_evidence(original_path, path)
    completed(original_path, original)
    assert S.combine_development_evidence(original_path, path).is_file()


@pytest.mark.parametrize("which", ["original", "supplemental"])
def test_combination_refuses_unfinished_evidence(queue, which):
    original_path, original = queue
    path, supplement = S.register_conditioning_controls(original_path)
    original_state = completed(original_path, original)
    supplement_state = completed(path, supplement)
    target = original_state if which == "original" else supplement_state
    value = S.workflow.read(target); value["status"] = "RUNNING"
    S.workflow.write(target, value)
    with pytest.raises(ValueError, match="unfinished"):
        S.combine_development_evidence(original_path, path)
    assert not (Path(original["run_root"]) / "workflow/combined_development_queue.json").exists()


def test_combination_retains_unfavorable_and_failed_results_and_is_immutable(queue):
    original_path, original = queue
    path, supplement = S.register_conditioning_controls(original_path)
    original_state = completed(original_path, original)
    failed = supplement["jobs"][-1]["id"]
    supplemental_state = completed(path, supplement, numerical=(failed,))
    original_bytes, receipt_bytes = original_path.read_bytes(), original_state.read_bytes()
    combined_path = S.combine_development_evidence(original_path, path)
    combined = S.workflow.read(combined_path)
    combined_state = S.workflow.read(combined_path.with_name(combined_path.stem + "_execution.json"))
    assert combined["jobs"] == original["jobs"] + supplement["jobs"]
    assert len(combined["jobs"]) == 46
    assert combined_state["jobs"][failed]["status"] == "FAIL_NUMERICAL_EXECUTION"
    assert combined["execution_allowed"] is False and combined["all_outcomes_retained"] is True
    assert original_path.read_bytes() == original_bytes and original_state.read_bytes() == receipt_bytes
    frozen = combined_path.read_bytes()
    assert S.combine_development_evidence(original_path, path) == combined_path
    assert combined_path.read_bytes() == frozen
    changed = S.workflow.read(supplemental_state)
    changed["jobs"][failed]["detail"] = "changed after the evidence snapshot"
    S.workflow.write(supplemental_state, changed)
    with pytest.raises(ValueError, match="overwrite"):
        S.combine_development_evidence(original_path, path)
    assert combined_path.read_bytes() == frozen


@pytest.mark.parametrize("mutation", ["hash", "coverage", "registered_factor", "execution_hash", "skipped"])
def test_rejects_hash_coverage_factor_and_unexecuted_control_mismatches(queue, mutation):
    original_path, original = queue
    path, supplement = S.register_conditioning_controls(original_path)
    completed(original_path, original)
    state_path = completed(path, supplement)
    if mutation in ("hash", "coverage", "registered_factor"):
        altered = copy.deepcopy(supplement)
        if mutation == "coverage":
            altered["jobs"].pop()
        else:
            altered["jobs"][0]["normalize_predictors"] = False
        if mutation != "hash":
            altered["sha256"] = canonical_sha256({key: value for key, value in altered.items() if key != "sha256"})
        S.workflow.write(path, altered)
    else:
        state = S.workflow.read(state_path)
        if mutation == "execution_hash":
            state["queue_sha256"] = "f" * 64
        else:
            state["jobs"][supplement["jobs"][0]["id"]]["status"] = "CONDITION_NOT_TRIGGERED"
        S.workflow.write(state_path, state)
    with pytest.raises(ValueError, match="hash|coverage|unfinished"):
        S.combine_development_evidence(original_path, path)


def test_refuses_late_registration_or_overwriting_existing_registration(queue):
    original_path, original = queue
    expected_path, supplement, registration_path, _ = S._definition(original_path)
    checkpoint = Path(supplement["jobs"][0]["output"]) / "last.ckpt"
    checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b"existing training")
    with pytest.raises(ValueError, match="before any supplemental training"):
        S.register_conditioning_controls(original_path)
    assert not expected_path.exists() and not registration_path.exists()
    checkpoint.rename(checkpoint.with_name("preserved.ckpt"))
    assert checkpoint.with_name("preserved.ckpt").read_bytes() == b"existing training"


def test_registration_cannot_change_after_freezing(queue):
    original_path, _ = queue
    path, supplement = S.register_conditioning_controls(original_path)
    registration_path = Path(supplement["supplemental_registration"])
    altered = S.workflow.read(registration_path); altered["training_seed"] = 999
    S.workflow.write(registration_path, altered)
    frozen = registration_path.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        S.register_conditioning_controls(original_path)
    assert registration_path.read_bytes() == frozen


def test_normalized_selected_recipe_reaches_full_seeds_and_control_stays_unchanged(queue):
    original_path, original = queue
    path, supplement = S.register_conditioning_controls(original_path)
    original_failures = [job["id"] for job in original["jobs"] if not job["id"].startswith("native/")]
    completed(original_path, original, numerical=original_failures)
    for job in original["jobs"]:
        if job["id"].startswith("native/"):
            selection(job, .4)
    completed(path, supplement)
    for job in supplement["jobs"]:
        selection(job, .1 if job["id"].startswith("normalized_native/") else .6)
    combined_path = S.combine_development_evidence(original_path, path)
    full = S.workflow.read(prepare_full(combined_path))
    assert len(full["jobs"]) == 24
    for job in full["jobs"]:
        assert job["seed"] in (101, 202, 303)
        if job["comparison_group"] == "selected":
            assert job["recipe"] == "native" and job["normalize_predictors"] is True
            assert "--normalize-predictors" in S.workflow.train_arguments(job, full)
        else:
            assert job["recipe"] == "existing_full_recipe" and job["formulation"] == "direct"
            assert "normalize_predictors" not in job
    selection_record = S.workflow.read(Path(full["run_root"]) / "workflow/selected_development_recipes.json")
    assert set(selection_record["failed_experiments"]) == set(original_failures)
    assert all(value["job"]["id"].startswith("normalized_native/") for value in selection_record["selected"].values())


def test_parent_hash_and_original_head_coverage_are_required(queue):
    path, original = queue
    missing = copy.deepcopy(original)
    missing["jobs"] = [job for job in missing["jobs"] if job["id"] != "native/diffusion_unet"]
    missing["sha256"] = canonical_sha256({key: value for key, value in missing.items() if key != "sha256"})
    S.workflow.write(path, missing)
    with pytest.raises(ValueError, match="cover each"):
        S.register_conditioning_controls(path)
    altered = copy.deepcopy(original); altered["registered_plan_sha256"] = "0" * 64
    altered["sha256"] = canonical_sha256({key: value for key, value in altered.items() if key != "sha256"})
    S.workflow.write(path, altered)
    with pytest.raises(ValueError, match="parent plan"):
        S.register_conditioning_controls(path)
