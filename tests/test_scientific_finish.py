"""Completion orchestration uses synthetic receipts; no scientific jobs run."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from examples.CORDEX_ML import cordex_scientific_finish as F


@pytest.fixture
def registered(tmp_path, monkeypatch):
    plan = {"screening_seed": 101, "training_seeds": [101, 202, 303],
            "acceptance_contract_sha256": "a"*64, "validation": {"sampling_seed": 7301977}}
    plan_path = tmp_path/"registered_experiment_plan.json"
    text = json.dumps(plan, indent=2)+"\n"
    plan_path.write_text(text, encoding="utf8")
    plan_path.with_suffix(".sha256").write_text(hashlib.sha256(text.encode()).hexdigest())
    path, queue = F.workflow.create_queue(tmp_path)
    registration = {"version": "paired_selected_fresh_stochastic_control_v1",
        "acceptance_contract_sha256": queue["acceptance_contract_sha256"],
        "parent_experiment_plan_sha256": queue["registered_plan_sha256"]}
    registration["sha256"] = F.canonical_sha256(registration)
    F.workflow.write(tmp_path/"fresh_control_contrast_registration.json", registration)
    statuses = {job["id"]: {"status": "CONDITION_NOT_TRIGGERED"} for job in queue["jobs"]}
    for head in F.workflow.HEADS:
        job = next(job for job in queue["jobs"] if job["id"] == "native/"+head)
        folder = Path(job["output"]); folder.mkdir(parents=True)
        checkpoint = folder/"nearest_provisional_candidate.ckpt"; checkpoint.write_bytes(head.encode())
        selection = {"score": [0, .2, .1], "checkpoint": str(checkpoint),
                     "selected_checkpoint": str(checkpoint), "checkpoint_sha256": F.file_sha256(checkpoint)}
        F.workflow.write(folder/"scientific_selection.json", {"nearest_provisional": selection})
        statuses[job["id"]] = {"status": "COMPLETED_REGISTERED_STOP"}
    F.workflow.write(path.with_name(path.stem+"_execution.json"),
                     {"queue_sha256": queue["sha256"], "status": "REGISTERED_QUEUE_COMPLETED", "jobs": statuses})
    (tmp_path/"cache").mkdir()
    (tmp_path/"cache"/"phase1.h5").write_bytes(b"synthetic cache for orchestration fixture")
    F.workflow.write(tmp_path/"cache"/"phase1.manifest.json",
                     {"complete": True, "cache_sha256": F.file_sha256(tmp_path/"cache"/"phase1.h5"), "phase1_checkpoint_sha256": "c"*64,
                      "variables": ["pr", "tasmax"],
                      "splits": {"dates": ["1977-01-07T12:00:00", "1977-07-21T12:00:00"],
                                 "validation_indices": [0, 1]}})
    monkeypatch.setattr(F, "free_disk_bytes", lambda paths: {"C:": 100*1024**3, "D:": 100*1024**3})
    monkeypatch.setattr(F, "active_processes", lambda paths: [])
    args = SimpleNamespace(development_queue=str(path), poll_seconds=1., minimum_free_gib=30., once=True)
    return SimpleNamespace(root=tmp_path, path=path, queue=queue, args=args)


def fake_full_training(path, **kwargs):
    queue = F.read_queue(path)
    states = {}
    for job in queue["jobs"]:
        folder = Path(job["output"]); folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder/"nearest_provisional_candidate.ckpt"; checkpoint.write_bytes(job["id"].encode())
        F.workflow.write(folder/"scientific_selection.json", {"nearest_provisional": {
            "selected_checkpoint": str(checkpoint), "checkpoint_sha256": F.file_sha256(checkpoint)}})
        F.workflow.write(folder/"execution_status.json", {"status": "STOPPED_REGISTERED_PATIENCE_PROVISIONAL"})
        states[job["id"]] = {"status": "COMPLETED_REGISTERED_STOP"}
    F.workflow.write(Path(path).with_name(Path(path).stem+"_execution.json"),
                     {"queue_sha256": queue["sha256"], "status": "REGISTERED_QUEUE_COMPLETED", "jobs": states})
    return 0


def write_prediction(path, identity, queue, *, partial=False):
    path = Path(path)
    destination = path.with_suffix(".partial.h5") if partial else path
    destination.parent.mkdir(parents=True, exist_ok=True)
    cache = F.workflow.read(Path(queue["run_root"])/"cache"/"phase1.manifest.json")
    dates = cache["splits"]["dates"]
    opaque = F.canonical_sha256(identity)
    with h5py.File(destination, "w") as f:
        f.create_dataset("dates", data=np.asarray(dates, dtype=object), dtype=h5py.string_dtype())
        f.attrs.update(ensemble_size=identity["members"], training_seed=identity["training_seed"],
                       sampling_seed=identity["sampling_seed"], random_stream_version=identity["random_stream_version"],
                       variables=json.dumps(identity["variables"]), prediction_identity=opaque,
                       data_scope="nested_member_diagnostic" if identity["diagnostic"] else "component_validation",
                       completed_dates=1 if partial else len(dates), complete=not partial)
    if not partial:
        F.workflow.write(path.with_suffix(".manifest.json"), {
            "complete": True, "prediction_identity": opaque, "sha256": F.file_sha256(path),
            "authentication": {key: identity[key] for key in ("checkpoint_sha256", "source_cache_sha256",
                "phase1_checkpoint_sha256", "acceptance_contract_sha256", "plan_sha256")}})


def fake_launch_factory(registered, calls):
    def launch(argv, log, state, state_path, root, poll_seconds):
        calls.append(list(argv))
        path = Path(F.option(argv, "--output"))
        queue = F.read_queue(root/"workflow"/"full_queue.json")
        if argv[5].endswith("cordex_scientific_prediction.py"):
            job = next(job for job in queue["jobs"] if job["output"] == F.option(argv, "--training-output"))
            entry = {"members": int(F.option(argv, "--members")), "diagnostic": "--diagnostic-dates" in argv}
            identity = F.prediction_identity(entry, job, queue)
            partial = path.with_suffix(".partial.h5")
            if partial.exists():
                assert "--resume" in argv
                # Simulate the native sampler's successful identity-checked
                # completion, preserving bytes until this synthetic transition.
                partial.rename(path.with_suffix(".retained_partial_fixture"))
            write_prediction(path, identity, queue)
        elif argv[6] == "collect":
            source = Path(F.option(argv, "--prediction"))
            job = next(job for job in queue["jobs"] if str(source).startswith(str(Path(queue["artifact_root"])/"workflow"/"assessments"/job["id"])))
            path.mkdir(parents=True)
            statistics = path/"statistics.npz"; statistics.write_bytes(b"synthetic-statistic-reference")
            checkpoint, _ = F.resolve_selected_checkpoint(job["output"])
            authentication = {"authenticated": True, "prediction_path": str(source), "prediction_sha256": F.file_sha256(source),
                "cache_path": str(root/"cache"/"phase1.h5"), "cache_sha256": F.file_sha256(root/"cache"/"phase1.h5"),
                "checkpoint_path": str(checkpoint), "checkpoint_sha256": F.file_sha256(checkpoint)}
            F.workflow.write(path/"authentication.json", authentication)
            reference = {"path": str(statistics), "sha256": F.file_sha256(statistics)}
            F.workflow.write(path/"evidence.json", {"contract_sha256": queue["acceptance_contract_sha256"],
                "authentication": {"artifact": str(path/"authentication.json"), "artifact_sha256": F.file_sha256(path/"authentication.json")},
                "records": {job["head"]: {str(job["seed"]): {"pr": reference, "tasmax": reference}}}})
            F.workflow.write(path/"completion.json", {"operation": "collect", "complete": True,
                "contract_sha256": queue["acceptance_contract_sha256"], "artifact_sha256": F.file_sha256(path/"evidence.json")})
        elif argv[5] == F.ATTRIBUTION:
            selected, control = Path(F.option(argv, "--selected")), Path(F.option(argv, "--control"))
            registration = F.workflow.read(F.option(argv, "--registration"))
            F.workflow.write(path/"evidence.json", {"contract_sha256": queue["acceptance_contract_sha256"],
                "input_evidence": [{"path": str(item), "sha256": F.file_sha256(item)} for item in (selected, control)]})
            F.workflow.write(path/"comparison.json", {
                "status": "COMPLETE", "contract_sha256": queue["acceptance_contract_sha256"],
                "registration_sha256": registration["sha256"], "acceptance_decisions_unchanged": True,
                "acceptance_families_combined": False, "production_promotion": False})
            F.workflow.write(path/"completion.json", {"operation": "attribution", "complete": True,
                "contract_sha256": queue["acceptance_contract_sha256"],
                "artifact_sha256": F.file_sha256(path/"comparison.json")})
        elif argv[6] == "compare-families":
            selected, control = Path(F.option(argv, "--selected")), Path(F.option(argv, "--control"))
            F.workflow.write(path/"evidence.json", {"contract_sha256": queue["acceptance_contract_sha256"],
                "input_evidence": [{"path": str(item), "sha256": F.file_sha256(item)} for item in (selected, control)]})
            F.workflow.write(path/"comparison.json", {"source_count": 24, "acceptance_families_combined": False})
            F.workflow.write(path/"completion.json", {"operation": "compare", "complete": True,
                "contract_sha256": queue["acceptance_contract_sha256"],
                "artifact_sha256": F.file_sha256(path/"comparison.json")})
        else:
            assert argv[6] == "merge"
            inputs = argv[argv.index("--evidence")+1:argv.index("--contract")]
            assert len(inputs) == 12
            head_seeds = {(head, seed) for item in inputs for head, seeds in F.workflow.read(item)["records"].items() for seed in seeds}
            assert len(head_seeds) == 12
            F.workflow.write(path/"acceptance_report"/"scientific_acceptance_results.json",
                             {"status": "INCONCLUSIVE", "production_promotion": False,
                              "contract_sha256": queue["acceptance_contract_sha256"], "synthetic_controller_fixture": True})
            F.workflow.write(path/"evidence.json", {"contract_sha256": queue["acceptance_contract_sha256"],
                "input_evidence": [{"path": item, "sha256": F.file_sha256(item)} for item in inputs]})
            F.workflow.write(path/"completion.json", {"operation": "merge", "complete": True,
                "contract_sha256": queue["acceptance_contract_sha256"],
                "artifact_sha256": F.file_sha256(path/"acceptance_report"/"scientific_acceptance_results.json")})
        return 0
    return launch


def test_waits_for_existing_development_without_launching_it(registered, monkeypatch):
    r = registered
    execution_path = r.path.with_name(r.path.stem+"_execution.json")
    value = F.workflow.read(execution_path); value["status"] = "RUNNING"
    F.workflow.write(execution_path, value)
    monkeypatch.setattr(F.workflow, "run_queue", lambda *a, **k: pytest.fail("Duplicate development launch"))
    assert F.finish(r.args) == 75
    state = F.workflow.read(r.root/"workflow"/"finish_execution.json")
    assert not state["operations"] and state["production_promotion"] is False


def test_stop_and_each_volume_floor_block_new_operations(registered, monkeypatch):
    r = registered
    monkeypatch.setattr(F.workflow, "run_queue", lambda *a, **k: pytest.fail("Operation launched while paused"))
    monkeypatch.setattr(F, "free_disk_bytes", lambda paths: {"C:": 29*1024**3, "D:": 100*1024**3})
    assert F.finish(r.args) == 75
    monkeypatch.setattr(F, "free_disk_bytes", lambda paths: {"C:": 100*1024**3, "D:": 29*1024**3})
    assert F.finish(r.args) == 75
    monkeypatch.setattr(F, "free_disk_bytes", lambda paths: {"C:": 100*1024**3, "D:": 100*1024**3})
    (r.root/"workflow"/"STOP").write_text("user stop")
    assert F.finish(r.args) == 75


def test_os_lease_prevents_a_second_finish_controller(registered):
    with F.exclusive_lease(registered.root/"workflow"/"finish.lock"):
        with pytest.raises(F.FinishPause, match="active lease"):
            F.finish(registered.args)


def test_existing_full_process_is_never_duplicated(registered, monkeypatch):
    monkeypatch.setattr(F, "active_processes", lambda paths: [{"pid": 123, "created": 123.}])
    monkeypatch.setattr(F.workflow, "run_queue", lambda *a, **k: pytest.fail("Duplicate full training launch"))
    assert F.finish(registered.args) == 75


def test_full_pipeline_has_96_predictions_24_collectors_two_separate_families_and_resume(registered, monkeypatch):
    monkeypatch.setattr(F.workflow, "run_queue", fake_full_training)
    calls = []
    launch = fake_launch_factory(registered, calls)
    assert F.finish(registered.args, launch=launch) == 0
    assert len(calls) == 124
    assert sum(argv[5].endswith("cordex_scientific_prediction.py") for argv in calls) == 96
    assert sum(argv[6] == "collect" for argv in calls if argv[5] == F.EVIDENCE) == 24
    state = F.workflow.read(registered.root/"workflow"/"finish_execution.json")
    assert state["status"] == "SCIENTIFIC_EVALUATION_COMPLETED"
    assert {group: value["status"] for group, value in state["scientific_results"].items()} == {"selected": "INCONCLUSIVE", "control": "INCONCLUSIVE"}
    assert state["production_promotion"] is False
    assert "/matched_selected_control_maps/comparison.json" in state["matched_selected_control_maps"]["artifact"].replace(chr(92), "/")
    assert state["fresh_control_attribution"]["status"] == "COMPLETE"
    assert state["scientific_results"]["selected"]["status"] == "INCONCLUSIVE"
    assert len(state["checkpoint_config_manifest"]) == 24
    assert state["data_scope"]["historical_1981_2000"].startswith("UNEXECUTED")
    planned = F.workflow.read(registered.root/"workflow"/"finish_planned_commands.json")
    assert len(planned["predictions"]) == 96 and len(planned["collect"]) == 24 and len(planned["merge"]) == 2
    assert all(not item["executed"] for item in planned["predictions"] + planned["collect"] + planned["merge"])
    assert F.finish(registered.args, launch=launch) == 0
    assert len(calls) == 124, "Completed predictions/evidence must not be regenerated."
    # Simulate controller death after successful child completion but before
    # collection/merge receipt commits. The final child markers permit recovery.
    state = F.workflow.read(registered.root/"workflow"/"finish_execution.json")
    for key, record in state["operations"].items():
        if key.startswith(("collect|", "merge|", "compare|", "attribution|")):
            record["status"] = "AWAITING_ARTIFACT_VERIFICATION"
            record.pop("receipt")
    F.workflow.write(registered.root/"workflow"/"finish_execution.json", state)
    assert F.finish(registered.args, launch=launch) == 0
    assert len(calls) == 124
    state = F.workflow.read(registered.root/"workflow"/"finish_execution.json")
    first = next(record for key, record in state["operations"].items() if key.startswith("collect|"))
    statistics = next(Path(path) for path in first["receipt"]["bundle_files"] if path.endswith("statistics.npz"))
    statistics.write_bytes(b"changed statistics after completion")
    assert F.finish(registered.args, launch=launch) == 2
    assert F.workflow.read(registered.root/"workflow"/"finish_execution.json")["status"] == "STOPPED_EXECUTION_ERROR"


def test_exit_zero_without_prediction_does_not_count_as_complete(registered, monkeypatch):
    monkeypatch.setattr(F.workflow, "run_queue", fake_full_training)
    assert F.finish(registered.args, launch=lambda *args: 0) == 75
    state = F.workflow.read(registered.root/"workflow"/"finish_execution.json")
    assert state["status"] != "SCIENTIFIC_EVALUATION_COMPLETED"
    assert not any(record["status"] == "COMPLETED" for record in state["operations"].values())


def test_exact_partial_resume_and_changed_source_rejection(registered):
    full_path = F.prepare_full(registered.path); fake_full_training(full_path)
    full = F.read_queue(full_path)
    command = F.workflow.read(registered.root/"workflow"/"full_inference_commands.json")[0]
    job = next(job for job in full["jobs"] if job["id"] == command["training_job"])
    identity = F.prediction_identity(command, job, full)
    path = Path(F.option(command["argv"], "--output"))
    write_prediction(path, identity, full, partial=True)
    result = F.check_prediction(path, identity, full)
    assert result["native_resume"] and result["state"] == "PARTIAL_REQUIRES_NATIVE_IDENTITY_CHECK"
    altered = dict(identity, training_seed=202)
    with pytest.raises(ValueError, match="seed identity"):
        F.check_prediction(path, altered, full)
    path.with_suffix(".partial.h5").rename(path.with_suffix(".retained_fixture"))
    write_prediction(path, identity, full)
    assert F.check_prediction(path, identity, full)["state"] == "COMPLETED_AUTHENTICATED"
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        F.check_prediction(path, dict(identity, checkpoint_sha256="d"*64), full)


def test_partial_collection_attempts_are_preserved(registered):
    base = registered.root/"evidence"
    first = base/"attempt_0001"; first.mkdir(parents=True)
    (first/"partial.json").write_text("{}")
    record = {"attempts": [{"argv": ["--output", str(first)]}]}
    next_path = F.new_attempt_directory(base, record)
    assert next_path.name == "attempt_0002"
    assert (first/"partial.json").exists()


def test_persistent_main_retries_resource_pause_with_one_lease(registered, monkeypatch):
    calls, slept = [], []
    def inspect(args, **kwargs):
        calls.append(kwargs)
        return 75 if len(calls) == 1 else 0
    monkeypatch.setattr(F, "finish", inspect)
    monkeypatch.setattr(F.time, "sleep", slept.append)
    assert F.main(["--development-queue", str(registered.path), "--poll-seconds", "1"]) == 0
    assert calls == [{"acquire_lease": False}, {"acquire_lease": False}] and slept == [1.]

@pytest.mark.parametrize("message,code", [
    ("", 75), ("numpy.core._exceptions._ArrayMemoryError: Unable to allocate 5 GiB", 1),
    ("[Errno 28] No space left on device", 1), ("There is not enough space on the disk", 1)])
def test_exact_resource_failures_remain_resumable(tmp_path, message, code):
    log = tmp_path/"resource.log"; log.write_text(message)
    with pytest.raises(F.FinishPause) as error:
        F.command_failed(code, log)
    assert error.value.code == 75


def test_completed_scientific_input_drift_rejected_before_reuse(registered):
    state = {"operations": {"collect|fixture": {"status": "COMPLETED", "attempts": [],
                                              "input_fingerprint": "old"}}}
    with pytest.raises(ValueError, match="input identity changed"):
        F.run_scientific_operation("collect|fixture", "collect", registered.root/"unused", [],
            "new", "a"*64, state, registered.root/"unused.json", registered.root, registered.root,
            registered.args, lambda *unused: pytest.fail("Should not launch after input drift"))

def test_resume_command_preserves_supplemental_and_process_identity(registered, monkeypatch):
    args = registered.args
    args.supplemental_queue = str(registered.root/"supplement.json")
    args.development_pid = 123
    args.development_created = 1234.5
    path = registered.path.with_name(registered.path.stem+"_execution.json")
    state = F.workflow.read(path); state["status"] = "RUNNING"
    F.workflow.write(path, state)
    monkeypatch.setattr(F, "check_development_process", lambda *unused: None)
    assert F.finish(args) == 75
    command = F.workflow.read(registered.root/"workflow"/"finish_execution.json")["resume_command"]
    for expected in ("--supplemental-queue", str(args.supplemental_queue), "--development-pid", "123",
                     "--development-created", "1234.5"):
        assert expected in command


def test_finish_supplemental_order_waits_original_then_combines_before_selection(registered, monkeypatch):
    from examples.CORDEX_ML import cordex_scientific_supplemental as S
    supplemental = copy.deepcopy(registered.queue)
    supplemental["jobs"] = []
    supplemental["sha256"] = F.canonical_sha256({k: v for k, v in supplemental.items() if k != "sha256"})
    path = registered.root/"workflow"/"supplement_fixture.json"
    F.workflow.write(path, supplemental)
    registered.args.supplemental_queue = str(path)
    order = []
    def run_supplemental(given):
        assert Path(given) == path
        order.append("supplement")
        return fake_full_training(path)
    def combine(original, supplement):
        assert Path(original) == registered.path and Path(supplement) == path
        order.append("combine")
        return registered.path
    def prepare(original):
        order.append("prepare")
        raise F.FinishPause("Synthetic fixture stops before real full preparation.")
    monkeypatch.setattr(S, "run_supplemental", run_supplemental)
    monkeypatch.setattr(S, "combine_development_evidence", combine)
    monkeypatch.setattr(F, "prepare_full", prepare)
    assert F.finish(registered.args) == 75
    assert order == ["supplement", "combine", "prepare"]


def test_attribution_registration_drift_blocks_full_training(registered, monkeypatch):
    full_path = F.prepare_full(registered.path)
    full = F.read_queue(full_path)
    first = F.attribution_registration(registered.root, full)
    F.workflow.write(registered.root/"workflow/finish_execution.json", {
        "version": F.VERSION, "development_queue_sha256": registered.queue["sha256"],
        "operations": {}, "attribution_registration": first})
    path = registered.root/"fresh_control_contrast_registration.json"
    value = F.workflow.read(path); value["unregistered_change"] = True
    value["sha256"] = F.canonical_sha256({key: item for key, item in value.items() if key != "sha256"})
    F.workflow.write(path, value)
    monkeypatch.setattr(F.workflow, "run_queue", lambda *a, **k: pytest.fail("Full training started after registration drift"))
    assert F.finish(registered.args) == 2
    state = F.workflow.read(registered.root/"workflow/finish_execution.json")
    assert "registration changed" in state["reason"]
