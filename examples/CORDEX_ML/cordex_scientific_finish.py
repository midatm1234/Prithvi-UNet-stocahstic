#!/usr/bin/env python
"""Resume the registered workflow after an already-running development queue.

No development jobs are launched here. Training, prediction and collection keep
their existing interfaces. STOP prevents new operations and allows an active
atomic operation to finish; no training process is forcibly terminated.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.CORDEX_ML import cordex_scientific_workflow as workflow
from examples.CORDEX_ML.cordex_scientific_full_plan import prepare_full
from examples.CORDEX_ML.cordex_scientific_prediction import resolve_selected_checkpoint
from granitewxc.refinement.scientific_data import canonical_sha256, file_sha256
from granitewxc.refinement.randomness import STREAM_VERSION
VERSION = "registered_scientific_finish_controller_v1"
TERMINAL_DEVELOPMENT = {"COMPLETED_REGISTERED_STOP", "CONDITION_NOT_TRIGGERED", "FAIL_NUMERICAL_EXECUTION"}
EVIDENCE = "examples/CORDEX_ML/cordex_scientific_evidence.py"
ATTRIBUTION = "examples/CORDEX_ML/cordex_scientific_attribution.py"


class FinishPause(RuntimeError):
    def __init__(self, reason, *, code=75):
        super().__init__(reason)
        self.code = code


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def exclusive_lease(path):
    """OS-released lock survives crashes without stale PID guessing."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        if path.stat().st_size == 0:
            stream.write(b"0"); stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise FinishPause("Another finish controller holds the active lease.") from error
        yield
    finally:
        stream.close()


def read_queue(path):
    queue = workflow.read(path)
    if canonical_sha256({key: value for key, value in queue.items() if key != "sha256"}) != queue.get("sha256"):
        raise ValueError("Registered queue bytes/identity changed.")
    return queue


def queue_execution(path, queue):
    state_path = Path(path).with_name(Path(path).stem + "_execution.json")
    if not state_path.exists():
        return {"status": "NOT_EXECUTED", "jobs": {}}
    state = workflow.read(state_path)
    if state.get("queue_sha256") != queue["sha256"]:
        raise ValueError("Execution receipt identifies a different queue.")
    return state


def development_complete(path, queue):
    state = queue_execution(path, queue)
    if state.get("status") == "REGISTERED_QUEUE_COMPLETED":
        missing = [job["id"] for job in queue["jobs"]
                   if state["jobs"].get(job["id"], {}).get("status") not in TERMINAL_DEVELOPMENT]
        if missing:
            raise ValueError("Completed development receipt has unfinished jobs: " + ", ".join(missing))
        return True, state
    if state.get("status") in ("INCONCLUSIVE_RESOURCE_PAUSE", "STOPPED_EXECUTION_ERROR",
                               "DEVELOPMENT_FINISHED_WITH_PENDING_DEPENDENCIES",
                               "REQUESTED_STAGE_COMPLETE_OTHER_JOBS_UNEXECUTED"):
        raise FinishPause("Development must be resumed by its existing controller: " + state["status"])
    return False, state


def check_development_process(args, queue_path):
    """Optional PID/create-time attestation; never terminate or restart it."""
    pid = getattr(args, "development_pid", None)
    if pid is None:
        return
    try:
        process = psutil.Process(pid)
        expected = getattr(args, "development_created", None)
        if expected is not None and abs(process.create_time()-expected) > .001:
            raise FinishPause("Registered development PID was reused; original process identity is gone.")
        if not process.is_running():
            raise psutil.NoSuchProcess(pid)
    except psutil.NoSuchProcess as error:
        if active_processes([queue_path]):
            return  # An explicitly restarted controller is already running.
        raise FinishPause("Registered development process ended before its queue completed; no automatic development relaunch.") from error
    except psutil.AccessDenied as error:
        raise FinishPause("Development process identity could not be verified read-only.") from error


def free_disk_bytes(paths):
    result = {}
    for value in paths:
        path = Path(value).resolve()
        while not path.exists():
            path = path.parent
        volume = Path(path.anchor) if path.anchor else path
        result[str(volume)] = shutil.disk_usage(path).free
    return result


def active_processes(paths):
    """Read-only scoped check for already-running jobs/predictions/collectors."""
    wanted = {str(Path(path).resolve()).casefold() for path in paths}
    found = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.info["pid"] == os.getpid():
            continue
        args = process.info["cmdline"] or []
        try:
            matched = any(
                args[index] in ("--queue", "--supplemental-queue", "--output", "--training-output")
                and str(Path(args[index+1]).resolve()).casefold() in wanted
                for index in range(len(args)-1))
            if matched:
                found.append({"pid": process.info["pid"], "created": process.create_time()})
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    return found


def ensure_ready(root, artifact_root, state, state_path, minimum_free_gib):
    if (root/"workflow"/"STOP").exists():
        raise FinishPause("STOP exists; no new operation was launched.")
    available = free_disk_bytes((root, artifact_root))
    state["free_bytes"] = available
    workflow.write(state_path, state)
    if min(available.values()) < minimum_free_gib * 1024**3:
        raise FinishPause("Disk reserve reached on metadata or prediction volume.")


def wait_idle(paths, state, state_path, root, artifact_root, args):
    while True:
        ensure_ready(root, artifact_root, state, state_path, args.minimum_free_gib)
        processes = active_processes(paths)
        if not processes:
            return
        state.update(status="WAITING_FOR_EXISTING_OPERATION", existing_processes=processes, updated_utc=now())
        workflow.write(state_path, state)
        if args.once:
            raise FinishPause("Existing operation is running; duplicate launch suppressed.")
        time.sleep(args.poll_seconds)


def dispatch(argv, log, state, state_path, root, poll_seconds):
    """Prithvi's current Python executes the exact normal script argument list."""
    if argv[:5] != ["mamba", "run", "-n", "Prithvi", "python"]:
        raise ValueError("Only registered Prithvi Python entry points may be dispatched.")
    actual = [sys.executable, *argv[5:]]
    log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf8") as output:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(actual, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                   creationflags=flags)
        state.update(child_pid=process.pid, actual_execution_argv=actual)
        workflow.write(state_path, state)
        while True:
            try:
                return process.wait(timeout=poll_seconds)
            except subprocess.TimeoutExpired:
                state.update(updated_utc=now(), stop_requested=(root/"workflow"/"STOP").exists())
                workflow.write(state_path, state)


def option(argv, flag):
    return argv[argv.index(flag)+1]


def prediction_identity(entry, job, queue):
    checkpoint, selected_kind = resolve_selected_checkpoint(job["output"])
    cache_manifest = workflow.read(Path(queue["run_root"])/"cache"/"phase1.manifest.json")
    plan = workflow.read(Path(queue["run_root"])/"registered_experiment_plan.json")
    return {"sampling_seed": plan["validation"]["sampling_seed"], "random_stream_version": STREAM_VERSION,
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
            "selection": selected_kind, "training_job": job["id"], "training_seed": job["seed"],
            "source_cache_sha256": cache_manifest["cache_sha256"],
            "phase1_checkpoint_sha256": cache_manifest["phase1_checkpoint_sha256"],
            "acceptance_contract_sha256": queue["acceptance_contract_sha256"],
            "plan_sha256": queue["registered_plan_sha256"], "members": entry["members"],
            "diagnostic": entry["diagnostic"], "variables": cache_manifest["variables"]}


def check_prediction(path, identity, queue):
    """Validate completed files; partial identity is finally checked natively."""
    path = Path(path)
    partial = path.with_suffix(".partial.h5")
    if path.exists() and partial.exists():
        raise ValueError("Both completed and partial prediction files exist; preserve and investigate.")
    if not path.exists() and not partial.exists():
        return {"state": "NEW", "native_resume": False}
    candidate = path if path.exists() else partial
    cache = workflow.read(Path(queue["run_root"])/"cache"/"phase1.manifest.json")
    dates = [cache["splits"]["dates"][index] for index in cache["splits"]["validation_indices"]]
    if identity["diagnostic"]:
        from examples.CORDEX_ML.cordex_scientific_experiments import diagnostic_dates
        dates = diagnostic_dates(dates)
    with h5py.File(candidate, "r") as f:
        if f["dates"].asstr()[:].tolist() != dates:
            raise ValueError("Existing prediction dates differ from the registered inference request.")
        for name, expected in (("ensemble_size", identity["members"]), ("training_seed", identity["training_seed"]),
                               ("sampling_seed", identity["sampling_seed"]),
                               ("random_stream_version", identity["random_stream_version"])):
            if f.attrs.get(name) != expected:
                raise ValueError("Existing prediction member/seed identity differs.")
        if json.loads(f.attrs["variables"]) != identity["variables"]:
            raise ValueError("Existing prediction target ordering differs.")
        scope = "nested_member_diagnostic" if identity["diagnostic"] else "component_validation"
        if f.attrs.get("data_scope") != scope:
            raise ValueError("Existing prediction scope differs.")
        opaque = str(f.attrs.get("prediction_identity", ""))
        completed = int(f.attrs.get("completed_dates", 0))
        if not opaque or completed < 0 or completed > len(dates):
            raise ValueError("Unversioned or invalid partial prediction coverage.")
        if path.exists() and (not bool(f.attrs.get("complete")) or completed != len(dates)):
            raise ValueError("Completed prediction filename has incomplete contents.")
    if candidate == partial:
        return {"state": "PARTIAL_REQUIRES_NATIVE_IDENTITY_CHECK", "native_resume": True,
                "completed_dates": completed, "prediction_identity": opaque,
                "note": "Normal runner must exactly verify its model/target/conditioning/date/random-stream identity before resuming."}
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise ValueError("Final prediction exists without its completed manifest; do not overwrite.")
    manifest = workflow.read(manifest_path)
    if not manifest.get("complete") or manifest.get("prediction_identity") != opaque:
        raise ValueError("Completed prediction manifest identity differs.")
    if file_sha256(path) != manifest.get("sha256"):
        raise ValueError("Completed prediction bytes changed.")
    authentication = manifest.get("authentication") or {}
    for key in ("checkpoint_sha256", "source_cache_sha256", "phase1_checkpoint_sha256",
                "acceptance_contract_sha256", "plan_sha256"):
        if authentication.get(key) != identity[key]:
            raise ValueError("Completed prediction provenance differs: " + key)
    return {"state": "COMPLETED_AUTHENTICATED", "native_resume": False,
            "artifact": str(path.resolve()), "artifact_sha256": manifest["sha256"],
            "manifest_sha256": file_sha256(manifest_path), "completed_dates": completed}


def command_failed(code, log):
    if code == 0:
        return
    tail = Path(log).read_text(encoding="utf8", errors="replace")[-12000:] if Path(log).exists() else ""
    resource = any(word in tail.lower() for word in (
        "out of memory", "insufficient disk", "disk reserve", "keyboardinterrupt", "interruptederror",
        "memoryerror", "unable to allocate", "no space left on device", "disk full", "not enough space on the disk"))
    raise FinishPause(f"Operation exited with code {code}; see {log}.", code=75 if code == 75 or resource else 2)


def execute_attempt(key, argv, fingerprint, state, state_path, root, args, *, launch=dispatch):
    record = state["operations"].setdefault(key, {"status": "NOT_EXECUTED", "attempts": []})
    if record.get("input_fingerprint") not in (None, fingerprint):
        raise ValueError("Operation inputs changed after an earlier attempt: " + key)
    record["input_fingerprint"] = fingerprint
    attempt = {"number": len(record["attempts"])+1, "started_utc": now(), "argv": argv,
               "command": workflow.shell_command(argv), "status": "RUNNING"}
    attempt["log"] = str(root/"workflow"/"finish_logs"/(key.replace("/", "__").replace("|", "__")
                                                      + f"__attempt_{attempt['number']:04d}.log"))
    record["attempts"].append(attempt)
    record["status"] = "RUNNING"; state.update(status="RUNNING", active_operation=key, updated_utc=now())
    workflow.write(state_path, state)
    code = launch(argv, attempt["log"], state, state_path, root, args.poll_seconds)
    attempt.update(returncode=code, finished_utc=now(), status="RETURNED")
    record["status"] = "AWAITING_ARTIFACT_VERIFICATION"
    workflow.write(state_path, state)
    command_failed(code, attempt["log"])
    return record


def new_attempt_directory(base, record):
    if record.get("use_exact_first_output"):
        base = Path(base)
        if not base.exists() and not record.get("attempts"):
            return base
        index = 2
        while True:
            path = base.with_name(base.name + f"_attempt_{index:04d}")
            if not path.exists() and all(str(path) != option(item["argv"], "--output") for item in record.get("attempts", [])):
                return path
            index += 1
    previous = {str(Path(option(item["argv"], "--output")).resolve()) for item in record.get("attempts", [])}
    index = 1
    while True:
        path = Path(base)/f"attempt_{index:04d}"
        if not path.exists() and str(path.resolve()) not in previous:
            return path
        index += 1


def checked_file(path, expected):
    path = Path(path)
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError("Referenced scientific artifact missing or changed: " + str(path))
    return str(path.resolve()), expected


def referenced_files(value):
    """Read explicit path/hash pairs without guessing filenames or array fields."""
    result = {}
    if isinstance(value, dict):
        for path_key, hash_key in (("path", "sha256"), ("artifact", "artifact_sha256")):
            if path_key in value and hash_key in value:
                path, digest = checked_file(value[path_key], value[hash_key])
                result[path] = digest
        for child in value.values():
            result.update(referenced_files(child))
    elif isinstance(value, list):
        for child in value:
            result.update(referenced_files(child))
    return result


def scientific_bundle_receipt(artifact, operation, contract_sha256):
    """Require the child's final marker and authenticate its complete bundle."""
    artifact = Path(artifact)
    output = artifact.parent.parent if operation == "merge" else artifact.parent
    marker_path = output/"completion.json"
    if not marker_path.is_file():
        return None
    marker = workflow.read(marker_path)
    if (marker.get("complete") is not True or marker.get("operation") != operation
            or marker.get("contract_sha256") != contract_sha256):
        raise ValueError("Scientific child completion marker identifies another operation.")
    checked_file(artifact, marker["artifact_sha256"])
    evidence_path = output/"evidence.json"
    evidence = workflow.read(evidence_path)
    if evidence.get("contract_sha256") != contract_sha256:
        raise ValueError("Scientific evidence contract changed.")
    files = referenced_files(evidence)
    for variable, products in evidence.get("diagnostic_outputs", {}).items():
        for name, path in products.items():
            location, digest = checked_file(path, evidence["diagnostic_output_sha256"][variable][name])
            files[location] = digest
    if operation == "collect":
        authentication = workflow.read(evidence["authentication"]["artifact"])
        if authentication.get("authenticated") is not True:
            raise ValueError("Collector did not authenticate its source arrays.")
        for path_key, hash_key in (("prediction_path", "prediction_sha256"),
                                   ("cache_path", "cache_sha256"),
                                   ("checkpoint_path", "checkpoint_sha256")):
            location, digest = checked_file(authentication[path_key], authentication[hash_key])
            files[location] = digest
        files.update(referenced_files(authentication.get("completed_run_authentication", {})))
        mean = authentication.get("mean_control_authentication")
        if mean:
            location, digest = checked_file(mean["prepared_cache_path"], mean["prepared_cache_sha256"])
            files[location] = digest
    for path in output.rglob("*"):
        if path.is_file():
            files[str(path.resolve())] = file_sha256(path)
    return {"artifact": str(artifact.resolve()), "artifact_sha256": file_sha256(artifact),
            "bundle_files": files, "operation": operation, "completion_marker": str(marker_path.resolve())}


def recover_attempt(record, operation, contract_sha256):
    """Reuse a completed orphan child; preserve all genuinely partial attempts."""
    for attempt in reversed(record.get("attempts", [])):
        output = Path(option(attempt["argv"], "--output"))
        artifact = scientific_artifact_path(output, operation)
        receipt = scientific_bundle_receipt(artifact, operation, contract_sha256)
        if receipt is not None:
            record.update(status="COMPLETED", receipt=receipt, recovered_completed_attempt=True)
            return True
    return False


def completed_receipt(record):
    if record.get("status") != "COMPLETED":
        return False
    receipt = record["receipt"]
    if not Path(receipt["artifact"]).is_file() or file_sha256(receipt["artifact"]) != receipt["artifact_sha256"]:
        raise ValueError("Completed controller artifact changed.")
    for path, digest in receipt.get("bundle_files", {}).items():
        checked_file(path, digest)
    if "operation" in receipt and not receipt.get("bundle_files"):
        raise ValueError("Completed scientific operation lacks its authenticated bundle.")
    return True


def scientific_artifact_path(output, operation):
    names = {"collect": "evidence.json", "merge": "acceptance_report/scientific_acceptance_results.json",
             "compare": "comparison.json", "attribution": "comparison.json"}
    return Path(output)/names[operation]


def run_scientific_operation(key, operation, base, argv, fingerprint, contract_sha256,
                             state, state_path, root, artifact_root, args, launch):
    record = state["operations"].setdefault(key, {"status": "NOT_EXECUTED", "attempts": []})
    if record.get("input_fingerprint") not in (None, fingerprint):
        raise ValueError("Scientific operation input identity changed: " + key)
    record["input_fingerprint"] = fingerprint
    if operation == "compare":
        record["use_exact_first_output"] = True
    if completed_receipt(record):
        return record
    wait_idle([option(item["argv"], "--output") for item in record["attempts"]],
              state, state_path, root, artifact_root, args)
    if recover_attempt(record, operation, contract_sha256):
        workflow.write(state_path, state)
        return record
    output = new_attempt_directory(base, record)
    argv = list(argv)
    argv[argv.index("--output")+1] = str(output)
    record = execute_attempt(key, argv, fingerprint, state, state_path, root, args, launch=launch)
    artifact = scientific_artifact_path(output, operation)
    receipt = scientific_bundle_receipt(artifact, operation, contract_sha256)
    if receipt is None:
        raise FinishPause("Child returned without a complete authenticated scientific bundle.", code=2)
    record.update(status="COMPLETED", receipt=receipt, finished_utc=now())
    workflow.write(state_path, state)
    return record


def attribution_registration(root, full):
    path = Path(root)/"fresh_control_contrast_registration.json"
    value = workflow.read(path)
    expected = canonical_sha256({key: item for key, item in value.items() if key != "sha256"})
    if (value.get("version") != "paired_selected_fresh_stochastic_control_v1"
            or value.get("sha256") != expected
            or value.get("acceptance_contract_sha256") != full["acceptance_contract_sha256"]
            or value.get("parent_experiment_plan_sha256") != full["registered_plan_sha256"]):
        raise ValueError("Fresh-control attribution registration identifies another protocol.")
    return {"path": str(path.resolve()), "sha256": file_sha256(path),
            "registration_sha256": expected}


def build_finish_plan(full, commands):
    """Freeze reviewable first-attempt commands; retries get separate receipts."""
    root = Path(full["run_root"])
    artifact_root = Path(full.get("artifact_root", root))
    jobs = {job["id"]: job for job in full["jobs"] if job["head"] in workflow.HEADS}
    assigned = {key: [] for key in jobs}
    outputs = set()
    for entry in commands:
        key = entry["training_job"]
        if key not in assigned:
            raise ValueError("Inference command references an unknown full job.")
        output = str(Path(option(entry["argv"], "--output")).resolve())
        if output in outputs:
            raise ValueError("Repeated inference output would overwrite an ensemble.")
        outputs.add(output); assigned[key].append(entry)
    result = {"version": VERSION, "full_queue_sha256": full["sha256"], "predictions": commands,
              "collect": [], "merge": [], "production_promotion": False}
    groups = {"selected": [], "control": []}
    for key, job in jobs.items():
        entries = assigned[key]
        primary = [entry for entry in entries if not entry["diagnostic"]]
        nested = sorted((entry for entry in entries if entry["diagnostic"]), key=lambda item: item["members"])
        if len(primary) != 1 or primary[0]["members"] != 20 or [entry["members"] for entry in nested] != [10, 20, 50]:
            raise ValueError("Every full job requires primary20 plus nested10/20/50.")
        base = artifact_root/"workflow"/"evidence"/key
        output = base/"attempt_0001"
        argv = ["mamba", "run", "-n", "Prithvi", "python", EVIDENCE, "collect",
                "--prediction", option(primary[0]["argv"], "--output"),
                "--cache", str(root/"cache"/"phase1.h5"), "--contract", str(root/"frozen_scientific_acceptance_v2.json"),
                "--output", str(output), "--nested", *[option(entry["argv"], "--output") for entry in nested]]
        result["collect"].append({"training_job": key, "base_output": str(base), "argv": argv,
                                  "command": workflow.shell_command(argv), "executed": False})
        groups[job["comparison_group"]].append(str(output/"evidence.json"))
    for group, evidence in groups.items():
        if len(evidence) != 12:
            raise ValueError("Separate selected/control families require twelve evidence files each.")
        base = artifact_root/"workflow"/"scientific_reports"/group
        argv = ["mamba", "run", "-n", "Prithvi", "python", EVIDENCE, "merge", "--evidence", *evidence,
                "--contract", str(root/"frozen_scientific_acceptance_v2.json"),
                "--output", str(base/"attempt_0001")]
        result["merge"].append({"comparison_group": group, "base_output": str(base), "argv": argv,
                                "command": workflow.shell_command(argv), "executed": False})
    comparison_base = artifact_root/"workflow"/"component_acceptance"/"matched_selected_control_maps"
    compare_argv = ["mamba", "run", "-n", "Prithvi", "python", EVIDENCE, "compare-families",
        "--selected", str(artifact_root/"workflow"/"scientific_reports"/"selected"/"attempt_0001"/"evidence.json"),
        "--control", str(artifact_root/"workflow"/"scientific_reports"/"control"/"attempt_0001"/"evidence.json"),
        "--contract", str(root/"frozen_scientific_acceptance_v2.json"), "--output", str(comparison_base)]
    result["matched_maps"] = {"base_output": str(comparison_base), "argv": compare_argv,
                              "command": workflow.shell_command(compare_argv), "executed": False}
    registration = attribution_registration(root, full)
    attribution_base = artifact_root/"workflow"/"component_acceptance"/"selected_fresh_control_attribution"
    attribution_argv = ["mamba", "run", "-n", "Prithvi", "python", ATTRIBUTION, "compare",
        "--selected", option(compare_argv, "--selected"), "--control", option(compare_argv, "--control"),
        "--contract", str(root/"frozen_scientific_acceptance_v2.json"),
        "--registration", registration["path"], "--output", str(attribution_base/"attempt_0001")]
    result["attribution"] = {"base_output": str(attribution_base), "argv": attribution_argv,
        "registration": registration, "command": workflow.shell_command(attribution_argv), "executed": False}
    return result


def finish(args, *, launch=dispatch, acquire_lease=True):
    development_path = Path(args.development_queue).resolve()
    development = read_queue(development_path)
    root = Path(development["run_root"]).resolve()
    artifact_root = Path(development.get("artifact_root", root)).resolve()
    state_path = root/"workflow"/"finish_execution.json"
    with exclusive_lease(root/"workflow"/"finish.lock") if acquire_lease else nullcontext():
        state = workflow.read(state_path) if state_path.exists() else {
            "version": VERSION, "development_queue_sha256": development["sha256"],
            "operations": {}, "production_promotion": False, "started_utc": now()}
        if state.get("version") != VERSION or state.get("development_queue_sha256") != development["sha256"]:
            raise ValueError("Finish receipt belongs to another registered queue.")
        try:
            while True:
                ensure_ready(root, artifact_root, state, state_path, args.minimum_free_gib)
                ready, development_state = development_complete(development_path, development)
                if ready:
                    break
                check_development_process(args, development_path)
                state.update(status="WAITING_FOR_EXISTING_DEVELOPMENT", development_status=development_state.get("status"),
                             updated_utc=now())
                workflow.write(state_path, state)
                if args.once:
                    raise FinishPause("Existing development queue has not completed; no jobs duplicated.")
                time.sleep(args.poll_seconds)
            selection_queue = development_path
            supplemental_path = getattr(args, "supplemental_queue", None)
            if supplemental_path:
                from examples.CORDEX_ML.cordex_scientific_supplemental import run_supplemental, combine_development_evidence
                supplemental_path = Path(supplemental_path).resolve()
                supplemental = read_queue(supplemental_path)
                if state.get("supplemental_queue_sha256") not in (None, supplemental["sha256"]):
                    raise ValueError("Supplemental controls changed after controller registration.")
                state.update(supplemental_queue=str(supplemental_path), supplemental_queue_sha256=supplemental["sha256"])
                wait_idle([supplemental_path, *[job["output"] for job in supplemental["jobs"]]],
                          state, state_path, root, artifact_root, args)
                supplement_state = queue_execution(supplemental_path, supplemental)
                if supplement_state.get("status") != "REGISTERED_QUEUE_COMPLETED":
                    state.update(status="RUNNING_REGISTERED_SUPPLEMENTAL_CONTROLS", updated_utc=now())
                    workflow.write(state_path, state)
                    code = run_supplemental(supplemental_path)
                    supplement_state = queue_execution(supplemental_path, supplemental)
                    if code != 0 or supplement_state.get("status") != "REGISTERED_QUEUE_COMPLETED":
                        raise FinishPause("Supplemental matched controls are unfinished.", code=code or 75)
                selection_queue = combine_development_evidence(development_path, supplemental_path)
                state["combined_selection_queue"] = str(selection_queue)
            elif state.get("supplemental_queue_sha256"):
                raise ValueError("A previously registered supplemental queue cannot be omitted on resume.")
            full_path = prepare_full(selection_queue)
            full = read_queue(full_path)
            registration = attribution_registration(root, full)
            if state.get("attribution_registration") not in (None, registration):
                raise ValueError("Fresh-control attribution registration changed after execution began.")
            state["attribution_registration"] = registration
            state.update(full_queue=str(full_path), full_queue_sha256=full["sha256"],
                data_scope={"component_validation": "1977-1980 and 2096-2099",
                            "historical_1981_2000": "UNEXECUTED: regeneration is not scheduled or supported by this experimental cache/CLI",
                            "midcentury_2041_2060": "QUARANTINED: target values are never opened"})
            wait_idle([full_path, *[job["output"] for job in full["jobs"]]],
                      state, state_path, root, artifact_root, args)
            execution = queue_execution(full_path, full)
            code = 0
            if execution.get("status") != "REGISTERED_QUEUE_COMPLETED":
                state.update(status="RUNNING_FULL_TRAINING", full_training_dispatch={
                    "api": "cordex_scientific_workflow.run_queue", "queue": str(full_path),
                    "only_stage": "full", "started_utc": now()})
                workflow.write(state_path, state)
                code = workflow.run_queue(full_path, only_stage="full")
                execution = queue_execution(full_path, full)
            if code != 0 or execution.get("status") != "REGISTERED_QUEUE_COMPLETED":
                raise FinishPause("Full training is unfinished: " + str(execution.get("status")), code=code or 75)
            jobs = {job["id"]: job for job in full["jobs"] if job["head"] in workflow.HEADS}
            if len(jobs) != 24:
                raise ValueError("Expected separate selected/control four-head three-seed families.")
            if any(execution["jobs"].get(job_id, {}).get("status") != "COMPLETED_REGISTERED_STOP" for job_id in jobs):
                raise FinishPause("At least one required full stochastic job did not complete its registered stop.", code=2)
            commands = workflow.read(root/"workflow"/"full_inference_commands.json")
            if len(commands) != 96:
                raise ValueError("Registered inference plan must contain all 96 operations.")
            plan = build_finish_plan(full, commands)
            planned_path = root/"workflow"/"finish_planned_commands.json"
            if planned_path.exists() and workflow.read(planned_path) != plan:
                raise ValueError("Previously planned finish operations changed.")
            workflow.write(planned_path, plan)
            by_job = {key: [] for key in jobs}
            for entry in commands:
                key = entry["training_job"]
                if key not in jobs:
                    raise ValueError("Inference references an unregistered full job.")
                by_job[key].append(entry)
                path = Path(option(entry["argv"], "--output"))
                ensure_ready(root, artifact_root, state, state_path, args.minimum_free_gib)
                wait_idle([path], state, state_path, root, artifact_root, args)
                identity = prediction_identity(entry, jobs[key], full)
                config_path = ROOT/Path(jobs[key]["config"])
                state.setdefault("checkpoint_config_manifest", {})[key] = {
                    **identity, "configuration": str(config_path.resolve()), "configuration_sha256": file_sha256(config_path)}
                operation = "predict|" + key + ("|nested_" if entry["diagnostic"] else "|full_") + str(entry["members"])
                fingerprint = canonical_sha256(identity)
                old = state["operations"].get(operation, {})
                if old.get("input_fingerprint") not in (None, fingerprint):
                    raise ValueError("Selected prediction inputs changed after an earlier controller attempt.")
                receipt = check_prediction(path, identity, full)
                if receipt["state"] != "COMPLETED_AUTHENTICATED":
                    argv = list(entry["argv"])
                    if receipt["native_resume"]:
                        argv.append("--resume")
                    state["operations"].setdefault(operation, {"status": "NOT_EXECUTED", "attempts": []})["resume_preflight"] = receipt
                    workflow.write(state_path, state)
                    record = execute_attempt(operation, argv, fingerprint, state, state_path, root, args, launch=launch)
                    receipt = check_prediction(path, identity, full)
                    if receipt["state"] != "COMPLETED_AUTHENTICATED":
                        raise FinishPause("Inference returned without a complete authenticated artifact.")
                record = state["operations"].setdefault(operation, {"attempts": [], "input_fingerprint": fingerprint})
                record.update(status="COMPLETED", receipt=receipt, finished_utc=now())
                workflow.write(state_path, state)
            groups = {"selected": [], "control": []}
            for item in plan["collect"]:
                job_id = item["training_job"]; job = jobs[job_id]
                ensure_ready(root, artifact_root, state, state_path, args.minimum_free_gib)
                fingerprint = canonical_sha256({"job": job_id, "full_queue": full["sha256"],
                    "predictions": [file_sha256(option(entry["argv"], "--output")) for entry in by_job[job_id]]})
                record = run_scientific_operation(
                    "collect|"+job_id, "collect", item["base_output"], item["argv"], fingerprint,
                    full["acceptance_contract_sha256"], state, state_path, root, artifact_root, args, launch)
                evidence_path = record["receipt"]["artifact"]
                value = workflow.read(evidence_path)
                if (set(value.get("records", {})) != {job["head"]}
                        or set(value["records"][job["head"]]) != {str(job["seed"])}):
                    raise ValueError("Collected evidence identifies another head or training seed.")
                groups[job["comparison_group"]].append(evidence_path)
            results = {}
            for item in plan["merge"]:
                group = item["comparison_group"]; paths = groups[group]
                ensure_ready(root, artifact_root, state, state_path, args.minimum_free_gib)
                argv = list(item["argv"])
                argv[argv.index("--evidence")+1:argv.index("--contract")] = paths
                fingerprint = canonical_sha256({"family": group, "evidence": [file_sha256(path) for path in paths]})
                record = run_scientific_operation(
                    "merge|"+group, "merge", item["base_output"], argv, fingerprint,
                    full["acceptance_contract_sha256"], state, state_path, root, artifact_root, args, launch)
                report = workflow.read(record["receipt"]["artifact"])
                if (report.get("production_promotion") is not False
                        or report.get("contract_sha256") != full["acceptance_contract_sha256"]
                        or report.get("status") not in ("PASS", "FAIL", "INCONCLUSIVE")):
                    raise ValueError("Scientific report contract/status is invalid; promotion is forbidden.")
                results[group] = {"status": report["status"], **record["receipt"]}
            ensure_ready(root, artifact_root, state, state_path, args.minimum_free_gib)
            comparison = plan["matched_maps"]; compare_argv = list(comparison["argv"])
            family_evidence = {group: str(Path(value["artifact"]).parent.parent/"evidence.json")
                               for group, value in results.items()}
            for group, path in family_evidence.items():
                compare_argv[compare_argv.index("--"+group)+1] = path
            comparison_record = run_scientific_operation(
                "compare|selected_control", "compare", comparison["base_output"], compare_argv,
                canonical_sha256({group: file_sha256(path) for group, path in family_evidence.items()}),
                full["acceptance_contract_sha256"], state, state_path, root, artifact_root, args, launch)
            state["matched_selected_control_maps"] = comparison_record["receipt"]
            state["scientific_results"] = results
            workflow.write(state_path, state)
            ensure_ready(root, artifact_root, state, state_path, args.minimum_free_gib)
            attribution = plan["attribution"]; attribution_argv = list(attribution["argv"])
            for group, path in family_evidence.items():
                attribution_argv[attribution_argv.index("--"+group)+1] = path
            attribution_record = run_scientific_operation(
                "attribution|selected_control", "attribution", attribution["base_output"], attribution_argv,
                canonical_sha256({"families": {group: file_sha256(path) for group, path in family_evidence.items()},
                                  "registration": registration}),
                full["acceptance_contract_sha256"], state, state_path, root, artifact_root, args, launch)
            attribution_result = workflow.read(attribution_record["receipt"]["artifact"])
            if (attribution_result.get("status") not in ("COMPLETE", "INCONCLUSIVE")
                    or attribution_result.get("acceptance_decisions_unchanged") is not True
                    or attribution_result.get("acceptance_families_combined") is not False
                    or attribution_result.get("production_promotion") is not False
                    or attribution_result.get("contract_sha256") != full["acceptance_contract_sha256"]
                    or attribution_result.get("registration_sha256") != registration["registration_sha256"]):
                raise ValueError("Supplemental attribution changed acceptance or its registered identity.")
            state["fresh_control_attribution"] = {
                "status": attribution_result["status"], **attribution_record["receipt"]}
            state.update(status="SCIENTIFIC_EVALUATION_COMPLETED", scientific_results=results,
                         finished_utc=now(), production_promotion=False)
            workflow.write(state_path, state)
            return 0
        except (FinishPause, KeyboardInterrupt, ValueError, OSError, KeyError) as error:
            code = error.code if isinstance(error, FinishPause) else 75 if isinstance(error, KeyboardInterrupt) else 2
            resume_argv = ["mamba", "run", "-n", "Prithvi", "python",
                "examples/CORDEX_ML/cordex_scientific_finish.py", "--development-queue", str(development_path),
                "--poll-seconds", str(args.poll_seconds), "--minimum-free-gib", str(args.minimum_free_gib)]
            for flag, name in (("--supplemental-queue", "supplemental_queue"),
                               ("--development-pid", "development_pid"),
                               ("--development-created", "development_created")):
                if getattr(args, name, None) is not None:
                    resume_argv.extend([flag, str(getattr(args, name))])
            state.update(status="INCONCLUSIVE_RESOURCE_OR_STOP_PAUSE" if code == 75 else "STOPPED_EXECUTION_ERROR",
                         reason=str(error), updated_utc=now(), production_promotion=False,
                         resume_command=workflow.shell_command(resume_argv))
            workflow.write(state_path, state)
            return code


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-queue", required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.)
    parser.add_argument("--minimum-free-gib", type=float, default=30.)
    parser.add_argument("--supplemental-queue")
    parser.add_argument("--development-pid", type=int)
    parser.add_argument("--development-created", type=float)
    parser.add_argument("--once", action="store_true", help="Inspect once and pause if a prerequisite is still running.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 1 <= args.poll_seconds <= 60 or args.minimum_free_gib < 30:
        raise ValueError("Polling must be 1 to 60 seconds and both-volume reserve at least 30 GiB.")
    if Path(sys.prefix).name.casefold() != "prithvi":
        raise ValueError("Run this controller with mamba run -n Prithvi.")
    queue = read_queue(args.development_queue)
    root = Path(queue["run_root"])
    try:
        with exclusive_lease(root/"workflow"/"finish.lock"):
            while True:
                code = finish(args, acquire_lease=False)
                if code != 75 or args.once:
                    return code
                # Keep exact state and arguments; no step-count, batch-size or
                # training-rule change is made to work around a resource pause.
                time.sleep(args.poll_seconds)
    except FinishPause as error:
        print(str(error), flush=True)
        return error.code


if __name__ == "__main__":
    raise SystemExit(main())
