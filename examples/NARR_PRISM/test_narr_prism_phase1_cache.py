"""Focused synthetic tests for the reusable NARR--PRISM Phase-1 cache."""

from __future__ import annotations

import json
import os
import signal
from datetime import date
from subprocess import PIPE, Popen, TimeoutExpired
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import narr_prism_phase1_cache as cache_module
from narr_prism_phase1_cache import (
    BUILD_LOCK_NAME,
    CACHE_SCHEMA,
    CACHE_SCHEMA_VERSION,
    CACHE_STATE_INCOMPLETE,
    RESIDUAL_DEFINITION,
    TARGET_VALID_MASK_CONTENT_SHA256_ATTR,
    TARGET_VALID_MASK_CRITERION_ATTR,
    TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR,
    TARGET_VALID_MASK_SHA256_ATTR,
    TARGET_VALID_MASK_SOURCE_SPLIT_ATTR,
    TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR,
    CacheBuildLock,
    CacheBuildLockError,
    CacheBuildNotActiveError,
    CacheWorkerLease,
    CacheWorkerLeaseError,
    Phase1ResidualCacheReader,
    _atomic_json,
    _coverage_multiplicity,
    _parallel_build,
    _install_worker_signal_handlers,
    _restore_signal_handlers,
    _target_valid_mask_content_sha256,
    _terminate_processes,
    _wait_for_processes,
    _write_daily_cache,
    build_parser,
    cache_build_activity,
    contract_digest,
    daily_cache_path,
    describe_cache,
    finalize_cache,
    load_and_validate_manifest,
    validate_daily_cache,
    wait_for_cache_completion,
)
from netCDF4 import Dataset as NetCDFDataset

from granitewxc.utils.normalization import TARGET_VALID_MASK_CRITERION
from granitewxc.utils.prism_grid import validate_prism_grid
from granitewxc.utils.prism_tiling import TilePlan


def _manifest(tmp_path):
    geometry = {
        "domain_shape": [4, 5],
        "core_shape": [3, 3],
        "stride": [2, 2],
        "overlap": [1, 1],
        "halo": [0, 0],
        "blend_mode": "hann",
        "skip_empty_target_tiles": False,
        "min_valid_target_fraction": 0.0,
    }
    lat = np.linspace(30.0, 33.0, 4)
    lon = np.linspace(-124.0, -120.0, 5)
    grid_entry = validate_prism_grid(
        lat, lon, context="synthetic cache grid"
    ).manifest_entry()
    mask_entry = {
        "sha256": "c" * 64,
        "criterion": TARGET_VALID_MASK_CRITERION,
        "source_split": "training",
        "grid_fingerprint": grid_entry["fingerprint"],
        "training_source_artifact_split_signature": None,
    }
    contract = {
        "schema": CACHE_SCHEMA,
        "schema_version": CACHE_SCHEMA_VERSION,
        "case_name": "synthetic",
        "phase1_checkpoint": {
            "sha256": "a" * 64,
            "phase1_fingerprint": "b" * 64,
            "fingerprint_algorithm": "phase1_state_sha256",
        },
        "target_variables": ["ppt", "tmax", "tmin"],
        "prism_pipeline_contract": {
            "coordinates_and_artifacts": {
                "grid": grid_entry,
                "target_valid_mask": mask_entry,
            }
        },
        "cache_geometry": geometry,
        "baseline_semantics": "synthetic_test",
        "target_space": "phase1_normalized_target_space",
        "residual_definition": RESIDUAL_DEFINITION,
        "statistics": {
            "fit_split": "training",
            "weighting": "training_tile_overlap_multiplicity",
            "residual_normalization_enabled": True,
            "residual_normalization_epsilon": 1.0e-6,
        },
    }
    digest = contract_digest(contract)
    root = tmp_path / digest[:32]
    split_dates = {
        "training": ["2000-01-01", "2000-01-02"],
        "validation": ["2000-01-03"],
    }
    splits = {
        split: {
            "expected_dates": dates,
            "expected_count": len(dates),
            "completed_count": 0,
            "inventory_digest_sha256": None,
        }
        for split, dates in split_dates.items()
    }
    manifest = {
        "schema": CACHE_SCHEMA,
        "schema_version": CACHE_SCHEMA_VERSION,
        "state": CACHE_STATE_INCOMPLETE,
        "contract_digest": digest,
        "contract": contract,
        "splits": splits,
        "residual_normalization": None,
    }
    manifest_path = root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest, manifest_path, split_dates


def _write_days(manifest, manifest_path, split_dates):
    shape = (4, 5)
    lat = np.linspace(30.0, 33.0, shape[0])
    lon = np.linspace(-124.0, -120.0, shape[1])
    grid = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    valid = np.ones((3, *shape), dtype=np.uint8)
    valid[1, 0, 0] = 0
    static_mask = np.ones(shape, dtype=np.uint8)
    mask_entry = manifest["contract"]["prism_pipeline_contract"][
        "coordinates_and_artifacts"
    ]["target_valid_mask"]
    mask_provenance = {
        TARGET_VALID_MASK_SHA256_ATTR: mask_entry["sha256"],
        TARGET_VALID_MASK_CONTENT_SHA256_ATTR: (
            _target_valid_mask_content_sha256(static_mask)
        ),
        TARGET_VALID_MASK_CRITERION_ATTR: mask_entry["criterion"],
        TARGET_VALID_MASK_SOURCE_SPLIT_ATTR: mask_entry["source_split"],
        TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR: (
            "not-applicable:raw-training-source-mode"
        ),
        TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR: mask_entry[
            "grid_fingerprint"
        ],
    }
    for split, dates in split_dates.items():
        for day_index, sample_date in enumerate(dates):
            baseline = np.stack([grid + channel for channel in range(3)])
            residual = np.stack(
                [grid * (channel + 1) + day_index for channel in range(3)]
            ).astype(np.float32)
            residual[valid == 0] = 0.0
            _write_daily_cache(
                daily_cache_path(manifest_path.parent, split, sample_date),
                manifest=manifest,
                split=split,
                sample_date=sample_date,
                lat=lat,
                lon=lon,
                deterministic_physical=baseline + 10.0,
                deterministic_normalized=baseline,
                residual_target_normalized=residual,
                residual_valid_mask=valid,
                prism_valid_mask=static_mask,
                mask_provenance=mask_provenance,
            )


def test_writer_reader_and_finalize_round_trip(tmp_path):
    manifest, manifest_path, split_dates = _manifest(tmp_path)
    _write_days(manifest, manifest_path, split_dates)
    first_path = daily_cache_path(
        manifest_path.parent, "training", date(2000, 1, 1)
    )
    validate_daily_cache(
        first_path,
        manifest,
        expected_split="training",
        expected_date="2000-01-01",
        verify_content=True,
    )
    with NetCDFDataset(first_path) as dataset:
        for name in (
            "deterministic_physical",
            "deterministic_normalized",
            "residual_target_normalized",
            "residual_valid_mask",
            "prism_valid_mask",
        ):
            assert dataset.variables[name].filters()["fletcher32"] is True

    finalized = finalize_cache(manifest_path)
    assert finalized["state"] == "complete"
    assert finalized["splits"]["training"]["completed_count"] == 2
    assert finalized["splits"]["validation"]["completed_count"] == 1
    assert finalized["splits"]["training"]["expected_count"] == 2
    assert finalized["splits"]["validation"]["expected_count"] == 1
    stats = finalized["residual_normalization"]
    assert stats["fit_split"] == "training"
    assert stats["fit_days"] == 2
    assert all(count > 0 for count in stats["count"])
    plan = TilePlan.build((4, 5), (3, 3), overlap=(1, 1), halo=(0, 0))
    weights = _coverage_multiplicity((4, 5), plan.positions, (3, 3)).astype(
        np.float64
    )
    grid = np.arange(20, dtype=np.float64).reshape(4, 5)
    valid = np.ones((3, 4, 5), dtype=bool)
    valid[1, 0, 0] = False
    expected_mean = []
    expected_count = []
    for channel in range(3):
        channel_values = []
        channel_weights = []
        for day_index in range(2):
            values = grid * (channel + 1) + day_index
            channel_values.append(values[valid[channel]])
            channel_weights.append(weights[valid[channel]])
        expected_mean.append(
            np.average(
                np.concatenate(channel_values),
                weights=np.concatenate(channel_weights),
            )
        )
        expected_count.append(int(np.concatenate(channel_weights).sum()))
    np.testing.assert_allclose(stats["mean"], expected_mean)
    assert stats["count"] == expected_count
    assert np.isfinite(stats["mean"]).all()
    assert np.isfinite(stats["std"]).all()

    reader = Phase1ResidualCacheReader(finalized)
    cached = reader._load_day("training", "2000-01-01")
    assert "deterministic_physical" not in cached
    crop = reader.load_crop("training", "2000-01-01", 1, 1, 2, 3)
    assert set(crop) == {
        "__phase1_normalized",
        "__residual_target_normalized",
        "__residual_valid_mask",
    }
    assert tuple(crop["__phase1_normalized"].shape) == (3, 2, 3)
    physical = reader.load_crop(
        "training",
        "2000-01-01",
        1,
        1,
        2,
        3,
        include_physical=True,
    )
    assert "__phase1_physical" in physical
    assert not list(manifest_path.parent.rglob("*.tmp"))


def test_overlap_multiplicity_matches_tile_plan():
    plan = TilePlan.build((4, 5), (3, 3), overlap=(1, 1), halo=(0, 0))
    observed = _coverage_multiplicity((4, 5), plan.positions, (3, 3))
    assert observed.shape == (4, 5)
    assert observed.min() == 1
    assert observed.max() == 4


def test_incomplete_manifest_and_tampered_daily_file_are_rejected(tmp_path):
    manifest, manifest_path, split_dates = _manifest(tmp_path)
    with pytest.raises(RuntimeError, match="not complete"):
        load_and_validate_manifest(manifest_path, validate_inventory=False)

    _write_days(manifest, manifest_path, split_dates)
    first_path = daily_cache_path(
        manifest_path.parent, "training", "2000-01-01"
    )
    with NetCDFDataset(first_path, "r+") as dataset:
        dataset.cache_contract_digest = "0" * 64
    with pytest.raises(RuntimeError, match="different cache contract"):
        validate_daily_cache(first_path, manifest)

    with NetCDFDataset(first_path, "r+") as dataset:
        dataset.cache_contract_digest = manifest["contract_digest"]
        variable = dataset.variables["deterministic_normalized"]
        variable[0, 0, 0] = float(variable[0, 0, 0]) + 1.0
    with pytest.raises(RuntimeError, match="content digest mismatch"):
        validate_daily_cache(first_path, manifest, verify_content=True)
    with pytest.raises(RuntimeError, match="content digest mismatch"):
        finalize_cache(manifest_path)


def test_repair_invalid_deep_validates_existing_file_and_rebuilds(
    monkeypatch, tmp_path
):
    manifest_path = tmp_path / "digest" / "manifest.json"
    sample_date = date(2000, 1, 1)
    output_path = daily_cache_path(
        manifest_path.parent, "training", sample_date
    )
    output_path.parent.mkdir(parents=True)
    output_path.write_text("synthetic-corrupt-cache", encoding="utf-8")
    manifest = {
        "state": CACHE_STATE_INCOMPLETE,
        "contract": {
            "cache_geometry": {
                "domain_shape": [4, 5],
                "core_shape": [3, 3],
                "overlap": [1, 1],
                "halo": [0, 0],
                "skip_empty_target_tiles": False,
                "min_valid_target_fraction": 0.0,
            }
        },
    }
    validations = []
    rebuilt = []

    class FakeModel:
        @staticmethod
        def parameters():
            return iter(())

    def validate(_path, _manifest, **kwargs):
        validations.append(bool(kwargs.get("verify_content")))
        if kwargs.get("verify_content"):
            raise RuntimeError("synthetic content digest mismatch")

    monkeypatch.setattr(cache_module, "load_yaml", lambda _path: {})
    monkeypatch.setattr(
        cache_module, "get_config", lambda _path: SimpleNamespace()
    )
    monkeypatch.setattr(
        cache_module,
        "_prepare_manifest",
        lambda *_args, **_kwargs: (manifest, manifest_path),
    )
    monkeypatch.setattr(
        cache_module, "_load_model", lambda *_args, **_kwargs: FakeModel()
    )
    monkeypatch.setattr(
        cache_module,
        "load_target_valid_mask",
        lambda *_args, **_kwargs: np.ones((4, 5), dtype=bool),
    )
    monkeypatch.setattr(
        cache_module, "_target_valid_mask_provenance", lambda *_args: {}
    )
    monkeypatch.setattr(
        cache_module,
        "_cap_tile_batch_for_cuda_indexing",
        lambda **kwargs: kwargs["batch_size"],
    )
    monkeypatch.setattr(
        cache_module,
        "_build_tasks",
        lambda _cfg, _splits: [("training", sample_date)],
    )
    monkeypatch.setattr(
        cache_module, "NarrPrismDataset", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(cache_module, "validate_daily_cache", validate)
    monkeypatch.setattr(
        cache_module,
        "_build_day",
        lambda **kwargs: rebuilt.append(kwargs["output_path"]),
    )
    args = SimpleNamespace(
        config="fixture.yaml",
        checkpoint="last.ckpt",
        phase1_fingerprint="f" * 64,
        splits="training",
        output_root=str(tmp_path),
        worker=True,
        device="cpu",
        batch_size=4,
        date_shard_index=0,
        date_shard_count=1,
        no_progress=True,
        repair_invalid=True,
    )

    assert cache_module._build_worker(args) == 0
    assert validations == [True]
    assert rebuilt == [output_path]


def test_parallel_worker_argv_is_registered_and_parseable(
    monkeypatch, tmp_path
):
    parser = build_parser()
    direct = parser.parse_args(
        [
            "build",
            "--config",
            "fixture.yaml",
            "--checkpoint",
            "last.ckpt",
            "--worker",
        ]
    )
    assert direct.worker is True

    captured = []

    class FinishedProcess:
        def __init__(self, command, **kwargs):
            captured.append((command, kwargs))

        def poll(self):
            return 0

        def wait(self, timeout=None):
            del timeout
            return 0

    manifest_path = tmp_path / "digest" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    monkeypatch.setattr(cache_module, "load_yaml", lambda _path: {})
    monkeypatch.setattr(
        cache_module, "get_config", lambda _path: SimpleNamespace()
    )
    monkeypatch.setattr(
        cache_module,
        "checkpoint_state_fingerprint",
        lambda _path: "f" * 64,
    )
    monkeypatch.setattr(
        cache_module,
        "_resolve_output_root",
        lambda _cfg, _explicit: str(tmp_path),
    )
    monkeypatch.setattr(
        cache_module,
        "_prepare_manifest",
        lambda *_args, **_kwargs: ({}, manifest_path),
    )
    monkeypatch.setattr(cache_module.subprocess, "Popen", FinishedProcess)
    monkeypatch.setattr(
        cache_module,
        "finalize_cache",
        lambda _path: {
            "_manifest_path": str(manifest_path),
            "residual_normalization": {},
        },
    )
    args = parser.parse_args(
        [
            "build",
            "--config",
            "fixture.yaml",
            "--checkpoint",
            "last.ckpt",
            "--output-root",
            str(tmp_path),
            "--parallel-gpus",
            "2,3",
        ]
    )
    assert _parallel_build(args) == 0
    assert len(captured) == 2
    for shard, (command, kwargs) in enumerate(captured):
        child = parser.parse_args(command[2:])
        assert child.worker is True
        assert child.date_shard_index == shard
        assert child.date_shard_count == 2
        assert kwargs["pass_fds"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == str(shard + 2)


class _FakeProcess:
    def __init__(self, returncode=None, *, timeout=False):
        self.returncode = returncode
        self.timeout = timeout
        self.terminated = False
        self.killed = False
        self.reaped = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if timeout is not None and self.timeout and not self.killed:
            raise TimeoutExpired("synthetic-worker", timeout)
        self.reaped = True
        if self.returncode is None:
            self.returncode = -15
        return self.returncode


def test_worker_failure_is_fail_fast_and_cleanup_kills_then_reaps():
    failed = _FakeProcess(returncode=3)
    running = _FakeProcess(returncode=None, timeout=True)
    with pytest.raises(RuntimeError, match="worker failures"):
        _wait_for_processes([("0", failed), ("1", running)])
    _terminate_processes([("0", failed), ("1", running)], timeout=0.0)
    assert running.terminated
    assert running.killed
    assert running.reaped
    assert failed.reaped


def test_partial_date_shard_stays_incomplete_and_resumes(tmp_path):
    manifest, manifest_path, split_dates = _manifest(tmp_path)
    _write_days(
        manifest,
        manifest_path,
        {"training": [split_dates["training"][0]]},
    )
    with pytest.raises(FileNotFoundError, match="Missing daily"):
        finalize_cache(manifest_path)

    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored["state"] == CACHE_STATE_INCOMPLETE
    assert stored["splits"]["training"]["completed_count"] == 0
    assert stored["splits"]["validation"]["completed_count"] == 0

    _write_days(manifest, manifest_path, split_dates)
    resumed = finalize_cache(manifest_path)
    assert resumed["state"] == "complete"
    assert resumed["splits"]["training"]["completed_count"] == 2
    assert resumed["splits"]["validation"]["completed_count"] == 1


def test_build_lock_reports_owner_and_releases_for_resume(tmp_path):
    lock_path = tmp_path / BUILD_LOCK_NAME
    with CacheBuildLock(tmp_path, command=["first-parent"]) as held:
        assert held.path == lock_path
        active = json.loads(lock_path.read_text(encoding="utf-8"))
        assert active["state"] == "active"
        assert active["pid"] == os.getpid()
        assert active["command"] == ["first-parent"]
        with pytest.raises(CacheBuildLockError) as caught:
            CacheBuildLock(tmp_path, command=["duplicate-parent"]).acquire()
        message = str(caught.value)
        assert "pid={}".format(os.getpid()) in message
        assert "host={}".format(active["host"]) in message
        assert active["started_utc"] in message
        assert "first-parent" in message
        assert "resumable" in message

    released = json.loads(lock_path.read_text(encoding="utf-8"))
    assert released["state"] == "released"
    assert released["released_utc"]

    with pytest.raises(ValueError, match="synthetic interruption"):
        with CacheBuildLock(tmp_path, command=["interrupted-parent"]):
            raise ValueError("synthetic interruption")
    with CacheBuildLock(tmp_path, command=["resumed-parent"]):
        pass


def test_cache_build_activity_uses_live_locks_not_stale_metadata(tmp_path):
    digest_dir = tmp_path / "digest"
    assert not cache_build_activity(tmp_path, cache_dir=digest_dir)["active"]
    assert not (tmp_path / BUILD_LOCK_NAME).exists()

    with CacheBuildLock(tmp_path, command=["active-parent"]):
        status = cache_build_activity(tmp_path, cache_dir=digest_dir)
        assert status["parent_active"]
        assert status["parent_owner"]["pid"] == os.getpid()
        assert status["active"]

    status = cache_build_activity(tmp_path, cache_dir=digest_dir)
    assert not status["parent_active"]
    assert status["parent_owner"]["state"] == "released"
    with CacheWorkerLease(digest_dir, command=["active-workers"]):
        status = cache_build_activity(tmp_path, cache_dir=digest_dir)
        assert not status["parent_active"]
        assert status["workers_active"]
        assert status["active"]


def test_wait_for_cache_completion_polls_active_owner_then_validates(
    monkeypatch, tmp_path
):
    state = {"complete": False}
    incomplete = {
        "state": CACHE_STATE_INCOMPLETE,
        "_manifest_path": str(tmp_path / "digest" / "manifest.json"),
        "contract": {"identity": "stable"},
    }
    complete = {
        "state": "complete",
        "_manifest_path": incomplete["_manifest_path"],
        "contract": incomplete["contract"],
    }
    validated_calls = []

    def fake_load(*args, require_complete, **kwargs):
        validated_calls.append((require_complete, "cfg" in kwargs))
        if require_complete:
            assert state["complete"]
            return complete
        return complete if state["complete"] else incomplete

    monkeypatch.setattr(cache_module, "load_and_validate_manifest", fake_load)
    monkeypatch.setattr(
        cache_module,
        "describe_cache",
        lambda *args, **kwargs: {
            "splits": {"training": {"observed_present_count": 7}}
        },
    )
    monkeypatch.setattr(
        cache_module,
        "cache_build_activity",
        lambda *args, **kwargs: {"active": not state["complete"]},
    )
    statuses = []

    def finish_after_poll(seconds):
        assert seconds == 2.0
        state["complete"] = True

    result = wait_for_cache_completion(
        tmp_path,
        cfg={},
        config=object(),
        phase1_checkpoint=tmp_path / "last.ckpt",
        poll_seconds=2.0,
        status_callback=statuses.append,
        sleep_fn=finish_after_poll,
    )
    assert result is complete
    assert len(statuses) == 1
    assert (
        statuses[0]["summary"]["splits"]["training"]["observed_present_count"]
        == 7
    )
    assert validated_calls == [
        (False, True),
        (False, False),
        (True, True),
    ]


def test_wait_for_cache_completion_requests_resume_when_unowned(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        cache_module,
        "load_and_validate_manifest",
        lambda *args, require_complete, **kwargs: (
            (_ for _ in ()).throw(RuntimeError("not complete"))
            if require_complete
            else {
                "state": CACHE_STATE_INCOMPLETE,
                "_manifest_path": str(tmp_path / "digest" / "manifest.json"),
                "contract": {"identity": "stable"},
            }
        ),
    )
    monkeypatch.setattr(
        cache_module, "describe_cache", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        cache_module,
        "cache_build_activity",
        lambda *args, **kwargs: {"active": False},
    )
    with pytest.raises(CacheBuildNotActiveError, match="launch/resume"):
        wait_for_cache_completion(
            tmp_path,
            cfg={},
            config=object(),
            phase1_checkpoint=tmp_path / "last.ckpt",
            poll_seconds=1.0,
            sleep_fn=lambda seconds: pytest.fail("must not sleep"),
        )


def test_build_lock_is_released_by_kernel_after_crash(tmp_path):
    script = (
        "import os, sys\n"
        "from narr_prism_phase1_cache import CacheBuildLock\n"
        "lock = CacheBuildLock({!r}, command=['crash-holder'])\n"
        "lock.acquire()\n"
        "print('READY', flush=True)\n"
        "sys.stdin.readline()\n"
        "os._exit(7)\n"
    ).format(str(tmp_path))
    process = Popen(
        [os.sys.executable, "-c", script],
        cwd=os.path.dirname(__file__),
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "READY"
        with pytest.raises(CacheBuildLockError) as caught:
            CacheBuildLock(tmp_path, command=["contender"]).acquire()
        assert "pid={}".format(process.pid) in str(caught.value)
        process.stdin.write("\n")
        process.stdin.flush()
        assert process.wait(timeout=15) == 7
        with CacheBuildLock(tmp_path, command=["post-crash-resume"]):
            pass
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def test_describe_reports_observed_progress_without_claiming_completion(
    tmp_path,
):
    manifest, manifest_path, split_dates = _manifest(tmp_path)
    _write_days(
        manifest,
        manifest_path,
        {"training": [split_dates["training"][0]]},
    )
    malformed = daily_cache_path(
        manifest_path.parent,
        "validation",
        split_dates["validation"][0],
    )
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text("not a NetCDF file", encoding="utf-8")

    summary = describe_cache(manifest_path)
    training = summary["splits"]["training"]
    validation = summary["splits"]["validation"]
    assert summary["state"] == CACHE_STATE_INCOMPLETE
    assert training["completed_count"] == 0
    assert training["observed_present_count"] == 1
    assert training["observed_missing_count"] == 1
    assert training["observed_invalid_count"] is None
    assert training["observed_validation"] == "not_scanned"
    assert validation["completed_count"] == 0
    assert validation["observed_present_count"] == 1
    assert validation["observed_missing_count"] == 0
    assert validation["observed_invalid_count"] is None

    validated = describe_cache(manifest_path, validate_present=True)
    assert validated["splits"]["training"]["observed_valid_count"] == 1
    assert validated["splits"]["training"]["observed_invalid_count"] == 0
    assert validated["splits"]["validation"]["observed_valid_count"] == 0
    assert validated["splits"]["validation"]["observed_invalid_count"] == 1
    assert validated["splits"]["validation"]["observed_invalid_examples"]


def test_inherited_worker_lease_blocks_resume_until_orphan_exits(tmp_path):
    lease = CacheWorkerLease(tmp_path, command=["original-parent"]).acquire()
    child = Popen(
        [
            os.sys.executable,
            "-c",
            "import sys, time; print('LEASED', flush=True); time.sleep(60)",
        ],
        pass_fds=(lease.fileno(),),
        stdout=PIPE,
        stderr=PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "LEASED"
        lease.release()
        with pytest.raises(CacheWorkerLeaseError) as caught:
            CacheWorkerLease(tmp_path, command=["resumed-parent"]).acquire()
        message = str(caught.value)
        assert "original-parent" in message
        assert "duplicate daily writers" in message
    finally:
        child.terminate()
        child.wait(timeout=15)
        lease.release()

    with CacheWorkerLease(tmp_path, command=["post-orphan-resume"]):
        pass


def test_sigterm_handler_stops_children_before_releasing_worker_lease():
    child = _FakeProcess(returncode=None)
    released = []
    previous = _install_worker_signal_handlers(
        [("0", child)],
        release_worker_lease=lambda: released.append(True),
    )
    try:
        handler = signal.getsignal(signal.SIGTERM)
        with pytest.raises(SystemExit) as caught:
            handler(signal.SIGTERM, None)
        assert caught.value.code == 128 + int(signal.SIGTERM)
        assert child.terminated
        assert child.reaped
        assert released == [True]
    finally:
        _restore_signal_handlers(previous)


def test_cache_builder_numeric_policy_is_explicit(monkeypatch):
    prior_matmul_precision = torch.get_float32_matmul_precision()
    prior_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    prior_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    prior_benchmark = bool(torch.backends.cudnn.benchmark)
    prior_deterministic = bool(torch.backends.cudnn.deterministic)
    prior_algorithms = torch.are_deterministic_algorithms_enabled()
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    try:
        policy = cache_module.configure_phase1_cache_numerics()
        assert policy == {
            "dtype": "float32",
            "autocast": False,
            "allow_tf32": False,
            "float32_matmul_precision": "highest",
            "deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8",
        }
    finally:
        torch.set_float32_matmul_precision(prior_matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = prior_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prior_cudnn_tf32
        torch.backends.cudnn.benchmark = prior_benchmark
        torch.backends.cudnn.deterministic = prior_deterministic
        torch.use_deterministic_algorithms(prior_algorithms)


def test_cache_builder_rejects_conflicting_cublas_policy(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(RuntimeError, match="requires CUBLAS_WORKSPACE_CONFIG"):
        cache_module.configure_phase1_cache_numerics()


def test_complete_inventory_validation_can_report_progress(
    tmp_path, monkeypatch
):
    manifest, manifest_path, split_dates = _manifest(tmp_path)
    _write_days(manifest, manifest_path, split_dates)
    finalize_cache(manifest_path)
    calls = []

    def fake_tqdm(values, **kwargs):
        materialized = list(values)
        calls.append((kwargs["desc"], kwargs["unit"], len(materialized)))
        return materialized

    monkeypatch.setattr(cache_module, "tqdm", fake_tqdm)
    loaded = load_and_validate_manifest(
        manifest_path, show_inventory_progress=True
    )
    assert loaded["state"] == "complete"
    assert calls == [
        ("Validate Phase1 cache (training)", "day", 2),
        ("Validate Phase1 cache (validation)", "day", 1),
    ]
