"""GPU occupancy selection and the notebook's subprocess launch contract."""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.NARR_PRISM.narr_prism_utils import select_idle_gpus


REPO_ROOT = Path(__file__).resolve().parents[1]
NARR_DIR = REPO_ROOT / "examples" / "NARR_PRISM"
GPU_UUIDS = [f"GPU-{index:08d}-aaaa-bbbb-cccc-dddddddddddd" for index in range(4)]


def mock_inventory(monkeypatch, *, busy=(), memory=None, utilization=None):
    memory = memory or {}
    utilization = utilization or {}
    inventory = "\n".join(
        f"{index}, {uuid}, {memory.get(index, 0)}, {utilization.get(index, 0)}"
        for index, uuid in enumerate(GPU_UUIDS)
    )

    def run(command, **kwargs):
        assert command[0] == "nvidia-smi"
        assert kwargs["timeout"] == 10
        output = (
            "\n".join(GPU_UUIDS[index] for index in busy)
            if command[1] == "--query-compute-apps=gpu_uuid"
            else inventory
        )
        return SimpleNamespace(stdout=output)

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(subprocess, "run", run)


def test_uses_only_idle_gpu_when_other_gpus_have_compute_processes(monkeypatch):
    # Even a sleeping compute process with zero sampled utilization is busy.
    mock_inventory(monkeypatch, busy=(0, 2, 3))
    assert select_idle_gpus(4) == [GPU_UUIDS[1]]


def test_available_gpus_are_capped_by_requested_worker_count(monkeypatch):
    mock_inventory(monkeypatch)
    assert select_idle_gpus(4) == GPU_UUIDS
    assert select_idle_gpus(1) == GPU_UUIDS[:1]


def test_rejects_high_memory_active_or_unknown_occupancy(monkeypatch):
    mock_inventory(
        monkeypatch,
        memory={0: 12000, 3: "[N/A]"},
        utilization={2: 100},
    )
    assert select_idle_gpus(4) == [GPU_UUIDS[1]]


def test_respects_inherited_visibility_order_and_explicit_override(monkeypatch):
    mock_inventory(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    assert select_idle_gpus(4) == [GPU_UUIDS[3], GPU_UUIDS[1]]
    assert select_idle_gpus(1, "2") == [GPU_UUIDS[2]]
    assert select_idle_gpus(4, GPU_UUIDS[1][:12]) == [GPU_UUIDS[1]]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3,1"


@pytest.mark.parametrize("visible", [None, "0,2,3", "", "-1"])
def test_all_eligible_gpus_busy_or_hidden_does_not_select_busy_gpu(
    monkeypatch, visible
):
    mock_inventory(monkeypatch, busy=(0, 1, 2, 3))
    with pytest.raises(RuntimeError, match="No idle GPUs"):
        select_idle_gpus(4, visible)


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 10)],
)
def test_unavailable_gpu_monitor_reports_actionable_error(monkeypatch, error):
    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Unable to check GPU occupancy"):
        select_idle_gpus(4)


@pytest.mark.parametrize("device_target", ["cuda:0", "cpu"])
def test_notebook_launch_passes_one_worker_and_child_visibility(
    monkeypatch, tmp_path, device_target
):
    notebook = json.loads(
        (NARR_DIR / "notebooks" / "narr_prism_inference.ipynb").read_text()
    )
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "subprocess.run(command," in "".join(cell.get("source", []))
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    queries = []
    launches = []

    def select(max_gpus, visible_devices):
        queries.append((max_gpus, visible_devices))
        return [GPU_UUIDS[1]]

    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: launches.append((command, kwargs))
    )
    namespace = {
        "os": os,
        "sys": SimpleNamespace(executable="python"),
        "Path": Path,
        "cfg": {},
        "OUTPUT_DIR": tmp_path,
        "PHASE1_CACHE_DIR": tmp_path / "phase1_cache",
        "NUM_GPUS": 4,
        "force_visible_devices": None,
        "device": torch.device(device_target),
        "select_idle_gpus": select,
        "NARR_PRISM_DIR": NARR_DIR,
        "REPO_ROOT": REPO_ROOT,
        "config_path": "config.yaml",
        "phase1_checkpoint_path": "phase1.ckpt",
        "refinement_checkpoint_path": "phase2.ckpt",
        "BATCH_SIZE": 8,
        "PREDICTION_SPLIT": "inference",
        "RESUME_EXISTING": True,
        "ENSEMBLE_SIZE": 10,
        "SEED": None,
        "LIMIT_DAYS": 0,
        "case_output_dir": lambda root, case: root / case,
        "case_name": "test_case",
    }
    exec(compile(source, "inference-launch-cell", "exec"), namespace)
    command, kwargs = launches[0]
    assert command[command.index("--num-gpus") + 1] == "1"
    assert command[command.index("--device") + 1] == device_target
    assert command[command.index("--ensemble-size") + 1] == "10"
    assert command[command.index("--phase1-cache-dir") + 1] == str(tmp_path / "phase1_cache")
    assert "--resume-existing" in command
    if device_target.startswith("cuda"):
        assert queries == [(4, None)]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == GPU_UUIDS[1]
    else:
        assert queries == []
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3,1"
