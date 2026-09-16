"""Exercise the user-selected workflow without starting training or inference."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from granitewxc.temporal.config import parse_temporal_config
from granitewxc.temporal.entrypoints import build_parser


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples/CORDEX_ML/notebooks/Temporal_selected_training_v2.ipynb"


def code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def setup_workflow(monkeypatch, tmp_path, case, backend):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "Prithvi"))
    cells = code_cells()
    settings = ast.parse(cells[0])
    for node in settings.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"CASE", "BACKEND"}:
                node.value = ast.Constant(case if name == "CASE" else backend)
            elif name.startswith("RUN_"):
                # Saved user choices may enable an action; tests choose their own.
                node.value = ast.Constant(False)
    ast.fix_missing_locations(settings)
    namespace = {}
    exec(compile(settings, str(NOTEBOOK), "exec"), namespace)
    exec(compile(cells[1], str(NOTEBOOK), "exec"), namespace)
    return cells, namespace


@pytest.mark.parametrize("case", ["SA", "NARR"])
@pytest.mark.parametrize("backend", ["recurrent", "mamba", "native_pair", "native_pair_pretext"])
def test_all_workflow_choices_use_real_temporal_configs_and_cli(monkeypatch, tmp_path, case, backend):
    cells, ns = setup_workflow(monkeypatch, tmp_path, case, backend)
    config = yaml.safe_load(ns["CONFIG"].read_text(encoding="utf-8"))
    temporal = parse_temporal_config(config["temporal"])
    assert temporal.enabled
    assert temporal.backend == ("native_pair" if backend.startswith("native_pair") else backend)
    if backend == "native_pair_pretext":
        assert temporal.native_pair.pretext.masked_reconstruction_enabled
        assert temporal.native_pair.pretext.transition_enabled
    assert ns["CLI"].name == ("cordex_temporal_training.py" if case == "SA" else "narr_prism_temporal.py")
    assert ns["UPDATES_PER_EPOCH"] is ns["VALIDATION_WINDOWS"] is ns["TOTAL_EPOCHS"] is None

    calls = []
    parser = build_parser()

    def capture(command, *, cwd, seen_warnings):
        assert cwd == ROOT and isinstance(seen_warnings, set)
        assert command[:3] == [sys.executable, "-u", str(ns["CLI"])]
        calls.append(parser.parse_args(command[3:]))

    ns["run_notebook_command"] = capture
    ns["RUN"] = tmp_path / "selected_run"
    ns["CHECKPOINT"] = ns["RUN"] / "checkpoints/last.ckpt"
    ns["RUN_TRAIN"] = True
    exec(compile(cells[2], str(NOTEBOOK), "exec"), ns)
    trained = calls[-1]
    assert trained.command == "train" and trained.resume is None
    assert trained.config == str(ns["CONFIG"])
    assert trained.epochs is trained.max_steps is trained.max_val_steps is None
    assert trained.num_workers == 0
    assert trained.notebook_output is True

    ns["CHECKPOINT"].parent.mkdir(parents=True)
    ns["CHECKPOINT"].write_bytes(b"checkpoint existence fixture; never loaded")
    ns["RUN_TRAIN"], ns["RUN_RESUME"], ns["RUN_INFER"] = False, True, True
    ns["RESUME_TOTAL_EPOCHS"], ns["UPDATES_PER_EPOCH"] = 8, 25
    exec(compile(cells[3], str(NOTEBOOK), "exec"), ns)
    resumed = calls[-1]
    assert resumed.resume == str(ns["CHECKPOINT"])
    assert resumed.epochs == 8 and resumed.max_steps == 25
    exec(compile(cells[4], str(NOTEBOOK), "exec"), ns)
    assert calls[-1].command == "infer" and calls[-1].checkpoint == str(ns["CHECKPOINT"])
    assert calls[-1].split == "test"
    predictions = ns["RUN"] / "inference/predictions.npz"
    (ns["RUN"] / "inference_summary.json").write_text(json.dumps({"npz": str(predictions)}))
    ns["RUN_EVALUATE"] = True
    exec(compile(cells[5], str(NOTEBOOK), "exec"), ns)
    assert calls[-1].command == "evaluate" and calls[-1].predictions == str(predictions)


def test_default_run_all_never_starts_a_process(monkeypatch, tmp_path):
    cells, ns = setup_workflow(monkeypatch, tmp_path, "SA", "recurrent")

    def unexpected(*args, **kwargs):
        pytest.fail("Default Run All started a subprocess")

    monkeypatch.setattr(subprocess, "Popen", unexpected)
    ns["run_notebook_command"] = unexpected
    for source in cells[2:]:
        exec(compile(source, str(NOTEBOOK), "exec"), ns)


def test_notebook_generation_does_not_create_yaml_files_by_default(monkeypatch, tmp_path):
    path = ROOT / "examples/CORDEX_ML/cordex_temporal_workflow_assets.py"
    spec = importlib.util.spec_from_file_location("workflow_assets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    configurations = module.configuration_paths()
    for relative in configurations.values():
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    (tmp_path / "examples/CORDEX_ML/notebooks").mkdir()
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*.yaml"))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    module.generate()
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*.yaml")) == before

    generated = json.loads((tmp_path / "examples/CORDEX_ML/notebooks/Temporal_selected_training_v2.ipynb").read_text(encoding="utf-8"))
    settings = ast.parse("".join(generated["cells"][1]["source"]))
    switches = {node.targets[0].id: ast.literal_eval(node.value) for node in settings.body
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.startswith("RUN_")}
    assert switches == {"RUN_TRAIN": False, "RUN_RESUME": False, "RUN_INFER": False, "RUN_EVALUATE": False}
