"""Generate the selected temporal notebook; create extended YAMLs only on request."""
from pathlib import Path
import argparse
import json
import yaml

ROOT = Path(__file__).resolve().parents[2]


def configuration_paths():
    configurations = {}
    for case, folder, stem in (
        ("SA", "CORDEX_ML", "SA_downscaling_refinement_T2_ACCESS-CM2_static"),
        ("NARR", "NARR_PRISM", "NARR_PRISM_subdomain"),
    ):
        for backend in ("recurrent", "mamba", "native_pair", "native_pair_pretext"):
            suffix = f"prithvi_{backend}" if backend.startswith("native_pair") else backend
            source = ROOT / f"examples/{folder}/{stem}_temporal_{suffix}.yaml"
            if not source.is_file():
                raise FileNotFoundError(f"Missing workflow configuration: {source}")
            configurations[f"{case}/{backend}"] = source.relative_to(ROOT).as_posix()
    return configurations


def generate(*, include_extended_configs=False):
    configurations = configuration_paths()
    if include_extended_configs:
        for selection, relative in configurations.items():
            if not selection.split("/")[1].startswith("native_pair"):
                continue
            source = ROOT / relative
            name = source.stem + "_extended_v2"
            target = source.with_name(name + ".yaml")
            # Preserve any user's edits to an existing training schedule.
            if target.exists():
                continue
            raw = yaml.safe_load(source.read_text(encoding="utf-8"))
            raw["num_epochs"] = 20
            raw["temporal"]["freeze"]["unfreeze_schedule"] = [
                {"epoch": 4, "modules": ["encoder"], "lr_scale": 0.1},
                {"epoch": 10, "modules": ["backbone"], "lr_scale": 0.05},
            ]
            folder = source.parent.name
            output_root = f"./examples/{folder}/runs_temporal/{name}"
            raw["job_id"] = name
            if selection.startswith("SA/"):
                raw["case_name"] = name
            raw["path_experiment"] = output_root
            raw["checkpoint_dir"] = output_root + "/checkpoints"
            raw["run_dir"] = output_root + "/runs"
            raw["inference"]["output_dir"] = output_root + "/inference"
            raw["temporal"]["inference"]["output_dir"] = output_root + "/inference"
            target.write_text(
                "# Optional extended training; never launched by asset generation.\n"
                "# Existing Adam state is retained; physical scalers remain frozen.\n"
                + yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cells = []
    def markdown(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})
    def code(text):
        cells.append({"cell_type": "code", "metadata": {}, "source": text.splitlines(True),
                      "outputs": [], "execution_count": None})
    markdown("""# Selected temporal model: train → resume → infer → evaluate

Select recurrent (ConvGRU), Mamba, native pair, or native pair with pretext losses.
Unlike the spatial-only SA fine-tuning notebook, this workflow dispatches to
`train_temporal_model` through the temporal CLI. It runs in one process on one
selected device; it does not force the spatial notebook's two-GPU FSDP setup.

Select one case/configuration below. Each action uses that configuration and its
selected checkpoint. Training and full-period inference run only when their
switches are enabled. Backend comparisons belong to the separate experiment CLI.

Use the **Prithvi** kernel. These are experimental workflows; passing an engineering
test does not establish scientific acceptance. NARR requires actual local data,
scalers and checkpoint paths. Existing Linux symlink stubs are preserved.
""")
    code("""from pathlib import Path
import json, subprocess, sys, yaml
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'granitewxc/temporal').is_dir())
CASE = 'SA'                    # 'SA' or 'NARR'
BACKEND = 'recurrent'          # 'recurrent', 'mamba', 'native_pair', 'native_pair_pretext'
""" + "CONFIGURATIONS = " + repr(configurations) + "\n" + """CONFIG = ROOT / CONFIGURATIONS[f'{CASE}/{BACKEND}']
# Native variants also offer sibling *_extended_v2.yaml training schedules.
RUN = ROOT / f'examples/{"CORDEX_ML" if CASE == "SA" else "NARR_PRISM"}/runs_temporal/selected_{CASE}_{BACKEND}_v2'
CHECKPOINT = RUN / 'checkpoints/last.ckpt'
CLI = ROOT / ('examples/CORDEX_ML/cordex_temporal_training.py' if CASE == 'SA' else 'examples/NARR_PRISM/narr_prism_temporal.py')
OVERRIDES = {}  # Existing dotted keys → values; see examples below.
# NARR examples (actual paths required):
# OVERRIDES = {'data.preprocessed_dir': '/data/preprocessed',
#              'temporal.init_from_spatial_checkpoint': '/data/checkpoints/last.ckpt',
#              'model.input_mu': '/data/scalars/inputs_mean.npy', ...}
DEVICE = None                 # None = YAML/auto; or 'cuda' / 'cpu'
UPDATES_PER_EPOCH = None       # Optional successful-update cap; None = full epoch
VALIDATION_WINDOWS = None      # None = full configured validation split
TOTAL_EPOCHS = None            # None = selected YAML's num_epochs
RESUME_TOTAL_EPOCHS = None     # Total desired epochs; None = selected YAML
RUN_TRAIN = False
RUN_RESUME = False
RUN_INFER = False
RUN_EVALUATE = False
assert CONFIG.is_file(), CONFIG
assert not (RUN_TRAIN and RUN_RESUME), 'Select train OR resume for this execution'
assert Path(sys.prefix).name.casefold() == 'prithvi', 'Select the Prithvi kernel before running this workflow'
print('Interpreter:', sys.executable)
print('Configuration:', CONFIG)
print('Checkpoint:', CHECKPOINT)
""")
    code("""if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from granitewxc.temporal.progress import run_notebook_command
# Remember identical warnings across train/resume/infer and helper-cell reruns.
_TEMPORAL_SEEN_WARNINGS = globals().get('_TEMPORAL_SEEN_WARNINGS', set())

def invoke(action, *arguments):
    command = [sys.executable, '-u', str(CLI), action, '--config', str(CONFIG), '--notebook-output']
    if DEVICE is not None:
        command.extend(['--device', DEVICE])
    for key, value in OVERRIDES.items():
        command.extend(['--set', f'{key}={json.dumps(value)}'])
    command.extend(str(a) for a in arguments)
    print(subprocess.list2cmdline(command))
    run_notebook_command(command, cwd=ROOT, seen_warnings=_TEMPORAL_SEEN_WARNINGS)

def training_arguments(epochs):
    arguments = ['--num-workers', 0]
    for flag, value in (('--epochs', epochs), ('--max-steps', UPDATES_PER_EPOCH),
                        ('--max-val-steps', VALIDATION_WINDOWS)):
        if value is not None:
            arguments.extend([flag, value])
    return arguments
""")
    markdown("## Train the selected model\nDefaults use the selected YAML epoch count, accumulation and complete data splits. Optional `--max-steps` caps successful optimizer updates per epoch. One tqdm bar per epoch updates after each training microbatch, showing loss and successful optimizer updates. The same bar shows validation status and its final loss. Identical warnings appear once per notebook session. Validation selects `best.ckpt`; `last.ckpt` preserves continuation state.")
    code("""if RUN_TRAIN:
    assert not CHECKPOINT.exists(), 'Choose a new RUN directory or use resume'
    invoke('train', '--output-dir', RUN / 'checkpoints', *training_arguments(TOTAL_EPOCHS),
           '--output', RUN / 'training_summary.json')
""")
    markdown("## Resume the selected checkpoint\nKeep the data, loss, accumulation and step protocol identical for exact resume. `RESUME_TOTAL_EPOCHS` is the desired total; increase it explicitly to extend a completed run. The trainer rejects incompatible protocols. Legacy checkpoints lacking RNG and data order are explicitly reported as approximate restarts.")
    code("""if RUN_RESUME:
    assert CHECKPOINT.is_file(), CHECKPOINT
    invoke('train', '--resume', CHECKPOINT, '--output-dir', RUN / 'checkpoints',
           *training_arguments(RESUME_TOTAL_EPOCHS),
           '--output', RUN / 'resume_summary.json')
""")
    markdown("## Infer with the selected checkpoint\nDefaults to the configured full test period. SA bounded comparison uses 1981-01-01 through 1983-12-31. Daily means remain daily means. Chunking supplies real history; unavailable history at a true run start follows the declared cold-start rule.")
    code("""if RUN_INFER:
    assert CHECKPOINT.is_file(), CHECKPOINT
    invoke('infer', '--checkpoint', CHECKPOINT, '--split', 'test', '--chunk-length', 30,
           '--output-dir', RUN / 'inference', '--output', RUN / 'inference_summary.json')
""")
    markdown("## Evaluate the emitted predictions\nThe summary identifies the exact prediction file. Metrics use matching timestamps, masks and physical units. The same test period must not become a repeated development target.")
    code("""if RUN_EVALUATE:
    summary = json.loads((RUN / 'inference_summary.json').read_text())
    invoke('evaluate', '--predictions', summary['npz'], '--event-aligned',
           '--output', RUN / 'evaluation.json')
""")
    markdown("""## Separate backend comparison and refinement

The six-variant comparison is `cordex_temporal_experiment.py --steps 600
--val-steps 60 --epochs 1 --test-years 3 --out <new-directory>`; it is intentionally
not run by this notebook. Optional `*_extended_v2.yaml` schedules are separate
development experiments, not the pre-registered comparison.

Recurrent and Mamba entry points and all four stochastic-refinement interfaces
remain available. Interface compatibility does not show a refiner remains
calibrated after changing Phase-1 predictions. Refinement retraining is separate.
""")
    for index, cell in enumerate(cells):
        cell["id"] = f"selected-workflow-{index}"
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Prithvi", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
    target = ROOT / "examples/CORDEX_ML/notebooks/Temporal_selected_training_v2.ipynb"
    target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-extended-configs", action="store_true",
                        help="Also create missing optional native extended-training YAMLs")
    args = parser.parse_args()
    generate(include_extended_configs=args.include_extended_configs)
