# NARR-to-PRISM Downscaling Workflow

This directory contains a complete YAML-driven workflow for statistical
downscaling of **NARR** reanalysis predictors to the **PRISM** 800 m
daily observation grid over the contiguous United States.

The pipeline follows a strict, case-scoped artifact contract:

```
training preprocessing → training-only scalars → validation/inference preprocessing
→ training / fine-tuning → tiled inference → evaluation
```

---

## Required Input Data

### NARR Predictors

| Item | Value |
|------|-------|
| **Directory** | `/data2/NARR/data/subset` |
| **Format** | Monthly NetCDF files under `<root>/<variable>/<variable>.YYYYMM.nc` |
| **Variables** | Configurable via `data.predictor_variables` in the YAML |
| **Default variables** | `QV`, `U`, `V`, `T`, `H` at 500, 700, and 850 hPa, mapped to NARR `shum`, `uwnd`, `vwnd`, `air`, and `hgt` |

### PRISM Targets

| Item | Value |
|------|-------|
| **Root directory** | `/data/PRISM/prism_daily_800m_an` |
| **Layout** | `<root>/<variable>/<YYYY>/*.nc` |
| **Variables** | `ppt`, `tmax`, `tmin` (configurable via `data.target_variables`) |
| **Resolution** | ~800 m (30 arc-seconds) |

Example file path:
```
/data/PRISM/prism_daily_800m_an/ppt/2000/prism_ppt_us_30s_20000101.nc
```

---

## YAML Configuration (`NARR_PRISM_subdomain.yaml`)

All paths, variables, date ranges, model parameters, and training
hyper-parameters are controlled by a single YAML file.

### Key sections

| Section | Purpose |
|---------|---------|
| `data.predictor_dir` | Path to NARR monthly variable files |
| `data.target_dir` | Path to PRISM root directory |
| `data.target_variables` | PRISM target sub-folders (`ppt`, `tmax`, `tmin`) |
| `data.predictor_variables` | NARR variable names to use as predictors |
| `data.preprocessed_dir` | Case-scoped root; every output lands under `<preprocessed_dir>/<case_name>/` |
| `data.scalar_dir` | Legacy scalar location (read-only, same-case fallback only); new scalars are written under `<preprocessed_dir>/<case_name>/scalars/` |
| `data.scalers.*` | Paths to the four `.npy` scalar files |
| `dates.training.start / end` | Training date range |
| `dates.validation.start / end` | Optional held-out validation date range |
| `dates.inference.start / end` | Inference date range |
| `preprocess.*` | Controls target inclusion in each preprocessed split |
| `normalization.*` | Predictor and target normalization contract |
| `case_name` | Required run/case name used as an output subfolder |
| `predictands.*` | Per-variable normalization & loss options |
| `model.*` | Architecture hyper-parameters |
| `training.*` | Distributed / batch / accumulation settings |
| `loss.*` | Loss function configuration |
| `inference.*` | Tiling, blending, output directory |
| `path_experiment` | Root for checkpoints and run artifacts |
| `path_model_weights` | Pre-trained backbone weights |
| `batch_size`, `num_epochs`, `learning_rate` | Standard training knobs |
| `gradient_accumulation_steps` | Effective batch size control |

---

## Workflow Steps

### 1. Preprocess and Compute Scalars

Use the launcher to generate training products, recompute scalars from those
products, then generate optional validation and inference products in the
required order:

```bash
cd examples/NARR_PRISM

bash run_preprocess.sh \
    --config NARR_PRISM_subdomain.yaml \
    --shards 1
```

Use `--overwrite` to rebuild existing daily NetCDF products. Scalars are always
recomputed. `--shards N` partitions dates across parallel preprocessing
processes without changing the resulting artifact contract.

**Outputs:** `inputs_mean.npy`, `inputs_std.npy`, `targets_mean.npy`,
`targets_std.npy`, and `metadata.json` in `<preprocessed_dir>/<case_name>/scalars/`.
The case directory also contains the persisted canonical PRISM coordinate
contract. Each daily product records preprocessing and source-artifact
signatures; cache hits are accepted only when those signatures still match.

To run one stage manually, invoke `preproc_narr_prism.py --mode` with
`training`, `validation`, or `inference`. With `data.use_preprocessed: true`,
`compute_scalars_narr_prism.py` must run only after all configured
training products exist; it never falls back to raw data.

---

### 2. Training / Fine-Tuning

Fine-tune the downscaling model using preprocessed data.

```bash
python narr_prism_finetune.py \
    --config NARR_PRISM_subdomain.yaml \
    --num-gpus 1 \
    --save-every 5
```

- The script loads all settings from the YAML file.
- Supports single-GPU and multi-GPU (DDP / FSDP) training.
- Checkpoints are saved under `checkpoint_dir/<case_name>/`.

---

### 3. Inference

Run inference over the YAML-defined inference date range.

```bash
python narr_prism_inference.py \
    --config NARR_PRISM_subdomain.yaml \
    [--checkpoint path/to/last.ckpt]
```

Optional flags:
- `--checkpoint` — explicit path to a trained checkpoint (auto-detected
  from `inference.checkpoint_path` if set, otherwise auto-discovered from
  experiment/checkpoint directories with **`last.ckpt` preferred over `best.ckpt`**).
- `--output-dir` — override the output directory.
- `--batch-size` — inference batch size (default 1).
- `--device cpu` — force CPU inference.

**Outputs:** Daily NetCDF files under
`<inference.output_dir>/<case_name>/` containing:

| Dimension | Description |
|-----------|-------------|
| `time` | Inference dates |
| `lat` | PRISM latitude grid |
| `lon` | PRISM longitude grid |
| Variables | `ppt`, `tmax`, `tmin` (denormalized) |

---

### 4. Evaluate Inference

The shared evaluator streams prediction/truth pairs on the exact canonical
PRISM grid; it does not regrid either field.

```bash
python ../evaluate_prism_inference.py \
    --config NARR_PRISM_subdomain.yaml \
    --run-label narr_prism
```

It writes a flat metrics CSV, a provenance-rich JSON report, and a four-panel
climatology/bias/RMSE PNG. Metrics include RMSE, correlation, overlap-boundary
gradient error, native-grid boundary error, and spatial block power.

---

## Expected Outputs Summary

| Step | Output location |
|------|-----------------|
| Scalars | `<preprocessed_dir>/<case_name>/scalars/` (`.npy` + `metadata.json`) |
| Grid contract | `<preprocessed_dir>/<case_name>/prism_grid.{npz,json}` |
| Preprocessing | `<preprocessed_dir>/<case_name>/{training,validation,inference}/` |
| Training | `checkpoint_dir/<case_name>/` |
| Inference | `inference.output_dir/<case_name>/` |
| Evaluation | Evaluator `--output-dir` (CSV, JSON, and PNG) |

---

## Scripts

| File | Purpose |
|------|---------|
| `NARR_PRISM_subdomain.yaml` | Master configuration |
| `run_preprocess.sh` | Ordered, optionally sharded preprocessing/scalar launcher |
| `narr_prism_utils.py` | Shared helpers (YAML, dates, file discovery) |
| `narr_prism_dataset.py` | PyTorch Dataset class |
| `compute_scalars_narr_prism.py` | Normalization statistics |
| `preproc_narr_prism.py` | Preprocessing / regridding |
| `narr_prism_training.py` | Training loop utilities |
| `narr_prism_finetune.py` | Fine-tuning CLI entry-point |
| `narr_prism_inference.py` | Inference CLI entry-point |
| `../evaluate_prism_inference.py` | Shared exact-grid inference evaluator |

---

## Notes

- **YAML-driven dates.** Training, validation, and inference date ranges are
  never hard-coded; change them in `NARR_PRISM_subdomain.yaml`.
- **Strict artifact reuse.** Training, validation, and inference consume only
  signed daily products from the active case. Scalar manifests bind channel
  order, normalization semantics, training dates, source artifacts, and the
  exact PRISM coordinate fingerprint.
- **Checkpoint compatibility.** New checkpoints persist the coordinate-sensitive
  PRISM contract. Resume and inference reject incompatible grids, scalars,
  preprocessing signatures, crop geometry, or decoder semantics.
- **Halo-aware tiling.** Training and inference use overlapping output cores
  with predictor context halos. Windowed-local attention and blend windows keep
  tiled predictions spatially aligned while reducing seam artifacts.
- **Case isolation.** Every preprocessing artifact (scalars, normalized
  predictors/targets, cached/tiled outputs) is written under
  `<preprocessed_dir>/<case_name>/`, so preprocessing one case (or YAML)
  never overwrites another. Scalars are **never** read from a shared/flat
  directory: fine-tuning and inference resolve `<preprocessed_dir>/<case_name>/scalars/`
  (with a same-case fallback to a legacy `data.scalar_dir/<case_name>/`) and
  raise a clear error if the per-case scalars are missing. Each entry point
  logs the active `case_name` and the exact preprocessing/scalar directories.


## Temporal (sequence-conditioned) workflow

`NARR_PRISM_subdomain_temporal_{recurrent,mamba}.yaml` add explicit temporal
dependence to the `narr_prism_California` case. They are derived from
`NARR_PRISM_subdomain.yaml` and preserve:

- **all three targets in order -- `ppt`, `tmax`, `tmin`** (target variables come
  from the config, never from a filename)
- the 31 predictor channels and their ordering, including `elev` and the 16
  predictor masks, with `num_static_channels: 0` -- elevation and masks travel in
  the *dynamic* channel list, so no separate static tensor is split off
- `case_name: narr_prism_California`, which is both the case identity and what
  locates the preprocessed inputs under `preprocessed/<case_name>/<mode>/`
- the California subset bounds, the Gregorian daily calendar, `scalars_with_H`
  normalization, the 1996-2013 / 2014-2015 / 2016-2025 periods, the tiled 256-px
  training geometry with a 32-px halo, and the hurdle precipitation head

Because both `tmax` and `tmin` are configured, a physical-consistency term
penalizing `tmin > tmax` is active automatically; it contributes exactly zero for a
consistent pair.

`context_length` is 5 here rather than the SA case's 7: the PRISM crop is 256x256
against SA's 128x128, so each frame costs ~4x as much and the U-Net bottleneck is
32x32. `state.tbptt_chunk: 3` bounds backprop memory. See the config comments.

> **Not executed end-to-end.** The NARR and PRISM archives and the
> `narr_prism_California` Phase-1 checkpoint are not present on the machine this
> branch was developed on (`preprocessed/` and `experiments/` are both empty). The
> configs are schema-complete and every key in them is consumed and validated by
> real code, and `describe` reports the missing preprocessed directory explicitly,
> but no training or evaluation numbers exist for this case. Re-run the
> event-alignment diagnostic before drawing scientific conclusions.

```bash
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_temporal.py describe \
    --config examples/NARR_PRISM/NARR_PRISM_subdomain_temporal_recurrent.yaml \
    --splits train validation test
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_temporal.py check    --config <yaml>
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_temporal.py train    --config <yaml>
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_temporal.py infer    --config <yaml> --split test
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_temporal.py evaluate --config <yaml> --predictions <npz>
```

Outputs go to `runs_temporal/`, so the existing `experiments/` tree is untouched.
