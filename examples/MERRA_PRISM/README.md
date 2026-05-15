# MERRA2-to-PRISM Downscaling Workflow

This directory contains a complete YAML-driven workflow for statistical
downscaling of **MERRA2** reanalysis predictors to the **PRISM** 800 m
daily observation grid over the contiguous United States.

The pipeline follows the same pattern as the CORDEX_ML workflow:

```
compute scalars → preprocessing / regridding → training / fine-tuning → inference
```

---

## Required Input Data

### MERRA2 Predictors

| Item | Value |
|------|-------|
| **Directory** | `/data/merra2_daily_subset` |
| **Format** | One NetCDF file per day, filename contains `YYYYMMDD` |
| **Variables** | Configurable via `data.predictor_variables` in the YAML |
| **Default variables** | `T2M`, `U10M`, `V10M`, `QV2M`, `PS`, `SLP`, `PRECTOT`, `PRECTOTCORR`, `T500`, `T850`, `U500`, `U850`, `V500`, `V850`, `H500`, `H850` |

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

## YAML Configuration (`MERRA_PRISM.yaml`)

All paths, variables, date ranges, model parameters, and training
hyper-parameters are controlled by a single YAML file.

### Key sections

| Section | Purpose |
|---------|---------|
| `data.predictor_dir` | Path to MERRA2 daily files |
| `data.target_dir` | Path to PRISM root directory |
| `data.target_variables` | PRISM target sub-folders (`ppt`, `tmax`, `tmin`) |
| `data.predictor_variables` | MERRA2 variable names to use as predictors |
| `data.preprocessed_dir` | Where preprocessed files are written |
| `data.scalar_dir` | Where normalization statistics are saved |
| `data.scalers.*` | Paths to the four `.npy` scalar files |
| `dates.training.start / end` | Training date range |
| `dates.inference.start / end` | Inference date range |
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

### 1. Compute Scalars

Compute channel-wise mean and standard deviation over the **training period only**.

```bash
cd examples/MERRA_PRISM

python compute_scalars_merra_prism.py \
    --config MERRA_PRISM.yaml
```

Optional flags:
- `--output-dir <dir>` — override the scalar output directory.
- `--progress-interval 100` — log every N dates.

**Outputs:** `inputs_mean.npy`, `inputs_std.npy`, `targets_mean.npy`,
`targets_std.npy`, and `metadata.json` in the configured `data.scalar_dir`.

---

### 2. Preprocessing

Align MERRA2 and PRISM by date, regrid MERRA2 to the PRISM grid, and
save preprocessed NetCDF files.

```bash
python preproc_merra_prism.py \
    --config MERRA_PRISM.yaml \
    --mode both
```

`--mode` accepts `training`, `inference`, or `both` (default).

**Outputs:** One NetCDF per aligned date in `<preprocessed_dir>/training/`
and `<preprocessed_dir>/inference/`.

---

### 3. Training / Fine-Tuning

Fine-tune the downscaling model using preprocessed data.

```bash
python merra_prism_finetune.py \
    --config MERRA_PRISM.yaml \
    --num-gpus 1 \
    --save-every 5
```

- The script loads all settings from the YAML file.
- Supports single-GPU and multi-GPU (DDP / FSDP) training.
- Checkpoints and logs are saved under `path_experiment`.

---

### 4. Inference

Run inference over the YAML-defined inference date range.

```bash
python merra_prism_inference.py \
    --config MERRA_PRISM.yaml \
    [--checkpoint path/to/best.ckpt]
```

Optional flags:
- `--checkpoint` — explicit path to a trained checkpoint (auto-detected
  from `inference.checkpoint_path` or `path_experiment` if omitted).
- `--output-dir` — override the output directory.
- `--batch-size` — inference batch size (default 1).
- `--device cpu` — force CPU inference.

**Outputs:** A single NetCDF file
`<inference.output_dir>/merra_prism_inference.nc` containing:

| Dimension | Description |
|-----------|-------------|
| `time` | Inference dates |
| `lat` | PRISM latitude grid |
| `lon` | PRISM longitude grid |
| Variables | `ppt`, `tmax`, `tmin` (denormalized) |

---

## Expected Outputs Summary

| Step | Output location |
|------|-----------------|
| Scalars | `data.scalar_dir` (`.npy` + `metadata.json`) |
| Preprocessing | `data.preprocessed_dir/{training,inference}/` |
| Training | `path_experiment/checkpoints/` |
| Inference | `inference.output_dir/merra_prism_inference.nc` |

---

## Scripts

| File | Purpose |
|------|---------|
| `MERRA_PRISM.yaml` | Master configuration |
| `merra_prism_utils.py` | Shared helpers (YAML, dates, file discovery) |
| `merra_prism_dataset.py` | PyTorch Dataset class |
| `compute_scalars_merra_prism.py` | Normalization statistics |
| `preproc_merra_prism.py` | Preprocessing / regridding |
| `merra_prism_training.py` | Training loop utilities |
| `merra_prism_finetune.py` | Fine-tuning CLI entry-point |
| `merra_prism_inference.py` | Inference CLI entry-point |

---

## Notes

- **No wrapper scripts.** Unlike CORDEX_ML, this workflow targets one
  domain and one model — no `preproc_*_wrapper` scripts are needed.
- **YAML-driven dates.** Training and inference date ranges are never
  hard-coded; change them in `MERRA_PRISM.yaml`.
- **Scalar reuse.** The same scalars computed over the training period
  are used for both training normalisation and inference denormalization.
