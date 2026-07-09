# NARR-to-PRISM Downscaling Workflow

This directory contains a complete YAML-driven workflow for statistical
downscaling of **NARR** reanalysis predictors to the **PRISM** 800 m
daily observation grid over the contiguous United States.

The pipeline follows the same pattern as the CORDEX_ML workflow:

```
compute scalars → preprocessing / regridding → training / fine-tuning → inference
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

## YAML Configuration (`NARR_PRISM.yaml`)

All paths, variables, date ranges, model parameters, and training
hyper-parameters are controlled by a single YAML file.

### Key sections

| Section | Purpose |
|---------|---------|
| `data.predictor_dir` | Path to NARR monthly variable files |
| `data.target_dir` | Path to PRISM root directory |
| `data.target_variables` | PRISM target sub-folders (`ppt`, `tmax`, `tmin`) |
| `data.predictor_variables` | NARR variable names to use as predictors |
| `data.preprocessed_dir` | Where preprocessed files are written |
| `data.scalar_dir` | Where normalization statistics are saved |
| `data.scalers.*` | Paths to the four `.npy` scalar files |
| `dates.training.start / end` | Training date range |
| `dates.inference.start / end` | Inference date range |
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

### 1. Compute Scalars

Compute channel-wise mean and standard deviation over the **training period only**.

```bash
cd examples/NARR_PRISM

python compute_scalars_narr_prism.py \
    --config NARR_PRISM.yaml
```

Optional flags:
- `--output-dir <dir>` — override the scalar output directory.
- `--progress-interval 100` — log every N dates.

**Outputs:** `inputs_mean.npy`, `inputs_std.npy`, `targets_mean.npy`,
`targets_std.npy`, and `metadata.json` in the configured `data.scalar_dir`.

---

### 2. Preprocessing

Align NARR and PRISM by date, regrid NARR to the PRISM grid, and
save preprocessed NetCDF files.

```bash
python preproc_narr_prism.py \
    --config NARR_PRISM.yaml \
    --mode both
```

`--mode` accepts `training`, `inference`, or `both` (default).

**Outputs:** One NetCDF per aligned date in
`<preprocessed_dir>/training/<case_name>/` and
`<preprocessed_dir>/inference/<case_name>/`.

---

### 3. Training / Fine-Tuning

Fine-tune the downscaling model using preprocessed data.

```bash
python narr_prism_finetune.py \
    --config NARR_PRISM.yaml \
    --num-gpus 1 \
    --save-every 5
```

- The script loads all settings from the YAML file.
- Supports single-GPU and multi-GPU (DDP / FSDP) training.
- Checkpoints are saved under `checkpoint_dir/<case_name>/`.

---

### 4. Inference

Run inference over the YAML-defined inference date range.

```bash
python narr_prism_inference.py \
    --config NARR_PRISM.yaml \
    [--checkpoint path/to/best.ckpt]
```

Optional flags:
- `--checkpoint` — explicit path to a trained checkpoint (auto-detected
  from `inference.checkpoint_path` or `path_experiment` if omitted).
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

## Expected Outputs Summary

| Step | Output location |
|------|-----------------|
| Scalars | `data.scalar_dir` (`.npy` + `metadata.json`) |
| Preprocessing | `data.preprocessed_dir/{training,inference}/<case_name>/` |
| Training | `checkpoint_dir/<case_name>/` |
| Inference | `inference.output_dir/<case_name>/` |

---

## Scripts

| File | Purpose |
|------|---------|
| `NARR_PRISM.yaml` | Master configuration |
| `narr_prism_utils.py` | Shared helpers (YAML, dates, file discovery) |
| `narr_prism_dataset.py` | PyTorch Dataset class |
| `compute_scalars_narr_prism.py` | Normalization statistics |
| `preproc_narr_prism.py` | Preprocessing / regridding |
| `narr_prism_training.py` | Training loop utilities |
| `narr_prism_finetune.py` | Fine-tuning CLI entry-point |
| `narr_prism_inference.py` | Inference CLI entry-point |

---

## Notes

- **No wrapper scripts.** Unlike CORDEX_ML, this workflow targets one
  domain and one model — no `preproc_*_wrapper` scripts are needed.
- **YAML-driven dates.** Training and inference date ranges are never
  hard-coded; change them in `NARR_PRISM.yaml`.
- **Scalar reuse.** The same scalars computed over the training period
  are used for both training normalisation and inference denormalization.
