# MERRA2-to-PRISM Downscaling Workflow

This directory contains a complete YAML-driven workflow for statistical
downscaling of **MERRA2** reanalysis predictors to the **PRISM** 800 m
daily observation grid over the contiguous United States.

## Artifact portal safety

The tracked `experiments`, `preprocessed`, and `scalars_with_H` entries are
relative symbolic links through the ignored per-clone `artifacts` portal. Do
not replace them with absolute links back into this source directory: such a
self-link can cause Git to remove ignored checkpoints and outputs during a
branch checkout. Configure a clone with an external root, for example:

```bash
mkdir -p /data2/granite-wxc-artifacts/MERRA_PRISM/{experiments,preprocessed,scalars_with_H}
ln -s /data2/granite-wxc-artifacts/MERRA_PRISM examples/MERRA_PRISM/artifacts
```

The tracked links must remain exactly `artifacts/experiments`,
`artifacts/preprocessed`, and `artifacts/scalars_with_H`. The local `artifacts`
portal is ignored and must never be committed.

The pipeline follows a strict, case-scoped artifact contract:

```
training preprocessing → training-only scalars → validation/inference preprocessing
→ training / fine-tuning → tiled inference → evaluation
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

## YAML Configuration (`MERRA_PRISM_subdomain.yaml`)

All paths, variables, date ranges, model parameters, and training
hyper-parameters are controlled by a single YAML file.

### Key sections

| Section | Purpose |
|---------|---------|
| `data.predictor_dir` | Path to MERRA2 daily files |
| `data.target_dir` | Path to PRISM root directory |
| `data.target_variables` | PRISM target sub-folders (`ppt`, `tmax`, `tmin`) |
| `data.predictor_variables` | MERRA2 variable names to use as predictors |
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
cd examples/MERRA_PRISM

bash run_preprocess.sh \
    --config MERRA_PRISM_subdomain.yaml \
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

To run one stage manually, invoke `preproc_merra_prism.py --mode` with
`training`, `validation`, or `inference`. With `data.use_preprocessed: true`,
`compute_scalars_merra_prism.py` must run only after all configured
training products exist; it never falls back to raw data.

---

### 2. Training / Fine-Tuning

Fine-tune the downscaling model using preprocessed data.

```bash
python merra_prism_finetune.py \
    --config MERRA_PRISM_subdomain.yaml \
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
python merra_prism_inference.py \
    --config MERRA_PRISM_subdomain.yaml \
    [--checkpoint path/to/last.ckpt]
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

### 4. Evaluate Inference

The shared evaluator streams prediction/truth pairs on the exact canonical
PRISM grid; it does not regrid either field.

```bash
python ../evaluate_prism_inference.py \
    --config MERRA_PRISM_subdomain.yaml \
    --run-label merra_prism
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
| `MERRA_PRISM_subdomain.yaml` | Master configuration |
| `run_preprocess.sh` | Ordered, optionally sharded preprocessing/scalar launcher |
| `merra_prism_utils.py` | Shared helpers (YAML, dates, file discovery) |
| `merra_prism_dataset.py` | PyTorch Dataset class |
| `compute_scalars_merra_prism.py` | Normalization statistics |
| `preproc_merra_prism.py` | Preprocessing / regridding |
| `merra_prism_training.py` | Training loop utilities |
| `merra_prism_finetune.py` | Fine-tuning CLI entry-point |
| `merra_prism_inference.py` | Inference CLI entry-point |
| `../evaluate_prism_inference.py` | Shared exact-grid inference evaluator |

---

## Notes

- **YAML-driven dates.** Training, validation, and inference date ranges are
  never hard-coded; change them in `MERRA_PRISM_subdomain.yaml`.
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
