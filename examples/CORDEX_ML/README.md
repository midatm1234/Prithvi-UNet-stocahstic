# CORDEX-ML Downscaling: ALPS, NZ, and SA

## Overview
This folder hosts the CORDEX-ML benchmark workflows for multiple regional domains—European Alps (ALPS), New Zealand (NZ), and South Africa (SA)—demonstrating how to fine-tune the Prithvi WxC UNet on coarse CORDEX predictors and produce high-resolution precipitation (`pr`) and maximum temperature (`tasmax`) forecasts. The assets here reuse the helper scripts (`preproc_cordex.py`, `compute_scalars_cordex.py`, `cordex_training.py`, and the notebooks in `notebooks/`) to cover the full loop: preprocess/regrid → compute scalars → fine-tune → inference → persist predictions as NetCDF.

## Case-specific output folders (`case_name`)

Every YAML config in this folder **must** start with a `case_name:` on line 1.
`case_name` is the single source of truth for where all generated artifacts are
written and read back; it is normally the config's file stem, e.g.:

```yaml
case_name: NZ_T1_ACCESS-CM2_static_v6
data:
  ...
```

`granitewxc.utils.config.get_config()` parses and validates it. If `case_name`
is missing it raises a clear `MissingCaseNameError` asking you to add it as the
first line of the YAML.

All generated outputs for a config are organized under
`<path_experiment>/<case_name>/`:

```text
<path_experiment>/<case_name>/
├── scalars/       # inputs_mean.npy, inputs_std.npy, targets_mean.npy, targets_std.npy, metadata.json
├── preproc/       # optional regridded predictors from preproc_cordex.py (--config)
├── checkpoints/   # best.ckpt / last.ckpt written by the trainer
├── inference/     # NetCDF predictions (pr, tasmax) + diagnostics
└── logs/          # optional run logs / evaluation summaries
```

The config exposes these as properties (`config.path_scalars`,
`config.path_preproc`, `config.path_checkpoints`, `config.path_inference`,
`config.path_logs`) and repoints the scalar files
(`config.model.{input_mu,input_sigma,target_mu,target_sigma}` and
`config.data.scalers`) at `<case_name>/scalars` automatically, so training and
inference always agree on where the statistics live. In practice:

* **Scalars** — `compute_scalars_cordex.py --config <cfg.yaml>` writes to
  `<case_name>/scalars` by default; `--output-dir` is now optional (override only).
* **Checkpoints** — the trainer defaults to `<case_name>/checkpoints` when
  `config.checkpoint_dir` is not set explicitly.
* **Inference** — the inference scripts locate the fine-tune run by `case_name`
  and write predictions under the case-named run directory (never a hard-coded path).

To opt out of repointing the scalar file paths (for example to reuse a shared
statistics folder) set `derive_output_paths: false` in the YAML; the
`config.path_*` helpers still resolve to the case sub-folders.

## Diffusion decoder head (optional)

For the residual-refinement design, exact configuration defaults, checkpoint
compatibility, tensor dimensions, and current limitations, see
[`DIFFUSION_RESIDUAL_CORRECTION.md`](./DIFFUSION_RESIDUAL_CORRECTION.md).

By default the CORDEX-ML model uses the deterministic convolutional / UNet
decoder head. An **optional score-based diffusion head** (adapted from
[`mlde`](https://github.com/midatm1234/mlde)) can be selected from config to turn
the model into a *generative* downscaler that samples high-resolution `pr` and
`tasmax` fields conditioned on the Prithvi WxC backbone features. The
deterministic workflow is untouched — the diffusion head activates only when the
config asks for it.

### How to enable it

Add a `head_type: diffusion` key under `model:` (the alias
`model.decoder_type: diffusion` also works) plus an optional `model.diffusion:`
block of hyper-parameters. Ready-to-use examples are
[`NZ_T1_ACCESS-CM2_static_diffusion.yaml`](./NZ_T1_ACCESS-CM2_static_diffusion.yaml)
and [`SA_T2_ACCESS-CM2_static_diffusion.yaml`](./SA_T2_ACCESS-CM2_static_diffusion.yaml):

```yaml
model:
  # ... all existing deterministic keys are unchanged ...
  head_type: diffusion          # <-- the only required switch
  diffusion:
    sde: subvpsde               # vpsde | subvpsde | vesde
    beta_min: 0.1               # VP/subVP noise schedule
    beta_max: 20.0
    sigma_min: 0.01             # VE noise schedule
    sigma_max: 50.0
    num_scales: 1000            # (a.k.a. num_diffusion_steps)
    sampling_method: pc         # pc (predictor-corrector) | ode
    predictor: euler_maruyama   # euler_maruyama | reverse_diffusion | none
    corrector: none             # langevin | none
    num_sampling_steps: 128     # reverse steps used at inference
    base_channels: 64           # score-network width
    channel_multipliers: [1, 2, 2]
    output_channels: 2          # pr + tasmax (inferred from the model too)
```

To go back to the deterministic head, delete `head_type` (or set it to
`deterministic`). No other change is needed.

> **Note:** the hurdle precipitation model (`precip_model: hurdle`) is not
> compatible with the diffusion head, which models `pr`/`tasmax` jointly. Use
> `precip_model: single_head` (as in the example config). Requesting the hurdle
> model together with the diffusion head raises a clear error at model build
> time.

### How training differs

* **Deterministic head** minimizes a regression loss (RMSE / composite predictand
  loss) between the decoded prediction and the target.
* **Diffusion head** minimizes a **score-matching (denoising) loss**. The model's
  `forward(batch)` returns the scalar diffusion loss directly, and
  `build_loss_fn` returns a thin passthrough so the existing trainer loop
  (`prediction = model(batch); loss = loss_func(prediction, batch)`) is unchanged.
  The loss is computed in **standardized target space** (the same scaling that the
  deterministic decoder inverts), so the existing scalers and per-variable scaling
  methods (`zscore` / `divide_only` / `log1p`) are reused.

Training is launched exactly like the deterministic workflow — just point it at
the diffusion config:

```bash
python examples/CORDEX_ML/cordex_training.py \
    --config examples/CORDEX_ML/NZ_T1_ACCESS-CM2_static_diffusion.yaml
```

Both training and validation call `model(batch)` without return flags and
therefore compute the (cheap) score-matching loss — validation does **not** run
the expensive sampler.

### How inference sampling works

Inference is unchanged at the call site: the inference helpers call the model
with `return_pre_inverse=True, return_raw_output=True`. In diffusion mode this
triggers the **reverse-diffusion sampler** instead of a single forward pass:

1. The backbone produces conditioning features from the coarse predictors
   (+ optional static/orography fields).
2. Gaussian noise at the target resolution is denoised over
   `num_sampling_steps` reverse SDE steps (predictor–corrector or ODE),
   conditioned on those features.
3. The standardized sample is inverse-scaled back to physical `pr`/`tasmax`,
   including **precipitation non-negativity** (configured non-negative channels
   are clamped at 0 after decoding).

The sampler returns one stochastic tensor of shape
`[B, output_channels, H_target, W_target]` (i.e.
`[B, 2, target_size_lat, target_size_lon]`). The inference workflow detects
diffusion checkpoints and runs the sampler repeatedly to build an ensemble.
Each member uses an independent, reproducible seed:

```text
ensemble_seed = base_seed + ensemble_index
```

### Using the existing notebooks and scripts

The diffusion head plugs into the **existing** CORDEX-ML finetune and inference
workflows — no Python edits are required, only a config switch (and, for
inference, a diffusion-trained checkpoint):

**Finetuning** (`cordex_training.py` and the `notebooks/*finetune*` notebooks):

* Point the run at a diffusion config. In the notebooks this is the
  `config_path` in the *USER PARAMETERS* cell (e.g. change
  `SA_T2_ACCESS-CM2_static_v6.yaml` → `SA_T2_ACCESS-CM2_static_diffusion.yaml`);
  for the CLI pass `--config .../<name>_diffusion.yaml`.
* The model is built through `create_finetune_model` → `get_finetune_model_UNET`,
  which constructs the diffusion head automatically when `head_type: diffusion`.
  `build_loss_fn` returns the diffusion loss passthrough, and the trainer loop
  (`model(batch)` → scalar loss) is unchanged. The notebooks' default
  `precip_model` fallback keeps the config's `single_head` setting.

**Inference** (`notebooks/*inference*` scripts/notebooks):

* Set `CONFIG_PATH` to the diffusion config so the script locates the matching
  training-run directory. The script reloads the run's resolved-config snapshot,
  so `head_type: diffusion` carries through and `get_finetune_model_UNET` rebuilds
  the diffusion head.
* **Use a checkpoint from a diffusion finetune run.** Inference validates weights
  strictly (missing/unexpected keys raise). A deterministic checkpoint has no
  `diffusion_head.*` weights and will fail with a clear mismatch error — this is
  expected, not a bug.
* **Tiling/boundary blending:** if you enable `inference.boundary_mitigation`
  with tiles, the sampler runs per tile with independent noise, producing seams.
  Prefer a single full-frame sampler call (the default when no `inference:` block
  is present, or set `boundary_mitigation.force_full_frame: true`).
* **Cost & stochasticity:** each ensemble member runs `num_sampling_steps`
  reverse steps. Lower `num_sampling_steps` to trade quality for speed.

Example inference settings:

```yaml
inference:
  ensemble_size: 10
  base_seed: 42
```

`ensemble_size` defaults to 1 when omitted. An explicit positive value is used
as configured; diffusion inference does not impose a minimum ensemble of 10.
Deterministic convolutional-head checkpoints keep the previous one-member
deterministic behavior unless an explicit downstream workflow adds its own
ensemble support.

### Expected output locations and shapes

* Output NetCDF files are written to the same case-name-based locations as the
  deterministic workflow.
* Deterministic checkpoints keep variables shaped `[time, lat, lon]`.
* Diffusion checkpoints write `pr` and `tasmax` with dimensions
  `[time, ensemble, lat, lon]`, plus an `ensemble = 0..ensemble_size-1`
  coordinate and global attrs including `head_type=diffusion`,
  `ensemble_size`, `ensemble_generation=diffusion_sampling`, and `base_seed`.
* Postprocessing that needs a single field should explicitly call
  `ensemble_mean_xr()` from `examples/CORDEX_ML/utils/postprocess_outputs.py`;
  the workflow does not silently squeeze or drop the ensemble dimension.

### Smoke test

A dummy-tensor smoke test (no data or GPU required) verifies the deterministic
head still imports, the diffusion training loss is finite, and the sampler
returns the expected shape:

```bash
python examples/CORDEX_ML/diffusion_smoke_test.py
```

## v6 Block Artifact Root-Cause Fixes

The remaining coarse block patterns in `pr` and `tasmax` are now addressed in the training/inference pipeline itself, not by cosmetic post-smoothing.

Primary root causes found and fixed:

1. **Grid mismatch in v6 configs**: several `*_v6.yaml` files used `target_size=256` while v6 scalers are `128x128`, causing resizing and normalization inconsistencies that amplify block structure.
2. **Decoder reconstruction path instability**: UNET decode flow could create coarse artifacts from upsample/downsample mismatch; decode path now aligns bottleneck-to-skip scales correctly and supports stable interpolation-based decoder upsampling.
3. **Inference reconstruction sensitivity**: tiling behavior now supports explicit overlap + weighted blending (`hann`/`cosine`/`gaussian`/`linear`) with configurable overlap/weights, and optional full-frame inference bypass.
4. **Training alignment bias**: when crop size equals full field, random crop is disabled; v6 now supports `data.train_random_crop_offset` to randomize spatial alignment without shrinking the crop.
5. **Scaler resize fallback**: output scaler interpolation for mismatched grids now supports smooth interpolation modes (default `bilinear`) instead of nearest-only fallback.

### Primary controls (recommended)

Use these first to reduce block boundaries while preserving physical gradients:

- `data.target_size_lat/lon` and `train_crop_size_lat/lon` must match scaler grid.
- `data.train_random_crop_offset`: small random shift (for example `[8, 8]`).
- `inference.boundary_mitigation.overlap` + blend window (`hann` by default).
- `model.decoder_upsampling_mode: bilinear` (or `nearest`) instead of artifact-prone decode settings.
- Optional low-weight structure-aware losses:
  - `loss.spatial_gradient`
  - `loss.multiscale`
  - `loss.boundary_continuity`

### Fallback control (not primary)

- `use_seam_deblocking` / `seam_deblocking_strength` remains available, but should only be used as a conservative fallback after overlap blending and reconstruction settings are tuned.

### New/updated v6 YAML knobs

```yaml
data:
  train_random_crop_offset: [8, 8]

model:
  decoder_upsampling_mode: bilinear
  output_scaler_resize_mode: bilinear
  output_scaler_align_corners: false

inference:
  inference_tile_size: [96, 96]
  inference_overlap: [32, 32]
  inference_blend_window: hann
  inference_blend_sigma: 0.35
  use_seam_deblocking: false
  seam_deblocking_strength: 0.1
  boundary_mitigation:
    enabled: true
    tile_size: [96, 96]
    overlap: [32, 32]
    blend_window: hann
    blend_sigma: 0.35
    deblock:
      enabled: false

loss:
  use_gradient_loss: true
  gradient_loss_weight: 0.03
  use_tv_loss: false
  tv_loss_weight: 0.0005
  use_multiscale_loss: true
  multiscale_loss_weight: 0.015
  boundary_continuity:
    enabled: true
    weight: 0.01
    predictands: {pr: 0.2, tasmax: 1.0}
    stride_lat: 16
    stride_lon: 16
    match_target: true
```

### Artifact diagnosis utility

Use the new diagnostic script to separate stitching artifacts from model artifacts:

```bash
python examples/CORDEX_ML/utils/inference_artifact_diagnostics.py \
  --config examples/CORDEX_ML/NZ_T1_ACCESS-CM2_static_v6.yaml \
  --checkpoint <path-to-checkpoint> \
  --num-samples 16 \
  --output-dir examples/CORDEX_ML/evaluations/artifact_diag_nz_t1
```

This runs no-overlap and overlap-blended reconstructions, writes comparison figures/difference maps, and reports seam-aligned metrics so you can verify whether artifacts are dominated by stitching, decoder behavior, or training/crop strategy.

## CORDEX v4 updates (new default path)

The repository now supports a reproducible v4 CORDEX path with updated normalization, optional distribution-aware losses, and multi-GPU execution. The v4 YAMLs (`*_v4.yaml`) are the recommended defaults for new training and inference runs.

### 1) Gridpoint-wise target normalization

v4 supports per-predictand normalization settings under `predictands.<name>.normalization`:

- `method`: `standardize` or `log1p_standardize`
- `mode`: `global` or `gridpoint`
- `eps_std`: numerical floor for small std values

For `mode: gridpoint`, target statistics are computed at each `(lat, lon)` grid point over time and saved in `targets_mean.npy` / `targets_std.npy` with shape `[C, H, W]` (instead of `[C]`).

Why this helps: climate fields are spatially heterogeneous, so a single global mean/std can over-normalize some regions and under-normalize others; gridpoint-wise stats preserve local climatology and improve conditioning.

### 2) `pr` log1p normalization and inverse transform

For precipitation in v4, use:

```yaml
predictands:
  pr:
    normalization:
      method: log1p_standardize
      mode: gridpoint
      eps_std: 1.0e-6
      allow_negative_value: false
  tasmax:
    normalization:
      method: standardize
      mode: gridpoint
      eps_std: 1.0e-6
```

Transform and inverse are:

- forward: `log1p(max(pr, 0))`, then standardize
- inverse: de-standardize, then `expm1`, then clamp tiny negative roundoff to zero

This keeps `pr=0` valid and yields non-negative denormalized precipitation.

### 3) Optional distribution-aware losses

Loss is now configurable with a base RMSE plus optional per-predictand distribution penalties:

- `moment`: mean/std/skewness matching
- `quantile`: selected quantile matching
- `cdf`: soft CDF matching

Example:

```yaml
loss:
  base: rmse
  predictands:
    tasmax:
      distribution_loss:
        enabled: true
        method: moment
        weight: 0.05
        use_mean: true
        use_std: true
        use_skewness: false
    pr:
      distribution_loss:
        enabled: false
```

Recommended first setting: `tasmax` with `moment` loss and a small weight (for example `0.01` to `0.05`).

### 4) Block-boundary artifact mitigation policy

v4/v6 treat seam deblocking as fallback only. Primary controls target root causes in tiled reconstruction:

- overlap size
- blend window (`hann`, `cosine`, `gaussian`, or `uniform`)
- tile size / stride consistency
- scaler alignment per tile (`__scaler_offset`) for gridpoint denormalization

Optional seam deblock is boundary-local and conservative:

```yaml
inference:
  inference_tile_size: [96, 96]
  inference_overlap: [32, 32]
  inference_blend_window: hann
  inference_blend_sigma: 0.35
  boundary_mitigation:
    enabled: true
    tile_size: [96, 96]
    overlap: [32, 32]
    blend_window: hann
    blend_sigma: 0.35
    deblock:
      enabled: false
      boundary_width: 2
      strength: 0.1
      kernel_size: 3
```

### 4b) Training-side artifact suppression (recommended for v6 retrains)

In addition to inference stitching controls, v6 now supports training knobs that directly
reduce checkerboard/block boundaries and over-strong terrain imprint:

- `loss.spatial_gradient`: matches output spatial gradients to target gradients.
- `model.static_embedding_scale`: scales static-terrain embedding contribution.
- `model.static_skip_scale` (UNET): scales static skip-path contribution.
- `model.static_dropout_p`: randomly drops static channels during training.
- `data.train_random_crop_offset`: applies random spatial shifts even when crop size equals full grid.

Example:

```yaml
data:
  target_size_lat: 128
  target_size_lon: 128
  train_crop_size_lat: 128
  train_crop_size_lon: 128
  train_random_crop_offset: [8, 8]

model:
  static_embedding_scale: 0.6
  static_skip_scale: 0.5
  static_dropout_p: 0.1

loss:
  base: rmse
  spatial_gradient:
    enabled: true
    weight: 0.03
    predictands:
      tasmax: 1.0
      pr: 0.25
    precipitation_wet_only: true
    precipitation_wet_threshold: 0.1
```

Practical starting point:
- First tune `static_embedding_scale/static_skip_scale` in `[0.4, 0.8]` and `static_dropout_p` in `[0.05, 0.15]`.
- Then add `loss.spatial_gradient.weight` in `[0.01, 0.05]`.
- Keep validation RMSE as the primary early-stop metric.

### 5) Multi-GPU fine-tuning and effective batch size

Fine-tuning supports DDP/FSDP execution and gradient accumulation. The effective batch size is:

`effective_batch_size = per_device_batch_size * world_size * gradient_accumulation_steps`

YAML example:

```yaml
training:
  distributed: ddp
  num_gpus: 4
  per_device_batch_size: 8
  gradient_accumulation_steps: 4
```

Launch on selected GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  examples/CORDEX_ML/cordex_finetune.py \
  --config examples/CORDEX_ML/NZ_T1_ACCESS-CM2_static_v4.yaml
```

Training logs print per-device batch size, world size, accumulation steps, and effective batch size.

### 6) Multi-GPU inference

Inference scripts support one process per GPU via `torchrun`. Work is sharded by run index to increase throughput without spatially splitting individual fields.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  examples/CORDEX_ML/notebooks/NZ_downscaling_inference_T1_ACCESS-CM2_static.py
```

This avoids duplicate predictions across ranks and preserves deterministic output writing per run index.

### 7) v4 config and run backups

- All CORDEX YAMLs have `*_v4.yaml` copies.
- Existing runs/checkpoints/metadata were copied to sibling `runs_v4` directories.
- Prediction outputs are intentionally excluded from `runs_v4` backups.

### 8) Compatibility notes

- Old configs remain supported (`predictands.<name>.scaling` still works).
- Old checkpoints can be loaded with v4 configs; scaler tensors from checkpoints are ignored so current config/scalar files remain authoritative.
- Scientific comparability: models trained with legacy global normalization are not directly equivalent to v4 gridpoint normalization. Compare metrics side-by-side before drawing conclusions.

### 9) Reproducible validation commands

```bash
cd /mnt/data2/kyo/granite-wxc

# Syntax
python -m py_compile $(find examples/CORDEX_ML -name "*.py" -type f)

# YAML parse
python - <<'PY'
from pathlib import Path
import yaml
for p in Path("examples/CORDEX_ML").rglob("*.yaml"):
    with open(p) as f:
        yaml.safe_load(f)
print("YAML validation passed")
PY

# v4 setup validation (writes JSON summary)
python examples/CORDEX_ML/utils/validate_v4_setup.py \
  --output-json examples/CORDEX_ML/evaluations/v4_validation_summary.json

# Baseline vs v4 metric comparison (requires numpy/torch runtime)
python examples/CORDEX_ML/utils/evaluate_v4_outputs.py \
  --baseline-config examples/CORDEX_ML/NZ_T1_ACCESS-CM2_static.yaml \
  --v4-config examples/CORDEX_ML/NZ_T1_ACCESS-CM2_static_v4.yaml \
  --checkpoint examples/CORDEX_ML/runs_v4/NZ_T1_ACCESS-CM2_static_train/NZ_T1_ACCESS-CM2_static/checkpoints/best.ckpt \
  --num-samples 64 \
  --batch-size 1 \
  --output-json examples/CORDEX_ML/evaluations/NZ_T1_static_v4_eval.json
```

## Data (Zenodo)
- **Source**: [CORDEX-ML Benchmark](https://zenodo.org/records/17517423) provides data for multiple domains. The Zenodo record includes coarse predictors, high-resolution targets, and static fields for ALPS, NZ, and SA regions.
- **Inputs**: CORDEX coarse predictors (multi-level dynamics plus static orography) stored under the benchmark's domain-specific `predictors` folders (e.g., `CORDEX/ALPS_domain/`, `CORDEX/NZ_domain/`, `CORDEX/SA_domain/`).
- **Targets**: Downscaled `tasmax` and `pr` fields packaged alongside the predictors in the benchmark `target` NetCDF files for each domain.
- **Setup**: Download the needed predictor/target tiles for your region of interest plus `Static_fields.nc`, then update the YAML/config paths (e.g., `ALPS_T1_*.yaml`, `NZ_T1_*.yaml`, `SA_T1_*.yaml`) to match your local layout.

### End-to-end workflow (explicit files)
1. **Normalization (compute_scalars.py)** – Run `compute_scalars.py` (this repo: `compute_scalars_cordex.py`) to compute predictor stats plus per-predictand target scalars for `{Region}` (for example, `zscore` mean/std for `tasmax`, and `divide_only` scale for `pr`).
2. **Regridding (preproc_cordex.py)** – Use `preproc_cordex.py` to regrid coarse `{Region}` predictors; `*_wrapper` scripts regrid multiple predictor files to the target high-res grids in batch.
3. **Prepare YAML configs** – Create `{Region}_T1` (or `{Region}_T2`) config files:
  - `{Region}_T1(or T2)_{ModelName}_{static|no_static}.yaml`
  - **T1**: `ESD_pseudo_reality`
  - **T2**: `Emulator_hist_future`
  - **static/no_static**: with or without orography
4. **Fine-tune** – Run the domain-specific `{Region}` notebooks:
  - `notebooks/{Region}_downscaling_finetune_T1(T2)_{ModelName}_{static|no_static}.ipynb`
5. **Inference** – Run the `{Region}` inference drivers:
  - `notebooks/{Region}_downscaling_inference_T1(T2)_{ModelName}_{static|no_static}.py`
  - Run 12 predictor configurations (perfect/imperfect; two models).
  - Time periods: hist (1981–2000), mid (2041–2060), end (2080–2099).

## Notebooks
### Regional Downscaling Notebooks
The repository provides domain-specific notebooks for each region:

#### ALPS (European Alps)
- **`ALPS_downscaling_finetune.ipynb`** – Fine-tune on the ALPS training split starting from a generic checkpoint.
  - **Inputs/targets**: Regridded CORDEX predictors from `granite-geospatial-wxc-downscaling/CORDEX/ALPS_domain/train/ESD_pseudo_reality/predictors/*_regridded.nc` and targets from `.../target/pr_tasmax_*.nc`.
  - **Outputs**: Run directory under `examples/CORDEX_ML/runs/alps_finetune/<run_name>/` with checkpoints, manifest, scalars, and logs.
- **`ALPS_downscaling_inference.ipynb`** – Load a fine-tuned checkpoint and generate downscaled forecasts on the ALPS test split.
  - **Data loader paths**: Configure `predictor_root` to `granite-geospatial-wxc-downscaling/CORDEX/ALPS_domain/test/historical_perfect/predictors`.
  - **Outputs**: Writes `predictions/*.nc` files containing `pr` and `tasmax` on the high-resolution ALPS domain.

#### New Zealand (NZ)
- **`NZ_downscaling_finetune.ipynb`** – Walk through fine-tuning Prithvi WxC on the CORDEX-ML NZ training split, starting from a generic checkpoint and adapting it to the tasmax/pr targets.
  - **Inputs/targets**: Points the dataloader to the regridded CORDEX predictors (`granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/train/ESD_pseudo_reality/predictors/*_regridded.nc`) and the paired high-resolution targets under `.../target/pr_tasmax_*.nc`.
  - **Key knobs**: `cordex_config_small.yaml` exposes GPU count, crop size, pressure levels, FSDP toggle, learning rate, and scaler paths; the notebook highlights which fields to edit for smaller runs.
  - **Outputs**: Produces a run directory under `examples/CORDEX_ML/runs/nz_finetune/<run_name>/` with checkpoints (`best.ckpt`, `last.ckpt`), run manifest, scalars copy, and logs ready for inference.

- **`NZ_downscaling_inference.ipynb`** – Load a fine-tuned checkpoint and generate downscaled forecasts on the Zenodo test split.
  - **Data loader paths**: Configure `predictor_root` to the benchmark "perfect predictors" folder (e.g., `granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/test/historical_perfect/predictors`) and reuse the manifest from the fine-tune run to locate scalars and checkpoints.
  - **Outputs**: Writes `predictions/*.nc` and matching `.pkl` files under the fine-tune run directory; the NetCDFs contain `pr` and `tasmax` grids on the high-resolution NZ domain.

#### South Africa (SA)
- **`SA_downscaling_finetune.ipynb`** – Fine-tune on the SA training split starting from a generic checkpoint.
  - **Inputs/targets**: Regridded CORDEX predictors from `granite-geospatial-wxc-downscaling/CORDEX/SA_domain/train/ESD_pseudo_reality/predictors/*_regridded.nc` and targets from `.../target/pr_tasmax_*.nc`.
  - **Outputs**: Run directory under `examples/CORDEX_ML/runs/sa_finetune/<run_name>/` with checkpoints, manifest, scalars, and logs.
- **`SA_downscaling_inference.ipynb`** – Load a fine-tuned checkpoint and generate downscaled forecasts on the SA test split.
  - **Data loader paths**: Configure `predictor_root` to `granite-geospatial-wxc-downscaling/CORDEX/SA_domain/test/historical_perfect/predictors`.
  - **Outputs**: Writes `predictions/*.nc` files containing `pr` and `tasmax` on the high-resolution SA domain.

## Quickstart
```bash
# Environment (Conda shown, pip/uv work too)
conda create -n cordex-ml python=3.10
conda activate cordex-ml
pip install '.[examples]'   # installs Prithvi WxC + notebook deps
pip install cartopy         # optional, only needed for map plots

# Download CORDEX-ML benchmark data from Zenodo and unpack to ./granite-geospatial-wxc-downscaling/CORDEX/{ALPS|NZ|SA}_domain

# Example for NZ domain: Regrid coarse predictors to the high-res target grid
python examples/CORDEX_ML/preproc_cordex.py \
  --predictor-files ./granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/train/ESD_pseudo_reality/predictors/ACCESS-CM2_1961-1980.nc \
  --target-sample ./granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/train/ESD_pseudo_reality/target/pr_tasmax_ACCESS-CM2_1961-1980.nc \
  --orography-file ./granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/train/ESD_pseudo_reality/predictors/Static_fields.nc \
  --output-dir ./granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/train/ESD_pseudo_reality/predictors

# Or for ALPS domain:
# python examples/CORDEX_ML/preproc_cordex.py \
#   --predictor-files ./granite-geospatial-wxc-downscaling/CORDEX/ALPS_domain/train/ESD_pseudo_reality/predictors/CNRM-CM5_1961-1980.nc \
#   --target-sample ./granite-geospatial-wxc-downscaling/CORDEX/ALPS_domain/train/ESD_pseudo_reality/target/pr_tasmax_CNRM-CM5_1961-1980.nc \
#   --orography-file ./granite-geospatial-wxc-downscaling/CORDEX/ALPS_domain/train/ESD_pseudo_reality/predictors/Static_fields.nc \
#   --output-dir ./granite-geospatial-wxc-downscaling/CORDEX/ALPS_domain/train/ESD_pseudo_reality/predictors

# Compute normalization scalars for tasmax/pr fine-tuning (static/orography included).
# Scalars are written to <path_experiment>/<case_name>/scalars automatically
# (case_name is read from line 1 of the YAML); pass --output-dir only to override.
python examples/CORDEX_ML/compute_scalars_cordex.py \
  --config ./examples/CORDEX_ML/NZ_T1_ACCESS-CM2_static.yaml \
  --predictor-files ./granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/train/ESD_pseudo_reality/predictors/*_regridded.nc \
  --target-files ./granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/train/ESD_pseudo_reality/target/pr_tasmax_*.nc \
  --use-static \
  --orography-file ./granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/train/ESD_pseudo_reality/predictors/Static_fields.nc

# For no-static scalars, drop orography (still writes under <case_name>/scalars):
# python examples/CORDEX_ML/compute_scalars_cordex.py \
#   --config ./examples/CORDEX_ML/NZ_T1_ACCESS-CM2_no_static.yaml \
#   --predictor-files ./granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/train/ESD_pseudo_reality/predictors/*_regridded.nc \
#   --target-files ./granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/train/ESD_pseudo_reality/target/pr_tasmax_*.nc \
#   --no-static

# Launch notebooks (Jupyter or VS Code works)
# For NZ domain:
jupyter lab examples/CORDEX_ML/notebooks/NZ_downscaling_finetune.ipynb
jupyter lab examples/CORDEX_ML/notebooks/NZ_downscaling_inference.ipynb

# For ALPS or SA domains, similarly use ALPS_downscaling_*.ipynb or SA_downscaling_*.ipynb
```
> Tip: set `distributed_strategy: fsdp` in the YAML when fine-tuning on multiple GPUs; switch `device_target: cpu` when only CPUs are available.
> Tip: use the provided wrapper scripts (`examples/CORDEX_ML/ALPS_preprocess`, `NZ_preprocess`, `SA_preprocess`) for batch pre-processing and scalar generation; they pass `--config` to `compute_scalars_cordex.py` so scalars land in each case's `<case_name>/scalars` folder automatically.
> Tip: if scalar logs print `[predictands] no --config provided; using defaults (pr -> divide_only + p95, others -> zscore)`, your run is not using the intended YAML predictand settings.

## I/O Variables
- **Predictor channels**: CORDEX coarse atmospheric fields referenced in the domain-specific YAML config files (`u`, `v`, `q`, `t`, `z` across multiple pressure levels) plus static orography.
- **Target variables**: Consistent across all domains (ALPS, NZ, SA):
  - `pr`: precipitation rate (mm/day in the benchmark files).
  - `tasmax`: daily max 2 m air temperature (°C).

Domain-specific notebooks expect predictors/targets to follow the CORDEX-ML benchmark naming convention and will emit NetCDFs with these same `pr` and `tasmax` variables.

## Model Loading
- Each domain's inference notebook enumerates the fine-tune runs under the domain-specific directory (e.g., `examples/CORDEX_ML/runs/nz_finetune` for NZ, `examples/CORDEX_ML/runs/alps_finetune` for ALPS, `examples/CORDEX_ML/runs/sa_finetune` for SA), picks the latest checkpoint (or one set via environment variable), and stores the resolved path in `CHECKPOINT_PATH`.
- When `CHECKPOINT_PATH` is loaded, the notebook prints the location, trainable parameter count, and a quick checksum (mean/std of the first tensor) so you can confirm the fine-tuned weights—not the generic base encoder—are active before running inference.

## Outputs
- Domain-specific notebooks assert that `config.data.output_vars` provides `['pr', 'tasmax']` and that the dataloaders emit two-channel targets before training/inference begins.
- Generated NetCDFs therefore always include `pr` and `tasmax` variables on each domain's high-resolution grid, matching the CORDEX-ML benchmark naming and order.

## Time Handling
- The inference writer inspects the predictor NetCDF(s) used for inference to build the `time` coordinate and raises if the predicted sequence length differs from the predictor timestamps.
- The resulting NetCDF inherits the predictor time metadata verbatim, so e.g. `ACCESS-CM2_1981-2000_regridded.nc` yields a 1981–2000 `time` axis rather than reusing the training (1961–1980) window.
- The notebook logs the detected start/end times to make it obvious which forcing period produced a set of predictions.

## Inference Safety Defaults
- All `*_downscaling_inference_*` scripts now automatically repair invalid predictor values (`NaN`, `inf`, and `_FillValue`/`missing_value`) by nearest-valid replacement in lat/lon space before dataloader normalization.
- `pr` non-negativity is now enforced by model decoding (`softplus` + divide-only scaling) when configured via YAML `predictands.pr`.
- No clamp-based precipitation post-processing is required.
- The same behavior is enabled in `*_downscaling_inference_*.ipynb` notebooks.
- Inference runs print diagnostics for:
  - Invalid predictor counts before/after repair (per predictor variable).
  - Non-negativity checks (min/max) for predictands with `nonnegativity.enabled: true`.
  - A failed non-negativity assertion now indicates a configuration/decoding bug (rather than being silently clipped).

## Tasmax Quantization Note
- **Root cause found**: inference was using CUDA autocast (`fp16/bf16`) by default and writing outputs without explicit float encoding. For `tasmax` near ~280–320 K, half precision introduces coarse increments that can appear as histogram spikes.
- **What changed**:
  - Added stage diagnostics in all `*_downscaling_inference_*.py` scripts for:
    - raw predictors (pre-normalization)
    - normalized predictors (fed to model)
    - model outputs before inverse scaling (reconstructed)
    - inverse-transformed outputs (`pr`, `tasmax`)
  - Added quantization detector (`5` timesteps × `20x20` spatial subset), compact stats table printing, and per-run diagnostics JSON export.
  - Added distribution plot export (`target`, `predicted`, `predicted pre-inverse`) with histograms and Q-Q plots for `tasmax` and `pr`.
  - Set inference defaults to keep output continuous:
    - `ENABLE_MIXED_PRECISION = False`
    - `FORCE_OUTPUT_FLOAT32 = True`
    - NetCDF write encoding uses explicit floating point dtype (`float32`) and no integer packing.
- **How to verify**:
  1. Run one inference script with `NUM_RUNS = 1` (or a single small split/file).
  2. Check printed diagnostic table in the script/notebook output (`approx_min_nonzero_step` and `approx_unique_count`).
  3. Inspect generated artifacts next to the prediction file:
     - `*.diagnostics.json`
     - `*.distribution.png`
  4. Confirm `tasmax` `approx_min_nonzero_step` is no longer close to fixed coarse steps (`0.25`, `0.5`, `1.0`) unless the model itself truly learns discrete modes.

### Notebook Cell (Distribution Check)
```python
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

diag_path = Path(".../Predictions_pr_tasmax_*.diagnostics.json")
pred_path = Path(".../Predictions_pr_tasmax_*.nc")
pred_pre_inverse_path = Path(".../Predictions_pr_tasmax_*.pre_inverse.pkl")

diag = json.loads(diag_path.read_text())
print(json.dumps(diag["quantization_detector"]["inverse_outputs"]["tasmax"], indent=2))

with open(pred_pre_inverse_path, "rb") as f:
    pred_pre_inverse = pickle.load(f)  # shape: [time, channel, lat, lon]

ds_pred = xr.open_dataset(pred_path)
target_path = Path(diag["target_template_paths"][0])
ds_target = xr.open_dataset(target_path)

pred_tasmax = ds_pred["tasmax"].values.ravel()
pred_tasmax_raw = pred_pre_inverse[:, 1].ravel()  # assumes [pr, tasmax]
n_time = ds_pred.sizes["time"]
tgt_tasmax = ds_target["tasmax"].isel(time=slice(0, n_time)).values.ravel()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(tgt_tasmax, bins=100, alpha=0.4, density=True, label="target tasmax")
axes[0].hist(pred_tasmax, bins=100, alpha=0.4, density=True, label="pred tasmax")
axes[0].hist(pred_tasmax_raw, bins=100, alpha=0.3, density=True, label="pred tasmax (pre-inverse)")
axes[0].legend()
axes[0].set_title("tasmax distribution")

q = np.linspace(0.01, 0.99, 199)
axes[1].scatter(np.quantile(tgt_tasmax, q), np.quantile(pred_tasmax, q), s=6, alpha=0.5)
mn = min(tgt_tasmax.min(), pred_tasmax.min())
mx = max(tgt_tasmax.max(), pred_tasmax.max())
axes[1].plot([mn, mx], [mn, mx], "k--", lw=1)
axes[1].set_title("tasmax Q-Q (pred vs target)")
axes[1].set_xlabel("target quantiles")
axes[1].set_ylabel("pred quantiles")
plt.tight_layout()
```

## Model Information

### Foundation Model
- **Prithvi WxC**: A vision transformer (ViT) based foundation model designed for weather and climate downscaling tasks.
- **Reference**: [Prithvi WxC arXiv paper](https://arxiv.org/abs/2409.13598)
- **Model Repository**: [NASA-IMPACT/Prithvi-WxC](https://github.com/NASA-IMPACT/Prithvi-WxC)
- **CORDEX downscaling scripts**: https://github.com/midatm1234/granite-wxc/tree/CORDEX_ML 

### Research Team
- **Key Contributor**: Hugo Kyo Lee (huikyo.lee@jpl.nasa.gov)

### Model Architecture
The downscaling framework uses a fine-tuned Prithvi WxC encoder as the backbone with domain-specific decoder heads for CORDEX downscaling:

**Encoder (Prithvi WxC Backbone)**:
- **Type**: Vision Transformer (ViT) encoder
- **Embedding dimension** (`embed_dim`): 1024
- **Number of transformer blocks** (`n_blocks_encoder`): 8
- **Attention heads** (`n_heads`): 16
- **MLP multiplier** (`mlp_multiplier`): 4
- **Dropout rate**: 0.0
- **Drop path**: 0.0

**Decoder (Convolutional Head for CORDEX)**:
- **Type**: Convolutional decoder for upsampling transformer latent features
- **Downscaling patch size**: [2, 2]
- **Downscaling embed dimension**: 512
- **Encoder-decoder type**: Convolutional
- **Upsampling mode**: Pixel shuffle
- **Conv channels**: 128
- **Kernel sizes per stage**: [[3], [3]]
- **Upscaling factors per stage**: [[2], [3]] (total upscaling: 6×)
- **Residual connection type**: Channel-wise (applied to output of the decoder stages)

The architecture enables coarse CORDEX predictors (~16×16 patches) to be upscaled to high-resolution targets (~256×256 pixels), achieving 16× downscaling enhancement via convolutional upsampling of the transformer encoder's latent representations.

## Training Details

### Predictor Variables
The model accepts multi-level atmospheric predictors from CORDEX regional climate models:
- **Dynamic variables** (`input_vars`): `u` (eastward wind), `v` (northward wind), `q` (specific humidity), `t` (temperature), `z` (geopotential height)
- **Pressure levels** (`input_levels`): 850 hPa, 700 hPa, 500 hPa (3 levels)
- **Static variable** (when `use_static=true`): `orog` (orography/surface elevation)
- **Number of input channels**: 16 (5 variables × 3 levels = 15 dynamic + 1 static)
- **Temporal dimension**: Single timestamp (`n_input_timestamps=1`)

### Target Variables
- `pr`: Precipitation rate (mm/day)
- `tasmax`: Daily maximum 2 m air temperature (°C)
- **Output shape**: 256×256 pixels per variable (2 channels total)

### Normalization Scheme
Current CORDEX configs use per-predictand `normalization` plus optional precipitation hurdle settings:

- `tasmax`: `method: standardize`, `mode: gridpoint`
- `pr` (single-head legacy): `method: log1p_standardize`, `mode: gridpoint`
- `pr` (hurdle mode): `method: divide_only`, `scale_stat: p95`, `mode: global`
- `eps_std` avoids division by near-zero local std values

Scalars are computed from training targets over the time axis and saved as `.npy` arrays:

- global mode: shape `[C]`
- gridpoint mode: shape `[C, H, W]`

Inputs remain channel-wise standardized by `input_mu` and `input_sigma`.

When normalization settings change, recompute scalars and retrain/re-evaluate before using for production inference.

### Softplus vs Zscore, and `scale_stat` choices
`softplus` and `zscore` are not interchangeable knobs; they act at different steps:

| Setting | What it does | Typical use |
|---|---|---|
| `nonnegativity.method: softplus` | Applies `softplus(raw_output)` in decoding, so decoded values are always `>= 0` before inverse scaling. | Physically non-negative predictands (for example `pr`). |
| `normalization.method: standardize` | Scales target as `(y - mean) / std`; inverse is `y = y_scaled * std + mean`. This can produce negative values. | Temperature-like predictands (`tasmax`) and other approximately symmetric variables. |
| `normalization.method: log1p_standardize` | Scales transformed target `(log1p(max(y,0)) - mean) / std`; inverse is `expm1(y_scaled * std + mean)`. | Precipitation (`pr`) with zero-heavy and skewed distribution. |
| `scaling.method: divide_only` | Scale-only normalization (`target_mu=0`, `y = y_scaled * scale`). In hurdle mode, `scale_stat: p95` supplies the positive-amount `q95`. | Hurdle precipitation amount head (`pr / q95`). |
| `precip_model: hurdle` | Uses Bernoulli wet-day logits + positive amount head (`softplus`) and outputs exact 0 when `sigmoid(wet_logits) < precip_wet_threshold`. | Precipitation with frequent dry days. |

For `divide_only`, `scale_stat` controls the scale magnitude:

| `scale_stat` | Scale source | Effect on extremes |
|---|---|---|
| `fixed` | User-provided `fixed_scale` in YAML (for example `100.0`). | Most reproducible across runs/periods; preserves linear mapping and avoids dataset-dependent rescaling drift. |
| `mean` | Mean precipitation over training targets. | Smallest typical scale; yields larger normalized values, which can make heavy events numerically larger in training. |
| `p95` | 95th percentile of sampled training precipitation. | Larger scale than `mean`; moderate compression of normalized tail values. |
| `p99` | 99th percentile of sampled training precipitation. | Larger scale than `p95`; strongest tail compression in normalized space and often the most conservative numerically. |

All four `scale_stat` options still allow arbitrarily large physical precipitation after inverse scaling; they mainly change training/inference numeric conditioning in scaled space.

### `allow_negative_value` Semantics
- `allow_negative_value: false` is the default for every predictand.
- For physically non-negative predictands (at minimum `pr`), `allow_negative_value: false` keeps precipitation non-negative after inverse transform (`expm1` path with tiny-negative clamp).
- For temperature-like predictands (for example `tasmax` in degC), nonnegativity remains disabled by default even when `allow_negative_value: false`; this is intentional and logged as a warning.
- **Scalars are computed separately** for static and dynamic inputs when applicable
- Normalization files are stored as `.npy` arrays (NumPy format) with shape `[num_channels]` or `[num_channels, lat, lon]` in gridpoint mode

### Framework and Libraries
- **Deep learning framework**: PyTorch (with PyTorch Lightning for training orchestration)
- **Distributed training**: Fully Sharded Data Parallel (FSDP) support for multi-GPU training
- **Precision**: Mixed precision training (bfloat16/float16 on compatible GPUs, float32 fallback)
- **Key dependencies**: PyTorch, xarray, NetCDF4 (h5netcdf), PyYAML, scipy

### Training Configuration

#### General Settings
- **Training type** (`type_training`): `full_dataset` (train on full training data per config)
- **Batch size**: 1–4 (configured per region/model combination)
- **Number of epochs**: 5 (typical; can be adjusted per config)
- **Number of workers for data loading** (`dl_num_workers`): 2
- **Prefetch size**: 1

#### Learning Rate Schedule
- **Initial learning rate** (`learning_rate`): 5.0×10⁻⁵
- **Minimum learning rate** (`min_lr`): 1.0×10⁻⁶
- **Maximum learning rate** (`max_lr`): 1.0×10⁻⁴
- **Warmup steps** (`warm_up_steps`): 0 (no learning rate warmup by default)
- **Learning rate scheduler**: Cosine annealing with min/max bounds

#### Gradient Accumulation & Optimization
- **Gradient accumulation** (`gradient_accumulation_steps`): configurable in YAML
- **Effective batch size**: `per_device_batch_size * num_gpus * gradient_accumulation_steps`
- **Optimizer**: Adam (default PyTorch Lightning optimizer)

#### Step Limits (for faster iteration during debugging)
- **Training step limit** (`limit_steps_train`): 200 steps per epoch
- **Validation step limit** (`limit_steps_valid`): 50 steps per epoch
- *Note*: Set to 0 (or omitted) to use full dataset

### Loss Function
- **Base loss**: Root Mean Squared Error (RMSE)
- **Hurdle precipitation loss** (when `precip_model: hurdle`):
  - `BCEWithLogitsLoss(wet_logits, wet_target)` on all pixels
  - amount loss (`smoothl1` or `mse`) on wet pixels only, using normalized amount target `pr / q95`
  - weighted sum controlled by `precip_lambda_occurrence` and `precip_lambda_amount`
- **Optional per-predictand distribution losses**:
  - moment (`mean`, `std`, optional `skewness`)
  - quantile
  - CDF (soft empirical CDF matching)
- **Final loss**:
  - non-hurdle: `rmse + sum(weight_i * distribution_loss_i)`
  - hurdle: `rmse(non-precip vars) + precip_hurdle_loss + sum(non-precip distribution losses)`
- **Logging**: training logs include active loss terms and weighted contributions

### Masking Strategy (Optional)
- **Mask unit size** (`mask_unit_size`): [16, 16] (adaptive masking support)
- **Mask ratio for inputs** (`mask_ratio_inputs`): 0.0 (no masking of inputs by default)
- **Mask ratio for targets** (`mask_ratio_targets`): 0.0 (no masking of targets by default)

### Data Sampling
- **Input spatial size** (`input_size_lat`, `input_size_lon`): 16×16 (coarse CORDEX resolution after regridding)
- **Target spatial size** (`target_size_lat`, `target_size_lon`): 256×256 (high-resolution downscaled output)
- **Crop factor**: 256 (crops training samples to this size)
- **Downsample factor**: 8 (ratio between high-res target and coarse predictor)
- **Random windows per sample** (`n_random_windows`): 1

## Dataset and Metadata Guidance

### Data Organization
Each regional domain (ALPS, NZ, SA) follows the same directory structure:
```
granite-geospatial-wxc-downscaling/CORDEX/{REGION}_domain/
├── train/
│   └── ESD_pseudo_reality/          # Training split
│       ├── predictors/
│       │   ├── Static_fields.nc     # Orography field (same for all models)
│       │   ├── {ModelName}_1961-1980_regridded.nc
│       │   └── ...
│       └── target/
│           └── pr_tasmax_{ModelName}_1961-1980.nc
├── test/
│   ├── historical_perfect/          # Perfect predictors (observation-based)
│   │   └── predictors/{ModelName}_1981-2000_regridded.nc
│   ├── historical_imperfect/        # Imperfect predictors (model-based, if available)
│   └── future_projections/          # Mid- and end-century scenarios
└── {optional additional splits}     # e.g., Emulator_hist_future for T2
```

### File Formats
- **Predictor/Target files**: NetCDF4 (`.nc`) format with dimensions `(time, lat, lon)` or `(time, channels, lat, lon)`
- **Scalar statistics**: NumPy binary (`.npy`) arrays with shape `[num_channels]`
- **Checkpoints**: PyTorch (`.pt` or `.ckpt`) format

### Metadata and Attributes
NetCDF files should include:
- **Time coordinate**: `time` (with CF-compliant units, e.g., "days since 1961-01-01")
- **Spatial coordinates**: `lat`, `lon` (latitude/longitude in degrees)
- **Variable attributes**: Standard names, units, long descriptions (CF conventions)
- **Global attributes**: Data source, model name, version, creation date

### Splitting Scheme
- **T1 (ESD_pseudo_reality)**: Pseudo-observations with perfect model predictors; used for initial fine-tuning
  - Training: 1961–1980 (20 years)
  - Test: 1981–2000 (20 years, perfect predictors)
- **T2 (Emulator_hist_future)**: Historical and future climate scenarios with imperfect model predictors
  - Training: Historical period (typically 1961–2000)
  - Test: Multiple scenarios (historical, mid-century 2041–2060, end-century 2080–2099)

### Climate Model Variations
Different global or regional climate models provide the coarse predictors (e.g., ACCESS-CM2, CNRM-CM5, EC-Earth3). Each model:
- Has separate predictor and target files
- Can be trained with or without static orography (`static` vs. `no_static` configurations)
- Produces domain-specific downscaled outputs matching the high-resolution target grid

### Checklist for Custom Domains
When extending this workflow to new regions or datasets:
1. Ensure predictors and targets are co-located and temporally aligned
2. Compute and store normalization scalars using `compute_scalars_cordex.py`
   - Always pass `--config <your_yaml>` (or run one of the `*_preprocess` wrappers, which already pass it).
3. Create domain-specific YAML configs with correct file paths and variable names
4. Verify that `input_vars`, `input_levels`, `output_vars`, and `static_path` match your data
5. Optionally adjust `batch_size`, `num_epochs`, `learning_rate`, and step limits based on GPU memory and desired iteration speed
6. Store predictions in a consistent NetCDF format with metadata for reproducibility
