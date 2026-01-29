# CORDEX-ML Downscaling: ALPS, NZ, and SA

## Overview
This folder hosts the CORDEX-ML benchmark workflows for multiple regional domains—European Alps (ALPS), New Zealand (NZ), and South Africa (SA)—demonstrating how to fine-tune the Prithvi WxC UNet on coarse CORDEX predictors and produce high-resolution precipitation (`pr`) and maximum temperature (`tasmax`) forecasts. The assets here reuse the helper scripts (`preproc_cordex.py`, `compute_scalars_cordex.py`, `cordex_training.py`, and the notebooks in `notebooks/`) to cover the full loop: preprocess/regrid → compute scalars → fine-tune → inference → persist predictions as NetCDF.

## Data (Zenodo)
- **Source**: [CORDEX-ML Benchmark](https://zenodo.org/records/17517423) provides data for multiple domains. The Zenodo record includes coarse predictors, high-resolution targets, and static fields for ALPS, NZ, and SA regions.
- **Inputs**: CORDEX coarse predictors (multi-level dynamics plus static orography) stored under the benchmark's domain-specific `predictors` folders (e.g., `CORDEX/ALPS_domain/`, `CORDEX/NZ_domain/`, `CORDEX/SA_domain/`).
- **Targets**: Downscaled `tasmax` and `pr` fields packaged alongside the predictors in the benchmark `target` NetCDF files for each domain.
- **Setup**: Download the needed predictor/target tiles for your region of interest plus `Static_fields.nc`, then update the YAML/config paths (e.g., `ALPS_T1_*.yaml`, `NZ_T1_*.yaml`, `SA_T1_*.yaml`) to match your local layout.

### End-to-end workflow (explicit files)
1. **Normalization (compute_scalars.py)** – Run `compute_scalars.py` (this repo: `compute_scalars_cordex.py`) to compute predictors/targets mean and std for `{Region}`.
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

# Compute normalization scalars for tasmax/pr fine-tuning (static/orography included)
python examples/CORDEX_ML/compute_scalars_cordex.py \
  --predictor-files ./granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/train/ESD_pseudo_reality/predictors/*_regridded.nc \
  --target-files ./granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/train/ESD_pseudo_reality/target/pr_tasmax_*.nc \
  --use-static \
  --orography-file ./granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/train/ESD_pseudo_reality/predictors/Static_fields.nc \
  --output-dir ./examples/CORDEX_ML/experiments/NZ_T1_ACCESS-CM2_scalars

# For no-static scalars, drop orography and use a separate output dir:
# python examples/CORDEX_ML/compute_scalars_cordex.py \
#   --predictor-files ./granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/train/ESD_pseudo_reality/predictors/*_regridded.nc \
#   --target-files ./granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/train/ESD_pseudo_reality/target/pr_tasmax_*.nc \
#   --no-static \
#   --output-dir ./examples/CORDEX_ML/experiments/NZ_T1_ACCESS-CM2_no_static_scalars

# Launch notebooks (Jupyter or VS Code works)
# For NZ domain:
jupyter lab examples/CORDEX_ML/notebooks/NZ_downscaling_finetune.ipynb
jupyter lab examples/CORDEX_ML/notebooks/NZ_downscaling_inference.ipynb

# For ALPS or SA domains, similarly use ALPS_downscaling_*.ipynb or SA_downscaling_*.ipynb
```
> Tip: set `distributed_strategy: fsdp` in the YAML when fine-tuning on multiple GPUs; switch `device_target: cpu` when only CPUs are available.

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
The model uses channel-wise standardization with pre-computed statistics:
- **Mean normalization** (`input_mu`, `target_mu`): Per-channel mean values computed on training data
- **Standard deviation normalization** (`input_sigma`, `target_sigma`): Per-channel standard deviation values
- **Normalization method**: `(x - mean) / (std + epsilon)` where epsilon is typically 1e-6
- **Scalars are computed separately** for static and dynamic inputs when applicable
- Normalization files are stored as `.npy` arrays (NumPy format) with shape `[num_channels]`

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
- **Maximum batch size** (`max_batch_size`): 4 (effective batch size via gradient accumulation)
- **Optimizer**: Adam (default PyTorch Lightning optimizer)

#### Step Limits (for faster iteration during debugging)
- **Training step limit** (`limit_steps_train`): 200 steps per epoch
- **Validation step limit** (`limit_steps_valid`): 50 steps per epoch
- *Note*: Set to 0 (or omitted) to use full dataset

### Loss Function
- **Primary loss**: Root Mean Squared Error (RMSE)
- **Formula**: $\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2}$
- **Evaluation metrics**: RMSE on validation set; best model checkpoint saved based on lowest validation RMSE

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
3. Create domain-specific YAML configs with correct file paths and variable names
4. Verify that `input_vars`, `input_levels`, `output_vars`, and `static_path` match your data
5. Optionally adjust `batch_size`, `num_epochs`, `learning_rate`, and step limits based on GPU memory and desired iteration speed
6. Store predictions in a consistent NetCDF format with metadata for reproducibility
