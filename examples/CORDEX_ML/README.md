# CORDEX-ML NZ Downscaling

## Overview
This folder hosts the CORDEX-ML benchmark workflows for the New Zealand domain, showing how to fine-tune the Prithvi WxC UNet on coarse CORDEX predictors and produce high-resolution precipitation (`pr`) and maximum temperature (`tasmax`) forecasts. The assets here reuse the helper scripts (`preproc_cordex.py`, `compute_scalars_cordex.py`, `cordex_training.py`, and the notebooks in `notebooks/`) to cover the full loop: preprocess/regrid → compute scalars → fine-tune → inference → persist predictions as NetCDF.

## Data (Zenodo)
- **Source**: [CORDEX-ML Benchmark – New Zealand domain](https://zenodo.org/records/17517423). The Zenodo record provides the coarse predictors, high-resolution targets, static fields, and splits referenced in the notebooks and configs.
- **Inputs**: CORDEX coarse predictors (multi-level dynamics plus static orography) stored under the benchmark’s `predictors` folders.
- **Targets**: Downscaled `tasmax` and `pr` fields packaged alongside the predictors in the benchmark `target` NetCDF files.
- Download the needed predictor/target tiles plus `Static_fields.nc`, then update the YAML/config paths (e.g., `cordex_config_small.yaml`) to match your local layout.

## Notebooks
### `NZ_downscaling_finetune.ipynb`
- **Purpose**: Walk through fine-tuning Prithvi WxC on the CORDEX-ML NZ training split, starting from a generic checkpoint and adapting it to the tasmax/pr targets.
- **Inputs/targets**: Points the dataloader to the regridded CORDEX predictors (`granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/train/ESD_pseudo_reality/predictors/*_regridded.nc`) and the paired high-resolution targets under `.../target/pr_tasmax_*.nc`.
- **Key knobs**: `cordex_config_small.yaml` exposes GPU count, crop size, pressure levels, FSDP toggle, learning rate, and scaler paths; the notebook highlights which fields to edit for smaller runs.
- **Outputs**: Produces a run directory under `examples/CORDEX_ML/runs/nz_finetune/<run_name>/` with checkpoints (`best.ckpt`, `last.ckpt`), run manifest, scalars copy, and logs ready for inference.

### `NZ_downscaling_inference.ipynb`
- **Purpose**: Load a fine-tuned checkpoint and generate downscaled forecasts on the Zenodo test split.
- **Data loader paths**: Configure `predictor_root` to the benchmark “perfect predictors” folder (e.g., `granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/test/historical_perfect/predictors`) and reuse the manifest from the fine-tune run to locate scalars and checkpoints.
- **Outputs**: Writes `predictions/*.nc` and matching `.pkl` files under the fine-tune run directory; the NetCDFs contain `pr` and `tasmax` grids on the high-resolution NZ domain.

## Workflow
1. **Normalization (compute_scalars.py)** – Run `compute_scalars_cordex.py` to derive per-channel mean/std values for predictors and targets. Use `--use-static` for orography-inclusive scalars (e.g., `experiments/NZ_T1_ACCESS-CM2_static_scalars/`) or `--no-static` for dynamic-only scalars (e.g., `experiments/NZ_T1_ACCESS-CM2_no_static_scalars/`).
2. **Regridding (preproc_cordex.py)** – Use `preproc_cordex.py` to interpolate the coarse CORDEX predictors and static orography onto the NZ high-resolution grid. The `*_wrapper` script (e.g., `preproc_cordex_nz_wrapper.py`) batches this over multiple predictor files to target the high-res grid in one go.
3. **Prepare YAML configs** – Create `NZ_T1_{ModelName}_{static|no_static}.yaml` or `NZ_T2_{ModelName}_{static|no_static}.yaml`:
   - **T1** uses `ESD_pseudo_reality` splits.
   - **T2** uses `Emulator_hist_future` splits.
   - **static/no_static** toggles orography (with/without `Static_fields.nc`).
4. **Fine-tune** – Run `notebooks/NZ_downscaling_finetune_T1(T2)_{ModelName}_{static|no_static}.ipynb` to train and save checkpoints for the chosen config/model.
5. **Inference (12 runs)** – Run `notebooks/NZ_downscaling_inference_T1(T2)_{ModelName}_{static|no_static}.py`:
   - Executes 12 predictor configurations (perfect/imperfect × ACCESS-CM2 & EC-Earth3).
   - Time periods: historical (1981–2000), mid-century (2041–2060), end-century (2080–2099).

## Quickstart
```bash
# Environment (Conda shown, pip/uv work too)
conda create -n cordex-ml python=3.10
conda activate cordex-ml
pip install '.[examples]'   # installs Prithvi WxC + notebook deps
pip install cartopy         # optional, only needed for map plots

# Download CORDEX-ML benchmark data from Zenodo and unpack to ./granite-geospatial-wxc-downscaling/CORDEX/NZ_domain

# Regrid coarse predictors to the high-res target grid
python examples/CORDEX_ML/preproc_cordex.py \
  --predictor-files ./granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/train/ESD_pseudo_reality/predictors/ACCESS-CM2_1961-1980.nc \
  --target-sample ./granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/train/ESD_pseudo_reality/target/pr_tasmax_ACCESS-CM2_1961-1980.nc \
  --orography-file ./granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/train/ESD_pseudo_reality/predictors/Static_fields.nc \
  --output-dir ./granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/train/ESD_pseudo_reality/predictors

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
jupyter lab examples/CORDEX_ML/notebooks/NZ_downscaling_finetune.ipynb
jupyter lab examples/CORDEX_ML/notebooks/NZ_downscaling_inference.ipynb
```
> Tip: set `distributed_strategy: fsdp` in the YAML when fine-tuning on multiple GPUs; switch `device_target: cpu` when only CPUs are available.

## I/O Variables
- **Predictor channels**: CORDEX coarse atmospheric fields referenced in `cordex_config*.yaml` (`u`, `v`, `q`, `t`, `z` across multiple pressure levels) plus static orography.
- **Target variables**:
  - `pr`: precipitation rate, NZ high-resolution grid (mm/day in the benchmark files).
  - `tasmax`: daily max 2 m air temperature, NZ high-resolution grid (°C).

Both notebooks expect predictors/targets to follow the CORDEX-ML benchmark naming and will emit NetCDFs with these same `pr` and `tasmax` variables.

## Model Loading
- The inference notebook (`NZ_downscaling_inference.ipynb`) now enumerates the fine-tune runs under `examples/CORDEX_ML/runs/nz_finetune`, picks the latest (or the one set via `NZ_FINETUNE_RUN`), and stores the resolved checkpoint path in `CHECKPOINT_PATH`.
- When `CHECKPOINT_PATH` is loaded the notebook prints the location, trainable parameter count, and a quick checksum (mean/std of the first tensor) so you can confirm the fine-tuned weights—not the generic base encoder—are active before running inference.

## Outputs
- Both notebooks assert that `config.data.output_vars` provides `['pr', 'tasmax']` and that the dataloaders emit two-channel targets before training/inference begins.
- Generated NetCDFs therefore always include `pr` and `tasmax` variables, matching the CORDEX-ML benchmark naming and order.

## Time Handling
- The inference writer inspects the predictor NetCDF(s) used for inference to build the `time` coordinate and now raises if the predicted sequence length differs from the predictor timestamps.
- The resulting NetCDF inherits the predictor time metadata verbatim, so e.g. `ACCESS-CM2_1981-2000_regridded.nc` yields a 1981–2000 `time` axis rather than reusing the training (1961–1980) window.
- The notebook logs the detected start/end times to make it obvious which forcing period produced a set of predictions.
