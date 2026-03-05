# MERRA2 O3 Rollout: Preprocess, Fine-Tune, Inference

## Overview
This folder provides an end-to-end workflow to train and run a Prithvi-WxC-based next-step chemistry model on MERRA-2:

1. Preprocess raw MERRA-2 chemistry + meteorology files into aligned supervised pairs.
2. Compute predictor/target normalization scalars.
3. Fine-tune an O3 next-step model checkpoint.
4. Run autoregressive 3-hourly rollout inference and save NetCDF outputs plus Cartopy maps.

The default setup predicts `O3_sfc(t + 3h)`, but target/predictor variables are configurable.

## Data and Directory Layout
Expected base directory layout:

```text
<base_dir>/
  merra2/                          # raw MERRA-2 *.nc4 files
  preprocessed/                    # generated train/val pair NetCDFs
  experiments/o3_scalars/          # generated scalers
  checkpoints/                     # fine-tuned checkpoints
  outputs/                         # notebook inference outputs
  weights/                         # downloaded foundation weights
```

Base directory resolution order:
- `MERRA2_PREDICTION_BASE` environment variable (if set)
- `paths.base_dir` in `o3_pipeline_config.yaml` (if set)
- auto-detection of a directory containing `merra2/`

## End-to-End Workflow (Explicit Scripts)
1. `preprocess_o3_pairs.py`
   - Loads chemistry + meteorology fields over `[start_time, end_time]`.
   - Extracts surface values from 3D fields.
   - Builds aligned `(input_time=t, target_time=t+delta_hours)` samples.
   - Writes `o3_pairs_train.nc` and `o3_pairs_val.nc`.
2. `compute_scalars_o3.py`
   - Computes channel-wise predictor mean/std from train pairs.
   - Computes target mean/std for `O3_sfc_target`.
   - Writes `.npy` scaler files and metadata JSON.
3. `finetune_o3_next_step.py`
   - Builds predictor tensor from config or preprocessed dataset metadata.
   - Initializes from a fine-tune checkpoint or foundation backbone.
   - Trains and saves `best.ckpt`, `last.ckpt`, epoch checkpoints, and run manifest.
4. `PrithviWxC_MERRA2_O3_inference.ipynb`
   - Runs 3-hourly inference with optional autoregressive rollout.
   - In autoregressive mode, only the configured rollout species input channel is replaced by previous prediction.
   - Meteorological channels are still read from MERRA-2 at each step.

## Key Files
- `o3_pipeline_config.yaml`: central pipeline configuration.
- `o3_pipeline_utils.py`: shared path/data/model helpers.
- `preprocess_o3_pairs.py`: pair generation.
- `compute_scalars_o3.py`: scaler generation.
- `finetune_o3_next_step.py`: training script.
- `run_o3_pipeline.sh`: wrapper to run preprocess -> scalers -> fine-tune.
- `PrithviWxC_MERRA2_O3_inference.ipynb`: inference + output writing + map plotting.

## Quickstart
```bash
cd examples/merra2_prediction

python preprocess_o3_pairs.py --config o3_pipeline_config.yaml --overwrite
python compute_scalars_o3.py --config o3_pipeline_config.yaml
python finetune_o3_next_step.py --config o3_pipeline_config.yaml
```

Or use the wrapper:

```bash
cd examples/merra2_prediction
./run_o3_pipeline.sh o3_pipeline_config.yaml
```

Optional interpreter override:

```bash
PYTHON_BIN=/mnt/data2/kyo/.mamba/envs/Prithvi/bin/python ./run_o3_pipeline.sh o3_pipeline_config.yaml
```

## Configuration Notes (`o3_pipeline_config.yaml`)
Main configurable blocks:
- `paths`: base/data/output/checkpoint/scaler locations
- `data`:
  - `chem_pattern`, `met_pattern`
  - `start_time`, `end_time`, `delta_hours`
  - `target_var`, `target_input_name`, `target_transform`, `target_name`
  - `include_target_as_predictor`
  - `chem_predictors` (optional additional chemistry predictors)
  - `met_vars`, `met_suffix`
  - `predictor_vars` (optional explicit predictor channel order)
- `preprocess`: split mode (`chronological` or `random`), val fraction, seed
- `scalers`: epsilon and optional explicit scaler paths
- `model`: encoder-decoder architecture knobs
- `training`: optimizer, epochs, AMP, checkpoint init strategy, crop options

## Inference Notebook Behavior
`PrithviWxC_MERRA2_O3_inference.ipynb` includes:
- robust base-dir detection
- 3-hour output naming (`o3_sfc_delta{delta_hours}h_*.nc`)
- safe output write path handling with NetCDF fallback logic
- optional tiling for memory safety
- progress bars (`Inference (AR)` and `Inference (batched)`)
- 15-map Cartopy plotting helpers for predicted and observed O3

Safety defaults:
- If no fine-tuned checkpoint is found, inference stops unless explicitly overridden.
- Foundation-only backbone + random head is allowed only for architecture/debug checks.

## Predictor/Predictand Definition
- Predictand:
  - `target_name = target_input_name(t + delta_hours)`
- Predictors:
  - optionally `target_input_name(t)` (if `include_target_as_predictor: true`)
  - optional additional chemistry predictors (`chem_predictors`)
  - optional meteorological predictors (`met_vars` + `met_suffix`)

### Example: Switch to Another Target Species
Use config-only changes, no code edits needed.

```yaml
data:
  # Example 1: surface CO next-step forecast
  target_var: CO
  target_input_name: CO_sfc
  target_transform: surface
  target_name: CO_sfc_target

  include_target_as_predictor: true
  chem_predictors:
    - {var: O3, output_name: O3_sfc, transform: surface}
  met_vars: [T, U, V, PS]

  # Optional explicit predictor order override
  predictor_vars: [CO_sfc, O3_sfc, T_sfc, U_sfc, V_sfc, PS_sfc]
```

For 2D column variables (for example total-column ozone in files where it is already 2D), set:
- `target_transform: none`

## Outputs
Training outputs:
- `checkpoints/<run_name>/best.ckpt`
- `checkpoints/<run_name>/last.ckpt`
- `checkpoints/<run_name>/epoch_*.ckpt`
- `checkpoints/<run_name>/manifest.json`

Inference outputs:
- `outputs/o3_sfc_delta{delta_hours}h_<start>_<end>.nc` (or `.zarr`)
- map figures from notebook plotting cells

## Troubleshooting
- `ValueError: Expecting tensor with last dimension size 2560`
  - Ensure backbone config (`embed_dim`, `n_blocks`, `n_heads`) matches checkpoint/weights.
- NetCDF `RuntimeError: NetCDF: HDF error`
  - Confirm output path exists and is writable.
  - Retry with `OUTPUT_FORMAT = "zarr"` to isolate backend issues.
- Missing progress bars
  - Install `tqdm` in your runtime environment.
- Output scale mismatch
  - Verify a real fine-tuned checkpoint is loaded (not untrained head fallback).
  - Check that `target_transform` matches variable dimensionality (`surface` for 3D, `none` for 2D columns).

## References
- Foundation model: Prithvi-WxC
  - Paper: https://arxiv.org/abs/2409.13598
  - Repository: https://github.com/NASA-IMPACT/Prithvi-WxC
