# granite-wxc

This repository contains code and examples to apply the [Prithvi WxC foundation model](https://github.com/NASA-IMPACT/Prithvi-WxC) to downscaling tasks. In particular, the repository contains both code and instructions for generic fine-tuning tasks as well as fine-tuned models for MERRA2 2m temperature and the ECCC v10 and u10 wind components as reference.

<p align="center">
   <img src="downscaling_T2M_coolwarm_animated.gif" alt="6x downscaling of MERRA-2 2m temperature" width="70%"/>
   <br><em>Figure 1: 6x downscaling of MERRA-2 2m temperature</em>
</p>

</br>

<p align="center">
   <img src="downscaling_eccc_u10.png" alt="8x downscaling of ECCC's u10 wind" width="70%"/>
   <br><em>Figure 2: 8x downscaling of ECCC's u10 wind component</em>
</p>

## Getting started

1. Create a virtual environment
2. Clone this repository as well as that of the foundation model. Install both in the virtual environment.:
   ```
   git clone https://github.com/NASA-IMPACT/Prithvi-WxC
   git clone https://github.com/IBM/granite-wxc.git
   cd Prithvi-WxC
   pip install '.[examples]'
   cd ../granite-wxc
   pip install '.[examples]'
   ```
3. Run the notebooks in the examples directory
   - [MERRA2 example](examples/merra2_downscaling/notebooks/merra2_downscaling_inference.ipynb):
      This notebook will download model weights as well as sample data for basic illustration from [Hugging Face](https://huggingface.co/ibm-granite/granite-geospatial-wxc-downscaling).
   - [ECCC example](examples/eccc_downscaling/):
         This directory contains notebooks for both fine-tuning and inference. It also includes instructions for downloading and setting up the data, model and the required files from [Hugging Face](https://huggingface.co/ibm-granite/granite-geospatial-wxc-downscaling/tree/main/ECCC).
   - [CORDEX ML example](examples/CORDEX_ML/):
         This directory demonstrates fine-tuning and inference on regional climate data for the CORDEX Machine Learning Task Force benchmark (https://github.com/WCRP-CORDEX/ml-benchmark). It includes workflows for multiple domains (European Alps, New Zealand, and South Africa). Detailed documentation can be found in [examples/CORDEX_ML/README.md](examples/CORDEX_ML/README.md).

## CORDEX v4 pipeline (recommended)

The CORDEX workflow now includes a v4 path with:

- gridpoint-wise target normalization (`mode: gridpoint`) for spatially heterogeneous fields;
- precipitation-aware `log1p_standardize` normalization for `pr` with consistent inverse transform at inference;
- optional distribution-aware loss terms (moment, quantile, CDF) per predictand;
- overlap/blending-based boundary mitigation for tiled inference, with seam deblock kept optional and fallback-only;
- multi-GPU fine-tuning and multi-GPU inference (up to 4 GPUs) plus gradient accumulation and effective batch size reporting.

Use the CORDEX v4 YAMLs (`*_v4.yaml`) and `runs_v4` paths documented in [examples/CORDEX_ML/README.md](examples/CORDEX_ML/README.md).

## Fine-tuned model

The fine-tuned model for MERRA-2 2m temperature data is available via [Hugging Face](https://huggingface.co/ibm-granite/granite-geospatial-wxc-downscaling).

The fine-tuned model for ECCC v10 and u10 wind component data is available via [Hugging Face](https://huggingface.co/ibm-granite/granite-geospatial-wxc-downscaling/tree/main/ECCC).

For applications to CORDEX regional climate data, please refer to the [CORDEX ML example](examples/CORDEX_ML/README.md), which demonstrates fine-tuning on the [CORDEX Machine Learning Task Force benchmark](https://github.com/WCRP-CORDEX/ml-benchmark) for multiple regional domains.

## Two-phase Prithvi-UNet with stochastic residual refinement

The CORDEX_ML, MERRA_PRISM and NARR_PRISM workflows share an optional second
phase that refines the deterministic Prithvi-UNet prediction with a stochastic
*residual* model, selected through `model.refinement.type`:

- `none` — deterministic Prithvi-UNet (default, unchanged behaviour)
- `diffusion_unet` — conditional convolutional UNet, DDPM training / DDIM sampling
- `flow_matching_unet` — conditional convolutional UNet, rectified flow matching
- `diffusion_transformer` — spatial-token Transformer, DDPM / DDIM
- `flow_matching_transformer` — spatial-token Transformer, rectified flow matching

The residual is defined, predicted and added in the Phase-1 normalized target
space; inverse normalization, precipitation constraints and masking are then
applied exactly once. Existing YAML files without a `refinement` section keep
running as deterministic Prithvi-UNet models, and existing deterministic
checkpoints load unchanged (verified bitwise against the NARR_PRISM checkpoint).

This is a spatial downscaling / bias-correction problem: predictors and targets
always share the same timestamp, there is no forecast lead time, and Transformer
attention operates over two-dimensional spatial tokens only.

See [docs/STOCHASTIC_REFINEMENT.md](docs/STOCHASTIC_REFINEMENT.md) for the full
description, configuration schema, example YAMLs, commands, checkpoint
compatibility notes and benchmarks.
