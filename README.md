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

The CORDEX_ML, MERRA_PRISM and NARR_PRISM configurations share an optional
second-phase library contract that refines the deterministic Prithvi-UNet prediction with a stochastic
*residual* model, selected through `model.refinement.type`:

- `none` — deterministic Prithvi-UNet (default, unchanged behaviour)
- `diffusion_unet` — conditional convolutional UNet, DDPM training / DDIM sampling
- `flow_matching_unet` — conditional convolutional UNet, rectified flow matching
- `diffusion_transformer` — spatial-token Transformer, DDPM / DDIM
- `flow_matching_transformer` — spatial-token Transformer, rectified flow matching

Every active head follows one physical residual contract:
`residual = ground_truth - frozen_phase1`. Per-channel residual statistics are
fitted on the training split, saved in the strict schema-2 Phase-2 checkpoint,
and inverted exactly once before the physical correction is added to Phase 1.
Each ensemble member is reconstructed in physical units before aggregation;
precipitation constraints and prediction masking are applied only after that
reconstruction.

The Transformer heads retain full-domain spatial attention and now use
convolutional stems, overlapping patch embeddings, conditioning at every block,
and an artifact-free convolutional decoder. A zero-initialized output projection
and learnable correction gate make initialization exactly reproduce Phase 1.
Existing YAML files without a `refinement` section and deterministic
checkpoints remain unchanged. Legacy schema-1 **Phase-2** checkpoints require
retraining and are rejected rather than partially loaded.

This is a spatial downscaling / bias-correction problem: predictors and targets
always share the same timestamp, there is no forecast lead time, and Transformer
attention operates over two-dimensional spatial tokens only.

See [docs/STOCHASTIC_REFINEMENT.md](docs/STOCHASTIC_REFINEMENT.md) for the full
description, configuration schema, example YAMLs, commands, checkpoint
compatibility notes, domain-runner availability and benchmarks.


## Temporal (sequence-conditioned) extension

Branch `Prithvi-UNet_temporal_model` adds **explicit temporal dependence** to the
otherwise frame-independent model, so it learns temporally evolving temperature and
precipitation events instead of predicting each date in isolation. Date embeddings
alone are not sufficient and are not what this does: temporal state is carried in a
spatially organized latent at the U-Net bottleneck by one of two interchangeable
backends, selected by `temporal.backend`:

- `recurrent` -- a multi-layer **ConvGRU** (or ConvLSTM) with dilated kernels
- `mamba` -- a **Mamba-2 / SSD** block whose selective scan runs over the **time**
  axis, with explicit convolutional spatial mixing

The task stays *causal, same-day, sequence-conditioned downscaling*: output date
`t` uses the coarse predictors at `t` and earlier, with **zero predictor-to-target
lead time**. Targets are never shifted and observed high-resolution fields never
enter the model's input path.

Everything is opt-in. With `temporal.enabled: false` -- or with no `temporal:` block
at all, which is every existing YAML -- the legacy spatial computation path is
preserved **bit-for-bit**; a test asserts exact equality on the real
246M-parameter SA model with real checkpoint weights. All four existing refinement
heads and their Phase-2 checkpoints remain valid.

Case configurations:

| case | recurrent | mamba |
|---|---|---|
| SA T2 ACCESS-CM2 static | `examples/CORDEX_ML/SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_recurrent.yaml` | `..._temporal_mamba.yaml` |
| NARR/PRISM California | `examples/NARR_PRISM/NARR_PRISM_subdomain_temporal_recurrent.yaml` | `..._temporal_mamba.yaml` |

```bash
# audit a config without loading weights
mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_training.py describe --config <yaml>
# engineering checks on real data (parity, causality, gradients, chunk exactness)
mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_training.py check    --config <yaml>
# fine-tune / infer / evaluate
mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_training.py train    --config <yaml>
mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_training.py infer    --config <yaml> --split test
mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_training.py evaluate --config <yaml> --predictions <npz>
```

See [docs/temporal_model_research.md](docs/temporal_model_research.md) for the
literature review and architecture decisions,
[docs/temporal_model_architecture.md](docs/temporal_model_architecture.md) for
tensor shapes, causality, state management, losses and checkpoint compatibility,
and [docs/temporal_model_results.md](docs/temporal_model_results.md) for measured
outcomes and remaining scientific limitations.
