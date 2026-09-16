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
spatially organized latent, or by pairing two dates *before* the transformer.
Three interchangeable pathways, selected by `temporal.backend`:

- `recurrent` -- a multi-layer **ConvGRU** (or ConvLSTM) with dilated kernels,
  carrying hidden state at the U-Net bottleneck
- `mamba` -- a **Mamba-2 / SSD** block whose selective scan runs over the **time**
  axis, with explicit convolutional spatial mixing, likewise at the bottleneck
- `native_pair` -- **no new memory module.** Uses the backbone's own
  multi-timestamp input axis, so the predictors at `t-1` and `t` are embedded
  together and meet each other *inside* the shared transformer, exactly as
  upstream Prithvi-WxC combines its two input states (it folds the time axis into
  patch-embed channels before anything learned happens). Adds **32,768**
  parameters -- 233x fewer than the ConvGRU adapter -- and costs *fewer* backbone
  evaluations per window, because a stateless finite-history model has no warm-up
  frames to run. Optional pretraining-aligned auxiliary objectives (masked
  atmospheric reconstruction, atmospheric transition prediction) are available as
  a separate, independently switchable ablation.

⚠️ `native_pair` tests an **architectural** claim, not foundation-model transfer.
This pipeline contains no Prithvi-WxC foundation weights: the configs pair
`model.embed_dim: 1024` with an `embed_dim 2560` checkpoint, so all 170 backbone
tensors shape-mismatch and are dropped by the loader's shape filter, and the
transformer was trained from random initialization on the downscaling task. See
[docs/temporal_native_pair.md](docs/temporal_native_pair.md) §2 for the measurement.

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

| case | recurrent | mamba | native pair | native pair + pretext |
|---|---|---|---|---|
| SA T2 ACCESS-CM2 static | `examples/CORDEX_ML/SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_recurrent.yaml` | `..._temporal_mamba.yaml` | `..._temporal_prithvi_native_pair.yaml` | `..._temporal_prithvi_native_pair_pretext.yaml` |
| NARR/PRISM California | `examples/NARR_PRISM/NARR_PRISM_subdomain_temporal_recurrent.yaml` | `..._temporal_mamba.yaml` | `..._temporal_prithvi_native_pair.yaml` | `..._temporal_prithvi_native_pair_pretext.yaml` |

Every config runs through the same CLI below with no code edits. The NARR/PRISM
case has **never been executed** — its archives, scalers and Phase-1 checkpoint are
absent from the development machine — so its configs are schema-complete and
test-validated only.

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
and [docs/temporal_model_results.md](docs/temporal_model_results.md) for the
measured outcomes of the two bottleneck backends — both of which were
**NOT ACCEPTED** against the pre-registered criteria, a verdict that stands.

For the `native_pair` pathway see
[docs/temporal_native_pair.md](docs/temporal_native_pair.md) (mechanism, the
checkpoint-provenance measurement, the pretext objectives, and limitations) and
[docs/temporal_native_pair_results.md](docs/temporal_native_pair_results.md) (the
bounded comparison and its verdict, scored by the **unchanged** pre-registered
scorecard plus one additional capacity control).


### Selected temporal workflow (takeover v2)

Use `examples/CORDEX_ML/notebooks/Temporal_selected_training_v2.ipynb` to select
one SA or NARR configuration and checkpoint for train, resume, infer and evaluate.
Set `BACKEND` to `recurrent`, `mamba`, `native_pair`, or `native_pair_pretext`.
The notebook uses each YAML's epoch count, accumulation and complete data splits;
short-run update/validation caps are optional. All action switches default to false.
Mamba's shipped `implementation: reference` uses the repository's PyTorch backend.
The spatial-only `SA_downscaling_finetune_T2_ACCESS-CM2_static.ipynb` rejects temporal
YAMLs early: its frame trainer and two-GPU FSDP setup do not implement the temporal
workflow. The temporal notebook invokes the dedicated CLI from the repository root.
The shared CLI accepts `train --resume CHECKPOINT` and repeated
`--set dotted.path=YAML_value` overrides for existing configuration keys.
`--max-steps` counts actual successful optimizer updates per epoch; microbatches
and temporal-parameter updates are recorded separately.

Optional `*_temporal_prithvi_native_pair*_extended_v2.yaml` configurations add
20-epoch development schedules. They are not launched automatically and are
separate from the frozen-encoder 600-update comparison. NARR retains
`case_name: narr_prism_California`, its real NetCDF source contracts, 32 predictor
channels, `ppt,tmax,tmin` output order, masks and hurdle precipitation.

See `docs/temporal_codex_handoff.md` for current execution/evidence and
`docs/temporal_pretrained_transfer_takeover.md` for the separate transfer path.
Existing recurrent/Mamba results remain not accepted. Refinement interface
compatibility does not establish calibration after changing Phase-1 predictions.


Fourteen obsolete/standalone transfer-control YAMLs were removed from the active
example directories. One current `*_regional_pretrained_transfer_v2.yaml` remains
per case; select controls using `--initialization` and `--history-mode` instead.
Archived copies and hashes: `artifacts/config_cleanup_20260915/cleanup_manifest.json`.
The notebook generator creates no YAMLs by default; `--include-extended-configs`
explicitly creates missing optional native training schedules. Existing experiment
snapshots, resolved configurations, scientific recipes and outputs are preserved.
