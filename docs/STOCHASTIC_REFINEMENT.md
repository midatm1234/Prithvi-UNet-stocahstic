# Two-phase Prithvi-UNet with stochastic residual refinement

This document describes the unified two-phase downscaling architecture shared by
the **CORDEX_ML**, **MERRA_PRISM** and **NARR_PRISM** workflows.

---

## 1. Scientific framing (read this first)

This is a **spatial downscaling and bias-correction** problem, not a temporal
forecasting problem.

* For every training and inference sample, the predictors and the target refer
  to the **same date/time**.
* There is **no forecast lead time** between predictors and targets, and none is
  introduced anywhere in this branch.
* There are **no** lead-time embeddings, temporal attention, causal masks,
  recurrent/autoregressive components or temporal token sequences.
* Transformer attention operates over **2-D spatial tokens only** — patches of
  the target latitude/longitude grid from a single sample at a single timestamp.

Two *mathematical* process variables do appear:

| symbol | meaning | not |
| --- | --- | --- |
| diffusion timestep `k` in `{0..K-1}` | index of a Gaussian noising process | a timestamp, a lead time, a sequence position |
| flow interpolation time `t` in `[0,1]` | integration coordinate of a probability path | a timestamp, a lead time, a sequence position |

Both are generated internally by the refiner and never read from the dataset.
`tests/test_refinement_timestamps.py` and `tests/test_refinement_transformer.py`
enforce these properties automatically.

---

## 2. Architecture

```
                    +---------- Phase 1 (deterministic) -------------+
 predictors  x ---> | patch embedding -> Prithvi-WxC backbone ->     | ---> y_phys
 static      s ---> | UNet decoder -> output conv -> constraints     |      y_norm
                    +-----------------------------------------------+
                                       |  y_norm (+ optional feature maps)
                                       v
                    +------- Phase 2 (optional, stochastic) ---------+
 conditioning  ---> | diffusion_unet | flow_matching_unet            | ---> r_hat
 (see section 4)    | diffusion_transformer | flow_matching_transf.  |
                    +-----------------------------------------------+
                                       |
        refined_norm = y_norm + r_hat  ->  decode once  ->  constraints once
```

### Phase 1 — deterministic downscaling

`granitewxc.models.cordex_finetune_model.ClimateDownscaleFinetuneUNETModel`,
built by `granitewxc.models.model.get_finetune_model_UNET`. **Unchanged.** The
only addition is an *opt-in* feature-capture hook
(`set_feature_capture` / `get_last_phase1_features`) which is inert by default,
so deterministic runs are bit-for-bit identical to the `NARR_PRISM` branch.

### Phase 2 — optional stochastic residual refinement

Selected with `model.refinement.type`:

| value | model | process variable |
| --- | --- | --- |
| `none` | *(disabled)* — deterministic Prithvi-UNet | – |
| `diffusion_unet` | conditional convolutional UNet, DDPM training / DDIM sampling | diffusion timestep |
| `flow_matching_unet` | conditional convolutional UNet, rectified flow | flow time |
| `diffusion_transformer` | spatial-token Transformer, DDPM/DDIM | diffusion timestep |
| `flow_matching_transformer` | spatial-token Transformer, rectified flow | flow time |

The four options are **alternatives**. Flow matching does *not* require the
diffusion head to run first, and the two are never stacked. Construction goes
through a registry (`granitewxc.refinement.base.build_refiner`), so training and
inference contain no per-type `if/elif` chains.

---

## 3. Residual target and reconstruction

Refinement happens entirely in the **Phase-1 normalized target space**
(`granitewxc.refinement.target_space.NormalizedTargetSpace`), never in physical
units.

```
residual_target  = encode(y) - y_norm                     # training target
refined_norm     = y_norm + r_hat                         # add in normalized space
refined_physical = decode(refined_norm)                   # inverse-normalize ONCE
refined_physical = apply_constraints(refined_physical)    # non-negativity, ONCE
refined_physical = apply_mask(refined_physical)           # invalid cells, ONCE
```

`encode` / `decode` are derived from the Phase-1 output scalers and the
per-predictand scaling method, so all dataset families are supported without
special-casing:

| `predictands.<name>.scaling.method` | `encode` | `decode` |
| --- | --- | --- |
| `zscore` | `(y - mu) / sigma` | `n*sigma + mu` |
| `divide_only` | `y / sigma` | `n*sigma` |
| `log1p_zscore` | `(log1p(y) - mu) / sigma` | `expm1(n*sigma + mu)` |

`y_norm` is `encode(y_phys)`, where `y_phys` is the *final* deterministic
prediction. This is deliberately not assumed to equal the raw
`x_pre_inverse`: the NARR--PRISM hurdle head returns an ungated amount latent
but a wet/dry-gated physical precipitation field. Re-encoding the final field
guarantees `decode(y_norm) == y_phys`, including dry pixels, so a zero residual
is an exact no-op.

When `residual_normalization.enabled` is true, the residual is additionally
standardized one explicitly ordered output channel at a time. Mean/std are fit
only by streaming the Phase-2 **training** loader with frozen Phase 1, stored as
persistent refiner buffers, and inverted once before residual addition. A model
cannot train or sample with enabled-but-unfitted statistics.

**A normalized residual is never added to a field in physical units.**
Inverse normalization, the precipitation inverse transform, non-negativity
clipping and masking are each applied exactly once, after the addition.

Invalid target cells (NaN) get a residual of exactly zero and are excluded from
the masked loss denominator, so masks never bias the loss magnitude. Masked
cells stay NaN in the output.

---

## 4. Conditioning

`model.refinement.conditioning` selects which spatial fields are concatenated
along the channel axis and resampled/cropped to the target grid:

| key | tensor |
| --- | --- |
| `deterministic_output` | Phase-1 prediction in normalized space |
| `input_predictors` | `batch['x']` after the exact Phase-1 input standardization |
| `static_fields` | `batch['static_x']`, `batch['static_y']` |
| `masks` | predictor validity mask (1 channel) |
| `coordinates` | absolute normalized PRISM row/column channels |
| `prithvi_features` | post-backbone convolution output |
| `unet_features` | final UNet decoder activation |

Every haloed conditioning field is cropped with the per-sample
`__output_crop=(top,left,height,width)` metadata; a symmetric crop is only a
legacy fallback. This prevents the historical 32-pixel NARR predictor shift.
NARR already appends 16 explicit predictor-mask channels, so its YAML disables
the redundant post-fill `isfinite(x)` mask (which would be all ones) and enables
absolute coordinates to keep overlapping Transformer tiles geographically
consistent.

Only the enabled sources are computed — an unused Phase-1 feature map is never
materialised. The **target is never part of the conditioning**
(`tests/test_refinement_timestamps.py::test_conditioning_never_includes_the_target`).

`model.refinement.conditioning` does **not** include lead time, because no
predictor/target lead time exists.

---

## 5. Diffusion implementation

* Discrete DDPM with `training_timesteps` steps and a `cosine` (default),
  `linear` or `scaled_linear` beta schedule. All coefficient tensors are built
  once per configuration and stored as non-persistent device buffers, so nothing
  is rebuilt inside the sampling loop and refinement checkpoints stay small.
* Forward noising: `r_k = sqrt(abar_k) * r + sqrt(1 - abar_k) * eps`, with `k`
  drawn uniformly per batch element and `eps ~ N(0, I)`.
* `prediction_type` is `epsilon` (default), `velocity` (v-prediction) or
  `sample`. All three are inter-convertible through the schedule helpers, which
  is verified in `tests/test_refinement_performance.py`.
* Loss: masked MSE/L1/Huber against the configured parameterisation, reduced in
  float32. Optional reconstruction, bias, gradient, Laplacian, multiscale and
  tail terms are applied only after converting the network output to a valid
  clean-residual estimate (`x0`), never directly to epsilon/velocity.
* Sampling: generalized DDIM over `inference_steps` timesteps drawn from the
  *same* schedule. `eta = 0` (default) is deterministic. With the complete,
  consecutive training grid, `eta = 1` has the DDPM posterior variance; with a
  strided inference grid it remains a stochastic DDIM transition rather than
  an exact ancestral DDPM step. The configured sampler is never silently
  replaced with a different approximation.
* Optional `clip_sample` clamps the predicted clean residual to
  `+/- clip_sample_range` in normalized space.

The convolutional score network, the Fourier/sinusoidal process-time embedding
and the conditioning-by-concatenation design are carried over from the
`CORDEX_ML_diffusion_head` branch; the model was re-implemented as a *residual*
refiner with a configurable prediction type so that one implementation serves
all three dataset families.

---

## 6. Flow-matching implementation

Conditional (rectified) flow matching, following the Aurora
`aurora_finetune_flow_matching` reference:

* **Source distribution** `p0`: isotropic Gaussian `N(0, I)`
  (`source_distribution: gaussian`).
* **Target distribution** `p1`: the conditional residual distribution.
* **Path**: straight optimal-transport interpolation `x_t = (1-t)*x0 + t*r`.
* **Velocity target**: `u_t = r - x0` (constant along each conditional path).
* **Loss**: masked MSE between `v_theta(x_t, t, cond)` and `u_t`. Optional
  clean-residual terms use the valid endpoint estimate
  `x1_hat = x_t + (1-t)*v_theta`.
* **Flow-time sampling**: `uniform`, or `logit_normal` (sigmoid of
  `N(mean, std)`, default `mean = -0.5`, `std = 1.2`) which concentrates
  training on informative mid-noise levels.
* **Flow-time embedding**: the same sinusoidal embedding used by the diffusion
  refiners, applied to `t * 1000` so a single implementation serves both.
* **Initial state**: `x0 ~ N(0, I)` drawn from the caller's `torch.Generator`
  when `stochastic_initialization: true` (default), otherwise `x0 = 0` for a
  deterministic mean-path corrector.
* **Integration**: `dx/dt = v_theta(x, t, cond)` from `t = 0` to `t = 1` over
  `integration_steps` uniform steps. Solvers: `euler` (default, safe),
  `midpoint`, `heun`. The integration grid is built once per call on the target
  device.
* `x1` **is** the predicted residual.

---

## 7. Spatial Transformer and 2-D positional encoding

`granitewxc.refinement.backbones.SpatialResidualTransformer` is a DiT-style
network. Per sample:

1. Concatenate the noisy/interpolated residual with the conditioning stack.
2. Replicate-pad the trailing edges to a multiple of the patch size.
3. Vectorised `patchify_2d` into `grid_h x grid_w` tokens in row-major
   `(lat, lon)` order — token `i` maps to grid cell `(i // grid_w, i % grid_w)`.
4. Linear token projection into `embedding_dim`.
5. Add 2-D positional information:
   * `learned_2d` (default): a separable pair of learned tables, one per axis,
     summed — parameter count is linear in the grid extent while every
     `(lat, lon)` token still gets a distinct embedding.
   * `sincos_2d`: fixed sin/cos, half the width per axis, cached per grid shape.
6. `num_blocks` x [ multi-head **spatial** self-attention -> MLP ], pre-norm,
   with residual connections and **adaptive layer normalisation** carrying the
   diffusion timestep / flow time (zero-initialised gates => identity at init).
7. Linear projection back to patch pixels, `unpatchify_2d`, then an **exact
   crop** to the original `(H, W)`.

Guarantees (all covered by `tests/test_refinement_transformer.py`):

* attention mixes only spatial tokens — perturbing one batch element cannot
  change another;
* attention is bidirectional (no causal mask);
* rectangular and non-patch-divisible grids work; output `(H, W)` is exact;
* process-time conditioning modulates values but never permutes token positions;
* `optimized_attention: sdpa` and the reference `math` path agree to
  `atol = rtol = 1e-5` in fp32.

Configurable: `patch_size` (shared or `[height, width]`), `embedding_dim`,
`num_heads`, `num_blocks`, `mlp_ratio`, `dropout`, `positional_encoding`,
`max_tokens_lat/lon`, `gradient_checkpointing`, `optimized_attention`.
Validation rejects `embedding_dim % num_heads != 0`, sin/cos embedding widths
that are not divisible by four, malformed patch settings, and token grids
larger than the learned positional tables. Learned 2-D positions do not impose
an unrelated head-dimension parity restriction.

---

## 8. Configuration schema

The section may live at the top level or under `model:`. A file **without** it
resolves to `enabled: false, type: none` and behaves exactly like the current
deterministic branch.

```yaml
model:
  phase1:
    checkpoint: ./examples/NARR_PRISM/experiments/checkpoints/narr_prism_California/last.ckpt
    freeze: true

  refinement:
    enabled: true
    type: diffusion_unet        # none | diffusion_unet | flow_matching_unet
                                # | diffusion_transformer | flow_matching_transformer
    checkpoint: null
    freeze_phase1: true
    joint_finetuning: false     # must be requested explicitly
    train_on_residual: true
    correction_scale: 1.0
    trainable_phase1_patterns: []  # e.g. ["head.*"]; mutually exclusive with joint mode
    ensemble_size: 10
    loss: mse                   # mse | l1 | huber
    seed: 1234

    conditioning:
      deterministic_output: true
      input_predictors: true
      prithvi_features: false
      unet_features: false
      static_fields: false
      masks: false
      coordinates: true

    residual_normalization:
      enabled: true
      epsilon: 1.0e-6
      fit_batches: 0        # 0 = complete training loader

    auxiliary_loss:
      reconstruction_weight: 0.10
      reconstruction_loss: huber
      bias_weight: 0.02
      gradient_weight: 0.05
      laplacian_weight: 0.01
      multiscale_weight: 0.02
      tail_weight: 0.02
      tail_threshold: 2.0
      variable_weights: [1.0, 1.0, 1.0]

    diffusion:                  # diffusion_* only
      training_timesteps: 1000
      inference_steps: 50
      prediction_type: epsilon  # epsilon | velocity | sample
      schedule: cosine          # cosine | linear | scaled_linear
      eta: 0.0                  # 0 = deterministic; >0 = stochastic DDIM
      clip_sample: false
      clip_sample_range: 10.0

    flow_matching:              # flow_matching_* only
      integration_steps: 50
      solver: euler             # euler | midpoint | heun
      source_distribution: gaussian
      stochastic_initialization: true
      time_sampling: uniform    # uniform | logit_normal
      sigma_min: 1.0e-4

    transformer:                # *_transformer only
      patch_size: 8             # int, or [height, width]
      embedding_dim: 256
      num_heads: 8
      num_blocks: 6
      mlp_ratio: 4.0
      dropout: 0.0
      positional_encoding: learned_2d   # learned_2d | sincos_2d
      max_tokens_lat: 256
      max_tokens_lon: 256
      gradient_checkpointing: false
      optimized_attention: auto         # auto | sdpa | math

    unet:                       # *_unet only
      hidden_channels: 64
      num_levels: 3
      time_embedding_dim: 128
      bottleneck_attention: true
      attention_heads: 4
      zero_init_output: true

performance:                    # workflow only - never changes the science
  profile: false
  dataloader:
    num_workers: auto
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 2
    non_blocking_transfer: true
  precision:
    mode: fp32                  # fp32 | bf16 | fp16
    allow_tf32: false
  compile:
    enabled: false
    phase1: false
    refinement: false
  phase1_cache:
    # NARR--PRISM production setting. Other workflows may leave this disabled.
    enabled: true
    path: ./examples/NARR_PRISM/experiments/phase1_residual_cache
    include_deterministic_output: true
    include_prithvi_features: false
    include_unet_features: false
    validate_cache: true
  ensemble:
    batch_members: true
    chunk_size: auto
  io:
    atomic_checkpoints: true
    netcdf_compression: true
    netcdf_compression_level: 4
```

Validation guarantees:

* unknown refinement names, unknown keys, invalid Transformer geometry, invalid
  patch settings, `inference_steps > training_timesteps`, unsupported
  performance options and mutually incompatible settings all raise
  `ConfigValidationError` with an explicit message;
* `joint_finetuning: true` conflicts with an explicit `freeze_phase1: true` or
  selected-component patterns; active `train_on_residual: false` is rejected
  because inference would otherwise add an absolute target as a residual;
* `phase1_cache.enabled` requires `freeze_phase1: true` and rejects joint
  fine-tuning;
* fitted residual normalization requires a completely frozen Phase 1 and is
  rejected for joint or selected-component fine-tuning because changing the
  baseline would immediately stale those statistics;
* scientific settings (`model.refinement`) and workflow settings (`performance`)
  are strictly separate, and performance settings never change architecture,
  loss, sampler, solver, ensemble size or evaluation data;
* the fully resolved configuration is stored in every checkpoint under
  `resolved_config`.

### Example configurations

| file | purpose |
| --- | --- |
| `examples/NARR_PRISM/NARR_PRISM_deterministic.yaml` | Phase-1 only, explicit `type: none` |
| `examples/NARR_PRISM/NARR_PRISM_diffusion_unet.yaml` | Phase-2 diffusion UNet |
| `examples/NARR_PRISM/NARR_PRISM_flow_matching_unet.yaml` | Phase-2 flow-matching UNet |
| `examples/NARR_PRISM/NARR_PRISM_diffusion_transformer.yaml` | Phase-2 diffusion Transformer |
| `examples/NARR_PRISM/NARR_PRISM_flow_matching_transformer.yaml` | Phase-2 flow-matching Transformer |
| `examples/MERRA_PRISM/MERRA_PRISM_diffusion_unet.yaml` | MERRA-PRISM diffusion UNet |
| `examples/MERRA_PRISM/MERRA_PRISM_flow_matching_transformer.yaml` | MERRA-PRISM flow-matching Transformer |
| `examples/CORDEX_ML/NZ_T1_ACCESS-CM2_static_diffusion_unet.yaml` | CORDEX diffusion UNet |
| `examples/CORDEX_ML/NZ_T1_ACCESS-CM2_static_flow_matching_unet.yaml` | CORDEX flow-matching UNet |

Existing deterministic model/training configurations remain usable. The
NARR--PRISM production inference contract has one intentional artifact
migration: recompute the original training-period scalars once to persist the
authenticated `target_valid_mask.npy`, then regenerate held-out predictor
products without embedded targets. Scaler values and the Phase-1 architecture
do not change. Phase-2 schema-1 heads require retraining because their residual
baseline semantics cannot be migrated losslessly.

---

## 9. Checkpoints and NARR_PRISM compatibility

### Migration

`TwoPhaseDownscalingModel` stores Phase 1 under the `phase1.` prefix and Phase 2
under `refiner.`. The only migration needed for an existing deterministic
checkpoint is that prefix, applied explicitly by
`granitewxc.refinement.checkpoint.migrate_phase1_state_dict`, which also strips
`module.` / `_orig_mod.` wrappers and uniform Lightning roots such as `model.`.
`LEGACY_PHASE1_KEY_RENAMES` is the single
place where any future rename must be registered — never `strict=False`.

### Controlled loading

`load_phase1_state_dict` raises on any unexpected key, any shape mismatch, and
any missing key that does **not** belong to `refiner.*`. `strict=False` is used
only after that proof.

### Historical parity record and current artifact status

```
python examples/NARR_PRISM/narr_prism_refinement_parity.py \
    --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml \
    --checkpoint examples/NARR_PRISM/experiments/checkpoints/narr_prism_California/last.ckpt \
    --device cuda:0 --size 256
```

`examples/NARR_PRISM/refinement_parity.json` records a prior run against
fingerprint `4f9c5eb7...` (208 tensors, 208 renames, 0 missing / unexpected /
mismatched):

| space | max abs diff | mean abs diff | max rel diff | verdict |
| --- | --- | --- | --- | --- |
| normalized (pre-Phase-2) | 0.0 | 0.0 | 0.0 | bitwise identical |
| physical (post-decode) | 0.0 | 0.0 | 0.0 | bitwise identical |

That record remains historical evidence rather than verification of the current
checkpoint. The local artifact portal now resolves
`examples/NARR_PRISM/{experiments,preprocessed,scalars_with_H}` to the durable
external root `/data2/granite-wxc-artifacts/NARR_PRISM`. The recovered
`last.ckpt` has file SHA-256
`1d352491ab32a4069696673f504edcd9c67d48325e83dcf3b042fe3d3eb80efa`
and deterministic tensor fingerprint
`84c8509ce161f049d8999e5d44b43b83d36cc316a579d052df1bd0f6173a31c7`.
Phase 2 runs `validate_prism_checkpoint_contract` before tensor loading;
matching shapes alone cannot bypass variable/grid/scaler/date/topology checks.

### Checkpoint kinds

| kind | contents |
| --- | --- |
| `phase1` | `phase1.*` only |
| `refinement` | `refiner.*` only, plus the referenced Phase-1 path **and SHA-256 fingerprint** |
| `combined` | both — used to resume joint or selected-component fine-tuning |

Every payload carries a Phase-2 schema and residual-semantics marker, model
state, optimizer, scheduler, gradient scaler,
epoch, global step, RNG states, the resolved configuration, the case name, the
precision settings, the refinement type and the Phase-1 identity. New
NARR--PRISM checkpoints also persist the effective optimizer, base/minimum LR,
warmup, accumulation, clipping, cosine horizon, total epoch budget, and
train/validation step limits. Schema-1 Phase-2 payloads are rejected because
they were trained against the hurdle precipitation amount latent, whereas
schema 2 uses `encode(final_phase1_physical)` and therefore has a different
residual target. There is no lossless automatic weight migration; those
refinement heads must be retrained. Resume compares the remaining fields plus
the complete
refinement/performance/precision contracts before loading state, validates the
Phase-1 fingerprint, and treats configured epochs as a total budget rather than
additional epochs. Incomplete schema-2 model, performance, precision, or
optimizer/schedule metadata is rejected before state loading. Checkpoints are
written atomically
(`os.replace`) by default.

The production NARR--PRISM inference entry point intentionally accepts only a
refinement-only Phase-2 checkpoint plus the separately validated deterministic
checkpoint. It rejects `combined` checkpoints so a fine-tuned Phase-1 cannot be
silently substituted for the configured `last.ckpt`. Combined checkpoints are
therefore a training/resume format until a separately audited joint-inference
contract is added.

---

## 10. Training and inference workflows

```bash
# 1. Deterministic Phase-1 training (unchanged)
python examples/NARR_PRISM/narr_prism_finetune.py --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml

# 2. Resume deterministic Phase-1 training (unchanged: resume_training in the YAML)

# 2a. Persist/authenticate training-only output support. This recomputes the
# existing scalers from the same 1996--2013 split and adds target_valid_mask.npy.
python examples/NARR_PRISM/compute_scalars_narr_prism.py \
    --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml

# 2b. Build target-free 2016--2025 predictor products. Existing inference
# products containing target_* fields must be regenerated under this contract.
python examples/NARR_PRISM/preproc_narr_prism.py \
    --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml \
    --mode inference --overwrite

# 3. Phase-2 training from an existing deterministic checkpoint (Phase 1 frozen)
python examples/NARR_PRISM/narr_prism_refinement.py train \
    --config examples/NARR_PRISM/NARR_PRISM_diffusion_unet.yaml --device cuda:0

# 4. Resume Phase-2 training
python examples/NARR_PRISM/narr_prism_refinement.py train \
    --config examples/NARR_PRISM/NARR_PRISM_diffusion_unet.yaml --resume

# 5. Joint Phase-1 + Phase-2 fine-tuning: set in the YAML
#    model.refinement.joint_finetuning: true

# 6. Deterministic inference (same Phase-1 weights; authenticated target-free IO)
python examples/NARR_PRISM/narr_prism_inference.py --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml

# 6a. Held-out 2014--2015 predictions for method selection. Validation files
# are placed below <output>/validation/ so they cannot overwrite 2016--2025.
# Even when validation daily products contain target_* variables, prediction
# opens a predictor-only dataset. Truth is read separately by evaluation.
python examples/NARR_PRISM/narr_prism_inference.py \
    --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml \
    --split validation --output-dir ./deterministic_daily

# 7/8. Refined inference, one or many ensemble members
python examples/NARR_PRISM/narr_prism_refinement.py infer \
    --config examples/NARR_PRISM/NARR_PRISM_flow_matching_unet.yaml \
    --refinement-checkpoint <refinement_checkpoints>/flow_matching_unet/best.ckpt \
    --ensemble-size 10 --seed 1234 --output ./refined_daily

# 8a. Use the same Phase-1/Phase-2 artifacts on held-out validation dates.
python examples/NARR_PRISM/narr_prism_refinement.py infer \
    --config examples/NARR_PRISM/NARR_PRISM_flow_matching_unet.yaml \
    --refinement-checkpoint <refinement_checkpoints>/flow_matching_unet/best.ckpt \
    --split validation --ensemble-size 10 --seed 1234 \
    --output ./refined_daily

# 8b. Compare methods only on the held-out validation split. The evaluator
# rejects products whose dataset_split or full configured split bounds differ
# from dates.validation. Optional --start/--end subsets must be supplied
# together and remain inside that split.
python examples/NARR_PRISM/evaluate_refinement.py \
    --config examples/NARR_PRISM/NARR_PRISM_flow_matching_unet.yaml \
    --split validation \
    --phase1-dir ./deterministic_daily/validation \
    --method flow=./refined_daily/validation \
    --output-dir ./validation_metrics

# 9. Inspect the resolved configuration
python examples/NARR_PRISM/narr_prism_refinement.py describe --config <config>.yaml

# Synthetic smoke test (all four real YAMLs; writes only requested diagnostics)
python examples/NARR_PRISM/refinement_smoke.py \
    --output-dir /tmp/narr_prism_refinement_smoke

# Deterministic checkpoint parity
python examples/NARR_PRISM/narr_prism_refinement_parity.py --config ... --checkpoint ...

# Performance benchmarks
python examples/refinement_benchmark.py --device cuda:0 --size 256 --ensemble 8 --steps 20
```

The default Phase-2 workflow loads a deterministic checkpoint, freezes Phase 1,
keeps it in `eval()`, generates the conditioning under `torch.no_grad()`, and
trains the refiner on residuals.

---

## 11. Output

The generic `granitewxc.refinement.io` writer can write a single aggregate
NetCDF containing, per target variable:

`<var>` (deterministic), `<var>_residual` (normalized space), `<var>_refined`,
`<var>_members` (with an explicit `member` dimension in draw order),
`<var>_ensemble_mean`, `<var>_ensemble_spread` (unbiased) and `<var>_truth` when
the workflow already saves it.

Timestamps, coordinates, variable names, units, calendar, masks, fill values,
attributes, latitude/longitude orientation and ensemble ordering are preserved.
Compression and chunking are configurable and lossless; the file is written in a
single pass and, by default, atomically.

Ensemble mean and spread are accumulated in float32 and skip NaN (masked) cells
so masked points never contaminate the statistics.

Production NARR--PRISM inference instead writes one canonical-grid file for
each exact date under
`<root>/<case>/<case>_<type>_refined_YYYYMMDD.nc`. There, the exact names
`ppt`, `tmax`, and `tmin` are always refined ensemble means; `_phase1`,
`_member_NNN`, and `_ensemble_spread` suffixes identify the baseline, members,
and spread. The predictor-only inference dataset never discovers or loads
held-out PRISM targets. Its static domain mask is a separately persisted,
signed boolean intersection of finite training-period targets in the explicit
output order. Unsupported scaler cells are neutral-filled, so scaler finiteness
is deliberately never used as a support mask.

Historical PRISM rasters may omit `units`. The NARR YAMLs therefore carry the
explicit `evaluation.truth_units` contract. The streaming evaluator validates
all present source-unit attributes and records every file for which that exact
configured convention was required; incompatible metadata always fails.

---

## 12. Performance

Measured on an NVIDIA A100 80GB PCIe, PyTorch 2.12.0+cu130 / CUDA 13.0, fp32,
TF32 off, batch size 1, 256x256 domain
(`examples/NARR_PRISM/refinement_benchmark.json`):

| optimization | case | baseline | optimized | gain | peak GPU (base -> opt) | parity |
| --- | --- | --- | --- | --- | --- | --- |
| batched ensemble generation (N=8, 20 steps) | diffusion_unet | 1.315 s | 0.840 s | **36.1 %** | 651 -> 5022 MiB | bitwise |
| batched ensemble generation | flow_matching_unet | 1.221 s | 0.831 s | **31.9 %** | 662 -> 5022 MiB | bitwise |
| batched ensemble generation | diffusion_transformer | 2.262 s | 1.929 s | **14.7 %** | 111 -> 472 MiB | bitwise |
| batched ensemble generation | flow_matching_transformer | 2.196 s | 1.932 s | **12.0 %** | 109 -> 460 MiB | bitwise |
| fused SDPA vs reference attention | diffusion_transformer | 0.622 s | 0.286 s | **54.1 %** | 1101 -> 94 MiB | bitwise |
| fused SDPA vs reference attention | flow_matching_transformer | 0.609 s | 0.274 s | **55.0 %** | 1102 -> 94 MiB | bitwise |
| frozen Phase-1 (eval + `no_grad`) | real Prithvi-UNet | 1.755 s | 1.756 s | 0 % (time) | 5708 -> 4882 MiB (**-14 %**) | bitwise |
| precomputed frozen Phase-1 cache | real Prithvi-UNet, one Phase-2 step | 1.763 s | 0.010 s | **99.4 %** | 4885 -> 1902 MiB (**-61 %**) | bitwise |

Notes:

* Ensemble batching trades memory for speed; `performance.ensemble.chunk_size`
  bounds the trade-off, and member results are **bitwise identical** for any
  chunk size because each member owns a dedicated `torch.Generator`
  (`granitewxc.refinement.base.ChunkNoiseSource`).
* The Phase-1 cache removes the entire frozen Prithvi-UNet forward pass from the
  Phase-2 training step. It is only valid for a frozen Phase 1 and is rejected
  for joint fine-tuning.
* Every entry was verified for numerical parity in the same run;
  `tests/test_refinement_performance.py` re-checks all of them.

### Other applied optimizations

* Diffusion schedules and flow integration grids are built once per
  configuration and kept on the target device (no host sync inside the loops).
* Non-persistent schedule buffers keep refinement checkpoints small.
* Conditioning is computed once and reused across all stochastic steps and all
  ensemble members.
* Loss reduction, residual construction, ensemble mean and spread run in float32
  regardless of the compute dtype.
* Atomic checkpoint writes; single-pass, chunked, compressed NetCDF output.

### Rejected / not enabled by default

| candidate | status | reason |
| --- | --- | --- |
| global TF32 | off by default, opt-in with a warning | changes matmul precision; needs per-case parity validation |
| AMP (`bf16` / `fp16`) | opt-in via `performance.precision.mode` | must be validated against the fp32 baseline for each case |
| `torch.compile` | off by default | recompilation cost dominates for the current domain sizes; kept configurable with an eager fallback |
| approximate / sparse / local attention | **not implemented** | would change the mathematical operation; full spatial attention is mandatory |
| reducing diffusion/flow steps, ensemble size or resolution | **not implemented as an optimization** | scientifically meaningful trade-offs, not workflow tuning |

---

## 13. Phase-1 conditioning cache

The production NARR--PRISM integration uses
`examples/NARR_PRISM/narr_prism_phase1_cache.py`. It runs the authenticated
deterministic `last.ckpt` once for the 1996--2013 training and 2014--2015
validation dates. One atomic NetCDF is written per date beneath the immutable
directory `<performance.phase1_cache.path>/<contract-digest-prefix>/`.

Each daily file contains the canonical full grid and ordered `ppt`, `tmax`,
`tmin` channels for:

* `deterministic_physical`;
* `deterministic_normalized`, obtained by encoding the final blended physical
  Phase-1 field exactly once;
* `residual_target_normalized = encode(PRISM) - deterministic_normalized`;
* the channel-wise residual-valid mask and static PRISM support mask; and
* exact date, split, coordinates and cache-contract identity.

The manifest binds the exact checkpoint-file SHA-256 and tensor-state
fingerprint, Phase-1 semantic contract, ordered predictors and targets,
normalization/scaler signatures, grid, preprocessing, dates, mask, tile core,
stride, halo and blend settings. It remains `incomplete` until the exact daily
inventory is validated. Finalization computes one set of per-channel residual
mean/std/count values from the training split only, with training-tile overlap
multiplicity. The 2014--2015 validation and 2016--2025 inference periods never
fit these statistics.

Build the shared cache once (change the GPU list to the visible devices), then
validate it and train any or all of the four heads:

```bash
PHASE1_CKPT=examples/NARR_PRISM/experiments/checkpoints/narr_prism_California/last.ckpt

mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_phase1_cache.py build \
  --config examples/NARR_PRISM/NARR_PRISM_diffusion_transformer.yaml \
  --checkpoint "$PHASE1_CKPT" --parallel-gpus 0,1,2,3

mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_phase1_cache.py validate \
  --cache examples/NARR_PRISM/experiments/phase1_residual_cache

mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_refinement.py train \
  --config examples/NARR_PRISM/NARR_PRISM_diffusion_transformer.yaml \
  --phase1-checkpoint "$PHASE1_CKPT" --device cuda:0
```

Interrupted builds are resumable: valid completed dates are skipped. Training
fails closed if the manifest is missing, incomplete or incompatible. On a
fresh run, manifest statistics are installed directly and the old complete
live-loader normalization scan is skipped. On resume, checkpoint statistics
are restored first and then required to match the manifest. A configurable
small live parity sample (four tiles by default) compares cached baselines and
residuals with the frozen model before training. The Phase-2 CLI applies the
configured TF32 policy to both CUDA matmul and cuDNN before this check. This is
required: leaving cuDNN TF32 enabled while the cache was built with TF32 off
caused a `0.00225335` normalized mismatch; honoring `allow_tf32: false` reduced
it to `2.74181e-06` without weakening the `1e-5` tolerance. Date-grouped
sampling and a per-worker daily reader ensure that all same-day tiles share one
NetCDF open; Phase 1 is not executed during refinement epochs.

The trainer displays the one-time complete-cache inventory validation before
epoch 1. The notebook avoids a duplicate scan and saves every streamed child
command under its output directory's `command_logs/`; failures include that log
path and a retained traceback tail.

`granitewxc.refinement.cache.Phase1ConditioningCache`, the older generic
one-`.pt`-entry API, remains available to library callers and parity tests. It
is not the cache consumed by the NARR--PRISM production trainer.

---

## 14. Tests

| file | scope |
| --- | --- |
| `tests/test_refinement_config.py` | schema, aliases, validation errors, backward compatibility |
| `tests/test_refinement_models.py` | shapes, training/inference steps, residual construction, normalization round trip, masks, NaNs, rectangular domains, ensembles, freezing |
| `tests/test_refinement_transformer.py` | spatial-only attention, no causal mask, no temporal constructs, patchify/unpatchify, 2-D positional encoding, exact cropping, SDPA-vs-reference parity |
| `tests/test_refinement_timestamps.py` | predictor/target timestamp identity, no offset, no forecast index, process time is internal |
| `tests/test_refinement_checkpoint.py` | key migration, strict loading, refinement-only checkpoints, Phase-1 identity, full resume |
| `tests/test_refinement_performance.py` | serial-vs-batched ensembles, attention kernels, gradient checkpointing, schedules, cached-vs-online conditioning, masked loss |
| `examples/NARR_PRISM/test_narr_prism_phase1_cache.py` | atomic daily files, content hashes, completion/finalization, repair, sharding, build locks and worker cleanup |
| `tests/test_narr_prism_phase1_cache_training.py` | daily cache reader locality, manifest statistics, fail-closed loading and live parity |
| `tests/test_narr_prism_cached_target_io.py` | opt-in one-day target caching, exact tile crops and unchanged direct-I/O fallback |
| `tests/test_prism_training_metadata.py` | date-grouped tile sampling, distributed length and metadata preservation |
| `tests/test_refinement_io.py` | NetCDF products, dimensions, units, coordinates, masks, member ordering, lossless compression |

Run them from the repository root in the project environment with
`mamba run -n Prithvi python -m pytest -q`. Using `python -m pytest` is
intentional: it keeps the repository root on `sys.path` for tests that import
the example entry points.

---

## 15. Known limitations

* Residual standardization is enabled in the four NARR YAMLs. Its one-time
  Phase-1 work is now performed by the resumable shared daily cache builder,
  not repeated by each refinement run.
* The Phase-2 trainer is single-process. DDP/FSDP wrapping of the refiner is not
  implemented; Phase-1 training keeps its existing FSDP path.
* `torch.compile` and mixed precision are wired into the configuration but are
  off by default and have not been parity-validated for every case.
* Production refinement inference uses the Phase-1 canonical tile/halo/blend
  geometry and coordinate-aligned stateless noise. It writes one exact-date,
  exact-grid file per day; `<var>` is the refined ensemble mean and
  `<var>_phase1` is the deterministic baseline.
* The checkpoint, scalers, preprocessing products and deterministic 2016--2025
  daily outputs are available through the external artifact portal. Full
  four-method 1996--2013 training and independent Phase-2 scientific evaluation
  remain to be run. No scientific improvement is claimed from mechanics tests.
* The 7,305 daily training/validation cache files contain about 285 GiB of
  arrays before NetCDF compression. Actual disk use depends on compression and
  must be sized before the one-time build.
