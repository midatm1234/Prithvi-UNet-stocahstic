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
                                       |  y_phys, y_norm (+ optional features)
                                       v
                    +------- Phase 2 (optional, stochastic) ---------+
 conditioning  ---> | diffusion_unet | flow_matching_unet            | ---> r_model
 (see section 4)    | diffusion_transformer | flow_matching_transf.  |
                    +-----------------------------------------------+
                                       |
 residual denormalize once -> physical correction gate -> y_phys + correction
                                       -> constraints/mask once
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

Phase 2 has one explicit, shared contract for all four heads. The quantity being
corrected is the frozen Phase-1 prediction in **physical units**; Phase 2 never
regenerates the complete predictand field.

```
base_physical            = frozen_phase1(predictors)
residual_physical        = ground_truth_physical - base_physical
residual_model_target    = (residual_physical - residual_mean) / residual_scale
process_output           = refiner(noised_or_interpolated_state, process_time, conditioning)
residual_model_predicted = reverse_sample_or_integrate(process_output, conditioning)
correction_physical      = residual_model_predicted * residual_scale + residual_mean
final_physical           = base_physical + correction_physical
final_physical           = apply_constraints_and_prediction_mask(final_physical)
```

`ResidualNormalizer` fits one mean and standard deviation per output channel
from finite **training-split physical residuals only**. It uses a streaming
estimator, floors the scale by `minimum_scale`, and stores the fitted buffers in
the Phase-2 checkpoint. Validation, test and inference refuse to recompute or
silently substitute these statistics. The round trip
`denormalize(normalize(residual))` is tested independently of the Phase-1 target
scaler.

The Phase-1 normalized tensor is still available for deterministic-output
conditioning and for reporting. It is not used to define or reconstruct the
residual. This distinction is important for nonlinear target transforms such as
precipitation `log1p`: subtracting two values after that transform is not the
same physical correction as `ground_truth - base`.

### Signed-log residual compression for non-negative predictands

`residual_normalization.signed_log_nonnegative_channels` (default `false`,
requires `method: standardize`) applies a signed, invertible compression to
the physical residual of every channel with an enabled non-negativity
constraint (precipitation) **before** standardization:

```
z = sign(r) * log1p(|r| / signed_log_scale)          # forward, fit/normalize
r = sign(z) * expm1(|z|) * signed_log_scale           # exact inverse, denormalize
```

This is the "symlog" transform of Hafner et al., *Mastering Diverse Domains
through World Models* (DreamerV3, arXiv:2301.04104; exact formula verified
against `danijar/dreamerv3`, `embodied/jax/nets.py`), adapted here from
large-dynamic-range RL regression targets to heavy-tailed precipitation
residuals. Precipitation residuals span several orders of magnitude between
dry cells and extreme wet cells; a handful of the heaviest cells otherwise
dominate a per-channel MSE/Huber process loss (measured contribution of the
top ~4% heaviest cells: 77-80% of the diffusion process loss, 41-42% of the
flow process loss — see the 2026-09-09 audit's `ACTUAL_LOSS_ATTRIBUTION.md`),
crowding out gradient signal for the widespread, modest corrections needed at
the domain boundary and in dry regions. The transform is fit and applied
per-channel from training-split residuals only, exactly like the existing
standardization step, and is fully invertible so training, sampling and
physical reconstruction remain mathematically consistent. Channels without a
non-negativity constraint (temperature) are bit-for-bit unaffected, and the
setting changes the refinement checkpoint's scientific-contract fingerprint
only when enabled, so existing (disabled-by-default) checkpoints keep loading
unchanged. See `signed_log_20260910/` under
`artifacts/refinement_validation/` for the compact before/after comparison.

The predicted residual is converted back to physical units exactly once and is
added to `base_physical` exactly once. Each ensemble member is reconstructed in
physical units before aggregation. The ensemble mean is formed as the physical
base plus the mean realized physical correction, and the spread is calculated
from those physical corrections. No normalized/noise-space average is treated
as a physical field.

Transformer refiners additionally have a learnable per-channel correction gate
initialized to zero. Their final convolutional projection is also
zero-initialized. At initialization the effective correction is therefore
exactly zero and `final_physical == base_physical`, even though the raw
diffusion/flow sampler still exposes its latent output to diagnostics. Raw
Huber/multiscale/gradient/mean-bias terms train the sampler output directly.
The gate is calibrated against the target using a **detached** raw prediction,
so it must become nonzero for a useful checkpoint without pushing the sampler
to inflate by `1/gate`. It is not applied inside the sampler, and acceptance
diagnostics inspect the ungated residual so it cannot conceal process errors.

Invalid target cells (NaN) get a residual of exactly zero and are excluded from
the masked loss denominator, so masks never bias the loss magnitude. Inference
does not derive its mask from `batch['y']`: it uses an explicit prediction mask,
or predictor validity when no explicit mask is supplied. Precipitation
non-negativity and other predictand constraints are applied only after final
physical reconstruction.

### Final physical non-negativity and ensemble means

`refinement.nonnegative_ensemble_strategy` controls that one final operation for
channels whose predictand configuration enables non-negativity:

| value | behavior |
| --- | --- |
| `memberwise` (default) | independently replace each final physical member `x` with `max(x, 0)`; this is the backward-compatible behavior |
| `mean_preserving` | project the unbounded physical ensemble to non-negative members whose mean is `max(mean(unbounded_members), 0)` |

The `mean_preserving` projection uses a common-offset, zero-floor water-filling
solve at each channel/grid cell. It preserves member ordering and retains
nonzero spread where the desired non-negative mean permits it; channels without
a non-negativity constraint are bit-for-bit unchanged. This avoids the positive
Jensen shift of independent clipping,
`mean(max(x, 0)) - mean(x)`, without changing the sampler, latent distribution or
residual normalization.

Both strategies first construct every *unbounded physical* member as
`base_physical + correction_physical`. Physical non-negativity is then applied
once. It is never applied to the flow state, diffusion latent, normalized
residual or Phase-1 conditioning. Programmatic inference exposes
`members_unbounded`, `unbounded_ensemble_mean`, `memberwise_clipped_mean` and
`memberwise_clipping_mean_shift`; the SA CORDEX driver persists the last three
mean fields in `<prediction-stem>.constraint_diagnostics.nc`. These diagnostics
make a large post-reconstruction clipping shift visible rather than allowing a
constraint to hide a broken scale or sampler.

A mathematical limitation is explicit: a non-negative ensemble with exactly
zero mean can only contain zeros. Consequently `mean_preserving` has zero spread
at cells where the projected unbounded mean is zero, even if the unbounded
sampler had both positive and negative members. This is not evidence by itself
of stochastic collapse; inspect the unbounded mean, memberwise-clipping shift
and wet-cell spread together. At positive means, projection can still reduce
spread when some members reach the zero floor.

---

## 4. Conditioning

`model.refinement.conditioning` selects which spatial fields are concatenated
along the channel axis and resampled/cropped to the target grid:

| key | tensor |
| --- | --- |
| `deterministic_output` | Phase-1 prediction in its existing normalized target space (conditioning only) |
| `input_predictors` | `batch['x']` |
| `static_fields` | `batch['static_x']`, `batch['static_y']` |
| `masks` | predictor validity mask (1 channel) |
| `prithvi_features` | post-backbone convolution output |
| `unet_features` | final UNet decoder activation |

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
* `prediction_type` is `sample` (default), `epsilon` or `velocity`
  (v-prediction). `sample` directly predicts the clean normalized physical
  residual and is the schema-v2 default because clean reconstruction from a
  small epsilon error is ill-conditioned at the highest-noise cosine
  timesteps. All three parameterisations remain supported and are
  inter-convertible through the schedule helpers, which is verified in
  `tests/test_refinement_performance.py`.
* Loss: masked MSE/L1/Huber against the configured parameterisation, reduced in
  float32.
* Sampling: DDIM over `inference_steps` timesteps drawn from the *same* schedule,
  explicitly including trained indices `training_timesteps - 1` and `0`.
  `eta = 0` (default) gives a deterministic path for a fixed initial latent;
  positive `eta` adds the DDIM variance term on non-terminal steps. The
  configured sampler is never silently replaced with a faster approximation.
* Optional `clip_sample` clamps the predicted clean residual to
  `+/- clip_sample_range` in residual-model space.
* The terminal DDIM step adds no fresh noise. Tests exercise the reverse formula
  with an oracle epsilon target and all supported prediction parameterisations.

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
* **Target distribution** `p1`: the standardized physical-residual distribution.
* **Path**: regularized straight interpolation
  `x_t = [1-(1-sigma_min)t]*x0 + t*r`.
* **Velocity target**: `u_t = r - (1-sigma_min)*x0` (constant along each
  conditional path).
* **Loss**: masked MSE between `v_theta(x_t, t, cond)` and `u_t`.
* **Optional deterministic mean-path anchor**:
  `mean_path_loss_weight` is non-negative and defaults to `0.0`, which leaves
  the standard stochastic Gaussian flow-matching objective unchanged. When it
  is positive, the same vector field is additionally evaluated on the
  zero-source trajectory `x0 = 0`, `x_t = t*r`, `u_t = r`, with a masked Huber
  loss. This gives the zero-source/conditional-mean diagnostic path a coherent
  residual anchor without gating samples, changing source variance or changing
  the inference ODE.
* **Flow-time sampling**: `uniform`, or `logit_normal` (sigmoid of
  `N(mean, std)`, default `mean = -0.5`, `std = 1.2`) which concentrates
  training on informative mid-noise levels.
* **Flow-time embedding**: the same sinusoidal embedding used by the diffusion
  refiners, applied to `t * 1000` so a single implementation serves both.
* **Initial state**: `x0 ~ N(0, I)` drawn from the caller's `torch.Generator`
  when `stochastic_initialization: true` (default), otherwise `x0 = 0` for a
  deterministic zero-source-path corrector.
* **Integration**: `dx/dt = v_theta(x, t, cond)` from `t = 0` to `t = 1` over
  `integration_steps` uniform steps. Solvers: `euler` (default, safe),
  `midpoint`, `heun`. The integration grid is built once per call on the target
  device.
* At the endpoint, the same regularized-path algebra converts state and
  velocity to the clean residual:
  `r_hat = (1-sigma_min)*x_1 + sigma_min*v_theta(x_1,1,cond)`.
  It is inverse standardized exactly once before physical reconstruction.

---

## 7. Spatial Transformer and 2-D positional encoding

`granitewxc.refinement.backbones.SpatialResidualTransformer` is a DiT-style
network. Per sample:

1. Assert `[B, C, H, W]` state and conditioning layouts, equal batch/grid
   shapes and the configured channel counts. Replicate-pad only the trailing
   latitude/longitude edges to a multiple of the patch stride.
2. Pass the state and conditioning through separate local 3x3 convolutional
   stems.
3. Apply overlapping convolutional patch embeddings with
   `kernel = 2 * patch_size - 1` and `stride = patch_size`. This keeps one
   token per padded patch cell while neighboring tokens see overlapping spatial
   context.
4. Flatten the two token grids in row-major `(lat, lon)` order and add the
   conditioning tokens to the state tokens.
5. Add 2-D positional information:
   * `learned_2d` (default): a separable pair of learned tables, one per axis,
     summed — parameter count is linear in the grid extent while every
     `(lat, lon)` token still gets a distinct embedding.
   * `sincos_2d`: fixed sin/cos, half the width per axis, cached per grid shape.
6. `num_blocks` x [ multi-head **spatial** self-attention -> MLP ], pre-norm,
   with residual connections and **adaptive layer normalisation** carrying the
   diffusion timestep / flow time. Spatial conditioning is re-injected at every
   block instead of being supplied only once.
7. Reshape tokens back to the 2-D grid, bilinearly resize to the padded full
   resolution, and decode with local convolutions plus full-resolution state and
   conditioning stem skips.
8. Apply a zero-initialized 3x3 output projection and crop trailing padding
   exactly to the original `(H, W)`.

There is no independent linear decoder for pixels inside non-overlapping
patches, no transposed convolution and no post-hoc smoothing. Full-domain
bidirectional attention provides the global receptive field; the overlapping
stem/decoder supplies the local spatial inductive bias needed for coherent
precipitation corrections.

Guarantees (all covered by `tests/test_refinement_transformer.py`):

* attention mixes only spatial tokens — perturbing one batch element cannot
  change another;
* attention is bidirectional (no causal mask);
* rectangular and non-patch-divisible grids work; padding/cropping preserves the
  exact output `(H, W)` and latitude/longitude orientation;
* process-time conditioning modulates values but never permutes token positions;
* the public `patchify_2d` / `unpatchify_2d` helpers are exact inverses and
  lock down the canonical BCHW, channel and row-major ordering;
* `optimized_attention: sdpa` and the reference `math` path agree to
  `atol = rtol = 1e-5` in fp32.

Configurable: `patch_size` (shared or `[height, width]`), `embedding_dim`,
`num_heads`, `num_blocks`, `mlp_ratio`, `dropout`, `positional_encoding`,
`max_tokens_lat/lon`, `gradient_checkpointing`, `optimized_attention` and
`zero_init_output` (default `true`).
Validation rejects `embedding_dim % num_heads != 0`, odd head dimensions,
malformed patch settings, and token grids larger than the learned positional
tables.

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
    joint_finetuning: false     # required for active Phase 2
    train_on_residual: true
    ensemble_size: 10
    loss: mse                   # mse | l1 | huber
    reconstruction_loss_weight: 0.25
    multiscale_loss_weight: 0.10
    gradient_loss_weight: 0.05
    mean_bias_loss_weight: 0.01
    nonnegative_ensemble_strategy: memberwise  # memberwise | mean_preserving
    seed: 1234

    residual_normalization:
      method: standardize       # standardize | identity
      epsilon: 1.0e-6
      minimum_scale: 1.0e-4
      require_fitted: true

    conditioning:
      deterministic_output: true
      input_predictors: true
      prithvi_features: false
      unet_features: false
      static_fields: true
      masks: true

    diffusion:                  # diffusion_* only
      training_timesteps: 1000
      inference_steps: 50
      prediction_type: sample   # sample | epsilon | velocity
      schedule: cosine          # cosine | linear | scaled_linear
      cosine_s: 0.008
      eta: 0.0                  # 0 deterministic; >0 adds nonterminal DDIM variance
      clip_sample: false
      clip_sample_range: 10.0

    flow_matching:              # flow_matching_* only
      integration_steps: 50
      solver: euler             # euler | midpoint | heun
      source_distribution: gaussian
      stochastic_initialization: true
      time_sampling: uniform    # uniform | logit_normal
      sigma_min: 1.0e-4
      logit_normal_mean: -0.5
      logit_normal_std: 1.2
      mean_path_loss_weight: 0.0

    transformer:                # *_transformer only
      patch_size: 8             # shipped recipe override; code default is [4, 4]
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
      zero_init_output: true

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
    enabled: false
    path: null
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
* every active refinement requires `train_on_residual: true`,
  `freeze_phase1: true` and `joint_finetuning: false`; changing the base while
  defining residual targets and fitting their statistics is rejected;
* `phase1_cache.enabled` requires `freeze_phase1: true` and rejects joint
  fine-tuning;
* scientific settings (`model.refinement`) and workflow settings (`performance`)
  are strictly separate, and performance settings never change architecture,
  loss, sampler, solver, ensemble size or evaluation data;
* the fully resolved configuration is stored in every checkpoint under
  `resolved_config`.

The four spatial weights apply to the clean residual reconstructed from the
diffusion/flow training state. The reconstruction, 2x/4x pooled residual,
horizontal/vertical residual-gradient and per-sample/channel mean-bias terms use
Huber comparisons against the **true residual structure**. They do not penalize
high-frequency content simply for existing and are not total-variation or
post-processing smoothers. Set a weight to `0` to disable that specific term.
The configured `loss` remains the primary diffusion-parameter or vector-field
objective.

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

Deterministic case configurations remain usable unchanged. The shipped active
Phase-2 examples now state the residual normalization, reconstruction-loss
weights and Transformer zero initialization explicitly.

---

## 9. Checkpoints, strict loading and migration

### Migration

`TwoPhaseDownscalingModel` stores Phase 1 under the `phase1.` prefix and Phase 2
under `refiner.`. The only migration needed for an existing deterministic
checkpoint is that prefix, applied explicitly by
`granitewxc.refinement.checkpoint.migrate_phase1_state_dict`, which also strips
`module.` / `_orig_mod.` wrappers. `LEGACY_PHASE1_KEY_RENAMES` is the single
place where any future rename must be registered — never `strict=False`.

### Controlled loading

`load_phase1_state_dict` raises on any unexpected key, any shape mismatch, and
any missing key that does **not** belong to the new `refiner.*` or
`residual_normalizer.*` modules. `strict=False` is used internally only after
that proof.

`load_refinement_state_dict` is strict. It reports missing, unexpected and
shape-mismatched keys and refuses to load unless every Phase-2 trainable tensor
and residual-normalizer buffer is present. It also verifies:

* checkpoint kind and schema;
* exact refinement type;
* the scientific-contract version and fingerprint;
* process parameterization and sampler/solver configuration;
* Transformer or U-Net architecture configuration;
* residual-normalization configuration, state keys and state fingerprint; and
* the SHA-256 identity of the frozen Phase-1 weights.

### Schema-2 boundary

The physical-residual repair introduces refinement checkpoint schema 2 and
scientific contract `physical_ground_truth_minus_phase1_v1`. Schema-1
**Phase-2** checkpoints modeled a different normalized-target delta and are
deliberately rejected. They cannot be converted by renaming keys or loaded with
`strict=False`; retrain those refinement heads with the revised YAML and frozen
Phase-1 checkpoint. In particular, the repaired overlapping-patch Transformer
has intentionally different parameter shapes from the former independent
patch-pixel decoder.

Versioned schema-1 deterministic **Phase-1** checkpoints remain valid read-only
inputs. They are migrated explicitly, fingerprinted, frozen and verified for
numerical parity; they are not rewritten.

### Verified parity

```
python examples/NARR_PRISM/narr_prism_refinement_parity.py \
    --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml \
    --checkpoint examples/NARR_PRISM/experiments/checkpoints/narr_prism_California/last.ckpt \
    --device cuda:0 --size 256
```

Result on the checkpoint that is currently training
(`narr_prism_California/last.ckpt`, epoch 17, fingerprint `4f9c5eb7...`,
208 tensors, 208 renames, 0 missing / 0 unexpected / 0 mismatched):

| space | max abs diff | mean abs diff | max rel diff | verdict |
| --- | --- | --- | --- | --- |
| normalized (pre-Phase-2) | 0.0 | 0.0 | 0.0 | bitwise identical |
| physical (post-decode) | 0.0 | 0.0 | 0.0 | bitwise identical |

### Checkpoint kinds

| kind | contents |
| --- | --- |
| `phase1` | `phase1.*` only |
| `refinement` | `refiner.*` and `residual_normalizer.*`, plus the referenced Phase-1 path and **SHA-256 fingerprint** |
| `combined` | both Phase 1 and Phase 2 for a portable strict resume; the active residual contract still keeps Phase 1 frozen |

Every Phase-2 payload carries: model state, optimizer, scheduler, gradient
scaler, epoch, global step, RNG states, the resolved configuration, the case
name, precision settings, refinement type, frozen Phase-1 identity, scientific
contract/fingerprint, and residual-normalizer keys/fingerprint. Resume validates
all of these before loading any Phase-2 state and refuses to continue against a
different Phase 1. Checkpoints are written atomically (`os.replace`) by
default.

---

## 10. Training and inference workflows

Run commands from the repository root. The examples below use the tested
`Prithvi` mamba environment.

```bash
# 1. Deterministic Phase-1 training (unchanged)
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_finetune.py \
    --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml

# 2. Resume deterministic Phase-1 training (unchanged: resume_training in the YAML)

# 3. Phase-2 training from an existing deterministic checkpoint (Phase 1 frozen)
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_refinement.py train \
    --config examples/NARR_PRISM/NARR_PRISM_diffusion_unet.yaml --device cuda:0

# 4. Resume Phase-2 training
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_refinement.py train \
    --config examples/NARR_PRISM/NARR_PRISM_diffusion_unet.yaml --resume

# 5. Deterministic inference (unchanged)
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_inference.py \
    --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml

# 6. Refined inference, one or many ensemble members
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_refinement.py infer \
    --config examples/NARR_PRISM/NARR_PRISM_flow_matching_unet.yaml \
    --refinement-checkpoint <refinement_checkpoints>/flow_matching_unet/best.ckpt \
    --ensemble-size 10 --seed 1234 --output narr_refined.nc

# 7. Inspect the resolved configuration
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_refinement.py describe \
    --config <config>.yaml

# Smoke test (all four refiners, minimal steps, nothing written)
mamba run -n Prithvi python examples/refinement_smoke_test.py \
    --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml \
    --checkpoint examples/NARR_PRISM/experiments/checkpoints/narr_prism_California/last.ckpt \
    --device cuda:0 --size 128

# Deterministic checkpoint parity
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_refinement_parity.py \
    --config ... --checkpoint ...

# Performance benchmarks
mamba run -n Prithvi python examples/refinement_benchmark.py \
    --device cuda:0 --size 256 --ensemble 8 --steps 20
```

The default Phase-2 workflow loads a deterministic checkpoint, freezes Phase 1,
keeps it in `eval()`, generates the conditioning under `torch.no_grad()`, and
trains the refiner on physical residuals. Before the first epoch it makes one
training-only pass to fit the residual statistics. Inference calls
`model.eval()` and `torch.no_grad()`; dropout and stochastic depth are off,
and randomness comes only from the explicit diffusion/flow initial state.

### CORDEX Phase-2 training

The non-notebook entry point reuses the same CORDEX dataset/model construction,
`RefinementTrainer` and schema-2 checkpoint format. It refuses to overwrite an
existing checkpoint directory unless resume is requested. When the YAML points
training and validation at the same files, it creates a chronological holdout
instead of reporting training data as validation (default final 10%, configurable
with `--validation-fraction`). The `--tiny-overfit` mode is the deliberate
exception.

```bash
# Fresh diffusion-Transformer training into a new schema-2 directory
mamba run -n Prithvi python examples/CORDEX_ML/cordex_refinement_training.py \
    --config examples/CORDEX_ML/SA_T2_ACCESS-CM2_static_diffusion_transformer.yaml \
    --device cuda:0 \
    --checkpoint-dir runs/SA_T2_ACCESS-CM2_static_train/SA_T2_ACCESS-CM2_static_twophase/checkpoints/SA_T2_ACCESS-CM2_static_diffusion_transformer_schema2_fixed

# Fresh flow-Transformer training with the fixed SA precipitation-mean recipe
mamba run -n Prithvi python examples/CORDEX_ML/cordex_refinement_training.py \
    --config examples/CORDEX_ML/SA_T2_ACCESS-CM2_static_flow_matching_transformer.yaml \
    --device cuda:0 \
    --checkpoint-dir runs/SA_T2_ACCESS-CM2_static_train/SA_T2_ACCESS-CM2_static_twophase/checkpoints/SA_T2_ACCESS-CM2_static_flow_matching_transformer_schema2_precip_mean_fixed

# Resume last.ckpt in the same directory, targeting 100 total epochs
mamba run -n Prithvi python examples/CORDEX_ML/cordex_refinement_training.py \
    --config examples/CORDEX_ML/SA_T2_ACCESS-CM2_static_diffusion_transformer.yaml \
    --device cuda:0 \
    --checkpoint-dir runs/SA_T2_ACCESS-CM2_static_train/SA_T2_ACCESS-CM2_static_twophase/checkpoints/SA_T2_ACCESS-CM2_static_diffusion_transformer_schema2_fixed \
    --resume --epochs 100

# One-batch coherence/overfit diagnostic (writes to checkpoint_dir/tiny_overfit)
mamba run -n Prithvi python examples/CORDEX_ML/cordex_refinement_training.py \
    --config examples/CORDEX_ML/SA_T2_ACCESS-CM2_static_diffusion_transformer.yaml \
    --device cuda:0 --tiny-overfit --epochs 20 --learning-rate 1e-3

# Bounded smoke run on configured predictor/target pair zero
mamba run -n Prithvi python examples/CORDEX_ML/cordex_refinement_training.py \
    --config examples/CORDEX_ML/SA_T2_ACCESS-CM2_static_diffusion_transformer.yaml \
    --device cuda:0 --case-index 0 --max-train-batches 2 \
    --max-val-batches 2 --checkpoint-dir runs/refinement_smoke/schema2
```

The four shipped SA defaults use distinct checkpoint directories:

| head | checkpoint-directory suffix |
| --- | --- |
| `diffusion_transformer` | `diffusion_transformer_schema2_fixed` |
| `diffusion_unet` | `diffusion_unet_schema2_fixed` |
| `flow_matching_transformer` | `flow_matching_transformer_schema2_precip_mean_fixed` |
| `flow_matching_unet` | `flow_matching_unet_schema2_precip_mean_fixed` |

The two fixed SA flow recipes deliberately set
`flow_matching.mean_path_loss_weight: 0.25` and
`nonnegative_ensemble_strategy: mean_preserving`. The auxiliary path weight
anchors coherent central precipitation corrections, while the final projection
prevents independent clipping from turning symmetric dry-cell variability into
a positive ensemble-mean bias. Both are recipe-specific overrides; the library
defaults remain `0.0` and `memberwise`. These settings change the serialized
scientific contract, so start in the new `*_schema2_precip_mean_fixed`
directories rather than resuming an older flow checkpoint. The shipped YAMLs
also disable legacy notebook resume fields.
Use `--help` for the complete CLI. `--batch-size`,
`--gradient-accumulation-steps` and `--learning-rate` are explicit runtime
overrides; when omitted, their YAML values are retained and recorded in the
schema-2 checkpoint.

### CORDEX inference and strict evaluation

The SA inference driver accepts an explicit YAML and schema-2 checkpoint through
environment variables. This PowerShell example runs its configured active case
(historical perfect ACCESS-CM2 by default):

```powershell
$env:GRANITE_REFINEMENT_CONFIG = (Resolve-Path "examples/CORDEX_ML/SA_T2_ACCESS-CM2_static_diffusion_transformer.yaml").Path
$env:GRANITE_REFINEMENT_CHECKPOINT = (Resolve-Path "examples/CORDEX_ML/runs/SA_T2_ACCESS-CM2_static_train/SA_T2_ACCESS-CM2_static_twophase/checkpoints/SA_T2_ACCESS-CM2_static_diffusion_transformer_schema2_fixed/best.ckpt").Path
$env:GRANITE_REFINEMENT_ENSEMBLE_SIZE = "10"  # optional; otherwise use YAML
# Optional; the default already appends _schema2_fixed and preserves legacy outputs.
$env:GRANITE_REFINEMENT_OUTPUT_HEADER = "SA_T2_ACCESS-CM2_static_diffusion_transformer_schema2_fixed"
mamba run -n Prithvi python examples/CORDEX_ML/notebooks/SA_downscaling_inference_T2_ACCESS-CM2_static.py
```

The driver writes physical ensemble members and a paired `.baseline.nc`
Phase-1 file with provenance attributes. It joins predictor and target samples
by exact civil timestamp rather than positional index; target-only leap days are
not paired to no-leap predictor data. Precipitation flux truth is converted to
`mm/day` exactly once. Before inference it preflights the complete NetCDF/pickle
artifact bundle and refuses any existing destination; choose a new
`GRANITE_REFINEMENT_OUTPUT_HEADER` instead of overwriting another run.

All four fixed SA YAMLs request full-frame inference, and the SA driver enforces
and logs it. The complete 128 x 128 domain therefore receives one spatially
coherent stochastic draw per member; the configured 96 x 96 tiling, Hann blend
and seam-deblocking settings are bypassed. No independently sampled tiles are
stitched together.

For every run the artifact bundle also contains
`<prediction-stem>.constraint_diagnostics.nc`, with the physical unbounded
ensemble mean, the mean that ordinary independent memberwise clipping would
produce, and their clipping-induced difference for every output channel. The
main prediction NetCDF always contains the selected, finally constrained
physical members; ensemble aggregation never uses normalized or latent values.

Evaluate the matched 1981-2000 period with the same variables, units, mask and
color scales:

```powershell
$checkpoint = (Resolve-Path "examples/CORDEX_ML/runs/SA_T2_ACCESS-CM2_static_train/SA_T2_ACCESS-CM2_static_twophase/checkpoints/SA_T2_ACCESS-CM2_static_diffusion_transformer_schema2_fixed/best.ckpt").Path
$prediction = "examples/CORDEX_ML/runs/SA_T2_ACCESS-CM2_static_train/SA_T2_ACCESS-CM2_static_twophase/predictions/SA_T2_ACCESS-CM2_static_diffusion_transformer_schema2_fixed/historical/perfect/Predictions_pr_tasmax_ACCESS-CM2_1981-2000.nc"
$baseline = $prediction -replace "\.nc$", ".baseline.nc"
mamba run -n Prithvi python examples/CORDEX_ML/utils/evaluate_refinement_outputs.py `
    --prediction $prediction --baseline $baseline `
    --target D:/CORDEX/SA_domain/test/historical/target/pr_tasmax_ACCESS-CM2_1981-2000.nc `
    --variables pr tasmax --expected-refinement-type diffusion_transformer `
    --expected-checkpoint $checkpoint `
    --expected-case SA_T2_ACCESS-CM2_static_diffusion_transformer_schema2_fixed `
    --start-date 1981-01-01 --end-date 2000-12-31 `
    --wet-day-threshold 1.0 --spectral-bins 8 `
    --output-dir examples/CORDEX_ML/evaluations/SA_diffusion_transformer_schema2_fixed
```

The evaluator requires `--expected-checkpoint` and refuses the wrong output
kind, refinement type, case, checkpoint, checkpoint schema, residual-contract
version, unfitted/missing or channel-reordered residual normalization,
refinement-checkpoint file SHA-256, scientific-contract or normalizer-state
fingerprint, Phase-1 checkpoint/fingerprint, dimension order, missing
physical units or timestamp mismatch. It writes a
JSON summary, CSV metrics, matched-period eight-panel climatology/correction
maps and representative member/spread maps. Metrics include bias, MAE, RMSE,
spatial pattern correlation, climatological bias, precipitation wet-day
frequency/intensity, quantile and 95th/99th-percentile errors, spatial-gradient
agreement, spatial autocorrelation, spectral power by scale, ensemble-mean
skill, spread and member diversity.

For another head, change the inference config, refinement checkpoint, distinct
schema-2 `*_fixed` output header/prediction path, `--expected-refinement-type`,
`--expected-checkpoint`, `--expected-case`, and evaluation output directory as
one consistent set. The driver accepts collision-safe `_schema2_fixed` and
`_schema2_<case>_fixed` names (including `_schema2_precip_mean_fixed`) and
rejects legacy or unsafe overrides.

After training, the CORDEX CLI also prints a fixed-seed validation-batch report
from `granitewxc.refinement.diagnostics` for ground truth, Phase 1, true
physical/normalized residual, sampled source and intermediate process state,
process target and prediction, configured physical ensemble members, unbounded
members/mean, predicted correction, final refinement, ensemble spread and the
historical memberwise-clipping shift. Each entry includes shape,
channel names, units, dimension names, dtype, min/max, mean/std,
1st/5th/25th/50th/75th/95th/99th percentiles, zero fraction,
lag-one spatial autocorrelation,
NaN/Inf counts, spatial-gradient magnitude and high-frequency spectral-power
fraction. Persisted reports also contain sampled timestep/loss summaries,
per-channel negative-member fractions, fixed-seed Phase-1 and effective
refinement correlation/MAE/RMSE/bias with refined-minus-Phase-1 deltas, while
keeping raw ungated sampler metrics explicitly separate. Every report records
the exact split/sample timestamp and paths, resolved configuration, epoch/step,
checkpoint schema and SHA-256, contract/Phase-1/normalizer fingerprints and
normalizer payload. A full-hash-named immutable JSON companion is retained next
to the convenience `diagnostics.json`; stale `best.ckpt` files are not selected
silently. The tensor helper accepts NumPy arrays or PyTorch tensors and can be
reused by all four heads without changing sampling or outputs.

---

## 11. Output

`granitewxc.refinement.io` writes a single NetCDF file containing, per target
variable:

`<var>` (deterministic), `<var>_residual` (effective physical correction,
`refined - deterministic`), `<var>_refined`,
`<var>_members` (with an explicit `member` dimension in draw order),
`<var>_ensemble_mean`, `<var>_ensemble_spread` (unbiased) and `<var>_truth` when
the workflow already saves it.

`<var>_residual` inherits the predictand's physical units. Raw
residual-model-space samples remain available through `TwoPhaseOutput.residual`
for diagnostics but are never mislabeled or written as a physical correction.

Timestamps, coordinates, variable names, units, calendar, masks, fill values,
attributes, latitude/longitude orientation and ensemble ordering are preserved.
Compression and chunking are configurable and lossless; the file is written in a
single pass and, by default, atomically.

Each member is inverse residual-normalized and reconstructed in physical units
before ensemble aggregation. Mean and spread are accumulated in float32 and
skip NaN (masked) cells so masked points never contaminate the statistics.

---

## 12. Performance

The table below records the original workflow-performance experiment. It
predates the schema-2 residual contract and repaired overlapping-patch
Transformer, so rerun `examples/refinement_benchmark.py` before using the
Transformer timings for capacity planning. These timings are not scientific
skill evidence.

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

* Diffusion schedule coefficients are built once per configuration and kept on
  the target device. A flow integration grid is allocated once per integration
  call on that device (no host-built per-step grid).
* Non-persistent schedule buffers keep refinement checkpoints small.
* Conditioning is computed once and reused across all stochastic steps and all
  ensemble members.
* Loss reduction, physical residual construction, ensemble mean and spread run
  in float32 regardless of the compute dtype.
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

Enabled with `performance.phase1_cache`. One file per sample under
`<path>/<digest>/`, where `<digest>` covers:

Phase-1 checkpoint fingerprint, Phase-1 architecture, case name, dataset split,
predictor definitions, target definitions, pressure levels, normalization
configuration, preprocessing configuration, spatial domain, grid shape and the
cache-schema version.

Each entry stores the sample identifier, timestamp, coordinates, the
deterministic normalized output, the requested Prithvi/UNet features, masks and
metadata.

Invalidation rules:

* any change to the fields above changes the digest, hence a different directory;
* a manifest that disagrees with the current run raises `RuntimeError`;
* a schema-version change raises;
* an entry whose stored digest disagrees with the expected one raises;
* `validate_cache: true` (default) compares cached tensors with a live frozen
  Phase-1 pass and fails above `validate_tolerance`;
* the cache is **never** usable with `joint_finetuning: true` or an unfrozen
  Phase 1 — the configuration validator rejects that combination.

---

## 14. Tests

| file | scope |
| --- | --- |
| `tests/test_refinement_config.py` | schema, aliases, validation errors, backward compatibility |
| `tests/test_refinement_models.py` | shapes, training/inference steps, physical residual construction, zero-refinement identity, diffusion oracle reverse step, flow ODE direction/integration, masks, deterministic seeds, ensembles, freezing |
| `tests/test_refinement_normalization.py` | training-only streaming residual statistics, fitted-state requirement, masked values and physical round trip |
| `tests/test_refinement_transformer.py` | spatial-only attention, no causal mask, overlapping stems/decoder, patchify/unpatchify ordering, 2-D positions, exact cropping, zero projection/gate, SDPA-vs-reference parity |
| `tests/test_refinement_timestamps.py` | predictor/target timestamp identity, no offset, no forecast index, process time is internal |
| `tests/test_refinement_checkpoint.py` | Phase-1 migration, schema-2 contract/type/config/normalizer fingerprints, strict trainable-key loading, Phase-1 identity, full resume |
| `tests/test_refinement_performance.py` | serial-vs-batched ensembles, attention kernels, gradient checkpointing, schedules, cached-vs-online conditioning, masked loss |
| `tests/test_refinement_io.py` | physical correction metadata/units, NetCDF products, dimensions, coordinates, masks, member ordering, lossless compression |
| `tests/test_refinement_diagnostics.py` | tensor statistics, quantiles, invalid counts, gradients and high-frequency spectral-power fraction |
| `tests/test_cordex_refinement_evaluation.py` | strict output/checkpoint selection, exact timestamp join, units, metrics and evaluation artifacts |
| `tests/test_cordex_dataset_contract.py` | timestamp pairing, canonical layout and one-time precipitation-unit conversion |
| `tests/test_refinement_structure_losses.py` | truth-matched multiscale, gradient and mean-bias objectives |
| `tests/test_refinement_transformer_overfit.py` | ungated full-sampler/integrator coherent-residual overfit for both Transformers |
| `tests/test_cordex_refinement_training_cli.py` | bounded loaders, chronological holdout and explicit CLI controls |

Run them with `mamba run -n Prithvi pytest tests/ -q`.

---

## 15. Known limitations

* The Phase-2 trainer is single-process. DDP/FSDP wrapping of the refiner is not
  implemented; Phase-1 training keeps its existing FSDP path.
* `torch.compile` and mixed precision are wired into the configuration but are
  off by default and have not been parity-validated for every case.
* Tiled/halo inference for the Phase-2 refiners reuses the Phase-1 tiling of the
  existing inference scripts; a dedicated tiled stochastic sampler (with
  seam-consistent noise across tiles) is not implemented.
* The generic CORDEX Phase-2 trainer consumes both SA and NZ recipes, but the
  strict Phase-2 NetCDF inference driver is currently SA-specific. MERRA_PRISM
  refinement recipes do not yet have a shipped trainer/inference entry point;
  NARR_PRISM has `narr_prism_refinement.py`. Library contracts and tests are
  shared, but those missing domain runners remain explicit workflow gaps.
* The shipped NZ recipes point to the available `D:/CORDEX/NZ_domain` data in
  this workspace, but their configured deterministic NZ Phase-1 checkpoint is
  not present here. Supply that prerequisite before training either NZ refiner;
  the CLI intentionally fails instead of substituting an SA or generic model.
* The `infer` subcommand writes index-based `time`/`lat`/`lon` coordinates when
  the dataloader does not attach real coordinate arrays; the existing
  deterministic inference script remains the reference for fully
  georeferenced NetCDF output.
* The Phase-1 conditioning cache stores dense tensors per sample; disk usage
  scales with the dataset and is the user's responsibility to size.
* Schema-1 Phase-2 weights cannot be scientifically reinterpreted under the
  physical-residual contract and require retraining.
* Unit, formulation and regression tests establish implementation correctness;
  they do not replace held-out scientific validation. Treat a new checkpoint as
  provisional until bias/RMSE, pattern correlation, wet-day/extreme statistics,
  spatial gradients/autocorrelation/spectra and ensemble behavior have been
  compared against the unchanged Phase-1 baseline.


## 2026-09-09 four-option CORDEX audit

The [four-option audit report](../artifacts/refinement_validation/four_option_20260909/REPORT.md) records verified fixes, all-eight paired subset results, existing-checkpoint non-regression, full-period flow diagnostics, and unresolved scientific acceptance. The short conditioning ablation does not justify promoting a new default. [Reproduction commands](../artifacts/refinement_validation/four_option_20260909/COMMANDS.md) and isolated configurations cover all four options.


## Temporal branch compatibility

Branch `Prithvi-UNet_temporal_model` adds a sequence-conditioned deterministic
model. It does **not** change any refinement head, and the temporal deterministic
model is a usable baseline on its own -- stochastic refinement is not a
prerequisite for using it.

All four heads (`diffusion_unet`, `diffusion_transformer`, `flow_matching_unet`,
`flow_matching_transformer`) are reused unchanged. Compatibility is governed by one
key:

| `temporal.refinement.temporal_conditioning` | extra `cond_channels` | existing Phase-2 checkpoints |
|---|---|---|
| `none` (**shipped default**) | 0 | valid, byte-identical path |
| `time_features` | +5 | invalid -- retrain Phase 2 |
| `latent_state` | +21 | invalid -- retrain Phase 2 |

Widening the conditioning changes the refiner's input projection, so the weights no
longer describe the same model. `check_refiner_temporal_compatibility` raises with
the exact channel counts and names the two valid actions; the existing
`cond_channels` guard in `two_phase.py::initialize_refiner` provides the same check
independently. Nothing is loaded partially.

Refinement is **causal per date**, not whole-sequence: each date is refined
conditioned on its temporal metadata and, optionally, the Phase-1 temporal latent,
with the noise state carried forward. Noise is `iid_per_frame` by default
(bit-for-bit the existing behaviour). `ar1_correlated` with `0 < rho < 1` gives each
ensemble member a temporally coherent trajectory while keeping every frame's
marginal exactly N(0,1), so the refiner still sees the distribution it was trained
against. `rho = 1` (identical noise at every date) is **rejected**: it manufactures
persistence rather than modelling it. Each member keeps its own generator *and* its
own AR(1) state, the same contract as `ChunkNoiseSource`.

See [temporal_model_architecture.md](temporal_model_architecture.md) section 10.
