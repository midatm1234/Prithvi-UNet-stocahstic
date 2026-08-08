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

`y_norm` is the `x_pre_inverse` tensor Phase 1 already returns, so
`decode(y_norm) == y_phys` exactly and the round trip introduces no drift.

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
* `prediction_type` is `epsilon` (default), `velocity` (v-prediction) or
  `sample`. All three are inter-convertible through the schedule helpers, which
  is verified in `tests/test_refinement_performance.py`.
* Loss: masked MSE/L1/Huber against the configured parameterisation, reduced in
  float32.
* Sampling: DDIM over `inference_steps` timesteps drawn from the *same* schedule.
  `eta = 0` (default) gives the deterministic probability-flow path; `eta = 1`
  recovers the ancestral DDPM update. The configured sampler is never silently
  replaced with a faster approximation.
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
* **Loss**: masked MSE between `v_theta(x_t, t, cond)` and `u_t`.
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
    joint_finetuning: false     # must be requested explicitly
    train_on_residual: true
    ensemble_size: 10
    loss: mse                   # mse | l1 | huber
    seed: 1234

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
      prediction_type: epsilon  # epsilon | velocity | sample
      schedule: cosine          # cosine | linear | scaled_linear
      eta: 0.0                  # 0 = DDIM, 1 = ancestral DDPM
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
* `joint_finetuning: true` conflicts with an explicit `freeze_phase1: true`;
* `phase1_cache.enabled` requires `freeze_phase1: true` and rejects joint
  fine-tuning;
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

All existing case configurations remain usable unchanged.

---

## 9. Checkpoints and NARR_PRISM compatibility

### Migration

`TwoPhaseDownscalingModel` stores Phase 1 under the `phase1.` prefix and Phase 2
under `refiner.`. The only migration needed for an existing deterministic
checkpoint is that prefix, applied explicitly by
`granitewxc.refinement.checkpoint.migrate_phase1_state_dict`, which also strips
`module.` / `_orig_mod.` wrappers. `LEGACY_PHASE1_KEY_RENAMES` is the single
place where any future rename must be registered — never `strict=False`.

### Controlled loading

`load_phase1_state_dict` raises on any unexpected key, any shape mismatch, and
any missing key that does **not** belong to `refiner.*`. `strict=False` is used
only after that proof.

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
| `refinement` | `refiner.*` only, plus the referenced Phase-1 path **and SHA-256 fingerprint** |
| `combined` | both — portable, used for joint fine-tuning |

Every payload carries: model state, optimizer, scheduler, gradient scaler,
epoch, global step, RNG states, the resolved configuration, the case name, the
precision settings, the refinement type and the Phase-1 identity. Resume
validates the Phase-1 fingerprint and refuses to continue against a different
Phase 1. Checkpoints are written atomically (`os.replace`) by default.

---

## 10. Training and inference workflows

```bash
# 1. Deterministic Phase-1 training (unchanged)
python examples/NARR_PRISM/narr_prism_finetune.py --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml

# 2. Resume deterministic Phase-1 training (unchanged: resume_training in the YAML)

# 3. Phase-2 training from an existing deterministic checkpoint (Phase 1 frozen)
python examples/NARR_PRISM/narr_prism_refinement.py train \
    --config examples/NARR_PRISM/NARR_PRISM_diffusion_unet.yaml --device cuda:0

# 4. Resume Phase-2 training
python examples/NARR_PRISM/narr_prism_refinement.py train \
    --config examples/NARR_PRISM/NARR_PRISM_diffusion_unet.yaml --resume

# 5. Joint Phase-1 + Phase-2 fine-tuning: set in the YAML
#    model.refinement.joint_finetuning: true

# 6. Deterministic inference (unchanged)
python examples/NARR_PRISM/narr_prism_inference.py --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml

# 7/8. Refined inference, one or many ensemble members
python examples/NARR_PRISM/narr_prism_refinement.py infer \
    --config examples/NARR_PRISM/NARR_PRISM_flow_matching_unet.yaml \
    --refinement-checkpoint <refinement_checkpoints>/flow_matching_unet/best.ckpt \
    --ensemble-size 10 --seed 1234 --output narr_refined.nc

# 9. Inspect the resolved configuration
python examples/NARR_PRISM/narr_prism_refinement.py describe --config <config>.yaml

# Smoke test (all four refiners, minimal steps, nothing written)
python examples/refinement_smoke_test.py \
    --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml \
    --checkpoint examples/NARR_PRISM/experiments/checkpoints/narr_prism_California/last.ckpt \
    --device cuda:0 --size 128

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

`granitewxc.refinement.io` writes a single NetCDF file containing, per target
variable:

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
| `tests/test_refinement_models.py` | shapes, training/inference steps, residual construction, normalization round trip, masks, NaNs, rectangular domains, ensembles, freezing |
| `tests/test_refinement_transformer.py` | spatial-only attention, no causal mask, no temporal constructs, patchify/unpatchify, 2-D positional encoding, exact cropping, SDPA-vs-reference parity |
| `tests/test_refinement_timestamps.py` | predictor/target timestamp identity, no offset, no forecast index, process time is internal |
| `tests/test_refinement_checkpoint.py` | key migration, strict loading, refinement-only checkpoints, Phase-1 identity, full resume |
| `tests/test_refinement_performance.py` | serial-vs-batched ensembles, attention kernels, gradient checkpointing, schedules, cached-vs-online conditioning, masked loss |
| `tests/test_refinement_io.py` | NetCDF products, dimensions, units, coordinates, masks, member ordering, lossless compression |

Run them with `pytest tests/ -q`.

---

## 15. Known limitations

* Flow-matching training uses the raw normalized residual scale. If a target
  variable has a very large normalized residual variance the velocity loss will
  be correspondingly large; residual standardization (as in the Aurora
  reference) is **not** enabled here because it changes the objective.
* The Phase-2 trainer is single-process. DDP/FSDP wrapping of the refiner is not
  implemented; Phase-1 training keeps its existing FSDP path.
* `torch.compile` and mixed precision are wired into the configuration but are
  off by default and have not been parity-validated for every case.
* Tiled/halo inference for the Phase-2 refiners reuses the Phase-1 tiling of the
  existing inference scripts; a dedicated tiled stochastic sampler (with
  seam-consistent noise across tiles) is not implemented.
* The `infer` subcommand writes index-based `time`/`lat`/`lon` coordinates when
  the dataloader does not attach real coordinate arrays; the existing
  deterministic inference script remains the reference for fully
  georeferenced NetCDF output.
* The Phase-1 conditioning cache stores dense tensors per sample; disk usage
  scales with the dataset and is the user's responsibility to size.
