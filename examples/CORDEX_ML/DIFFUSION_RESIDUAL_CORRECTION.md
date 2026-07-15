# Diffusion residual correction for CORDEX-ML

This document describes the implemented optional diffusion decoder and its
`residual_diffusion` mode in the CORDEX-ML downscaling models. Residual mode is
intended to model errors around a deterministic baseline instead of generating
the complete target field directly. It is currently experimental; see
[Known limitations](#known-limitations) before using it for production runs.

## Motivation

A deterministic downscaler produces one conditional estimate and is usually
optimized for a point loss. It can smooth small-scale variability and does not
directly represent multiple plausible high-resolution fields. A conditional
diffusion head instead learns a score model in standardized target space and
can generate repeatable stochastic ensemble members.

Residual diffusion narrows the generative task further. Rather than learning
the full high-resolution target distribution, it learns the distribution of
the error around a deterministic baseline. In principle this lets the baseline
carry the large-scale signal while diffusion represents unresolved detail,
bias corrections, and conditional uncertainty.

## Definitions and implemented data flow

Let:

- `c` be the decoder conditioning features with shape `[B, C_cond, h, w]`;
- `y` be the physical target with shape `[B, C_out, H, W]`;
- `S(y)` be the configured target transform into standardized space;
- `b_std` be the deterministic baseline in standardized target space; and
- `r_std = S(y) - stop_gradient(b_std)` be the training residual.

The target transform is the existing per-predictand `zscore`, `divide_only`, or
`log1p` transform. The diffusion head is scaler-agnostic and always operates in
this standardized space.

With `model.diffusion.residual_diffusion: false`, score matching is applied to
`S(y)` and sampling returns a complete standardized prediction.

With `model.diffusion.residual_diffusion: true`, the model:

1. builds a separate two-convolution `baseline_head` from the decoder feature
   channels to `C_out` channels;
2. constrains and standardizes its output to obtain `b_std`;
3. concatenates `b_std` with `c` along the channel dimension;
4. trains the diffusion score model on `r_std`; and
5. returns `b_std + r_sample` from the reverse-diffusion sampler.

The returned inference tensor is therefore the full prediction, not a residual.
It is inverse-transformed into physical units, and configured non-negative
predictands such as precipitation are clamped to zero after decoding.

Important: the baseline is computed under `torch.no_grad()` and is detached in
the residual target. The current training path has no separate baseline loss.
Consequently, the built-in trainer does **not** train `baseline_head`; residual
mode is not presently equivalent to refining the output of a pretrained
deterministic head. This is the primary limitation of the option as currently
implemented.

## Configuration

Select the diffusion decoder with `model.head_type: diffusion` (the older
`model.decoder_type: diffusion` selector is also accepted). Diffusion settings
belong under `model.diffusion`. The implementation also accepts the same keys
directly under `model`, but the nested block is recommended.

### Diffusion defaults

The defaults below come from `DiffusionHeadConfig` in
`granitewxc/decoders/diffusion_head.py`.

| Key | Default | Purpose |
| --- | ---: | --- |
| `sde` | `subvpsde` | Forward SDE: `vpsde`, `subvpsde`, or `vesde`. |
| `beta_min` | `0.1` | VP/subVP minimum beta. |
| `beta_max` | `20.0` | VP/subVP maximum beta. |
| `sigma_min` | `0.01` | VE minimum sigma. |
| `sigma_max` | `50.0` | VE maximum sigma. |
| `num_scales` | `1000` | Training SDE discretization; `num_diffusion_steps` is an alias. |
| `continuous` | `true` | Resolved configuration value for continuous training. |
| `likelihood_weighting` | `false` | Enable likelihood-weighted score loss. |
| `reduce_mean` | `true` | Use a mean rather than summed spatial loss reduction. |
| `eps` | `1.0e-5` | Minimum training time sampled by score matching. |
| `noise_conditioning_scale` | `999.0` | VP/subVP time-label scale passed to the score network. |
| `sampling_method` | `pc` | Sampler: `pc`, `ode`, or `ddim`. |
| `predictor` | `euler_maruyama` | PC predictor: `euler_maruyama`, `reverse_diffusion`, or `none`. |
| `corrector` | `none` | PC corrector: `langevin` or `none`. |
| `snr` | `0.16` | Langevin corrector signal-to-noise ratio. |
| `n_corrector_steps` | `1` | Corrector updates per reverse step. |
| `num_sampling_steps` | `256` | Reverse steps per ensemble member. |
| `probability_flow` | `false` | Use probability-flow dynamics in the PC predictor. |
| `denoise` | `true` | Return the denoised sample where supported. |
| `sampling_eps` | `1.0e-3` | Final reverse-sampling time. |
| `eta` | `0.0` | DDIM stochasticity (`0` deterministic, `1` DDPM-like). |
| `projected_cond_channels` | `128` | Project wide conditioning features to this width; `0` disables projection. |
| `residual_diffusion` | `false` | Model a correction around `baseline_head`. |
| `base_channels` | `64` | Score U-Net base width. |
| `channel_multipliers` | `[1, 2, 2]` | Score U-Net level width multipliers; `channel_mults` is an alias. |
| `num_res_blocks` | `2` | Residual blocks per score U-Net level. |
| `time_embed_dim` | `128` | Diffusion-time embedding width. |
| `dropout` | `0.1` | Score U-Net dropout probability. |
| `fourier_scale` | `16.0` | Gaussian Fourier time-feature scale. |

`continuous` is parsed and retained by the configuration object, but the
current score-loss call does not branch on it. `output_channels` and
`conditioning_channels` may appear as descriptive YAML entries, but model
construction infers both dimensions and does not use those entries to size the
head.

Inference ensemble settings are separate:

| Key | Default | Purpose |
| --- | ---: | --- |
| `inference.ensemble_size` | `1` | Number of diffusion samples. Explicit values are used without a minimum clamp. |
| `inference.base_seed` | `42` | Seed for member 0; `ensemble_seed` is a supported fallback key. |

`ensemble_size` must be at least 1. Deterministic-head inference always makes
one model pass in the included inference helper.

## Enabling and disabling residual correction

Enable both the diffusion decoder and residual mode:

```yaml
model:
  head_type: diffusion
  diffusion:
    residual_diffusion: true
```

Generate full fields with diffusion, without residual mode:

```yaml
model:
  head_type: diffusion
  diffusion:
    residual_diffusion: false
```

Return to the existing deterministic decoder by removing `head_type` or using:

```yaml
model:
  head_type: deterministic
```

Changing `residual_diffusion` changes the module structure because the
`baseline_head` exists only when the value is `true`.

## Training behavior

The same CORDEX model builder creates the selected head. Training and
validation call `model(batch)` without inference return flags:

- deterministic mode returns a prediction for the configured regression loss;
- diffusion mode returns the scalar score-matching loss directly; and
- residual diffusion uses `S(y) - b_std` as the score-matching target and adds
  `b_std` to the score-network conditioning.

Validation computes the score loss and does not run reverse sampling. The
score model receives a noised tensor `[B, C_out, H, W]` and conditioning
features. In residual mode, the conditioning input before optional projection
has `C_cond + C_out` channels.

The current SA workflow is notebook-driven. Set the training notebook's
`config_path` to the diffusion YAML, then execute the notebook:

```text
examples/CORDEX_ML/notebooks/SA_downscaling_finetune_T2_ACCESS-CM2_static.ipynb
```

## Checkpoints and compatibility

Trainer checkpoints save the complete model and include `head_type` and
`diffusion_head` metadata. Resume loading is strict, so the reconstructed model
configuration must match the checkpoint:

- deterministic checkpoints contain deterministic output-head parameters, not
  `diffusion_head.*` parameters;
- full-field diffusion checkpoints contain `diffusion_head.*` but no
  `baseline_head.*` parameters;
- residual-diffusion checkpoints contain both `diffusion_head.*` and
  `baseline_head.*` parameters.

Accordingly, toggling deterministic/diffusion mode or toggling
`residual_diffusion` is not compatible with strict resume loading. The SA
inference script loads with `strict=False` only to omit scaler tensors, then
explicitly rejects other missing or unexpected keys. Use a checkpoint trained
with the same head structure as the inference configuration.

A generic pretrained backbone may still initialize a fine-tuning run through
the repository's existing non-strict backbone-loading path. That does not
convert a deterministic output head into a trained residual baseline.

## Inference and ensemble generation

The inference helper calls the model with `return_pre_inverse=True` and
`return_raw_output=True`, which selects reverse sampling. A single sample has:

```text
physical output:     [B, C_out, H, W]
standardized output: [B, C_out, H, W]
```

For a diffusion checkpoint, `infer_batch_ensemble` repeats sampling
`inference.ensemble_size` times and stacks the result as:

```text
[B, ensemble, C_out, H, W]
```

Before each member, the helper seeds PyTorch, CUDA (when used), and NumPy with
`base_seed + ensemble_index`. DDIM with `eta: 0` remains deterministic for the
same initial random state and conditioning; changing the member seed changes
the initial prior sample. Increasing `eta` adds noise during DDIM updates.

For the SA example, edit the YAML and run the inference script from the
repository root:

```yaml
model:
  head_type: diffusion
  diffusion:
    sampling_method: ddim
    eta: 0.2
    num_sampling_steps: 128
    residual_diffusion: true

inference:
  ensemble_size: 3
  base_seed: 42

precip_model: single_head
```

```powershell
python examples/CORDEX_ML/notebooks/SA_downscaling_inference_T2_ACCESS-CM2_static.py
```

The script's `CONFIG_PATH` currently points to
`SA_T2_ACCESS-CM2_static_diffusion.yaml`. It also forces full-frame inference,
so tiled boundary mitigation is disabled in that script.

## NetCDF dimensions and uncertainty

The included writer maps a diffusion output `[T, E, C_out, H, W]` to one
NetCDF variable per predictand. For the SA `pr` and `tasmax` case, each variable
has dimensions:

```text
(time, ensemble, lat, lon)
```

The file contains an integer `ensemble` coordinate from `0` through
`ensemble_size - 1` and global attributes:

- `head_type = "diffusion"`;
- `ensemble_size`;
- `ensemble_generation = "diffusion_sampling"`; and
- `base_seed`.

Uncertainty is represented by variation across the `ensemble` dimension; the
writer does not calculate or store a separate uncertainty variable. Consumers
must explicitly compute statistics such as ensemble mean, standard deviation,
quantiles, or probabilities. `utils/postprocess_outputs.py` provides
`ensemble_mean_xr()` when a single ensemble-mean field is required.

Deterministic outputs remain `(time, lat, lon)` without an ensemble dimension.

## Known limitations and recommended settings

- **The baseline head is not trained by the current loss path.** Do not assume
  `residual_diffusion: true` refines an existing deterministic prediction. Use
  full-field diffusion (`false`) for supported training, or add and validate a
  baseline supervision/freeze-and-load workflow before relying on residual mode.
- Residual mode cannot be enabled on an existing full-field diffusion or
  deterministic checkpoint with strict loading because its parameter structure
  differs.
- The diffusion head is incompatible with `precip_model: hurdle`; use
  `precip_model: single_head`.
- Sampling cost is approximately proportional to
  `ensemble_size * num_sampling_steps`. Start with 1--3 members and fewer
  reverse steps for pipeline tests, then evaluate quality before increasing
  either value.
- The supplied SA YAML uses DDIM with `eta: 0.2`, 256 sampling steps, and
  `residual_diffusion: false`. These are example choices, not dataclass defaults.
- PC settings `predictor`, `corrector`, `snr`, `n_corrector_steps`, and
  `probability_flow` do not affect `ddim` or `ode` sampling.
- DDIM is implemented for VP/subVP SDEs. With `vesde`, selecting `ddim` falls
  back to the ODE sampler.
- Tiled diffusion can create seams because tiles sample independent noise.
  Prefer full-frame inference when memory permits.
- Ensemble spread is sampling variability, not guaranteed calibrated
  uncertainty. Evaluate spread-skill and probabilistic metrics on held-out data.

For a lightweight shape and finite-loss check, run:

```powershell
python examples/CORDEX_ML/diffusion_smoke_test.py
```
