# Joint residual diffusion for CORDEX-ML

> **Legacy CORDEX pipeline.** This document and its companion YAML/scripts are
> retained for backward compatibility with `origin/CORDEX_ML_diffusion_head`.
> New NARR–PRISM, MERRA–PRISM, and CORDEX work should use the shared
> `granitewxc.refinement` package and the explicitly named
> `*_diffusion_unet`, `*_diffusion_transformer`, and flow-matching configs.
> These files coexist additively and do not replace the newer implementation.

This document defines the implemented `residual_diffusion` contract. Residual
mode jointly trains the existing deterministic CORDEX U-Net and a conditional
diffusion head. The U-Net supplies the deployed baseline; there is no separate
random `baseline_head`.

## Audit status: the existing checkpoint failed

The July 2026 repository and artifact audit found an implementation-level
boundary defect and a checkpoint that makes the deterministic prediction
worse. These are not hyperparameter-tuning results:

- The checkpoint was trained with zero padding in every spatial convolution of
  the score U-Net. Its exact theoretical receptive field is 189 pixels
  (radius 94), while the South Africa target is only 128 x 128 pixels. There is
  consequently no location whose score prediction is independent of the
  artificial zero-valued exterior. Reapplying that boundary-conditioned score
  during every reverse-diffusion step produces the rectangular domain-edge
  response seen in the precipitation plots.
- The affected inference run used one full 128 x 128 frame
  (`force_full_frame: true`); tile extraction, overlap blending, and stitching
  were not executed. The rectangles in that artifact therefore were **not**
  stitching seams. Independent tile-level diffusion is also not a valid repair:
  separate tile calls draw different reverse-process noise fields. Diffusion
  remains full-frame unless score evaluation is tiled *inside each global
  sampler step* while sharing one global latent and RNG stream. The inference
  helper now rejects independent diffusion tiling. Halo-expanded, valid-core
  tiling is supported for the deterministic path: only the central tile core is
  retained and overlap blending never accumulates padding-contaminated internal
  edges.
- The old ensemble dispatcher reset `base_seed + member` on every batch. Each
  member therefore replayed the same prior/noise pattern on every group of
  dates, which can imprint a fixed spatial noise pattern into a climatology.
  Member-specific RNG streams now advance across batches and are reset at the
  start of every scenario run, so scenario output no longer depends on run
  order. Exact stochastic replay currently requires the same base seed **and
  the same inference batch partitioning**; the effective batch size and this
  limitation are recorded in JSON, diagnostics, and NetCDF provenance.
- The training loader and target scalars use precipitation in `mm/day`.
  Historical verification truth is stored as precipitation flux in
  `kg m-2 s-1`, and earlier prediction/diagnostic metadata did not reliably
  preserve that distinction. One early diagnostic consequently subtracted raw
  flux truth from an `mm/day` prediction. The audited metrics convert the truth
  to `mm/day` first. This was a diagnostics/output-metadata defect, not evidence
  of a second residual normalization transform.
- The old training and validation predictor/target paths were identical, and
  the loader traversed all 14,600 samples for both roles. Reported validation
  loss was therefore an in-sample training metric, not an independent
  validation estimate; it cannot support checkpoint selection or a claim of
  held-out skill. The active run now disables validation entirely and traverses
  the 14,600 configured training samples once per epoch. Its post-training
  sampled inference uses the separate 1981--2000 files.
- The scalar set created on 2026-07-16 reports `num_samples: 14600`. That is
  the correct normalization population for the active full-data training mode.
  The optional contiguous-tail validation implementation still records and
  checks source/used index ranges when another experiment enables it.
- The implemented residual equation, sign, epsilon-to-score conversion, and
  DDIM reconstruction algebra are correct: the target is `target - baseline`
  and inference adds the sampled residual. Sign/oracle tests do not support a
  sign-flip or sampler-coefficient explanation for the failure.
- Bernoulli-Gamma supplies a non-negative precipitation baseline, the diffusion
  residual is Gaussian and signed, and precipitation is clamped only after the
  correction is added. With an inaccurate, high-variance correction,
  `max(0, baseline + residual)` removes the negative tail while retaining the
  positive tail. Ensemble averaging therefore cannot cancel that tail and can
  introduce a positive precipitation bias. This is a structural failure mode,
  not justification for clipping the residual itself.

On the matched 1981--2000 evaluation fields, applying the three-member
diffusion ensemble mean increased RMSE as follows:

| Variable | Deterministic baseline RMSE | Diffusion ensemble-mean RMSE |
| --- | ---: | ---: |
| `pr` | 8.3228 mm/day | 10.0344 mm/day |
| `tasmax` | 1.3668 °C | 1.8511 °C |

A post-hoc least-squares application scale computed on this same evaluation
period was approximately 0.0249 for `pr` and clipped to 0 for `tasmax`. Those
are test-period oracle diagnostics, **not** deployable calibration values; do
not put them in a production YAML or tune alpha on the test set. They show that
the current sampled correction contributes essentially no trustworthy skill.

The active residual contract therefore explicitly sets
`residual_application_scale` (alpha) to zero, uses replicate padding throughout
the score U-Net, uses continuous-time scale `1.0`, zero-initializes its output
convolution, and records these choices in checkpoint metadata schema v2. The
global padding/time defaults remain the historical `zeros`/`999.0` so legacy
full-field configs that omitted these keys are not silently reinterpreted.
Alpha zero bypasses reverse diffusion entirely and recovers the deterministic
baseline exactly. The old
checkpoint was trained with the defective boundary condition and lacks the v2
safety contract, so it must be retrained; changing its YAML is not a conversion.

The deterministic U-Net remains the deployment default. Residual diffusion is
experimental until a newly trained checkpoint improves **both bias and RMSE**
over that baseline on an untouched held-out evaluation. No such retrained
checkpoint or skill evaluation exists yet, so the code and invariant tests
below must not be described as a model-quality fix.

## Model and loss contract

Let:

- `c_theta(x)` be the backbone/U-Net decoder features;
- `b_theta(x)` be the deterministic prediction in physical units, including
  the configured precipitation head;
- `m_phi(x, b)` be a directly supervised conditional residual mean;
- `T` and `T^-1` be the configured physical-to-standardized target transform
  and its inverse; and
- `y` be the physical target.

The baseline and true residual are

```text
b_std = T(b_theta(x))
y_std = T(y)
r_std = y_std - stop_gradient(b_std)
```

The sign is always **target minus baseline**. Both `y` and the deterministic
baseline enter standardized target space exactly once. In particular, raw
U-Net output logits are first decoded by the ordinary deterministic path to a
physical prediction and only that physical prediction is passed through `T`.
Scaler means and standard deviations must never be applied a second time to an
already-standardized tensor.

The mean-corrected prediction and innovation target are

```text
u_std = r_std - stop_gradient(m_phi)
y_mean = T^-1(stop_gradient(b_std) + m_phi)
```

The joint training objective is

```text
L_base = L_yaml(b_theta(x), y)
L_mean = L_yaml(y_mean, y)
L_gate = relu(L_mean - stop_gradient(L_base) * (1 - delta))
L = lambda_base * L_base + lambda_mean * L_mean
  + lambda_gate * L_gate + lambda_diff * L_epsilon(u_std, condition)
```

`L_yaml` is one `CompositePredictandLoss` constructed from the active YAML.
The same object is called for baseline and corrected mean, so configured RMSE,
distribution, spatial-gradient, multiscale, and boundary terms cannot diverge
between branches or be hard-wired by the diffusion implementation. The active
run jointly updates the initialized U-Net through `L_base`, updates `m_phi`
through `L_mean`/`L_gate`, and updates the score network through score matching.
The baseline threshold used by `L_gate` is detached so the U-Net cannot game the
comparison by becoming worse. Decoder features, baseline, and `m_phi` remain
detached from score loss.

The residual forward process is epsilon-parameterized. For VPSDE,

```text
epsilon ~ Normal(0, I)
x_t = alpha(t) * u_std + sigma(t) * epsilon
L_epsilon = E[||epsilon_theta(x_t, condition, t) - epsilon||^2]
x0_theta = (x_t - sigma(t) * epsilon_theta) / alpha(t)
SNR(t) = alpha(t)^2 / sigma(t)^2
L_x0 = E[min(1, x0_inverse_snr_cap * SNR(t))
         * ||x0_theta - u_std||^2]
L_diffusion = L_epsilon + clean_x0_reconstruction_weight * L_x0
```

Internally the epsilon prediction is converted to a score as
`score_theta = -epsilon_theta / sigma(t)`. `prediction_type: epsilon` is the
only supported parameterization; unsupported values fail during construction
instead of silently changing the training/sampling contract.

The clean-target term is required for the residual run because plain epsilon
MSE weakly constrains clean reconstruction at the high-noise end of the
`beta_max: 20` schedule. Algebraically,
`||x0_theta-u_std||^2 = inverse_SNR * ||epsilon_theta-epsilon||^2`; the
`min(1, cap*SNR)` factor caps that inverse-SNR *training weight* at a finite
value. It does not clip or rescale a sampled residual. An isolated tiny-data
gate failed with epsilon loss alone despite a low epsilon MSE, and passed across
four reverse-process seeds after enabling this clean objective. That result
checks that the objective can fit a tiny case; it is not evidence that the
audited production checkpoint generalizes or improves the deterministic model.

## Sampling and precipitation semantics

Reverse diffusion produces a signed correction in standardized target space:

```text
r_hat_std = reverse_diffusion(condition, seed)
y_hat_std = b_std + alpha * r_hat_std
y_hat = T^-1(y_hat_std)
```

`alpha` is the recorded `residual_application_scale` deployment gate in
`[0, 1]`; its safe default is 0. It is intentionally a runtime calibration knob,
not an immutable learned-checkpoint semantic, so held-out evaluation may test a
nonzero value without rewriting the checkpoint. The inference script accepts an
explicit `SA_RESIDUAL_ALPHA` environment override after loading the immutable
resolved snapshot. It rejects nonnumeric, nonfinite, out-of-range, and
non-residual uses. When the variable is unset, the snapshot value (or safe
default 0) is retained. Every output records the actual runtime alpha and its
source. At alpha zero the sampler is not called and the returned raw and applied
corrections are zero. At nonzero alpha, the sampled diagnostic residual is raw
and signed: it is never passed through a positivity link or independently
clipped. The correction is added to the baseline before a single inverse target
transform.
Negative precipitation corrections are necessary to correct a wet baseline
downward. Configured precipitation non-negativity is enforced only on the final
decoded physical prediction `y_hat`, after baseline and residual have been
combined. The sampled validation notebook fixes alpha at one to test the full
learned correction; it must not tune alpha on those same samples. Any later
production alpha must be frozen before an untouched final evaluation.

During training, the diffusion head checkpoints per-channel count, mean,
standard deviation, RMS, minimum, and maximum of the exact normalized residual
tensors used for forward noising. Because the deterministic U-Net changes
during joint training, those online moments pool residuals from many different
baseline states; they are diagnostics, **not** a calibrated distribution for
the final checkpoint. The active config therefore disables the RMS magnitude
guard with a non-positive multiple. Enabling it requires a separate
training-only pass through the frozen final model that records a versioned
calibration artifact. If enabled after such calibration,
`residual_magnitude_guard_multiple` compares generated residual RMS with that
reference after `residual_guard_min_count` observations and raises on an
out-of-scale correction; it never rescales, clips, or hides it. Until then,
alpha zero is the actual fail-closed safety gate.

## Required configuration

The residual run selects the diffusion head while retaining the v6
deterministic loss options:

```yaml
validation_enabled: false

data:
  validation_predictor_paths: []
  validation_target_paths: []
  validation_holdout_fraction: 0.0

model:
  head_type: diffusion
  diffusion:
    residual_diffusion: true
    residual_mean_enabled: true
    residual_mean_channels: 64
    # Fail closed: alpha=0 exactly returns the deterministic baseline. Enable
    # only after held-out gates have been passed by a retrained model.
    residual_application_scale: 0.0
    prediction_type: epsilon
    padding_mode: replicate
    zero_init_output: true
    sde: vpsde
    beta_max: 20.0
    eps: 1.0e-5
    noise_conditioning_scale: 1.0
    fourier_scale: 4.0
    clean_x0_reconstruction_weight: 1.0
    clean_x0_inverse_snr_cap: 100.0
    sampling_method: ddim
    eta: 0.0
    num_sampling_steps: 256
    sampling_eps: 1.0e-3
    # Disabled: online joint-training moments are not final-model calibration.
    residual_magnitude_guard_multiple: 0.0
    residual_guard_min_count: 1024

loss:
  deterministic_weight: 1.0
  diffusion:
    weight: 1.0
    corrected_mean_weight: 1.0
    improvement_penalty_weight: 1.0
    minimum_relative_improvement: 0.02
  # Keep the remaining deterministic loss options aligned with v6.

training:
  # Update the pretrained U-Net together with both diffusion-head branches.
  freeze_deterministic_baseline: false

validation:
  residual_correction:
    # Sampled inference performs the post-training comparison.
    enabled: false
    ensemble_size: 3
    application_scale: 1.0
    require_each_configured_term_non_degradation: true
```

The corrected-field objective uses the pretrained single-head U-Net:

```yaml
precip_model: single_head
```

Hurdle and Bernoulli-Gamma auxiliaries parameterize their baseline head; they
cannot evaluate an arbitrary corrected precipitation field with the same loss.
The builder therefore rejects those modes when corrected-mean supervision is
enabled instead of silently omitting corrected precipitation.

Continuous SDE time `t` lies in `[0, 1]` and is passed directly to the Fourier
embedding, so `noise_conditioning_scale` is `1.0`. Scaling time by the number
of diffusion steps makes adjacent times effectively unrelated at the configured
Fourier frequencies and is not the implemented residual-run contract.
The residual run also uses `fourier_scale: 4.0`; scale 16 failed to interpolate
the continuous noise-time dependence reliably in the held-out-seed overfit
gate.

For DDIM:

- `eta` must be in `[0, 1]`;
- stochastic DDIM (`eta > 0`) is implemented only for `vpsde`;
- `subvpsde` requires `eta: 0` because its marginal variance differs from VP;
  and
- selecting DDIM with `vesde` falls back to the ODE sampler.

DDIM coefficients are derived from the selected SDE's actual marginal signal
coefficient and noise standard deviation. The current residual YAML uses
`vpsde` with `eta: 0`; independently seeded terminal priors still produce the
ensemble, while qualification remains exactly reproducible.

The model training forward returns `baseline_prediction`,
`corrected_mean_prediction`, and scalar `diffusion_loss`.
`JointResidualDiffusionLoss` evaluates both physical fields with `L_yaml` and
reports every dynamically returned configured term under baseline and corrected
prefixes. The active run does not construct a validation loader or sample a
diffusion ensemble during training. It always writes `last.ckpt`; the sampled
inference workflow loads that checkpoint and performs the decoded ensemble-mean
comparison at full alpha after training. No `best.ckpt` is selected in this
mode.

## Checkpoints and compatibility

New residual checkpoints use metadata schema v2 and contain two related
contracts. The complete recorded provenance covers the head and residual
definition, output variables, resolved SDE/score and sampler settings,
noise-time scale, joint loss configuration, predictand and precipitation
semantics, joint-versus-frozen U-Net training policy, normalization-scalar
paths and SHA-256 hashes (including the
scalar-selection `metadata.json`), initializer identities and hashes,
training/validation/test paths, static-field identity, target/crop geometry,
case/job identity, inference randomness, and source-control provenance. It has
its own integrity hash. A separate strict fingerprint contains only portable
architecture/training semantics: absolute locations are replaced by portable
training/validation filenames, scalar files are identified by content hash,
and case/job identity, initializer identity, test paths, ensemble size, base
seed, and alpha are excluded. These excluded values remain recorded in full;
moving a run or changing an evaluation seed does not reinterpret trained
weights.

Residual inference loads the immutable resolved configuration snapshot and
validates its strict semantic projection before accepting model weights.
Missing or unsupported metadata, damage to either integrity hash, changed
scalar contents, changed portable training-data identity, changed residual
semantics, or an architecture/loss mismatch is a hard error. Provenance-only
differences are allowed but still written to inference artifacts. State-dict
loading may omit the separately reconstructed scaler buffers, but other missing
or unexpected parameters are rejected.

Old full-field diffusion checkpoints are not residual checkpoints. They were
trained on `T(y)`, do not contain the jointly supervised deterministic baseline
contract, and cannot be resumed or used for residual inference. Likewise, a
checkpoint from the obsolete random/untrained-baseline implementation is not
convertible by changing YAML. A compatible deterministic checkpoint may be
used only through the normal initialization path when its parameter shapes are
explicitly checked; the corrected joint residual model must then be retrained.
The audited residual checkpoint is also obsolete: it learned the zero-padded
score field and predates schema v2. It must not be loaded under a replicate-
padding YAML or used with a post-hoc test-period alpha.

The NetCDF preprocessing products do not need to be regenerated: their units,
grid, and numerical contents are unchanged. Because the active no-validation
run trains on all 14,600 samples, its existing full-period scalar metadata is
the appropriate normalization contract and does not need a holdout-specific
recalculation. Reuse remains safe only while sources, units, transforms,
full-period selection, and recorded hashes remain unchanged; training and
checkpoint compatibility both fail closed otherwise.

## Affected files and functions

This inventory covers the production code, audit tools, configuration, and
regression tests intentionally changed for the residual-diffusion repair.

- `SA_T2_ACCESS-CM2_static_residual_diffusion.yaml` defines the corrected joint
  objective, single-head precipitation baseline, full-data/no-validation mode,
  supervised residual mean, replicate score padding, safe deployment alpha,
  and sampler contract.
- `notebooks/SA_downscaling_finetune_T2_ACCESS-CM2_static.ipynb` synchronizes the
  notebook launch path with that resolved training configuration.
- `cordex_dataset.py` changes `CordexDownscaleDataset.__getitem__`,
  `_compute_time_lengths`, `_build_exact_target_time_index`,
  `_read_time_coordinate`, and `_load_targets`, and adds
  `_time_coordinate_key`, `_format_time_key`, and `_exact_time_index`. Together
  they require exact civil-date target pairing, reject duplicate/missing dates,
  and reject non-finite targets instead of silently shifting, repeating, or
  filling them.
- `cordex_inference.py:build_inference_dataset` documents and activates that
  exact calendar-date pairing for inference files with different calendar
  lengths.
- `compute_scalars_cordex.py:_select_scalar_training_partition` and `main`
  support an optional leading-training/contiguous-tail-validation split and
  write auditable selection metadata when that mode is enabled.
- `cordex_training.py` adds `CordexIndexSubset`,
  `_contiguous_holdout_indices`, `_validate_train_validation_source_paths`, and
  `_validate_scalar_holdout_metadata`; `get_dataloaders` skips validation and
  uses the complete training dataset when explicitly disabled, while still
  rejecting full-range validation, partial source-list overlap, and stale
  all-sample scalars when validation is enabled.
  `load_pretrained_weights` and `create_finetune_model` enforce residual
  checkpoint compatibility rather than silently accepting a partial load.
- `granitewxc/decoders/diffusion_head.py` changes `_ResBlock`,
  `ConditionalScoreUNet`, `DiffusionHeadConfig`, and `DiffusionHead`. These
  implement explicit padding, epsilon-only prediction semantics, continuous
  noise-time scaling, zero output initialization, signed standardized residual
  construction, a supervised residual-mean plus innovation decomposition,
  alpha-gated application, sampler bypass at alpha zero, residual-stage
  exposure, and diagnostic residual moments.
- `granitewxc/models/cordex_finetune_model.py` changes
  `ClimateECCCFinetuneWrapper` and `ClimateDownscaleFinetuneUNETModel` so the
  ordinary deterministic prediction is the residual baseline, exposes the
  differentiable mean-corrected field during training, and passes the physical
  baseline directly to diagnostics without reconstructing it by subtraction.
- `granitewxc/models/diffusion_loss.py:score_matching_loss`,
  `DiffusionLossPassthrough`, and `JointResidualDiffusionLoss` implement the
  epsilon objective, optional clean-x0 reconstruction term, and the YAML-driven
  baseline/corrected/diffusion objective with explicit gradient routing.
- `granitewxc/models/diffusion_sampling.py:get_ddim_sampler` and
  `build_sampler` derive DDIM coefficients from the configured SDE, enforce
  VP/subVP/VE restrictions, and accept the caller-owned random generator.
- `granitewxc/models/loss.py:CompositePredictandLoss`,
  `_resolve_precip_wet_threshold`, and `build_loss_fn` retain the v6
  configured deterministic losses and fail closed when an auxiliary-only
  precipitation likelihood cannot score the corrected field.
- `granitewxc/models/model.py:get_finetune_model_UNET` and
  `get_finetune_model` construct the requested deterministic or residual head
  without changing scaler application or output-channel order.
- `granitewxc/utils/predictands.py:canonicalize_precip_model` recognizes the
  Bernoulli-Gamma aliases retained for configurations that do not supervise an
  arbitrary corrected precipitation field.
- `granitewxc/utils/checkpoint_metadata.py` adds
  `build_checkpoint_compatibility`, `build_checkpoint_metadata`, and
  `validate_checkpoint_compatibility`, including hashes and a versioned strict
  training/model semantic fingerprint plus separately recorded runtime and
  provenance fields.
- `granitewxc/utils/trainer.py:validate_one_epoch`, `save_checkpoint`, and
  `train_model` support the optional decoded-ensemble gate and reserve
  `best.ckpt` for qualified corrections. The active YAML disables that path,
  so training saves `last.ckpt` and sampled inference performs validation.
- `utils/diffusion_inference.py` changes `infer_batch_ensemble`,
  `residual_transformation_stages`, and `as_ensemble_mean`, and adds
  `PersistentEnsembleGenerators` and `reset_ensemble_generators`. Member RNG
  streams now advance across batches, reset once per scenario, and expose the
  ensemble mean and transformation stages without a float32 cancellation-based
  baseline recovery.
- `utils/inference_blending.py:resolve_boundary_mitigation_settings` and
  `infer_batch_with_boundary_mitigation` implement halo-expanded valid-core
  deterministic tiling, shifted origins, scaler offsets, and smooth overlap
  weights, while rejecting independent diffusion tiling even through common
  model wrappers.
- `utils/diffusion_diagnostics.py` adds distribution, deterministic, CRPS,
  spread-skill, climatological-bias, and transformation-stage reports used by
  the saved audit artifacts.
- `notebooks/SA_downscaling_inference_T2_ACCESS-CM2_static.py` changes
  `_run_full_inference`, `_load_model_and_config`, `_save_run_outputs`, and
  `main`, and adds the unit, scalar-hash, checkpoint, boundary, RNG, alpha, and
  NetCDF provenance helpers. It converts precipitation truth flux to `mm/day`,
  keeps output attrs numerically honest, resets stochastic streams per
  scenario, sanitizes unsupported Boolean NetCDF attrs, saves the paired
  deterministic baseline, and never silently forces a boundary mode.
- `residual_validation_report.py:validate_residual_correction`,
  `_active_geometry_report`, `_plot_residual_composite`, and
  `_plot_residual_scatter_triptych` produce the direct 7,300-day paired metrics,
  member/ensemble results, seven climatology maps, geometry overlays, three
  requested residual relationships, and the fail-closed deployment decision.
- `residual_diffusion_real_batch_preflight.py`,
  `residual_diffusion_overfit_gate.py`, and `diffusion_residual_tests.py` are
  executable preflight/oracle tools; they do not replace held-out forecast
  evaluation.
- `tests/test_diffusion_head.py`, `test_diffusion_inference_workflow.py`,
  `test_distribution_loss.py`, `test_checkpoint_metadata.py`,
  `test_compute_scalars_holdout.py`, `test_cordex_time_alignment.py`,
  `test_cordex_validation_holdout.py`, `test_diffusion_diagnostics.py`,
  `test_inference_blending.py`, and `test_residual_production_pipeline.py`
  cover the corresponding residual algebra, sampler, serialization, unit,
  split, RNG, tiling, boundary, and production NetCDF invariants.
- `DIFFUSION_RESIDUAL_CORRECTION.md` is this mathematical contract, root-cause
  record, affected-code inventory, and retraining/deployment checklist.

## Mandatory preflight before a production retrain

Do not start or accept a full run until all of the following pass in the
`Prithvi` environment:

1. **Data-separation audit:** require `validation_enabled: false`, empty
   validation path lists, and a zero holdout fraction for this full-data run.
   Verify that all 14,600 configured training samples are used and that the
   separate 1981--2000 sampled-inference dates are absent from those training
   sources. Do not tune alpha or sampler settings on the sampled comparison.
2. **Transform and sign tests:** verify `T^-1(T(y)) == y`,
   `T^-1(b_std + (T(y) - b_std)) == y`, target-minus-baseline sign, and signed
   residual behavior without intermediate precipitation clipping.
3. **Gradient-routing tests:** verify the YAML corrected-field loss reaches only
   the residual-mean predictor in staged training; verify score loss reaches
   only the score network and not the mean, baseline, or conditioning graph.
4. **SDE/sampler tests:** verify epsilon/score equivalence, forward/reverse
   coefficient parity, oracle DDIM reconstruction, VP stochasticity, and
   rejection of nonzero `eta` for subVP.
5. **Boundary and tiling tests:** verify replicate-padding behavior on constant
   fields, exact full-frame/deterministic halo-valid-tile agreement where the
   receptive field permits it, shifted tile origins, seam masks, boundary
   normalization, and hard rejection of independent tile-level diffusion.
6. **Real-batch transformation audit:** on an actual training sample, log
   physical truth and baseline, normalized truth and baseline, true normalized
   residual, generated normalized residual, de-normalized correction, and final
   physical output. Check finite values, scalar shapes/hashes, channel order,
   units, coordinates, and time alignment. The diagnostic baseline must be the
   direct tensor published by the model; do not recover it as `full - residual`
   in float32. Explicitly verify precipitation-flux to `mm/day` conversion and
   saved output unit metadata.
7. **Safe-application tests:** require alpha zero to reproduce the deterministic
   output bit-for-bit without constructing a sampler, verify
   `baseline + alpha * residual` for nonzero alpha,
   pass exact zero and true residuals through the production transformation
   path, and test correction sign in both directions.
8. **Tiny-data overfit:** overfit one sample or a very small subset and require
   the epsilon loss to decrease, the sampled residual scale to remain comparable
   with the true residual, and reconstructed fields to improve over the
   deterministic baseline.
9. **Serialization and compatibility tests:** save/reload the joint checkpoint,
   reproduce seeded samples, and confirm that residual-sign, scalar, loss, SDE,
   prediction-type, padding, initialization, and schema mismatches are rejected.
   Confirm separately that alpha, case/job names, filesystem relocation,
   initializer provenance, ensemble size, and base seed can change without a
   strict mismatch while their runtime values remain recorded.

Unit tests now cover the residual sign and transform identities, sampler oracle,
safe alpha application, score-network padding/output initialization,
deterministic halo-valid tiling (including shifted origins and seam coverage),
wrapper-aware rejection of independent diffusion tiling, singleton-ensemble
mean reduction, and direct-baseline residual diagnostics. Passing these tests establishes
implementation invariants only; it does not establish forecast skill.

After preflight, train a new residual-diffusion checkpoint from the corrected
configuration and load its `last.ckpt` in the sampled inference notebook.
Compare the paired U-Net and full-alpha ensemble mean on identical 1981--2000
samples using physical bias, MAE, RMSE, spatial correlation, residual
percentiles and maps, ensemble spread-skill, and CRPS. The 20-date, 16-step CPU
run is a smoke validation; confirm any apparent improvement with a full-period,
checkpoint-native sampler evaluation. Use shared color limits for before/after
bias maps and do not tune alpha or sampler settings on the reported comparison.
A production result is not valid merely because it looks bounded: the
transformation audit, residual distributions, and metrics must agree with the
documented contract.
Keep alpha at zero and deploy the deterministic baseline unless the jointly
trained, nonzero-alpha model improves both bias and RMSE on that untouched test
data.
