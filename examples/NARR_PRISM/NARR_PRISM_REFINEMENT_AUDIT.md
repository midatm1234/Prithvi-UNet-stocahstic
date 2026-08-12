# NARR–PRISM Phase-2 stochastic-refinement audit

Date: 2026-08-12
Repository branch audited: `Prithvi-UNet-stochastic_refinement`
Environment used for every Python/test command: `Prithvi`

## Executive result

The shared Phase-2 implementation, all four NARR–PRISM configurations, the
training/resume CLI, daily tiled ensemble inference, evaluation code, and the
generic notebook have been repaired and covered by automated tests. The four
configured refinement choices are:

| Requested name | Repository file | Resolved implementation |
| --- | --- | --- |
| Diffusion | `NARR_PRISM_diffusion_unet.yaml` | `diffusion_unet` |
| Diffusion Transformer | `NARR_PRISM_diffusion_transformer.yaml` | `diffusion_transformer` |
| Flow Matching | `NARR_PRISM_flow_matching_unet.yaml` | `flow_matching_unet` |
| Flow Matching Transformer | `NARR_PRISM_flow_matching_transformer.yaml` | `flow_matching_transformer` |

All four YAMLs parse, instantiate, complete a finite forward/backward pass,
sample or integrate, preserve the `[ppt, tmax, tmin]` output order, and pass a
fixed-tiny-example overfit test. In addition, all four wrappers have now loaded
the recovered *real* Phase-1 `last.ckpt`: 208/208 deterministic state entries,
zero missing, unexpected, or shape-mismatched entries, and the same Phase-1
tensor fingerprint `84c8509ce161f049d8999e5d44b43b83d36cc316a579d052df1bd0f6173a31c7`.
Phase 1 is completely frozen in every default Phase-2 wrapper. The final
full-suite result is recorded in the test section below.

### Artifact-loss incident and recovery

Commit `c538165` had mistakenly tracked `experiments`, `preprocessed`, and
`scalars_with_H` as absolute links back to the same paths in the primary
worktree. At 2026-08-12 17:40 UTC, checking out that branch replaced the real,
ignored artifact directories with those self-referential links. Reflog times,
link inode birth times, and earlier directory listings establish this sequence;
it was not an external-mount failure.

`/data/granite-wxc-stochastic` is a small, clean, detached secondary Git
worktree at the older `NARR_PRISM` commit; it contains no experiment artifacts
and was not used for this recovery. The former `granite-wxc` GitHub URL and the
current `Prithvi-UNet-stocahstic` URL identify the same repository after an
in-place rename, not two repositories. The working checkout remains
`/data/granite-wxc`, and publication targets lowercase `origin/narr_prism`.

The deleted Phase-1 inode and extent tree were recovered read-only from the
ext4 journal. The reconstructed checkpoint is 2,982,857,829 bytes with SHA-256
`1d352491ab32a4069696673f504edcd9c67d48325e83dcf3b042fe3d3eb80efa`.
All 814 ZIP members pass CRC validation, `torch.load(..., mmap=True)` succeeds,
and its metadata reports epoch index 19 (20 completed epochs), 208 model state
entries, the expected NARR–PRISM case/channel/grid/normalization contract, and
ordered outputs `[ppt, tmax, tmin]`. Its saved final training loss is
1.6487374, final validation loss 2.5783734, and best validation loss 2.3550687;
these are checkpoint training objectives, not physical-unit skill metrics.

The four normalization arrays were recovered byte-for-byte from the checkpoint
buffers, including the exact `.npy` container bytes; all four SHA-256 values
match the checkpoint contract. The original normalization manifest and scalar
metadata were also recovered from the journal. Independently recovered 1996,
2005, and 2013 daily training products were compared with their deterministic
regenerations: all 15 predictors, three targets, elevation, latitude,
longitude, and NaN payloads/masks are bitwise equal (maximum absolute
difference 0.0). The apparent raw-source signature
change was traced across all 6,575 training dates solely to Linux device
renumbering from 2097 to 2065; paths, inodes, sizes, and nanosecond mtimes are
unchanged. A strict attestation verifies all 52,600 source entries and permits
only this one schema-4 source-split relocation; all other contract differences
still fail.

The durable local artifact root is now outside the Git worktree at
`/data2/granite-wxc-artifacts/NARR_PRISM`. Tracked compatibility links route
through the ignored per-clone `examples/NARR_PRISM/artifacts` portal rather
than pointing at themselves. `narr_prism_artifacts.py` validates/configures the
portal and rejects cycles or source-tree targets. The intact raw NARR, PRISM,
and elevation inputs were used to regenerate the complete corpus under that
external root: 6,575 training days, 730 target-bearing validation days, and
3,653 predictor-only inference days, with no missing or extra configured
dates. No Phase-2 scientific improvement is claimed until the four real
refinement runs and paired validation evaluation are complete.

## Acceptance matrix

| Criterion | Result | Evidence |
| --- | --- | --- |
| Four YAMLs parse | Pass | CLI `describe` on all four files |
| Same configured Phase-1 checkpoint | Pass (configuration) | All point to the same `last.ckpt` |
| Current real `last.ckpt` loads in all four | Pass | Recovered checkpoint; each wrapper loads 208/208 keys and fingerprint `84c8509c…` |
| No unexplained deterministic keys | Pass | Real load has zero missing/unexpected/shape mismatches; prefix tests also cover `model.`, `module.`, and `_orig_mod.` roots |
| Semantic checkpoint compatibility | Pass | Real schema-4 PRISM contract is validated before tensor loading; source relocation requires a narrow audited rebind |
| Phase 1 frozen by default | Pass | `eval`, `requires_grad=False`, `no_grad`, detached baseline/features |
| Selected unfreezing remains configurable | Pass for training/resume | `trainable_phase1_patterns`; mutually exclusive with joint fine-tuning and fitted residual normalization |
| Residual sign/reconstruction | Pass | Synthetic exact residual and hurdle-precipitation no-op tests |
| Finite training backward for every head | Pass | Parameterized tests plus one regenerated real-data batch for all four heads |
| Sampling/integration for every head | Pass | Diffusion/flow tests plus two-step real-data sampling/integration for all four heads |
| Transformer patch round trip | Pass | Rectangular, non-divisible padding/cropping tests |
| Tiny-subset overfit | Pass | All four reduce a fixed stochastic objective by more than 50% |
| Phase-2 save/resume | Pass | Model/refiner/optimizer/scheduler/scaler/RNG tests |
| Residual-semantics migration | Pass | Schema 2 marker required; schema-1 inference/resume rejected before state loading |
| Complete preprocessed corpus | Pass | 6,575/730/3,653 exact training/validation/inference dates; inference products contain no targets |
| Production daily inference contract | Pass synthetically | Exact dates, canonical coordinates, mask, units, filenames, and provenance tests |
| Generic notebook execution | Pass | All four synthetic helpers, headless smoke, and real production setup against `last.ckpt` and a real batch |
| Existing Phase-1 behavior/regressions | Pass automated suite | Final full-suite command in the test section |
| Full 2016–2025 Phase-2 evaluation | Not run | Complete data products are ready; four trained Phase-2 checkpoints are not yet available |

## Audited data and model contract

### Dates

All splits are inclusive and disjoint:

| Split | Dates | Days |
| --- | --- | ---: |
| Training | 1996-01-01 through 2013-12-31 | 6,575 |
| Validation | 2014-01-01 through 2015-12-31 | 730 |
| Independent inference | 2016-01-01 through 2025-12-31 | 3,653 |

The raw-file audit found all 10,958 daily target dates, all expected monthly
predictor files, and all eight leap days from 1996 through 2025. Dataset
selection is by exact timestamp, not a positional offset. Training-only scalar
generation remains restricted to 1996–2013.

### Ordered variables and channels

The output order is explicitly configured and enforced as:

```text
[ppt, tmax, tmin]
```

The 16 value predictors are the 500/700/850-hPa channels for `shum`, `uwnd`,
`vwnd`, `air`, and `hgt`, in that order, followed by elevation. The next 16
channels are their corresponding masks, giving 32 Phase-1 input channels.
Elevation is already an input channel; it is not a separate `static_x` tensor
in this workflow. The Phase-2 YAMLs therefore disable duplicate static and
post-fill validity conditioning, retain the explicit mask channels, and add
absolute PRISM-grid coordinate conditioning.

Checkpoint semantic validation now binds the ordered predictors/outputs,
channel counts, case name, train/validation dates, grid and crop/halo contract,
scaler/source signatures, preprocessing semantics, precipitation settings, and
model topology before any weight is applied.

The recovered schema-4 checkpoint records the same Phase-1 topology as the
authoritative YAML: embedding width 1024, eight backbone blocks, MLP ratio 4,
16 attention heads, 2 x 2 patches, 512 downscaling embedding channels, 128
convolution channels, convolutional encoder/decoder, UNet depths `[2, 2, 2]`,
dynamic decoder skips, pre-convolution backbone residuals, windowed-local
attention, and bilinear decoder/output scaling with `align_corners: false`.
The observed training tensors are `[B, 32, 320, 320]` predictors (32-pixel
halo) and `[B, 3, 256, 256]` targets. Precipitation uses the hurdle head with a
0.01 wet threshold and 0.5 occurrence threshold; temperatures are degrees C.

### Grid, crop, masks, and alignment

The canonical California grid is 1024 x 1024 with ascending latitude and
longitude, approximately 1/120 degree spacing, and bounds 32.55–41.075 N and
-124.45–-115.925 E. Training uses 256 x 256 target cores, stride 192, and a
32-pixel predictor halo. Target NaNs remain masked and never enter a loss or
metric denominator. Invalid residuals and stochastic process states are reset
to zero, including between sampler/integrator steps, so masked geography cannot
feed generated state into neighboring valid pixels.

Previously, a haloed 320 x 320 conditioning tensor was top-left cropped to
256 x 256. This shifted Phase-2 predictors by 32 pixels relative to the
deterministic prediction and PRISM. Conditioning now uses each sample's exact
`__output_crop=(top,left,height,width)`; a centered crop is only a legacy
fallback. Daily production inference uses the same canonical tile plan and
Hann stitching as deterministic NARR–PRISM inference. Its random source is
keyed by seed, date, member, draw, channel, and global grid cell, so overlapping
tiles receive identical stochastic forcing regardless of tile order or batch
size. Production inference opens the dataset in predictor-only mode: it never
discovers or loads 2016–2025 PRISM observations. Its static output mask is a
separate signed boolean artifact derived from finite 1996–2013 targets for all
three ordered variables. It cannot be reconstructed from the spatial scalers,
whose unsupported cells are deliberately neutral-filled with mean 0/std 1.

### Units and transforms

| Variable | Physical convention | Phase-1 target transform |
| --- | --- | --- |
| `ppt` | mm/day | training-period global p95 `divide_only` |
| `tmax` | degrees C | per-grid-cell training mean/std z-score |
| `tmin` | degrees C | per-grid-cell training mean/std z-score |

The authenticated 1996–2013 scalar artifacts are `inputs_mean`/`inputs_std`
with shape `[32]` and SHA-256 prefixes `27ffd1e2d9c7`/`aef98b46969e`, plus
`targets_mean`/`targets_std` with shape `[3, 1024, 1024]` and prefixes
`dd1d2f1eed9d`/`53498f61544f`. The precipitation p95 scale is 7.42578125
mm/day. The canonical-grid fingerprint begins `f062e76bd115`; all are bound by
the recovered checkpoint and active scalar manifest. The regenerated manifest
SHA-256 is `dd4a51d631d37601fad3c4f528036c5a34722a534469c1d5129d6b1522e46f12`.
The signed boolean support mask has SHA-256
`8ecc8b537827db900961f5c66e128ffe7e86d1acfeaed3fec0e23cc30e937ee7`
and marks 714,742 of 1,048,576 cells valid. The schema-4 relocation
attestation's signed digest is
`ffef6674e51c3cfadbea79f86b4a7771fbf83c8165e1a0639551a39e9f3b171f`;
it binds the historical `5588ce4c…` source split to the regenerated
`3e41bdcc…` split and cannot authorize any other checkpoint-contract change.

NARR air predictors remain Kelvin before predictor standardization. No target
unit conversion is performed by preprocessing. The raw PRISM GDAL files do not
carry a reliable `units` attribute. All five YAMLs therefore state the exact
`evaluation.truth_units` contract. Evaluation validates any source attribute
that is present; only an absent attribute may fall back to that explicit
contract, and every such fallback is counted in report provenance. Production
outputs explicitly write `mm/day` and `degC` attributes. A future signed source
manifest would make this convention independently verifiable.

Residual mean/std are now fit, one ordered target channel at a time, by
streaming only the 1996–2013 training loader with frozen Phase 1. They are
persistent buffers in the Phase-2 checkpoint, are required before training or
sampling when enabled, and are restored exactly on resume/inference. Validation
and 2016–2025 observations are never used to fit them.

## Root causes and repairs

| Root cause | Why it produced incorrect/noisy results | Repair |
| --- | --- | --- |
| Artifact directories were committed as absolute links to themselves | A branch checkout replaced ignored checkpoints/products with cyclic links and removed the local artifacts | Recover the exact checkpoint/scalars, regenerate daily products, and route tracked relative links through an ignored external portal |
| `data.use_static` contradicted the recovered Phase-1 contract | Elevation was already input channel 16; claiming a separate static tensor made the YAML semantically incompatible | Set `use_static: false` in Phase 1 and all four refiners while retaining `elev` and `mask_elev` in the ordered 32-channel input |
| Phase-2 accepted an absent Phase-1 checkpoint | Training/inference could run against random deterministic weights | Active training/inference require an accessible regular Phase-1 file |
| Only tensor shape/key checks were used | Same-shaped weights with different variables, scalers, dates, or grid could be accepted | Run `validate_prism_checkpoint_contract` before tensor loading |
| Key-level `model.`/Lightning roots were not normalized | Valid deterministic checkpoints failed or were partially mapped | Strip only uniform known roots, then require complete deterministic coverage |
| Unrestricted non-strict semantics were possible | Missing deterministic weights could be hidden as “new Phase 2” | Non-strict load occurs only after proving every missing key is `refiner.*`; shapes/unexpected keys fail |
| NARR hurdle precipitation returned an ungated normalized amount latent | A zero residual changed dry Phase-1 precipitation into positive precipitation | Define the baseline as `encode(final_phase1_physical)`; assert decode/no-op parity |
| Haloed predictors were top-left cropped | Conditioning was spatially shifted 32 pixels | Use exact per-sample output crop metadata |
| Raw physical predictors conditioned Phase 2 | Predictor channel scales differed by orders of magnitude and from Phase 1 | Reuse the exact Phase-1 input scaler resolver and offsets |
| Filled predictors were tested with `isfinite` | The additional mask was all ones | Disable this duplicate for NARR; retain the 16 explicit mask channels |
| `train_on_residual: false` trained absolute targets but inference always added output | Deterministic prediction/target could be added twice | Reject active absolute-target mode; one residual convention only |
| No residual-specific training-period standardization | Precipitation and temperatures had poorly balanced stochastic targets | Add fitted ordered residual mean/std metadata and inverse exactly once |
| Untrained Phase-2 inference was allowed | Zero network outputs do not imply zero diffusion/flow samples | Require and validate a Phase-2 checkpoint; use YAML fallback or explicit override |
| Old inference consumed the 2014–2015 validation loader | Tiles were concatenated as time, coordinates were integers, and 2016–2025 was not evaluated | New daily inference uses `NarrPrismDataset(mode='inference')`, exact dates/coords/mask, tiling, and daily files |
| Inference derived its domain mask from the first held-out PRISM day | A transient observation gap could erase that cell on every date and made inference depend on truth | Persist and authenticate joint finite support from 1996–2013 target counts; both deterministic and refined inference require it |
| Existing Phase-2 payloads did not identify residual-baseline semantics | A checkpoint trained against the ungated hurdle amount latent could load after the baseline changed to the final physical Phase-1 field | Phase-2 schema 2 stores an explicit semantics marker; schema-1 checkpoints are rejected and must be retrained |
| Refinement output named the deterministic field as the canonical variable | Existing evaluation could silently score Phase 1 instead of refinement | Canonical `ppt/tmax/tmin` now hold refined means; baseline uses `_phase1` |
| Tile-local random draws differed in overlaps | Stitching could create stochastic seams and depend on tile order | Coordinate-aligned stateless random field plus Hann blending |
| Evaluation cache omitted model/checkpoint identity | Phase-2 evaluation could silently reuse Phase-1 climatology | Cache identity includes file signature and Phase-1/Phase-2/config fingerprints |
| Resume compared only partial model metadata | Changed warmup, accumulation, precision, or cosine horizon could alter an interrupted trajectory | New checkpoints persist and exactly compare refinement, performance, precision, optimizer, schedule, and total epoch/step contracts before state loading |
| Each method used a differently seeded bounded reservoir | Sampled tail/distribution metrics compared different target cells | Reuse one deterministic priority stream per variable so every method retains the same paired cells |
| Partial gradient-accumulation groups were dropped/underweighted | Small datasets and smoke runs could make no update | Scale by actual group size and flush the final partial group |

## Mathematical contracts

### Residual and reconstruction

The single training and inference convention is:

```text
baseline_norm   = encode(final_phase1_physical)
residual_target = encode(PRISM_target) - baseline_norm
z               = (residual_target - residual_training_mean) / residual_training_std
pred_residual   = inverse_residual_standardization(predicted_z)
refined_norm    = baseline_norm + correction_scale * pred_residual
refined         = decode(refined_norm)                 # exactly once
refined         = physical_constraints_and_mask(refined)
```

Precipitation is made nonnegative only after reconstruction in physical space;
the normalized residual is not broadly clipped. Temperatures are not clipped,
and `tmin > tmax` rates/excess are emitted as diagnostics rather than hidden.

### Diffusion

Forward noising is
`x_t=sqrt(alpha_bar_t)*z+sqrt(1-alpha_bar_t)*epsilon`. The configured objective
can predict epsilon, velocity, or clean sample, and the loss target and clean
estimate use the matching schedule conversion. Optional reconstruction, bias,
gradient, Laplacian, multiscale, and tail terms are applied only to a valid
clean-residual estimate. Sampling uses the same schedule and generalized DDIM;
`eta=0` is deterministic, while nonzero eta adds stochasticity. Exact DDPM
posterior variance at eta 1 applies only to a consecutive full timestep grid.

### Flow matching

The implemented path is:

```text
x0 ~ N(0, I)
x1 = normalized residual
xt = (1-t) * x0 + t * x1
target_velocity = x1 - x0
```

Inference integrates from `t=0` to `t=1`; Euler, midpoint, and Heun direction
are tested with an analytic constant-velocity field. The terminal state is
interpreted once as the residual.

### Transformer geometry

Patchification is row-major over `[B,C,H,W]`, trailing dimensions are
replicate-padded to patch multiples, and unpatchification removes that padding
exactly. Rectangular and non-divisible grids round-trip to numerical precision.
Absolute coordinate channels and coordinate-aligned overlap noise make the
geographic conditioning consistent across tiled inference; Hann stitching
downweights tile boundaries. Actual checkerboard/stripe skill diagnostics still
require trained production checkpoints and are included in the evaluation via
gradient/spatial-structure metrics and maps.

## Objective and stability changes

The mathematically required diffusion/flow objective remains primary. The four
YAMLs now enable configurable, backward-compatible clean-residual terms:

- Huber reconstruction (`0.10`);
- mean-bias penalty (`0.02`);
- horizontal/vertical gradient loss (`0.05`);
- Laplacian loss (`0.01`);
- multiscale loss (`0.02`); and
- normalized-tail loss above magnitude 2 (`0.02`).

All default to zero for older configurations. Per-variable weights are explicit
in `[ppt,tmax,tmin]` order. Other stabilization includes training-period
residual standardization, gradient clipping, optional warmup, deterministic
seeds, configurable correction scale, stable cosine diffusion defaults, Heun
flow integration, and mandatory trained checkpoints. EMA was not added: doing
so would introduce another checkpoint/inference contract without real
validation evidence. It remains a possible measured follow-up, not a cosmetic
post-processing correction.

## Production output and evaluation

Refined inference writes one file per exact configured day. Independent
inference keeps the historical path, while validation is isolated below a
`validation/` child:

```text
<output>/<case>/<case>_<refinement_type>_refined_YYYYMMDD.nc
<output>/validation/<case>/<case>_<refinement_type>_refined_YYYYMMDD.nc
```

Canonical variables are refined ensemble means. Each file also contains
`<var>_phase1`, `<var>_member_NNN`, `<var>_ensemble_spread`, and the canonical
mask. Dates, ascending lat/lon, variable order, physical units, seed scheme,
Phase-1 and Phase-2 paths/fingerprints, config path/fingerprint, and discovery
pattern are stored as provenance. Cross-day provenance and the embedded Phase-1
baseline are checked by the evaluator.

The streaming evaluator supports the deterministic baseline plus any or all
four named refiners on exactly paired days. It reports, separately for each
variable:

- bias, absolute bias, MAE, RMSE, centered RMSE, Pearson correlation, temporal
  correlation, daily and climatological spatial-pattern correlation, standard
  deviation ratio, quantile errors, empirical Wasserstein-1 and KS distribution
  distances, distribution plots, gradient agreement/correlation/RMSE, and a
  mask-aware adjacent-difference roughness ratio/spatial-smoothing index;
- precipitation dry/wet frequency, precision, recall, false-alarm rate/ratio,
  wet-day intensity, target-wet RMSE, maxima, q90/q95/q99/q99.9 tail errors,
  heavy-event bias, and extreme RMSE;
- temperature q1/q5/q95/q99 errors, cold/warm extreme RMSE, optional threshold
  exceedance frequencies, and `tmin > tmax` violation diagnostics;
- month, season, exact target wet/dry precipitation, paired-reservoir
  moderate/extreme target regimes, optional elevation bins, and arbitrary
  coordinate-aligned region masks (including user-provided coastal/inland
  masks); and
- empirical ensemble CRPS, spread, RMS spread/skill ratio, interval coverage
  and width, coverage reliability error, and member diversity.

Scalar moments and maps are exact. Distribution/tail diagnostics use a seeded,
documented reservoir only when the paired sample count exceeds its configured
capacity; one priority stream per variable retains identical truth cells for
every method. The evaluator generates Phase-1/refined/PRISM climatologies, bias,
RMSE, RMSE improvement, absolute-bias improvement, and distribution/extreme
figures with shared comparison scales.

## Tests and measured implementation results

### Commands executed

```bash
mamba run -n Prithvi python -m pytest -q
# 423 passed, 2 warnings in 444.86s

mamba run -n Prithvi python -m pytest -q \
  tests/test_refinement_evaluation.py \
  tests/test_refinement_evaluation_cli.py
# 26 passed, 1 dependency warning in 9.00s

mamba run -n Prithvi python -m pytest -q \
  tests/test_narr_prism_refinement_notebook.py
# 19 passed in 48.25s

mamba run -n Prithvi python -m pytest -q \
  tests/test_prism_source_rebind.py \
  tests/test_prism_checkpoint_contract.py \
  tests/test_narr_prism_refinement_notebook.py
# 29 passed in 39.26s

mamba run -n Prithvi python -m ruff check \
  granitewxc/refinement \
  granitewxc/utils/prism_checkpoint.py \
  granitewxc/utils/prism_source_rebind.py \
  examples/NARR_PRISM/narr_prism_refinement.py \
  examples/NARR_PRISM/narr_prism_refinement_inference.py \
  examples/NARR_PRISM/narr_prism_artifacts.py \
  examples/NARR_PRISM/refinement_smoke.py \
  examples/NARR_PRISM/evaluate_refinement.py \
  tests/test_refinement_phase2_contract.py \
  tests/test_narr_prism_refinement_inference.py \
  tests/test_narr_prism_refinement_notebook.py \
  tests/test_prism_source_rebind.py \
  tests/test_refinement_evaluation.py \
  tests/test_refinement_evaluation_cli.py \
  --select E9,F,B905
# All checks passed

git diff --check
# clean
```

The full Ruff profile was not used as a completion claim because this
repository currently contains many pre-existing 79-column/style findings. No
unrelated mass formatting was performed.

### Real artifact and notebook verification

The recovery/rebuild and the final production-mode notebook check used these
commands in the same `Prithvi` environment:

```bash
mamba run -n Prithvi env PYTHON_BIN=python PREPROCESS_SHARDS=48 \
  bash examples/NARR_PRISM/run_preprocess.sh \
  --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml --shards 48

mamba run -n Prithvi python -m granitewxc.utils.prism_source_rebind \
  examples/NARR_PRISM/artifacts/recovery_provenance/2026-08-12/normalization_manifest_historical.json \
  examples/NARR_PRISM/preprocessed/narr_prism_California/scalars/normalization_manifest.json \
  examples/NARR_PRISM/artifacts/recovery_provenance/2026-08-12/source_rebind_contract.json

mamba run -n Prithvi jupyter nbconvert --to notebook --execute \
  examples/NARR_PRISM/notebooks/narr_prism_refinement.ipynb \
  --output /tmp/narr_prism_refinement_production_defaults.ipynb \
  --ExecutePreprocessor.timeout=1800
```

The preprocessing command produced exactly 6,575 training, 730 validation,
and 3,653 inference files. The last split is predictor-only. The notebook ran
with its unmodified production defaults (`SMOKE_TEST=False`), resolved the
external artifact portal, accepted only the signed device-number rebind, loaded
all 208 Phase-1 tensors with no missing/unexpected/shape-mismatched keys, and
confirmed the Phase-1 fingerprint `84c8509c…`. Its real first batch had input
shape `[1, 32, 320, 320]`, target shape `[1, 3, 256, 256]`, and residual-valid
fraction 0.651718. The selected diffusion Transformer initialized 8,200,128
trainable refinement parameters while all 252,744,266 Phase-1 parameters
remained frozen. Training/inference/evaluation switches were deliberately off;
this was an executable production setup check, not a trained Phase-2 result.

### Per-method synthetic evidence

The overfit test fixes the same sample, diffusion noise/timestep or flow path,
and seed for 30 optimizer steps. It is a capacity/direction test, not a
generalization or climate-skill result.

| Head | Initial objective | Final objective | Reduction | Forward/backward/sample |
| --- | ---: | ---: | ---: | --- |
| Diffusion UNet | 0.906346 | 0.048519 | 94.65% | Pass |
| Diffusion Transformer | 1.534092 | 0.009425 | 99.39% | Pass |
| Flow Matching UNet | 1.077023 | 0.297376 | 72.39% | Pass |
| Flow Matching Transformer | 1.292314 | 0.007635 | 99.41% | Pass |

### Per-method real-data mechanics evidence

Each head was also exercised sequentially on the same regenerated 1996-01-01
training tile on `cuda:1`: real Phase 1, real scalers, real predictors and
PRISM target, one objective/backward pass, and a two-step sample or ODE
integration. All four produced `[1, 3, 256, 256]` outputs, finite values on the
valid mask, zeros for invalid normalized residuals, NaNs for invalid physical
outputs, and nonnegative physical precipitation. Every Phase-1 parameter
remained frozen with no gradient; every refiner had finite nonzero gradients.

| Head | Real-batch objective | Refiner gradient norm | Two-step reconstruction |
| --- | ---: | ---: | --- |
| Diffusion UNet | 1.113680 | 0.145386 | Pass |
| Diffusion Transformer | 1.477615 | 3.881091 | Pass |
| Flow Matching UNet | 1.827922 | 0.924171 | Pass |
| Flow Matching Transformer | 2.193870 | 3.955415 | Pass |

For this bounded mechanics check only, residual statistics were fitted to that
single batch and were not saved. The evidence JSON is explicitly marked
`scientific_validation=false` (SHA-256
`0594964adc88faf2f60dbb7182ba5977e82d48d9dcf4cc267e2eb5f8829ef6e0`).
Production training still fits residual normalization over the complete
1996–2013 loader before optimization.

Additional tests cover residual sign, exact zero-residual reconstruction,
normalization inversion, invalid masks, finite gradients, output dimensions,
fixed-seed reproducibility, all diffusion parameterizations, diffusion clean
oracle sampling, flow integration direction/solvers, patch ordering,
non-divisible padding/crop, Phase-2 save/resume, partial accumulation flush,
daily leap-date inference, overlap-identical noise, coordinates/units/masks,
and evaluation provenance rejection.

### Available real Phase-1 metrics

The existing 2016–2025 comparison notebook contains historical Phase-1
aggregates. They were not treated as newly measured evidence in this audit
because those daily Phase-1 prediction files were deleted in the artifact
incident and the old notebook cache identity was too weak. Predictor-only
inference inputs are fully regenerated, but the deterministic prediction
command in the production section must recreate those outputs. The historical
aggregates are recorded only as baseline context:

| Variable | Phase-1 mean | PRISM mean | Mean bias | Mean cellwise RMSE |
| --- | ---: | ---: | ---: | ---: |
| `ppt` (mm/day) | 1.086763 | 1.379408 | -0.292645 | 3.362612 |
| `tmax` (degC) | 21.140944 | 20.506580 | +0.634364 | 2.322303 |
| `tmin` (degC) | 6.348609 | 6.852247 | -0.503639 | 2.240355 |

No Phase-2 validation metric exists yet. Synthetic loss reduction must not be
used to select a scientific method.

## Files added or materially changed

Core implementation:

- `granitewxc/refinement/{base,checkpoint,config,diffusion,flow_matching,target_space,training,two_phase}.py`
- `granitewxc/refinement/evaluation.py`
- `granitewxc/refinement/{__init__,cache,io}.py`
- `granitewxc/utils/{normalization,prism_checkpoint}.py` (signed support mask
  and Phase-1 semantic contract);
- `granitewxc/utils/prism_source_rebind.py` (strictly authenticated legacy
  device-number relocation)

NARR–PRISM workflow:

- `NARR_PRISM_subdomain.yaml` (explicit evaluation-unit metadata) and the four
  refinement YAMLs listed at the start of this report;
- `preproc_narr_prism.py` and `compute_scalars_narr_prism.py` (target-free
  inference products and training-only support-mask persistence);
- `narr_prism_dataset.py` (observation-free validation/inference modes);
- `narr_prism_inference.py` (target-free deterministic daily inference,
  authenticated support masks, and isolated validation/inference splits);
- `narr_prism_refinement.py`;
- `narr_prism_refinement_inference.py`;
- `evaluate_refinement.py`;
- `refinement_smoke.py`;
- `narr_prism_artifacts.py`, the tracked artifact links, and `.gitignore`
  (safe external artifact portal);
- `notebooks/narr_prism_refinement.ipynb`; and
- cache hardening in `notebooks/Compare_inference_prism.ipynb`.

Documentation and tests:

- `docs/STOCHASTIC_REFINEMENT.md`;
- `tests/test_refinement_phase2_contract.py`;
- `tests/test_narr_prism_refinement_inference.py`;
- `tests/test_narr_prism_refinement_notebook.py`;
- `tests/test_narr_prism_target_free.py`;
- `tests/test_refinement_evaluation.py`;
- `tests/test_refinement_evaluation_cli.py`;
- `tests/test_narr_prism_inference_selection.py`,
  `tests/test_prism_normalization.py`, and
  `tests/test_prism_checkpoint_contract.py`;
- `tests/test_prism_source_rebind.py`; and
- focused fixture/config/checkpoint test updates.

## Commands for full production runs

Run from the repository root after configuring and validating the local
artifact portal. The four training commands use the same Phase-1 checkpoint:

```bash
PHASE1_CKPT=examples/NARR_PRISM/experiments/checkpoints/narr_prism_California/last.ckpt

mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_artifacts.py

# Regenerate training products, then training-only scalars/support mask, then
# target-bearing validation and predictor-only independent inference products.
# This ordering is required because training constructs both loaders.
mamba run -n Prithvi env PYTHON_BIN=python PREPROCESS_SHARDS=48 \
  bash examples/NARR_PRISM/run_preprocess.sh \
  --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml --shards 48

# This recovered schema-4 checkpoint was trained before the raw filesystem's
# Linux device number changed. Authenticate the proven device-only relocation;
# this command rejects changes to any path, inode, size, mtime, day, scalar,
# grid, channel, transform, or other immutable contract field.
mamba run -n Prithvi python -m granitewxc.utils.prism_source_rebind \
  examples/NARR_PRISM/artifacts/recovery_provenance/2026-08-12/normalization_manifest_historical.json \
  examples/NARR_PRISM/preprocessed/narr_prism_California/scalars/normalization_manifest.json \
  examples/NARR_PRISM/artifacts/recovery_provenance/2026-08-12/source_rebind_contract.json

mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_refinement.py train \
  --config examples/NARR_PRISM/NARR_PRISM_diffusion_unet.yaml \
  --phase1-checkpoint "$PHASE1_CKPT" --device cuda:0

mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_refinement.py train \
  --config examples/NARR_PRISM/NARR_PRISM_diffusion_transformer.yaml \
  --phase1-checkpoint "$PHASE1_CKPT" --device cuda:0

mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_refinement.py train \
  --config examples/NARR_PRISM/NARR_PRISM_flow_matching_unet.yaml \
  --phase1-checkpoint "$PHASE1_CKPT" --device cuda:0

mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_refinement.py train \
  --config examples/NARR_PRISM/NARR_PRISM_flow_matching_transformer.yaml \
  --phase1-checkpoint "$PHASE1_CKPT" --device cuda:0
```

Resume a run with `--resume` (the YAML checkpoint directory's `last.ckpt`) or
`--resume /explicit/path.ckpt`.

Use the complete 2014–2015 validation products generated by the pipeline above
for method selection. Prediction never receives validation observations; the
evaluator opens PRISM separately. The output split and its configured bounds
are embedded in every daily file and validated during evaluation:

```bash
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_inference.py \
  --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml \
  --checkpoint "$PHASE1_CKPT" --split validation \
  --output-dir examples/NARR_PRISM/experiments/inference_output \
  --device cuda

mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_refinement.py infer \
  --config examples/NARR_PRISM/NARR_PRISM_diffusion_unet.yaml \
  --phase1-checkpoint "$PHASE1_CKPT" \
  --refinement-checkpoint examples/NARR_PRISM/experiments/refinement_checkpoints/diffusion_unet/best.ckpt \
  --split validation --ensemble-size 10 --seed 1234 --device cuda:0 \
  --output examples/NARR_PRISM/experiments/validation_daily/diffusion_unet

mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_refinement.py infer \
  --config examples/NARR_PRISM/NARR_PRISM_diffusion_transformer.yaml \
  --phase1-checkpoint "$PHASE1_CKPT" \
  --refinement-checkpoint examples/NARR_PRISM/experiments/refinement_checkpoints/diffusion_transformer/best.ckpt \
  --split validation --ensemble-size 10 --seed 1234 --device cuda:0 \
  --output examples/NARR_PRISM/experiments/validation_daily/diffusion_transformer

mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_refinement.py infer \
  --config examples/NARR_PRISM/NARR_PRISM_flow_matching_unet.yaml \
  --phase1-checkpoint "$PHASE1_CKPT" \
  --refinement-checkpoint examples/NARR_PRISM/experiments/refinement_checkpoints/flow_matching_unet/best.ckpt \
  --split validation --ensemble-size 10 --seed 1234 --device cuda:0 \
  --output examples/NARR_PRISM/experiments/validation_daily/flow_matching_unet

mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_refinement.py infer \
  --config examples/NARR_PRISM/NARR_PRISM_flow_matching_transformer.yaml \
  --phase1-checkpoint "$PHASE1_CKPT" \
  --refinement-checkpoint examples/NARR_PRISM/experiments/refinement_checkpoints/flow_matching_transformer/best.ckpt \
  --split validation --ensemble-size 10 --seed 1234 --device cuda:0 \
  --output examples/NARR_PRISM/experiments/validation_daily/flow_matching_transformer

mamba run -n Prithvi python examples/NARR_PRISM/evaluate_refinement.py \
  --config examples/NARR_PRISM/NARR_PRISM_diffusion_unet.yaml \
  --split validation \
  --phase1-dir examples/NARR_PRISM/experiments/inference_output/validation/narr_prism_California \
  --method diffusion_unet=examples/NARR_PRISM/experiments/validation_daily/diffusion_unet/validation/narr_prism_California \
  --method diffusion_transformer=examples/NARR_PRISM/experiments/validation_daily/diffusion_transformer/validation/narr_prism_California \
  --method flow_matching_unet=examples/NARR_PRISM/experiments/validation_daily/flow_matching_unet/validation/narr_prism_California \
  --method flow_matching_transformer=examples/NARR_PRISM/experiments/validation_daily/flow_matching_transformer/validation/narr_prism_California \
  --output-dir examples/NARR_PRISM/experiments/refinement_validation \
  --reservoir-size 200000 --seed 1234
```

After selection, run full daily ensemble inference with the chosen settings
(all four commands are retained here for the requested final comparison):

```bash
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_refinement.py infer \
  --config examples/NARR_PRISM/NARR_PRISM_diffusion_unet.yaml \
  --phase1-checkpoint "$PHASE1_CKPT" \
  --refinement-checkpoint examples/NARR_PRISM/experiments/refinement_checkpoints/diffusion_unet/best.ckpt \
  --ensemble-size 10 --seed 1234 --device cuda:0 \
  --output examples/NARR_PRISM/experiments/refined_daily/diffusion_unet

mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_refinement.py infer \
  --config examples/NARR_PRISM/NARR_PRISM_diffusion_transformer.yaml \
  --phase1-checkpoint "$PHASE1_CKPT" \
  --refinement-checkpoint examples/NARR_PRISM/experiments/refinement_checkpoints/diffusion_transformer/best.ckpt \
  --ensemble-size 10 --seed 1234 --device cuda:0 \
  --output examples/NARR_PRISM/experiments/refined_daily/diffusion_transformer

mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_refinement.py infer \
  --config examples/NARR_PRISM/NARR_PRISM_flow_matching_unet.yaml \
  --phase1-checkpoint "$PHASE1_CKPT" \
  --refinement-checkpoint examples/NARR_PRISM/experiments/refinement_checkpoints/flow_matching_unet/best.ckpt \
  --ensemble-size 10 --seed 1234 --device cuda:0 \
  --output examples/NARR_PRISM/experiments/refined_daily/flow_matching_unet

mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_refinement.py infer \
  --config examples/NARR_PRISM/NARR_PRISM_flow_matching_transformer.yaml \
  --phase1-checkpoint "$PHASE1_CKPT" \
  --refinement-checkpoint examples/NARR_PRISM/experiments/refinement_checkpoints/flow_matching_transformer/best.ckpt \
  --ensemble-size 10 --seed 1234 --device cuda:0 \
  --output examples/NARR_PRISM/experiments/refined_daily/flow_matching_transformer
```

Run deterministic daily Phase-1 inference first if its daily directory does not
already exist. Then evaluate all five predictions against PRISM on the exact
2016–2025 dates:

```bash
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_inference.py \
  --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml \
  --checkpoint "$PHASE1_CKPT" \
  --output-dir examples/NARR_PRISM/experiments/inference_output \
  --device cuda

mamba run -n Prithvi python examples/NARR_PRISM/evaluate_refinement.py \
  --config examples/NARR_PRISM/NARR_PRISM_diffusion_unet.yaml \
  --split inference \
  --phase1-dir examples/NARR_PRISM/experiments/inference_output/narr_prism_California \
  --method diffusion_unet=examples/NARR_PRISM/experiments/refined_daily/diffusion_unet/narr_prism_California \
  --method diffusion_transformer=examples/NARR_PRISM/experiments/refined_daily/diffusion_transformer/narr_prism_California \
  --method flow_matching_unet=examples/NARR_PRISM/experiments/refined_daily/flow_matching_unet/narr_prism_California \
  --method flow_matching_transformer=examples/NARR_PRISM/experiments/refined_daily/flow_matching_transformer/narr_prism_California \
  --start 2016-01-01 --end 2025-12-31 \
  --output-dir examples/NARR_PRISM/experiments/refinement_evaluation \
  --reservoir-size 200000 --seed 1234
```

Optional `--elevation-file`, `--elevation-bins`, and repeated
`--region-mask coast=... --region-mask inland=...` arguments add terrain and
geographic strata. Repeated `--temperature-lower-threshold NAME=DEGC` and
`--temperature-upper-threshold NAME=DEGC` arguments add configured threshold
exceedance diagnostics; `--wet-day-threshold MM_PER_DAY` controls the exact
precipitation occurrence strata (default 1 mm/day). Use the generic notebook as a front end by setting
`REFINEMENT_CONFIG` and the checkpoint/run parameters near its start. CI smoke
execution is:

```bash
NARR_PRISM_SMOKE_TEST=1 NARR_PRISM_REFINEMENT_DEVICE=cpu \
  NARR_PRISM_REFINEMENT_OUTPUT_DIR=/tmp/narr_prism_refinement_smoke \
  mamba run -n Prithvi jupyter nbconvert --to notebook --execute \
  examples/NARR_PRISM/notebooks/narr_prism_refinement.ipynb \
  --output /tmp/NARR_PRISM_refinement_smoke.ipynb
```

## Remaining limitations and recommendation

1. Train four independent Phase-2 checkpoints and run the paired 2014–2015
   validation evaluation before selecting hyperparameters. Keep 2016–2025 as
   the independent inference period; do not fit normalizers or thresholds on it.
2. Run the full 2016–2025 evaluator and inspect precipitation occurrence/tails,
   temperature extremes, terrain strata, ensemble calibration, and boundary
   maps. A lower aggregate RMSE alone is insufficient.
3. Raw PRISM target units should eventually be backed by a signed source
   manifest because the source files themselves omit unit attributes.
4. Joint/selected Phase-1 fine-tuning can train and resume with combined
   checkpoints, but frozen production inference intentionally rejects that
   format until a separate audited combined-checkpoint inference contract is
   implemented.
5. Phase-2 schema-1 checkpoints are intentionally not loadable: their
   precipitation residual baseline was the ungated hurdle amount latent. There
   is no lossless weight migration to schema 2's encoded final physical
   baseline, so such refinement heads must be retrained.
6. Legacy raw-source signatures include Linux `st_dev`, which can change after
   disk re-enumeration. The recovered schema-4 checkpoint is protected by the
   exact device-only attestation above; a future signature schema should omit
   volatile device numbers or add signed content hashes while retaining a
   legacy reader.

**Recommendation:** no one of the four refiners is scientifically recommended
yet. All four are operational under automated tests and real-data one-batch
forward/backward/sampling checks, but no trained Phase-2 checkpoint or
independent paired validation result is available in this checkout. Select the
method only after running the same paired evaluator on validation-split daily
products and obtaining measured 2014–2015 metrics; use the 2016–2025 command
above only for the final independent evaluation. Do not infer a winner from
the synthetic overfit or mechanics tables.
