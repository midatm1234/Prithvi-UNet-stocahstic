> **2026-09-15 takeover correction.** The historical experiments and verdicts below
> are preserved. The later native-pair audit found frozen historical weights,
> mutable normalization, dropped partial accumulation, and a missing quantile
> guardrail. See [current takeover evidence](temporal_codex_handoff.md),
> [native/NARR corrections](temporal_native_correctness_takeover.md), and
> [pretrained provenance](temporal_pretrained_transfer_takeover.md).
> Architectural history gains and verified pretrained transfer are separate claims.

# Temporal Prithvi-UNet: architecture, tensor shapes, and contracts

Branch `Prithvi-UNet_temporal_model`, based on `Prithvi-UNet-stochastic_refinement`
@ `dd48c6f`. Companion documents: `docs/temporal_model_research.md` (why these
choices), `docs/temporal_model_results.md` (measured outcomes and limitations).

---

## 1. The task, stated precisely

**Causal, same-day, sequence-conditioned downscaling.** For each output date `t`:

```
z_t          = spatial_encoder(predictors_t, static)          # existing, freezable
h_t, S_t     = temporal_backend(z_t, S_{t-1}, tau_t, r_t)     # new
y_hat_t      = spatial_decoder(z_t + g ⊙ proj(h_t), skips_t)  # existing decoder
```

* `z_t` is the U-Net **bottleneck** latent.
* `tau_t` is the calendar/temporal metadata vector, `r_t = Δt_actual / cadence`.
* `S_t` is the recurrent/SSM hidden state; `g` is a learnable per-channel gate.

What this is **not**, and the code-level reason it cannot become one:

| Property | Guarantee |
|---|---|
| zero predictor→target lead time | `mode: downscaling` forces `lead_time_days == 0`; a non-zero value raises `TemporalConfigError`. Targets are joined to predictors **by timestamp**, never by index. |
| not a free-running forecast | The model consumes the coarse predictors at every date. It advances state through them; it does not integrate its own output forward. |
| no target leakage | `y` is used only by the loss and the NetCDF writer. `TemporalSequenceModel._frame` places `batch["y"]` into the frame dict solely because the base model reads its *shape* for output cropping; no target value reaches any learned layer. |
| no predicted-output feedback | Not implemented. There is no path from `y_hat_{t-1}` to the input of frame `t`. |
| causality | Structural: the loop is step-wise and the state only ever moves forward. Verified by `test_causality_future_inputs_do_not_affect_earlier_outputs`, which perturbs the last frame and requires every earlier output to be **bit-for-bit** unchanged. |

A `forecasting` mode exists in the schema and requires an explicit positive
`lead_time_days` plus `time_features.include_lead_time`. **No forecasting
configuration is shipped**, because defining initialization time, available future
forcing and predicted-state feedback is a separate piece of work; the mode is
present so that the downscaling guarantees above are enforced by an explicit
branch rather than by silence.

---

## 2. Two distinct time axes

| Axis | Extent | Meaning |
|---|---|---|
| `data.n_input_timestamps` | **1** in both cases | The Prithvi-WxC backbone's *native* multi-timestamp axis, flattened into channels before patch embedding. |
| `temporal.context_length` | 7 (SA), 5 (NARR) | The **outer** sequence axis this branch adds. |

They are never merged, and history is never faked by duplicating a snapshot into
the native axis. This is safe here for a verified reason: `PrithviWxCEncoderDecoder.forward(x)`
takes only the token tensor — no `input_time`, no `lead_time`, no `time_encoding`
— so the native axis carries no temporal metadata at all and is size 1. See
`docs/temporal_model_research.md` §2.2 for the check.

---

## 3. Tensor shapes, end to end

### 3.1 Dataset output (per sample, before collation)

```
x                    [T, C_pred,   H, W]     predictors, already on the fine grid
y                    [T, C_target, H, W]     targets, SAME dates as x
static_x, static_y    [C_static, H, W]        time-invariant (omitted if C_static == 0)
time_features        [T, F]                  calendar metadata
interval_ratio       [T]                     Δt_actual / cadence
reset                [T]  bool               True where state must reset
__target_valid_mask  [T, C_target, H, W]     finite-target mask
```

Collation adds a leading batch axis. Concrete values:

| | SA (`cordex`) | NARR/PRISM |
|---|---|---|
| `T` | 7 | 5 |
| `C_pred` | 15 (u,v,q,t,z × 850,700,500) | 31 (5 vars × 3 levels + elev + 16 masks) |
| `C_target` | 2 (`pr` mm/day, `tasmax` K) | 3 (`ppt`, `tmax`, `tmin`) |
| `C_static` | 1 (`orog`) | **0** — elevation and masks travel in `C_pred` |
| `H, W` | 128 × 128 | 256 × 256 (crop) |
| `F` | 5 | 5 |

⚠️ `num_static_channels: 0` is a **meaningful value** for NARR/PRISM, not a missing
one. Coercing it to 1 (e.g. via `x or 1`) silently strips a real predictor channel
off every frame. Three call sites guard against this explicitly.

### 3.2 Inside the model (SA numbers)

```
x                       [B, 15, 128, 128]
  ├─ normalize, PatchEmbed (stride 1, padding 'same')   -> [B, 512, 128, 128]
  ├─ + static embedding                                 -> [B, 512, 128, 128]
  ├─ downsampling_layers x3 (skips)                     -> 128, 64, 32, 16
  ├─ cat(deepest skip) -> conv_before_backbone          -> [B, 1024, 128, 128]
  ├─ PrithviWxCEncoderDecoder (8 blocks, 16x16 mask units)
  ├─ conv_after_backbone                                -> [B, 1024, 128, 128]
  ├─ interpolate to bottleneck-skip resolution          -> [B, 1024,  16,  16]
  │
  ├──── ###  TEMPORAL HOOK  ###   _apply_temporal_latent
  │        delta, S_t = temporal_adapter(out, S_{t-1}, tau_t, r_t)
  │        out = out + delta
  │
  ├─ upsample x3 with skips                             -> [B, 640, 128, 128]
  └─ output_conv_block                                  -> [B, 2, 128, 128]
```

The bottleneck is where the temporal state lives. Measured with a spy on the hook:

```
BOTTLENECK tensor at temporal hook: (2, 1024, 16, 16)
```

**Why here.** The post-backbone map is `[B, 1024, 128, 128]` — 64× larger. Seven
frames of state at the bottleneck cost ~29 MB in fp32 at batch 4; at full
resolution ~1.9 GB before autograd. The bottleneck still has real spatial extent
(16 × 16 latent cells, each ~8 target pixels ≈ 80 km on the SA grid), so a 3 × 3
convolutional kernel moves information ~240 km per day — the right order for daily
synoptic advection. Multi-scale injection is deliberately **not** enabled by
default: `latent.site` accepts only `bottleneck`, so the option cannot be set
without implementing it.

### 3.3 The adapter

```
TemporalLatentAdapter
  pre_norm  GroupNorm(8, 1024)
  in_proj   Conv2d(1024 -> 256, k=1)
  backend   ConvGRUBackend | TemporalMambaBackend      on [B, 256, 16, 16]
  out_proj  Conv2d(256 -> 1024, k=1)
  gate      Parameter [1, 1024, 1, 1]   init = latent.adapter_init_gate
```

Parameter counts on the real SA model (246,408,968 total):

| backend | temporal params | share |
|---|---|---|
| ConvGRU (2 layers, dilations 1/2) | 7,642,752 | 3.10% |
| Mamba (2 SSD blocks) | 1,516,576 | 0.62% |

### 3.4 ConvGRU backend

```
h_t = (1 - z) ⊙ h_{t-1} + z ⊙ tanh(W_h * [x_t, r ⊙ h_{t-1}])
```

Gates are 2-D convolutions, so the state *is* a spatial field and the kernel
couples neighbouring cells every step. Gate biases are initialized toward
retention (reset gate open, update gate closed) so an untrained cell starts near
"carry the previous state". `dilations: [1, 2]` gives multi-timescale/multi-scale
memory: layer 1 couples adjacent latent cells, layer 2 reaches ~3 cells, without
extra parameters.

State: `[[B, 256, 16, 16]]` per layer (ConvLSTM: a `(h, c)` pair per layer, which
is why ConvGRU is the default).

**Physical-time handling — an honest asymmetry.** A ConvGRU has no continuous-time
interpretation and there is no principled way to "advance it by 2 days". Rather
than invent one, this backend relies on the fact that **sequences are built only
from strictly contiguous runs**, so within a sequence the step is always exactly
the cadence. The elapsed-time and sequence-start metadata still reach the cell
through FiLM. This is a real difference from the Mamba backend, which does admit a
correct rescaling (§3.5).

### 3.5 Temporal Mamba (SSD) backend

**Scan axis is time.** `(H, W)` folds into the batch; the scan runs over `t`, one
`C`-vector per grid cell. Spatial coupling comes from a depthwise 3 × 3 + pointwise
1 × 1 convolution applied to every frame *before* the scan, interleaved across
`n_layers`. Flattening `H·W` into the scan axis — what VMamba/Vim/MambaVision do —
would model no temporal dependence at all.

Recurrence (Mamba-2 / SSD; `A` is scalar-×-identity per head, one `Δ` per head):

```
Ā_t = exp(dt_t · A_h)
h_t = Ā_t · h_{t-1} + dt_t · x_t ⊗ B_t
y_t = ⟨h_t, C_t⟩ + D_h · x_t
```

`in_proj` width `2·d_inner + 2·d_state + nheads`, matching upstream's fused layout.
With `hidden_channels 256`, `expand 2`, `headdim 32`: `d_inner = 512`, `nheads = 16`.

State per block:

```
ssm  [B·H·W, nheads, headdim, d_state]   = [256, 16, 32, 16] at B=1, 16x16
conv [B·H·W, d_inner + 2·d_state, d_conv - 1]
```

The `conv` entry is the causal depthwise time-convolution's history. **It must be
carried across chunk boundaries too** — omitting it is a silent source of
chunk/single-pass disagreement.

**Physical elapsed time.** `dt` is a learned, input-dependent, dimensionless gate;
the Mamba papers are explicit that it has no physical-time meaning. Because the
block is a zero-order-hold discretization, integrating over an interval `r` times
longer means using `dt·r`, so with `time_scale_from_metadata: true`:

```python
dt = F.softplus(dt_raw + self.dt_bias)     # learned
dt = dt * interval_ratio                    # measured Δt / cadence
```

A step across a removed 29 February decays the state by `exp(2·dt·A)`. The learned
timescale and the physical interval stay separate quantities that are multiplied.
**This is our adaptation, not endorsed by the Mamba papers** (see research doc §3.3).

**Kernel policy.** `implementation: fused` requires `mamba_ssm` and raises
`MambaUnavailableError` if absent — it never substitutes a different model.
`reference` is the in-repo pure-PyTorch SSD recurrence: same mathematics, same
parameters, slower. `auto` prefers fused and falls back with a loud warning plus a
recorded `implementation_used` provenance field. In this environment `mamba_ssm`
is not importable (Windows/CUDA), so the shipped configs set `reference`
explicitly.

### 3.6 Near-identity initialization, and the zero-init trap

`gate` is a learnable per-channel vector initialized to `adapter_init_gate`.

| setting | behaviour | use |
|---|---|---|
| `0.0` | **exact** identity: emitted field is bit-for-bit the frame-independent prediction. But every temporal parameter's gradient is exactly zero, so the module can never start learning. | legacy-parity proof, and the `spatial_ft` control |
| `1e-3` (default) | perturbation ~1e-5 relative; **every** temporal parameter receives non-zero gradient from step 1. | training |

Measured on the real SA model with real checkpoint weights:

```
legacy parity (gate=0)     : bitwise identical = True   maxdiff = 0.0
near-identity (gate=1e-3)  : mean relative deviation = 8.8e-06 (ConvGRU) / 9.4e-06 (Mamba)
temporal params with zero gradient : 0 / 23 (ConvGRU), 0 / 37 (Mamba)
```

A related trap inside `TimeFeatureFiLM`: its output layer is initialized to std
`1e-3` rather than exactly zero. With exact zeros the *first* linear layer's
gradient is identically zero (`dL/dW1 ∝ W2ᵀ = 0`), so half the module could not
train. This was caught by the gradient-flow test before any training run.

---

## 4. Temporal metadata (`F = 5`)

| channel | definition |
|---|---|
| `season_sin`, `season_cos` | `sin/cos(2π · year_fraction)`, where `year_fraction` divides day-of-year by the length of **that** year in **that** calendar (366 in a Gregorian leap year, 365 in a no-leap year, 360 in a 360-day calendar) |
| `log_gap_excess` | `sign(e)·log1p(|e|)` with `e = (Δt − cadence)/cadence`. `0` for a regular step, `log 2 ≈ 0.693` for a dropped day; compresses the 36,160-day SA jump so it cannot dominate the feature scale |
| `is_sequence_start` | `1` only at a genuine state reset |
| `state_age` | `min(position_since_reset, L−1)/(L−1)`: how much history stands behind this frame, saturating at 1 |

Hour-of-day sin/cos are emitted **only** when the archive actually resolves time
of day. Both cases stamp every record `12:00:00`, so they are correctly omitted;
requesting them explicitly raises rather than training a dead feature.

`state_age` and `is_sequence_start` are defined relative to the **last reset**, not
to the extent of the current array. This is load-bearing — see §7.

Available but unused: `lead_time_steps` (forecasting mode only). Geographic and
terrain context already reaches the model through the existing static pathway and
the predictor masks, so it is not duplicated here.

---

## 5. Sequences, splits, and discontinuities

### 5.1 What the SA time axis actually looks like [measured]

```
n_times 14600  calendar standard  step_histogram {1.0: 14588, 2.0: 10, 36160.0: 1}
n_discontinuities 12   n_duplicate_timestamps 0
discontinuity dates:
  idx      0  (1961, 1, 1)  gap=nan       <- sequence start
  idx   1154  (1964, 3, 1)  gap=2.0       <- 29 Feb removed
  idx   2614  (1968, 3, 1)  gap=2.0
  idx   4074  (1972, 3, 1)  gap=2.0
  idx   5534  (1976, 3, 1)  gap=2.0
  idx   6994  (1980, 3, 1)  gap=2.0
  idx   7300  (2080, 1, 1)  gap=36160.0   <- historical -> end-century concatenation
  idx   7359  (2080, 3, 1)  gap=2.0
  ... four more leap-day gaps
```

The file **declares** a Gregorian `standard` calendar but physically omits
29 February. A model that assumed "adjacent index ⇒ adjacent day" would be wrong
in 11 places, silently. Every leap-day gap lands exactly on 1 March, which is the
signature confirming the diagnosis.

### 5.2 The pipeline

```
frames (from actual time coordinates, sorted)
  └─ date_in_range(split)                 <- SPLIT FIRST
       └─ build_runs()                    <- cut at every discontinuity
            └─ build_windows()            <- windows live strictly inside one run
```

Splitting before windowing is what makes leakage structurally impossible: a window
cannot straddle a split boundary, so no validation target is ever training
context. Measured for SA:

| split | dates | frames | runs | run length | windows | unique emitted dates |
|---|---|---|---|---|---|---|
| train | 1961-01-01 … 1976-12-31 | 5840 | 5 | 306–1460 | 1163 | 5815 |
| validation | 1977-01-01 … 1980-12-31 | 1460 | 2 | 306–1154 | 290 | 1450 |
| test | 1981-01-01 … 2000-12-31 | 7300 | 6 | 306–1460 | 1454 | 7270 |

No pair of ranges overlaps. The 2080–2099 block is excluded from training and
validation; it is the climate-change application target (§9).

Other guarantees:

* **Duplicate timestamps raise.** Guessing which copy to keep would hide a wrong
  file list.
* **Runs shorter than the window yield nothing** rather than being padded — padding
  would feed the model fabricated context.
* **One crop per window**, drawn from a `(seed, epoch, window_index)`-keyed
  generator and applied to every frame. A per-frame crop would make the state
  meaningless. Reproducible across runs and workers.
* **`stride == output_length`**, so training windows tile the record: each date is
  supervised exactly once per epoch.
* An assertion inside `__getitem__` re-checks that every step in a window is
  regular, so a bug in `build_runs` surfaces rather than silently training on a gap.

### 5.3 Predictor/target calendar mismatch

The held-out SA test files pair a **365-day** predictor axis (7300 records) with a
**Gregorian** target axis (7305 records). The underlying
`CordexDownscaleDataset` resolves this by **exact timestamp join**, dropping the
five unmatched target days; positional indexing and final-target repetition are
refused. `CordexFrameSource` sets `allow_time_mismatch=True` for exactly this case.

### 5.4 Warm-up policy at a split boundary

With `warmup_length: 2`, the first two frames of each window advance the state but
are not supervised, so every emitted frame has 2–6 days of history. The first two
dates of each *run* are therefore never emitted during training — 5815 of 5840
training dates, and 7270 of 7300 test dates.

At inference, state is carried across chunks, so warm-up is unnecessary within a
run. At a **run start** the state is genuinely zero, and the first frames are
emitted with a cold state. That is the documented deployment policy: a cold start
produces a prediction closer to the frame-independent baseline rather than
refusing to predict, and `is_sequence_start` / `state_age` tell the model that is
the situation it is in.

---

## 6. Losses

The existing per-frame spatial loss (`granitewxc/models/loss.build_loss_fn`) is
called once per emitted frame with exactly the batch layout the frame trainer
uses, and remains the dominant weight. Temporal terms are added on top. Invalid
targets are **restored to NaN** before that call, because the existing loss does
its own `isfinite` masking — this reproduces the frame trainer's behaviour rather
than approximating it.

Per-variable residuals are divided by a fixed scale from the **training** target
sigma before weighting, so units cannot decide the balance:

```
variable scales (from the SA training scalers): {'pr': 16.066, 'tasmax': 6.340}
```

| term | form | weight (SA) |
|---|---|---|
| per-frame | existing composite spatial loss | 1.0 |
| tendency | `|Δŷ − Δy| / Δt`, masked pairwise | 0.15 |
| accumulation | 3- and 5-day `pr` totals | 0.10 |
| occurrence | `P(wet)` **and** `P(wet\|wet)`, `P(wet\|dry)` | 0.05 |
| lag autocorrelation | lag-`k` ACF of anomalies | **0.0 (off)** |
| Tmax/Tmin | `relu(tmin − tmax)` | automatic when both exist |

**Tendency is not smoothness.** `|Δŷ|` is minimized by a constant field; `|Δŷ − Δy|`
is minimized by the correct evolution. `test_tendency_loss_does_not_prefer_a_constant_field`
requires a constant-output model to score strictly worse than the truth.

**Interval-aware.** `Δt` normalization means a 2-day step is not charged as a
1-day change. A shape guard raises if the ratio is passed for the whole context
window instead of the emitted frames — a bug that occurred and is now impossible
to reintroduce silently.

**Masks.** A tendency pair is valid only if *both* endpoints are; an accumulation
window only if every day in it is. A missing day is never read as 0 mm.

**Lag autocorrelation is off by default.** With 5 supervised frames per window the
per-cell lag-`k` estimate is a 5-sample statistic, i.e. mostly noise.
Autocorrelation is an *evaluation* metric instead, computed over the full
multi-year test series. The term is implemented, tested, and gated by
`min_samples`, so it can be enabled for a longer window.

**No spectral loss.** By Parseval, an unweighted complex FFT squared error equals
the spatial squared error up to a constant, so it cannot be an independent
anti-blurring mechanism. See research doc §6.

**Precipitation audit.** Units are converted to `mm/day` exactly once by the
existing dataset, which *rejects* unknown precipitation units rather than assuming
daily totals (`target_units == ['mm/day', 'K']` for SA). `pr` keeps its
`divide_only` / `p95` scaling and `softplus` non-negativity from the source
config, so physical validity is enforced at the output and signed residuals are
preserved in refinement space. The occurrence threshold is applied in **physical**
units (mm/day), so `wet_threshold: 1.0` means what it says.

---

## 7. Chunked inference — and the bug that exactness caught

Chunks tile each run with **no overlap**, so every date is written exactly once
(`run_sequence_inference` raises if the emitted count ≠ the date count). State is
carried across chunk boundaries, which makes chunked inference **exactly** equal
to a single pass rather than approximately equal.

Two real bugs were found by demanding *exact* equality on real data, both of which
a tolerance-based check would have passed:

1. `is_sequence_start` was set on the first frame of every **chunk**, mislabelling
   mid-run chunks as sequence starts.
2. `state_age` was normalized by the **chunk length** instead of the configured
   `context_length`.

Both made the time features depend on chunking, producing a ~0.02 K discrepancy.
Fixed by passing `position_offset` (absolute index within the run) and
`context_length` (from config) explicitly. After the fix:

```
recurrent: CHUNK exact=True maxdiff=0.0  per-chunk={'12': 0.0, '4': 0.0}
mamba    : CHUNK exact=True maxdiff=0.0  per-chunk={'12': 0.0, '4': 0.0}
```

Regression test: `test_time_features_are_chunk_invariant`.

**State never leaks.** Hidden state is per batch element, and a reset mask zeroes
only the selected samples. Verified: perturbing sample 0's history leaves sample
1 bit-for-bit unchanged, and resetting mid-sequence equals starting a fresh
sequence from that frame. Runs, tiles, splits and (for refinement) ensemble members
each own their state.

---

## 8. Checkpoint compatibility

Two distinct operations. Neither uses an unrestricted `strict=False`:
`load_state_dict(strict=False)` is used only so this module can *adjudicate* the
missing/unexpected lists itself.

### `initialize_from_spatial_checkpoint` — start a new temporal run

Raises unless every checkpoint tensor is consumed **and** every uninitialized model
tensor is under `temporal_adapter.`. Measured against the real SA Phase-1
checkpoint:

```
[checkpoint] spatial_init from .../checkpoints/phase1_unet/best.ckpt
  loaded            : 208 tensor(s)
  missing (temporal): 23   (ConvGRU) / 37 (Mamba)  -- all temporal_adapter.*
  missing (other)   : 0
  unexpected        : 0
  shape mismatch    : 0
  provenance        : epoch=9 step=4570 trained_temporal_steps=0
```

Negative control: renaming one decoder key makes it **reject** with that key listed
under `missing (other)` and `unexpected`. Refuses outright if the checkpoint
already contains `temporal_adapter.*` weights.

The validated contract covers variable ordering, dynamic/static channel counts,
output channels, embed dims, U-Net stage count, **target grid shape**, and scaler
digests. Grid shape matters because gridpoint target normalization bakes it into
`output_scalers_mu` (`[1, 2, 128, 128]` for SA) — loading against a different crop
would silently apply the wrong per-cell mean.

### `resume_temporal_checkpoint` — continue a trained run

Nothing may be missing. Additionally validates `architecture_version`, the backend
and geometry settings, the latent/backend blocks, and the **time-feature channel
names** — if the feature layout changed, the module's first-layer weights no longer
mean the same thing. `implementation` and `adapter_init_gate` are exempt: they
select a kernel and an initialization, not an architecture.

### Saving

`save_temporal_checkpoint` writes atomically (temp file + `os.replace`) and records
architecture version, the full temporal config, backend and
`mamba_implementation_used`, time-feature names, output variables, the contract,
normalization scales, per-split data provenance, metrics, environment, git
commit/branch/dirty, and optimizer + scheduler state.

**`trained_temporal_steps` is recorded explicitly and is 0 for a merely migrated
checkpoint.** A migrated checkpoint is not a trained temporal model, and anything
reading a checkpoint can tell the difference without inspecting weights.

### Staged fine-tuning

```
epoch 0-2 : backbone frozen, encoder frozen, decoder + temporal trainable
epoch 3+  : backbone unfrozen at lr_backbone = 1e-6
lrs       : temporal 2e-4, decoder 5e-5, encoder 1e-5, backbone 1e-6
```

The policy is re-applied every epoch and the optimizer rebuilt when the trainable
set changes, so parameter groups can never disagree with `requires_grad`. Frozen
parameters are not handed to the optimizer at all. The actual trainable set is
logged per epoch:

```
[epoch 0] trainable: backbone=frozen, encoder=frozen, decoder=train, temporal=train
          groups: temporal@2.00e-04, decoder@5.00e-05, other@5.00e-05
```

---

## 9. Event alignment and the climate-change application

Supervised day-by-day losses are only valid if predictors and targets describe the
same weather. **Measured** for SA (domain-mean deseasonalized anomalies, 1961–1970,
each target probed with its most lag-0-correlated predictor):

| lag | corr(t_850, tasmax) | corr(q_700, pr) |
|---:|---:|---:|
| −3 | 0.305 | 0.233 |
| −2 | 0.503 | 0.352 |
| −1 | 0.800 | **0.535** |
| **0** | **0.953** | 0.531 |
| +1 | 0.740 | 0.211 |
| +2 | 0.457 | 0.024 |

`tasmax` peaks sharply at lag 0 (0.953, clearing its neighbours by 0.152).

⚠️ **`pr` peaks at lag −1, by 0.004.** Lag 0 retains 99.3% of the peak. The honest
reading is that 700 hPa humidity leads daily precipitation by *less than one
sampling interval*, so lag 0 and lag −1 are tied to within sampling noise — and the
curve is clearly unimodal and centred near zero, falling to 0.21 at +1 and 0.35 at
−2. A genuinely unaligned pair (two different realizations of the same climate)
would be flat and near zero at *every* lag.

The decisive evidence for alignment is the experiment design, not the probe
correlation: this is CORDEX-ML-Bench's perfect-predictor framework, in which the
predictors are coarsened from the same ACCESS-CM2 realization as the target. The lag
test is a sanity check on that, and it passes in the sense that matters.

`cordex_temporal_diagnostics.py` therefore reports **two** verdicts and hides
neither:

* `strict_lag0_aligned` — peak exactly at lag 0 with margin ≥ 0.05. `tasmax` passes;
  `pr` does not.
* `same_day_defensible` — peak within ±1 day **and** lag 0 retaining ≥ 90% of the
  peak. Both pass. This is the field callers act on (`event_aligned`).

The strict test firing on `pr` is a property of the test, not evidence of
misalignment; the criterion was widened *after* seeing it fire on a
physically-explicable case, and that revision is recorded here rather than
silently applied. The printed lag curve lets a reader judge independently.

**The SA ACCESS-CM2 configuration is both a training case and an application
case**, and the two roles use different evaluation:

| period | files | role | evaluation |
|---|---|---|---|
| 1961–1976 | train file | training | — |
| 1977–1980 | train file | validation | loss, checkpoint selection |
| 1981–2000 | `test/historical/.../perfect` | held-out test, event-aligned | **event-paired** metrics |
| 2041–2060, 2080–2099 | `test/mid_century`, `test/end_century` | climate-change application | **distributional only** |
| `test/.../imperfect` (NorESM2-MM predictors) | — | cross-model transfer | **distributional only** |

`evaluate_predictions(event_aligned=False)` omits date-paired scores entirely and
attaches an explicit note, rather than computing an RMSE between two unrelated
realizations of the same climate and leaving the reader to notice.

⚠️ **Inherited normalization limitation.** The Phase-1 scalers were fitted over the
whole 1961–1980 + 2080–2099 record, so they saw the validation window. They cannot
be recomputed without invalidating the Phase-1 checkpoint, because
`output_scalers_mu/sigma` are stored *inside* it. The effect is second-order
(global per-channel statistics over ~14,600 days) and identical for the baseline
and both temporal backends, so the comparison between them is unaffected — but the
absolute validation numbers are mildly optimistic.

---

## 10. Stochastic refinement compatibility

All four existing refiners (`diffusion_unet`, `diffusion_transformer`,
`flow_matching_unet`, `flow_matching_transformer`) are reused unchanged. The
temporal deterministic model is a working baseline on its own; refinement is not a
prerequisite.

**Semantics.** Refinement is **causal per date**, not whole-sequence: each date is
refined conditioned on its temporal metadata and (optionally) the Phase-1 temporal
latent, with the noise state carried forward. `SequenceRefinementPlan.refine_whole_sequence`
records this explicitly; a whole-sequence refiner is not implemented and selecting
it raises rather than silently doing the causal thing.

**Conditioning and checkpoint compatibility.**

| `temporal_conditioning` | extra `cond_channels` | existing Phase-2 checkpoints |
|---|---|---|
| `none` (**shipped default**) | 0 | **valid, byte-identical path** |
| `time_features` | `F` (=5) | invalid — retrain required |
| `latent_state` | `F` + 16 | invalid — retrain required |

Widening the conditioning changes the refiner's first projection, so
`check_refiner_temporal_compatibility` raises with the exact channel counts and
names the two valid actions. The existing `cond_channels` guard in
`two_phase.py::initialize_refiner` provides the same check independently.

**Noise.** `AR1NoiseSource` uses the stationary form
`e_t = ρ·e_{t-1} + √(1−ρ²)·z_t`, so **every frame's marginal remains exactly
N(0,1)** — the refiner still sees the distribution it was trained against and only
the between-frame correlation changes. `ρ = 0` reproduces the legacy i.i.d. path
bit-for-bit and is the default. `ρ = 1` (identical noise every date) is **rejected
by the validator**: it manufactures persistence instead of modelling it. Per-member
`torch.Generator` *and* per-member AR(1) state, so a member's trajectory is
identical whether drawn alone or inside any batch — the same contract as the
existing `ChunkNoiseSource`.

**Residual semantics preserved.** Residual-space normalization, the signed-log
precipitation compression, and reconstruction are untouched. Precipitation
non-negativity is enforced at the *final field*, not by forcing residuals positive.

⚠️ Temporal modelling does **not** automatically fix existing spatial or refinement
errors. Boundary/interior RMSE and chunk-seam ratios are reported separately so a
change that improves the interior while degrading the rim, or that introduces a
temporal seam, is visible rather than averaged away.

---

## 11. Files

| File | Role |
|---|---|
| `granitewxc/temporal/calendar.py` | calendar-aware time features, gap/duplicate detection |
| `granitewxc/temporal/config.py` | validated `temporal:` schema; rejects unknown keys |
| `granitewxc/temporal/backends.py` | ConvGRU / ConvLSTM / temporal-Mamba (SSD) |
| `granitewxc/temporal/model.py` | adapter, sequence runner, freeze policy, param groups |
| `granitewxc/temporal/sequence_dataset.py` | split → runs → windows, crop consistency |
| `granitewxc/temporal/sources.py` | frame-source adapters over the existing datasets |
| `granitewxc/temporal/losses.py` | temporal objectives |
| `granitewxc/temporal/metrics.py` | event-paired and distributional evaluation |
| `granitewxc/temporal/checkpoint.py` | migration, validation, provenance |
| `granitewxc/temporal/training.py` | training loop, staged unfreezing |
| `granitewxc/temporal/inference.py` | chunked inference, NetCDF output |
| `granitewxc/temporal/refinement.py` | AR(1) noise, temporal conditioning, compat checks |
| `granitewxc/temporal/entrypoints.py` | shared CLI (`describe`/`check`/`train`/`infer`/`evaluate`) |
| `granitewxc/models/cordex_finetune_model.py` | **+45 lines, 0 deletions**: the bottleneck hook and its guard |

The only change to existing code is the hook in `cordex_finetune_model.py`, which
returns its input object unchanged when no adapter is attached.
