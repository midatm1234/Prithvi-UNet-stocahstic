# Prithvi-native paired state: architecture, provenance, and what is actually being tested

> **2026-09-15 provenance correction:** Historical Phase-1 foundation
> provenance remains unresolved. Current ECCC/Phase-1 shape incompatibility does
> not prove random historical initialization, and weight histograms do not prove
> untrained or useless time embeddings. The general/rollout cadence difference
> is a transfer limitation, not a proof of impossibility. The original statements
> below are retained as the handoff record and superseded on these points by
> [the takeover evidence report](temporal_pretrained_transfer_takeover.md).


Branch `Prithvi-UNet_temporal_model`. Companion to the three existing temporal
documents, which are **unchanged**: `temporal_model_research.md`,
`temporal_model_architecture.md`, and `temporal_model_results.md` (whose
`NOT ACCEPTED` verdicts for ConvGRU and Mamba stand and are not revised here).

Environment: `Prithvi` mamba env at
`C:\Users\huikyole\AppData\Local\miniforge3\envs\Prithvi`, Python 3.12.13,
torch 2.11.0+cu128, one NVIDIA RTX PRO 6000 Blackwell Max-Q (97 GiB).

> **Read this first — what changed, and what did not.**
>
> The previous experiment added a randomly initialized recurrent/SSM module
> *after* the transformer and failed the pre-registered criteria. This work tests
> a different hypothesis: that letting two temporally separated atmospheric states
> interact **inside** the shared transformer — using the mechanism the
> architecture already has for that — is more useful than a separate memory
> module bolted onto the end.
>
> **A finding that reframes the brief.** This pipeline contains **no Prithvi-WxC
> foundation weights at all**. The configs pair `model.embed_dim: 1024` with
> `path_model_weights: .../ECCC/weights/best_rmse_UNET.pt`, which is
> `embed_dim 2560`. All 170 backbone tensors shape-mismatch and are silently
> dropped by the loader's shape filter. The transformer in the Phase-1 checkpoint
> was trained **from random initialization** on the downscaling task. §2 gives the
> measurement. Consequently this is a test of the *architectural* claim, not of
> foundation-model transfer, and nothing below claims otherwise.
>
> Engineering verification on real data with the real checkpoint **passes**: 208
> tensors load, exactly one is explicitly adapted, legacy parity is bit-exact,
> causality is structural, chunked inference is bit-exact, and every temporal
> parameter trains except one whose deadness is structural and declared (§5).
>
> Scientific acceptance is decided by the **unchanged** pre-registered scorecard,
> plus one additional control. See `temporal_native_pair_results.md`.

---

## 1. The mechanism, and why it is the native one

Upstream `PrithviWxC.forward` takes `x` of shape
`[batch, time, parameter, lat, lon]` and, before anything learned happens, folds
the time axis into convolution channels:

```python
# PrithviWxC/model.py:1364-1369
x_rescaled = x_rescaled.flatten(1, 2)   # [B, time x parameter, lat, lon]
x_rescaled = self.parameter_dropout(x_rescaled)
x_embedded = self.patch_embedding(x_rescaled)
```

`input_size_time` — 2 in every released checkpoint — has exactly **one**
functional use in the whole upstream model:

```python
# PrithviWxC/model.py:1015-1019
self.patch_embedding = PatchEmbed(
    patch_size=patch_size_px,
    channels=in_channels * input_size_time,
    embed_dim=embed_dim,
)
```

There is no temporal attention, no temporal position embedding and no recurrence
anywhere in `PrithviWxC`. So "the paired-time input structure" *is* channel
stacking, and the two input states first meet each other inside the transformer's
self-attention. All additional temporal information is two scalars,
`input_time` and `lead_time`, consumed once as a global additive bias:

```python
# PrithviWxC/model.py:1386-1388
time_encoding = self.time_encoding(batch['input_time'], batch['lead_time'])
tokens = x_embedded + static_embedded + time_encoding
```

**This repository's downstream model already reproduces the channel-stacking
mechanism verbatim:**

```python
# granitewxc/models/cordex_finetune_model.py:971
x_sep_time = batch['x'].view(B, self.n_input_timestamps, -1, H, W)
```

It was simply never used. Every shipped config sets `n_input_timestamps: 1`, and
`docs/temporal_model_architecture.md` §2 records the previous decision to leave it
at 1 and never "fake history by duplicating a snapshot into this axis" — correct,
because at the time nothing put a *real* earlier date there. This work puts a real
earlier date there.

### 1.1 Where each pathway enters

```
x -> normalize -> PatchEmbed -> +static -> U-Net encoder (skips)
                                        -> conv_before_backbone
                                        -> tokens
                                        -> [ NATIVE-PAIR HOOK ]        <-- new
                                        -> PrithviWxCEncoderDecoder
                                        -> conv_after_backbone
                                        -> [ BOTTLENECK HOOK ]         <-- existing
                                        -> U-Net decoder -> heads
```

The two hooks are on opposite sides of the transformer, and that is the entire
point. The bottleneck hook can only mix latents that have *already* been through
the transformer separately; the token hook lets two dates enter it together.

`backend: native_pair` uses the token hook and leaves the bottleneck hook an exact
identity (`injects_at_bottleneck = False`). The `recurrent` and `mamba` backends
have no `apply_tokens` method, so the token hook returns its input object
unchanged for them — the existing results stay reproducible.

### 1.2 Shapes (SA case)

| | frame-independent / recurrent | `native_pair` |
|---|---|---|
| `data.n_input_timestamps` | 1 | **2** |
| `x` per emitted frame | `[B, 15, 128, 128]` | `[B, 30, 128, 128]` |
| `embedding.proj.weight` | `(512, 15, 2, 2)` | `(512, 30, 2, 2)` |
| `input_scalers_mu` | `(1, 15, 1, 1)` | `(1, 15, 1, 1)` — unchanged |
| tokens into the transformer | `[B, 64, 256, 1024]` | identical |
| transformer cost | — | **identical** |

Channel order is **oldest first**, matching the native flatten order: block 0 is
`t-1`, block 1 is `t`.

The input scalers deliberately stay at 15 channels. They are per-parameter
statistics that the model broadcasts *over* the time axis
(`input_mu.unsqueeze(1)` against `[B, T, P, H, W]`), because it is the same
variable observed on a different date.

### 1.3 Cost

Because the time axis becomes channels rather than tokens, the transformer's
sequence length does not change. The only added compute is the patch-embedding
convolution's wider input.

| | new parameters | share of 246,408,968 |
|---|---|---|
| ConvGRU adapter | 7,642,752 | 3.10% |
| Mamba adapter | 1,516,576 | 0.62% |
| **`native_pair`** | **32,768** | **0.0133%** |

That is 233× fewer new parameters than ConvGRU. Of the 32,768: 30,720 is the
widened patch embedding, 2,048 is the time-conditioning module (5 tensors —
`input_time_embedding.{weight,bias}`, `lead_time_embedding.{weight,bias}`, `gate`).

Backbone evaluations per training window (SA geometry, 7 frames, 5 supervised):

| variant | backbone evals / step | why |
|---|---|---|
| `convgru`, `mamba`, `spatial_ft`, `time_only` | 7 | every frame must run to advance hidden state |
| `native_pair`, `native_pair_nohistory` | **5** | stateless: warm-up frames have no effect, so they are not run |
| `native_pair_pretext` | **7** | 5 + one masked-reconstruction + one transition pass |

Measured, not assumed: `train.backbone_evaluations_per_step` is recorded in every
training log. Equal optimizer steps do **not** imply equal compute, and the
paired pathway is *cheaper*, not more expensive, than the backends it is compared
against.

### 1.4 How much history reaches each output

**Exactly one earlier date.** With `history_offsets: [1]`, the prediction for date
`t` is a function of the coarse predictors at `t-1` and `t`, the static field, and
the time scalars. Nothing else.

Explicitly, this is **not**:

* an unbounded recurrent memory — there is no state, and `final_state is None`;
* an autoregressive rollout — no predicted output is ever an input; a sequence of
  paired predictions is a sequence of independent conditional predictions;
* a forecast — `mode: downscaling` forces `lead_time_days == 0`, and the config
  layer raises for `native_pair` in any other mode.

Under `pretext.transition`, date `t-2` influences the *weights* through the
auxiliary loss, but never the prediction for `t`.

---

## 2. Pretrained-component and checkpoint-compatibility report

### 2.1 There are no foundation weights in this pipeline [MEASURED]

The downstream model instantiates `PrithviWxCEncoderDecoder`, a bare transformer
stack, not `PrithviWxC`:

```python
# granitewxc/models/model.py:246-253
backbone = PrithviWxCEncoderDecoder(
    embed_dim=config.model.embed_dim,      # 1024 for SA
    n_blocks=config.model.n_blocks_encoder,  # 8 -> 2*8+1 = 17 transformers
    ...
)
```

Weight loading is `examples/CORDEX_ML/cordex_training.py:1131`, a whole-model
loader with a hand-rolled shape filter:

```python
target = model_state.get(key)
if target is None or target.shape != value.shape:
    skipped += 1
    continue
```

Comparing the checkpoint the configs point at against the model they build:

```
ECCC      best_rmse_UNET.pt : 208 tensors, 170 backbone, 17 blocks, qkv (7680, 2560)
SA phase1 best.ckpt         : 208 tensors, 170 backbone, 17 blocks, qkv (3072, 1024)

common keys 199 | same-shape 15 | SHAPE-MISMATCHED 184  (170 of them backbone)
same-shape keys are all downsampling_layers.* — not one transformer weight
```

`strict_matching: false` in every YAML is a dead key read by no code, so nothing
surfaced this. **The SA Phase-1 transformer was trained from random
initialization.** `freeze: backbone: true` in the temporal configs therefore froze
a randomly-initialized, downscaling-trained transformer — not a foundation model.

This is reported, not fixed. Fixing it would mean building an `embed_dim 2560`
model, which the Phase-1 checkpoint cannot initialize, so it would need its own
baseline and its own budget and would not be a matched comparison.

### 2.2 Foundation checkpoints on this machine

| checkpoint | status | path | notes |
|---|---|---|---|
| `prithvi.wxc.rollout.2300m.v1` | **present** | `D:\Prithvi-WxC\data\{weights\large_rollout,large_rollout}\` | 26.5 GiB ×2, 320 tensors, 2,413,912,182 params, `embed_dim 2560`, 12 encoder + 2 decoder blocks |
| `prithvi.wxc.2300m.v1` (general, masked reconstruction) | **absent** | — | searched all of `D:\` and `C:\Users\huikyole` |
| `granite-geospatial-wxc-downscaling` ECCC | present | `granite-geospatial-wxc-downscaling\ECCC\weights\best_rmse_UNET.pt` | 16.2 GiB, 208 tensors, `embed_dim 2560` |

**The general checkpoint is the right one for this task and it is the missing
one.** Pretraining drew `lead_time` from `[0, 6, 12, 24]` h, so zero lead — our
case — is inside its trained support. The rollout checkpoint *fixed* both the
input delta and the lead time to 6 h, so it has never seen `lead_time = 0`. The
rollout checkpoint is not a substitute, and it is not used.

No YAML in this repository references any `prithvi.wxc.*` checkpoint.

### 2.3 Native time-embedding weights are not transferable [MEASURED]

Two independent reasons, either sufficient:

1. **Shape.** The checkpoint stores `input_time_embedding.weight` as `(640, 1)`
   (`embed_dim // 4` for `embed_dim 2560`). This backbone needs `(256, 1)`.
2. **Content.** Measured over the checkpoint's 640 values: range
   `[-0.951, +0.946]`, `mean|w| 0.467`, flat 8-bin histogram — statistically
   indistinguishable from a fresh `nn.Linear(1, 640)` default `U(-1, 1)`. A
   freshly initialized layer in the same environment gave `[-0.998, +0.9997]`
   with a comparably flat histogram. There is no learned time structure to
   transfer; the "pretrained temporal machinery" is effectively a frozen random
   Fourier feature map of two scalars.

Additionally, at a 24-hour interval 72% of the 640 units alias with period
< 24 h and the median implied period is ~13 h, so the code is a hash rather than
a smooth metric: it *distinguishes* 24 h but supports no interpolation to a new
cadence.

`NativeTimeConditioning` therefore reproduces the native **form** and **injection
point** with newly initialized values, and says so.

### 2.4 The one adapted tensor

`embedding.proj.weight` is the only tensor whose shape depends on
`n_input_timestamps`. Migration widens it explicitly and reports it under a
separate `adapted` list, never merged into `loaded`:

```
[checkpoint] spatial_init from .../phase1_unet/best.ckpt
  loaded            : 208 tensor(s)
  adapted (reshaped): 1 -> ['embedding.proj.weight']
  missing (temporal): 5 e.g. ['temporal_adapter.time_conditioning.gate', ...]
  missing (other)   : 0
  unexpected        : 0
  shape mismatch    : 0
  provenance        : epoch=9 step=4570 trained_temporal_steps=0
```

The pretrained filter goes on the **last** channel block (date `t`) and every
history block starts at **exactly zero** (measured: history-half `absmax = 0.0`).
Two properties follow, and the distinction between them matters:

* **Baseline-preserving.** At initialization the convolution computes
  `0 * x_history + W_pretrained * x_t`, so the paired model departs from the
  established spatial skill rather than from a perturbation of it.
* **Immediately trainable.** A zero *weight* is not a zero *gradient*:
  `dL/dW_history = dL/dout * x_history`, non-zero from step 1. Verified on real
  data (`history_half_receives_gradient: true`). The zero-init trap that applies
  to a multiplicative gate does not apply to an input weight.

**Precision of the parity claim.** With the history weights zeroed and the same
30-channel input shape, parity with the frame-independent prediction is
**bit-exact** (`max_abs_diff: 0.0`, real data, real checkpoint). Comparing the
*wider* convolution against the original *15-channel* convolution is exact only in
float64; in float32 it differs by `3e-7` relative, because widening a
convolution's input channel count changes the order in which the products are
accumulated. Both statements are true and neither is rounded up: the pathway does
not claim the unconditional bit-exactness that `adapter_init_gate: 0` has.

### 2.5 A latent defect this work had to fix

`n_input_timestamps > 1` was unreachable before this branch. `get_scalers`
multiplied the dynamic channel count by `n_input_timestamps`, which broke two
things at once for any value above 1:

```
n_input_timestamps = 2 (before the fix)
  input_mu            -> (16,)   # wrongly absorbed the static channel
  input_static_sigma  -> 1.0     # real orography sigma is 576.938
  forward             -> RuntimeError: size of tensor a (15) must match b (16)
```

The silent half is the dangerous half: the orography normalization was destroyed
and only the shape error made it visible. The fix makes the input scalers
per-parameter, which is what the forward pass already assumed, and is provably a
no-op for every configuration in the repository — all of which set
`n_input_timestamps: 1`.

---

## 3. Pretraining-aligned auxiliary objectives (separate ablation)

Opt-in, independently switchable, and **off** in
`..._prithvi_native_pair.yaml` so the architectural change can be measured
alone. Both are enabled only in `..._prithvi_native_pair_pretext.yaml`.

Both train the **shared** encoder, transformer and post-backbone convolution;
only the small output heads are private. `test_pretext_losses_reach_the_shared_trunk`
asserts non-zero gradient at `embedding.proj.weight`,
`conv_before_backbone.weight`, `conv_after_backbone.weight` and inside
`backbone.*`. An auxiliary head trained in isolation would be no evidence of an
improved representation, so this is checked rather than assumed.

The heads do **not** participate in the downscaling forward pass at all, so the
*prediction* model of `native_pair_pretext` is identical to `native_pair`; the
heads only shape the shared weights through their gradients.

### 3.1 Masked atmospheric reconstruction

Whole mask-units (16×16 px, matching `mask_unit_size`) of the raw normalized
predictor field are replaced by the per-channel input mean — the value that
normalizes to exactly zero, i.e. what a dropped token contributes upstream — and
the masked cells are reconstructed from the shared representation.

**Masking is applied to every input timestamp simultaneously.** This is what makes
the task honest, and it closes two leakage paths that masking date `t` alone would
leave wide open:

1. the history channels hold the same variables one day earlier, and daily fields
   are strongly autocorrelated, so a near-copy of the answer would remain in the
   input;
2. the U-Net skip pyramid is built from `self.embedding(x)` on this same tensor,
   so a mask applied only to the transformer's input would leave the masked region
   fully visible to the decoder through the skips.

Upstream has the identical property for the identical reason: it drops whole mask
units of *tokens*, and a token already contains both timestamps because the time
axis was folded into channels before embedding.

Two tests pin this down: `test_mask_is_applied_to_every_timestamp`, and
`test_masked_content_cannot_influence_the_masked_forward`, which perturbs the
truth inside the masked region by 1000 and requires the model's input to be
bit-identical.

The loss is scored on masked pixels **only**. Including visible pixels would let
the term be driven down by copying the input.

⚠️ **This is not native masked-autoencoder training, and is not described as
such.** Upstream drops masked tokens from the encoder's sequence entirely and
restores them with a learned `mask_token` before a decoder. `PrithviWxCEncoderDecoder`
has no masking capability, no `mask_token` and no reconstruction head — and
`mask_ratio_inputs` / `mask_ratio_targets` in the YAMLs are dead keys read by no
model. What is implemented is masked *field* reconstruction through the shared
trunk: the same objective in spirit, at the same granularity, but with the tokens
kept and the masked content removed from the input rather than from the sequence.

### 3.2 Atmospheric transition prediction

Predict the atmospheric state at `t` from strictly earlier states
(`x[t-2], x[t-1]` with `lead_steps: 1`), through the shared representation, with
`lead_time = +24 h`. The main downscaling pass legitimately still uses `x[t]`.

The verification state is never an input to this pass. Guaranteed structurally —
the pair is built ending at `t - lead_steps`, so every input date is strictly
earlier than the target — and checked two ways:
`test_transition_pass_does_not_see_its_target` perturbs `x[t]` by 1000 and
requires the transition pass's *input tensor* to be bit-identical, and
`test_transition_pretext_prediction_is_independent_of_the_target` requires the
loss to move only through its target.

The target is a **physical predictor field** (in normalized units, so the 15
channels are commensurate), not a latent, so there is no representation collapse
and no identity shortcut to guard against.

⚠️ This is a **regional, reduced-variable** transition model over 15 predictor
channels on a 128×128 grid. It is not native full-state MERRA-2 rollout, and no
high-resolution precipitation or temperature output is ever fed into an
atmospheric-state input interface.

### 3.3 Cost control

Each objective costs one extra forward pass, applied to the **last emitted frame
of each window** rather than to all five. That keeps the pretext variant at 7
backbone evaluations per window — the same as ConvGRU and Mamba — so the
comparison is not bought with extra compute.

---

## 4. Data contract [MEASURED, and one inherited hazard]

Verified against the files, not the config:

| property | training/validation file | held-out test file |
|---|---|---|
| predictors | 15 × `(14600, 128, 128)`, **`cell_methods = 'time: mean'` on all 15** | 15 × `(7300, 16, 16)`, same `cell_methods` |
| `pr` | `units = 'mm/day'`, no `cell_methods` | **`units = 'kg m-2 s-1'`**, `cell_methods = 'time: mean'` |
| `tasmax` | `cell_methods = 'time: maximum (interval: 1 day time: mean'` | identical |
| calendar | `standard`, steps `{1.0: 14588, 2.0: 10, 36160.0: 1}` | `standard`, steps `{1.0: 7294, 2.0: 5}` |
| stamps | every record 12:00:00 | every record 12:00:00 |

Consequences that shaped this pathway:

* **The predictors are daily means, not instantaneous states.** Pretraining used
  3-hourly instantaneous MERRA-2 states with `input_time` in
  `[-3, -6, -9, -12]` h. A 24-hour interval on daily means is outside that support
  in two independent ways, and both are recorded rather than relabelled to fit the
  checkpoint. `input_time` is passed as `+24 h` because 24 h is the truth.
* **No sub-daily structure is invented.** Every stamp is 12:00:00, so
  `include_hour_of_day: auto` correctly omits hour-of-day; requesting it
  explicitly raises.
* **The interval is measured, not assumed.** `pair_time_scalars` sums
  `interval_ratio` over the spanned steps, so a step across a removed 29 February
  is charged as the two days it really is. Windows are cut at every discontinuity,
  so within a window the sum is exactly the cadence, but the arithmetic does not
  rely on that.

The four separated time quantities:

| quantity | value here |
|---|---|
| input-history interval | 24 h (`history_offsets: [1]` × `cadence_days: 1.0`) |
| main downscaling lead | **0** — enforced; `native_pair` raises outside `mode: downscaling` |
| auxiliary forecast lead | 0 (off) or +24 h (`pretext.transition.lead_steps: 1`) |
| calendar phase | `season_sin/cos`; no sub-daily timing exists to use |

### 4.1 An inherited data hazard found while verifying the contract

The training target `pr` is `mm/day` while the held-out test target `pr` is
`kg m-2 s-1` — a factor of 86400. The scalers, the `wet_threshold: 1.0` and every
occurrence/spell diagnostic are stated in mm/day. `cordex_dataset.py` rejects
unknown precipitation units rather than assuming, so this is converted at load
rather than silently mis-scaled; it is recorded here because it is **not** in the
existing documents' list of known limitations and it affects every variant's
absolute `pr` numbers equally (so the comparison is unaffected, but the absolute
values should not be read as calibrated).

### 4.2 Splits, and no leakage

Splits are unchanged from the recurrent config and are applied **before** windowing,
so a window cannot straddle a boundary. `train 1961-1976 / validation 1977-1980 /
test 1981-2000`, no overlap. Every emitted date, warm-up exclusion and evaluated
date is identical to the archived experiment: 600 training windows, 60 validation
windows, test period `1981-01-01 … 1983-12-31`, 1095 dates.

Pretraining-year overlap is a separate question from downstream leakage and is
recorded separately: MERRA-2 pretraining covers 1980-2019, which overlaps the
evaluated 1981-1983. It is **moot here** because no foundation weights are in the
pipeline (§2.1). It would need addressing before any genuine foundation-transfer
run.

The inherited scaler limitation is unchanged: Phase-1 scalers were fitted over the
whole 1961-1980 + 2080-2099 record and saw the validation window. Identical across
all variants, so the comparison is unaffected; absolute validation numbers are
mildly optimistic.

---

## 5. Engineering verification on real data

```bash
mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_training.py check \
    --config examples/CORDEX_ML/SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_prithvi_native_pair.yaml \
    --split validation --chunk-frames 12
```

| check | result |
|---|---|
| spatial tensors loaded | **208 / 208** |
| explicitly adapted | **1** (`embedding.proj.weight`) |
| missing outside `temporal_adapter.*` | **0** |
| unexpected keys / shape mismatches | 0 / 0 |
| missing `temporal_adapter.*` (expected) | 5 |
| temporal parameters | 2,048 (+30,720 in the widened embedding) |
| **legacy parity, contribution zeroed: bitwise identical** | **True, max_abs_diff 0.0** |
| near-identity at `time_conditioning_gate_init: 1e-3` | mean rel. deviation **1.43e-05** |
| temporal params with zero gradient | 1 / 5, **all declared** (`unexpected_dead: []`) |
| history half of the embedding receives gradient | **True** |
| causality: earlier frames bitwise unchanged | **True** (and the last frame does change) |
| history sensitivity (contribution amplified) | 100.74 on a field of scale 151.83 |
| **chunked ≡ single pass** | **exact, max_abs_diff 0.0** (chunk lengths 12 and 4) |

Full JSON: `docs/temporal_experiment_records/temporal_check_SA_native_pair.json`.

### 5.1 One parameter is structurally untrainable, and it is declared

For the downscaling task `lead_time` is identically `0`, so

```
dL/d(lead_time_embedding.weight) = dL/d(lt) * lead_time = 0
```

That weight matrix cannot train; only its bias adapts. This is a property of a
zero-lead task, not a defect, and it is **declared** by
`NativePairAdapter.structurally_dead_parameters()` so the check asserts *exactly
the declared set is dead* — which catches both a new dead parameter and a stale
exemption. The weaker "nothing is dead" assertion would have had to be relaxed
into meaninglessness.

Enabling `pretext.transition` supplies a positive lead and revives it;
`test_the_transition_objective_revives_the_lead_time_weight` asserts that. This is
one concrete way the pretraining-aligned objective exercises more of the native
mechanism than the downscaling task alone can.

### 5.2 Chunked inference needed a real change

A stateless finite-history model has nothing to "carry across" a chunk boundary:
the earlier **dates themselves** must be in the batch. `run_sequence_inference`
therefore extends each chunk backwards by the deepest history offset and does not
re-emit those frames — the direct analogue of carrying hidden state.

`build_pair_input` distinguishes the two genuinely different failure modes, which
a clamped index would have merged into one silent wrong answer:

* absolute index `< 0` — the date does not exist because `t` is at the start of a
  run. This is the **cold start**: the history slot repeats date `t`, the honest
  "no history available" input, and the analogue of the recurrent backends' zero
  initial state. Permitted **only** at inference, and it keeps the emitted date set
  identical to the baseline, which the matched comparison requires.
* absolute index `>= 0` but the frame is not in the batch — the date exists and was
  not supplied. That is a chunking bug and raises `MissingHistoryError`.

During training the default forbids cold starts, so a too-short window raises
instead of quietly training a supervised frame on fabricated context.

### 5.3 Tests

```
tests/test_temporal_native_pair.py .................................. 47 passed
tests/test_temporal_{model,configs,calendar_and_losses,refinement}.py 126 passed
tests/ (whole repository)                                    980 passed, 2 failed
```

Both failures are unrelated to this work:

* `test_prism_grid.py::test_concurrent_first_writers_publish_one_complete_contract`
  — documented as pre-existing in `temporal_model_results.md` §3 (two concurrent
  `os.replace` calls; POSIX permits it, Windows raises `WinError 32`).
* `test_prism_preprocess_launchers.py::test_failed_training_shard_aborts_before_scalar_recomputation`
  — **passes in isolation** (`2 passed in 3.11s`); it fails only under full-suite
  timing and exercises preproc subprocess launchers, which this branch does not
  touch.

---

## 6. Focused literature-to-implementation table

Only decisions that changed code. Sources verified by reading the primary
artifact, not a summary.

| Source | What it establishes | What was implemented, and where |
|---|---|---|
| [R2] `PrithviWxC/model.py:1364-1369, 1015-1019` | The paired-time input structure **is** channel stacking; `input_size_time` only sets the patch-embed channel count; no temporal attention or recurrence exists | `data.n_input_timestamps: 2` with a real `t-1`; `build_pair_input` stacks oldest-first to match the native flatten order |
| [R2] `PrithviWxC/model.py:1233-1253, 1386-1388` | Time conditioning is two `Linear(1, D//4)` maps combined as `cat(cos,cos,sin,sin)`, added **once** to the token stream before the encoder; no FiLM/adaLN anywhere | `NativeTimeConditioning` reproduces the form exactly; injected at the new token hook, which is the same position |
| [R2] `PrithviWxC/dataloaders/merra2.py:288, 328-357` | `input_time` is the gap between the two input states, in **hours**, and the tensor handed to the model is **positive** (opposite in sign to the constructor argument) | `pair_time_scalars` returns positive hours, measured from `interval_ratio` |
| [R3] model card | Pretraining drew `input_time` from `[-3,-6,-9,-12]` h and `lead_time` from `[0,6,12,24]` h; the rollout checkpoint **fixed both to 6 h**; 50% masking | The **general** checkpoint is the correct one for zero-lead downscaling (0 is in its support) and the rollout one is not a substitute (§2.2); `mask_ratio: 0.5` matches pretraining |
| [R1] abstract | "trained with a mixed objective that combines the paradigms of masked reconstruction with forecasting"; downscaling is a named downstream task | Both halves implemented as the two separable pretext objectives (§3), rather than only one |
| [R2] `PrithviWxC/model.py:1390-1428` | Masking drops whole **global mask units** of tokens, a fixed count per sample; masked tokens are restored with `mask_token + static_embedded` | `sample_mask_units` draws a fixed count; masking is at mask-unit granularity, applied to every timestamp because a token contains both (§3.1) |
| [R2] `PrithviWxC/model.py:947-948` | `mask_ratio_targets > 0` raises `NotImplementedError` upstream | Target masking is not implemented here either; only inputs are masked |
| [R5] abstract | "autoregressive rollout training produces substantially more accurate forecasts than direct conditioning on forecast lead time" | A caution about lead-time conditioning **for forecasting**. Our main lead is 0, so it does not apply to the downscaling task; it is a reason not to over-invest in the auxiliary lead, which is why `transition_weight` is 0.05 and untuned |
| [R4] granite-wxc | Downscaling weights are released separately from the foundation model, and the repo does not state the backbone's provenance | Motivated checking it directly, which is how §2.1 was found |

Deliberately **not** cited as support: the MetMamba / WSSM / MambaRain line of work
that motivated the previous backends. It is not evidence for or against this
pathway, and `temporal_model_research.md` already covers it.

---

## 7. Limitations

**Scope**

1. **This is not a test of foundation-model transfer.** No Prithvi-WxC foundation
   weights are in the pipeline (§2.1). The hypothesis under test is architectural:
   two dates interacting inside a shared representation versus a separate memory
   module after it. The representation they share happens to be
   downscaling-trained rather than pretrained.
2. **The general masked-reconstruction 2300M checkpoint is absent** (§2.2), so a
   genuine transfer experiment could not be run even at a different budget without
   a ~26 GB download and an `embed_dim 2560` model that the Phase-1 checkpoint
   cannot initialize.
3. **Native time-embedding weights are not inherited** (§2.3) — form and injection
   point are native, values are new.
4. **24 h is outside the pretraining cadence** and the predictors are daily means,
   not instantaneous states (§4).
5. **One earlier date.** `context_length: 14` and deeper `history_offsets` are
   untested by design: testing them first would confound "does native pairing
   help" with "does more history help". `history_offsets` accepts a list, so the
   extension is a config change, and it stays a finite-history conditional model
   rather than reintroducing a generic memory module.
6. **Bounded, not converged.** 600 optimizer steps, batch 1, one epoch, backbone
   frozen throughout (the schedule unfreezes at epoch 3, which one epoch never
   reaches). Whether longer training helps is **untested**, here as before.
7. **Loss weights untuned.** The per-frame/tendency/accumulation/occurrence block
   is byte-identical to the recurrent config so the objective is not confounded
   with the architecture; the two pretext weights were fixed a priori at 0.05 and
   never tuned, so the acceptance test is not fitted to the test set.
8. **Not native MAE training** (§3.1), and the transition task is regional and
   reduced-variable, not full-state rollout (§3.2).
9. **NARR/PRISM is not executed.** Configs are schema-complete and validated by the
   config layer and unit tests (including `num_static_channels: 0` and the three
   `ppt`/`tmax`/`tmin` targets), but the NARR and PRISM archives, the preprocessed
   products, the scalers and the `narr_prism_California` Phase-1 checkpoint are all
   absent — `examples/NARR_PRISM/{preprocessed,experiments,scalars_with_H}` are git
   symlinks to non-existent `/data/...` Linux paths. Data availability, not an
   implementation gap. The pretext variant there also needs `context_length: 6`
   and `warmup_length: 2` for the auxiliary lead, which shifts its emitted frames
   by one relative to the other NARR variants, so it would need its own matched
   controls — unlike the SA case, whose `warmup_length: 2` already suffices with no
   geometry change at all.
10. **Stochastic refinement untouched.** All four refiners and their checkpoints are
    unaffected: `temporal.refinement.temporal_conditioning: none` keeps the
    per-date Phase-2 path byte-identical. No diffusion or flow-matching head was
    retrained, and no four-head factorial was attempted.

**Explicitly not claimed**

* That the paired pathway will pass the pre-registered criteria. It is reported
  exactly as scored.
* That reusing a pretrained *mechanism* implies reusing pretrained *knowledge* —
  §2.1–2.3 are the reasons that inference would be wrong here.
* That sensitivity to history is skill. A prediction changing under perturbation
  establishes that the pathway is wired, which is why §5's history-sensitivity
  number is labelled as plumbing and the scorecard is what decides anything.
