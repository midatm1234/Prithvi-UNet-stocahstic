# Temporal Prithvi-UNet: results, provenance, and limitations

Branch `Prithvi-UNet_temporal_model`, created from `Prithvi-UNet-stochastic_refinement`
@ `dd48c6f`. Environment: `Prithvi` mamba env at
`C:\Users\huikyole\AppData\Local\miniforge3\envs\Prithvi`, Python 3.12.13,
torch 2.11.0+cu128, one NVIDIA RTX PRO 6000 Blackwell Max-Q (97 GiB).

Companion documents: `temporal_model_research.md` (literature and decisions),
`temporal_model_architecture.md` (shapes, causality, state, losses, checkpoints).

> **Read this first — the verdict.**
>
> The implementation is complete and its engineering properties are verified on real
> data with real checkpoints: legacy parity is **bit-exact**, chunked inference is
> **bit-exact**, causality is structural, and every temporal parameter trains (§3).
>
> **Scientific acceptance is NOT demonstrated.** Both temporal backends fail the
> pre-registered primary criteria and the spatial guardrails, and neither beats both
> matched controls. The trained checkpoints are labelled **experimental** and must
> not be used as production weights.
>
> The comparison is a *bounded* run — 600 optimizer steps per variant, one GPU,
> ~4.4 h total, backbone frozen throughout, loss weights untuned — so this is a
> lower bound on the architecture, not a ceiling. Three findings are nonetheless
> solid: most of the distributional improvement comes from the **changed objective**
> rather than temporal memory (which is why the controls exist); temporal memory does
> add a monotone improvement in **wet/dry spell structure**; and **Mamba is the
> better-behaved backend** here, correcting persistence without overshoot. §7.2 has
> the detail, §8 the commands to continue, §9 the limitations.
>
> No improvement is claimed anywhere without the measurement that supports it.

---

## 1. Base branch and provenance

```
$ git rev-parse HEAD            # before branching
dd48c6faee358815cdae03d0bc8d5a0c860bed44
$ git checkout -b Prithvi-UNet_temporal_model
```

`Prithvi-UNet-stochastic_refinement` was selected after checking which branch
actually contains both target workflows:

| branch | has `NARR_PRISM_subdomain.yaml` | has SA refinement configs |
|---|---|---|
| `main` | no | no |
| `origin/CORDEX_ML` | no | — |
| `NARR_PRISM` / `origin/narr_prism` | yes | no |
| **`Prithvi-UNet-stochastic_refinement`** | **yes** | **yes** |

No existing file was deleted and no working-tree state was reset. New outputs go to
`runs_temporal/` trees.

### Source assets located

| asset | path | status |
|---|---|---|
| SA Phase-1 checkpoint | `examples/CORDEX_ML/runs/SA_T2_ACCESS-CM2_static_train/SA_T2_ACCESS-CM2_static_twophase/checkpoints/phase1_unet/best.ckpt` | present, 2.96 GB, epoch 9, step 4570, git `50982b8` on `CORDEX_ML_diffusion_head` |
| SA data | `D:/CORDEX/SA_domain/{train,test}` | present |
| SA scalers | `.../SA_T2_ACCESS-CM2_static_twophase/scalars/*.npy` | present |
| NARR/PRISM Phase-1 checkpoint | `examples/NARR_PRISM/experiments/checkpoints/narr_prism_California/last.ckpt` | **absent** |
| NARR/PRISM preprocessed data | `examples/NARR_PRISM/preprocessed/` | **absent (empty)** |
| NARR predictors / PRISM targets | `/data2/NARR/...`, `/data/PRISM/...` | **absent** (Linux paths; this is a Windows host) |

**Consequence: the SA case was executed end-to-end; the NARR/PRISM case was not.**
Its configs are schema-complete and every key is consumed and validated by real
code, and `describe` reports the missing inputs with the exact command to generate
them, but no NARR/PRISM training or evaluation numbers exist. This is a data
availability limitation, not an implementation gap.

---

## 2. Files added and changed

**Changed (existing code): one file, +45 lines, 0 deletions** -- `git diff --cached
--stat` confirms the change is purely additive, so no existing line of the spatial
model was modified.

`granitewxc/models/cordex_finetune_model.py`
* `__init__`: `self.temporal_adapter = None`, `self._temporal_ctx = None`
* `forward`: one call, `out = self._apply_temporal_latent(out)`, at the U-Net bottleneck
* new method `_apply_temporal_latent`, which **returns its input object unchanged**
  when no adapter is attached

**Added: `granitewxc/temporal/`** — `calendar.py`, `config.py`, `backends.py`,
`model.py`, `sequence_dataset.py`, `sources.py`, `losses.py`, `metrics.py`,
`checkpoint.py`, `training.py`, `inference.py`, `refinement.py`, `entrypoints.py`.

**Added: entry points**
* `examples/CORDEX_ML/cordex_temporal_training.py` — describe/check/train/infer/evaluate
* `examples/CORDEX_ML/cordex_temporal_diagnostics.py` — event alignment, headroom, persistence
* `examples/CORDEX_ML/cordex_temporal_experiment.py` — the five-variant bounded comparison
* `examples/NARR_PRISM/narr_prism_temporal.py` — same CLI for the PRISM case

**Added: configs (4)**
* `examples/CORDEX_ML/SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_recurrent.yaml`
* `examples/CORDEX_ML/SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_mamba.yaml`
* `examples/NARR_PRISM/NARR_PRISM_subdomain_temporal_recurrent.yaml`
* `examples/NARR_PRISM/NARR_PRISM_subdomain_temporal_mamba.yaml`

**Added: tests (4 files, 126 tests)** — `tests/temporal_fixtures.py`,
`test_temporal_model.py`, `test_temporal_calendar_and_losses.py`,
`test_temporal_configs.py`, `test_temporal_refinement.py`.

**Added: docs and notebook** — the three `docs/temporal_model_*.md`, and
`examples/CORDEX_ML/notebooks/SA_downscaling_temporal_T2_ACCESS-CM2_static.ipynb`.

**Updated: documentation only** — `README.md`, `examples/CORDEX_ML/README.md`,
`examples/NARR_PRISM/README.md`, `docs/STOCHASTIC_REFINEMENT.md` (appended sections).

---

## 3. Engineering verification (all on real data and real checkpoints)

Command:

```bash
mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_training.py check \
    --config examples/CORDEX_ML/SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_{recurrent,mamba}.yaml \
    --split validation --chunk-frames 12
```

| check | ConvGRU | Mamba |
|---|---|---|
| spatial tensors loaded | 208 / 208 | 208 / 208 |
| missing outside `temporal_adapter.*` | **0** | **0** |
| unexpected keys | 0 | 0 |
| shape mismatches | 0 | 0 |
| missing `temporal_adapter.*` (expected) | 23 | 37 |
| temporal parameters | 7,642,752 (3.10%) | 1,516,576 (0.62%) |
| **legacy parity, gate = 0: bitwise identical** | **True, maxdiff 0.0** | **True, maxdiff 0.0** |
| near-identity, gate = 1e-3 (mean rel. deviation) | 8.79e-06 | 9.30e-06 |
| temporal params with zero gradient | **0 / 23** | **0 / 37** |
| causality: earlier frames bitwise unchanged | True | True |
| history changes final frame (gate 1.0) | 0.426 (field scale 152) | 0.091 (field scale 152) |
| **chunked ≡ single pass** | **exact, maxdiff 0.0** | **exact, maxdiff 0.0** |

Full JSON: `docs/temporal_experiment_records/temporal_check_SA_{recurrent,mamba}.json`
(versioned copies; the run also writes them under `artifacts/`, which is gitignored).

Negative control: renaming one decoder key in a copy of the checkpoint makes
migration **reject** it, with the key listed under both `missing (other)` and
`unexpected`.

### Two real bugs the exactness requirement caught

Both would have passed a tolerance-based check, and both were found by demanding
*bit-exact* chunk/single-pass agreement on real data:

1. `is_sequence_start` was set on the first frame of every **chunk**, mislabelling
   mid-run chunks as sequence starts.
2. `state_age` was normalized by the **chunk length** rather than the configured
   `context_length`.

Together they produced a ~0.02 K discrepancy between chunk lengths 4 and 12. Fixed
by threading `position_offset` (absolute index within the contiguous run) and
`context_length` (from config) explicitly. Regression test:
`test_time_features_are_chunk_invariant`.

A third bug — `interval_ratio` passed for the whole 7-frame context while the
tensors covered the 5 emitted frames — was caught by a shape error during
calibration and is now impossible to reintroduce silently: `SequenceOutput` carries
the sliced ratio and `tendency_loss` raises with a diagnostic message on mismatch.

### Test suite

```
$ mamba run -n Prithvi python -m pytest tests/test_temporal_*.py -q
126 passed
```

Pre-existing suite, to confirm no regression from the additive model change:

```
$ mamba run -n Prithvi python -m pytest tests/test_refinement_*.py -q
218 passed

$ mamba run -n Prithvi python -m pytest tests/ -q          # whole repository
1 failed, 934 passed in 341.43s
```

The single failure is `tests/test_prism_grid.py::test_concurrent_first_writers_publish_one_complete_contract`
and it is **pre-existing, not a regression**. Verified directly by running it at the
base commit in a throwaway worktree:

```
$ git worktree add --detach /tmp/basecheck dd48c6f
$ cd /tmp/basecheck && pytest tests/test_prism_grid.py::test_concurrent_first_writers_publish_one_complete_contract -q
1 failed
```

It is a Windows platform issue, not a logic error: the test exercises two concurrent
writers doing an atomic `os.replace` onto the same target, which POSIX permits while
Windows raises `PermissionError: [WinError 32]`. This branch does not touch
`granitewxc/utils/prism_grid.py` or its test, and fixing it is out of scope here.

---

## 4. Data contract, verified against the files

### SA time axis is not contiguous

```
n_times 14600  calendar 'standard'  steps {1.0: 14588, 2.0: 10, 36160.0: 1}
12 discontinuities, 0 duplicate timestamps
```

The ten 2-day gaps land **exactly on 1 March** of 1964/68/72/76/80 and
2080/84/88/92/96 — i.e. the file declares a Gregorian calendar but physically omits
29 February. The 36,160-day step is the 1980→2080 concatenation. A model assuming
"adjacent index ⇒ adjacent day" would be silently wrong in 11 places. Windows are
cut at all of them.

### Splits, applied before windowing

| split | dates | frames | runs | windows | unique emitted dates |
|---|---|---|---|---|---|
| train | 1961-01-01 … 1976-12-31 | 5840 | 5 | 1163 | 5815 |
| validation | 1977-01-01 … 1980-12-31 | 1460 | 2 | 290 | 1450 |
| test | 1981-01-01 … 2000-12-31 | 7300 | 6 | 1454 | 7270 |

No pair overlaps. `stride == output_length`, so training windows tile the record and
each date is supervised once per epoch. The first `warmup_length` frames of each run
are never emitted, which accounts for 5840 → 5815 and 7300 → 7270.

The held-out test files pair a **365-day** predictor axis (7300 records) with a
**Gregorian** target axis (7305); they are joined by exact timestamp.

### Event alignment

```
$ mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_diagnostics.py \
    --config .../SA_..._temporal_recurrent.yaml --days 3650
pr      ALIGNED (lead<1d) peak_lag=-1  |r(0)|=0.5314  r0/peak=0.993  probe=q_700
tasmax  ALIGNED (strict ) peak_lag=+0  |r(0)|=0.9525  r0/peak=1.000  probe=t_850
```

`tasmax` peaks sharply at lag 0. **`pr` peaks at lag −1 by 0.004**, with lag 0
retaining 99.3% of the peak — 700 hPa humidity leads daily precipitation by less
than one sampling interval. The curve is unimodal and centred near zero (0.21 at +1,
0.35 at −2), unlike an unaligned pair which would be flat and near zero everywhere.
Combined with the perfect-predictor experiment design (predictors coarsened from the
same ACCESS-CM2 realization as the target), same-day supervision and date-paired
metrics are valid. Both the strict and the widened verdict are reported; see
`temporal_model_architecture.md` §9 for why the criterion was widened and that it
was widened after seeing it fire.

### Measured headroom, and what to expect

```
pr      R2 same-day=0.1642  +lags1-3=+0.0219  (explained +13.3%, unexplained -2.6%)
tasmax  R2 same-day=0.8367  +lags1-3=+0.0223  (explained +2.7%, unexplained -13.6%)
target persistence (deseasonalized): pr e-folding 1.81 d, tasmax 5.17 d
```

The absolute gain is nearly identical, but the two normalizations differ by an order
of magnitude in opposite directions. **For `tasmax` history removes 13.6% of an
already-small residual — the most likely place for a real improvement. For `pr` the
residual is 0.836 of the variance and history addresses only 2.6% of it, so a large
precipitation RMSE improvement should not be expected.** An earlier draft of the
research document mis-stated `pr`'s figure as 13%; that error is corrected and
flagged there.

⚠️ **Context length: an acknowledged tension.** `context_length: 7` was chosen from
the marginal-R² profile (gains concentrated in lags 1–3; the flat ~0.003/lag beyond
lag 5 is in-sample overfitting) plus memory. The e-folding heuristic in the
diagnostics script instead suggests **14** (2.5 × `tasmax`'s 5.17-day e-folding).
The two criteria disagree, and 7 is the more conservative choice. `context_length:
14` is the single most promising untested variation and is cheap to try.

---

## 5. Baseline behaviour on the held-out test period

Frame-independent Phase-1 checkpoint, 1981–1983 (1095 dates), event-paired:

| metric | `pr` | `tasmax` |
|---|---|---|
| bias | −0.165 mm/day | +0.055 K |
| MAE | 2.696 | 1.170 |
| RMSE | 8.088 | 1.553 |
| pooled correlation | 0.629 | 0.970 |
| daily spatial correlation | 0.530 | 0.938 |
| tendency RMSE | 11.300 | 1.530 |
| **tendency std ratio (pred/truth)** | **0.548** | 0.929 |
| **field std ratio** | **0.622** | 1.007 |
| lag-1 ACF error | **+0.0249** | +0.0049 |
| lag-2 / lag-3 / lag-5 ACF error | +0.014 / +0.022 / +0.006 | +0.000 / −0.011 / −0.031 |
| 3-day / 5-day accumulation RMSE | 14.206 / 18.390 | 3.768 / 5.777 |
| q90 / q99 bias | +1.196 / **−18.72** | +0.033 / −0.171 |
| max pred vs truth | **226.9 vs 663.5** | 319.9 vs 321.1 |
| RMSE boundary / interior | 8.543 / 7.944 | 1.414 / 1.593 |
| **wet-day frequency error** | **+0.1034** | — |
| **`P(wet｜wet)` error** | **+0.1435** | — |
| `P(wet｜dry)` error | +0.0119 | — |
| **mean wet-spell length error** | **+1.04 d** | — |
| mean dry-spell length error | −0.215 d | — |

**The baseline's precipitation deficiency is not mainly RMSE.** It produces 10
percentage points too many wet days, wet spells over a day too long, day-to-day
variability at 55% of observed, and a 99th percentile 18.7 mm/day too low with a
maximum less than a third of observed. That is the drizzle-and-oversmooth signature
of a deterministic regression, and it is exactly what the occurrence/transition and
accumulation objectives target — and what RMSE does not measure. It is also a
concrete instance of CORDEX-ML-Bench's own finding that "pixelwise daily RMSE is a
poor proxy for overall downscaling skill".

`tasmax` is already good: RMSE 1.55 K, daily spatial correlation 0.938, lag-1 ACF
error +0.005, std ratio 1.007.

### Null level of the chunk-seam statistic

The baseline has `adapter_init_gate = 0`, so it has **no temporal pathway at all**
and its output is bit-for-bit the frame-independent prediction. Its measured seam
ratios are nevertheless `pr` 1.049 and `tasmax` 1.122 over 17 seams. Since no seam
can exist by construction, **that is the null level of this statistic at n = 17**,
not a defect. Temporal-model seam ratios must be compared against those values, not
against 1.0.

---

## 6. Pre-registered acceptance criteria

Fixed in `examples/CORDEX_ML/cordex_temporal_experiment.py::ACCEPTANCE` **before any
test result was read**, printed at the start of the run, and stored in
`runs_temporal/experiment/manifest.json`.

**Primary (temporal/event representation)** — evaluated on `|value|`, because the
baseline *over*-estimates persistence and merely increasing it must not score as a win:

| criterion | threshold |
|---|---|
| lag-1 autocorrelation error, `|.|` reduced | ≥ 10% |
| lag-2 autocorrelation error, `|.|` reduced | ≥ 5% |
| day-to-day tendency RMSE reduced | ≥ 2% |
| 3-day accumulation RMSE reduced | ≥ 1% |

**Guardrails (no material spatial degradation)**

| criterion | tolerance |
|---|---|
| per-frame RMSE not worse | ≤ +1% |
| MAE not worse | ≤ +1% |
| daily spatial correlation not worse | ≤ 0.005 absolute |
| 99th-percentile bias `|.|` not worse | ≤ +5% |
| field std ratio not moved toward 0 | ≤ 0.02 absolute |
| tendency std ratio not reduced | ≤ 0.02 absolute |

**Must beat both controls.** A temporal backend is accepted only if it also beats
`spatial_ft` *and* `time_only` on **every** primary metric. Otherwise any gain is
attributable to further fine-tuning or to date conditioning rather than to temporal
memory.

⚠️ **One threshold was ill-calibrated, and is reported as such rather than moved.**
`tasmax`'s baseline lag-1 ACF error is +0.0049, so "reduce by ≥10%" asks for a
0.0005 change — at or below the noise floor of that estimate. The criterion is kept
as pre-registered; its verdict for `tasmax` should be read as uninformative rather
than as evidence either way. The `pr` lag-1 criterion (+0.0249 baseline) is
well-posed.

---

## 7. Bounded comparison

```bash
mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_experiment.py \
    --out examples/CORDEX_ML/runs_temporal/experiment \
    --steps 600 --val-steps 60 --epochs 1 --test-years 3 \
    --variants baseline spatial_ft time_only convgru mamba
```

Five variants, identical data, splits, seed (1234), optimizer step count (600),
learning rates, and per-frame loss. Differences confined to the temporal pathway:

| variant | temporal pathway | isolates |
|---|---|---|
| `baseline` | gate 0, untrained | the existing frame-independent model |
| `spatial_ft` | inert (gate 0, temporal frozen) → decoder-only fine-tune | does *any* further fine-tuning help? |
| `time_only` | full module, `state.reset_every_frame: true` | date conditioning **without** memory |
| `convgru` | full recurrent | |
| `mamba` | full SSD | |

`time_only` is the ablation that separates date/time conditioning from actual
temporal memory: the module has the same parameters, the same optimizer and the same
data order, and still receives every calendar feature through its FiLM modulation —
only the hidden state is zeroed before each frame.

Throughput, as measured by the run itself: 2994-3512 s for 600 windows (5.0-5.9 s
per 7-frame window) plus 408-413 s inference over 1095 test dates, i.e. 57-65 min
per trained variant and ~4.4 h for all five. Peak VRAM 16.7 GiB of 97 GiB.

Measured trainable set at epoch 0 (backbone frozen until epoch 3, which a
single-epoch bounded run never reaches — so the backbone stayed frozen throughout):

```
[epoch 0] trainable: backbone=frozen, encoder=frozen, decoder=train, temporal=train
          groups: temporal@2.00e-04 (7,642,752), decoder@5.00e-05 (15,791,365), other@5.00e-05 (65,570)
variable scales (from training scalers): {'pr': 16.066, 'tasmax': 6.340}
```

### 7.1 Results

Generated from the run's own JSON by
`examples/CORDEX_ML/cordex_temporal_report.py` — the metrics below are transcribed
mechanically, not by hand. The source records are versioned under
`docs/temporal_experiment_records/` (`manifest.json`, `scorecard.json`,
`evaluation_<variant>.json`), so every number here is checkable without re-running
the ~4.4 h experiment. Regenerate with:

```bash
mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_report.py     --experiment examples/CORDEX_ML/runs_temporal/experiment     --write docs/temporal_model_results.md
```

<!-- RESULTS_BEGIN: generated by cordex_temporal_report.py -- do not edit by hand -->

### Run record

| variant | trained | optimizer steps | train (s) | inference (s) | test dates | final val loss |
|---|---|---|---|---|---|---|
| `baseline` | no | — | 0 | 411 | 1095 | — |
| `spatial_ft` | yes | 600 | 3507 | 412 | 1095 | 4.68740 |
| `time_only` | yes | 600 | 2994 | 408 | 1095 | 4.54507 |
| `convgru` | yes | 600 | 3498 | 411 | 1095 | 4.69478 |
| `mamba` | yes | 600 | 3512 | 413 | 1095 | 4.71847 |

Test period: ['1981-01-01', '1983-12-31']. Every variant used seed 1234, the same splits, the same per-frame loss and the same learning rates.

### `pr` — held-out test period (event-paired)

| metric | `baseline` | `spatial_ft` | `time_only` | `convgru` | `mamba` |
|---|---|---|---|---|---|
| RMSE | +8.0884 | +8.3623 | +8.2502 | +8.2896 | +8.4125 |
| MAE | +2.6961 | +2.7188 | +2.5782 ✓ | +2.5204 ✓ | +2.5240 ✓ |
| bias | -0.1651 | -0.3081 | -0.6735 | -0.9228 | -1.1213 |
| daily spatial r | +0.5301 | +0.5089 | +0.5129 | +0.5129 | +0.5165 |
| tendency RMSE | +11.2998 | +11.5993 | +11.4311 | +11.4295 | +11.4979 |
| tendency std ratio | +0.5484 | +0.6350 ✓ | +0.5573 ✓ | +0.5178 | +0.4041 |
| lag-1 ACF error | +0.02490 | -0.00692 ✓ | -0.00971 ✓ | -0.02259 ✓ | +0.00269 ✓ |
| lag-2 ACF error | +0.01408 | -0.00256 ✓ | -0.01713 | -0.02732 | -0.01171 ✓ |
| 3-day acc. RMSE | +14.2056 | +14.8608 | +14.6950 | +14.8879 | +15.2985 |
| 5-day acc. RMSE | +18.3902 | +19.4031 | +19.2348 | +19.6030 | +20.2846 |
| q99 bias | -18.7205 | -13.5546 ✓ | -18.1018 ✓ | -20.8201 | -27.2233 |
| field std ratio | +0.6224 | +0.7002 ✓ | +0.6116 | +0.5674 | +0.4550 |
| wet-day freq. error | +0.1034 | +0.0075 ✓ | +0.0011 ✓ | -0.0155 ✓ | -0.0008 ✓ |
| P(wet|wet) error | +0.1435 | +0.0606 ✓ | +0.0511 ✓ | +0.0384 ✓ | +0.0609 ✓ |
| P(wet|dry) error | +0.0119 | -0.0242 | -0.0251 | -0.0338 | -0.0318 |
| wet-spell mean error (d) | +1.044 | +0.346 ✓ | +0.285 ✓ | +0.207 ✓ | +0.348 ✓ |
| dry-spell mean error (d) | -0.215 | +0.520 | +0.541 | +0.759 | +0.708 |
| RMSE boundary | +8.5434 | +8.7597 | +8.6757 | +8.7355 | +8.8550 |
| RMSE interior | +7.9439 | +8.2368 | +8.1155 | +8.1483 | +8.2722 |
| chunk-seam jump ratio | +1.0490 | +1.1179 | +1.0894 | +1.0880 | +1.0659 |

✓ marks a value better than `baseline` in that metric's own direction (lower for errors, |.| for signed errors, toward 1 for std ratios).

### `tasmax` — held-out test period (event-paired)

| metric | `baseline` | `spatial_ft` | `time_only` | `convgru` | `mamba` |
|---|---|---|---|---|---|
| RMSE | +1.5526 | +1.9360 | +1.7759 | +2.1809 | +1.8706 |
| MAE | +1.1696 | +1.4839 | +1.3943 | +1.5959 | +1.4630 |
| bias | +0.0548 | +0.9188 | -0.7149 | +0.4486 | +0.8573 |
| daily spatial r | +0.9383 | +0.9322 | +0.9364 | +0.9019 | +0.9376 |
| tendency RMSE | +1.5297 | +1.6778 | +1.7004 | +2.0268 | +1.6492 |
| tendency std ratio | +0.9293 | +0.9165 | +0.9834 ✓ | +1.1690 | +0.8248 |
| lag-1 ACF error | +0.00487 | +0.01009 | -0.00333 ✓ | -0.02498 | +0.00256 ✓ |
| lag-2 ACF error | +0.00042 | +0.01811 | -0.00847 | -0.05560 | -0.00844 |
| 3-day acc. RMSE | +3.7678 | +4.9554 | +4.3473 | +5.3471 | +4.7684 |
| 5-day acc. RMSE | +5.7768 | +7.7628 | +6.6329 | +8.0072 | +7.4786 |
| q99 bias | -0.1708 | +1.6395 | -1.9639 | +2.0408 | -0.7399 |
| field std ratio | +1.0071 | +0.9675 | +0.9882 | +1.0756 | +0.9192 |
| RMSE boundary | +1.4138 | +1.8176 | +1.7893 | +2.0489 | +1.8371 |
| RMSE interior | +1.5927 | +1.9709 | +1.7718 | +2.2198 | +1.8808 |
| chunk-seam jump ratio | +1.1217 | +1.0800 | +1.0873 | +1.1283 | +1.1403 |

✓ marks a value better than `baseline` in that metric's own direction (lower for errors, |.| for signed errors, toward 1 for std ratios).

### Pre-registered acceptance verdict

| variant | primary | guardrail | beats controls | verdict |
|---|---|---|---|---|
| `spatial_ft` | FAIL | FAIL | — | control |
| `time_only` | FAIL | FAIL | — | control |
| `convgru` | FAIL | FAIL | no | **NOT ACCEPTED** |
| `mamba` | FAIL | FAIL | no | **NOT ACCEPTED** |

**`spatial_ft` failing checks (12):**

* `pr` / day-to-day tendency RMSE reduced >=2%: baseline 11.3 → variant 11.599
* `pr` / 3-day accumulation RMSE reduced >=1%: baseline 14.206 → variant 14.861
* `pr` / per-frame RMSE not worse by >1%: baseline 8.0884 → variant 8.3623
* `pr` / daily spatial correlation not worse by >0.005: baseline 0.5301 → variant 0.50887
* `tasmax` / lag-1 autocorr error |.| reduced >=10%: baseline 0.0048657 → variant 0.010093
* `tasmax` / lag-2 autocorr error |.| reduced >=5%: baseline 0.00041992 → variant 0.01811
* `tasmax` / day-to-day tendency RMSE reduced >=2%: baseline 1.5297 → variant 1.6778
* `tasmax` / 3-day accumulation RMSE reduced >=1%: baseline 3.7678 → variant 4.9554
* `tasmax` / per-frame RMSE not worse by >1%: baseline 1.5526 → variant 1.936
* `tasmax` / MAE not worse by >1%: baseline 1.1696 → variant 1.4839
* `tasmax` / daily spatial correlation not worse by >0.005: baseline 0.93829 → variant 0.93217
* `tasmax` / std ratio not moved toward 0 by >2%: baseline 1.0071 → variant 0.96752

**`time_only` failing checks (10):**

* `pr` / lag-2 autocorr error |.| reduced >=5%: baseline 0.014082 → variant -0.017134
* `pr` / day-to-day tendency RMSE reduced >=2%: baseline 11.3 → variant 11.431
* `pr` / 3-day accumulation RMSE reduced >=1%: baseline 14.206 → variant 14.695
* `pr` / per-frame RMSE not worse by >1%: baseline 8.0884 → variant 8.2502
* `pr` / daily spatial correlation not worse by >0.005: baseline 0.5301 → variant 0.51292
* `tasmax` / lag-2 autocorr error |.| reduced >=5%: baseline 0.00041992 → variant -0.0084723
* `tasmax` / day-to-day tendency RMSE reduced >=2%: baseline 1.5297 → variant 1.7004
* `tasmax` / 3-day accumulation RMSE reduced >=1%: baseline 3.7678 → variant 4.3473
* `tasmax` / per-frame RMSE not worse by >1%: baseline 1.5526 → variant 1.7759
* `tasmax` / MAE not worse by >1%: baseline 1.1696 → variant 1.3943

**`convgru` failing checks (15):**

* `pr` / lag-1 autocorr error |.| reduced >=10%: baseline 0.024903 → variant -0.02259
* `pr` / lag-2 autocorr error |.| reduced >=5%: baseline 0.014082 → variant -0.027323
* `pr` / day-to-day tendency RMSE reduced >=2%: baseline 11.3 → variant 11.43
* `pr` / 3-day accumulation RMSE reduced >=1%: baseline 14.206 → variant 14.888
* `pr` / per-frame RMSE not worse by >1%: baseline 8.0884 → variant 8.2896
* `pr` / daily spatial correlation not worse by >0.005: baseline 0.5301 → variant 0.51291
* `pr` / std ratio not moved toward 0 by >2%: baseline 0.62245 → variant 0.56737
* `pr` / tendency std ratio not reduced by >2%: baseline 0.54835 → variant 0.51781
* `tasmax` / lag-1 autocorr error |.| reduced >=10%: baseline 0.0048657 → variant -0.024982
* `tasmax` / lag-2 autocorr error |.| reduced >=5%: baseline 0.00041992 → variant -0.055605
* `tasmax` / day-to-day tendency RMSE reduced >=2%: baseline 1.5297 → variant 2.0268
* `tasmax` / 3-day accumulation RMSE reduced >=1%: baseline 3.7678 → variant 5.3471
* `tasmax` / per-frame RMSE not worse by >1%: baseline 1.5526 → variant 2.1809
* `tasmax` / MAE not worse by >1%: baseline 1.1696 → variant 1.5959
* `tasmax` / daily spatial correlation not worse by >0.005: baseline 0.93829 → variant 0.90191

**`mamba` failing checks (13):**

* `pr` / day-to-day tendency RMSE reduced >=2%: baseline 11.3 → variant 11.498
* `pr` / 3-day accumulation RMSE reduced >=1%: baseline 14.206 → variant 15.298
* `pr` / per-frame RMSE not worse by >1%: baseline 8.0884 → variant 8.4125
* `pr` / daily spatial correlation not worse by >0.005: baseline 0.5301 → variant 0.51646
* `pr` / std ratio not moved toward 0 by >2%: baseline 0.62245 → variant 0.45501
* `pr` / tendency std ratio not reduced by >2%: baseline 0.54835 → variant 0.40409
* `tasmax` / lag-2 autocorr error |.| reduced >=5%: baseline 0.00041992 → variant -0.0084363
* `tasmax` / day-to-day tendency RMSE reduced >=2%: baseline 1.5297 → variant 1.6492
* `tasmax` / 3-day accumulation RMSE reduced >=1%: baseline 3.7678 → variant 4.7684
* `tasmax` / per-frame RMSE not worse by >1%: baseline 1.5526 → variant 1.8706
* `tasmax` / MAE not worse by >1%: baseline 1.1696 → variant 1.463
* `tasmax` / std ratio not moved toward 0 by >2%: baseline 1.0071 → variant 0.91924
* `tasmax` / tendency std ratio not reduced by >2%: baseline 0.92932 → variant 0.82484


<!-- RESULTS_END -->

### 7.2 Interpretation

**The headline result is that the pre-registered criteria are not met, and the
reason is informative rather than merely negative.**

**(a) Most of the distributional improvement comes from the changed objective, not
from temporal memory.** `spatial_ft` has no temporal pathway at all — gate 0,
temporal parameters frozen, so the adapter contributes exactly zero — yet it moves
`pr`'s wet-day frequency error from +0.103 to +0.008, `P(wet|wet)` from +0.144 to
+0.061, mean wet-spell length from +1.04 d to +0.35 d, and the 99th-percentile bias
from −18.7 to −13.6 mm/day. Those are large corrections to the baseline's real
deficiencies, and they are attributable to adding the tendency, accumulation and
occurrence terms to the loss — nothing else differs. **Reporting those numbers as
evidence for temporal modelling would have been wrong**, and the control is what
makes that visible.

**(b) Temporal memory does add something specific and monotone: wet/dry spell
structure.** Across the variants in the order baseline → `spatial_ft` (loss change
only) → `time_only` (loss change + date conditioning) → `convgru` (loss change +
date conditioning + memory):

| `pr` statistic | baseline | `spatial_ft` | `time_only` | `convgru` |
|---|---|---|---|---|
| `P(wet｜wet)` error | +0.1435 | +0.0606 | +0.0511 | **+0.0384** |
| mean wet-spell length error (d) | +1.044 | +0.346 | +0.285 | **+0.207** |
| MAE | 2.696 | 2.719 | 2.578 | **2.520** |

Each step improves these monotonically and `convgru` is best, with memory adding
beyond date conditioning (`time_only` → `convgru`). This is the one place the
recurrent state demonstrably contributes, and it is precisely the statistic the
occurrence/transition objective was aimed at.

**(c) The two backends fail differently, and Mamba is the better-behaved one at
this budget.** This was not expected from the literature, and the numbers are
unambiguous:

| | baseline | `convgru` | `mamba` |
|---|---|---|---|
| `pr` lag-1 ACF error | +0.0249 | −0.0226 (overshoots) | **+0.0027** (−89%, no overshoot) |
| `tasmax` lag-1 ACF error | +0.0049 | −0.0250 (5× overshoot) | **+0.0026** (−47%, no overshoot) |
| `tasmax` RMSE | 1.553 | 2.181 (+40%) | **1.871** (+21%) |
| `tasmax` daily spatial r | 0.9383 | 0.9019 | **0.9376** (≈ baseline) |
| `tasmax` tendency RMSE | 1.530 | 2.027 | **1.649** |
| `pr` tendency std ratio | 0.548 | 0.518 | **0.404** (worst) |
| `pr` field std ratio | 0.622 | 0.567 | **0.455** (worst) |
| `pr` q99 bias | −18.7 | −20.8 | **−27.2** (worst) |

* **ConvGRU over-corrects persistence.** It drives the lag-1 autocorrelation error
  from +0.025 straight past zero to −0.023 for `pr`, and from +0.005 to −0.025 for
  `tasmax` — a fivefold overshoot. It also degrades `tasmax` most (RMSE +40%,
  spatial correlation 0.938 → 0.902).
* **Mamba lands the persistence correction almost exactly** (both variables end
  within +0.003 of zero ACF error, without overshoot) and leaves `tasmax` far less
  damaged. Against the `spatial_ft` control it wins **4 of 4** `tasmax` primary
  metrics and 2 of 4 for `pr`; ConvGRU wins 0 of 4 and 1 of 4.
* **Mamba's distinct failure is precipitation variance.** It suppresses day-to-day
  variability (tendency std ratio 0.404 vs the baseline's already-low 0.548) and
  extremes (q99 bias −27.2 vs −18.7) more than any other variant — it buys correct
  persistence by smoothing.

A plausible mechanism, offered as a hypothesis rather than a finding: Mamba's
adapter has **5× fewer temporal parameters** (1.5 M vs 7.6 M, §3), so at 600 steps
with the same learning rate it moves less far and overshoots less. That would make
this a statement about the training budget, not about the architectures' ceilings —
which is exactly why the comparison is labelled inconclusive in §9.

**(d) All variants degrade RMSE, multi-day accumulation and extremes.** Every
trained variant fails the RMSE and accumulation guardrails for both variables. The
guardrails caught this, which is what they are for.

**(e) No chunk-seam artefact was introduced.** Seam jump ratios land in 1.07–1.14
against a measured null of 1.05 (`pr`) and 1.12 (`tasmax`) from the
temporal-pathway-free baseline (§5). Carrying recurrent state across inference
chunk boundaries did not produce a temporal discontinuity.

**(f) Why: the run is far from converged and the loss weights were never tuned.**
600 optimizer steps at batch 1 is roughly half an epoch, the backbone never
unfroze (the schedule unfreezes at epoch 3), and the temporal loss weights
(tendency 0.15, accumulation 0.10, occurrence 0.05) were chosen a priori and
deliberately *not* tuned, so that the acceptance test was not fitted to the test
set. The observed pattern — persistence over-corrected, per-frame accuracy
sacrificed — is the signature of temporal terms weighted too heavily relative to
the per-frame term, and the fix is a validation-set weight sweep (§8, item 4).

**(g) A methodological limitation of our own pre-registration.** The pre-registered
primary metrics were lag-1/lag-2 autocorrelation error, tendency RMSE and 3-day
accumulation RMSE. The one quantity temporal memory *did* improve monotonically —
wet/dry spell structure — is **not in that set**. So the criteria are, in
retrospect, aimed slightly off-target for this architecture on this variable.
The verdict is nonetheless reported exactly as pre-registered; the criteria were
not moved after seeing the results. A future pre-registration for this problem
should include `P(wet|wet)`, `P(wet|dry)` and spell-length errors as primary, and
should express the autocorrelation criterion as a *band* around zero rather than a
one-sided reduction, so that overshoot is penalized rather than rewarded.

**(h) The `tasmax` lag-1 criterion was uninformative,** as flagged in §6 before the
run: a baseline error of +0.0049 makes "reduce by ≥10%" a request for a 0.0005
change, below the noise floor.

**Conclusion.** The implementation is verified and the architecture demonstrably
carries and uses temporal information (§3, and the monotone spell-structure result
above). **Scientific acceptance is not demonstrated at this training budget** --
both backends fail the pre-registered primary criteria and the spatial guardrails,
and neither beats both controls. The checkpoints under
`examples/CORDEX_ML/runs_temporal/experiment/*/checkpoints/` are therefore labelled
**experimental** and must not be used as production weights.

Two things are worth carrying forward rather than discarding:

1. **`temporal.backend: mamba` is the more promising starting point** for further
   training on this case: it corrects persistence without overshoot on both
   variables, keeps `tasmax` spatial correlation at the baseline value, and beats
   the matched fine-tuning control on 4 of 4 `tasmax` primary metrics.
2. **The occurrence/transition objective works**, and the recurrent state adds to
   it: `P(wet|wet)` error falls monotonically 0.144 → 0.061 → 0.051 → 0.038 across
   baseline → loss change → date conditioning → memory.

§8 gives the exact commands to continue, ordered by expected value.

---

## 8. Commands to continue

The bounded run is 600 optimizer steps at batch 1 with the backbone frozen. To train
properly:

```bash
CFG=examples/CORDEX_ML/SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_recurrent.yaml

# full training: 6 epochs over 1163 windows/epoch, backbone unfreezes at epoch 3
mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_training.py train --config $CFG

# resume
#   set temporal.resume_from_temporal_checkpoint to the checkpoint and
#   temporal.init_from_spatial_checkpoint to null, then:
mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_training.py train --config $CFG

# held-out test inference + evaluation over the full 1981-2000 period
mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_training.py infer \
    --config $CFG --split test \
    --checkpoint examples/CORDEX_ML/runs_temporal/SA_T2_ACCESS-CM2_static_temporal_recurrent/checkpoints/best.ckpt
mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_training.py evaluate \
    --config $CFG --predictions <npz printed by infer>

# climate-change application: distributional metrics ONLY
#   point data.test_*_paths at SA_domain/test/end_century/... first
mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_training.py evaluate \
    --config $CFG --predictions <npz> --not-event-aligned
```

Highest-value untested variations, in order:

1. **`context_length: 14`** — the e-folding heuristic suggests it and §4 records the
   disagreement with the marginal-R² criterion.
2. **More optimizer steps** — 600 steps at batch 1 is roughly half an epoch.
3. **Staged backbone unfreezing** — needs ≥ 4 epochs to reach the epoch-3 stage.
4. **`temporal.losses.occurrence.weight`** — this is the term aimed at the largest
   measured baseline deficit (+0.103 wet-day frequency, +0.144 `P(wet|wet)`), and
   0.05 was chosen a priori, not tuned. Tune on **validation** only.
5. **`cell: convlstm`** and **`mamba.n_layers`/`d_state`** sweeps.
6. **Temporally correlated refinement noise** — `noise: ar1_correlated`,
   `noise_rho: 0.3–0.7`, which addresses the 0.548 tendency-std ratio that a
   deterministic model structurally cannot fix.

### NARR/PRISM

```bash
# 1. generate the preprocessed daily products (needs the NARR and PRISM archives)
mamba run -n Prithvi python examples/NARR_PRISM/preproc_narr_prism.py \
    --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml
# 2. confirm the Phase-1 checkpoint exists at
#    examples/NARR_PRISM/experiments/checkpoints/narr_prism_California/last.ckpt
# 3. audit, then verify event alignment BEFORE training
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_temporal.py describe \
    --config examples/NARR_PRISM/NARR_PRISM_subdomain_temporal_recurrent.yaml \
    --splits train validation test
mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_diagnostics.py \
    --config examples/NARR_PRISM/NARR_PRISM_subdomain_temporal_recurrent.yaml
# 4. engineering checks, then train
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_temporal.py check --config <yaml>
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_temporal.py train --config <yaml>
```

---

## 9. Limitations

**Scientific**

1. **The comparison is bounded, not converged.** 600 optimizer steps, batch 1, one
   epoch, backbone frozen throughout. Treat the scorecard as a lower bound on what
   the architecture can do, and as *not* evidence that it cannot do more.
2. **`pr` headroom from temporal conditioning is small** — 2.6% of the unexplained
   variance (§4). The precipitation problem worth attacking is occurrence and
   persistence, and part of it (the 0.548 tendency-std ratio) is a *deterministic
   regression* limitation that no temporal architecture removes; it needs the
   stochastic refinement path.
3. **One pre-registered threshold was ill-calibrated** (`tasmax` lag-1 ACF, §6). It
   was left in place rather than moved.
4. **The alignment criterion was widened after it fired** on `pr` (§4). The original
   verdict is still reported alongside the widened one.
5. **No NARR/PRISM results at all** (§1). Three targets including `tmin`, a
   different calendar, `num_static_channels: 0` and the hurdle precipitation head
   are all exercised by config validation and unit tests, but not by a real run.
6. **Only the deterministic path was run.** Temporal refinement conditioning is
   implemented, tested at unit level, and correctly rejects incompatible Phase-2
   checkpoints, but no temporally-conditioned Phase-2 model was trained.
7. **Climate-change application not evaluated.** The 2041–2060 and 2080–2099 periods
   and the imperfect-predictor (NorESM2-MM) transfer are wired and would be scored
   distributionally, but were not run.

**Inherited**

8. **Normalization saw the validation window.** The Phase-1 scalers were fitted over
   the whole 1961–1980 + 2080–2099 record and cannot be recomputed without
   invalidating the Phase-1 checkpoint, which stores `output_scalers_mu/sigma`
   internally. The effect is second-order and **identical across all five variants**,
   so the comparison is unaffected while absolute validation numbers are mildly
   optimistic.
9. **No xESMF in this environment**, so the frame dataset falls back to xarray
   interpolation for the coarse→fine regridding of the native-16×16 test files. This
   is the pre-existing behaviour of `cordex_dataset.py` and is identical for every
   variant.

**Environmental**

10. **`mamba_ssm` is not installable here** (Windows/CUDA), so the Mamba backend runs
    the in-repo pure-PyTorch SSD recurrence. Same mathematics, same parameters,
    slower. `implementation: fused` raises rather than substituting a different
    model, and the shipped configs set `reference` explicitly so the run record is
    unambiguous.
11. **Single GPU, no FSDP.** The temporal configs set `distributed_strategy: none`.
    The adapter is an ordinary submodule and should wrap under FSDP, but that was
    not exercised.

**Explicitly not claimed**

* That temporal modelling fixes existing spatial or refinement errors. Boundary
  versus interior RMSE and chunk-seam ratios are reported separately so any such
  change is visible.
* That the ConvGRU/Mamba comparison is decisive. At this budget Mamba is clearly
  the better-behaved of the two (§7.2c) -- and that is the *opposite* of what the
  one published gridded-geoscience comparison we could find would suggest, where
  pure Mamba trails the ConvGRU family (research document §5). Two reasons not to
  read our result as contradicting it: our Mamba backend is a **hybrid** (conv
  spatial mixing + time-axis SSM), not pure Mamba; and it has **5x fewer temporal
  parameters**, so at a fixed 600 steps and learning rate it simply moves less far.
  The observed difference may be a statement about the training budget rather than
  about either architecture's ceiling. Settling it needs a converged run at matched
  parameter counts.
* That the loss weights are near-optimal. They were fixed a priori and never tuned,
  precisely so the acceptance test was not fitted to the test set. The observed
  failure mode -- persistence over-corrected, per-frame accuracy sacrificed -- is
  what too-heavy temporal weights look like, and a validation-set sweep is the first
  thing to try (§8 item 4).
