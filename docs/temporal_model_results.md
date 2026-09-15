# Temporal Prithvi-UNet: results, provenance, and limitations

Branch `Prithvi-UNet_temporal_model`, created from `Prithvi-UNet-stochastic_refinement`
@ `dd48c6f`. Environment: `Prithvi` mamba env at
`C:\Users\huikyole\AppData\Local\miniforge3\envs\Prithvi`, Python 3.12.13,
torch 2.11.0+cu128, one NVIDIA RTX PRO 6000 Blackwell Max-Q (97 GiB).

Companion documents: `temporal_model_research.md` (literature and decisions),
`temporal_model_architecture.md` (shapes, causality, state, losses, checkpoints).

> **Read this first.** The implementation is complete and its engineering
> properties are verified on real data and real checkpoints. The **scientific**
> comparison is a *bounded* fine-tuning run (600 optimizer steps per variant, one
> GPU, ~4 h total), not a converged experiment. Where the pre-registered acceptance
> criteria are not met, the checkpoints are labelled experimental and the exact
> commands to continue are given in §8. No improvement is claimed without the
> measurement that supports it.

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

Full JSON: `artifacts/temporal_check_SA_{recurrent,mamba}.json`.

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

Throughput: ~4.3 s per 7-frame window, 16.7 GiB peak, so 600 windows ≈ 43 min per
variant plus ~7 min inference over 1095 test dates.

Measured trainable set at epoch 0 (backbone frozen until epoch 3, which a
single-epoch bounded run never reaches — so the backbone stayed frozen throughout):

```
[epoch 0] trainable: backbone=frozen, encoder=frozen, decoder=train, temporal=train
          groups: temporal@2.00e-04 (7,642,752), decoder@5.00e-05 (15,791,365), other@5.00e-05 (65,570)
variable scales (from training scalers): {'pr': 16.066, 'tasmax': 6.340}
```

### 7.1 Results

<!-- RESULTS_TABLE_PLACEHOLDER -->

*This section is completed by the run; see
`examples/CORDEX_ML/runs_temporal/experiment/scorecard.json` for the machine-readable
verdict and `<variant>/evaluation.json` for every metric.*

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
* That the ConvGRU/Mamba comparison is decisive at this training budget. The
  literature (§5 of the research document) finds pure Mamba *behind* the ConvGRU
  family on gridded fields while hybrids lead; our Mamba backend is a hybrid, and
  600 steps is not enough to separate them.
