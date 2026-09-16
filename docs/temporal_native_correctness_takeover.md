# Native-pair and NARR correctness corrections - 2026-09-15

The inherited artifacts remain preserved in `artifacts/temporal_takeover/20260915T113310`.
This is an additive correction record; earlier reports are historical evidence.

## Historical adaptation and auxiliary objectives

The inherited widened `embedding.proj.weight` belongs to the encoder optimizer group.
With the shipped encoder freeze policy its zero historical slice cannot learn.
The corrected adapter adds `temporal_adapter.history_projection.weight`, a separate
zero-initialized convolution with the same spatial kernel, padding, and stride.
A projection hook adds its historical contribution before the shared transformer.
The historical contribution belongs to the temporal optimizer group. Frozen original
embedding weights are excluded from the optimizer, including AdamW momentum and decay.

Both auxiliary heads remain registered under `temporal_adapter`. Their gradients pass
through frozen computations into the historical projection and time conditioning.
Frozen transformer parameters do not update. The positive-lead transition objective
exercises the lead-time slope; zero-lead main downscaling alone cannot do so.
Auxiliary weights are applied once inside `compute_pretext_losses`.

Auxiliary normalization now uses the same crop-aware scaler resolver as the model.
NARR has **32 inputs**: 15 atmospheric fields, elevation, and 16 validity indicators.
Its auxiliary heads predict the 15 atmospheric fields only and score their finite,
observed cells using the matching masks. Elevation and masks remain predictor inputs.

Native main and auxiliary forwards pass explicit output geometry rather than observed
high-resolution target values. The runner also accepts no `y` tensor when geometry is
provided. This establishes prediction independence from verification values; it does
not make the current NARR file adapter a predictor-only data archive reader.

## Evidence

- `tests/test_temporal_native_pair.py`: 53 tests passed before NARR integration.
  These include independent one-timestamp Phase-1 checkpoint initialization,
  direct auxiliary prediction comparisons under hidden-target perturbations,
  frozen-backbone/shared-adapter gradients, required YAML presence, NARR masks,
  and target-free runner geometry. Initialization uses stated numerical tolerances;
  no universal bit-exactness is asserted across kernels or devices.
- The root task exercised actual SA optimizer updates, save/resume, and history
  sensitivity for plain and pretext variants. Machine-readable proof is under
  `artifacts/temporal_takeover/optimizer_proof_v1`.
- `tests/test_temporal_narr_adapter.py`: four strict synthetic NetCDF tests passed,
  checking the actual spatial loader contract and full-domain tile blending at
  chunk lengths two and four. These are engineering fixtures, not scientific NARR
  validation or trained-model skill measurements.

Shared regression check after NARR integration: 85 passed across
`test_temporal_narr_adapter.py`, `test_temporal_model.py`, and
`test_temporal_calendar_and_losses.py`.

## NARR executable data and inference pathway

The inherited temporal factory only globbed `.npz` files and used `train` as a
mode directory. The established NARR preprocessor writes per-day `.nc` files
under `training`, `validation`, and `inference`.

`NarrPrismDatasetFrameSource` now wraps the existing strict NARR spatial dataset.
It retains canonical coordinates, preprocessing/scaler signatures, field ordering,
physical units, mean-filled missing predictors, and their masks. Temporal training
enumerates the existing valid tiled cores, holding core/halo/scaler offsets fixed
across every date. Unique emission plans enumerate each date once per spatial tile;
spatially overlapping tiles must be blended before temporal scoring.

Temporal NARR inference now covers the full canonical domain with the configured
core, overlap, halo, and Hann blending. Each spatial tile carries independent temporal
state/history. It no longer reports one center crop as full-domain inference.
The generic `.npz` adapter remains available for older explicit consumers.
Full-domain NARR results retain canonical latitude/longitude for NetCDF export.

Actual NARR data/checkpoints were unavailable on this execution host when these
changes were implemented. No synthetic fixture result establishes NARR skill.

## Provenance qualification

Current checkpoint dimension incompatibility does not establish the historical
initialization of an existing Phase-1 checkpoint. The native adapter reuses the
downscaling checkpoint and initializes historical, time, and auxiliary adaptation
parameters anew. Earlier foundation-weight provenance remains unresolved unless
independent records verify it. Similar parameter histograms cannot prove that time
conditioning never learned useful information.


## Regional saved-artifact diagnostic

`examples/CORDEX_ML/cordex_temporal_regional_report.py` reads saved predictions,
targets, masks, and dates from `predictions.npz`, with `resolved.yaml` supplying
case and threshold metadata. It can read latitude/longitude metadata only from
the resolved target NetCDF when the prediction archive has no coordinates; it
saves a coordinate-only NPZ for fully offline regeneration. It does not reload
verification values from the original files or rerun a model.

The inherited baseline report is at
`artifacts/temporal_takeover/20260915T113310/regional_baseline/regional_metrics.json`.
Two PNGs show prediction mean, target mean, and signed spatial bias on physical
coordinates, and `spatial_bias_fields.npz` preserves the numeric fields.
Statistics cover full domain, an eight-cell boundary, interior, and geographic
southeast defined by latitude below the domain median and longitude above it.
The southeast bounds are -34.7 to -28.4 degrees north and 26.9 to 33.2 degrees east.
Threshold comparison is strictly `> 1 mm/day`, matching the archived scorer.

For the inherited baseline, precipitation RMSE is 8.088 full-domain, 8.543
boundary, 7.944 interior, and 8.387 southeast mm/day. In the southeast,
wet-day frequency is 53.63% predicted versus 40.70% target, while the pooled
99th-percentile bias is -18.04 mm/day. These are additional diagnostics,
not acceptance decisions or a retuning target.

`tests/test_temporal_regional_report.py`: two known-example tests passed,
including descending-latitude southeast selection and exact wet-day semantics.
Final NARR coordinate-export regression: four tests passed in 13.44 seconds.


## Final legacy-constructor regression repair

The broad regression run exposed a compatibility regression in the added model
metadata: legacy direct `SimpleNamespace` constructors omit `input_vars`,
`input_levels`, and target grid sizes. These fields now default to empty predictor
metadata and absent configured output geometry. Legacy calls still obtain geometry
from the provided target tensor; target-free calls can provide `__output_shape`.
No learned computation changed for the resolved SA/NARR configurations.

Focused repair verification: all nine spatial-alignment tests passed (5.44 seconds).
The native-pair suite was rerun after this repair. Active bounded-experiment source
snapshots were not edited.


## SA physical-coordinate export correction

`CordexFrameSource.spatial_coordinates()` now reads latitude/longitude metadata
from the same target template used by the spatial dataset. It returns the exact
inference crop, and `InferenceResult` carries these coordinates into NetCDF export.
Previously SA exports could use zero-based array indices as latitude/longitude.
The writer also supports two-dimensional physical latitude/longitude grids.

Target-value independence and target-file independence are different claims.
The temporal model receives geometry rather than verification values, and supports
explicit geometry without a `y` tensor. CORDEX/NARR sources still require target
files for date joins, grids, units, and verification loading. Selecting the inference
split for time-feature probing removes an unrelated training-archive dependency;
it does not remove this source-level target-file dependency. A fully
predictor-only deployment file adapter is not implemented by this change.

Focused fixtures check actual source-to-inference-to-NetCDF coordinates for full
and cropped CORDEX grids, the missing-leap-day discontinuity, units, and a 2D-grid
writer case, alongside the existing NARR coordinate-export checks.

Final coordinate/source export check: seven tests passed in 11.44 seconds.

## Partial-period history and explicit cold-start time convention

A CPU regression with a structurally real tiny native model reproduced a history
error: selecting January 5 as the first requested output dropped available
January 4 context and silently treated January 5 as a cold start. Native inference
now builds its chronology within the declared split, loads only the required prior
predictor context for each chunk, and emits only requested dates. Actual split
boundaries and missing-date discontinuities still prevent history crossing.
A single requested date no longer requires a complete training-length window.
Limits count emitted dates; temporary cold-start permission is restored on errors.
This correction applies to finite-history native inference. Stateful backends keep
their existing requested-period cold-start protocol.

A second regression exposed an IndexError at a true cold start with deeper offsets
such as [1, 3]. The explicit native_pair.cold_start_time_mode now defines the time
conditioning used while unavailable historical slots duplicate the current input:

- legacy_nominal (default, including old configs/checkpoints with the field absent)
  preserves a synthetic full configured interval: 24 hours for [1], 72 for [1, 3].
  This is a cold-start initialization convention, not an observed elapsed time.
- selected_timestamps (opt-in) measures the span of actually available selected
  history timestamps; duplicated current slots contribute zero lag. For [1, 3],
  run positions 0/1/2/3 use 0/24/24/72 hours on a daily archive.

Once all selected history exists, both modes use measured date intervals. Neither
mode allows omitted context inside a contiguous observed run. The duplicate_current
capacity control deliberately retains the matching real-history time metadata,
separately from how true cold starts are represented. Training windows require
complete main and auxiliary history, so no trained window uses this cold rule.

The convention is serialized and part of strict resume compatibility. CPU loading
of the preserved native_pair and native_pair_pretext optimizer proof checkpoints
confirmed that their absent field resolves to legacy_nominal, all 214/222 state
tensors remain exact, optimizer step 2 restores, and attempting selected_timestamps
is rejected with an explicit contract mismatch. Evidence is additive in
artifacts/temporal_takeover/20260915T113310/final_checkpoint_load_validation.json.

Frozen bounded-experiment sources and archived numerical arrays were not edited.
Their default [1] starts at the actual source boundary and retains its original
24-hour synthetic cold condition. The deeper-span crash and arbitrary partial
start were outside that run's protocol; these fixes do not imply all trained
steps or its existing scorecard were invalidated.

Focused CPU coverage includes full-versus-partial exact predictions under both
time modes, near-boundary partial starts, single-date context, declared splits,
true gaps, chunk context, output budgets, NARR tiled partial-date equivalence, and
legacy config parsing. All 79 combined native/source/partial-inference tests passed
in 16.99 seconds.

The auxiliary-loss docstring now distinguishes backbone call counts from measured
compute. The SA training window uses five main native calls, seven with both
auxiliary objectives, and seven recurrent calls. Equal call counts do not prove
equal FLOPs, retained activations, memory, or runtime.
