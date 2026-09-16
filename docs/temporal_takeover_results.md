# Temporal takeover results

Status captured: 2026-09-16T00:35:22.367022+00:00

**Scientific status: not accepted.** The bounded comparison is separate
from optimizer engineering checks and the pretrained-transfer branch.

## Execution and preservation

Branch `Prithvi-UNet_temporal_model`, HEAD `2f237147f24f29b3146701b6d4ccb1b1fcbf1c21`.
Inherited changes/output inventory preserved under `artifacts/temporal_takeover/20260915T113310`.
No old experiment output was deleted or overwritten; no push was made.

| variant | inherited status | current status | saved optimizer updates |
|---|---|---|---|
| baseline | completed | completed | 0 |
| spatial_ft | interrupted | completed | 600 |
| time_only | not started | completed | 600 |
| native_pair | not started | completed | 600 |
| native_pair_nohistory | not started | completed | 600 |
| native_pair_pretext | not started | completed | 600 |

Active process IDs: .
Current process commands, paths, and checkpoint counters: `artifacts/temporal_takeover/latest_status.json`.
A running checkpoint counter is a durable lower bound, not an estimate of live updates.

The frozen initial driver checks the wrong counter for spatial_ft. Its valid checkpoint
is recovered by the separately versioned continuation driver using the same frozen model
sources and original 600-update, 60-validation-window, one-epoch protocol.

## Actual learning through the real trainer

| variant | actual updates including resume | max history weight change, first step | save/load max difference | chunk max difference |
|---|---|---|---|---|
| native_pair | 2 | 0.0001999996 | 0.0 | 0.0 |
| native_pair_pretext | 2 | 0.0001999996 | 0.0 | 0.0 |

### Saved SA reference metrics (1981–1983)

| variable | units | RMSE | tendency RMSE | lag-1 error | 3-day accumulation RMSE |
|---|---|---|---|---|---|
| pr | mm/day | 8.08836 | 11.2998 | 0.0249026 | 14.2056 |
| tasmax | K | 1.5526 | 1.52971 | 0.00486572 | 3.76775 |

The full 1,095-date baseline archive matched the inherited predictions, targets,
masks, dates, variables and seams exactly; max prediction difference was zero.

Historical parameters occur exactly once at LR 2e-4. Frozen encoder/backbone and
all eight physical normalization tensors stayed unchanged. History affects predictions
after optimization without manual amplification. Both auxiliary losses reach the shared
historical projection; the positive-lead transition exercises the lead-time slope.
Auxiliary heads are optimized and checkpointed. Frozen transformer weights do not update.
Zero differences above are measurements on this host; portable tests also use explicit tolerances.
Evidence: `artifacts/temporal_takeover/optimizer_proof_v1/optimizer_proof.json`.

## Physical units, splits and NARR

SA precipitation matches file values converted exactly once to mm/day; applying the
converter twice is an identity after the first conversion. Temperature stays Kelvin.
Timestamp joins and discontinuity-aware windows preserve actual daily means and gaps.
Phase-1 saw temporal validation years 1977–1980 and the inherited scalers include those
years. Validation is not independent from Phase-1; shared exposure does not remove its effect.
NARR has 32 inputs, ppt/tmax/tmin outputs, zero separate static channels and a hurdle head.
Actual NARR assets remain absent; strict NetCDF/tile/mask tests use synthetic fixtures.
See `data_contracts.json` and `docs/temporal_native_correctness_takeover.md`.

## Provenance and separate transfer

Current 2560-to-1024 loading matches no backbone learned parameters. Historical Phase-1
foundation provenance remains unresolved; a related notebook records loading a smaller
checkpoint. Weight histograms do not settle provenance or whether time weights learned.
The separate transfer path loads the official rollout checkpoint's first local/global
transformer pair and time maps, with new regional input/latent adapters. It explicitly
adapts reduced daily predictors and regional token geometry. It does not claim full-state
MERRA-2 compatibility or full-model transfer. Negative-time v1 audits are invalidated;
use corrected v2 evidence in `docs/temporal_pretrained_transfer_takeover.md`.

## Scores, figures and runnable workflow

Original ACCEPTANCE thresholds, MUST_BEAT and extra no-history/pretext controls remain
unchanged. Corrected scoring handles dotted metric names, missing/NaN metrics, near-zero
baselines, missing controls and invalidated/partial experiments. Archived ConvGRU/Mamba
verdicts remain not accepted. Versioned additional scorecards are under the takeover evidence.
Eleven baseline figures: `artifacts/temporal_takeover/20260915T113310/inherited_baseline_figures`.
Geographic full/boundary/interior/southeast metrics and mean-bias maps: `regional_baseline/`.
These figures describe the saved baseline, not a completed native-pair comparison.

Selected workflow: `examples/CORDEX_ML/notebooks/Temporal_selected_training_v2.ipynb`.
CLI: `train --resume CHECKPOINT`, repeated `--set dotted.path=YAML_value`, then `infer`/`evaluate`.
Four optional extended YAMLs exist for both cases and both native variants; none launched.
Staged unfreezing retains existing Adam moments and never unfreezes normalization.
All four stochastic-refinement interfaces remain; their scientific recalibration is separate.

## Continuation

Consult `docs/temporal_codex_handoff.md` before acting. Do not launch duplicate GPU work.
Refresh this report with `python examples/CORDEX_ML/cordex_temporal_takeover_summary.py`.
Final scientific comparison, figures and verdict require all planned variant artifacts.
A separate CPU-only report process waits for the bounded controller, then generates
regional maps/metrics for all six variants under
`artifacts/temporal_takeover/bounded_final_reports_v1_20260915/`. Its execution.json
records completion or the precise blocker; it never starts training.

## Saved comparison results

Only completed variants with saved evaluations appear below; missing variants remain unscored.

### pr (mm/day)

| variant | RMSE | tendency RMSE | lag-1 error | lag-2 error | 3-day accumulation RMSE |
|---|---:|---:|---:|---:|---:|
| baseline | 8.08836 | 11.2998 | 0.0249026 | 0.0140823 | 14.2056 |
| spatial_ft | 8.17956 | 11.3901 | 0.015659 | 0.00610964 | 14.4406 |
| time_only | 8.24477 | 11.4662 | 0.0373553 | 0.0230475 | 14.5718 |
| native_pair | 8.25728 | 11.4602 | 0.0628959 | 0.0414763 | 14.6559 |
| native_pair_nohistory | 8.22554 | 11.4214 | 0.00138336 | -0.0147742 | 14.5877 |
| native_pair_pretext | 8.34942 | 11.538 | 0.0387944 | 0.0246678 | 14.928 |

### tasmax (K)

| variant | RMSE | tendency RMSE | lag-1 error | lag-2 error | 3-day accumulation RMSE |
|---|---:|---:|---:|---:|---:|
| baseline | 1.5526 | 1.52971 | 0.00486572 | 0.000419922 | 3.76775 |
| spatial_ft | 1.57164 | 1.61318 | 0.0106091 | 0.0150363 | 3.71694 |
| time_only | 1.74483 | 1.61322 | 0.00642452 | 0.00274992 | 4.36478 |
| native_pair | 1.83662 | 1.68712 | 0.00790465 | 0.000627532 | 4.60588 |
| native_pair_nohistory | 1.774 | 1.71655 | 0.0174847 | 0.0348165 | 4.32705 |
| native_pair_pretext | 1.60083 | 1.72119 | 0.00857706 | 0.0179744 | 3.67448 |

### Complete comparison verdicts

- **native_pair: not accepted.**
  - pr: lag-1 autocorr error |.| reduced >=10%; lag-2 autocorr error |.| reduced >=5%; day-to-day tendency RMSE reduced >=2%; 3-day accumulation RMSE reduced >=1%; per-frame RMSE not worse by >1%; MAE not worse by >1%; daily spatial correlation not worse by >0.005; 99th-pct bias |.| not worse by >5%; tendency std ratio not reduced by >2%.
  - tasmax: lag-1 autocorr error |.| reduced >=10%; lag-2 autocorr error |.| reduced >=5%; day-to-day tendency RMSE reduced >=2%; 3-day accumulation RMSE reduced >=1%; per-frame RMSE not worse by >1%; MAE not worse by >1%; std ratio not moved toward 0 by >2%; tendency std ratio not reduced by >2%.
  - Required control comparisons did not all pass; see the versioned scorecard.
- **native_pair_pretext: not accepted.**
  - pr: lag-1 autocorr error |.| reduced >=10%; lag-2 autocorr error |.| reduced >=5%; day-to-day tendency RMSE reduced >=2%; 3-day accumulation RMSE reduced >=1%; per-frame RMSE not worse by >1%; MAE not worse by >1%; daily spatial correlation not worse by >0.005; 99th-pct bias |.| not worse by >5%; tendency std ratio not reduced by >2%.
  - tasmax: lag-1 autocorr error |.| reduced >=10%; lag-2 autocorr error |.| reduced >=5%; day-to-day tendency RMSE reduced >=2%; per-frame RMSE not worse by >1%; MAE not worse by >1%; 99th-pct bias |.| not worse by >5%; std ratio not moved toward 0 by >2%.
  - Required control comparisons did not all pass; see the versioned scorecard.
