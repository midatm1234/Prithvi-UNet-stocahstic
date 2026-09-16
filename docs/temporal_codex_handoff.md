# Temporal takeover — 2026-09-15

## Current entry point for continuation

Read `artifacts/temporal_takeover/latest_status.json` for timestamped process and
checkpoint state; regenerate it with
`python examples/CORDEX_ML/cordex_temporal_takeover_summary.py`.

The original bounded driver PID61848 exited after valid600-update spatial training
and its known counter assertion. Controller **PID20944** observed that exit,
strictly loaded all231 tensors from the spatial best checkpoint at global600 /
temporal0. Baseline and spatial_ft now have complete1095-date evaluations.
Time_only is training; the three native variants remain queued. This is
an active, incomplete scientific comparison, not a completed scorecard.

CPU report process **PID23068** waits for the controller, then generates all six
regional reports and refreshes the summary. Never start a second GPU experiment
while the recorded controller is active. Current source is ahead of the immutable
running source snapshot by tested inference/compatibility/launcher improvements;
those changes were not injected into the run. All milestone sections below are
chronological evidence; the latest machine-readable state overrides old progress
counts. See the final sections for compatibility limits and exact next actions.

## Preserved state

- Branch: `Prithvi-UNet_temporal_model`; HEAD `2f237147f24f29b3146701b6d4ccb1b1fcbf1c21`.
- Before edits, staged/unstaged binary patches, all modified/untracked source files,
  process list and inherited output inventory saved in
  `artifacts/temporal_takeover/20260915T113310/`.
- No repository/parent AGENTS.md or CLAUDE.md found by hidden-file search.
- Original `runs_temporal/experiment`, `experiment_native_pair`, and
  `docs/temporal_experiment_records` preserved. No pushes or branch changes.
- Prithvi interpreter exists at
  `C:/Users/huikyole/AppData/Local/miniforge3/envs/Prithvi/python.exe`, Python 3.12.13.
- RTX PRO 6000 Blackwell: 97,887 MiB VRAM; approximately 188 GB free on D:.
- No temporal experiment process at takeover. Unrelated scientific-finish watcher
  PID 4420 and notebook kernel PID 47884 exist; left untouched.
- Windows sandbox process startup returns error 1327; approved elevated execution
  is used for repository operations. No environment replacement.

## Initial findings

- Inherited baseline completed: 1,095 test dates, predictions and evaluation saved.
- Inherited spatial_ft has resolved config only; no saved training checkpoint.
  Its actual updates before interruption cannot be reconstructed from that fact.
- Other native-pair experiment variants have no output directories.
- Trainer inspection confirms max-steps counted microbatches and dropped partial
  accumulation cycles. The experiment runner explicitly sets accumulation to 1;
  the standalone pretext YAML uses 16, so the defects affect budgets differently.
- Missing scorecard metrics were skipped without failing aggregate booleans.
  Corrected scorecards will be versioned; acceptance thresholds remain unchanged.

## Work in progress

- Trainer: immutable scalers, optimizer budgets/counters, remainder accumulation,
  optimizer/RNG restoration and honest resume classification.
- Native pair: separate historical adaptation parameter, auxiliary evidence,
  target-value independence, inference and regression coverage.
- Transfer: local assets, historical provenance and separate executable integration.
- Root: inventory, real-data optimizer proof, corrected comparison runner/scorer,
  workflow notebooks/configs, report and figures.

## Next executable milestone

Run focused affected tests, then an instrumented real-data call to
`train_temporal_model` with at least one optimizer update. Only after this proof,
launch the corrected 600-update comparison in a fresh versioned directory.

Scientific status remains **incomplete / not accepted**. Hypothesis A (history)
and hypothesis B (verified pretrained transfer) remain separate.

## Milestone: real optimizer proof and bounded execution

- `artifacts/temporal_takeover/optimizer_proof_v1/optimizer_proof.json`: BOTH
  native_pair and native_pair_pretext ran one real SA optimizer update and one
  stateful resume update, global steps 1 -> 2. Historical parameter changes
  approximately 2e-4 per update; history optimizer membership exactly once at LR
  2e-4; no change to frozen spatial/backbone tensors or eight physical scalers.
- Save/load and chunk sizes 8 versus 3 had max difference 0 on this GPU; target-free
  model inference matched exactly. This is device-specific evidence, not a
  universal floating-point guarantee.
- Both auxiliary terms produced gradients in the shared historical projection;
  transition produced positive-lead slope gradient; masked reconstruction at zero
  lead correctly did not. Auxiliary heads had nonzero gradients and were saved.
- Inherited pretext smoke directly compared with Phase1: 207 same-shape spatial
  tensors, zero changes. Its 3 microbatches, accumulation16, zero optimizer steps
  are now explained and recorded as invalidated engineering evidence.
- Frozen source snapshot (263 files with SHA256 manifest):
  `artifacts/temporal_takeover/bounded_source_v2_20260915/`.
- Active bounded launch PID **61848**, started 2026-09-15 11:51:38 local;
  output `examples/CORDEX_ML/runs_temporal/experiment_native_pair_v2_20260915`.
  Logs `artifacts/temporal_takeover/bounded_v2_20260915.{stdout,stderr}.log`.
  Original budget: 600 actual optimizer updates per trained variant, 60 validation
  windows, one epoch, test 1981-01-01 through 1983-12-31.
- Review found an orchestration-only assertion bug in that frozen driver: it
  checks trained_temporal_steps for spatial_ft, which legitimately stays zero
  while decoder global_step reaches600. Current source fixed to global_step and
  regression-tested. Model/training results remain valid. A continuation driver
  is being built to wait for original exit, reuse baseline and trained spatial
  checkpoint, and finish remaining variants from THE SAME frozen model sources.
  Do not start a competing GPU process or edit the frozen snapshot.

## Additional completed work

- 53 strengthened native tests; 10 trainer tests; actual NARR strict NetCDF source
  and tiled mosaics tested with synthetic fixtures. NARR real inputs/checkpoint
  still absent; Linux symlink stubs preserved. 32 channels verified (15 weather,
  elevation,16 masks); auxiliary targets use only weather with validity masks.
- 6 corrected scorer regressions. Additive corrected archived scorecards saved
  under snapshot; original results untouched. ConvGRU/Mamba remain not accepted.
- Eight normalization tensors explicitly immutable, including staged unfreezing;
  exact RNG/Adam/data-order resume tested at epoch and optimizer-update boundaries.
- Windows PRISM first-writer concurrency lacked any lock when fcntl unavailable;
  msvcrt lock added. 27 combined trainer/grid/launcher regressions passed after
  supplying installed Git Bash on test-process PATH. Original later launcher
  failure's traceback unavailable, so its historical cause remains unresolved.
- Dedicated runnable notebook:
  `examples/CORDEX_ML/notebooks/Temporal_selected_training_v2.ipynb`.
  Select ONE config/checkpoint and explicit train/resume/infer/evaluate switches.
  Four additive extended YAMLs generated; none launched. Existing notebooks kept.
- Eleven saved-baseline figures generated under
  `artifacts/temporal_takeover/20260915T113310/inherited_baseline_figures`.
- Transfer first audit found wrong time-interval sign during review; v1 retained
  but invalidated. Corrected v2 and matched CPU controls in progress; do not cite
  v1 as valid native time-conditioning evidence.

## Remaining priorities

1. Launch reviewed continuation controller after its CPU checks; record exact PID.
2. Final scoped regression and corrected transfer audit/control evidence.
3. Geographic regional diagnostics and mechanically generated current status/report.
4. Monitor the bounded run only while doing independent useful work; before
   session end record exact live variants/checkpoints and continuation commands.

## Milestone: continuation controller launched and transfer audited

- Reviewed and launched frozen controller copy
  `artifacts/temporal_takeover/continue_bounded_driver_v1.py`, PID **20944**,
  started 2026-09-15 12:07:29 local. It waits on original PID61848 creation
  FILETIME134339718983595352, holding the OS process handle to reject PID reuse.
  It uses no GPU while waiting. Recovery record:
  `examples/CORDEX_ML/runs_temporal/experiment_native_pair_v2_20260915/recovery_execution_20260915T190735Z.json`.
  Logs `artifacts/temporal_takeover/bounded_continuation_v1.{stdout,stderr}.log`.
- Five controller CPU tests passed including simulated spatial_ft global600 /
  temporal0 reuse and exactly four later trained variants. All263 snapshot hashes
  and resolved scientific contracts validated. Only orchestration changes; frozen
  trainer, model, losses, data order and acceptance thresholds remain identical.
- Full1,095-date recomputed baseline is bitwise identical to Claude's saved
  predictions, targets, masks, timestamps, variable order and chunk seams.
  Evidence `baseline_full_period_parity.json`.
- Separate transfer corrected v2 runs ALL completed: pretrained+observed history,
  random+observed history, pretrained+duplicate-current; one actual optimizer
  update each. Same Phase1, adapter initialization, selected dates and parameter
  count; all9 trainable adapter tensors updated; frozen source/base unchanged.
  Positive24h input delta verified against actual upstream SampleSpec.
  v1 negative-time audits remain explicitly invalidated. Scientific benefit still
  unevaluated. Report `docs/temporal_pretrained_transfer_takeover.md` and
  `transfer_engineering_comparison.json`.
- Larger regression:442 passes/6 failures. Five were a real newly introduced
  minimal-config input_vars constructor regression, now fixed; all9 spatial
  alignment and53 native tests passed afterward. The remaining launcher timeout
  is preserved: Bash had exited1 but a stdout reader stayed open beyond10s.
  Six captured repeats passed, so a suspected surviving descendant is NOT
  established as the definitive cause. See `prism_failure_review.md`.
- New regional report preserves coordinates and all valid gridcell-days. Baseline
  southeastern precipitation RMSE8.387 mm/day, predicted wet frequency53.63%
  versus40.70% observed, q99 bias-18.04 mm/day. This identifies a baseline weakness,
  not a claim that the temporal model corrected it.
- Current report/status generated by
  `python examples/CORDEX_ML/cordex_temporal_takeover_summary.py` at
  `docs/temporal_takeover_results.md` and `artifacts/temporal_takeover/latest_status.json`.

### Exact continuation actions

1. Inspect current `latest_status.json`, the original manifest, and the recovery
   record above. If PID61848 or PID20944 is alive with the recorded command line,
   **do not launch another experiment**.
2. The controller reuses completed baseline and spatial_ft training, then runs
   inference and the four remaining variants automatically after the original
   process exits. It records actual updates and validates equal scored dates,
   target/mask digests and output geometry before complete-only acceptance.
3. If the controller is no longer alive and its recovery record reports failure,
   inspect its exact error before resuming. The source command is:
   `python examples/CORDEX_ML/cordex_temporal_continue_bounded.py --snapshot artifacts/temporal_takeover/bounded_source_v2_20260915 --out examples/CORDEX_ML/runs_temporal/experiment_native_pair_v2_20260915`.
   It refuses a competing original process and an existing recovery lock.
   Never remove a lock before verifying its recorded owning PID has exited.
4. After all variants finish, read manifest.scorecard_path and the recovery
   report-view directory; refresh the takeover summary and generate per-variant
   regional figures with `cordex_temporal_regional_report.py`. Do not tune again
   against this same test scorecard and call it untouched confirmatory evidence.
5. NARR scientific execution still requires actual mounted data/scalers/checkpoint.
   Use explicit existing-key `--set` path overrides; do not replace symlink stubs.

## Milestone: final inference, contract and subprocess checks

- Final focused regression after the earlier constructor correction: **240 passed**,
  56 warnings, 49.79 seconds (`postfix_focused_regression.xml`). This preceded the
  later partial-period inference and launcher fixes; their own targeted results
  are recorded separately below rather than claiming this run covered future code.
- Current-source inference now preserves available prior predictors when a requested
  output range starts inside a declared split. True gaps/split boundaries remain
  cold starts. Full bounded SA starts at the actual test boundary, so the frozen
  experiment is unaffected. Additive cold-start time policy defaults to the existing
  nominal convention; an explicit measured-timestamp mode cannot silently load
  an old nominal checkpoint. Both preserved real native/pretext optimizer-proof
  checkpoints reload exactly on CPU, with optimizer state; incompatible time mode
  is rejected. See `final_checkpoint_load_validation.json`.
- Independent comparison audit found no additional active-run blocker. It records
  the same 600 successful updates, 3,000 main target samples when none are skipped,
  and 60 sequential validation windows (300 targets, not the entire 1977-1980
  archive). Spatial_ft trains the decoder in this epoch. Native main paths use
  five backbone calls/window, recurrent and pretext seven; equal calls do not
  establish equal compute. Time-only receives calendar features; native pair's
  24/0 scalars do not encode season. See `final_comparison_contract_review.md/.json`.
- Launcher timeout is now a demonstrated cleanup defect: a controlled shard child
  survived its leader and held stdout open. Original code timed out after10s;
  corrected process-group cleanup completed the identical fixture in2.297s.
  Both NARR/MERRA launchers fixed; entire launcher suite **10 passed**. Exact
  historical child identity remains unavailable. No diagnostic process survived.
- New CPU resume regression interrupts exactly at the maximum update budget,
  before validation. Resume performs zero further updates and matches uninterrupted
  weights, counters, and best loss. The frozen/working trainer text matches modulo
  line endings; no numerical snapshot modification was needed.
- Five report/status tests passed: recovered manifest prediction paths supersede
  stale canonical filenames; missing recorded artifacts fail clearly; a running
  label whose owning PID exited becomes interrupted. Regional reports require a
  new/empty output directory, preserving existing artifacts.

## Final report process

A CPU-only bounded report process was launched, PID **23068**, at
2026-09-15 12:33:00 local. It waits on controller PID20944 with exact creation
FILETIME134339728499133917, for at most12h, and never starts training. Frozen
report sources: `artifacts/temporal_takeover/final_report_source_v1_20260915`.
It generates regional maps/metrics for all six completed variants and refreshes
the summary; incomplete training/evaluation produces a precise failed report
record instead of fabricated figures or acceptance.

- Record/output: `artifacts/temporal_takeover/bounded_final_reports_v1_20260915/`.
- Logs: `artifacts/temporal_takeover/bounded_final_reports_v1.{stdout,stderr}.log`.
- Latest observed durable spatial_ft count was500 at19:29:41UTC; it was still
  training. Re-read the current status instead of treating that lower bound as live.
- Transfer restore review found an additional active-representation identity gap:
  official source SHA alone cannot distinguish a randomized pair from pretrained
  weights. A focused correction is in progress; original one-update evidence is
  retained and no additional GPU work is being launched.


## Milestone: transfer representation identity repaired

- Actual saved v2 adapters could previously restore into a different randomized or
  pretrained representation if only the official source-file SHA matched. A tiny
  fixture demonstrated changed predictions after such a restore.
- New checkpoint schema v3 binds initialization mode, actual complete frozen-pair
  tensor-state SHA256, architecture and time-conditioning semantics. Mismatches
  reject before adapter/optimizer mutation. All23 focused transfer tests passed.
- Three real corrected-v2 checkpoint files remain unchanged and retain their
  one-update engineering evidence. Current strict restore rejects them explicitly:
  they lack the active representation identity now required. No validated legacy
  migration is implemented and no extra training was run for this serialization
  correction. Do not call these existing v2 files production-resumable through v3.
- Evidence: `transfer_restore_identity_bug_reproduction.json`,
  `transfer_restore_identity_validation.json`, and `transfer_tests_v3_identity.log`.
  Native/pretext production resume is independently verified and unaffected.
- Final inference/source test result: **79 passed** in16.99s, including both time
  conventions, near-boundary partial ranges, single dates, true gaps, declared
  split limits, and tiled NARR partial-range equivalence. Both real native/pretext
  proof checkpoints still restore exactly; explicit timing-mode changes reject.


## Final scoped regression and actual spatial accounting

The final corrected25-file CPU regression completed successfully: **493 passed**,
84 warnings,89.60s. It includes the original448-test failure scope plus all new
current temporal tests, all four stochastic-refinement interfaces, both launcher
regressions and checkpoint contracts. Command/environment, full log and JUnit
results are `final_corrected_scoped_regression.command.json`, `.log`, and `.xml`
in the takeover evidence directory. Git Bash was added only to the test-process
PATH; CUDA was hidden for this run. No further full-suite repetition was needed.

Recovered spatial checkpoint accounting is exact:600 successful updates,
600 microbatches/backward passes/accumulation cycles,3,000 target samples,
4,200 backbone calls,zero nonfinite or AMP skips,15,791,365 decoder trainable
parameters at LR5e-5. Its recorded epoch runtime is1824.9463s, train total4.9452095,
validation total4.4424811. Historical-projection and temporal parameters are frozen
in this control. Its original training peak allocated GPU memory was not recorded;
point-in-time GPU observations are not substituted for that missing peak. Later
controller training/inference records include explicit peak allocated memory.

Final working-source snapshot (including inherited modifications, without reset):
`artifacts/temporal_takeover/final_working_source_v1_20260915/`. Its manifest pins
all changed/untracked source files; git HEAD/status/staged/unstaged patches are
saved alongside. The running experiment still uses its own earlier263-file
snapshot, which was rehashed with **zero changed files** after all repairs.


## Latest recorded execution state

Captured 2026-09-15T19:47:05.647514+00:00. Scientific status: **incomplete**.

- baseline: completed; phase=completed; saved updates=0.
- spatial_ft: completed; phase=completed; saved updates=600.
- time_only: running; phase=training; saved updates=0.
- native_pair: not started; phase=None; saved updates=None.
- native_pair_nohistory: not started; phase=None; saved updates=None.
- native_pair_pretext: not started; phase=None; saved updates=None.

Exact active commands/creation times and output paths are preserved in
`artifacts/temporal_takeover/20260915T113310/session_handoff_execution.json`.
The controller remains the sole GPU experiment; the report process remains
CPU-only. No completed scientific comparison or native/pretext acceptance is
claimed at this checkpoint. The next executable action is to inspect the same
controller and its manifest, without launching a duplicate; if it has failed,
read its saved traceback and validate checkpoint compatibility before resuming.


## Completed spatial-control evaluation and final reporting correction

At19:47:05UTC, baseline and spatial_ft were completed; time_only was actively
training under PID20944; native_pair, native_pair_nohistory and native_pair_pretext
had not started. All compared date/target/mask/geometry digests matched.

The600-update spatial control worsened full-domain precipitation RMSE from
8.088363 to8.179560mm/day and temperature RMSE from1.552598 to1.571643K. Its
precipitation autocorrelation errors improved, but tendencies and accumulations
worsened. Temperature accumulation improved, while other primary and several
extreme/spatial checks failed. This is a control result, not a candidate verdict;
no test-driven hyperparameter changes were made. See
`spatial_control_completed_comparison.json` and
`bounded_partial_scorecard_20260915T1947.json` (complete=false, four evaluations
missing, original thresholds intact).

Additional rescoring initially searched canonical evaluation.json only. It now
uses the exact manifest-selected recovery evaluation path and never falls back
to a stale canonical file if that recorded path is missing. Seven affected
scorecard tests passed after this reporting-only repair. This occurred after the
493-test final scoped run; no training/model source changed. Final source snapshot
v2 supplements the retained v1 snapshot with this final repair and current docs.


## Completed spatial regional evidence and final handoff

`artifacts/temporal_takeover/20260915T113310/regional_spatial_ft/` contains
regional_metrics.json, comparison_to_baseline.json/.md, both physical-coordinate
maps and numeric bias fields. All30 saved-array pairing checks passed. Precipitation
RMSE increased1.13% full domain and1.62% southeast. Temperature RMSE increased
5.90% at boundaries while decreasing3.34% southeast. Southeastern precipitation
wet frequency moved53.63%->45.38% versus40.70% observed, but q99bias worsened
-18.04->-19.83mm/day. The saved report presents both improvements and failures;
this control has no candidate acceptance verdict.

Canonical final working source snapshot:
`artifacts/temporal_takeover/final_working_source_v2_20260915/` (v1 retained).
Training still uses the separately preserved bounded_source_v2_20260915 snapshot;
source/evaluation fixes were not silently injected into that scientific run.

Final status capture: 2026-09-15T19:53:09.785290+00:00; see
`artifacts/temporal_takeover/20260915T113310/session_handoff_execution_v2.json`.
- baseline: completed, completed; durable updates=0.
- spatial_ft: completed, completed; durable updates=600.
- time_only: running, training; durable updates=100.
- native_pair: not started, None; durable updates=None.
- native_pair_nohistory: not started, None; durable updates=None.
- native_pair_pretext: not started, None; durable updates=None.

GPU controllerPID20944 and CPU report waiterPID23068 were alive at this capture.
No scientific acceptance is established while the planned comparison is incomplete.
No files were pushed and inherited outputs/scorecards remain preserved.


## Follow-up: temporal notebook selection and YAML cleanup

User requested whether the spatial SA fine-tuning notebook can use recurrent/Mamba
YAMLs and removal of unnecessary generated test YAMLs. Its legacy frame trainer
has no temporal dispatch; merely changing P.config_path would not train temporal
adapters, sequence losses or state. The original notebook now rejects enabled
temporal configs before GPU selection, dependency installation and checkpoint
cleanup, with a link to the dedicated workflow. Its spatial/FSDP path is preserved.

`Temporal_selected_training_v2.ipynb` now offers recurrent, Mamba, native_pair and
native_pair_pretext for both SA and NARR. It invokes the actual temporal CLI from
the repository root in one process. Defaults honor YAML epochs, accumulation and
complete splits; optional update/validation caps default to None. Train/resume/
infer/evaluate switches remain false. Resume epochs mean the total desired count.

Fourteen obsolete v1/duplicate standalone transfer-control YAMLs were removed from
active example folders after verified archival. One current transfer YAML per
case remains. `--initialization` and `--history-mode` reproduce the controls and
are recorded in effective checkpoint/report configs. Existing native extended
training schedules, recurrent/Mamba configs, scientific recipes and run records
remain. The notebook generator creates YAMLs only when explicitly asked for
missing optional extended configs, and never overwrites existing schedules.

Archive/hash inventory: `artifacts/config_cleanup_20260915/cleanup_manifest.json`.
Validation:35 workflow/config tests plus30 transfer tests passed; the original
notebook guard was exercised using actual spatial/recurrent/Mamba YAMLs; all31
code cells across both notebooks compile. The263-file active scientific snapshot
still matches its manifest. PIDs20944 and23068 remained alive; no new GPU work or
training was launched by this cleanup. See cleanup_validation.json in that archive.


## Notebook progress and repeated warnings (2026-09-15)

The selected temporal notebook now forwards unbuffered subprocess output and displays one tqdm bar per epoch, with microbatch progress, optimizer update count, running loss, validation status and final validation loss. Identical warning category/message pairs appear once across notebook commands; predictand notices now use Python warnings. The trainer emits optional progress events without changing the numerical or checkpoint protocol. User settings in the notebook were preserved, including the saved `RUN_TRAIN=True`. Reopen the notebook and rerun the setup/helper cells before the next train/resume invocation.

Validation: 44 targeted CPU tests passed, including exact optimizer/RNG resume regressions. A real Prithvi Jupyter kernel produced exactly two bars for two synthetic epochs, eight in-place display updates, and one repeated-warning output; that synthetic kernel was shut down. Evidence and pre-edit backups: `artifacts/temporal_progress_20260915/`. No real training was launched for this UI change; existing frozen training sources and processes were not modified.
