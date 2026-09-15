# Authenticated scientific evidence collection

The collector is implemented in examples/CORDEX_ML/cordex_scientific_evidence.py. It evaluates retained experimental HDF5 predictions without fitting statistics, training a model, sampling new members, or accessing quarantined target files. Each output directory must be new or empty. Existing predictions, statistics and diagnostic products are preserved.

The scientific contract remains version 2, SHA256 f649faeb83c7a5722cd975e5df8cf6effa6145d3d1822f4a410bb9c9a7245b30. The explicit pre-experiment amendment and unchanged version-1 snapshot remain in the run directory. This collector does not alter any acceptance threshold, region, year, seed, or stopping rule.

## Authentication and scope

Before computing results, collection verifies the complete immutable Phase-1 cache and checkpoint hashes, split manifest, complete prediction hash, strict restored stochastic candidate state, target-normalizer fingerprint, variable order, physical units, experiment/acceptance contracts, implementation fingerprints and resolved sampling settings. It then compares the actual baseline, target, validity-mask, coordinate and date values against the common cache. Physical member values must be finite on the paired valid support; stored ensemble means must agree with averages of the physical members within their documented floating-point summation tolerance. All metrics recompute the means in float64.

Full component scope additionally requires a checkpoint from the registered full stage, all registered fitting dates and the plan-defined days-7/21 validation-selection subset, an authenticated completed stopping-rule receipt, complete fitting calendar years in the source cache, and every validation date. Merely sampling a screening checkpoint on a larger date range does not satisfy these requirements. The final execution_status.json must match last.ckpt progress and the selected checkpoint must match scientific_selection.json. The minimum epoch rule applies to the completed run; an earlier selected best checkpoint remains legitimate, and selected_checkpoint_epoch is reported separately from completed run epochs and updates. The 192 selection dates are used only for provisional checkpoint ranking; full component acceptance still requires all 2920 validation dates. Single-variable controls remain incomplete for the joint two-variable family. The collector refuses fitting, historical-development and quarantined target dates as acceptance evidence.

Mean-remainder and cross-fit-remainder candidates use the postprocessed deterministic point prediction as their CRPS control. The signed baseline-plus-mean value is retained separately as mean_control_unbounded. Both are replayed using the strictly restored frozen mean and authenticated saved conditioning; Phase 1 is not executed. The replay tolerance is 1e-5 absolute plus 1e-6 relative to accommodate CPU/GPU convolution roundoff. Cached baseline, target, coordinates and masks require exact equality. Scientific metric thresholds do not use this replay tolerance.

The fitting-source NetCDF headers were checked without reading target values: D:/CORDEX/SA_domain/train/Emulator_hist_future/target/pr_tasmax_ACCESS-CM2_1961-1980_2080-2099.nc has 14600 dates, a 128 by 128 grid, pr units mm/day and tasmax units K. No extra conversion is applied to these already-physical fields. This check does not apply to the earlier historical-development source, whose precipitation rate units differ.

## Diagnostics and inference

The canonical evaluate_variable implementation provides climatology, daily errors, pattern agreement, seasonal diagnostics, empirical CRPS, spread/error, finite-ensemble interval coverage, spatial coherence, distance-from-boundary diagnostics and numerical maps. Registered geographic masks and canonical sufficient-statistic reducers supply all widths 1, 2, 4, 8 and 16, each edge, corner, interior, full domain, historical/future regime and southeastern precipitation box. New predictions are not smoothed or cropped for evaluation.

Additional diagnostics compute exact quantiles for every retained physical member, exact regional ensemble-mean quantiles, observed-q99 event errors, raw/postprocessed CRPS and variance, signed corrections and changes from physical constraints. Precipitation uses the configured 0.01 mm/day wet threshold as primary and labels 1 mm/day as a sensitivity check. Wet-day Brier scores and ten-bin reliability tables retain valid dry-day zeros. Negative-value fractions and magnitudes use the full valid member-observation denominator. Temperature wet-day diagnostics are explicitly inapplicable.

Whole-year exact extreme draws use YearExtremeStatistics, linear pooled quantiles, and inclusive truth >= resampled truth q99 events. Independent training-seed metrics are averaged before taking relative changes, using the same paired year/seed resampling plan as the core gates. Ensembles are retained intact; members are not resampled or pooled across trained models.

The nested-member option requires independently issued 10-, 20- and 50-member diagnostic HDF5 files for the same checkpoint, settings, seed stream, variable order and registered first/middle dates per season and climate regime. Raw prefixes must agree exactly. A whole-ensemble mean-preserving physical projection can change the processed prefix as ensemble size changes; those effects are reported. Missing nested products stay missing.

Collection writes per-variable year-statistic NPZ files, raw-statistic NPZ files, exact extreme-draw JSON, numerical maps, readable metric summaries, and a per-seed evidence.json. All retained diagnostic products are hashed. Merge verifies these hashes and the retained physical prediction files again, reports independent training-seed metrics, and produces comparable maps using shared scales across submitted heads and seeds.

Missing required dates, diagnostics, heads or independently trained seeds prevent complete-family acceptance. Eight year blocks and three training seeds limit uncertainty resolution. Finite-ensemble order-statistic coverage is descriptive and does not establish population calibration. End-to-end production promotion remains blocked by the registered independence limitation even if component validation eventually passes.

## Commands prepared for completed predictions

Run from D:/granite-wxc. Set the prediction and optional nested paths to the actual completed products emitted by cordex_scientific_experiments.py. They may reside in the separate C: artifact directory; the common cache remains on D:.

PowerShell:

    $contract = "D:/granite-wxc/artifacts/refinement_validation/scientific_acceptance_20260909T230807Z/frozen_scientific_acceptance_v2.json"
    $cache = "D:/granite-wxc/artifacts/refinement_validation/scientific_acceptance_20260909T230807Z/cache/phase1.h5"
    $prediction = "<completed full-validation prediction.h5>"
    $output = "<new per-head per-training-seed evidence directory>"
    mamba run -n Prithvi python examples/CORDEX_ML/cordex_scientific_evidence.py collect --prediction $prediction --cache $cache --contract $contract --output $output --chunk-time 16

To include nested diagnostics, append:

    --nested "<10-member diagnostic.h5>" "<20-member diagnostic.h5>" "<50-member diagnostic.h5>"

To collect all four heads and all three registered training seeds, repeat collection once for each completed head/seed artifact. Each invocation processes one variable at a time. With 2920 days, 20 members and 128 by 128 cells, each float32 member array for one variable is approximately 3.83 GB; raw and processed arrays are both retained in memory while that variable is evaluated. Diagnostic output is separate from prediction storage.

Merge all twelve resulting evidence.json paths into a new directory. Do not mix competing recipes for the same head/seed:

    $evidenceFiles = @("<head1-seed101/evidence.json>", "<head1-seed202/evidence.json>", "<head1-seed303/evidence.json>", "<remaining nine evidence.json paths>")
    mamba run -n Prithvi python examples/CORDEX_ML/cordex_scientific_evidence.py merge --evidence $evidenceFiles --contract $contract --output "<new combined scientific report directory>"

The canonical report is acceptance_report/scientific_acceptance_results.json with its CSV and SCIENTIFIC_ACCEPTANCE.md beside it. The merge also writes evidence.json, independent_training_seed_results.json, shared_plot_limits.json and comparable_maps/. Supplying --no-plots retains numerical maps and metrics while omitting raster rendering. Incomplete submitted evidence produces an INCONCLUSIVE report; it does not certify a smaller statistical family.

## Validation actually executed

The commands above are prepared interfaces, not claims that new full CORDEX predictions have been evaluated. Synthetic CPU artifacts exercise real native sampling, strict checkpoint loading and the HDF5 schema. Tests cover altered baseline/target/mask values even after rehashing, nonfinite members, incorrect physical means, immutable diagnostics, both frozen-mean replay paths, exact quantiles and extreme year draws, paired training-seed reduction, nested 10/20/50 raw prefixes, valid-zero wet diagnostics and incomplete-scope rejection.

The combined regression run passed 101 tests in 34.87 seconds:

    mamba run -n Prithvi python -m pytest tests/test_cordex_scientific_evidence.py tests/test_scientific_acceptance.py tests/test_scientific_extremes.py tests/test_refinement_regional_validation.py tests/test_cordex_refinement_evaluation.py tests/test_refinement_diagnostics.py tests/test_refinement_io.py -q --tb=short --junitxml=artifacts/refinement_validation/scientific_acceptance_20260909T230807Z/scientific_evidence_tests.xml

The final isolated collector run includes the subsequent diagnostic-file tamper check and parameterized cross-fit replay, plus registered ranking-subset/full-acceptance separation and earlier-best/completed-run receipt checks, with results in evidence_collector_final_tests.xml:

    mamba run -n Prithvi python -m pytest tests/test_cordex_scientific_evidence.py -q --tb=short --junitxml=artifacts/refinement_validation/scientific_acceptance_20260909T230807Z/evidence_collector_final_tests.xml

The preregistered core simultaneous-bootstrap benchmark remains 37.36 seconds for 1999 draws over four heads, two variables, three seeds and a complete synthetic 128 by 128 domain; maximum difference from the canonical reference was 2.05e-13. This is a core bootstrap timing, not a runtime estimate for full HDF5 I/O, exact distribution diagnostics or map rendering. Those additional operations are deliberately retained.
