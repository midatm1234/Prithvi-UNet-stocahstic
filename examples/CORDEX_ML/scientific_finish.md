# Persistent registered completion controller

Implemented entry point: examples/CORDEX_ML/cordex_scientific_finish.py.

The controller has been tested with synthetic execution receipts and real CPU evidence-collector fixtures. It was not launched by the implementing agent. Existing original-development and supplemental workers remain under their own controllers.

From D:/granite-wxc, start or resume the same completion service with:

    mamba run -n Prithvi python examples/CORDEX_ML/cordex_scientific_finish.py --development-queue artifacts/refinement_validation/scientific_acceptance_20260909T230807Z/workflow/development_queue_artifacts.json --supplemental-queue artifacts/refinement_validation/scientific_acceptance_20260909T230807Z/workflow/supplemental_queue.json --poll-seconds 15 --minimum-free-gib 30

The current absolute Prithvi Python interpreter executes child script argument lists. Reviewable receipts retain the corresponding mamba commands. Subprocesses use argument lists and hidden Windows execution. No shell-generated commands delete or move outputs.

Optional --development-pid and --development-created arguments bind monitoring to a known original process and its psutil creation timestamp. This is a read-only check; it never terminates or relaunches original development. All supplied monitoring, supplemental and resource flags are preserved in the saved resume command.

The controller keeps one operating-system process lock throughout persistent waits. It reads the existing original development queue until all original outcomes are terminal. It then waits for or runs the separately registered supplemental queue, combines the immutable original and supplemental outcomes, and calls prepare_full on that combined selection evidence. A process scan recognizes --queue, --supplemental-queue, --output and --training-output, including an independently running supplemental parent between child jobs.

Full training uses the existing run_queue API and original registered stopping rules. The completed queue receipt and every required stochastic job are checked; process exit zero alone is insufficient. Common full-seed mean routing and any required fold means are handled by the existing frozen routing registration and queue. No training rule, batch size, sampler step count or acceptance threshold is changed on a resource pause.

The downstream sequence is:

- 96 normal inference commands: one full 20-member validation product and nested 10/20/50-member diagnostic products for each of 24 selected/control head/seed cases.
- 24 authenticated evidence collections.
- Separate selected and control family merges, each with all four heads and all three independently trained seeds.
- One matched-map comparison using shared_map_limits over all 24 source cases. A single scale set applies across both groups, heads and seeds; scientific acceptance gates and families remain separate.

The ordinary shared-map output is under the large artifact root at workflow/component_acceptance/matched_selected_control_maps/{selected|control}/{head}/{seed}. If rendering is interrupted before completion, the unfinished directory is preserved and a new explicitly numbered sibling attempt is used. Actual final paths are always recorded in finish_execution.json.

The independent rendering command is:

    mamba run -n Prithvi python examples/CORDEX_ML/cordex_scientific_evidence.py compare-families --selected "<selected merged evidence.json>" --control "<control merged evidence.json>" --contract artifacts/refinement_validation/scientific_acceptance_20260909T230807Z/frozen_scientific_acceptance_v2.json --output "<new matched_selected_control_maps directory>"

## Preservation and resume behavior

The metadata receipt is workflow/finish_execution.json beneath the original run root. It includes checkpoint/configuration hashes, selected-checkpoint identities, sampler seeds, scope, executed arguments, logs, attempts, artifact hashes, bundle hashes, current scientific result paths and the exact resume command. workflow/finish_planned_commands.json contains separate, unexecuted command plans; it is not rewritten to imply execution.

Completed predictions require matching selected-checkpoint, cache, contract, plan, member, variable, seed, stream and date provenance. Their bytes must match the completed manifest. Partial predictions retain their recorded state and are resumed through the normal entry point with --resume; the native runner must verify the exact opaque model/target/conditioning/random-stream identity before continuing. Incompatible or unversioned files are preserved and rejected.

Collectors and mergers write completion.json only after every required output and requested plot is finished. On restart, a completed orphan child can be recovered from that marker and authenticated bundle. Incomplete evaluation directories are preserved in separate attempts. Current input fingerprints are checked before reuse, and changes or deletion of referenced statistics, diagnostics, checkpoints or source artifacts prevent a completed-controller claim.

A STOP file at the original run root's workflow/STOP prevents new operations. An already active operation is allowed to finish; the controller does not forcibly terminate training, inference or plotting. The persistent service waits while STOP remains and resumes after its removal. --once instead returns code 75 when a prerequisite is pending. This is an inspection/wait option, not a dry-run option: if all prerequisites are ready it executes authorized remaining work.

Both the D: metadata/cache volume and C: prediction volume retain at least the configured 30 GiB reserve. Explicit exit 75, allocation failures and disk-full failures are recorded as resumable pauses. Default persistent operation rechecks unchanged conditions and arguments. Other execution or provenance errors stop with code 2 and retain an exact resume command. No arbitrary wall-clock stop or silent numerical fallback is introduced.

SCIENTIFIC_EVALUATION_COMPLETED means the registered execution and reporting sequence finished. It does not mean any head passed the scientific contract. Each selected/control report retains its actual PASS, FAIL or INCONCLUSIVE result. Production promotion is always false, and members from independently trained models are not pooled into a larger ensemble.

## Scope and executed validation

Full component evaluation uses 1977-1980 and 2096-2099. Regeneration of the full 1981-2000 development period is not scheduled or supported by this experimental cache/CLI and remains UNEXECUTED. The 2041-2060 target values remain quarantined and are never opened by this controller.

The final focused run passed 48 tests in 86.34 seconds:

    mamba run -n Prithvi python -m pytest tests/test_scientific_finish.py tests/test_cordex_scientific_evidence.py tests/test_scientific_supplemental.py -q --tb=short --junitxml=artifacts/refinement_validation/scientific_acceptance_20260909T230807Z/finish_controller_tests.xml

These checks cover queue coordination, the operating-system lock, STOP and both volume floors, exact partial identity checks, exit-zero-without-artifact rejection, all downstream operations, separate scientific families, complete orphan recovery, changed bundle dependencies, current input drift, resource error classes, resume flags, supplemental ordering, earlier-best/completed-run scope, direct and cross-fit mean replay, nested-to-primary raw-member equality, exact extreme resampling and identical plotting scales across the selected/control families. The orchestration fixtures generate no real climate training or accepted scientific results.


## Supplemental paired attribution continuation

A separately registered selected-versus-fresh-control comparison now follows both family merges and the common-scale maps. Before full training starts, the controller verifies fresh_control_contrast_registration.json against the parent experiment plan and frozen acceptance contract and binds its file hash in the execution receipt. A changed registration is rejected on resume. The separate acceptance families and their thresholds remain unchanged.

The new operation is attribution|selected_control. Its output is workflow/component_acceptance/selected_fresh_control_attribution/attempt_N under the large artifact directory, with the actual path and bundle hashes recorded as fresh_control_attribution in finish_execution.json. Interrupted output attempts are preserved, and completed orphan operations use the same authenticated recovery as collections and merges. COMPLETE means that the supplemental calculation finished; it does not mean scientific acceptance or production promotion. Its paired year/training-seed intervals are conditional on saved ensembles and the selected component-validation procedure.

The independent reproduction command is:

    mamba run -n Prithvi python examples/CORDEX_ML/cordex_scientific_attribution.py compare --selected "<selected merged evidence.json>" --control "<control merged evidence.json>" --contract artifacts/refinement_validation/scientific_acceptance_20260909T230807Z/frozen_scientific_acceptance_v2.json --registration artifacts/refinement_validation/scientific_acceptance_20260909T230807Z/fresh_control_contrast_registration.json --output "<new supplemental attribution directory>"

The live report renderer links the actual selected and control metric CSVs, all eight combinations' physical metrics and relative-change intervals, separate three-training-seed metrics, matched maps and supplemental attribution results. Completed family merge results can be reported while later plotting or attribution remains pending; controller completion is displayed separately.

The added continuation regressions passed 17 tests in 54.11 seconds (finish_attribution_controller_tests.xml). The final incomplete/completed report fixtures passed 9 tests in 6.84 seconds (execution_report_tests.xml). These overlap earlier tests and must not be summed as distinct repository coverage. No real climate training or final scientific comparison was executed by these tests. An already-running controller retains its previously loaded implementation until explicitly restarted; its actual restart receipt determines which code is active.
