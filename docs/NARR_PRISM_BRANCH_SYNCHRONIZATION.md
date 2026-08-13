# NARR–PRISM branch-content audit and synchronization

This repository preserves deterministic Phase 1 and stochastic Phase 2 work as
one additive history. A branch with the newest model code is not necessarily a
superset of every useful notebook, configuration, launcher, test, or document
on an older workflow branch. Synchronization therefore requires a tree-content
audit and explicit reconciliation; comparing only commit dates is insufficient.

## Read-only content audit

Run the audit from the repository root with the existing `Prithvi`
environment:

```bash
mamba run -n Prithvi python scripts/audit_repository_content.py \
  --json-output /tmp/narr-prism-content-audit.json
```

By default, the script compares `HEAD` with locally available branch and
remote-tracking refs whose names indicate NARR, PRISM, CORDEX, deterministic,
UNet, diffusion, flow matching, or stochastic refinement work. It also includes
the default branch. Discovery uses only the local Git object database: the
script never fetches, checks out, merges, stages, commits, resets, cleans, or
updates a ref.

Pass refs explicitly when provenance is known or when an important branch name
does not match the discovery terms:

```bash
mamba run -n Prithvi python scripts/audit_repository_content.py \
  --no-discover \
  --source main \
  --source NARR_PRISM \
  --source origin/NARR_PRISM \
  --source origin/Prithvi-UNet-stochastic_refinement \
  --source origin/CORDEX_ML_diffusion_head \
  --source origin/MERRA_PRISM \
  --source origin/CORDEX_ML \
  --source origin/test_on_aws \
  --source origin/Merra2-rollout \
  --json-output /tmp/narr-prism-explicit-audit.json
```

`--source` accepts any locally resolvable ref or commit. If a requested local
branch is absent but a matching remote-tracking ref exists, that ref is used and
identified in the report. An unavailable ref is reported rather than silently
ignored. Remote-tracking refs are snapshots from the last fetch, not proof of
the current remote state.

Use `--path-prefix examples/NARR_PRISM` to inspect one subtree. Repeat the
option to inspect several subtrees. `--format json` emits JSON on standard
output, while `--json-output PATH` writes the complete JSON report in addition
to the readable terminal report. JSON is never truncated. `--max-paths` only
limits terminal display. `--fail-on-differences` returns status 1 when any
compared trees differ or a requested ref is unavailable, which is useful for a
purpose-built CI policy but is too strict for branches expected to diverge.

The report separates:

- `missing_from_current` / `present_only_in_other`: tracked by a source ref but
  absent from the current committed tree;
- `unexpectedly_deleted`: a conservative subset that existed at the merge base
  and remains on the source ref but is absent from current;
- `current_only`: present only in the current committed tree;
- `renames`: Git similarity-based rename or copy candidates;
- `modified_differently`: paths present in both trees with different blob or
  file-mode identities;
- `likely_generated_artifacts`: explainable path/suffix heuristics among the
  differences, such as checkpoints, NetCDF output, caches, or log files.

These are review candidates, not automatic decisions. In particular, an
“unexpected delete” may be intentional, and a generated-looking fixture may be
small, intentional test data. The report records uncommitted working-tree
changes for context but comparisons use committed trees so transient edits
cannot masquerade as preserved branch content.

## Safe additive synchronization procedure

1. Start from a clean understanding of local state. Do not discard edits:

   ```bash
   git status --short
   git branch --show-current
   git log -1 --oneline --decorate
   ```

2. Before any content change, create a recoverable backup ref at the exact
   starting commit. Choose a unique descriptive name:

   ```bash
   git branch backup/narr-prism-sync-YYYYMMDD HEAD
   ```

   If the working tree has valuable uncommitted changes, commit them on an
   appropriately named safety branch before synchronization. Do not hide them
   with a destructive reset or cleanup.

3. If current remote state is required, fetch it explicitly before the audit
   and record that action. A fetch updates remote-tracking refs but does not
   alter the working tree. Do not use pruning as part of a preservation audit.

4. Run both the discovered and provenance-specific audits. Keep the JSON report
   with the work record. Inspect ancestry as well as tree content:

   ```bash
   git merge-base HEAD SOURCE_REF
   git log --left-right --cherry-pick --oneline HEAD...SOURCE_REF
   git diff --find-renames --summary HEAD SOURCE_REF
   git diff --find-renames --name-status HEAD SOURCE_REF
   ```

5. Classify every source-only path before restoring it:

   - source/configuration, notebook, test, script, environment, or documentation;
   - a compatible older implementation already superseded in current;
   - a genuinely branch-specific asset that needs a clearly named directory;
   - generated data, checkpoint, cache, plot, or inference output that belongs
     in external artifact storage rather than Git.

6. Restore genuinely missing source-only paths additively and in small groups.
   `git restore --source SOURCE_REF -- path` is appropriate only after verifying
   that the path is absent in current. Review and commit each logical group.
   Never use a repository-wide checkout to solve a file-level omission.

7. For a path modified on both sides, inspect all three versions before editing:

   ```bash
   git show MERGE_BASE:path/to/file
   git show HEAD:path/to/file
   git show SOURCE_REF:path/to/file
   ```

   Retain the newest compatible implementation and manually incorporate useful
   behavior from the other ref. Preserve existing YAML keys and defaults when
   possible. Put irreconcilable workflow-specific launchers or examples in a
   clearly named subdirectory rather than overwriting either workflow.

8. Treat rename candidates as hypotheses. Confirm imports, documentation links,
   CLI references, and Git history before adding an alias or moving anything.
   Do not delete, rename, or move an existing path merely to match another ref.

9. Validate the combined tree in `Prithvi`. At minimum, parse affected YAML and
   notebooks, run focused tests for restored workflows, then run the repository
   test suite. Rerun the audit against every source ref and explain each
   remaining source-only or differently modified path.

10. Commit the additive reconciliation with its audit provenance. Push normally
    only after confirming the intended destination and remote divergence.

Never use `git reset --hard`, `git clean -fd`, a force-push, or history
rewriting for branch synchronization. Never replace a newer working module with
an older whole-file version merely because the older branch has additional
assets. Generated checkpoints, inference data, and large scientific datasets
must be preserved in the configured artifact location, not reintroduced into
Git.

## Review checklist

- Backup branch points to the exact pre-change commit.
- Local and remote NARR–PRISM refs were compared independently when their
  commits differ.
- Default/original and deterministic Phase 1 refs were inspected.
- Diffusion, diffusion Transformer, flow-matching, and flow-matching Transformer
  refs were inspected.
- Source-only scripts, notebooks, YAML, tests, and documentation were either
  restored or explicitly explained.
- Differently modified files were reconciled, not blindly replaced.
- Generated artifacts were kept outside Git with paths and checksums documented
  where required.
- Focused and full tests ran in `Prithvi`.
- A final JSON audit was retained with the synchronization record.

## 2026-08-13 reconciliation record

The preservation audit that produced this document started from commit
`1ce98bd` and created the local recovery ref
`backup/narr-prism-content-audit-20260812-1ce98bd` before editing anything.
Remote-tracking refs were refreshed without pruning. The relevant histories
and their disposition were:

| Ref | Result |
|---|---|
| `NARR_PRISM` at `c218d85` | Ancestor of current; no branch-only path |
| `origin/NARR_PRISM` at `2c3ce4f` | Ancestor of current; no branch-only path |
| `origin/Prithvi-UNet-stochastic_refinement` at `de715f1` | Ancestor of current; no branch-only path |
| `main` / `origin/main` at `a6c4f31` | Original ECCC implementation had been renamed and substantially extended as `cordex_finetune_model.py`; the original import path is retained as a compatibility alias |
| `origin/MERRA_PRISM` at `73c6ed8` | No branch-only path; its `last.ckpt`-before-`best.ckpt` discovery behavior was manually reapplied to the newer MERRA inference implementation |
| `origin/CORDEX_ML_diffusion_head` at `eb9d005` | Divergent legacy diffusion implementation; 35 source/config/notebook/test assets restored additively, while newer same-path implementations were retained |
| `origin/CORDEX_ML` at `1259561` | Ancestor of current; no branch-only path |
| `origin/test_on_aws` at `989620a` | Ancestor of current except the original ECCC model path, which is preserved by a compatibility module pointing at the newer CORDEX implementation |
| `origin/Merra2-rollout` at `74e97b7` | Nine branch-only ozone/chemistry rollout examples; inspected and retained on that branch because they are unrelated to NARR–PRISM, deterministic downscaling, or stochastic refinement |

The CORDEX diffusion branch also contains 52 generated evaluation, scalar,
diagnostic, PNG, and NPZ artifacts (about 49 MB). They remain recoverable from
`origin/CORDEX_ML_diffusion_head@eb9d005` and are intentionally not copied into
the active source tree. The audit reports them as generated artifacts rather
than silently treating them as lost source. The unrelated
`origin/Merra2-rollout` chemistry example was inspected but is outside the
NARR–PRISM/deterministic/refinement lineage and was not imported.

The restored legacy CORDEX modules coexist with the newer
`granitewxc.refinement` package. Compatibility seams were ported into the
newest shared CORDEX files rather than replacing those files with older branch
versions. The generic refinement package and the four current NARR–PRISM
refinement YAMLs remain authoritative for new work.

The same-path comparison against `origin/CORDEX_ML_diffusion_head` was also
resolved explicitly rather than being dismissed as ordinary branch drift:

- The branch's reusable preprocessing fallbacks, case/run-path helpers,
  headless diagnostics, optional plotting imports, explicit ensemble-mean
  postprocessing, calendar-pairing contract, and `case_name` run naming were
  retained. These are compatible extensions to the common deterministic path.
- The legacy diffusion training/model/checkpoint behavior was ported into the
  current shared CORDEX modules and is covered by restored unit and production
  pipeline tests. Its full mathematical and operational contract is preserved
  in [the CORDEX residual-diffusion audit](../examples/CORDEX_ML/DIFFUSION_RESIDUAL_CORRECTION.md).
- `SA_downscaling_inference_T2_ACCESS-CM2_static.py` is the reconciled
  production inference entry point: it supports deterministic output, the
  legacy diffusion ensemble, and the newer shared refinement package while
  preserving boundary, seed, checkpoint, and NetCDF provenance checks.
- The other eleven domain/task inference exports remain their latest
  deterministic launchers. The divergent branch duplicated its legacy
  score-head dispatch into every export; copying that large implementation
  again would create eleven independent code paths and would not add a new
  asset. Their reusable diffusion pieces are preserved in
  `examples/CORDEX_ML/utils/diffusion_inference.py`, and the reconciled SA T2
  static entry point is the maintained stochastic front end. The eleven files
  remain useful and tracked for their domain-specific deterministic scenarios.
- Notebook-output-only changes and branch-generated evaluation/scalar/plot
  artifacts were not copied. Those outputs remain recoverable at the recorded
  source ref and are classified as generated by the audit report.

This disposition means a `modified_differently` result is not by itself an
omission: it may identify a newer implementation, a deliberately preserved
domain-specific launcher, or generated notebook state. Future synchronizations
must repeat the three-way behavioral review instead of forcing blob equality.

The audit also found that MERRA–PRISM still tracked the same absolute
self-referential artifact links that caused the NARR–PRISM checkout deletion.
Those three links now use the relative `artifacts/...` portal contract as well,
and the local portal resolves outside the Git worktree. A regression test
rejects self-links so a future synchronization cannot silently reintroduce the
destructive topology.
