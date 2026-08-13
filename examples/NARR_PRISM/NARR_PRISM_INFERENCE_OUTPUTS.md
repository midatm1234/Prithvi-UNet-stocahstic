# NARR–PRISM deterministic inference outputs

This document records where deterministic output from
`NARR_PRISM_subdomain.yaml` belongs, what happened to the historical output,
and how to reproduce and validate it without confusing a partial run with a
complete 2016–2025 product.

## Output contract

For the current configuration, deterministic inference covers every calendar
day from 2016-01-01 through 2025-12-31, inclusively: 3,653 daily files. The
configured case is `narr_prism_California`, so the output contract is:

```text
<inference.output_dir>/narr_prism_California/
  narr_prism_California_inference_20160101.nc
  ...
  narr_prism_California_inference_20251231.nc
```

The repository path is
`examples/NARR_PRISM/experiments/inference_output/narr_prism_California`.
The repository `experiments` entry is an artifact portal, so its resolved
storage location may be outside the Git worktree. Use the audit command below
instead of assuming that either spelling proves the product is complete.

Each daily file must contain one time sample on the ordered PRISM latitude and
longitude grid, the configured target variables in the explicit order
`ppt`, `tmax`, `tmin`, and the training-derived `prism_valid_mask`. Required
attributes bind the file to its case, split, inclusive date interval,
checkpoint, grid fingerprint, and mask provenance.

## Historical output deletion (2026-08-12)

The original daily files are no longer accessible. This was a repository
checkout incident, not an inference failure or an external mount outage.
Read-only ext4 journal and inode inspection established the following:

- Journal directory entries contain all 3,653 unique expected daily names,
  from `narr_prism_California_inference_20160101.nc` through
  `narr_prism_California_inference_20251231.nc`, at the historical path
  `/data/granite-wxc/examples/NARR_PRISM/experiments/inference_output/narr_prism_California`.
- The final files' creation-time window was 2026-08-11 22:34:08 through
  2026-08-12 08:37:15 UTC.
- Their deletion-time window was 2026-08-12 17:39:58 through 17:40:00 UTC.
- Git reflog records checkout from `NARR_PRISM` to
  `Prithvi-UNet-stochastic_refinement` at 2026-08-12 17:40:54 UTC. That branch
  was then at commit `c538165`. The commit tracked `experiments`,
  `preprocessed`, and `scalars_with_H` as absolute links back to those same
  worktree paths. Checkout replaced the real directories seconds after the
  file deletions.
- No surviving copy of the historical daily NetCDF data was found in the
  repository, artifact portal, alternate worktree, or other searched local
  paths. Journal names and inode metadata prove the complete set existed, but
  they cannot restore overwritten file data.

The recovered Phase-1 `last.ckpt` used for regeneration has SHA-256:

```text
1d352491ab32a4069696673f504edcd9c67d48325e83dcf3b042fe3d3eb80efa
```

Notebook outputs preserve historical evidence of the completed 3,653-day run
and historical evaluation summaries. Those summaries are not validation of a
newly generated product and must not be presented as new results.

The evidence sources agree independently:

- `notebooks/narr_prism_inference.ipynb` in commit `2c3ce4f` records the exact
  checkpoint, two A100 workers (`0,1`), batch size 8, all workers finishing,
  and 3,653 files at the historical destination. Commit `c218d85`, made at
  2026-08-12 17:35:35 UTC, changes only notebook widget/plot output and retains
  the same successful run record.
- Read-only IPython history records the final full notebook session beginning
  at 2026-08-11 22:33:10 UTC. The journal's creation times establish when its
  first and last daily products were committed to disk.
- The journal also contains the deleted worker-log names
  `parallel_worker_0_gpu0.log` and `parallel_worker_1_gpu1.log`; their data is
  no longer recoverable.
- `notebooks/Compare_inference_prism.ipynb` records 3,653 common dates and the
  historical comparison products under `experiments/comparison_plots`. The
  NetCDF, CSV, PNG, and GIF products named there were deleted with the artifact
  directory and were not found elsewhere.
- The former ignored output tree had no run-level manifest. Output NetCDF,
  plots, arrays, and experiment directories were intentionally Git-ignored, so
  Git history contains notebook evidence but not the large daily products.
- Searches of the external artifact portal, the detached
  `/data/granite-wxc-stochastic` worktree, and other local recovery locations
  found no surviving historical prediction set. The detached worktree contains
  source only; it is not an artifact backup.

Recovery provenance is retained outside Git at
`/data2/granite-wxc-artifacts/NARR_PRISM/recovery_provenance/2026-08-12`.
That recovery restored the exact checkpoint and training artifacts, and the
complete predictor-only inference inputs, but not the deleted deterministic
prediction files. The durable output root now resolves through the repository
portal to `/data2/granite-wxc-artifacts/NARR_PRISM/experiments`.

## Current regeneration status

A new production inference invocation began at 2026-08-12 23:52:50 UTC and was
still in progress when this record was written. It uses the recovered
checkpoint and three date-sharded GPU workers (`1,2,3`). Its products are
currently accessible through the historical repository spelling and are
stored at
`/data2/granite-wxc-artifacts/NARR_PRISM/experiments/inference_output/narr_prism_California`.
It is a distinct regeneration under the current hardened output contract; it
is not a byte-for-byte recovery of the deleted files. Do not describe it as
complete until the audit reports exactly 3,653 files, no
missing/extra/duplicate dates, and valid NetCDF endpoint (or all-file) checks.

## Reproduce

Run from the repository root in the existing `Prithvi` environment. The
following command uses GPUs 1, 2, and 3; adjust that explicit list only for the
available hardware:

```bash
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_inference.py \
  --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml \
  --checkpoint examples/NARR_PRISM/experiments/checkpoints/narr_prism_California/last.ckpt \
  --output-dir examples/NARR_PRISM/experiments/inference_output \
  --split inference \
  --batch-size 8 \
  --device cuda \
  --parallel-gpus 1,2,3
```

The inference writer uses a temporary file and `os.replace` for each daily
NetCDF. Process-level shards write disjoint dates. A run intentionally writes
the full configured interval; do not launch a second run against the same
directory while one is active.

## Inventory, validate, and publish a manifest

The audit is read-only unless `--write-manifest` or `--json-out` is supplied.
It resolves the YAML output root and case directory, enumerates the inclusive
date range (including leap days), and exits nonzero for missing, extra,
duplicate, malformed, or invalid outputs.

During a run, obtain a machine-readable progress inventory without opening
NetCDF data:

```bash
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_output_audit.py \
  --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml \
  --validation-level none
```

After inference finishes, validate the first and last files as a quick
structural check. Endpoint validation deliberately cannot publish a final
manifest because it has not opened the intervening 3,651 files:

```bash
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_output_audit.py \
  --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml \
  --checkpoint examples/NARR_PRISM/experiments/checkpoints/narr_prism_California/last.ckpt \
  --validation-level endpoints
```

For final archival validation, inspect every daily NetCDF:

```bash
mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_output_audit.py \
  --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml \
  --checkpoint examples/NARR_PRISM/experiments/checkpoints/narr_prism_California/last.ckpt \
  --validation-level all \
  --write-manifest
```

Only the subsequent all-file command can write the final manifest. The
manifest contains the resolved config and output paths, config hash, checkpoint
path and SHA-256,
case/split/range, explicit target order, expected and observed counts,
missing/extra/duplicate diagnostics, validated-file provenance summaries, and
one name/size/mtime record per daily file. Its inventory signature detects a
later change in that run-level file inventory; it is not a substitute for a
cryptographic content hash of every multi-megabyte NetCDF.

Only after this audit succeeds should evaluation figures and metrics be
regenerated. Historical notebook metrics must remain labeled historical until
the new 3,653-file set is independently evaluated.
