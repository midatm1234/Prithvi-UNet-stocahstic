#!/usr/bin/env python3
"""Read-only content audit across related Git refs.

The audit compares committed trees.  It never checks out, merges, resets,
fetches, updates refs, stages files, or otherwise mutates Git state.  An
optional JSON report is the only file it writes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
RELATED_REF_TERMS = (
    "narr",
    "prism",
    "cordex",
    "deterministic",
    "diffusion",
    "flow_matching",
    "flow-matching",
    "phase1",
    "refinement",
    "stochastic",
    "unet",
)
GENERATED_DIRECTORY_NAMES = {
    ".cache",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "build",
    "cache",
    "checkpoints",
    "dist",
    "diagnostics",
    "evaluations",
    "evaluation_outputs",
    "experiments",
    "inference_output",
    "lightning_logs",
    "outputs",
    "predictions",
    "refinement_outputs",
    "runs",
    "scalars",
    "wandb",
}
GENERATED_SUFFIXES = {
    ".arrow",
    ".bin",
    ".cdf",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".log",
    ".nc",
    ".nc4",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".pyc",
    ".tmp",
    ".zarr",
}


class AuditError(RuntimeError):
    """Raised when the repository cannot be inspected safely."""


def _run_git(
    repo: Path,
    arguments: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = ["git", "-C", os.fspath(repo), *arguments]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
    )
    if check and completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise AuditError(
            f"Git inspection failed ({' '.join(arguments)}): {detail}"
        )
    return completed


def _repository_root(path: Path) -> Path:
    result = _run_git(path, ["rev-parse", "--show-toplevel"])
    return Path(result.stdout.strip()).resolve()


def _list_refs(repo: Path) -> list[dict[str, str]]:
    result = _run_git(
        repo,
        [
            "for-each-ref",
            "--format=%(refname)%09%(refname:short)%09%(objectname)%09%(symref)",
            "refs/heads",
            "refs/remotes",
        ],
    )
    refs: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        full_name, short_name, commit, symref = line.split("\t", 3)
        refs.append(
            {
                "full_name": full_name,
                "short_name": short_name,
                "commit": commit,
                "symbolic_target": symref,
            }
        )
    return refs


def _validate_ref_input(ref: str) -> str:
    ref = ref.strip()
    if not ref:
        raise AuditError("A Git ref cannot be empty")
    if ref.startswith("-") or "\x00" in ref or "\n" in ref:
        raise AuditError(f"Unsafe Git ref syntax: {ref!r}")
    return ref


def _resolve_requested_ref(
    repo: Path,
    requested: str,
    refs: Sequence[dict[str, str]],
) -> dict[str, str] | None:
    requested = _validate_ref_input(requested)
    candidates = [
        ref
        for ref in refs
        if requested in {ref["full_name"], ref["short_name"]}
    ]
    if not candidates and "/" not in requested:
        candidates = [
            ref
            for ref in refs
            if ref["short_name"].endswith(f"/{requested}")
        ]
    if candidates:
        candidates.sort(
            key=lambda item: (
                not item["full_name"].startswith("refs/heads/"),
                not item["short_name"].startswith("origin/"),
                item["short_name"],
            )
        )
        selected = candidates[0]
        return {
            "requested": requested,
            "ref": selected["short_name"],
            "full_ref": selected["full_name"],
            "commit": selected["commit"],
        }

    result = _run_git(
        repo,
        ["rev-parse", "--verify", "--quiet", f"{requested}^{{commit}}"],
        check=False,
    )
    if result.returncode:
        return None
    return {
        "requested": requested,
        "ref": requested,
        "full_ref": requested,
        "commit": result.stdout.strip(),
    }


def _default_ref_names(
    repo: Path, refs: Sequence[dict[str, str]]
) -> list[str]:
    names: list[str] = []
    symbolic = _run_git(
        repo,
        ["symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"],
        check=False,
    )
    if not symbolic.returncode and symbolic.stdout.strip():
        names.append(symbolic.stdout.strip())
    available = {item["short_name"] for item in refs}
    for name in ("main", "master", "origin/main", "origin/master"):
        if name in available and name not in names:
            names.append(name)
    return names


def discover_related_refs(
    repo: Path,
    *,
    current_commit: str,
    refs: Sequence[dict[str, str]] | None = None,
) -> list[str]:
    """Return local snapshots of likely relevant branch refs.

    Local and remote-tracking refs are considered independently when they point
    at different commits.  A local ref is preferred when both names point at
    the same commit.  No network access or fetch is performed.
    """

    refs = list(refs if refs is not None else _list_refs(repo))
    default_names = set(_default_ref_names(repo, refs))
    candidates: list[dict[str, str]] = []
    for ref in refs:
        short_name = ref["short_name"]
        lowered = short_name.lower()
        if ref["symbolic_target"] or lowered.startswith("backup/"):
            continue
        relevant = short_name in default_names or any(
            term in lowered for term in RELATED_REF_TERMS
        )
        if relevant and ref["commit"] != current_commit:
            candidates.append(ref)

    candidates.sort(
        key=lambda item: (
            not item["full_name"].startswith("refs/heads/"),
            item["short_name"].lower(),
        )
    )
    discovered: list[str] = []
    seen_commits: set[str] = set()
    for candidate in candidates:
        commit = candidate["commit"]
        if commit in seen_commits:
            continue
        seen_commits.add(commit)
        discovered.append(candidate["short_name"])
    return discovered


def _tree(repo: Path, commit: str) -> dict[str, dict[str, str]]:
    result = _run_git(repo, ["ls-tree", "-r", "-z", "--full-tree", commit])
    entries: dict[str, dict[str, str]] = {}
    for record in result.stdout.split("\x00"):
        if not record:
            continue
        metadata, path = record.split("\t", 1)
        mode, object_type, object_id = metadata.split(" ", 2)
        entries[path] = {
            "mode": mode,
            "type": object_type,
            "object": object_id,
        }
    return entries


def _merge_base(repo: Path, left: str, right: str) -> str | None:
    result = _run_git(repo, ["merge-base", left, right], check=False)
    if result.returncode:
        return None
    return result.stdout.strip() or None


def _renames(repo: Path, current: str, other: str) -> list[dict[str, Any]]:
    result = _run_git(
        repo,
        [
            "diff",
            "--name-status",
            "-z",
            "--find-renames=50%",
            current,
            other,
            "--",
        ],
    )
    records = result.stdout.split("\x00")
    renames: list[dict[str, Any]] = []
    index = 0
    while index < len(records):
        status = records[index]
        index += 1
        if not status:
            continue
        if status.startswith("R") or status.startswith("C"):
            if index + 1 >= len(records):
                raise AuditError("Git returned a truncated rename record")
            current_path = records[index]
            other_path = records[index + 1]
            index += 2
            renames.append(
                {
                    "kind": "rename" if status.startswith("R") else "copy",
                    "similarity_percent": int(status[1:] or "0"),
                    "current_path": current_path,
                    "other_path": other_path,
                }
            )
        else:
            index += 1
    return renames


def _normalize_prefixes(prefixes: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for prefix in prefixes:
        candidate = prefix.strip().replace("\\", "/").strip("/")
        parts = PurePosixPath(candidate).parts
        if not candidate or ".." in parts:
            raise AuditError(f"Invalid path prefix: {prefix!r}")
        normalized.append(candidate)
    return tuple(dict.fromkeys(normalized))


def _selected(path: str, prefixes: Sequence[str]) -> bool:
    if not prefixes:
        return True
    return any(
        path == prefix or path.startswith(f"{prefix}/")
        for prefix in prefixes
    )


def generated_artifact_reasons(path: str) -> list[str]:
    """Return explainable heuristics for a likely generated artifact."""

    pure_path = PurePosixPath(path)
    lowered_parts = [part.lower() for part in pure_path.parts]
    reasons: list[str] = []
    generated_parts = sorted(
        set(lowered_parts).intersection(GENERATED_DIRECTORY_NAMES)
    )
    if generated_parts:
        reasons.append(f"generated-directory:{generated_parts[0]}")

    suffixes = [suffix.lower() for suffix in pure_path.suffixes]
    generated_suffixes = [
        suffix for suffix in suffixes if suffix in GENERATED_SUFFIXES
    ]
    if generated_suffixes:
        reasons.append(f"generated-suffix:{generated_suffixes[-1]}")
    zarr_parts = [part for part in lowered_parts if part.endswith(".zarr")]
    if zarr_parts and "generated-suffix:.zarr" not in reasons:
        reasons.append("generated-suffix:.zarr")

    name = pure_path.name.lower()
    if name.endswith(("_benchmark.json", "_parity.json")):
        reasons.append("generated-report-name")
    if name == "metrics.json" or name.startswith("metrics_"):
        reasons.append("generated-metrics-name")
    return reasons


def _working_tree(repo: Path) -> dict[str, Any]:
    result = _run_git(
        repo,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
    )
    records = [record for record in result.stdout.split("\x00") if record]
    entries: list[dict[str, str]] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if len(record) < 4:
            entries.append({"status": "?", "path": record})
            continue
        status = record[:2]
        entry = {"status": status, "path": record[3:]}
        if ("R" in status or "C" in status) and index < len(records):
            entry["original_path"] = records[index]
            index += 1
        entries.append(entry)
    return {
        "clean": not entries,
        "change_count": len(entries),
        "changes": entries,
    }


def compare_refs(
    repo: Path,
    *,
    current_commit: str,
    source: dict[str, str],
    prefixes: Sequence[str] = (),
) -> dict[str, Any]:
    current_tree = {
        path: value
        for path, value in _tree(repo, current_commit).items()
        if _selected(path, prefixes)
    }
    other_tree = {
        path: value
        for path, value in _tree(repo, source["commit"]).items()
        if _selected(path, prefixes)
    }
    current_paths = set(current_tree)
    other_paths = set(other_tree)
    missing = sorted(other_paths - current_paths)
    current_only = sorted(current_paths - other_paths)

    merge_base = _merge_base(repo, current_commit, source["commit"])
    base_paths: set[str] = set()
    if merge_base:
        base_paths = {
            path
            for path in _tree(repo, merge_base)
            if _selected(path, prefixes)
        }
    unexpectedly_deleted = sorted(set(missing).intersection(base_paths))

    modified: list[dict[str, str]] = []
    for path in sorted(current_paths.intersection(other_paths)):
        current_entry = current_tree[path]
        other_entry = other_tree[path]
        if current_entry != other_entry:
            modified.append(
                {
                    "path": path,
                    "current_object": current_entry["object"],
                    "other_object": other_entry["object"],
                    "current_mode": current_entry["mode"],
                    "other_mode": other_entry["mode"],
                }
            )

    renames = [
        item
        for item in _renames(repo, current_commit, source["commit"])
        if _selected(item["current_path"], prefixes)
        or _selected(item["other_path"], prefixes)
    ]

    locations: dict[str, set[str]] = {}
    for path in missing:
        locations.setdefault(path, set()).add("other")
    for path in current_only:
        locations.setdefault(path, set()).add("current")
    for item in modified:
        locations.setdefault(item["path"], set()).update({"current", "other"})
    generated = []
    for path, present_in in sorted(locations.items()):
        reasons = generated_artifact_reasons(path)
        if reasons:
            generated.append(
                {
                    "path": path,
                    "present_in": sorted(present_in),
                    "reasons": reasons,
                }
            )

    counts = {
        "missing_from_current": len(missing),
        "unexpectedly_deleted": len(unexpectedly_deleted),
        "current_only": len(current_only),
        "renames_or_copies": len(renames),
        "modified_differently": len(modified),
        "likely_generated_artifacts": len(generated),
    }
    return {
        "requested_ref": source["requested"],
        "source_ref": source["ref"],
        "source_full_ref": source["full_ref"],
        "source_commit": source["commit"],
        "merge_base": merge_base,
        "counts": counts,
        "missing_from_current": missing,
        "present_only_in_other": list(missing),
        "unexpectedly_deleted": unexpectedly_deleted,
        "current_only": current_only,
        "renames": renames,
        "modified_differently": modified,
        "likely_generated_artifacts": generated,
    }


def audit_repository(
    repo: Path,
    *,
    current: str = "HEAD",
    sources: Sequence[str] = (),
    discover: bool = True,
    prefixes: Sequence[str] = (),
) -> dict[str, Any]:
    repo = _repository_root(repo)
    refs = _list_refs(repo)
    current_source = _resolve_requested_ref(repo, current, refs)
    if current_source is None:
        raise AuditError(
            f"Current ref does not resolve to a commit: {current}"
        )
    current_commit = current_source["commit"]

    requested_sources = list(sources)
    discovered_sources: list[str] = []
    if discover:
        discovered_sources = discover_related_refs(
            repo, current_commit=current_commit, refs=refs
        )
        requested_sources.extend(discovered_sources)
    requested_sources = list(dict.fromkeys(requested_sources))
    normalized_prefixes = _normalize_prefixes(prefixes)

    comparisons: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    for requested in requested_sources:
        resolved = _resolve_requested_ref(repo, requested, refs)
        if resolved is None:
            unavailable.append(
                {
                    "requested_ref": requested,
                    "reason": "not present in local refs or object database",
                }
            )
            continue
        comparisons.append(
            compare_refs(
                repo,
                current_commit=current_commit,
                source=resolved,
                prefixes=normalized_prefixes,
            )
        )

    totals = {
        key: sum(item["counts"][key] for item in comparisons)
        for key in (
            "missing_from_current",
            "unexpectedly_deleted",
            "current_only",
            "renames_or_copies",
            "modified_differently",
            "likely_generated_artifacts",
        )
    }
    branch_result = _run_git(
        repo, ["symbolic-ref", "--quiet", "--short", "HEAD"], check=False
    )
    current_branch = (
        branch_result.stdout.strip() if not branch_result.returncode else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": os.fspath(repo),
        "current": {
            "requested_ref": current,
            "resolved_ref": current_source["ref"],
            "commit": current_commit,
            "checked_out_branch": current_branch,
        },
        "path_prefixes": list(normalized_prefixes),
        "discovery": {
            "enabled": discover,
            "discovered_refs": discovered_sources,
            "note": "Uses local and remote-tracking refs only; never fetches.",
        },
        "working_tree": _working_tree(repo),
        "unavailable_refs": unavailable,
        "totals": totals,
        "comparisons": comparisons,
    }


def _append_paths(
    lines: list[str],
    title: str,
    paths: Sequence[str],
    *,
    limit: int,
) -> None:
    lines.append(f"  {title}: {len(paths)}")
    for path in paths[:limit]:
        lines.append(f"    - {path}")
    if len(paths) > limit:
        lines.append(f"    ... {len(paths) - limit} more (see JSON report)")


def format_text(report: dict[str, Any], *, limit: int = 100) -> str:
    current = report["current"]
    displayed_ref = current["checked_out_branch"] or current["resolved_ref"]
    lines = [
        "Repository content audit",
        f"Repository: {report['repository']}",
        f"Current: {displayed_ref} @ {current['commit'][:12]}",
        (
            "Working tree: clean"
            if report["working_tree"]["clean"]
            else (
                "Working tree: "
                f"{report['working_tree']['change_count']} "
                "uncommitted change(s)"
            )
        ),
        (
            "Scope: entire repository"
            if not report["path_prefixes"]
            else f"Scope: {', '.join(report['path_prefixes'])}"
        ),
        "",
    ]
    if report["unavailable_refs"]:
        lines.append("Unavailable requested refs:")
        for item in report["unavailable_refs"]:
            lines.append(
                f"  - {item['requested_ref']}: {item['reason']}"
            )
        lines.append("")

    if not report["comparisons"]:
        lines.append("No source refs were available for comparison.")
    for comparison in report["comparisons"]:
        lines.extend(
            [
                (
                    f"[{comparison['source_ref']}] "
                    f"{comparison['source_commit'][:12]}"
                ),
                (
                    "  Merge base: "
                    + (
                        comparison["merge_base"][:12]
                        if comparison["merge_base"]
                        else "none (unrelated histories)"
                    )
                ),
            ]
        )
        _append_paths(
            lines,
            "Missing from current / present only in source",
            comparison["missing_from_current"],
            limit=limit,
        )
        _append_paths(
            lines,
            "Unexpected-delete candidates",
            comparison["unexpectedly_deleted"],
            limit=limit,
        )
        _append_paths(
            lines,
            "Present only in current",
            comparison["current_only"],
            limit=limit,
        )
        lines.append(f"  Renames/copies: {len(comparison['renames'])}")
        for item in comparison["renames"][:limit]:
            lines.append(
                "    - "
                f"{item['current_path']} -> {item['other_path']} "
                f"({item['kind']}, {item['similarity_percent']}%)"
            )
        lines.append(
            "  Modified differently: "
            f"{len(comparison['modified_differently'])}"
        )
        for item in comparison["modified_differently"][:limit]:
            lines.append(f"    - {item['path']}")
        lines.append(
            "  Likely generated artifacts among differences: "
            f"{len(comparison['likely_generated_artifacts'])}"
        )
        for item in comparison["likely_generated_artifacts"][:limit]:
            reasons = ", ".join(item["reasons"])
            lines.append(f"    - {item['path']} ({reasons})")
        lines.append("")

    totals = report["totals"]
    lines.extend(
        [
            "Aggregate comparison counts "
            "(a path may appear for multiple refs):",
            *[f"  {key}: {value}" for key, value in totals.items()],
            "",
            (
                "This report is advisory. Inspect provenance and reconcile "
                "each candidate explicitly; do not restore generated "
                "artifacts or overwrite newer files automatically."
            ),
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the current committed Git tree with related source refs "
            "without modifying Git state."
        )
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Path inside the repository (default: current directory).",
    )
    parser.add_argument(
        "--current",
        default="HEAD",
        help="Commit/ref used as the current tree (default: HEAD).",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source ref to compare; repeat for multiple refs.",
    )
    parser.add_argument(
        "--no-discover",
        action="store_true",
        help="Compare only explicitly supplied --source refs.",
    )
    parser.add_argument(
        "--path-prefix",
        action="append",
        default=[],
        help="Restrict results to a repository-relative path; repeatable.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Standard-output format (default: text).",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Also write the complete machine-readable JSON report.",
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=100,
        help="Maximum paths per text section; JSON is never truncated.",
    )
    parser.add_argument(
        "--fail-on-differences",
        action="store_true",
        help="Return status 1 when refs are unavailable or differ.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_paths < 1:
        raise SystemExit("--max-paths must be at least 1")
    try:
        report = audit_repository(
            args.repo,
            current=args.current,
            sources=args.source,
            discover=not args.no_discover,
            prefixes=args.path_prefix,
        )
    except AuditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json_output:
        _write_json(args.json_output, report)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_text(report, limit=args.max_paths))

    if args.fail_on_differences:
        differs = bool(report["unavailable_refs"]) or any(
            any(comparison["counts"].values())
            for comparison in report["comparisons"]
        )
        return int(differs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
