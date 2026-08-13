from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.audit_repository_content import (
    _list_refs,
    _resolve_requested_ref,
    audit_repository,
    discover_related_refs,
    format_text,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = REPO_ROOT / "scripts" / "audit_repository_content.py"


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write(repo: Path, relative_path: str, content: str) -> None:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "--all")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repository"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main")
    _git(repo, "config", "user.name", "Audit Test")
    _git(repo, "config", "user.email", "audit@example.invalid")
    return repo


def _divergent_repository(tmp_path: Path) -> Path:
    repo = _repository(tmp_path)
    _write(repo, "shared.txt", "base\n")
    _write(repo, "retained_in_source.txt", "retain\n")
    _write(repo, "old_name.txt", "rename me\n")
    _commit(repo, "base")

    _git(repo, "switch", "--create", "NARR_PRISM")
    _write(repo, "shared.txt", "source version\n")
    _write(repo, "source_only.py", "VALUE = 1\n")
    _write(repo, "outputs/prediction.nc", "generated-placeholder\n")
    _write(repo, "evaluations/run/metrics.csv", "rmse,1.0\n")
    _git(repo, "mv", "old_name.txt", "new_name.txt")
    _commit(repo, "source content")

    _git(repo, "switch", "main")
    _git(repo, "switch", "--create", "current")
    _write(repo, "shared.txt", "current version\n")
    _write(repo, "current_only.py", "VALUE = 2\n")
    _git(repo, "rm", "retained_in_source.txt")
    _commit(repo, "current content")
    return repo


def test_audit_reports_missing_deletes_renames_modifications_and_artifacts(
    tmp_path: Path,
) -> None:
    repo = _divergent_repository(tmp_path)

    report = audit_repository(
        repo,
        sources=["NARR_PRISM"],
        discover=False,
    )

    assert report["schema_version"] == 1
    assert report["working_tree"]["clean"] is True
    comparison = report["comparisons"][0]
    assert {
        "new_name.txt",
        "evaluations/run/metrics.csv",
        "outputs/prediction.nc",
        "retained_in_source.txt",
        "source_only.py",
    }.issubset(comparison["missing_from_current"])
    assert comparison["present_only_in_other"] == comparison[
        "missing_from_current"
    ]
    assert comparison["unexpectedly_deleted"] == ["retained_in_source.txt"]
    assert "current_only.py" in comparison["current_only"]
    assert [item["path"] for item in comparison["modified_differently"]] == [
        "shared.txt"
    ]
    assert {
        (item["current_path"], item["other_path"])
        for item in comparison["renames"]
    } == {("old_name.txt", "new_name.txt")}
    generated = {
        item["path"]: item["reasons"]
        for item in comparison["likely_generated_artifacts"]
    }
    assert "generated-directory:outputs" in generated["outputs/prediction.nc"]
    assert "generated-suffix:.nc" in generated["outputs/prediction.nc"]
    assert "generated-directory:evaluations" in generated[
        "evaluations/run/metrics.csv"
    ]

    rendered = format_text(report, limit=10)
    assert "Missing from current / present only in source" in rendered
    assert "Unexpected-delete candidates" in rendered
    assert "old_name.txt -> new_name.txt" in rendered
    assert "Likely generated artifacts among differences" in rendered


def test_remote_tracking_ref_is_used_when_local_branch_is_absent(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    _write(repo, "base.txt", "base\n")
    _commit(repo, "base")
    _git(repo, "switch", "--create", "NARR_PRISM")
    _write(repo, "narr.py", "NARR = True\n")
    narr_commit = _commit(repo, "narr")
    _git(repo, "switch", "main")
    _git(repo, "update-ref", "refs/remotes/origin/NARR_PRISM", narr_commit)
    _git(repo, "update-ref", "-d", "refs/heads/NARR_PRISM")

    refs = _list_refs(repo)
    resolved = _resolve_requested_ref(repo, "NARR_PRISM", refs)

    assert resolved is not None
    assert resolved["ref"] == "origin/NARR_PRISM"
    assert resolved["commit"] == narr_commit
    discovered = discover_related_refs(
        repo,
        current_commit=_git(repo, "rev-parse", "HEAD"),
        refs=refs,
    )
    assert "origin/NARR_PRISM" in discovered


def test_cli_emits_complete_json_without_mutating_git_state(
    tmp_path: Path,
) -> None:
    repo = _divergent_repository(tmp_path)
    before_head = _git(repo, "rev-parse", "HEAD")
    before_status = _git(repo, "status", "--porcelain=v1")

    completed = subprocess.run(
        [
            sys.executable,
            str(AUDIT_SCRIPT),
            "--repo",
            str(repo),
            "--source",
            "NARR_PRISM",
            "--source",
            "absent-branch",
            "--no-discover",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert len(report["comparisons"]) == 1
    assert report["unavailable_refs"] == [
        {
            "reason": "not present in local refs or object database",
            "requested_ref": "absent-branch",
        }
    ]
    assert _git(repo, "rev-parse", "HEAD") == before_head
    assert _git(repo, "status", "--porcelain=v1") == before_status


def test_path_prefix_limits_tree_results(tmp_path: Path) -> None:
    repo = _divergent_repository(tmp_path)

    report = audit_repository(
        repo,
        sources=["NARR_PRISM"],
        discover=False,
        prefixes=["outputs"],
    )

    comparison = report["comparisons"][0]
    assert comparison["missing_from_current"] == ["outputs/prediction.nc"]
    assert comparison["current_only"] == []
    assert comparison["modified_differently"] == []
