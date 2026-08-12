"""Validate and configure the local NARR--PRISM artifact portal.

The repository tracks the stable ``experiments``, ``preprocessed``, and
``scalars_with_H`` symlinks.  Each one routes through the ignored local
``artifacts`` entry so a clone can select its own large-artifact location
without changing a tracked link or allowing Git to replace the data directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

MANAGED_LINKS = {
    "experiments": Path("artifacts/experiments"),
    "preprocessed": Path("artifacts/preprocessed"),
    "scalars_with_H": Path("artifacts/scalars_with_H"),
}


class ArtifactLinkError(RuntimeError):
    """Raised when an artifact link is missing, cyclic, or misdirected."""


def _lexists(path: Path) -> bool:
    """Return true for normal paths and dangling symbolic links."""

    return os.path.lexists(path)


def _resolved(path: Path, *, label: str, strict: bool) -> Path:
    try:
        return path.resolve(strict=strict)
    except RuntimeError as exc:
        raise ArtifactLinkError(
            f"{label} has a symbolic-link cycle or points to itself: {path}"
        ) from exc
    except OSError as exc:
        raise ArtifactLinkError(
            f"Cannot resolve {label} {path}: {exc}"
        ) from exc


def _validate_managed_link_layout(base: Path) -> None:
    for name, expected_target in MANAGED_LINKS.items():
        link = base / name
        if not link.is_symlink():
            kind = "missing" if not _lexists(link) else "not a symbolic link"
            raise ArtifactLinkError(
                f"Managed artifact entry {link} is {kind}; its tracked target "
                f"must be {expected_target}."
            )
        actual_target = Path(os.readlink(link))
        if actual_target != expected_target:
            raise ArtifactLinkError(
                f"Managed artifact link {link} targets {actual_target}, "
                f"expected the relative portal target {expected_target}."
            )


def validate_artifact_links(
    narr_prism_dir: str | Path | None = None,
    *,
    require_targets: bool = True,
) -> dict[str, Any]:
    """Validate the tracked link layout and return its resolved locations.

    ``require_targets=False`` validates a fresh checkout where the ignored
    ``artifacts`` entry has not yet been configured.  Production callers should
    retain the default so missing artifact data fails before model setup.
    """

    base = (
        Path(narr_prism_dir)
        if narr_prism_dir is not None
        else Path(__file__).resolve().parent
    ).absolute()

    _validate_managed_link_layout(base)

    portal = base / "artifacts"
    if not _lexists(portal):
        if require_targets:
            helper = base / "narr_prism_artifacts.py"
            raise ArtifactLinkError(
                f"Local artifact portal is not configured: {portal}. Run "
                f"`mamba run -n Prithvi python {helper} "
                "--artifact-root /path/to/NARR_PRISM` first."
            )
        return {
            "narr_prism_dir": str(base),
            "artifact_root": None,
            "configured": False,
            "links": {
                name: str(base / target)
                for name, target in MANAGED_LINKS.items()
            },
        }

    artifact_root = _resolved(
        portal, label="local artifact portal", strict=require_targets
    )
    repository_root = _resolved(
        base, label="NARR--PRISM directory", strict=True
    )
    if (
        artifact_root == repository_root
        or artifact_root in repository_root.parents
    ):
        raise ArtifactLinkError(
            f"Local artifact portal {portal} resolves to the NARR--PRISM "
            "source directory itself or one of its ancestors. Choose a "
            "separate artifact directory."
        )
    if portal.is_symlink() and repository_root in artifact_root.parents:
        raise ArtifactLinkError(
            f"Local artifact portal {portal} resolves inside the source tree "
            f"at {artifact_root}. Use an external directory, or make {portal} "
            "an ignored real directory."
        )
    if require_targets and not artifact_root.is_dir():
        raise ArtifactLinkError(
            f"Local artifact portal {portal} does not resolve to a directory: "
            f"{artifact_root}"
        )

    resolved_links: dict[str, str] = {}
    for name in MANAGED_LINKS:
        link = base / name
        resolved = _resolved(
            link, label=f"managed artifact link {name}", strict=require_targets
        )
        if require_targets and not resolved.is_dir():
            raise ArtifactLinkError(
                f"Managed artifact target for {name} is not a directory: "
                f"{resolved}"
            )
        resolved_links[name] = str(resolved)

    return {
        "narr_prism_dir": str(repository_root),
        "artifact_root": str(artifact_root),
        "configured": True,
        "links": resolved_links,
    }


def configure_artifact_root(
    artifact_root: str | Path,
    narr_prism_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create the ignored portal if absent, without replacing existing data."""

    base = (
        Path(narr_prism_dir)
        if narr_prism_dir is not None
        else Path(__file__).resolve().parent
    ).absolute()
    target = _resolved(
        Path(artifact_root).expanduser(),
        label="requested artifact root",
        strict=True,
    )
    if not target.is_dir():
        raise ArtifactLinkError(
            f"Requested artifact root is not a directory: {target}"
        )
    repository_root = _resolved(
        base, label="NARR--PRISM directory", strict=True
    )
    if target == repository_root or target in repository_root.parents:
        raise ArtifactLinkError(
            "The artifact root must not be the NARR--PRISM source directory "
            "or one of its ancestors."
        )
    if repository_root in target.parents:
        raise ArtifactLinkError(
            f"Requested artifact root {target} is inside the source tree. Use "
            "an external directory so a Git checkout cannot affect its "
            "contents."
        )

    portal = base / "artifacts"
    if _lexists(portal):
        current = _resolved(
            portal, label="existing artifact portal", strict=True
        )
        if current != target:
            raise ArtifactLinkError(
                f"Refusing to replace existing {portal}, which resolves to "
                f"{current}; requested {target}. Move or unlink it "
                "intentionally first."
            )
    else:
        portal.symlink_to(target, target_is_directory=True)

    return validate_artifact_links(base)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Existing per-clone artifact root to link through ./artifacts",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Only validate the tracked link layout; allow an unconfigured "
            "portal"
        ),
    )
    args = parser.parse_args()

    try:
        if args.artifact_root is not None:
            status = configure_artifact_root(args.artifact_root)
        else:
            status = validate_artifact_links(
                require_targets=not args.allow_missing
            )
    except ArtifactLinkError as exc:
        parser.exit(2, f"artifact-link error: {exc}\n")
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
