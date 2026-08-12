"""Auditable rebinds for legacy PRISM source-artifact device identifiers.

The version-1 PRISM source signature includes ``stat.st_dev``.  That is useful
for detecting replacement of a source file, but the identifier can change when
the *same* filesystem is enumerated under a different device number.  This
module provides a deliberately narrow migration contract for that case.

An attestation contains the immutable metadata for every source file and the
old/new device number.  Both legacy daily signatures and both split signatures
are reconstructed, so this is not a general-purpose checkpoint bypass.  The
runtime verifier also binds the attestation to the exact active scalar
manifest.  All JSON digests use canonical UTF-8 JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from granitewxc.utils import normalization
from granitewxc.utils.prism_preprocessed import (
    preprocessing_signature,
    split_source_artifact_signature,
)

ATTESTATION_FILENAME = "phase1_source_rebind.json"
ATTESTATION_SCHEMA_VERSION = 1
ATTESTATION_TYPE = "prism_phase1_source_device_rebind"
ATTESTATION_REASON = "filesystem_device_enumeration_drift"

_MAX_JSON_BYTES = 64 * 1024 * 1024
_IMMUTABLE_SOURCE_FIELDS = ("label", "path", "inode", "size", "mtime_ns")
_CHANGED_SOURCE_FIELDS = ("device",)
_SOURCE_ENTRY_FIELDS = (
    *_IMMUTABLE_SOURCE_FIELDS,
    "old_device",
    "new_device",
)
_IMMUTABLE_MANIFEST_FIELDS = (
    "schema_version",
    "case_name",
    "predictor_mode",
    "target_mode",
    "train_date_range",
    "scalers",
    "config_contract",
    "prism_grid",
    "scaling_statistics",
    "predictor_preprocessing_signature",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value!r}")


def _load_json_mapping(path: Path, *, role: str) -> dict[str, Any]:
    try:
        stat = path.lstat()
    except OSError as exc:
        raise ValueError(f"[{role}] cannot access {path}: {exc}") from exc
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            f"[{role}] expected a regular, non-symlink JSON file: {path}"
        )
    if stat.st_size > _MAX_JSON_BYTES:
        raise ValueError(
            f"[{role}] JSON file exceeds {_MAX_JSON_BYTES} bytes: {path}"
        )
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"[{role}] invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"[{role}] expected a JSON object in {path}")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: Sequence[str], *, role: str
) -> None:
    observed = set(value)
    wanted = set(expected)
    if observed != wanted:
        raise ValueError(
            f"[{role}] fields differ: missing={sorted(wanted - observed)}, "
            f"unexpected={sorted(observed - wanted)}"
        )


def _require_sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"[{role}] expected a lowercase SHA-256 digest")
    return value


def _require_int(value: Any, *, role: str, nonnegative: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"[{role}] expected an integer")
    if nonnegative and value < 0:
        raise ValueError(f"[{role}] expected a nonnegative integer")
    return value


def _manifest_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    missing = [
        field for field in _IMMUTABLE_MANIFEST_FIELDS if field not in manifest
    ]
    if missing:
        raise ValueError(
            "source-rebind manifest lacks immutable provenance fields: "
            f"{missing}"
        )
    return {field: manifest[field] for field in _IMMUTABLE_MANIFEST_FIELDS}


def _manifest_source_contract(
    manifest: Mapping[str, Any], *, role: str
) -> tuple[dict[str, str], str]:
    raw_daily = manifest.get(
        normalization.TRAINING_SOURCE_ARTIFACT_SIGNATURES_KEY
    )
    raw_split = manifest.get(
        normalization.TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY
    )
    if not isinstance(raw_daily, Mapping) or not raw_daily:
        raise ValueError(
            f"[{role}] manifest has no daily source-artifact signatures"
        )
    daily: dict[str, str] = {}
    for raw_day, raw_signature in raw_daily.items():
        day = str(raw_day)
        if day in daily:
            raise ValueError(f"[{role}] duplicate source-artifact date {day}")
        daily[day] = _require_sha256(
            raw_signature, role=f"{role}:daily_source_artifacts:{day}"
        )
    split = _require_sha256(raw_split, role=f"{role}:split")
    computed = split_source_artifact_signature(daily)
    if split != computed:
        raise ValueError(
            f"[{role}] stored source split {split} does not match computed "
            f"{computed}"
        )
    return daily, split


def _normalize_proof(  # noqa: C901 - strict schema validation is intentionally explicit
    proof: Mapping[str, Any], *, role: str
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    _require_exact_keys(
        proof,
        (
            "immutable_fields",
            "changed_fields",
            "device_transitions",
            "daily_source_artifacts",
        ),
        role=role,
    )
    if proof["immutable_fields"] != list(_IMMUTABLE_SOURCE_FIELDS):
        raise ValueError(
            f"[{role}] immutable source fields are not the required set"
        )
    if proof["changed_fields"] != list(_CHANGED_SOURCE_FIELDS):
        raise ValueError(f"[{role}] only the device field may change")
    raw_daily = proof["daily_source_artifacts"]
    if not isinstance(raw_daily, Mapping) or not raw_daily:
        raise ValueError(
            f"[{role}] daily_source_artifacts must be a nonempty mapping"
        )

    normalized_daily: dict[str, list[dict[str, Any]]] = {}
    old_daily: dict[str, str] = {}
    new_daily: dict[str, str] = {}
    transition_counts: Counter[tuple[int, int]] = Counter()

    for raw_day in sorted(raw_daily):
        day = str(raw_day)
        if raw_day != day:
            raise ValueError(f"[{role}] date keys must be strings")
        raw_entries = raw_daily[raw_day]
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ValueError(f"[{role}:{day}] source list must be nonempty")
        normalized_entries: list[dict[str, Any]] = []
        labels: list[str] = []
        old_entries: list[dict[str, Any]] = []
        new_entries: list[dict[str, Any]] = []
        for index, raw_entry in enumerate(raw_entries):
            entry_role = f"{role}:{day}:{index}"
            if not isinstance(raw_entry, Mapping):
                raise ValueError(
                    f"[{entry_role}] source entry must be a mapping"
                )
            _require_exact_keys(
                raw_entry, _SOURCE_ENTRY_FIELDS, role=entry_role
            )
            label = raw_entry["label"]
            path = raw_entry["path"]
            if not isinstance(label, str) or not label:
                raise ValueError(
                    f"[{entry_role}] label must be a nonempty string"
                )
            if (
                not isinstance(path, str)
                or not path
                or not Path(path).is_absolute()
            ):
                raise ValueError(
                    f"[{entry_role}] path must be a nonempty absolute path"
                )
            inode = _require_int(
                raw_entry["inode"], role=f"{entry_role}:inode"
            )
            size = _require_int(raw_entry["size"], role=f"{entry_role}:size")
            mtime_ns = _require_int(
                raw_entry["mtime_ns"], role=f"{entry_role}:mtime_ns"
            )
            old_device = _require_int(
                raw_entry["old_device"], role=f"{entry_role}:old_device"
            )
            new_device = _require_int(
                raw_entry["new_device"], role=f"{entry_role}:new_device"
            )
            if old_device == new_device:
                raise ValueError(
                    f"[{entry_role}] old_device and new_device must differ"
                )
            labels.append(label)
            common = {
                "label": label,
                "path": path,
                "inode": inode,
                "size": size,
                "mtime_ns": mtime_ns,
            }
            normalized_entries.append(
                {**common, "old_device": old_device, "new_device": new_device}
            )
            old_entries.append({**common, "device": old_device})
            new_entries.append({**common, "device": new_device})
            transition_counts[(old_device, new_device)] += 1
        if labels != sorted(labels) or len(labels) != len(set(labels)):
            raise ValueError(
                f"[{role}:{day}] source labels must be unique and sorted"
            )
        normalized_daily[day] = normalized_entries
        old_daily[day] = preprocessing_signature(
            {"source_artifacts": old_entries}
        )
        new_daily[day] = preprocessing_signature(
            {"source_artifacts": new_entries}
        )

    computed_transitions = [
        {
            "old_device": old_device,
            "new_device": new_device,
            "source_count": count,
        }
        for (old_device, new_device), count in sorted(
            transition_counts.items()
        )
    ]
    if proof["device_transitions"] != computed_transitions:
        raise ValueError(
            f"[{role}] device_transitions do not match the daily proof"
        )
    return (
        {
            "immutable_fields": list(_IMMUTABLE_SOURCE_FIELDS),
            "changed_fields": list(_CHANGED_SOURCE_FIELDS),
            "device_transitions": computed_transitions,
            "daily_source_artifacts": normalized_daily,
        },
        old_daily,
        new_daily,
    )


def _build_attestation(
    old_manifest: Mapping[str, Any],
    new_manifest: Mapping[str, Any],
    proof: Mapping[str, Any],
    *,
    old_manifest_sha256: str,
    new_manifest_sha256: str,
) -> dict[str, Any]:
    old_projection = _manifest_projection(old_manifest)
    new_projection = _manifest_projection(new_manifest)
    if old_projection != new_projection:
        mismatch = [
            field
            for field in _IMMUTABLE_MANIFEST_FIELDS
            if old_projection[field] != new_projection[field]
        ]
        raise ValueError(
            "source rebind cannot change immutable scalar-manifest fields: "
            f"{mismatch}"
        )
    immutable_sha = _canonical_sha256(old_projection)
    old_manifest_daily, old_manifest_split = _manifest_source_contract(
        old_manifest, role="source-rebind:old-manifest"
    )
    new_manifest_daily, new_manifest_split = _manifest_source_contract(
        new_manifest, role="source-rebind:new-manifest"
    )
    normalized_proof, old_proof_daily, new_proof_daily = _normalize_proof(
        proof, role="source-rebind:proof"
    )
    if old_proof_daily != old_manifest_daily:
        raise ValueError(
            "source-rebind proof does not reconstruct every old daily "
            "signature"
        )
    if new_proof_daily != new_manifest_daily:
        raise ValueError(
            "source-rebind proof does not reconstruct every new daily "
            "signature"
        )
    if old_manifest_split == new_manifest_split:
        raise ValueError(
            "source-rebind old and new split signatures must differ"
        )

    source_count = sum(
        len(entries)
        for entries in normalized_proof["daily_source_artifacts"].values()
    )
    document: dict[str, Any] = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "attestation_type": ATTESTATION_TYPE,
        "reason": ATTESTATION_REASON,
        "manifest_binding": {
            "old_manifest_sha256": _require_sha256(
                old_manifest_sha256, role="source-rebind:old-manifest-sha256"
            ),
            "new_manifest_sha256": _require_sha256(
                new_manifest_sha256, role="source-rebind:new-manifest-sha256"
            ),
            "immutable_fields": list(_IMMUTABLE_MANIFEST_FIELDS),
            "immutable_sha256": immutable_sha,
        },
        "source_binding": {
            "old_split_sha256": old_manifest_split,
            "new_split_sha256": new_manifest_split,
            "date_count": len(old_manifest_daily),
            "source_count": source_count,
        },
        "proof_sha256": _canonical_sha256(normalized_proof),
        "proof": normalized_proof,
    }
    document["attestation_sha256"] = _canonical_sha256(document)
    return document


def write_phase1_source_rebind_attestation(  # noqa: C901
    old_manifest_path: Path | str,
    new_manifest_path: Path | str,
    proof_path: Path | str,
    output_path: Path | str | None = None,
) -> Path:
    """Validate a device-only proof and atomically write its attestation."""
    old_path = Path(old_manifest_path)
    new_path = Path(new_manifest_path)
    raw_proof_path = Path(proof_path)
    old_manifest = _load_json_mapping(
        old_path, role="source-rebind:old-manifest"
    )
    new_manifest = _load_json_mapping(
        new_path, role="source-rebind:new-manifest"
    )
    proof_input = _load_json_mapping(
        raw_proof_path, role="source-rebind:proof-input"
    )
    _require_exact_keys(
        proof_input,
        ("schema_version", "days"),
        role="source-rebind:proof-input",
    )
    if proof_input["schema_version"] != ATTESTATION_SCHEMA_VERSION:
        raise ValueError("source-rebind proof-input schema_version must be 1")
    raw_days = proof_input["days"]
    if not isinstance(raw_days, Mapping) or not raw_days:
        raise ValueError(
            "source-rebind proof-input days must be a nonempty mapping"
        )
    raw_daily: dict[str, list[dict[str, Any]]] = {}
    transition_counts: Counter[tuple[int, int]] = Counter()
    source_fields = (*_IMMUTABLE_SOURCE_FIELDS, "device")
    for raw_day in sorted(raw_days):
        day = str(raw_day)
        if day != raw_day:
            raise ValueError(
                "source-rebind proof-input date keys must be strings"
            )
        pair = raw_days[raw_day]
        if not isinstance(pair, Mapping):
            raise ValueError(
                f"source-rebind proof-input {day} must be a mapping"
            )
        _require_exact_keys(
            pair, ("old", "new"), role=f"source-rebind:proof-input:{day}"
        )
        old_entries = pair["old"]
        new_entries = pair["new"]
        if (
            not isinstance(old_entries, list)
            or not isinstance(new_entries, list)
            or not old_entries
            or len(old_entries) != len(new_entries)
        ):
            raise ValueError(
                f"source-rebind proof-input {day} old/new source lists must "
                "be "
                "nonempty and have equal length"
            )
        combined: list[dict[str, Any]] = []
        for index, (old_entry, new_entry) in enumerate(
            zip(old_entries, new_entries, strict=True)
        ):
            entry_role = f"source-rebind:proof-input:{day}:{index}"
            if not isinstance(old_entry, Mapping) or not isinstance(
                new_entry, Mapping
            ):
                raise ValueError(
                    f"{entry_role} old/new entries must be mappings"
                )
            _require_exact_keys(
                old_entry, source_fields, role=f"{entry_role}:old"
            )
            _require_exact_keys(
                new_entry, source_fields, role=f"{entry_role}:new"
            )
            for field in _IMMUTABLE_SOURCE_FIELDS:
                if old_entry[field] != new_entry[field]:
                    raise ValueError(
                        f"{entry_role} immutable field {field!r} changed"
                    )
            old_device = _require_int(
                old_entry["device"], role=f"{entry_role}:old-device"
            )
            new_device = _require_int(
                new_entry["device"], role=f"{entry_role}:new-device"
            )
            transition_counts[(old_device, new_device)] += 1
            combined.append(
                {
                    **{
                        field: old_entry[field]
                        for field in _IMMUTABLE_SOURCE_FIELDS
                    },
                    "old_device": old_device,
                    "new_device": new_device,
                }
            )
        raw_daily[day] = combined
    proof = {
        "immutable_fields": list(_IMMUTABLE_SOURCE_FIELDS),
        "changed_fields": list(_CHANGED_SOURCE_FIELDS),
        "device_transitions": [
            {
                "old_device": old_device,
                "new_device": new_device,
                "source_count": count,
            }
            for (old_device, new_device), count in sorted(
                transition_counts.items()
            )
        ],
        "daily_source_artifacts": raw_daily,
    }
    document = _build_attestation(
        old_manifest,
        new_manifest,
        proof,
        old_manifest_sha256=_file_sha256(old_path),
        new_manifest_sha256=_file_sha256(new_path),
    )
    destination = (
        Path(output_path)
        if output_path is not None
        else new_path.parent / ATTESTATION_FILENAME
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                document, stream, indent=2, ensure_ascii=False, allow_nan=False
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def validate_phase1_source_rebind_attestation(  # noqa: C901
    scalar_dir: Path | str,
    *,
    old_split_sha256: str,
    new_split_sha256: str,
    role: str,
) -> dict[str, Any]:
    """Authenticate the one permitted legacy source-signature rebind."""
    old_split = _require_sha256(old_split_sha256, role=f"{role}:old-split")
    new_split = _require_sha256(new_split_sha256, role=f"{role}:new-split")
    if old_split == new_split:
        raise ValueError(
            f"[{role}] source rebind requires different split hashes"
        )
    directory = Path(scalar_dir)
    manifest_path = directory / normalization.MANIFEST_NAME
    attestation_path = directory / ATTESTATION_FILENAME
    manifest = _load_json_mapping(manifest_path, role=f"{role}:manifest")
    document = _load_json_mapping(attestation_path, role=f"{role}:attestation")
    _require_exact_keys(
        document,
        (
            "schema_version",
            "attestation_type",
            "reason",
            "manifest_binding",
            "source_binding",
            "proof_sha256",
            "proof",
            "attestation_sha256",
        ),
        role=f"{role}:attestation",
    )
    if document["schema_version"] != ATTESTATION_SCHEMA_VERSION:
        raise ValueError(
            f"[{role}] unsupported source-rebind attestation schema"
        )
    if document["attestation_type"] != ATTESTATION_TYPE:
        raise ValueError(f"[{role}] invalid source-rebind attestation type")
    if document["reason"] != ATTESTATION_REASON:
        raise ValueError(f"[{role}] source-rebind reason is not device drift")

    recorded_attestation_sha = _require_sha256(
        document["attestation_sha256"], role=f"{role}:attestation-sha256"
    )
    unsigned = dict(document)
    unsigned.pop("attestation_sha256")
    computed_attestation_sha = _canonical_sha256(unsigned)
    if recorded_attestation_sha != computed_attestation_sha:
        raise ValueError(
            f"[{role}] source-rebind attestation SHA-256 mismatch"
        )

    manifest_binding = document["manifest_binding"]
    source_binding = document["source_binding"]
    proof = document["proof"]
    if not isinstance(manifest_binding, Mapping):
        raise ValueError(f"[{role}] manifest_binding must be a mapping")
    if not isinstance(source_binding, Mapping):
        raise ValueError(f"[{role}] source_binding must be a mapping")
    if not isinstance(proof, Mapping):
        raise ValueError(f"[{role}] proof must be a mapping")
    _require_exact_keys(
        manifest_binding,
        (
            "old_manifest_sha256",
            "new_manifest_sha256",
            "immutable_fields",
            "immutable_sha256",
        ),
        role=f"{role}:manifest-binding",
    )
    _require_exact_keys(
        source_binding,
        ("old_split_sha256", "new_split_sha256", "date_count", "source_count"),
        role=f"{role}:source-binding",
    )
    if manifest_binding["immutable_fields"] != list(
        _IMMUTABLE_MANIFEST_FIELDS
    ):
        raise ValueError(f"[{role}] immutable manifest fields differ")
    _require_sha256(
        manifest_binding["old_manifest_sha256"],
        role=f"{role}:old-manifest-sha256",
    )
    active_manifest_sha = _file_sha256(manifest_path)
    if manifest_binding["new_manifest_sha256"] != active_manifest_sha:
        raise ValueError(
            f"[{role}] source rebind is not bound to the active manifest"
        )
    active_projection_sha = _canonical_sha256(_manifest_projection(manifest))
    if manifest_binding["immutable_sha256"] != active_projection_sha:
        raise ValueError(f"[{role}] immutable manifest provenance has changed")

    normalized_proof, reconstructed_old_daily, reconstructed_new_daily = (
        _normalize_proof(proof, role=f"{role}:proof")
    )
    recorded_proof_sha = _require_sha256(
        document["proof_sha256"], role=f"{role}:proof-sha256"
    )
    if recorded_proof_sha != _canonical_sha256(normalized_proof):
        raise ValueError(f"[{role}] source-rebind proof SHA-256 mismatch")
    reconstructed_old_split = split_source_artifact_signature(
        reconstructed_old_daily
    )
    reconstructed_new_split = split_source_artifact_signature(
        reconstructed_new_daily
    )
    if source_binding["old_split_sha256"] != old_split:
        raise ValueError(
            f"[{role}] attested old split does not match checkpoint"
        )
    if source_binding["new_split_sha256"] != new_split:
        raise ValueError(
            f"[{role}] attested new split does not match active config"
        )
    if reconstructed_old_split != old_split:
        raise ValueError(
            f"[{role}] proof does not reconstruct checkpoint source split"
        )
    if reconstructed_new_split != new_split:
        raise ValueError(
            f"[{role}] proof does not reconstruct active source split"
        )
    if source_binding["date_count"] != len(reconstructed_old_daily):
        raise ValueError(f"[{role}] attested date count is invalid")
    reconstructed_source_count = sum(
        len(entries)
        for entries in normalized_proof["daily_source_artifacts"].values()
    )
    if source_binding["source_count"] != reconstructed_source_count:
        raise ValueError(f"[{role}] attested source count is invalid")

    manifest_daily, manifest_split = _manifest_source_contract(
        manifest, role=f"{role}:active-manifest"
    )
    if (
        manifest_daily != reconstructed_new_daily
        or manifest_split != new_split
    ):
        raise ValueError(
            f"[{role}] proof does not match every active-manifest daily "
            "source signature"
        )
    warnings.warn(
        f"[{role}] accepted attested PRISM source-device rebind "
        f"{old_split[:12]}->{new_split[:12]} using {attestation_path} "
        f"attestation_sha256={recorded_attestation_sha}",
        RuntimeWarning,
        stacklevel=2,
    )
    return document


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a strict device-only Phase-1 source rebind attestation"
        )
    )
    parser.add_argument("old_manifest")
    parser.add_argument("new_manifest")
    parser.add_argument("proof")
    parser.add_argument("--output")
    args = parser.parse_args()
    path = write_phase1_source_rebind_attestation(
        args.old_manifest, args.new_manifest, args.proof, args.output
    )
    print(path)


if __name__ == "__main__":
    _main()
