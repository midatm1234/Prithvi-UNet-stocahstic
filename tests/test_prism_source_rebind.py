import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from granitewxc.utils import normalization
from granitewxc.utils.prism_checkpoint import (
    CONTRACT_KEY,
    build_prism_checkpoint_contract,
    validate_prism_checkpoint_contract,
)
from granitewxc.utils.prism_preprocessed import (
    preprocessing_signature,
    split_source_artifact_signature,
)
from granitewxc.utils.prism_source_rebind import (
    ATTESTATION_FILENAME,
    write_phase1_source_rebind_attestation,
)


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        case_name="narr_case",
        data=SimpleNamespace(
            type="narr_prism",
            preprocessed_dir=str(tmp_path / "preprocessed"),
            scalar_dir=str(tmp_path / "unused_scalars"),
            use_preprocessed=True,
            train_crop_size_lat=8,
            train_crop_size_lon=8,
            training_tile_stride_lat=8,
            training_tile_stride_lon=8,
            training_halo_lat=0,
            training_halo_lon=0,
            regrid_method="bilinear",
            input_vars=["air_850", "mask_air_850"],
            input_levels=[1],
            output_vars=["ppt", "tmax"],
            target_variables=["ppt", "tmax"],
            predictor_variables={"air": [850]},
            n_input_timestamps=1,
            use_static=False,
        ),
        model=SimpleNamespace(
            decoder_skip_source="dynamic",
            backbone_attention_scope="windowed_local",
            backbone_residual_mode="pre_conv_add",
            residual_connection=True,
            decoder_upsampling_mode="bilinear",
            embed_dim=32,
        ),
        dates=SimpleNamespace(
            training=SimpleNamespace(start="2000-01-01", end="2000-01-02"),
            validation=SimpleNamespace(start="2001-01-01", end="2001-01-01"),
        ),
        mask_unit_size=[4, 4],
        backbone_use=True,
        predictands={
            "ppt": {
                "normalization": {
                    "method": "divide_only",
                    "mode": "global",
                    "scale_stat": "p95",
                }
            },
            "tmax": {
                "nonnegativity": {"enabled": False, "method": "none"},
                "normalization": {
                    "method": "standardize",
                    "mode": "gridpoint",
                },
            },
        },
        precip_model="hurdle",
    )


def _source_entry(*, day: int, label: str, device: int) -> dict:
    label_offset = 0 if label == "predictor:air" else 100
    return {
        "device": device,
        "inode": 10_000 + day + label_offset,
        "label": label,
        "mtime_ns": 1_700_000_000_000_000_000 + day + label_offset,
        "path": f"/archive/{label.replace(':', '_')}/2000010{day}.nc",
        "size": 1_000_000 + day + label_offset,
    }


def _daily_signature(entries: list[dict]) -> str:
    return preprocessing_signature({"source_artifacts": entries})


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_and_proof(tmp_path: Path):
    old_device, new_device = 2097, 2065
    proof_days = {}
    old_daily = {}
    new_daily = {}
    for day_index, day in enumerate(("2000-01-01", "2000-01-02"), start=1):
        old_entries = [
            _source_entry(
                day=day_index, label="predictor:air", device=old_device
            ),
            _source_entry(
                day=day_index, label="target:ppt", device=old_device
            ),
        ]
        new_entries = [
            {**entry, "device": new_device} for entry in old_entries
        ]
        proof_days[day] = {"old": old_entries, "new": new_entries}
        old_daily[day] = _daily_signature(old_entries)
        new_daily[day] = _daily_signature(new_entries)

    immutable = {
        "schema_version": 3,
        "created_at": "ignored-during-device-rebind",
        "case_name": "narr_case",
        "predictor_mode": "global",
        "target_mode": "spatial",
        "train_date_range": ["2000-01-01", "2000-01-02"],
        "scalers": {},
        "config_contract": {"sentinel": "same"},
        "prism_grid": {"fingerprint": "a" * 64},
        "scaling_statistics": {"ppt": {"p95": 7.0}},
        "predictor_preprocessing_signature": "b" * 64,
    }
    old_manifest = {
        **immutable,
        normalization.TRAINING_SOURCE_ARTIFACT_SIGNATURES_KEY: old_daily,
        normalization.TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY: (
            split_source_artifact_signature(old_daily)
        ),
    }
    new_manifest = {
        **immutable,
        "created_at": "new-generation-time",
        normalization.TRAINING_SOURCE_ARTIFACT_SIGNATURES_KEY: new_daily,
        normalization.TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY: (
            split_source_artifact_signature(new_daily)
        ),
    }
    old_path = tmp_path / "old_manifest.json"
    proof_path = tmp_path / "source_rebind_contract.json"
    old_path.write_text(json.dumps(old_manifest, indent=2), encoding="utf-8")
    proof_path.write_text(
        json.dumps({"schema_version": 1, "days": proof_days}, indent=2),
        encoding="utf-8",
    )
    return old_manifest, new_manifest, old_path, proof_path


def _write_active_manifest(config, manifest: dict) -> Path:
    scalar_dir = normalization.resolve_scalar_dir(config, for_writing=True)
    scalar_dir.mkdir(parents=True, exist_ok=True)
    path = scalar_dir / normalization.MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _schema4_checkpoint(config, old_split: str) -> tuple[dict, dict]:
    current = build_prism_checkpoint_contract(config)
    assert current is not None
    legacy = copy.deepcopy(current)
    legacy["schema_version"] = 4
    legacy["coordinates_and_artifacts"].pop("target_valid_mask", None)
    legacy["coordinates_and_artifacts"][
        normalization.TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY
    ] = old_split
    return {CONTRACT_KEY: legacy}, legacy


def test_schema4_checkpoint_accepts_only_exact_attested_device_rebind(
    tmp_path,
):
    config = _config(tmp_path)
    old_manifest, new_manifest, old_path, proof_path = _manifest_and_proof(
        tmp_path
    )
    new_path = _write_active_manifest(config, new_manifest)
    checkpoint, legacy = _schema4_checkpoint(
        config,
        old_manifest[
            normalization.TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY
        ],
    )

    with pytest.raises(ValueError, match="attestation is missing or invalid"):
        validate_prism_checkpoint_contract(config, checkpoint, role="phase1")

    attestation_path = write_phase1_source_rebind_attestation(
        old_path, new_path, proof_path
    )
    assert attestation_path.name == ATTESTATION_FILENAME
    with pytest.warns(
        RuntimeWarning, match="accepted attested.*source-device rebind"
    ):
        assert (
            validate_prism_checkpoint_contract(
                config, checkpoint, role="phase1"
            )
            == legacy
        )

    changed_channel = copy.deepcopy(checkpoint)
    changed_channel[CONTRACT_KEY]["channels"]["output_vars"] = ["tmax", "ppt"]
    with pytest.raises(ValueError, match="Only an exactly attested"):
        validate_prism_checkpoint_contract(
            config, changed_channel, role="phase1"
        )

    wrong_old_split = copy.deepcopy(checkpoint)
    wrong_old_split[CONTRACT_KEY]["coordinates_and_artifacts"][
        normalization.TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY
    ] = "f" * 64
    with pytest.raises(ValueError, match="attested old split does not match"):
        validate_prism_checkpoint_contract(
            config, wrong_old_split, role="phase1"
        )


def test_schema4_checkpoint_rejects_tampered_attestation_or_manifest(tmp_path):
    config = _config(tmp_path)
    old_manifest, new_manifest, old_path, proof_path = _manifest_and_proof(
        tmp_path
    )
    new_path = _write_active_manifest(config, new_manifest)
    checkpoint, _ = _schema4_checkpoint(
        config,
        old_manifest[
            normalization.TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY
        ],
    )
    attestation_path = write_phase1_source_rebind_attestation(
        old_path, new_path, proof_path
    )
    original_attestation = attestation_path.read_text(encoding="utf-8")
    tampered = json.loads(original_attestation)
    tampered["proof"]["daily_source_artifacts"]["2000-01-01"][0]["size"] += 1
    attestation_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="attestation SHA-256 mismatch"):
        validate_prism_checkpoint_contract(config, checkpoint, role="phase1")

    # Re-hashing a modified document cannot forge the old source signatures:
    # those are cryptographic preimages embedded in the Phase-1 checkpoint.
    tampered["proof_sha256"] = _canonical_sha256(tampered["proof"])
    unsigned = dict(tampered)
    unsigned.pop("attestation_sha256")
    tampered["attestation_sha256"] = _canonical_sha256(unsigned)
    attestation_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(
        ValueError, match="does not reconstruct checkpoint source split"
    ):
        validate_prism_checkpoint_contract(config, checkpoint, role="phase1")

    attestation_path.write_text(original_attestation, encoding="utf-8")
    active = json.loads(new_path.read_text(encoding="utf-8"))
    active["created_at"] = "tampered-after-attestation"
    new_path.write_text(json.dumps(active, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="not bound to the active manifest"):
        validate_prism_checkpoint_contract(config, checkpoint, role="phase1")


def test_attestation_writer_rejects_any_non_device_metadata_change(tmp_path):
    config = _config(tmp_path)
    _, new_manifest, old_path, proof_path = _manifest_and_proof(tmp_path)
    new_path = _write_active_manifest(config, new_manifest)
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["days"]["2000-01-01"]["new"][0]["inode"] += 1
    proof_path.write_text(json.dumps(proof), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable field 'inode' changed"):
        write_phase1_source_rebind_attestation(old_path, new_path, proof_path)
