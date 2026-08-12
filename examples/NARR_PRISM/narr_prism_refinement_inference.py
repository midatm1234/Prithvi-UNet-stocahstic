"""Production daily inference for NARR--PRISM Phase-2 residual refiners.

The deterministic and stochastic models operate tile-by-tile, but all retained
products are stitched on the canonical PRISM grid.  Stochastic source fields
are generated from global pixel coordinates, so overlapping tiles see exactly
the same Gaussian value at the same location.  This prevents tile boundaries
from becoming boundaries in the stochastic forcing while still giving every
date and ensemble member an independent, reproducible stream.

Daily files are written below ``<output root>/<case_name>/`` using
``<case_name>_<refinement_type>_refined_YYYYMMDD.nc``. Evaluators should either
use this exact glob or consume the directory returned by
:func:`run_refined_inference`; it intentionally differs from deterministic
``*_inference_YYYYMMDD.nc`` so products cannot be mixed accidentally.
"""

from __future__ import annotations

import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for import_root in (REPO_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from granitewxc.refinement.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    REFINEMENT_SEMANTICS_VERSION,
    extract_model_state,
    load_refinement_state_dict,
    phase1_state_fingerprint,
    validate_phase1_reference,
)
from granitewxc.refinement.config import config_fingerprint
from granitewxc.refinement.io import write_refined_netcdf
from granitewxc.utils import normalization as normalization_contract
from granitewxc.utils.prism_grid import validate_prism_grid
from granitewxc.utils.prism_tiling import (
    TilePlan,
    WeightedTileStitcher,
    blend_window,
    extract_halo_context,
)

from narr_prism_dataset import NarrPrismDataset
from narr_prism_inference import (
    TARGET_VALID_MASK_CONTENT_SHA256_ATTR,
    TARGET_VALID_MASK_CRITERION_ATTR,
    TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR,
    TARGET_VALID_MASK_SHA256_ATTR,
    TARGET_VALID_MASK_SOURCE_SPLIT_ATTR,
    TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR,
    VAR_UNITS,
    _pad_multiple_from_config,
    _pad_to_multiple,
    _sanity_check_outputs,
    _split_output_root,
    _target_valid_mask_content_sha256,
    _target_valid_mask_provenance,
    _validate_dataset_dates,
    _validate_prediction_split,
)
from narr_prism_utils import (
    case_output_dir,
    get_case_name,
)

__all__ = [
    "CoordinateAlignedNoiseSource",
    "build_daily_refined_dataset",
    "derive_tile_seed",
    "run_refined_inference",
    "training_scaler_valid_mask",
    "validate_refinement_checkpoint_metadata",
]


def _date_key(value: Any) -> str:
    return str(value)[:10]


def _hash_seed(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "little"
    )


def derive_tile_seed(
    base_seed: int,
    sample_date: Any,
    tile_origin: Sequence[int],
    member: int,
) -> int:
    """Return a stable date/tile/member seed for provenance and diagnostics."""
    y0, x0 = (int(value) for value in tile_origin)
    return _hash_seed(
        "narr-prism-refinement",
        int(base_seed),
        _date_key(sample_date),
        y0,
        x0,
        int(member),
    )


def _coordinate_stream_seed(
    base_seed: int,
    sample_date: Any,
    member: int,
    draw_index: int,
) -> int:
    return _hash_seed(
        "narr-prism-global-noise",
        int(base_seed),
        _date_key(sample_date),
        int(member),
        int(draw_index),
    )


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 mixer with intentional unsigned wraparound."""
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    values = (values + np.uint64(0x9E3779B97F4A7C15)) & mask
    values = (
        (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    ) & mask
    values = (
        (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    ) & mask
    return values ^ (values >> np.uint64(31))


class CoordinateAlignedNoiseSource:
    """Stateless Gaussian draws aligned in the global PRISM coordinate frame.

    A fresh instance is used for each member and tile batch.  Its draw counter
    advances once for the diffusion initial state and once for every stochastic
    reverse step.  Because the counter and stream seed are date/member based,
    and pixels are keyed by their global indices, overlap values are independent
    of tile ordering, tile batching and tile origin.
    """

    def __init__(
        self,
        *,
        base_seed: int,
        sample_date: Any,
        member: int,
        origins: Sequence[Sequence[int]],
        domain_shape: Sequence[int],
    ) -> None:
        self.base_seed = int(base_seed)
        self.sample_date = _date_key(sample_date)
        self.member = int(member)
        self.origins = tuple((int(pos[0]), int(pos[1])) for pos in origins)
        self.domain_shape = (int(domain_shape[0]), int(domain_shape[1]))
        self.draw_index = 0

    def randn(self, shape: tuple[int, ...], device, dtype) -> torch.Tensor:
        if len(shape) != 4:
            raise ValueError(
                f"coordinate-aligned noise expects [B,C,H,W], got {shape}"
            )
        batch, channels, height, width = (int(value) for value in shape)
        if batch != len(self.origins):
            raise ValueError(
                f"noise batch {batch} does not match {len(self.origins)} tile origins"
            )
        domain_h, domain_w = self.domain_shape
        stream_seed = np.uint64(
            _coordinate_stream_seed(
                self.base_seed,
                self.sample_date,
                self.member,
                self.draw_index,
            )
        )
        result = np.empty((batch, channels, height, width), dtype=np.float32)
        channel_stride = np.uint64(domain_h * domain_w)
        for batch_index, (y0, x0) in enumerate(self.origins):
            if (
                y0 < 0
                or x0 < 0
                or y0 + height > domain_h
                or x0 + width > domain_w
            ):
                raise ValueError(
                    f"noise tile {(y0, x0, height, width)} exceeds domain {self.domain_shape}"
                )
            global_y = np.arange(y0, y0 + height, dtype=np.uint64)[:, None]
            global_x = np.arange(x0, x0 + width, dtype=np.uint64)[None, :]
            pixel = global_y * np.uint64(domain_w) + global_x
            for channel in range(channels):
                counter = pixel + np.uint64(channel) * channel_stride
                first = _splitmix64(counter ^ stream_seed)
                second = _splitmix64(
                    counter ^ stream_seed ^ np.uint64(0xD2B74407B1CE6E93)
                )
                # Convert the upper 53 bits into open-interval uniforms, then
                # use Box--Muller.  The same operations run for every tile.
                u1 = (
                    (first >> np.uint64(11)).astype(np.float64) + 0.5
                ) / float(2**53)
                u2 = (
                    (second >> np.uint64(11)).astype(np.float64) + 0.5
                ) / float(2**53)
                result[batch_index, channel] = (
                    np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
                ).astype(np.float32)
        self.draw_index += 1
        return torch.from_numpy(result).to(device=device, dtype=dtype)


def validate_refinement_checkpoint_metadata(payload: Any, model: Any) -> None:
    """Reject a Phase-2 checkpoint whose stochastic semantics do not match."""
    if not isinstance(payload, Mapping):
        raise ValueError(
            "Phase-2 checkpoint must be a mapping with provenance metadata"
        )
    schema = payload.get("checkpoint_schema_version")
    semantics = payload.get("refinement_semantics_version")
    if (
        schema != CHECKPOINT_SCHEMA_VERSION
        or semantics != REFINEMENT_SEMANTICS_VERSION
    ):
        raise ValueError(
            "Phase-2 checkpoint uses incompatible residual semantics: "
            f"schema={schema!r}, semantics={semantics!r}; expected "
            f"schema={CHECKPOINT_SCHEMA_VERSION}, "
            f"semantics={REFINEMENT_SEMANTICS_VERSION!r}. Legacy checkpoints "
            "trained against the hurdle amount latent must be retrained or "
            "explicitly migrated."
        )
    kind = payload.get("checkpoint_kind")
    if kind not in {"refinement", "combined"}:
        raise ValueError(
            "Phase-2 checkpoint_kind must be 'refinement' or 'combined', "
            f"got {kind!r}"
        )
    if kind == "combined":
        raise ValueError(
            "Combined Phase-1/Phase-2 checkpoints are not accepted by frozen "
            "NARR_PRISM refinement inference. Supply the original deterministic "
            "Phase-1 checkpoint plus a refinement-only Phase-2 checkpoint."
        )
    resolved = payload.get("resolved_config")
    observed = (
        resolved.get("refinement") if isinstance(resolved, Mapping) else None
    )
    if not isinstance(observed, Mapping):
        raise ValueError(
            "Phase-2 checkpoint has no resolved_config.refinement metadata; "
            "its stochastic objective cannot be verified"
        )
    expected = model.refinement_config.to_dict()
    if not bool(expected.get("train_on_residual", True)):
        raise ValueError(
            "Production NARR_PRISM inference requires "
            "model.refinement.train_on_residual=true"
        )
    if "type" not in observed:
        raise ValueError(
            "Phase-2 checkpoint metadata is missing refinement.type"
        )
    if "train_on_residual" not in observed:
        raise ValueError(
            "Phase-2 checkpoint metadata is missing refinement.train_on_residual"
        )
    if not bool(observed["train_on_residual"]):
        raise ValueError(
            "Phase-2 checkpoint was trained as an absolute target model; it cannot "
            "be used by residual-addition inference"
        )
    canonical_saved = dict(observed)
    for mutable in ("checkpoint", "ensemble_size"):
        canonical_saved.pop(mutable, None)
        expected.pop(mutable, None)
    if canonical_saved != expected:
        def differing_paths(
            saved: Mapping[str, Any],
            current: Mapping[str, Any],
            prefix: str = "refinement",
        ) -> list[tuple[str, Any, Any]]:
            differences: list[tuple[str, Any, Any]] = []
            for key in sorted(set(saved) | set(current)):
                path = f"{prefix}.{key}"
                saved_value = saved.get(key)
                current_value = current.get(key)
                if isinstance(saved_value, Mapping) and isinstance(
                    current_value, Mapping
                ):
                    differences.extend(
                        differing_paths(saved_value, current_value, path)
                    )
                elif saved_value != current_value:
                    differences.append((path, saved_value, current_value))
            return differences

        differing = differing_paths(canonical_saved, expected)
        details = "; ".join(
            f"{path}: checkpoint={saved_value!r}, config={current_value!r}"
            for path, saved_value, current_value in differing
        )
        raise ValueError(
            f"Phase-2 resolved refinement configuration mismatch ({details})"
        )


def _refinement_state_fingerprint(payload: Mapping[str, Any]) -> str:
    state = extract_model_state(payload)
    refiner_state = {
        key: value
        for key, value in state.items()
        if key.startswith("refiner.")
    }
    if not refiner_state:
        refiner_state = state
    return phase1_state_fingerprint(refiner_state)


def _verify_phase2_artifact_fingerprint(
    payload: Mapping[str, Any], computed: str
) -> None:
    """Validate a stored Phase-2 tensor fingerprint when the artifact has one."""
    saved = payload.get("refinement_fingerprint") or payload.get(
        "phase2_fingerprint"
    )
    if saved is not None and str(saved) != computed:
        raise ValueError(
            "Phase-2 checkpoint tensor fingerprint mismatch: "
            f"stored={saved!r}, computed={computed!r}"
        )


def training_scaler_valid_mask(
    config: Any, domain_shape: Sequence[int]
) -> np.ndarray:
    """Load the signed static output mask derived from training targets.

    Spatial target scalers neutral-fill unsupported cells with mean=0/std=1,
    so their finiteness cannot encode PRISM support. Production inference must
    authenticate the separate mask artifact fitted from 1996--2013 and must
    never inspect a 2016--2025 observed target to decide output validity.
    """
    expected_shape = (int(domain_shape[0]), int(domain_shape[1]))
    return normalization_contract.load_target_valid_mask(
        config,
        role="NARR refinement inference",
        expected_shape=expected_shape,
    )


def _inference_plan(
    cfg: Mapping[str, Any], config: Any, domain_shape: Sequence[int]
) -> TilePlan:
    inf_cfg = dict(cfg.get("inference", {}) or {})
    boundary = dict(inf_cfg.get("boundary_mitigation", {}) or {})
    tile = (
        inf_cfg.get("inference_tile_size")
        or boundary.get("tile_size")
        or [256, 256]
    )
    overlap = (
        inf_cfg.get("inference_overlap") or boundary.get("overlap") or [64, 64]
    )
    halo = inf_cfg.get("inference_halo") or boundary.get("halo") or [32, 32]
    if bool(
        inf_cfg.get(
            "force_full_frame", boundary.get("force_full_frame", False)
        )
    ):
        tile = list(domain_shape)
        overlap = [0, 0]
    plan = TilePlan.build(domain_shape, tile, overlap=overlap, halo=halo)
    if (
        str(
            getattr(config.model, "backbone_attention_scope", "legacy_global")
        ).lower()
        == "windowed_local"
    ):
        plan.assert_globally_aligned(
            getattr(config, "mask_unit_size", [16, 16])
        )
    return plan


def _make_tile_batch(
    day_predictors: torch.Tensor,
    positions: Sequence[tuple[int, int]],
    plan: TilePlan,
    *,
    n_targets: int,
    pad_multiple: int,
    device: torch.device,
    domain_valid_mask: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    contexts = [
        _pad_to_multiple(
            extract_halo_context(
                day_predictors, origin, plan.core_shape, plan.halo
            ).unsqueeze(0),
            pad_multiple,
        )
        for origin in positions
    ]
    x_full = torch.cat(contexts, dim=0)
    offsets = torch.tensor(positions, dtype=torch.long, device=device)
    halo = torch.tensor(plan.halo, dtype=torch.long, device=device)
    batch = {
        "x": x_full.to(device, non_blocking=True),
        # Phase 1 uses y only for the exact output-core shape. No observed PRISM
        # values are loaded or exposed during independent inference.
        "y": torch.zeros(
            (len(positions), n_targets, *plan.core_shape),
            dtype=x_full.dtype,
            device=device,
        ),
        "__scaler_offset": offsets,
        "__input_scaler_offset": offsets - halo,
        "__output_scaler_offset": offsets,
        "__output_crop": torch.tensor(
            [plan.output_crop] * len(positions),
            dtype=torch.long,
            device=device,
        ),
    }
    if domain_valid_mask is not None:
        spatial_mask = np.asarray(domain_valid_mask, dtype=bool)
        mask_tiles = [
            spatial_mask[
                top : top + plan.core_shape[0],
                left : left + plan.core_shape[1],
            ]
            for top, left in positions
        ]
        batch["__target_valid_mask"] = (
            torch.as_tensor(np.stack(mask_tiles), device=device)
            .unsqueeze(1)
            .expand(-1, n_targets, -1, -1)
        )
    return batch


def _predict_tile_ensemble(
    model: Any,
    batch: Mapping[str, torch.Tensor],
    *,
    positions: Sequence[tuple[int, int]],
    domain_shape: Sequence[int],
    sample_date: Any,
    ensemble_size: int,
    base_seed: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    deterministic, normalized, conditioning = model._prepare(batch)
    if model.refiner is None:
        raise RuntimeError(
            "active refinement model did not initialize its refiner"
        )
    expected_shape = tuple(batch["y"].shape[-2:])
    if tuple(deterministic.shape[-2:]) != expected_shape:
        raise ValueError(
            f"Phase-1 tile output {tuple(deterministic.shape[-2:])} != core {expected_shape}"
        )
    member_fields: list[torch.Tensor] = []
    offset = batch["__output_scaler_offset"]
    target_mask = batch.get("__target_valid_mask")
    for member in range(ensemble_size):
        source = CoordinateAlignedNoiseSource(
            base_seed=base_seed,
            sample_date=sample_date,
            member=member,
            origins=positions,
            domain_shape=domain_shape,
        )
        with model.refiner.use_noise_source(source):
            residual = model.refiner.sample_target_space(
                conditioning, valid_mask=target_mask
            )
        # Refiners standardize Phase-1 residuals with training-period-only
        # statistics. sample_target_space applies that inverse exactly once.
        residual = residual * float(model.refinement_config.correction_scale)
        _, physical = model.target_space.reconstruct(
            normalized,
            residual.to(normalized.dtype),
            scaler_offset=offset,
            target_mask=None,
        )
        member_fields.append(physical)
    return deterministic, member_fields


def build_daily_refined_dataset(  # noqa: C901
    *,
    variables: Sequence[str],
    sample_date: Any,
    lat: np.ndarray,
    lon: np.ndarray,
    valid_mask: np.ndarray,
    phase1: np.ndarray,
    members: np.ndarray,
    attrs: Mapping[str, Any],
):
    """Build one canonical daily dataset using the production naming contract.

    ``<var>`` is always the refined ensemble mean.  The deterministic baseline
    and individual members are explicit suffixed variables, never aliases that
    depend on reader-side interpretation.
    """
    import xarray as xr

    variables = [str(value) for value in variables]
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    phase1 = np.asarray(phase1, dtype=np.float32)
    members = np.asarray(members, dtype=np.float32)
    expected_phase1 = (len(variables), len(lat), len(lon))
    if phase1.shape != expected_phase1:
        raise ValueError(f"phase1 shape {phase1.shape} != {expected_phase1}")
    if members.ndim != 4 or members.shape[1:] != expected_phase1:
        raise ValueError(
            f"members must be [member,C,H,W] ending in {expected_phase1}, got {members.shape}"
        )
    if members.shape[0] < 1:
        raise ValueError("at least one refinement ensemble member is required")
    if valid_mask.shape != expected_phase1[-2:]:
        raise ValueError(
            f"valid_mask shape {valid_mask.shape} != {expected_phase1[-2:]}"
        )
    required_mask_attrs = (
        TARGET_VALID_MASK_SHA256_ATTR,
        TARGET_VALID_MASK_CRITERION_ATTR,
        TARGET_VALID_MASK_SOURCE_SPLIT_ATTR,
        TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR,
        TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR,
    )
    missing_mask_attrs = [
        name for name in required_mask_attrs if name not in attrs
    ]
    if missing_mask_attrs:
        raise ValueError(
            "refined daily output lacks target-valid-mask provenance "
            "attributes: "
            f"{missing_mask_attrs}"
        )
    attrs = dict(attrs)
    computed_mask_digest = _target_valid_mask_content_sha256(valid_mask)
    supplied_mask_digest = attrs.get(TARGET_VALID_MASK_CONTENT_SHA256_ATTR)
    if (
        supplied_mask_digest is not None
        and str(supplied_mask_digest) != computed_mask_digest
    ):
        raise ValueError(
            "refined daily output target-valid-mask content digest does not "
            "match "
            "the supplied mask cells"
        )
    attrs[TARGET_VALID_MASK_CONTENT_SHA256_ATTR] = computed_mask_digest
    output_grid_fingerprint = validate_prism_grid(
        lat, lon, context="NARR refined inference NetCDF"
    ).fingerprint
    if (
        str(attrs[TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR])
        != output_grid_fingerprint
    ):
        raise ValueError(
            "refined daily output target-valid-mask grid fingerprint does not "
            "match its output coordinates"
        )

    refined_mean = np.mean(members.astype(np.float64), axis=0).astype(
        np.float32
    )
    spread = (
        np.std(members.astype(np.float64), axis=0, ddof=1).astype(np.float32)
        if members.shape[0] > 1
        else None
    )
    phase1 = phase1.copy()
    members = members.copy()
    refined_mean = refined_mean.copy()
    phase1[:, ~valid_mask] = np.nan
    members[:, :, ~valid_mask] = np.nan
    refined_mean[:, ~valid_mask] = np.nan
    if spread is not None:
        spread[:, ~valid_mask] = np.nan

    coords = {
        "time": np.asarray([np.datetime64(_date_key(sample_date), "ns")]),
        "lat": lat,
        "lon": lon,
    }
    data_vars: dict[str, Any] = {
        "prism_valid_mask": xr.DataArray(
            valid_mask.astype(np.uint8),
            dims=("lat", "lon"),
            attrs={
                "long_name": "canonical PRISM valid-cell mask",
                "flag_values": np.asarray([0, 1], dtype=np.uint8),
                "flag_meanings": "invalid valid",
            },
        )
    }
    for channel, variable in enumerate(variables):
        unit = VAR_UNITS.get(variable, "")
        data_vars[variable] = xr.DataArray(
            refined_mean[None, channel],
            dims=("time", "lat", "lon"),
            attrs={
                "units": unit,
                "long_name": f"{variable} refined ensemble mean",
            },
        )
        data_vars[f"{variable}_phase1"] = xr.DataArray(
            phase1[None, channel],
            dims=("time", "lat", "lon"),
            attrs={
                "units": unit,
                "long_name": f"{variable} deterministic Phase-1 baseline",
            },
        )
        for member in range(members.shape[0]):
            data_vars[f"{variable}_member_{member:03d}"] = xr.DataArray(
                members[member : member + 1, channel],
                dims=("time", "lat", "lon"),
                attrs={
                    "units": unit,
                    "long_name": f"{variable} refined member {member}",
                },
            )
        if spread is not None:
            data_vars[f"{variable}_ensemble_spread"] = xr.DataArray(
                spread[None, channel],
                dims=("time", "lat", "lon"),
                attrs={
                    "units": unit,
                    "long_name": f"{variable} ensemble standard deviation",
                },
            )

    dataset = xr.Dataset(data_vars=data_vars, coords=coords)
    dataset.attrs.update({str(key): value for key, value in attrs.items()})
    dataset.attrs["prism_grid_fingerprint"] = output_grid_fingerprint
    dataset.attrs["output_variable_contract"] = (
        "<var>=refined ensemble mean; <var>_phase1=deterministic baseline; "
        "<var>_member_NNN=refined member"
    )
    return dataset


def _load_phase2(
    model: Any,
    checkpoint_path: str,
) -> tuple[str, Mapping[str, Any]]:
    payload = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    # Check stochastic objective/config before applying any Phase-2 tensor.
    validate_refinement_checkpoint_metadata(payload, model)
    phase1_state = {
        key[len("phase1.") :]: value
        for key, value in model.state_dict().items()
        if key.startswith("phase1.")
    }
    validate_phase1_reference(payload, phase1_state, strict=True)
    report = load_refinement_state_dict(model, payload)
    print(f"[refinement] Phase-2 load: {report.summary()}")
    fingerprint = _refinement_state_fingerprint(payload)
    _verify_phase2_artifact_fingerprint(payload, fingerprint)
    return fingerprint, payload


def run_refined_inference(
    *,
    config_path: str,
    cfg: Mapping[str, Any],
    config: Any,
    model: Any,
    phase1_checkpoint: str,
    phase1_fingerprint: str,
    refinement_checkpoint: str,
    output_dir: str,
    device: torch.device,
    ensemble_size: int,
    base_seed: int | None,
    batch_size: int = 1,
    limit_days: int = 0,
    split: str = "inference",
) -> Path:
    """Run exact-date daily Phase-1 + Phase-2 tiled inference."""
    split = _validate_prediction_split(split)
    ensemble_size = int(ensemble_size)
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be >= 1")
    if base_seed is None:
        raise ValueError(
            "production refinement inference requires an explicit reproducible seed"
        )
    base_seed = int(base_seed)
    batch_size = max(1, int(batch_size))
    resolved_config_path = str(Path(config_path).expanduser().resolve())
    active_config_fingerprint = config_fingerprint(cfg)

    dataset = NarrPrismDataset(
        config_path, mode=split, load_observed_targets=False
    )
    configured_split_dates = _validate_dataset_dates(
        cfg, dataset.dates, split
    )
    inference_dates = configured_split_dates
    if limit_days:
        inference_dates = inference_dates[: int(limit_days)]
    if not inference_dates:
        raise ValueError(f"configured {split} period contains no samples")

    variables = list(config.data.output_vars)
    if variables != list(cfg.get("data", {}).get("target_variables", [])):
        raise ValueError(
            "data.output_vars and data.target_variables must have identical explicit order"
        )
    target_lat = np.asarray(dataset.fine_lat, dtype=np.float64)
    target_lon = np.asarray(dataset.fine_lon, dtype=np.float64)
    domain_shape = tuple(dataset.fine_shape)
    valid_mask = training_scaler_valid_mask(config, domain_shape)
    if valid_mask.shape != domain_shape:
        raise ValueError(
            f"canonical mask {valid_mask.shape} != domain {domain_shape}"
        )
    target_valid_mask_provenance = _target_valid_mask_provenance(
        config, valid_mask
    )

    plan = _inference_plan(cfg, config, domain_shape)
    inf_cfg = dict(cfg.get("inference", {}) or {})
    boundary = dict(inf_cfg.get("boundary_mitigation", {}) or {})
    blend_mode = str(
        inf_cfg.get(
            "inference_blend_window",
            inf_cfg.get("blend_window", boundary.get("blend_mode", "hann")),
        )
    ).lower()
    window = blend_window(plan.core_shape, plan.overlap, mode=blend_mode)
    min_valid = float(
        cfg.get("data", {}).get("min_valid_target_fraction", 1.0e-4)
    )
    skip_empty = bool(cfg.get("data", {}).get("skip_empty_target_tiles", True))
    positions = [
        position
        for position in plan.positions
        if not skip_empty
        or float(
            valid_mask[
                position[0] : position[0] + plan.core_shape[0],
                position[1] : position[1] + plan.core_shape[1],
            ].mean()
        )
        >= min_valid
    ]
    if not positions:
        raise ValueError(
            "no refinement inference tiles remain after mask filtering"
        )

    # Initialize the lazy head with production inference geometry, then verify
    # and load its checkpoint. This never uses validation targets.
    first_predictors = dataset._load_predictor_day(inference_dates[0])
    probe = _make_tile_batch(
        first_predictors,
        positions[:1],
        plan,
        n_targets=len(variables),
        pad_multiple=_pad_multiple_from_config(config),
        device=device,
        domain_valid_mask=valid_mask,
    )
    model.initialize_from_batch(probe)
    model.to(device).eval()
    phase2_fingerprint, _ = _load_phase2(
        model,
        refinement_checkpoint,
    )
    del probe, first_predictors

    case_name = get_case_name(cfg)
    output_path = case_output_dir(_split_output_root(output_dir, split), case_name)
    output_path.mkdir(parents=True, exist_ok=True)
    pad_multiple = _pad_multiple_from_config(config)
    mixed_precision = device.type == "cuda" and str(
        model.performance_config.precision.mode
    ).lower() in {"bf16", "fp16"}
    amp_dtype = (
        torch.bfloat16
        if str(model.performance_config.precision.mode).lower() == "bf16"
        else torch.float16
    )

    print(
        f"[refinement] split={split} dates={len(inference_dates)} "
        f"grid={domain_shape} "
        f"tiles={len(positions)} core={plan.core_shape} halo={plan.halo} "
        f"ensemble={ensemble_size}"
    )
    with torch.inference_mode():
        for sample_date in inference_dates:
            day_predictors = dataset._load_predictor_day(sample_date)
            baseline_stitcher = WeightedTileStitcher(
                len(variables), domain_shape
            )
            member_stitchers = [
                WeightedTileStitcher(len(variables), domain_shape)
                for _ in range(ensemble_size)
            ]
            for start in range(0, len(positions), batch_size):
                chunk = positions[start : start + batch_size]
                batch = _make_tile_batch(
                    day_predictors,
                    chunk,
                    plan,
                    n_targets=len(variables),
                    pad_multiple=pad_multiple,
                    device=device,
                    domain_valid_mask=valid_mask,
                )
                amp_context = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if mixed_precision
                    else nullcontext()
                )
                with amp_context:
                    phase1_tiles, member_tiles = _predict_tile_ensemble(
                        model,
                        batch,
                        positions=chunk,
                        domain_shape=domain_shape,
                        sample_date=sample_date,
                        ensemble_size=ensemble_size,
                        base_seed=base_seed,
                    )
                phase1_np = phase1_tiles.detach().float().cpu().numpy()
                members_np = [
                    value.detach().float().cpu().numpy()
                    for value in member_tiles
                ]
                for local_index, position in enumerate(chunk):
                    baseline_stitcher.add(
                        phase1_np[local_index], position, window
                    )
                    for member, stitcher in enumerate(member_stitchers):
                        stitcher.add(
                            members_np[member][local_index], position, window
                        )
                del batch, phase1_tiles, member_tiles, phase1_np, members_np

            if np.any(valid_mask & (baseline_stitcher.weight <= 0.0)):
                raise ValueError(
                    f"{sample_date}: tiled Phase-1 output left valid cells uncovered"
                )
            for member, stitcher in enumerate(member_stitchers):
                if np.any(valid_mask & (stitcher.weight <= 0.0)):
                    raise ValueError(
                        f"{sample_date}: refined member {member} left valid cells uncovered"
                    )
            phase1 = baseline_stitcher.finalize(
                require_full_coverage=False
            ).astype(np.float32)
            members = np.stack(
                [
                    stitcher.finalize(require_full_coverage=False)
                    for stitcher in member_stitchers
                ]
            ).astype(np.float32)
            mean = members.astype(np.float64).mean(axis=0).astype(np.float32)
            _sanity_check_outputs(
                phase1[None], variables, f"{sample_date} Phase-1"
            )
            _sanity_check_outputs(
                mean[None], variables, f"{sample_date} refined mean"
            )
            temperature_order_diagnostics: dict[str, float] = {}
            if "tmin" in variables and "tmax" in variables:
                tmin_index = variables.index("tmin")
                tmax_index = variables.index("tmax")
                valid_temperature = (
                    valid_mask
                    & np.isfinite(mean[tmin_index])
                    & np.isfinite(mean[tmax_index])
                )
                if np.any(valid_temperature):
                    temperature_order_diagnostics[
                        "refined_tmin_gt_tmax_rate"
                    ] = float(
                        np.mean(
                            mean[tmin_index][valid_temperature]
                            > mean[tmax_index][valid_temperature]
                        )
                    )
                    member_rates = []
                    for member_field in members:
                        member_valid = (
                            valid_mask
                            & np.isfinite(member_field[tmin_index])
                            & np.isfinite(member_field[tmax_index])
                        )
                        if np.any(member_valid):
                            member_rates.append(
                                float(
                                    np.mean(
                                        member_field[tmin_index][member_valid]
                                        > member_field[tmax_index][member_valid]
                                    )
                                )
                            )
                    if member_rates:
                        temperature_order_diagnostics[
                            "max_member_tmin_gt_tmax_rate"
                        ] = max(member_rates)

            coordinate_seed_digest = hashlib.sha256(
                json.dumps(
                    [
                        _coordinate_stream_seed(
                            base_seed, sample_date, member, 0
                        )
                        for member in range(ensemble_size)
                    ]
                ).encode("utf-8")
            ).hexdigest()
            attrs = {
                "description": "NARR-to-PRISM Phase-2 residual refinement",
                "case_name": case_name,
                "refinement_type": model.refinement_config.type,
                "ensemble_size": ensemble_size,
                "base_seed": base_seed,
                "seed_scheme": (
                    "BLAKE2b(base_seed,date,member,draw)+global PRISM pixel index; "
                    "coordinate-aligned across overlapping tiles"
                ),
                "coordinate_seed_digest": coordinate_seed_digest,
                "phase1_checkpoint": str(phase1_checkpoint),
                "phase2_checkpoint": str(refinement_checkpoint),
                # Compatibility alias used by the existing PRISM evaluator:
                # for a refined product, the active checkpoint is Phase 2.
                "checkpoint": str(refinement_checkpoint),
                "phase1_fingerprint": str(phase1_fingerprint),
                "phase2_fingerprint": phase2_fingerprint,
                "config_path": resolved_config_path,
                "config_fingerprint": active_config_fingerprint,
                "dataset_split": split,
                "split_start": _date_key(configured_split_dates[0]),
                "split_end": _date_key(configured_split_dates[-1]),
                "inference_date": _date_key(sample_date),
                "variable_order": json.dumps(variables),
                "daily_file_pattern": (
                    f"{case_name}_{model.refinement_config.type}_refined_YYYYMMDD.nc"
                ),
                **target_valid_mask_provenance,
                **temperature_order_diagnostics,
            }
            daily = build_daily_refined_dataset(
                variables=variables,
                sample_date=sample_date,
                lat=target_lat,
                lon=target_lon,
                valid_mask=valid_mask,
                phase1=phase1,
                members=members,
                attrs=attrs,
            )
            date_token = _date_key(sample_date).replace("-", "")
            output_file = output_path / (
                f"{case_name}_{model.refinement_config.type}_refined_{date_token}.nc"
            )
            io_cfg = model.performance_config.io
            write_refined_netcdf(
                daily,
                output_file,
                compression=io_cfg.netcdf_compression,
                compression_level=io_cfg.netcdf_compression_level,
                chunk_sizes={"time": 1},
                atomic=True,
            )
            daily.close()
            print(f"[refinement] wrote {output_file}")
            del (
                day_predictors,
                baseline_stitcher,
                member_stitchers,
                phase1,
                members,
                mean,
            )

    return output_path
