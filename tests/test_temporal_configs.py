"""The four shipped temporal case configs must be real, valid, and comparable.

Checks:

* every one parses through the *actual* validator (so no unimplemented or
  silently-ignored keys),
* each pair differs ONLY in backend selection and output paths, so a backend
  comparison is not confounded by anything else,
* each config preserves its source case's variables, ordering, domain, periods
  and checkpoint references,
* the legacy spatial configs still parse and still resolve to "no temporal".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from granitewxc.temporal.config import parse_temporal_config

REPO = Path(__file__).resolve().parents[1]

SA_RECURRENT = REPO / "examples/CORDEX_ML/SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_recurrent.yaml"
SA_MAMBA = REPO / "examples/CORDEX_ML/SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_mamba.yaml"
NARR_RECURRENT = REPO / "examples/NARR_PRISM/NARR_PRISM_subdomain_temporal_recurrent.yaml"
NARR_MAMBA = REPO / "examples/NARR_PRISM/NARR_PRISM_subdomain_temporal_mamba.yaml"

ALL_TEMPORAL = [SA_RECURRENT, SA_MAMBA, NARR_RECURRENT, NARR_MAMBA]

SA_SOURCE = REPO / "examples/CORDEX_ML/SA_T2_ACCESS-CM2_static_diffusion_unet.yaml"
NARR_SOURCE = REPO / "examples/NARR_PRISM/NARR_PRISM_subdomain.yaml"


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, inner in value.items():
            out.update(_flatten(inner, f"{prefix}.{key}" if prefix else str(key)))
    else:
        out[prefix] = value
    return out


@pytest.mark.parametrize("path", ALL_TEMPORAL, ids=lambda p: p.stem)
def test_config_file_exists(path):
    assert path.is_file(), f"missing shipped config: {path}"


@pytest.mark.parametrize("path", ALL_TEMPORAL, ids=lambda p: p.stem)
def test_temporal_block_parses_through_the_real_validator(path):
    """No key may be unimplemented: the validator rejects anything it cannot consume."""
    raw = _load(path)
    cfg = parse_temporal_config(raw.get("temporal"))
    assert cfg is not None and cfg.enabled
    assert cfg.mode == "downscaling"
    assert cfg.causal is True
    assert cfg.lead_time_days == 0.0
    assert cfg.cadence_days == 1.0
    assert cfg.backend in {"recurrent", "mamba"}
    # Geometry is self-consistent.
    assert cfg.warmup_length + cfg.output_length <= cfg.context_length
    assert cfg.output_length <= cfg.context_length
    # A checkpoint source is declared.
    assert cfg.init_from_spatial_checkpoint or cfg.resume_from_temporal_checkpoint
    # Round-trips: to_dict output re-parses to the same thing.
    again = parse_temporal_config({**cfg.to_dict(), **{
        "time_features": cfg.to_dict()["time_features"]}})
    assert again.backend == cfg.backend
    assert again.context_length == cfg.context_length


@pytest.mark.parametrize("path", ALL_TEMPORAL, ids=lambda p: p.stem)
def test_case_name_and_required_top_level_keys(path):
    raw = _load(path)
    assert raw.get("case_name"), "case_name is required by get_config"
    assert raw.get("job_id")
    for key in ("data", "model", "predictands", "loss", "mask_unit_size"):
        assert key in raw, f"{key} missing from {path.name}"


@pytest.mark.parametrize(
    "a,b,label",
    [(SA_RECURRENT, SA_MAMBA, "SA"), (NARR_RECURRENT, NARR_MAMBA, "NARR")],
)
def test_backend_pair_differs_only_in_backend_and_paths(a, b, label):
    """A backend comparison must not be confounded.

    Data, splits, sequence geometry, losses, freezing schedule, learning rates
    and seed must be byte-identical between the two backends of a case.
    """
    fa, fb = _flatten(_load(a)), _flatten(_load(b))
    differing = {k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)}

    # case_name legitimately differs for the SA pair (two separate experiments);
    # for the NARR pair it is deliberately IDENTICAL, because it also locates the
    # preprocessed inputs. Both are correct, so it is permitted rather than pinned.
    allowed_exact = {"temporal.backend", "job_id", "case_name"}
    path_like = {
        "path_experiment",
        "checkpoint_dir",
        "run_dir",
        "inference.output_dir",
        "temporal.inference.output_dir",
    }
    unexpected = differing - allowed_exact - path_like
    assert not unexpected, f"{label}: unexpected differences between backends: {sorted(unexpected)}"

    assert fa["temporal.backend"] != fb["temporal.backend"]
    assert {fa["temporal.backend"], fb["temporal.backend"]} == {"recurrent", "mamba"}

    # Explicitly pin the things that would silently invalidate a comparison.
    for key in (
        "temporal.context_length",
        "temporal.output_length",
        "temporal.warmup_length",
        "temporal.sequence_stride",
        "temporal.seed",
        "temporal.latent.hidden_channels",
        "temporal.latent.adapter_init_gate",
        "temporal.freeze.lr_temporal",
        "temporal.freeze.lr_decoder",
        "temporal.losses.tendency.weight",
        "temporal.losses.accumulation.weight",
        "temporal.losses.occurrence.weight",
        "temporal.init_from_spatial_checkpoint",
    ):
        assert fa.get(key) == fb.get(key), f"{label}: {key} differs between backends"


def test_sa_config_preserves_the_source_case():
    src, new = _load(SA_SOURCE), _load(SA_RECURRENT)
    for key in (
        "input_vars",
        "input_levels",
        "output_vars",
        "input_static_surface_vars",
        "vertical_pres_vars",
        "input_level_pres",
        "n_input_timestamps",
        "target_size_lat",
        "target_size_lon",
        "input_size_lat",
        "input_size_lon",
        "use_static",
        "static_path",
        "training_predictor_paths",
        "training_target_paths",
    ):
        assert new["data"][key] == src["data"][key], f"data.{key} changed"

    assert new["data"]["scalers"] == src["data"]["scalers"], "normalization scalers changed"
    assert new["predictands"] == src["predictands"], "predictand definitions changed"
    for key in ("embed_dim", "n_blocks_encoder", "num_static_channels",
                "unet_upsample_scales", "downscaling_embed_dim"):
        assert new["model"][key] == src["model"][key], f"model.{key} changed"
    assert new["data"]["output_vars"] == ["pr", "tasmax"]
    # Held-out test files must actually be declared (the source left them empty).
    assert new["data"]["test_predictor_paths"], "test split has no predictor files"
    assert new["data"]["test_target_paths"], "test split has no target files"


def test_narr_config_preserves_the_source_case_including_tmin():
    src, new = _load(NARR_SOURCE), _load(NARR_RECURRENT)
    # All three targets, in order. tmin must not be dropped.
    assert new["data"]["output_vars"] == ["ppt", "tmax", "tmin"] == src["data"]["output_vars"]
    assert "tmin" in new["predictands"]
    assert new["predictands"] == src["predictands"]

    assert new["case_name"] == "narr_prism_California" == src["case_name"], (
        "the narr_prism_California case identity must be preserved -- it also "
        "locates the preprocessed inputs"
    )
    for key in (
        "input_vars",
        "input_levels",
        "target_variables",
        "predictor_variables",
        "spatial_subset",
        "preprocessed_dir",
        "scalar_dir",
        "scalers",
        "train_crop_size_lat",
        "train_crop_size_lon",
        "training_halo_lat",
        "training_halo_lon",
        "n_input_timestamps",
        "regrid_method",
    ):
        if key in src["data"]:
            assert new["data"][key] == src["data"][key], f"data.{key} changed"

    assert new["dates"] == src["dates"], "training/validation/inference periods changed"
    assert new["model"]["num_static_channels"] == 0 == src["model"]["num_static_channels"]
    assert new["precip_model"] == src["precip_model"] == "hurdle"
    for key in ("decoder_skip_source", "backbone_residual_mode", "backbone_attention_scope"):
        assert new["model"][key] == src["model"][key], f"model.{key} changed"


def test_new_outputs_go_to_separate_directories():
    """Nothing existing may be overwritten."""
    for path in ALL_TEMPORAL:
        raw = _load(path)
        for key in ("path_experiment", "checkpoint_dir", "run_dir"):
            value = str(raw.get(key, ""))
            assert "runs_temporal" in value, (
                f"{path.name}: {key}={value!r} does not point at a separate "
                "runs_temporal tree"
            )
        out = str(raw["temporal"]["inference"]["output_dir"])
        assert "runs_temporal" in out

    # And they must not collide with each other.
    dirs = [str(_load(p)["checkpoint_dir"]) for p in ALL_TEMPORAL]
    assert len(set(dirs)) == len(dirs), "two configs share a checkpoint directory"


def test_refinement_defaults_preserve_existing_phase2_checkpoints():
    for path in ALL_TEMPORAL:
        cfg = parse_temporal_config(_load(path)["temporal"])
        assert cfg.refinement.temporal_conditioning == "none", (
            f"{path.name}: temporal refinement conditioning must default to 'none' so "
            "the existing Phase-2 checkpoints stay loadable"
        )
        assert cfg.refinement.noise == "iid_per_frame"
        assert cfg.refinement.noise_rho == 0.0


def test_legacy_spatial_configs_still_resolve_to_no_temporal():
    """Existing configs must be untouched and must take the legacy path."""
    for path in (
        SA_SOURCE,
        NARR_SOURCE,
        REPO / "examples/CORDEX_ML/SA_T2_ACCESS-CM2_static_flow_matching_unet.yaml",
        REPO / "examples/NARR_PRISM/NARR_PRISM_diffusion_unet.yaml",
        REPO / "examples/NARR_PRISM/NARR_PRISM_flow_matching_transformer.yaml",
    ):
        if not path.is_file():
            pytest.skip(f"{path.name} not present")
        raw = _load(path)
        assert parse_temporal_config(raw.get("temporal")) is None, (
            f"{path.name} unexpectedly resolves to an enabled temporal config"
        )


@pytest.mark.parametrize("path", ALL_TEMPORAL, ids=lambda p: p.stem)
def test_context_length_is_justified_and_bounded(path):
    """Guard against an accidentally huge window."""
    cfg = parse_temporal_config(_load(path)["temporal"])
    assert 2 <= cfg.context_length <= 31, (
        "context_length outside the range the memory/statistics argument covers"
    )
    assert cfg.sequence_stride == cfg.output_length, (
        "stride must equal output_length so training windows tile the record and "
        "no date is supervised twice per epoch"
    )


# ---------------------------------------------------------------------------
# geometry resolution (regression: the two cases declare geometry differently)
# ---------------------------------------------------------------------------
def test_crop_and_static_resolution_for_both_cases():
    """Regression test for an AttributeError on the NARR configs.

    CORDEX declares ``target_size_lat/lon`` *and* ``train_crop_size_lat/lon``;
    NARR/PRISM declares only the training crop, because the full target grid comes
    from the PRISM file. Code that reads ``target_size_lat`` unconditionally raises
    on the NARR configs, which is exactly what happened.
    """
    from granitewxc.utils.config import ExperimentConfig
    from granitewxc.temporal.training import resolve_crop_size, resolve_static_channels

    sa = ExperimentConfig.from_dict(_load(SA_RECURRENT))
    assert resolve_crop_size(sa) == (128, 128)
    # SA carries orography as one trailing static channel.
    assert resolve_static_channels(sa) == 1

    narr = ExperimentConfig.from_dict(_load(NARR_RECURRENT))
    assert not hasattr(narr.data, "target_size_lat"), (
        "the NARR config is expected to omit target_size_lat; if that changed, this "
        "test no longer covers the case it was written for"
    )
    assert resolve_crop_size(narr) == (256, 256)
    # 0 must survive: elevation and masks live in the dynamic channel list.
    assert resolve_static_channels(narr) == 0


def test_crop_resolution_falls_back_to_the_frame_source_grid():
    from granitewxc.utils.config import ExperimentConfig
    from granitewxc.temporal.training import resolve_crop_size

    raw = _load(NARR_RECURRENT)
    for key in ("train_crop_size_lat", "train_crop_size_lon"):
        raw["data"].pop(key, None)
    stripped = ExperimentConfig.from_dict(raw)

    class _Source:
        def fine_shape(self):
            return (321, 481)

    assert resolve_crop_size(stripped, _Source()) == (321, 481)
    with pytest.raises(ValueError, match="Cannot resolve a crop size"):
        resolve_crop_size(stripped, None)
