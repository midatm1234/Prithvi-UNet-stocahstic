from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

xr = pytest.importorskip("xarray")

from examples.NARR_PRISM import narr_prism_refinement as refinement_cli
from examples.NARR_PRISM import narr_prism_training as training


def _checkpoint_config(checkpoint_dir: str):
    return SimpleNamespace(
        checkpoint_dir=checkpoint_dir,
        model=SimpleNamespace(refinement={"checkpoint": None}),
    )


def test_refinement_checkpoint_resolves_shipped_repo_relative_path(
    tmp_path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    config_path = repo / "examples" / "NARR_PRISM" / "case.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("case_name: test\n", encoding="utf-8")
    checkpoint = (
        repo
        / "examples"
        / "NARR_PRISM"
        / "experiments"
        / "refinement_checkpoints"
        / "flow_matching_transformer"
        / "best.ckpt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(refinement_cli, "REPO_ROOT", repo)

    resolved = refinement_cli._refinement_checkpoint(
        _checkpoint_config(
            "./examples/NARR_PRISM/experiments/refinement_checkpoints/"
            "flow_matching_transformer"
        ),
        str(config_path),
        None,
    )

    assert Path(resolved) == checkpoint.resolve()


def test_refinement_checkpoint_also_resolves_yaml_relative_path(
    tmp_path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    config_path = repo / "configs" / "case.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("case_name: test\n", encoding="utf-8")
    checkpoint = config_path.parent / "local_checkpoints" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(refinement_cli, "REPO_ROOT", repo)

    resolved = refinement_cli._refinement_checkpoint(
        _checkpoint_config("./local_checkpoints"),
        str(config_path),
        None,
    )

    assert Path(resolved) == checkpoint.resolve()


def test_inference_loader_is_full_domain_and_preserves_dates(monkeypatch) -> None:
    calls = []

    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, config_path, mode, full_domain=False):
            calls.append((config_path, mode, full_domain))
            self.mode = mode
            self.full_domain = full_domain
            self.fine_shape = (2, 3)
            self.crop_size = self.fine_shape if full_domain else (1, 1)
            self.fine_lat = np.array([40.0, 39.0])
            self.fine_lon = np.array([-120.0, -119.0, -118.0])
            self._tile_slices = None
            self._elevation = None

        def __len__(self):
            return 1

        def __getitem__(self, index):
            h, w = self.crop_size
            return {
                "x": torch.zeros(1, h, w),
                "y": torch.zeros(1, h, w),
                "date": "2016-01-02",
                "__scaler_offset": torch.tensor([0, 0]),
                "__input_scaler_offset": torch.tensor([0, 0]),
                "__output_scaler_offset": torch.tensor([0, 0]),
                "__output_crop": torch.tensor([0, 0, h, w]),
            }

    monkeypatch.setattr(training, "NarrPrismDataset", FakeDataset)
    config = SimpleNamespace(
        batch_size=1,
        dl_num_workers=0,
        mask_unit_size=[1, 1],
        model=SimpleNamespace(
            num_static_channels=0,
            downscaling_patch_size=[1, 1],
        ),
    )

    loader = training.get_inference_dataloader("case.yaml", config)
    batch = next(iter(loader))

    assert calls == [("case.yaml", "inference", True)]
    assert loader.dataset.base.crop_size == loader.dataset.base.fine_shape
    assert batch["date"] == ["2016-01-02"]
    assert tuple(batch["y"].shape[-2:]) == (2, 3)

    validation_loader = training._build_dataloader(
        "case.yaml",
        config,
        "validation",
        shuffle=False,
        distributed=False,
        rank=0,
        world_size=1,
    )
    validation_batch = next(iter(validation_loader))
    assert calls[-1] == ("case.yaml", "validation", False)
    assert "date" not in validation_batch
    assert tuple(validation_batch["y"].shape[-2:]) == (1, 1)


class _FakeRefinementModel:
    def __init__(self) -> None:
        self.refiner = torch.nn.Identity()
        self.refinement_config = SimpleNamespace(
            ensemble_size=2,
            seed=17,
            type="flow_matching_transformer",
        )
        self.performance_config = SimpleNamespace(
            io=SimpleNamespace(
                netcdf_compression=False,
                netcdf_compression_level=1,
            )
        )

    def to(self, device):
        return self

    def eval(self):
        self.refiner.eval()
        return self

    def initialize_from_batch(self, batch):
        return self

    def state_dict(self):
        return {"phase1.weight": torch.zeros(1)}

    def residual_normalization_metadata(self):
        return {
            "fitted": True,
            "method": "standardize",
            "mean": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
            "count": [1, 1, 1],
        }

    def predict(self, batch, *, ensemble_size, generator):
        batch_size, _, height, width = batch["y"].shape
        channels = 3
        deterministic = torch.ones(batch_size, channels, height, width)
        member_offsets = torch.arange(
            ensemble_size, dtype=deterministic.dtype
        ).reshape(1, ensemble_size, 1, 1, 1)
        members = deterministic.unsqueeze(1) + member_offsets
        ensemble_mean = members.mean(dim=1)
        ensemble_spread = members.std(dim=1, unbiased=True)
        return SimpleNamespace(
            deterministic=deterministic,
            residual_physical=ensemble_mean - deterministic,
            members=members,
            ensemble_mean=ensemble_mean,
            ensemble_spread=ensemble_spread,
        )


def test_cmd_infer_writes_source_dates_canonical_grid_and_physical_units(
    tmp_path, monkeypatch
) -> None:
    fine_lat = np.array([40.5, 39.5], dtype=np.float64)
    fine_lon = np.array([-121.0, -120.0, -119.0], dtype=np.float64)
    base_dataset = SimpleNamespace(
        mode="inference",
        full_domain=True,
        fine_shape=(2, 3),
        crop_size=(2, 3),
        fine_lat=fine_lat,
        fine_lon=fine_lon,
        _tile_slices=None,
    )
    batches = [
        {
            "x": torch.zeros(1, 1, 2, 3),
            "y": torch.zeros(1, 3, 2, 3),
            "date": [date],
        }
        for date in ("2016-01-01", "2016-01-02")
    ]

    class FakeLoader:
        dataset = SimpleNamespace(base=base_dataset)

        def __iter__(self):
            return iter(batches)

    config = SimpleNamespace(
        case_name="synthetic_narr",
        data=SimpleNamespace(output_vars=["ppt", "tmax", "tmin"]),
    )
    model = _FakeRefinementModel()
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.write_bytes(b"strict-refinement-checkpoint")
    payload = {
        "checkpoint_schema_version": refinement_cli.CHECKPOINT_SCHEMA_VERSION,
        "refinement_contract": {
            "contract_version": refinement_cli.REFINEMENT_CONTRACT_VERSION,
            "residual_contract": "physical_ground_truth_minus_phase1_v1",
        },
        "refinement_contract_fingerprint": "contract-fingerprint",
    }
    monkeypatch.setattr(refinement_cli, "get_config", lambda path: config)
    monkeypatch.setattr(
        refinement_cli, "_phase1_checkpoint", lambda config, override: "phase1.ckpt"
    )
    monkeypatch.setattr(
        refinement_cli,
        "build_model",
        lambda config, config_path, phase1_checkpoint, device: (model, "phase1-fingerprint"),
    )
    monkeypatch.setattr(
        refinement_cli, "get_inference_dataloader", lambda config_path, config: FakeLoader()
    )
    monkeypatch.setattr(
        refinement_cli,
        "_refinement_checkpoint",
        lambda config, config_path, override: str(checkpoint),
    )
    monkeypatch.setattr(refinement_cli.torch, "load", lambda *args, **kwargs: payload)
    monkeypatch.setattr(refinement_cli, "validate_phase1_reference", lambda *args, **kwargs: None)
    monkeypatch.setattr(refinement_cli, "load_refinement_state_dict", lambda *args, **kwargs: None)

    output = tmp_path / "refinement.nc"
    args = SimpleNamespace(
        config="case.yaml",
        device="cpu",
        phase1_checkpoint=None,
        refinement_checkpoint=None,
        ensemble_size=2,
        seed=23,
        limit_batches=0,
        output=str(output),
    )

    assert refinement_cli.cmd_infer(args) == 0

    with xr.open_dataset(output) as dataset:
        np.testing.assert_array_equal(
            dataset["time"].values,
            np.array(["2016-01-01", "2016-01-02"], dtype="datetime64[ns]"),
        )
        np.testing.assert_array_equal(dataset["lat"].values, fine_lat)
        np.testing.assert_array_equal(dataset["lon"].values, fine_lon)
        assert dataset["ppt_members"].dims == ("time", "member", "lat", "lon")
        for variable, unit in {
            "ppt": "mm/day",
            "tmax": "degC",
            "tmin": "degC",
        }.items():
            for suffix in (
                "",
                "_residual",
                "_refined",
                "_members",
                "_ensemble_mean",
                "_ensemble_spread",
                "_truth",
            ):
                assert dataset[f"{variable}{suffix}"].attrs["units"] == unit


def test_unknown_narr_output_unit_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="No physical output-unit contract"):
        refinement_cli._physical_output_units(["unsupported_variable"])


def test_canonical_inference_metadata_rejects_tiles_as_time() -> None:
    tiled = SimpleNamespace(
        mode="inference",
        full_domain=True,
        fine_shape=(4, 4),
        crop_size=(4, 4),
        fine_lat=np.arange(4),
        fine_lon=np.arange(4),
        _tile_slices=[(slice(0, 2), slice(0, 2))],
    )
    loader = SimpleNamespace(dataset=SimpleNamespace(base=tiled))

    with pytest.raises(RuntimeError, match="tiled dataset"):
        refinement_cli._canonical_inference_dataset(loader)
