from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from granitewxc.utils.config import SCALAR_FILENAMES, get_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CORDEX_CONFIG_DIR = REPO_ROOT / "examples" / "CORDEX_ML"


def _cordex_yamls() -> list[Path]:
    return sorted(CORDEX_CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("path", _cordex_yamls(), ids=lambda path: path.stem)
def test_all_cordex_yamls_parse_with_one_case_scoped_identity(path: Path):
    text = path.read_text(encoding="utf-8")
    declarations = [
        line for line in text.splitlines() if line.startswith("case_name:")
    ]
    assert len(declarations) == 1

    config = get_config(str(path))
    assert config.case_name
    case_root = Path(config.path_experiment) / config.case_name
    assert Path(config.case_dir) == case_root
    assert Path(config.path_scalars) == case_root / "scalars"
    assert Path(config.path_preproc) == case_root / "preproc"
    assert Path(config.path_checkpoints) == case_root / "checkpoints"
    assert Path(config.path_inference) == case_root / "inference"
    assert Path(config.path_logs) == case_root / "logs"


def test_case_path_derivation_is_explicit_and_uses_canonical_scalar_names(tmp_path):
    source = CORDEX_CONFIG_DIR / "SA_T2_ACCESS-CM2_static.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["path_experiment"] = str(tmp_path / "runs")
    payload["case_name"] = "derived_case"
    payload["derive_output_paths"] = True
    config_path = tmp_path / "derived.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = get_config(str(config_path))
    expected = tmp_path / "runs" / "derived_case"
    assert Path(config.checkpoint_dir) == expected / "checkpoints"
    for key, filename in SCALAR_FILENAMES.items():
        path = expected / "scalars" / filename
        assert Path(config.data.scalers[key]) == path
    assert Path(config.model.input_mu) == expected / "scalars" / "inputs_mean.npy"
    assert Path(config.model.input_sigma) == expected / "scalars" / "inputs_std.npy"
    assert Path(config.model.target_mu) == expected / "scalars" / "targets_mean.npy"
    assert Path(config.model.target_sigma) == expected / "scalars" / "targets_std.npy"


def test_narr_config_preserves_authenticated_yaml_scalar_paths():
    path = REPO_ROOT / "examples" / "NARR_PRISM" / "NARR_PRISM_subdomain.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = get_config(str(path))

    assert config.data.type == "narr_prism"
    for name in ("input_mu", "input_sigma", "target_mu", "target_sigma"):
        assert getattr(config.model, name) == raw["model"][name]
    assert config.data.scalers == raw["data"]["scalers"]
    assert config.checkpoint_dir == raw["checkpoint_dir"]


def test_compute_scalars_uses_same_explicit_directory_as_training(monkeypatch):
    import argparse

    monkeypatch.syspath_prepend(str(CORDEX_CONFIG_DIR))
    from compute_scalars_cordex import _resolve_output_dir

    path = CORDEX_CONFIG_DIR / "NZ_T1_ACCESS-CM2_static_v6.yaml"
    config = get_config(str(path))
    expected = Path(config.model.input_mu).parent
    assert Path(_resolve_output_dir(argparse.Namespace(output_dir=None), config)) == expected


def test_compute_scalars_rejects_inconsistent_explicit_scalar_directories(monkeypatch):
    import argparse

    monkeypatch.syspath_prepend(str(CORDEX_CONFIG_DIR))
    from compute_scalars_cordex import _resolve_output_dir

    config = get_config(str(CORDEX_CONFIG_DIR / "SA_T2_ACCESS-CM2_static.yaml"))
    config.model.target_sigma = "./different/place/targets_std.npy"
    with pytest.raises(ValueError, match="Inconsistent configured scalar path"):
        _resolve_output_dir(argparse.Namespace(output_dir=None), config)
