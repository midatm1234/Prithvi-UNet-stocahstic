from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[1]
CORDEX_DIR = REPO_ROOT / "examples" / "CORDEX_ML"


def _load_module(relative_path: str, module_name: str):
    path = CORDEX_DIR / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_run_paths_include_inference_and_logs(tmp_path: Path) -> None:
    run_utils = _load_module("run_utils.py", "cordex_run_utils_compat")

    paths = run_utils.prepare_run_paths(tmp_path, "case-a")

    assert paths.inference == tmp_path / "case-a" / "inference"
    assert paths.logs == tmp_path / "case-a" / "logs"
    assert paths.inference.is_dir()
    assert paths.logs.is_dir()


def test_prepare_case_paths_rebases_config_without_double_nesting(tmp_path: Path) -> None:
    run_utils = _load_module("run_utils.py", "cordex_run_utils_case_compat")
    config = SimpleNamespace(
        case_name="case-a",
        path_experiment=str(tmp_path / "case-a"),
        require_case_name=lambda: "case-a",
    )

    paths = run_utils.prepare_case_paths(config)

    assert paths.run_dir == tmp_path / "case-a"
    assert config.path_experiment == str(paths.run_dir)
    assert config.inference_dir == str(paths.inference)
    assert config.log_dir == str(paths.logs)


def test_ensemble_mean_is_explicit_and_preserves_provenance() -> None:
    postprocess = _load_module(
        "utils/postprocess_outputs.py", "cordex_postprocess_compat"
    )
    values = xr.DataArray(
        np.arange(12, dtype=np.float32).reshape(2, 3, 2),
        dims=("time", "ensemble", "x"),
        attrs={"head_type": "diffusion"},
    )

    result = postprocess.ensemble_mean_xr(values)

    xr.testing.assert_allclose(result, values.mean("ensemble"))
    assert result.attrs["head_type"] == "diffusion"
    assert result.attrs["ensemble_reduction"] == "mean"
    assert result.attrs["ensemble_dimension"] == "ensemble"


def test_case_name_is_default_run_name_for_all_domains(tmp_path: Path) -> None:
    config = SimpleNamespace(
        case_name="case-a",
        job_id="legacy-job-id",
        append_timestamp_to_run_name=False,
    )

    for domain in ("alps", "nz", "sa"):
        params_module = _load_module(
            f"{domain}_params.py", f"cordex_{domain}_params_compat"
        )
        params = params_module.UserParams(
            repo_root=REPO_ROOT,
            project_dir=CORDEX_DIR,
            runs_root=tmp_path,
            config_path=CORDEX_DIR / "cordex_config.yaml",
        )
        run_name, run_dir = params_module.resolve_run_dir(params, config)
        assert run_name == "case-a"
        assert run_dir == tmp_path / "case-a"


def test_rectilinear_regrid_fallback_uses_physical_coordinates() -> None:
    preproc = _load_module("preproc_cordex.py", "cordex_preproc_rectilinear")
    grid_in = xr.Dataset(
        coords={"lat": np.array([0.0, 2.0]), "lon": np.array([10.0, 14.0])}
    )
    grid_out = xr.Dataset(
        coords={"lat": np.array([0.0, 1.0, 2.0]), "lon": np.array([10.0, 12.0, 14.0])}
    )
    values = xr.DataArray(
        np.array([[10.0, 14.0], [12.0, 16.0]]),
        dims=("lat", "lon"),
        coords=grid_in.coords,
    )

    result = preproc.XarrayRegridder(grid_in, grid_out, method="bilinear")(values)

    assert result.sel(lat=1.0, lon=12.0).item() == pytest.approx(13.0)
    np.testing.assert_array_equal(result.lat, grid_out.lat)
    np.testing.assert_array_equal(result.lon, grid_out.lon)


def test_curvilinear_regrid_fallback_fails_instead_of_index_resizing() -> None:
    preproc = _load_module("preproc_cordex.py", "cordex_preproc_curvilinear")
    grid_in = xr.Dataset(
        {
            "lat": (("y", "x"), np.array([[0.0, 0.1], [1.0, 1.1]])),
            "lon": (("y", "x"), np.array([[10.0, 11.0], [10.1, 11.1]])),
        }
    )
    grid_out = xr.Dataset(
        {
            "lat": (("y", "x"), np.array([[0.0, 0.05], [0.5, 0.55]])),
            "lon": (("y", "x"), np.array([[10.0, 10.5], [10.05, 10.55]])),
        }
    )

    with pytest.raises(RuntimeError, match="curvilinear.*scientifically invalid"):
        preproc.XarrayRegridder(grid_in, grid_out, method="bilinear")
