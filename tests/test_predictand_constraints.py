from __future__ import annotations

from types import SimpleNamespace

import torch

from granitewxc.utils.predictands import build_predictand_specs
from granitewxc.utils.target_transforms import apply_pr_positive_link


def _make_config(output_vars: list[str], predictands: dict | None = None):
    return SimpleNamespace(
        data=SimpleNamespace(output_vars=output_vars),
        predictands=predictands or {},
    )


def test_predictand_defaults_enable_softplus_divide_only_for_pr():
    config = _make_config(["pr", "tasmax"])
    specs = build_predictand_specs(config, output_vars=["pr", "tasmax"])
    by_name = {spec.name: spec for spec in specs}

    assert by_name["pr"].allow_negative_value is False
    assert by_name["pr"].nonnegativity.enabled is True
    assert by_name["pr"].nonnegativity.method == "softplus"
    assert by_name["pr"].scaling.method == "divide_only"
    assert by_name["pr"].scaling.mode == "global"
    assert by_name["pr"].scaling.eps_std > 0.0
    assert by_name["pr"].scaling.scale_stat == "p95"

    assert by_name["tasmax"].allow_negative_value is False
    assert by_name["tasmax"].nonnegativity.enabled is False
    assert by_name["tasmax"].scaling.method == "zscore"


def test_predictand_nonnegative_link_and_inverse_scale_never_negative():
    raw = torch.tensor([[[[-1000.0]], [[3.0]]]], dtype=torch.float32)  # [B, V, H, W]
    linked = apply_pr_positive_link(raw, var_names=["pr", "tasmax"], pr_name="pr")
    scale = torch.tensor([10.0, 2.0], dtype=torch.float32).view(1, 2, 1, 1)
    physical = linked * scale

    assert float(physical[:, 0, ...].min()) >= 0.0
    assert torch.allclose(linked[:, 1, ...], raw[:, 1, ...])


def test_nonnegative_with_zscore_scaling_raises():
    config = _make_config(
        ["pr"],
        predictands={
            "pr": {
                "allow_negative_value": False,
                "nonnegativity": {"enabled": True, "method": "softplus"},
                "scaling": {"method": "zscore", "scale_stat": "mean"},
            }
        },
    )
    try:
        build_predictand_specs(config, output_vars=["pr"])
    except ValueError as exc:
        assert "incompatible" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for nonnegativity + zscore scaling.")


def test_log1p_standardize_alias_is_canonicalized():
    config = _make_config(
        ["pr"],
        predictands={
            "pr": {
                "allow_negative_value": False,
                "nonnegativity": {"enabled": True, "method": "softplus"},
                "scaling": {"method": "log1p_zscore", "scale_stat": "mean"},
            }
        },
    )
    specs = build_predictand_specs(config, output_vars=["pr"])
    assert specs[0].scaling.method == "log1p_standardize"


def test_gridpoint_normalization_block_is_parsed():
    config = _make_config(
        ["pr", "tasmax"],
        predictands={
            "pr": {
                "normalization": {
                    "method": "log1p_standardize",
                    "mode": "gridpoint",
                    "eps_std": 1e-5,
                    "allow_negative_value": False,
                },
                "nonnegativity": {"enabled": True, "method": "softplus"},
            },
            "tasmax": {
                "normalization": {
                    "method": "standardize",
                    "mode": "gridpoint",
                    "eps_std": 1e-6,
                }
            },
        },
    )
    specs = build_predictand_specs(config, output_vars=["pr", "tasmax"])
    by_name = {spec.name: spec for spec in specs}
    assert by_name["pr"].scaling.method == "log1p_standardize"
    assert by_name["pr"].scaling.mode == "gridpoint"
    assert by_name["pr"].scaling.eps_std == 1e-5
    assert by_name["tasmax"].scaling.method == "zscore"
    assert by_name["tasmax"].scaling.mode == "gridpoint"


def test_pr_zero_roundtrip_with_log1p_standardize():
    # Simulate model-space decode: y = expm1(z * sigma + mu)
    z = torch.tensor([[[[0.0]]]], dtype=torch.float32)
    mu = torch.tensor([[[[0.0]]]], dtype=torch.float32)
    sigma = torch.tensor([[[[1.0]]]], dtype=torch.float32)

    y = torch.expm1(z * sigma + mu)
    y = torch.where((y < 0.0) & (y > -1e-7), torch.zeros_like(y), y)

    assert float(y.min()) >= 0.0
    assert torch.allclose(y, torch.zeros_like(y))
