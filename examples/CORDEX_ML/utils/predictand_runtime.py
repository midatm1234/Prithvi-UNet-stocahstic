"""Runtime helpers for predictand configuration and output checks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from granitewxc.utils.predictands import PredictandSpec, build_predictand_specs, specs_by_name


def resolve_predictand_specs(
    config: Any, target_vars: Sequence[str]
) -> dict[str, PredictandSpec]:
    specs = build_predictand_specs(config, output_vars=list(target_vars))
    return specs_by_name(specs)


def assert_nonnegative_outputs(
    predicted_values: Mapping[str, np.ndarray],
    predictand_specs: Mapping[str, PredictandSpec],
    eps: float = 1e-8,
) -> None:
    for var_name, values in predicted_values.items():
        spec = predictand_specs.get(var_name)
        if spec is None or not spec.nonnegativity.enabled:
            continue
        min_value = float(np.nanmin(values))
        max_value = float(np.nanmax(values))
        print(f"[nonneg] {var_name}: min={min_value:.6g}, max={max_value:.6g}")
        if min_value < -eps:
            raise AssertionError(
                f"{var_name} contains negative values ({min_value:.6g}) "
                f"despite nonnegativity.enabled=True"
            )

