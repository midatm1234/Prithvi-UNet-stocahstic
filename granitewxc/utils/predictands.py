"""Predictand configuration helpers for CORDEX training/inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, MutableMapping, Sequence


PHYSICALLY_NONNEGATIVE_VARS = {"pr", "precip", "precipitation"}
ALLOWED_NONNEGATIVITY_METHODS = {"softplus", "exp", "none"}
ALLOWED_SCALING_METHODS = {"divide_only", "zscore", "log1p_zscore"}
ALLOWED_SCALE_STATS = {"mean", "p90", "p95", "p99", "fixed"}


@dataclass
class NonnegativitySpec:
    enabled: bool
    method: str


@dataclass
class ScalingSpec:
    method: str
    scale_stat: str
    fixed_scale: float | None


@dataclass
class PredictandSpec:
    name: str
    allow_negative_value: bool
    nonnegativity: NonnegativitySpec
    scaling: ScalingSpec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, MutableMapping):
        return dict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _warn(message: str) -> None:
    print(f"[predictands] {message}")


def _parse_predictand_spec(name: str, raw_cfg: Mapping[str, Any]) -> PredictandSpec:
    lower_name = name.lower()
    raw = _coerce_mapping(raw_cfg)

    allow_negative_value = bool(raw.get("allow_negative_value", False))
    raw_nonnegativity = _coerce_mapping(raw.get("nonnegativity"))
    raw_scaling = _coerce_mapping(raw.get("scaling"))

    physically_nonnegative = lower_name in PHYSICALLY_NONNEGATIVE_VARS
    default_nonnegativity_enabled = physically_nonnegative and not allow_negative_value

    nonnegativity_enabled = bool(
        raw_nonnegativity.get("enabled", default_nonnegativity_enabled)
    )
    nonnegativity_method = str(
        raw_nonnegativity.get(
            "method",
            "softplus" if nonnegativity_enabled else "none",
        )
    ).lower()

    if nonnegativity_method not in ALLOWED_NONNEGATIVITY_METHODS:
        raise ValueError(
            f"predictands.{name}.nonnegativity.method='{nonnegativity_method}' is invalid; "
            f"expected one of {sorted(ALLOWED_NONNEGATIVITY_METHODS)}"
        )
    if nonnegativity_enabled and nonnegativity_method == "none":
        raise ValueError(
            f"predictands.{name}.nonnegativity.enabled=True requires method softplus or exp."
        )
    if not nonnegativity_enabled:
        nonnegativity_method = "none"

    default_scaling_method = (
        "divide_only" if default_nonnegativity_enabled else "zscore"
    )
    scaling_method = str(
        raw_scaling.get("method", default_scaling_method)
    ).lower()
    if scaling_method not in ALLOWED_SCALING_METHODS:
        raise ValueError(
            f"predictands.{name}.scaling.method='{scaling_method}' is invalid; "
            f"expected one of {sorted(ALLOWED_SCALING_METHODS)}"
        )

    default_scale_stat = "p95" if scaling_method == "divide_only" else "mean"
    scale_stat = str(raw_scaling.get("scale_stat", default_scale_stat)).lower()
    if scale_stat not in ALLOWED_SCALE_STATS:
        raise ValueError(
            f"predictands.{name}.scaling.scale_stat='{scale_stat}' is invalid; "
            f"expected one of {sorted(ALLOWED_SCALE_STATS)}"
        )

    fixed_scale = raw_scaling.get("fixed_scale")
    if fixed_scale is not None:
        fixed_scale = float(fixed_scale)

    if scale_stat == "fixed" and (fixed_scale is None or fixed_scale <= 0):
        raise ValueError(
            f"predictands.{name}.scaling.scale_stat='fixed' requires a positive fixed_scale."
        )

    if nonnegativity_enabled and scaling_method == "zscore":
        raise ValueError(
            f"predictands.{name}: nonnegativity.enabled=True is incompatible with "
            "scaling.method='zscore'. Use divide_only."
        )

    if lower_name in PHYSICALLY_NONNEGATIVE_VARS and allow_negative_value:
        _warn(
            f"{name}: allow_negative_value=True for a physically non-negative variable."
        )

    if not allow_negative_value and not nonnegativity_enabled:
        _warn(
            f"{name}: allow_negative_value=False but nonnegativity.enabled=False. "
            "This is expected for temperature-like variables."
        )

    return PredictandSpec(
        name=name,
        allow_negative_value=allow_negative_value,
        nonnegativity=NonnegativitySpec(
            enabled=nonnegativity_enabled,
            method=nonnegativity_method,
        ),
        scaling=ScalingSpec(
            method=scaling_method,
            scale_stat=scale_stat,
            fixed_scale=fixed_scale,
        ),
    )


def build_predictand_specs(
    config: Any, output_vars: Sequence[str] | None = None
) -> list[PredictandSpec]:
    """Resolve predictand specs with defaults and attach the normalized block to config."""

    if output_vars is None:
        output_vars = list(getattr(config.data, "output_vars", []))

    raw_predictands = _coerce_mapping(getattr(config, "predictands", {}))

    specs: list[PredictandSpec] = []
    for var_name in output_vars:
        raw_cfg = raw_predictands.get(var_name, {})
        specs.append(_parse_predictand_spec(var_name, _coerce_mapping(raw_cfg)))

    setattr(config, "predictands", {spec.name: spec.to_dict() for spec in specs})
    return specs


def specs_by_name(specs: Sequence[PredictandSpec]) -> dict[str, PredictandSpec]:
    return {spec.name: spec for spec in specs}
