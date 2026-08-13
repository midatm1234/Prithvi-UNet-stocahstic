"""Predictand configuration helpers for CORDEX training/inference."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, MutableMapping, Sequence


PHYSICALLY_NONNEGATIVE_VARS = {"pr", "precip", "precipitation", "ppt"}
ALLOWED_NONNEGATIVITY_METHODS = {"softplus", "exp", "none"}
ALLOWED_SCALING_METHODS = {"divide_only", "zscore", "log1p_standardize"}
ALLOWED_NORMALIZATION_MODES = {"global", "gridpoint"}
SCALING_METHOD_ALIASES = {
    "divide_only": "divide_only",
    "zscore": "zscore",
    "standardize": "zscore",
    "log1p_zscore": "log1p_standardize",
    "log1p_standardize": "log1p_standardize",
}
ALLOWED_SCALE_STATS = {"mean", "p90", "p95", "p99", "fixed"}
ALLOWED_PRECIP_MODELS = {
    "single",
    "single_head",
    "default",
    "legacy",
    "hurdle",
    "bernoulli_positive",
    "bernoulli_plus_positive",
    "bernoulli_positive_amount",
    "bernoulli-plus-positive-amount",
    "bernoulli_gamma",
    "bernoulli-gamma",
    "bg",
    "zero_inflated_gamma",
}


@dataclass
class NonnegativitySpec:
    enabled: bool
    method: str


@dataclass
class ScalingSpec:
    method: str
    mode: str
    eps_std: float
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


def canonicalize_scaling_method(value: Any) -> str:
    method = str(value).lower().strip()
    if method not in SCALING_METHOD_ALIASES:
        raise ValueError(
            f"Unsupported scaling method '{value}'. "
            f"Expected one of {sorted(SCALING_METHOD_ALIASES)}."
        )
    canonical = SCALING_METHOD_ALIASES[method]
    if canonical not in ALLOWED_SCALING_METHODS:
        raise ValueError(
            f"Canonical scaling method '{canonical}' is invalid. "
            f"Expected one of {sorted(ALLOWED_SCALING_METHODS)}."
        )
    return canonical


def canonicalize_precip_model(value: Any) -> str:
    model = str(value or "single_head").strip().lower()
    if model not in ALLOWED_PRECIP_MODELS:
        raise ValueError(
            f"Unsupported precip_model '{value}'. "
            f"Expected one of {sorted(ALLOWED_PRECIP_MODELS)}."
        )
    aliases = {
        "single": "single_head",
        "single_head": "single_head",
        "default": "single_head",
        "legacy": "single_head",
        "hurdle": "hurdle",
        "bernoulli_positive": "hurdle",
        "bernoulli_plus_positive": "hurdle",
        "bernoulli_positive_amount": "hurdle",
        "bernoulli-plus-positive-amount": "hurdle",
        "bernoulli_gamma": "bernoulli_gamma",
        "bernoulli-gamma": "bernoulli_gamma",
        "bg": "bernoulli_gamma",
        "zero_inflated_gamma": "bernoulli_gamma",
    }
    return aliases[model]


def _parse_predictand_spec(name: str, raw_cfg: Mapping[str, Any]) -> PredictandSpec:
    lower_name = name.lower()
    raw = _coerce_mapping(raw_cfg)

    raw_normalization = _coerce_mapping(raw.get("normalization"))
    raw_scaling = _coerce_mapping(raw.get("scaling"))
    allow_negative_value = bool(
        raw.get("allow_negative_value", raw_normalization.get("allow_negative_value", False))
    )
    raw_nonnegativity = _coerce_mapping(raw.get("nonnegativity"))
    # Backward compatibility: prefer normalization{} but accept scaling{}.
    normalization_block = raw_normalization or raw_scaling

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
    scaling_method_raw = normalization_block.get("method", default_scaling_method)
    try:
        scaling_method = canonicalize_scaling_method(scaling_method_raw)
    except ValueError as exc:
        raise ValueError(
            f"predictands.{name}.normalization.method='{scaling_method_raw}' is invalid; "
            f"expected one of {sorted(SCALING_METHOD_ALIASES)}"
        ) from exc

    mode = str(normalization_block.get("mode", raw_scaling.get("mode", "global"))).lower()
    if mode not in ALLOWED_NORMALIZATION_MODES:
        raise ValueError(
            f"predictands.{name}.normalization.mode='{mode}' is invalid; "
            f"expected one of {sorted(ALLOWED_NORMALIZATION_MODES)}"
        )

    eps_std = float(normalization_block.get("eps_std", raw_scaling.get("eps_std", 1e-6)))
    if not math.isfinite(eps_std) or eps_std <= 0.0:
        raise ValueError(
            f"predictands.{name}.normalization.eps_std must be > 0, got {eps_std!r}"
        )

    default_scale_stat = "p95" if scaling_method == "divide_only" else "mean"
    scale_stat = str(normalization_block.get("scale_stat", raw_scaling.get("scale_stat", default_scale_stat))).lower()
    if scale_stat not in ALLOWED_SCALE_STATS:
        raise ValueError(
            f"predictands.{name}.scaling.scale_stat='{scale_stat}' is invalid; "
            f"expected one of {sorted(ALLOWED_SCALE_STATS)}"
        )

    fixed_scale = normalization_block.get("fixed_scale", raw_scaling.get("fixed_scale"))
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
            mode=mode,
            eps_std=eps_std,
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

    precip_model = canonicalize_precip_model(
        getattr(config, "precip_model", getattr(config, "precip_head_type", "single_head"))
    )
    if precip_model == "hurdle":
        precip_spec = next(
            (spec for spec in specs if spec.name.lower() in PHYSICALLY_NONNEGATIVE_VARS),
            None,
        )
        if precip_spec is None:
            raise ValueError(
                "precip_model='hurdle' requires a precipitation predictand (e.g. 'pr')."
            )
        if precip_spec.scaling.method != "divide_only":
            raise ValueError(
                "precip_model='hurdle' requires precipitation scaling.method='divide_only'."
            )
        if precip_spec.scaling.scale_stat != "p95":
            raise ValueError(
                "precip_model='hurdle' requires precipitation scaling.scale_stat='p95'."
            )
        if precip_spec.scaling.mode != "global":
            raise ValueError(
                "precip_model='hurdle' requires precipitation normalization.mode='global' "
                "so q95 is quantile-based."
            )
        if precip_spec.nonnegativity.method != "softplus" or not precip_spec.nonnegativity.enabled:
            raise ValueError(
                "precip_model='hurdle' requires precipitation nonnegativity.enabled=True "
                "and nonnegativity.method='softplus'."
            )

    setattr(config, "predictands", {spec.name: spec.to_dict() for spec in specs})
    return specs


def specs_by_name(specs: Sequence[PredictandSpec]) -> dict[str, PredictandSpec]:
    return {spec.name: spec for spec in specs}
