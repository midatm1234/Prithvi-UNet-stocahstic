"""Validated schema for the ``temporal:`` block of an experiment YAML.

Design rules enforced here:

* **Unknown keys are rejected.** A silently ignored option is worse than a
  crash: it produces a run that does not match its own configuration file.
  Every key accepted by this parser is consumed by real code.
* **Disabled means legacy.** ``temporal.enabled: false`` (or an absent block)
  yields ``None``, and every call site then takes the original spatial path.
* **Downscaling stays zero-lead.** ``mode: downscaling`` forces
  ``lead_time_days == 0`` and ``causal: true``; asking for a non-zero lead in
  downscaling mode is an error rather than a silent target shift.
* **No fallback substitution.** Requesting ``backend: mamba`` with
  ``mamba.implementation: fused`` when the fused kernels are absent raises at
  construction time instead of quietly running a different model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "TemporalConfig",
    "TemporalConfigError",
    "TemporalStateConfig",
    "TemporalLatentConfig",
    "RecurrentBackendConfig",
    "MambaBackendConfig",
    "TemporalLossConfig",
    "TemporalFreezeConfig",
    "TemporalInferenceConfig",
    "TemporalRefinementConfig",
    "TemporalEvaluationConfig",
    "parse_temporal_config",
]


class TemporalConfigError(ValueError):
    """Raised for any malformed or unrecognized ``temporal:`` configuration."""


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _require_mapping(value: Any, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TemporalConfigError(f"{where} must be a mapping, got {type(value).__name__}.")
    return dict(value)


def _reject_unknown(data: Mapping[str, Any], allowed: Iterable[str], where: str) -> None:
    allowed_set = set(allowed)
    unknown = sorted(set(data) - allowed_set)
    if unknown:
        raise TemporalConfigError(
            f"Unknown key(s) {unknown} under {where}. Allowed: {sorted(allowed_set)}. "
            "Unrecognized keys are rejected so a run always matches its config."
        )


def _as_bool(value: Any, where: str, default: bool | None = None) -> bool:
    if value is None:
        if default is None:
            raise TemporalConfigError(f"{where} is required.")
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "yes", "on", "1"}:
            return True
        if low in {"false", "no", "off", "0"}:
            return False
    raise TemporalConfigError(f"{where} must be a boolean, got {value!r}.")


def _as_int(value: Any, where: str, default: int | None = None, *, minimum: int | None = None) -> int:
    if value is None:
        if default is None:
            raise TemporalConfigError(f"{where} is required.")
        result = int(default)
    else:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise TemporalConfigError(f"{where} must be an integer, got {value!r}.") from exc
    if minimum is not None and result < minimum:
        raise TemporalConfigError(f"{where} must be >= {minimum}, got {result}.")
    return result


def _as_float(value: Any, where: str, default: float | None = None, *, minimum: float | None = None) -> float:
    if value is None:
        if default is None:
            raise TemporalConfigError(f"{where} is required.")
        result = float(default)
    else:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise TemporalConfigError(f"{where} must be a number, got {value!r}.") from exc
    if minimum is not None and result < minimum:
        raise TemporalConfigError(f"{where} must be >= {minimum}, got {result}.")
    return result


def _as_choice(value: Any, where: str, choices: Sequence[str], default: str | None = None) -> str:
    if value is None:
        if default is None:
            raise TemporalConfigError(f"{where} is required (one of {list(choices)}).")
        return default
    text = str(value).strip().lower()
    if text not in choices:
        raise TemporalConfigError(f"{where} must be one of {list(choices)}, got {value!r}.")
    return text


def _as_weight_map(value: Any, where: str) -> dict[str, float]:
    data = _require_mapping(value, where)
    out: dict[str, float] = {}
    for key, weight in data.items():
        out[str(key)] = _as_float(weight, f"{where}.{key}", minimum=0.0)
    return out


def _as_int_list(value: Any, where: str, *, minimum: int = 1) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise TemporalConfigError(f"{where} must be a list, got {type(value).__name__}.")
    out: list[int] = []
    for i, item in enumerate(value):
        out.append(_as_int(item, f"{where}[{i}]", minimum=minimum))
    return out


# ----------------------------------------------------------------------------
# sub-blocks
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class TemporalStateConfig:
    """Recurrent/SSM hidden-state lifecycle policy."""

    init: str = "zeros"
    reset_on_discontinuity: bool = True
    carry_across_windows: bool = False
    detach_between_chunks: bool = True
    tbptt_chunk: int = 0  # 0 => backpropagate through the whole window
    #: Ablation switch. When true the hidden state is zeroed before *every* frame,
    #: so the temporal module still receives and uses the calendar/date features
    #: (through its FiLM modulation) but has no memory of previous frames. This is
    #: the control that separates "date conditioning" from "temporal memory": it
    #: has the same parameter count, the same optimizer and the same data order as
    #: the full model, differing only in whether history is available.
    reset_every_frame: bool = False

    @staticmethod
    def parse(value: Any, where: str) -> "TemporalStateConfig":
        data = _require_mapping(value, where)
        allowed = {
            "init",
            "reset_on_discontinuity",
            "carry_across_windows",
            "detach_between_chunks",
            "tbptt_chunk",
            "reset_every_frame",
        }
        _reject_unknown(data, allowed, where)
        return TemporalStateConfig(
            init=_as_choice(data.get("init"), f"{where}.init", ("zeros", "learned"), "zeros"),
            reset_on_discontinuity=_as_bool(
                data.get("reset_on_discontinuity"), f"{where}.reset_on_discontinuity", True
            ),
            carry_across_windows=_as_bool(
                data.get("carry_across_windows"), f"{where}.carry_across_windows", False
            ),
            detach_between_chunks=_as_bool(
                data.get("detach_between_chunks"), f"{where}.detach_between_chunks", True
            ),
            tbptt_chunk=_as_int(data.get("tbptt_chunk"), f"{where}.tbptt_chunk", 0, minimum=0),
            reset_every_frame=_as_bool(
                data.get("reset_every_frame"), f"{where}.reset_every_frame", False
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "init": self.init,
            "reset_on_discontinuity": self.reset_on_discontinuity,
            "carry_across_windows": self.carry_across_windows,
            "detach_between_chunks": self.detach_between_chunks,
            "tbptt_chunk": self.tbptt_chunk,
            "reset_every_frame": self.reset_every_frame,
        }


@dataclass(frozen=True)
class TemporalLatentConfig:
    """Where and how the temporal state is injected into the spatial network."""

    site: str = "bottleneck"
    hidden_channels: int = 256
    adapter_init_gate: float = 1.0e-3
    norm: str = "group"
    groups: int = 8
    time_feature_hidden: int = 64

    @staticmethod
    def parse(value: Any, where: str) -> "TemporalLatentConfig":
        data = _require_mapping(value, where)
        allowed = {
            "site",
            "hidden_channels",
            "adapter_init_gate",
            "norm",
            "groups",
            "time_feature_hidden",
        }
        _reject_unknown(data, allowed, where)
        return TemporalLatentConfig(
            site=_as_choice(data.get("site"), f"{where}.site", ("bottleneck",), "bottleneck"),
            hidden_channels=_as_int(
                data.get("hidden_channels"), f"{where}.hidden_channels", 256, minimum=8
            ),
            adapter_init_gate=_as_float(
                data.get("adapter_init_gate"), f"{where}.adapter_init_gate", 1.0e-3, minimum=0.0
            ),
            norm=_as_choice(data.get("norm"), f"{where}.norm", ("group", "none"), "group"),
            groups=_as_int(data.get("groups"), f"{where}.groups", 8, minimum=1),
            time_feature_hidden=_as_int(
                data.get("time_feature_hidden"), f"{where}.time_feature_hidden", 64, minimum=4
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "site": self.site,
            "hidden_channels": self.hidden_channels,
            "adapter_init_gate": self.adapter_init_gate,
            "norm": self.norm,
            "groups": self.groups,
            "time_feature_hidden": self.time_feature_hidden,
        }


@dataclass(frozen=True)
class RecurrentBackendConfig:
    """ConvGRU / ConvLSTM settings."""

    cell: str = "convgru"
    kernel_size: int = 3
    num_layers: int = 1
    dilations: tuple[int, ...] = (1,)

    @staticmethod
    def parse(value: Any, where: str) -> "RecurrentBackendConfig":
        data = _require_mapping(value, where)
        allowed = {"cell", "kernel_size", "num_layers", "dilations"}
        _reject_unknown(data, allowed, where)
        num_layers = _as_int(data.get("num_layers"), f"{where}.num_layers", 1, minimum=1)
        dilations = _as_int_list(data.get("dilations"), f"{where}.dilations") or [1] * num_layers
        if len(dilations) != num_layers:
            raise TemporalConfigError(
                f"{where}.dilations has {len(dilations)} entries but num_layers={num_layers}. "
                "Provide one dilation per layer (multi-timescale memory) or omit the key."
            )
        kernel = _as_int(data.get("kernel_size"), f"{where}.kernel_size", 3, minimum=1)
        if kernel % 2 == 0:
            raise TemporalConfigError(f"{where}.kernel_size must be odd, got {kernel}.")
        return RecurrentBackendConfig(
            cell=_as_choice(data.get("cell"), f"{where}.cell", ("convgru", "convlstm"), "convgru"),
            kernel_size=kernel,
            num_layers=num_layers,
            dilations=tuple(dilations),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell": self.cell,
            "kernel_size": self.kernel_size,
            "num_layers": self.num_layers,
            "dilations": list(self.dilations),
        }


@dataclass(frozen=True)
class MambaBackendConfig:
    """Temporal-Mamba (selective state space) settings.

    ``implementation`` controls kernel selection and is deliberately explicit:

    ``fused``
        Require ``mamba_ssm``. Raises if unavailable -- no substitution.
    ``reference``
        Use the in-repo pure-PyTorch SSD scan. Same mathematics and same
        parameters as the fused path, just slower.
    ``auto``
        Prefer ``fused``, fall back to ``reference`` with a loud warning and a
        recorded provenance field. Never falls back to a *different*
        architecture.
    """

    implementation: str = "auto"
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 2
    headdim: int = 32
    spatial_mixing: str = "conv3x3"
    scan_axis: str = "time"
    time_scale_from_metadata: bool = True
    dt_min: float = 0.001
    dt_max: float = 0.1

    @staticmethod
    def parse(value: Any, where: str) -> "MambaBackendConfig":
        data = _require_mapping(value, where)
        allowed = {
            "implementation",
            "d_state",
            "d_conv",
            "expand",
            "n_layers",
            "headdim",
            "spatial_mixing",
            "scan_axis",
            "time_scale_from_metadata",
            "dt_min",
            "dt_max",
        }
        _reject_unknown(data, allowed, where)
        scan_axis = _as_choice(data.get("scan_axis"), f"{where}.scan_axis", ("time",), "time")
        dt_min = _as_float(data.get("dt_min"), f"{where}.dt_min", 0.001, minimum=1e-8)
        dt_max = _as_float(data.get("dt_max"), f"{where}.dt_max", 0.1, minimum=1e-8)
        if dt_max <= dt_min:
            raise TemporalConfigError(f"{where}.dt_max must exceed dt_min.")
        return MambaBackendConfig(
            implementation=_as_choice(
                data.get("implementation"),
                f"{where}.implementation",
                ("auto", "fused", "reference"),
                "auto",
            ),
            d_state=_as_int(data.get("d_state"), f"{where}.d_state", 16, minimum=1),
            d_conv=_as_int(data.get("d_conv"), f"{where}.d_conv", 4, minimum=1),
            expand=_as_int(data.get("expand"), f"{where}.expand", 2, minimum=1),
            n_layers=_as_int(data.get("n_layers"), f"{where}.n_layers", 2, minimum=1),
            headdim=_as_int(data.get("headdim"), f"{where}.headdim", 32, minimum=1),
            spatial_mixing=_as_choice(
                data.get("spatial_mixing"),
                f"{where}.spatial_mixing",
                ("conv3x3", "conv5x5", "none"),
                "conv3x3",
            ),
            scan_axis=scan_axis,
            time_scale_from_metadata=_as_bool(
                data.get("time_scale_from_metadata"), f"{where}.time_scale_from_metadata", True
            ),
            dt_min=dt_min,
            dt_max=dt_max,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "n_layers": self.n_layers,
            "headdim": self.headdim,
            "spatial_mixing": self.spatial_mixing,
            "scan_axis": self.scan_axis,
            "time_scale_from_metadata": self.time_scale_from_metadata,
            "dt_min": self.dt_min,
            "dt_max": self.dt_max,
        }


@dataclass(frozen=True)
class TendencyLossConfig:
    enabled: bool = False
    weight: float = 0.0
    predictands: dict[str, float] = field(default_factory=dict)
    normalize_by_interval: bool = True

    @staticmethod
    def parse(value: Any, where: str) -> "TendencyLossConfig":
        data = _require_mapping(value, where)
        _reject_unknown(data, {"enabled", "weight", "predictands", "normalize_by_interval"}, where)
        return TendencyLossConfig(
            enabled=_as_bool(data.get("enabled"), f"{where}.enabled", False),
            weight=_as_float(data.get("weight"), f"{where}.weight", 0.0, minimum=0.0),
            predictands=_as_weight_map(data.get("predictands"), f"{where}.predictands"),
            normalize_by_interval=_as_bool(
                data.get("normalize_by_interval"), f"{where}.normalize_by_interval", True
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "weight": self.weight,
            "predictands": dict(self.predictands),
            "normalize_by_interval": self.normalize_by_interval,
        }


@dataclass(frozen=True)
class AccumulationLossConfig:
    enabled: bool = False
    weight: float = 0.0
    windows: tuple[int, ...] = (3,)
    predictands: tuple[str, ...] = ()

    @staticmethod
    def parse(value: Any, where: str) -> "AccumulationLossConfig":
        data = _require_mapping(value, where)
        _reject_unknown(data, {"enabled", "weight", "windows", "predictands"}, where)
        windows = tuple(_as_int_list(data.get("windows"), f"{where}.windows", minimum=2) or (3,))
        preds = data.get("predictands") or []
        if not isinstance(preds, (list, tuple)):
            raise TemporalConfigError(f"{where}.predictands must be a list of variable names.")
        return AccumulationLossConfig(
            enabled=_as_bool(data.get("enabled"), f"{where}.enabled", False),
            weight=_as_float(data.get("weight"), f"{where}.weight", 0.0, minimum=0.0),
            windows=windows,
            predictands=tuple(str(p) for p in preds),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "weight": self.weight,
            "windows": list(self.windows),
            "predictands": list(self.predictands),
        }


@dataclass(frozen=True)
class OccurrenceLossConfig:
    enabled: bool = False
    weight: float = 0.0
    variable: str = "pr"
    wet_threshold: float = 1.0
    transition_weight: float = 0.5
    softness: float = 0.5

    @staticmethod
    def parse(value: Any, where: str) -> "OccurrenceLossConfig":
        data = _require_mapping(value, where)
        _reject_unknown(
            data,
            {"enabled", "weight", "variable", "wet_threshold", "transition_weight", "softness"},
            where,
        )
        return OccurrenceLossConfig(
            enabled=_as_bool(data.get("enabled"), f"{where}.enabled", False),
            weight=_as_float(data.get("weight"), f"{where}.weight", 0.0, minimum=0.0),
            variable=str(data.get("variable") or "pr"),
            wet_threshold=_as_float(data.get("wet_threshold"), f"{where}.wet_threshold", 1.0, minimum=0.0),
            transition_weight=_as_float(
                data.get("transition_weight"), f"{where}.transition_weight", 0.5, minimum=0.0
            ),
            softness=_as_float(data.get("softness"), f"{where}.softness", 0.5, minimum=1e-6),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "weight": self.weight,
            "variable": self.variable,
            "wet_threshold": self.wet_threshold,
            "transition_weight": self.transition_weight,
            "softness": self.softness,
        }


@dataclass(frozen=True)
class LagAutocorrLossConfig:
    enabled: bool = False
    weight: float = 0.0
    lags: tuple[int, ...] = (1,)
    min_samples: int = 64
    predictands: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def parse(value: Any, where: str) -> "LagAutocorrLossConfig":
        data = _require_mapping(value, where)
        _reject_unknown(data, {"enabled", "weight", "lags", "min_samples", "predictands"}, where)
        return LagAutocorrLossConfig(
            enabled=_as_bool(data.get("enabled"), f"{where}.enabled", False),
            weight=_as_float(data.get("weight"), f"{where}.weight", 0.0, minimum=0.0),
            lags=tuple(_as_int_list(data.get("lags"), f"{where}.lags") or (1,)),
            min_samples=_as_int(data.get("min_samples"), f"{where}.min_samples", 64, minimum=8),
            predictands=_as_weight_map(data.get("predictands"), f"{where}.predictands"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "weight": self.weight,
            "lags": list(self.lags),
            "min_samples": self.min_samples,
            "predictands": dict(self.predictands),
        }


@dataclass(frozen=True)
class TemporalLossConfig:
    """Temporal objectives layered on top of the existing per-frame spatial loss."""

    per_frame_weight: float = 1.0
    predictand_weights: dict[str, float] = field(default_factory=dict)
    tendency: TendencyLossConfig = field(default_factory=TendencyLossConfig)
    accumulation: AccumulationLossConfig = field(default_factory=AccumulationLossConfig)
    occurrence: OccurrenceLossConfig = field(default_factory=OccurrenceLossConfig)
    lag_autocorr: LagAutocorrLossConfig = field(default_factory=LagAutocorrLossConfig)

    @staticmethod
    def parse(value: Any, where: str) -> "TemporalLossConfig":
        data = _require_mapping(value, where)
        allowed = {
            "per_frame_weight",
            "predictand_weights",
            "tendency",
            "accumulation",
            "occurrence",
            "lag_autocorr",
        }
        _reject_unknown(data, allowed, where)
        return TemporalLossConfig(
            per_frame_weight=_as_float(
                data.get("per_frame_weight"), f"{where}.per_frame_weight", 1.0, minimum=0.0
            ),
            predictand_weights=_as_weight_map(
                data.get("predictand_weights"), f"{where}.predictand_weights"
            ),
            tendency=TendencyLossConfig.parse(data.get("tendency"), f"{where}.tendency"),
            accumulation=AccumulationLossConfig.parse(
                data.get("accumulation"), f"{where}.accumulation"
            ),
            occurrence=OccurrenceLossConfig.parse(data.get("occurrence"), f"{where}.occurrence"),
            lag_autocorr=LagAutocorrLossConfig.parse(
                data.get("lag_autocorr"), f"{where}.lag_autocorr"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_frame_weight": self.per_frame_weight,
            "predictand_weights": dict(self.predictand_weights),
            "tendency": self.tendency.to_dict(),
            "accumulation": self.accumulation.to_dict(),
            "occurrence": self.occurrence.to_dict(),
            "lag_autocorr": self.lag_autocorr.to_dict(),
        }


@dataclass(frozen=True)
class UnfreezeStage:
    epoch: int
    modules: tuple[str, ...]
    lr_scale: float = 1.0

    MODULES = ("temporal", "decoder", "encoder", "backbone", "all")

    @staticmethod
    def parse(value: Any, where: str) -> "UnfreezeStage":
        data = _require_mapping(value, where)
        _reject_unknown(data, {"epoch", "modules", "lr_scale"}, where)
        mods = data.get("modules")
        if not isinstance(mods, (list, tuple)) or not mods:
            raise TemporalConfigError(f"{where}.modules must be a non-empty list.")
        resolved: list[str] = []
        for i, m in enumerate(mods):
            resolved.append(_as_choice(m, f"{where}.modules[{i}]", UnfreezeStage.MODULES))
        return UnfreezeStage(
            epoch=_as_int(data.get("epoch"), f"{where}.epoch", minimum=0),
            modules=tuple(resolved),
            lr_scale=_as_float(data.get("lr_scale"), f"{where}.lr_scale", 1.0, minimum=0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "modules": list(self.modules), "lr_scale": self.lr_scale}


@dataclass(frozen=True)
class TemporalFreezeConfig:
    """Staged unfreezing so the pretrained spatial weights are not wrecked."""

    backbone: bool = True
    encoder: bool = True
    decoder: bool = False
    #: Normally false -- the temporal module is the thing being trained. Setting it
    #: true (together with ``latent.adapter_init_gate: 0``) makes the run a pure
    #: spatial fine-tuning control: identical data order, identical decoder
    #: learning rate and identical step count, with the temporal pathway inert.
    temporal: bool = False
    unfreeze_schedule: tuple[UnfreezeStage, ...] = ()
    lr_temporal: float = 1.0e-4
    lr_decoder: float = 5.0e-5
    lr_encoder: float = 1.0e-5
    lr_backbone: float = 1.0e-6

    @staticmethod
    def parse(value: Any, where: str) -> "TemporalFreezeConfig":
        data = _require_mapping(value, where)
        allowed = {
            "backbone",
            "encoder",
            "decoder",
            "temporal",
            "unfreeze_schedule",
            "lr_temporal",
            "lr_decoder",
            "lr_encoder",
            "lr_backbone",
        }
        _reject_unknown(data, allowed, where)
        raw_schedule = data.get("unfreeze_schedule") or []
        if not isinstance(raw_schedule, (list, tuple)):
            raise TemporalConfigError(f"{where}.unfreeze_schedule must be a list.")
        stages = tuple(
            UnfreezeStage.parse(item, f"{where}.unfreeze_schedule[{i}]")
            for i, item in enumerate(raw_schedule)
        )
        epochs = [s.epoch for s in stages]
        if epochs != sorted(epochs):
            raise TemporalConfigError(
                f"{where}.unfreeze_schedule must be ordered by non-decreasing epoch."
            )
        return TemporalFreezeConfig(
            backbone=_as_bool(data.get("backbone"), f"{where}.backbone", True),
            encoder=_as_bool(data.get("encoder"), f"{where}.encoder", True),
            decoder=_as_bool(data.get("decoder"), f"{where}.decoder", False),
            temporal=_as_bool(data.get("temporal"), f"{where}.temporal", False),
            unfreeze_schedule=stages,
            lr_temporal=_as_float(data.get("lr_temporal"), f"{where}.lr_temporal", 1.0e-4, minimum=0.0),
            lr_decoder=_as_float(data.get("lr_decoder"), f"{where}.lr_decoder", 5.0e-5, minimum=0.0),
            lr_encoder=_as_float(data.get("lr_encoder"), f"{where}.lr_encoder", 1.0e-5, minimum=0.0),
            lr_backbone=_as_float(data.get("lr_backbone"), f"{where}.lr_backbone", 1.0e-6, minimum=0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone,
            "encoder": self.encoder,
            "decoder": self.decoder,
            "temporal": self.temporal,
            "unfreeze_schedule": [s.to_dict() for s in self.unfreeze_schedule],
            "lr_temporal": self.lr_temporal,
            "lr_decoder": self.lr_decoder,
            "lr_encoder": self.lr_encoder,
            "lr_backbone": self.lr_backbone,
        }


@dataclass(frozen=True)
class TemporalInferenceConfig:
    """Chunked, state-carrying inference over a long continuous sequence."""

    chunk_length: int = 64
    chunk_warmup: int = 7
    output_dir: str | None = None
    write_netcdf: bool = True
    carry_state_across_chunks: bool = True

    @staticmethod
    def parse(value: Any, where: str) -> "TemporalInferenceConfig":
        data = _require_mapping(value, where)
        allowed = {
            "chunk_length",
            "chunk_warmup",
            "output_dir",
            "write_netcdf",
            "carry_state_across_chunks",
        }
        _reject_unknown(data, allowed, where)
        return TemporalInferenceConfig(
            chunk_length=_as_int(data.get("chunk_length"), f"{where}.chunk_length", 64, minimum=1),
            chunk_warmup=_as_int(data.get("chunk_warmup"), f"{where}.chunk_warmup", 7, minimum=0),
            output_dir=None if data.get("output_dir") is None else str(data.get("output_dir")),
            write_netcdf=_as_bool(data.get("write_netcdf"), f"{where}.write_netcdf", True),
            carry_state_across_chunks=_as_bool(
                data.get("carry_state_across_chunks"), f"{where}.carry_state_across_chunks", True
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_length": self.chunk_length,
            "chunk_warmup": self.chunk_warmup,
            "output_dir": self.output_dir,
            "write_netcdf": self.write_netcdf,
            "carry_state_across_chunks": self.carry_state_across_chunks,
        }


@dataclass(frozen=True)
class TemporalRefinementConfig:
    """How stochastic residual refinement is made sequence-aware.

    ``temporal_conditioning``
        ``none`` keeps the existing per-date refinement path unchanged (and is
        the only setting compatible with existing Phase-2 checkpoints).
        ``time_features`` appends the temporal metadata vector to the refiner's
        conditioning. ``latent_state`` additionally supplies the Phase-1
        temporal latent.
    ``noise``
        ``iid_per_frame`` is the legacy behaviour. ``ar1_correlated`` draws
        temporally correlated noise with lag-1 coefficient ``noise_rho`` so a
        member's trajectory is coherent in time. ``noise_rho: 1.0`` (identical
        noise every date) is rejected: it manufactures persistence rather than
        modelling it.
    """

    temporal_conditioning: str = "none"
    noise: str = "iid_per_frame"
    noise_rho: float = 0.0
    ensemble_size: int = 1
    per_member_state: bool = True

    @staticmethod
    def parse(value: Any, where: str) -> "TemporalRefinementConfig":
        data = _require_mapping(value, where)
        allowed = {
            "temporal_conditioning",
            "noise",
            "noise_rho",
            "ensemble_size",
            "per_member_state",
        }
        _reject_unknown(data, allowed, where)
        noise = _as_choice(
            data.get("noise"), f"{where}.noise", ("iid_per_frame", "ar1_correlated"), "iid_per_frame"
        )
        rho = _as_float(data.get("noise_rho"), f"{where}.noise_rho", 0.0, minimum=0.0)
        if rho >= 1.0:
            raise TemporalConfigError(
                f"{where}.noise_rho must be < 1.0, got {rho}. rho=1 reuses identical noise at "
                "every date, which imposes artificial persistence instead of modelling it."
            )
        if noise == "iid_per_frame" and rho != 0.0:
            raise TemporalConfigError(
                f"{where}.noise_rho={rho} is only meaningful with noise='ar1_correlated'."
            )
        return TemporalRefinementConfig(
            temporal_conditioning=_as_choice(
                data.get("temporal_conditioning"),
                f"{where}.temporal_conditioning",
                ("none", "time_features", "latent_state"),
                "none",
            ),
            noise=noise,
            noise_rho=rho,
            ensemble_size=_as_int(data.get("ensemble_size"), f"{where}.ensemble_size", 1, minimum=1),
            per_member_state=_as_bool(
                data.get("per_member_state"), f"{where}.per_member_state", True
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "temporal_conditioning": self.temporal_conditioning,
            "noise": self.noise,
            "noise_rho": self.noise_rho,
            "ensemble_size": self.ensemble_size,
            "per_member_state": self.per_member_state,
        }


@dataclass(frozen=True)
class TemporalEvaluationConfig:
    """Evaluation controls, including the event-alignment guard."""

    event_paired: bool = True
    wet_threshold: float = 1.0
    autocorr_lags: tuple[int, ...] = (1, 2, 3, 5)
    accumulation_windows: tuple[int, ...] = (3, 5)
    deseasonalize: bool = True
    boundary_width: int = 8
    block_bootstrap_length: int = 30
    n_bootstrap: int = 500

    @staticmethod
    def parse(value: Any, where: str) -> "TemporalEvaluationConfig":
        data = _require_mapping(value, where)
        allowed = {
            "event_paired",
            "wet_threshold",
            "autocorr_lags",
            "accumulation_windows",
            "deseasonalize",
            "boundary_width",
            "block_bootstrap_length",
            "n_bootstrap",
        }
        _reject_unknown(data, allowed, where)
        return TemporalEvaluationConfig(
            event_paired=_as_bool(data.get("event_paired"), f"{where}.event_paired", True),
            wet_threshold=_as_float(data.get("wet_threshold"), f"{where}.wet_threshold", 1.0, minimum=0.0),
            autocorr_lags=tuple(
                _as_int_list(data.get("autocorr_lags"), f"{where}.autocorr_lags") or (1, 2, 3, 5)
            ),
            accumulation_windows=tuple(
                _as_int_list(data.get("accumulation_windows"), f"{where}.accumulation_windows", minimum=2)
                or (3, 5)
            ),
            deseasonalize=_as_bool(data.get("deseasonalize"), f"{where}.deseasonalize", True),
            boundary_width=_as_int(data.get("boundary_width"), f"{where}.boundary_width", 8, minimum=0),
            block_bootstrap_length=_as_int(
                data.get("block_bootstrap_length"), f"{where}.block_bootstrap_length", 30, minimum=1
            ),
            n_bootstrap=_as_int(data.get("n_bootstrap"), f"{where}.n_bootstrap", 500, minimum=0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_paired": self.event_paired,
            "wet_threshold": self.wet_threshold,
            "autocorr_lags": list(self.autocorr_lags),
            "accumulation_windows": list(self.accumulation_windows),
            "deseasonalize": self.deseasonalize,
            "boundary_width": self.boundary_width,
            "block_bootstrap_length": self.block_bootstrap_length,
            "n_bootstrap": self.n_bootstrap,
        }


# ----------------------------------------------------------------------------
# top-level
# ----------------------------------------------------------------------------
#: Bumped whenever the temporal state_dict layout changes incompatibly.
TEMPORAL_ARCHITECTURE_VERSION = "temporal-1"


@dataclass(frozen=True)
class TemporalConfig:
    """Fully validated ``temporal:`` block."""

    enabled: bool
    backend: str
    mode: str
    context_length: int
    output_length: int
    warmup_length: int
    sequence_stride: int
    cadence_days: float
    causal: bool
    lead_time_days: float
    include_hour_of_day: bool | str
    include_lead_time: bool
    state: TemporalStateConfig
    latent: TemporalLatentConfig
    recurrent: RecurrentBackendConfig
    mamba: MambaBackendConfig
    losses: TemporalLossConfig
    freeze: TemporalFreezeConfig
    inference: TemporalInferenceConfig
    refinement: TemporalRefinementConfig
    evaluation: TemporalEvaluationConfig
    init_from_spatial_checkpoint: str | None
    resume_from_temporal_checkpoint: str | None
    seed: int
    architecture_version: str = TEMPORAL_ARCHITECTURE_VERSION

    # -- derived ------------------------------------------------------------
    @property
    def supervised_length(self) -> int:
        """Frames that contribute to the loss / are written at inference."""
        return self.output_length

    @property
    def window_length(self) -> int:
        """Total frames loaded per training window."""
        return self.context_length

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "mode": self.mode,
            "context_length": self.context_length,
            "output_length": self.output_length,
            "warmup_length": self.warmup_length,
            "sequence_stride": self.sequence_stride,
            "cadence_days": self.cadence_days,
            "causal": self.causal,
            "lead_time_days": self.lead_time_days,
            "time_features": {
                "include_hour_of_day": self.include_hour_of_day,
                "include_lead_time": self.include_lead_time,
            },
            "state": self.state.to_dict(),
            "latent": self.latent.to_dict(),
            "recurrent": self.recurrent.to_dict(),
            "mamba": self.mamba.to_dict(),
            "losses": self.losses.to_dict(),
            "freeze": self.freeze.to_dict(),
            "inference": self.inference.to_dict(),
            "refinement": self.refinement.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "init_from_spatial_checkpoint": self.init_from_spatial_checkpoint,
            "resume_from_temporal_checkpoint": self.resume_from_temporal_checkpoint,
            "seed": self.seed,
            "architecture_version": self.architecture_version,
        }


_TOP_LEVEL_KEYS = {
    "enabled",
    "backend",
    "mode",
    "context_length",
    "output_length",
    "warmup_length",
    "sequence_stride",
    "cadence_days",
    "causal",
    "lead_time_days",
    "time_features",
    "state",
    "latent",
    "recurrent",
    "mamba",
    "losses",
    "freeze",
    "inference",
    "refinement",
    "evaluation",
    "init_from_spatial_checkpoint",
    "resume_from_temporal_checkpoint",
    "seed",
    # Optional, and normally absent from a hand-written YAML. Accepted so that
    # ``TemporalConfig.to_dict()`` round-trips through this parser (checkpoint
    # metadata stores exactly that dict), and so a config carried over from an
    # incompatible code version is rejected loudly rather than mis-parsed.
    "architecture_version",
}


def parse_temporal_config(raw: Any, *, where: str = "temporal") -> TemporalConfig | None:
    """Parse and validate a ``temporal:`` block.

    Returns ``None`` when the block is absent or ``enabled: false``, which is
    the signal for every call site to take the unmodified legacy spatial path.
    """
    if raw is None:
        return None
    data = _require_mapping(raw, where)
    if not data:
        return None
    _reject_unknown(data, _TOP_LEVEL_KEYS, where)

    enabled = _as_bool(data.get("enabled"), f"{where}.enabled", False)
    if not enabled:
        return None

    declared_version = data.get("architecture_version")
    if declared_version is not None and str(declared_version) != TEMPORAL_ARCHITECTURE_VERSION:
        raise TemporalConfigError(
            f"{where}.architecture_version is {declared_version!r} but this code implements "
            f"{TEMPORAL_ARCHITECTURE_VERSION!r}. The temporal state_dict layout differs; "
            "do not reuse this config against these weights."
        )

    backend = _as_choice(data.get("backend"), f"{where}.backend", ("recurrent", "mamba"))
    mode = _as_choice(data.get("mode"), f"{where}.mode", ("downscaling", "forecasting"), "downscaling")

    context_length = _as_int(data.get("context_length"), f"{where}.context_length", 7, minimum=1)
    output_length = _as_int(
        data.get("output_length"), f"{where}.output_length", context_length, minimum=1
    )
    warmup_length = _as_int(data.get("warmup_length"), f"{where}.warmup_length", 0, minimum=0)
    if output_length > context_length:
        raise TemporalConfigError(
            f"{where}.output_length ({output_length}) cannot exceed context_length "
            f"({context_length}); the model cannot emit more frames than it reads."
        )
    if warmup_length >= context_length:
        raise TemporalConfigError(
            f"{where}.warmup_length ({warmup_length}) must be < context_length ({context_length})."
        )
    if warmup_length + output_length > context_length:
        raise TemporalConfigError(
            f"{where}.warmup_length ({warmup_length}) + output_length ({output_length}) "
            f"exceeds context_length ({context_length}). Warm-up frames are consumed for "
            "state only and cannot also be supervised."
        )

    cadence_days = _as_float(data.get("cadence_days"), f"{where}.cadence_days", 1.0, minimum=1e-9)
    causal = _as_bool(data.get("causal"), f"{where}.causal", True)
    lead_time_days = _as_float(data.get("lead_time_days"), f"{where}.lead_time_days", 0.0, minimum=0.0)

    if mode == "downscaling":
        if not causal:
            raise TemporalConfigError(
                f"{where}.causal must be true in downscaling mode. A non-causal temporal "
                "module would let output date t depend on predictors after t."
            )
        if lead_time_days != 0.0:
            raise TemporalConfigError(
                f"{where}.lead_time_days must be 0 in downscaling mode (got {lead_time_days}). "
                "Existing downscaling cases pair predictors and targets on the same date; "
                "select mode='forecasting' to introduce a lead time explicitly."
            )
    else:  # forecasting
        if lead_time_days <= 0.0:
            raise TemporalConfigError(
                f"{where}.lead_time_days must be > 0 in forecasting mode; otherwise use "
                "mode='downscaling'."
            )

    tf = _require_mapping(data.get("time_features"), f"{where}.time_features")
    _reject_unknown(tf, {"include_hour_of_day", "include_lead_time"}, f"{where}.time_features")
    raw_hour = tf.get("include_hour_of_day", "auto")
    if isinstance(raw_hour, str) and raw_hour.strip().lower() == "auto":
        include_hour: bool | str = "auto"
    else:
        include_hour = _as_bool(raw_hour, f"{where}.time_features.include_hour_of_day", False)
    include_lead = _as_bool(
        tf.get("include_lead_time"), f"{where}.time_features.include_lead_time", mode == "forecasting"
    )
    if include_lead and mode == "downscaling":
        raise TemporalConfigError(
            f"{where}.time_features.include_lead_time is only valid in forecasting mode."
        )

    state = TemporalStateConfig.parse(data.get("state"), f"{where}.state")
    if state.tbptt_chunk and state.tbptt_chunk > context_length:
        raise TemporalConfigError(
            f"{where}.state.tbptt_chunk ({state.tbptt_chunk}) exceeds context_length "
            f"({context_length})."
        )

    latent = TemporalLatentConfig.parse(data.get("latent"), f"{where}.latent")
    if latent.norm == "group" and latent.hidden_channels % latent.groups != 0:
        raise TemporalConfigError(
            f"{where}.latent.hidden_channels ({latent.hidden_channels}) must be divisible by "
            f"{where}.latent.groups ({latent.groups}) for group normalization."
        )

    mamba = MambaBackendConfig.parse(data.get("mamba"), f"{where}.mamba")
    if backend == "mamba":
        inner = latent.hidden_channels * mamba.expand
        if inner % mamba.headdim != 0:
            raise TemporalConfigError(
                f"{where}: latent.hidden_channels*mamba.expand ({inner}) must be divisible by "
                f"mamba.headdim ({mamba.headdim})."
            )

    inference = TemporalInferenceConfig.parse(data.get("inference"), f"{where}.inference")
    if inference.chunk_warmup >= inference.chunk_length and inference.chunk_length > 1:
        raise TemporalConfigError(
            f"{where}.inference.chunk_warmup ({inference.chunk_warmup}) must be < chunk_length "
            f"({inference.chunk_length}) or no frame would be emitted."
        )

    init_spatial = data.get("init_from_spatial_checkpoint")
    resume_temporal = data.get("resume_from_temporal_checkpoint")
    if init_spatial is None and resume_temporal is None:
        raise TemporalConfigError(
            f"{where} requires either init_from_spatial_checkpoint (start from an existing "
            "frame-independent Prithvi-UNet checkpoint) or resume_from_temporal_checkpoint "
            "(continue a trained temporal run). Training a temporal model from scratch is not "
            "the intent of this branch."
        )

    return TemporalConfig(
        enabled=True,
        backend=backend,
        mode=mode,
        context_length=context_length,
        output_length=output_length,
        warmup_length=warmup_length,
        sequence_stride=_as_int(
            data.get("sequence_stride"), f"{where}.sequence_stride", 1, minimum=1
        ),
        cadence_days=cadence_days,
        causal=causal,
        lead_time_days=lead_time_days,
        include_hour_of_day=include_hour,
        include_lead_time=include_lead,
        state=state,
        latent=latent,
        recurrent=RecurrentBackendConfig.parse(data.get("recurrent"), f"{where}.recurrent"),
        mamba=mamba,
        losses=TemporalLossConfig.parse(data.get("losses"), f"{where}.losses"),
        freeze=TemporalFreezeConfig.parse(data.get("freeze"), f"{where}.freeze"),
        inference=inference,
        refinement=TemporalRefinementConfig.parse(data.get("refinement"), f"{where}.refinement"),
        evaluation=TemporalEvaluationConfig.parse(data.get("evaluation"), f"{where}.evaluation"),
        init_from_spatial_checkpoint=None if init_spatial is None else str(init_spatial),
        resume_from_temporal_checkpoint=None if resume_temporal is None else str(resume_temporal),
        seed=_as_int(data.get("seed"), f"{where}.seed", 1234),
    )


def temporal_config_from_experiment(config: Any) -> TemporalConfig | None:
    """Extract the temporal block from an :class:`ExperimentConfig`-like object."""
    raw = getattr(config, "temporal", None)
    return parse_temporal_config(raw)
