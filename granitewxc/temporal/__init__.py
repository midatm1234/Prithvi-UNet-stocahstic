"""Spatiotemporal extension of the Prithvi-UNet downscaling models.

This package adds *explicit* temporal dependence to the otherwise
frame-independent :class:`~granitewxc.models.cordex_finetune_model.ClimateDownscaleFinetuneUNETModel`.
Temporal state is carried in a spatially organized latent at the U-Net
bottleneck by one of two interchangeable backends:

``recurrent``
    A ConvGRU cell (see :class:`granitewxc.temporal.backends.ConvGRUBackend`).

``mamba``
    A selective state-space (Mamba-2 / SSD) block scanning the **time** axis
    with explicit spatial mixing (see
    :class:`granitewxc.temporal.backends.TemporalMambaBackend`).

Everything in this package is inert unless ``temporal.enabled`` is true in the
experiment YAML. With it disabled the legacy spatial computation path is
preserved bit-for-bit; see :func:`granitewxc.temporal.model.attach_temporal_adapter`.
"""

from granitewxc.temporal.calendar import (
    CalendarSpec,
    TimeFeatureSpec,
    build_time_features,
    elapsed_days,
    find_discontinuities,
    resolve_calendar,
    time_key,
    year_fraction,
)
from granitewxc.temporal.config import (
    TemporalConfig,
    TemporalConfigError,
    parse_temporal_config,
)

__all__ = [
    "CalendarSpec",
    "TimeFeatureSpec",
    "TemporalConfig",
    "TemporalConfigError",
    "build_time_features",
    "elapsed_days",
    "find_discontinuities",
    "parse_temporal_config",
    "resolve_calendar",
    "time_key",
    "year_fraction",
]
