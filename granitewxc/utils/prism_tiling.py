"""Shared halo/core tiling helpers for the PRISM downscaling workflows.

The NARR-PRISM and MERRA-PRISM inference paths intentionally use the same
implementation.  A tile describes the *output core*.  ``extract_halo_context``
adds predictor context around that core, while only the core prediction is
blended into the domain.  This keeps model padding away from retained pixels
and makes the coordinate frame of spatial target scalers unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def _pair(value: int | Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(value, (int, np.integer)):
        pair = (int(value), int(value))
    else:
        values = tuple(int(v) for v in value)
        if len(values) != 2:
            raise ValueError(f"{name} must contain two integers, got {value!r}")
        pair = values
    if pair[0] < 0 or pair[1] < 0:
        raise ValueError(f"{name} must be non-negative, got {pair}")
    return pair


def tile_origins(total: int, tile: int, stride: int) -> list[int]:
    """Return deterministic fixed-size tile origins with complete coverage."""
    total, tile, stride = int(total), int(tile), int(stride)
    if total <= 0 or tile <= 0 or stride <= 0:
        raise ValueError(
            f"total, tile and stride must be positive, got {total}, {tile}, {stride}"
        )
    tile = min(tile, total)
    if tile == total:
        return [0]
    origins = list(range(0, total - tile + 1, stride))
    final = total - tile
    if not origins or origins[-1] != final:
        origins.append(final)
    return origins


@dataclass(frozen=True)
class TilePlan:
    """Geometry for fixed-size overlapping output cores with input halos."""

    domain_shape: tuple[int, int]
    core_shape: tuple[int, int]
    overlap: tuple[int, int]
    halo: tuple[int, int]
    lat_origins: tuple[int, ...]
    lon_origins: tuple[int, ...]

    @classmethod
    def build(
        cls,
        domain_shape: Sequence[int],
        core_shape: int | Sequence[int],
        overlap: int | Sequence[int] = 0,
        halo: int | Sequence[int] = 0,
    ) -> "TilePlan":
        domain = _pair(domain_shape, "domain_shape")
        core_requested = _pair(core_shape, "core_shape")
        overlap_pair = _pair(overlap, "overlap")
        halo_pair = _pair(halo, "halo")
        if domain[0] == 0 or domain[1] == 0:
            raise ValueError(f"domain_shape must be positive, got {domain}")
        core = (min(core_requested[0], domain[0]), min(core_requested[1], domain[1]))
        if overlap_pair[0] >= core[0] or overlap_pair[1] >= core[1]:
            raise ValueError(
                f"overlap {overlap_pair} must be smaller than core_shape {core}"
            )
        strides = (core[0] - overlap_pair[0], core[1] - overlap_pair[1])
        return cls(
            domain_shape=domain,
            core_shape=core,
            overlap=overlap_pair,
            halo=halo_pair,
            lat_origins=tuple(tile_origins(domain[0], core[0], strides[0])),
            lon_origins=tuple(tile_origins(domain[1], core[1], strides[1])),
        )

    @property
    def context_shape(self) -> tuple[int, int]:
        return (
            self.core_shape[0] + 2 * self.halo[0],
            self.core_shape[1] + 2 * self.halo[1],
        )

    @property
    def positions(self) -> list[tuple[int, int]]:
        return [(y, x) for y in self.lat_origins for x in self.lon_origins]

    @property
    def output_crop(self) -> tuple[int, int, int, int]:
        return (*self.halo, *self.core_shape)

    def assert_globally_aligned(
        self, alignment: int | Sequence[int]
    ) -> None:
        """Require all tile/context origins to share the model's spatial phase."""
        align = _pair(alignment, "alignment")
        if align[0] == 0 or align[1] == 0:
            raise ValueError(f"alignment must be positive, got {align}")
        bad_lat = [origin for origin in self.lat_origins if origin % align[0]]
        bad_lon = [origin for origin in self.lon_origins if origin % align[1]]
        if self.halo[0] % align[0] or self.halo[1] % align[1] or bad_lat or bad_lon:
            raise ValueError(
                "Tile origins and halo must align with the model mask/decoder "
                f"phase {align}; halo={self.halo}, bad_lat={bad_lat}, "
                f"bad_lon={bad_lon}. Choose core-overlap strides and a halo "
                "that are integer multiples of this alignment."
            )


def _axis_blend_weights(length: int, overlap: int, mode: str, floor: float) -> np.ndarray:
    if length <= 0:
        raise ValueError(f"blend length must be positive, got {length}")
    if overlap < 0 or overlap >= length:
        raise ValueError(f"overlap must be in [0, {length}), got {overlap}")
    if mode in {"uniform", "boxcar", "none"} or overlap == 0:
        return np.ones(length, dtype=np.float64)
    if mode not in {"hann", "cosine"}:
        raise ValueError(f"Unsupported blend mode {mode!r}; use hann/cosine/uniform")

    weights = np.ones(length, dtype=np.float64)
    # Half-sample locations keep every weight strictly positive.  Adjacent
    # sin^2/cos^2 ramps are complementary when tiles use the configured stride.
    phase = (np.arange(overlap, dtype=np.float64) + 0.5) / overlap
    ramp = np.sin(0.5 * np.pi * phase) ** 2
    weights[:overlap] = ramp
    weights[-overlap:] = ramp[::-1]
    return np.maximum(weights, float(floor))


def blend_window(
    core_shape: int | Sequence[int],
    overlap: int | Sequence[int],
    *,
    mode: str = "hann",
    floor: float = 1.0e-8,
) -> np.ndarray:
    """Return a separable float64 overlap window for an output core."""
    core = _pair(core_shape, "core_shape")
    overlap_pair = _pair(overlap, "overlap")
    wy = _axis_blend_weights(core[0], overlap_pair[0], mode.lower(), floor)
    wx = _axis_blend_weights(core[1], overlap_pair[1], mode.lower(), floor)
    return np.multiply.outer(wy, wx)


def extract_halo_context(
    field: object,
    origin: Sequence[int],
    core_shape: int | Sequence[int],
    halo: int | Sequence[int],
    *,
    pad_mode: str = "reflect",
) -> object:
    """Extract ``core + 2*halo`` from a ``[..., H, W]`` NumPy/Torch field.

    The full domain is padded first, so every tile has an identical context
    shape and edge cores retain the same position within the model input.
    """
    y0, x0 = _pair(origin, "origin")
    core = _pair(core_shape, "core_shape")
    halo_pair = _pair(halo, "halo")
    shape = tuple(int(v) for v in getattr(field, "shape"))
    if len(shape) < 2:
        raise ValueError(f"field must have at least two dimensions, got {shape}")
    h, w = shape[-2:]
    if y0 + core[0] > h or x0 + core[1] > w:
        raise ValueError(
            f"core origin {(y0, x0)} and shape {core} exceed field shape {(h, w)}"
        )
    hy, hx = halo_pair
    if hy == 0 and hx == 0:
        return field[..., y0 : y0 + core[0], x0 : x0 + core[1]]

    try:
        import torch
        import torch.nn.functional as torch_f
    except ImportError:  # pragma: no cover - NumPy-only installations
        torch = None
        torch_f = None

    if torch is not None and torch.is_tensor(field):
        effective_mode = pad_mode
        if effective_mode == "reflect" and (hy >= h or hx >= w):
            effective_mode = "replicate"
        padded = torch_f.pad(field, (hx, hx, hy, hy), mode=effective_mode)
    else:
        np_mode = {"replicate": "edge"}.get(pad_mode, pad_mode)
        if np_mode == "reflect" and (hy >= h or hx >= w):
            np_mode = "edge"
        pad_width = [(0, 0)] * (len(shape) - 2) + [(hy, hy), (hx, hx)]
        padded = np.pad(np.asarray(field), pad_width, mode=np_mode)
    return padded[..., y0 : y0 + core[0] + 2 * hy, x0 : x0 + core[1] + 2 * hx]


def halo_crop_slices(
    domain_shape: Sequence[int],
    origin: Sequence[int],
    core_shape: int | Sequence[int],
    halo: int | Sequence[int],
) -> tuple[tuple[slice, slice], tuple[int, int, int, int]]:
    """Return clamped source slices and ``(left,right,top,bottom)`` padding.

    This is useful for datasets that can efficiently load only the requested
    coordinate subset instead of materializing the full fine grid first.
    """
    domain = _pair(domain_shape, "domain_shape")
    y0, x0 = _pair(origin, "origin")
    core = _pair(core_shape, "core_shape")
    hy, hx = _pair(halo, "halo")
    if y0 + core[0] > domain[0] or x0 + core[1] > domain[1]:
        raise ValueError(
            f"core origin {(y0, x0)} and shape {core} exceed domain {domain}"
        )
    desired_y0, desired_x0 = y0 - hy, x0 - hx
    desired_y1, desired_x1 = y0 + core[0] + hy, x0 + core[1] + hx
    source_y0, source_x0 = max(0, desired_y0), max(0, desired_x0)
    source_y1, source_x1 = min(domain[0], desired_y1), min(domain[1], desired_x1)
    padding = (
        source_x0 - desired_x0,
        desired_x1 - source_x1,
        source_y0 - desired_y0,
        desired_y1 - source_y1,
    )
    return (slice(source_y0, source_y1), slice(source_x0, source_x1)), padding


def pad_spatial_context(
    field: object,
    padding: Sequence[int],
    *,
    pad_mode: str = "reflect",
) -> object:
    """Pad ``[...,H,W]`` by ``(left,right,top,bottom)`` for a halo crop."""
    values = tuple(int(v) for v in padding)
    if len(values) != 4 or any(v < 0 for v in values):
        raise ValueError(f"padding must be four non-negative integers, got {padding}")
    left, right, top, bottom = values
    if not any(values):
        return field
    shape = tuple(int(v) for v in getattr(field, "shape"))
    if len(shape) < 2:
        raise ValueError(f"field must have at least two dimensions, got {shape}")
    h, w = shape[-2:]
    try:
        import torch
        import torch.nn.functional as torch_f
    except ImportError:  # pragma: no cover
        torch = None
        torch_f = None
    if torch is not None and torch.is_tensor(field):
        effective_mode = pad_mode
        if effective_mode == "reflect" and (
            top >= h or bottom >= h or left >= w or right >= w
        ):
            effective_mode = "replicate"
        return torch_f.pad(field, values, mode=effective_mode)
    np_mode = {"replicate": "edge"}.get(pad_mode, pad_mode)
    if np_mode == "reflect" and (
        top >= h or bottom >= h or left >= w or right >= w
    ):
        np_mode = "edge"
    pad_width = [(0, 0)] * (len(shape) - 2) + [(top, bottom), (left, right)]
    return np.pad(np.asarray(field), pad_width, mode=np_mode)


def crop_prediction_core(
    prediction: object,
    core_shape: int | Sequence[int],
    halo: int | Sequence[int],
) -> object:
    """Crop the retained output core from a halo-context prediction."""
    core = _pair(core_shape, "core_shape")
    hy, hx = _pair(halo, "halo")
    return prediction[..., hy : hy + core[0], hx : hx + core[1]]


class WeightedTileStitcher:
    """Numerically stable weighted accumulator for overlapping tile cores."""

    def __init__(self, n_channels: int, domain_shape: Sequence[int]) -> None:
        self.n_channels = int(n_channels)
        self.domain_shape = _pair(domain_shape, "domain_shape")
        if self.n_channels <= 0:
            raise ValueError(f"n_channels must be positive, got {self.n_channels}")
        self.accumulator = np.zeros(
            (self.n_channels, *self.domain_shape), dtype=np.float64
        )
        self.weight = np.zeros(self.domain_shape, dtype=np.float64)

    def add(self, tile: np.ndarray, origin: Sequence[int], window: np.ndarray) -> None:
        values = np.asarray(tile, dtype=np.float64)
        weights = np.asarray(window, dtype=np.float64)
        if values.ndim == 2 and self.n_channels == 1:
            values = values[np.newaxis]
        if values.ndim != 3 or values.shape[0] != self.n_channels:
            raise ValueError(
                f"tile must have shape [C,H,W] with C={self.n_channels}, got {values.shape}"
            )
        if tuple(values.shape[-2:]) != tuple(weights.shape):
            raise ValueError(
                f"tile spatial shape {values.shape[-2:]} != window shape {weights.shape}"
            )
        y0, x0 = _pair(origin, "origin")
        y1, x1 = y0 + weights.shape[0], x0 + weights.shape[1]
        if y1 > self.domain_shape[0] or x1 > self.domain_shape[1]:
            raise ValueError(
                f"tile {(y0, x0, y1, x1)} exceeds domain {self.domain_shape}"
            )
        finite = np.all(np.isfinite(values), axis=0)
        effective_weight = np.where(finite, weights, 0.0)
        self.accumulator[:, y0:y1, x0:x1] += (
            np.where(np.isfinite(values), values, 0.0) * effective_weight[np.newaxis]
        )
        self.weight[y0:y1, x0:x1] += effective_weight

    def finalize(self, *, require_full_coverage: bool = True) -> np.ndarray:
        uncovered = self.weight <= 0.0
        if require_full_coverage and bool(uncovered.any()):
            raise ValueError(
                f"Tile stitching left {int(uncovered.sum())} domain cells uncovered"
            )
        out = np.full_like(self.accumulator, np.nan, dtype=np.float64)
        valid = ~uncovered
        out[:, valid] = self.accumulator[:, valid] / self.weight[valid]
        return out


def overlap_crossovers(origins: Iterable[int], tile_size: int) -> list[int]:
    """Return centers of the overlap shared by each adjacent tile pair."""
    ordered = sorted({int(v) for v in origins})
    positions: list[int] = []
    for left, right in zip(ordered[:-1], ordered[1:]):
        overlap_start = right
        overlap_stop = left + int(tile_size)
        if overlap_start < overlap_stop:
            positions.append((overlap_start + overlap_stop - 1) // 2)
    return positions


def boundary_gradient_ratio(
    field: np.ndarray,
    positions: Iterable[int],
    axis: int,
    *,
    half_width: int = 2,
) -> float:
    """Mean absolute gradient in boundary bands divided by the global mean."""
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or axis not in (0, 1):
        raise ValueError(f"field must be 2-D and axis 0/1, got {values.shape}, {axis}")
    grad = np.abs(np.diff(values, axis=axis))
    bands: list[np.ndarray] = []
    for position in sorted({int(v) for v in positions}):
        lo = max(0, position - int(half_width))
        hi = min(grad.shape[axis], position + int(half_width) + 1)
        if lo >= hi:
            continue
        bands.append(grad[lo:hi, :] if axis == 0 else grad[:, lo:hi])
    if not bands:
        return float("nan")
    band_mean = np.nanmean(np.concatenate([band.ravel() for band in bands]))
    global_mean = np.nanmean(grad)
    if not np.isfinite(global_mean) or global_mean == 0.0:
        return float("nan")
    return float(band_mean / global_mean)


def overlap_disagreement(
    first: np.ndarray,
    first_origin: Sequence[int],
    second: np.ndarray,
    second_origin: Sequence[int],
) -> dict[str, float]:
    """Compare two ``[C,H,W]`` predictions over their shared domain pixels."""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.ndim == 2:
        a = a[np.newaxis]
    if b.ndim == 2:
        b = b[np.newaxis]
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0]:
        raise ValueError(f"predictions must be compatible [C,H,W], got {a.shape}, {b.shape}")
    ay, ax = _pair(first_origin, "first_origin")
    by, bx = _pair(second_origin, "second_origin")
    y0, x0 = max(ay, by), max(ax, bx)
    y1, x1 = min(ay + a.shape[-2], by + b.shape[-2]), min(
        ax + a.shape[-1], bx + b.shape[-1]
    )
    if y0 >= y1 or x0 >= x1:
        return {"rmse": float("nan"), "mae": float("nan"), "count": 0}
    a_shared = a[:, y0 - ay : y1 - ay, x0 - ax : x1 - ax]
    b_shared = b[:, y0 - by : y1 - by, x0 - bx : x1 - bx]
    delta = a_shared - b_shared
    finite = np.isfinite(delta)
    if not finite.any():
        return {"rmse": float("nan"), "mae": float("nan"), "count": 0}
    return {
        "rmse": float(np.sqrt(np.mean(np.square(delta[finite])))),
        "mae": float(np.mean(np.abs(delta[finite]))),
        "count": int(finite.sum()),
    }
