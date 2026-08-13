"""Lightweight quantization diagnostics for CORDEX inference outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _finite(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return flat[np.isfinite(flat)]


@dataclass
class RunningStats:
    max_samples: int = 250_000
    round_decimals: int = 6

    def __post_init__(self) -> None:
        self._sum = 0.0
        self._sumsq = 0.0
        self._count = 0
        self._min = float("inf")
        self._max = float("-inf")
        self._samples: list[np.ndarray] = []

    def add(self, values: np.ndarray) -> None:
        finite = _finite(values)
        if finite.size == 0:
            return
        self._sum += float(finite.sum())
        self._sumsq += float(np.square(finite).sum())
        self._count += int(finite.size)
        self._min = min(self._min, float(finite.min()))
        self._max = max(self._max, float(finite.max()))
        if sum(arr.size for arr in self._samples) < self.max_samples:
            budget = max(0, self.max_samples - sum(arr.size for arr in self._samples))
            self._samples.append(finite[:budget].copy())

    def finalize(self) -> dict[str, Any]:
        if self._count == 0:
            return {
                "count": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        mean = self._sum / self._count
        var = max(0.0, self._sumsq / self._count - mean * mean)
        std = float(np.sqrt(var))
        return {
            "count": int(self._count),
            "mean": round(float(mean), self.round_decimals),
            "std": round(std, self.round_decimals),
            "min": round(float(self._min), self.round_decimals),
            "max": round(float(self._max), self.round_decimals),
        }


def quantization_detector(
    values: np.ndarray,
    *,
    time_window: int = 5,
    spatial_window: int = 20,
    round_decimals: int = 6,
) -> dict[str, Any]:
    del time_window, spatial_window  # kept for API compatibility
    finite = _finite(values)
    if finite.size == 0:
        return {"unique_count": 0, "rounded_unique_count": 0, "is_quantized_suspect": False}
    rounded = np.round(finite, round_decimals)
    unique_count = int(np.unique(finite).size)
    rounded_unique = int(np.unique(rounded).size)
    ratio = float(rounded_unique / max(1, unique_count))
    return {
        "unique_count": unique_count,
        "rounded_unique_count": rounded_unique,
        "rounded_to_raw_unique_ratio": ratio,
        "is_quantized_suspect": bool(unique_count > 0 and ratio < 0.1),
    }


def format_compact_table(rows: Iterable[dict[str, Any]]) -> str:
    lines = ["stage,var,count,mean,std,min,max,quantized_suspect"]
    for row in rows:
        stats = row.get("stats", {})
        quant = row.get("quantization", {})
        lines.append(
            ",".join(
                [
                    str(row.get("stage", "")),
                    str(row.get("var", "")),
                    str(stats.get("count", "")),
                    str(stats.get("mean", "")),
                    str(stats.get("std", "")),
                    str(stats.get("min", "")),
                    str(stats.get("max", "")),
                    str(quant.get("is_quantized_suspect", "")),
                ]
            )
        )
    return "\n".join(lines)


def save_distribution_plot(
    *,
    target_values: dict[str, np.ndarray],
    predicted_values: dict[str, np.ndarray],
    predicted_raw_values: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    var_names = sorted(set(target_values) | set(predicted_values) | set(predicted_raw_values))
    ncols = len(var_names)
    fig, axes = plt.subplots(1, max(1, ncols), figsize=(5 * max(1, ncols), 4), squeeze=False)
    for idx, name in enumerate(var_names):
        ax = axes[0, idx]
        t = _finite(target_values.get(name, np.array([], dtype=np.float32)))
        p = _finite(predicted_values.get(name, np.array([], dtype=np.float32)))
        r = _finite(predicted_raw_values.get(name, np.array([], dtype=np.float32)))
        if t.size > 0:
            ax.hist(t, bins=80, alpha=0.4, density=True, label=f"target:{name}")
        if p.size > 0:
            ax.hist(p, bins=80, alpha=0.4, density=True, label=f"pred:{name}")
        if r.size > 0:
            ax.hist(r, bins=80, alpha=0.4, density=True, label=f"pred_raw:{name}")
        ax.set_title(name)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_json(payload: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
