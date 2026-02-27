"""Postprocessing helpers for CORDEX inference outputs."""

from __future__ import annotations

import numpy as np
import xarray as xr


def _to_stats(array: xr.DataArray) -> tuple[float, int]:
    values = np.asarray(array.values)
    valid = np.isfinite(values)
    if not valid.any():
        return float("nan"), 0
    min_value = float(np.nanmin(values))
    neg_count = int(np.sum(values[valid] < 0.0))
    return min_value, neg_count


def enforce_pr_nonnegative_xr(
    pred: xr.Dataset | xr.DataArray,
    *,
    pr_var_candidates: tuple[str, ...] = ("pr", "precip", "precipitation"),
    pr_index: int | None = None,
) -> xr.Dataset | xr.DataArray:
    """Clamp precipitation output to be nonnegative and print diagnostics."""
    candidates = tuple(name.lower() for name in pr_var_candidates)

    if isinstance(pred, xr.Dataset):
        pr_name = next((name for name in pred.data_vars if name.lower() in candidates), None)
        if pr_name is None:
            print("[clamp] No precipitation variable found; skipping nonnegative clamp.")
            return pred

        min_before, neg_before = _to_stats(pred[pr_name])
        pred[pr_name] = pred[pr_name].clip(min=0)
        min_after, neg_after = _to_stats(pred[pr_name])
        print(
            f"[clamp] {pr_name}: min_before={min_before:.6g}, min_after={min_after:.6g}, "
            f"negative_count_before={neg_before}, negative_count_after={neg_after}"
        )
        return pred

    if not isinstance(pred, xr.DataArray):
        raise TypeError(f"Unsupported prediction type: {type(pred)}")

    if pred.name and pred.name.lower() in candidates:
        min_before, neg_before = _to_stats(pred)
        clamped = pred.clip(min=0)
        min_after, neg_after = _to_stats(clamped)
        print(
            f"[clamp] {pred.name}: min_before={min_before:.6g}, min_after={min_after:.6g}, "
            f"negative_count_before={neg_before}, negative_count_after={neg_after}"
        )
        return clamped

    for dim in ("variable", "channel"):
        if dim not in pred.dims:
            continue
        if dim in pred.coords:
            labels = [str(value).lower() for value in pred.coords[dim].values]
            for i, label in enumerate(labels):
                if label in candidates:
                    min_before, neg_before = _to_stats(pred.isel({dim: i}))
                    out = pred.copy()
                    out.loc[{dim: pred.coords[dim].values[i]}] = out.isel({dim: i}).clip(min=0)
                    min_after, neg_after = _to_stats(out.isel({dim: i}))
                    print(
                        f"[clamp] {dim}={pred.coords[dim].values[i]}: min_before={min_before:.6g}, "
                        f"min_after={min_after:.6g}, negative_count_before={neg_before}, "
                        f"negative_count_after={neg_after}"
                    )
                    return out
        if pr_index is not None:
            min_before, neg_before = _to_stats(pred.isel({dim: pr_index}))
            out = pred.copy()
            out[{dim: pr_index}] = out.isel({dim: pr_index}).clip(min=0)
            min_after, neg_after = _to_stats(out.isel({dim: pr_index}))
            print(
                f"[clamp] {dim}_index={pr_index}: min_before={min_before:.6g}, min_after={min_after:.6g}, "
                f"negative_count_before={neg_before}, negative_count_after={neg_after}"
            )
            return out

    print("[clamp] Could not locate precipitation channel in DataArray; skipping nonnegative clamp.")
    return pred

