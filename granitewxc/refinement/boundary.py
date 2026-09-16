"""Physical-field diagnostics on every valid domain-edge cell."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_cdt


def boundary_distance(valid_domain: np.ndarray) -> np.ndarray:
    """Distance in cells to a valid-domain boundary; invalid cells are -1.

    The perimeter has distance zero. Chebyshev distance treats diagonal
    adjacency to missing cells as a boundary too. No geographic cells are cut.
    """
    valid = np.asarray(valid_domain, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("valid_domain must have [height, width] shape")
    distance = distance_transform_cdt(np.pad(valid, 1), metric="chessboard")[1:-1, 1:-1] - 1
    return np.where(valid, distance, -1)


def boundary_regions(valid_domain, latitude=None, longitude=None):
    distance = boundary_distance(valid_domain)
    valid = distance >= 0
    regions = {"domain": valid, "interior": distance >= 16}
    for lo, hi in ((0, 0), (1, 1), (2, 2), (3, 3), (4, 7), (8, 15)):
        regions[f"distance_{lo}" if lo == hi else f"distance_{lo}_{hi}"] = (distance >= lo) & (distance <= hi)
    yy, xx = np.indices(valid.shape)
    height, width = valid.shape
    north_first = latitude is None or latitude[0] > latitude[-1]
    west_first = longitude is None or longitude[0] < longitude[-1]
    north = yy if north_first else height - 1 - yy
    west = xx if west_first else width - 1 - xx
    south, east = height - 1 - north, width - 1 - west
    for width in (1, 2, 4, 8, 16):
        regions[f"outer_{width}"] = valid & (distance < width)
        for name, edge in (("north", north), ("south", south), ("west", west), ("east", east)):
            regions[f"{name}_{width}"] = valid & (edge < width)
        regions[f"corners_{width}"] = valid & (np.minimum(north, south) < width) & (np.minimum(east, west) < width)
    return distance, regions


def physical_metrics(prediction, truth, valid=None, *, region=None, precipitation=False, wet_threshold=1.0):
    """Paired physical metrics for [time,H,W], retaining valid dry-day zeros."""
    prediction, truth = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError("prediction and truth must share [time,height,width] shape")
    mask = np.isfinite(truth) if valid is None else np.asarray(valid, dtype=bool) & np.isfinite(truth)
    if region is not None:
        mask = mask & np.asarray(region, dtype=bool)[None]
    if np.any(mask & ~np.isfinite(prediction)):
        raise ValueError("Nonfinite prediction on valid physical cells")
    if not mask.any():
        return {"count": 0}
    error = np.where(mask, prediction - truth, 0.0)
    counts = mask.sum(axis=0)
    bias_map = np.divide(error.sum(axis=0), counts, out=np.full(counts.shape, np.nan), where=counts > 0)
    result = {"count": int(mask.sum()), "bias": float(error.sum() / mask.sum()),
              "mae": float(np.abs(error).sum() / mask.sum()),
              "rmse": float(np.sqrt(np.square(error).sum() / mask.sum())),
              "climatological_absolute_bias": float(np.nanmean(np.abs(bias_map)))}
    result["sample_days"] = int(mask.any(axis=(1,2)).sum())
    result["mean_absolute_bias"] = result["climatological_absolute_bias"]
    pred_climate = np.divide(np.where(mask,prediction,0.).sum(0),counts,out=np.zeros(counts.shape),where=counts>0)
    true_climate = np.divide(np.where(mask,truth,0.).sum(0),counts,out=np.zeros(counts.shape),where=counts>0)
    covered = counts>0
    pmean,ymean = pred_climate[covered],true_climate[covered]
    result["climatological_spatial_correlation"] = float(np.corrcoef(pmean,ymean)[0,1]) if len(pmean)>1 and pmean.std()>0 and ymean.std()>0 else None
    p, y = prediction[mask], truth[mask]
    # Pooled space-time quantiles are descriptive; they are not estimates of
    # per-gridcell return levels or a claim of independent spatial samples.
    for q in (95,99):
        result[f"pooled_p{q}_error"] = float(np.percentile(p,q)-np.percentile(y,q)) if len(y)>=100 else None
    result["temporal_p95_supported"] = bool(np.min(counts[covered])>=200)
    result["temporal_p99_supported"] = bool(np.min(counts[covered])>=1000)
    if precipitation:
        pwet, ywet = p >= wet_threshold, y >= wet_threshold
        result["wet_frequency_bias"] = float(pwet.mean() - ywet.mean())
        result["wet_intensity_bias"] = float(p[pwet].mean() - y[ywet].mean()) if pwet.any() and ywet.any() else None
        for q in (95, 99):
            result[f"p{q}_error"] = float(np.percentile(p, q) - np.percentile(y, q)) if len(y) >= 100 else None
    return result
