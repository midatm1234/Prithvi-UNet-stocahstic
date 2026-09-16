"""Regenerate regional diagnostics from saved temporal experiment arrays.

Prediction/target values are read only from the archive selected by manifest.json
(or predictions.npz for older experiments without artifact paths). Latitude/longitude
are read from that archive, --coordinates, or coordinate metadata in the target
NetCDF named in resolved.yaml. No model execution or training occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _meshes(lat, lon, shape):
    lat, lon = np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)
    if lat.ndim == lon.ndim == 1:
        if (len(lat), len(lon)) != tuple(shape):
            raise ValueError(f"Coordinates {(len(lat), len(lon))} do not match saved grid {shape}.")
        lon, lat = np.meshgrid(lon, lat)
    if lat.shape != tuple(shape) or lon.shape != tuple(shape):
        raise ValueError("Physical coordinates must be 1D axes or matching 2D latitude/longitude grids.")
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("Physical coordinate arrays must be finite.")
    if np.ptp(lon) > 180:
        raise ValueError("This quadrant diagnostic requires a non-dateline-crossing longitude domain.")
    return lat, lon


def _coordinates(archive, raw, shape, sidecar, output):
    if "lat" in archive and "lon" in archive:
        lat, lon = archive["lat"], archive["lon"]
        provenance = {"kind": "prediction_archive"}
    elif sidecar is not None:
        with np.load(sidecar, allow_pickle=False) as coordinates:
            lat, lon = coordinates["lat"], coordinates["lon"]
        provenance = {"kind": "coordinate_sidecar", "path": str(sidecar.resolve()), "sha256": _digest(sidecar)}
    else:
        import xarray as xr
        paths = raw.get("data", {}).get("test_target_paths") or []
        if not paths:
            raise ValueError("Saved arrays have no physical coordinates; supply --coordinates with lat/lon arrays.")
        source = Path(paths[0])
        with xr.open_dataset(source, decode_times=False) as dataset:
            lat_name = next((key for key in ("lat", "latitude") if key in dataset), None)
            lon_name = next((key for key in ("lon", "longitude") if key in dataset), None)
            if lat_name is None or lon_name is None:
                raise ValueError(f"No physical latitude/longitude in {source}; supply --coordinates.")
            # Only coordinate values are accessed; no verification data values.
            lat = dataset[lat_name].values.copy()
            lon = dataset[lon_name].values.copy()
        provenance = {"kind": "target_file_coordinate_metadata_only", "path": str(source.resolve()),
                      "latitude_name": lat_name, "longitude_name": lon_name}
    lat_grid, lon_grid = _meshes(lat, lon, shape)
    coordinates_path = output / "coordinates.npz"
    np.savez_compressed(coordinates_path, lat=lat, lon=lon)
    provenance["saved_coordinate_sha256"] = _digest(coordinates_path)
    return lat_grid, lon_grid, provenance


def regions_for_grid(lat, lon, boundary_width):
    height, width = lat.shape
    if boundary_width < 1 or 2 * boundary_width >= min(height, width):
        raise ValueError("boundary_width must leave a nonempty interior.")
    boundary = np.ones(lat.shape, dtype=bool)
    boundary[boundary_width:-boundary_width, boundary_width:-boundary_width] = False
    lat_median, lon_median = float(np.median(lat)), float(np.median(lon))
    regions = {"full_domain": np.ones(lat.shape, dtype=bool), "boundary": boundary,
               "interior": ~boundary, "southeast": (lat < lat_median) & (lon > lon_median)}
    definitions = {}
    for name, mask in regions.items():
        if not mask.any():
            raise ValueError(f"Region {name} has no cells.")
        definitions[name] = {"grid_cells": int(mask.sum()),
            "latitude_bounds_degrees_north": [float(lat[mask].min()), float(lat[mask].max())],
            "longitude_bounds_degrees_east": [float(lon[mask].min()), float(lon[mask].max())]}
    definitions["boundary"]["width_grid_cells"] = int(boundary_width)
    definitions["southeast"]["definition"] = "latitude < domain median AND longitude > domain median"
    definitions["southeast"]["latitude_median"] = lat_median
    definitions["southeast"]["longitude_median"] = lon_median
    return regions, definitions


def region_metrics(prediction, target, valid_mask, region, *, wet_threshold=None):
    valid = valid_mask.astype(bool) & np.isfinite(prediction) & np.isfinite(target) & region[None]
    count = int(valid.sum())
    if count == 0:
        return {"status": "no_valid_samples", "n_valid_gridcell_days": 0}
    pred = prediction[valid].astype(np.float64)
    truth = target[valid].astype(np.float64)
    error = pred - truth
    qp, qt = float(np.quantile(pred, 0.99)), float(np.quantile(truth, 0.99))
    result = {"status": "complete", "n_valid_gridcell_days": count,
              "n_valid_cells": int(np.any(valid, axis=0).sum()),
              "bias": float(error.mean()), "rmse": float(np.sqrt(np.square(error).mean())),
              "q99_pred": qp, "q99_target": qt, "q99_error": qp - qt}
    if wet_threshold is not None:
        wet_p, wet_t = pred > wet_threshold, truth > wet_threshold
        result["wet_day"] = {
            "threshold": float(wet_threshold), "comparison": ">",
            "frequency_pred": float(wet_p.mean()), "frequency_target": float(wet_t.mean()),
            "frequency_error": float(wet_p.mean() - wet_t.mean()),
            "mean_wet_amount_pred": float(pred[wet_p].mean()) if wet_p.any() else None,
            "mean_wet_amount_target": float(truth[wet_t].mean()) if wet_t.any() else None,
        }
    return result


def _time_mean(field, valid):
    count = valid.sum(axis=0)
    total = np.where(valid, field, 0).sum(axis=0, dtype=np.float64)
    return np.divide(total, count, out=np.full(count.shape, np.nan), where=count > 0)


def _figure(pred, truth, valid, lat, lon, region_masks, *, variable, units, variant, date_range, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pred_mean, target_mean = _time_mean(pred, valid), _time_mean(truth, valid)
    bias = pred_mean - target_mean
    finite = np.concatenate((pred_mean[np.isfinite(pred_mean)], target_mean[np.isfinite(target_mean)]))
    lower, upper = float(finite.min()), float(finite.max())
    bound = max(float(np.nanmax(np.abs(bias))), 1e-8)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for i, (field, label) in enumerate(zip((pred_mean, target_mean, bias), ("Prediction mean", "Target mean", "Mean signed bias"))):
        ax = axes[i]
        im = ax.pcolormesh(lon, lat, np.ma.masked_invalid(field), shading="auto",
            cmap="RdBu_r" if i == 2 else "viridis", vmin=-bound if i == 2 else lower,
            vmax=bound if i == 2 else upper)
        ax.contour(lon, lat, region_masks["boundary"].astype(float), levels=[0.5], colors="black", linewidths=0.7)
        ax.contour(lon, lat, region_masks["southeast"].astype(float), levels=[0.5], colors="white", linewidths=1.0, linestyles="dashed")
        ax.set(xlabel="Longitude (degrees east)", ylabel="Latitude (degrees north)", title=label)
        fig.colorbar(im, ax=ax, shrink=0.86, label=units)
    fig.suptitle(f"{variant}: {variable}, {date_range[0]} to {date_range[1]}\nBlack: boundary/interior; white dashed: geographic southeast")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return bias, valid.sum(axis=0)


def resolve_artifacts(experiment, variant):
    """Honor the manifest's recovered artifacts instead of stale canonical files."""
    experiment = Path(experiment)
    folder = experiment / variant
    manifest_path = experiment / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.is_file() else {}
    entry = manifest.get("variants", {}).get(variant, {})
    paths = []
    for key, fallback in (("predictions", folder / "predictions.npz"),
                          ("configuration", folder / "resolved.yaml")):
        recorded = entry.get(key)
        path = Path(recorded) if recorded else fallback
        if recorded and not path.is_absolute() and not path.is_file():
            path = experiment / path
        if not path.is_file():
            raise FileNotFoundError(f"{variant}: {key} artifact is missing: {path}")
        paths.append(path)
    return tuple(paths)


def generate_report(experiment, variant, output, *, boundary_width=8, coordinates=None):
    experiment, output = Path(experiment), Path(output)
    prediction_path, config_path = resolve_artifacts(experiment, variant)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Report output is nonempty; choose a new directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction, target, mask = archive["pred"], archive["target"], archive["mask"].astype(bool)
        variables = [str(v) for v in archive["output_vars"]]
        dates = archive["dates"]
        if prediction.shape != target.shape or prediction.shape != mask.shape or prediction.ndim != 4:
            raise ValueError("Saved prediction, target and mask tensors must have matching [T,C,H,W] shapes.")
        if variables != list(raw["data"]["output_vars"]):
            raise ValueError("Prediction archive and resolved config disagree on variable order.")
        lat, lon, coordinate_provenance = _coordinates(archive, raw, prediction.shape[-2:], coordinates, output)
    regions, definitions = regions_for_grid(lat, lon, boundary_width)
    date_range = ["-".join(f"{int(v):02d}" for v in row[:3]) for row in (dates[0], dates[-1])]
    threshold = float(raw.get("temporal", {}).get("evaluation", {}).get("wet_threshold", 1.0))
    report = {"schema_version": 1, "variant": variant, "scientific_status": "additional regional diagnostic; no acceptance verdict",
        "case": raw.get("case_name", "SA_downscaling_refinement_T2_ACCESS-CM2_static"),
        "date_range": date_range, "n_dates": int(len(dates)), "array_shape": list(prediction.shape),
        "weighting": "equally weighted valid gridcell-days; no latitude area weighting",
        "quantiles": "pooled valid gridcell-days within each variable/region",
        "input_prediction_path": str(prediction_path.resolve()),
        "input_prediction_sha256": _digest(prediction_path), "resolved_config_sha256": _digest(config_path),
        "coordinate_provenance": coordinate_provenance, "regions": definitions, "variables": {}}
    maps = {"lat": lat, "lon": lon}
    for channel, variable in enumerate(variables):
        is_precip = variable.lower() in {"pr", "ppt", "precip", "precipitation"}
        units = "mm/day" if is_precip else ("K" if raw["data"]["type"] == "cordex" else "physical units")
        values = {"units": units, "units_provenance": "existing case/loader contract; saved NPZ lacks unit attributes",
            "regions": {name: region_metrics(prediction[:, channel], target[:, channel], mask[:, channel], region,
                wet_threshold=threshold if is_precip else None) for name, region in regions.items()}}
        valid = mask[:, channel] & np.isfinite(prediction[:, channel]) & np.isfinite(target[:, channel])
        figure = output / f"{variant}_{variable}_spatial_means_bias.png"
        if valid.any():
            bias, count = _figure(prediction[:, channel], target[:, channel], valid, lat, lon, regions,
                variable=variable, units=units, variant=variant, date_range=date_range, path=figure)
            values["figure"] = figure.name
            maps[f"{variable}_mean_bias"] = bias
            maps[f"{variable}_valid_count"] = count
        report["variables"][variable] = values
    np.savez_compressed(output / "spatial_bias_fields.npz", **maps)
    destination = output / "regional_metrics.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(destination.resolve()), "variant": variant, "n_dates": len(dates)}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--boundary-width", type=int, default=8)
    parser.add_argument("--coordinates", type=Path, help="Optional NPZ with physical lat/lon; extracted coordinates are saved for reuse.")
    args = parser.parse_args()
    generate_report(args.experiment, args.variant, args.out, boundary_width=args.boundary_width, coordinates=args.coordinates)


if __name__ == "__main__":
    main()
