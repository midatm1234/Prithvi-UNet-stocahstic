"""Compare saved physical validation ensembles without changing any prediction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from granitewxc.refinement.boundary import boundary_regions, physical_metrics


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    runs = {}
    for specification in args.run:
        label, directory = specification.split("=", 1)
        folder = Path(directory)
        arrays = dict(np.load(folder / "validation_members.npz", allow_pickle=True))
        manifest = json.loads((folder / "manifest.json").read_text())
        if runs:
            reference = next(iter(runs.values()))[0]
            for key in ("dates", "truth", "baseline", "valid"):
                if not np.array_equal(reference[key], arrays[key], equal_nan=key != "dates"):
                    raise ValueError(f"Cannot pair {label}: differing {key}")
        runs[label] = arrays, manifest
    arrays, manifest = next(iter(runs.values()))
    with h5py.File(manifest["args"]["cache"], "r") as cache:
        lat, lon = cache["latitude"][:], cache["longitude"][:]
    _, regions = boundary_regions(arrays["valid"].any((0, 1)), lat, lon)
    distance_names = ["distance_0", "distance_1", "distance_2", "distance_3", "distance_4_7", "distance_8_15", "interior"]
    report = {"dates": arrays["dates"].tolist(), "days": len(arrays["dates"]),
              "scope": "validation screening; not final historical climatology",
              "runs": {}}
    normalization_figure, normalization_axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for label, (_, run_manifest) in runs.items():
        normalizer = run_manifest["normalizer"]
        for channel, variable in enumerate(("pr", "tasmax")):
            for column, statistic in enumerate(("mean", "scale", "count")):
                value = float(normalizer[statistic][channel])
                ax = normalization_axes[channel, column]
                ax.plot(range(len(distance_names)), [value] * len(distance_names), label=label)
                ax.set_title(f"{variable}: training residual {statistic}")
                ax.set_xticks(range(len(distance_names)), ["0", "1", "2", "3", "4–7", "8–15", "≥16"])
                ax.set_xlabel("Distance from valid boundary (cells)")
                ax.legend(fontsize=7)
    normalization_figure.suptitle("Global per-channel statistics; mean/scale use each run's transformed residual units")
    normalization_figure.savefig(output / "normalization_distance.png", dpi=160)
    plt.close(normalization_figure)
    for channel, variable in enumerate(("pr", "tasmax")):
        offset = 273.15 if variable == "tasmax" else 0
        truth = arrays["truth"][:, channel]
        baseline = arrays["baseline"][:, channel]
        valid = arrays["valid"][:, channel].astype(bool)
        truth_mean = np.nanmean(np.where(valid, truth, np.nan), axis=0)
        base_mean = np.nanmean(np.where(valid, baseline, np.nan), axis=0)
        climates = [truth_mean, base_mean] + [np.nanmean(np.where(valid, a["mean"][:, channel], np.nan), axis=0) for a, _ in runs.values()]
        low = min(np.nanmin(x) for x in climates) - offset
        high = max(np.nanmax(x) for x in climates) - offset
        biases = [base_mean - truth_mean]
        for a, _ in runs.values():
            refined = np.nanmean(np.where(valid, a["mean"][:, channel], np.nan), axis=0)
            biases.extend((refined - truth_mean, refined - base_mean))
        bound = max(float(np.nanmax(np.abs(x))) for x in biases)
        requested = args.pr_bias_limit if channel == 0 else args.tasmax_bias_limit
        if requested is not None: bound = requested
        if channel == 0 and args.pr_climatology_range is not None: low,high = args.pr_climatology_range
        report.setdefault("plot_limits",{})[variable] = {"climatology":[float(low),float(high)],"bias":[-bound,bound],"unclipped_bias_extrema":[float(min(np.nanmin(x) for x in biases)),float(max(np.nanmax(x) for x in biases))]}
        curve, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        p95_products = {}
        if channel == 0 and len(truth) >= 200:
            observed_p95 = np.nanpercentile(np.where(valid, truth, np.nan), 95, axis=0)
            for run_label, (values, _) in runs.items():
                mean_p95 = np.nanpercentile(np.where(valid, values["mean"][:, channel], np.nan), 95, axis=0)
                member_p95 = np.nanpercentile(np.where(valid[:, None], values["members"][:, :, channel], np.nan), 95, axis=(0, 1))
                p95_products[run_label] = (mean_p95-observed_p95, member_p95, member_p95-observed_p95)
            mean_p95_limit = max(1e-6, max(float(np.nanmax(np.abs(item[0]))) for item in p95_products.values()))
            member_p95_limit = max(1e-6, max(float(np.nanmax(np.abs(item[2]))) for item in p95_products.values()))
            report["plot_limits"][variable].update(ensemble_mean_p95_bias=[-mean_p95_limit,mean_p95_limit],
                                                   member_p95_bias=[-member_p95_limit,member_p95_limit])
        for label, (a, run_manifest) in runs.items():
            prediction = a["mean"][:, channel]
            mean = np.nanmean(np.where(valid, prediction, np.nan), axis=0)
            metrics = {region: physical_metrics(prediction, truth, valid, region=mask, precipitation=channel == 0)
                       for region, mask in regions.items()}
            report["runs"].setdefault(label, {})[variable] = metrics
            members = a["members"][:, :, channel]
            member_variance = members.var(1,ddof=1) if members.shape[1]>1 else None
            sample_spread = np.sqrt(member_variance) if member_variance is not None else np.full(prediction.shape,np.nan)
            member_metrics = {}
            for name, region in regions.items():
                selected = valid & region[None]
                member_mask = np.broadcast_to(selected[:,None],members.shape)
                member_values,truth_values = members[member_mask],truth[selected]
                if not len(truth_values): continue
                values = {"mean_bias":float(member_values.mean()-truth_values.mean()),
                          "pooled_p95_error":float(np.percentile(member_values,95)-np.percentile(truth_values,95)),
                          "pooled_p99_error":float(np.percentile(member_values,99)-np.percentile(truth_values,99)),
                          "mean_ensemble_variance":float(member_variance[selected].mean()) if member_variance is not None else None}
                if channel == 0:
                    wet,truth_wet=member_values>=1.,truth_values>=1.
                    values["wet_frequency_bias"]=float(wet.mean()-truth_wet.mean())
                    values["wet_intensity_bias"]=float(member_values[wet].mean()-truth_values[truth_wet].mean()) if wet.any() and truth_wet.any() else None
                member_metrics[name]=values
            report["runs"][label].setdefault("member_distribution",{})[variable]=member_metrics
            report["runs"][label].setdefault("ensemble_spread", {})[variable] = {
                name: float(sample_spread[valid & mask[None]].mean()) if member_variance is not None and (valid & mask[None]).any() else None
                for name, mask in regions.items() if mask.any()}
            figure, panels = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
            fields = [truth_mean-offset, base_mean-offset, mean-offset, base_mean-truth_mean, mean-truth_mean, mean-base_mean]
            titles = ["Ground truth", "Phase-1 U-Net", label, "Phase 1 − truth", "Refinement − truth", "Refinement − Phase 1"]
            for i, (ax, field, title) in enumerate(zip(panels.flat, fields, titles)):
                cmap = ("YlGnBu" if channel == 0 else "magma_r") if i < 3 else ("BrBG" if channel == 0 else "RdBu_r")
                mesh = ax.pcolormesh(lon, lat, field, shading="nearest", cmap=cmap,
                                     vmin=low if i < 3 else -bound, vmax=high if i < 3 else bound)
                ax.set_title(title); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
                figure.colorbar(mesh, ax=ax, extend="both", label="mm/day" if channel == 0 else "°C")
            figure.suptitle(f"{variable}: {len(truth)} validation dates (screening)")
            figure.savefig(output / f"{label}_{variable}_climatology.png", dpi=160)
            plt.close(figure)
            x = np.arange(len(distance_names))
            for ax, metric in zip(axes.flat[:2], ("bias", "rmse")):
                ax.plot(x, [metrics[name][metric] for name in distance_names], marker="o", label=label)
                ax.set_ylabel(metric)
            correction = np.where(valid, prediction - baseline, np.nan)
            axes.flat[2].plot(x, [float(np.nanmean(correction[:, regions[name]])) for name in distance_names], marker="o", label=label)
            axes.flat[2].set_ylabel("Mean physical correction")
            spread = np.where(valid, sample_spread, np.nan)
            axes.flat[3].plot(x, [float(np.nanmean(spread[:, regions[name]])) for name in distance_names], marker="o", label=label)
            axes.flat[3].set_ylabel("Sample ensemble standard deviation")
            if channel == 0:
                wet_bias = np.nanmean(np.where(valid, prediction >= 1, np.nan), axis=0) - np.nanmean(np.where(valid, truth >= 1, np.nan), axis=0)
                fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
                image = ax.pcolormesh(lon, lat, wet_bias, shading="nearest", cmap="BrBG", vmin=-1, vmax=1)
                fig.colorbar(image, ax=ax, label="Ensemble-mean wet-day frequency bias (threshold 1 mm/day)")
                ax.set_title(f"{label}: validation screening")
                fig.savefig(output / f"{label}_wet_frequency.png", dpi=150); plt.close(fig)
                member_values = a["members"][:, :, channel]
                observed_frequency = np.nanmean(np.where(valid, truth >= 1., np.nan), axis=0)
                member_frequency = np.nanmean(np.where(valid[:, None], member_values >= 1., np.nan), axis=(0, 1))
                member_wet_bias = member_frequency - observed_frequency
                fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
                image = ax.pcolormesh(lon, lat, member_wet_bias, shading="nearest", cmap="BrBG", vmin=-1, vmax=1)
                fig.colorbar(image, ax=ax, label="Member wet-day frequency bias (threshold 1 mm/day)")
                ax.set_title(f"{label}: {len(truth)} dates, {member_values.shape[1]} members")
                ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
                fig.savefig(output / f"{label}_member_wet_frequency.png", dpi=150); plt.close(fig)
                distribution_maps = {"observed_wet_frequency": observed_frequency,
                    "member_wet_frequency": member_frequency, "member_wet_frequency_bias": member_wet_bias,
                    "ensemble_mean_wet_frequency_bias": wet_bias}
                report["runs"][label]["distribution_map_scope"] = (
                    "Member wet-day/P95 maps pool physical members and dates; ensemble-mean maps are separate. "
                    "Ensemble averaging generally changes wet frequency and suppresses daily extremes.")
                # Temporal extremes need many independent dates. Do not label a
                # sparse screening quantile as a climatological extreme metric.
                if len(truth) >= 200:
                    p95_bias, member_p95, member_p95_bias = p95_products[label]
                    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
                    limit = mean_p95_limit
                    mesh = ax.pcolormesh(lon, lat, p95_bias, shading="nearest", cmap="BrBG", vmin=-limit, vmax=limit)
                    fig.colorbar(mesh, ax=ax, label="Temporal P95 bias (mm/day; ensemble mean)")
                    ax.set_title(f"{label}: {len(truth)} validation dates")
                    fig.savefig(output / f"{label}_p95_bias.png", dpi=150)
                    plt.close(fig)
                    distribution_maps.update(observed_p95=observed_p95, member_p95=member_p95,
                                             member_p95_bias=member_p95_bias, ensemble_mean_p95_bias=p95_bias)
                    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
                    member_limit = member_p95_limit
                    mesh = ax.pcolormesh(lon, lat, member_p95_bias, shading="nearest", cmap="BrBG", vmin=-member_limit, vmax=member_limit)
                    fig.colorbar(mesh, ax=ax, label="Member marginal P95 bias (mm/day)")
                    ax.set_title(f"{label}: {len(truth)} dates, {member_values.shape[1]} members")
                    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
                    fig.savefig(output / f"{label}_member_p95_bias.png", dpi=150); plt.close(fig)
                else:
                    report["runs"][label]["temporal_p95_map"] = "Not estimated: fewer than 200 validation dates"
                np.savez_compressed(output / f"{label}_precipitation_distribution_maps.npz", **distribution_maps, latitude=lat, longitude=lon)

        baseline_metrics = {name: physical_metrics(baseline, truth, valid, region=regions[name], precipitation=channel == 0)
                            for name in distance_names}
        for ax, metric in zip(axes.flat[:2], ("bias", "rmse")):
            ax.plot(np.arange(len(distance_names)), [baseline_metrics[name][metric] for name in distance_names], "k--", label="Phase 1")
        for ax in axes.flat:
            ax.set_xticks(np.arange(len(distance_names)), ["0", "1", "2", "3", "4–7", "8–15", "≥16"])
            ax.set_xlabel("Distance from valid boundary (cells)"); ax.legend(); ax.grid(alpha=.25)
        curve.savefig(output / f"{variable}_distance.png", dpi=160); plt.close(curve)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="label=training_output_directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pr-bias-limit", type=float)
    parser.add_argument("--tasmax-bias-limit", type=float)
    parser.add_argument("--pr-climatology-range", type=float, nargs=2)
    main(parser.parse_args())
