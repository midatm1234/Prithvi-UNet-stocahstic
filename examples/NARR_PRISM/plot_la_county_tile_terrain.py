"""Plot the suggested California 1024x1024 PRISM tile with terrain."""

from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[2]
ELEVATION_PATH = REPO_ROOT / "examples/NARR_PRISM/prism_elevation.nc"
OUTPUT_PATH = REPO_ROOT / "examples/NARR_PRISM/california_1024_tile_terrain.png"
TILE_SIZE = 1024

# PRISM-aligned 1024x1024 bounds covering most of California within the US.
LAT_MIN = 32.55
LAT_MAX = 41.075
LON_MIN = -124.45
LON_MAX = -115.925


def main() -> None:
    with xr.open_dataset(ELEVATION_PATH) as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values
        lat_start = int(np.abs(lat - LAT_MIN).argmin())
        lon_start = int(np.abs(lon - LON_MIN).argmin())
        lat_idx = np.arange(lat_start, lat_start + TILE_SIZE)
        lon_idx = np.arange(lon_start, lon_start + TILE_SIZE)

        elev = ds["elevation"].isel(
            lat=slice(int(lat_idx[0]), int(lat_idx[-1]) + 1),
            lon=slice(int(lon_idx[0]), int(lon_idx[-1]) + 1),
        )
        elev_arr = elev.values.astype(float)
        elev_arr[elev_arr <= -9990] = np.nan
        tile_lat = elev["lat"].values
        tile_lon = elev["lon"].values
        lat_min = float(tile_lat[0])
        lat_max = float(tile_lat[-1])
        lon_min = float(tile_lon[0])
        lon_max = float(tile_lon[-1])

    ls = LightSource(azdeg=315, altdeg=45)
    shaded = ls.shade(
        np.nan_to_num(elev_arr, nan=np.nanmedian(elev_arr)),
        cmap=plt.get_cmap("terrain"),
        vert_exag=0.7,
        blend_mode="overlay",
    )

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(9, 8), dpi=180)
    ax = plt.axes(projection=proj)
    pad = 0.35
    ax.set_extent(
        [lon_min - pad, lon_max + pad, lat_min - pad, lat_max + pad],
        crs=proj,
    )

    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#c8dce8", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#eeeeee", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.8, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4, zorder=3)
    ax.add_feature(cfeature.STATES.with_scale("10m"), linewidth=0.5, zorder=3)

    ax.imshow(
        shaded,
        origin="lower",
        extent=[tile_lon[0], tile_lon[-1], tile_lat[0], tile_lat[-1]],
        transform=proj,
        zorder=2,
    )
    mesh = ax.pcolormesh(
        tile_lon,
        tile_lat,
        elev_arr,
        transform=proj,
        cmap="terrain",
        shading="auto",
        alpha=0.0,
        zorder=2,
    )
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.72, pad=0.03)
    cbar.set_label("Elevation (m)")

    xs = [lon_min, lon_max, lon_max, lon_min, lon_min]
    ys = [lat_min, lat_min, lat_max, lat_max, lat_min]
    ax.plot(xs, ys, color="#d7191c", linewidth=2.0, transform=proj, zorder=5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.35", alpha=0.45)
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(
        "California Candidate PRISM Tile\n"
        f"{TILE_SIZE}x{TILE_SIZE} cells: {lat_min:.3f}..{lat_max:.3f}N, "
        f"{lon_min:.3f}..{lon_max:.3f}E",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
