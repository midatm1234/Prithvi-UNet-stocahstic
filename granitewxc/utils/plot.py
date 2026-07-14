import time
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from granitewxc.utils.eccc_contants import RLAT_GDPS, RLON_GDPS, RLAT_HRDPS, RLON_HRDPS 
from granitewxc.utils.eccc_contants import GRID_NORTH_POLE_LATITUDE, GRID_NORTH_POLE_LONGITUDE


def _get_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as exc:
        raise ImportError(
            "Cartopy is required for map plotting functions in granitewxc.utils.plot. "
            "Install or repair cartopy/pyproj before calling these functions."
        ) from exc
    return ccrs, cfeature


def _get_rapsd():
    try:
        from pysteps.utils.spectral import rapsd
    except ImportError as exc:
        raise ImportError(
            "pysteps is required for RAPSD power spectrum plots in granitewxc.utils.plot."
        ) from exc
    return rapsd


def plot_spatial(sample, ax, title, **kwargs):
    
    vmin=kwargs.get('vmin', np.min(sample))
    vmax=kwargs.get('vmax', np.max(sample))
    
    if kwargs.get('c_lognorm', False):
        vmin = max(vmin, 0)
        e = kwargs.get('e', 1e-44)
        vmin+=e
        norm=colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm=colors.Normalize(vmin=vmin, vmax=vmax)
        
    im = ax.imshow(
        sample,
        cmap=kwargs.get('cmap', 'coolwarm'),
        norm=norm,
        origin='lower',
        animated=kwargs.get('animated', False),
    )
    ax.set_xlabel('Longitudes')
    ax.set_ylabel('Latitudes')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.title.set_text(title)
    
    return im

def plot_model_residual(sample, sample_id, title, **kwargs):

    fig, ax = plt.subplots(figsize=(15, 5))
    plt.suptitle(title)
    
    im = plot_spatial(sample, ax, sample_id,  **kwargs)
    
    label = f"{kwargs.get('var_name_tile', '')} [{kwargs.get('var_unit', '')}]"
    fig.colorbar(im, ax=ax, orientation='vertical', label=label, fraction=0.05, aspect=50)
    
    

def plot_model_results(samples, samples_id, title, **kwargs):
    
    fig, axes = plt.subplots(1, len(samples), figsize=(20, 5))
    plt.suptitle(title)
    
    for i, ax in enumerate(axes):
        if samples_id[i] == 'Residual':
            im_res = plot_spatial(samples[i], ax, samples_id[i],  **kwargs.get('plot_residual_kwargs'))
            label = f"{kwargs.get('var_name_tile', '')} [{kwargs.get('var_unit', '')}]"
            fig.colorbar(im_res, ax=ax, orientation='vertical', label=label, fraction=0.015)
        else:
            im = plot_spatial(samples[i], ax, samples_id[i],  **kwargs)
        
    label = f"{kwargs.get('var_name_tile', '')} [{kwargs.get('var_unit', '')}]"
    fig.colorbar(im, ax=axes, orientation='horizontal', label=label, fraction=0.05, aspect=50)
 
    plt.close()
    
    return fig

def plot_power_spectrum(img, ax, label=None, save_fig=False):
    """
    A power spctrum mesaures the strength of features at different resolutions

    :param img: H x W
    """

    npix = img.shape[-2], img.shape[-1]

    fft_img = np.fft.fftn(img)
    fft_amp = np.abs(fft_img)**2
    fft_amp = fft_amp.flatten()

    kfreq_x = np.fft.fftfreq(npix[1]) * npix[1] # wave vector
    kfreq_y = np.fft.fftfreq(npix[0]) * npix[0] # wave vector
    kfreq2D = np.meshgrid(kfreq_x, kfreq_y)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()

    kbins = np.arange(0.5, min(*npix)//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(
        knrm,
        fft_amp,
        statistic='mean',
        bins=kbins
    )

    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    ax.loglog(kvals, Abins, label=label)
    ax.set_xlabel("Wavelength (km)", fontsize=13)
    ax.set_ylabel("Power (db)", fontsize=13)
    if label:
        ax.legend()
    # plt.tight_layout()

    if save_fig:
        timestr = time.strftime("%Y%m%d-%H%M")
        plt.savefig(f'power_spectrum_{timestr}.png', dpi=300, bbox_inches='tight')

def spatial_rmse(y_hat, y):
    return np.mean((y_hat - y) ** 2) ** 0.5
    
def spatial_bias(y_hat, y):
    return y_hat.mean() - y.mean()


def plot_sample(data):

    x = data['x']
    y = data['y']

    num_rows, num_cols = 2, 2
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(6, 6))
    
    # flatten the axes for easy iteration
    axs = np.ravel(axs)
    
    images = [
        axs[0].imshow(x[0, 1, :, :], cmap='coolwarm'), axs[1].imshow(y[0, 0, :, :], cmap='coolwarm'), 
        axs[2].imshow(x[0, 2, :, :], cmap='coolwarm'), axs[3].imshow(y[0, 1, :, :], cmap='coolwarm')
    ]
    
    titles = ["GDPS - UUVE", "HRDPS - UUVE", "GDPS - VVSN", "HRDPS - VVSN"]
    for ax, title in zip(axs, titles):
        ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=14, pad=10)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])  
    
    # colorbar
    for i in range(num_rows):
        cbar_ax = fig.add_axes([0.92, [0.56, 0.073][i], 0.02, 0.35])
        fig.colorbar(axs[i * num_cols].images[0], cax=cbar_ax, orientation='vertical', label=["UUVE [m/s]", "VVSN [m/s]"][i])

    plt.show()

def plot_loss(train_loss, val_loss):
    
    plt.figure(figsize=(7, 4)) 
    plt.plot(train_loss, label='Training Loss', color='blue', linestyle='-', linewidth=1)
    plt.plot(val_loss, label='Validation Loss', color='orange', linestyle='-', linewidth=1)

    plt.title('Training and Validation Loss', fontsize=13)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.show()


def plot_with_bbox(ax, data, rotated_crs, bounding_boxes, rlat, rlon, cmap='coolwarm', vmin=-10, vmax=10):
    ccrs, cfeature = _get_cartopy()
    mesh = ax.pcolormesh(rlon, rlat, data, cmap=cmap, transform=rotated_crs, vmin=vmin, vmax=vmax)
    
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True)
    
    rotated_crs = ccrs.RotatedPole(
        pole_latitude=GRID_NORTH_POLE_LATITUDE,
        pole_longitude=GRID_NORTH_POLE_LONGITUDE
    )
    
    geographic_crs = ccrs.PlateCarree()

    for name, (lon_min, lon_max, lat_min, lat_max) in bounding_boxes.items():
        lon_points = np.array([lon_min, lon_max, lon_max, lon_min])
        lat_points = np.array([lat_min, lat_min, lat_max, lat_max])

        rotated_points = rotated_crs.transform_points(geographic_crs, lon_points, lat_points)
        rotated_lon = rotated_points[:, 0]
        rotated_lat = rotated_points[:, 1]

        lon_min, lon_max = rotated_lon.min(), rotated_lon.max()
        lat_min, lat_max = rotated_lat.min(), rotated_lat.max()
        
        box_lon = [lon_min, lon_max, lon_max, lon_min, lon_min]
        box_lat = [lat_min, lat_min, lat_max, lat_max, lat_min]
        ax.plot(box_lon, box_lat, linestyle="--", color="black", label=name)

        text_x = (lon_min + lon_max) / 2
        if name == 'Prairies (Alberta)':
            text_y = lat_min - 1.1  # offset below the box
        else:
            text_y = lat_max + 0.2  # offset above the box
        ax.text(
            text_x, text_y, name,
            transform=rotated_crs,
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
        )
    return mesh

def plot_power_spectrum_with_rapsd(img, ax, label=None, save_fig=False, fs=1/2.5):
    rapsd = _get_rapsd()
    spectrum, k = rapsd(img, fft_method = np.fft, return_freq=True, d=1/fs, normalize=False)

    spectrum_db = 10 * np.log10(spectrum + 1e-10)  # add epsilon to avoid log(0)

    ax.plot(k, spectrum_db, label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('linear')
    ax.set_xlabel("Wavelength (pixels)", fontsize=13)
    ax.set_ylabel("Power (dB)", fontsize=13)

    if label:
        ax.legend()

    if save_fig:
        timestr = time.strftime("%Y%m%d-%H%M")
        plt.savefig(f'power_spectrum_rapsd_{timestr}.png', dpi=300, bbox_inches='tight')

def crop_region(data, rlon, rlat, lon_min, lon_max, lat_min, lat_max):
    rlon2d, rlat2d = np.meshgrid(rlon, rlat)

    mask = (
        (rlon2d >= lon_min) & (rlon2d <= lon_max) &
        (rlat2d >= lat_min) & (rlat2d <= lat_max)
    )

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # add 1 to max indices to include the boundary
    return (
        data[row_min:row_max+1, col_min:col_max+1],
        rlon2d[row_min:row_max+1, col_min:col_max+1],
        rlat2d[row_min:row_max+1, col_min:col_max+1]
    )

def convert_bbox_to_rotated(lon_min, lon_max, lat_min, lat_max, rotated_crs):
    ccrs, _ = _get_cartopy()
    geographic_crs = ccrs.PlateCarree()
    lon = np.array([lon_min, lon_max, lon_max, lon_min])
    lat = np.array([lat_min, lat_min, lat_max, lat_max])
    transformed = rotated_crs.transform_points(geographic_crs, lon, lat)
    rlon = transformed[:, 0]
    rlat = transformed[:, 1]
    return rlon.min(), rlon.max(), rlat.min(), rlat.max()


def plot_maps(plot_input, plot_pred, plot_target, regions, rotated_crs):
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), subplot_kw={'projection': rotated_crs})

    im1 = plot_with_bbox(
        axes[0, 0], plot_input, rotated_crs, bounding_boxes=regions,
        rlat=RLAT_GDPS, rlon=RLON_GDPS,
    )
    axes[0, 0].set_title("GDPS", fontsize=18)
    axes[0, 0].axis('off')

    im2 = plot_with_bbox(
        axes[0, 1], plot_pred, rotated_crs, bounding_boxes=regions,
        rlat=RLAT_HRDPS, rlon=RLON_HRDPS,
    )
    axes[0, 1].set_title("GDPS Downscaled", fontsize=18)
    axes[0, 1].axis('off')

    im3 = plot_with_bbox(
        axes[1, 0], plot_target, rotated_crs, bounding_boxes=regions,
        rlat=RLAT_HRDPS, rlon=RLON_HRDPS,
    )
    axes[1, 0].set_title("HRDPS", fontsize=18)
    axes[1, 0].axis('off')
    fig.delaxes(axes[1, 1]) 
    axes[1, 1] = fig.add_subplot(2, 2, 4) 

    plot_power_spectrum_with_rapsd(plot_pred, axes[1, 1], label="GDPS Downscaled")
    plot_power_spectrum_with_rapsd(plot_target, axes[1, 1], label="HRDPS")

    axes[1, 1].set_title("Power Spectrum", fontsize=18)

    cbar = fig.colorbar(
        im1,
        ax=axes.ravel().tolist(),
        orientation='horizontal',
        pad=0.08,         
        aspect=60,        
        shrink=0.9,     
        location='bottom',
        label='[m/s]'
    )
    cbar.ax.tick_params(labelsize=11)
    plt.show()

def plot_by_region(plot_input, plot_pred, plot_target, regions, rotated_crs):
    _, cfeature = _get_cartopy()
    n_regions = len(regions)
    
    fig, axes = plt.subplots(n_regions, n_regions, figsize=(22, 20), subplot_kw={'projection': rotated_crs})

    for col in range(n_regions):
        axes[n_regions-1, col] = fig.add_subplot(n_regions, n_regions, n_regions*(n_regions-1) + col + 1)  # replace with non-map axes

    datasets = [
        (plot_input, RLON_GDPS, RLAT_GDPS, "GDPS"),
        (plot_pred, RLON_HRDPS, RLAT_HRDPS, "GDPS Downscaled"),
        (plot_target, RLON_HRDPS, RLAT_HRDPS, "HRDPS")
    ]

    for col, (region_name, (lon_min, lon_max, lat_min, lat_max)) in enumerate(regions.items()):
        rlon_min, rlon_max, rlat_min, rlat_max = convert_bbox_to_rotated(
            lon_min, lon_max, lat_min, lat_max, rotated_crs
        )

        cropped_maps = []  

        for row, (data, rlon, rlat, title) in enumerate(datasets):
            ax = axes[row, col]

            try:
                cropped_data, cropped_rlon, cropped_rlat = crop_region(
                    data, rlon, rlat, rlon_min, rlon_max, rlat_min, rlat_max
                )
                cropped_maps.append((cropped_data, title))
                pcm = ax.pcolormesh(cropped_rlon, cropped_rlat, cropped_data,
                                    cmap='coolwarm', transform=rotated_crs, vmin=-10, vmax=10)

                ax.coastlines()
                ax.add_feature(cfeature.BORDERS, linestyle=':')

                gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.6, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': 9}
                gl.ylabel_style = {'size': 9}
            except Exception as e:
                print(f"Skipping region '{region_name}' for dataset '{title}': {e}")
                ax.set_visible(False)
                continue

            if row == 0:
                ax.set_title(region_name, fontsize=16, pad=10)
            if col == 0:
                ax.text(-0.2, 0.5, title, va='center', ha='right', rotation='vertical',
                        fontsize=16, transform=ax.transAxes)

        # power spectra (row 3)
        ax_psd = axes[3, col]
        for data_cropped, label in cropped_maps:
            if label != 'GDPS':
                plot_power_spectrum_with_rapsd(data_cropped, ax_psd, label=label)
        ax_psd.set_title(f"Power Spectrum", fontsize=16)

    # cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.55])
    # fig.colorbar(pcm, cax=cbar_ax, label='[m/s]')

    plt.show()

def plot_eccc_results(plot_input, plot_pred, plot_target):
    ccrs, _ = _get_cartopy()
    # define bounding boxes for each region
    regions = {
        "Mountains (BC)": (-116 - 5.88/2,
                            -116 + 5.88/2, 
                            52 - 3.6/2, 
                            52 + 3.6/2),  # Centered at (116ºW, 52ºN)
        "Prairies (Alberta)": (-110 - 5.88/2, 
                                -110 + 5.88/2, 
                                52 - 3.6/2, 
                                52 + 3.6/2),  # Centered at (110ºW, 52ºN)
        "Lakes (Ontario)": (-78 - 5.12/2, 
                            -78 +  5.12/2, 
                            45.5 - 3.6/2, 
                            45.5 + 3.6/2),  # Centered at (78ºW, 45.5ºN)
        "Oceans (Atlantic)": (-60 - 4.82/2, 
                                -60 + 4.82/22, 
                                42 - 3.6/2, 
                                42 + 3.6/2)  # Centered at (60ºW, 42ºN)
    }
    rotated_crs = ccrs.RotatedPole(pole_latitude=GRID_NORTH_POLE_LATITUDE, pole_longitude=GRID_NORTH_POLE_LONGITUDE)

    plot_maps(plot_input, plot_pred, plot_target, regions, rotated_crs)
    plot_by_region(plot_input, plot_pred, plot_target, regions, rotated_crs)
    
