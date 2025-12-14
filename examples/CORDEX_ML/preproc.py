import fstd2nc
import numpy as np
import xesmf as xe
import argparse

import os, os.path
import downgan.config.hyperparameters as hp
import warnings
import json


# Load config
parser = argparse.ArgumentParser(description="Just an example",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
parser.add_argument('--config', dest='config', required=True, type=str, help='Path to the config file')
args = vars(parser.parse_args())
hp.load_config(args['config'])


# Path of hrdps and gdps data
with open(hp.TRAIN_INDEX_PATH, 'r') as fp:
    dataIndexTrain = json.load(fp)

with open(hp.VAL_INDEX_PATH, 'r') as fp:
    dataIndexVal = json.load(fp)

with open(hp.TEST_INDEX_PATH, 'r') as fp:
    dataIndexTest = json.load(fp)

#Path of netcdf data
gdps_netcdf_path = "/dccstor/wfm/shared/datasets/eccc/netcdf/gdps"
hrdps_netcdf_path = "/dccstor/wfm/shared/datasets/eccc/netcdf/hrdps"

# Regrid function
def regrid(ds_gdps, ds_hrdps):
    # returns a regrided gdps as a function of hdrps coordinates
    if os.path.isfile('gdps_hrdps.nc'):
        #regridder = xesmf.Regridder(gdps_ds, hrdps_ds, method="conservative", weights='gdps_hrdps.nc')
        try:
            regridder = xe.Regridder(ds_gdps, ds_hrdps, method="nearest_s2d",weights='gdps_hrdps.nc')
        except Exception as error:
            print("An exception occurred:", type(error).__name__, "–", error)
    else:
        try:
            regridder = xe.Regridder(ds_gdps, ds_hrdps, method="nearest_s2d").to_netcdf('gdps_hrdps.nc')
        except Exception as error:
            print("An exception occurred:", type(error).__name__, "–", error)

    return  regridder(ds_gdps)

# Load funtion
def dataload(paths):
    gdps_path, hrdps_path = paths

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds_hrdps = fstd2nc.Buffer(hrdps_path,forecast_axis=True).to_xarray()
    ds_hrdps = ds_hrdps.isel(height1=0,height2=0,pres=0, forecast=0,time=0)
    ds_hrdps = ds_hrdps.rename({'rlon': 'lon_b','rlat': 'lat_b'})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds_gdps = fstd2nc.Buffer(gdps_path,forecast_axis=True).to_xarray()
    ds_gdps = ds_gdps.isel(height1=0,height2=0,pres=0, forecast=0,time=0)
    ds_gdps = ds_gdps.rename({'rlon': 'lon_b','rlat': 'lat_b'})

    return ds_hrdps, ds_gdps

#Save netcdf file
def datasave(paths):
    gdps_path, hrdps_path = paths
    file_gdps = os.path.split(gdps_path)[1]
    folder1_gdps = os.path.split(os.path.split(gdps_path)[0])[1]
    folder2_gdps = os.path.split(os.path.split(os.path.split(gdps_path)[0])[0])[1]
    gdps_netcdf_folder = os.path.join(gdps_netcdf_path,folder2_gdps,folder1_gdps)

    file_hrdps = os.path.split(hrdps_path)[1]
    folder1_hrdps = os.path.split(os.path.split(hrdps_path)[0])[1]
    folder2_hrdps = os.path.split(os.path.split(os.path.split(hrdps_path)[0])[0])[1]
    hrdps_netcdf_folder = os.path.join(hrdps_netcdf_path,folder2_hrdps,folder1_hrdps)

    try:
        os.makedirs(gdps_netcdf_folder)
    except Exception as error:
        print("An exception occurred:", type(error).__name__, "–", error)

    try:
        os.makedirs(hrdps_netcdf_folder)
    except Exception as error:
        print("An exception occurred:", type(error).__name__, "–", error)

    gdps_netcdf_file = os.path.join(gdps_netcdf_folder,file_gdps)
    ds_gdps.to_netcdf(gdps_netcdf_file)

    hrdps_netcdf_file = os.path.join(hrdps_netcdf_folder,file_hrdps)
    ds_hrdps.to_netcdf(hrdps_netcdf_file)

# Training data
print("Processing training data")
for idx_train in range(0,100):
    # Load data
    ds_hrdps, ds_gdps = dataload(dataIndexTrain[str(idx_train)])

    # Perform regridding
    ds_gdps = regrid(ds_gdps, ds_hrdps)

    #Save data
    datasave(dataIndexTrain[str(idx_train)])

# Validation data
print("Processing validation data")
for idx_val in range(0,25):
    # Load data
    ds_hrdps, ds_gdps = dataload(dataIndexVal[str(idx_val)])

    # Perform regridding
    ds_gdps = regrid(ds_gdps, ds_hrdps)

    #Save data
    datasave(dataIndexVal[str(idx_val)])

# Test data
print("Processing test data")
for idx_test in range(0,25):
    # Load data
    ds_hrdps, ds_gdps = dataload(dataIndexTest[str(idx_test)])

    # Perform regridding
    ds_gdps = regrid(ds_gdps, ds_hrdps)

    #Save data
    datasave(dataIndexTest[str(idx_test)])

# HRDPS covariates
HRDPS_COVARIATES_PATH = hp.HRDPS_COVARIATES_PATH
HRDPS_COVARIATES = fstd2nc.Buffer(HRDPS_COVARIATES_PATH, forecast_axis=True).to_xarray()
HRDPS_COVARIATES = HRDPS_COVARIATES.rename({'rlon': 'lon_b', 'rlat': 'lat_b'})

# standardization
for covariate in ['ME', 'MG', 'Z0']:
    min_val = np.min(HRDPS_COVARIATES[covariate].values)
    max_val = np.max(HRDPS_COVARIATES[covariate].values)
    HRDPS_COVARIATES[covariate] = (HRDPS_COVARIATES[covariate] - min_val) / (max_val - min_val)

hrdps_covariates_path = '/dccstor/wfm/shared/datasets/eccc/netcdf/covariates/hrdps_covariates'
HRDPS_COVARIATES.to_netcdf(hrdps_covariates_path)