from cartopy.util import add_cyclic_point
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import warnings
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore")
import matplotlib as mpl
mpl.use("Agg")
from scipy.linalg import norm
from scipy import stats
from scipy.stats.mstats import ttest_ind
from scipy.signal import detrend
import cartopy
import os
from numpy import nan as NaN
import glob
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
#-*- coding:utf-8 –*-

from multiprocessing import Pool
import math
import time

 
def trend_result(data_3D,nt,nlat,nlon):
    ##--##--## calculate trend --##--##--##
    ngrd=nlat*nlon
    ano_grd = data_3D.values.reshape((nt, ngrd), order='F')
    x = np.linspace(1, nt, nt)

    # construct slope and significant matrix
    ano_rate = np.empty((ngrd,))
    p_values = np.empty((ngrd,))

    ano_rate[:] = np.nan
    p_values[:] = np.nan


    # for each grid point  
    for i in range(ngrd):
        slope, _, _, p_value, _ = stats.linregress(x, ano_grd[:, i])
        ano_rate[i] = slope
        p_values[i] = p_value


    # return back to the (nlat, nlon)
    day_shape = (nlat, nlon)
    anom_grd = ano_rate.reshape(day_shape, order='F') * 10
    p_values_grd = p_values.reshape(day_shape, order='F')
    return anom_grd,p_values_grd

def trend_1D_result(data_1D,nt):
    slope, _, _, _, _ = stats.linregress(np.linspace(1, nt, nt), data_1D)
    slope = slope*10
    return slope 
def trend_1D_result(data_1D,nt):
    slope, _, _, _, _ = stats.linregress(np.linspace(1, nt, nt), data_1D)
    slope = slope*10
    return slope 


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def find_max_patch(data,lon1, lat_vals, lat_win=6, lon_win=20):
    max_val = -np.inf
    best_patch = None

    
    for lat_start in range(min(lat_vals), max(lat_vals) - lat_win + 2, 2):
        for lon_start in range((lon1-40), (lon1+40) - lon_win + 2, 2):

            lat_slice = np.arange(lat_start, lat_start + lat_win+2,2)
            lon_slice = np.arange(lon_start % 360, (lon_start + lon_win+2) % 360,2)

            # print(lat_slice);print(lon_slice)
            patch = data.sel(lat2=lat_slice, lon2=lon_slice)
            patch_mean = patch.mean().item()
            
            if patch_mean > max_val:
                max_val = patch_mean
                best_patch_lat_slice=lat_slice
                best_patch_lon_slice=lon_slice
    # print(best_patch_lat_slice)
    # print(best_patch_lon_slice)
    # print(max_val)
                
    return best_patch_lat_slice,best_patch_lon_slice , max_val


## -------------1. trends in index

#---------------------------------------------

Fut1 = xr.open_dataset("/public/home/fcai/abc/2Paper_blocking/CMIP6_data/streamfunction/His_Dynamical/4MODEL_T_dyn_anom_20152099_JJA_NH.nc")
fut_th75_1 = Fut1.T_dyn_anom
Fut2 = xr.open_dataset("/public/home/fcai/abc/2Paper_blocking/CMIP6_data/streamfunction/His_Dynamical/16MODEL_T_dyn_anom_20152099_JJA_NH.nc")
fut_th75_2 = Fut2.T_dyn_anom

t1 = fut_th75_1.sel(lat=slice(-30,90), lon=slice(-179,180))
t2 = fut_th75_2.sel(lat=slice(-30,90), lon=slice(-179,180))
ta1 = t1.sel(time=Fut1['time'].dt.year.isin(range(2015,2099+1))&Fut1['time'].dt.month.isin(range(7,8+1)));print(np.shape(ta1))
ta1_75 = ta1.groupby("time.year").map(lambda x: x.quantile(0.75, dim="time", skipna=True))
 

ta2 = t2.sel(time=Fut2['time'].dt.year.isin(range(2015,2099+1))&Fut2['time'].dt.month.isin(range(7,8+1)));print(np.shape(ta2))
ta2_75 = ta2.groupby("time.year").map(lambda x: x.quantile(0.75, dim="time", skipna=True))
print(np.shape(ta2_75))
print(np.shape(ta1_75))


lat = fut_th75_1.lat.sel(lat=slice(-30,90))
lon = fut_th75_1.lon.sel(lon=slice(-179,180))

T =np.zeros((20,85,71,360))
T[0:16,:,:,:] = ta2_75
T[16:,:,:,:]  = ta1_75

# DataArray
t_day= xr.DataArray(T,
    dims=["model","year", "lat", "lon"],
    coords={
        "model":["ACCESS-CM2","CanESM5","EC-Earth3","EC-Earth3-Veg","FGOALS-g3","GFDL-CM4","INM-CM4-8","INM-CM5-0","IPSL-CM6A-LR",\
                "MIROC6","MPI-ESM1-2-HR","MPI-ESM1-2-LR","MRI-ESM2-0","NorESM2-LM",\
                "NorESM2-MM","TaiESM1","HadGEM3-GC31-LL", "HadGEM3-GC31-MM","KACE-1-0-G","UKESM1-0-LL"],
        "year": np.linspace(2015,2099,85),
        "lat": lat.data,
        "lon": lon.data,
    })


##--##--##-- save --##--##--##
air_th75 = xr.DataArray(data= t_day, dims=['model','year','lat','lon'],
              coords= {'model':["ACCESS-CM2","CanESM5","EC-Earth3","EC-Earth3-Veg","FGOALS-g3","GFDL-CM4","INM-CM4-8","INM-CM5-0","IPSL-CM6A-LR",\
                "MIROC6","MPI-ESM1-2-HR","MPI-ESM1-2-LR","MRI-ESM2-0","NorESM2-LM",\
                "NorESM2-MM","TaiESM1","HadGEM3-GC31-LL", "HadGEM3-GC31-MM","KACE-1-0-G","UKESM1-0-LL"],'year':np.linspace(2015,2099,85), 'lat':lat.data , 'lon':lon.data})

ds3 = xr.Dataset(data_vars= dict(air_th75 = air_th75))
ds3.to_netcdf("/public/home/fcai/abc/2Paper_blocking/CMIP6_data/complex_network/6_pattern/data/Dynamical_T75_model_day_JA_20152099.nc")
exit()



# After calculating the Tdyn, pick the 75th percentile