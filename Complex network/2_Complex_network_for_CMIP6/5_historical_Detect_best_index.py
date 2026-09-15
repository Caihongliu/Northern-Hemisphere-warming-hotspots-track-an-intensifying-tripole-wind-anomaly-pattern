from cartopy.util import add_cyclic_point
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from copy import copy
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import shapely.geometry as sgeom
import matplotlib.colorbar as colorbar
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from matplotlib.patches import Rectangle
import warnings
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore")
import matplotlib as mpl
mpl.use("Agg")
from cartopy.util import add_cyclic_point
from scipy.linalg import norm
from scipy import stats
from scipy.stats.mstats import ttest_ind
from scipy.signal import detrend
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
from cartopy.util import add_cyclic_point
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import warnings
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore")
import matplotlib as mpl
mpl.use("Agg")
from cartopy.util import add_cyclic_point
from scipy.linalg import norm
from scipy import stats
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
from scipy import signal
import os
import glob
#-*- coding:utf-8 –*-

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


# model list
model_name=("ACCESS-CM2","CanESM5","EC-Earth3","EC-Earth3-Veg",
            "FGOALS-g3","GFDL-CM4","HadGEM3-GC31-LL",\
            "HadGEM3-GC31-MM","INM-CM4-8","INM-CM5-0","IPSL-CM6A-LR",\
            "KACE-1-0-G","MIROC6","MPI-ESM1-2-HR","MPI-ESM1-2-LR",\
            "MRI-ESM2-0","NorESM2-LM","NorESM2-MM","TaiESM1","UKESM1-0-LL") #24s
 

input_base_dir1="/public/home/fcai/abc/2Paper_blocking/CMIP6_data/2x2/U300_daily_19792014/"
output_base_dir="/public/home/fcai/abc/2Paper_blocking/CMIP6_data/complex_network/1_threshold/"

Model_triple_index = np.zeros((len(model_name),36,28+1,180))
 
 

# loop model name
imodel=0
for model in model_name:

    print(imodel)
    
    pattern1=os.path.join(input_base_dir1,f"2x2_300_JA-ua_{model}_historical_1950-2014_NH.nc"); print(pattern1)
    file1= glob.glob(pattern1)
    ds1 = xr.open_dataset(file1[0])
    u = ds1.ua.sel(latitude=slice(-40,90), longitude=slice(0,360))[:,0,:,:] ;print(np.shape(u))

    ua = u.sel(time=ds1['time'].dt.year.isin(range(1979,2014+1))) ;print(ua)
    lat = ds1.latitude.sel(latitude=slice(-40,90))
    lon = ds1.longitude

    ua_grouped = ua.groupby("time.year")
    n_years = 36; n_days = int(np.shape(ua)[0]/n_years); print(n_days)
    n_lat, n_lon = len(ua.latitude), len(ua.longitude)

    data = np.zeros((n_years, n_days, n_lat, n_lon))
    for i , (year, group) in enumerate(ua_grouped):
        data[i] = group.values

    ## ------------- 1.  U250 anomaly and std
    ua_year_run_90 = np.mean(data[:,:,:,:],1)
    u_JJA_clm = np.mean(ua_year_run_90[2:32,:,:],0)
    tile_u_JJA_clm =np.tile(u_JJA_clm ,(36,1,1)) 
    u_ano = np.subtract(ua_year_run_90,tile_u_JJA_clm) 

    u_ano = xr.DataArray(data= u_ano, dims=['time','lat','lon' ],
                coords= {'time':range(36),'lat':lat.data ,'lon':lon.data })
    print(np.shape(u_ano))


    ## ------------- b. complex network
    ds0=xr.open_dataset(f"/public/home/fcai/abc/2Paper_blocking/CMIP6_data/complex_network/2_U300positive_network/significant/{model}_U300Positive_significant_99.nc")
    
    print(ds0)
  
    for lon_name in ['lon1', 'lon2']:
        ds0['longitude_adjusted'] = xr.where(ds0[lon_name] < 0, ds0[lon_name] + 360, ds0[lon_name])
        ds0 = ds0.swap_dims({lon_name: 'longitude_adjusted'}).sel(
            **{'longitude_adjusted': sorted(ds0.longitude_adjusted)}
        ).drop(lon_name)
        ds0 = ds0.rename({'longitude_adjusted': lon_name})

    lat=ds0.lat1; lon=ds0.lon1
    network=ds0.significant_99
    print(network)
    

    ds1=xr.open_dataset(f"/public/home/fcai/abc/2Paper_blocking/CMIP6_data/complex_network/3_U300negative_network/significant/{model}_U300Negative_significant_99.nc")
    
  
    for lon_name in ['lon1', 'lon2']:
        ds1['longitude_adjusted'] = xr.where(ds1[lon_name] < 0, ds1[lon_name] + 360, ds1[lon_name])
        ds1 = ds1.swap_dims({lon_name: 'longitude_adjusted'}).sel(
            **{'longitude_adjusted': sorted(ds1.longitude_adjusted)}
        ).drop(lon_name)
        ds1 = ds1.rename({'longitude_adjusted': lon_name})

    network_easterly=ds1.significant_99
 

    # ------------- c. construct TWAI


    triple_index = np.zeros((36,28+1,np.shape(u)[2]))

    for ilat in range(12,68+2,2):   #69
        print(ilat)
        
        for ilon in range(0,358+2,2):    # 359
            
            lat_range = np.arange(ilat-2,ilat+18+2)    
            lat_range_val = [(x // 2) * 2 + 1 for x in lat_range]
            lat2_range_val = list(set(lat_range_val)) 

            lon_range = np.arange(ilon-40,ilon+40+2) %360  
            lon_range_val = [(x // 2) * 2 + 1 for x in lon_range]
            lon2_range_val = list(set(lon_range_val)) 
                                                                

            # 1. Northern westerly  
            patch_north = network.sel(lat1=ilat, lon1=ilon, lat2=lat2_range_val, lon2 = lon2_range_val)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
            north_best_lat, north_best_lon , north_max_val = find_max_patch(patch_north,lon1=ilon,lat_vals=lat2_range_val)
        

            # Easterly region from ilat-15 to ilat+5
            lat_range_east = np.arange(ilat - 16, ilat + 4+2)  # -14   6
            lat_range_east_val = [(x // 2) * 2 + 1 for x in lat_range_east]
            lat2_range_east_val = list(set(lat_range_east_val)) 

            # 2. central easterly
            patch_center = network_easterly.sel(lat1=ilat, lon1=ilon, lat2=lat2_range_east_val, lon2 = lon2_range_val)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
            center_best_lat, center_best_lon, center_max_val = find_max_patch(patch_center, lon1=ilon,lat_vals=lat2_range_east_val)


            # Southern westerly region form ilat-30 to ilat-15
            lat_range_center = np.arange(ilat - 32, ilat - 12+2) #-28-8
            lat_range_center_val = [(x // 2) * 2 + 1 for x in lat_range_center]
            lat2_range_center_val = list(set(lat_range_center_val)) 

            # 3. Southern westerly
            patch_south = network.sel(lat1=ilat, lon1=ilon, lat2=lat2_range_center_val, lon2 = lon2_range_val)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
            south_best_lat, south_best_lon, south_max_val = find_max_patch(patch_south, lon1=ilon,lat_vals=lat2_range_center_val)

    
            triple_index[:,(ilat-11)//2, (ilon-1)//2] =np.mean(u_ano.sel(lat=north_best_lat,lon=north_best_lon),axis=(1,2))\
                        + np.mean(u_ano.sel(lat=south_best_lat,lon=south_best_lon),axis=(1,2))\
                        - 2* np.mean(u_ano.sel(lat=center_best_lat,lon=center_best_lon),axis=(1,2))
        
#
    triple_index_scale = standardization(triple_index)

    Model_triple_index[imodel,:,:,:] = triple_index_scale
    imodel +=1


##--##--##-- save --##--##--##
tripole = xr.DataArray(data= Model_triple_index, dims=['Model','year','lat','lon'],
            coords= {'Model':range(20),'year':range(36) , 'lat':range(12,68+2,2),'lon':range(0,358+2,2)})

ds3 = xr.Dataset(data_vars= dict(tripole = tripole))
ds3.to_netcdf("/public/home/fcai/abc/2Paper_blocking/CMIP6_data/complex_network/6_pattern/standize_historical_Model_tripole_wind_index_based_on_automatic_finding.nc")
ds3.close()
exit()



