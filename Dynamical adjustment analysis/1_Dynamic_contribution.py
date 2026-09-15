# coding by Caihong
# Nov 13 2024
# refer to https://medium.com/@creatrohit9/lasso-ridge-elastic-net-regression-a-complete-understanding-2021-b335d9e8ca3; https://glmnet-python.readthedocs.io/en/latest/glmnet_vignette.html#Usage
# Ridge regression 

# ===========================================
# Here we only show how to calculate dynamical adjustment analysis over -159 ~ -119
# The rest is the same
# ===========================================


from multiprocessing import Pool
import math
import time
 
    

import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs


import sys
module_path = '/public/home/fcai/abc/2Paper_blocking/historical_Dynamic/BASE_Heat_code'
sys.path.append(module_path)
from dyn_decomp import preprocess_data,decompose_one_gridpoint
import matplotlib.pyplot as plt

sys.path.append('/public/home/fcai/abc/1Paper_2023/data0/')
from mymodule import change_lon



#----------------------分割线（至此，地图地图加载完毕）------------------------

ds0 = xr.open_dataset("/public/home/fcai/abc/2Paper_blocking/historical_Dynamic/STREAM500_era5_Global_1degr_19400101_20240229.nc")
ds0 = change_lon(ds0, 'longitude')
lat = ds0.latitude[::-1]
lon = ds0.longitude

stream=ds0.stream[:,::-1,:];print(stream) 

stream_JJA = stream.sel(time= (stream.time.dt.year>= 1950)&(stream.time.dt.month.isin([6,7,8])))
stream_JJA= stream_JJA.sel(time=stream_JJA.time.dt.year<= 2023)

print(stream_JJA)

#---------------------- load ERA5 Tmax (1979-2023(45 y), JA) ------------------------

ds1 = xr.open_dataset("/public/home/fcai/abc/2Paper_blocking/historical_Dynamic/T2M_era5_Global_1degr_19400101_20240229.nc")
ds1 = change_lon(ds1, 'longitude')
t2m = ds1.t2m[:,::-1,:];print(t2m)

t2m_JJA = t2m.sel(time= (t2m.time.dt.year>= 1950)&(t2m.time.dt.month.isin([6,7,8])))
t2m_JJA= t2m_JJA.sel(time=t2m_JJA.time.dt.year<= 2023)


print(t2m_JJA)

#----------------------------------------------

ds = xr.open_dataset("/public/data2/model/fcai/data0/topography/topo_adapt_era5_air2m.nc")
ds = change_lon(ds, 'lon')
topo = ds.topo
topo_NH = topo.loc[0:90,:]
print(topo_NH.values)




#----------------------------------  GMST  -----------------------------------

data_path='/public/home/fcai/abc/2Paper_blocking/historical_Dynamic/'
yearly_GMST = pd.read_csv(f"{data_path}GMST_LOWESS.csv",delim_whitespace=True, skiprows=1, index_col=0,usecols=['Year','No_Smoothing','Lowess(5)'])






#---------------------------------------------
# point

T_thermo_anom = np.zeros((6808, 71, 360))
T_dyn_anom = np.zeros((6808, 71, 360))

window=20;start_year=1950; end_year=2023


for ilat in range(0,71):
    print(ilat)

    for ilon in range(-159,-119):
        print(ilon)
        j=ilon+179

        # mask ocean
        if topo_NH[ilat, j] <= 0: 
            T_dyn_anom[:,ilat, j]=0
            continue

        # lon condition
        # if -160 <= ilon <= 160:
        lon_sel = (ilon - window <= lon) & (lon <= ilon + window)

        
       ## first process the data correctly
        Y_, X_, penalty, t2m_mean, t2m_std, stream500_x2, GMST_x1 = preprocess_data(
            ilon, ilat, t2m, stream, yearly_GMST,
            lower_year=start_year, upper_year=end_year, window=window)
        
        ## fit the model for this gridpoint
        optimal_lambda, MSE, R2, T2M_thermodynamical, _, T2M_dynamical, T2M_combined = decompose_one_gridpoint(
            Y_, X_, penalty, t2m_mean, t2m_std, stream500_x2, GMST_x1, cross_validate = False,
                            alpha = 4.3890,
                            K=5)

        print(T2M_thermodynamical.shape); print(t2m_std.shape)
        T_dyn_anom[:, ilat, j] = T2M_dynamical
        T_thermo_anom[:, ilat, j] = T2M_thermodynamical* t2m_std.values
        
 
        print(f"best_alpha: {optimal_lambda}")
        print(f"MSE: {MSE}")
        print(f"R2: {R2}")
        

LAT = ds0.latitude.loc[70:0][::-1]



# ##--##--##--##--  save .nc  --##--##--##--##

# T_all_pred = lch.xr.DataArray(data=T_all_pred,dims=["time","lat","lon"], coords={"time":t2m.time, "lat":lat.values, "lon":lon.values})
T_dyn_anom = xr.DataArray(data=T_dyn_anom,dims=["time","lat","lon"], coords={"time":stream_JJA.time, "lat":LAT.values, "lon":lon.values})
T_thermo_anom = xr.DataArray(data=T_thermo_anom,dims=["time","lat","lon"], coords={"time":stream_JJA.time, "lat":LAT.values, "lon":lon.values})

ds = xr.Dataset(data_vars=dict( T_dyn_anom=T_dyn_anom,T_thermo_anom= T_thermo_anom )) #T_all_pred=T_all_pred,
output_dir = "/public/home/fcai/abc/2Paper_blocking/historical_Dynamic/"
ds.to_netcdf(output_dir + "-159_-120_Dynamic_contribution_daily_19502023_JJA.nc")