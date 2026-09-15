import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl
mpl.use("Agg")


#--##--##--##--  U250   --##--##--##--##

ds4 = xr.open_dataset("/public/home/fcai/data0/ERA5_ua_daily2x2/ua.1979.nc")
u = ds4.u.loc[:,300,:,:]
lat = u.lat.loc[:]; lon = u.lon;print(lat) 
print(np.shape(u))
u250 = np.zeros((45,62,np.shape(u)[1],np.shape(u)[2]))
 
for iyear in range(45):
  print('iyear = ',iyear+1979)
  zJuly = xr.open_dataset("/public/home/fcai/data0/ERA5_ua_daily2x2/ua."+str(iyear+1979)+".nc")
  u250[iyear,0:31,:,:] = zJuly.u.loc[181:212,300,:,:] 
  zAugust = xr.open_dataset("/public/home/fcai/data0/ERA5_ua_daily2x2/ua."+str(iyear+1979)+".nc")
  u250[iyear,31:62,:,:] = zAugust.u.loc[212:243,300,:,:]
  zJuly.close(); zAugust.close()

## ------------- 1.  U250 anomaly and std  
u_JJA_clm = np.mean(u250[2:32,:,:,:],0)
tile_u_JJA_clm =np.tile(u_JJA_clm ,(45,1,1,1))
u_ano = np.subtract(u250[:,:,:,:],tile_u_JJA_clm)
u_std = np.std(u250[2:32,:,:,:],0)

## ------------- 2. select westerly day

tile_u_std = np.tile(u_std,(45,1,1,1))
westerly_day_data = u_ano <= (tile_u_std)* -1.0    # Westerly or Easterly
print(np.shape(westerly_day_data))

##--##--##--##--  save  --##--##--##--##
westerly_day_data= xr.DataArray(data=westerly_day_data.data, dims=['year','day','lat','lon'],
                      coords={'year':np.arange(1979,2023+1,1), 'day':np.arange(1,62+1,1), 'lat':lat, 'lon':lon})
ds2 = xr.Dataset(data_vars=dict(easterly_day_data=westerly_day_data))

ds2.to_netcdf("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/1_threshold/Easterly_01_day_JA_79_23.nc")   
ds2.close()