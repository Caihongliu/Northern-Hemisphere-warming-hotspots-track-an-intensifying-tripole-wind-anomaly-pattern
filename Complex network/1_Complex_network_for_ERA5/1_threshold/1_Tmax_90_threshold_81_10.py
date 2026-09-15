
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import time
import datetime as dt  

##--##--##--##-- T data  --##--##--##--##
ds0 = xr.open_dataset("/public/data2/model/fcai/data0/ERA5_Tmax_daily2x2/tmax.1979.nc")
t0 = ds0.tmax.loc[:,:,:]
lat = t0.lat; lon = t0.lon ; print(lat); print(lon)
print(np.shape(t0))
t = np.zeros((30, 122, np.shape(t0)[1],np.shape(t0)[2]))



##--##--##--##--  load ERA5T2m (1979-2022, 5-10月)  --##--##--##--##
for iyear in range(30):
  print('iyear = ',iyear+1981)
  # ds1 = xr.open_dataset("/public/home/fcai/data0/ERA5_Tmax_daily2x2/tmax."+str(iyear+1981)+".nc")
  # t[iyear,0:31,:,:] = ds1.tmax.loc[120:151,:,:]
  ds2 = xr.open_dataset("/public/data2/model/fcai/data0/ERA5_Tmax_daily2x2/tmax."+str(iyear+1981)+".nc")
  t[iyear,0:30,:,:] = ds2.tmax.loc[151:181,:,:];print("151:181 ->", ds2.tmax.loc[151:181,:,:].shape)
  ds3 = xr.open_dataset("/public/data2/model/fcai/data0/ERA5_Tmax_daily2x2/tmax."+str(iyear+1981)+".nc")
  t[iyear,30:61,:,:] = ds3.tmax.loc[181:212,:,:]
  ds4 = xr.open_dataset("/public/data2/model/fcai/data0/ERA5_Tmax_daily2x2/tmax."+str(iyear+1981)+".nc")
  t[iyear,61:92,:,:] = ds4.tmax.loc[212:243,:,:]
  ds5 = xr.open_dataset("/public/data2/model/fcai/data0/ERA5_Tmax_daily2x2/tmax."+str(iyear+1981)+".nc")
  t[iyear,92:122,:,:] = ds5.tmax.loc[243:273,:,:]
  ds2.close(); ds3.close(); ds4.close(); ds5.close()



##--##--##--##--  90th   --##--##--##--##
t_15days = np.zeros((30*15, 62, np.shape(t0)[1],np.shape(t0)[2]))
print(t_15days.size * t_15days.itemsize / 1073741824)    
for iday in range(62):
  if iday%30==0: print('iday = ',iday)
  for lag in range(15):
    t_15days[30*lag:30*lag+30,iday,:,:] = t[:,30+iday+lag-7,:,:]   

t_th90 = np.percentile(t_15days, 90, axis=0) 
print(np.shape(t_th90))  



##--##--##--##--  save  --##--##--##--##
air_th90= xr.DataArray(data=t_th90.data, dims=['day','lat','lon'],
                      coords={'day':np.arange(1,63,1), 'lat':lat, 'lon':lon})
ds2 = xr.Dataset(data_vars=dict(air_th90=air_th90))

ds2.to_netcdf("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/1_threshold/Tmax_2x2_th90_JA_81_10.nc")   
ds2.close()

