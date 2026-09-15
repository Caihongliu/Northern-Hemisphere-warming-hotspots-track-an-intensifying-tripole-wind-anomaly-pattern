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



def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma





##--##--##--##--  创建 Heatwave 数据  --##--##--##--##
ds5 = xr.open_dataset("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/1_threshold/T_01_day_JA_79_23.nc")
indexhw = ds5.heatwave.loc[:,:,-21:89,:]
print(indexhw)

#--##--##--##-- 创建 U250 数据  --##--##--##--##

ds4 = xr.open_dataset("/public/home/fcai/data0/ERA5_ua_daily2x2/ua.1979.nc")
u = ds4.u.loc[:,300,-21:89,:]
lat = u.lat.loc[-21:89]; lon = u.lon;print(lat) 
print(np.shape(u))
u250 = np.zeros((45,62,np.shape(u)[1],np.shape(u)[2]))
 
for iyear in range(45):
  print('iyear = ',iyear+1979)
  # zJune = xr.open_dataset("/public/home/fcai/data0/ERA5_ua_daily2x2/ua."+str(iyear+1979)+".nc")
  # u250[iyear,0:30,:,:] = zJune.u.loc[151:181,300,:,:]
  zJuly = xr.open_dataset("/public/home/fcai/data0/ERA5_ua_daily2x2/ua."+str(iyear+1979)+".nc")
  u250[iyear,0:31,:,:] = zJuly.u.loc[181:212,300,-21:89,:] 
  zAugust = xr.open_dataset("/public/home/fcai/data0/ERA5_ua_daily2x2/ua."+str(iyear+1979)+".nc")
  u250[iyear,31:62,:,:] = zAugust.u.loc[212:243,300,-21:89,:]
  zJuly.close(); zAugust.close()

## ------------- 1. 计算eddy U250异常 和 标准差  
# ua_run_90 = np.zeros((45,92,np.shape(u)[1],np.shape(u)[2]))
 

# for iyear in range(45):
#     for iday in range(92):
#         for ilat in range(45):
#             ua_run_90[iyear,iday,ilat,:]= runavg(u250[iyear,iday,ilat,:],45)  #2x2 so 90/2
          

u_JJA_clm = np.mean(u250[2:32,:,:,:],0)
tile_u_JJA_clm =np.tile(u_JJA_clm ,(45,1,1,1))
u_ano = np.subtract(u250[:,:,:,:],tile_u_JJA_clm)
u_std = np.std(u250[2:32,:,:,:],0)


## ------------- 2. 挑选 westerly day

tile_u_std = np.tile(u_std,(45,1,1,1))
westerly_day_data = u_ano >= (tile_u_std)*1.0

print(np.shape(westerly_day_data)); print(np.shape(indexhw))

# ## ------------- 3. 把 不是 heatwave 的都标记为0

# westerly_day_True2 = westerly_day_data.copy()
# for ilon in range(180):
#     print(" ilon = " + str(ilon))
#     for ilat in range(45): #固定一点看与全球的关系
#        print(" ilat = " + str(ilat))
#        if np.isnan(indexhw[:,:,ilat,ilon]).all():
#         break
#        else:
#         for jlon in range(180):
#             print(" jlon = " + str(jlon))
#             for jlat in range(45):
#                 westerly_day_True2[:,:,jlat,jlon][indexhw[:,:,ilat,ilon] ==0] = 0
       
# ## ------------- 3. 对于每一个点去求 他和全球的联系
# westerly_day_True2 = westerly_day_data.copy()



for iyear in range(45):
    network=np.zeros((np.shape(lat)[0],np.shape(lon)[0],np.shape(lat)[0],np.shape(lon)[0]))
    print(" iyear = " + str(iyear))
    westerly=westerly_day_data[iyear,:,:,:]
    hw=indexhw[iyear,:,:,:]
    
    #确定一个点
    for ilon in range(180):
        print(" ilon = " + str(ilon))
        for ilat in range(-11,45): #固定一点看与全球的关系
            print(" ilat = " + str(ilat))
            if np.isnan(indexhw[:,:,ilat,ilon]).all():
                break
            else:
                hw_3D=np.tile( hw[:,ilat,ilon],(np.shape(u)[1],np.shape(u)[2],1)).transpose(2,0,1)
                concurrent=np.sum(westerly[:,:,:]+ hw_3D[:,:,:]==2, axis=0)
                network[ilat,ilon,:,:] = concurrent 
                # for jlon in range(180):
                #     print(" jlon = " + str(jlon))
                # for jlat in range(45):
                #     westerly_day_True2[:,:,jlat,jlon][ ==0] = 0

    ##--##--##-- 存储 --##--##--##
    network = xr.DataArray(data= network, dims=['lat1','lon1','lat2','lon2'],
                coords= {'lat1':lat.data , 'lon1':lon.data, 'lat2':lat.data , 'lon2':lon.data})

    ds3 = xr.Dataset(data_vars= dict(network = network))
    ds3.to_netcdf("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/2_U300positive_network/Westerly_network/concurrent_day_"+str(iyear+1979)+".nc")
    ds3.close()
    del network
