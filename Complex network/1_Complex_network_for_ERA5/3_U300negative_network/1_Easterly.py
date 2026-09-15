import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl
mpl.use("Agg")
import sys
sys.path.append("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/Function")
from Function_Complex_network import computing_link_bwt_two_array,significant_links_at_each_grid, get_significant_links


##--##--##--##--  load Heatwave and easterly  --##--##--##--##
ds1 = xr.open_dataset("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/1_threshold/T_01_day_JA_79_23.nc")
heatwave = ds1.heatwave.loc[:,:,-21:89,:]
lat = ds1.lat.loc[-21:89]
lon = ds1.lon

ds2 = xr.open_dataset("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/1_threshold/Easterly_01_day_JA_79_23.nc")
easterly_day_data = ds2.easterly_day_data.loc[:,:,-21:89,:]


# computing_link_bwt_two_array(heatwave, easterly_day_data, 45, lat, lon,
#                              "/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/Easterly_network/")

significant_links_at_each_grid(heatwave, easterly_day_data,lat,lon,\
                  "/public/home/fcai/abc/2Paper_jet/Fig2/1_threshold/45y_62d_matrix99_network.nc",\
                  "/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/Function/JA_U300Negative_matrix4D_99.nc")
get_significant_links("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/Easterly_network/",
                      "/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/Function/JA_U300Negative_matrix4D_99.nc"
                      ,lat,lon,45,"/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/Function/U300Negative_significant_99.nc")