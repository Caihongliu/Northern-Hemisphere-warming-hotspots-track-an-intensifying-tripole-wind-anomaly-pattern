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
from pylab import *
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
module_path = '/public/home/fcai/abc/1Paper_2023/data0'
sys.path.append(module_path)
import glob
import os
import lch
from scipy.stats import pearsonr
from mymodule import make_map_left_below,runavg,make_axvline,make_map_left


def runavg(x, width):
 
    n = len(x)
    x = np.append(x, np.append(x, x))
    x_smooth = np.convolve(x, np.ones(width)/width, mode='valid')
    xs = x_smooth[n:2*n]
    return xs


# =========================
# 1. function
# =========================
def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def weighted_cov(x, y, w):
    mx = weighted_mean(x, w)
    my = weighted_mean(y, w)
    return np.sum(w * (x - mx) * (y - my)) / np.sum(w)

def weighted_corr(x, y, w):
    cov_xy = weighted_cov(x, y, w)
    var_x = weighted_cov(x, x, w)
    var_y = weighted_cov(y, y, w)
    if var_x <= 0 or var_y <= 0:
        return np.nan
    return cov_xy / np.sqrt(var_x * var_y)

def pearson_significance_from_neff(r, neff):
    """
    """
    if np.isnan(r) or neff is None or neff <= 2:
        return np.nan, np.nan

   
    r = np.clip(r, -0.999999, 0.999999)
    tval = r * np.sqrt((neff - 2) / (1 - r**2))
    pval = 2 * (1 - stats.t.cdf(np.abs(tval), df=neff - 2))
    return tval, pval
     
 



ds0=xr.open_dataset("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/Function/U300Positive_significant_99.nc")
lat=ds0.lat1; lon=ds0.lon1
network=ds0.significant_99
print(network)

##--##--## Different regions --##--##--##


concur_WNA = np.sum(network.loc[42:42+12,-125:-110,:,:], axis=(0,1))     
concur_EE  = np.sum(network.loc[45:45+20,22:60,:,:], axis=(0,1)) 
concur_CA  = np.sum(network.loc[32:32+18,88:88+18,:,:], axis=(0,1)) 
concur_NA  = np.sum(network.loc[58:58+11,135:135+40,:,:], axis=(0,1)) 


network_WNA = (concur_WNA)
network_EE = (concur_EE)
network_CA = (concur_CA)
network_NA = (concur_NA)

# network_all = network_WNA+network_EE+network_CA+network_NA

network_EE = xr.DataArray(data= network_EE, dims=['lat','lon'],  
            coords= {'lat':lat.data , 'lon':lon.data})    ; print(network_EE.shape)
network_CA = xr.DataArray(data= network_CA, dims=['lat','lon'],
            coords= {'lat':lat.data , 'lon':lon.data})
network_NA = xr.DataArray(data= network_NA, dims=['lat','lon'],
                coords= {'lat':lat.data , 'lon':lon.data})
network_WNA = xr.DataArray(data= network_WNA, dims=['lat','lon'],
            coords= {'lat':lat.data , 'lon':lon.data})
# network_All = xr.DataArray(data= network_all, dims=['lat','lon'],
#             coords= {'lat':lat.data , 'lon':lon.data})

 




#----------------------Plot function ------------------------
with open('/public/home/fcai/abc/data/CN-border-La.dat') as src:
    context = src.read()
    blocks = [cnt for cnt in context.split('>') if len(cnt) > 0]
    borders = [np.fromstring(block, dtype=float, sep=' ') for block in blocks]


# Create a figure and axis
fig = plt.figure(figsize=[8, 12],frameon=True)
ax1 = fig.add_axes([0, 0.8, 0.6, 0.2], projection=ccrs.PlateCarree(central_longitude=163) )
ax2 = fig.add_axes([0, 0.58, 0.6, 0.2], projection=ccrs.PlateCarree(central_longitude=163) )
ax3 = fig.add_axes([0, 0.36, 0.6, 0.2], projection=ccrs.PlateCarree(central_longitude=163) )
ax4 = fig.add_axes([0, 0.14, 0.6, 0.2], projection=ccrs.PlateCarree(central_longitude=163) )
# ax5 = fig.add_axes([0, 0.06, 0.6, 0.2], projection=ccrs.PlateCarree(central_longitude=163) )


# prepare 
box = [-180, 180, 10, 80]
scale = '50m'            
xstep, ystep = 90, 35

def make_map_Eurasia(ax):
    ax.set_extent(box, crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(box[0],box[1]+xstep, xstep),crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(box[2], box[3]+ystep, ystep),crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label =False)) 
    ax.tick_params(which='major', 
                    direction='out', 
                    length=4,
                    width=0.99, 
                    pad=3, 
                    labelsize=12,
                    bottom=True, left=True, right=False, top=False)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
    return ax

# # prepare 
# box = [0, 360, 10, 80]  
# scale = '50m'            
# xstep, ystep = 90, 35
# xminor_step,yminor_step = 30,15
# make_map_left(ax1,box,xstep, ystep,xminor_step,yminor_step,border_width=1, border_color='k') 
# make_map_left(ax2,box,xstep, ystep,xminor_step,yminor_step,border_width=1, border_color='k') 
# make_map_left(ax3,box,xstep, ystep,xminor_step,yminor_step,border_width=1, border_color='k') 
# make_map_left_below(ax4,box,xstep, ystep,xminor_step,yminor_step,border_width=1, border_color='k') 

make_map_Eurasia(ax1)
make_map_Eurasia(ax2)
make_map_Eurasia(ax3)
make_map_Eurasia(ax4)
# make_map_Eurasia(ax5)

def make_composite(composite,lat,lon,ax,title,levels):#,right_title

    #colorbar
    cbar_kwargs = {
        'orientation': 'horizontal',   
        'label': '',
        'ticks':levels,
        'pad': 0.05,
        'shrink': 0.65,
        'extend': 'both',
        'extendfrac': 0.1  
    }
    # levels
    levels = levels
    Colors=('#ffffff','#ffe101','#feaa00',\
            '#ff8000','#ff3234','#da0003') 
    # Colors = ('#ffffff', '#fff2e6','#ffe0cc','#ffc2a3','#ffa07a', '#ff7f50','#e65c4f','#cc3b3b' )


            
    global_t_composite2, cycle_lon = add_cyclic_point(composite, coord=lon)
    c=ax.contourf(cycle_lon,lat ,global_t_composite2, cbar_kwargs=cbar_kwargs,transform=ccrs.PlateCarree(),colors=Colors,levels=levels,extend='both')
    ax.set_title(title,loc='center',color='black',fontsize=13,y=1.03)
    # ax.set_title(right_title,loc='right',color='black',fontsize=10,y=1.03)
    ax.set_xlabel(' ')
    ax.set_ylabel(' ')
    ax.set_aspect(1.3) 
    # ax.add_feature(cfeature.COASTLINE,  linewidth=0.5 , edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linestyle = 'dotted',lw = 0.35, edgecolor='black')

    return c

# #---------------------- Plotting ------------------------

c1=make_composite(network_EE,lat,lon,ax1,'Westerly event related to EE heatwaves',levels=[1000,2000,3000,4000,6000]) 
c2=make_composite(network_CA,lat,lon,ax2,'Westerly event related to NWC heatwaves',levels=[300,500,1000,1500,2000])
c3=make_composite(network_NA,lat,lon,ax3,'Westerly event related to ES heatwaves',levels=[500,1000,1500,2000,3000])
c4=make_composite(network_WNA,lat,lon,ax4,'Westerly event related to WNA heatwaves',levels=[300,500,1000,1500,2000])
# c5=make_composite(network_All,lat,lon,ax5,'Westerly event related to All hotspot heatwaves',levels=[100,130,150,170,190])

cax = fig.add_axes([0.04,0.8,0.4,0.012])
cbar=fig.colorbar(c1,cax = cax,ax=[ax1],orientation= 'horizontal', shrink=0.6 )
cbar.outline.set_color('none')
cbar.update_ticks()
cbar.set_ticks([1000,2000,3000,4000,6000])
cbar.set_ticklabels([ 10,20,30,40,60])
cbar.ax.xaxis.set_tick_params(labelsize=10)

cax1 = fig.add_axes([0.04,0.58,0.4,0.012])
cbar1=fig.colorbar(c2,cax = cax1,ax=[ax2],orientation= 'horizontal' )
cbar1.outline.set_color('none')
cbar1.update_ticks()
cbar1.set_ticks([300,500,1000,1500,2000])
cbar1.set_ticklabels([3,5,10,15,20])
cbar1.ax.xaxis.set_tick_params(labelsize=10)

cax2 = fig.add_axes([0.04,0.36,0.4,0.012])
cbar2=fig.colorbar(c3,cax = cax2,ax=[ax3],orientation= 'horizontal', shrink=0.6 )
cbar2.outline.set_color('none')
cbar2.update_ticks()
cbar2.set_ticks([500,1000,1500,2000,3000])
cbar2.set_ticklabels([5,10,15,20,30])
cbar2.ax.xaxis.set_tick_params(labelsize=10)

cax4 = fig.add_axes([0.04,0.14,0.4,0.012])
cbar4=fig.colorbar(c4,cax = cax4,ax=[ax4],orientation= 'horizontal', shrink=0.6 )
cbar4.outline.set_color('none')
cbar4.update_ticks()
cbar4.set_ticks([300,500,1000,1500,2000])
cbar4.set_ticklabels([3,5,10,15,20])
cbar4.ax.xaxis.set_tick_params(labelsize=10)







ax1.text(0,1.15, 'a', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax1.transAxes)
ax2.text(0,1.15, 'c', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax2.transAxes)
ax3.text(0,1.15, 'e', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax3.transAxes)
ax4.text(0,1.15, 'g', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax4.transAxes)
# ax5.text(0,1.1, 'i', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax5.transAxes)


#  (NE Asia)


#East Europe
def create_east_europe_rectangle():
    return Rectangle(
        (22, 45),
        38, 20, 
        linewidth=1.5, edgecolor='blue',
        facecolor='none',
        transform=ccrs.PlateCarree()
    )

ax1.add_patch(create_east_europe_rectangle())
# ax5.add_patch(create_east_europe_rectangle())
 
#Central Asia
def create_northwest_china_rectangle():
    return Rectangle(
        (88, 32), 
        18, 18, 
        linewidth=1.5,edgecolor='blue',
        facecolor='none',
        transform=ccrs.PlateCarree()
    )

ax2.add_patch(create_northwest_china_rectangle())
# ax5.add_patch(create_northwest_china_rectangle())
# 
 
#North Asia
def create_eastern_siberia_rectangle():
    return Rectangle(
        (135, 58), 
        40, 11, 
        linewidth=1.5,edgecolor='blue',
        facecolor='none',
        transform=ccrs.PlateCarree()
    )
ax3.add_patch(create_eastern_siberia_rectangle()) 
# ax5.add_patch(create_eastern_siberia_rectangle()) 

#West NA
def create_western_america_rectangle():
    return Rectangle(
        (235, 42), 
        15, 12, 
        linewidth=1.5,edgecolor='blue',
        facecolor='none',
        transform=ccrs.PlateCarree()
    )
ax4.add_patch(create_western_america_rectangle()) 
# ax5.add_patch(create_western_america_rectangle()) 

### 
# for i in range(1,5):
#     locals()['ax' + str(i)].plot([143-180,143-180],[10,80],linewidth=1,color='#cdd0cd',linestyle = 'dashed')
#     locals()['ax' + str(i)].plot([62+18,62+18],[10,80],linewidth=1,color='#cdd0cd',linestyle = 'dashed')
#     locals()['ax' + str(i)].plot([90+5-180,90+5-180],[10,80],linewidth=1,color='#cdd0cd',linestyle = 'dashed')
#     locals()['ax' + str(i)].plot([-155,-155],[10,80],linewidth=1,color='#cdd0cd',linestyle = 'dashed')

for i in range(1, 5):
    ax = locals()['ax' + str(i)]
    
    ax.plot(
        [242, 242],      
        [10, 80],  
        linestyle='--',
        color='#cdd0cd',
        linewidth=2,
        transform=ccrs.PlateCarree()
    )

    ax.plot(
        [38, 38],      
        [10, 80],      
        linestyle='--',
        color='#cdd0cd',
        linewidth=2,
        transform=ccrs.PlateCarree()
    )
  
    
    ax.plot(
        [97, 97],      
        [10, 80],      
        linestyle='--',
        color='#cdd0cd',
        linewidth=2,
        transform=ccrs.PlateCarree()
    )

    ax.plot(
        [155, 155],      
        [10, 80],      
        linestyle='--',
        color='#cdd0cd',
        linewidth=2,
        transform=ccrs.PlateCarree()
    )

     


plt.savefig("/public/home/fcai/abc/2Paper_blocking/Final_figures/Last_sig99_westerly_network.png", dpi=300, bbox_inches='tight')
plt.show()