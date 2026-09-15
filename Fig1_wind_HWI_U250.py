from cartopy.util import add_cyclic_point
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from copy import copy
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import shapely.geometry as sgeom
import matplotlib.colorbar as colorbar
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from matplotlib.patches import Rectangle,Polygon
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
from global_land_mask import globe


from multiprocessing import Pool
import math
import time
 
def mask_land(ds, label='land', lonname='lon'):
    if lonname == 'lon':
        lat = ds.lat.data
        lon = ds.lon.data
        if np.any(lon > 180):
            lon = lon - 180
            lons, lats = np.meshgrid(lon, lat)
            mask = globe.is_ocean(lats, lons)
            temp = []
            temp = mask[:, 0:(len(lon) // 2)].copy()
            mask[:, 0:(len(lon) // 2)] = mask[:, (len(lon) // 2):]
            mask[:, (len(lon) // 2):] = temp
        else:
            lons, lats = np.meshgrid(lon, lat)# Make a grid
            mask = globe.is_ocean(lats, lons)# Get whether the points are on ocean.
        ds.coords['mask'] = (('lat', 'lon'), mask)
    elif lonname == 'longitude':
        lat = ds.latitude.data
        lon = ds.longitude.data
        if np.any(lon > 180):
            lon = lon - 180
            lons, lats = np.meshgrid(lon, lat)
            mask = globe.is_ocean(lats, lons)
            temp = []
            temp = mask[:, 0:(len(lon) // 2)].copy()
            mask[:, 0:(len(lon) // 2)] = mask[:, (len(lon) // 2):]
            mask[:, (len(lon) // 2):] = temp
        else:
            lons, lats = np.meshgrid(lon, lat)
            mask = globe.is_ocean(lats, lons)
        lons, lats = np.meshgrid(lon, lat)
        mask = globe.is_ocean(lats, lons)
        ds.coords['mask'] = (('latitude', 'longitude'), mask)
    if label == 'land':
        ds = ds.where(ds.mask == True)
    elif label == 'ocean':
        ds = ds.where(ds.mask == False)
    return ds
 
# make_map_Eurasia(ax1) 

def spatial_corr(field1, field2):
    """Spatial correlation coefficient"""
    # flatten
    f1 = field1.flatten()
    f2 = field2.flatten()
    # remove NaN
    mask = ~np.isnan(f1) & ~np.isnan(f2)
    f1 = f1[mask]
    f2 = f2[mask]
    
    return np.corrcoef(f1, f2)[0, 1]

def pearson_significance(r, n):
    """Student-t test"""
    from scipy.stats import t
    t_stat = r * np.sqrt((n-2) / (1-r**2))
    p_value = 2 * t.sf(np.abs(t_stat), df=n-2)  # two-t 
    return p_value


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

##--##--##--##--  Load temperature data --##--##--##--##

ds1 = xr.open_dataset("/public/home/fcai/abc/2Paper_blocking/historical_Dynamic/NEW_T_dyn_therm_anom_19502023_JJA_NH.nc")
ds1.coords['lon'] = np.mod(ds1['lon'], 360)
ds1 = ds1.reindex({ 'lon' : np.sort(ds1['lon'])})
 
T_dyn_anom = ds1.T_dyn_anom 
lat = ds1.lat.loc[0:70];   lon = ds1.lon 
year=T_dyn_anom.time.dt.year
month=T_dyn_anom.time.dt.month
 
## ------------- 1. read Tdyn (1979-2023, JA)
year_ind=(year>=1979)&(year<=2023)

T_dyn_Jul=T_dyn_anom.where(year_ind&(month==7),drop=True)
T_dyn_Aug=T_dyn_anom.where(year_ind&(month==8),drop=True)

T_dyn_4D=np.zeros((45,62,71,360))

for iyear in range(1979,2024):
    i=iyear-1979
    T_dyn_4D[i,0:31,:,:]=T_dyn_Jul[i*31+0:i*31+31,:,:]
    T_dyn_4D[i,31:62,:,:]=T_dyn_Aug[i*31+0:i*31+31,:,:]
   

##--##--##--##--    75th percentile  --##--##--##--##
t_th75 = np.percentile(T_dyn_4D, 75, axis=1)     
print(np.shape(t_th75))   



##--##--## trends --##--##--##
nt = 45;nlat=71;nlon=360
T_trend=trend_result(t_th75,nt,nlat,nlon)
p_values_grd = T_trend[1]
anom_grd = T_trend[0]
anom_grd = np.where(anom_grd==0,np.nan,anom_grd )

anom_grd = xr.DataArray(
    data=anom_grd,
    dims=[ "lat", "lon"],
    coords={ "lat": lat.values , "lon": lon.values}
)

t_th75 = xr.DataArray(data= t_th75, dims=['year','lat','lon'],
              coords= {'year':np.linspace(1979,2023,45) ,'lat':lat.data , 'lon':lon.data})



## ------------- 2. multi-model mean of trend in Dyn T75

ds1 = xr.open_dataset("/public/home/fcai/abc/2Paper_blocking/CMIP6_data/complex_network/6_pattern/data/Dynamical_T75_model_day_JA_19792023.nc") #"New_Dynamical_T75_model_day_JA_19502023.nc"
ds1.coords['lon'] = np.mod(ds1['lon'], 360)
ds1 = ds1.reindex({ 'lon' : np.sort(ds1['lon'])})
 
air_75 = ds1.air_th75;print(air_75);print(air_75)

Model_T_trend=np.zeros((20,71,360))

##--##--## trends --##--##--##
nt = 45;nlat=71;nlon=360

for i in range(20):
    trend=trend_result(air_75[i,:,:,:].to_numpy(),nt,nlat,nlon)
    Model_T_trend[i,:,:] = trend[0]

# ds6 = xr.open_dataset("/public/home/fcai/abc/2Paper_blocking/Fig1/MMM_T_dyn_therm_anom_19792023_JA_NH.nc")
# Model_T_trend = ds6.Model_T_trend


Model_T_trend = xr.DataArray(
    data=Model_T_trend,
    dims=["Model", "lat", "lon"],
    coords={"Model":range(20), "lat": air_75.lat.values , "lon": air_75.lon.values}
)


MMM_T_anom = np.nanmean(Model_T_trend,0)
MMM_T_anom = xr.DataArray(
    data=MMM_T_anom,
    dims=["lat", "lon"],
    coords={"lat": air_75.lat.values , "lon": air_75.lon.values}
)
 
# -------------- test --------------
T_test=np.zeros((20,71,360))
T_test=np.where(Model_T_trend>0, 1, 0)
T_test_Neg=np.where(Model_T_trend<0, 1, 0)


####20*90%=18
p_t_test_pos=np.sum(T_test,0)
p_t_test_neg=np.sum(T_test_Neg,0)


 
 

## ------------- 4. Total T and dynamical T over four regions 


print(np.shape(t_th75))
dyn_WNA = np.mean(t_th75.loc[:,42:42+12,235:235+15], axis=(1,2))  ;print(np.shape(dyn_WNA))
dyn_EE  = np.mean(t_th75.loc[:,45:45+20,22:22+38], axis=(1,2))  ;print(np.shape(dyn_EE))
dyn_CA  = np.mean(t_th75.loc[:,30:30+15,88:88+18], axis=(1,2)) 
dyn_NA  = np.mean(t_th75.loc[:,58:58+11,135:135+40], axis=(1,2)) 

nt=45
dyn_WNA_trend = trend_1D_result(dyn_WNA.values ,nt)
dyn_EE_trend = trend_1D_result(dyn_EE.values ,nt)
dyn_CA_trend = trend_1D_result(dyn_CA.values ,nt)
dyn_NA_trend = trend_1D_result(dyn_NA.values ,nt)



#raw
 
ds2 = xr.open_dataset("/public/data2/model/fcai/data-observation/ERA5_daily/t2m/t2m.1979-01.daily.nc")
print(ds2)
t = ds2.t2m.loc[:,70:0,:][:,::-1,:]
lat = t.latitude; lon = t.longitude;print(lat) 

t2m = np.zeros((45,62,np.shape(t)[1],np.shape(t)[2]))
 
for iyear in range(45):
  print('iyear = ',iyear+1979)

  tJuly = xr.open_dataset("/public/data2/model/fcai/data-observation/ERA5_daily/t2m/t2m."+str(iyear+1979)+"-07.daily.nc")
  t2m[iyear,0:31,:,:] = tJuly.t2m.loc[:,70:0,:][:,::-1,:]
  tAugust = xr.open_dataset("/public/data2/model/fcai/data-observation/ERA5_daily/t2m/t2m."+str(iyear+1979)+"-08.daily.nc")
  t2m[iyear,31:62,:,:] = tAugust.t2m.loc[:,70:0,:][:,::-1,:]
  tJuly.close(); tAugust.close()

 

##--##--##--##-- calculate 75th percentile for each summer  --##--##--##--##
t2m_th75 = np.percentile(t2m, 75, axis=1)    
print(np.shape(t2m_th75))    

t2m_th75 = xr.DataArray(data= t2m_th75, dims=['year','lat','lon'],
              coords= {'year':np.linspace(1979,2023,45) ,'lat':lat.data , 'lon':lon.data})


# Raw
T_WNA = np.mean(t2m_th75.loc[:,42:42+12,235:235+15],axis=(1,2))  
T_EE  = np.mean(t2m_th75.loc[:,45:45+20,22:22+38],axis=(1,2)) 
T_CA  = np.mean(t2m_th75.loc[:,30:30+15,88:88+18],axis=(1,2))
T_NA  = np.mean(t2m_th75.loc[:,58:58+11,135:135+40],axis=(1,2))

nt=45
T_WNA_trend = trend_1D_result(T_WNA ,nt)
T_EE_trend = trend_1D_result(T_EE ,nt)
T_CA_trend = trend_1D_result(T_CA ,nt)
T_NA_trend = trend_1D_result(T_NA ,nt)

print("dyn_WNA",dyn_WNA_trend,"      T_WNA",T_WNA_trend)# ,"      rate",(dyn_WNA_trend/T_WNA_trend ))
print("dyn_EE",dyn_EE_trend,"      T_EE",T_EE_trend)#,"      rate",str(dyn_EE_trend/T_EE_trend ))
print("dyn_CA",dyn_CA_trend,"      T_CA",T_CA_trend)#,"      rate",str(dyn_CA_trend/T_CA_trend ))
print("dyn_NA",dyn_NA_trend,"      T_NA",T_NA_trend)#,"      rate",str(dyn_NA_trend/T_NA_trend ))

print("      rate",(dyn_WNA_trend/T_WNA_trend ))
print("      rate",str(dyn_EE_trend/T_EE_trend ))
print("      rate",str(dyn_CA_trend/T_CA_trend ))
print("      rate",str(dyn_NA_trend/T_NA_trend ))



# Model


Model_dyn_WNA = np.mean(Model_T_trend.loc[:,42:42+12,235:235+15], axis=(1,2))   # only trend left
Model_dyn_EE  = np.mean(Model_T_trend.loc[:,45:45+20,22:22+38], axis=(1,2))
Model_dyn_CA  = np.mean(Model_T_trend.loc[:,30:30+15,88:88+18], axis=(1,2)) 
Model_dyn_NA  = np.mean(Model_T_trend.loc[:,58:58+11,135:135+40], axis=(1,2)) 


#raw
 
# ds5 = xr.open_dataset("/public/home/fcai/abc/2Paper_blocking/historical_Dynamic/Model_raw_T75_19502014_JA.nc")
ds5 = xr.open_dataset("/public/home/fcai/abc/2Paper_blocking/historical_Dynamic/Model_raw_T75_19792023_JA.nc")
Model_raw_th75 = ds5.t2m_th75 
lat = ds5.lat; lon = ds5.lon


# Raw
Model_T_WNA = np.mean(Model_raw_th75.loc[:,:,42:42+12,-125:-110],axis=(2,3))  
Model_T_EE  = np.mean(Model_raw_th75.loc[:,:,45:45+20,22:22+38],axis=(2,3)) 
Model_T_CA  = np.mean(Model_raw_th75.loc[:,:,30:30+15,88:88+18],axis=(2,3))
Model_T_NA  = np.mean(Model_raw_th75.loc[:,:,58:58+11,135:135+40],axis=(2,3))


nt=45
Model_raw_WNA_trend =np.zeros((20))
Model_raw_EE_trend =np.zeros((20))
Model_raw_CA_trend =np.zeros((20))
Model_raw_NA_trend =np.zeros((20))

print( Model_raw_NA_trend)

for i in range(20):
    Model_raw_WNA_trend[i] = trend_1D_result(Model_T_WNA[i,:] ,nt)
    Model_raw_EE_trend[i] = trend_1D_result(Model_T_EE[i,:] ,nt)
    Model_raw_CA_trend[i] = trend_1D_result(Model_T_CA[i,:] ,nt)
    Model_raw_NA_trend[i] = trend_1D_result(Model_T_NA[i,:] ,nt)


#---------------------- Plot function ------------------------
with open('/public/home/fcai/abc/data/CN-border-La.dat') as src:
    context = src.read()
    blocks = [cnt for cnt in context.split('>') if len(cnt) > 0]
    borders = [np.fromstring(block, dtype=float, sep=' ') for block in blocks]


# Create a figure and axis

projection = ccrs.Robinson(central_longitude=150)

fig = plt.figure(figsize=[8, 12],frameon=True,dpi=300)
ax1 = fig.add_axes([0,0.7,1,0.7],projection=projection)
ax2 = fig.add_axes([0,0.4,1,0.7],projection=projection)
 

ax22 = fig.add_axes([1.11, 0.76,0.34, 0.4])


# # prepare 
box = [0, 360, 10, 70]  
scale = '50m'            
xstep, ystep = 90, 30

def make_map_Eurasia(ax):
    ax.set_extent(box, crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(box[0],box[1]+xstep, xstep),crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(box[2], box[3]+ystep, ystep),crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label =False))
    ax.tick_params(which='major', 
                    direction='out', 
                    length=8,
                    width=0.99, 
                    pad=3, 
                    labelsize=12,
                    bottom=True, left=True, right=False, top=False)
    ax.add_feature(cfeature.COASTLINE,  linewidth=0.4 , edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linestyle = 'dotted',lw = 0.25, edgecolor='black')
    return ax



def make_box(x,y,ax,colors,title,lat1,lat2):
    # add some docoration
    ax.plot(x, y,color = colors)
    ax.tick_params(axis='both',labelsize=8, length = 3,direction = 'in', width=0.49)

    ax.axhspan(lat1, lat2, facecolor='#e8e8e8', alpha=0.7)
    ax.axvline(x=0,linestyle='--',color='black',linewidth=1 )
    ax.set_facecolor('none')
    
    ax.set_ylim(10,80)
    ax.set_yticks(np.arange(10, 91, 20))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(title, loc='left', color='black', fontsize=8)

#---------------------- Plotting ------------------------
import matplotlib as mpl
mpl.rcParams['hatch.color'] = 'white'

#colorbar
cbar_kwargs = {
    'orientation': 'horizontal',  
    'label': '°C/decade',
    'ticks': np.arange(-0.4, 0.4 + 0.2, 0.2),
    'pad': 0.03,
    'shrink': 0.55,
    'extend': 'both',
    'extendfrac': 0.1  
}

#level
levels = np.arange(-0.4, 0.4 + 0.05, 0.05)

# 'RdYlGn'
cmap = plt.get_cmap('RdBu')
cmap_colors = cmap(np.arange(cmap.N))
cmap_reversed = mcolors.LinearSegmentedColormap.from_list('Reversed', cmap_colors[::-1])


global_t_composite2, cycle_lon = add_cyclic_point(anom_grd, coord=lon)  
mappable = ax1.contourf(global_t_composite2[:,:],transform=ccrs.PlateCarree(),cmap=cmap_reversed,levels=levels,extend='both') #subplot_kw={'projection':projection,'frameon':False},extend='both')

p_t_value = p_values_grd
c1p = ax1.contourf( p_t_value[:,:], levels =[0,0.1,1],hatches=[ '.....',None],colors="none", transform=ccrs.PlateCarree())

for collection in c1p.collections:
    collection.set_edgecolor('white')
for collection in c1p.collections:
    collection.set_linewidth(0)

 


ax1.axis('off') 
new_position = fig.add_axes([1.08, 0.65,0.39, 0.015])   
ax1.add_feature(cfeature.COASTLINE,  linewidth=0.5 , edgecolor='black')
ax1.add_feature(cfeature.BORDERS, linestyle = 'dotted',lw = 0.35, edgecolor='black')

cbar = plt.colorbar(mappable,cmap=cmap_reversed,norm=norm,  cax=new_position,**cbar_kwargs)
cbar.outline.set_color('none')
cbar.update_ticks()
cbar.ax.xaxis.set_tick_params(labelsize=9)
ax1.set_title(' Trends of T$_{dyn}$ in JA',loc='center',color='black',fontsize=13,y=1.03)
ax1.text(0.06,1.1, 'a', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax1.transAxes)

ax1.set_xlabel(' ')
ax1.set_ylabel(' ')
ax1.set_aspect(1.5)



### ============== Fig.b ======================



MMM_T_anom = mask_land(MMM_T_anom,'ocean','lon')
global_t_composite3, cycle_lon = add_cyclic_point(MMM_T_anom, coord=lon)  
mappable = ax2.contourf(global_t_composite3[:,:],transform=ccrs.PlateCarree(),cmap=cmap_reversed,levels=levels,extend='both') #subplot_kw={'projection':projection,'frameon':False},extend='both')

p_t_value = p_t_test_pos
c2p = ax2.contourf( p_t_value[:,:], levels =[0,13.5,20],hatches=[ None,'.....'],colors="none", transform=ccrs.PlateCarree())

for collection in c2p.collections:
    collection.set_edgecolor('white')
for collection in c2p.collections:
    collection.set_linewidth(0)

c3p = ax2.contourf( p_t_test_neg[:,:], levels =[0,13.5,20],hatches=[ None,'.....'],colors="none", transform=ccrs.PlateCarree())

for collection in c3p.collections:
    collection.set_edgecolor('white')
for collection in c3p.collections:
    collection.set_linewidth(0)
 
ax2.axis('off') 

ax2.add_feature(cfeature.COASTLINE,  linewidth=0.5 , edgecolor='black')
ax2.add_feature(cfeature.BORDERS, linestyle = 'dotted',lw = 0.35, edgecolor='black')
ax2.set_title(' Multi-model mean of trends in T$_{dyn}$',loc='center',color='black',fontsize=13,y=1.03)
ax2.text(0.06,1.1, 'b', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax2.transAxes)


############
# spatial correlation
Spatial_r = spatial_corr(global_t_composite2.data, global_t_composite3.data)
print("figure1 & figure 2 similarity:",Spatial_r)
n =len(global_t_composite2.data.flatten())
p = pearson_significance(Spatial_r,n)
print(f"NA Spatial correlation: {Spatial_r:.2f}")

ax2.set_xlabel(' ')
ax2.set_ylabel(' ')
ax2.set_aspect(1.5)
 
#square markers (NE US)


for k in range(1,3):

    # West NA
    locals()['rectangle' + str(k)] = Rectangle(
        (235, 42),  #  (longitude, latitude)
        15, 12, #  (longitude, latitude)
        linewidth=1.5,edgecolor='black',
        facecolor='none',linestyle='--',
        transform=ccrs.PlateCarree()
    )
    locals()['ax' + str(k)].add_patch(locals()['rectangle' + str(k)]) 


    #East Europe
    locals()['rectangle' + str(k)] = Rectangle(
        (22, 45),
        38, 20, 
        linewidth=1.3,edgecolor='black',
        facecolor='none',linestyle='--',
        transform=ccrs.PlateCarree()
    )
    locals()['ax' + str(k)].add_patch(locals()['rectangle' + str(k)])


    #Central Asia
    locals()['rectangle' + str(k)] = Rectangle(
        (88, 30),  #  (longitude, latitude)
        18, 15, #  (longitude, latitude)
        linewidth=1.3,edgecolor='black',
        facecolor='none',linestyle='--',
        transform=ccrs.PlateCarree()
    )
    locals()['ax' + str(k)].add_patch(locals()['rectangle' + str(k)])  

    #North Asia
    locals()['rectangle' + str(k)] = Rectangle(
        (135, 58), 
        40, 11, #  (longitude, latitude)
        linewidth=1.3,edgecolor='black',
        facecolor='none',linestyle='--',
        transform=ccrs.PlateCarree()
    )
    locals()['ax' + str(k)].add_patch(locals()['rectangle' + str(k)]) 



 


###----------------------------------------------------------------
### double box
###----------------------------------------------------------------


data_raw = {"EE":Model_raw_EE_trend,
            "CC":Model_raw_CA_trend,
            "ES":Model_raw_NA_trend,
            "WNA":Model_raw_WNA_trend}; data_raw = pd.DataFrame(data_raw) 

data_dyn = {"EE":Model_dyn_EE,
            "CC":Model_dyn_CA,
            "ES":Model_dyn_NA,
            "WNA":Model_dyn_WNA} ; data_dyn = pd.DataFrame(data_dyn) 

Era5_raw = [T_EE_trend,T_CA_trend,T_NA_trend,T_WNA_trend]

Era5_dyn = [dyn_EE_trend,dyn_CA_trend,dyn_NA_trend,dyn_WNA_trend]

x1 = np.arange(1,12,3) 
x2 = x1+1
# x3 = x2+1
print(Model_raw_WNA_trend)
print(Model_raw_CA_trend)
print(Model_raw_EE_trend)
print(Model_raw_NA_trend)

print(Model_dyn_WNA)
print(Model_dyn_CA)
print(Model_dyn_EE)
print(Model_dyn_NA)

box1 = ax22.boxplot(data_raw.dropna(), widths=0.4,positions=x1, patch_artist=True,showmeans=True,
            boxprops={"facecolor": "#fce2d2",
                      "edgecolor": "none",
                      "linewidth": 0.5},
            medianprops={"color": "#ec9374", "linewidth": 2},
            meanprops={'marker':'o',
                       'markerfacecolor':'#ec9374',
                       'markeredgecolor':'#ec9374',
                       'markersize':6},
            capprops={'color': "#fce2d2", 'linewidth': 1},
            flierprops={'marker': 'o','markerfacecolor':'none','markeredgecolor':"#fce2d2", 'markersize': 5},
            whiskerprops={'color':"#fce2d2",'linestyle':'--','linewidth':1.2})



box2 = ax22.boxplot(data_dyn.dropna(),widths=0.4,positions=x2,patch_artist=True,showmeans=True,
            boxprops={"facecolor": "#dae9f2",
                      "edgecolor": "none",
                      "linewidth": 0.5},
            medianprops={"color": "#569fc9", "linewidth": 2},
            meanprops={'marker':'o',
                       'markerfacecolor':'#569fc9',
                       'markeredgecolor':'#569fc9',
                       'markersize':6},
            capprops={'color': "#dae9f2", 'linewidth': 1},
            flierprops={'marker': 'o','markerfacecolor':'none','markeredgecolor':"#dae9f2", 'markersize': 5},
            whiskerprops={'color': "#dae9f2",'linestyle':'--','linewidth':1.2})

box3 = ax22.scatter(x1, Era5_raw, marker='s',s=50, facecolor="none",zorder=10,edgecolors="purple",linewidth=2,label='ERA5 - origin trend')
box4 = ax22.scatter(x2, Era5_dyn, marker='x',s=50, color="blue",zorder=10,label='ERA5 - Dynamic trend')
print("WNA Q1 =", box2['boxes'][3].get_path().vertices[:,1].min(),"          WNA Q3 =", box2['boxes'][3].get_path().vertices[:,1].max(),  "   mean",box2['means'][3].get_ydata()[0])

city = ["EE","NWC","ES","WNA"]
ax22.set_xticks([1.5,4.5,7.5,10.5],city,fontsize=12)
ax22.set_xticklabels(city)  #,rotation=30)
ax22.set_title('Temperature trends',loc='center',color='black',fontsize=13,y=1.01)
ax22.text(-0.115,1.02, 'c', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax22.transAxes)
     
ax22.axhline(y=0 , color='k', linestyle='--',linewidth=1.5,zorder=0)
# ax22.set_ylim(-0.2,0.5)
ax22.set_ylabel(' T (°C/decade)',fontsize=13)
ax22.tick_params(axis='both', which='major', labelsize=11)
 
ax22.legend(handles=[box1['boxes'][0],box2['boxes'][0],box3,box4],labels=['20 model Total','20 model Dyn','ERA5 Total','ERA5 Dyn'], frameon=False)
# box1['boxes'][0],

plt.savefig("/public/home/fcai/abc/2Paper_blocking/Final_figures/Fig1_75_HW_trend.pdf", dpi=300, bbox_inches='tight')
plt.show()