import numpy as np
import xarray as xr



##--##--##-- --##--##--##
file = xr.open_dataset("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/1_threshold/Tmax_2x2_th90_JA_81_10.nc")
t_90the = file.air_th90

##--##--##--##--  T  --##--##--##--##
ds0 = xr.open_dataset("/public/home/fcai/data0/ERA5_Tmax_daily2x2/tmax.1979.nc")
t0 = ds0.tmax.loc[:,:,:]
lat = t0.lat; lon = t0.lon ; print(lat); print(lon)
print(np.shape(t0))
t = np.zeros((45, 62, np.shape(t0)[1],np.shape(t0)[2]))



##--##--##--##--  T2m (1979-2022, 5-10)  --##--##--##--##
for iyear in range(45):
  print('iyear = ',iyear+1979)

  # ds2 = xr.open_dataset("/public/home/fcai/data0/ERA5_Tmax_daily2x2/tmax."+str(iyear+1979)+".nc")
  # t[iyear,0:30,:,:] = ds2.tmax.loc[151:181,:,:]
  ds3 = xr.open_dataset("/public/home/fcai/data0/ERA5_Tmax_daily2x2/tmax."+str(iyear+1979)+".nc")
  t[iyear,0:31,:,:] = ds3.tmax.loc[181:212,:,:]
  ds4 = xr.open_dataset("/public/home/fcai/data0/ERA5_Tmax_daily2x2/tmax."+str(iyear+1979)+".nc")
  t[iyear,31:62,:,:] = ds4.tmax.loc[212:243,:,:]
  ds3.close(); ds4.close()


#--##--##--##--  U250   --##--##--##--##

ds4 = xr.open_dataset("/public/home/fcai/data0/ERA5_ua_daily2x2/ua.1979.nc")
u = ds4.u.loc[:,300,:,:]
lat = u.lat.loc[:]; lon = u.lon;print(lat) ;print(lat)
print(np.shape(u))
u250 = np.zeros((45,62,np.shape(u)[1],np.shape(u)[2]))
 
for iyear in range(45):
  print('iyear = ',iyear+1979)
  # zJune = xr.open_dataset("/public/home/fcai/data0/ERA5_ua_daily2x2/ua."+str(iyear+1979)+".nc")
  # u250[iyear,0:30,:,:] = zJune.u.loc[151:181,300,:,:]
  zJuly = xr.open_dataset("/public/home/fcai/data0/ERA5_ua_daily2x2/ua."+str(iyear+1979)+".nc")
  u250[iyear,0:31,:,:] = zJuly.u.loc[181:212,300,:,:] 
  zAugust = xr.open_dataset("/public/home/fcai/data0/ERA5_ua_daily2x2/ua."+str(iyear+1979)+".nc")
  u250[iyear,31:62,:,:] = zAugust.u.loc[212:243,300,:,:]
  zJuly.close(); zAugust.close()

## ------------- 1. U250

          

u_JJA_clm = np.mean(u250[2:32,:,:,:],0)
tile_u_JJA_clm =np.tile(u_JJA_clm ,(45,1,1,1))
u_ano = np.subtract(u250[:,:,:,:],tile_u_JJA_clm)
u_std = np.std(u250[2:32,:,:,:],0)




##--##--##-- 判断是否大于阈值 --##--##--##
t_True = np.zeros((45,62,np.shape(t0)[1],np.shape(t0)[2]))
Intensity_True = np.zeros((45,62,np.shape(t0)[1],np.shape(t0)[2]))
u_True= np.zeros((45,62,np.shape(t0)[1],np.shape(t0)[2]))
for i in range(45):
  t_True[i,:,:,:] = np.array(t[i,:,:,:]) >= np.array(t_90the)
  Intensity_True[i,:,:,:] = np.array(t[i,:,:,:]) - np.array(t_90the[:,:,:])
  u_True[i,:,:,:] = np.array(u250[i,:,:,:]) - np.array(u_JJA_clm[:,:,:])

print("check" , np.max(t), np.max(t_90the), np.min(t_90the), np.percentile(t_90the, 50))
del t; print("heat day = ", np.sum(t_True), np.size(t_True))
print("heat Intensity (day)" , np.sum(Intensity_True), np.max(Intensity_True), np.min(Intensity_True), np.percentile(Intensity_True, 50))
print("u300 Intensity (day)" , np.sum(u_True), np.max(u_True), np.min(u_True), np.percentile(u_True, 50))




##--##--##-- Heatwave > threshold for at least 3 day in a row --##--##--##

t_True2 = t_True.copy()
t_True2[:,2:-2,:,:][t_True[:,:-4,:,:]+t_True[:,1:-3,:,:]+t_True[:,3:-1,:,:]+t_True[:,4:,:,:]<=1] = 0.0
t_True2[:,2:-2,:,:][t_True[:,1:-3,:,:]+t_True[:,3:-1,:,:]==0] = 0.0
t_True2[:,2:-2,:,:][(t_True[:,1:-3,:,:]+t_True[:,3:-1,:,:]==1)&(t_True[:,3:-1,:,:]+t_True[:,4:,:,:]<=1)&(t_True[:,:-4,:,:]+t_True[:,1:-3,:,:]<=1)] = 0.0

t_True2[:,0,:,:][t_True[:,1,:,:]+t_True[:,2,:,:]<=1] = 0.0
t_True2[:,1,:,:][t_True[:,0,:,:]+t_True[:,2,:,:]+t_True[:,3,:,:]<=1] = 0.0
t_True2[:,1,:,:][(t_True[:,0,:,:]==1)&(t_True[:,2,:,:]==0)&(t_True[:,3,:,:]==1)] = 0.0
t_True2[:,-1,:,:][t_True[:,-2,:,:]+t_True[:,-3,:,:]<=1] = 0.0
t_True2[:,-2,:,:][t_True[:,-1,:,:]+t_True[:,-3,:,:]+t_True[:,-4,:,:]<=1] = 0.0
t_True2[:,-2,:,:][(t_True[:,-1,:,:]==1)&(t_True[:,-3,:,:]==0)&(t_True[:,-4,:,:]==1)] = 0.0
del t_True



##--##--##-- save  --##--##--##
heatwave = xr.DataArray(data= t_True2, dims=['year','day','lat','lon'],
              coords= {'year':np.linspace(1979,2023,45),'day':np.linspace(1,62,62), 'lat':lat.data , 'lon':lon.data})

ds3 = xr.Dataset(data_vars= dict(heatwave = heatwave))


ds3.to_netcdf("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/1_threshold/T_01_day_JA_79_23.nc")
# ds3.close()
 

##--##--##--  --##--##--##
Intensity_True = Intensity_True * np.array(t_True2)
print("heat Intensity (day)" , np.sum(Intensity_True), np.max(Intensity_True), np.min(Intensity_True))

u_True = u_True * np.array(t_True2)
print("heat u (day)" , np.sum(u_True), np.max(u_True), np.min(u_True))



##--##--##-- heatwave day and intensity (year,lat,lon) --##--##--##
heatwave_day = np.nansum(np.array(t_True2),axis=1)
print(np.shape(heatwave_day), np.mean(heatwave_day),np.max(heatwave_day) , np.min(heatwave_day))

heatwave_intensity = np.nansum(np.array(Intensity_True) ,axis=1)
print(np.shape(heatwave_intensity) ,np.mean(heatwave_intensity), np.max(heatwave_intensity) , np.min(heatwave_intensity))

u300_anom_intensity = np.nansum(np.array(u_True) ,axis=1)
print(np.shape(u300_anom_intensity) ,np.mean(u300_anom_intensity), np.max(u300_anom_intensity) , np.min(u300_anom_intensity))


##--##--##-- save --##--##--##

u300_intensity_array = xr.DataArray(data= u300_anom_intensity, dims=['year','lat','lon'],
              coords= {'year':np.linspace(1979,2023,45), 'lat':lat , 'lon':lon})
print("u300 Intensity (day)" , np.sum(u300_intensity_array), np.max(u300_intensity_array), np.min(u300_intensity_array), np.percentile(u300_intensity_array, 50))

ds2 = xr.Dataset(data_vars= dict(u300_intensity = u300_intensity_array))


ds2.to_netcdf("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/1_threshold/U300_accumulate_inten_JA_79_23.nc")
ds2
ds2.close()


