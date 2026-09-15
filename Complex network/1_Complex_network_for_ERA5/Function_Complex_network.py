import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl
mpl.use("Agg")


def computing_link_bwt_two_array(array1, array2,years_num,lat,lon,outfile):
    """Computing the links between two metrix.

    Args:
        array1 (bool): climatological data 1;
        array2 (bool): climatological data 2;
        years_num (num): total year number;
        lat (DataArray): latitude information;
        lon (DataArray): longitude information;
        outfile (str): files need to be stored.
    
    Note: here we only calculate from 1979; 
    due to the limited calculation memory, we have to store each year seperately.
    
    """

    for iyear in range(years_num):
        network=np.zeros((np.shape(lat)[0],np.shape(lon)[0],np.shape(lat)[0],np.shape(lon)[0]),dtype=np.float32)
        print(" iyear = " + str(iyear))
        Metrix1=array1[iyear,:,:,:]
        Metrix2=array2[iyear,:,:,:]
        
        
        #for each grid 
        for ilon in range(np.shape(lon)[0]):
            print(" ilon = " + str(ilon))
            for ilat in range(np.shape(lat)[0]): 
                print(" ilat = " + str(ilat))
                # if np.isnan(Metrix1[:,:,ilat,ilon]).all():
                #     break
                # else:
                Metrix1_3D=np.tile( Metrix1[:,ilat,ilon],(np.shape(lat)[0],np.shape(lon)[0],1)).transpose(2,0,1)
                concurrent=np.sum(Metrix2[:,:,:]+ Metrix1_3D[:,:,:]==2, axis=0)
                network[ilat,ilon,:,:] = concurrent 
        
        ##--##--##-- save --##--##--##
        network = xr.DataArray(data= network, dims=['lat1','lon1','lat2','lon2'],
                    coords= {'lat1':lat.data , 'lon1':lon.data, 'lat2':lat.data , 'lon2':lon.data})

        ds3 = xr.Dataset(data_vars= dict(network = network))
        ds3.to_netcdf(outfile+f"concurrent_day_"+str(iyear+1979)+".nc")
        ds3.close()
        del network


def significant_links_at_each_grid(array1, array2,lat,lon,Test_file,outfile):
    """We have a threshold sheet, now we use this table to calculate the threshold for each grid cell.
    
    Args:
        array1 (bool): climatological data 1;
        array2 (bool): climatological data 2;
        lat (DataArray): latitude information;
        lon (DataArray): longitude information;
        outfile (str): files need to be stored.
    
    """
    ## ------------- 1. total days for each grid point
    # heatwave day for each grid
    array1_days =  np.sum(array1[:,:,:,:], axis=(0,1))
    array1_days = np.where(array1_days>1000, 999,array1_days)
    print(int(np.nanmax(array1_days)))

    # westerly event for each grid
    array2_days = np.sum(array2[:,:,:,:], axis=(0,1))
    array2_days = np.where(array2_days>1000, 999,array2_days)
    print(int(np.nanmax(array2_days)))

    ## -------------  2. construct network   

    threshold_values = np.zeros((np.shape(lat)[0],np.shape(lon)[0],np.shape(lat)[0],np.shape(lon)[0]))

    # load threshold sheet
    TEST = xr.open_dataset(Test_file)
    matrix99 = TEST.matrix99

    #  （ilat1,ilon1, ilat2,ilon2）
    for ilat in range(np.shape(lat)[0]):
        print(f"ilat{ilat}")
        for ilon in range(np.shape(lon)[0]):
            print(f"ilon{ilon}")
            

            # 
            hw_event_days = int(array1_days[ilat, ilon])
            
            for b_days in np.unique(array2_days.astype(int)):
                concurrent_days = matrix99[hw_event_days,b_days]
                threshold_values[ilat,ilon,:,:][np.where(array2_days==b_days)] =concurrent_days
                del concurrent_days
            del hw_event_days
            
 

    ##--##--##--##--  simultaneous network, save nc  --##--##--##--##
    matrix4D_99_array0 = xr.DataArray(data=threshold_values, dims=['lat1', 'lon1', 'lat2', 'lon2'],
                                    coords={'lat1':lat.data,'lon1':lon.data,
                                            'lat2':lat.data,'lon2':lon.data})
    ds0 = xr.Dataset(data_vars=dict(matrix4D_99=matrix4D_99_array0))
    ds0.to_netcdf(outfile)
    ds0.close()


# def convert_lon(ds0):
#     """Convert 0-360 to -180-180

#     Args:
#         ds0 (str): the variable representing the whole data file
#     """
#     for lon_name in ['lon1', 'lon2']:
#         ds0['longitude_adjusted'] = xr.where(ds0[lon_name] > 180, ds0[lon_name] - 360, ds0[lon_name])
#         ds0 = ds0.swap_dims({lon_name: 'longitude_adjusted'}).sel(
#             **{'longitude_adjusted': sorted(ds0.longitude_adjusted)}
#         ).drop(lon_name)
#         ds0 = ds0.rename({'longitude_adjusted': lon_name})
    
#     return(ds0)

    




def get_significant_links(infile,significant_point_eachgrid_file,lat,lon,years_num,outfile):
    """Calculate the significant links based on threshold sheet.

    Args:

        infile (str): files storing the links between two arrays;
        significant_point_eachgrid_file (str): Threshold table for each grid point
        lat (DataArray): latitude information;
        lon (DataArray): longitude information;
        years_num (num): total year number;
        outfile (str): files need to be stored.
    
    """

    ##--##--##--##--  construct network   --##--##--##--##
    concurrent_day=np.zeros((years_num,np.shape(lat)[0],np.shape(lon)[0],np.shape(lat)[0],np.shape(lon)[0]))
    for iyear in range(years_num):
        print('iyear = ', iyear + 1979)
        ds0 = xr.open_dataset(infile+f"/concurrent_day_{iyear+1979}.nc")
       
        for lon_name in ['lon1', 'lon2']:
            ds0['longitude_adjusted'] = xr.where(ds0[lon_name] > 180, ds0[lon_name] - 360, ds0[lon_name])
            ds0 = ds0.swap_dims({lon_name: 'longitude_adjusted'}).sel(
                **{'longitude_adjusted': sorted(ds0.longitude_adjusted)}
            ).drop(lon_name)
            ds0 = ds0.rename({'longitude_adjusted': lon_name})
        
        network1  = ds0.network
        lat = ds0.lat1.loc[-21:89]; lon = ds0.lon1
        concurrent_day[iyear,:,:,:,:]= network1  

    
    All_year_concurr= np.sum(concurrent_day,axis=0)
    print(np.shape(All_year_concurr))

    # # 
    TEST = xr.open_dataset(significant_point_eachgrid_file)
    for lon_name in ['lon1', 'lon2']:
        TEST['longitude_adjusted'] = xr.where(TEST[lon_name] > 180, TEST[lon_name] - 360, TEST[lon_name])
        TEST = TEST.swap_dims({lon_name: 'longitude_adjusted'}).sel(
            **{'longitude_adjusted': sorted(TEST.longitude_adjusted)}
        ).drop(lon_name)
        TEST = TEST.rename({'longitude_adjusted': lon_name})

    matrix99 = TEST.matrix4D_99
    All_year_concurr[:,:,:,:]  = np.where(All_year_concurr[:,:,:,:] < matrix99[:,:,:,:], 0, All_year_concurr[:,:,:,:] )     
    matrix4D_99_array0 = xr.DataArray(data=All_year_concurr, dims=['lat1', 'lon1', 'lat2', 'lon2'],
                                coords={'lat1':lat.data,'lon1':lon.data,
                                        'lat2':lat.data,'lon2':lon.data})
    ds1 = xr.Dataset(data_vars=dict(significant_99=matrix4D_99_array0))
    ds1.to_netcdf(outfile)
    ds1.close()
