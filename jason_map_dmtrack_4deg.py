"""
jason_map_dmtrack_4deg.py

Oct 11 2020

Produces power spectra of 8 degree boxes for points on (4 deg)^2 grid
Tracks in each patch selected according to protocol in Callies and Wu
https://journals.ametsoc.org/view/journals/phoc/49/9/jpo-d-18-0272.1.xml
Includes treatment of wraparound issues ner lon = 0.
Each pass demeaned for every cycle.

Author: Albion Lawrence

"""

import numpy as np
from netCDF4 import Dataset#, num2date
import glob
import geopy.distance
import scipy.signal as sig
#import sys
#import os

#geophysical parameters
M2 = 2*np.pi/(12.42*3600)

low_cycle = 1
high_cycle = 300
num_passes = 254
num_empty_pass = 0
test_cycle = 17
num_cycles = high_cycle - low_cycle + 1

#load climatological data
#speed of sound for low-lying linear modes (for tidal wavenumbers)
cn_global = np.load('../submesoscale_seasonality_data/vertical_modes/cn_monthly_4deg.npy',encoding='latin1',allow_pickle=True)
cn_annual = np.mean(cn_global,axis=2)
#global climatology
mdt_aviso = Dataset('../submesoscale_seasonality_data/AVISO/mdt_cnes_cls18_global.nc')
lat_aviso = mdt_aviso['latitude'][:].data #1d array
lon_aviso = mdt_aviso['longitude'][:].data
uga = mdt_aviso['u'][0,:,:].data #indices are [lat,lon]
vga = mdt_aviso['v'][0,:,:].data
tga = np.sqrt(uga**2 + vga**2) #total geostrophic velocity (sqrt of KE)

lat_ranges = np.arange(-60,64,4) #31-element array of latitudes (defining edges of 30 bands) in units of 4
seg_length_allowed = 160
window = np.sqrt(8./3.)*np.sin(np.pi*np.arange(seg_length_allowed)/seg_length_allowed)**2
fft_len = int(seg_length_allowed/2) + 1

power_spec_jason = np.zeros((30,90,12,fft_len))
power_spec_jason[:] = np.nan
num_segs_map = np.zeros((30,90,12))
dx_global = np.zeros((30,90))#eventually, average track length in km
dx_global[:] = np.nan
num_good_passes = np.zeros((30,90))
geovel_aviso = np.zeros((3,30,90)) #0: mean, 1:median, 2:max
geovel_aviso[:] = np.nan


for lat_band in range(30):
#for lat_band in range(2,3):
    lat_min = lat_ranges[lat_band] - 2
    lat_max = lat_ranges[lat_band + 1] + 2
    #note we are moving in units of 4 but want 8 degree squares, so 
    #the range extends 2 degrees past each band edge
    for lon_band in range(90):
        #print(lat_band,lon_band)
    #for lon_band in range(12,13):
        #don't bother if patch doesn't satisfy same criteria used to compute c_n (too shallow etc)
        if np.ma.is_masked(cn_annual[lon_band,lat_band,:] == True):
            print(lat_band,lon_band)
            continue        
        lon_min = lon_band*4 - 2
        lon_max = (lon_band+1)*4 + 2
        pass_list = np.empty(0)
        dx_cx_list = np.empty(0)
        
        #print(lat_min,lon_min,'start',pass_list,dx_cx_list)
        #figure out which passes actually contribute to the patch at hand.
        #Also compute interval size
        for p in range(1,num_passes+1):
            path = glob.glob('../submesoscale_seasonality_data/jason2_ssh_data/cycle'+str(test_cycle).zfill(3)+'/JA2_GPR_2PdP'+str(test_cycle).zfill(3)+'_'+str(p).zfill(3)+'_*.nc')
            if path == []:
                continue
            pass_data = Dataset(path[0],'r')
            lat = pass_data.variables['lat'][:]
            lon = pass_data.variables['lon'][:]
            bath = pass_data.variables['bathymetry'][:]
            if lon_band == 0:
                indx_high = np.where(lon >= 358)
                lon[indx_high] -= 360
            if lon_band == 89:
                indx_lo = np.where(lon <= 2)
                lon[indx_lo] += 360
            indx = np.nonzero((lat >= lat_min) & (lat <= lat_max))
            seg_length = len(indx[0])
            #make sure track is long enough
            if seg_length <= seg_length_allowed:
                continue    
            if (((lon[indx[0][0]] >= lon_min) & (lon[indx[0][0]] <= lon_max)) or ((lon[indx[0][seg_length_allowed - 1]] >= lon_min) & (lon[indx[0][seg_length_allowed - 1]] <= lon_max))):
                #print(lat_band,lon_band,'test')
                indx_cx = indx[0][0:seg_length_allowed]
                lon_seg = lon[indx_cx]
                lat_seg = lat[indx_cx]
                bath_seg = bath[indx_cx]
                if np.max(bath_seg) > -1000:
                    continue
                if len(np.where(lon_seg > lon_max)[0]) > seg_length_allowed/2 or len(np.where(lon_seg < lon_min)[0]) > seg_length_allowed/2:
                    continue
                #print(lat_band,lon_band,p,'test')
                pass_list = np.append(pass_list,p)
                dx_cx_list = np.append(dx_cx_list,0.)
                for seg_pt in range(seg_length_allowed):
                    dx_cx_list[-1] += geopy.distance.distance((lat[indx_cx[seg_pt]],lon[indx_cx[seg_pt]]), (lat[indx_cx[seg_pt]-1],lon[indx_cx[seg_pt]-1])).km/seg_length_allowed
            #print('test',lat_band,lon_band,p)
                    
        if pass_list.shape[0] == 0:
            continue
        dx_global[lat_band,lon_band] = np.mean(dx_cx_list)
        #print(dx_global[lat_band,lon_band])
        pass_list = pass_list.astype(int)
        num_accepted_passes = len(pass_list)        
        num_segs_cx = np.zeros((12,num_accepted_passes)) #counter for number of segments accepted, CX's criterion
        
        ssh_array = np.empty((num_passes,num_cycles,seg_length_allowed))
        ssh_array[:] = np.nan
        good_cycles = np.zeros((num_passes,num_cycles))
        month = np.empty((num_passes,num_cycles))
        month[:] = np.nan
        bath_tracks = np.zeros((num_passes,seg_length_allowed))
        
        #Filter out passes with missing data
        for cycle in range(low_cycle,high_cycle+1):
            for k,p in enumerate(pass_list):
                path = glob.glob('../submesoscale_seasonality_data/jason2_ssh_data/cycle'+str(cycle).zfill(3)+'/JA2_GPR_2PdP'+str(cycle).zfill(3)+'_'+str(p).zfill(3)+'_*.nc')
                if path == []:
                    continue
                month[k,cycle-low_cycle] = int(path[0][-30:-28]) - 1
                #month of start of pass, converted to index
                pass_data = Dataset(path[0],'r')
                lat = pass_data.variables['lat'][:]
                lon = pass_data.variables['lon'][:]
                bath = pass_data.variables['bathymetry'][:]
                if lon_band == 0:
                    indx_high = np.where(lon >= 358)
                    lon[indx_high] -= 360
                if lon_band == 89:
                    indx_lo = np.where(lon <= 2)
                    lon[indx_lo] += 360
                indx = np.nonzero((lat >= lat_min) & (lat <= lat_max))
                seg_length = len(indx[0][:])
                if seg_length < seg_length_allowed:
                    continue
                indx_cx = indx[0][0:seg_length_allowed]
                lon_seg = lon[indx_cx]
                lat_seg = lat[indx_cx]
                #check segment is mostly inside the box
                if len(np.where(lon_seg > lon_max)[0]) > seg_length_allowed/2 or len(np.where(lon_seg < lon_min)[0]) > seg_length_allowed/2:
                    continue
                ssha = pass_data.variables['ssha'][:]
                ssh_seg = ssha[indx_cx]
                #skip if missing data
                if (np.ma.is_masked(ssh_seg) == True):
                    continue
                #print(lat_band,lon_band,cycle,p,'enuf')
                num_segs_cx[int(month[k,cycle-low_cycle]),k] += 1
                ssh_array[k,cycle - low_cycle,:] = ssh_seg
                good_cycles[k,cycle - low_cycle] = 1.
                bath_seg = bath[indx_cx]
                bath_tracks[k,:] = bath_seg

        good_passes = np.where(np.amin(num_segs_cx,axis=0) > 4)[0]
        #print(lat_band,lon_band,np.amin(num_segs_cx,axis=0))
        good_passes = np.unique(good_passes) 
        pass_good = pass_list[good_passes]
        #print(lat_band,lon_band,pass_good)
        pass_reject = np.delete(pass_list,good_passes)
        month_good = month[good_passes,:]
        bath_good = bath_tracks[good_passes,:]

        num_good_segs = num_segs_cx[:,good_passes]
        #num_segs_total = np.sum(num_segs_cx[:,good_passes],axis=1)
        num_segs_map[lat_band,lon_band,:] = np.sum(num_segs_cx[:,good_passes],axis=1)
        
        ssh_array_good = ssh_array[good_passes,:,:]
        good_cycles_good_passes = good_cycles[good_passes,:]
        num_good_passes[lat_band,lon_band] = len(good_passes)
        
        power_spec_box = np.zeros((12,len(pass_good),fft_len))
        dk_cx = 1/(2*dx_global[lat_band,lon_band]*fft_len)
        #print(dk_cx)
        
        for k,p in enumerate(pass_good):
            cycle_indx = np.nonzero(good_cycles_good_passes[k,:])[0]
            for n,cycle in enumerate(cycle_indx):
                ssh_array_good[k,cycle,:] = sig.detrend(ssh_array_good[k,cycle,:],type='constant')
                jason_fft = np.fft.rfft(window*ssh_array_good[k,cycle,:])
                power_spec_box[int(month_good[k,cycle]),k,:] += dx_global[lat_band,lon_band]*np.absolute(jason_fft)**2/seg_length_allowed
        
        power_spec_jason[lat_band,lon_band,:,:] = 2*np.sum(power_spec_box,axis=1)/np.sum(num_good_segs,axis=1)[:,np.newaxis]
        #factor of 2 for 1-sided PSD. See page 447 of Thomson and Emery.
        #power_spec_jason[lat_band,lon_band,:,:] = 2*np.mean(power_spec_box/num_good_segs[:,:,np.newaxis],axis=1)
        
        #Finally, compute characteristics of geostrophic velocity inside patch
        #first again be careful if in initial or final patch
        if lon_band == 0:
            indx_hi_aviso = np.where(lon_aviso >= 358)
            lon_aviso[indx_hi_aviso] -= 360
        if lon_band == 89:
            indx_lo_aviso = np.where(lon_aviso <= 2)
            lon_aviso[indx_lo_aviso] += 360
        lat_indx_aviso = np.nonzero((lat_aviso >= lat_min) & (lat_aviso <= lat_max))[0]
        lon_indx_aviso = np.nonzero((lon_aviso >= lon_min) & (lon_aviso <= lon_max))[0]
        tga_square = tga[lat_indx_aviso[0]:(lat_indx_aviso[-1]+1),lon_indx_aviso[0]:(lon_indx_aviso[-1]+1)]
        geovel_aviso[0,lat_band,lon_band] = np.mean(tga_square)
        geovel_aviso[1,lat_band,lon_band] = np.median(tga_square)
        geovel_aviso[2,lat_band,lon_band] = np.max(tga_square)
        
        
        print('completed',lat_band,lon_band)
        
power_spec_jason.dump('powerspectdm_4deg.npy')
dx_global.dump('dx_4deg.npy')
num_segs_map.dump('num_segs_4deg.npy')
num_good_passes.dump('num_good_passes_4deg.npy')
geovel_aviso.dump('geovel_aviso_4deg.npy')
