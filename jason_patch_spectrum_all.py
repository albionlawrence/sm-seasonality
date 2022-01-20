#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jason_patch_pectrum_all.py lat lon size

Generates (windowed) monthly spectra for all tracks in a (size degree)^2
patch sentered at lat N lon E, also saves 
record of which tracks are any good, statistics,
ALSO spatial averages pre-windowing.

Uses Jason-2 data taken from

https://podaac.jpl.nasa.gov/dataset/OSTM_L2_OST_OGDR_GPS

Updated 10-13-21

@author: albionlawrence
"""

import numpy as np
import netCDF4 as nc
import glob
import geopy.distance
import scipy.signal as sig
#import scipy.interpolate as interpolate
import sys
import scipy.io as sio


##Defining the Jason-2 dataset we use
low_cycle = 1 #first cycle of dataset (cycle = full coverage of earth)
high_cycle = 300 #final cycle of dataset
num_passes = 254 #number of passes per cycle
num_cycles = high_cycle - low_cycle + 1

##Input specifying patch
lat_center = int(sys.argv[1])#coordinates of center of patch
lon_center = int(sys.argv[2])
size = int(sys.argv[3])
#size = 8#linear size of patch in degrees
half_size = int(size/2)
#region = 'Gulf'
#pass_number = 126 #track number for single track spectra

#track length 
seg_length_allowed = 160 #number of points we will keep on this track
fft_length = int(seg_length_allowed/2) + 1 #number of independent Fourier components


#Physics parameters
grav = 9.8
f = 2*np.pi*np.sin(2*np.pi*lat_center/360)/(12*3600)

#check if center of box is close to 0 longitude; define 
#new lon_center that is negative if center is up to 8 degrees
#west of 0 longitude
if lon_center > 352:
    lon_center_f = lon_center - 360
else:
    lon_center_f = lon_center

lat_min = lat_center - half_size
lat_max = lat_center + half_size
lon_min = lon_center_f - half_size
lon_max = lon_center_f + half_size

test_cycle = 17#a cycle which doesn't have any missing passes, used to select passes for a patch
pass_list = np.empty(0)
dx_list = np.empty(0)

#figure out which passes actually contribute to the patch at hand, using test cycle
#Also compute interval size
for p in range(1,num_passes+1):
    path = glob.glob('../submesoscale_seasonality_data/jason2_ssh_data/cycle'+str(test_cycle).zfill(3)+'/JA2_GPR_2PdP'+str(test_cycle).zfill(3)+'_'+str(p).zfill(3)+'_*.nc')
    if path == []:
        continue
    pass_data = nc.Dataset(path[0],'r')
    lat = pass_data.variables['lat'][:]
    lon = pass_data.variables['lon'][:]
    indx = np.nonzero((lat >= lat_min) & (lat <= lat_max))
    seg_length = len(indx[0][:])
    #make sure track is long enough
    if seg_length <= seg_length_allowed:
        continue    
    indx_seg = indx[0][0:seg_length_allowed]
    if (lon_center_f < 8) and (lon_center_f > - 8):
        indx_high = np.where(lon > 180)[0]
        lon[indx_high] -= 360
    if (((lon[indx[0][0]] >= lon_min) & (lon[indx[0][0]] <= lon_max)) or ((lon[indx[0][seg_length_allowed - 1]] >= lon_min) & (lon[indx[0][seg_length_allowed - 1]] <= lon_max))):       
        lon_seg = lon[indx_seg]
        lat_seg = lat[indx_seg]
        if len(np.where(lon_seg > lon_max)[0]) > seg_length_allowed/2 or len(np.where(lon_seg < lon_min)[0]) > seg_length_allowed/2:
            continue
        pass_list = np.append(pass_list,p)
        dx_list = np.append(dx_list,0.)
        for seg_pt in range(seg_length_allowed):
            dx_list[-1] += geopy.distance.distance((lat[indx_seg[seg_pt]],lon[indx_seg[seg_pt]]), (lat[indx_seg[seg_pt]-1],lon[indx_seg[seg_pt]-1])).km/seg_length_allowed

dx_seg = np.mean(dx_list)
pass_list = pass_list.astype(int)
num_passes = len(pass_list)       

num_segs = np.zeros((12,num_passes)) #counter for number of segments accepted, CX's criterion
num_empty_pass = np.zeros((num_passes)) #counter for passes that are simply missing
#path_means = np.zeros((12,num_passes))
ssh_array = np.empty((num_passes,num_cycles,seg_length_allowed))
ssh_array[:] = np.nan
constant_mode_array = np.empty((num_passes,num_cycles))
constant_mode_array[:] = np.nan
num_skipped = 0 #counter for number of skipped cycles in a track; used to do interpolation.
good_cycles = np.zeros((num_passes,num_cycles))
#good_cycles[:] = np.nan
month = np.empty((num_passes,num_cycles))
month[:] = np.nan
bath_tracks = np.zeros((num_passes,seg_length_allowed))

#filter out passes with missing data
#
for k,p in enumerate(pass_list):
    for cycle in range(low_cycle,high_cycle+1): 
        path = glob.glob('../submesoscale_seasonality_data/jason2_ssh_data/cycle'+str(cycle).zfill(3)+'/JA2_GPR_2PdP'+str(cycle).zfill(3)+'_'+str(p).zfill(3)+'_*.nc')
        if path == []:
            #num_skipped += 1
            continue
        month[k,cycle-low_cycle] = int(path[0][-30:-28]) - 1 #month of start of pass, converted to index
        if np.isnan(month[k,cycle-low_cycle]) == True:
            print(p.cycle)
        pass_data = nc.Dataset(path[0],'r')
        lat = pass_data.variables['lat'][:]
        lon = pass_data.variables['lon'][:]
        bath = pass_data.variables['bathymetry'][:]
        indx = np.nonzero((lat >= lat_min) & (lat <= lat_max))
        seg_length = len(indx[0][:])
        if seg_length < seg_length_allowed:
            #num_skipped +=1
            continue
        indx_seg = indx[0][0:seg_length_allowed]
        lon_seg = lon[indx_seg]
        lat_seg = lat[indx_seg]
        #check segment is mostly inside the box
        if len(np.where(lon_seg > lon_center + half_size)[0]) > seg_length_allowed/2 or len(np.where(lon_seg < lon_center - half_size)[0]) > seg_length_allowed/2:
            #num_skipped += 1
            continue
        ssha = pass_data.variables['ssha'][:]
        ssh_seg = ssha[indx_seg]
        #skip if missing data
        if (np.ma.is_masked(ssh_seg) == True):
            #num_skipped +=1
            continue
        num_segs[int(month[k,cycle-low_cycle]),k] += 1
        ssh_array[k,cycle - low_cycle,:] = ssh_seg
        good_cycles[k,cycle - low_cycle] = 1.
        bath_seg = bath[indx_seg]
        bath_tracks[k,:] = bath_seg
        

window_seg = np.sqrt(8./3.)*np.sin(np.pi*np.arange(seg_length_allowed)/seg_length_allowed)**2
#Hanning window for Fourier analysis
fft_len = int(seg_length_allowed/2) + 1
power_spec_jason = np.zeros((num_passes,num_cycles,fft_len))
dk_seg = 1/(2*dx_seg*(fft_len))
 
#bath = np.zeros((num_good_passes,))

#Compute power spectrum for each track, and constant mode.
for k,p in enumerate(pass_list):
    #pass_array = ssh_array_good[k,:,:]
    cycle_indx = np.nonzero(good_cycles[k,:])[0]#here we will look at all acceptable cycles
    for n,cycle in enumerate(cycle_indx):
        constant_mode_array[k,cycle] = np.mean(ssh_array[k,cycle,:])
        ssh_array[k,cycle,:] = sig.detrend(ssh_array[k,cycle,:],type='constant')
        jason_fft = np.fft.rfft(window_seg*ssh_array[k,cycle,:])
        power_spec_jason[k,cycle,:] = dx_seg*np.absolute(jason_fft)**2/seg_length_allowed
        
sio.savemat("../submesoscale_seasonality_data/patch_spectra/jason_lat{:n}_lon{:n}_{:n}deg_unfiltered.mat".format(lat_center,lon_center,size), {'high_cycle': high_cycle, 'low_cycle': low_cycle, 'dx': dx_seg,'ps_all': power_spec_jason, 'num_segs': num_segs, 'good_cycles': good_cycles, 'pass_list': pass_list, 'month_list':month, 'bath_tracks': bath_tracks,'constant_mode_array':constant_mode_array})
