#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 12:27:19 2021

seasonality_maps_4deg.py

global maps of phases of spectral slope and bandpassed KE
(as well as of errors in phase) with respect to MLD.
Uses output from monthly_speed_4deg.py,
jason_map_dmtrack_4deg.py, jason_reduction_4deg.py

@author: albionlawrence, modifications by joernc
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
#import scipy.stats as stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.optimize import least_squares
from scipy.stats import chi2

powerspec_map = np.load('../submesoscale_seasonality_data/global_maps/powerspectdm_4deg.npy',encoding='latin1',allow_pickle=True)[:,:,:,1:]
param_maps = np.load('../submesoscale_seasonality_data/global_maps/jason_map_parameters_4deg.npy',encoding='latin1',allow_pickle="True")
error_params = np.load('../submesoscale_seasonality_data/global_maps/error_map_4deg.npy',encoding='latin1',allow_pickle=True)
ann_param_map = np.load('../submesoscale_seasonality_data/global_maps/jason_ann_parameters_4deg.npy',encoding='latin1',allow_pickle=True)
num_segs = np.load('../submesoscale_seasonality_data/global_maps/num_segs_4deg.npy',encoding='latin1',allow_pickle=True)
energy_maps = np.load('../submesoscale_seasonality_data/global_maps/fitted_energy_4deg.npy',encoding='latin1',allow_pickle=True)
energy_err = np.load('../submesoscale_seasonality_data/global_maps/fitted_energy_err_4deg.npy',encoding='latin1',allow_pickle=True)
ann_segs = np.sum(num_segs,axis=2)
powerspec_map_ann = np.sum(powerspec_map*num_segs[:,:,:,np.newaxis],axis=2)/ann_segs[:,:,np.newaxis]

dx_global = np.load('../submesoscale_seasonality_data/global_maps/dx_4deg.npy',encoding='latin1',allow_pickle=True)
mld_map= np.load('../submesoscale_seasonality_data/global_maps/mld_map_4deg.npy',encoding='latin1',allow_pickle=True)
mld_map_ann = np.mean(mld_map,axis=2)
tides_yes = np.load('../submesoscale_seasonality_data/global_maps/tides_yes_4deg.npy',encoding='latin1',allow_pickle=True)
geovel_map = np.load('../submesoscale_seasonality_data/global_maps/geovel_aviso_4deg.npy',encoding='latin1',allow_pickle=True)

cn_global = np.load('../submesoscale_seasonality_data/vertical_modes/cn_monthly_4deg.npy',encoding='latin1',allow_pickle=True)
cn_annual = np.mean(cn_global,axis=2)
c0_annual = np.transpose(cn_annual[:,:,0])

seg_length_allowed = 160
fft_len = int(seg_length_allowed/2) + 1
nbins = int(seg_length_allowed/2)

lat_ranges = np.arange(-60,64,4)
#M2 = 2*np.pi/(12.42*3600)
grav = 9.8

KE1_map = np.empty((30,90,12))
KE1_amp_map = np.empty((30,90))
KE1_phase_map = np.empty((30,90))
KE2_map = np.empty((30,90,12))
KE2_amp_map = np.empty((30,90))
KE2_phase_map = np.empty((30,90))
noise_map = np.empty((30,90,12))
noise_amp_map = np.empty((30,90))
mld_phase2_map = np.empty((30,90))
KE1_map[:] = np.nan
KE1_amp_map[:] = np.nan
KE1_phase_map[:] = np.nan
KE2_map[:] = np.nan
KE2_amp_map[:] = np.nan
KE2_phase_map[:] = np.nan
noise_map[:] = np.nan
noise_amp_map[:] = np.nan

KE2_annav_map = np.empty((30,90))
noise_annav_map = np.empty((30,90))
KE2_vs_noise = np.empty((30,90))
k0_vs_noise = np.empty((30,90))
KE2_annav_map[:] = np.nan
noise_annav_map[:] = np.nan
KE2_vs_noise[:] = np.nan
k0_vs_noise[:] = np.nan
mld_phase2_map[:] = np.nan


for lat_band in range(30):
    for lon_band in range(90):
        if np.isnan(ann_param_map[lat_band,lon_band,0]) == True:
            continue
        lat_center = lat_ranges[lat_band] + 2
        f = 2*np.pi*np.sin(2*np.pi*lat_center/360)/(12*3600)
        #M2 = 
        k_n = 1/(2*dx_global[lat_band,lon_band])
        dk = k_n/fft_len
        resm = k_n/(nbins*1000)
        wn = np.arange(1,fft_len)/(2*dx_global[lat_band,lon_band]*(fft_len-1))
        k0_ann = ann_param_map[lat_band,lon_band,1]
        k0ann_bin = int(k0_ann*nbins/k_n) - 1
        k1_ann = 2*k0_ann #don't confuse with k1, k2 which are todal wavenumbers; these are band edges
        k1ann_bin = int(k1_ann*(nbins)/k_n) - 1
        k2_ann = 2*k1_ann
        k2ann_bin = int(k2_ann*(nbins)/k_n) - 1
        #ke_subtracted = (powerspec_map[lat_band,lon_band,:,:] - param_maps[lat_band,lon_band,:,3,np.newaxis])*(wn[np.newaxis,:]/1000)**2*grav**2/f**2
        noise_ke = param_maps[lat_band,lon_band,:,np.newaxis,3]*(wn[np.newaxis,:]/1000)**2*grav**2/f**2
        #KE1_map[lat_band,lon_band,:] = np.sum(ke_subtracted[:,k0ann_bin:k1ann_bin],axis=1)*resm
        #KE2_map[lat_band,lon_band,:] = np.sum(ke_subtracted[:,k1ann_bin:k2ann_bin],axis=1)*resm
        noise_map[lat_band,lon_band,:] = np.sum(noise_ke[:,k1ann_bin:k2ann_bin],axis=1)*resm
        ke_ann_subtracted = (powerspec_map_ann[lat_band,lon_band,:] - ann_param_map[lat_band,lon_band,3])*(wn/1000)**2*grav**2/f**2
        noise_ke = ann_param_map[lat_band,lon_band,3]*(wn/1000)**2*grav**2/f**2
        KE2_annav_map[lat_band,lon_band] = np.sum(ke_ann_subtracted[k1ann_bin:k2ann_bin])*resm
        noise_annav_map[lat_band,lon_band] = np.sum(noise_ke[k1ann_bin:k2ann_bin])*resm
        KE2_vs_noise[lat_band,lon_band] = KE2_annav_map[lat_band,lon_band]/noise_annav_map[lat_band,lon_band]
        k0_vs_noise[lat_band,lon_band] = (powerspec_map_ann[lat_band,lon_band,k0ann_bin]/ann_param_map[lat_band,lon_band,3]) - 1
        
sinusoid = np.exp(2j*np.pi*np.arange(12)/12)
KE1_ann_map = np.sum(energy_maps[:,:,:,1]*sinusoid[np.newaxis,np.newaxis,:],axis=2)
KE2_ann_map = np.sum(energy_maps[:,:,:,2]*sinusoid[np.newaxis,np.newaxis,:],axis=2)
KE1_amp_map = np.abs(KE1_ann_map)
KE1_phase_map = np.angle(KE1_ann_map)
KE2_amp_map = np.abs(KE2_ann_map)
KE2_phase_map = np.angle(KE2_ann_map)

mld_mode1_map = np.sum(mld_map*sinusoid[np.newaxis,np.newaxis,:],axis=2)/6
mld_phase1_map = 12*np.angle(mld_mode1_map)/(2*np.pi) % 12
mld_phase2_map = np.angle(mld_mode1_map)
#mld_phase2_map[:15,:] = np.pi
#mld_phase2_map[15:,:] = 0.
mld_amp_map = np.abs(mld_mode1_map)

# KE1_phase_map[0:15,:] = 12*(KE1_phase_map[:15,:] + np.pi)/(2*np.pi) % 12
# KE1_phase_map[15:,:] = 12*(KE1_phase_map[15:,:])/(2*np.pi) % 12
# KE2_phase_map[0:15,:] = 12*(KE2_phase_map[:15,:] + np.pi)/(2*np.pi) % 12
# KE2_phase_map[15:,:] = 12*(KE2_phase_map[15:,:])/(2*np.pi) % 12

KE1_phase_map = 12*(KE1_phase_map - mld_phase2_map)/(2*np.pi) % 12
KE2_phase_map = 12*(KE2_phase_map - mld_phase2_map)/(2*np.pi) % 12

amp_mode1_map = np.sum(param_maps[:,:,:,0]*sinusoid[np.newaxis,np.newaxis,:],axis=2)
amp_amp1_map = np.abs(amp_mode1_map)
amp_phase1_map = 12*np.angle(amp_mode1_map)/(2*np.pi) % 12

k0_mode1_map = np.sum(param_maps[:,:,:,1]*sinusoid[np.newaxis,np.newaxis,:],axis=2)
k0_amp1_map = np.abs(k0_mode1_map)
k0_phase1_map = 12*np.angle(k0_mode1_map)/(2*np.pi) % 12


slope_mode1_map = np.sum(param_maps[:,:,:,2]*sinusoid[np.newaxis,np.newaxis,:],axis=2)
slope_amp1_map = np.abs(slope_mode1_map)
slope_phase1_map = np.angle(slope_mode1_map) + np.pi % 2*np.pi
# slope_phase1_map[:15,:] = 12*(slope_phase1_map[:15,:])/(2*np.pi) % 12
# slope_phase1_map[15:,:] = 12*(slope_phase1_map[15:,:] + np.pi)/(2*np.pi) % 12
slope_phase1_map = 12*(slope_phase1_map - mld_phase2_map)/(2*np.pi) % 12
slope_amp_err = np.sqrt(2*np.sum(np.cos(2*np.pi*np.arange(12)[np.newaxis,np.newaxis,:]/12 - slope_phase1_map[:,:,np.newaxis])**2*error_params[:,:,:,2]**2,axis=2))
slope_phase_err = 12*(slope_amp_err/slope_amp1_map)/(2*np.pi) % 12

noise_mode1_map = np.sum(param_maps[:,:,:,3]*sinusoid[np.newaxis,np.newaxis,:],axis=2)
noise_amp1_map = np.abs(noise_mode1_map)
noise_phase1_map = np.angle(noise_mode1_map)
# noise_phase1_map[:15,:] = 12*(noise_phase1_map[:15,:] + np.pi)/(2*np.pi) % 12
# noise_phase1_map[15:,:] = 12*(noise_phase1_map[15:,:])/(2*np.pi) % 12
noise_phase1_map = 12*(noise_phase1_map - mld_phase2_map)/(2*np.pi) % 12


spike = mld_amp_map > 400
mld_amp_map[spike] = np.nan
mld_phase1_map[spike] = np.nan

# ampspike = amp_amp1_map > 10000
# amp_amp1_map[ampspike] = np.nan
# amp_phase1_map[ampspike] = np.nan

def no_annual(x):
    """
    Cosine series for monthly signal without annual component
    """
    return x[0] + x[1]*np.cos(4*np.pi*np.arange(12)/12 + x[6])/6 + x[2]*np.cos(6*np.pi*np.arange(12)/12 + x[7])/6 + x[3]*np.cos(8*np.pi*np.arange(12)/12 + x[8])/6 + x[4]*np.cos(10*np.pi*np.arange(12)/12 + x[9])/6 + x[5]*np.cos(12*np.pi*np.arange(12)/12)/6

def resid_noann(x,data,err):
    """
    residual for monthly series fit without annual component
    """
    return (data - no_annual(x))/err

likelihood_KE2 = np.empty((30,90))
likelihood_KE2[:] = np.nan
likelihood_KE2_yes = np.empty((30,90))
likelihood_KE2_yes[:] = np.nan
likelihood_KE1 = np.empty((30,90))
likelihood_KE1[:] = np.nan
likelihood_KE1_yes = np.empty((30,90))
likelihood_KE1_yes[:] = np.nan
likelihood_amp = np.empty((30,90))
likelihood_amp[:] = np.nan
likelihood_amp_yes = np.empty((30,90))
likelihood_amp_yes[:] = np.nan
likelihood_slope = np.empty((30,90))
likelihood_slope[:] = np.nan
likelihood_slope_yes = np.empty((30,90))
likelihood_slope_yes[:] = np.nan
likelihood_noise = np.empty((30,90))
likelihood_noise[:] = np.nan
likelihood_noise_yes = np.empty((30,90))
likelihood_noise_yes[:] = np.nan

CL95 = chi2.interval(0.95,2)[1]

#map out likelihood for KE2
for lat_band in range(30):
    for lon_band in range(90):
        if (np.isnan(KE2_amp_map[lat_band,lon_band]) == True) or (geovel_map[2,lat_band,lon_band] > 1000):
            continue
        KE2_monthly = energy_maps[lat_band,lon_band,:,2]
        KE2_err_monthly = energy_err[lat_band,lon_band,:,2]
        KE2NA_init = np.array([np.mean(KE2_monthly), 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        resd_KE2 = least_squares(resid_noann,KE2NA_init,args=(KE2_monthly,KE2_err_monthly))
        KE2NA_params = resd_KE2.x
        likelihood_KE2[lat_band,lon_band] = 2*np.sum(resid_noann(KE2NA_params,KE2_monthly,KE2_err_monthly)**2)
        if likelihood_KE2[lat_band,lon_band] > CL95:
            likelihood_KE2_yes[lat_band,lon_band] = 2.
        else:
            likelihood_KE2_yes[lat_band,lon_band] = 1.

#map out likelihood for KE1
for lat_band in range(30):
    for lon_band in range(90):
        if (np.isnan(KE1_amp_map[lat_band,lon_band]) == True) or (geovel_map[2,lat_band,lon_band] > 1000):
            continue
        KE1_monthly = energy_maps[lat_band,lon_band,:,1]
        KE1_err_monthly = energy_err[lat_band,lon_band,:,1]
        KE1NA_init = np.array([np.mean(KE2_monthly), 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        resd_KE1 = least_squares(resid_noann,KE2NA_init,args=(KE1_monthly,KE1_err_monthly))
        KE1NA_params = resd_KE1.x
        likelihood_KE1[lat_band,lon_band] = 2*np.sum(resid_noann(KE1NA_params,KE1_monthly,KE1_err_monthly)**2)
        if likelihood_KE1[lat_band,lon_band] > CL95:
            likelihood_KE1_yes[lat_band,lon_band] = 2.
        else:
            likelihood_KE1_yes[lat_band,lon_band] = 1.



#map out likelihood for slope, noise, amp
for lat_band in range(30):
    for lon_band in range(90):
        if (np.isnan(slope_amp1_map[lat_band,lon_band]) == True) or (geovel_map[2,lat_band,lon_band] > 1000):
            continue
        amp_monthly = param_maps[lat_band,lon_band,:,0]
        slope_monthly = param_maps[lat_band,lon_band,:,2]
        noise_monthly = param_maps[lat_band,lon_band,:,3]
        amp_err_monthly = error_params[lat_band,lon_band,:,0]
        slope_err_monthly = error_params[lat_band,lon_band,:,2]
        noise_err_monthly = error_params[lat_band,lon_band,:,3]
        amp_init = np.array([np.mean(amp_monthly), 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        slope_init = np.array([np.mean(slope_monthly), 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        noise_init = np.array([np.mean(noise_monthly), 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        resd_amp = least_squares(resid_noann,amp_init,args=(amp_monthly,amp_err_monthly))
        amp_params = resd_amp.x
        resd_slope = least_squares(resid_noann,slope_init,args=(slope_monthly,slope_err_monthly))
        slope_params = resd_slope.x
        resd_noise = least_squares(resid_noann,noise_init,args=(noise_monthly,noise_err_monthly))
        noise_params = resd_noise.x
        likelihood_amp[lat_band,lon_band] = 2*np.sum(resid_noann(amp_params,amp_monthly,amp_err_monthly)**2)
        likelihood_slope[lat_band,lon_band] = 2*np.sum(resid_noann(slope_params,slope_monthly,slope_err_monthly)**2)
        likelihood_noise[lat_band,lon_band] = 2*np.sum(resid_noann(noise_params,noise_monthly,noise_err_monthly)**2)
        if likelihood_amp[lat_band,lon_band] > CL95:
            likelihood_amp_yes[lat_band,lon_band] = 2.
        else:
            likelihood_amp_yes[lat_band,lon_band] = 1.
        if likelihood_slope[lat_band,lon_band] > CL95:
            likelihood_slope_yes[lat_band,lon_band] = 2.
        else:
            likelihood_slope_yes[lat_band,lon_band] = 1.
        if likelihood_noise[lat_band,lon_band] > CL95:
            likelihood_noise_yes[lat_band,lon_band] = 2.
        else:
            likelihood_noise_yes[lat_band,lon_band] = 1.

#Filter out grids with low likelihood for KE2
KE2_phase_KE2filtered = np.empty((30,90))
KE1_phase_KE2filtered = np.empty((30,90))
KE1_phase_KE1filtered = np.empty((30,90))
slope_phase_KE2filtered = np.empty((30,90))
KEdelta_phase_filtered = np.empty((390,90))
KE2_phase_KE2filtered[:] = np.nan
KE1_phase_KE2filtered[:] = np.nan
KE1_phase_KE1filtered[:] = np.nan
slope_phase_KE2filtered[:] = np.nan
KEdelta_phase_filtered[:] = np.nan

lowKE2 = likelihood_KE2_yes == 2.
lowKE1 = likelihood_KE1_yes == 2.
KE2_phase_KE2filtered[lowKE2] = KE2_phase_map[lowKE2]
KE1_phase_KE2filtered[lowKE2] = KE1_phase_map[lowKE2]
KE1_phase_KE1filtered[lowKE1] = KE1_phase_map[lowKE1]
slope_phase_KE2filtered[lowKE2] = slope_phase1_map[lowKE2]
KEdelta_phase_filtered = (KE1_phase_KE2filtered - KE2_phase_KE2filtered) % 12

#filter for low likelihood in slope
slope_phase_slopefiltered = np.empty((30,90))
slope_phase_slopefiltered[:] = np.nan
KE1_phase_slopefiltered = np.empty((30,90))
KE1_phase_slopefiltered[:] = np.nan

slope_perr_hiCL = np.empty((30,90))
slope_perr_hiCL[:] = np.nan

hi_slope = likelihood_slope_yes == 2.
slope_phase_slopefiltered[hi_slope] = slope_phase1_map[hi_slope]
slope_perr_hiCL[hi_slope] = slope_phase_err[hi_slope]


KE1_phase_slopefiltered[hi_slope] = KE1_phase_map[hi_slope]
 

x = np.linspace(0, 360, 91)
y = np.linspace(-60, 60, 31)

land = cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor="lightgray")

fig = plt.figure(figsize=(8, 10))
gs = fig.add_gridspec(4, 1, hspace=0.2, left=0.05, right=0.95, bottom=0.05, top=0.95)

ax0 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
ax0.add_feature(land, zorder=0)
img = ax0.pcolormesh(x, y, slope_phase_slopefiltered, cmap="twilight", vmin=0, vmax=12)
plt.colorbar(img, ax=ax0)
ax0.set_title("(a)", loc="left")
ax0.set_title("$-s$ phase $-$ MLD phase (months)")

ax1 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
ax1.add_feature(land, zorder=0)
img = ax1.pcolormesh(x, y, slope_perr_hiCL, cmap="Reds", vmin=0, vmax=2)
plt.colorbar(img, ax=ax1)
ax1.set_title("(b)", loc="left")
ax1.set_title("$-s$ phase error (months)")

ax2 = fig.add_subplot(gs[2], projection=ccrs.PlateCarree())
ax2.add_feature(land, zorder=0)
img = ax2.pcolormesh(x, y, KE1_phase_KE1filtered, cmap="twilight", vmin=0, vmax=12)
plt.colorbar(img, ax=ax2)
ax2.set_title("(c)", loc="left")
ax2.set_title("KE1 phase $-$ MLD phase (months)")

ax3 = fig.add_subplot(gs[3], projection=ccrs.PlateCarree())
ax3.add_feature(land, zorder=0)
img = ax3.pcolormesh(x, y, KE2_phase_KE2filtered, cmap="twilight", vmin=0, vmax=12)
plt.colorbar(img, ax=ax3)
ax3.set_title("(d)", loc="left")
ax3.set_title("KE2 phase $-$ MLD phase (months)")

plt.savefig("fig/slope_phase_maps.pdf")

plt.show()
