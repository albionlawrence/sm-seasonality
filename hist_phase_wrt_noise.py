#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hist_phase_wrt_noise.py

Creates stacked histograms of phases WRT phase of noise
*as computed from ECCO-4 climatology*,
color coded by log likelihood
Uses output from jason_reduction_4deg.py

Created on Sat April 24

@author: albionlawrence, modifications by joernc
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scipy.stats as stats
from scipy.optimize import least_squares
from scipy.stats import chi2
from string import ascii_lowercase

param_maps = np.load('../submesoscale_seasonality_data/global_maps/jason_map_parameters_4deg.npy',encoding='latin1',allow_pickle=True)
err_maps = np.load('../submesoscale_seasonality_data/global_maps/error_map_4deg.npy',encoding='latin1',allow_pickle=True)
mld_map = np.load('../submesoscale_seasonality_data/global_maps/mld_map_4deg.npy',encoding='latin1',allow_pickle=True)
energy_map = np.load('../submesoscale_seasonality_data/global_maps/energy_map_4deg.npy',encoding='latin1',allow_pickle=True)
energy_map_err = np.load('../submesoscale_seasonality_data/global_maps/energy_map_err_4deg.npy',encoding='latin1',allow_pickle=True)
ann_param_maps = np.load('../submesoscale_seasonality_data/global_maps/jason_ann_parameters_4deg.npy',encoding='latin1',allow_pickle=True)
num_passes_map = np.load('../submesoscale_seasonality_data/global_maps/num_passes_4deg.npy',encoding='latin1',allow_pickle=True)
#KE_datamap = np.load('KE_datamap.npy',encoding='latin1')

months_delay = np.arange(0,13)


sinusoid1 = np.exp(2j*np.pi*np.arange(12)/12)
sinusoid1 = sinusoid1[np.newaxis,np.newaxis,:]
sinusoidarg_year = 2*np.pi*np.arange(12)/12
sinusoidarg = sinusoidarg_year[np.newaxis,np.newaxis,:]

mld_mode1_map = np.sum(mld_map*sinusoid1,axis=2)
mld_phase1_map = np.angle(mld_mode1_map)
mld_amp_map = np.abs(mld_mode1_map)

num_passes_south = np.ndarray.flatten(num_passes_map[:15,:])
num_passes_north = np.ndarray.flatten(num_passes_map[15:,:])

noise_mode1_map = np.sum(param_maps[:,:,:,3]*sinusoid1,axis=2)/6
noise_amp1_map = np.abs(noise_mode1_map)
noise_phase1_map = np.angle(noise_mode1_map)
noise_mean_map = np.mean(param_maps[:,:,:,3],axis=2)
noise_phase_diff = 12*(noise_phase1_map - mld_phase1_map)/(2*np.pi) % 12

slope_mode1_map = np.sum(param_maps[:,:,:,2]*sinusoid1,axis=2)/6
slope_amp1_map = np.abs(slope_mode1_map)
slope_phase1_map = np.angle(slope_mode1_map)
slope_mean_map = np.mean(param_maps[:,:,:,2], axis=2)
slope_phase_diff = 12*(slope_phase1_map + np.pi - mld_phase1_map)/(2*np.pi) % 12
slope_vs_noise = 12*(slope_phase1_map + np.pi - noise_phase1_map)/(2*np.pi) % 12
# slope_phase_diff = np.empty((30,90))
# slope_phase_diff[:] = np.nan
# slope_phase_diff[:15,:] = 12*slope_phase1_map[:15,:]/(2*np.pi) % 12
# slope_phase_diff[15:,:] = 12*(slope_phase1_map[15:,:] + np.pi)/(2*np.pi) % 12

KE1_mode1_map = np.sum(energy_map[:,:,:,1]*sinusoid1,axis=2)/6
KE1_mode1_amp = np.abs(KE1_mode1_map)
KE1_mode1_amp_south = np.ndarray.flatten(KE1_mode1_amp[:15,:])
KE1_mode1_amp_north = np.ndarray.flatten(KE1_mode1_amp[15:,:])
KE1_phase = np.angle(KE1_mode1_map)
KE1_phasediff = np.empty((30,90))
KE1_phasediff[:] = np.nan
KE1_phasediff = 12*(KE1_phase - mld_phase1_map)/(2*np.pi) % 12
KE1_vs_noise = 12*(KE1_phase - noise_phase1_map)/(2*np.pi) % 12
# KE1_phasediff[:15,:] = 12*(KE1_phase[:15,:] - np.pi)/(2*np.pi) % 12
# KE1_phasediff[15:,:] = 12*(KE1_phase[15:,:])/(2*np.pi) % 12
#KE1_phasediff_south = np.ndarray.flatten(12*(KE1_phase[:15,:] - np.pi)/(2*np.pi) % 12)
#KE1_phasediff_north = np.ndarray.flatten(12*KE1_phase[15:,:]/(2*np.pi) % 12)

KE2_mode1_map = np.sum(energy_map[:,:,:,2]*sinusoid1,axis=2)/6
KE2_mode1_amp = np.abs(KE2_mode1_map)
KE2_mode1_amp_south = np.ndarray.flatten(KE2_mode1_amp[:15,:])
KE2_mode1_amp_north = np.ndarray.flatten(KE2_mode1_amp[15:,:])
KE2_phase = np.angle(KE2_mode1_map)
KE2_phasediff = np.empty((30,90))
KE2_phasediff[:] = np.nan
KE2_phasediff = 12*(KE2_phase - mld_phase1_map)/(2*np.pi) % 12
KE2_vs_noise = 12*(KE2_phase - noise_phase1_map)/(2*np.pi) % 12
# KE2_phasediff[:15,:] = 12*(KE2_phase[:15,:] - np.pi)/(2*np.pi) % 12
# KE2_phasediff[15:,:] = 12*(KE2_phase[15:,:])/(2*np.pi) % 12
#KE2_phasediff = 12*(KE2_phase - mld_phase1_map)/(2*np.pi) % 12
#KE2_phasediff_south = np.ndarray.flatten(12*(KE2_phase[:15,:] - np.pi)/(2*np.pi) % 12)
#KE2_phasediff_north = np.ndarray.flatten(12*KE2_phase[15:,:]/(2*np.pi) % 12)

KE12_phasediff = 12*(KE1_phase - KE2_phase)/(2*np.pi) % 12

filtered_KE1 = np.empty((30,90))
filtered_KE2 = np.empty((30,90))
filtered_slope = np.empty((30,90))
filtered_KE12 = np.empty((30,90))
filtered_noise = np.empty((30,90))
filtered_KE1[:] = np.nan
filtered_KE2[:] = np.nan
filtered_slope[:] = np.nan
filtered_KE12[:] = np.nan
filtered_noise[:] = np.nan

lowCL_KE1 = np.empty((30,90))
lowCL_KE2 = np.empty((30,90))
lowCL_slope = np.empty((30,90))
lowCL_KE12 = np.empty((30,90))
lowCL_noise = np.empty((390,90))
lowCL_KE1[:] = np.nan
lowCL_KE2[:] = np.nan
lowCL_slope[:] = np.nan
lowCL_KE12[:] = np.nan
lowCL_noise[:] = np.nan

likelihood_noise = np.empty((30,90))
likelihood_noise[:] = np.nan

slope_noise_diff = np.empty((30,90))
slope_noise_diff[:] = np.nan
KE2_noise_diff = np.empty((30,90))
KE2_noise_diff[:] = np.nan
KE1_noise_diff = np.empty((30,90))
KE1_noise_diff[:] = np.nan

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

CL95 = chi2.interval(0.95,2)[1]

#separating out the noise phase by likelihood of annual signal
for lat_band in range(30):
    for lon_band in range(90):
        if np.isnan(noise_amp1_map[lat_band,lon_band]) == True:
            continue
        noise = param_maps[lat_band,lon_band,:,3]
        noise_err = err_maps[lat_band,lon_band,:,3]
        noiseNA_init = np.array([np.mean(noise),0.,0.,0.,0.,0.,0.,0.,0.,0.])
        resd_noise = least_squares(resid_noann,noiseNA_init,args=(noise,noise_err))
        noiseNA_params = resd_noise.x
        likelihood_noise[lat_band,lon_band] = 2*np.sum(resid_noann(noiseNA_params,noise,noise_err)**2)
        if likelihood_noise[lat_band,lon_band] > CL95:
            filtered_noise[lat_band,lon_band] = noise_phase_diff[lat_band,lon_band]
        else:
            lowCL_noise[lat_band,lon_band] = noise_phase_diff[lat_band,lon_band]

#separating out slope phase by likelihood of annual signal
for lat_band in range(30):
    for lon_band in range(90):
        if np.isnan(slope_amp1_map[lat_band,lon_band]) == True:
            continue
        slope = param_maps[lat_band,lon_band,:,2]
        slope_err = err_maps[lat_band,lon_band,:,2]
        slopeNA_init = np.array([np.mean(slope), 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        resd_slope = least_squares(resid_noann,slopeNA_init,args=(slope,slope_err))
        slopeNA_params = resd_slope.x
        likelihood_slope = 2*np.sum(resid_noann(slopeNA_params,slope,slope_err)**2)
        if likelihood_slope > CL95:
            #filtered_KE1[lat_band,lon_band] = KElo_phasediff[lat_band,lon_band]
            #filtered_KE2[lat_band,lon_band] = KEhi_phasediff[lat_band,lon_band]
            filtered_slope[lat_band,lon_band] = slope_phase_diff[lat_band,lon_band]
            if likelihood_noise[lat_band,lon_band] > CL95:
                slope_noise_diff[lat_band,lon_band] = slope_vs_noise[lat_band,lon_band]
                #= 12*(slope_phase1_map[lat_band,lon_band] + np.pi - noise_phase1_map[lat_band,lon_band])/(2*np.pi) % 12
                
        else:
            #lowCL_KE1[lat_band,lon_band] = KElo_phasediff[lat_band,lon_band]
            #lowCL_KE2[lat_band,lon_band] = KEhi_phasediff[lat_band,lon_band]
            lowCL_slope[lat_band,lon_band] = slope_phase_diff[lat_band,lon_band]

for lat_band in range(30):
    for lon_band in range(90):
        if np.isnan(KE1_mode1_amp[lat_band,lon_band]) == True:
            continue
        KE1 = energy_map[lat_band,lon_band,:,1]
        KE1_err = energy_map_err[lat_band,lon_band,:,1]
        KE1NA_init = np.array([np.mean(KE1), 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        resd_KE1 = least_squares(resid_noann,KE1NA_init,args=(KE1,KE1_err))
        KE1NA_params = resd_KE1.x
        likelihood_KE1 = 2*np.sum(resid_noann(KE1NA_params,KE1,KE1_err)**2)
        
        KE2 = energy_map[lat_band,lon_band,:,2]
        KE2_err = energy_map_err[lat_band,lon_band,:,2]
        KE2NA_init = np.array([np.mean(KE2), 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        resd_KE2 = least_squares(resid_noann,KE2NA_init,args=(KE2,KE2_err))
        KE2NA_params = resd_KE2.x
        likelihood_KE2 = 2*np.sum(resid_noann(KE2NA_params,KE2,KE2_err)**2)
        
        if likelihood_KE1 > CL95:
            #filtered_KE1[lat_band,lon_band] = KElo_phasediff[lat_band,lon_band]
            #filtered_KE2[lat_band,lon_band] = KEhi_phasediff[lat_band,lon_band]
            filtered_KE1[lat_band,lon_band] = KE1_phasediff[lat_band,lon_band]
            if likelihood_noise[lat_band,lon_band] > CL95:
                KE1_noise_diff[lat_band,lon_band] = KE1_vs_noise[lat_band,lon_band]
        else:
            #lowCL_KE1[lat_band,lon_band] = KElo_phasediff[lat_band,lon_band]
            #lowCL_KE2[lat_band,lon_band] = KEhi_phasediff[lat_band,lon_band]
            lowCL_KE1[lat_band,lon_band] = KE1_phasediff[lat_band,lon_band]       
        
        if likelihood_KE2 > CL95:
            filtered_KE2[lat_band,lon_band] = KE2_phasediff[lat_band,lon_band]
            if likelihood_noise[lat_band,lon_band] > CL95:
                KE2_noise_diff[lat_band,lon_band] = KE2_vs_noise[lat_band,lon_band]
                #= 12*(KE2_phase[lat_band,lon_band] - noise_phase1_map[lat_band,lon_band])/(2*np.pi) % 12
        else:
            lowCL_KE2[lat_band,lon_band] = KE2_phasediff[lat_band,lon_band]
            
        if (likelihood_KE2 > CL95 and likelihood_KE1 > CL95):
            filtered_KE12[lat_band,lon_band] = KE12_phasediff[lat_band,lon_band]
        else:
            lowCL_KE12[lat_band,lon_band] = KE12_phasediff[lat_band,lon_band]

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 5.4))

ax[0,0].hist(np.ndarray.flatten(filtered_noise), align='mid', bins=np.arange(13))
ax[0,0].set_xlim(0, 12)
ax[0,0].set_xticks(range(13))
ax[0,0].set_xticklabels([i if i%3 == 0 else None for i in range(13)])
ax[0,0].set_title('noise phase $-$ MLD phase (months)')

ax[0,1].hist(np.ndarray.flatten(slope_vs_noise), align='mid',bins=np.arange(13))
ax[0,1].set_title('$-s$ phase $-$ noise phase (months)')

ax[1,0].hist(np.ndarray.flatten(KE2_vs_noise), align='mid',bins=np.arange(13))
ax[1,0].set_title('KE2 phase $-$ noise phase (months)')

ax[1,1].hist(np.ndarray.flatten(KE1_vs_noise), align='mid',bins=np.arange(13))
ax[1,1].set_title('KE1 phase $-$ noise phase (months)')

for i in range(4):
    ax[i//2,i%2].set_title("({:s})".format(ascii_lowercase[i]), loc="left")

fig.tight_layout()
fig.savefig("fig/phase_diff_noise_hist.pdf")

plt.show()
