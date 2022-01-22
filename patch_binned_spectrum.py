#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_binned_spectrum.py lat lon track

Plots JFM and JAS spectra for a whole patch
of sie (8 deg)^2 centered on lat N lon E, 
and again for selected track, with confidence intervals.

Also computes ratio with confidence intervals based on
the F distribution

uses output from jason_patch_spectrum_all.py

Created on Sun Aug 15 09:24:54 2021

@author: albionlawrence modifications by joernc
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.io import loadmat
import matplotlib.cm as cm
from scipy.optimize import least_squares
import netCDF4 as nc
from scipy.stats import chi2, f
import scipy.integrate as sint
#import seaborn as sns
#from scipy.optimize import least_squares

lat = int(sys.argv[1])
lon = int(sys.argv[2])
track = int(sys.argv[3])
x_int = int(lon/8)
y_int = int((lat+60)/8)
M2 = 2*np.pi/(12.42*3600)
grav = 9.8
f_inertial = 2*np.pi*np.sin(2*np.pi*lat/360)/(12*3600)
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

spec = loadmat('../submesoscale_seasonality_data/patch_spectra/jason_lat{:n}_lon{:n}_8deg_unfiltered.mat'.format(lat,lon))

ps_all = spec['ps_all'][:,:,1:]
dx = spec['dx'][0][0]
num_segs = spec['num_segs']
good_cycles = spec['good_cycles']
pass_list = spec['pass_list'][0,:]
month_list = spec['month_list']
high_cycle = spec['high_cycle'][0][0]
low_cycle = spec['low_cycle'][0][0]
constant_mode_array = spec['constant_mode_array']
fft_len = ps_all.shape[2] + 1

#multiplier for converting SSH power spectra in m^2/cpkm to KE spectrum in m^3
track_length = ps_all.shape[2]*2
dk = 1/(track_length*dx)
k_n = 1/(2*dx)
wn_km = np.arange(1,fft_len)*dk
wn_m = np.arange(1,fft_len)*dk/1000 #wavenumber in 1/m
awn_m= 2*np.pi*wn_m #angular wavenumber
ssh_to_ke = 2*awn_m**2*grav**2/f_inertial**2
#factor of 2 to get u**2 + v**2; factor of 1000 to convert the dx
#used in cmoputing the SSH PSD (which is importd in m^2/(cpkm))

#Data for "good" passes with enough data
good_passes = np.where(np.amin(num_segs,axis=0) > 4)[0]
good_passes = np.unique(good_passes) 
pass_list_good = pass_list[good_passes]
month_list_good = month_list[good_passes,:]
num_good_segs = num_segs[:,good_passes]
num_segs_patch = np.sum(num_good_segs,axis=1)
num_good_passes = len(pass_list_good)
good_cycles_good_passes = good_cycles[good_passes,:]
ps_good = ps_all[good_passes,:,:]

#extract power spectra for individual track
pass_indx =  np.where(pass_list == track)[0][0]
ps_track = ps_all[pass_indx,:,:]

#Now let's generate monthly spectra and binned spectra for patch
#Note "unfiltered" because I'm ot rejecting passes with less than 4 
#gap-free tracks in a calendar month, as binning should build up statistics.
#Commented out "filtered" specra apply that crireria before binning.
num_unfiltered_jfm = np.sum(num_segs[0:3,:])
num_unfiltered_jas = np.sum(num_segs[6:9,:])
#num_filtered_jfm = np.sum(num_good_segs[6:9,:])
#num_filtered_jas = np.sum(num_good_segs[6:9,:])

monthly_unfiltered_spectrum = np.zeros((12,high_cycle - low_cycle + 1,fft_len-1))
#monthly_filtered_spectrum = np.zeros((12,high_cycle - low_cycle + 1,fft_len-1))

for k, p in enumerate(pass_list):
    cycle_indx = np.nonzero(good_cycles[k,:])[0]
    for n,cycle in enumerate(cycle_indx):
        if month_list[k,cycle] == np.nan:
            print('hah')
        monthly_unfiltered_spectrum[int(month_list[k,cycle]),k,:] += ps_all[k,cycle,:]
        #if p in pass_list_good:
         #   kg = np.where(pass_list_good == p)[0][0]
         #  monthly_filtered_spectrum[int(month_list_good[kg,cycle]),kg,:] += ps_good[kg,cycle,:]

#binned spectra
unfiltered_ke_jfm = ssh_to_ke*np.sum(monthly_unfiltered_spectrum[0:3,:,:],axis=(0,1))/num_unfiltered_jfm
#filtered_ke_jfm = ssh_to_ke*np.sum(monthly_filtered_spectrum[0:3,:,:],axis=(0,1))/num_filtered_jfm
unfiltered_ke_jas = ssh_to_ke*np.sum(monthly_unfiltered_spectrum[5:8,:,:],axis=(0,1))/num_unfiltered_jas
#filtered_ke_jas = ssh_to_ke*np.sum(monthly_filtered_spectrum[5:8,:,:],axis=(0,1))/num_filtered_jas

#spectra along track
num_track_jfm = np.sum(num_segs[0:3,pass_indx])
num_track_jas = np.sum(num_segs[6:9,pass_indx])

monthly_track_spectrum = np.zeros((12,fft_len-1))
cycle_indx_track = np.nonzero(good_cycles[pass_indx,:])[0]
for n,cycle in enumerate(cycle_indx_track):
    monthly_track_spectrum[int(month_list[pass_indx,cycle]),:] += ps_track[cycle,:]

track_ke_jfm = ssh_to_ke*np.sum(monthly_track_spectrum[0:3,:],axis=0)/num_track_jfm
track_ke_jas = ssh_to_ke*np.sum(monthly_track_spectrum[6:9,:],axis=0)/num_track_jas

##Now lets compute some likelihoods

# hiCL_filtered_jfm = (num_filtered_jfm)*filtered_ke_jfm/chi2.interval(.95,num_filtered_jfm)[0]
# loCL_filtered_jfm = (num_filtered_jfm)*filtered_ke_jfm/chi2.interval(.95,num_filtered_jfm)[1]
# hiCL_filtered_jas = (num_filtered_jas)*filtered_ke_jas/chi2.interval(.95,num_filtered_jas)[0]
# loCL_filtered_jas = (num_filtered_jas)*filtered_ke_jas/chi2.interval(.95,num_filtered_jas)[1]

hiCL_unfiltered_jfm = (num_unfiltered_jfm)*unfiltered_ke_jfm/chi2.interval(.95,num_unfiltered_jfm)[0]
loCL_unfiltered_jfm = (num_unfiltered_jfm)*unfiltered_ke_jfm/chi2.interval(.95,num_unfiltered_jfm)[1]
hiCL_unfiltered_jas = (num_unfiltered_jas)*unfiltered_ke_jas/chi2.interval(.95,num_unfiltered_jas)[0]
loCL_unfiltered_jas = (num_unfiltered_jas)*unfiltered_ke_jas/chi2.interval(.95,num_unfiltered_jas)[1]
hiCL_track_jfm = (num_track_jfm)*track_ke_jfm/chi2.interval(.95,num_track_jfm)[0]
loCL_track_jfm = (num_track_jfm)*track_ke_jfm/chi2.interval(.95,num_track_jfm)[1]
hiCL_track_jas = (num_track_jas)*track_ke_jas/chi2.interval(.95,num_track_jas)[0]
loCL_track_jas = (num_track_jas)*track_ke_jas/chi2.interval(.95,num_track_jas)[1]

#finally, let's compute he ratio of JFM/JAS and the associated confidence limits
ratio_unfiltered = unfiltered_ke_jfm/unfiltered_ke_jas
F_low = f.interval(.95,num_unfiltered_jas,num_unfiltered_jfm)[0]
F_hi = f.interval(.95,num_unfiltered_jas,num_unfiltered_jfm)[1]

ratio_track = track_ke_jfm/track_ke_jas
F_low_track = f.interval(.95,num_track_jas,num_track_jfm)[0]
F_hi_track = f.interval(.95,num_track_jas,num_track_jfm)[1]

#before going on, I would like to compute SSH slopes and errors for the binned spectra

def model_notides(x):
    """
    Model shape without the tidal components
    """
    return x[0]/(1 + (wn_km/x[1])**x[2]) + x[3]

def resid_notides(x,data,ndat):
    """
    Residuals for nonlinear east squares, with k^{-s} slope
    """
    return (model_notides(x)/np.abs(data) - 1)*np.sqrt(ndat - 1)*np.sqrt(dk)/np.sqrt(wn_km)
  
ssh_jfm = np.sum(monthly_unfiltered_spectrum[0:3,:,:],axis=(0,1))/num_unfiltered_jfm
ssh_jas = np.sum(monthly_unfiltered_spectrum[6:9,:,:],axis=(0,1))/num_unfiltered_jas
ssh_track_jfm = np.sum(monthly_track_spectrum[0:3,:],axis=0)/num_track_jfm
ssh_track_jas = np.sum(monthly_track_spectrum[6:9,:],axis=0)/num_track_jas
  
x_init_jfm = np.array([ssh_jfm[0],.001,4.,.001])
x_init_jas = np.array([ssh_jas[0],.001,4.,.001])
x_init_track_jfm = np.array([ssh_track_jfm[0],.001,4.,.001])
x_init_track_jas = np.array([ssh_track_jas[0],.001,4.,.001])
lb = np.array([0.,0.,0.,0.])
ub = np.array([np.inf,np.inf,np.inf,np.inf])

resd_jfm = least_squares(resid_notides, x_init_jfm, args = (ssh_jfm,num_unfiltered_jfm),bounds = (lb,ub))
resd_jas = least_squares(resid_notides, x_init_jas, args = (ssh_jas,num_unfiltered_jas),bounds = (lb,ub))
resd_jfm_track = least_squares(resid_notides, x_init_track_jfm, args = (ssh_track_jfm,num_track_jfm),bounds = (lb,ub))
resd_jas_track = least_squares(resid_notides, x_init_track_jas, args = (ssh_track_jas,num_track_jas),bounds = (lb,ub))
slope_jfm = resd_jfm.x[2]
slope_jas = resd_jas.x[2]
slope_jfm_track = resd_jfm_track.x[2]
slope_jas_track = resd_jas_track.x[2]

jac_cost_jfm = resd_jfm.jac
jac_cost_jas = resd_jas.jac
jac_cost_jfm_track = resd_jfm_track.jac
jac_cost_jas_track = resd_jas_track.jac

jac_cost_jfm_mod = jac_cost_jfm*dk/wn_km[:,np.newaxis]
jac_cost_jas_mod = jac_cost_jas*dk/wn_km[:,np.newaxis]
jac_cost_jfm_track_mod = jac_cost_jfm_track*dk/wn_km[:,np.newaxis]
jac_cost_jas_track_mod = jac_cost_jas_track*dk/wn_km[:,np.newaxis]

Hess_cost_jfm = np.matmul(np.transpose(jac_cost_jfm),jac_cost_jfm)
Hess_cost_jas = np.matmul(np.transpose(jac_cost_jas),jac_cost_jas)
Hess_cost_jfm_track = np.matmul(np.transpose(jac_cost_jfm_track),jac_cost_jfm_track)
Hess_cost_jas_track = np.matmul(np.transpose(jac_cost_jas_track),jac_cost_jfm_track)

Hess_mod_jfm = np.matmul(np.transpose(jac_cost_jfm),jac_cost_jfm_mod)
Hess_mod_jas = np.matmul(np.transpose(jac_cost_jas),jac_cost_jas_mod)
Hess_mod_jfm_track = np.matmul(np.transpose(jac_cost_jfm_track),jac_cost_jfm_track_mod)
Hess_mod_jas_track = np.matmul(np.transpose(jac_cost_jas_track),jac_cost_jas_track_mod)

try:
    C_cost_jfm = np.linalg.inv(Hess_cost_jfm)
except np.linalg.linalg.LinAlgError as err:
    C_cost_jfm = np.zeros(Hess_cost_jfm.shape)

try:
    C_cost_jas = np.linalg.inv(Hess_cost_jas)
except np.linalg.linalg.LinAlgError as err:
    C_cost_jas = np.zeros(Hess_cost_jas.shape)
    
try:
    C_cost_jfm_track = np.linalg.inv(Hess_cost_jfm_track)
except np.linalg.linalg.LinAlgError as err:
    C_cost_jfm_track = np.zeros(Hess_cost_jfm_track.shape)

try:
    C_cost_jas_track = np.linalg.inv(Hess_cost_jas_track)
except np.linalg.linalg.LinAlgError as err:
    C_cost_jas_track = np.zeros(Hess_cost_jas_track.shape)


C_mod_jfm = np.matmul(np.matmul(C_cost_jfm,Hess_mod_jfm),np.transpose(C_cost_jfm))
C_mod_jas = np.matmul(np.matmul(C_cost_jas,Hess_mod_jas),np.transpose(C_cost_jas))
C_mod_jfm_track = np.matmul(np.matmul(C_cost_jfm_track,Hess_mod_jfm_track),np.transpose(C_cost_jfm_track))
C_mod_jas_track = np.matmul(np.matmul(C_cost_jas_track,Hess_mod_jas_track),np.transpose(C_cost_jas_track))

slope_err_jfm = np.sqrt(C_mod_jfm[2,2])
slope_err_jas = np.sqrt(C_mod_jas[2,2])
slope_err_jfm_track = np.sqrt(C_mod_jfm_track[2,2])
slope_err_jas_track = np.sqrt(C_mod_jas_track[2,2])

#errors in spectral slope
print('JFM slope {:0.2f} +/- {:0.2f}'.format(slope_jfm,slope_err_jfm))
print('JAS slope {:0.2f} +/- {:0.2f}'.format(slope_jas,slope_err_jas))
print('JFM slope Track {:n} {:0.2f} +/- {:0.2f}'.format(track,slope_jfm_track,slope_err_jfm_track))
print('JAS slope Track {:n} {:0.2f} +/- {:0.2f}'.format(track,slope_jas_track,slope_err_jas_track))

lat_s = "{:d}°N".format(lat) if lat > 0 else "{:d}°S".format(-lat)
lon_s = "{:d}°E".format(lon) if lon < 180 else "{:d}°W".format(360-lon)

fig, ax = plt.subplots(2, 2, sharex=True, sharey="row", figsize=(8, 7.6))

ax[0,0].loglog(wn_km, unfiltered_ke_jfm, label='JFM', color='tab:blue', zorder=1)
ax[0,0].loglog(wn_km, unfiltered_ke_jas, label='JAS', color='tab:orange', zorder=2)
ax[0,0].fill_between(wn_km, loCL_unfiltered_jfm, hiCL_unfiltered_jfm, color='tab:blue', alpha=0.3, linewidths=0, zorder=1)
ax[0,0].fill_between(wn_km, loCL_unfiltered_jas, hiCL_unfiltered_jas, color='tab:orange', alpha=0.3, linewidths=0, zorder=2)
ax[0,0].set_xlim(1e-3, 1e-1)
ax[0,0].set_title('(a)', loc="left")
ax[0,0].set_title('patch {:s} {:s}'.format(lat_s, lon_s))
ax[0,0].set_ylabel('KE specral density (m$^2$ s$^{-2}$ cpkm$^{-1}$)')
ax[0,0].legend(frameon=False, loc=2)

ax[0,1].loglog(wn_km,track_ke_jfm, label='JFM', color='tab:blue', zorder=1)
ax[0,1].loglog(wn_km,track_ke_jas, label='JAS', color='tab:orange', zorder=2)
ax[0,1].fill_between(wn_km, loCL_track_jfm, hiCL_track_jfm, color='tab:blue', alpha=0.3, linewidths=0, zorder=1)
ax[0,1].fill_between(wn_km, loCL_track_jas, hiCL_track_jas, color='tab:orange', alpha=0.3, linewidths=0, zorder=2)
ax[0,1].set_title('(b)', loc="left")
ax[0,1].set_title('pass {:n}'.format(track))
ax[0,1].legend(frameon=False, loc=2)

ax[1,0].loglog(wn_km, ratio_unfiltered)
ax[1,0].fill_between(wn_km, F_low*ratio_unfiltered, F_hi*ratio_unfiltered, color='tab:blue', alpha=0.3, linewidths=0)
ax[1,0].axhline(y=1, color='black', linewidth=0.8)
ax[1,0].set_xlabel('wavenumber (cpkm)')
ax[1,0].set_ylabel('ratio of JFM to JAS spectrum')
ax[1,0].set_title('(c)', loc="left")
ax[1,0].set_title('patch {:s} {:s}'.format(lat_s, lon_s))

ax[1,1].loglog(wn_km,ratio_track)
ax[1,1].fill_between(wn_km, F_low_track*ratio_track, F_hi_track*ratio_track, color='tab:blue', alpha=0.3, linewidths=0)
ax[1,1].axhline(y=1, color='black', linewidth=0.8)
ax[1,1].set_xlabel('wavenumber (cpkm)')
ax[1,1].set_title('(d)', loc="left")
ax[1,1].set_title('pass {:n}'.format(track))

ax[1,0].set_yticks([1])
ax[1,0].set_yticks([i/10 for i in range(2, 10)] + [i for i in range(2, 10)], minor=True)
ax[1,0].set_yticklabels([1.0])
ax[1,0].set_yticklabels([0.2] + [None]*10 + [5.0] + [None]*4, minor=True)
ax[1,0].set_ylim(0.2, 9)

fig.align_ylabels()
fig.tight_layout()
if lat == 32 and lon == 292: fig.savefig("fig/binned_spectra_gs.pdf")

plt.show()
