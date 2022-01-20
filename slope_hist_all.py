#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 23:25:40 2021

slope_hist_all.py

Maps (and draws histograms for) SSH pectral slopes slopes for 
annually averaged and binned JFM and JJA spectra. Uses output from
jason_reduction_4deg.py

@author: albionlawrence, modifications by @joernc 
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.optimize import least_squares
import matplotlib.gridspec as gridspec

ann_param_maps = np.load('../submesoscale_seasonality_data/global_maps/jason_ann_parameters_4deg.npy',encoding='latin1',allow_pickle=True)

powerspec_map = np.load('../submesoscale_seasonality_data/global_maps/powerspectdm_4deg.npy',encoding='latin1',allow_pickle=True)[:,:,:,1:]#(30.90,12,80)
num_segs_map = np.load('../submesoscale_seasonality_data/global_maps/num_segs_4deg.npy',encoding='latin1',allow_pickle=True)#[30,90,12]
dx_global = np.load('../submesoscale_seasonality_data/global_maps/dx_4deg.npy',encoding='latin1',allow_pickle=True)#[30,90]
tides_yes = np.load('../submesoscale_seasonality_data/global_maps/tides_yes_4deg.npy',allow_pickle=True)

seg_length_allowed = 160
fft_len = int(seg_length_allowed/2) + 1 #number of independent fourier modes including constant
nbins = int(seg_length_allowed/2)#number of nonconstant Fourier modes
wn_un = np.arange(1,fft_len)

powerspec_winter = np.empty((30,90,80))
powerspec_summer = np.empty((30,90,80))
powerspec_winter[:] = np.nan
powerspec_summer[:] = np.nan

num_segs_jfm = np.sum(num_segs_map[:,:,0:3],axis=2)
num_segs_jas = np.sum(num_segs_map[:,:,6:9],axis=2)

powerspec_winter[:15,:,:] = np.sum(powerspec_map[:15,:,5:8,:]*num_segs_map[:15,:,6:9,np.newaxis],axis=2)/num_segs_jas[:15,:,np.newaxis]
powerspec_winter[15:,:,:] = np.sum(powerspec_map[15:,:,0:3,:]*num_segs_map[15:,:,0:3,np.newaxis],axis=2)/num_segs_jfm[15:,:,np.newaxis]
powerspec_summer[:15,:,:] = np.sum(powerspec_map[:15,:,0:3,:]*num_segs_map[:15,:,0:3,np.newaxis],axis=2)/num_segs_jfm[:15,:,np.newaxis]
powerspec_summer[15:,:,:] = np.sum(powerspec_map[15:,:,6:9,:]*num_segs_map[15:,:,6:9,np.newaxis],axis=2)/num_segs_jas[15:,:,np.newaxis]

slope_winter = np.empty((30,90))
slope_summer = np.empty((30,90))
slope_winter_err = np.empty((30,90))
slope_summer_err = np.empty((30,90))
slope_winter[:] = np.nan
slope_summer[:] = np.nan
slope_winter_err[:] = np.nan
slope_summer_err[:] = np.nan

def model_notides(x,deltak):
    """
    Model shape without the tidal components
    """
    return x[0]/(1 + (wn_un*deltak/x[1])**x[2]) + x[3]

def resid_fn_notides(x,data,ndat,deltak):
    """
    Residuals for nonlinear least squares, with k^{-s} slope
    Note the wavenumber depdendence -> log weighting
    
    """
    return (model_notides(x,deltak)/np.abs(data) - 1)*np.sqrt(ndat - 1)/np.sqrt(wn_un)

for lat_band in range(15):
    for lon_band in range(90):
        if (num_segs_jfm[lat_band,lon_band] > 0) and (tides_yes[lat_band,lon_band] == 0.):           
            dx_s = dx_global[lat_band,lon_band]           
            dk_s = 1/(2*dx_s*nbins)          
            spectrum_s = powerspec_summer[lat_band,lon_band]          
            lower_bounds_s = np.array([0.,0.,0.,0.])
            upper_bounds_s = np.array([np.inf,np.inf,np.inf,np.inf])           
            x_init_s = np.array([spectrum_s[0],.001,4.,.001])
            resd_s = least_squares(resid_fn_notides, x_init_s, args = (spectrum_s,num_segs_jfm[lat_band,lon_band],dk_s),bounds = (lower_bounds_s,upper_bounds_s))
            slope_summer[lat_band,lon_band] = resd_s.x[2]
            jac_cost = resd_s.jac
            #modified Jacobian used for computing error bars (based on actual likelihood)
            jac_cost_mod = jac_cost/wn_un[:,np.newaxis]
            Hess_cost = np.matmul(np.transpose(jac_cost),jac_cost)
            Hess_cost_mod = np.matmul(np.transpose(jac_cost),jac_cost_mod)
            try:
                C_cost = np.linalg.inv(Hess_cost)
            except np.linalg.linalg.LinAlgError as err:
                C_cost = np.zeros(Hess_cost.shape)
            C_cost_mod = np.matmul(np.matmul(C_cost,Hess_cost_mod),np.transpose(C_cost))
            #Cost matrix for te balanced parameters
            C_cost_balanced = C_cost_mod[:3,:3]
            slope_summer_err[lat_band,lon_band] = np.sqrt(np.abs(C_cost_balanced[2,2]))           
        if (num_segs_jfm[lat_band+15,lon_band] > 0) and (tides_yes[lat_band+15,lon_band] == 0.):
            dx_w = dx_global[lat_band+15,lon_band]
            dk_w = 1/(2*dx_w*nbins)
            spectrum_w = powerspec_winter[lat_band+15,lon_band,:]
            lower_bounds_w = np.array([0.,0.,0.,0.])
            upper_bounds_w = np.array([np.inf,np.inf,np.inf,np.inf])
            x_init_w = np.array([spectrum_w[0],.001,4.,.001])
            resd_w = least_squares(resid_fn_notides, x_init_w, args = (spectrum_w,num_segs_jfm[lat_band+15,lon_band],dk_w),bounds = (lower_bounds_w,upper_bounds_w))
            slope_winter[lat_band+15,lon_band] = resd_w.x[2]
            jac_cost = resd_w.jac
            #modified Jacobian used for computing error bars (based on actual likelihood)
            jac_cost_mod = jac_cost/wn_un[:,np.newaxis]
            Hess_cost = np.matmul(np.transpose(jac_cost),jac_cost)
            Hess_cost_mod = np.matmul(np.transpose(jac_cost),jac_cost_mod)
            try:
                C_cost = np.linalg.inv(Hess_cost)
            except np.linalg.linalg.LinAlgError as err:
                C_cost = np.zeros(Hess_cost.shape)
            C_cost_mod = np.matmul(np.matmul(C_cost,Hess_cost_mod),np.transpose(C_cost))
            #Cost matrix for te balanced parameters
            C_cost_balanced = C_cost_mod[:3,:3]
            slope_winter_err[lat_band+15,lon_band] = np.sqrt(np.abs(C_cost_balanced[2,2]))  
        if (num_segs_jas[lat_band+15,lon_band] > 0) and (tides_yes[lat_band+15,lon_band] == 0.):           
            dx_s = dx_global[lat_band+15,lon_band]           
            dk_s = 1/(2*dx_s*nbins)          
            spectrum_s = powerspec_summer[lat_band+15,lon_band]          
            lower_bounds_s = np.array([0.,0.,0.,0.])
            upper_bounds_s = np.array([np.inf,np.inf,np.inf,np.inf])           
            x_init_s = np.array([spectrum_s[0],.001,4.,.001])
            resd_s = least_squares(resid_fn_notides, x_init_s, args = (spectrum_s,num_segs_jas[lat_band+15,lon_band],dk_s),bounds = (lower_bounds_s,upper_bounds_s))    
            slope_summer[lat_band+15,lon_band] = resd_s.x[2]
            jac_cost = resd_s.jac
            #modified Jacobian used for computing error bars (based on actual likelihood)
            jac_cost_mod = jac_cost/wn_un[:,np.newaxis]
            Hess_cost = np.matmul(np.transpose(jac_cost),jac_cost)
            Hess_cost_mod = np.matmul(np.transpose(jac_cost),jac_cost_mod)
            try:
                C_cost = np.linalg.inv(Hess_cost)
            except np.linalg.linalg.LinAlgError as err:
                C_cost = np.zeros(Hess_cost.shape)
            C_cost_mod = np.matmul(np.matmul(C_cost,Hess_cost_mod),np.transpose(C_cost))
            #Cost matrix for te balanced parameters
            C_cost_balanced = C_cost_mod[:3,:3]
            slope_summer_err[lat_band+15,lon_band] = np.sqrt(np.abs(C_cost_balanced[2,2]))
        if (num_segs_jas[lat_band,lon_band] > 0) and (tides_yes[lat_band,lon_band] == 0.):
            dx_w = dx_global[lat_band,lon_band]
            dk_w = 1/(2*dx_w*nbins)
            spectrum_w = powerspec_winter[lat_band,lon_band,:]
            lower_bounds_w = np.array([0.,0.,0.,0.])
            upper_bounds_w = np.array([np.inf,np.inf,np.inf,np.inf])
            x_init_w = np.array([spectrum_w[0],.001,4.,.001])
            resd_w = least_squares(resid_fn_notides, x_init_w, args = (spectrum_w,num_segs_jas[lat_band,lon_band],dk_w),bounds = (lower_bounds_w,upper_bounds_w))
            slope_winter[lat_band,lon_band] = resd_w.x[2]
            jac_cost = resd_w.jac
            #modified Jacobian used for computing error bars (based on actual likelihood)
            jac_cost_mod = jac_cost/wn_un[:,np.newaxis]
            Hess_cost = np.matmul(np.transpose(jac_cost),jac_cost)
            Hess_cost_mod = np.matmul(np.transpose(jac_cost),jac_cost_mod)
            try:
                C_cost = np.linalg.inv(Hess_cost)
            except np.linalg.linalg.LinAlgError as err:
                C_cost = np.zeros(Hess_cost.shape)
            C_cost_mod = np.matmul(np.matmul(C_cost,Hess_cost_mod),np.transpose(C_cost))
            #Cost matrix for te balanced parameters
            C_cost_balanced = C_cost_mod[:3,:3]
            slope_winter_err[lat_band,lon_band] = np.sqrt(np.abs(C_cost_balanced[2,2]))

x = np.linspace(0, 360, 91)
y = np.linspace(-60, 60, 31)

land = cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor="lightgray")

fig = plt.figure(figsize=(9.6, 7.2))
gs = fig.add_gridspec(3, 2, width_ratios=(3.5, 1), hspace=0.2, wspace=0, left=0.02, right=0.98, bottom=0.05, top=0.95)

ax00 = fig.add_subplot(gs[0,0], projection=ccrs.PlateCarree())
ax00.add_feature(land, zorder=0)
img = ax00.pcolormesh(x, y, ann_param_maps[:,:,2], vmin=2, vmax=7)
plt.colorbar(img, ax=ax00)
ax00.set_title("(a)", loc="left")
ax00.set_title('annual mean spectral slope $s$')

ax10 = fig.add_subplot(gs[1,0], projection=ccrs.PlateCarree())
ax10.add_feature(land, zorder=0)
img = ax10.pcolormesh(x, y, slope_summer, vmin=2, vmax=7)
plt.colorbar(img, ax=ax10)
ax10.set_title("(c)", loc="left")
ax10.set_title('summer spectral slope $s$')

ax20 = fig.add_subplot(gs[2,0], projection=ccrs.PlateCarree())
ax20.add_feature(land, zorder=0)
img = ax20.pcolormesh(x, y, slope_winter, vmin=2, vmax=7)
plt.colorbar(img, ax=ax20)
ax20.set_title("(e)", loc="left")
ax20.set_title('winter spectral slope $s$')

b = np.linspace(0.75, 8.25, 16)

ax01 = fig.add_subplot(gs[0,1])
ax01.hist(ann_param_maps[:,:,2].flatten(), bins=b)
ax01.set_title("(b)", loc="left")
ax01.set_title('annual mean')

ax11 = fig.add_subplot(gs[1,1])
ax11.hist(slope_summer.flatten(), bins=b)
ax11.set_title("(d)", loc="left")
ax11.set_title('summer')

ax21 = fig.add_subplot(gs[2,1])
ax21.hist(slope_winter.flatten(), bins=b)
ax21.set_title("(f)", loc="left")
ax21.set_title('winter')

ax01.set_xlim(b[0], b[-1])
ax11.set_xlim(b[0], b[-1])
ax21.set_xlim(b[0], b[-1])

ax01.set_xticks(b[::2]+0.25)
ax11.set_xticks(b[::2]+0.25)
ax21.set_xticks(b[::2]+0.25)

m = max([ax01.get_ylim()[1], ax11.get_ylim()[1], ax21.get_ylim()[1]])
ax01.set_ylim(0, m)
ax11.set_ylim(0, m)
ax21.set_ylim(0, m)

ax01.set_xticklabels([])
ax11.set_xticklabels([])

fig.savefig("fig/slope_maps.pdf")

plt.show()
