#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jason_reduction_4deg.py

Based on map of overlapping 8 degree squares spaced 4 degrees apart.
Takes powerspectrum output from jason_map_dmtrack_4deg.py;
MXLDEPTH from ECCO-4 climatology; 

https://ecco.jpl.nasa.gov/drive/files/Version4/Release3/interp_climatology

cn_monthly_4deg.npy is output from monthly_speed_4deg.py

Takes annual average at each grid box, rejects if mode-1 tide too big,
that is if peak is outside of 95% CL of estimated background,
then fits without tides, creates maps of fitting parameters, integrated KE,
likelihoods

Created on Fri Apr 19 10:53:53 2019

@author: albion
"""

import numpy as np
from scipy.optimize import least_squares
from netCDF4 import Dataset#, num2date
import scipy.integrate as sint
from scipy.stats import chi2
#import glob
#import geopy.distance
#from scipy import stats
#import scipy.signal as sig
#import sys
#import os
##import scipy.io as sio

M2 = 2*np.pi/(12.42*3600)
grav = 9.8
cn_global = np.load('../submesoscale_seasonality_data/vertical_modes/cn_monthly_4deg.npy',encoding='latin1',allow_pickle=True)
cn_annual_t = np.mean(cn_global,axis=2)
cn_annual = np.transpose(cn_annual_t,(1,0,2))

#speed of sound of 1st 2 modes
#also masked when ocean too shallow

##get map of mixed layer depths
ecco_data = Dataset('../submesoscale_seasonality_data/ECCO-4/MXLDEPTH.0001.nc','r')
ecco_lat = ecco_data.variables['lat'][:,:]
ecco_lon = ecco_data.variables['lon'][:,:]
ecco_lon += 360*(1 - np.sign(ecco_lon))/2
ecco_mld = ecco_data.variables['MXLDEPTH'][:,:,:]
monthly_mld = 0.

#Load from jason_map_dmtrack_4deg.npy
powerspec_map = np.load('powerspectdm_4deg.npy',encoding='latin1',allow_pickle=True)[:,:,:,1:]
#powerspec_map_ann = np.mean(powerspec_map,axis=2)
num_segs_map = np.load('num_segs_4deg.npy',encoding='latin1',allow_pickle=True)
num_segs_annual = np.sum(num_segs_map,axis=2)
dx_global = np.load('dx_4deg.npy',encoding='latin1',allow_pickle=True)
#distance_global_annual = np.sum(distance_global_map,axis=0)
tides_yes = np.zeros((30,90))
num_good_passes = np.load('num_good_passes_4deg.npy',encoding='latin1',allow_pickle=True)
powerspec_map_ann = np.sum(powerspec_map*num_segs_map[:,:,:,np.newaxis],axis=2)/num_segs_annual[:,:,np.newaxis]

seg_length_allowed = 160
fft_len = int(seg_length_allowed/2) + 1 #number of independent fourier modes including constant
nbins = int(seg_length_allowed/2)#number of nonconstant Fourier modes
wn_un = np.arange(1,fft_len) #k/dk in the end

lat_ranges = np.arange(-60,64,4)
param_map = np.empty((30,90,12,4))
ann_param_map = np.empty((30,90,4))
error_map = np.empty((30,90,12,4))
ann_error_map = np.empty((30,90,4))
mld_map = np.empty((30,90,12))
energy_map = np.empty((30,90,12,4))
energy_map_err = np.empty((30,90,12,4))
fitted_energy_map = np.empty((30,90,12,4))
fitted_energy_err = np.empty((30,90,12,4))#0 = total KE from model; 1 = k0:2*k0, 2= 2*k0:4*k0,3 = 1/2k0:k0
noise_bins = np.empty((30,90,12,4))
num_passes_map = np.empty((30,90))
#energy_map_err = np.empty((15,45,12,3))
#KE_datamap = np.empty((15,45,12,3))
#last index k: 0-3: x[k]
param_map[:] = np.nan
ann_param_map[:] = np.nan
error_map[:] = np.nan
ann_error_map[:] = np.nan
mld_map[:] = np.nan
energy_map[:] = np.nan
energy_map_err[:] = np.nan
fitted_energy_map[:] = np.nan
fitted_energy_err[:] = np.nan
noise_bins[:] = np.nan
num_passes_map[:] = np.nan
#energy_map_err[:] = np.nan

C_cost = np.zeros((4,4))
C_cost_balanced = np.zeros((3,3))
pd_ke = np.zeros((3,fft_len-1))


def model_func(x,deltak):
    """
    function used to fit data for k^(-s) slope
    """
    return x[0]/(1 + (wn_un*deltak/x[1])**x[2]) + x[4]*np.exp(-(wn_un*deltak - k1)**2/(2*x[6]**2)) + x[5]*np.exp(-(wn_un*deltak - k2)**2/(2*x[7]**2)) + x[3]

def model_notides(x,deltak):
    """
    Model shape without the tidal components
    """
    return x[0]/(1 + (wn_un*deltak/x[1])**x[2]) + x[3]

def model_balanced(x,deltak):
    return x[0]/(1 + (wn_un*deltak/x[1])**x[2])

# def SSH_balanced(x,A,knee,slope):
#     """
#     Model SSH without the tidal components or the noise. Note that x is in 
#     units of km, but A is in units of m^2/km
#     """
#     return A*(grav**2/f**2)/(1 + (x/knee)**slope)

# def KE_balanced(x,A,knee,slope):
#     """
#     Model KE without the tidal components or the noise. Note that x is in 
#     units of km, but A is in units of m^2/km
#     """
#     return A*(x/1000)**2*(grav**2/f**2)/(1 + (x/knee)**slope)

def resid_fn(x,data,ndat,deltak):
    """
    Residuals for nonlinear least squares, with k^{-s} slope
    Note the wavenumber depdendence -> log weighting
    """
    return (model_func(x,deltak)/np.abs(data) - 1)*np.sqrt(ndat - 1)/np.sqrt(wn_un)

def resid_fn_notides(x,data,ndat,deltak):
    """
    Residuals for nonlinear least squares, with k^{-s} slope
    Note the wavenumber depdendence -> log weighting
    
    """
    return (model_notides(x,deltak)/np.abs(data) - 1)*np.sqrt(ndat - 1)/np.sqrt(wn_un)

fout = open('4deg_rejects.txt','a')

for lat_band in range(30):
#for lat_band in range(12,13):
    lat_min = lat_ranges[lat_band] - 2
    lat_max = lat_ranges[lat_band + 1] + 2
    lat_center = lat_ranges[lat_band]+2
    f = 2*np.pi*np.sin(2*np.pi*lat_center/360)/(12*3600)
    for lon_band in range(90):
    #for lon_band in range(19,20):
        if np.ma.is_masked(cn_annual[lat_band,lon_band,:]):
            tides_yes[lat_band,lon_band] = -1.
            continue
        lon_min = lon_band*4 - 2
        lon_max = (lon_band+1)*4 + 2
        lon_center = lon_band*4 + 2
        #Make sure there are no months with zero data
        if np.count_nonzero(num_segs_map[lat_band,lon_band,:]) < 12:
            continue        
        #print(lat_band,lon_band,num_segs_map[lat_band,lon_band,:]
        #compute tides (for annual averaged signal)
        k1 = 1000*np.sqrt(M2**2 - f**2)/(2*np.pi*cn_annual[lat_band,lon_band,0])
        k2 = 1000*np.sqrt(M2**2 - f**2)/(2*np.pi*cn_annual[lat_band,lon_band,1])
        #Compute annual Fourier transform
        dx = dx_global[lat_band,lon_band]#/(num_segs_map[lat_band,lon_band,mnth]*160)
        k_n = 1/(2*dx)
        dk = k_n/nbins
        resolution_meters = dk/1000
        wn = wn_un*dk
        ##Definition of wavenumber bands based on 'hard' estimation...
        #band_low = np.where((wn < 1/100))
        #band1 = np.where((wn < 1/100) & (wn > 1/200))[0]
        #band2 = np.where((wn < 1/200) & (wn > 1/300))[0]
        wn_meters = wn_un*dk/1000
        spectrum = powerspec_map_ann[lat_band,lon_band,:]
        ns_ann = num_segs_annual[lat_band,lon_band]
        #fit annual transform
        lower_bounds = np.array([0.,0.,0.,0.,0.,0.,dk/2,dk/2])
        upper_bounds = np.array([np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,dk*2,dk*2])
        x_init = np.array([spectrum[0],.001,4.,.001,.01,.01,dk,dk])
        resd = least_squares(resid_fn, x_init, args = (spectrum,ns_ann,dk),bounds = (lower_bounds,upper_bounds))
        fit_array = resd.x
        test_fn = model_balanced(fit_array,dk)
        k0_check = fit_array[1]
        if 1/k0_check > 900:
            tides_yes[lat_band,lon_band] = -1.
            print('noknee', 'lat = ',lat_center,'lon = ',lon_center, 'knee = ',1/k0_check,file=fout)
            continue
        k1_bin = int(k1*nbins/k_n) - 1#first term would be index if constant mode = 0
        k2_bin = int(k2*nbins/k_n) - 1#-1 shift because spectrum array drops constanr mode
        geospectrum = model_notides(fit_array[0:4],dk)
        fitvar_mode1 = chi2.interval(0.95,ns_ann-1)[1]*geospectrum[k1_bin]/(ns_ann-1)
        if spectrum[k1_bin] > fitvar_mode1:#if tide recorded as significant, check if it changes slope any
            tides_yes[lat_band,lon_band] = 1.
            print('yes','lat = ',lat_center, 'lon = ',lon_center,"dk = ", dk, 'k1 = ', k1, 'k1 power = ', spectrum[k1_bin], 'background = ', fitvar_mode1, 'params = ', fit_array[0:4], file=fout)
            continue
        #For no tides, compute k0 in annual average
        #spectrum, and bands for analysis of KE spectral bands
        lower_bounds_ann = np.array([0.,0.,0.,0.])
        upper_bounds_ann = np.array([np.inf,np.inf,np.inf,np.inf])
        x_init_ann = np.array([spectrum[0],.001,4.,.001])
        resd_yearly = least_squares(resid_fn_notides,x_init_ann,args=(spectrum,num_segs_annual[lat_band,lon_band],dk),bounds = (lower_bounds_ann,upper_bounds_ann))
        fit_array_ann = resd_yearly.x
        ann_param_map[lat_band,lon_band,:] = fit_array_ann
        k0_ann = fit_array_ann[1]
        k0_half = k0_ann/2
        k1_ann = 2*k0_ann
        k2_ann = 2*k1_ann
        k0ann_bin = int(k0_ann*(nbins)/k_n) - 1
        k0half_bin = int(k0_half*(nbins)/k_n) - 1
        k1ann_bin = int(k1_ann*(nbins)/k_n) - 1
        k2ann_bin = int(k2_ann*(nbins)/k_n) - 1
        signal_gt_noise = True #flag for signal > noise
        num_passes_map[lat_band,lon_band] = num_good_passes[lat_band,lon_band]
        #print('no')
        for mnth in range(12):
            lower_bounds_month = np.array([0.,0.,0.,0.])
            upper_bounds_month = np.array([np.inf,np.inf,np.inf,np.inf])
            spectrum_month = powerspec_map[lat_band,lon_band,mnth,:]
            x_init_month = np.array([spectrum_month[0],.001,4.,.001])
            resd_month = least_squares(resid_fn_notides,x_init_month,args = (spectrum_month,num_segs_map[lat_band,lon_band,mnth],dk),bounds=(lower_bounds_month,upper_bounds_month))
            param_map[lat_band,lon_band,mnth,:] = resd_month.x
            #compute MLD in grid box
            ecco_grid_points = np.transpose(np.nonzero((ecco_lat >= lat_min) & (ecco_lat <= lat_max) & (ecco_lon >= lon_min) & (ecco_lon <= lon_max)))
            num_ecco_points = ecco_grid_points.shape[0]
            for pts in range(num_ecco_points):
                monthly_mld += ecco_mld[mnth,ecco_grid_points[pts,0],ecco_grid_points[pts,1]]/num_ecco_points
            mld_map[lat_band,lon_band,mnth] = monthly_mld
            monthly_mld = 0.
            #compute error bars in parameters
            #cost function based on residual used for least squares fitting.
            #Note this has an extra factor of 1/k to weight low wavenumbers
            #but this factor should not be there in estimating the likelihood
            jac_cost = resd_month.jac
            #modified Jacobian used for computing error bars (based on actual likelihood)
            jac_cost_mod = jac_cost/wn_un[:,np.newaxis]
            Hess_cost = np.matmul(np.transpose(jac_cost),jac_cost)
            Hess_cost_mod = np.matmul(np.transpose(jac_cost),jac_cost_mod)
            try:
                C_cost = np.linalg.inv(Hess_cost)
                C_cost = np.linalg.inv(Hess_cost)
            except np.linalg.linalg.LinAlgError as err:
                C_cost = np.zeros(Hess_cost.shape)
            C_cost_mod = np.matmul(np.matmul(C_cost,Hess_cost_mod),np.transpose(C_cost))
            #Cost matrix for te balanced parameters
            C_cost_balanced = C_cost_mod[:3,:3]
            error_map[lat_band,lon_band,mnth,:] = np.sqrt(np.abs(C_cost_mod.diagonal()))
            #compute model kinetic energy and errors
            A,knee,slope = resd_month.x[:3]
            #if (SSH_balanced(k2_ann,A,knee,slope) > resd_month.x[3]):
            ke_spectrum = spectrum_month*wn_meters**2*grav**2/f**2
            ke_subtracted = (spectrum_month - param_map[lat_band,lon_band,mnth,3])*wn_meters**2*grav**2/f**2
            noise_ke = param_map[lat_band,lon_band,mnth,3]*wn_meters**2*grav**2/f**2
            fitted_ke_spectrum = model_balanced(param_map[lat_band,lon_band,mnth,:3],dk)*wn_meters**2*grav**2/f**2
            #KE from actual spectrum with fitted noise subtracted
            energy_map[lat_band,lon_band,mnth,0] = np.sum(ke_subtracted[:k2ann_bin])*resolution_meters
            energy_map[lat_band,lon_band,mnth,1] = np.sum(ke_subtracted[k0ann_bin:k1ann_bin])*resolution_meters
            energy_map[lat_band,lon_band,mnth,2] = np.sum(ke_subtracted[k1ann_bin:k2ann_bin])*resolution_meters
            energy_map[lat_band,lon_band,mnth,3] = np.sum(ke_subtracted[k0half_bin:k0ann_bin])*resolution_meters
            ## Error, computed statistically. Note that the  error in estimating the subtracted noise is also propagated through.
            energy_map_err[lat_band,lon_band,mnth,0] = np.sqrt(np.sum(ke_spectrum[:k2ann_bin]**2*2/(num_segs_map[lat_band,lon_band,mnth] - 1) + noise_ke[:k2ann_bin]**2)*resolution_meters**2)
            energy_map_err[lat_band,lon_band,mnth,1] = np.sqrt(np.sum(ke_spectrum[k0ann_bin:k1ann_bin]**2*2/(num_segs_map[lat_band,lon_band,mnth] - 1) + noise_ke[k0ann_bin:k1ann_bin]**2)*resolution_meters**2)
            energy_map_err[lat_band,lon_band,mnth,2] = np.sqrt(np.sum(ke_spectrum[k1ann_bin:k2ann_bin]**2*2/(num_segs_map[lat_band,lon_band,mnth] - 1) + noise_ke[k1ann_bin:k2ann_bin]**2)*resolution_meters**2)
            energy_map_err[lat_band,lon_band,mnth,3] = np.sqrt(np.sum(ke_spectrum[k0half_bin:k0ann_bin]**2*2/(num_segs_map[lat_band,lon_band,mnth] - 1) + noise_ke[k0half_bin:k0ann_bin]**2)*resolution_meters**2)
            #KE from fitted noise
            noise_bins[lat_band,lon_band,mnth,0] = np.sum(noise_ke[:k2ann_bin])*resolution_meters
            noise_bins[lat_band,lon_band,mnth,1] = np.sum(noise_ke[k0ann_bin:k1ann_bin])*resolution_meters
            noise_bins[lat_band,lon_band,mnth,2] = np.sum(noise_ke[k1ann_bin:k2ann_bin])*resolution_meters
            noise_bins[lat_band,lon_band,mnth,3] = np.sum(noise_ke[k0half_bin:k0ann_bin])**resolution_meters
            #KE from fitted balanced spectrum
            fitted_energy_map[lat_band,lon_band,mnth,0] = np.sum(fitted_ke_spectrum[:k2ann_bin])*resolution_meters
            fitted_energy_map[lat_band,lon_band,mnth,1] = np.sum(fitted_ke_spectrum[k0ann_bin:k1ann_bin])*resolution_meters
            fitted_energy_map[lat_band,lon_band,mnth,2] = np.sum(fitted_ke_spectrum[k1ann_bin:k2ann_bin])*resolution_meters
            fitted_energy_map[lat_band,lon_band,mnth,3] = np.sum(fitted_ke_spectrum[k0half_bin:k0ann_bin])*resolution_meters
            #Error for fitten energy
            #pd = 'partial derivative' of fitted KE wrt parameters
            pd_ke[0,:] = fitted_ke_spectrum*resolution_meters/A
            pd_ke[1,:] = (grav**2/f**2)*(wn_meters**2*wn**slope/knee**(slope+1))*slope*A*resolution_meters/(1 + (wn/knee)**(slope))**2
            pd_ke[2,:] = - (grav**2/f**2)*(wn_meters**2*wn**slope*resolution_meters/knee**slope)*np.log(wn/knee)*A/(1 + (wn/knee)**(slope))**2
            fitted_energy_err[lat_band,lon_band,mnth,0] = np.sqrt(np.matmul(np.sum(pd_ke,axis=1),np.matmul(C_cost_balanced,np.transpose(np.sum(pd_ke,axis=1)))))
            fitted_energy_err[lat_band,lon_band,mnth,1] = np.sqrt(np.matmul(np.sum(pd_ke[:,k0ann_bin:k1ann_bin],axis=1),np.matmul(C_cost_balanced,np.transpose(np.sum(pd_ke[:,k0ann_bin:k1ann_bin],axis=1)))))
            fitted_energy_err[lat_band,lon_band,mnth,2] = np.sqrt(np.matmul(np.sum(pd_ke[:,k1ann_bin:k2ann_bin],axis=1),np.matmul(C_cost_balanced,np.transpose(np.sum(pd_ke[:,k1ann_bin:k2ann_bin],axis=1)))))
            fitted_energy_err[lat_band,lon_band,mnth,3] = np.sqrt(np.matmul(np.sum(pd_ke[:,k0half_bin:k0ann_bin],axis=1),np.matmul(C_cost_balanced,np.transpose(np.sum(pd_ke[:,k0half_bin:k0ann_bin],axis=1)))))
            #print(energy_map_err[lat_band,lon_band,mnth,2]/fitted_energy_err[lat_band,lon_band,mnth,2],energy_map[lat_band,lon_band,mnth,2]/fitted_energy_map[lat_band,lon_band,mnth,2])
            
            #print(np.sqrt(np.sum(np.sum(pd_ke[:,k1ann_bin:k2ann_bin],axis=1),np.sum(C_cost_balanced,np.sum(pd_ke[:,k1ann_bin:k2ann_bin],axis=1)[np.newaxis,:],axis=1),axis=0)))
#            pd_ke[0,:] = KE/resd.x[0]
#            pd_ke[1,:] = (wn_meters**2*wn**resd_month.x[2]/resd_month.x[1]**(resd_month.x[2]+1))*resd_month.x[2]*resd_month.x[0]/(1 + (wn/resd_month.x[1])**(resd_month.x[2]))**2
#            pd_ke[2,:] = - (wn_meters**2*wn**resd_month.x[2]/resd_month.x[1]**resd_month.x[2])*np.log(wn/resd_month.x[1])*resd_month.x[0]/(1 + (wn/resd_month.x[1])**(resd_month.x[2]))**2
            
            
fout.close()
            
param_map.dump('jason_map_parameters_4deg.npy')
ann_param_map.dump('jason_ann_parameters_4deg.npy')
error_map.dump('error_map_4deg.npy')
mld_map.dump('mld_map_4deg.npy')
tides_yes.dump('tides_yes_4deg.npy')
energy_map.dump('energy_map_4deg.npy')
num_passes_map.dump('num_passes_4deg.npy')
energy_map_err.dump('energy_map_err_4deg.npy')
noise_bins.dump('noise_bins_4deg.npy')
fitted_energy_map.dump('fitted_energy_4deg.npy')
fitted_energy_err.dump('fitted_energy_err_4deg.npy')
#KE_datamap.dump('KE_datamap.npy')
