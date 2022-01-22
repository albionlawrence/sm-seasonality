#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jason_annspectrum.py lat lon region

Plots the annual spectrum and the fit to Callies and Wu's model 
(with tides) from 

http://dx.doi.org/10.1175/JPO-D-18-0272.1

for a patch centered
on (lat N, lon E)

Uses output from jason_patch_spectrum_all.py
and from monthly_speed.py (thus this is
                           for grid squares from the
                           nonoverlaping (8deg)^2 grid
                           between 60S and 60N; for the
                           overlapping grids used in the paper you
                           would have to tweak to take output from
                           monthly_speed_4deg.py)


Created on Tue Mar 19 09:19:02 2019

@author: albion modified by joernc
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.io import loadmat
#import matplotlib.cm as cm
from scipy.optimize import least_squares
from scipy.stats import chi2
#import netCDF4 as nc
#import seaborn as sns
#from scipy.optimize import least_squares

lat = int(sys.argv[1])
lon = int(sys.argv[2])
#month = int(sys.argv[3])
region = sys.argv[3]

M2 = 2*np.pi/(12.42*3600)
f_inertial = 2*np.pi*np.sin(2*np.pi*lat/360)/(12*3600)
#months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

spec = loadmat('../submesoscale_seasonality_data/patch_spectra/jason_lat{:n}_lon{:n}_8deg_unfiltered.mat'.format(lat,lon))
cns = np.load('../submesoscale_seasonality_data/vertical_modes/cn_new.npy',encoding='latin1',allow_pickle=True)

x_int = int(lon/8)
y_int = int((lat+60)/8)
#wavenumbers of mode-1 and -2 M2 tides
k1 = 1000*np.sqrt(M2**2 - f_inertial**2)/(2*np.pi*cns[x_int,y_int,0])
k2 = 1000*np.sqrt(M2**2 - f_inertial**2)/(2*np.pi*cns[x_int,y_int,1])

#pass_spectrum_cx = spec['ps_cx'][:,1:]
#box_spectrum = np.mean(spec['ps_cx'][:,1:],axis=0)
ps_all = spec['ps_all'][:,:,1:]
dx = spec['dx'][0][0]
num_segs = spec['num_segs']
good_cycles = spec['good_cycles']
pass_list = spec['pass_list'][0,:]
high_cycle = spec['high_cycle'][0][0]
low_cycle = spec['low_cycle'][0][0]
fft_len = ps_all.shape[2] + 1
nbins = ps_all.shape[2]

dx = spec['dx'][0][0]
k_n = 1/(2*dx)
wn = np.arange(1,fft_len)/(2*dx*(fft_len-1))
#pass_list = spec['pass_good'][0]
#bins/array elements conatining tidal peaks.
k1_bin = int(k1*(nbins+1)/k_n) - 1 #(nbins)+1 = length of fft array; -1 shift because we drop lowest point
k2_bin = int(k2*(nbins+1)/k_n) - 1
track_length = ps_all.shape[2]*2
dk = 1/(track_length*dx)

#Generate annual spectrum
annual_spectrum = np.zeros(fft_len-1)
num_data = 0

for k, p in enumerate(pass_list):
    cycle_indx = np.nonzero(good_cycles[k,:])[0]
    for n,cycle in enumerate(cycle_indx):
        annual_spectrum += ps_all[k,cycle,:]
        num_data += 1

annual_spectrum /= num_data
hi_CL = num_data*annual_spectrum/chi2.interval(.95,num_data)[0]
lo_CL = num_data*annual_spectrum/chi2.interval(.95,num_data)[1]


k_n = 1/(2*dx)
k1_bin = int(k1*(nbins+1)/k_n) - 1 #(nbins)+1 = length of fft array; -1 shift because we drop lowest point
k2_bin = int(k2*(nbins+1)/k_n) - 1

def model_func(x):
    """
    function used to fit data for k^(-s) slope
    """
    return x[0]/(1 + (wn/x[1])**x[2]) + x[4]*np.exp(-(wn - k1)**2/(2*x[6]**2)) + x[5]*np.exp(-(wn - k2)**2/(2*x[7]**2)) + x[3]

def resid_fn(x,data,ndat):
    """
    Residuals for nonlinear east squares, with k^{-s} slope
    """
    return (model_func(x)/np.abs(data) - 1)*np.sqrt(ndat - 1)*np.sqrt(resolution)/np.sqrt(wn)

resolution = k_n/(nbins+1)
lower_bounds = np.array([0.,0.001,0.,0.,0.,0.,resolution/2,resolution/2])
upper_bounds = np.array([np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,resolution*2,resolution*2])

x_init = np.array([annual_spectrum[0],.001,4.,.001,.01,.01,resolution,resolution])
resd_annual = least_squares(resid_fn, x_init, args = (annual_spectrum,num_data),bounds = (lower_bounds,upper_bounds))
fit_array_annual = resd_annual.x
jac_cost = resd_annual.jac
jac_cost_mod = jac_cost*resolution/wn[:,np.newaxis]
Hess_cost = np.matmul(np.transpose(jac_cost),jac_cost)
Hess_mod = np.matmul(np.transpose(jac_cost),jac_cost_mod)
try:
    C_cost = np.linalg.inv(Hess_cost)
except np.linalg.linalg.LinAlgError as err:
    C_cost = np.zeros(Hess_cost.shape)
C_mod = np.matmul(np.matmul(C_cost,Hess_mod),np.transpose(C_cost))
slope = fit_array_annual[2]
slope_err = np.sqrt(C_mod[2,2])

#output annual slope 
print('annual slope {:f} +/- {:f}'.format(slope,slope_err))

lat_s = "{:d}°N".format(lat) if lat > 0 else "{:d}°S".format(-lat)
lon_s = "{:d}°E".format(lon) if lon < 180 else "{:d}°W".format(360-lon)

plt.figure(figsize=(4.6, 4.3))

plt.loglog(wn, annual_spectrum, color='tab:blue', label='data', zorder=1)
plt.loglog(wn, model_func(fit_array_annual), color='tab:orange', label='model fit', zorder=2)
plt.fill_between(wn, lo_CL, hi_CL, color="tab:blue", alpha=0.3, linewidths=0, zorder=1)
plt.xlabel('wavenumber (cpkm)')
plt.ylabel('SSH power spectral density (m$^2$ cpkm$^{-1}$)')
plt.axvline(k1, 0.05, 0.1, color='black')
plt.axvline(k2, 0.05, 0.1, color='black')
plt.legend(frameon=False)
plt.title('{:s} ({:s} {:s})'.format(region, lat_s, lon_s))
plt.xlim(1e-3, 1e-1)

plt.tight_layout()
if lat == 16 and lon == 196: plt.savefig("fig/annual_spectrum_hr.pdf")

plt.show()
