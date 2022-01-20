#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 10:11:59 2021

monthly_spectrum.py lat lon 

Plots along-track SSH power spectrum by calendar month for a given 8 degree square patch centered on lat N lon E.
Takes data from jason_patch_spectrum_all.py

@author: albionlawrence
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.io import loadmat
import matplotlib.cm as cm
from scipy.optimize import least_squares

lat = int(sys.argv[1])
lon = int(sys.argv[2])
region = sys.argv[3]
x_int = int(lon/8)
y_int = int((lat+60)/8)
M2 = 2*np.pi/(12.42*3600)
grav = 9.8
f = 2*np.pi*np.sin(2*np.pi*lat/360)/(12*3600)
months = ['J','F','M','A','M','J','J','A','S','O','N','D']
#seasons = ['JFM','AMJ','JAS','OND']
#seasons = ['FMA','MJJ','ASO','NDJ']
#seasons = ['MAM','JJA','SOD','NJF']


##Download spectrum
spec = loadmat('../submesoscale_seasonality_data/patch_spectra/jason_lat{:n}_lon{:n}_8deg_unfiltered.mat'.format(lat,lon))
ps_all = ps_all = spec['ps_all'][:,:,1:]#Note we are ignoring the zero-wavenumber bin
num_segs = spec['num_segs']
good_cycles = spec['good_cycles']
pass_list = spec['pass_list'][0,:]
month_list = spec['month_list']
high_cycle = spec['high_cycle']
low_cycle = spec['low_cycle']

##wavenumber range and resolution
fft_len = ps_all.shape[2] + 1
dx = spec['dx'][0][0]
k_n = 1/(2*dx)
wn = np.arange(1,fft_len)/(2*dx*(fft_len-1))
nbins = np.size(wn)
resolution = k_n/nbins

##Generate monthly spectrum
#enumerate passes which pass a basic data quality criterion
good_passes = np.where(np.amin(num_segs,axis=0) > 4)[0]#Keep pass if every month has at least 4 good tracks
good_passes = np.unique(good_passes) 
pass_list_good = pass_list[good_passes]#array of pass identifiers
month_list_good = month_list[good_passes,:]
num_good_segs = num_segs[:,good_passes]
num_segs_patch = np.sum(num_good_segs,axis=1)
num_good_passes = len(pass_list_good)
good_cycles_good_passes = good_cycles[good_passes,:]
ps_good = ps_all[good_passes,:,:]

monthly_spectrum_by_pass = np.zeros((12,num_good_passes,fft_len-1))

#Now let's generae monthly spectra and averages for "good" months
for k, p in enumerate(pass_list_good):
    cycle_indx = np.nonzero(good_cycles_good_passes[k,:])[0]
    for n,cycle in enumerate(cycle_indx):
        monthly_spectrum_by_pass[int(month_list_good[k,cycle]),k,:] += ps_good[k,cycle,:]

monthly_spectrum = 2*np.sum(monthly_spectrum_by_pass,axis=1)/np.sum(num_good_segs[:,:,np.newaxis],axis=1)


#Compute tidal wavenumbers
cns = np.load('../submesoscale_seasonality_data/vertical_modes/cn_new.npy',encoding='latin1',allow_pickle=True)
k1 = 1000*np.sqrt(M2**2 - f**2)/(2*np.pi*cns[x_int,y_int,0])
k2 = 1000*np.sqrt(M2**2 - f**2)/(2*np.pi*cns[x_int,y_int,1])

def model_notides(x):
    """
    Model shape without the tidal components
    """
    return x[0]/(1 + (wn/x[1])**x[2]) + x[3]

def resid_notides(x,data,ndat):
    """
    Residuals for nonlinear east squares, with k^{-s} slope
    """
    return (model_notides(x)/np.abs(data) - 1)*np.sqrt(ndat - 1)*np.sqrt(resolution)/np.sqrt(wn)

amplitude_init = np.mean(monthly_spectrum[:,0])

lower_bounds_ntann = np.array([0.,0.,0.,0.])
upper_bounds_ntann = np.array([np.inf,np.inf,np.inf,np.inf])
x_init_ntann = np.array([amplitude_init,.001,4.,.001])
fitted_spectrum = np.zeros((12,nbins))

for mnth in range(12):
    #fit spectrum to Joern and Roger's model
    x_init = np.array([monthly_spectrum[mnth,0],.001,4.,.001])
    spectra = monthly_spectrum[mnth,:]
    resd = least_squares(resid_notides, x_init, args = (spectra,num_segs_patch[mnth]),bounds = (lower_bounds_ntann,upper_bounds_ntann))
    #fit_array[mnth,:] = resd.x
    fitted_spectrum[mnth,:] = model_notides(resd.x)

colors = cm.twilight(np.linspace(0,1,13))

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4.2))

for mn in range(12):
    ax[0].loglog(wn,monthly_spectrum[mn,:],color = colors[mn],label=months[mn])
ax[0].set_title("(a)", loc="left")
ax[0].set_title('data')
ax[0].set_xlabel('wavenumber (cpkm)')
ax[0].set_ylabel('SSH power spectral density (m$^2$ cpkm$^{-1}$)')
ax[0].axvline(k1, 0.05, 0.1, color='black')
ax[0].axvline(k2, 0.05, 0.1, color='black')
ax[0].legend(frameon=False, ncol=2)
ax[0].set_xlim(1e-3, 1e-1)

for mn in range(12):
    ax[1].loglog(wn,fitted_spectrum[mn,:],color = colors[mn],label=months[mn])
ax[1].set_title("(b)", loc="left")
ax[1].set_title('model fit')
ax[1].set_xlabel('wavenumber (cpkm)')
ax[1].axvline(k1, 0.05, 0.1, color='black')
ax[1].axvline(k2, 0.05, 0.1, color='black')

#plt.suptitle('{:s} lat {:n}{:s} lon {:n}{:s}'.format(region,lat,lat_dir,lon,lon_dir),fontsize=24)            

fig.tight_layout()

print(lat, lon)
if lat == 32 and lon == 292: plt.savefig("fig/monthly_spectra_gs.pdf")

plt.show()
