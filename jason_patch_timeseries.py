#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jason_patch_timeseries lat lon region

Extracts monthly spectrum, spatial average for 8 deg^2 box centered at lat N
lon E; region is name to put on plots. 
Fits model parameters (from Callies and Wu 
https://journals.ametsoc.org/view/journals/phoc/49/9/jpo-d-18-0272.1.xml)
without tides.  Plots with error bars

Created on Aug 12, 2021. Modified from previous code.

Locations:
north pacific lat 12-20, lon 196 =/- 4
MAR 28-36, LON 319 +/- 4
Gulf 28-36, LON 290 +/- 4
Gulf2 33-41, LON 299 +/- 4
kuroshio1 32-40, LON 154 +/- 4

@author: Albion Lawrence (with some modifications by Joern Callies)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.io import loadmat
import matplotlib.cm as cm
from scipy.optimize import least_squares
import netCDF4 as nc
from datetime import datetime, timedelta

##input parameters
lat = int(sys.argv[1])
lon = int(sys.argv[2])
region = sys.argv[3] #Name of region for plot
##Matrix positions
x_int = int(lon/8)
y_int = int((lat+60)/8)
##Geophysical parameters.
M2 = 2*np.pi/(12.42*3600)
grav = 9.8
f = 2*np.pi*np.sin(2*np.pi*lat/360)/(12*3600)
#months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
months = ['J','F','M','A','M','J','J','A','S','O','N','D']
#seasons = ['JFM','AMJ','JAS','OND']
#seasons = ['FMA','MJJ','ASO','NDJ']
#seasons = ['MAM','JJA','SOD','NJF']

#Load spectrum for patch, and ECCO mixed layer depth
spec = loadmat('../submesoscale_seasonality_data/patch_spectra/jason_lat{:n}_lon{:n}_8deg_unfiltered.mat'.format(lat,lon))
ecco_data = nc.Dataset('../submesoscale_seasonality_data/ECCO-4/MXLDEPTH.0001.nc','r')
ecco_lat = ecco_data.variables['lat'][:,:]
ecco_lon = ecco_data.variables['lon'][:,:]
ecco_lon += 360*(1 - np.sign(ecco_lon))/2
ecco_mld = ecco_data.variables['MXLDEPTH'][:,:,:]

#Load climatology for Significant Wave Height
swh_data = nc.Dataset('../submesoscale_seasonality_data/SWH_climatology/era5_swh.nc','r')
swh_lat = swh_data['latitude'][:]
swh_lon = swh_data['longitude'][:]
swh_time = swh_data['time'][:]
swh = swh_data['swh'][:,0,:,:]
#k1 = 1000*np.sqrt(M2**2 - f**2)/(2*np.pi*cns[x_int,y_int,0])
#k2 = 1000*np.sqrt(M2**2 - f**2)/(2*np.pi*cns[x_int,y_int,1])

ps_all = spec['ps_all'][:,:,1:]#Note we are ignoring the zeo-wavenumber bin
dx = spec['dx'][0][0]
num_segs = spec['num_segs']
good_cycles = spec['good_cycles']
pass_list = spec['pass_list'][0,:]
month_list = spec['month_list']
high_cycle = spec['high_cycle']
low_cycle = spec['low_cycle']
constant_mode_array = spec['constant_mode_array']

##Additional parameters for Fourier transform
fft_len = ps_all.shape[2] + 1
k_n = 1/(2*dx)
wn = np.arange(1,fft_len)/(2*dx*(fft_len-1))
wn_meters = wn/1000 #wn is given in km^{-1}
nbins = np.size(wn)
resolution = k_n/nbins


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
cm_good = constant_mode_array[good_passes,:]

monthly_spectrum_by_pass = np.zeros((12,num_good_passes,fft_len-1))
monthly_constant_by_pass = np.zeros((12,num_good_passes))

#Now let's generae monthly spectra and averages for "good" months
for k, p in enumerate(pass_list_good):
    cycle_indx = np.nonzero(good_cycles_good_passes[k,:])[0]
    for n,cycle in enumerate(cycle_indx):
        monthly_spectrum_by_pass[int(month_list_good[k,cycle]),k,:] += ps_good[k,cycle,:]
        monthly_constant_by_pass[int(month_list_good[k,cycle]),k] += cm_good[k,cycle]
        
monthly_spectrum = 2*np.sum(monthly_spectrum_by_pass,axis=1)/np.sum(num_good_segs[:,:,np.newaxis],axis=1)
monthly_constant = np.sum(monthly_constant_by_pass,axis=1)/np.sum(num_good_segs[:,:],axis=1)
monthly_constant_trackwise = monthly_constant_by_pass/num_good_segs

#Compute standard deviations
monthly_constant_sd = np.zeros((12))
monthly_constant_trackwise_sd = np.zeros((12,num_good_passes))
num_month_by_pass = np.zeros((12,num_good_passes))
for k, p in enumerate(pass_list_good):
    cycle_indx = np.nonzero(good_cycles_good_passes[k,:])[0]
    for n,cycle in enumerate(cycle_indx):
        mn = int(month_list_good[k,cycle])
        monthly_constant_sd[mn] += (monthly_constant_trackwise[mn,k] - monthly_constant[mn])**2
        monthly_constant_trackwise_sd[mn,k] += (monthly_constant_trackwise[mn,k] - monthly_constant[mn])**2
        num_month_by_pass[mn,k] += 1
num_month = np.sum(num_month_by_pass,axis=1)
monthly_constant_sd = np.sqrt(monthly_constant_sd/(num_month-1))
monthly_constant_trackwise_sd = np.sqrt(monthly_constant_trackwise_sd/(num_month_by_pass - 1))


amplitude = np.zeros(12)
slope = np.zeros(12)
monthly_mld = np.zeros(12)
jac_cost = np.zeros((nbins,8))
Hess_cost = np.zeros((8,8))
C_cost = np.zeros((8,8))
slope_err = np.zeros(12)
amp_err = np.zeros(12)
k0_err= np.zeros(12)
noise_err = np.zeros(12)
resolution = k_n/nbins
#resm = resolution/1000


#model functions and residuals from Callies and Wu

def model_notides(x):
    """
    Model shape without the tidal components
    """
    return x[0]/(1 + (wn/x[1])**x[2]) + x[3]

def model_balanced(x):
    """
    Model shape without the tidal components or the noise
    """
    return x[0]/(1 + (wn/x[1])**x[2])


def resid_notides(x,data,ndat):
    """
    Residuals for nonlinear east squares, with k^{-s} slope. Note the weighting by 1/wn to
    favor low wavenumers in fit
    """
    return (model_notides(x)/np.abs(data) - 1)*np.sqrt(ndat - 1)*np.sqrt(resolution)/np.sqrt(wn)

# def loglikelihood_tides(x,data,ndat):
#     return - np.sum((ndat - 1)*(np.abs(data)/model_func(x) - 1)**2)

def SSH_balanced(x,A,knee,slope):
    """
    Model SSH without the tidal components or the noise. Note that x is in 
    units of 1/km, but A is in units of m^2/cpkm
    """
    return A*(grav**2/f**2)/(1 + (x/knee)**slope)

##compute annual average Fourier spectrumto find wavenumber bins for bandpassed KE
ann_spectrum = np.zeros(fft_len-1)
num_segs_ann = 0

for k, p in enumerate(pass_list):
    cycle_indx = np.nonzero(good_cycles[k,:])[0]
    for n,cycle in enumerate(cycle_indx):
        ann_spectrum += ps_all[k,cycle,:]
        num_segs_ann += 1

ann_spectrum /= num_segs_ann

##Fit to tide-free model and compute locations of wavenumber bins
lower_bounds_ntann = np.array([0.,0.,0.,0.])
upper_bounds_ntann = np.array([np.inf,np.inf,np.inf,np.inf])
x_init_ntann = np.array([ann_spectrum[0],.001,4.,.001])
resd_ntann = least_squares(resid_notides,x_init_ntann,args=(ann_spectrum,num_segs_ann),bounds=(lower_bounds_ntann,upper_bounds_ntann))
ntann_param = resd_ntann.x
k0_ann = ntann_param[1]
k0ann_bin = int(k0_ann*(nbins)/k_n) - 1
k0_half = k0_ann/2
k0half_bin = int(k0_half*(nbins)/k_n) - 1
k1_ann = 2*k0_ann #don't confuse with k1, k2 which are todal wavenumbers; these are band edges
k1ann_bin = int(k1_ann*(nbins)/k_n) - 1
k2_ann = 2*k1_ann
k2ann_bin = int(k2_ann*(nbins)/k_n) - 1




lower_bounds_ntann = np.array([0.,0.,0.,0.])
upper_bounds_ntann = np.array([np.inf,np.inf,np.inf,np.inf])
x_init_ntann = np.array([ann_spectrum[0],.001,4.,.001])
resd_ntann = least_squares(resid_notides,x_init_ntann,args=(ann_spectrum,num_segs_ann),bounds=(lower_bounds_ntann,upper_bounds_ntann))
ntann_param = resd_ntann.x
k0_ann = ntann_param[1]
k0ann_bin = int(k0_ann*(nbins)/k_n) - 1
k0_half = k0_ann/2
k0half_bin = int(k0_half*(nbins)/k_n) - 1
k1_ann = 2*k0_ann #don't confuse with k1, k2 which are todal wavenumbers; these are band edges
k1ann_bin = int(k1_ann*(nbins)/k_n) - 1
k2_ann = 2*k1_ann
k2ann_bin = int(k2_ann*(nbins)/k_n) - 1


fit_array = np.zeros((12,4))
lower_bounds_nt = np.array([0.,0.,0.,0.])
upper_bounds_nt = np.array([np.inf,np.inf,np.inf,np.inf])
pd_ke = np.zeros((3,fft_len-1))
energy_bins = np.zeros((12,2)) #0 = k0:2k0, 1 = 2k0:4k0
energy_bins_err = np.zeros((12,2))
noise_bins = np.zeros((12,2))

##Compute fits to Callies-Wu model, with error bars
for mnth in range(12):
    x_init = np.array([monthly_spectrum[mnth,0],.001,4.,.001])
    spectra = monthly_spectrum[mnth,:]
    resd = least_squares(resid_notides, x_init, args = (spectra,num_segs_patch[mnth]),bounds = (lower_bounds_nt,upper_bounds_nt))
    fit_array[mnth,:] = resd.x
    A, knee, slope = resd.x[:3]
    #compute Hessian of cost function
    jac_cost = resd.jac
    jac_mod = jac_cost*resolution/wn[:,np.newaxis]
    #jac_mod, Hess_mod, C_mod are the Jacobian, Hessian, and covariance matrix 
    #for the cost function *without*
    #the 1/wn weighting: this is what I use to compute actual error bars.
    #Reference: Press et al, "Numerical Recipes"
    Hess_cost = np.matmul(np.transpose(jac_cost),jac_cost)
    Hess_mod = np.matmul(np.transpose(jac_cost),jac_mod)
    try:
        C_cost = np.linalg.inv(Hess_cost)
    except np.linalg.linalg.LinAlgError as err:
        C_cost = np.zeros(Hess_cost.shape)
    C_mod = np.matmul(np.matmul(C_cost,Hess_mod),np.transpose(C_cost))
    C_cost_balanced = C_mod[:3,:3]
    #C_cost = np.linalg.inv(Hess_cost)
    slope_err[mnth] = np.sqrt(C_mod[2,2])
    amp_err[mnth] = np.sqrt(C_mod[0,0])
    k0_err[mnth] = np.sqrt(C_mod[1,1])
    noise_err[mnth] = np.sqrt(C_mod[3,3])
    ##compute binned kinetic energy for balanced component and noise component
    fitted_ke = model_balanced(resd.x[:3])*wn_meters**2*grav**2/f**2
    noise_ke = fit_array[mnth,3]*(wn_meters)**2*grav**2/f**2
    #partial defivatives wrt parameters, to compute errors in estimates of KE
    pd_ke[0,:] = fitted_ke*resolution/A
    pd_ke[1,:] = (grav**2/f**2)*(wn_meters**2*wn**slope/knee**(slope+1))*slope*A*resolution/(1 + (wn/knee)**(slope))**2
    pd_ke[2,:] = - (grav**2/f**2)*(wn_meters**2*wn**slope*resolution/knee**slope)*np.log(wn/knee)*A/(1 + (wn/knee)**(slope))**2
    energy_bins[mnth,0] = np.sum(fitted_ke[k0ann_bin:k1ann_bin])*resolution
    energy_bins[mnth,1] = np.sum(fitted_ke[k1ann_bin:k2ann_bin])*resolution
    energy_bins_err[mnth,0] = np.sqrt(np.matmul(np.sum(pd_ke[:,k0ann_bin:k1ann_bin],axis=1),np.matmul(C_cost_balanced,np.transpose(np.sum(pd_ke[:,k0ann_bin:k1ann_bin],axis=1)))))
    energy_bins_err[mnth,1] = np.sqrt(np.matmul(np.sum(pd_ke[:,k1ann_bin:k2ann_bin],axis=1),np.matmul(C_cost_balanced,np.transpose(np.sum(pd_ke[:,k1ann_bin:k2ann_bin],axis=1)))))
    noise_bins[mnth,0] = np.sum(noise_ke[k0ann_bin:k1ann_bin])*resolution
    noise_bins[mnth,1] = np.sum(noise_ke[k1ann_bin:k2ann_bin])*resolution

#extract monthly mixed layer depth
ecco_grid_points = np.transpose(np.nonzero((ecco_lat >= lat - 4) & (ecco_lat <= lat + 4) & (ecco_lon >= lon - 4) & (ecco_lon <= lon + 4)))
num_ecco_points = ecco_grid_points.shape[0]
for pts in range(num_ecco_points):
    monthly_mld += ecco_mld[:,ecco_grid_points[pts,0],ecco_grid_points[pts,1]]/num_ecco_points

#Extract monthly SWH
#convert time into months

start_date = '1900-01-01'
cal_origin = datetime.strptime(start_date,'%Y-%m-%d')
month_array = np.zeros(len(swh_time))
swh_mean = np.zeros(12)
month_count = np.zeros(12)

## Now extract SWH mean on specified 8 degree grid
#find indices with max and min lat; coordinates of this
#file count down from 90 in half degrees
swh_lat_max_index = int((90 - lat - 4)*2)
swh_lat_min_index = int((90 - lat + 4)*2)

if (lon >=4) and (lon <= 355.5):
    swh_lon_min_index = int((lon - 4)*2)
    swh_lon_max_index = int((lon+4)*2)
    swh_grid_ma = swh[:,swh_lat_max_index:(swh_lat_min_index+1),swh_lon_min_index:(swh_lon_max_index+1)]
else:
    swh_lon_min_index = int(((lon - 4) % 360)*2)
    swh_lon_max_index = int(((lon + 4) % 360)*2)
    swh_grid_ma = np.concatenate(swh[:,swh_lat_max_index:(swh_lat_min_index+1),swh_lon_min_index:-1],swh[:,swh_lat_max_index:(swh_lat_min_index+1),0:swh_lon_max_index+1],axis=2)

#create an object array of SWH time slices for each month.
#This will allow us to take nanmeans over all NaNs in space and time.
#(An issue is that for some locations, we will have NaNs over all time at a point in space,
#AND NaNs at all points in space at a given time)
swh_monthly = np.empty(12,object)
swh_grid = swh_grid_ma.filled(fill_value=np.nan)

for n in range(len(swh_time)):
    b = cal_origin + timedelta(hours = int(swh_time[n]))
    month_index = b.month - 1 #reset for Python indexing
    #First check if we are away from longitude = 0    
    if month_count[month_index] == 0.:
        swh_monthly[month_index] = swh_grid[n,:,:]
    else:
        swh_monthly[month_index] = np.dstack((swh_monthly[month_index],swh_grid[n,:,:]))
    month_count[month_index] += 1
    

for month in range(12):
    swh_mean[month] = np.nanmean(swh_monthly[month])

np.savez("../submesoscale_seasonality_data/timeseries/{:+03d}_{:03d}.npz".format(lat, lon),
        region=region,
        monthly_mld=monthly_mld,
        fit_array=fit_array,
        amp_err=amp_err,
        slope_err=slope_err,
        k0_err=k0_err,
        energy_bins=energy_bins,
        energy_bins_err=energy_bins_err,
        noise_bins=noise_bins,
        swh_mean=swh_mean)

#reset so that longitude reported from 180W to 180E instead of 0 to 360E
if lon > 180:
    lon_dir = 'W'
    lon = 360-lon
else:
    lon_dir = 'E'

#set lat to be reported from 90S to 90N instead of [-90,90]
if lat < 0:
    lat_dir = 'S'
else:
    lat_dir = 'N'

plt.figure(figsize=(12,12))
#plt.tight_layout

#sns.set_palette('coolwarm',12)
#colors = cm.twilight(np.linspace(0,1,13))

colors = cm.twilight(np.linspace(0,1,13))

plt.subplot(2,3,1)
plt.plot(np.arange(12), monthly_mld)
plt.title('(a) Mixed Layer Depth (m)')
plt.xticks(np.arange(12),months)

plt.subplot(2,3,2)
plt.errorbar(np.arange(12),fit_array[:,0],amp_err)
plt.xticks(np.arange(12),months)
plt.title('(c) Amplitude $m^2/cpkm$')

ax1 = plt.subplot(2,3,3)
ax1.errorbar(np.arange(12),energy_bins[:,1],energy_bins_err[:,1],label = 'KE2')
ax1.plot(np.arange(12),noise_bins[:,0], label="noise KE")
ax1.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
#plt.yscale('log')
plt.xticks(np.arange(12),months)
plt.title('(e) KE2 $m^2/s^2$')
ax1.legend(loc = 'lower left')
if np.count_nonzero(np.isnan(swh_mean)) < 6:
    ax2= ax1.twinx()
    ax2.plot(np.arange(12),swh_mean, label = "swh (m)",color='red')
    ax2.legend(loc='upper right')

plt.subplot(2,3,4)
plt.errorbar(np.arange(12),fit_array[:,2],slope_err)
plt.xticks(np.arange(12),months)
plt.title('(b) Slope $s$')

plt.subplot(2,3,5)
plt.errorbar(np.arange(12),1/fit_array[:,1],k0_err/fit_array[:,1]**2)
plt.xticks(np.arange(12),months)
plt.title('(d) transition scale $1/k_0$ (km)')

plt.subplot(2,3,6)
plt.errorbar(np.arange(12),energy_bins[:,0],energy_bins_err[:,0])
#plt.plot(np.arange(12),noise_bins[:,1], label="noise")
#plt.yscale('log')
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.xticks(np.arange(12),months)
plt.title('(f) KE1 $m^2/s^2$')

plt.suptitle('{:s} lat {:n}{:s} lon {:n}{:s}'.format(region,np.abs(lat),lat_dir,lon,lon_dir),fontsize=24)


## Time series of constant modes (estimating steric effect)

plt.figure(figsize=(10,15))

plt.subplot(2,1,1)
plt.errorbar(np.arange(12),monthly_constant,monthly_constant_sd)
plt.xticks(np.arange(12),months)
plt.ylabel('Mean SSH in patch (m)')

plt.subplot(2,1,2)
for k in range(num_good_passes):
    plt.errorbar(np.arange(12),monthly_constant_trackwise[:,k],monthly_constant_trackwise_sd[:,k],label=pass_list[k])
plt.xticks(np.arange(12),months)
plt.ylabel('SSH by track (m)')
plt.legend()

plt.suptitle('Mean SSH {:s} lat {:n}{:s} lon {:n}{:s}'.format(region, np.abs(lat),lat_dir,lon,lon_dir),fontsize=24)


##plots I had made for Joern and Daniela

# plt.subplot(2,2,1)
# plt.errorbar(np.arange(12),fit_array[:,2],slope_err,label='slope')
# plt.xticks(np.arange(12),months)
# plt.title('(a) Slope $s$ AND MLD (m)')
# ax1 = plt.gca()
# ax1.tick_params(axis='y',color='green')
# ax1.set_ylabel('slope s')
# ax1.legend(loc='upper left')
# ax2 = ax1.twinx()
# ax2.plot(np.arange(12), monthly_mld,label = 'MLD',color='green')
# ax2.set_ylabel('MLD(m)$')
# ax2.legend(loc = 'upper right')

# plt.subplot(2,2,2)
# plt.errorbar(np.arange(12),fit_array[:,0],amp_err)
# plt.xticks(np.arange(12),months)
# plt.title('(b) Amplitude $m^2/cpkm$')

# plt.subplot(2,2,3)
# plt.errorbar(np.arange(12),1/fit_array[:,1],k0_err/fit_array[:,1]**2)
# plt.xticks(np.arange(12),months)
# plt.title('(c) transition scale $1/k_0$ (km)')

# plt.subplot(2,2,4)
# plt.errorbar(np.arange(12),fit_array[:,3],noise_err)
# plt.xticks(np.arange(12),months)
# plt.title('(d) Noise scale $m^2/cpkm$')


# plt.suptitle('Model parameters {:s} lat {:n} lon {:n}{:s}'.format(region, lat,lon,lon_dir),fontsize=24)

plt.show()


