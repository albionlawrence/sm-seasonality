#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monthly_speed.py

Computes speed of sound month-by-month over global ocean, for
analyzing SSH data, based on non-overlapping (8deg)^2 grid
between 60S and 60N.

Code adopted from that generously provided by Weiguang Wu

Created on Fri Feb  8 14:58:45 2019

@author: albion
"""
#from netCDF4 import Dataset#, num2date
import numpy as np
import netCDF4 as nc
#import matplotlib.pyplot as plt
#import scipy.io as sio
import gsw
import scipy.sparse as sps
from scipy.sparse.linalg import eigs
#from mpl_toolkits.basemap import Basemap

#load monthly ECCO version4, release3, climatology
nc1 = nc.Dataset('SALT.0001.nc','r')
nc2 = nc.Dataset('THETA.0001.nc','r')
## mask nan value
SP = np.ma.masked_invalid(nc1.variables['SALT'][:])
theta = np.ma.masked_invalid(nc2.variables['THETA'][:])

#arrays for converting the -180 to 180 longitude system
#into 0 to 360 longitude system
#note this has half degree resolution
SP_cor = np.ma.masked_all([50,360,720])
theta_cor = np.ma.masked_all([50,360,720])

lat_md = nc1.variables['lat'][:]#coordinates
lon_md = nc1.variables['lon'][:]
depth = nc1.variables['dep'][:]

lon_cor = np.ma.masked_all([360,720])
lon_cor[:,:360] = lon_md[:,360:]
lon_cor[:,360:] = lon_md[:,:360]+360

# set the empty matrix for global phase speed, cn, for first three modes
# and the vertical mode distribution, f0, at sea surface
f0_global = np.ma.masked_all([3,360,720])
cn_global = np.ma.masked_all([3,360,720])

# within 8*8 boxes, the median of all data is used as the one for the box.
#(lonblock,latblock,month,mode)
cn_global_median = np.ma.masked_all([45,15,12,3])
f0_global_median = np.ma.masked_all([45,15,12,3])

for month in range(12):
    SP_cor[:,:,:360] = SP[month,:,:,360:] #salt
    SP_cor[:,:,360:] = SP[month,:,:,:360]
    theta_cor[:,:,:360] = theta[month,:,:,360:] #temperature
    theta_cor[:,:,360:] = theta[month,:,:,:360]
    for x,lat in enumerate(lat_md[:,0]):#x will be an integer counting up from 1, lat the latitude
        for y,lon in enumerate(lon_cor[0,:]):
        
            SP_1 = SP_cor[:,x,y][::-1] #salt
            theta_1 = theta_cor[:,x,y][::-1]
            length = np.sum(~SP_1.mask)
            #length of the valid data

            if length > 10: # corresponding depth around 500m, discard shallow water profiles
                SP_2 = SP_1[~SP_1.mask]
                theta_2 = theta_1[~theta_1.mask]
                depth_2 = depth[:length] # depth of the ocean at x,y
                z = -depth_2[::-1] #set upward z-axis
                
                #calculate buoyancy from potential density referenced to 2000 m
                p = gsw.p_from_z(z,lat)
                g = gsw.grav(lat, p)
                sa = gsw.SA_from_SP(SP_2,p,lon,lat)
                ct = gsw.CT_from_pt(sa, theta_2)
                rho = gsw.sigma2(sa, ct) + 1000
                rho_mean = np.mean(rho)
                
                b = -g*(rho - rho_mean)/rho_mean
                
                # augment z with ghost points
                z = np.hstack(([2*z[0]-z[1]], z, [-z[-1]]))
                # set up matrix
                ap = 2/(z[2:-1]-z[:-3])/(b[1:]-b[:-1])
                am = 2/(z[3:]-z[1:-2])/(b[1:]-b[:-1])
                A = sps.diags((ap, -np.hstack((ap, [0])) -np.hstack(([0], am)), am), offsets=(1, 0, -1))
                # compute eigenvalues and eigenvectors for first 5 modes
                w, v = eigs(A, k=5, sigma=-np.pi**2/z[0]**2)
                v = np.real(v)
                w = np.real(w)
                idx = w.argsort()[::-1]
                w = w[idx]
                #modal phase speed for first three modes
                cn = np.sqrt(-1/w)
                cn_global[:,x,y] = cn[1:4]
                
    
                # normalize modes
                v = v[:,idx]
                v /= np.sqrt(np.trapz(v**2, x=z[1:-1], axis=0))
                v *= np.sign(v[-1,:])
                # mode expression at sea surface
                f0_global[:,x,y] = v[-1,1:4]
                
                # mask the physically unreasonable value (>4) and nan value
                cn_1_global= np.ma.masked_where(np.ma.masked_invalid(cn_global)>4., np.ma.masked_invalid(cn_global))

                # mask all first three modes, if any of the first three mode is masked by the reason above.
                cn_2_global = np.ma.masked_all([3,360,720])

                for x in range(360):
                    for y in range(720):
                        modes = cn_1_global[:,x,y]
                        if np.ma.is_masked(modes) == False:
                            cn_2_global[:,x,y] = modes
                
                # mask the surface expression of modes in the same way
                f0_1_global = np.ma.masked_where(cn_2_global.mask,f0_global)

                # split data into 8*8 grid boxes from latitude 60S to 60N
                c = np.ma.array(np.split(np.ma.array(np.split(cn_2_global[:,60:-60,:],15,axis = 1)),45, axis = 3))

                f = np.ma.array(np.split(np.ma.array(np.split(f0_1_global[:,60:-60,:],15,axis = 1)),
                  45, axis = 3))

                for x in range(45):
                    for y in range(15):
                        cn_global_median[x,y,month,0] = np.ma.median(c[x,y,0,:,:])
                        cn_global_median[x,y,month,1] = np.ma.median(c[x,y,1,:,:])
                        cn_global_median[x,y,month,2] = np.ma.median(c[x,y,2,:,:])
        
                for x in range(45):
                    for y in range(15):
                        f0_global_median[x,y,month,0] = np.ma.median(f[x,y,0,:,:])
                        f0_global_median[x,y,month,1] = np.ma.median(f[x,y,1,:,:])
                        f0_global_median[x,y,month,2] = np.ma.median(f[x,y,2,:,:])

#np.save('cn_monthly.npy',cn_global_median)
#np.save('f0_monthly.npy',f0_global_median)

cn_global_median.dump('cn_monthly')
f0_global_median.dump('f0_monthly')



