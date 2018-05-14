__author__ = 'omniscope+jline'
import numpy as np
import optparse, sys, os
import healpy as hp

import matplotlib
##Protects clusters where no $DISPLAY is set when running PBS/SLURM
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib.pyplot import close
from os import environ

MAJICK_DIR = environ['MAJICK_DIR']


labels = ['Synchrotron', 'CMB', 'HI', 'Dust1', 'Dust2', 'Free-Free']
n_comp = len(labels)
kB = 1.38065e-23
C = 2.99792e8
h = 6.62607e-34
T = 2.725
hoverk = h / kB

def K_CMB2MJysr(K_CMB, nu):#in Kelvin and Hz
    B_nu = 2 * (h * nu)* (nu / C)**2 / (np.exp(hoverk * nu / T) - 1)
    conversion_factor = (B_nu * C / nu / T)**2 / 2 * np.exp(hoverk * nu / T) / kB
    return  K_CMB * conversion_factor * 1e20#1e-26 for Jy and 1e6 for MJy

def K_RJ2MJysr(K_RJ, nu):#in Kelvin and Hz
    conversion_factor = 2 * (nu / C)**2 * kB
    return  K_RJ * conversion_factor * 1e20#1e-26 for Jy and 1e6 for MJy

def generate_gsm_2016(freq=None,this_date=None,observer=None,xsize=225):
    '''Generates an orthographic view of the 2016 GSM for a given
    observer, for a given date, and given frequency (MHz)'''
    
    freq *= 1e-9
    
    ##This is directly lifted from github code - dunno what it's doing!
    nside = 1024
    if freq < 1000:
        op_resolution = 48
    else:
        op_resolution = 24
        
    map_ni = np.array([np.fromfile('%s/imager_lib/gsm_2016_data/highres_%s_map.bin' %(MAJICK_DIR,lb), dtype='float32') for lb in labels])
    spec_nf = np.loadtxt('%s/imager_lib/gsm_2016_data/spectra.txt' %MAJICK_DIR)
    nfreq = spec_nf.shape[1]

    left_index = -1
    for i in range(nfreq - 1):
        if freq >= spec_nf[0, i] and freq <= spec_nf[0, i + 1]:
            left_index = i
            break
    if left_index < 0:
        print "FREQUENCY ERROR: %.2e GHz is outside supported frequency range of %.2e GHz to %.2e GHz."%(freq, spec_nf[0, 0], spec_nf[0, -1])
        
        
    interp_spec_nf = np.copy(spec_nf)
    interp_spec_nf[0:2] = np.log10(interp_spec_nf[0:2])
    x1 = interp_spec_nf[0, left_index]
    x2 = interp_spec_nf[0, left_index + 1]
    y1 = interp_spec_nf[1:, left_index]
    y2 = interp_spec_nf[1:, left_index + 1]
    x = np.log10(freq)
    interpolated_vals = (x * (y2 - y1) + x2 * y1 - x1 * y2) / (x2 - x1)
    result = np.sum(10.**interpolated_vals[0] * (interpolated_vals[1:, None] * map_ni), axis=0)
    
    ##Put into healpix ring order, works with the orthographic projection
    result = hp.reorder(result, n2r=True)

    ##format the date, and change the date of the observer to now
    date,time = this_date.split('T')
    observer.date = '/'.join(date.split('-'))+' '+time

    ##Here be code that I lifted from PyGSM
    n_pix  = hp.get_map_size(result)
    n_side = hp.npix2nside(n_pix)

    theta, phi = hp.pix2ang(n_side, np.arange(n_pix))


    # Get RA and DEC of zenith
    ra_rad, dec_rad = observer.radec_of(0, np.pi/2)
    ra_deg  = ra_rad / np.pi * 180
    dec_deg = dec_rad / np.pi * 180


    # Apply rotation
    hrot = hp.Rotator(rot=[ra_deg, dec_deg], coord=['G', 'C'], inv=True)
    g0, g1 = hrot(theta, phi)
    pix0 = hp.ang2pix(n_side, g0, g1)
    sky_rotated = result[pix0]

    ## Generate a mask for below horizon
    #mask1 = phi + np.pi / 2 > 2 * np.pi
    #mask2 = phi < np.pi / 2
    #mask = np.invert(np.logical_or(mask1, mask2))

    #observed_sky = hp.ma(sky_rotated)
    #observed_sky.mask = mask

    ###The original GSM is set at 0.8 deg so stick with that - 180.0 / 0.8 = 225 pixels
    #sky_view = hp.orthview(observed_sky, half_sky=True, return_projected_map=True,xsize=xsize) #xsize=IMAGE_SIZE
    #close()
    ##plt.show()

    #sky_view.set_fill_value(0.0)

    #l_reso = 2.0 / sky_view.shape[0]

    ###resolution is hard coded to 0.8 deg
    ### * 1e+6 because in MJy - then need to convert from sterrad**-1 to pixel**-1
    ### the observers sky covers 2pi sterrads
    ### there are pi*(d/2)**2 pixels in sky, where d is width of image
    
    #sky_view = np.array(sky_view.filled()) * 1e+6 * ((2*np.pi) / (np.pi * (float(sky_view.shape[0])/2.0)**2))
    
    #sky_view[sky_view == -np.inf] = 0
    #sky_view = sky_view[:,::-1]
    
    
    
    #return sky_view, l_reso
    
    #Need to convert from sterrad**-1 to pixel**-1
    pixel_area = hp.nside2pixarea(nside)
    sky_rotated *= pixel_area
    
    ##Then convert from MJy to Jy
    sky_rotated *= 1e+6
    
    return nside,sky_rotated