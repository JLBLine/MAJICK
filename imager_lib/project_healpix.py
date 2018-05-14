'''Functions to take healpix images and project them to l,m space '''
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import healpy as hp
from os import environ
import pickle
from numpy import pi,inf,invert,logical_or,array,arange,where
from matplotlib.pyplot import close

MAJICK_DIR = environ['MAJICK_DIR']
with open('%s/imager_lib/MAJICK_variables.pkl' %MAJICK_DIR) as f:  # Python 3: open(..., 'rb')
    D2R, R2D, VELC, MWA_LAT, KERNEL_SIZE, W_E, SOLAR2SIDEREAL = pickle.load(f)
    
    
def convert_healpix2lm(healpix_array=None,observer=None,max_uv=None):
    '''Takes a healpix array which is in celestial coords and an ephem observer
    (which should alread have the correct date within) and spits out an all sky 
    image of the observer's sky, with a resolution such that the maximum uv point 
    can be sampled upon an FT of the image'''
    
    n_side = hp.npix2nside(len(healpix_array))
    theta, phi = hp.pix2ang(n_side, arange(len(healpix_array)))
    
    # Get RA and DEC of zenith
    ra_rad, dec_rad = observer.radec_of(0, pi/2)
    ra_deg  = ra_rad * R2D
    dec_deg = dec_rad * R2D

    # Apply rotation
    hrot = hp.Rotator(rot=[ra_deg, dec_deg], inv=True)
    g0, g1 = hrot(theta, phi)
    pix0 = hp.ang2pix(n_side, g0, g1)
    sky_rotated = healpix_array[pix0]

    ## Generate a mask for below horizon
    #mask1 = phi + pi / 2 > 2 * pi
    #mask2 = phi < pi / 2
    #mask = invert(logical_or(mask1, mask2))

    #observed_sky = hp.ma(sky_rotated)
    #observed_sky.mask = mask
    
    observed_sky = sky_rotated
    
    ##Max uv is absolute distance on the uv plane and so
    
    #if xsize % 2 == 0:
        #xsize += 1
        #l_reso = 2.0 / xsize
        
    min_lreso = 1.0 / (2*max_uv)
    min_xsize = 2.0 / min_lreso
    
    ##TODO Make this a GPU option only?
    ##FFTs work best when powers of two (at least for GPUs)
    xsizes = [2**power for power in arange(30)]
    xsize = None
    for ind in arange(len(xsizes)-1):
        low_xsize = xsizes[ind]
        high_xsize = xsizes[ind+1]
        if low_xsize < min_xsize and high_xsize > min_xsize:
            xsize = high_xsize
    
    if xsize % 2 == 0:
        l_reso = 2.0 / (xsize + 1)
    else:
        l_reso = 2.0 / xsize
    
    sky_view = hp.orthview(observed_sky, half_sky=True, return_projected_map=True,xsize=xsize) #xsize=IMAGE_SIZE
    close()





    ##resolution is hard coded to 0.8 deg
    ## * 1e+6 because in MJy - then need to convert from sterrad**-1 to pixel**-1
    ## the observers sky covers 2pi sterrads
    ## there are pi*(d/2)**2 pixels in sky, where d is width of image
    
    #sky_view = array(sky_view.filled()) * 1e+6 * ((2*pi) / (pi * (float(sky_view.shape[0])/2.0)**2))
    sky_view[sky_view == -inf] = sky_view[where(sky_view != -inf)].mean()
    #sky_view = sky_view[:,::-1]
    
    return sky_view,l_reso

def find_healpix_zenith_offset(nside=None,observer=None):
    '''Takes a heapix nside, and for a celestial projection, finds the
    closest pixel to zenith for the given ephem Observer. Returns the ra and
    dec offset to the closest pixel in degress'''
    
    # Get RA and DEC of zenith
    ra_rad, dec_rad = observer.radec_of(0, pi/2)
    ra_zen  = ra_rad * R2D
    dec_zen = dec_rad * R2D
    
    #print('ra_zen,dec_zen',ra_zen,dec_zen)
    
    ##Find nearest healpixel
    pixel = hp.ang2pix(nside,dec_rad+pi/2,ra_rad)
    
    ##Find ra/dec of heallpixel
    dec,ra = hp.pix2ang(nside,pixel)
    ra_pix,dec_pix = ra*R2D,dec*R2D-90.0
    #print('ra_pix,dec_pix',ra_pix,dec_pix)
    #print(ra_pix-ra_zen, dec_pix-dec_zen)
    ##Return the difference
    return ra_pix-ra_zen, dec_pix-dec_zen
    
    #pix = hp.pix2ang(nside)
