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
    ra_deg  = ra_rad / pi * 180
    dec_deg = dec_rad / pi * 180

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
    l_reso = 1.0 / (2*max_uv)
    xsize = 2.0 / l_reso
    
    #if xsize % 2 == 0:
        #xsize += 1
        #l_reso = 2.0 / xsize
        
    xsize = 2**14
    l_reso = 2.0 / xsize
    
    ##The original GSM is set at 0.8 deg so stick with that - 180.0 / 0.8 = 225 pixels
    sky_view = hp.orthview(observed_sky, half_sky=True, return_projected_map=True,xsize=xsize) #xsize=IMAGE_SIZE
    close()
    #plt.show()

    #sky_view.set_fill_value(sky_view.mean())

    #l_reso = 2.0 / sky_view.shape[0]

    ##resolution is hard coded to 0.8 deg
    ## * 1e+6 because in MJy - then need to convert from sterrad**-1 to pixel**-1
    ## the observers sky covers 2pi sterrads
    ## there are pi*(d/2)**2 pixels in sky, where d is width of image
    
    #sky_view = array(sky_view.filled()) * 1e+6 * ((2*pi) / (pi * (float(sky_view.shape[0])/2.0)**2))
    
    sky_view[sky_view == -inf] = sky_view[where(sky_view != -inf)].mean()
    #sky_view = sky_view[:,::-1]
    
    return sky_view,l_reso
