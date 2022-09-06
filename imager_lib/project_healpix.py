'''Functions to take healpix images and project them to l,m space '''
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import healpy as hp
from os import environ
import pickle
from numpy import pi,inf,invert,logical_or,array,arange,where,zeros,arcsin,nan
from matplotlib.pyplot import close
import matplotlib.pyplot as plt

MAJICK_DIR = environ['MAJICK_DIR']
with open('%s/imager_lib/MAJICK_variables.pkl' %MAJICK_DIR, 'rb') as f:  # Python 3: open(..., 'rb')
    D2R, R2D, VELC, MWA_LAT, KERNEL_SIZE, W_E, SOLAR2SIDEREAL = pickle.load(f)


def convert_healpix2lm(healpix_array=None,observer=None,max_uv=None,rotate=False,unit='Jy',freq=None):
    '''Takes a healpix array which is in celestial coords and an ephem observer
    (which should alread have the correct date within) and spits out an all sky
    image of the observer's sky, with a resolution such that the maximum uv point
    can be sampled upon an FT of the image'''

    k_B = 1.38064852e-23
    VELC = 299792458.

    nside = hp.get_nside(healpix_array)

    if unit == 'Jy':
        pass
    else:
        solid_angle = hp.nside2pixarea(nside)
        wavelen = VELC / freq
        ##10e-26 because Jy
        healpix_array = (2*float(k_B)*healpix_array*solid_angle) / (wavelen**2*10e-26)

        if unit == 'K':
            pass
        elif unit == 'mK':
            healpix_array *= 1e-3

    theta, phi = hp.pix2ang(nside, arange(len(healpix_array)))

    # Get RA and DEC of zenith
    ra_rad, dec_rad = observer.radec_of(0, pi/2)
    ra_deg  = ra_rad * R2D
    dec_deg = dec_rad * R2D

    # Apply rotation
    if rotate == True:
        hrot = hp.Rotator(rot=[ra_deg, dec_deg], inv=True)
        g0, g1 = hrot(theta, phi)
        pix0 = hp.ang2pix(nside, g0, g1)
        observed_sky = healpix_array[pix0]
        norm = healpix_array.sum() / observed_sky.sum()
        observed_sky *= norm
    else:
        observed_sky = healpix_array

    ## Generate a mask for below horizon
    #mask1 = phi + pi / 2 > 2 * pi
    #mask2 = phi < pi / 2
    #mask = invert(logical_or(mask1, mask2))

    #observed_sky = hp.ma(sky_rotated)
    #observed_sky.mask = mask

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

    # xsize = 501

    sky_view = hp.orthview(observed_sky, half_sky=True, return_projected_map=True,xsize=xsize) #xsize=IMAGE_SIZE
    close()

    sky_view[sky_view == -inf] = 0.0
    # sky_view[sky_view == -inf] = nan
    sky_view = sky_view[::-1,:]


    solid_angle_healpix = hp.nside2pixarea(nside)
    soild_angle_projection = arcsin(l_reso)**2
    # print(solid_angle_healpix,soild_angle_projection)

    norm = soild_angle_projection/solid_angle_healpix
    sky_view *= norm

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
