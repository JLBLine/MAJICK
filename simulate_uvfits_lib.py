#!/usr/bin/python
import optparse
#import matplotlib.pyplot as plt
#from imager_lib import *
from calibrator_classes import *
from gridding_functions import *
from uvdata_classes import *
from time import time
from generate_gsm_2016 import generate_gsm_2016
from astropy.io import fits
from ephem import Observer,degrees
from os import environ
from numpy import floor,random
from os import environ
import pickle

MAJICK_DIR = environ['MAJICK_DIR']
with open('%s/imager_lib/MAJICK_variables.pkl' %MAJICK_DIR, 'rb') as f:  # Python 3: open(..., 'rb')
    D2R, R2D, VELC, MWA_LAT, KERNEL_SIZE, W_E, SOLAR2SIDEREAL = pickle.load(f)

def enh2xyz(east,north,height,latitiude):
    sl = sin(latitiude)
    cl = cos(latitiude)
    X = -north*sl + height*cl
    Y = east
    Z = north*cl + height*sl
    return X,Y,Z

def create_uv_kernel(image_kernel=False):
    '''Takes a image kernel and turns into a uv kernel'''
    ##FFT shift the image ready for FFT
    image_kernel = fft.ifftshift(image_kernel)
    ##Do the forward FFT as we define the inverse FFT for u,v -> l,m.
    ##Scale the output correctly for the way that numpy does it, and remove FFT shift
    #uv_kernel = fft.fft2(image_kernel) #/ (image_kernel.shape[0] * image_kernel.shape[1])
    uv_kernel = fft.fft2(image_kernel) / (KERNEL_SIZE * KERNEL_SIZE)
    uv_kernel = fft.fftshift(uv_kernel)

    return uv_kernel

def undo_phase_track(visibilities=None,new_wws=None):
    '''Undoes phase tracking for the given w-coords - these w-coords
    should be calculated towards phase centre'''

    ##no phase track = exp(-2*pi*j*(ul + vm + wn))
    ##phase track = exp(-2*pi*j*(ul + vm + w(n-1))
    ##so just multiply by exp(-2*pi*j*w) to undo
    sign = -1
    PhaseConst = 1j * 2 * pi * sign

    phase_rotate = n_exp(PhaseConst * new_wws)
    rotated_visis *= phase_rotate


def create_uvfits(v_container=None,freq_cent=None, ra_point=None, dec_point=MWA_LAT,
                  output_uvfits_name=None,uu=None,vv=None,ww=None,
                  baselines_array=None,date_array=None,date=None,
                  central_freq_chan=None,ch_width=None,template_uvfits=None,
                  int_jd=None):
    '''Takes visibility date and writes out a uvfits format file'''

    ##UU, VV, WW don't actually get read in by RTS - might be an issue with
    ##miriad/wsclean however, as it looks like oskar w = negative maps w
    uvparnames = ['UU','VV','WW','BASELINE','DATE']
    parvals = [uu,vv,ww,baselines_array,date_array]

    uvhdu = fits.GroupData(v_container,parnames=uvparnames,pardata=parvals,bitpix=-32)
    uvhdu = fits.GroupsHDU(uvhdu)

    ###Try to copy MAPS as sensibly as possible
    uvhdu.header['CTYPE2'] = 'COMPLEX '
    uvhdu.header['CRVAL2'] = 1.0
    uvhdu.header['CRPIX2'] = 1.0
    uvhdu.header['CDELT2'] = 1.0

    ##This means it's linearly polarised
    uvhdu.header['CTYPE3'] = 'STOKES '
    uvhdu.header['CRVAL3'] = -5.0
    uvhdu.header['CRPIX3'] =  1.0
    uvhdu.header['CDELT3'] = -1.0

    uvhdu.header['CTYPE4'] = 'FREQ'
    uvhdu.header['CRVAL4'] = freq_cent  ##Middle pixel value in Hz
    uvhdu.header['CRPIX4'] = int(central_freq_chan) + 1 ##Middle pixel number
    uvhdu.header['CDELT4'] = ch_width

    uvhdu.header['CTYPE5'] = template_uvfits[0].header['CTYPE5']
    uvhdu.header['CRVAL5'] = template_uvfits[0].header['CRVAL5']
    uvhdu.header['CRPIX5'] = template_uvfits[0].header['CRPIX5']
    uvhdu.header['CDELT5'] = template_uvfits[0].header['CDELT5']

    uvhdu.header['CTYPE6'] = template_uvfits[0].header['CTYPE6']
    uvhdu.header['CRVAL6'] = ra_point
    uvhdu.header['CRPIX6'] = template_uvfits[0].header['CRPIX6']
    uvhdu.header['CDELT6'] = template_uvfits[0].header['CDELT6']

    uvhdu.header['CTYPE7'] = template_uvfits[0].header['CTYPE7']
    uvhdu.header['CRVAL7'] = dec_point
    uvhdu.header['CRPIX7'] = template_uvfits[0].header['CRPIX7']
    uvhdu.header['CDELT7'] = template_uvfits[0].header['CDELT7']

    ## Write the parameters scaling explictly because they are omitted if default 1/0
    uvhdu.header['PSCAL1'] = 1.0
    uvhdu.header['PZERO1'] = 0.0
    uvhdu.header['PSCAL2'] = 1.0
    uvhdu.header['PZERO2'] = 0.0
    uvhdu.header['PSCAL3'] = 1.0
    uvhdu.header['PZERO3'] = 0.0
    uvhdu.header['PSCAL4'] = 1.0
    uvhdu.header['PZERO4'] = 0.0
    uvhdu.header['PSCAL5'] = 1.0

    uvhdu.header['PZERO5'] = float(int_jd)

    uvhdu.header['OBJECT']  = 'Undefined'
    uvhdu.header['OBSRA']   = ra_point
    uvhdu.header['OBSDEC']  = dec_point
    # uvhdu.header['TELESCOPE'] = 'MWA'

    ##ANTENNA TABLE MODS======================================================================

    template_uvfits[1].header['FREQ'] = freq_cent
    # template_uvfits[1].header['TELESCOPE'] = 'MWA'
    template_uvfits[1].header['ARRNAME'] = 'MWA'

    ###MAJICK uses this date to set the LST
    #dmy, hms = date.split()
    #day,month,year = map(int,dmy.split('-'))
    #hour,mins,secs = map(float,hms.split(':'))

    #rdate = "%d-%02d-%02dT%02d:%02d:%.2f" %(year,month,day,hour,mins,secs)

    template_uvfits[1].header['RDATE'] = date

    ## Create hdulist and write out file
    hdulist = fits.HDUList(hdus=[uvhdu,template_uvfits[1]])
    hdulist.writeto(output_uvfits_name,overwrite=True)
    hdulist.close()
