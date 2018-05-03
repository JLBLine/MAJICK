'''Functions to handle uvdata '''
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
#from calibrator_classes import *
#from imager_classes import *
#from uvdata_classes import *
#from astropy.io import fits
try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits
    
from ephem import Observer,degrees
from numpy import sin,cos,pi,array,sqrt,arange,zeros,fft,meshgrid,where,arcsin,mod,real,ndarray,ceil,imag,floor,tile,savez_compressed
from numpy import abs as np_abs
from numpy import exp as np_exp
from cmath import phase,exp
from sys import exit
from jdcal import gcal2jd
#from astropy.wcs import WCS
#from time import time

from os import environ
import pickle

MAJICK_DIR = environ['MAJICK_DIR']
with open('%s/imager_lib/MAJICK_variables.pkl' %MAJICK_DIR) as f:  # Python 3: open(..., 'rb')
    D2R, R2D, VELC, MWA_LAT, KERNEL_SIZE, W_E, SOLAR2SIDEREAL = pickle.load(f)

##ephem Observer class, use this to compute LST from the date of the obs 
MRO = Observer()
##Set the observer at Boolardy
MRO.lat, MRO.long, MRO.elevation = '-26:42:11.95', '116:40:14.93', 0

###METHOD - grid u,v by baseline, do the FFT to give you l,m
###Apply inverse of decorrelation factor to l,m grid

def get_uvw(X,Y,Z,d,H):
    u = sin(H)*X + cos(H)*Y
    v = -sin(d)*cos(H)*X + sin(d)*sin(H)*Y + cos(d)*Z
    w = cos(d)*cos(H)*X - cos(d)*sin(H)*Y + sin(d)*Z
    return u,v,w


def get_uvw_freq(x_length=None,y_length=None,z_length=None,dec=None,ha=None,freq=None):
    '''Takes the baseline length in meters and uses the frequency'''
    
    scale = freq / VELC
    X = x_length * scale
    Y = y_length * scale
    Z = z_length * scale
    
    u = sin(ha)*X + cos(ha)*Y
    v = -sin(dec)*cos(ha)*X + sin(dec)*sin(ha)*Y + cos(dec)*Z
    w = cos(dec)*cos(ha)*X - cos(dec)*sin(ha)*Y + sin(dec)*Z
    
    return u,v,w

def add_time_uvfits(date_time,time_step):
    '''Take the time string format that uvfits uses (DIFFERENT TO OSKAR!! '2013-08-23 17:54:32.0'), and add a time time_step (seconds).
    Return in the same format - NO SUPPORT FOR CHANGES MONTHS CURRENTLY!!'''
    date,time = date_time.split('T')
    year,month,day = map(int,date.split('-'))
    hours,mins,secs = map(float,time.split(':'))
    ##Add time
    secs += time_step
    if secs >= 60.0:
        ##Find out full minutes extra and take away
        ext_mins = int(secs / 60.0)
        secs -= 60*ext_mins
        mins += ext_mins
        if mins >= 60.0:
            ext_hours = int(mins / 60.0)
            mins -= 60*ext_hours
            hours += ext_hours
            if hours >= 24.0:
                ext_days = int(hours / 24.0)
                hours -= 24*ext_days
                day += ext_days
            else:
                pass
        else:
            pass
    else:
        pass
    return '%d-%d-%dT%d:%02d:%05.2f' %(year,month,day,int(hours),int(mins),secs)

def calc_jdcal_old(date):
    dmy, hms = date.split('T')
    
    year,month,day = map(int,dmy.split('-'))
    hour,mins,secs = map(float,hms.split(':'))

    ##For some reason jdcal gives you the date in two pieces
    ##Gives you the time up until midnight of the day
    jd1,jd2 = gcal2jd(year,month,day)
    jd3 = (hour + (mins / 60.0) + (secs / 3600.0)) / 24.0

    jd = jd1 + jd2 + jd3
    
    ##The header of the uvdata file takes the integer, and
    ##then the fraction goes into the data array for PTYPE5
    return floor(jd), jd - floor(jd)

def calc_jdcal(date):
    dmy, hms = date.split('T')
    
    year,month,day = map(int,dmy.split('-'))
    hour,mins,secs = map(float,hms.split(':'))

    ##For some reason jdcal gives you the date in two pieces
    ##Gives you the time up until midnight of the day
    jd1,jd2 = gcal2jd(year,month,day)
    jd3 = (hour + (mins / 60.0) + (secs / 3600.0)) / 24.0

    jd = jd1 + jd2 + jd3
    
    jd_day = jd1 + floor(jd2)
    jd_fraction = (jd2 - floor(jd2)) + jd3
    
    ##The header of the uvdata file takes the integer, and
    ##then the fraction goes into the data array for PTYPE5
    return jd_day, jd_fraction


def make_complex(re=None,im=None):
    '''Takes two arrays, and returns a complex array with re real values and im imarginary values'''
    comp = array(re,dtype=complex)
    comp += 1j * im
    
    return comp

def rotate_phase(wws=None,visibilities=None):
    '''Adds in phase tracking for the w terms specified'''
    
    ##theory - so normal phase delay is caused by path difference across
    ##a base line, which is u*l + v*m + w*n
    ##To phase track, you insert a phase to make sure there is no w contribution at
    ##phase centre; this is when n = 1, so you insert a phase thus:
    ## u*l + v*m + w*(n - 1)
    PhaseConst = -1j * 2 * pi
    phase_rotate = np_exp(PhaseConst * wws)
    rotated_visis = visibilities * phase_rotate
    return rotated_visis
    #return visibilities

class UVData(object):
    #@profile
    def __init__(self,uvfits=None):
        '''A single time and frequency step of uvdata. Includes the u,v,w coords and
        full pol visibility data'''
        self.uvfits = uvfits
        self.uu = None
        self.vv = None
        self.ww = None
        self.data = None
        
class UVContainer(object):
    def __init__(self,uvfits_files=None,add_phase_track=False,date=False,time_res=None,phase_centre=False):
        '''An array containing UVData objects in shape = (num time steps, num freq steps'''
        ##TODO do an error check for all required uvfits files
        ##Have a custom error?
        
        self.freqs = []
        self.times = []
        self.uv_data = {}
        if phase_centre:
            self.ra_phase,self.dec_phase = map(float,phase_centre.split(','))
        else:
            self.ra_phase = None
            self.dec_phase = None
        self.LST = None
        self.central_LST = None
        self.add_phase_track = add_phase_track
        self.date = date
        self.time_res = time_res
        self.freq_res = None
        self.antennas = None
        self.antenna_pairs = None
        self.xyz_lengths = None
        self.first_date = None
        
        ##TODO set the 0 time step from the date of the first obs??
        for uvfits_ind,uvfits in enumerate(uvfits_files):
            if uvfits_ind == 0:
                first_uvfits = True
            else:
                first_uvfits = False
            
            self.add_data(uvfits=uvfits,first_uvfits=first_uvfits,date=self.date)
        
    def add_data(self,uvfits=None,first_uvfits=False,date=False):
        ##Open up that data
        HDU = fits.open(uvfits)
        ##Retrieve some useful header informations
        
        if len(HDU[0].data.data.shape) == 7:
            seven_len = True
        else:
            seven_len = False
        
        ##Use the first loaded uvfits file to set pointing centre
        ##and other global obs properties
        if first_uvfits:
            
            self.first_date = HDU[1].header['RDATE']
            
            ##TODO Could do some kind of check here that all the data
            ##has the same frequency
            self.freq_res = HDU[0].header['CDELT4']
            
            ##User doesn't care (or know what is going on ha)
            ##Make phase centre the pointing centre
            if self.ra_phase == None:
                ##Find the pointing centre
                if seven_len:
                    self.ra_phase = HDU[0].header['CRVAL6']
                    self.dec_phase = HDU[0].header['CRVAL7']
                else:
                    self.ra_phase = HDU[0].header['CRVAL5']
                    self.dec_phase = HDU[0].header['CRVAL6']
                    
                    
            ##User wants a specific phase centre and has already set it
            else:
                pass
                    
            ##TODO For now assume that all data is coming from the
            ##the same instrument and array layout, so only need to do
            ##this calculation once
            ##Setup a dictionary that contains antenna info
            self.antennas = {}
            xyz = HDU[1].data['STABXYZ'].copy()
            for i in xrange(len(xyz)):
                self.antennas['ANT%03d' %(i+1)] = xyz[i]
                
            self.num_baselines = int((len(xyz) - 1) * (len(xyz) / 2.0))
            
            ##The field BASELINE is the baseline number (256ant1 + ant2 +
            ##subarray/100.)
            ##Found that out from here:
            ##https://www.mrao.cam.ac.uk/~bn204/alma/memo-turb/uvfits.py
            baselines = HDU[0].data['BASELINE'].copy()
            #print('Length of baselines is',len(baselines))

            self.antenna_pairs = []
            self.xyz_lengths = []
            
            ##These lengths are in the instrumental plane so only need to calculate
            ##once for each baseline - uvfits format stores a baseline number
            ##for all time steps
            for baseline in baselines[:self.num_baselines]:
                ant2 = mod(baseline, 256)
                ant1 = (baseline - ant2)/256
                ##Make the antenna pairs - stored in meters which is coool
                self.antenna_pairs.append(('ANT%03d' %ant1,'ANT%03d' %ant2))
                x_length,y_length,z_length = self.antennas['ANT%03d' %ant1] - self.antennas['ANT%03d' %ant2]
                self.xyz_lengths.append([x_length,y_length,z_length])
            self.xyz_lengths = array(self.xyz_lengths)
            
        if date:
            ##Use a user supplied date for LST rather than what's in the uvfits writer
            ##Reformat date from header into something
            ##readble by Observer to calcualte LST
            date,time = date.split('T')
            MRO.date = '/'.join(date.split('-'))+' '+time
            initial_LST = float(MRO.sidereal_time())*R2D
            if first_uvfits:
                self.initial_LST = initial_LST
            
        else:
            ##Reformat date from header into something
            ##readble by Observer to calcualte LST
            date,time = HDU[1].header['RDATE'].split('T')
            MRO.date = '/'.join(date.split('-'))+' '+time
            initial_LST = float(MRO.sidereal_time())*R2D
            if first_uvfits:
                self.initial_LST = initial_LST
        
        ##To work out the time resolution need to subtract
        ##two different time steps from one another
        num_timesteps = int(HDU[0].header['GCOUNT'] / self.num_baselines)
        if num_timesteps == 1:
            print('Only one timestep - cannot auto calculate the time resolution')
            print('Setting time resolution to 0.0 - things might go wrong')
            time_res = 0.0
            if first_uvfits:
                self.time_res = 0.0
        else:
            date_array = HDU[0].data['DATE']
            time_res = round((date_array[self.num_baselines] - date_array[0])* (24.0*60.*60.),2)
            if first_uvfits:
                self.time_res = time_res
        
        ##add on half a time resolution to find the LST
        ##at the centre of this
        
        #central_LST = initial_LST + ((self.time_res/2.0)*(15.0/3600.0)*SOLAR2SIDEREAL)
        central_LST = initial_LST + ((self.time_res/2.0)*(15.0/3600.0))

        if first_uvfits:
            self.central_LST = central_LST
            
        time_range = arange(num_timesteps)*time_res
        
        if self.add_phase_track:
            ##Calculate u,v,w coords towards the new phase centre
            ##Need to calculate the LST for all time steps to do the u,v,w 
            ##calc in one go
            
            ##initilise array
            these_LSTs = zeros(self.num_baselines*num_timesteps)
            
            for time_ind,time in enumerate(time_range):
                these_LSTs[time_ind*self.num_baselines:(time_ind+1)*self.num_baselines] = central_LST + (time*(15.0/3600.0)*SOLAR2SIDEREAL) #+ ((time_res/2.0)*(15.0/3600.0)*SOLAR2SIDEREAL)
                #these_LSTs[time_ind*self.num_baselines:(time_ind+1)*self.num_baselines] = initial_LST + (time*(15.0/3600.0)*SOLAR2SIDEREAL)
                
            ##Need to repeat the x,y,z coords by the number of timesteps
            x_lens = tile(self.xyz_lengths[:,0],num_timesteps)
            y_lens = tile(self.xyz_lengths[:,1],num_timesteps)
            z_lens = tile(self.xyz_lengths[:,2],num_timesteps)
            
            print(self.ra_phase,MWA_LAT)
            
            uu_meters,vv_meters,ww_meters = get_uvw(x_lens,y_lens,z_lens,self.dec_phase*D2R,(these_LSTs - self.ra_phase)*D2R)
        else:
            ##Assume that the correct u,v,w coords are in the uvfits file
            ##if we don't want to add phase tracking (either already tracked,
            ##or the u,v,ws point toward zenith and visis have no phase tracking)
            
            ##Stored in seconds, want it in meters so * VELC
            uu_meters = HDU[0].data['UU'].copy() * VELC
            vv_meters = HDU[0].data['VV'].copy() * VELC
            ww_meters = HDU[0].data['WW'].copy() * VELC
        
        ##Uvfits files are 1 indexed, python is not
        central_freq = HDU[0].header['CRVAL4']
        central_freq_index = HDU[0].header['CRPIX4'] - 1
        num_freqs = HDU[0].header['NAXIS4']
        freq_res = HDU[0].header['CDELT4']
        freqs = central_freq + (arange(num_freqs) - central_freq_index)*freq_res
        
        max_us = []
        min_us = []
        max_vs = []
        min_vs = []
        
        for chan_ind,freq in enumerate(freqs):
            self.freqs.append(freq)
            if seven_len:
                visi_data = HDU[0].data['DATA'][:,0,0,0,chan_ind,:,:].copy()
            else:
                visi_data = HDU[0].data['DATA'][:,0,0,chan_ind,:,:].copy()
                
            ##Scale the uv coords by the wavelength
            uu_scaled = uu_meters / (VELC / freq)
            vv_scaled = vv_meters / (VELC / freq)
            ww_scaled = ww_meters / (VELC / freq)
                
            if self.add_phase_track:
                ##Laborisouly phase rotate the data
                chan_xx_res = visi_data[:,0,0]
                chan_xx_ims = visi_data[:,0,1]
                chan_yy_res = visi_data[:,1,0]
                chan_yy_ims = visi_data[:,1,1]
                chan_xy_res = visi_data[:,2,0]
                chan_xy_ims = visi_data[:,2,1]
                chan_yx_res = visi_data[:,3,0]
                chan_yx_ims = visi_data[:,3,1]
                
                ##Make complex numpy arrays
                comp_xx = make_complex(chan_xx_res,chan_xx_ims)
                comp_xy = make_complex(chan_xy_res,chan_xy_ims)
                comp_yx = make_complex(chan_yx_res,chan_yx_ims)
                comp_yy = make_complex(chan_yy_res,chan_yy_ims)
                
                ##Add in the phase tracking
                rotated_xx = rotate_phase(wws=ww_scaled,visibilities=comp_xx)
                rotated_xy = rotate_phase(wws=ww_scaled,visibilities=comp_xy)
                rotated_yx = rotate_phase(wws=ww_scaled,visibilities=comp_yx)
                rotated_yy = rotate_phase(wws=ww_scaled,visibilities=comp_yy)
                
                visi_data[:,0,0] = rotated_xx.real
                visi_data[:,0,1] = rotated_xx.imag
                visi_data[:,1,0] = rotated_yy.real
                visi_data[:,1,1] = rotated_yy.imag
                visi_data[:,2,0] = rotated_xy.real
                visi_data[:,2,1] = rotated_xy.imag
                visi_data[:,3,0] = rotated_yx.real
                visi_data[:,3,1] = rotated_yx.imag
                
            for time_ind,time in enumerate(time_range):
                self.times.append(time)
                uvdata = UVData(uvfits)
                
                uvdata.uu = uu_scaled[time_ind*self.num_baselines:(time_ind+1)*self.num_baselines]
                uvdata.vv = vv_scaled[time_ind*self.num_baselines:(time_ind+1)*self.num_baselines]
                uvdata.ww = ww_scaled[time_ind*self.num_baselines:(time_ind+1)*self.num_baselines]
                
                uvdata.data = visi_data[time_ind*self.num_baselines:(time_ind+1)*self.num_baselines,:,:]
                
                max_us.append(uvdata.uu.max())
                max_vs.append(uvdata.vv.max())
                min_us.append(uvdata.uu.min())
                min_vs.append(uvdata.vv.min())
                self.uv_data['%.3f_%05.2f' %(freq,time)] = uvdata
                
        
        ##Use these later on to work out how big the u,v grid needs to be for imaging
        self.max_u = max(max_us)
        self.max_v = max(max_vs)
        self.min_u = min(min_us)
        self.min_v = min(min_vs)
        
        self.times = sorted(list(set(self.times)))
        self.freqs = sorted(list(set(self.freqs)))