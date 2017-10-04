'''Functions to handle uvdata '''
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
#from calibrator_classes import *
#from imager_classes import *
#from uvdata_classes import *
#from astropy.io import fits
try:
    import pyfits as fits
except ImportError:
    from astropy.io import fits
from ephem import Observer,degrees
from numpy import sin,cos,pi,array,sqrt,arange,zeros,fft,meshgrid,where,arcsin,mod,real,ndarray,ceil,imag,floor
from numpy import abs as np_abs
from numpy import exp as np_exp
from cmath import phase,exp
from sys import exit
from jdcal import gcal2jd
#from astropy.wcs import WCS
#from time import time

D2R = pi/180.0
R2D = 180.0/pi
VELC = 299792458.0
MWA_LAT = -26.7033194444
#MWA_LAT = 0.0
SOLAR2SIDEREAL = 1.00274

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


class UVData(object):
    def __init__(self,uvfits=None,time_res=None,rts=False,date=False):
        '''A single time and frequency step of uvdata. Includes a dictioary
        containing all telecope X,Y,Z and antenna pairs from which to calcualte
        baseline lengths'''
        self.uvfits = uvfits
        HDU = fits.open(uvfits)
        ##Find the pointing centre
        if rts:
            self.ra_point = HDU[0].header['CRVAL5']
            self.dec_point = HDU[0].header['CRVAL6']
        else:
            self.ra_point = HDU[0].header['CRVAL6']
            self.dec_point = HDU[0].header['CRVAL7']
        self.freq = HDU[0].header['CRVAL4']
        
        if date:
            ##Use a user supplied date for LST rather than what's in the uvfits writer
            ##Reformat date from header into something
            ##readble by Observer to calcualte LST
            date,time = date.split('T')
            MRO.date = '/'.join(date.split('-'))+' '+time
            self.LST = float(MRO.sidereal_time())*R2D
        else:
            ##Reformat date from header into something
            ##readble by Observer to calcualte LST
            date,time = HDU[1].header['RDATE'].split('T')
            MRO.date = '/'.join(date.split('-'))+' '+time
            self.LST = float(MRO.sidereal_time())*R2D
        
        ##add on half a time resolution to find the LST
        ##at the centre of this
        self.central_LST = float(MRO.sidereal_time())*R2D + ((time_res/2.0)*(15.0/3600.0)*SOLAR2SIDEREAL)
        #print(self.ra_point,self.dec_point,HDU[1].header['RDATE'],self.LST)

        ##Requires u,v,w in units of wavelength, stored in seconds
        ##(u * c) / (c / freq) = u * freq
        self.uu = HDU[0].data['UU'].copy() * HDU[0].header['CRVAL4']
        self.vv = HDU[0].data['VV'].copy() * HDU[0].header['CRVAL4']
        self.ww = HDU[0].data['WW'].copy() * HDU[0].header['CRVAL4']
        
        self.antennas = {}
        xyz = HDU[1].data['STABXYZ'].copy()
        for i in xrange(len(xyz)):
            self.antennas['ANT%03d' %(i+1)] = xyz[i]
        
        ##The field BASELINE is the baseline number (256ant1 + ant2 +
        ##subarray/100.)
        ##Found that out from here:
        ##https://www.mrao.cam.ac.uk/~bn204/alma/memo-turb/uvfits.py
        baselines = HDU[0].data['BASELINE'].copy()
        self.baselines = baselines
        self.antenna_pairs = []
        self.xyz_lengths = []
        self.uvw_calc = []
        self.uvw_zenith = []
        
        for baseline in baselines:
            ant2 = mod(baseline, 256)
            ant1 = (baseline - ant2)/256
            ##Make the antenna pairs and then scale by the wavelength
            self.antenna_pairs.append(('ANT%03d' %ant1,'ANT%03d' %ant2))
            x_length,y_length,z_length = self.antennas['ANT%03d' %ant1]*(HDU[0].header['CRVAL4'] / VELC) - self.antennas['ANT%03d' %ant2]*(HDU[0].header['CRVAL4'] / VELC)
            self.xyz_lengths.append([x_length,y_length,z_length])
            
            self.uvw_calc.append(get_uvw(x_length,y_length,z_length,self.dec_point*D2R,self.central_LST*D2R - self.ra_point*D2R))
            self.uvw_zenith.append(get_uvw(x_length,y_length,z_length,MWA_LAT*D2R,0.0))
            
        ##TODO read in 'CRVAL3' to determine data shape
        ##and set num_polar
        self.num_polar = 4
        
        if rts:
            self.data = HDU[0].data['DATA'][:,0,0,0,:,:].copy()
        else:
            self.data = HDU[0].data['DATA'][:,0,0,0,0,:,:].copy()
        
        self.max_u = self.uu.max()
        self.max_v = self.vv.max()
        self.min_u = self.uu.min()
        self.min_v = self.vv.min()
        
        self.cal_data = None
        
        ##Do this as memmap (some python memory map thing) makes
        ##a copy of the hdu everytime you reference it, and so
        ##HDU.close() doesn't shut everything down - you can quite
        ##easily end up with two many HDUs open
        del HDU[0].data
        del HDU[1].data
        HDU.close()
        
class UVContainer(object):
    def __init__(self,uv_tag=None,freq_start=None,num_freqs=None,freq_res=None,time_start=None,num_times=None,time_res=None,add_phase_track=False,rts=False,date=False):
        '''An array containing UVData objects in shape = (num time steps, num freq steps'''
        ##TODO do an error check for all required uvfits files
        ##Have a custom error?
        self.freq_res = freq_res
        self.time_res = time_res
        self.rts = rts
        
        times = time_start + arange(num_times) * time_res
        freqs = freq_start + arange(num_freqs) * freq_res
        
        self.freqs = list(freqs)
        self.times = list(times)
        self.uv_data = {}
        self.cal_uv_data = {}
        self.kernel_params = None
        self.xyz_lengths_unscaled = []
        
        max_us = []
        max_vs = []
        min_us = []
        min_vs = []
        
        print("Now loading uvfits data....")
        
        ##Set the phase centre from the ra_point, dec_point from the first time step
        ##TODO get this from somewhere smarter
        ##Also grab the unscaled antenna lengths from the first uvfits file
        if time_res < 1:
            first_uvdata = UVData('%s_%.3f_%05.2f.uvfits' %(uv_tag,freqs[0],times[0]),time_res=time_res,rts=rts,date=date)
        else:
            first_uvdata = UVData('%s_%.3f_%02d.uvfits' %(uv_tag,freqs[0],times[0]),time_res=time_res,rts=rts,date=date)
            
        self.ra_phase = first_uvdata.ra_point
        self.dec_phase = first_uvdata.dec_point
        antennas = first_uvdata.antennas
        baselines = first_uvdata.baselines
        
        for baseline in baselines:
            ant2 = mod(baseline, 256)
            ant1 = (baseline - ant2)/256
            x_length,y_length,z_length = antennas['ANT%03d' %ant1] - antennas['ANT%03d' %ant2]
            self.xyz_lengths_unscaled.append([x_length,y_length,z_length])
        
        
        
        for i in xrange(len(freqs)):
            for j in xrange(len(times)):
                #print("Loading and rotating uvfits time %03d freq %.3f" %(times[j],freqs[i]))
                #from subprocess import call
                if time_res < 1:
                    uvdata = UVData('%s_%.3f_%05.2f.uvfits' %(uv_tag,freqs[i],times[j]),time_res=time_res,rts=rts,date=date)
                else:
                    uvdata = UVData('%s_%.3f_%02d.uvfits' %(uv_tag,freqs[i],times[j]),time_res=time_res,rts=rts,date=date)
                    
                ###If the correlator wasn't phase tracking, add in phase tracking
                ####Here we try to phase rotate everything to the first pointing 
                #print(uvdata.central_LST,self.ra_phase)
                if add_phase_track:
                    for k in xrange(len(uvdata.uu)):
                        ##Find the u,v,w coordinates for the phase centre for the LST of the given uv data
                        x_length,y_length,z_length = uvdata.xyz_lengths[k]
                        
                        ##central_LST has half the time resolution added to the intial LST for the obs
                        u_phase, v_phase, w_phase = get_uvw(x_length,y_length,z_length,self.dec_phase*D2R,uvdata.central_LST*D2R - self.ra_phase*D2R)
                        
                        ##Update the u,v,w data to the u,v,w associated with phase centre
                        uvdata.uu[k] = u_phase
                        uvdata.vv[k] = v_phase
                        uvdata.ww[k] = w_phase
                        
                        ##This rotation works if there has been NO PHASE TRACKING IN THE CORRELATOR
                        ##If there was phase tracking, would need to undo that, I think by
                        ##multiplying by exp(+PhaseConst * w_phase_correlator) - or you just leave
                        ##the phase centre as it was lolz
                        ##Basically this multiplication sets w to zero in the direction
                        ##of your phase centre (the old w[n-1])
                        xx_complex = complex(uvdata.data[k][0,0],uvdata.data[k][0,1])
                        yy_complex = complex(uvdata.data[k][1,0],uvdata.data[k][1,1])
                        PhaseConst = -1j * 2 * pi
                        rotate_xx_complex = xx_complex * exp(PhaseConst * w_phase)
                        rotate_yy_complex = yy_complex * exp(PhaseConst * w_phase)
                        
                        uvdata.data[k][0,0] = real(rotate_xx_complex)
                        uvdata.data[k][0,1] = imag(rotate_xx_complex)
                        uvdata.data[k][1,0] = real(rotate_yy_complex)
                        uvdata.data[k][1,1] = imag(rotate_yy_complex)
                    
                max_us.append(uvdata.max_u)
                max_vs.append(uvdata.max_v)
                min_us.append(uvdata.min_u)
                min_vs.append(uvdata.min_v)
                self.uv_data['%.3f_%05.2f' %(freqs[i],times[j])] = uvdata
        
        ##Use these later on to work out how big the u,v grid needs to be for imaging
        self.max_u = max(max_us)
        self.max_v = max(max_vs)
        self.min_u = min(min_us)
        self.min_v = min(min_vs)