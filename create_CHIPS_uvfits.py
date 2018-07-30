#!/usr/bin/env python
from optparse import OptionParser
from sys import exit
from imager_lib import *
from numpy import exp as np_exp
from numpy import tile
#from os import environ
import pickle

MAJICK_DIR = environ['MAJICK_DIR']
#with open('%s/imager_lib/MAJICK_variables.pkl' %MAJICK_DIR) as f:  # Python 3: open(..., 'rb')
    #D2R, R2D, VELC, MWA_LAT, KERNEL_SIZE, W_E, SOLAR2SIDEREAL = pickle.load(f)

VELC = 299792458.0

def add_time(date_time,time_step):
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

parser = OptionParser()

#parser.add_option('-f', '--freq_start',
    #help='Enter lowest frequency (MHz) to simulate - this is lower band edge')
    
#parser.add_option('-n', '--num_freqs',
    #help='Enter number of frequency channels to simulate')

#parser.add_option('-y', '--freq_res', default=0.04,
    #help='Enter frequency resolution (MHz) of observations, default=0.04')
    
#parser.add_option('-t', '--time_start', 
    #help='Enter lowest time offset from start date to simulate (s)')

#parser.add_option('-x', '--time_res', default=2.0,
    #help='Enter time resolution (s) of observations, default=2.0')
    
#parser.add_option('-m', '--num_times', 
    #help='Enter number of times steps to simulate')
    
#parser.add_option('-d', '--date', default='2000-01-01T00:00:00',
    #help="Enter date to start the observation on (YYYY-MM-DDThh:mm:ss), default='2000-01-01T00:00:00'")

parser.add_option('-a', '--time_int', default=False,
    help='Enter time cadence to average data to (s)')

parser.add_option('-b', '--freq_int', default=False,
    help='Enter freq cadence to average data to (kHz)')

parser.add_option('-c', '--output_name', 
    help='Enter tag name for output uvfits files')

parser.add_option('-d', '--uvfits_to_average', 
    help='Base fits file name and location (e.g. /location/file/uvfits_tag)')

parser.add_option('-e', '--telescope', default='MWA_phase1',
    help='Gets the base uvfits file used for the antenna table - defaults to MWA_phase1')

parser.add_option('-f', '--data_loc', default='./',
    help='Location to output the uvfits to')

parser.add_option('-g', '--band_num',
    help='RTS band number to name the output uvfits with')

parser.add_option('-i', '--phase_track', default=False,
    help='Enter the phase centre as --phase_track=ra,dec (deg). Add phase tracking to the base uvfits files - phase tracking applied in the native time cadence, before average')

parser.add_option('-j', '--rephase', default=False,
    help='NOT IMPLEMENTED YET!! Enter the phase centre as --phase_track=ra,dec (deg). Unwraps current phase tracking of the current time cadence, and apply phase tracking for the new averaged time - i.e. if data is 2s cadence, undo any phase tracking average over new time cadence, say 8s, and apply phase trakcing using the w-term of the centre of the 8s time cadence')
    
options, args = parser.parse_args()

###Setup some options---------------------------

uvfits_files_list = [options.uvfits_to_average]

if options.phase_track:
    add_phase_track = True
    phase_centre = options.phase_track

else:
    add_phase_track = False
    phase_centre = False

uv_container = UVContainer(uvfits_files = uvfits_files_list,add_phase_track=add_phase_track,phase_centre=phase_centre,time_res=2.0)
print('uvfits data finished loading......')

##-----------------------
freq_int = float(options.freq_int)*1e+3
time_int = float(options.time_int)
freq_res = uv_container.freq_res
time_res = uv_container.time_res
num_freqs = float(len(uv_container.freqs))
num_times = float(len(uv_container.times))
data_loc = options.data_loc
if data_loc[-1] == '/': data_loc = data_loc[:-1]
band_num = 1

##Get some basic uvfits file structure from template uvfits----------
base_uvfits_loc = "%s/telescopes/%s/%s_template.uvfits" %(MAJICK_DIR,options.telescope,options.telescope)
base_uvfits = fits.open(base_uvfits_loc)
    
base_data = base_uvfits[0].data
antenna_header = base_uvfits[1].header
    
##Get some parameters about the averaging we are doing---------------------------
num_freq_avg = int(freq_int/uv_container.freq_res)
num_time_avg = int(time_int/uv_container.time_res)

num_time_groups = int((num_times*time_res) / time_int)
num_freq_groups = int((num_freqs*freq_res) / freq_int)

num_baselines = uv_container.num_baselines
##The number of random groups is set by number of baselines and averaged time steps
n_data = num_baselines * num_time_groups

ra_phase = uv_container.ra_phase
dec_phase = uv_container.dec_phase
template_baselines = base_data['BASELINE']

##This gives us the Julian Date for the first integrated time step
##Use this to set the PZERO5 (int_jd) - then we can add time on
##time increments to float_jd for each subsequent time integration

##OLD WAY - add half a time step on - think this is wrong
#half_time_cadence = num_time_avg * (uv_container.time_res / 2.0)
#first_date = add_time(intial_date,half_time_cadence)
##NEW WAY - just convert initial date into int, float days
int_jd, float_jd = calc_jdcal(uv_container.first_date)
#print int_jd, float_jd

##Need an array the length of number of baselines worth of the fractional jd date
float_jd_array = ones(num_baselines)*float_jd

##Create empty data structures for final uvfits file
v_container = zeros((n_data,1,1,num_freq_groups,4,3))
uu = zeros(n_data)
vv = zeros(n_data)
ww = zeros(n_data)
baselines_array = zeros(n_data)
date_array = zeros(n_data)

print('Now averaging data......')

for time_step,time_start in enumerate(range(0,len(uv_container.times),num_time_avg)):
    ##First time step of this cadence
    time = uv_container.times[time_start]
    #print('time_step,time_start,time',time_step,time_start,time)
    ##Start point of data in the final uvfits structure
    array_time_loc = num_baselines*time_step
    
    ###In the following, find the LST and frequency at the centre of the set of
    ###visis being averaged over
    ###If averaging more than one time step together, need to find the offset of the
    ###central LST of the averaged time from the start of the set of times
    if num_time_avg > 1:
        ##The centre of the averaged time cadence, in time after the very first time step
        half_time_cadence = time + ((num_time_avg * uv_container.time_res) / 2.0)
        ##Initial LST is for the centre of the intital time step, so half a time resolution
        ##after the beginning of the averaged time cadence
        half_time_cadence -= uv_container.time_res / 2.0
        half_time_cadence *= SOLAR2SIDEREAL*(15.0/3600.0)
        #half_time_cadence = num_time_avg * (uv_container.time_res / 2.0) * SOLAR2SIDEREAL*(15.0/3600.0)
    ##the initial_lst is the central lst of the first time step, so if not averaging, don't
    ##need to add anything
    else:
        half_time_cadence = time * SOLAR2SIDEREAL*(15.0/3600.0)
    
    #central_lst = uv_container.central_LST + (time * SOLAR2SIDEREAL*(15.0/3600.0))
    central_lst = uv_container.central_LST + half_time_cadence

    ##Get some relevant positions and data
    ra0,dec0 =  ra_phase*D2R,dec_phase*D2R
    #print('ra_phase,dec_phase',ra0,dec0)
    h0 = central_lst*D2R - ra0
    
    for freq_step,freq_start in enumerate(range(0,len(uv_container.freqs),num_freq_avg)):
        ##First freq step of this cadence
        freq = uv_container.freqs[freq_start]
        
        ##Empty uvdata array for this time,freq integration
        sum_uvdata = zeros((num_baselines,4,3))
        ##Actual averaging loop----------------------
        for time_avg in range(time_start,time_start+num_time_avg):
            
            for freq_avg in range(freq_start,freq_start+num_freq_avg):
                freq_label = uv_container.freqs[freq_avg]
                time_label = uv_container.times[time_avg]
                #print("time step %.2f, freq step %.3f"%(time,freq))
                uvdata = uv_container.uv_data['%.3f_%05.2f' %(freq_label,time_label)].data
                
                ##TODO sort this shit out--------------------------------
                #if options.rephase:
                    ###if we need to rephase, undo original phase tracking.
                    #w = uv_container.uv_data['%.3f_%05.2f' %(freq,time)].ww
                    #PhaseConst = -1 * 1j * 2 * pi
                    #for i in xrange(len(w)):
                        #new_XX = complex(uvdata[i,0,0],uvdata[i,0,1]) * np_exp(PhaseConst * w[i])
                        #new_YY = complex(uvdata[i,1,0],uvdata[i,1,1]) * np_exp(PhaseConst * w[i])
                        #uvdata[i,0,0] = real(new_XX)
                        #uvdata[i,0,1] = imag(new_XX)
                        #uvdata[i,1,0] = real(new_YY)
                        #uvdata[i,1,1] = imag(new_YY)
                ##TODO sort this shit out--------------------------------

                sum_uvdata += uvdata
        ##Actual averaging loop finish----------------------
        

        ###Central frequency of the the first freq step of this cadence
        #freq_cent = freq + (uv_container.freq_res / 2.0)
        
        ###If averaging over more than one frequeny, work out distance
        ###of cadence centre to start of cadence
        #if num_freq_avg > 1:
            #half_freq_cadence = (num_freq_avg * uv_container.freq_res) / 2.0
            ##half_freq_cadence -= uv_container.freq_res / 2.0
        #else:
            #half_freq_cadence = 0
            
        #central_frequency = freq_cent + half_freq_cadence
        
        ##These are the non frequency scaled lengths in X,Y,Z
        xyzs = array(uv_container.xyz_lengths)
        ##Seperate out into x,y,z
        x_lengths = xyzs[:,0]
        y_lengths = xyzs[:,1]
        z_lengths = xyzs[:,2]
        
        ##Calculate the u,v,w coords for all baselines at the centre of the integration
        avg_uu, avg_vv, avg_ww = get_uvw(x_lengths,y_lengths,z_lengths,dec0,h0)
        
        ##Add the u,v,w coords from the central time step into uvfits
        ##Stored in seconds, have calculated in meters, so divide by VELC
        
        uu[array_time_loc:array_time_loc+num_baselines] = avg_uu / VELC
        vv[array_time_loc:array_time_loc+num_baselines] = avg_vv / VELC
        ww[array_time_loc:array_time_loc+num_baselines] = avg_ww / VELC

        ##TODO sort this shit out--------------------------------
        ##If rephasing, phase track to the new phase centre
        #if options.rephase:
            #PhaseConst = 1j * 2 * pi
            #for i in xrange(len(w)):
                #new_XX = complex(sum_uvdata[i,0,0],uvdata[i,0,1]) * np_exp(PhaseConst * avg_ww[i])
                #new_YY = complex(sum_uvdata[i,1,0],uvdata[i,1,1]) * np_exp(PhaseConst * avg_ww[i])
                #sum_uvdata[i,0,0] = real(new_XX)
                #sum_uvdata[i,0,1] = imag(new_XX)
                #sum_uvdata[i,1,0] = real(new_YY)
                #sum_uvdata[i,1,1] = imag(new_YY)
        ##TODO sort this shit out--------------------------------

        ##Average the summed data by the number of summed
        ##visibilities
        sum_uvdata[:,:,0] /= num_time_avg*num_freq_avg
        sum_uvdata[:,:,1] /= num_time_avg*num_freq_avg
        sum_uvdata[:,:,2] /= num_time_avg*num_freq_avg
        
        ##Add data in order of baselines, then time step in axes 0 of v_container
        ##Each frequency average goes axes 4 of the v_container
        v_container[array_time_loc:array_time_loc+num_baselines,0,0,freq_step,:,:] = sum_uvdata
        
        
    ##Fill in the baselines the template - these are baseline tag numbers, so
    ##only need one set per time step - not per freqency step
    
    baselines_array[array_time_loc:array_time_loc+num_baselines] = template_baselines[:num_baselines]
    
    ##Fill in the fractional julian date, after adding on the appropriate amount of
    ##time - /(24*60*60) because julian number is a fraction of a whole day
    adjust_float_jd_array = float_jd_array + ((time_step * time_int) / (24.0*60.0*60.0))
    date_array[array_time_loc:array_time_loc+num_baselines] = adjust_float_jd_array
        

    
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

middle_pixel_number = ceil(num_freq_groups / 2.0)
middle_pixel_value = uv_container.freqs[0] + (freq_res * num_freq_avg * (middle_pixel_number - 1))

uvhdu.header['CTYPE4'] = 'FREQ'
uvhdu.header['CRVAL4'] = middle_pixel_value  ##Middle pixel value in Hz
uvhdu.header['CRPIX4'] = middle_pixel_number ##Middle pixel number
uvhdu.header['CDELT4'] = freq_int

##Don't inlcude the extra axes that helps no one
#uvhdu.header['CTYPE5'] = base_uvfits[0].header['CTYPE5']
#uvhdu.header['CRVAL5'] = base_uvfits[0].header['CRVAL5']
#uvhdu.header['CRPIX5'] = base_uvfits[0].header['CRPIX5']
#uvhdu.header['CDELT5'] = base_uvfits[0].header['CDELT5']

##RA phase information
uvhdu.header['CTYPE5'] = base_uvfits[0].header['CTYPE6']
uvhdu.header['CRVAL5'] = uv_container.ra_phase
uvhdu.header['CRPIX5'] = base_uvfits[0].header['CRPIX6']
uvhdu.header['CDELT5'] = base_uvfits[0].header['CDELT6']

##DEC phase information
uvhdu.header['CTYPE6'] = base_uvfits[0].header['CTYPE7']
uvhdu.header['CRVAL6'] = uv_container.dec_phase
uvhdu.header['CRPIX6'] = base_uvfits[0].header['CRPIX7']
uvhdu.header['CDELT6'] = base_uvfits[0].header['CDELT7']

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

uvhdu.header['PZERO5'] = int_jd

uvhdu.header['OBJECT']  = 'Undefined'                                                           
uvhdu.header['OBSRA']   = uv_container.ra_phase
uvhdu.header['OBSDEC']  = uv_container.dec_phase

##Number of baselines * number of time steps
uvhdu.header['GCOUNT'] = n_data

antenna_header['FREQ'] = middle_pixel_value

## Create hdulist and write out file
write_uvfits = fits.HDUList(hdus=[uvhdu,base_uvfits[1]])

#if options.rephase:
    #uvfits_name = "%s_rephase_t%02d_f%.3f_%02d.uvfits" %(tag_name,time_int,freq_int,band_num)
#else:
uvfits_name = "%s_t%02d_f%.3f_%02d.uvfits" %(options.output_name,time_int,freq_int/1e+6,band_num)

write_uvfits.writeto('%s/%s' %(data_loc,uvfits_name) ,overwrite=True)
