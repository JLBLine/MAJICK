#!/usr/bin/env python
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
from os import environ,getcwd,chdir,makedirs,path
from numpy import floor,random,savez_compressed,load
from os import environ
import pickle
from simulate_uvfits_lib import *
from subprocess import call
import healpy as hp
from project_healpix import convert_healpix2lm,find_healpix_zenith_offset

MAJICK_DIR = environ['MAJICK_DIR']
with open('%s/imager_lib/MAJICK_variables.pkl' %MAJICK_DIR) as f:  # Python 3: open(..., 'rb')
    D2R, R2D, VELC, MWA_LAT, KERNEL_SIZE, W_E, SOLAR2SIDEREAL = pickle.load(f)

def enh2xyz(east,north,height,latitiude):
    sl = sin(latitiude)
    cl = cos(latitiude)
    X = -north*sl + height*cl
    Y = east
    Z = north*cl + height*sl
    return X,Y,Z

##---------------------------------------------------##
##----------OBSERVATION SETTINGS----------------------##

parser = optparse.OptionParser()

parser.add_option('-a', '--time_decor', default=True, action='store_false',
    help='Add to include time_decorrelation in the simulation')

parser.add_option('-b', '--degrid', default=False, action='store_true',
    help='Add to include the 2016 gsm sky model')

parser.add_option('-c', '--tag_name',
    help='Enter tag name for output uvfits files')

parser.add_option('-d', '--date', default='2000-01-01T00:00:00',
    help="Enter date to start the observation on (YYYY-MM-DDThh:mm:ss), default='2000-01-01T00:00:00'")

#parser.add_option('-e', '--base_uvfits', default=False,
    #help='Base fits file name and location (e.g. /location/file/uvfits_tag) tag to add diffuse model to (not needed if generating uvfits from srclist)')

parser.add_option('-f', '--freq_start',default=False,
    help='Enter lowest frequency (MHz) to simulate (if not using metafits) - this is lower band edge i.e. for freq_res=0.04, freq_start=167.035 will be simulated at 167.055')

parser.add_option('-g', '--no_beam', default=False, action='store_true',
    help='Add to switch off the beam')

parser.add_option('-i', '--data_loc', default='./data',
    help='Location to output the uvfits to. Default = ./data')

parser.add_option('-j', '--telescope', default='MWA_phase1',
    help='Uses the array layout and primary beam model as stored in MAJICK_DIR/telescopes - defaults to MWA_phase1')

parser.add_option('-k', '--fix_beam', default=False, action='store_true',
    help='Forces the MWA beam to be fixed to 186.235MHz, to be used in conjection with CHIPS_FIXBEAM')

parser.add_option('-l', '--multi_process', default=False,
    help='Switches on multiprocessing, using the number of processes given e.g. --multi_process=8')

parser.add_option('-m', '--num_times',
    help='Enter number of times steps to simulate')

parser.add_option('-n', '--num_freqs',default=False,
    help='Enter number of frequency channels per band to simulate - defaults to only simulate channels that are not flagged so 27 fine chans in total')

parser.add_option('-o', '--no_wproj',default=True,action='store_false',
    help='Add to switch off w-projection')

parser.add_option('-p', '--phase_centre', default=False,
    help='Phase centre of the observation in degrees as RA,DEC - as a default tracks the intial zenith point')

parser.add_option('-q', '--l_value',
    help='l offset value for testing')

#parser.add_option('-r', '--new_uvfits', default=False, action='store_true',
    #help='Add to create a fully new uvfits file for the diffuse simulation')

parser.add_option('-s', '--srclist', default=False,
    help='Enter name of srclist from which to add point sources')

parser.add_option('-t', '--time_start',
    help='Enter lowest time offset from start date to simulate (s)')

parser.add_option('-u', '--clobber',default=False,action='store_true',
    help='Add to change "overwrite" to "clobber" - astropy on gstar is old version')

parser.add_option('-v', '--no_phase_tracking',default=False,action='store_true',
    help='Add to turn off phase tracking')

#parser.add_option('-w', '--add_to_existing',default=False,action='store_true',
    #help='Add to add simulations to previous uvfits')

parser.add_option('-x', '--time_res', default=None,
    help='Enter time resolution (s) of observations, default to what is in metafits')

parser.add_option('-y', '--freq_res', default=0.04,
    help='Enter frequency resolution (MHz) of observations, default=0.04')

parser.add_option('-z', '--freq_decor', default=True, action='store_false',
    help='Add to include freq decorrelation in the simulation')

parser.add_option('--over_sampled_kernel', default=False, action='store_true',
    help='Add to switch the kernel to the oversampled version rather than the phase shifting niceness')

parser.add_option('--degrid_test', default=False, action='store_true',
    help='Add to make a single point source image for testing degrid rather than use the GSM')

parser.add_option('--single_baseline', default=False, action='store_true',
    help='Add to only simulate one baseline - good for testing')

parser.add_option('--oversampling_factor', default=99,
    help='Set the oversampling factor if using an oversampled kernel')

parser.add_option('--metafits',default=False,
    help='Enter name of metafits file to base obs on')

parser.add_option('--band_num',default=False,
    help='Enter band number to simulate')

parser.add_option('--bandwidth',default=30.72,
    help='Enter the full instrument bandwidth in MHz (defaults to 30.72MHz)')

parser.add_option('--healpix',default=False,
    help='Enter a healpix map for degridding sims - needs to be in celestial coordinates')

parser.add_option('--chips_settings', default=False, action='store_true',
    help='Swtiches on a default CHIPS resolution and uvfits weightings - 8s, 80kHz integration with the normal 5 40kHz channels missing. OVERRIDES other time/freq int settings')

parser.add_option('--full_chips', default=False, action='store_true',
    help='Instead of missing freq channels, do a complete simulation when doing a CHIPS simulation')

options, args = parser.parse_args()

if options.degrid_test:
    MWA_LAT = 0

#freq_start = float(options.freq_start)
#freq_res = float(options.freq_res)


time_start = float(options.time_start)
num_times = int(options.num_times)

srclist = options.srclist
#initial_date = options.date
tag_name = options.tag_name
time_decor = options.time_decor
freq_decor = options.freq_decor
over_sampled = options.over_sampled_kernel
oversampling_factor = int(options.oversampling_factor)

data_loc = options.data_loc

if options.no_beam:
    beam = False
else:
    beam = True

if data_loc[-1] == '/': data_loc = data_loc[:-1]


if options.metafits:
    try:
        f=fits.open(options.metafits)
    except Exception,e:
        print 'Unable to open metafits file %s: %s' % (options.metafits,e)
        exit(1)

    def test_avail(key):
        if not key in f[0].header.keys():
            print 'Cannot find %s in %s' % (key,options.metafits)
            exit(1)

    for key in ['DATE-OBS','FREQCENT','FINECHAN','INTTIME','BANDWDTH','AZIMUTH','ALTITUDE','DELAYS','RA','DEC']:
        test_avail(key)


    initial_date = f[0].header['DATE-OBS']

    time_res = float(f[0].header['INTTIME'])
    if options.time_res: time_res = float(options.time_res)

    ch_width = float(f[0].header['FINECHAN'])*1e+3
    freqcent = float(f[0].header['FREQCENT'])*1e+6
    b_width = float(f[0].header['BANDWDTH'])*1e+6
    base_low_freq = freqcent - (b_width/2) - (ch_width/2)

    freq_res = ch_width / 1.0e+6

    delay_str = f[0].header['DELAYS']
    print delay_str

    if delay_str == "32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32":
        delay_str = "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"

    delays = array([map(int,delay_str.split(',')),map(int,delay_str.split(','))])

    initial_ra_point = f[0].header['RA']
    dec_point = f[0].header['DEC']

    if options.chips_settings:
        ch_width = 80e+3
        time_res = 8.0
        low_freq = base_low_freq - (ch_width / 2.0)
    else:
        low_freq = base_low_freq

    num_freq_channels = int(1.28e+6 / ch_width)

else:
    initial_date = options.date
    time_res = float(options.time_res)
    ch_width = float(options.freq_res)*1e+6
    freq_res = ch_width / 1e+6
    b_width = float(options.bandwidth)*1e+6
    low_freq = float(options.freq_start) * 1e+6
    delays = zeros((2,16))
    num_freq_channels = int(options.num_freqs)

##TODO - make this generic, so you can use any telescope
##ephem Observer class, use this to compute LST from the date of the obs
MRO = Observer()
##Set the observer at Boolardy
MRO.lat, MRO.long, MRO.elevation = '-26:42:11.95', '116:40:14.93', 0
date,time = initial_date.split('T')
MRO.date = '/'.join(date.split('-'))+' '+time
initial_lst = float(MRO.sidereal_time())*R2D

print 'lst',initial_lst

if options.metafits:
    pass
else:
    initial_ra_point = initial_lst
    dec_point = MWA_LAT

ha_point = initial_lst - initial_ra_point

##delays = zeros((2,16))
##ha_point = 0.0
##dec_point = MWA_LAT
##initial_ra_point = initial_lst - ha_point


##====================================================================================

if options.chips_settings:
    if options.full_chips:
        good_chans = range(0,16)
    ##Ignores first and last channels for CHIPS settings
    else:
        good_chans = range(1,15)
    central_freq_chan = 8

elif options.num_freqs:
    good_chans = arange(int(options.num_freqs))
    central_freq_chan = 0

else:
    ##Unflagged channel numbers
    good_chans = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27,28,29]
    #good_chans = xrange(32)
    central_freq_chan = 15

##Flagged channel numbers
#bad_chans = [0,1,16,30,31]

band_num = int(options.band_num)
base_freq = ((band_num - 1)*(b_width/24.0)) + low_freq

#start_tstep,end_tstep = map(float,options.time.split(','))
#tsteps = arange(start_tstep,end_tstep,time_int)

cwd = getcwd()
tmp_dir = cwd+'/tmp'
if not path.exists(tmp_dir):
    makedirs(tmp_dir)

if options.data_loc:
    data_loc = options.data_loc
else:
    data_loc = cwd + '/data'
##Check to see if data directory exists; if not create it
if not path.exists(data_loc):
    makedirs(data_loc)

###---------------------------------------------------##
###---------------------------------------------------##


#print('initial_ra_point', initial_ra_point)

if options.phase_centre:
    ra_phase, dec_phase = map(float,options.phase_centre.split(','))
else:
    ra_phase, dec_phase = initial_ra_point + ((time_res / 2.0)*SOLAR2SIDEREAL*(15.0/3600.0)), dec_point

##Use a template uvfits file because uvfits are stoopid
base_uvfits_loc = "%s/telescopes/%s/%s_template.uvfits" %(MAJICK_DIR,options.telescope,options.telescope)
base_uvfits = fits.open(base_uvfits_loc)

base_data = base_uvfits[0].data
base_header = base_uvfits[0].header
antenna_table = base_uvfits[1].data
antenna_header = base_uvfits[1].header

##Get the local topocentric X,Y,Z values for the MWA using the local topocentric e,n,h
##values from the antenna locs in MWA_Tools
array_layout = "%s/telescopes/%s/antenna_locations_%s.txt" %(MAJICK_DIR,options.telescope,options.telescope)

anntenna_locs = loadtxt(array_layout)
X,Y,Z = enh2xyz(anntenna_locs[:,0],anntenna_locs[:,1],anntenna_locs[:,2],MWA_LAT*D2R)

num_baselines = (len(X)*(len(X)-1)) / 2

##Go through the uvfits file and replace the current X,Y,Z locations
##so we know for sure what array layout we have
for i in xrange(len(X)):
    antenna_table['STABXYZ'][i] = array([X[i],Y[i],Z[i]])

##Make an antenna dict to work out baseline lengths
antennas = {}
for i in xrange(len(X)):
    antennas['ANT%03d' %(i+1)] = array([X[i],Y[i],Z[i]])

baselines = base_data['BASELINE']

##This should be in meters
xyz_lengths = []

for baseline in baselines:
    ant2 = mod(baseline, 256)
    ant1 = (baseline - ant2)/256
    #self.antenna_pairs.append(('ANT%03d' %ant1,'ANT%03d' %ant2))
    x_length,y_length,z_length = antennas['ANT%03d' %ant1] - antennas['ANT%03d' %ant2]
    xyz_lengths.append([x_length,y_length,z_length])

xyz_lengths = array(xyz_lengths)


time_range = time_start + arange(num_times)*time_res
num_time_steps = len(time_range)

##Find the current JD and split it into int a float
##like the uvfits file likes
int_jd, float_jd = calc_jdcal(initial_date)


sim_freqs = []
for chan in good_chans:
    ##Take the band base_freq and add on fine channel freq
    freq = base_freq + (chan*ch_width)+ (ch_width / 2.0)
    freq_cent = freq + (ch_width / 2.0)
    sim_freqs.append(freq)

    ##Set the band central frequency
    if chan == central_freq_chan:
        band_freq_cent = freq_cent


sim_freqs = array(sim_freqs)


##Calc the julian date and split in the way uvfits likes
int_jd, float_jd = calc_jdcal(initial_date)

##Need an array the length of number of baselines worth of the fractional jd date
float_jd_array = ones(num_baselines)*float_jd

##Create empty data structures for final uvfits file

n_data = num_baselines*num_time_steps

v_container = zeros((n_data,1,1,1,num_freq_channels,4,3))
uus = zeros(n_data)
vvs = zeros(n_data)
wws = zeros(n_data)
baselines_array = zeros(n_data)
date_array = zeros(n_data)



##For each saved frequency channel visi data, read in and stick in uvfits container
for freq,chan in zip(sim_freqs,good_chans):
    visi_data = load('%s/%s_%.3f.npz' %(tmp_dir,tag_name,freq))['visi_data']

    ##visi_data should contain all timesteps and baselines for one frequecy channel
    ##in the correct order, so just bung it in
    v_container[:,0,0,0,chan,:,:] = visi_data

    cmd = 'rm %s/%s_%.3f.npz' %(tmp_dir,tag_name,freq)
    call(cmd,shell=True)

##Only set the u,v,w, time arrays etc once
for time_ind,time in enumerate(time_range):

    array_time_loc = num_baselines*time_ind

    ##Convert the time offset into a sky offset in degrees
    ##Add in half a time resolution step to give the central LST
    sky_offset = (((time + (time_res / 2.0))*SOLAR2SIDEREAL)*(15.0/3600.0))

    ##Currently always point to zenith
    #ra_point = initial_ra_point + sky_offset
    #if ra_point >=360.0: ra_point -= 360.0
    lst = initial_lst + sky_offset
    if lst >=360.0: lst -= 360.0

    ##TODO get ha_point from metafits
    ra_point = lst - ha_point
    ha_phase = lst - ra_phase

    ##Calc u,v,w in meters
    if options.no_phase_tracking:
        u,v,w = get_uvw(xyz_lengths[:,0],xyz_lengths[:,1],xyz_lengths[:,2],dec_point*D2R,ha_point*D2R)
    else:
        u,v,w = get_uvw(xyz_lengths[:,0],xyz_lengths[:,1],xyz_lengths[:,2],dec_phase*D2R,ha_phase*D2R)

    ##u,v,w stored in seconds by uvfits files
    chan_uu = u / VELC
    chan_vv = v / VELC
    chan_ww = w / VELC

    uus[array_time_loc:array_time_loc+num_baselines] = chan_uu
    vvs[array_time_loc:array_time_loc+num_baselines] = chan_vv
    wws[array_time_loc:array_time_loc+num_baselines] = chan_ww

    ##Fill in the baselines using the first time and freq uvfits
    baselines_array[array_time_loc:array_time_loc+num_baselines] = base_data['BASELINE']

    ##Fill in the fractional julian date, after adding on the appropriate amount of
    ##time - /(24*60*60) because julian number is a fraction of a whole day
    adjust_float_jd_array = float_jd_array + (time / (24.0*60.0*60.0))
    date_array[array_time_loc:array_time_loc+num_baselines] = adjust_float_jd_array


output_uvfits_name = "%s/%s_t%02d_f%.3f_band%02d.uvfits" %(data_loc,tag_name,time_res,ch_width/1e+6,band_num)

create_uvfits(v_container=v_container,freq_cent=band_freq_cent, ra_point=ra_phase, dec_point=dec_phase,
    output_uvfits_name=output_uvfits_name,uu=uus,vv=vvs,ww=wws,
    baselines_array=baselines_array,date_array=date_array,date=initial_date,
    central_freq_chan=central_freq_chan,ch_width=ch_width,template_uvfits=base_uvfits,
    int_jd=int_jd)
