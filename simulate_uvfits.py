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
from os import environ,getcwd,chdir,makedirs,path
from numpy import floor,random,savez_compressed,load
from os import environ
import pickle
from simulate_uvfits_lib import *
from subprocess import call

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

parser.add_option('-b', '--diffuse', default=False, action='store_true',
    help='Add to include the 2016 gsm sky model')

parser.add_option('-c', '--tag_name', 
    help='Enter tag name for output uvfits files')

parser.add_option('-d', '--date', default='2000-01-01T00:00:00',
    help="Enter date to start the observation on (YYYY-MM-DDThh:mm:ss), default='2000-01-01T00:00:00'")

parser.add_option('-e', '--base_uvfits', default=False, 
    help='Base fits file name and location (e.g. /location/file/uvfits_tag) tag to add diffuse model to (not needed if generating uvfits from srclist)')

#parser.add_option('-f', '--freq_start',
    #help='Enter lowest frequency (MHz) to simulate - this is lower band edge i.e. for freq_res=0.04, freq_start=167.035 will be simulated at 167.055')

parser.add_option('-g', '--no_beam', default=False, action='store_true',
    help='Add to switch off the beam')

parser.add_option('-i', '--data_loc', default='./data',
    help='Location to output the uvfits to OR location of uvfits if just adding diffuse model. Default = ./data')

parser.add_option('-j', '--telescope', default='MWA_phase1',
    help='Uses the array layout and primary beam model as stored in MAJICK_DIR/telescopes - defaults to MWA_phase1')

parser.add_option('-k', '--fix_beam', default=False, action='store_true',
    help='Forces the MWA beam to be fixed to 186.235MHz, to be used in conjection with CHIPS_FIXBEAM')

parser.add_option('-l', '--multi_process', default=False,
    help='Switches on multiprocessing, using the number of processes given e.g. --multi_process=8')

parser.add_option('-m', '--num_times', 
    help='Enter number of times steps to simulate')

parser.add_option('-n', '--num_freqs',default=32,
    help='Enter number of frequency channels per band to simulate - set to usual 32 for MWA')

parser.add_option('-o', '--wproj',default=False,action='store_true',
    help='Add to enable w-projection')

parser.add_option('-p', '--phase_centre', default=False,
    help='Phase centre of the observation in degrees as RA,DEC - as a default tracks the intial zenith point')

parser.add_option('-q', '--l_value',
    help='l offset value for testing')

parser.add_option('-r', '--new_uvfits', default=False, action='store_true',
    help='Add to create a fully new uvfits file for the diffuse simulation')

parser.add_option('-s', '--srclist', default=False,
    help='Enter name of srclist from which to add point sources')

parser.add_option('-t', '--time_start', 
    help='Enter lowest time offset from start date to simulate (s)')

parser.add_option('-u', '--clobber',default=False,action='store_true',
    help='Add to change "overwrite" to "clobber" - astropy on gstar is old version')

parser.add_option('-v', '--no_phase_tracking',default=False,action='store_true',
    help='Add to turn off phase tracking')

parser.add_option('-w', '--add_to_existing',default=False,action='store_true',
    help='Add to add simulations to previous uvfits')

parser.add_option('-x', '--time_res', default=False,
    help='Enter time resolution (s) of observations, default to what is in metafits')

parser.add_option('-y', '--freq_res', default=0.04,
    help='Enter frequency resolution (MHz) of observations, default=0.04')

parser.add_option('-z', '--freq_decor', default=True, action='store_false',
    help='Add to include freq decorrelation in the simulation')

parser.add_option('--over_sampled_kernel', default=False, action='store_true',
    help='Add to switch the kernel to the oversampled version rather than the phase shifting niceness')

parser.add_option('--diffuse_test', default=False, action='store_true',
    help='Add to make a single point source image for testing diffuse rather than use the GSM')

parser.add_option('--single_baseline', default=False, action='store_true',
    help='Add to only simulate one baseline - good for testing')

parser.add_option('--oversampling_factor', default=99,
    help='Set the oversampling factor if using an oversampled kernel')

parser.add_option('--metafits',
    help='Enter name of metafits file to base obs on')

parser.add_option('--band_num',
    help='Enter band number to simulate')

options, args = parser.parse_args()

if options.diffuse_test:
    MWA_LAT = 0

#freq_start = float(options.freq_start)
#freq_res = float(options.freq_res)
num_freq_channels = int(options.num_freqs)

time_start = float(options.time_start)
#time_res = float(options.time_res)
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
low_freq = freqcent - (b_width/2) - (ch_width/2)

freq_res = ch_width / 1.0e+6

delay_str = f[0].header['DELAYS']
print delay_str

if delay_str == "32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32":
    delay_str = "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"

delays = array([map(int,delay_str.split(',')),map(int,delay_str.split(','))])

##TODO - make this generic, so you can use any telescope
##ephem Observer class, use this to compute LST from the date of the obs 
MRO = Observer()
##Set the observer at Boolardy
MRO.lat, MRO.long, MRO.elevation = '-26:42:11.95', '116:40:14.93', 0
date,time = initial_date.split('T')
MRO.date = '/'.join(date.split('-'))+' '+time
initial_lst = float(MRO.sidereal_time())*R2D

initial_ra_point = f[0].header['RA']
ha_point = initial_lst - initial_ra_point
dec_point = f[0].header['DEC']

##delays = zeros((2,16))
##ha_point = 0.0
##dec_point = MWA_LAT
##initial_ra_point = initial_lst - ha_point


##====================================================================================

##Unflagged channel numbers
good_chans = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27,28,29]
central_freq_chan = 15
#good_chans = xrange(32)
#good_chans = [2,3]
#central_freq_chan = 2

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

if srclist:
    try:
        source_src_info = open(srclist,'r').read().split('ENDSOURCE')
        del source_src_info[-1]
    except:
        print("Cannot open %s, cannot calibrate without source list" %srclist)
        exit(1)
        
    sources = {}
    for source_info in source_src_info:
        source = create_calibrator(source_info)
        sources[source.name] = source
        
if options.diffuse:
    ##If using fix_beam, don't need to load beam images multiple times:
    ##big saving computationally
    if options.fix_beam:
        beam_loc = '%s/telescopes/%s/primary_beam/data' %(MAJICK_DIR,options.telescope)
        ##If using CHIPS in fix beam mode, set to 186.235MHz (+0.02 for half channel width)
        image_XX = my_loadtxt('%s/beam_%s_186255000.000_XX.txt' %(beam_loc,delay_str))
        image_YY = my_loadtxt('%s/beam_%s_186255000.000_YY.txt' %(beam_loc,delay_str))
    else:
        image_XX = False
        image_YY = False
        
if over_sampled:
    #print 'Begun creating oversampled kernel'
    beam_loc = '%s/telescopes/%s/primary_beam/data' %(MAJICK_DIR,options.telescope)
    ##If using CHIPS in fix beam mode, set to 186.235MHz (+0.02 for half channel width)
    image_XX = my_loadtxt('%s/beam_%s_186255000.000_XX.txt' %(beam_loc,delay_str))
    image_YY = my_loadtxt('%s/beam_%s_186255000.000_YY.txt' %(beam_loc,delay_str))
    
    oversamp_XX = zeros(((oversampling_factor)*KERNEL_SIZE,(oversampling_factor)*KERNEL_SIZE))
    oversamp_YY = zeros(((oversampling_factor)*KERNEL_SIZE,(oversampling_factor)*KERNEL_SIZE))
    
    lower = (oversampling_factor*KERNEL_SIZE) / 2 - (KERNEL_SIZE / 2)
    
    oversamp_XX[lower:lower+KERNEL_SIZE,lower:lower+KERNEL_SIZE] = image_XX
    oversamp_YY[lower:lower+KERNEL_SIZE,lower:lower+KERNEL_SIZE] = image_YY
    
    uv_kernel_XX = create_uv_kernel(image_kernel=oversamp_XX)
    uv_kernel_YY = create_uv_kernel(image_kernel=oversamp_YY)
    
    num_pixel = uv_kernel_XX.shape[0]
    
    l_reso = 2.0 / KERNEL_SIZE
    n2max = 1.0 / l_reso
    l_mesh, m_mesh = sample_image_coords(n2max=n2max,l_reso=l_reso)
    l_reso = l_mesh[0,1] - l_mesh[0,0]
    
    max_u = (0.5 / l_reso)
    u_reso = (2*max_u) / float(num_pixel)
    u_offset = -(u_reso / 2.0)
    
    u_range_kernel = linspace(-max_u-u_offset,max_u+u_offset,num_pixel)
    v_range_kernel = linspace(-max_u-u_offset,max_u+u_offset,num_pixel)
    
    #u_range_kernel = linspace(-max_u+u_offset,max_u-u_offset,num_pixel)
    #v_range_kernel = linspace(-max_u+u_offset,max_u-u_offset,num_pixel)
    
    print 'Done creating oversampled kernel'
    print 'Oversampling %d u_reso %.5f' %(oversampling_factor,u_range_kernel[1] - u_range_kernel[0])
    
else:
    uv_kernel_XX = None
    uv_kernel_YY = None
    oversampling_factor = None
    u_range_kernel = None
    v_range_kernel = None


if options.diffuse_test:
    half_width = 1.0
    image_size = 3051
    l_range = linspace(-half_width,half_width,2.0*image_size + 1)
    m_range = linspace(-half_width,half_width,2.0*image_size + 1)
    #l_reso_test = (2.0*half_width) / (2.0*image_size + 1)
    l_reso_test = l_range[1] - l_range[0]
    
    l_off = int(options.l_value)
    m_off = 0
    
    image = zeros((int(2.0*image_size+1),int(2.0*image_size+1)))
    l = l_range[image_size+l_off]
    m = m_range[image_size+m_off]
    image[image_size+m_off,image_size-l_off] = 1.0
    
    IMGSIZE = image.shape[0]
    img_oversamp = 3
    
    oversamp_image = zeros(((img_oversamp)*IMGSIZE,(img_oversamp)*IMGSIZE))
    lower = (img_oversamp*IMGSIZE) / 2 - (IMGSIZE / 2)
    oversamp_image[lower:lower+IMGSIZE,lower:lower+IMGSIZE] = image
    
    uv_data_array_test, u_sim_test, v_sim_test, u_reso_test = convert_image_lm2uv(image=oversamp_image,l_reso=l_reso_test)
    

time_range = time_start + arange(num_times)*time_res
num_time_steps = len(time_range)

##Find the current JD and split it into int a float
##like the uvfits file likes
int_jd, float_jd = calc_jdcal(initial_date)

#@profile
def simulate_frequency_channel(all_args=None):
    
    freq_chan_index,freq = all_args
    freq_cent = freq + (ch_width / 2.0)
    n_data = num_baselines * num_time_steps
    
    ##Create empty data structures to save data into
    visi_data = zeros((n_data,4,3))
    
    for time_ind,time in enumerate(time_range):
        #print 'Doing band %02d freq %.3f time %02d'  %(band_num,freq,time)
        
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
        if ra_point < 0.0: ra_point += 360.0
        
        #ha_point = lst - ra_point
        ha_phase = lst - ra_phase
        
        
        ##If not phase tracking, stick the zero point of l,m,n at zenith
        ##This means as observation goes on, a source dirfts through the
        ##l,m plane
        if options.no_phase_tracking:
            coord_centre_ra = ra_point
            coord_centre_dec = dec_point
        else:
            ##If phase tracking, sets the zero point of l,m,n to ra_phase,dec_phase
            ##So really for a source, l,m never changes as moves with the sky
            coord_centre_ra = ra_phase
            coord_centre_dec = dec_phase
        
        if options.single_baseline:
            baselines = xrange(1)
        else:
            baselines = xrange(num_baselines)

        for baseline in baselines:
            
            #xyz = antenna_table['STABXYZ'] * (freq_cent / VELC)
            
            x_length,y_length,z_length = xyz_lengths[baseline,:]
            ##The old way of non-phase tracking
            if options.no_phase_tracking:
                u,v,w = get_uvw_freq(x_length,y_length,z_length,dec_point*D2R,ha_point*D2R,freq=freq_cent)
            else:
                u,v,w = get_uvw_freq(x_length,y_length,z_length,dec_phase*D2R,ha_phase*D2R,freq=freq_cent)
        
            ##Container for this particular baseline uv data
            uv_baseline = zeros((4,3))
            ##Set weights to one
            uv_baseline[:,2] = 1.0
            
            if srclist:
                ##For every source in the sky
                for name,source in sources.iteritems():
                    ##If source is below the horizon, forget about it
                    if source.skip[time_ind]:
                        pass
                    ##Otherwise, proceed
                    else:
                        if options.no_phase_tracking:
                            phasetrack = False
                        else:
                            phasetrack = True
                        model_xxpol,model_xypol,model_yxpol,model_yypol = model_vis(u=u,v=v,w=w,source=source,coord_centre_ra=coord_centre_ra,
                            coord_centre_dec=coord_centre_dec,LST=lst,x_length=x_length,y_length=y_length,z_length=z_length,
                            freq_decor=freq_decor,freq=freq_cent,time_decor=time_decor,time_res=time_res,chan_width=freq_res*1e+6,beam=beam,
                            phasetrack=phasetrack,freqcent=freq_cent,freq_chan_index=freq_chan_index)
                        
                        uv_baseline[0,:] += array([model_xxpol.real,model_xxpol.imag,0.0000])
                        uv_baseline[1,:] += array([model_yypol.real,model_yypol.imag,0.0000])
                        uv_baseline[2,:] += array([model_xypol.real,model_xypol.imag,0.0000])
                        uv_baseline[3,:] += array([model_yxpol.real,model_yxpol.imag,0.0000])
            
            visi_data[array_time_loc+baseline,:,:] = uv_baseline
            
    savez_compressed('%s/%s_%.3f.npz' %(tmp_dir,tag_name,freq),visi_data=visi_data)
            
    ###Add degridding visibilities
    #if options.diffuse:
        ###If doing tests, already made image at start of options
        #if options.diffuse_test:
            #l_off = int(options.l_value)
            #m_off = 0
            ###Figure out where the point source is so can compare to analytic
            #test_srclist = open('srclist_%03d.txt' %l_off,'w+')
            
            #l = l_range[image_size+l_off]
            #m = m_range[image_size+m_off]
            #ra_source = lst + arcsin(l)*R2D #- ((time_res / 2.0)*SOLAR2SIDEREAL*(15.0/3600.0))
            #dec_source = MWA_LAT + arcsin(m)*R2D
            #print 'l offset %d is %.2f deg off zenith' %(l,arcsin(l)*R2D)
            #if ra_source > 360.0: ra_source -= 360.0
            
            #test_srclist.write('SOURCE bleh%d%d %.5f %.5f\n' %(l_off,m_off,ra_source/15.0,dec_source))
            #test_srclist.write('FREQ 160e+6 1.0 0 0 0\n')
            #test_srclist.write('FREQ 180e+6 1.0 0 0 0\n')
            #test_srclist.write('ENDSOURCE\n')
            #test_srclist.close()

            ###Need to assign these to the test values form above,
            ###because we also declare them when not using the test option and python
            ###throws a wobbly if we don't
            #uv_data_array, u_sim, v_sim, u_reso = uv_data_array_test, u_sim_test, v_sim_test, u_reso_test
            #l_reso = l_reso_test
        
        ###Otherwise, generate a diffuse sky image using the GSM, and FT to uv-space
        #else:
            ###TODO - I think this may be half a time step off, need to give the central time
            #image, l_reso = generate_gsm_2016(freq=freq_cent,this_date=this_date,observer=MRO)
            #uv_data_array, u_sim, v_sim, u_reso = convert_image_lm2uv(image=image,l_reso=l_reso)
        
        ###If we only want one baseline, make the range for one baseline
        #if options.single_baseline:
            #baseline_range = xrange(1)
        #else:
            #baseline_range = xrange(len(base_data))
            
        #ra0,dec0 = ra_phase*D2R,dec_phase*D2R
        #phase_centre = [ra0,dec0]
        #skipped_gsm = 0
        #outside_uv = 0
        
        ###For each baseline
        #for baseline in baseline_range:
            ###Due to the resolution of the GSM not all baselines will fall on the u,v
            ###plane (u_extent = 1 / l_reso), so skip those that fail
            ##try:
            ####u,v,w, stored in seconds in the uvfits
            ###make sure we use the same u,v,w aleady stored in the uvfits
            #u = write_data[baseline][0] * freq_cent
            #v = write_data[baseline][1] * freq_cent
            #w = write_data[baseline][2] * freq_cent
            
            #print 'u,v is',u,v
            
            #outside = False
            #half_grid_width = floor(KERNEL_SIZE / 2.0) * u_reso
            
            #if u < (u_sim.min() + half_grid_width) or u > (u_sim.max() - half_grid_width): outside = True
            #if v < (v_sim.min() + half_grid_width) or v > (v_sim.max() - half_grid_width): outside = True
            
            #if outside:
                #skipped_gsm += 1
                #if options.new_uvfits:
                    #write_data[baseline][5][0,0,0,0,0,:] = array([0,0,0.0000])
                    #write_data[baseline][5][0,0,0,0,1,:] = array([0,0,0.0000])
            #else:
                #if beam:
                    #uv_complex_XX,uv_complex_YY,this_image_XX,this_image_YY = reverse_grid(uv_data_array=uv_data_array, l_reso=l_reso, u=u, v=v,
                        #kernel=options.telescope,freq_cent=freq_cent,u_reso=u_reso,u_sim=u_sim,v_sim=v_sim,xyz_lengths=xyz_lengths[baseline,:],
                        #phase_centre=phase_centre,time_res=time_res,freq_int=freq_res,central_lst=lst*D2R,time_decor=time_decor,
                        #freq_decor=freq_decor,fix_beam=options.fix_beam,image_XX=image_XX,image_YY=image_YY,wproj=options.wproj,
                        #uv_kernel_XX=uv_kernel_XX,uv_kernel_YY=uv_kernel_YY,u_range_kernel=u_range_kernel,v_range_kernel=v_range_kernel,over_sampled=over_sampled,over_sampling=oversampling_factor)
                #else:
                    #uv_complex_XX,uv_complex_YY,this_image_XX,this_image_YY = reverse_grid(uv_data_array=uv_data_array, l_reso=l_reso, u=u, v=v,
                        #kernel='gaussian',freq_cent=freq_cent,u_reso=u_reso,u_sim=u_sim,v_sim=v_sim,xyz_lengths=xyz_lengths[baseline,:],
                        #phase_centre=phase_centre,time_res=time_res,freq_int=freq_res,central_lst=lst*D2R,time_decor=time_decor,
                        #freq_decor=freq_decor,fix_beam=options.fix_beam,image_XX=image_XX,image_YY=image_YY,wproj=options.wproj)
                
                #if options.new_uvfits:
                    #write_data[baseline][5][0,0,0,0,0,:] = array([real(uv_complex_XX),imag(uv_complex_XX),0.0000])
                    #write_data[baseline][5][0,0,0,0,1,:] = array([real(uv_complex_YY),imag(uv_complex_YY),0.0000])
                #else:
                    #write_data[baseline][5][0,0,0,0,0,:] += array([real(uv_complex_XX),imag(uv_complex_XX),0.0000])
                    #write_data[baseline][5][0,0,0,0,1,:] += array([real(uv_complex_YY),imag(uv_complex_YY),0.0000])
                


sim_freqs = []
for chan in good_chans:
    ##Take the band base_freq and add on fine channel freq
    freq = base_freq + (chan*ch_width)
    sim_freqs.append(freq)
    
    ##Set the band central frequency
    if chan == central_freq_chan:
        band_freq_cent = freq
    
    
sim_freqs = array(sim_freqs)

##Weight all of the 
if srclist:
    if beam:
        for name,source in sources.iteritems():
            #weight_by_beam(source=source,freqcent=freq_cent,LST=initial_lst,delays=delays,beam=beam,fix_beam=options.fix_beam,time_range)
            extrapolate_and_cal_beam(sources=sources,initial_lst=initial_lst,delays=delays,beam=beam,
                fix_beam=options.fix_beam,time_range=time_range,time_res=time_res,sim_freqs=sim_freqs,freqcent=band_freq_cent)

all_args_list = [[freq_ind,freq] for freq_ind,freq in enumerate(sim_freqs)]

##TODO - ONLY WORKS FOR CERTAIN PYTHON VERSIONS
##DAGNABIT
print 'Now calculating visibilities'
if options.multi_process:
    from multiprocessing import Pool
    from functools import partial
        
    pool = Pool(processes=int(options.multi_process))
    #func = partial(this_main, antenna_table, base_data, base_uvfits)
    pool.map(simulate_frequency_channel, all_args_list)

    pool.close()
    pool.join()
    
else:
    for all_args in all_args_list:
        simulate_frequency_channel(all_args)
        print '----------------------------------'
            
##Read in all the saved tmp npz files and write out a uvfits file

print 'Now pull in the data and create uvfits'

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




