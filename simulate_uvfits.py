#!/usr/bin/python
import optparse
#import matplotlib.pyplot as plt
#from imager_lib import *
from calibrator_classes import *
from gridding_functions import *
from time import time
from generate_gsm_2016 import generate_gsm_2016
try:
	import pyfits as fits
except ImportError:
	from astropy.io import fits
from ephem import Observer,degrees
from os import environ
from jdcal import gcal2jd
from numpy import floor

MAJICK_DIR = environ['MAJICK_DIR']
	
D2R = pi/180.0
R2D = 180.0/pi
VELC = 299792458.0
MWA_LAT = -26.7033194444

def enh2xyz(east,north,height,latitiude):
	sl = sin(latitiude)
	cl = cos(latitiude)
	X = -north*sl + height*cl
	Y = east
	Z = north*cl + height*sl
	return X,Y,Z

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

def calc_jdcal(date):
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

def create_uvfits(u_coords=None, v_coords=None, w_coords=None, central_frequency=None, ra_point=None, dec_point=None, oskar_vis_tag=None, output_uvfits_name=None,date=None):
	
	int_jd, float_jd = calc_jdcal(date)
	
	template_file = fits.open(template_uvfits)
	template_data = template_file[0].data
	antenna_table = template_file[1].data

	# Create uv structure by hand, probably there is a better way of doing this but the uvfits structure is kind of finicky
	n_freq = 1 # only one frequency per uvfits file as read by the RTS

	n_data = len(template_data)

	v_container = zeros((n_data,1,1,1,n_freq,4,3))
	uu = zeros(n_data)
	vv = zeros(n_data)
	ww = zeros(n_data)
	baseline = zeros(n_data)
	date_array = zeros(n_data)
	
	xx_us,xx_vs,xx_ws,xx_res,xx_ims = get_osk_data(oskar_vis_tag=oskar_vis_tag,polarisation='XX')
	yy_us,yy_vs,yy_ws,yy_res,yy_ims = get_osk_data(oskar_vis_tag=oskar_vis_tag,polarisation='YY')
	xy_us,xy_vs,xy_ws,xy_res,xy_ims = get_osk_data(oskar_vis_tag=oskar_vis_tag,polarisation='XY')
	yx_us,yx_vs,yx_ws,yx_res,yx_ims = get_osk_data(oskar_vis_tag=oskar_vis_tag,polarisation='YX')


	for i in range(len(template_data)):
		xx_list = [xx_res[i],xx_ims[i],1.0]
		yy_list = [yy_res[i],yy_ims[i],1.0]
		xy_list = [xy_res[i],xy_ims[i],1.0]
		yx_list = [yx_res[i],yx_ims[i],1.0]
		
		uvdata = [xx_list,yy_list,xy_list,yx_list]
		uvdata = array(uvdata)
		uvdata.shape = (4,3)
		
		v_container[i] = uvdata
		uu[i] = xx_us[i] / freq_cent
		vv[i] = xx_vs[i] / freq_cent
		ww[i] = xx_ws[i] / freq_cent
		baseline[i] = template_data[i][3]
		date_array[i] = float_jd
		rotate_phase(xx_ws[i],v_container[i][0,0,0,0,:,:])

	##UU, VV, WW don't actually get read in by RTS - might be an issue with
	##miriad/wsclean however, as it looks like oskar w = negative maps w
	uvparnames = ['UU','VV','WW','BASELINE','DATE']
	parvals = [uu,vv,ww,baseline,date_array]
		
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
	###Oskar/CASA for some reason adds half of the frequency specified in the 
	###simulation setup. I think this is happens because CASA is unsure
	###what 'channel' the data is - when you run with multiple channels, they
	###are all set to spw = 0, but the output freq is correct. Somethig funky anyway
	###For one channel, set by hand
	uvhdu.header['CRVAL4'] = central_frequency ##(sim freq + half channel width)
	uvhdu.header['CRPIX4'] = template_file[0].header['CRPIX4']
	uvhdu.header['CDELT4'] = template_file[0].header['CDELT4']

	uvhdu.header['CTYPE5'] = template_file[0].header['CTYPE5']
	uvhdu.header['CRVAL5'] = template_file[0].header['CRVAL5']
	uvhdu.header['CRPIX5'] = template_file[0].header['CRPIX5']
	uvhdu.header['CDELT5'] = template_file[0].header['CDELT5']

	uvhdu.header['CTYPE6'] = template_file[0].header['CTYPE6']
	uvhdu.header['CRVAL6'] = template_file[0].header['CRVAL6']
	uvhdu.header['CRPIX6'] = template_file[0].header['CRPIX6']
	uvhdu.header['CDELT6'] = template_file[0].header['CDELT6']

	uvhdu.header['CTYPE7'] = template_file[0].header['CTYPE7']
	uvhdu.header['CRVAL7'] = template_file[0].header['CRVAL7']
	uvhdu.header['CRPIX7'] = template_file[0].header['CRPIX7']
	uvhdu.header['CDELT7'] = template_file[0].header['CDELT7']

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
	
	##ANTENNA TABLE MODS======================================================================

	template_file[1].header['FREQ'] = freq_cent
	
	##MAJICK uses this date to set the LST
	dmy, hms = date.split()
	day,month,year = map(int,dmy.split('-'))
	hour,mins,secs = map(float,hms.split(':'))
	
	rdate = "%d-%02d-%2dT%2d:%2d:%.2f" %(year,month,day,hour,mins,secs)
	
	template_file[1].header['RDATE'] = rdate

	## Create hdulist and write out file
	hdulist = fits.HDUList(hdus=[uvhdu,template_file[1]])
	hdulist.writeto(output_uvfits_name,clobber=True)
	template_file.close()
	hdulist.close()




##---------------------------------------------------##
##----------OBSERVATION SETTINGS----------------------##

parser = optparse.OptionParser()

parser.add_option('-f', '--freq_start',
	help='Enter lowest frequency (MHz) to simulate - this is lower band edge i.e. for freq_res=0.04, freq_start=167.035 will be simulated at 167.055')
	
parser.add_option('-n', '--num_freqs',
	help='Enter number of frequency channels to simulate')

parser.add_option('-y', '--freq_res', default=0.04,
	help='Enter frequency resolution (MHz) of observations, default=0.04')
	
parser.add_option('-t', '--time_start', 
	help='Enter lowest time offset from start date to simulate (s)')

parser.add_option('-x', '--time_res', default=2.0,
	help='Enter time resolution (s) of observations, default=2.0')
	
parser.add_option('-m', '--num_times', 
	help='Enter number of times steps to simulate')

parser.add_option('-s', '--srclist', default=False,
	help='Enter name of srclist from which to add point sources')

parser.add_option('-g', '--beam', default=False, action='store_true',
	help='Add in the beam to all simulations')

parser.add_option('-d', '--date', default='2000-01-01T00:00:00',
	help="Enter date to start the observation on (YYYY-MM-DDThh:mm:ss), default='2000-01-01T00:00:00'")

parser.add_option('-c', '--tag_name', 
	help='Enter tag name for output uvfits files')

parser.add_option('-a', '--time_decor', default=False, action='store_true',
	help='Add to include time_decorrelation in the simulation')

parser.add_option('-z', '--freq_decor', default=False, action='store_true',
	help='Add to include freq decorrelation in the simulation')

parser.add_option('-b', '--diffuse', default=False, action='store_true',
	help='Add to include the 2016 gsm sky model')

parser.add_option('-e', '--base_uvfits', default=False, 
	help='Base fits file name and location (e.g. /location/file/uvfits_tag) tag to add diffuse model to (not needed if generating uvfits from srclist)')

parser.add_option('-j', '--telescope', default='MWA_phase1',
	help='Uses the array layout and primary beam model as stored in MAJICK_DIR/telescopes - defaults to MWA_phase1')

parser.add_option('-i', '--data_loc', default='./data',
	help='Location to output the uvfits to OR location of uvfits if just adding diffuse model. Default = ./data')

parser.add_option('-p', '--phase_centre', default=False,
	help='Phase centre of the observation in degrees as RA,DEC - as a default tracks the intial zenith point')

options, args = parser.parse_args()

freq_start = float(options.freq_start)
freq_res = float(options.freq_res)
num_freqs = int(options.num_freqs)

time_start = float(options.time_start)
time_res = float(options.time_res)
num_times = int(options.num_times)

srclist = options.srclist
intial_date = options.date
tag_name = options.tag_name
time_decor = options.time_decor
freq_decor = options.freq_decor

data_loc = options.data_loc

if data_loc[-1] == '/': data_loc = data_loc[:-1]

###---------------------------------------------------##
###---------------------------------------------------##

##TODO - make this generic, so you can use any telescope
##ephem Observer class, use this to compute LST from the date of the obs 
MRO = Observer()
##Set the observer at Boolardy
MRO.lat, MRO.long, MRO.elevation = '-26:42:11.95', '116:40:14.93', 0
date,time = intial_date.split('T')
MRO.date = '/'.join(date.split('-'))+' '+time
intial_lst = float(MRO.sidereal_time())*R2D
#intial_ra_point = 356.303842095
intial_ra_point = float(MRO.sidereal_time())*R2D
#print intial_ra_point
dec_point = MWA_LAT

if options.phase_centre:
	ra_phase, dec_phase = map(float,options.phase_centre.split(','))
else:
	ra_phase, dec_phase = intial_ra_point, dec_point

##Get the local topocentric X,Y,Z values for the MWA using the local topocentric e,n,h
##values from the antenna locs in MWA_Tools

array_layout = "%s/telescopes/%s/antenna_locations_%s.txt" %(MAJICK_DIR,options.telescope,options.telescope)

anntenna_locs = loadtxt(array_layout)
X,Y,Z = enh2xyz(anntenna_locs[:,0],anntenna_locs[:,1],anntenna_locs[:,2],MWA_LAT*D2R)

##TODO - get the delays in a smart way
##TODO - check whether the desired pointing exists in a smart way
##       when using the MWA beam as a gridding function
delays = zeros((2,16))


if srclist:
	try:
		source_src_info = open(srclist,'r').read().split('ENDSOURCE')
		del source_src_info[-1]
	except:
		print("Cannot open %s, cannot calibrate without source list")
		exit(1)
		
	sources = {}
	for source_info in source_src_info:
		source = create_calibrator(source_info)
		sources[source.name] = source
		
	base_uvfits_loc = "%s/telescopes/%s/%s_template.uvfits" %(MAJICK_DIR,options.telescope,options.telescope)
	base_uvfits = fits.open(base_uvfits_loc)
	
	base_data = base_uvfits[0].data
	base_header = base_uvfits[0].header
	antenna_table = base_uvfits[1].data
	antenna_header = base_uvfits[1].header
	
##Sidereal seconds per solar seconds - ie if 1s passes on
##the clock, sky has moved by 1.00274 secs of angle
SOLAR2SIDEREAL = 1.00274

freq_range = freq_start + arange(num_freqs)*freq_res
time_range = time_start + arange(num_times)*time_res

for freq in freq_range:
	for time in time_range:
		print freq,time
		freq_cent = ((freq + freq_res / 2.0)*1e+6)
		
		if not srclist:
			if time_res < 1:
				base_uvfits = fits.open("%s_%.3f_%05.2f.uvfits" %(options.base_uvfits,freq,time))
			else:
				base_uvfits = fits.open("%s_%.3f_%02d.uvfits" %(options.base_uvfits,freq,int(time)))

			base_data = base_uvfits[0].data
			base_header = base_uvfits[0].header
			antenna_table = base_uvfits[1].data
			antenna_header = base_uvfits[1].header
		
		##Go through the uvfits file and replace the current X,Y,Z locations
		for i in xrange(len(X)):
			antenna_table['STABXYZ'][i] = [X[i],Y[i],Z[i]]

		##Scale the X,Y,Z by the wavelength of channel centre
		antennas = {}
		xyz = antenna_table['STABXYZ'] * (freq_cent / VELC)

		for i in xrange(len(xyz)):
			antennas['ANT%03d' %(i+1)] = xyz[i]
			
		baselines = base_data['BASELINE']
			
		xyz_lengths = []
			
		for baseline in baselines:
			ant2 = mod(baseline, 256)
			ant1 = (baseline - ant2)/256
			#self.antenna_pairs.append(('ANT%03d' %ant1,'ANT%03d' %ant2))
			x_length,y_length,z_length = antennas['ANT%03d' %ant1] - antennas['ANT%03d' %ant2]
			xyz_lengths.append([x_length,y_length,z_length])
		
		##Convert the time offset into a sky offset in degrees
		##Add in half a time resolution step to give the central LST
		sky_offset = (((time + (time_res / 2.0))*SOLAR2SIDEREAL)*(15.0/3600.0))

		##Currently always point to zenith
		ra_point = intial_ra_point + sky_offset
		if ra_point >=360.0: ra_point -= 360.0
		lst = intial_lst + sky_offset
		if lst >=360.0: lst -= 360.0
		
		ha_point = lst - ra_point
		ha_phase = lst - ra_phase
		
		##DO NOT ADD HALF A TIME RESOLUTION
		##make sure this all happens when reading in the uvfits
		this_date = add_time(intial_date,time)# + (time_res / 2.0))
		
		int_jd, float_jd = calc_jdcal(this_date)
		
		print('intial_date',intial_date,'this_date',this_date)
		
		#print 'srclist has been weighted by freq and beam'
		##GSM image and uv_data_array are the same for all baselines, for each time and freq
		
		
		if srclist:
	
			# Create uv structure by hand, probably there is a better way of doing this but the uvfits structure is kind of finicky
			n_freq = 1 # only one frequency per uvfits file as read by the RTS
			n_data = len(base_data)

			v_container = zeros((n_data,1,1,1,n_freq,4,3))
			uu = zeros(n_data)
			vv = zeros(n_data)
			ww = zeros(n_data)
			baselines_array = zeros(n_data)
			date_array = zeros(n_data)
			
			##TODO Weight each source by the beam pattern - do this to
			##calculate the beam at each point, as well as extrapolate the source
			##flux density to the current frequency
			for name,source in sources.iteritems():
				if options.beam:
					weight_by_beam(source=source,freqcent=freq_cent,LST=lst,delays=delays,beam=True)
				else:
					weight_by_beam(source=source,freqcent=freq_cent,LST=lst,beam=False)
			
			for baseline in xrange(len(base_data)):
				#print 'Simulating baseline %04d' %baseline
			#for baseline in range(0,1):
				x_length,y_length,z_length = xyz_lengths[baseline]
				##The old way of non-phase tracking
				#u,v,w = get_uvw(x_length,y_length,z_length,dec_point*D2R,ha_point*D2R)
				u,v,w = get_uvw(x_length,y_length,z_length,dec_phase*D2R,ha_phase*D2R)
			
				#print 'Adding point sources'
				uv_data_XX = array([0.0,0.0,1.0])
				uv_data_YY = array([0.0,0.0,1.0])
				
				##For every source in the sky
				for name,source in sources.iteritems():
					##If source is below the horizon, forget about it
					if source.skip:
						pass
					##Otherwise, proceed
					else:
						if time_decor and freq_decor:
							if options.beam:
								model_xxpol,model_yypol = model_vis_phasetrack(u=u,v=v,w=w,source=source,phase_ra=ra_phase,phase_dec=dec_phase,LST=lst,
											x_length=x_length,y_length=y_length,z_length=z_length,freq_decor=freq_res*1e+6,freq=freq_cent,time_decor=time_res,beam=True)
							else:
								model_xxpol,model_yypol = model_vis_phasetrack(u=u,v=v,w=w,source=source,phase_ra=ra_phase,phase_dec=dec_phase,LST=lst,
											x_length=x_length,y_length=y_length,z_length=z_length,freq_decor=freq_res*1e+6,freq=freq_cent,time_decor=time_res)
						elif time_decor:
							if options.beam:
								model_xxpol,model_yypol = model_vis_phasetrack(u=u,v=v,w=w,source=source,phase_ra=ra_phase,phase_dec=dec_phase,LST=lst,
											x_length=x_length,y_length=y_length,z_length=z_length,time_decor=time_res,beam=True)
							else:
								model_xxpol,model_yypol = model_vis_phasetrack(u=u,v=v,w=w,source=source,phase_ra=ra_phase,phase_dec=dec_phase,LST=lst,
											x_length=x_length,y_length=y_length,z_length=z_length,time_decor=time_res)
						elif freq_decor:
							if options.beam:
								model_xxpol,model_yypol = model_vis_phasetrack(u=u,v=v,w=w,source=source,phase_ra=ra_phase,phase_dec=dec_phase,LST=lst,
											x_length=x_length,y_length=y_length,z_length=z_length,freq_decor=freq_res*1e+6,freq=freq_cent,beam=True)
							else:
								model_xxpol,model_yypol = model_vis_phasetrack(u=u,v=v,w=w,source=source,phase_ra=ra_phase,phase_dec=dec_phase,LST=lst,
											x_length=x_length,y_length=y_length,z_length=z_length,freq_decor=freq_res*1e+6,freq=freq_cent)
						else:
							if options.beam:
								model_xxpol,model_yypol = model_vis_phasetrack(u=u,v=v,w=w,source=source,phase_ra=ra_phase,phase_dec=dec_phase,LST=lst,beam=True)
							else:
								model_xxpol,model_yypol = model_vis_phasetrack(u=u,v=v,w=w,source=source,phase_ra=ra_phase,phase_dec=dec_phase,LST=lst)
							
						uv_data_XX += array([real(model_xxpol),imag(model_xxpol),0.0000])
						uv_data_YY += array([real(model_yypol),imag(model_yypol),0.0000])
				
				###Could add in crazy weightings here
				##uv_data_XX[2] = 1.0
				##uv_data_YY[2] = 1.0
				
				##Enter the XX and YY info. Leave XY, YX as zero for now
				uvdata = [list(uv_data_XX),list(uv_data_YY), [0.0,0.0,0.0],[0.0,0.0,0.0]]
				uvdata = array(uvdata)
				uvdata.shape = (4,3)
				
				v_container[baseline] = uvdata
				uu[baseline] = u / freq_cent
				vv[baseline] = v / freq_cent
				ww[baseline] = w / freq_cent
				baselines_array[baseline] = base_data[baseline][3]
				date_array[baseline] = float_jd
				
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

			##Frequency information
			uvhdu.header['CTYPE4'] = 'FREQ'
			uvhdu.header['CRVAL4'] = freq_cent 
			uvhdu.header['CRPIX4'] = base_uvfits[0].header['CRPIX4']
			uvhdu.header['CDELT4'] = freq_res * 1e+6
			
			uvhdu.header['CTYPE5'] = base_uvfits[0].header['CTYPE5']
			uvhdu.header['CRVAL5'] = base_uvfits[0].header['CTYPE5']
			uvhdu.header['CRPIX5'] = base_uvfits[0].header['CRPIX5']
			uvhdu.header['CDELT5'] = base_uvfits[0].header['CDELT5']

			##RA phase information
			uvhdu.header['CTYPE6'] = base_uvfits[0].header['CTYPE6']
			uvhdu.header['CRVAL6'] = ra_phase
			uvhdu.header['CRPIX6'] = base_uvfits[0].header['CRPIX6']
			uvhdu.header['CDELT6'] = base_uvfits[0].header['CDELT6']

			##DEC phase information
			uvhdu.header['CTYPE7'] = base_uvfits[0].header['CTYPE7']
			uvhdu.header['CRVAL7'] = dec_phase
			uvhdu.header['CRPIX7'] = base_uvfits[0].header['CRPIX7']
			uvhdu.header['CDELT7'] = base_uvfits[0].header['CDELT7']

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
			#uvhdu.header['PZERO5'] = 0.0

			uvhdu.header['OBJECT']  = 'Undefined'                                                           
			uvhdu.header['OBSRA']   = ra_phase                                          
			uvhdu.header['OBSDEC']  = dec_phase
			
			##ANTENNA TABLE MODS======================================================================

			base_uvfits[1].header['FREQ'] = freq_cent
			
			###MAJICK uses this date to set the LST
			#dmy, hms = date.split()
			#day,month,year = map(int,dmy.split('-'))
			#hour,mins,secs = map(float,hms.split(':'))
			
			#rdate = "%d-%02d-%2dT%2d:%2d:%.2f" %(year,month,day,hour,mins,secs)
			print(this_date)
			base_uvfits[1].header['RDATE'] = this_date

			## Create hdulist and write out file
			write_uvfits = fits.HDUList(hdus=[uvhdu,base_uvfits[1]])
			#hdulist.writeto(output_uvfits_name,clobber=True)
			#base_uvfits.close()
			#hdulist.close()
			write_data = write_uvfits[0].data
			write_header = write_uvfits[0].header
			antenna_table = write_uvfits[1].data
			antenna_header = write_uvfits[1].header
			
		if options.diffuse:
			
			##TODO - I think this may be half a time step off, need to give the central time
			image, l_reso = generate_gsm_2016(freq=freq_cent,this_date=this_date,observer=MRO)
			uv_data_array, u_sim, v_sim, u_reso = convert_image_lm2uv(image=image,l_reso=l_reso)
			
			##For each baseline
			skipped_gsm = 0
			outside_uv = 0
			
			if not options.srclist:
				write_uvfits = base_uvfits
				write_data = base_uvfits[0].data
				write_header = base_uvfits[0].header
				antenna_table = base_uvfits[1].data
				antenna_header = base_uvfits[1].header
			
			for baseline in xrange(len(base_data)):
				#print 'Adding GSM 2016'
				##Due to the resolution of the GSM not all baselines will fall on the u,v
				##plane (u_extent = 1 / l_reso), so skip those that fail
				try:
					###u,v,w, stored in seconds in the uvfits
					##make sure we use the same u,v,w aleady stored in the uvfits
					u = write_data[baseline][0] * freq_cent
					v = write_data[baseline][1] * freq_cent
					w = write_data[baseline][2] * freq_cent
					
					outside = False
					if u < u_sim.min() or u > u_sim.max(): outside = True
					if v < v_sim.min() or v > v_sim.max(): outside = True
					
					#if outside: outside_uv += 1
					
					#l,m,n = get_lm(ra_point*D2R,ra_phase*D2R, MWA_LAT*D2R, MWA_LAT*D2R)
					uv_complex_XX,uv_complex_YY = reverse_grid(uv_data_array=uv_data_array, l_reso=l_reso, u=u, v=v, kernel=options.telescope,freq_cent=freq_cent,u_reso=u_reso,u_sim=u_sim,v_sim=v_sim)
					PhaseConst = 1j * 2 * pi
					##Insert a w-term as the FFT doesn't include them??
					##Inserting the w for zenith pointing where n=1
					
					##TODO - is this phase tracking case? Or add in an option for phase tracking??
					n = 1
					uv_complex_XX *= exp(PhaseConst * w*n)
					
					write_data[baseline][5][0,0,0,0,0,:] += array([real(uv_complex_XX),imag(uv_complex_XX),0.0000])
					write_data[baseline][5][0,0,0,0,1,:] += array([real(uv_complex_YY),imag(uv_complex_YY),0.0000])
				except:
					skipped_gsm += 1
				
			print '%04d out of %04d baselines skipped in gsm, u,v point outside gsm uv data plane' %(skipped_gsm,len(base_data))
				
		if time_res < 1:
			uvfits_name = "%s_%.3f_%05.2f.uvfits" %(tag_name,freq,time)
		else:
			uvfits_name = "%s_%.3f_%02d.uvfits" %(tag_name,freq,int(time))
			
		write_uvfits.writeto('%s/%s' %(data_loc,uvfits_name) ,clobber=True)