import optparse
import matplotlib.pyplot as plt
#from sys import path
#path.append('/home/jline/Documents/time_decorrelation/dummy_imager/imager_lib')
#path.append('/home/jline/Documents/time_decorrelation/dummy_imager/')
from imager_lib import *
#import "/home/jline/Documents/time_decorrelation/dummy_imager/imager_lib"
from time import time
#from numpy import *
from generate_gsm_2016 import generate_gsm2016
try:
	import pyfits as fits
except ImportError:
	from astropy.io import fits
from ephem import Observer,degrees
	
D2R = pi/180.0
R2D = 180.0/pi
VELC = 299792458.0
MWA_LAT = -26.7033194444

#MWA_LAT = -26.78347242983789

def enh2xyz(east,north,height,latitiude):
	sl = sin(latitiude)
	cl = cos(latitiude)
	X = -north*sl + height*cl
	Y = east
	Z = north*cl + height*sl
	return X,Y,Z

def add_time(date_time,time_step):
	'''Take the time string format that uvfits uses ('23-08-2013 17:54:32.0'), and add a time time_step (seconds).
	Return in the same format - NO SUPPORT FOR CHANGES MONTHS CURRENTLY!!'''
	date,time = date_time.split('T')
	day,month,year = map(int,date.split('-'))
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
	return '%02d-%02d-%dT%d:%02d:%05.2f' %(day,month,year,int(hours),int(mins),secs)

##---------------------------------------------------##
##----------OBSERVATION SETTINGS----------------------##

parser = optparse.OptionParser()

parser.add_option('-f', '--freq_start',
	help='Enter lowest frequency to simulate')
	
parser.add_option('-n', '--num_freqs',
	help='Enter number of frequency channels to simulate')

parser.add_option('-y', '--freq_res', default=0.04,
	help='Enter frequency resolution (MHz) of observations, default=0.04')
	
parser.add_option('-t', '--time_start', 
	help='Enter lowest time offset from start date to simulate')

parser.add_option('-x', '--time_res', default=2.0,
	help='Enter time resolution (s) of observations, default=2.0')
	
parser.add_option('-m', '--num_times', 
	help='Enter number of times steps to simulate')

parser.add_option('-s', '--srclist', default=False,
	help='Enter name of srclist from which to add point sources')

parser.add_option('-g', '--beam', default=False,
	help='Add in the beam to all simulations')

parser.add_option('-d', '--date', default='2000-01-01T00:00:00',
	help="Enter date to start the observation on (YYYY-MM-DDThh:mm:ss), default='2000-01-01T00:00:00'")

parser.add_option('-c', '--tag_name', 
	help='Enter tag name for output uvfits files')

parser.add_option('-a', '--time_decor', default=False, action='store_true',
	help='Add to include time_decorrelation in the simulation')

parser.add_option('-b', '--diffuse', default=False, action='store_true',
	help='Add to include the 2016 gsm sky model')

parser.add_option('-e', '--base_uvfits', default=False, 
	help='Base fits file name tag to add diffuse model to')

parser.add_option('-l', '--layout', default='antenna_locations_mwa-phase1.txt',
	help='The array layout in local topocentric e,n,h coords - defaults to MWA phase 1')

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

#freq_start = 167.035
#freq_res = 0.04
#num_freqs = 1

#time_start = 0.0
#time_res = 2.0
#num_times = 1

#intial_date = '2013-08-23T17:50:32'
#srclist = '/home/jline/Documents/time_decorrelation/dummy_imager/oskar_singlesource/srclist_grid.txt'
##srclist = '/home/jline/Documents/srclists/srclist_puma-v2_complete_1061315448_patch1000.txt'

#tag_name = 'mymodel_single_offzen_orig-u-v-w'

###---------------------------------------------------##
###---------------------------------------------------##

##ephem Observer class, use this to compute LST from the date of the obs 
MRO = Observer()
##Set the observer at Boolardy
MRO.lat, MRO.long, MRO.elevation = '-26:42:11.95', '116:40:14.93', 0
date,time = intial_date.split('T')
MRO.date = '/'.join(date.split('-'))+' '+time
intial_lst = float(MRO.sidereal_time())*R2D
#intial_ra_point = 356.303842095
intial_ra_point = float(MRO.sidereal_time())*R2D
dec_point = MWA_LAT

##Get the local topocentric X,Y,Z values for the MWA using the local topocentric e,n,h
##values from the antenna locs in MWA_Tools
anntenna_locs = loadtxt(options.layout)
X,Y,Z = enh2xyz(anntenna_locs[:,0],anntenna_locs[:,1],anntenna_locs[:,2],MWA_LAT*D2R)

delays = zeros(32)
##Lookup an mwa_title (I don't really know precisely what it's doing)
d = mwa_tile.Dipole(type='lookup')
tile = mwa_tile.ApertureArray(dipoles=[d]*16)
#delays=repeat(reshape(delays,(1,16)),2,axis=0)

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
		#print source.name
		
	base_uvfits = fits.open("/home/jline/Documents/time_decorrelation/dummy_imager/oskar_singlesource/data/oskar_single_offzen_167.035_00.uvfits")
	base_data = base_uvfits[0].data
	base_header = base_uvfits[0].header
	antenna_table = base_uvfits[1].data
	antenna_header = base_uvfits[1].header

##Sidereal seconds per solar seconds - ie if 1s passes on
##the clock, sky has moved by 1.00274 secs of angle
SOLAR2SIDEREAL = 1.00274
for freq in arange(freq_start,freq_start+ freq_res*num_freqs,freq_res):
	for time in arange(time_start,time_start + time_res*num_times,time_res):
		print freq,time
		
		##Go through the uvfits file and replace the current X,Y,Z locations
		for i in xrange(len(X)):
			antenna_table['STABXYZ'][i] = [X[i],Y[i],Z[i]]

		##Scale the X,Y,Z by the wavelength of channel centre
		antennas = {}
		xyz = antenna_table['STABXYZ'] * (((freq + freq_res / 2.0)*1e+6) / VELC)

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

		ra_point = intial_ra_point + sky_offset
		if ra_point >=360.0: ra_point -= 360.0
		lst = intial_lst + sky_offset
		if lst >=360.0: lst -= 360.0
		
		ha_point = lst - ra_point
		#print(ha_point)
		
		
		if srclist:
			##TODO Weight each source by the beam pattern - do this to
			##calculate the beam at each point, as well as extrapolate the source
			##flux density to the current frequency
			for name,source in sources.iteritems():
				weight_by_beam(source=source,freqcent=(freq + freq_res / 2.0)*1e+6,LST=lst,tile=tile,delays=delays)
				
			##Set out instrument to always point at the same HA, DEC for now
			##TODO make an option to track an RA, DEC
			base_header['CRVAL6'] = ra_point
			base_header['CRVAL7'] = dec_point
			base_header['CRVAL4'] = (freq + freq_res / 2.0)*1e+6
			
			#print(ra_point,dec_point,add_time(intial_date,time + (time_res / 2.0)),lst)
			
			##Change the date by the time step - this is used
			##to get the LST
			
			antenna_header['RDATE'] = add_time(intial_date,time + (time_res / 2.0))
			
			
		if options.diffuse:
			if time_res < 1:
				fits.open("%s_%.3f_%05.2f.uvfits" %(options.base_uvfits,freq,time))
			else:
				fits.open("%s_%.3f_%02d.uvfits" %(options.base_uvfits,freq,int(time)))

			base_data = base_uvfits[0].data
			base_header = base_uvfits[0].header
			antenna_table = base_uvfits[1].data
			antenna_header = base_uvfits[1].header
			
		
		##For each baseline
		for baseline in range(len(base_data)):
		#for baseline in range(0,1):
			if srclist:
				x_length,y_length,z_length = xyz_lengths[baseline]
				u,v,w = get_uvw(x_length,y_length,z_length,dec_point*D2R,ha_point*D2R)
				
				#uv_data_XX = array([0.0,0.0,0.0])
				
				##For every source in the sky
				for name,source in sources.iteritems():
					##If source is below the horizon, forget about it
					if source.skip:
						pass
					##Otherwise, proceed
					else:
						if time_decor:
							model_xxpol,model_yypol = model_vis(u=u,v=v,w=w,source=source,phase_ra=ra_point,phase_dec=dec_point,LST=lst,x_length=x_length,y_length=y_length,z_length=z_length,time_decor=time_res)
						else:
							model_xxpol,model_yypol = model_vis(u=u,v=v,w=w,source=source,phase_ra=ra_point,phase_dec=dec_point,LST=lst)
							
						uv_data_XX += array([real(model_xxpol),imag(model_xxpol),1.0000])
						uv_data_YY += array([real(model_yypol),imag(model_yypol),1.0000])
				
				##Could add in crazy weightings here
				#uv_data_XX[2] = 1.0
				#uv_data_YY[2] = 1.0
				base_data[baseline][5][0,0,0,0,0,:] = uv_data_XX
				base_data[baseline][5][0,0,0,0,1,:] = uv_data_YY
				base_data[baseline][0] = u / ((freq + freq_res / 2.0)*1e+6)
				base_data[baseline][1] = v / ((freq + freq_res / 2.0)*1e+6)
				base_data[baseline][2] = w / ((freq + freq_res / 2.0)*1e+6)
				
			if options.diffuse:
				###u,v,w, stored in seconds in the uvfits
				u = base_data[baseline][0]*((freq + freq_res / 2.0)*1e+6)
				v = base_data[baseline][1]*((freq + freq_res / 2.0)*1e+6)
				w = base_data[baseline][2]*((freq + freq_res / 2.0)*1e+6)
				
				uv_data_XX = base_data[baseline][5][0,0,0,0,0,:]
				uv_data_YY = base_data[baseline][5][0,0,0,0,1,:]
				
				
			
		if time_res < 1:
			uvfits_name = "%s_%.3f_%05.2f.uvfits" %(tag_name,freq,time)
		else:
			uvfits_name = "%s_%.3f_%02d.uvfits" %(tag_name,freq,int(time))
			
		base_uvfits.writeto('./data/%s' %uvfits_name ,clobber=True)