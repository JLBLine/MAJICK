from optparse import OptionParser
from sys import exit
from imager_lib import *
from numpy import exp as np_exp

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

parser.add_option('-f', '--freq_start',
	help='Enter lowest frequency (MHz) to simulate - this is lower band edge')
	
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

parser.add_option('-s', '--time_int', default=False,
	help='Enter name of srclist from which to add point sources')

parser.add_option('-g', '--freq_int', default=False,
	help='Add in the beam to all simulations')

#parser.add_option('-d', '--date', default='2000-01-01T00:00:00',
	#help="Enter date to start the observation on (YYYY-MM-DDThh:mm:ss), default='2000-01-01T00:00:00'")

parser.add_option('-c', '--tag_name', 
	help='Enter tag name for output uvfits files')

parser.add_option('-e', '--uvfits_tag', default=False, 
	help='Base fits file name and location (e.g. /location/file/uvfits_tag)')

#parser.add_option('-j', '--telescope', default='MWA_phase1',
	#help='Uses the array layout and primary beam model as stored in MAJICK_DIR/telescopes - defaults to MWA_phase1')

parser.add_option('-i', '--data_loc', default='./',
	help='Location to output the uvfits to')

parser.add_option('-b', '--band_num',
	help='RTS band number to name the output uvfits with')

parser.add_option('-p', '--rephase', default=False,action='store_true',
	help='Add to unwrap phase tracking of the current time cadence, and apply phase tracking for the new averaged time - i.e. if data is 2s cadence, undo any phase tracking average over new time cadence, say 8s, and apply phase trakcing using the w-term of the centre of the 8s time cadence')

options, args = parser.parse_args()

##Setup some options---------------------------
freq_int = float(options.freq_int)
time_int = float(options.time_int)
first_freq = float(options.freq_start)
first_time = float(options.time_start)
freq_res = float(options.freq_res)
time_res = float(options.time_res)
num_freqs = float(options.num_freqs)
num_times = float(options.num_times)
uv_tag = options.uvfits_tag
data_loc = options.data_loc
if data_loc[-1] == '/': data_loc = data_loc[:-1]
band_num = int(options.band_num)
tag_name = options.tag_name

##Load all the uvdata into a class container--------------------------
uv_container = UVContainer(uv_tag=uv_tag,freq_start=first_freq,num_freqs=num_freqs,freq_res=freq_res,time_start=first_time,num_times=num_times,time_res=time_res)

print('uvfits data finished loading......')

##Open the first time and freq steal uvfits file to get the antenna table----------
if time_res < 1:
	base_uvfits = fits.open("%s_%.3f_%05.2f.uvfits" %(uv_tag,first_freq,first_time))
else:
	base_uvfits = fits.open("%s_%.3f_%02d.uvfits" %(uv_tag,first_freq,int(first_time)))
	
base_data = base_uvfits[0].data
base_header = base_uvfits[0].header
antenna_table = base_uvfits[1].data
antenna_header = base_uvfits[1].header
	
##Get some parameters about the averaging we are doing---------------------------
num_freq_avg = int(freq_int/uv_container.freq_res)
num_time_avg = int(time_int/uv_container.time_res)

num_time_groups = int((num_times*time_res) / time_int)
num_freq_groups = int((num_freqs*freq_res) / freq_int)

num_baselines = len(uv_container.xyz_lengths_unscaled)
##The number of random groups(?) is set by number of baselines and averaged time steps
n_data = num_baselines * num_time_groups

ra_phase = base_header['CRVAL6']
dec_phase = base_header['CRVAL7']
template_baselines = base_data['BASELINE']

##Initial date is the time the first observation started
intial_date = antenna_header['RDATE']
print('intial_date',intial_date)

##This calculates half a time cadence in seconds
half_time_cadence = num_time_avg * (uv_container.time_res / 2.0)

print  'half_time_cadence', half_time_cadence

##This gives us the Julian Date for the first integrated time step
##Use this to set the PZERO5 (int_jd) - then we can add time on
##time increments to float_jd for each subsequent time integration
first_date = add_time(intial_date,half_time_cadence)
int_jd, float_jd = calc_jdcal(first_date)
print int_jd, float_jd

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

time_step = 0
for time_start in range(0,len(uv_container.times),num_time_avg):
	freq_step = 0
	for freq_start in range(0,len(uv_container.freqs),num_freq_avg):
		##Empty uvdata array for this time,freq integration
		sum_uvdata = zeros((num_baselines,4,3))
		##Actual averaging loop----------------------
		for time_avg in range(time_start,time_start+num_time_avg):
			
			for freq_avg in range(freq_start,freq_start+num_freq_avg):
				freq = uv_container.freqs[freq_avg]
				time = uv_container.times[time_avg]
				#print("time step %.2f, freq step %.3f"%(time,freq))
				uvdata = uv_container.uv_data['%.3f_%05.2f' %(freq,time)].data
				if options.rephase:
					##if we need to rephase, undo original phase trakcing.
					w = uv_container.uv_data['%.3f_%05.2f' %(freq,time)].ww
					PhaseConst = -1 * 1j * 2 * pi
					for i in xrange(len(w)):
						new_XX = complex(uvdata[i,0,0],uvdata[i,0,1]) * np_exp(PhaseConst * w[i])
						new_YY = complex(uvdata[i,1,0],uvdata[i,1,1]) * np_exp(PhaseConst * w[i])
						uvdata[i,0,0] = real(new_XX)
						uvdata[i,0,1] = imag(new_XX)
						uvdata[i,1,0] = real(new_YY)
						uvdata[i,1,1] = imag(new_YY)

				sum_uvdata += uvdata
		##Actual averaging loop----------------------
		array_time_loc = num_baselines*time_step

		##Add data in order of baselines, then time step in axes 0 of v_container
		##Each frequency average goes axes 4 of the v_container
		
		##First time and freq step of this cadence
		freq = uv_container.freqs[freq_start]
		time = uv_container.times[time_start]
		##Central frequency of the the first freq step of this cadence
		freq_cent = freq + (uv_container.freq_res / 2.0)
		
		##This is the initial LST of this group of uvfits, and the centre of the first time step
		intial_lst = uv_container.uv_data['%.3f_%05.2f' %(freq,time)].central_LST #- SOLAR2SIDEREAL*(15.0/3600.0)
		print('intial_lst',intial_lst)

		##In the following, find the LST and frequency at the centre of the set of
		##visis being averaged over
		##If averaging more than one time step together, need to find the offset of the
		##central LST of the averaged time from the start of the set of times
		if num_time_avg > 1:
			##The centre of the averaged time cadence
			half_time_cadence = (num_time_avg * uv_container.time_res) / 2.0
			##Initial LST is for the centre of the intital time step, so half a time resolution
			##after the beginning of the averaged time cadence
			half_time_cadence -= uv_container.time_res / 2.0
			half_time_cadence *= SOLAR2SIDEREAL*(15.0/3600.0)
			#half_time_cadence = num_time_avg * (uv_container.time_res / 2.0) * SOLAR2SIDEREAL*(15.0/3600.0)
		##the intial_lst is the central lst of the first time step, so if not averaging, don't
		##need to add anything
		else:
			half_time_cadence = 0
		print('half_time_cadence',half_time_cadence)
		
		central_lst = intial_lst + half_time_cadence 
		if central_lst > 360: central_lst -= 360.0
		print('central_lst',central_lst)

		##Get some relevant positions and data
		ra0,dec0 =  ra_phase*D2R,dec_phase*D2R
		#print('ra_phase,dec_phase',ra0,dec0)
		h0 = central_lst*D2R - ra0
		
		##If averaging over more than one frequeny, work out distance
		##of cadence centre to start of cadence
		if num_freq_avg > 1:
			half_freq_cadence = num_freq_avg * (uv_container.freq_res / 2.0) * 1e+6
		else:
			half_freq_cadence = 0
			
		central_frequency = freq_cent*1e+6 + half_freq_cadence
		
		##These are the non frequency scaled lengths in X,Y,Z
		xyzs = array(uv_container.xyz_lengths_unscaled)
		##Seperate out into x,y,z
		x_lengths = xyzs[:,0]
		y_lengths = xyzs[:,1]
		z_lengths = xyzs[:,2]
		
		print('dec0,h0',dec0,h0)
		
		##Calculate the u,v,w coords for all baselines at the centre of the integration
		avg_uu, avg_vv, avg_ww = get_uvw_freq(x_length=x_lengths,y_length=y_lengths,z_length=z_lengths,dec=dec0,ha=h0,freq=central_frequency)
		
		##Add the u,v,w coords from the central time and frequency step into the final uvfits
		uu[array_time_loc:array_time_loc+num_baselines] = avg_uu / central_frequency
		vv[array_time_loc:array_time_loc+num_baselines] = avg_vv / central_frequency
		ww[array_time_loc:array_time_loc+num_baselines] = avg_ww / central_frequency

		##If rephasing, phase track to the new phase centre
		if options.rephase:
			PhaseConst = 1j * 2 * pi
			for i in xrange(len(w)):
		                new_XX = complex(sum_uvdata[i,0,0],uvdata[i,0,1]) * np_exp(PhaseConst * avg_ww[i])
                                new_YY = complex(sum_uvdata[i,1,0],uvdata[i,1,1]) * np_exp(PhaseConst * avg_ww[i])
                                sum_uvdata[i,0,0] = real(new_XX)
                                sum_uvdata[i,0,1] = imag(new_XX)
                                sum_uvdata[i,1,0] = real(new_YY)
                                sum_uvdata[i,1,1] = imag(new_YY)

		
		##Add data in order of baselines, then time step in axes 0 of v_container
                ##Each frequency average goes axes 4 of the v_container
		v_container[array_time_loc:array_time_loc+num_baselines,0,0,freq_step,:,:] = sum_uvdata
		
		##Fill in the baselines using the first time and freq uvfits
		baselines_array[array_time_loc:array_time_loc+num_baselines] = template_baselines
		
		##Fill in the fractional julian date, after adding on the appropriate amount of
		##time - /(24*60*60) because julian number is a fraction of a whole day
		
		adjust_float_jd_array = float_jd_array + (time_step * (time_int / (24.0*60.0*60.0)))
		date_array[array_time_loc:array_time_loc+num_baselines] = adjust_float_jd_array

		freq_step += 1
		
	time_step += 1
	
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

##TODO get the central frequency of all averages, and number of freqs
##Frequency information

middle_pixel_number = ceil(num_freq_groups / 2.0)

middle_pixel_value = first_freq + (freq_res * num_freq_avg * (middle_pixel_number - 1))
middle_pixel_value *= 1e+6

uvhdu.header['CTYPE4'] = 'FREQ'
uvhdu.header['CRVAL4'] = middle_pixel_value  ##Middle pixel value in Hz
uvhdu.header['CRPIX4'] = middle_pixel_number ##Middle pixel number
uvhdu.header['CDELT4'] = freq_int * 1e+6


##Don't inlcude the extra axes that helps no one
#uvhdu.header['CTYPE5'] = base_uvfits[0].header['CTYPE5']
#uvhdu.header['CRVAL5'] = base_uvfits[0].header['CRVAL5']
#uvhdu.header['CRPIX5'] = base_uvfits[0].header['CRPIX5']
#uvhdu.header['CDELT5'] = base_uvfits[0].header['CDELT5']

##RA phase information
uvhdu.header['CTYPE5'] = base_uvfits[0].header['CTYPE6']
uvhdu.header['CRVAL5'] = base_uvfits[0].header['CRVAL6']
uvhdu.header['CRPIX5'] = base_uvfits[0].header['CRPIX6']
uvhdu.header['CDELT5'] = base_uvfits[0].header['CDELT6']

##DEC phase information
uvhdu.header['CTYPE6'] = base_uvfits[0].header['CTYPE7']
uvhdu.header['CRVAL6'] = base_uvfits[0].header['CRVAL7']
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
uvhdu.header['OBSRA']   = base_uvfits[0].header['OBSRA']
uvhdu.header['OBSDEC']  = base_uvfits[0].header['OBSDEC']

##Number of visibilities * number of time steps
uvhdu.header['GCOUNT'] = n_data

antenna_header['FREQ'] = middle_pixel_value

## Create hdulist and write out file
write_uvfits = fits.HDUList(hdus=[uvhdu,base_uvfits[1]])

if options.rephase:
	uvfits_name = "%s_rephase_t%02d_f%.3f_%02d.uvfits" %(tag_name,time_int,freq_int,band_num)
else:
	uvfits_name = "%s_t%02d_f%.3f_%02d.uvfits" %(tag_name,time_int,freq_int,band_num)

write_uvfits.writeto('%s/%s' %(data_loc,uvfits_name) ,clobber=True)
