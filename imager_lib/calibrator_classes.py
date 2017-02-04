from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from mwapy import ephem_utils
from mwapy.pb import primary_beam
from mwapy.pb import beam_full_EE
#from mwapy.pb import mwa_tile
from numpy import pi,cos,sin,array,repeat,reshape,sqrt,dot,ones,real,imag,arctan2,zeros,conjugate,linalg,transpose,matrix,identity,arccos,log,argmin,swapaxes
from cmath import exp
from uvdata_classes import *
from gridding_functions import *
from scipy import interpolate
from os import environ
try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits

D2R = pi/180.0
R2D = 180.0/pi
VELC = 299792458.0
MWA_LAT = -26.7033194444
#MWA_LAT = 0
beam_freqs = arange(49920000,327680000+1.28e6,1.28e+6)

MAJICK_DIR = environ['MAJICK_DIR']
MWAPY_H5PATH = MAJICK_DIR + "/telescopes/MWA_phase1/mwa_full_embedded_element_pattern.h5" 

##Class to store calibrator source information with - set to lists
##to store component info in the same place
class Cali_source():
	def __init__(self):
		self.name = ''
		self.ras = []
		self.has = []
		self.decs = []
		self.freqs = []
		self.fluxs = []
		self.extrap_fluxs = []
		self.weighted_fluxs = -1
		self.offset = -1
		self.shapelets = []
		self.shapelet_indexes = []
		self.shapelet_coeffs = []
		self.gaussians = []
		self.gaussian_indexes = []
		self.ranking_flux = None
		self.XX_beam = []
		self.YY_beam = []
		self.skip = False
		
def extrap_flux(freqs,fluxs,extrap_freq):
	'''f1/f2 = (nu1/n2)**alpha
	   alpha = ln(f1/f2) / ln(nu1/nu2)
	   f1 = f2*(nu1/nu2)**alpha'''
	alpha = log(fluxs[0]/fluxs[1]) / log(freqs[0]/freqs[1])
	extrap_flux = fluxs[0]*(extrap_freq/freqs[0])**alpha
	return extrap_flux

def create_calibrator(cali_info=None):
	##TODO - maybe fit all information with
	##a 2nd order polynomial?? Then we can just
	##call the fits whenever we want to extrap
	
	source = Cali_source()
	
	primary_info = cali_info.split('COMPONENT')[0].split('\n')
	primary_info = [info for info in primary_info if info!='']
	meh,prim_name,prim_ra,prim_dec = primary_info[0].split()
	
	##Put in to the source class
	source.name = prim_name
	source.ras.append(float(prim_ra)*15.0)
	source.decs.append(float(prim_dec))
	#source.offset = offset
	##Find the fluxes and append to the source class
	prim_freqs = []
	prim_fluxs = []
	for line in primary_info:
		if 'FREQ' in line:
			prim_freqs.append(float(line.split()[1]))
			prim_fluxs.append(float(line.split()[2]))
	source.freqs.append(prim_freqs)
	source.fluxs.append(prim_fluxs)
	
	##Split all info into lines and get rid of blank entries
	lines = cali_info.split('\n')
	lines = [line for line in lines if line!='']
	##If there are components to the source, see where the components start and end
	comp_starts = [lines.index(line) for line in lines if 'COMPONENT' in line and 'END' not in line]
	comp_ends = [i for i in xrange(len(lines)) if lines[i]=='ENDCOMPONENT']
	
	##Check to see if the primary source is a gaussian or shapelet
	for line in primary_info:
		###Check here to see if primary source is a gaussian:
		if 'GAUSSIAN' in line:
			source.gaussians.append(line)
			source.gaussian_indexes.append(0)
		##As shapelet line comes after the 
		elif 'SHAPELET' in line:
			coeffs = []
			source.shapelets.append(line)
			source.shapelet_indexes.append(0)
			##If the primary source is a shapelet, search for shapelet coeffs in primary data,
			##gather in a list and append to the source class
			for line in primary_info:
				if 'COEFF' in line: coeffs.append(line)
			source.shapelet_coeffs.append(coeffs)
		
	##For each component, go through and find ra,dec,freqs and fluxs
	##Also check here if the component is a gaussian or shapelet
	for start,end in zip(comp_starts,comp_ends):
		freqs = []
		fluxs = []
		coeffs = []
		for line in lines[start:end]:
			if 'COMPONENT' in line:
				source.ras.append(float(line.split()[1])*15.0)
				source.decs.append(float(line.split()[2]))
			elif 'FREQ' in line:
				freqs.append(float(line.split()[1]))
				fluxs.append(float(line.split()[2]))
			elif 'GAUSSIAN' in line:
				source.gaussians.append(line)
				gaus_ind = comp_starts.index(start) + 1
				source.gaussian_indexes.append(gaus_ind)
			elif 'SHAPELET' in line:
				source.shapelets.append(line)
				shap_ind = comp_starts.index(start) + 1
				source.shapelet_indexes.append(shap_ind)
			elif 'COEFF' in line:
				coeffs.append(line)
		source.fluxs.append(fluxs)
		source.freqs.append(freqs)
		if len(coeffs) > 0:
			source.shapelet_coeffs.append(coeffs)
	return source

def local_beam(za, az, freq, delays=None, zenithnorm=True, power=True, jones=False, interp=True, pixels_per_deg=5):
	'''Code pulled my mwapy that generates the MWA beam response - removes unecessary extra code from mwapy/pb'''
	tile=beam_full_EE.ApertureArray(MWAPY_H5PATH,freq)
	mybeam=beam_full_EE.Beam(tile, delays)
	if interp:
		j=mybeam.get_interp_response(az, za, pixels_per_deg)
	else:
		j=mybeam.get_response(az, za)        
	if zenithnorm==True:      
		j=tile.apply_zenith_norm_Jones(j) #Normalise
		
	#Use swapaxis to place jones matrices in last 2 dimensions
	#insead of first 2 dims.
	if len(j.shape)==4:
		j=swapaxes(swapaxes(j,0,2),1,3)
	elif len(j.shape)==3: #1-D
		j=swapaxes(swapaxes(j,1,2),0,1)
	else: #single value
		pass

	if jones:        
		return j
		
	#Use mwa_tile makeUnpolInstrumentalResponse because we have swapped axes
	vis = mwa_tile.makeUnpolInstrumentalResponse(j,j)
	if not power:
		return (sqrt(vis[:,:,0,0].real),sqrt(vis[:,:,1,1].real))
	else:
		return (vis[:,:,0,0].real,vis[:,:,1,1].real)

def weight_by_beam(source=None,freqcent=None,LST=None,delays=None,beam=False,fix_beam=False):
	'''Takes a Cali_source() class and extrapolates to the given
	frequency, then weights by the beam'''
	
	beam_weights = []
	
	##Check if the primary RA,Dec is below the horizon - don't calibrate on
	##these sources
	ha_prim = LST - source.ras[0]
	Az_prim,Alt_prim = ephem_utils.eq2horz(ha_prim,source.decs[0],MWA_LAT)
	if Alt_prim < 0.0:
		source.skip = True
	else:
		source.skip = False
		extrap_fluxs = []
		##For each set of source infomation, calculate and extrapolated flux at the central flux value
		for freqs,fluxs in zip(source.freqs,source.fluxs):
			
			##If only one freq, extrap with an SI of -0.8:
			if len(freqs)==1:
				##f1 = c*v1**-0.8
				##c = f1 / (v1**-0.8)
				c = fluxs[0] / (freqs[0]**-0.8)
				ext_flux = c*freqcent**-0.8
			##If extrapolating below known freqs, choose two lowest frequencies
			elif min(freqs)>freqcent:
				ext_flux = extrap_flux([freqs[0],freqs[1]],[fluxs[0],fluxs[1]],freqcent)
			##If extrapolating above known freqs, choose two highest frequencies
			elif max(freqs)<freqcent:
				ext_flux = extrap_flux([freqs[-2],freqs[-1]],[fluxs[-2],fluxs[-1]],freqcent)
			##Otherwise, choose the two frequencies above and below, and extrap between them
			else:
				for i in xrange(len(freqs)-1):
					if freqs[i]<freqcent and freqs[i+1]>freqcent:
						ext_flux = extrap_flux([freqs[i],freqs[i+1]],[fluxs[i],fluxs[i+1]],freqcent)
			extrap_fluxs.append(ext_flux)
			
		source.extrap_fluxs = extrap_fluxs
		
		if beam:
			##For each component, work out it's position, convolve with the beam and sum for the source
			for ra,dec in zip(source.ras,source.decs):
				##HA=LST-RA in def of ephem_utils.py
				ha = LST - ra  ##RTS stores things in hours
				##Convert to zenith angle, azmuth in rad
				
				Az,Alt=ephem_utils.eq2horz(ha,dec,MWA_LAT)
				za=(90-Alt)*pi/180
				az=Az*pi/180
				
				if fix_beam:
					##If using CHIPS in fix_beam mode, force beam to 186.235MHz
					#XX,YY = primary_beam.MWA_Tile_full_EE([[za]], [[az]], freq=186.235e+6, delays=delays, zenithnorm=True, power=True, interp=False)
					XX,YY = local_beam([[za]], [[az]], freq=186.235e+6, delays=delays, zenithnorm=True, power=True, interp=False)
					
					final_XX,final_YY = XX[0][0],YY[0][0]
				else:
					##The beam only has a spectral resolution of 1.28MHz. The known beam
					##model frequencies are defined as beam_freqs at top of file. Find the
					##beam points within 2 course bands above and below, interpolate over 
					##them, and then find out the beam value at the desired frequency
					lower_freq = freqcent - 3*1.28e+6
					upper_freq = freqcent + 3*1.28e+6
					
					pos_lowest = argmin(np_abs(beam_freqs - lower_freq))
					pos_highest = argmin(np_abs(beam_freqs - upper_freq))
					
					freqs = beam_freqs[pos_lowest:pos_highest+1]
					
					this_XX = []
					this_YY = []
					
					for freq in freqs:
						#XX,YY = primary_beam.MWA_Tile_full_EE([[za]], [[az]], freq=freq, delays=delays, zenithnorm=True, power=True, interp=False)
						XX,YY = local_beam([[za]], [[az]], freq=186.235e+6, delays=delays, zenithnorm=True, power=True, interp=False)
						this_XX.append(XX[0][0])
						this_YY.append(YY[0][0])
					
					##interpolate over all XX,YY
					f_XX = interpolate.interp1d(freqs,this_XX,kind='cubic')
					f_YY = interpolate.interp1d(freqs,this_YY,kind='cubic')
					final_XX = f_XX(freqcent)
					final_YY = f_YY(freqcent)
				
				source.XX_beam.append(final_XX)
				source.YY_beam.append(final_YY)
		
#def model_vis(u=None,v=None,w=None,source=None,phase_ra=None,phase_dec=None,LST=None,x_length=None,y_length=None,z_length=None,time_decor=False,freq_decor=False,beam=False,freq=None):   ##,sources=None
	## V(u,v) = integral(I(l,m)*exp(i*2*pi*(ul+vm)) dl dm)
	#vis_XX = complex(0,0)
	#vis_YY = complex(0,0)
	#sign = +1
	#PhaseConst = 1j * 2 * pi * sign
	
	#tot_decor = 0

	#phase_ra *= D2R
	#phase_dec *= D2R

	###For each component in the source
	#for i in xrange(len(source.ras)):
		
		#ra,dec,flux = source.ras[i],source.decs[i],source.extrap_fluxs[i]
		
		#phase_ha = LST*D2R - phase_ra
		#ha = (LST - ra)*D2R
		###Here phase_ra, phase_dec should be zenith
		#l,m,n = get_lm(ra*D2R, phase_ra, dec*D2R, phase_dec)
		#this_vis = (flux * exp(PhaseConst*(u*l + v*m + w*n)))

		###Turn this on to add in phase tracking
		##this_vis = (flux * exp(PhaseConst*(u*l + v*m + w*(n-1))))
		
		###Add in decor if asked for
		#if time_decor:
			##l,m,n = get_lm(ra*D2R, phase_ra, dec*D2R, phase_dec)
			##phase_ha = LST*D2R - phase_ra
			###For MWA obs, need to input phase_ra,phase_dec as zenith
			##print(time_decor)
			##print(phase_dec,phase_ha)
			#tdecor = tdecorr_phasetrack(X=x_length,Y=y_length,Z=z_length,d0=phase_dec,h0=phase_ha,l=l,m=m,n=n,time_int=time_decor)
			##u, v, w = get_uvw(x_length,y_length,z_length,phase_dec,phase_ha)
			##tdecor = tdecorr_nophasetrack(u=u,d=dec*D2R,H=ha,t=time_decor)
			
			##print(tdecor) 
			#this_vis *= tdecor
			#tot_decor += tdecor
			
		###Add in decor if asked for
		#if freq_decor:
			##l,m,n = get_lm(ra*D2R, phase_ra, dec*D2R, phase_dec)
			##phase_ha = LST*D2R - phase_ra
			#u,v,w = get_uvw(x_length,y_length,z_length,phase_dec,phase_ha)
			#fdecor = fdecorr(u=u,v=v,w=w,l=l,m=m,n=n,chan_width=freq_decor,freq=freq,phasetrack=True)
			#this_vis *= fdecor
	
		#if beam:
			#vis_XX += (this_vis * source.XX_beam[i])
			#vis_YY += (this_vis * source.YY_beam[i])
		#else:
			#vis_XX += this_vis
			#vis_YY += this_vis
			
	#return vis_XX,vis_YY

def model_vis(u=None,v=None,w=None,source=None,phase_ra=None,
		phase_dec=None,LST=None,x_length=None,y_length=None,z_length=None,
		time_decor=False,freq_decor=False,beam=False,freq=None,time_int=None,
		chan_width=None,fix_beam=False,phasetrack=True):   ##,sources=None
	'''Generates model visibilities for a phase tracking correlator'''
	# V(u,v) = integral(I(l,m)*exp(i*2*pi*(ul+vm)) dl dm)
	vis_XX = complex(0,0)
	vis_YY = complex(0,0)
	sign = -1
	PhaseConst = 1j * 2 * pi * sign
	
	phase_ra *= D2R
	phase_dec *= D2R
	##For each component in the source
	for i in xrange(len(source.ras)):
		
		ra,dec,flux = source.ras[i],source.decs[i],source.extrap_fluxs[i]
		phase_ha = LST*D2R - phase_ra
		ha = (LST - ra)*D2R
		
		##TODO - l,m,n should be constant if phasetracking - pull out of loop somehow?
		l,m,n = get_lm(ra*D2R, phase_ra, dec*D2R, phase_dec)
		
		if phasetrack:
			this_vis = flux * exp(PhaseConst*(u*l + v*m + w*(n-1)))
		else:
			this_vis = flux * exp(PhaseConst*(u*l + v*m + w*n))
		##Add in decor if asked for
		if time_decor:
			tdecor = tdecorr_phasetrack(X=x_length,Y=y_length,Z=z_length,d0=phase_dec,h0=phase_ha,l=l,m=m,n=n,time_int=time_int)
			this_vis *= tdecor
			
		##Add in decor if asked for
		if freq_decor:
			fdecor = fdecorr(u=u,v=v,w=w,l=l,m=m,n=n,chan_width=chan_width,freq=freq,phasetrack=phasetrack)
			this_vis *= fdecor
	
		if beam:
			vis_XX += (this_vis * source.XX_beam[i])
			vis_YY += (this_vis * source.YY_beam[i])

		else:
			vis_XX += this_vis
			vis_YY += this_vis
			
	return vis_XX,vis_YY

def apply_gains(model,gains):
	'''Takes model visibilities and gains and applies the gains (multiples by) to the model'''
	updated_model = []
	model_ind = 0
	for i in xrange(len(gains)-1):
		for j in range(i+1,len(gains)):
			updated_model.append(gains[i]*model[model_ind]*conjugate(gains[j]))
			model_ind += 1
	return array(updated_model)

def remove_gains(visibilities,gains):
	'''Takes the real visibilities and estimated gains and removes the gains (divides by) from the visibilities'''
	updated_visibilities = []
	visibilities_ind = 0
	for i in xrange(len(gains)-1):
		for j in range(i+1,len(gains)):
			updated_visibilities.append(visibilities[visibilities_ind] / (gains[i]*conjugate(gains[j])))
			visibilities_ind += 1
	return array(updated_visibilities)
	
def minimise_using_a(visi_data,gains,model):
	'''Takes in visibility data, current gain estimates and a sky model.
    Using this it creates an 'A' array to estimate updates to the current gains.
    Returns updated gains'''
    
    ##Generate and populate the 'A' array
	visi_ind = 0
	##Set up empty array
	a_array = zeros((2*len(visi_data),2*len(gains)))
	for i in xrange(len(gains)-1):
		for j in range(i+1,len(gains)):
			a_array[2*visi_ind,2*i] = real(visi_data[visi_ind]*conjugate(gains[j]))
			a_array[2*visi_ind,2*i+1] = -imag(visi_data[visi_ind]*conjugate(gains[j]))
			a_array[2*visi_ind,2*j] = real(visi_data[visi_ind]*gains[i])
			a_array[2*visi_ind,2*j+1] = imag(visi_data[visi_ind]*gains[i])
			
			##Imag part goes in first row
			a_array[2*visi_ind+1,2*i] = imag(visi_data[visi_ind]*conjugate(gains[j]))
			a_array[2*visi_ind+1,2*i+1] = real(visi_data[visi_ind]*conjugate(gains[j]))
			a_array[2*visi_ind+1,2*j] = imag(visi_data[visi_ind]*gains[i])
			a_array[2*visi_ind+1,2*j+1] = -real(visi_data[visi_ind]*gains[i])
			visi_ind += 1
	a_array = matrix(a_array)
    
	##Start by giving everything a weight of 1.0
	##via an identity matrix
	weights = identity(a_array.shape[0])
	
	##Find the difference between the real visis and the model
	##with the gains applied
	diffs = visi_data - apply_gains(model,gains)
	
	##Populate a difference array with real and imag parts of the
	##the difference between
	diff_array = zeros((2*len(visi_data),1))
	for i in xrange(len(diffs)):
		diff_array[2*i] = real(diffs[i])
		diff_array[2*i+1] = imag(diffs[i])
		
	##The equation used to calucalte the update to the gains is
	##x_tilde = inverse(transpose(a_array)*weights*a_array) * transpose(a_array) * weights * diff_array
	x_tilde = linalg.inv((transpose(a_array)*weights)*a_array) * transpose(a_array) * weights * diff_array
	
	##Put the updates that we found back into complex form so that we can easily apply to the gains
	delta_gains = zeros(len(gains),dtype=complex)
	for i in xrange(len(delta_gains)):
		delta_gains[i] = complex(x_tilde[2*i],x_tilde[2*i+1])

	##TODO Add in a safe-guard against massive updates to avoid death of matrix?
	##Choose how much of the updates to apply to the gains
	update_step_size = 0.5

	return gains + update_step_size*delta_gains
		
class Calibrator(object):
	def __init__(self,uv_container=None,freq_int=None,time_int=None,srclist=None,num_cals=None,metafits=None):
		self.uv_container = uv_container
		self.freq_int = freq_int
		self.time_int = time_int
		self.ra = uv_container.ra
		self.dec = uv_container.dec
		self.gridded_uv = None
		
		##Try opening the metafits file and getting information
		try:
			f=fits.open(metafits)
		except Exception,e:
			print('Unable to open metafits file %s: %s' % (metafits,e))
			sys.exit(1)
		if not 'DELAYS' in f[0].header.keys():
			print('Cannot find DELAYS in %s' %metafits)
			sys.exit(1)
		#if not 'LST' in f[0].header.keys():
			#print('Cannot find LST in %s' %metafits)
			#sys.exit(1)
		
		#LST = float(f[0].header['LST'])	
		delays=array(map(int,f[0].header['DELAYS'].split(',')))
		
		##Try opening the source list
		try:
			cali_src_info = open(srclist,'r').read().split('ENDSOURCE')
			del cali_src_info[-1]
		except:
			print("Cannot open %s, cannot calibrate without source list")
			exit(1)
			
		calibrator_sources = {}
			
		for cali_info in cali_src_info:
			Cali_source = create_calibrator(cali_info)
			calibrator_sources[Cali_source.name] = Cali_source
			
		##Lookup an mwa_title (I don't really know precisely what it's doing)
		d = mwa_tile.Dipole(type='lookup')
		tile = mwa_tile.ApertureArray(dipoles=[d]*16)
		delays=repeat(reshape(delays,(1,16)),2,axis=0)
			
		##Set up some empty gains
		xx_gains = ones(128,dtype=complex)
		##Do some kind of averaging here
		for freq in self.uv_container.freqs:
			for time in self.uv_container.times:
				
				uvdata = self.uv_container.uv_data['%.3f_%05.2f' %(freq,time)]
				xxpol_real, xxpol_imag, xxpol_weight = uvdata.data[:,0,0],uvdata.data[:,0,1],uvdata.data[:,0,2]
				xxpol_data = array([complex(xxpol_re,xxpol_im) for xxpol_re,xxpol_im in zip(xxpol_real, xxpol_imag)],dtype=complex)
				
				##Should I make the srclist before time/freq averaging?
				cal_names = []
				cal_wfluxes = []
				
				for cal_name,cal_source in calibrator_sources.iteritems():
					weight_by_beam(source=cal_source,freqcent=freq*1e+6,LST=LST,tile=tile,delays=delays)
					cal_names.append(cal_name)
					cal_wfluxes.append(cal_source.ranking_flux)
					
				###Order sources by strength of flux when convolved with beam, and only retain desired number of calibrators
				#calibrators = [flux for (freq,flux) in sorted(zip(freqs,fluxs),key=lambda pair: pair[0])][:num_cals]
				calibrators = [calibrator_sources[name] for (name,flux) in sorted(zip(cal_names,cal_wfluxes),key=lambda pair: pair[1])][:num_cals]
				
				for calibrator in calibrators:
					model_xxpols = []
					for i in xrange(len(uvdata.uu)):
						u,v = uvdata.uu[i],uvdata.vv[i]
					
						##TODO always phase to LST?? Make it an option?
						model_xxpol,model_yypol = model_vis(u=u,v=v,source=calibrator,phase_ra=uvdata.ra_point,phase_dec=uvdata.dec_point,LST=uvdata.LST)
						#print(model_xxpol,model_yypol)
						model_xxpols.append(model_xxpol)
						
					print("Made model visi for %.3f_%05.2f" %(freq,time))
					#print(xxpol_data[0],model_xxpols[0])
					
					##How many times we loop and update the gains
					loop_nums = 10
					loop = 0
					
					while loop < loop_nums:
						print("On cal loop %03d for calibrator %04d" %(loop,calibrators.index(calibrator)))
						xx_gains = minimise_using_a(xxpol_data,xx_gains,model_xxpols)
						
						loop += 1
					
				out_file = open("Jones_%.3f_%05.2f.txt" %(freq,time),'w+')
				
				for re,img in zip(real(xx_gains),imag(xx_gains)):
					out_file.write("%.5f %.5f\n" %(re,img))
					
					
				#import matplotlib.pyplot as plt
				##diffs = minimize_visi_real(jacobs=xx_jacobians_real,visi_data=xxpol_data,visi_model=model_xxpols)
				##plt.plot(arctan2(imag(xxpol_data) , real(xxpol_data)),'bo')
				#plt.plot(real(xxpol_data),'bo')
				##diffs = minimize_visi_imag(jacobs=xx_jacobians_real,visi_data=xxpol_data,visi_model=model_xxpols)
				##plt.plot(arctan2(imag(model_xxpols) , real(model_xxpols)),'r^')
				#plt.plot(real(model_xxpols),'r^')
				#plt.show()
				
				#plt.plot(imag(xxpol_data),'bo')
				##diffs = minimize_visi_imag(jacobs=xx_jacobians_real,visi_data=xxpol_data,visi_model=model_xxpols)
				##plt.plot(arctan2(imag(model_xxpols) , real(model_xxpols)),'r^')
				#plt.plot(imag(model_xxpols),'r^')
				#plt.show()
					
					
		self.xx_gains = xx_gains
		#self.xx_jacobians_imag = xx_jacobians_imag