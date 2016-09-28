'''Dummy imager to image interferometric visibilities'''
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from calibrator_classes import *
from uvdata_classes import *
from gridding_functions import *
from astropy.io import fits
from ephem import Observer,degrees
from numpy import sin,cos,pi,array,sqrt,arange,zeros,fft,meshgrid,where,arcsin,mod,real,ndarray,ceil
from numpy import abs as np_abs
from numpy import exp as np_exp
from cmath import phase,exp
import matplotlib
##Protects clusters where no $DISPLAY is set when running PBS/SLURM
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sys import exit
from astropy.wcs import WCS
#from time import time

D2R = pi/180.0
R2D = 180.0/pi
VELC = 299792458.0
MWA_LAT = -26.7033194444
SOLAR2SIDEREAL = 1.00274

class Sum_Pixel(object):
	def __init__(self):
		self.time_int = None
		self.freq_int = None
		self.cal_names = []
		self.ras = []
		self.lsts = []
		self.decs = []
		self.sum_pixels = []
		self.time_step = None
		self.freq_step = None
		self.tdecorr_factors = []
		self.tdecorr_fluxes = []
		self.tdecorr_adj = []

class Imager(object):
	def __init__(self, uv_container=None, over_sampling=1.0,freq_int=None,time_int=None,kernel=None,num_cals=None,xx_jones=None,srclist=None,metafits=None):
		self.uv_container = uv_container
		self.over_sampling = over_sampling
		#self.gridded_uv = uv_container.gridded_uv
		self.ra_phase = uv_container.ra_phase
		self.dec_phase = uv_container.dec_phase
		self.kernel = kernel
		self.xx_jones = xx_jones
		self.srclist = srclist
		self.num_cals = num_cals
		self.metafits = metafits
		self.time_int = time_int
		self.freq_int = freq_int
		self.kernel = kernel
		
	def sum_visi_at_source(self,predict_tdecor=False,verbose=False,apply_tdecor=False):
		try:
			cali_src_info = open(self.srclist,'r').read().split('ENDSOURCE')
			del cali_src_info[-1]
		except:
			print("Cannot open %s, not sure where to sum the visibilities for you without positions" %self.srclist)
			exit(1)
			
		calibrator_sources = {}
			
		for cali_info in cali_src_info:
			Cali_source = create_calibrator(cali_info)
			calibrator_sources[Cali_source.name] = Cali_source
			
		num_freq_avg = int(self.freq_int/self.uv_container.freq_res)
		num_time_avg = int(self.time_int/self.uv_container.time_res)
		
		print("Beginning sum_visi_at_source now----------------------")
		
		##A dictionary to contain all the Sum_Pixel containers we will create
		##Function returns this at the end
		sum_pixels = {}
		
		for time_start in range(0,len(self.uv_container.times),num_time_avg):
			for freq_start in range(0,len(self.uv_container.freqs),num_freq_avg):
				if verbose: print("At time_cad %02d %02d" %(time_start,freq_start))
				
				##Set up a container for all the summed pixel info
				sum_pixel = Sum_Pixel()
				sum_pixel.time_int = self.time_int
				sum_pixel.freq_int = self.freq_int
				sum_pixel.time_step = time_start
				sum_pixel.freq_step = freq_start
				
				avg_uu = None
				avg_vv = None
				avg_ww = None
				sum_xxpol_comps = None
				
				#print("Gridding time cadence: %02d, freq cadence %02d" %(time_start,freq_start))
				for time_int in range(time_start,time_start+num_time_avg):
					for freq_int in range(freq_start,freq_start+num_freq_avg):
						freq = self.uv_container.freqs[freq_int]
						time = self.uv_container.times[time_int]
						#print("time step %.2f, freq step %.3f"%(time,freq))
						
						uvdata = self.uv_container.uv_data['%.3f_%05.2f' %(freq,time)]
						xxpol_real, xxpol_imag, xxpol_weight = uvdata.data[:,0,0],uvdata.data[:,0,1],uvdata.data[:,0,2]
						xxpol_comps = array([complex(xxpol_re,xxpol_im) for xxpol_re,xxpol_im in zip(xxpol_real, xxpol_imag)],dtype=complex)
						if type(avg_uu) == ndarray:
							avg_uu += uvdata.uu
							avg_vv += uvdata.vv
							avg_ww += uvdata.ww
							sum_xxpol_comps += xxpol_comps
						else:
							avg_uu = uvdata.uu
							avg_vv = uvdata.vv
							avg_ww = uvdata.ww
							sum_xxpol_comps = xxpol_comps
				#print('------------')
				##TODO is this a legit average? Add some kind of weighting in here?
				avg_uu /= (num_freq_avg * num_time_avg)
				avg_vv /= (num_freq_avg * num_time_avg)
				avg_ww /= (num_freq_avg * num_time_avg)
				sum_xxpol_comps /= (num_freq_avg * num_time_avg)
				
				sum_weights = 2*float(len(avg_uu))
				##UNDO CURRENT PHASE TRACK THEN APPLY NEW ONE?
				
				##Want to phase everything up to the LST at the centre of the time cadence
				##Convert to degrees
				#half_time_cadence = ((self.time_int / 2.0)*SOLAR2SIDEREAL)*(15.0/3600.0)
				
				half_time_cadence = ((self.time_int / 2.0)*SOLAR2SIDEREAL)*(15.0/3600.0)
				
				##First time and freq step of this cadence
				freq = self.uv_container.freqs[freq_start]
				time = self.uv_container.times[time_start]
				##This is the initial LST of this group of uvfits
				##TODO CHECK THIS LST, IS IT CORRECT?
				intial_lst = self.uv_container.uv_data['%.3f_%05.2f' %(freq,time)].LST
				
				##TODO Need to offset by half of the time resolution of observation - WHY????
				central_lst = intial_lst + half_time_cadence  + (-(self.uv_container.time_res/2.0)*(15.0/3600.0)*SOLAR2SIDEREAL)
				if central_lst > 360: central_lst -= 360.0
				
				sum_pixel.lst = central_lst
				
				out_uv = open("time_%02d_%02d_decor.txt" %(time_start,freq_start),'w+')
				
				##For each calibrator
				for cal_name,cal_source in calibrator_sources.iteritems():
					
					##If actually get beam stuff, change tile and delays below
					##Currently this gives us the extrapolated fluxes at our current frequency
					weight_by_beam(source=cal_source,freqcent=uvdata.freq,LST=central_lst,tile='meh',delays='wah')

					##If verbose, print out what is being seen at the command line	
					if verbose: print("For SOURCE %s:" %cal_name)
					
					##Get some relevant positions and data
					ra0,dec0 =  self.ra_phase*D2R,self.dec_phase*D2R
					h0 = central_lst*D2R - ra0
					
					##For each component in the calibrator
					for pos_ind in xrange(len(cal_source.ras)):
						
						ra,dec = cal_source.ras[pos_ind]*D2R,cal_source.decs[pos_ind]*D2R
						ha_source = central_lst*D2R - ra
						out_uv.write('#For LST RA DEC HA RA0 DEC0 HA0 (rads) %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n' %(central_lst,ra,dec,ha_source,ra0,dec0,h0))
						sum_uvw_cal = complex(0,0)
						sum_tdecor = 0
						sum_tdecor_factor = 0
						sum_tdecor_adj = 0
						##For each visibility
						for visi_ind in xrange(len(avg_uu)):
							x_length,y_length,z_length = uvdata.xyz_lengths[visi_ind]
							##Calcualte u,v,w coords for phase centre and source position
							u_phase, v_phase, w_phase = get_uvw(x_length,y_length,z_length,dec0,h0)
							u_src, v_src, w_src = get_uvw(x_length,y_length,z_length,dec,ha_source)
							
							##Undo phase tracking by muliplying by the inverse of the original phase track,
							##then phase to src position by multiplying by phase of w_src
							PhaseConst = -1j * 2 * pi
							rotate_xx_complex = sum_xxpol_comps[visi_ind] * exp(PhaseConst*(w_src - w_phase))
							
							out_uv.write('%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n' %(u_phase,v_phase,w_phase,real(rotate_xx_complex),imag(rotate_xx_complex),x_length,y_length,z_length))
							
							##If the simulation had no decorrelation, add it now
							if apply_tdecor:
								l,m,n = get_lm(ra,ra0,dec,dec0)
								time_decor_adj = []
								for time_int in range(time_start,time_start+num_time_avg):
									for freq_int in range(freq_start,freq_start+num_freq_avg):
										adj_half_time = ((self.uv_container.time_res / 2.0)*SOLAR2SIDEREAL)*(15.0/3600.0)
										adj_central_lst = intial_lst + adj_half_time + (-(self.uv_container.time_res/2.0)*(15.0/3600.0))
										h0_adj = adj_central_lst*D2R - ra0
										tdecorr_factor_adj = tdecorr_phasetrack(X=x_length,Y=y_length,Z=z_length,d0=dec0,h0=h0_adj,l=l,m=m,n=n,time_int=self.uv_container.time_res)
										time_decor_adj.append(tdecorr_factor_adj)
										
								time_decor_adj = array(time_decor_adj).mean()
								
								sum_uvw_cal += (rotate_xx_complex*time_decor_adj)
								##And then add in the conjugate - we know -u,-v sees the conjugate of u,v
								##so don't need to do other phasing, just add in the conjugate
								sum_uvw_cal += (conjugate(rotate_xx_complex)*time_decor_adj)
							##Otherwise just sum the visibilities
							else:
								sum_uvw_cal += rotate_xx_complex
								##And then add in the conjugate - we know -u,-v sees the conjugate of u,v
								##so don't need to do other phasing, just add in the conjugate
								sum_uvw_cal += conjugate(rotate_xx_complex)
							
							if predict_tdecor:
								#l,m,n = get_lm(ra,ra0,dec,dec0)
								#tdecorr_factor = tdecorr_phasetrack(X=x_length,Y=y_length,Z=z_length,d0=dec,h0=ha_source,l=0,m=0,n=1,time_int=self.time_int)
								l,m,n = get_lm(ra,ra0,dec,dec0)
								tdecorr_factor = tdecorr_phasetrack(X=x_length,Y=y_length,Z=z_length,d0=dec0,h0=h0,l=l,m=m,n=n,time_int=self.time_int)
								tdecorr_flux = cal_source.extrap_fluxs[pos_ind] * tdecorr_factor
								##Times two because also grid conjugates of data so double amounts
								sum_tdecor += (2 * tdecorr_flux)
								sum_tdecor_factor += (2 * tdecorr_factor)
								
								##=========================================================================
								##Little section to adjust time decor for the missing decor in simulations
								time_decor_adj = []
								for time_int in range(time_start,time_start+num_time_avg):
									for freq_int in range(freq_start,freq_start+num_freq_avg):
										adj_half_time = ((self.uv_container.time_res / 2.0)*SOLAR2SIDEREAL)*(15.0/3600.0)
										adj_central_lst = intial_lst + adj_half_time + (-(self.uv_container.time_res/2.0)*(15.0/3600.0))
										h0_adj = adj_central_lst*D2R - ra0
										tdecorr_factor_adj = tdecorr_phasetrack(X=x_length,Y=y_length,Z=z_length,d0=dec0,h0=h0_adj,l=l,m=m,n=n,time_int=self.uv_container.time_res)
										time_decor_adj.append(tdecorr_factor_adj)
								sum_tdecor_adj += (2*array(time_decor_adj).mean())
								##=========================================================================
										
						sum_uvw_cal /= sum_weights
						sum_pixel.cal_names.append(cal_name)
						sum_pixel.ras.append(ra)
						sum_pixel.decs.append(dec)
						sum_pixel.sum_pixels.append(float(real(sum_uvw_cal)))
						
						if predict_tdecor:
							sum_tdecor /= sum_weights
							sum_tdecor_factor /= sum_weights
							sum_tdecor_adj /= sum_weights
							sum_pixel.tdecorr_factors.append(sum_tdecor_factor)
							sum_pixel.tdecorr_fluxes.append(sum_tdecor)
							sum_pixel.tdecorr_adj.append(sum_tdecor_adj)
							
						##If verbose, print out what is being seen at the command line	
						if verbose:
							if predict_tdecor:	
								print("At %.2f %.2f see %.10f expect %.10f (%.10f with tdecorr)" %(cal_source.ras[pos_ind],cal_source.decs[pos_ind],real(sum_uvw_cal),cal_source.extrap_fluxs[pos_ind],sum_tdecor))
							else:
								print("At %.2f %.2f see %.10f expect %.10f" %(cal_source.ras[pos_ind],cal_source.decs[pos_ind],real(sum_uvw_cal),cal_source.extrap_fluxs[pos_ind]))
				sum_pixels['%03d_%03d' %(time_start,freq_start)] = sum_pixel
						
		print("Finished sum_visi_at_source ----------------------")
		return sum_pixels
						
							
	def grid(self,image_size=None,over_sampling=2.0):
		'''Grids the data without correcting for decorrelation'''
		self.image_size=image_size
		
		##Resolution in u,v plane is set by the size of the image
		##in l,m coords, so sine of the angle - as zero is set
		##by the field centre, need to divide by two before
		##putting into the sine function
		cell_reso = 1.0 / (2*sin((image_size / 2.0)*D2R))
		self.cell_reso = cell_reso
		##Size of the grid is set by your longest baseline,
		##and then the over_sampling increases the resolution in
		##l,m space; 2 is the default which is nyquist sampling
		
		extra = 1
		##Set max_base_line by finding maximum u or v value
		max_u = self.uv_container.max_u
		min_u = self.uv_container.min_u
		max_v = self.uv_container.max_v
		min_v = self.uv_container.min_v
		
		max_coord = max(abs(array([max_u,max_v,min_u,min_v])))*over_sampling
		n2max = int(max_coord / cell_reso) + 1
		
		uu_range = arange(-n2max,n2max+1,1.0)*cell_reso
		vv_range = arange(-n2max,n2max+1,1.0)*cell_reso
		
		print('Gridding with N-side',len(uu_range))
		
		self.naxis_u = len(uu_range)
		self.naxis_v = len(vv_range)
		self.uu_range = uu_range
		self.vv_range = vv_range
		
		empty_uv = zeros((len(vv_range),len(uu_range)),dtype=complex)
		
		num_freq_avg = int(self.freq_int/self.uv_container.freq_res)
		num_time_avg = int(self.time_int/self.uv_container.time_res)
		
		print("Now gridding u,v data....")
		
		for time_start in range(0,len(self.uv_container.times),num_time_avg):
			for freq_start in range(0,len(self.uv_container.freqs),num_freq_avg):
				avg_uu = None
				avg_vv = None
				avg_ww = None
				sum_xxpol_comps = None
				#print("Gridding time cadence: %02d, freq cadence %02d" %(time_start,freq_start))
				for time_int in range(time_start,time_start+num_time_avg):
					for freq_int in range(freq_start,freq_start+num_freq_avg):
						freq = self.uv_container.freqs[freq_int]
						time = self.uv_container.times[time_int]
						#print("time step %.2f, freq step %.3f"%(time,freq))
						uvdata = self.uv_container.uv_data['%.3f_%05.2f' %(freq,time)]
						xxpol_real, xxpol_imag, xxpol_weight = uvdata.data[:,0,0],uvdata.data[:,0,1],uvdata.data[:,0,2]
						xxpol_comps = array([complex(xxpol_re,xxpol_im) for xxpol_re,xxpol_im in zip(xxpol_real, xxpol_imag)],dtype=complex)
						if type(avg_uu) == ndarray:
							avg_uu += uvdata.uu
							avg_vv += uvdata.vv
							avg_ww += uvdata.ww
							sum_xxpol_comps += xxpol_comps
						else:
							avg_uu = uvdata.uu
							avg_vv = uvdata.vv
							avg_ww = uvdata.ww
							sum_xxpol_comps = xxpol_comps
				##TODO is this a legit average? Add some kind of weighting in here?
				avg_uu /= (num_freq_avg * num_time_avg)
				avg_vv /= (num_freq_avg * num_time_avg)
				avg_ww /= (num_freq_avg * num_time_avg)
				sum_xxpol_comps /= (num_freq_avg * num_time_avg)
				
				if self.kernel == 'gaussian':
					self.uv_container.kernel_params = [2.0,2.0]
					empty_uv = grid(container=empty_uv,u_coords=avg_uu, v_coords=avg_vv, u_range=uu_range, v_range=vv_range,complexes=sum_xxpol_comps, weights=None, resolution=cell_reso,kernel='gaussian',kernel_params=self.uv_container.kernel_params)
					empty_uv = grid(container=empty_uv,u_coords=-avg_uu, v_coords=-avg_vv, u_range=uu_range, v_range=vv_range,complexes=conjugate(sum_xxpol_comps), weights=None, resolution=cell_reso,kernel='gaussian',kernel_params=self.uv_container.kernel_params,conjugate=True)
					
				elif self.kernel == 'time_decor':
					##Want to calculate time decor for LST at the centre of the time cadence
					##Convert to degrees
					half_time_cadence = ((self.time_int / 2.0)*SOLAR2SIDEREAL)*(15.0/3600.0)
					
					##First time and freq step of this cadence
					freq = self.uv_container.freqs[freq_start]
					time = self.uv_container.times[time_start]
					##This is the initial LST of this group of uvfits
					intial_lst = self.uv_container.uv_data['%.3f_%05.2f' %(freq,time)].LST
					central_lst = intial_lst + half_time_cadence
					if central_lst > 360: central_lst -= 360.0
					
					self.uv_container.kernel_params = [2.0,2.0]
					empty_uv = grid(container=empty_uv,u_coords=avg_uu, v_coords=avg_vv, u_range=uu_range, v_range=vv_range,complexes=sum_xxpol_comps, weights=None, resolution=cell_reso,kernel='gaussian',kernel_params=self.uv_container.kernel_params,central_lst=central_lst,time_decor=True,xyz_lengths=uvdata.xyz_lengths,phase_centre=[self.ra_phase,self.dec_phase],time_int=self.time_int)
					empty_uv = grid(container=empty_uv,u_coords=-avg_uu, v_coords=-avg_vv, u_range=uu_range, v_range=vv_range,complexes=conjugate(sum_xxpol_comps), weights=None, resolution=cell_reso,kernel='gaussian',kernel_params=self.uv_container.kernel_params,conjugate=True,central_lst=central_lst,time_decor=True,xyz_lengths=uvdata.xyz_lengths,phase_centre=[self.ra_phase,self.dec_phase],time_int=self.time_int)
					
				else:
					empty_uv = grid(container=empty_uv,u_coords=avg_uu, v_coords=avg_vv, u_range=uu_range, v_range=vv_range,complexes=sum_xxpol_comps, weights=None, resolution=cell_reso)
					empty_uv = grid(container=empty_uv,u_coords=-avg_uu, v_coords=-avg_vv, u_range=uu_range, v_range=vv_range,complexes=conjugate(sum_xxpol_comps), weights=None, resolution=cell_reso)
				
				sum_weights = 2*float(len(avg_uu))
				empty_uv /= sum_weights
				
		self.gridded_uv = empty_uv
		self.cell_reso = cell_reso
		return self.gridded_uv
	
	def image(self, plot_name=False, fits_name=False):
		'''Images the gridded data - requires gridded_uv != None'''
		##RA is backwards in images / fits files so invert u_axis
		
		print("Now FTing gridded data....")
		
		gridded_shift = fft.ifftshift(self.gridded_uv[::-1,:])
		img_array = fft.ifft2(gridded_shift) * (self.naxis_u * self.naxis_v)
		img_array_shift = fft.fftshift(img_array)
		
		if self.kernel == 'gaussian' or self.kernel == 'time_decor':
			##Calcuate the l,m coords of the image array,
			##and populate a meshgrid with them
			l_extent = 1.0 / self.cell_reso
			l_reso = (l_extent / (self.naxis_u))
			n2max = int((l_extent/2) / l_reso)
			
			l_range = arange(-n2max,n2max+1, 1) * l_reso
			m_range = arange(-n2max,n2max+1, 1) * l_reso
			l_mesh, m_mesh = meshgrid(l_range,m_range)
			
			##Calcuate the guassian created in image space create by the kernel gaussian
			##1 over the image_gaussian as we want to multiply image by correction factor
			##Do this as we want to mask poor pixels and not correct for them by setting to False;
			## float * False = 0.0, whereas float / False = inf
			sig_x,sig_y = self.uv_container.kernel_params
			correct_image = 1/image_gaussian(kernel_sig_x=sig_x,kernel_sig_y=sig_y,l_mesh=l_mesh,m_mesh=m_mesh,cell_reso=self.cell_reso)
			
			###Don't correct the image if going to blow up to huge pixel value
			thresh = 1.0 / 1e-12
			correct_image *= (correct_image < thresh)
			img_array_shift *= correct_image
		
		if plot_name:
			fig = plt.figure(figsize=(10,10))
			#from mpl_toolkits.axes_grid1 import make_axes_locatable
			#l_extent = sin(self.image_size*D2R)
			l_extent = 1.0 / self.cell_reso
			
			
			header = { 'NAXIS'  : 2,           			##Number of data axis
			'CTYPE1' : 'RA---SIN',    					##Projection type of X axis
			'CRVAL1' : self.ra_phase,   		     	##Central X world coord value
			'CRPIX1' : int(self.naxis_u / 2.0) + 1,		##Central X Pixel value
			'CUNIT1' : 'deg',             				##Unit of X axes
			'CDELT1' : -(l_extent / self.naxis_u)*R2D,  ##Size of pixel in world co-ord - +1 because fits files are 1 indexed
			'CTYPE2' : 'DEC--SIN',    				    ##Projection along Y axis
			'CRVAL2' : self.dec_phase,            		##Central Y world coord value
			'CRPIX2' : int(self.naxis_v / 2.0) + 1,     ##Central Y Pixel value
			'CUNIT2' : 'deg',                			##Unit of Y world coord
			'CDELT2' : (l_extent / self.naxis_v)*R2D    ##Size of pixel in deg
			} 
			
			print('Max abs, real, imag:',np_abs(img_array_shift).max(),real(img_array_shift).max(),imag(img_array_shift).max())
			print('Min abs, real, imag:',np_abs(img_array_shift).min(),real(img_array_shift).min(),imag(img_array_shift).min())

			wcs = WCS(header=header)
			ax1 = fig.add_axes([0.1,0.1,0.8,0.8],projection=wcs)

			im = ax1.imshow(real(img_array_shift),cmap='Blues',interpolation='none',origin='lower')#  real(img_arr).max()/2.0)#,vmin=-1,vmax=10)#,)

			ax1.grid()
			cax = fig.add_axes([0.92,0.1,0.05,0.8])
			cax.set_xticks([])
			cax.set_yticks([])
			cax.set_xticklabels([])
			cax.set_yticklabels([])
			plt.colorbar(im, cax=cax)
			fig.savefig(plot_name,bbox_inches='tight')
			
		if fits_name:
			hdu = fits.PrimaryHDU(real(img_array_shift))
			hdulist = fits.HDUList([hdu])
			header = hdulist[0].header
			
			l_extent = 1.0 / self.cell_reso
			
			header['CRPIX1']  = int(self.naxis_u / 2.0) + 1  ##+1 because fits files are 1 indexed
			header['CRPIX2']  = int(self.naxis_v / 2.0) + 1
			#header['CRVAL1']  = self.ra_point
			header['CRVAL1']  = self.ra_phase
			header['CRVAL2']  = self.dec_phase
			header['CDELT1']  = -(l_extent / self.naxis_u)*R2D
			header['CDELT2']  = (l_extent / self.naxis_v)*R2D
			header['CTYPE1']  = 'RA---SIN'
			header['CTYPE2']  = 'DEC--SIN'
			header['RADECSYS'] = 'FK5     '
			header['EQUINOX'] =  2000.
			
			hdu.writeto(fits_name,clobber=True)
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
	#def image_cal(self, plot_name=False, fits_name=False):
		#'''Images the gridded data - requires gridded_uv != None'''
		###RA is backwards in images / fits files so invert u_axis
		#img_arr_cal = fft.ifft2(self.cal_gridded_uv[::-1,:])
		#img_arr_cal = fft.fftshift(img_arr_cal)
		
		
		#if plot_name:
			#fig = plt.figure(figsize=(15,15))
			#ax1 = fig.add_subplot(111)
			#im_cal = ax1.imshow(np_abs(img_arr_cal),cmap='Blues',interpolation='none',origin='lower',vmax=np_abs(img_arr_cal).max()/4.0)
			#plt.colorbar(im_cal)
			#fig.savefig(plot_name,bbox_inches='tight')

		#if fits_name:
			#hdu = fits.PrimaryHDU(np_abs(img_arr_cal))
			#hdulist = fits.HDUList([hdu])
			#header = hdulist[0].header
			
			#print(arcsin(max_l)*R2D / self.naxis_v)
			
			#max_l = 1 / self.cell_reso
			
			#header['CRPIX1']  = int(self.naxis_u / 2.0)
			#header['CRPIX2']  = int(self.naxis_v / 2.0)
			#header['CRVAL1']  = self.ra
			#header['CRVAL2']  = self.dec
			#header['CDELT1']  = -arcsin(max_l)*R2D / self.naxis_u
			#header['CDELT2']  = arcsin(max_l)*R2D / self.naxis_v
			#header['CTYPE1']  = 'RA---SIN'
			#header['CTYPE2']  = 'DEC--SIN'
			#header['RADECSYS'] = 'FK5     '
			#header['EQUINOX'] =  2000.
			
			#hdu.writeto(fits_name,clobber=True)