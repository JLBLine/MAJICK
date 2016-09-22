'''Functions to handle uvdata '''
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
#from calibrator_classes import *
#from imager_classes import *
#from uvdata_classes import *
from astropy.io import fits
from ephem import Observer,degrees
from numpy import sin,cos,pi,array,sqrt,arange,zeros,fft,meshgrid,where,arcsin,mod,real,ndarray,ceil,imag
from numpy import abs as np_abs
from numpy import exp as np_exp
from cmath import phase,exp
from sys import exit
#from astropy.wcs import WCS
#from time import time

D2R = pi/180.0
R2D = 180.0/pi
VELC = 299792458.0
MWA_LAT = -26.7033194444

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


class UVData(object):
	def __init__(self,uvfits=None):
		'''A single time and frequency step of uvdata. Includes a dictioary
		containing all telecope X,Y,Z and antenna pairs from which to calcualte
		baseline lengths'''
		self.uvfits = uvfits
		HDU = fits.open(uvfits)
		data0 = HDU[0].data
		header0 = HDU[0].header
		data1 = HDU[1].data
		header1 = HDU[1].header

		##Find the pointing centre
		self.ra_point = header0['CRVAL6']
		self.dec_point = header0['CRVAL7']
		self.freq = header0['CRVAL4']
		
		##Reformat date from header into something
		##readble by Observer to calcualte LST
		date,time = header1['RDATE'].split('T')
		MRO.date = '/'.join(date.split('-'))+' '+time
		self.LST = float(MRO.sidereal_time())*R2D
		
		#print(self.ra_point,self.dec_point,header1['RDATE'],self.LST)

		##Requires u,v,w in units of wavelength, stored in seconds
		##(u * c) / (c / freq) = u * freq
		self.uu = data0['UU'] * header0['CRVAL4']
		self.vv = data0['VV'] * header0['CRVAL4']
		self.ww = data0['WW'] * header0['CRVAL4']
		
		
		self.antennas = {}
		xyz = data1['STABXYZ'] * (header0['CRVAL4'] / VELC)
		for i in xrange(len(xyz)):
			self.antennas['ANT%03d' %(i+1)] = xyz[i]
		
		##The field BASELINE is the baseline number (256ant1 + ant2 +
		##subarray/100.)
		##Found that out from here:
		##https://www.mrao.cam.ac.uk/~bn204/alma/memo-turb/uvfits.py
		baselines = data0['BASELINE']
		self.antenna_pairs = []
		self.xyz_lengths = []
		self.uvw_calc = []
		self.uvw_zenith = []
		
		#test_xyz = open("test_xyz.txt",'w+')
		
		for baseline in baselines:
			ant2 = mod(baseline, 256)
			ant1 = (baseline - ant2)/256
			
			self.antenna_pairs.append(('ANT%03d' %ant1,'ANT%03d' %ant2))
			x_length,y_length,z_length = self.antennas['ANT%03d' %ant1] - self.antennas['ANT%03d' %ant2]
			self.xyz_lengths.append([x_length,y_length,z_length])
			#test_xyz.write('%.3f %.3f %.3f\n' %(x_length,y_length,z_length))
			#self.uvw_calc.append(get_uvw(x_length,y_length,z_length,self.dec_point*D2R,self.LST*D2R - self.ra_point*D2R))
			#self.uvw_zenith.append(get_uvw(x_length,y_length,z_length,MWA_LAT*D2R,0.0))
		#test_xyz.close()
			
		##TODO read in 'CRVAL3' to determine data shape
		##and set num_polar
		self.num_polar = 4
		self.data = data0['DATA'][:,0,0,0,0,:,:]
		
		self.max_u = self.uu.max()
		self.max_v = self.vv.max()
		self.min_u = self.uu.min()
		self.min_v = self.vv.min()
		
		self.cal_data = None
		
		##Do this as memmap (some python memory map thing) makes
		##a copy of the hdu everytime you reference it, and so
		##HDU.close() doesn't shut everything down - you can quite
		##easily end up with two many HDUs open
		del data0
		del header0
		del data1
		del header1
		del HDU[0].data
		#del HDU[0].header
		del HDU[1].data
		#del HDU[1].header
		HDU.close()
		
class UVContainer(object):
	def __init__(self,uv_tag=None,freq_start=None,freq_end=None,freq_res=None,time_start=None,time_end=None,time_res=None):
		'''An array containing UVData objects in shape = (num time steps, num freq steps'''
		##TODO do an error check for all required uvfits files
		##Have a custom error?
		self.freq_res = freq_res
		self.time_res = time_res
		
		freqs = arange(freq_start,freq_end,freq_res)
		times = arange(time_start,time_end,time_res)
		
		self.freqs = list(freqs)
		self.times = list(times)
		self.uv_data = {}
		self.cal_uv_data = {}
		self.kernel_params = None
		
		
		max_us = []
		max_vs = []
		min_us = []
		min_vs = []
		
		print("Now loading uvfits data....")
		
		##Set the phase centre from the ra_point, dec_point from the first time step
		##TODO get this from somewhere smarter
		
		if time_res < 1:
			self.ra_phase = UVData('%s_%.3f_%05.2f.uvfits' %(uv_tag,freqs[0],times[0])).ra_point
			self.dec_phase = UVData('%s_%.3f_%05.2f.uvfits' %(uv_tag,freqs[0],times[0])).dec_point
		else:
			self.ra_phase = UVData('%s_%.3f_%02d.uvfits' %(uv_tag,freqs[0],times[0])).ra_point
			self.dec_phase = UVData('%s_%.3f_%02d.uvfits' %(uv_tag,freqs[0],times[0])).dec_point
			#self.LST = UVData('%s_%.3f_%05.2f.uvfits' %(uv_tag,freqs[0],times[0])).LST
		
		for i in xrange(len(freqs)):
			for j in xrange(len(times)):
				
				#print("Loading and rotating uvfits time %03d freq %.3f" %(times[j],freqs[i]))
				#from subprocess import call
				if time_res < 1:
					uvdata = UVData('%s_%.3f_%05.2f.uvfits' %(uv_tag,freqs[i],times[j]))
				else:
					uvdata = UVData('%s_%.3f_%02d.uvfits' %(uv_tag,freqs[i],times[j]))
				
				import matplotlib.pyplot as plt
				from mpl_toolkits.axes_grid1 import make_axes_locatable
			
				def add_colorbar(im,ax):
					divider = make_axes_locatable(ax)
					cax = divider.append_axes("right", size="5%", pad=0.05)
					cbar = fig.colorbar(im, cax = cax)
				
				fig = plt.figure(figsize=(10,7))
			
				ax1 = fig.add_subplot(211)
				ax2 = fig.add_subplot(212)
				
				#print(len(uvdata.data[:,0,0]))
				#print(len(uvdata.uu))
				
				im1 = ax1.scatter(uvdata.uu,uvdata.vv,c=uvdata.data[:,0,0],cmap='gnuplot')
				ax1.set_title('Analytic and works (real)')
				add_colorbar(im1,ax1)
				
				im2 = ax2.scatter(uvdata.uu,uvdata.vv,c=uvdata.data[:,0,1],cmap='gnuplot')
				ax2.set_title('From point source image (imag)')
				add_colorbar(im2,ax2)
				
				plt.tight_layout()
				
				fig.savefig('read-in-data.png')
				plt.close()
				
				

				###Here we try to phase rotate everything to the first pointing 
				for k in xrange(len(uvdata.uu)):
					##Find the u,v,w coordinates for the phase centre for the LST of the given uv data
					x_length,y_length,z_length = uvdata.xyz_lengths[k]
					u_phase, v_phase, w_phase = get_uvw(x_length,y_length,z_length,self.dec_phase*D2R,uvdata.LST*D2R - self.ra_phase*D2R)
					
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
					PhaseConst = - 1j * 2 * pi
					rotate_xx_complex = xx_complex * exp(PhaseConst * w_phase)
					
					uvdata.data[k][0,0] = real(rotate_xx_complex)
					uvdata.data[k][0,1] = imag(rotate_xx_complex)
					
				
				#fig = plt.figure(figsize=(10,7))
			
				#ax1 = fig.add_subplot(211)
				#ax2 = fig.add_subplot(212)
				
				#im1 = ax1.scatter(uvdata.uu,uvdata.vv,c=uvdata.data[:,0,0],cmap='gnuplot')
				#ax1.set_title('Analytic and works (real)')
				#add_colorbar(im1,ax1)
				
				#im2 = ax2.scatter(uvdata.uu,uvdata.vv,c=uvdata.data[:,0,1],cmap='gnuplot')
				#ax2.set_title('From point source image (imag)')
				#add_colorbar(im2,ax2)
				
				#plt.tight_layout()
				
				#fig.savefig('phase-rotated-data.png')
				#plt.close()
					
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