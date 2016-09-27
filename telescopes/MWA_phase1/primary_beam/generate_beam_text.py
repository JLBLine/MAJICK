
from numpy import *
from mwapy.pb import primary_beam #import MWA_Tile_full_EE
KERNEL_SIZE = 31
from optparse import OptionParser
import matplotlib.pyplot as plt
import h5py
from mwapy import pb
import numpy as np

#parser = OptionParser()

#parser.add_option('-f', '--frequency',
	#help='Frequency at which to simulate the beam (Hz)')
	
#parser.add_option('-d', '--delays',default="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
	#help='Enter beam delays as 16 delays separated by commas - default is zenith: "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"')

#options, args = parser.parse_args()

#freq = float(options.frequency)
#delays = map(int,options.delays.split(','))

delays = "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
delays = map(int,delays.split(','))

def sample_image_coords(n2max=None,l_reso=None,num_samples=KERNEL_SIZE):
	'''Creates a meshgrid of l,m coords to give a specified 
	size array which covers a given range of l and m - always
	the same range of l,m'''

	##DO NOT TOOUCH THE MAGIC
	##So this makes you sample at zero and all the way up to half a
	##resolution element away from the edge of your range
	##n2max is half the l range, ie want -n2max <= l <= n2max
	offset = n2max*l_reso / num_samples
	l_sample = linspace(-n2max*l_reso + offset, n2max*l_reso - offset, num_samples)
	m_sample = linspace(-n2max*l_reso + offset, n2max*l_reso - offset, num_samples)
	l_mesh, m_mesh = meshgrid(l_sample,m_sample)
	
	return l_mesh, m_mesh

#def add_colorbar(im,ax):
#	divider = make_axes_locatable(ax)
#	cax = divider.append_axes("right", size="5%", pad=0.05)
#	cbar = fig.colorbar(im, cax = cax)

l_reso = 2.0 / KERNEL_SIZE
n2max = 1.0 / l_reso


l_mesh, m_mesh = sample_image_coords(n2max=n2max,l_reso=l_reso)  
			
mask = l_mesh**2 + m_mesh**2 <= 1
mask_zero = ones(l_mesh.shape) * mask
l_mesh *= mask_zero
m_mesh *= mask_zero
			
za = arcsin(sqrt(l_mesh*l_mesh + m_mesh*m_mesh))
az = arctan2(l_mesh,m_mesh)

##New beam model does NOT let you have -ve azimuth - so wherever
##you have negative, need to have 2*pi + -ve value, i.e. must
##make -pi into pi, and -0.1 => 2*pi - 0.1

neg_az_mask = az < 0
pi_add_az = ones(az.shape)*2*pi * neg_az_mask

az += pi_add_az


#base_freq = 167.055 + (29 * 0.04)
#upper_freq = 167.055 + (39 * 0.04)

#freqs = arange(base_freq,upper_freq,0.04)

freqs = linspace(167e+6,168.5e+6,5)

#freqs = arange(180e+6,191e+6,40e+3)

pixel_vals = []


#for freq in freqs:
	
h5f = h5py.File(pb.h5file,'r')

freqs=np.array([int(x[3:]) for x in h5f.keys() if 'X1_' in x])
freqs.sort()

#plt.plot(freqs)
#plt.show()

full_band = (32*24*40e3)

base_freq = 167.055e6 - full_band
upper_freq = 167.055e6 + full_band

pos_lowest = np.argmin(np.abs(freqs - base_freq))
pos_highest = np.argmin(np.abs(freqs - upper_freq))

for freq in freqs[pos_lowest-1:pos_highest+1]:
	
	XX,YY = primary_beam.MWA_Tile_full_EE(za, az, freq=freq, delays=delays, zenithnorm=True, power=True, interp=False)
	
	#XX,YY = primary_beam.MWA_Tile_advanced(za, az, freq=freq, delays=delays, zenithnorm=False, power=True, jones=False)
	
	#pixel_vals.append(XX[11,15])

	delay_str = str(delays[0])
	for delay in delays[1:]: delay_str += ','+str(delay)

	savetxt('./base_images/beam_%s_%.3f_XX.txt' %(delay_str,freq), XX*mask_zero)
	savetxt('./base_images/beam_%s_%.3f_YY.txt' %(delay_str,freq), YY*mask_zero)
	
	fig = plt.figure(figsize=(10,10))

	ax = fig.add_subplot(111)

	ax.imshow(log(XX)*mask_zero,cmap='Blues',interpolation='none')

	fig.savefig('./base_images/beam_%s_%.3f_XX.png' %(delay_str,freq/1e+6),bbox_inches='tight')
	plt.close()











