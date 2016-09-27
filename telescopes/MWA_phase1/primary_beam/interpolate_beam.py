from numpy import *
from mwapy import pb
import h5py
from scipy import interpolate

delay_str = "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"

h5f = h5py.File(pb.h5file,'r')

base_freqs=array([int(x[3:]) for x in h5f.keys() if 'X1_' in x])
base_freqs.sort()

fine_chan = 40e3

full_band = (32*24*fine_chan)

lower_freq = 167.055e6 - full_band
upper_freq = 167.055e6 + full_band

pos_lowest = argmin(abs(base_freqs - lower_freq))
pos_highest = argmin(abs(base_freqs - upper_freq))

freqs = base_freqs[pos_lowest-1:pos_highest+1]

beam_cube_XX = zeros((31,31,len(freqs)))
beam_cube_YY = zeros((31,31,len(freqs)))

for freq in xrange(len(freqs)):
	data_XX = loadtxt('./base_images/beam_%s_%.3f_XX.txt' %(delay_str,freqs[freq]))
	beam_cube_XX[:,:,freq] = data_XX
	data_YY = loadtxt('./base_images/beam_%s_%.3f_YY.txt' %(delay_str,freqs[freq]))
	beam_cube_YY[:,:,freq] = data_YY
	
interp_freqs = arange(lower_freq,upper_freq+fine_chan,fine_chan)
interp_beam_cube_XX = zeros((31,31,len(interp_freqs)))
interp_beam_cube_YY = zeros((31,31,len(interp_freqs)))

for y in xrange(31):
	for x in xrange(31):
		pixel_vals_XX = beam_cube_XX[y,x,:]
		f_XX = interpolate.interp1d(freqs,pixel_vals_XX,kind='cubic')
		interp_vals_XX = f_XX(interp_freqs)
		interp_beam_cube_XX[y,x,:] = interp_vals_XX
		
		pixel_vals_YY = beam_cube_YY[y,x,:]
		f_YY = interpolate.interp1d(freqs,pixel_vals_YY,kind='cubic')
		interp_vals_YY = f_YY(interp_freqs)
		interp_beam_cube_YY[y,x,:] = interp_vals_YY
		
		
for freq in xrange(len(interp_freqs)):
	image_XX = interp_beam_cube_XX[:,:,freq]
	savetxt('./data/beam_%s_%.3f_XX.txt' %(delay_str,interp_freqs[freq]), image_XX)
	
	image_YY = interp_beam_cube_YY[:,:,freq]
	savetxt('./data/beam_%s_%.3f_YY.txt' %(delay_str,interp_freqs[freq]), image_YY)
	