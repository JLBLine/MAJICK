#!/usr/bin/python
from subprocess import call
from sys import exit
from optparse import OptionParser
from numpy import pi, arange, ceil
import os
from glob import glob
from uvdata_classes import *

R2D = 180.0 / pi
D2R = pi / 180.0
MWA_LAT = -26.7033194444

parser = OptionParser()

parser.add_option('-n','--output_name', help='Enter prefix name for outputs')
parser.add_option('-d','--debug',default=False,action='store_true', help='Enable to debug with print statements')
parser.add_option('-m','--metafits', help='Enter name of metafits file to base obs on')
parser.add_option('-t','--time', help='Enter start,end of sim in seconds from the beginning of the observation (as set by metafits)')
parser.add_option('-x','--twosec', default=False, help='Enable to force a different time cadence - enter the time in seconds')
parser.add_option('-c','--beam', default=False, action='store_true', help='Enable to apply beam to simulations')
parser.add_option('-f','--freq_decor', default=False, action='store_true', help='Enable to switch on frequency decorrelation')
parser.add_option('-g','--time_decor', default=False, action='store_true', help='Enable to switch on time decorrelation')
parser.add_option('-p','--phase_centre', default=False, help='Specify phase centre; enter as ra_phase,dec_phase (deg)')
parser.add_option('-a','--telescope', default='MWA_phase1', help='Enter telescope used for simulation. Default = MWA_phase1')
parser.add_option('-b','--band_nums', help='Enter band numbers to simulate, separated by a comma eg 1,3,4')
parser.add_option('-i', '--data_loc', default='./data',	help='Location to output the uvfits to OR location of uvfits if just adding diffuse model. Default = ./data')
parser.add_option('-s','--srclist', help='Enter srclist to base sky model on')
parser.add_option('-z','--fix_beam', default=False, action='store_true', help='Enable to switch on fixed beam observation')

options, args = parser.parse_args()
debug = options.debug

def run_command(cmd):
	if debug: print cmd
	call(cmd,shell=True)
	
##Open the metafits file and get the relevant info
try:
	import pyfits
except ImportError:
	import astropy.io.fits as pyfits

try:
	f=pyfits.open(options.metafits)
except Exception,e:
	print 'Unable to open metafits file %s: %s' % (options.metafits,e)
	exit(1)
	
def test_avail(key):
	if not key in f[0].header.keys():
		print 'Cannot find %s in %s' % (key,options.metafits)
		exit(1)

for key in ['DATE-OBS','FREQCENT','FINECHAN','INTTIME','BANDWDTH']:
	test_avail(key)


intial_date = f[0].header['DATE-OBS']
dump_time = float(f[0].header['INTTIME'])

if options.twosec: dump_time = float(options.twosec)

ch_width = float(f[0].header['FINECHAN'])*1e+3
freqcent = float(f[0].header['FREQCENT'])*1e+6
b_width = float(f[0].header['BANDWDTH'])*1e+6
low_freq = freqcent - (b_width/2) - (ch_width/2)

band_nums = map(int,options.band_nums.split(','))
start_tstep,end_tstep = map(float,options.time.split(','))
tsteps = arange(start_tstep,end_tstep,dump_time)

##Find all of the successfully simulated files
found_files = glob('%s/%s*.uvfits' %(options.data_loc,options.output_name))

##Find out where we are and setup a place to store the qsub scripts

cwd = os.getcwd()
wd = cwd+'/qsub_majick'

if not os.path.exists(wd):
    os.makedirs(wd)
os.chdir(wd)

qsub_names = []

for band_num in band_nums:
	
	base_freq = ((band_num - 1)*(b_width/24.0)) + low_freq
	sim_commands = []
	
	freq_range = (base_freq / 1e+6) + (arange(32)*(ch_width / 1e+6))
	
	for freq in freq_range:
		for time in tsteps:
			if dump_time < 1:
				uvfits_name = "%s/%s_%.3f_%05.2f.uvfits" %(options.data_loc,options.output_name,freq,time)
			else:
				uvfits_name = "%s/%s_%.3f_%02d.uvfits" %(options.data_loc,options.output_name,freq,int(time))
			#print uvfits_name
			if uvfits_name not in found_files:
				this_date = add_time_uvfits(intial_date,time)
				
				sim_command = "python $MAJICK_DIR/simulate_uvfits.py"
				sim_command += " --freq_start=%.5f" %freq
				sim_command += " --num_freqs=1"
				sim_command += " --freq_res=%.5f" %(ch_width / 1e+6)
				sim_command += " --time_start=%.5f " %time
				sim_command += " --num_times=1"
				sim_command += " --time_res=%.5f" %dump_time
				sim_command += " --date=%s" %this_date
				sim_command += " --tag_name=%s" %options.output_name
				sim_command += " --data_loc=%s" %options.data_loc
				sim_command += " --telescope=%s" %options.telescope
				sim_command += " --srclist=%s" %options.srclist
				if options.beam:
					sim_command += " --beam"
				if options.phase_centre:
					sim_command += " --phase_centre=%s" %options.phase_centre
				if options.time_decor:
					sim_command += " --time_decor"
				if options.freq_decor:
					sim_command += " --freq_decor"
				if options.fix_beam:
					sim_command += " --fix_beam"
					
				sim_commands.append(sim_command)
				
	if len(sim_commands) > 0:

		file_name = 'qsub_missing-%s_band%02d_t%d-%d.sh' %(options.output_name,band_num,int(tsteps[0]),int(tsteps[-1]))
		qsub_names.append(file_name)
		out_file = open(file_name,'w+')
		out_file.write('#!/bin/bash\n')
		out_file.write('#PBS -l nodes=1\n')
		
		##Lets say 3 mins per sim so
		
		num_mins = len(sim_commands) * 3.0
		hours = ceil(num_mins / 60.0)
		
		out_file.write('#PBS -l walltime=%02d:00:00\n' %int(hours) )
		out_file.write('#PBS -m e\n')
		out_file.write('#PBS -q sstar\n')
		out_file.write('#PBS -A p048_astro\n')

		out_file.write('source /home/jline/.bash_profile\n')
		out_file.write('cd %s\n' %wd)
		
		for sim_command in sim_commands:
			out_file.write(sim_command+'\n')
		
		out_file.close()
	
os.chdir(cwd)

##Write out a controlling bash script to launch all the jobs
out_file = open('run_all_majicksim_missing-%s.sh' %options.output_name,'w+')
out_file.write('#!/bin/bash\n')
for qsub in qsub_names:
	out_file.write('MAIN_RUN=$(qsub ./qsub_majick/%s | cut -d "." -f 1)\n' %qsub)
	out_file.write('echo $MAIN_RUN\n')
out_file.close()
