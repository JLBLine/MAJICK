from optparse import OptionParser
from subprocess import call
import os

parser = OptionParser()

parser.add_option('-f', '--freq_start',
	help='Enter lowest frequency (MHz) - this is lowest fine channel')
	
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

parser.add_option('-b', '--band_nums',
	help='Which band numbers to process separated by a comma e.g. 1,5,9')

#parser.add_option('-p', '--phase_centre', default=False,
	#help='Phase centre of the observation in degrees as RA,DEC - as a default tracks the intial zenith point')

options, args = parser.parse_args()

band_nums = map(float,options.band_nums.split(','))
freq_int = float(options.freq_int)
time_int = float(options.time_int)

cwd = os.getcwd()
wd = cwd+'/qsub_uvfits'

if not os.path.exists(wd):
    os.makedirs(wd)
os.chdir(wd)

qsub_names = []

for band_num in band_nums:
	freq_start = float(options.freq_start) + (1.28 * (band_num - 1))
	cmd = "python $MAJICK_DIR/imager_lib/create_CHIPS_uvfits.py --freq_start=%.5f --num_freqs=32 --freq_int=%s --time_start=%s --num_times=%s --time_int=%s --tag_name=%s --uvfits_tag=%s --band_num=%d --data_loc=%s" %(freq_start,options.freq_int,options.time_start,options.num_times,options.time_int,options.tag_name,options.uvfits_tag,int(band_num),options.data_loc)
	
	file_name = 'qsub_%s_t%02d_f%.3f_%02d.sh' %(options.tag_name,time_int,freq_int,band_num)
	qsub_names.append(file_name)
	out_file = open(file_name,'w+')
	out_file.write('#!/bin/bash\n')
	out_file.write('#PBS -l nodes=1\n')
	
	out_file.write('#PBS -l walltime=01:00:00\n')
	out_file.write('#PBS -m e\n')
	out_file.write('#PBS -q sstar\n')
	out_file.write('#PBS -A p048_astro\n')

	out_file.write('source /home/jline/.bash_profile\n')
	out_file.write('cd %s\n' %wd)
	out_file.write(cmd+'\n')
	
	out_file.close()
	
os.chdir(cwd)

##Write out a controlling bash script to launch all the jobs
out_file = open('run_all_chipsuvfits_%s_t%02d_f%.3f.sh' %(options.tag_name,time_int,freq_int),'w+')
out_file.write('#!/bin/bash\n')
for qsub in qsub_names:
	out_file.write('MAIN_RUN=$(qsub ./qsub_uvfits/%s | cut -d "." -f 1)\n' %qsub)
	out_file.write('echo $MAIN_RUN\n')
out_file.close()