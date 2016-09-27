from numpy import *

data = loadtxt('MWA_Tools_phase1-antenna-locs.txt',usecols=(1,2,3))

savetxt('antenna_locations_MWA_phase1.txt',data)
