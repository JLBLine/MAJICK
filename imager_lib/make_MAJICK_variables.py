from numpy import *
import pickle

##Convert degrees/rads
D2R = pi/180.0
R2D = 180.0/pi
##Speed of light m/s
VELC = 299792458.0

##Latitude of the MWA
MWA_LAT = -26.7033194444

##Always set the kernel size to an odd value
##Makes all ranges set to zero at central values
KERNEL_SIZE = 31

##Rotational velocity of the Earth rad / sex
W_E = 7.292115e-5
##Sidereal seconds per solar seconds - ie if 1s passes on
##the clock, sky has moved by 1.00274 secs of angle
SOLAR2SIDEREAL = 1.00274


with open('MAJICK_variables.pkl', 'w') as f:
    pickle.dump([D2R, R2D, VELC, MWA_LAT, KERNEL_SIZE, W_E, SOLAR2SIDEREAL], f)