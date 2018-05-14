'''Gridding functions used in Jack's dummy imager'''
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
#from calibrator_classes import *
#from imager_classes import *
from uvdata_classes import *
#from astropy.io import fits
#from ephem import Observer,degrees
from numpy import sin,cos,pi,array,sqrt,arange,zeros,fft,meshgrid,where,arcsin,mod,real,ndarray,ceil,linspace,sinc,repeat,reshape,loadtxt,nan_to_num,ones,conjugate,complex128,complex64
from numpy import abs as np_abs
from numpy import exp as np_exp
from cmath import phase,exp
from sys import exit
#from astropy.wcs import WCS
#from time import time
#import matplotlib.pyplot as plt
import scipy as sp
from mwapy import ephem_utils
from mwapy.pb import primary_beam
from mwapy.pb import mwa_tile
from os import environ
import pickle
from numba import jit

try:
    from pyfft.cuda import Plan
    import pycuda.driver as cuda
    from pycuda.tools import make_default_context
    import pycuda.gpuarray as gpuarray
except:
    pass


MAJICK_DIR = environ['MAJICK_DIR']
with open('%s/imager_lib/MAJICK_variables.pkl' %MAJICK_DIR) as f:  # Python 3: open(..., 'rb')
    D2R, R2D, VELC, MWA_LAT, KERNEL_SIZE, W_E, SOLAR2SIDEREAL = pickle.load(f)


def add_kernel(uv_array,u_ind,v_ind,kernel):
    '''Takes v by u sized kernel and adds it into
    a numpy array at the u,v point u_ind, v_ind
    Kernel MUST be odd dimensions for symmetry purposes'''
    ker_v,ker_u = kernel.shape
    width_u = int((ker_u - 1) / 2)
    width_v = int((ker_v - 1) / 2)
    array_subsec = uv_array[v_ind - width_v:v_ind+width_v+1,u_ind - width_u:u_ind+width_u+1]
    array_subsec += kernel
    
def apply_kernel(uv_data_array=None,u_ind=None,v_ind=None,kernel=None):
    '''Takes an array containing uvdata, and samples a visisibility at the
    given u,v index with a given kernel. Kernel MUST be odd dimensions 
    for symmetry purposes'''
    ker_v,ker_u = kernel.shape
    width_u = ker_u // 2
    width_v = ker_v // 2
    
    ##Just return the central data point of the kernel
    array_subsec = uv_data_array[v_ind - width_v:v_ind+width_v+1,u_ind - width_u:u_ind+width_u+1]*kernel
    
    return array_subsec.sum()
    
def gaussian(sig_x=None,sig_y=None,gridsize=KERNEL_SIZE,x_offset=0,y_offset=0):
    '''Creates a gaussian array of a specified gridsize, with the
    the gaussian peak centred at an offset from the centre of the grid'''
    x_cent = int(gridsize / 2.0) + x_offset
    y_cent = int(gridsize / 2.0) + y_offset
    
    x = arange(gridsize)
    y = arange(gridsize)
    x_mesh, y_mesh = meshgrid(x,y)
    
    x_bit = (x_mesh - x_cent)*(x_mesh - x_cent) / (2*sig_x*sig_x)
    y_bit = (y_mesh - y_cent)*(y_mesh - y_cent) / (2*sig_y*sig_y)
    
    amp = 1 / (2*pi*sig_x*sig_y)
    gaussian = amp*np_exp(-(x_bit + y_bit))
    return gaussian

def image_gaussian(kernel_sig_x=None,kernel_sig_y=None,l_mesh=None,m_mesh=None,cell_reso=None):
    '''Takes desired properties for a kernel in u,v space (in pixel coords),
    and creates FT of this'''
    
    fiddle = 2*pi
    sig_l, sig_m = 1.0/(fiddle*cell_reso*kernel_sig_x), 1.0/(fiddle*cell_reso*kernel_sig_y)

    l_bit = l_mesh*l_mesh / (2*sig_l*sig_l)
    m_bit = m_mesh*m_mesh / (2*sig_m*sig_m)
    
    return np_exp(-(l_bit + m_bit))


def spheroidal_1D(x=None):
    '''Calculates a 1D spheroidal. x is -1.0 and 1.0 at the edge of the image  '''
    m = 1; n = 1;
    c = 1 # look this up in the scipy docs and change to whatever

    mask = (abs(x) < 1.0)
    (a_fn, sph_deriv) = sp.special.pro_ang1(m, n, c, x*mask)
    a_fn *= mask

    # x = -1 and 1 are undefined, so extrapolate from the neighbours
    # may want to test that the ends are if fact -1 and 1.
    a_fn[0] = 3. * (a_fn[1] - a_fn[2]) + a_fn[3]
    a_fn[-1] = a_fn[0]

    return a_fn

def spheroidal_2D(size=KERNEL_SIZE):
    '''Returns a 2D array of size x size using spheroidal_1D'''
    ##I think we always need an odd kernel size to get a 
    ##zero in this function
    values = linspace(-1,1,size)
    
    x_spheroidal = spheroidal_1D(values)
    y_spheroidal = spheroidal_1D(values)
    
    x_mesh, y_mesh = meshgrid(x_spheroidal,y_spheroidal)
    
    return x_mesh * y_mesh

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

#@profile
def image2kernel(image=None,cell_reso=None,u_off=0.0,v_off=0.0,l_mesh=None,m_mesh=None):
    '''Takes an input image array, and FTs to create a kernel
    Uses the u_off and v_off (given in pixels values), cell resolution
    and l and m coords to phase    shift the image, to create a kernel 
    with the given u,v offset'''
    
    ##TODO WARNING - if you want to use a complex image for the kernel,
    ##may need to either take the complex conjugate, or flip the sign in
    ##the phase shift, or reverse the indicies in l_mesh, m_mesh. Or some
    ##combo of all!! J. Line 20-07-2016
    phase_shift_image =  image * np_exp(2.0j * pi*(u_off*cell_reso*l_mesh + v_off*cell_reso*m_mesh))
    #phase_shift_image =  image * np_exp(2j * pi*(u_off*l_mesh + v_off*m_mesh))

    ##FFT shift the image ready for FFT
    phase_shift_image = fft.ifftshift(phase_shift_image)
    ##Do the forward FFT as we define the inverse FFT for u,v -> l,m. 
    ##Scale the output correctly for the way that numpy does it, and remove FFT shift
    recovered_kernel = fft.fft2(phase_shift_image) / (image.shape[0] * image.shape[1])
    recovered_kernel = fft.fftshift(recovered_kernel)
    #return recovered_kernel
    return recovered_kernel

@jit
def get_lm(ra,ra0,dec,dec0):
    '''Calculate l,m for a given phase centre ra0,dec0 and sky point ra,dec
    Enter angles in radians'''
    
    ##RTS way of doing it
    cdec0 = cos(dec0)
    sdec0 = sin(dec0)
    cdec = cos(dec)
    sdec = sin(dec)
    cdra = cos(ra-ra0)
    sdra = sin(ra-ra0)
    l = cdec*sdra
    m = sdec*cdec0 - cdec*sdec0*cdra
    n = sdec*sdec0 + cdec*cdec0*cdra
    
    return l,m,n

def do_sinc(x=None):
    '''In-built numpy sinc function is slooooow'''
    ##If zero dividing by zero so have an if test
    ##TODO this breaks with an array - I think we should never
    ##hit a situation where x == 0 but ya never know
    
    if type(x) == ndarray:
        ##Numpy is faster if we have a 2D array
        sinc_x = sinc(x)
    ##Otherwise for just a number, sin(x) / x is quicker
    else:
        if x == 0:
            sinc_x = 1
        else:
            sinc_x = sin(pi*x) / (pi*x)
            
    return sinc_x

def tdecorr_phasetrack(X=None,Y=None,Z=None,d0=None,h0=None,l=None,m=None,n=None,time_int=None):
    '''Find the time decorrelation factor at a specific l,m,n given the X,Y,Z length coords
    of a baseline in wavelengths, phase tracking at dec and hour angle d0, h0
    and integrating over time t'''
    part1 = l*(cos(h0)*X - sin(h0)*Y)
    part2 = m*(sin(d0)*sin(h0)*X + sin(d0)*cos(h0)*Y)
    part3 = (n-1)*(-cos(d0)*sin(h0)*X - cos(d0)*cos(h0)*Y)
    nu_pt = W_E*(part1+part2+part3)
    D_t = do_sinc(nu_pt*time_int)
    return D_t

def tdecorr_nophasetrack(X=None,Y=None,dec_s=None,ha_s=None,t=None):
    '''Calculates the time decorrelation factor using u, at a given
    sky position and time intergration'''
    nu_fu= -W_E*cos(dec_s)*(sin(ha_s)*X + cos(ha_s)*Y)
    D_t=do_sinc(nu_fu*t)
    return D_t

def fdecorr(u=None,v=None,w=None,l=None,m=None,n=None,chan_width=None,freq=None,phasetrack=False):
    '''Calculates the frequency decorrelation factor for a baseline u,v,w
    at a given sky position l,m,n, for a given channel width and frequency
    Assumes no phase tracking - add phasetrack=True to set w(n-1)'''
    
    if phasetrack:
        D_f = do_sinc((chan_width / freq) * (u*l + v*m + w*(n-1)))
    else:
        D_f = do_sinc((chan_width / freq) * (u*l + v*m + w*n))
    return D_f

#def fdecorr(x_length=None,y_length=None,z_length=None,dec=None,ha=None,freq=None,chan_width=None):
    #'''Take the X,Y,Z_lambda coords of a baseline, and calculates the 
    #frequency decorrelation in the direction of a source at location ha,dec'''
    ###Eq 2-12 in TCP
    
    #w = cos(dec)*cos(ha)*x_length - cos(dec)*sin(ha)*y_length + sin(dec)*z_length
    
    ###Geometric time delay across baseline
    #t_g = w/freq
    
    #return sinc(chan_width*t_g)
    
@jit(cache=True)
def find_closet_uv(u=None,v=None,u_range=None,v_range=None,resolution=None):
    #print(u,v)
    ##Find the difference between the gridded u coords and the desired u
    u_offs = np_abs(u_range - u)
    ##Find out where in the gridded u coords the current u lives;
    ##This is a boolean array of length len(u_offs)
    u_true = u_offs < resolution/2.0
    ##Find the index so we can access the correct entry in the container
    u_ind = where(u_true == True)[0]
    
    ##Use the numpy abs because it's faster (np_abs)
    v_offs = np_abs(v_range - v)
    v_true = v_offs < resolution/2
    v_ind = where(v_true == True)[0]
    
    ##If the u or v coord sits directly between two grid points,
    ##just choose the first one ##TODO choose smaller offset?
    if len(u_ind) == 0:
        u_true = u_offs <= resolution/2
        u_ind = where(u_true == True)[0]
        #print('here')
        #print(u_range.min())
    if len(v_ind) == 0:
        v_true = v_offs <= resolution/2
        v_ind = where(v_true == True)[0]
    u_ind,v_ind = u_ind[0],v_ind[0]
    
    ##TODO is this -ve legit??? Seems so...
    u_offs = u_range - u
    v_offs = v_range - v
    
    u_off = -(u_offs[u_ind] / resolution)
    v_off = -(v_offs[v_ind] / resolution)
    #u_off = -(u_range[u_ind] - u)
    #v_off = -(v_range[u_ind] - v)
    
    return u_ind,v_ind,u_off,v_off

#from sys import exit
#@jit(cache=True)
def grid(container=None,u_coords=None, v_coords=None, u_range=None, v_range=None,complexes=None, weights=None,resolution=None, kernel='gaussian', kernel_params=None, central_lst=None,time_decor=False,freq_decor=False,xyz_lengths=None,phase_centre=None,time_int=None,freq_int=None,central_frequency=None):
    '''A simple(ish) gridder - defaults to gridding with a gaussian '''
    for i in xrange(len(u_coords)):
        u,v,comp = u_coords[i],v_coords[i],complexes[i]
        ##Find the difference between the gridded u coords and the current u
        ##Get the u and v indexes in the uv grdding container
        u_ind,v_ind,u_off,v_off = find_closet_uv(u=u,v=v,u_range=u_range,v_range=v_range,resolution=resolution)
        
        if kernel == 'gaussian':
            sig_x,sig_y = kernel_params
            #kernel_array = gaussian(sig_x=sig_x,sig_y=sig_y,x_offset=u_off,y_offset=v_off)
            
            ##TODO TODO TODO DO THIS
            ##Think this needs to span different l,m space for an even array???????
            
            ##Calculate the extent of the pixels in l,m space
            l_extent = 1.0 / resolution
            l_reso = l_extent / len(u_range)
            n2max = int((l_extent/2) / l_reso)
            ##Create a meshgrid to sample over whole l,m range with default 31 samples
            
            l_mesh, m_mesh = sample_image_coords(n2max=n2max,l_reso=l_reso)
            ##Calculate the gaussian caused in image space due to desired params of gridding gaussian in u,v
            image_kernel = image_gaussian(kernel_sig_x=sig_x,kernel_sig_y=sig_y,l_mesh=l_mesh,m_mesh=m_mesh,cell_reso=resolution)
            
            ##FT the image kernel to create the gridding kernel
            
            if time_decor:
                ra0,dec0 = phase_centre
                h0 = central_lst - ra0
                
                X,Y,Z = xyz_lengths[i]
                n_mesh = sqrt(1 - (l_mesh*l_mesh + m_mesh*m_mesh))
                
                image_time = (tdecorr_phasetrack(X=X,Y=Y,Z=Z,d0=dec0*D2R,h0=h0*D2R,l=l_mesh,m=m_mesh,n=n_mesh,time_int=time_int))
                
                ###EXTENSION SECTION LEAVE COMMENTED-------------------------------------------
                ###If not phase tracking in the simulations, have to fiddle by the difference
                #for time_step in xrange(len(num_time_step)):
                    #this_lst = intial_lst + time_step*time_reso
                    #this_ha0 = this_lst - ra0
                    #this_ha = this_lst - ra
                    #image_time /= tdecorr_phasetrack(X=X,Y=Y,Z=Z,d0=dec0*D2R,h0=this_ha0*D2R,l=l_mesh,m=m_mesh,n=n_mesh,time_int=time_reso)
                    #u, v, w = get_uvw(X,Y,Z,dec0,this_ha0)
                    ###TODO - get nophasetrack defined in l,m,n
                    #image_time *= tdecorr_nophasetrack(u=u,d=dec,H=this_ha,t=time_reso)
                ###EXTENSION SECTION LEAVE COMMENTED-------------------------------------------
                
                image_time = 1 / image_time
                image_kernel *= image_time
                
            if freq_decor:
                ra0,dec0 = phase_centre
                h0 = central_lst - ra0
                
                X,Y,Z = xyz_lengths[i]
                n_mesh = sqrt(1 - (l_mesh*l_mesh + m_mesh*m_mesh))
                
                u,v,w = get_uvw(X,Y,Z,dec0*D2R,h0*D2R)
                
                image_freq = fdecorr(u=u,v=v,w=w,l=l_mesh,m=m_mesh,n=n_mesh,chan_width=freq_int*1e6,freq=central_frequency,phasetrack=True)
                
                ###EXTENSION SECTION LEAVE COMMENTED-------------------------------------------
                ###If not phase tracking in the simulations, have to fiddle by the difference
                #for freq_step in xrange(len(num_freq_step)):
                    #this_freq = intial_freq + freq_step*freq_reso
                    
                    #image_freq /= fdecorr(u=u,v=v,w=w,l=l_mesh,m=m_mesh,n=n_mesh,chan_width=freq_int*1e6,freq=central_frequency,phasetrack=False)
                    #u, v, w = get_uvw(X,Y,Z,dec0,this_ha0)
                    
                    ###TODO - get nophasetrack defined in l,m,n
                    #image_time *= tdecorr_nophaset
                ###EXTENSION SECTION LEAVE COMMENTED-------------------------------------------
                
                image_freq = 1 / image_freq
                image_kernel *= image_freq
            
            ##Take the image space kernel and FT to get the gridding kernel
            kernel_array = image2kernel(image=image_kernel,cell_reso=resolution,u_off=u_off,v_off=v_off,l_mesh=l_mesh,m_mesh=m_mesh)
            
        else:
            print('YOU ARE HERE YOU FOOL')
            kernel_array = array([[complex(1,0)]])
            
        ##Multiply the kernal by the complex value
        data_kernel = kernel_array * comp
        
        ##Add the multiplied kernel-uvdata values to the grid
        add_kernel(container,u_ind,v_ind,data_kernel)
        
    return container

#def convert_image_lm2uv(image=None,l_reso=None,mode='CPU'):
    #'''Takes an l,m projected all sky and ffts to give a u,v plane from which
    #to (re)grid from'''
    
    #num_pixel = image.shape[0]
    
    #max_u = (0.5 / l_reso)
    #u_reso = (2*max_u) / float(num_pixel)
    #u_offset = -(u_reso / 2.0)

    #if num_pixel % 2 == 0:
        #u_sim = linspace(-max_u-u_offset,max_u+u_offset,num_pixel) - u_reso/2.0
        #v_sim = linspace(-max_u-u_offset,max_u+u_offset,num_pixel) - u_reso/2.0
        
        #shift_image = fft.fftshift(image)
        #if mode == 'CPU':
            #uv_data_array = fft.fft2(shift_image) #/ (image.shape[0] * image.shape[1])

        #elif mode == 'GPU':
            ###GPU CODE-------------------------------------
            #cuda.init()
            #context = make_default_context()
            #stream = cuda.Stream()

            #plan = Plan(shift_image.shape, stream=stream)

            #gpu_data = gpuarray.to_gpu(shift_image.astype(complex64))

            #plan.execute(gpu_data)
            #uv_data_array = gpu_data.get()
            #context.pop()
        ###GPU CODE-------------------------------------
        
        #uv_data_array = fft.ifftshift(uv_data_array)
        
        #u_mesh, v_mesh = meshgrid(u_sim,v_sim)
        ###As an even data array has l,m = 0,0 outside of a pixel, need to shift centre
        ###of the image to account for this. An offset in l,m space is a phase change in u,v
        ###space so apply a phase shift of half a pixel's worth to the uv_data array
        ##uv_data_array *= np_exp(-2j*pi*(u_mesh*(l_reso/2.0) + v_mesh*(l_reso/2.0)))
    
    #else:
        #u_sim = linspace(-max_u-u_offset,max_u+u_offset,num_pixel)
        #v_sim = linspace(-max_u-u_offset,max_u+u_offset,num_pixel)
        
        #shift_image = fft.ifftshift(image)
        #uv_data_array = fft.fft2(shift_image) #/ (image.shape[0] * image.shape[1])
        #uv_data_array = fft.fftshift(uv_data_array)
        
    #return uv_data_array,u_sim,v_sim,u_reso
    
def convert_image_lm2uv(image=None,l_reso=None,mode='CPU',ra_offset=None,dec_offset=None):
    '''Takes an l,m projected all sky and ffts to give a u,v plane from which
    to (re)grid from'''
    num_pixel = image.shape[0]
    
    u_reso = (1/l_reso) / float(num_pixel)
    u_offset = -(u_reso / 2.0)
    
    max_u = (num_pixel*u_reso) / 2.0
    
    if num_pixel % 2 == 0:
        u_sim = linspace(-max_u-u_offset,max_u+u_offset,num_pixel) - u_reso/2.0
        v_sim = linspace(-max_u-u_offset,max_u+u_offset,num_pixel) - u_reso/2.0
        
    else:
        u_sim = linspace(-max_u-u_offset,max_u+u_offset,num_pixel)
        v_sim = linspace(-max_u-u_offset,max_u+u_offset,num_pixel)
    
    shift_image = fft.fftshift(image)
    if mode == 'CPU':
        uv_data_array = fft.fft2(shift_image) #/ (image.shape[0] * image.shape[1])

    elif mode == 'GPU':
        ##GPU CODE-------------------------------------
        cuda.init()
        context = make_default_context()
        stream = cuda.Stream()

        plan = Plan(shift_image.shape, stream=stream)

        gpu_data = gpuarray.to_gpu(shift_image.astype(complex64))

        plan.execute(gpu_data)
        uv_data_array = gpu_data.get()
        context.pop()
    ##GPU CODE-------------------------------------
    
    uv_data_array = fft.ifftshift(uv_data_array)
    
    shift_image = fft.ifftshift(image)
    uv_data_array = fft.fft2(shift_image) #/ (image.shape[0] * image.shape[1])
    uv_data_array = fft.fftshift(uv_data_array)
    
    if ra_offset != None and dec_offset != None:
    
        u_mesh, v_mesh = meshgrid(u_sim,v_sim)
        uv_data_array *= np_exp(-2.0j*pi*(u_mesh*-sin(ra_offset*D2R) + v_mesh*-sin(dec_offset*D2R)))
        
    return uv_data_array,u_sim,v_sim,u_reso

def my_loadtxt(image_loc):
    '''For small arrays, loadtxt is slooooow so made my own'''
    #my_im = loadtxt(image_loc)
    lines = open(image_loc).read().split('\n')
    my_im = array([map(float,line.split()) for line in lines if line != ''])
    my_im.shape = (KERNEL_SIZE,KERNEL_SIZE)
    
    return my_im

def wprojection(w=None,n=None):
    '''When degridding from a plane of l,m data, you have no w term information in
    the FFT. Add this is image space to do a quasi w-projection'''
    
    #print('w-term',np_exp(-2j*pi*(w*(n - 1))))
    #return conjugate(np_exp(-2j*pi*(w*(n - 1)))) 
    return np_exp(2j*pi*(w*(n - 1)))

#@profile
def reverse_grid(uv_data_array=None, l_reso=None, m_reso=None, u=None, v=None, weights=None, kernel='none',kernel_params=None,
                 conjugate=False,central_lst=None,time_decor=False,xyz_lengths=None,phase_centre=None,time_int=None,freq_cent=None,
                 beam_image=None,delay_str="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",u_sim=None,v_sim=None,image=None,u_reso=None,freq_decor=False,
                 freq_int=None,fix_beam=False,image_XX=False,image_YY=False,wproj=True,phase_shift_kernel=True,
                 uv_kernel_XX=None,uv_kernel_YY=None,u_range_kernel=None,v_range_kernel=None,over_sampled=False,over_sampling=None):
    '''A reverse gridded - takes a grid of uv data (must be square!!), and then samples the 
    u,v data at the given u,v coords, using the desired kerrnel'''
    
    sim_complexes = []
    
    ##TODO - get the ra of the source in there
    #l,m,n = get_lm(ra*D2R, phase_ra, dec*D2R, phase_dec)
    
    u_ind,v_ind,u_off,v_off = find_closet_uv(u=u,v=v,u_range=u_sim,v_range=v_sim,resolution=u_reso)
    #print('degrid plane resolution %.2f' %(u_sim[1] - u_sim[0]))
    
    l_extent = 0.5 / u_reso
    l_reso = 2.0 / KERNEL_SIZE
    offset = l_reso / 2.0
    
    l_range = linspace(-l_extent+offset,l_extent-offset,KERNEL_SIZE)
    m_range = linspace(-l_extent+offset,l_extent-offset,KERNEL_SIZE)
    l_mesh, m_mesh = meshgrid(l_range,m_range)
    
    ##Outside l**2 + m**2 = 1 is unphysical, so we get nans
    ##Find a mask based on where the l,m coords are physical
    n_mask = (l_mesh*l_mesh + m_mesh*m_mesh) <= 1
    n_mesh = sqrt(1 - (l_mesh*l_mesh*n_mask + m_mesh*m_mesh*n_mask))
    #n_mesh = nan_to_num(n_mesh)
    
    
    #ra0,dec0 = phase_centre
    #h0 = central_lst - ra0
    #X,Y,Z = xyz_lengths
    
    #u,v,w = get_uvw(X,Y,Z,dec0,h0)
    
    if kernel == 'gaussian':
        kernel_params = [2.0,2.0]
        sig_x,sig_y = kernel_params
        
        ##Calculate the gaussian caused in image space due to desired params of gridding gaussian in u,v
        image_XX = image_gaussian(kernel_sig_x=sig_x,kernel_sig_y=sig_y,l_mesh=l_mesh,m_mesh=m_mesh,cell_reso=u_reso)
        image_YY = image_XX
        
    elif kernel == 'MWA_phase1':
        beam_loc = '%s/telescopes/%s/primary_beam/data' %(MAJICK_DIR,kernel)
        
        if fix_beam:
            ##We have already passed image_XX, image_YY as an argument if using fix_beam
            pass
            ##If using CHIPS in fix beam mode, set to 186.235MHz (+0.02 for half channel width)
        else:
            image_XX = my_loadtxt('%s/beam_%s_%.3f_XX.txt' %(beam_loc,delay_str,freq_cent))
            image_YY = my_loadtxt('%s/beam_%s_%.3f_YY.txt' %(beam_loc,delay_str,freq_cent))
        
    else:
        print("You haven't entered a correct degridding kernel option")
        
    if time_decor:
        ra0,dec0 = phase_centre
        h0 = central_lst - ra0
        X,Y,Z = xyz_lengths
        image_time = (tdecorr_phasetrack(X=X,Y=Y,Z=Z,d0=dec0,h0=h0,l=l_mesh,m=m_mesh,n=n_mesh,time_int=time_int))
        #image_time = (tdecorr_phasetrack(X=X,Y=Y,Z=Z,d0=dec0,h0=15.0*D2R,l=l_mesh,m=m_mesh,n=n_mesh,time_int=time_int))
        ##Apply mask to image_time
        image_time *= n_mask
        
        image_XX *= image_time
        image_YY *= image_time

    if freq_decor:
        ra0,dec0 = phase_centre
        h0 = central_lst - ra0
        X,Y,Z = xyz_lengths
        u,v,w = get_uvw(X,Y,Z,dec0,h0)
        #print('ooosh')
        image_freq = fdecorr(u=u,v=v,w=w,l=l_mesh,m=m_mesh,n=n_mesh,chan_width=freq_int*1e6,freq=freq_cent,phasetrack=True)
        ##Apply mask to image_freq
        image_freq *= n_mask
        
        image_XX *= image_freq
        image_YY *= image_freq
        
    if over_sampled:
        ##Find closest u,v point in the oversample kernel that matches the offset from our grid point
        ##This will become the centre point of our 
        u_ind_ker,v_ind_ker,u_off_ker,v_off_ker = find_closet_uv(u=u_off,v=v_off,u_range=u_range_kernel,v_range=v_range_kernel,resolution=u_range_kernel[1]-u_range_kernel[0])
        
        u_ind_ker = int(u_ind_ker)
        v_ind_ker = int(v_ind_ker)
        over_sampling = int(over_sampling)
        
        ##Do some maths to work out which points about the central
        ##point u_ind_ker,v_ind_ker to sample
        low_samp_u = u_ind_ker % over_sampling
        low_samp_v = v_ind_ker % over_sampling

        if abs(int(uv_kernel_XX.shape[0]/2) - u_ind_ker) == over_sampling:
            low_samp_u = over_sampling - 1
            low_samp_v = over_sampling - 1

        offset_array = arange(-int(KERNEL_SIZE / 2), int(KERNEL_SIZE / 2) + 1) * over_sampling 
        kern_mid = int(uv_kernel_XX.shape[0]/2)

        offset_u = offset_array + (3*kern_mid) / 2 - u_ind_ker / 2
        offset_v = offset_array + (3*kern_mid) / 2 - v_ind_ker / 2

        ##Make a mesh grid of the coords to sample, then apply
        ##to the kernel to get back the KERNEL_SIZE*KERNEL_SIZE kernel
        mesh_u,mesh_v = meshgrid(offset_u.astype(int),offset_v.astype(int))

        kernel_array_XX = uv_kernel_XX[mesh_v,mesh_u]
        kernel_array_YY = uv_kernel_YY[mesh_v,mesh_u]
        
    ##If doing via the fourier shift kernel method
    else:
        #No matter what kernel used, l,m is a flat sky and so w terms are
        #absent. Need to account for this, so generate the w-term effects
        #image space and add to degridding kernel
        if wproj:
            wproj_image = wprojection(w=w,n=n_mesh)
            wproj_image *= n_mask
            
            image_XX = wproj_image * image_XX.astype(complex)
            image_YY = wproj_image * image_YY.astype(complex)
        else:
            pass
        
        kernel_array_XX = image2kernel(image_XX,cell_reso=u_reso,u_off=u_off,v_off=v_off,l_mesh=l_mesh,m_mesh=m_mesh)
        kernel_array_YY = image2kernel(image_YY,cell_reso=u_reso,u_off=u_off,v_off=v_off,l_mesh=l_mesh,m_mesh=m_mesh)
    
    ##Grab the u,v data point that you want using the given kernel
    sim_complex_XX = apply_kernel(uv_data_array=uv_data_array,u_ind=u_ind,v_ind=v_ind,kernel=kernel_array_XX)
    sim_complex_YY = apply_kernel(uv_data_array=uv_data_array,u_ind=u_ind,v_ind=v_ind,kernel=kernel_array_YY)
    
    return sim_complex_XX,sim_complex_YY,image_XX,image_YY