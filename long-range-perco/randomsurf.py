################# GENERATING RANDOM SURFACES USING FFT #######################

import numpy as np
from numba import jit,njit
#from scipy import special
import hoshen
import matplotlib.pyplot as plt

@njit
def berry(size):
    N = 1000
    spectrum=np.zeros((size,size))
    for w in prange(N):
        #choose a random direction for the wave vector:
        kx = np.random.random()*2-1
        ky = np.sqrt(1-kx**2)*(np.random.randint(2)*2-1) #last part gives random sign
        #choose a gaussian amplitude and phase
        amp = np.random.normal(0,1)
        phase = np.random.random()*2*np.pi
        #add contribution of chosen plane wave:
        for i in range(size):
            for j in range(size):
                spectrum[i,j] += amp * np.cos( kx*i + ky*j + phase) / np.sqrt(N)
    return 1*(spectrum > 0)

@njit
def kernel(L, alpha):
    #computes the spectral density in momentum space
    ker = np.zeros((L,L))
    for k1 in range(-L//2, L//2):
        for k2 in range(-L//2, L//2):
            ker[k1,k2] = np.abs(2*np.cos(2*np.pi*k1/L)+2*np.cos(2*np.pi*k2/L)-4)
    ker[0,0]=1
    final_ker = ker**(-alpha/2)
    return np.sqrt(final_ker), L/np.sqrt(np.sum(final_ker))

@njit
def kernel_smallk(size, alpha):
    #computes the spectral density in momentum space
    sp_den = np.zeros((size,size))
    for k1 in prange(-size//2,size//2):
        for k2 in prange(-size//2,size//2):
            sp_den[k1,k2] = np.sqrt(k1**2 + k2**2)
    sp_den[0,0]=1
    return 1/sp_den**(alpha)

def gaussian_field(L,alpha,kernel,ker_norm):
    '''Builds a correlated gaussian field on a surface LxL'''

    # FFT of gaussian noise:
    noise_real = np.random.normal(0, 1, size = (L, L))
    noise_fourier = np.fft.fft2(noise_real)

    # Add correlations by Fourier Filtering Method:
    convolution = noise_fourier * kernel

    # Take IFFT and exclude residual complex part
    correlated_noise = np.fft.ifft2(convolution).real

    # Return normalized field
    return correlated_noise * ker_norm
################################################################################

def uniform_field(L, alpha, spectral_density):
    '''Build a correlated field from a uniform noise on a surface LxL'''
    # FFT of uniform noise
    noise_real = np.random.uniform(low = -np.sqrt(3), high = np.sqrt(3), size=(L,L)) #real, distributed with mean 0 and std 1
    noise_fourier = np.fft.fft2(noise_real)

    # Add correlations by Fourier Filtering Method:
    convolution = noise_fourier*np.sqrt(spectral_density)

    # Take IFFT and exclude residual complex part
    correlated_noise = np.fft.ifft2(convolution).real

    # Return normalized field
    return correlated_noise * (L/np.sqrt(np.sum(spectral_density)) )

def percolation(L, p = 0.59274):
    '''Plain Percolation generation of surfaces'''
    surface = np.random.random((L,L))
    return (surface < p)*1

###### Rectangular Surfaces ###########
def rectangular_percolation(M, N, p = 0.59274):
    '''Plain Percolation generation of surfaces'''
    surface = np.random.random((N,M))
    return (surface < p)*1


@njit
def rectangular_kernel(M,N, alpha):
    '''
    computes the spectral density in momentum space:
    M : Vertical
    N : Horizontal
    '''
    sp_den = np.zeros((M,N))
    for k1 in prange(-M//2, M//2):
        for k2 in prange(-N//2, N//2):
            sp_den[k1,k2] = np.abs(2*(np.cos(2*np.pi*k1/M)+np.cos(2*np.pi*k2/N)-2))
    sp_den[0,0]=1
    return 1/sp_den**(alpha/2)


def rectangular_uniform_field(M, N, alpha, spectral_density):
    '''Build a correlated field from a uniform noise on a surface MxN'''
    # FFT of uniform noise
    noise_real = np.random.uniform(low = -np.sqrt(3), high = np.sqrt(3), size=(M,N)) #real, distributed with mean 0 and std 1
    noise_fourier = np.fft.fft2(noise_real)

    # Add correlations by Fourier Filtering Method:
    convolution = noise_fourier*np.sqrt(spectral_density)

    # Take IFFT and exclude residual complex part
    correlated_noise = np.fft.ifft2(convolution).real

    # Return normalized field
    return correlated_noise * (np.sqrt(M*N/np.sum(spectral_density)) )

def rectangular_gaussian_field(M, N,alpha,spectral_density):
    '''Builds a correlated gaussian field on a surface LxL'''

    # FFT of gaussian noise:
    noise_real = np.random.normal(0, 1, size = (M, N))
    noise_fourier = np.fft.fft2(noise_real)

    # Add correlations by Fourier Filtering Method:
    convolution = noise_fourier*np.sqrt(spectral_density)

    # Take IFFT and exclude residual complex part
    correlated_noise = np.fft.ifft2(convolution).real

    # Return normalized field
    return correlated_noise * (np.sqrt(M*N/np.sum(spectral_density)) )

###############################################################################
###############################################################################



########################################
##Check divergence of correlation length:
'''
#According to Ziff

alpha = 0.0
samples = 20

parameters = np.linspace(-2,0.5,30)
max_mass = np.zeros(len(parameters))

for L in [2**k for k in range(5,8)]:
    print(f'Building at size {L}...')
    spectral_density = build_spden(L,alpha)
    for _ in range(samples):
        correlated_surface = gaussian_field(L,alpha,spectral_density)
        i = 0
        for h in parameters:
            level_set = (correlated_surface>h)*1
            clusters = hoshen.get_clusters(level_set)
            counts = np.unique(clusters, return_counts=True)
            max_mass[i] += np.mean(counts[1][1:])/(samples)
            i += 1

    plt.plot(parameters,max_mass,'.',ls=":")
plt.show()
'''
#According to Hoshen
'''
def second_moment(masses):
    return np.sum([x**2 for x in masses])

@jit(nopython=True)
def rad_gyration(cluster_numbers):
    L = len(clusters)
    sum = 0
    for n in cluster_numbers: #i.e. for each cluster
        for i1 in range(L):
            for j1 in range(L):
                if clusters[i1,j1] == n:
                    for i2 in range(L):
                        for j2 in range(L):
                            if clusters[i2,j2] == n:
                                sum += ( (i1-i2)**2 + (j1-j2)**2 )/L**2
    return sum

@jit(nopython=True)
def rad_gyration2(clusters,masses):
    L = len(clusters)
    sum = 0
    t = 0
    for n in cluster_numbers: #i.e. for each cluster
        coordinates = np.where(clusters == n)
        m1x = np.sum(coordinates[0])
        m1y = np.sum(coordinates[1])
        m2x = np.sum(coordinates[0]**2)
        m2y = np.sum(coordinates[1]**2)
        sum += 2*(masses[t]*(m2x + m2y) - m1x**2 - m1y**2)
        t += 1
    return sum


sizes = [128,256,512]
alpha = 0.0

grid = 50
levels = np.linspace(-0.5,0.2,grid)
samples = 20

for L in sizes:
    print(f'Currently calculating {L}')
    corr_length = np.zeros(grid)
    sp_den = build_spden(L,alpha)
    i = 0
    for h in levels:
        for _ in range(samples):
            surface = (gaussian_field(L,alpha,sp_den) > h)*1
            clusters = hoshen.get_clusters( surface )

            counts = np.unique(clusters, return_counts=True)
            cluster_numbers = counts[0][1:]
            masses = counts[1][1:]
            #print(clusters)
            #print(cluster_numbers,masses)
            corr_length[i] += (rad_gyration2(clusters,masses)/second_moment(masses))/samples
            #corr_length[i] += second_moment(masses)/samples
        i += 1

    plt.plot(levels,corr_length,'.')

plt.show()
'''

##############################################################################
# Ziff-Binder Method!



####################################################

### Check Scaling of Occupation for alpha < 0.5 ###
'''
import time

sizes = np.sort([2**i for i in range(4,9)]+[3*2**i for i in range(2,8)])

samples = 2000
parameters = [[0.0,-0.2346], [0.25,-0.2240], [0.4,-0.2174]]

start = time.time()

for param in parameters:
    alpha = param[0]
    level = param[1]
    data = []
    for L in sizes:
        mass = 0
        sp_den = build_spden(L,alpha)
        print(f'Occupations: Alpha {alpha}, Size {L}...')
        for s in range(samples):
            surface = (gaussian_field(L,alpha,sp_den) > level)*1
            mass += np.sum(surface)/(samples*L**2)
        data.append(mass)
    plt.plot(sizes,data,'o',ls='--')
    print(f'Done! Elapsed time: {time.time() - start}')
plt.show()
'''


############### Variance Tests #################
'''
import time
from scipy import stats


samples = 10000
M = 24
N = 8*M
alpha = 0.0

sp_den = rectangular_spden(M,N,alpha)
data = [rectangular_gaussian_field(M,N,alpha,sp_den)[2,3] for _ in range(samples)]

print(stats.normaltest(data)) #if p value is smaller than ~0.05, the data is NOT normal
print(f'Mean = {np.mean(data)}, Variance = {np.var(data)}')#get the variance of the set

#end = time.time()
#print(f'Time = {round(end-start,3)} secs')

#plt.plot(range(samples), data, '.')
#plt.matshow(surface)
#plt.show()

'''
##################################


###Basic Tests##########

# Show surfaces with changing alpha
'''
M = 64
N = 2*M
alphas = [ 0.75]
levels = [ -0.19913]
result=[]

for i, alpha in enumerate(alphas):
    print(alpha, levels[i])
    spectral_density = rectangular_spden(M,N,alpha)
    #result = rectangular_gaussian_field(M,N,alpha,spectral_density)
    rfield = rectangular_gaussian_field(M,N,alpha,spectral_density)
    plt.matshow(rfield, cmap='bone')
    plt.axis('off')
    plt.savefig(f'pics/alpha={alpha}.pdf',bbox_inches = 'tight',
    pad_inches = 0)
    plt.colorbar()


    surf = 1*(rfield > levels[i])
    plt.matshow(surf, cmap='Greys_r')
    plt.axis('off')
    plt.savefig(f'pics/level_set-alpha={alpha}.pdf',bbox_inches = 'tight',
    pad_inches = 0)

    #result = berry(500)
    result = hoshen.rectangular_get_clusters(surf)
    plt.matshow(result, cmap='cividis')
    plt.axis('off')
    plt.savefig(f'pics/cluster-alpha={alpha}.pdf',bbox_inches = 'tight',
    pad_inches = 0)

plt.show()


'''

######## CORRELATION-LENGTH EXPONENT ########
'''
from scipy.interpolate import InterpolatedUnivariateSpline

sizes = [8,16,32,64,128]

parameters = np.linspace(-.3,-.2,10)
samples = 100
alpha = 0.25

results=[]

print(parameters)
for L in sizes:
    print(L)
    data = []
    for h in parameters:
        m2 = 0
        m4 = 0
        for _ in range(samples):
            surface = (gaussian(L,alpha) > h).astype(int)
            clusters = hoshen.get_clusters( surface )
            masses = np.unique(clusters, return_counts=True)[1][1:]/L**2
            m4 += sum([x**4 for x in masses])/samples
            m2 += sum([x**2 for x in masses])/samples
        data.append(m4/m2**2)
    spline_fit = InterpolatedUnivariateSpline(parameters, data, k=3)
    plt.plot(parameters, data, linestyle='-', marker='o')
    xs = np.linspace(-.3, -.2, 1000)
    plt.plot(xs, spline_fit(xs), lw=3)
    #derivative = spline_fit.derivative()
    #results.append(derivative(-.23461)) #evaluate at critical level
    np.savetxt(f'critical_h_size_{L}_alpha_{alpha}.txt',(sizes,np.round(data, 5)), fmt='%1.5f,')

plt.show()
'''
