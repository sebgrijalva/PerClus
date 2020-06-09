# Binder method to estimate critical level
import numpy as np
from numba import njit
import time, hoshen, randomsurf
import matplotlib.pyplot as plt

@njit
def stats(surface):
    '''calculates statistics of masses of clusters'''
    m4 = 0
    m2 = 0
    masses = hoshen.get_masses(surface)
    for x in masses:
        m4 += x**4
        m2 += x**2
    return m4,m2

sizes = [32,64,128,256]
#sizes = [24,48,96,192]
parameters = np.linspace(-0.24,-0.217,8)
samples = 2*10**3
alpha = 0.25

start = time.time()
ax = plt.axes()

print(parameters)
for L in sizes:
    ker,ker_norm = randomsurf.kernel(L,alpha)
    data = []
    error = []

    for i,h in enumerate(parameters):
        print(f'Binder Cumulants: Size {L}, measure {i+1} of {len(parameters)}...')
        m2 = np.zeros(samples)
        m4 = np.zeros(samples)
        for s in range(samples):
            surface = 1 * (randomsurf.gaussian_field(L,alpha,ker,ker_norm) > h)
            #surface = randomsurf.percolation(L,p=h)
            m4[s],m2[s] = stats(surface)

        mean_m2 = np.mean(m2)
        mean_m4 = np.mean(m4)
        std_m2 = np.std(m2)
        std_m4 = np.std(m4)
        data.append(mean_m4/mean_m2**2)
        error.append( (std_m4/mean_m2**2 + 2*mean_m4*std_m2/mean_m2**3)/np.sqrt(samples) )

    #np.savetxt(f'data/binder/L={L}-alpha={alpha}-s={samples}.txt',(parameters,np.round(data, 5),np.round(error, 5)), fmt='%1.5f,')
    # plot results

    ax.set_title(f'Alpha = {alpha}')
    ax.errorbar(parameters,data,yerr=error, capsize=2, ls ='solid', marker='.')
    ax.set_xlabel("level h")
    ax.set_ylabel("Binder cumulant B(h)")

    print(data,error)
print(f'Done! Elapsed time: {time.time() - start}')


plt.show()


### Binder Method - Rectangular Arrays ###
##########################################
'''
import time

sizes = [4,8,16,32,64,128]
#parameters = np.linspace(.57,.6,8)
parameters = np.linspace(-.25,-.15,10)
samples = 200
alpha = 0.25

factor = 2
start = time.time()
for L in sizes:

    data = []
    error = []
    sp_den = rectangular_spden(L,factor*L,alpha)

    i = 1 #just an index for displaying progresss
    for h in parameters:
        print(f'Binder Cumulants: Size {factor*L}x{L}, measure {i} of {len(parameters)}...')
        m2 = np.zeros(samples)
        m4 = np.zeros(samples)
        for s in range(samples):
            surface = (rectangular_gaussian_field(L,factor*L,alpha,sp_den) > h)*1
            #surface = rectangular_percolation(factor*L, L, p = h)
            clusters = hoshen.rectangular_get_clusters( surface )
            masses = np.unique(clusters, return_counts=True)[1][1:]/(L*factor*L)
            m4[s] = np.sum([x**4 for x in masses])
            m2[s] = np.sum([x**2 for x in masses])
        mean_m2 = np.mean(m2)
        mean_m4 = np.mean(m4)
        std_m2 = np.std(m2)
        std_m4 = np.std(m4)
        data.append(mean_m4/mean_m2**2)
        error.append( (std_m4/mean_m2**2 + 2*mean_m4*std_m2/mean_m2**3)/np.sqrt(samples) )
        i += 1
    #np.savetxt(f'data/binder/(rectangular)-MxN={factor}-L={L}-alpha={alpha}-s={samples}.txt',(parameters,np.round(data, 5),np.round(error, 5)), fmt='%1.5f,')
    #np.savetxt(f'data/binder/(rectangular)-MxN={factor}-L={L}-(percolation)-s={samples}.txt',(parameters,np.round(data, 5),np.round(error, 5)), fmt='%1.5f,')
    plt.plot(parameters,data,'o',ls='--')
print(f'Done! Elapsed time: {time.time() - start}')
#print(data)

plt.show()
'''
