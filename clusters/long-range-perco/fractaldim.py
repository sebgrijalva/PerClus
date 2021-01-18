import numpy as np
import time
import matplotlib.pyplot as plt

from numba import jit,prange

########################################"

import hoshen
import randomsurf


#############################################

def fractal_dim_measurement(L, alpha, height,ker,norm):
    '''Gather here all the calculations of the fractal dimension of a surface'''
    surface = 1 * (randomsurf.gaussian_field(L,alpha,ker,norm) > height)
    masses = hoshen.get_masses(surface)
    return np.max(masses)*L**2



###########################################


alphas = [0.,0.25]
heights = [-0.23461,-0.224]
samples = 4000
sizes = np.arange(6,200, dtype = np.int16)

ax = plt.axes()
#ax.set_title(f'Largest Clusters')
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("L")
ax.set_ylabel("Largest Cluster")

print(list(sizes))

for i, alpha in enumerate(alphas):

    print(f'alpha = {alpha}')

    largest_cluster = []
    error = []

    start=time.time()
    for L in sizes:
        #print(f'Fractal Dimension: Currently on size {L}/{sizes.max()}')
        ker, norm = randomsurf.kernel(L,alpha)
        fractal_dim_samples = [fractal_dim_measurement(L, alpha, heights[i], ker, norm) for _ in range(samples)]
        largest_cluster.append(np.mean(fractal_dim_samples))
        error.append(np.std(fractal_dim_samples)/np.sqrt(samples))

    print(f'Elapsed T: {time.time()-start}')

    ax.errorbar(sizes,largest_cluster,yerr=error, capsize=2, ls ='', marker='.', label=f'H = {(alpha-2)/2}')
    ax.legend()

    #np.set_printoptions(suppress=True) # No scientific notation
    '''
    np.savetxt(
        f'data/fractal_dim/alpha={alpha}-s={samples}-h={height}.txt',
        (sizes, largest_cluster, errors_largest_cluster),
        delimiter = ',', fmt = '%-8g' # left justify, 8 chars max, general type
        )
    '''
    print(largest_cluster,error)

plt.show()
