import numpy as np
import time
import matplotlib.pyplot as plt

from numba import jit,prange

########################################"

import hoshen
import randomsurf


#############################################

def max_cluster(cluster_array):
    '''Returns the size of the largest cluster'''
    return np.max(np.unique(cluster_array,return_counts=True)[1][1:])

def fractal_dim_measurement(L, alpha, height,sp_den):
    '''Gather here all the calculations of the fractal dimension of a surface'''
    surface = (randomsurf.gaussian_field(L,alpha,sp_den) > height)*1
    clusters = hoshen.get_clusters(surface)
    return max_cluster(clusters)



###########################################


alpha = 2/3.
heights = [-0.2035]
samples = 20000
sizes = np.linspace(6,200,140, dtype = np.int16)

for height in heights:

    largest_cluster = []
    errors_largest_cluster = []

    start=time.time()
    for L in sizes:
        print(f'Fractal Dimension: Currently on size {L}')
        sp_den = randomsurf.build_spden(L,alpha)
        fractal_dim_samples = [fractal_dim_measurement(L, alpha, height, sp_den) for _ in range(samples)]
        largest_cluster.append(np.mean(fractal_dim_samples))
        errors_largest_cluster.append(np.std(fractal_dim_samples)/np.sqrt(samples))

    print(f'Elapsed T: {time.time()-start}')

    plt.loglog(sizes,largest_cluster,'.')
    plt.show()
    #np.set_printoptions(suppress=True) # No scientific notation
    np.savetxt(
        f'data/fractal_dim/alpha={alpha}-s={samples}-h={height}.txt',
        (sizes, largest_cluster, errors_largest_cluster),
        delimiter = ',', fmt = '%-8g' # left justify, 8 chars max, general type
        )
