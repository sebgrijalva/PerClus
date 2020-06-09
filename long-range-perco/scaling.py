
import numpy as np
from numba import njit
import time
import matplotlib.pyplot as plt

import randomsurf, hoshen

#Test to see scaling of correlations in fractional gaussian surface:
'''
L = 2**10
alphas = [0.5,1., 1.5]
samples = 200

#radii = np.array([0] + [2**i for i in range(20)] + [3*2**i for i in range(20)] )
#radii = np.sort(radii[radii <= L//2])

radii = np.sort(
                np.concatenate(
                    (np.linspace(10, L//2-1, 30, dtype=np.int16),
                    np.arange(1,10))
                    )
                )
print(radii)

@njit
def measure_field(field, radii):
    size = field.shape[0]
    ans = np.zeros(len(radii))
    for x in range(size//2):
        for y in range(size//2):
            ans += np.array([(field[y,x]*field[y,x+r]) for r in radii] )
    return ans/(size*size/4)

corrs = np.zeros((samples, len(radii)))

for alpha in alphas:
    print(f'alpha = {alpha}...')
    ker = kernel(L, alpha)
    for s in range(samples):
        corrs[s,:] = measure_field(gaussian_field(L,alpha,ker),radii)

    result = [np.mean(corrs[:,i]) for i in range(len(radii))]
    error = [2*np.std(corrs[:,i])/np.sqrt(samples) for i in range(len(radii))]

    plt.figure()

    # plot results
    ax = plt.axes()
    ax.set_title(f'Alpha = {alpha}')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.errorbar(radii,result,yerr=error, capsize=2, ls ='', marker='.')
    ax.set_xlabel("radii")
    ax.set_ylabel("correlation")

    # add scaling line:
    r = np.linspace(0.9,L//2,400)
    f = result[0]*radii[0]**(2-alpha)*r**(alpha-2)
    plt.loglog(r, f,'k')

    #np.savetxt(f'data/corr/L={L}-alpha={alpha}-s={samples}.txt',(radii,np.round(result, 5),np.round(error, 5)), fmt='%1.5f')

plt.show()
'''
################################################################################
# Probability of active points in the level sets is long-range correlated

L = 2**10
alphas = [0.5,1.0,1.5]
crit_level = [-0.2136,-0.18409,-0.12229]
samples = 200

radii = np.sort(
                np.concatenate(
                    (np.linspace(10, L//2-1, 30, dtype=np.int16),
                    np.arange(1,10))
                    )
                )
print(radii)

@njit
def measure_field(field, radii):
    size = field.shape[0]
    ans = np.zeros(len(radii))
    for x in range(size//2):
        for y in range(size//2):
            ans += np.array([field[y,x]*field[y,x+r] for r in radii] )
    return ans/(size*size/4)

corrs = np.zeros((samples, len(radii)))

for i,alpha in enumerate(alphas):
    ker = randomsurf.kernel(L, alpha)
    for s in range(samples):
        surf = 1 * (randomsurf.gaussian_field(L,alpha,ker) > crit_level[i])
        corrs[s,:] = measure_field(surf,radii) - ( np.sum(surf)/(L**2) )**2

    result = [np.mean(corrs[:,i])  for i in range(len(radii))]
    error = [np.std(corrs[:,i])/np.sqrt(samples) for i in range(len(radii))]

    plt.figure()
    # plot results
    ax = plt.axes()
    ax.set_title(f'Alpha = {alpha}')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.errorbar(radii,result,yerr=error, capsize=2, ls ='', marker='.', color='Green')
    ax.set_xlabel("radii")
    ax.set_ylabel("correlation")

    # add scaling line:
    r = np.linspace(0.9,L//2,400)
    f = result[0]*radii[0]**(2-alpha)*r**(alpha-2)
    plt.loglog(r, f,'k')

    #np.savetxt(f'data/corr/L={L}-alpha={alpha}-s={samples}.txt',(radii,np.round(result, 5),np.round(error, 5)), fmt='%1.5f')

plt.show()
