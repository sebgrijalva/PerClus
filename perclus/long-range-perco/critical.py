from numba import njit
import numpy as np
import random, math, time

import randomsurf, hoshen
#from CorrelatedPercolation_square import*
#random.seed(-2)

from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

@njit
def find_critical(surface):
    '''dichotomy search of critical point of surface'''
    h = 0 # Initial guess level
    dh = 0.2 # Search Step

    while dh > 0.0001:
        surf = 1 * (surface > h)
        clusters = hoshen.get_clusters(surf, open=True) #get clusters (with OBC!)

        intersection = False
        for x in clusters[0]:
            for y in clusters[-1]:
                if x == y and x:
                    intersection = True
                    break
            if intersection: break

        if intersection:
            h += dh
            dh = dh / 10 # decrease step
        else:
            h -= dh

    return h

alphas = [0.25,0.5]
sizes=[8,12,16,24,32,48,64,96,128]
sample_set = [10**7,10**6,10**6,10**5,10**5,10**5,10**5,10**4,10**4]

start = time.time()
for i,alpha in enumerate(alphas):
    print(f'Alpha = {alpha}')
    Levels = []
    Err = []

    for j,L in enumerate(sizes):
        print(f'Currently on size {L}')
        samples = sample_set[j]
        ker = randomsurf.kernel(L, alpha)

        levels = []
        for _ in range(samples):
            rsurf = randomsurf.gaussian_field(L, alpha, ker)
            crit_level = find_critical(rsurf)
            levels.append(crit_level)

        Levels.append(np.mean(levels))
        Err.append(np.std(levels)/np.sqrt(samples))
        print(Levels[-1],Err[-1])

    print(Levels)
    print(Err)

    print(f'elapsed time = {time.time() - start}')
    plt.errorbar(
        x = sizes,
        y = Levels,
        yerr = Err,
        capsize = 2,
        ls = ':',
        marker = 'o')

plt.show()


# Rectangular Lattice:
'''
for r in range(1): # several runs
    print(f'Run {r+1}...')

    for alpha in alphas:
        print(alpha)
        Levels = []
        Err = []
        i = 0

        for L in sizes:
            print(f'Currently on size {L}')
            samples = sample_space[i]
            sp_den = randomsurf.rectangular_spden(L, L+2, alpha) #add two in v
            levels = []

            for _ in range(samples):
                rsurf = randomsurf.rectangular_gaussian_field(L, L+2, alpha, sp_den)
                h = 0 # Initial guess level
                dh = 0.1 # Search Step

                while dh > 0.0001:

                    #print(f'level = {h}')
                    surf = 1 * (rsurf > h)
                    clusters = hoshen.rectangular_get_clusters(surf, open = True) #get clusters with OBC

                    if sum(set(clusters[0]) & set(clusters[-1])) > 0: # Vertical wrapping
                        dh = dh / 2 # decrease step
                        h += dh
                    else:
                        h -= dh

                    #plt.matshow(clusters)
                    #plt.show()

                levels.append(h)

            Levels.append(np.mean(levels))
            Err.append(np.std(levels)/np.sqrt(samples))
            print(Levels[-1],Err[-1])
        i += 1

        print(Levels)
        print(Err)

        plt.errorbar(
            x = sizes,
            y = Levels,
            yerr = Err,
            capsize = 2,
            marker = '.')
        #plt.plot(np.arange(40,120),-0.23461-2.5*np.arange(40,120)**(-0.75))
plt.show()
'''
