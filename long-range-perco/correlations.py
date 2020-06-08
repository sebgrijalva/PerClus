import numpy as np
import time
import matplotlib.pyplot as plt

from numba import jit,prange

########################################"

import hoshen
import randomsurf


L = 1024
alpha = 0.4

samples = 100000
# Make surface:
SD = randomsurf.build_spden(L, alpha)
rg = 20
twopoint = np.zeros((samples,2,rg))
avg = np.zeros(samples)
for s in range(samples):
    surface = randomsurf.uniform_field(L, alpha, SD)
    twopoint[s][0] = [surface[0,0]*surface[0,r] for r in range(rg)]
    twopoint[s][1] = [surface[0,r] for r in range(rg)]
    avg[s] = surface[0,0]

results = [np.mean(twopoint[:,0,r])-np.mean(twopoint[:,1,r])*np.mean(avg) for r in range(rg)]
errors = [np.std(twopoint[:,0,r])/np.sqrt(samples) for r in range(rg)]
plt.errorbar(x = range(rg), y = results, yerr = errors, marker = 'o', mec = 'k', ls = '--', color = 'g', capsize = 2)
plt.show()
