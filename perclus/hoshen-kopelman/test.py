import hoshen
import numpy as np
import matplotlib.pyplot as plt
import time


np.random.seed(4)

L = 2**13

A = np.random.randint(0,5,(L,L))
#plt.matshow(A)

for _  in range(10): # 10 runs
    start = time.time()
    A = hoshen.get_clusters(A)
    print(f'elapsed time = {time.time() - start}')

#plt.matshow(A)

plt.show()
