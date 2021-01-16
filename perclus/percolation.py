import numpy as np

def percolation(M, N, p=0.59274):
    """Plain Percolation generation of surfaces"""
    surface = np.random.random((M, N))
    return int(surface < p)
