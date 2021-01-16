import numpy as np


class Percolation:
    """Plain Percolation generation of surfaces"""
    def __init__(self, M, N, p=0.59274):
        self.surface = int(np.random.random((M, N)) < p)
