import numpy as np


class Percolation:
    def __init__(self, M, N, H=-1):
        self.M = M
        self.N = N
        self.H = H

    def _build_kernel(M, N, H=-1):
        """Build kernel in momentum space"""
        ker = np.zeros((M, N))
        for k1 in range(-M//2, M//2):
            for k2 in range(-N//2, N//2):
                ker[k1, k2] = np.abs(
                            2*np.cos(2*np.pi*k1/M)+2 * np.cos(2*np.pi*k2/N) - 4
                            )
        ker[0, 0] = 1
        final_ker = ker**(-(H+1))
        return np.sqrt(final_ker), (M*N/np.sum(final_ker))**0.5


class Bernoulli(Percolation):
    """Plain Percolation generation of surfaces"""
    def __init__(self, M, N, p=0.59274):
        super.__init__(M, N)
        self.sample = np.random.binomial(n=1, p=p, size=(M, N))


class Uniform(Percolation):
    def __init__(self, M, N):
        """Uniform Distribution"""
        # real, distributed with mean 0 and std 1
        self.real = np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3),
                                      size=(M, N))
        self.fourier = np.fft.fft2(self.real)

        # Add correlations by Fourier Filtering Method:
        self.kernel, self.norm = self._build_kernel(M, N, H=-1)
        self.convolution = self.fourier*np.sqrt(self.kernel)

        # Take IFFT and exclude residual complex part
        self.correlated_noise = np.fft.ifft2(self.convolution).real

        # Return normalized field
        self.sample = self.correlated_noise * self.norm
