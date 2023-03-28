# Recursive algorithm for computing the BSCs from paraxial LG profiles

import mpmath as mp
import numpy as np
from functools import lru_cache

from .fsframe import FSFrame

pi = mp.pi

class RecLGFSFrame(FSFrame):
    def __init__(self, k, w0, wavelength=None, p=0, l=0):
        if wavelength is not None:
            self.wavelength = wavelength
            self.wavenumber = 2 * pi / wavelength
        elif k is not None:
            self.wavenumber = k
            self.wavelength = 2 * pi / k
        self.s = 1 / k / w0
        self.p, self.l = p, l
    
    @property
    def k(self):
        return self.wavenumber
    
    def G(self, l=None):
        if l is None: l = self.l
        s = self.s

        return 1j * s / mp.sqrt(2 * pi) * mp.power(s * mp.sqrt(2), l)
    
    def E(self, n, u):
        l, s = self.l, self.s
        nlu2 = (n - l - u) / 2

        return (mp.power(-s ** 2, nlu2) / mp.factorial(nlu2))
    
    # @np.vectorize
    def F_func(self, n, a):
        if n >= 0 and n <= (self.l + 2 * a):
            return 0

        return self.G() * self.E(n, 1 + 2 * a)
    
    def F_vec(self, n, u):
        us = np.arange(u + 1)
        F = np.vectorize(self.F_func)
        return F(n, us)

    @lru_cache(maxsize=None)
    def a_coeffs(self, u):
        if u == -1:
            return np.array([])
        if u == 0:
            return np.array([mp.mpf(1)])
        
        A = (2 * u - 1 + self.l) / u
        B = -(u - 1 + self.l) / u
        C = -2 * self.s ** 2 / u

        aum1 = self.a_coeffs(u - 1)
        aum2 = self.a_coeffs(u - 2)
        aa = np.append(aum1, 0)
        ab = np.append(aum2, [0, 0])
        ac = np.append(0, aum1)

        return A * aa + B * ab + C * ac

    def tm_maclaurin_even(self, n):
        return np.dot(self.F_vec(n, self.p), self.a_coeffs(self.p))

    def tm_maclaurin_odd(self, n):
        pass

    @lru_cache(maxsize=None)
    def tm_maclaurin(self, n, m):
        if (n - m) % 2 == 0:
            return self.tm_maclaurin_even(n)
        else:   
            return self.tm_maclaurin_odd(n)
    
    def te_maclaurin(self, n, m):
        pass

    def normalization_constant(self):
        p, l, k = self.p, self.l, self.k
        d = 2 if l == 0 else 1
        return k * mp.sqrt(2 * mp.gammaprod([p + 1], [p + l + 1]) / pi / d)
    
    def bsc(self, n, m, mode="tm"):
        if abs(m) > n or m not in (self.l + 1, self.l - 1): return 0
        norm = self.normalization_constant()
        if mode.lower() == "te":
            pm = m - self.l
            return -1j * norm * pm * super().bsc(n, m, mode="tm")
        
        return norm * super().bsc(n, m, mode=mode)