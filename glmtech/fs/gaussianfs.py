import mpmath as mp
# from functools import cache

from .fsframe import FSFrame

"""FSFrame for zeroth-order gaussian beams."""

class GaussianFSFrame(FSFrame):
    def __init__(self, k, w0):
        self.k = k,
        self.w0 = w0
        self.s = 1 / k / w0
    
    def xi(self, n, mu):
        if (n - mu) % 2 == 1 or n < mu: return 0
        return mp.power(-self.s ** 2, (n - mu) / 2) / mp.factorial((n - mu) / 2)
    
    def tm_maclaurin_even(self, n, m):
        return mp.pi * self.xi(n, 1)

    def tm_maclaurin_odd(self, n, m):
        s = self.s
        return 1j * mp.pi * (
            -2 * s ** 4 * self.xi(n, 4) \
            + (4 * s ** 2 - 1) * self.xi(n, 2)
        )
    
    def tm_maclaurin(self, n, m):
        if m not in [-1, 1]: return 0
        if (n - m) % 2 == 0: return self.tm_maclaurin_even(n, m)
        if (n - m) % 2 == 1: return self.tm_maclaurin_odd(n, m)
    
    def te_maclaurin(self, n, m):
        return -m * 1j * self.tm_maclaurin(n, m)
