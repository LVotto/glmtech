import mpmath as mp
# from functools import cache

from .fsframe import FSFrame

pi = mp.pi

"""FSFrame for freely-propagating Laguerre-Gaussian beams."""

class FreeLGFSFrame(FSFrame):
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
    
    def beta(self, n, mu):
        s = self.s
        if n >= mu and (n - mu) % 2 == 0:
            return (1j * s) ** (n - mu) / mp.factorial((n - mu) / 2)
        return 0
    
    def laguerre_coeff(self, j, p=None, l=None):
        if p is None: p = self.p
        if l is None: l = self.l
        s = self.s
        return mp.power((-2 * s ** 2), j) / mp.factorial(j) \
            * mp.binomial(p + l, p - j)
    
    def tm_maclaurin_even(self, n):
        s, l, p = self.s, self.l, self.p
        result = 0
        for j in range(0, p + 1):
            result += self.laguerre_coeff(j, p=p, l=l) \
                * self.beta(n, 2 * j + l + 1)
        return pi * 2 ** (l / 2) * s ** (l + 1) * result
    
    def tm_maclaurin_odd_a(self, n):
        s, l, p = self.s, self.l, self.p
        result = 0
        for j in range(0, p + 1):
            result += self.laguerre_coeff(j, p=p, l=l) \
                * self.beta(n, 2 * j + l + 2)
        return 1j * pi * 2 ** (l / 2) * s ** (l + 1) \
            * (2 * s ** 2 * (2 * p + l + 1) - 1) * result
    
    def tm_maclaurin_odd_b(self, n):
        s, l, p = self.s, self.l, self.p
        result = 0
        for j in range(0, p + 1):
            result += self.laguerre_coeff(j, p=p, l=l) \
                * self.beta(n, 2 * j + l + 4)
        return -pi * 2 ** (l / 2 + 1) * s ** (l + 5) * result
    
    def tm_maclaurin(self, n, m):
        if m in (self.l + 1, self.l - 1):
            if (n - m) % 2 == 0:
                return self.tm_maclaurin_even(n)
            else:
                return self.tm_maclaurin_odd_a(n) + self.tm_maclaurin_odd_b(n)
        return 0
    
    def te_maclaurin(self, *args, **kwargs):
        pass
    
    def normalization_constant(self):
        p, l, k = self.p, self.l, self.k
        d = 2 if l == 0 else 1
        return 2 * k * mp.sqrt(mp.gammaprod([p + 1], [p + l + 1]) / pi / d)
    
    def bsc(self, n, m, mode="tm"):
        if abs(m) > n or m not in (self.l + 1, self.l - 1): return 0
        norm = self.normalization_constant()
        if mode.lower() == "te":
            pm = m - self.l
            return -1j * norm * pm * super().bsc(n, m, mode="tm")
        
        return norm * super().bsc(n, m, mode=mode)
    
    def make_field(self, *args, **kwargs):
        return super().make_field(
            *args, k=self.k, degrees=(self.l - 1, self.l + 1), **kwargs
        )

class FreeLGEvenFSFrame(FreeLGFSFrame):
    def bsc(self, n, m, mode="tm"):
        g = 0
        non_zero = (self.l + 1, self.l - 1)
        if mode.lower() == "tm":
            if  m in non_zero:
                g += super().bsc(n, m, mode=mode) / 2
            if -m in non_zero:
                g += super().bsc(n, -m, mode=mode) / 2
        elif mode.lower() == "te":
            if  m in non_zero:
                g += super().bsc(n, m, mode=mode) / 2
            if -m in non_zero:
                g -= super().bsc(n, -m, mode=mode) / 2
        return g
    
    def make_field(self, *args, **kwargs):
        degrees = (-self.l - 1, -self.l + 1, self.self.l - 1, self.l + 1)
        return super().make_field(*args, degrees=degrees, **kwargs)

class FreeLGOddFSFrame(FreeLGFSFrame):
    def bsc(self, n, m, mode="tm"):
        g = 0
        non_zero = (self.l + 1, self.l - 1)
        if mode.lower() == "tm":
            if  m in non_zero:
                g += super().bsc(n, m, mode=mode) / 2 / 1j
            if -m in non_zero:
                g -= super().bsc(n, -m, mode=mode) / 2 / 1j
        elif mode.lower() == "te":
            if  m in non_zero:
                g += super().bsc(n, m, mode=mode) / 2 / 1j
            if -m in non_zero:
                g += super().bsc(n, -m, mode=mode) / 2 / 1j
        return g
        
    def make_field(self, *args, **kwargs):
        degrees = (-self.l - 1, -self.l + 1, self.self.l - 1, self.l + 1)
        return super().make_field(*args, degrees=degrees, **kwargs)