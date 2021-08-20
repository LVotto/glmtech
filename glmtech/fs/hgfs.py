import mpmath as mp
import numpy as np

from .fsframe import FSFrame
from .lgfs import FreeLGEvenFSFrame, FreeLGOddFSFrame

"""FSFrame for Hermite-Gaussian beams.

This is based on FreeLGFSFrame.
"""

class HGFSFrame(FSFrame):
    def __init__(self, k, w0, nx=0, ny=0):
        self.wavenumber = k
        self.k = k
        self.w0 = w0
        self.nx, self.ny = nx, ny
    
    def tm_maclaurin(self, *args, **kwargs):
        pass
    
    def te_maclaurin(self, *args, **kwargs):
        pass
    
    def S(self, w):
        nx, ny = self.nx, self.ny
        def s_term(s):
            return (-1) ** s * mp.binomial(nx, s) * mp.binomial(ny, w - s)
        return sum(map(s_term, [s for s in range(w + 1)]))
    
    def C(self, q):
        nx, ny = self.nx, self.ny
        delta = 2 if (2 * q) == (nx + ny) else 1
        return (-1) ** (ny // 2) * mp.sqrt(
            mp.gammaprod(
                [q + 1, nx + ny - q + 1],
                [nx + 1, ny + 1]
            ) / delta / mp.power(2, nx + ny + 1)
        )
    
    def B(self, q):
        nx, ny = self.nx, self.ny
        return self.C(q) * (
            self.S(q) + (-1) ** nx 
            * self.S(nx + ny - q)
        )
    
    def lg_mode(self, p, l, even=True):
        args = [self.k, self.w0]
        kwargs = {"p": p, "l": l}
        LGFrame = FreeLGEvenFSFrame if even else FreeLGOddFSFrame
        return LGFrame(*args, **kwargs)
    
    def bsc(self, n, m, mode="tm"):
        if n < abs(m): return 0
        nx, ny = self.nx, self.ny
        if ((nx + ny + 1) % 2) != (m % 2) or abs(nx + ny + 1) < abs(m):
            return 0

        even = (ny % 2) == 0
        lg = lambda p, l: self.lg_mode(p, l, even=even)
        def term(q):
            return self.B(q) * (lg(q, nx + ny - 2 * q).bsc(n, m, mode=mode))
        return sum(map(term, [q for q in range(nx // 2 + ny // 2 + 1)]))
    
    def make_field(self, *args, **kwargs):
        nx, ny = self.nx, self.ny
        max_m = nx + ny + 1
        degrees = np.arange(-max_m, max_m + 1, 2)
        return super().make_field(
            *args, k=self.k, degrees=degrees, **kwargs
        )