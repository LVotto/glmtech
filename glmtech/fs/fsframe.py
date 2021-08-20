from abc import ABC, abstractmethod
import mpmath as mp
from ..field import SphericalElectricField

class FSFrame(ABC):
    """A class that makes BSCs in the FS method."""
    
    def maclaurin(self, n, m, mode="tm"):
        """Directs to the correct polarization mode."""
        if mode.lower() == "tm":
            return self.tm_maclaurin(n, m)
        if mode.lower() == "te":
            return self.te_maclaurin(n, m)
    
    @abstractmethod
    def tm_maclaurin(self, *args, **kwargs):
        """TM mode Maclaurin coefficient."""
        pass
    
    @abstractmethod
    def te_maclaurin(self, *args, **kwargs):
        """TE mode Maclaurin coefficient."""
        pass
    
    def premul(self, n, m):
        """Repeated term before each FS sum which may only depend on n and m."""
        if (n - m) % 2 == 0:
            res  = mp.power(-1j, abs(m) - 1) / mp.pi
            res *= mp.gammaprod([(n - abs(m)) / 2 + 1], [(n + abs(m) + 1) / 2])
            res /= mp.power(2, abs(m) + 1)
        else:
            res  = mp.power(-1j, abs(m) - 2) / mp.pi
            res *= mp.gammaprod([(n - abs(m) + 1) / 2], [(n + abs(m)) / 2 + 1])
            res /= mp.power(2, abs(m) + 2)
        return res
    
    def bsc(self, n, m, mode="tm"):
        """The BSCs themselves as in the FS."""
        if n < abs(m): return 0
        fsum = 0
        q = 0
        while q <= n / 2:
            fsum += mp.power(2, n - 2 * q) \
                  * mp.gammaprod([.5 + n - q], [q + 1]) \
                  * self.maclaurin(n - 2 * q, m, mode=mode)
            q += 1
        return self.premul(n, m) * fsum
    
    def make_field(self, max_n=10, degrees=(-1, 1), k=1,
                   FieldClass=SphericalElectricField):
        bscs = {"tm": {}, "te": {}}
        for mode in bscs:
            for m in degrees:
                for n in range(1, max_n + 1):
                    bscs[mode][n, m] = self.bsc(n, m, mode=mode)
        return FieldClass(k, bscs=bscs)