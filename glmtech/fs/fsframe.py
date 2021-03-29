from abc import ABC, abstractmethod
import mpmath as mp

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
            res  = mp.power(1j, 2 * n + m + 1) / mp.pi ** 2
            res *= mp.gammaprod([(n - m) / 2 + 1], [(n + m + 1) / 2])
            res /= mp.power(2, m + 1)
        else:
            res  = mp.power(1j, 2 * n + m) / mp.pi ** 2
            res *= mp.gammaprod([(n - m + 1) / 2], [(n + m) / 2 + 1])
            res /= mp.power(2, m + 2)
        return res
    
    def bsc(self, n, m, mode="tm"):
        """The BSCs themselves as in the FS."""
        fsum = 0
        q = 0
        while q <= n / 2:
            fsum += mp.power(2, n - 2 * q) \
                  * mp.gammaprod([.5 + n - q], [q + 1]) \
                  * self.maclaurin(n - 2 * q, m, mode=mode)
            q += 1
        return self.premul(n, m) * fsum