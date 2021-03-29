import mpmath as mp
from functools import cache

from fsframe import FSFrame

class LGFS(FSFrame):
    def __init__(self, mu, nu, s):
        self.mu = mu
        self.nu = nu
        self.s = s
    
    def En(self, n, u):
        return (pow(1j, n - self.nu - u) * pow(self.s, n - u) / mp.gamma((n - self.nu - 1 - u) / 2 + 1))
    
    @cache
    def bn(self, n, mu):
        s, nu = self.s, self.nu
        G = mp.power(2, (nu - 1) / 2) / mp.sqrt(mp.pi)
        
        if mu < 0:
            return 0
        if mu == 0:
            if n <= nu: return 0
            return G * self.En(n, 0)
        
        bn  = (2 * mu  - 1 + nu) / mu * self.bn(n, mu - 1) 
        bn -= (mu - 1 + nu) / mu * self.bn(n, mu - 2)
        if n not in (0, 1):
            bn -= 2 * mp.power(s, 2) / mu * self.bn(n - 2, mu - 1)
        return bn
    
    @cache
    def tm_maclaurin(self, p, m):
        mu, nu = self.mu, self.nu
        s = self.s
        if m != nu + 1 or (p - m) % 2 != 0:
            raise NotImplementedError("nope")
        
        # m = nu + 1, (n - m) even
        return self.bn(p, self.mu)
    
    def te_maclaurin(self, p, m):
        return 0