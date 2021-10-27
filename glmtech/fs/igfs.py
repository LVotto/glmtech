import mpmath as mp
import numpy as np
# from numpy.core.fromnumeric import diagonal
from scipy.sparse import diags
from functools import lru_cache
# from scipy.integrate import quad

from .fsframe import FSFrame
from .lgfs import FreeLGEvenFSFrame, FreeLGOddFSFrame

"""FSFrame for Hermite-Gaussian beams.

This is based on FreeLGFSFrame.

IP := Ince polynomial
"""

def tridiagonal_ce(n, ell=1):
    """ Tridiagonal recurrence matrix for even IPs with even indices. """
    diagonals = [
        [(n - j + 1) * ell for j in range(1, n + 1)],
        [(4 * j ** 2) for j in range(n + 1)],
        [(n + j + 1) * ell for j in range(n)]
    ]
    diagonals = [np.array(d) for d in diagonals]
    diagonals[0] += np.array([int(j == 0) * n * ell for j in range(n)])
    return diags(diagonals, [-1, 0, 1]).toarray()


def tridiagonal_co(n, ell=1):
    diagonals = [
        [(n - j + 1) * ell for j in range(1, n + 1)],
        [(2 * j + 1) ** 2 for j in range(n + 1)],
        [(n + j + 2) * ell for j in range(n)]
    ]
    diagonals = [np.array(d) for d in diagonals]
    diagonals[1][0] += (n + 1) * ell
    return diags(diagonals, [-1, 0, 1]).toarray()

def tridiagonal_se(n, ell=1):
    diagonals = [
        [(n - j + 1) * ell for j in range(2, n + 1)],
        [4 * j ** 2 for j in range(1, n + 1)],
        [(n + j + 1) * ell for j in range(1, n)]
    ]
    return diags(diagonals, [-1, 0, 1]).toarray()

def tridiagonal_so(n, ell=1):
    diagonals = [
        [(n - j + 1) * ell for j in range(1, n + 1)],
        [(2 * j + 1) ** 2 for j in range(n + 1)],
        [(n + j + 2) * ell for j in range(n)]
    ]
    diagonals = [np.array(d) for d in diagonals]
    diagonals[1][0] -= (n + 1) * ell
    return diags(diagonals, [-1, 0, 1]).toarray()

def eig_ce(n, ell=1):
    """ Eigenvalues for even IPs with even indices """
    return sorted(np.real(np.linalg.eig(tridiagonal_ce(n, ell=ell))[0]))

def norm2_through_parseval_even(coeffs):
    return np.pi * (np.abs(coeffs[0]) ** 2 + sum(np.abs(coeffs[1:]) ** 2) / 2)

def norm2_through_parseval_odd(coeffs):
    return norm2_through_parseval_even(np.array([0, *coeffs]))

@lru_cache()
def cos_coeffs_e(n, m, ell=1, a0=None, normalize=False):
    eigenvals, eigenvecs = np.linalg.eig(tridiagonal_ce(n, ell=ell))
    eigenvals = np.real(eigenvals)
    eigenvecs = eigenvecs.T[eigenvals.argsort()]
    coeffs = eigenvecs[m]
    if normalize:
        coeffs *= 1 / coeffs[0]
        norm2 = norm2_through_parseval_even(coeffs) * 2
        a0 = np.sqrt(np.pi / norm2)
    if a0 is not None:
        coeffs *= a0 / coeffs[0]
    return coeffs

@lru_cache()
def cos_coeffs_o(n, m, ell=1, a0=None, normalize=False):
    eigenvals, eigenvecs = np.linalg.eig(tridiagonal_co(n, ell=ell))
    eigenvals = np.real(eigenvals)
    eigenvecs = eigenvecs.T[eigenvals.argsort()]
    coeffs = eigenvecs[m]
    if normalize:
        coeffs *= 1 / coeffs[0]
        norm2 = norm2_through_parseval_odd(coeffs) * 2
        a0 = np.sqrt(np.pi / norm2)
    if a0 is not None:
        coeffs *= a0 / coeffs[0]
    return coeffs

@lru_cache()
def sin_coeffs_e(n, m, ell=1, a0=None, normalize=False):
    if m == 0 or n == 0: return 0
    eigenvals, eigenvecs = np.linalg.eig(tridiagonal_se(n, ell=ell))
    eigenvals = np.real(eigenvals)
    eigenvecs = eigenvecs.T[eigenvals.argsort()]
    coeffs = eigenvecs[m - 1]
    if normalize:
        coeffs *= 1 / coeffs[0]
        norm2 = norm2_through_parseval_odd(coeffs) * 2
        a0 = np.sqrt(np.pi / norm2)
    if a0 is not None:
        coeffs *= a0 / coeffs[0]
    return coeffs
    
@lru_cache()
def sin_coeffs_o(n, m, ell=1, a0=None, normalize=False):
    if m == 0 or n == 0: return 0
    eigenvals, eigenvecs = np.linalg.eig(tridiagonal_so(n, ell=ell))
    eigenvals = np.real(eigenvals)
    eigenvecs = eigenvecs.T[eigenvals.argsort()]
    coeffs = eigenvecs[m]
    if normalize:
        coeffs *= 1 / coeffs[0]
        norm2 = norm2_through_parseval_odd(coeffs) * 2
        a0 = np.sqrt(np.pi / norm2)
    if a0 is not None:
        coeffs *= a0 / coeffs[0]
    return coeffs

def fourier_coeff_ce(j, n, m, ell=1, a0=1):
    if j == 0: return a0
    if j > n: return 0

    eigenvals = sorted(eig_ce(n, ell=ell))
    eig = eigenvals[m]

    if j == 1:
        return a0 * eig / ell / (n + 1)

    aj = lambda k: fourier_coeff_ce(k, n, m, ell=ell, a0=a0)

    if j == 2:
        a1 = aj(1)
        return -(a0 * 2 * n * ell + (4 - eig) * a1) / (n + 2) / ell
    
    ajm2 = aj(j - 2)
    ajm1 = aj(j - 1)
    return -((n - j + 2) * ell * ajm2 \
        + (4 * j ** 2 - eig) * ajm1) / ell / (n + j)

def ince_ce(x, ell, a, b, normalize=True):
    n, m = a // 2, b // 2
    ce = 0
    coeffs = cos_coeffs_e(n, m, ell=ell, a0=1, normalize=normalize)
    # for j in range(n + 1):
        # ce += fourier_coeff_ce(j, n, m, ell=ell, a0=1) * np.cos(2 * j * x)
    ce = np.dot(coeffs, np.array([np.cos(2 * j * x) for j in range(n + 1)]))
    return ce

def ince_co(x, ell, a, b, normalize=True):
    n, m = a // 2, b // 2
    co = 0
    coeffs = cos_coeffs_o(n, m, ell=ell, a0=1, normalize=normalize)
    co = np.dot(coeffs,
        np.array([np.cos((2 * j + 1) * x) for j in range(n + 1)]))
    return co

def ince_se(x, ell, a, b, normalize=True):
    n, m = a // 2, b // 2
    se = 0
    coeffs = sin_coeffs_e(n, m, ell=ell, a0=1, normalize=normalize)
    se = np.dot(coeffs, np.array([np.sin(2 * j * x) for j in range(1, n + 1)]))
    return se

def ince_so(x, ell, a, b, normalize=True):
    n, m = a // 2, b // 2
    so = 0
    coeffs = sin_coeffs_o(n, m, ell=ell, a0=1, normalize=normalize)
    so = np.dot(coeffs,
        np.array([np.sin((2 * j + 1) * x) for j in range(n + 1)]))
    return so

def ince_c(x, ell, a, b, normalize=True):
    if a % 2 == 0:
        if b % 2 != 0:
            return 0
        return ince_ce(x, ell, a, b, normalize=normalize)
    else:
        if b % 2 == 0:
            return 0
        return ince_co(x, ell, a, b, normalize=normalize)

def ince_s(x, ell, a, b, normalize=True):
    if a % 2 == 0:
        if b % 2 != 0:
            return 0
        return ince_se(x, ell, a, b, normalize=normalize)
    else:
        if b % 2 == 0:
            return 0
        return ince_so(x, ell, a, b, normalize=normalize)

class IGFSFrame(FSFrame):
    def __init__(self, k, w0, a=0, b=0, ellipticity=1, even=True):
        self.wavenumber = k
        self.k = k
        self.w0 = w0        # Beam waist radius
        self.a = a          # IP sub-index (a = 2n or a = 2n + 1)
        self.b = b          # IP super-index (b = 2m or b = 2m + 1)
        self.even = even    # Whether IP is even
        self.ellipticity = ellipticity
        self.ell = self.ellipticity
    
        if self.even:
            if self.a % 2 == 0:
                coeff_func = cos_coeffs_e
            else:
                coeff_func = cos_coeffs_o
        else:
            if self.a % 2 == 0:
                coeff_func = sin_coeffs_e
            else:
                coeff_func = sin_coeffs_o
        
        n, m = self.a // 2, self.b // 2
        self.fourier_coeffs = coeff_func(n, m, ell=self.ellipticity,
                                         normalize=True)
    
    def is_even(self):
        return self.even

    def tm_maclaurin(self, *args, **kwargs):
        pass
    
    def te_maclaurin(self, *args, **kwargs):
        pass

    def lg_coeff(self, q):
        a, b = self.a, self.b
        d_2q_a = 1 if a == 2 * q else 0
        d_even_false = 1 if not self.even else 0
        
        coeff = pow(-1, q + (b - a) / 2) \
              * mp.sqrt((1 + d_2q_a) * mp.gamma(a - q + 1) * mp.factorial(q)) \
              * self.fourier_coeffs[(a - 2 * q + d_even_false) // 2]
        return coeff

    def lg_frame(self, p, l, even=None):
        if even is None: even = self.even
        LGFrame = FreeLGEvenFSFrame if even else FreeLGOddFSFrame
        return LGFrame(self.k, self.w0, p=p, l=l)
    
    def bsc(self, n, m, mode="tm"):
        if n < abs(m): return 0
        a = self.a
        # For normalization
        coeff_sum = sum([self.lg_coeff(q) for q in range(a // 2 + 1)])
        def term(q):
            if abs(m) not in (a - 2 * q - 1, a - 2 * q + 1):
                return 0
            return self.lg_coeff(q) * self.lg_frame(q, a - 2 * q).bsc(
                n, m, mode=mode
            )
        return sum(map(term, [q for q in range(a // 2 + 1)])) / coeff_sum
    
    def make_field(self, *args, **kwargs):
        degrees = np.arange(-self.a - 1, self.a + 2, 2)
        return super().make_field(*args, k=self.k, degrees=degrees, **kwargs)