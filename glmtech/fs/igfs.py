from matplotlib.pyplot import uninstall_repl_displayhook
import mpmath as mp
import numpy as np
# from numpy.core.fromnumeric import diagonal
from scipy.sparse import diags
from functools import lru_cache
# from scipy.integrate import quad

from .fsframe import FSFrame
from .lgfs import FreeLGEvenFSFrame, FreeLGOddFSFrame

"""FSFrame for Ince-Gaussian beams.

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

@lru_cache()
def cos_coeffs_e(n, m, ell=1):
    eigenvals, eigenvecs = np.linalg.eig(tridiagonal_ce(n, ell=ell))
    eigenvals = np.real(eigenvals)
    eigenvecs = eigenvecs.T[eigenvals.argsort()]
    coeffs = eigenvecs[m]
    return coeffs

@lru_cache()
def cos_coeffs_o(n, m, ell=1):
    eigenvals, eigenvecs = np.linalg.eig(tridiagonal_co(n, ell=ell))
    eigenvals = np.real(eigenvals)
    eigenvecs = eigenvecs.T[eigenvals.argsort()]
    coeffs = eigenvecs[m]
    return coeffs

@lru_cache()
def sin_coeffs_e(n, m, ell=1):
    if m == 0 or n == 0: return np.array([0])
    eigenvals, eigenvecs = np.linalg.eig(tridiagonal_se(n, ell=ell))
    eigenvals = np.real(eigenvals)
    eigenvecs = eigenvecs.T[eigenvals.argsort()]
    coeffs = eigenvecs[m - 1]
    return np.array([0, *coeffs])


@lru_cache()
def sin_coeffs_o(n, m, ell=1):
    # if m == 0 or n == 0: return [0]
    eigenvals, eigenvecs = np.linalg.eig(tridiagonal_so(n, ell=ell))
    eigenvals = np.real(eigenvals)
    eigenvecs = eigenvecs.T[eigenvals.argsort()]
    coeffs = eigenvecs[m]
    return coeffs

def t_norm(v):
    u = v.copy()
    u[0] *= np.sqrt(2)
    return np.linalg.norm(u)

def fourier_coeffs(a, b, even=True, normalize=True,
                   **kwargs):
    if a % 2 != b % 2: return []
    args = (a // 2, b // 2)
    if a % 2 == 0:
        if even:
            coeffs = cos_coeffs_e(*args, **kwargs)
        else:
            coeffs = sin_coeffs_e(*args, **kwargs)
    else:
        if even:
            coeffs = cos_coeffs_o(*args, **kwargs)
        else:
            coeffs = sin_coeffs_o(*args, **kwargs)
    if normalize and len(coeffs[coeffs != 0]) > 0:
        if coeffs[coeffs != 0][0] < 0:
            coeffs = -coeffs
        if even and a % 2 == 0:
            the_norm = t_norm
        else:
            the_norm = np.linalg.norm
        coeffs = coeffs / the_norm(coeffs)
    return coeffs
    
def ince(x, ell, a, b, normalize=True, even=True):
    if a % 2 != b % 2:
        return 0
    sc = np.cos if even else np.sin

    coeffs = fourier_coeffs(a, b, ell=ell, normalize=normalize, even=even)
    
    val = np.dot(coeffs,
        np.array([sc((2 * j + (a % 2)) * x) for j in range(a // 2 + 1)])
    )
    if not even and not np.isreal(x):
        val = -1j * val
    return val

def ince_c(*args, **kwargs):
    return ince(*args, even=True, **kwargs)

def ince_s(*args, **kwargs):
    return ince(*args, even=False, **kwargs)


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
    
        self.fourier_coeffs = fourier_coeffs(a, b, ell=ellipticity,
                                         normalize=True, even=even)
    
    def is_even(self):
        return self.even

    def tm_maclaurin(self, *args, **kwargs):
        pass
    
    def te_maclaurin(self, *args, **kwargs):
        pass

    def unnormed_lg_coeff(self, q):
        a, b = self.a, self.b
        d_2q_a = 1 if a == 2 * q else 0
        # d_even_false = 1 if not self.even else 0
        d_even_false = 0
        
        coeff = pow(-1, q + (b - a) / 2) \
              * mp.sqrt((1 + d_2q_a) * mp.gamma(a - q + 1) * mp.factorial(q)) \
              * self.fourier_coeffs[(a - 2 * q + d_even_false) // 2]
        return coeff

    @lru_cache()    
    def lg_coeff(self, q):
        qs = [q for q in range(self.a // 2 + 1)]
        norm = np.linalg.norm([self.unnormed_lg_coeff(q) for q in qs])
        return self.unnormed_lg_coeff(q) / norm

    def lg_frame(self, p, l, even=None):
        if even is None: even = self.even
        LGFrame = FreeLGEvenFSFrame if even else FreeLGOddFSFrame
        return LGFrame(self.k, self.w0, p=p, l=l)
    
    def bsc(self, n, m, mode="tm"):
        if n < abs(m): return 0
        a = self.a
        # For normalization
        # coeff_sum = np.linalg.norm([self.lg_coeff(q) for q in range(a // 2 + 1)])
        def term(q):
            if abs(m) not in (a - 2 * q - 1, a - 2 * q + 1):
                return 0
            return self.lg_coeff(q) * self.lg_frame(q, a - 2 * q).bsc(
                n, m, mode=mode
            )
        return sum(map(term, [q for q in range(a // 2 + 1)]))
    
    def make_field(self, *args, **kwargs):
        degrees = np.arange(-self.a - 1, self.a + 2, 2)
        return super().make_field(*args, k=self.k, degrees=degrees, **kwargs)