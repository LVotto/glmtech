from difflib import get_close_matches
import scipy.special as spc
import numpy as np
import mpmath as mp
import pickle
from pathlib import Path
import os

# EPS_THETA = 1E-20


# LOOK_UP_FOLDER = Path("/cached/look_up/").absolute()

dir_name = os.path.join("cached", "look_up")
LOOK_UP_FOLDER = os.path.join(os.path.dirname(__file__), dir_name)

def look_up(func, folder=LOOK_UP_FOLDER):
    filepath = os.path.join(folder, (func.__name__ + ".pickle"))
    # filepath = Path(filename).absolute()
    try:
        with open(filepath, "rb") as f:
            CACHE = pickle.load(f)
    except FileNotFoundError:  
        CACHE = {}
        
    def looked_up_func(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if key not in CACHE:
            CACHE[key] = func(*args, **kwargs)
            with open(filepath, "wb") as f:
                pickle.dump(CACHE, f)
        return CACHE[key]
    
    return looked_up_func

@look_up
def lpmn(*args, **kwargs):
    return spc.lpmn(*args, **kwargs)

@look_up
def riccati_jn(*args, **kwargs):
    return spc.riccati_jn(*args, **kwargs)

@look_up
def riccati_yn(*args, **kwargs):
    return spc.riccati_yn(*args, **kwargs)

def plane_wave_coefficient(degree, wave_number_k):
    """ Computes plane wave coefficient :math:`c_{n}^{pw}` """
    return (1 / (1j * wave_number_k)) \
        * pow(-1j, degree) \
        * (2 * degree + 1) / (degree * (degree + 1))

def ltaumn(m, n, theta):
    # m = np.abs(m)
    return -np.sin(theta) * lpmn(m, n, np.cos(theta))[1]

def lpimn(m, n, theta):
    # m = np.abs(m)
    if np.sin(theta) == 0:
        return ltaumn(m, n, theta)
    return lpmn(m, n, np.cos(theta))[0] / np.sin(theta)

def mie_ans(n_max, k, radius, M, mu, musp):
    a = k * radius
    b = M * a
    psi_a = riccati_jn(n_max, a)
    psi_b = riccati_jn(n_max, b)
    ksi_a = riccati_yn(n_max, a)

    an = (musp * psi_a[0] * psi_b[1] - mu * M * psi_a[1] * psi_b[0]) \
       / (musp * ksi_a[0] * psi_b[1] - mu * M * ksi_a[1] * psi_b[0])
    return an

def mie_bns(n_max, k, radius, M, mu, musp):
    a = k * radius
    b = M * a
    psi_a = riccati_jn(n_max, a)
    psi_b = riccati_jn(n_max, b)
    ksi_a = riccati_yn(n_max, a)

    bn = (mu * M * psi_a[0] * psi_b[1] - musp * psi_a[1] * psi_b[0]) \
       / (mu * M * ksi_a[0] * psi_b[1] - musp * ksi_a[1] * psi_b[0])
    return bn

def mie_cns(n_max, k, radius, M, mu, musp):
    a = k * radius
    b = M * a
    psi_a = riccati_jn(n_max, a)
    psi_b = riccati_jn(n_max, b)
    ksi_a = riccati_yn(n_max, a)

    cn = M * musp * (ksi_a[0] * psi_a[1] - ksi_a[1] * psi_a[0]) \
       / (musp * ksi_a[0] * psi_b[1] - mu * M * ksi_a[1] * psi_b[0])
    return cn

def mie_dns(n_max, k, radius, M, mu, musp):
    a = k * radius
    b = M * a
    psi_a = riccati_jn(n_max, a)
    psi_b = riccati_jn(n_max, b)
    ksi_a = riccati_yn(n_max, a)

    dn = mu * M * M * (ksi_a[0] * psi_a[1] - ksi_a[1] * psi_a[0]) \
       / (mu * M * ksi_a[0] * psi_b[1] - musp * ksi_a[1] * psi_b[0])
    return dn

def eval_field_at(x, y, z, f=None):
    field_value = f.field_i(x, y, z)
    return field_value
eval_field_at = np.vectorize(eval_field_at)

def eval_norm(val):
    return float(mp.norm(val))
eval_norm = np.vectorize(eval_norm)

def eval_norm2(val):
    return eval_norm(val) ** 2

def bscs_at_m(sph, m=0, min_n=1, max_n=100, mode="tm"):
    ns = np.arange(min_n, max_n + 1)
    return np.vectorize(sph.bsc, otypes=[np.ndarray])(ns, m, mode)

factorial = np.math.factorial
def lg_mode(x, y, z=0, p=0, l=0, k=1e-7, w0=1e-5):
    dl0 = 2 if l == 0 else 1
    c_pl = np.sqrt(2 * factorial(p) / np.pi / dl0 / factorial(p + l))
    zr = k * w0 ** 2 / 2
    R_inv = z / (z ** 2 + zr ** 2)
    w = w0 * np.sqrt(1 + z ** 2 / zr ** 2)
    psi = np.arctan2(z, zr)

    rho, phi = np.linalg.norm([x, y]), np.arctan2(y, x)
    
    return (
        c_pl / w * (rho / w * np.sqrt(2)) ** l 
        * spc.assoc_laguerre(2 * rho ** 2 / w ** 2, p, l)
        * np.exp(1j * (l * phi - k * rho ** 2 / 2 * R_inv))
        * np.exp(1j * (2 * p + l + 1) * psi)
        * np.exp(-rho ** 2 / w ** 2)
    )
