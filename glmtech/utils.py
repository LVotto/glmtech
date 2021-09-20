from scipy.special import (
    loggamma, gammasgn, lpmn,
    riccati_jn, riccati_yn
)
import numpy as np
import mpmath as mp

# EPS_THETA = 1E-20


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