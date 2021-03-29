from scipy.special import (loggamma, gammasgn, lpmv, lpmn,
                           riccati_jn, riccati_yn
                        )
import scipy.special as special
import numpy as np

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

def factorial_quotient(num, den):
    # DEPRECATED: USE mpmath.gammaprod
    return gammasgn(num + 1) * gammasgn(den + 1) * np.exp(
            loggamma(num + 1) - loggamma(den + 1))

def fac_plus_minus(n, m):
    """ Calculates the expression below avoiding overflows.
    
    .. math::
        \\frac{(n + m)!}{(n - m)!}
    """
    return factorial_quotient(n + m, n - m)

def legendre_p(degree, order, argument):
    """ Associated Legendre function of integer order
    """

    if degree < np.abs(order):
        return 0
    if order < 0:
        return pow(-1, -order) / fac_plus_minus(degree, -order) \
               * legendre_p(degree, -order, argument)
    return special.lpmv(order, degree, argument)

def legendre_tau(degree, order, argument, mv=True):
    """ Returns generalized Legendre function tau

    Derivative is calculated based on relation 14.10.5:
    http://dlmf.nist.gov/14.10
    """
    if not mv:
        return -np.sqrt(1 - argument ** 2) * special.lpmn(order, degree, argument)[1][order][degree]
    '''
    if argument == 1:
        argument = 1 - 1E-16
    if argument == -1:
        argument = -1 + 1E-16
    '''
    return (degree * argument * legendre_p(degree, order, argument) \
            - (degree + order) * legendre_p(degree - 1, order, argument)) \
            / (np.sqrt(1 - argument * argument))

def legendre_pi(degree, order, argument, overflow_protection=False):
    """ Generalized associated Legendre function pi
    
    .. math::
        \\pi_n^m(x) = \\frac{P_n^m(x)}{\\sqrt{1-x^2}}
    """
    if overflow_protection:
        if argument == 1:
            argument = 1 - 1E-16
        if argument == -1:
            argument = -1 + 1E-16
    return legendre_p(degree, order, argument) \
           / (np.sqrt(1 - argument * argument))

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