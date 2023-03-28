""" For computing BSCs """

import numpy as np
import mpmath as mp
from utils import bp

def make_bsc_dict(func, n_max, ms, e0=1, n_min=1, **func_kwargs):
    bscs = {"tm": {}, "te": {}}
    for mode in bscs:
        for n in range(n_min, n_max + 1):
            for m in ms:
                bscs[mode][(n, m)] = e0 * func(n, m, mode=mode, **func_kwargs)
    return bscs

def pw_bsc(n, m, mode="tm"):
    if mode.lower() == "tm" and m in (-1, 1):
        return .5
    if mode.lower() == "te" and m in (-1, 1):
        if m == 1: return -1j / 2
        if m == -1: return 1j / 2
    return 0

def make_pw_bscs(n_max, e0=1):
    return make_bsc_dict(pw_bsc, n_max, (-1, 1), e0=e0)

@np.vectorize
def gaussian_tm_bsc(n, s):
    """Gaussian TM BSC for m = +-1"""
    if n < 1:
        raise ValueError("n cannot be less than one.")
    if n % 2 != 0:
        p, a = (n - 1) // 2, mp.mpmathify("5/2")
    else:
        p, a = n // 2 - 1, mp.mpmathify("7/2")

    return bp(p, -8 * s ** 2, a=a, b=2) / 2

@np.vectorize
def gaussian_bsc(n, m, mode="tm", s=.01):
    """Gaussian BSC"""
    if abs(m) != 1:
        return 0
    tm_bsc = gaussian_tm_bsc(n, s)
    if mode.lower() == "tm":
        return tm_bsc
    if mode.lower() == "te":
        if m == 1: return -1j * tm_bsc
        if m == -1: return 1j * tm_bsc
    else:
        raise ValueError("Mode {} not supported.".format(mode))

def make_gaussian_bscs(n_max, e0=1, s=.01):
    return make_bsc_dict(gaussian_bsc, n_max, (-1, 1), e0=e0, s=s)