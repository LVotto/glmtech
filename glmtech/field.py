from abc import ABC, abstractmethod
import numpy as np
import mpmath as mp
from functools import partialmethod

from .utils import (
    plane_wave_coefficient,
    riccati_yn, riccati_jn,
    lpmn, ltaumn, lpimn,
    mie_ans, mie_bns, mie_cns, mie_dns
)


MAX_IT = 1000
EPSILON = 1e-20

ELECTRIC_FIELD_NAMES = [
    "electric", "e"
]

MAGNETIC_FIELD_NAMES = [
    "magnetic", "h"
]

TM_MODE_NAME = "tm"
TE_MODE_NAME = "te"

def one(*args, **kwargs): return 1

def none(*args, **kwargs): return None


class Field(ABC):
    def __init__(self, wave_number, bscs={'tm': {}, 'te': {}}, f0=1, name=""):
        self.tm_bscs = bscs[TM_MODE_NAME]
        self.te_bscs = bscs[TE_MODE_NAME]
        self.degrees = set()
        self.wave_number = wave_number
        self.f0 = f0
        for n, m in self.tm_bscs:
            if self.tm_bscs[n, m] != 0: self.degrees.add(m)
        for n, m in self.te_bscs:
            if self.te_bscs[n, m] != 0: self.degrees.add(m)
        self.name = name
    
    @property
    def k(self):
        return self.wave_number
    
    @property
    def wavelength(self):
        return 2 * np.pi / self.k

    @property
    def coordinate_system_name(self):
        if isinstance(self, SphericalField):
            return "spherical"
        
        if isinstance(self, CartesianField):
            return "cartesian"

        return "unknown?"
    
    @classmethod
    def max_n(cls, x):
        return int(x + 4.05 * np.power(x, 1 / 3) + 2) if x >= .02 else 2

    def max_it(self, radius):
        x = self.wave_number * radius
        return Field.max_n(x)
    
    @classmethod
    def max_x(cls, n):
        sqrt = np.sqrt
        return (
            n - 81 * (500 * n + 5 * sqrt(5)
            * sqrt(2000 *n ** 2 - 8000*n + 27683) - 1000) ** (1/3) / 200
            - 2 + 2187 * 5 ** (2 / 3)
            / (200 * (100 * n + sqrt(5) 
                * sqrt(2000 * n ** 2 - 8000 * n + 27683) - 200) ** (1 / 3))
        )
    
    def max_r(self, n):
        return Field.max_x(n) / self.k

    @abstractmethod
    def field_i(self, x1, x2, x3, **kwargs):
        pass

    @abstractmethod
    def field_s(self, x1, x2, x3, **kwargs):
        pass

    @abstractmethod
    def field_sp(self, x1, x2, x3, **kwargs):
        pass

    def field_t(self, *point, radius=None, **kwargs):
        if radius == None: radius = self.wavelength
        if self.in_radius(*point, radius=radius):
            return self.field_sp(*point, radius=radius, **kwargs)
        fi = self.field_i(*point, **kwargs)
        fs = self.field_s(*point, radius=radius, **kwargs)
        return fi + fs

    def field(self, x1, x2, x3, part="i", **kwargs):
        if part.lower() == "i":
            return self.field_i(x1, x2, x3, **kwargs)
        if part.lower() == "s":
            return self.field_s(x1, x2, x3, **kwargs)
        if part.lower() == "sp":
            return self.field_sp(x1, x2, x3, **kwargs)
        if part.lower() == "t":
            return self.field_t(x1, x2, x3, **kwargs)
        else:
            raise ValueError("I don't recognize field type: " + str(part))

    def norm(self, x1, x2, x3, part="i", **kwargs):
        return np.linalg.norm(self.field(x1, x2, x3, part=part, **kwargs))
    
    @classmethod
    def sph2car_matrix(cls, theta, phi):
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        return np.array([
            [st * cp,   ct * cp,   -sp  ],
            [st * sp,   ct * sp,    cp  ],
            [ct,       -st,         0   ]
        ])
    
    @classmethod
    def sph2car(cls, v, theta, phi):
        return np.matmul(Field.sph2car_matrix(theta, phi), v)
    
    @classmethod
    def car2sph(cls, x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        t = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
        p = np.arctan2(y, x)
        return np.array([r, t, p])

    def transverse_mode(self, name):
        # TODO: prepare for invalid input
        return "t" + name.lower()[0]
    
    def other_mode(self, mode):
        if mode.lower() == "te":
            return "tm"
        if mode.lower() == "tm":
            return "te"
        raise ValueError("Cannot recognize mode: %s" % mode)

    def is_transverse(self, mode):
        return self.transverse_mode(self.name).lower() == mode.lower()
    
    def is_electric(self):
        if self.name in ELECTRIC_FIELD_NAMES:
            return True
        return False

    def is_magnetic(self):
        if self.name in MAGNETIC_FIELD_NAMES:
            return True
        return False
    
    @abstractmethod
    def in_radius(self, x1, x2, x3, radius=None):
        pass
    
    def bscs(self, mode="tm"):
        if mode.lower() == "tm":
            return self.tm_bscs
        if mode.lower() == "te":
            return self.te_bscs
        if mode.lower() not in ["tm", "te"]:
            raise ValueError(
                "I only recognize modes TM and TE. Received:" + str(mode)
            )

    def bsc(self, n, m, mode="tm"):
            if (n, m) not in self.bscs(mode): return 0
            return self.bscs(mode)[n, m]

    def is_same_type(self, o):
        names = {o.name, self.name}
        if (names.issubset(ELECTRIC_FIELD_NAMES)
            or names.issubset(MAGNETIC_FIELD_NAMES)):
            return True
        return False

    def __add__(self, o):
        if o.k != self.k:
            raise ValueError(
                "I cannot add two Fields with different wave numbers."
            )
        
        if type(self) != type(o):
            raise ValueError(
                "I cannot add fields with different types."
            )
        
        bscs = {
            TM_MODE_NAME: self.tm_bscs.copy(),
            TE_MODE_NAME: self.te_bscs.copy(), 
        }
        bscs[TM_MODE_NAME].update(o.tm_bscs)
        bscs[TE_MODE_NAME].update(o.te_bscs)
        
        for mode in bscs:
            for (n, m) in bscs[mode]:
                bscs[mode][n, m] = self.bsc(n, m, mode=mode) \
                                 + o.bsc(n, m, mode=mode)
        
        FieldClass = type(self)
        return FieldClass(self.k, bscs=bscs)
    
    def __mul__(self, a):
        bscs = {
            TM_MODE_NAME: self.tm_bscs.copy(),
            TE_MODE_NAME: self.te_bscs.copy(),
        }

        for mode in bscs:
            for n, m in bscs[mode]:
                bscs[mode][n, m] *= a

        FieldClass = type(self)
        return FieldClass(self.k, bscs=bscs)
    
    __rmul__ = __mul__

class SphericalField(Field):
    def eval_component(self, radial, theta, phi,
                  riccati_terms, legendre_terms, pre_mul, nm_func,
                  max_it, radius=None, mode="tm", mies=None):
        n, res = 1, 0
        
        bscs = self.bscs(mode)

        if mies is None:
            mies = [1 for _ in range(max_it + 1)]

        while n <= max_it:
            for m in self.degrees:
                if n >= abs(m):
                    inc = plane_wave_coefficient(n, self.wave_number) \
                        * bscs[(n, m)] \
                        * riccati_terms[n] \
                        * legendre_terms[np.abs(m)][n] \
                        * np.exp(1j * m * phi) \
                        * nm_func(n, m) \
                        * mies[n]
                    if not mp.isnan(inc):
                        res += inc
            n += 1
        return self.f0 * pre_mul * res


    def max_it(self, radius):
        x = self.wave_number * radius
        return int(x + 4.05 * np.power(x, 1 / 3) + 2) if x >= .02 else 2

    def component_funcs(self, direction="r", mode="tm",
                               part="i", M=2, mu=1, musp=1):
        direction = direction.lower()
        k_in = M * self.k if part.lower() == "sp" else self.k
        riccati = riccati_yn if part.lower() == "s" else riccati_jn
        R1 = lambda n, r: riccati(n, k_in * r)[0]
        R2 = lambda n, r: riccati(n, k_in * r)[1]
        P1 = lambda m, n, t: lpmn(m, n, np.cos(t))[0]
        ONE = lambda n, m: 1
        # TX
        if self.is_transverse(mode):
            if direction.startswith("r"):
                return None
            riccati_func = R1
            if direction.startswith("t"):
                legendre_func = lpimn
                nm_func = lambda n, m: m
            elif direction.startswith("p"):
                legendre_func = ltaumn
                nm_func = ONE
        #TY
        else:
            if direction.startswith("r"):
                riccati_func = R1
                legendre_func = P1
                nm_func = lambda n, m: n * (n + 1)
            else:
                riccati_func = R2
                if direction.startswith("t"):
                    legendre_func = ltaumn
                    nm_func = ONE
                elif direction.startswith("p"):
                    legendre_func = lpimn
                    nm_func = lambda n, m: m
        
        sgn = 1
        if mode == "tm" and self.name.lower() in MAGNETIC_FIELD_NAMES:
            sgn *= -1
        if part == "s":
            sgn *= -1

        sp_factor = 1
        if part == "sp":
            if not self.is_transverse(mode):
                sp_factor = 1 / M
            else:
                if self.is_magnetic():
                    sp_factor = mu / musp
                elif self.is_electric():
                    sp_factor = musp / mu / M ** 2

        if direction.startswith("r"):
            pre_func = lambda r: sgn * self.k / k_in ** 2 / r ** 2 * sp_factor
        if direction.startswith("t"):
            pre_func = lambda r: sgn / r * sp_factor
        if direction.startswith("p"):
            pre_func = lambda r: 1j * sgn / r * sp_factor

        mie_func = none
        if part == "s":
            mie_func = mie_ans if mode == "tm" else mie_bns
        if part == "sp":
            mie_func = mie_cns if mode == "tm" else mie_dns
        
        return (riccati_func, legendre_func, pre_func, nm_func, mie_func)
    
    def component(self, radial, theta, phi, max_r=None, mode="tm",
                 radius=None, M=2, mu=1, musp=1, small=False, part="i",
                 direction="r"):
        if direction.lower().startswith("r") and self.is_transverse(mode):
            return 0

        if part == "s":
            if radial < radius: return 0
        if part == "sp":
            if radial > radius: return 0

        wavelength = 2 * np.pi / self.k
        if radial < EPSILON:
            # print(radial)
            radial = wavelength * 1e-3
        # if radial == 0: radial = EPSILON

        if not small or radius is None:
            max_it = max(self.max_it(radial), 5)
        else:
            max_it = max(self.max_it(max_r), 5)
        
        max_degree = min(max([abs(d) for d in self.degrees]), max_it)

        rfunc, lfunc, pfunc, nfunc, mfunc =  self.component_funcs(
            direction=direction, mode=mode, part=part, M=M
        )

        riccati_terms = rfunc(max_it, radial)
        legendre_terms = lfunc(max_degree, max_it, theta)
        nm_func = nfunc
        pre_mul = pfunc(radial)
        mies = mfunc(max_it, self.wave_number, radius, M, mu, musp)

        return self.eval_component(
            radial, theta, phi,
            riccati_terms, legendre_terms, pre_mul, nm_func,
            max_it, mies=mies, mode=mode
        )

    radial_tm_i = partialmethod(component, direction="r", part="i", mode="tm")

    radial_te_i = partialmethod(component, direction="r", part="i", mode="te")

    def radial_i(self, *args, **kwargs):
        result  = self.radial_te_i(*args, **kwargs) 
        result += self.radial_tm_i(*args, **kwargs)
        return result
    
    radial_tm_s = partialmethod(component, direction="r", part="s", mode="tm")

    radial_te_s = partialmethod(component, direction="r", part="s", mode="te")

    def radial_s(self, *args, **kwargs):
        result  = self.radial_te_s(*args, **kwargs) 
        result += self.radial_tm_s(*args, **kwargs)
        return result

    radial_tm_sp = partialmethod(component,direction="r",
                                 part="sp", mode="tm")

    radial_te_sp = partialmethod(component, direction="r",
                                 part="sp", mode="te")

    def radial_sp(self, *args, **kwargs):
        result  = self.radial_te_sp(*args, **kwargs) 
        result += self.radial_tm_sp(*args, **kwargs)
        return result

    theta_tm_i = partialmethod(component, direction="t", part="i", mode="tm")

    theta_te_i = partialmethod(component, direction="t", part="i", mode="te")

    def theta_i(self, *args, **kwargs):
        result  = self.theta_te_i(*args, **kwargs) 
        result += self.theta_tm_i(*args, **kwargs)
        return result
    
    theta_tm_s = partialmethod(component, direction="t", part="s", mode="tm")

    theta_te_s = partialmethod(component, direction="t", part="s", mode="te")

    def theta_s(self, *args, **kwargs):
        result  = self.theta_te_s(*args, **kwargs) 
        result += self.theta_tm_s(*args, **kwargs)
        return result

    theta_tm_sp = partialmethod(component,direction="t",
                                 part="sp", mode="tm")

    theta_te_sp = partialmethod(component, direction="t",
                                 part="sp", mode="te")

    def theta_sp(self, *args, **kwargs):
        result  = self.theta_te_sp(*args, **kwargs) 
        result += self.theta_tm_sp(*args, **kwargs)
        return result
    
    phi_tm_i = partialmethod(component, direction="p", part="i", mode="tm")

    phi_te_i = partialmethod(component, direction="p", part="i", mode="te")

    def phi_i(self, *args, **kwargs):
        result  = self.phi_te_i(*args, **kwargs) 
        result += self.phi_tm_i(*args, **kwargs)
        return result
    
    phi_tm_s = partialmethod(component, direction="p", part="s", mode="tm")

    phi_te_s = partialmethod(component, direction="p", part="s", mode="te")

    def phi_s(self, *args, **kwargs):
        result  = self.phi_te_s(*args, **kwargs) 
        result += self.phi_tm_s(*args, **kwargs)
        return result

    phi_tm_sp = partialmethod(component,direction="p",
                                 part="sp", mode="tm")

    phi_te_sp = partialmethod(component, direction="p",
                                 part="sp", mode="te")

    def phi_sp(self, *args, **kwargs):
        result  = self.phi_te_sp(*args, **kwargs) 
        result += self.phi_tm_sp(*args, **kwargs)
        return result
    
    def field_i(self, radial, theta, phi, **kwargs):
        return np.array([
            self.radial_i(radial, theta, phi, **kwargs),
            self.theta_i(radial, theta, phi, **kwargs),
            self.phi_i(radial, theta, phi, **kwargs),
        ])
    
    def field_s(self, radial, theta, phi, **kwargs):
        return np.array([
            self.radial_s(radial, theta, phi, **kwargs),
            self.theta_s(radial, theta, phi, **kwargs),
            self.phi_s(radial, theta, phi, **kwargs),
        ])

    def field_sp(self, radial, theta, phi, **kwargs):
        return np.array([
            self.radial_sp(radial, theta, phi, **kwargs),
            self.theta_sp(radial, theta, phi, **kwargs),
            self.phi_sp(radial, theta, phi, **kwargs),
        ])
    
    def field_part(self, *sph_coords, part="i", **kwargs):
        if part == "i":
            return self.field_i(*sph_coords, **kwargs)
        if part == "s":
            return self.field_s(*sph_coords, **kwargs)
        if part == "sp":
            return self.field_sp(*sph_coords, **kwargs)

        raise ValueError("I don't recognize field part: {}".format(part))
    
    def in_radius(self, radial, theta, phi, radius=None):
        radius = radius or self.wavelength
        return radial <= radius


class SphericalElectricField(SphericalField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, name=ELECTRIC_FIELD_NAMES[0])


class SphericalMagneticField(SphericalField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, name=MAGNETIC_FIELD_NAMES[0])


class CartesianField(Field):
    def __init__(self, spherical, *args, **kwargs):
        k = spherical.k
        bscs = {
            TM_MODE_NAME: spherical.tm_bscs,
            TE_MODE_NAME: spherical.te_bscs
        }
        super().__init__(
            k, *args, bscs=bscs, name=spherical.name, **kwargs
        )
        self.spherical = spherical
    
    @classmethod
    def cartesian_at_coord(cls, x, y, z, sph_field, **kwargs):
        rtp = Field.car2sph(x, y, z)
        return Field.sph2car(sph_field(*rtp, **kwargs), rtp[1], rtp[2])
    
    def field_i(self, x, y, z, **kwargs):
        sph_field = self.spherical.field_i
        return CartesianField.cartesian_at_coord(x, y, z, sph_field, **kwargs)
    
    def field_s(self, x, y, z, **kwargs):
        sph_field = self.spherical.field_s
        return CartesianField.cartesian_at_coord(x, y, z, sph_field, **kwargs)
    
    def field_sp(self, x, y, z, **kwargs):
        sph_field = self.spherical.field_sp
        return CartesianField.cartesian_at_coord(x, y, z, sph_field, **kwargs)
    
    def sph_at_xyz(self, func, x, y, z, *args, **kwargs):
        rtp = Field.car2sph(x, y, z)
        return func(*rtp, *args, **kwargs)

    def in_radius(self, x, y, z, radius=None):
        radius = radius or self.wavelength
        return np.linalg.norm([x, y, z]) <= radius

