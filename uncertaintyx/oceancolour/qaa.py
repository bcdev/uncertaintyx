#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
"""
This module provides functions and empirical model functions used
in the Quasi-Analytical Algorithm (QAA).
"""

import jax.numpy as jnp
import numpy as np

from ..m.jax import ToM


class Qaa(ToM):
    """
    The Quasi-Analytical Algorithm (QAA).

    Let :math:`\lambda` denote spectral wavelength expressed in nm. Then
    The QAA algorithm computes detrital matter and gelbstoff absorption
    coefficients :math:`a_{\mathrm{dg}}(\lambda)` and phytoplankton
    absorption coefficients :math:`a_{\mathrm{ph}}(\lambda)` from remote
    sensing reflectance :math:`R_{\mathrm{rs}}(\lambda)` and absorption
    and back-scattering coefficients :math:`a_{\mathrm{w}}(\lambda)` and
    :math:`b_{\mathrm{bw}}(\lambda)` of pure water, respectively.

    For details refer to:

    Melin & Sclep (2015). Band shifting for ocean color multi-spectral
    reflectance data. https://doi.org/10.1364/OE.23.002262.

    Lee et al. (2010). An Update of the Quasi-Analytical Algorithm.
    https://www.ioccg.org/groups/Software_OCA/QAA_v5.pdf.

    Lee et al. (2014). An Update of the Quasi-Analytical Algorithm.
    https://www.ioccg.org/groups/Software_OCA/QAA_v6_2014209.pdf
    """

    def __init__(self, b412: int, b443: int, b490: int, b55x: int, b670: int):
        b412: int = 0
        b443: int = 0
        b490: int = 0
        b55x: int = 0
        b670: int = 0

        def _r(R, c1=0.52, c2=1.70):  # noqa: N806
            r"""
            Returns the spectral remote sensing reflectance
            :math:`r_{\mathrm{rs}}(\lambda)` just below the
            surface.

            :param R: :math:`R_{\mathrm{rs}}(\lambda)`
            :param c1: A coefficient.
            :param c2: A coefficient.
            :returns: :math:`r_{\mathrm{rs}}(\lambda)`.
            """
            return R / (c1 + c2 * R)

        def _u(r, g0=0.0890, g1=0.1245):
            r"""
            Returns the ratio of back-scattering coefficient to the sum
            of back-scattering and absorption coefficients.

            :param r: :math:`r_{\mathrm{rs}}(\lambda)`.
            :param g0: A coefficient.
            :param g1: A coefficient.
            :returns: :math:`u(\lambda)`.
            """
            return (jnp.sqrt(4.0 * g1 * r + g0**2) - g0) / (2.0 * g1)

        def _a_1(r, aw, h0=-1.146, h1=-1.366, h2=-0.469):
            r"""
            Returns the total absorption coefficient of Case 1-like water
            at :math:`\lambda_{0} = 55x~\text{nm}`.

            :param r: :math:`r_{\mathrm{rs}}(\lambda)`.
            :param aw: :math:`a_{\mathrm{w}}(\lambda)`.
            :param h0: A coefficient.
            :param h1: A coefficient.
            :param h2: A coefficient.
            :returns: :math:`a(\lambda_{0})`.
            """

            def g(r):
                i = r[b670] / r[b490]  # red to green spectral index
                return jnp.log10(
                    (r[b443] + r[b490]) / (r[b55x] + 5.0 * r[b670] * i)
                )

            def h(x, h0, h1, h2):
                return jnp.power(10.0, h0 + (h1 + h2 * x) * x)

            return aw[b55x] + h(g(r), h0, h1, h2)

        def _a_2(R, aw):  # noqa: N806
            r"""
            Returns the total absorption coefficient of Case 2-like water
            at :math:`\lambda_{0} = 670~\text{nm}`.

            :param R: :math:`R_{\mathrm{rs}}(\lambda)`.
            :param aw: :math:`a_{\mathrm{w}}(\lambda)`.
            :returns: :math:`a(\lambda_{0})`.
            """

            def g(R):  # noqa: N806
                return 0.39 * jnp.power(R[b670] / (R[b443] + R[b490]), 1.14)

            return aw[b670] + g(R)

        def _adg(W, aw, a, s, z):  # noqa: N806
            r"""
            Returns the absorption coefficient of detrital matter and
            gelbstoff.

            :param W: The spectral wavelengths :math:`\lambda`.
            :param aw: Pure water absorption :math:`a_\mathrm{w}(\lambda)`.
            :param a: Total absorption :math:`a(\lambda)`.
            :param s: The coefficient :math:`S`.
            :param z: The coefficient :math:`\zeta`.
            :returns: :math:`a_\mathrm{dg}(\lambda)`.
            """
            x = jnp.exp(s * (442.5 - 415.5))
            e = jnp.exp(s * (W - W[b443]))
            return (
                (a[b412] - aw[b412] - z * (a[b443] - aw[b443])) / (x - z)
            ) / e

        def _b(u, a, bw):
            r"""
            Returns the back-scattering coefficient of suspended particles
            at a reference wavelength :math:`\lambda_{0}`.

            :param u: :math:`u(\lambda_{0})`.
            :param a: :math:`a(\lambda_{0})`.
            :param bw: :math:`b_\mathrm{bw}(\lambda_{0})`.
            :returns: :math:`b_\mathrm{bp}(\lambda_{0})`.
            """
            return (u * a) / (1.0 - u) - bw

        def _e(i, e1=2.0, e2=1.2, e3=0.9):
            """
            Returns the coefficient :math:`\eta`.

            :param i: The blue to green spectral index.
            :param e1: A coefficient.
            :param e2: A coefficient.
            :return: The coefficient :math:`\eta`.
            """
            return e1 * (1.0 - e2 * jnp.exp(-e3 * i))

        def _s(i, s1=0.015, s2=0.002, s3=0.600):
            r"""
            Returns the coefficient :math:`S`.

            :param i: The blue to green spectral index.
            :param s1: A coefficient (:math:`\mathrm{nm}^{-1}`).
            :param s2: A coefficient (:math:`\mathrm{nm}^{-1}`).
            :param s3: A coefficient.
            :returns: The coefficient :math:`S` (:math:`\mathrm{nm}^{-1}`).
            """
            return s1 + s2 / (s3 + i)

        def _z(i):
            r"""
            Returns the coefficient :math:`\zeta`.

            :param i: The blue to green spectral index.
            :returns: The coefficient :math:`\zeta`.
            """
            return 0.74 + 0.2 / (0.8 + i)

        def f(p, x):
            c1, c2, g0, g1, h0, h1, h2, e1, e2, e3, s1, s2, s3 = p
            W = x[0]  # noqa: N806
            R = x[1]  # noqa: N806
            aw = x[2]
            bw = x[3]
            # 1
            r = _r(R, c1, c2)
            u = _u(r, g0, g1)
            # 2
            a = jnp.where(
                R[b670] < 0.0015,
                _a_1(r, aw, h0, h1, h2),
                _a_2(R, aw),
            )
            # 3
            b = jnp.where(
                R[b670] < 0.0015,
                _b(u[b55x], a, bw[b55x]),
                _b(u[b670], a, bw[b670]),
            )
            # 4
            i = r[b443] / r[b55x]
            e = _e(i, e1, e2, e3)
            # 5
            bbp = b * jnp.where(
                R[b670] < 0.0015,
                jnp.power(W[b55x] / W, e),
                jnp.power(W[b670] / W, e),
            )
            # 6
            a = (1.0 - u) * (bw + bbp) / u
            # 7 & 8
            z = _z(i)
            s = _s(i, s1, s2, s3)
            # 9 & 10
            adg = _adg(W, aw, a, s, z)
            aph = a - adg - aw[b443]

            return jnp.stack([adg, aph])

        super().__init__(f)

    def estimate(
        self, x: np.ndarray | None = None, y: np.ndarray | None = None
    ) -> np.ndarray:
        pars = dict(
            c1=1.7000,
            c2=0.5200,
            g0=0.0890,
            g1=0.1245,
            h0=-1.146,
            h1=-1.366,
            h2=-0.469,
            e1=2.0000,
            e2=1.2000,
            e3=0.9000,
            s1=0.0150,
            s2=0.0020,
            s3=0.6000,
        )
        return np.array([p for _, p in pars.items()])


class E(ToM):
    """
    Empirical model function to fit data in Lee et al. (2010, Figure 2)
    for the :math:`\eta` coefficient.
    """

    def __init__(self):
        def f(p, x):
            """The model function."""
            a, c = p
            return a * (1.0 - 1.2 * jnp.exp(-c * x))

        super().__init__(f)

    def estimate(
        self, x: np.ndarray | None = None, y: np.ndarray | None = None
    ) -> np.ndarray:
        return np.array([2.0, 0.9])


class S(ToM):
    """
    Empirical model function to fit data in Lee et al. (2010, Figure 3)
    for the :math:`S` coefficient.
    """

    def __init__(self):
        def f(p, x):
            """The model function."""
            a, b, c = p
            return a + b / (c + x)

        super().__init__(f)

    def estimate(
        self, x: np.ndarray | None = None, y: np.ndarray | None = None
    ) -> np.ndarray:
        return np.array([0.015, 0.002, 0.6])
