#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
"""
This module provides functions and empirical model functions used
in the Quasi-Analytical Algorithm (QAA).
"""

import jax.numpy as jnp
import numpy as np

from ..m.jax import ToM


class QAA(ToM):
    r"""
    The Quasi-Analytical Algorithm (QAA) to compute inherent
    optical properties.

    Let :math:`m \ge 5` denote the number of spectral wavebands
    with nomial wavelengths :math:`\lambda = (\lambda_1, \dots, \lambda_m`).
    QAA uses a minimum band set consisting of:
    :math:`\lambda_{i_{412}} \simeq 412~\mathrm{nm}`,
    :math:`\lambda_{i_{443}} \simeq 443~\mathrm{nm}`,
    :math:`\lambda_{i_{490}} \simeq 490~\mathrm{nm}`,
    :math:`\lambda_{i_{555}} \simeq 555~\mathrm{nm}`, and
    :math:`\lambda_{i_{670}} \simeq 670~\mathrm{nm}`.

    The input band set may include more than this minimal set;
    outputs are spectrally inter- and extrapolated.

    QAA requires inputs, which are available from either remote
    sensing observations or scientific literature:

    .. csv-table::
       :header: Symbol, Description, Unit

       :math:`\lambda`, Spectral wavelengths, nm
       :math:`R_{\mathrm{rs}}(\lambda)`, Remote sensing reflectance, sr-1
       :math:`a_{\mathrm{w}}(\lambda)`, Pure seawater absorption, m-1
       :math:`b_{\mathrm{bw}}(\lambda)`, Pure seawater back-scattering, m-1

    Under the same notation, QAA returns the following inherent
    optical properties:

    .. csv-table::
       :header: Symbol, Description, Unit

       :math:`a(\lambda)`, Total absorption, m-1
       :math:`a_{\mathrm{dg}}(\lambda)`, Detritus and gelbstoff absorption, m-1
       :math:`a_{\mathrm{ph}}(\lambda)`, Phytoplankton absorption, m-1
       :math:`b_{\mathrm{bp}}(\lambda)`, Particulate back-scattering, m-1

    QAA uses several configurable parameters, hereafter referred to by
    the model parameter vector :math:`p \in \mathbb{R}^{k}`:

    .. csv-table::
       :header: Symbol, Description, Value, Unit

       :math:`r_0`, Reflectance conversion coefficient, 0.5200, sr-1
       :math:`r_1`, Reflectance conversion coefficient, 1.7000,
       :math:`g_0`, Gordon coefficient, 0.0890,
       :math:`g_1`, Gordon coefficient, 0.1245,
       :math:`h_0`, Polynomial coefficient, -1.1459,
       :math:`h_1`, Polynomial coefficient, -1.3658,
       :math:`h_2`, Polynomial coefficient, -0.4693,
       :math:`\eta_0`, Coefficient, 2.0000,
       :math:`\eta_1`, Coefficient, 1.2000,
       :math:`\eta_2`, Coefficient, 0.9000,
       :math:`s_0`, Coefficient, 0.0150, nm-1
       :math:`s_1`, Coefficient, 0.0002, nm-1
       :math:`s_2`, Coefficient, 0.6000,
       :math:`t_1`, Threshold to separate Case 1 and 2 water, 0.0015, sr-1

    QAA requires inputs supplied in batches :math:`X` of size :math:M`
    and consisting of samples:

       .. math::
           x = ((\lambda_1, \dots, \lambda_m),
           R_\mathrm{rs}(\lambda_1, \dots, \lambda_m),
           a_\mathrm{w}(\lambda_1, \dots, \lambda_m),
           b_\mathrm{bw}(\lambda_1, \dots, \lambda_m))
           \in \mathbb{R}^{4 \times m}

    QAA returns outputs are returned in batches :math:`Y` of the same
    size and consisting of samples:

       .. math::
           y = (a(\lambda_1, \dots, \lambda_m),
           a_\mathrm{dg}(\lambda_1, \dots, \lambda_m),
           a_\mathrm{ph}(\lambda_1, \dots, \lambda_m),
           b_\mathrm{bp}(\lambda_1, \dots, \lambda_m))
           \in \mathbb{R}^{4 \times m}

    For details refer to:

    Lee et al. (2010). An Update of the Quasi-Analytical Algorithm.
    https://www.ioccg.org/groups/Software_OCA/QAA_v5.pdf.

    Lee et al. (2014). An Update of the Quasi-Analytical Algorithm.
    https://www.ioccg.org/groups/Software_OCA/QAA_v6_2014209.pdf

    Melin & Sclep (2015). Band shifting for ocean color multi-spectral
    reflectance data. https://doi.org/10.1364/OE.23.002262.
    """

    def __init__(
        self,
        i412: int = 0,
        i443: int = 1,
        i490: int = 2,
        i555: int = 4,
        i670: int = 5,
        jit: bool = True,
    ):
        """
        Creates a new QAA model function.

        :param i412: The index :math:`i_{412}` of a waveband near 412 nm.
        :param i443: The index :math:`i_{443}` of a waveband near 443 nm.
        :param i490: The index :math:`i_{490}` of a waveband near 490 nm.
        :param i555: The index :math:`i_{555}` of a waveband near 555 nm.
        :param i670: The index :math:`i_{670}` of a waveband near 670 nm.
        :param jit: Switches JIT compilation on and off (for debugging).
        """

        def _r(R, r0=0.52, r1=1.70):  # noqa: N806
            r"""
            Returns the spectral remote sensing reflectance
            :math:`r_{\mathrm{rs}}(\lambda)` just below the
            surface.

            :param R: :math:`R_{\mathrm{rs}}(\lambda)`
            :param r0: A coefficient.
            :param r1: A coefficient.
            :returns: :math:`r_{\mathrm{rs}}(\lambda)`.
            """
            return R / (r0 + r1 * R)

        def _u(r, g0=0.0890, g1=0.1245):
            r"""
            Returns the ratio of back-scattering coefficient to the sum
            of back-scattering and absorption coefficients.

            :param r: :math:`r_{\mathrm{rs}}(\lambda)`.
            :param g0: A coefficient.
            :param g1: A coefficient.
            :returns: :math:`u(\lambda)`.
            """
            return (jnp.sqrt(g0**2 + 4.0 * g1 * r) - g0) / (2.0 * g1)

        def _a_1(r, aw, h0=-1.1459, h1=-1.3658, h2=-0.4693):
            r"""
            Returns the total absorption coefficient of Case 1 water
            at :math:`\lambda_{0} = 555~\text{nm}`.

            :param r: :math:`r_{\mathrm{rs}}(\lambda)`.
            :param aw: :math:`a_{\mathrm{w}}(\lambda)`.
            :param h0: A coefficient.
            :param h1: A coefficient.
            :param h2: A coefficient.
            :returns: :math:`a(\lambda_{0})`.
            """

            def g(r):
                i = r[i670] / r[i490]
                return jnp.log10(
                    (r[i443] + r[i490]) / (r[i555] + 5.0 * r[i670] * i)
                )

            def h(x, h0, h1, h2):
                return jnp.power(10.0, h0 + (h1 + h2 * x) * x)

            return aw[i555] + h(g(r), h0, h1, h2)

        def _a_2(R, aw):  # noqa: N806
            r"""
            Returns the total absorption coefficient of Case 2 water
            at :math:`\lambda_{0} = 670~\text{nm}`.

            :param R: :math:`R_{\mathrm{rs}}(\lambda)`.
            :param aw: :math:`a_{\mathrm{w}}(\lambda)`.
            :returns: :math:`a(\lambda_{0})`.
            """

            def g(R):  # noqa: N806
                return 0.39 * jnp.power(R[i670] / (R[i443] + R[i490]), 1.14)

            return aw[i670] + g(R)

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
            e = jnp.exp(s * (W - W[i443]))
            adg = (a[i412] - aw[i412] - z * (a[i443] - aw[i443])) / (x - z)
            return adg / e

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

        def _e(i, e0=2.0, e1=1.2, e2=0.9):
            """
            Returns the coefficient :math:`\eta`.

            :param i: The blue to green spectral index.
            :param e0: A coefficient.
            :param e1: A coefficient.
            :param e2: A coefficient.
            :return: The coefficient :math:`\eta`.
            """
            return e0 * (1.0 - e1 * jnp.exp(-e2 * i))

        def _s(i, s0=0.015, s1=0.002, s2=0.600):
            r"""
            Returns the coefficient :math:`S`.

            :param i: The blue to green spectral index.
            :param s0: A coefficient.
            :param s1: A coefficient.
            :param s2: A coefficient.
            :returns: The coefficient :math:`S` (:math:`\mathrm{nm}^{-1}`).
            """
            return s0 + s1 / (s2 + i)

        def _z(i):
            r"""
            Returns the coefficient :math:`\zeta`.

            :param i: The blue to green spectral index.
            :returns: The coefficient :math:`\zeta`.
            """
            return 0.74 + 0.2 / (0.8 + i)

        def f(p, x):
            r"""
            The QAA model function.

            Let :math:`k` be the number of model parameters and let
            :math:`m \ge 5` denote the number of spectral wavebands
            with nomial wavelengths
            :math:`\lambda = (\lambda_1, \dots, \lambda_m)`.
            Further let

            .. math::
                x = (\lambda, R_\mathrm{rs}(\lambda),
                a_\mathrm{w}(\lambda),
                b_\mathrm{bw}(\lambda))
                \in \mathbb{R}^{4 \times m}

            denote the function inputs, and let

            .. math::
                y = (a(\lambda),
                a_\mathrm{dg}(\lambda), a_\mathrm{ph}(\lambda),
                b_\mathrm{bp}(\lambda))
                \in \mathbb{R}^{4 \times m}

            denote its outputs. Then:

            :param p: The parameters :math:`p \in \mathbb{R}^{k}`.
            :param x: :math:`x \in \mathbb{R}^{4 \times m}`.
            :returns: :math:`y \in \mathbb{R}^{4 \times m}`.
            """
            r0, r1, g0, g1, h0, h1, h2, e0, e1, e2, s0, s1, s2, t1 = p
            W = x[0]  # noqa: N806
            R = x[1]  # noqa: N806
            aw = x[2]
            bw = x[3]
            # 1
            r = _r(R, r0, r1)
            u = _u(r, g0, g1)
            # 2
            a = jnp.where(
                R[i670] < t1,
                _a_1(r, aw, h0, h1, h2),
                _a_2(R, aw),
            )
            # 3
            b = jnp.where(
                R[i670] < t1,
                _b(u[i555], a, bw[i555]),
                _b(u[i670], a, bw[i670]),
            )
            # 4
            i = r[i443] / r[i555]
            e = _e(i, e0, e1, e2)
            # 5
            bbp = b * jnp.where(
                R[i670] < t1,
                jnp.power(W[i555] / W, e),
                jnp.power(W[i670] / W, e),
            )
            # 6
            a = (1.0 - u) * (bw + bbp) / u
            # 7 & 8
            z = _z(i)
            s = _s(i, s0, s1, s2)
            # 9 & 10
            adg = _adg(W, aw, a, s, z)
            aph = a - adg - aw

            return jnp.stack([a, adg, aph, bbp])

        super().__init__(f, jit)

    def estimate(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        pars = dict(
            r0=0.5200,
            r1=1.7000,
            g0=0.0890,
            g1=0.1245,
            h0=-1.1459,
            h1=-1.3658,
            h2=-0.4693,
            e0=2.0000,
            e1=1.2000,
            e2=0.9000,
            s0=0.0150,
            s1=0.0020,
            s2=0.6000,
            t1=0.0015,
        )
        match kwargs.get("preset", None):
            case "case1":
                pars["t1"] = np.inf
            case "case2":
                pars["t1"] = 0.0
            case _:
                pass
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
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        **kwargs,
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
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        return np.array([0.015, 0.002, 0.6])
