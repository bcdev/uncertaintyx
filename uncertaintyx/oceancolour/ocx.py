#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
"""
Empirical model functions for NASA Ocean Colour algorithms. Refer to the
Algorithm Publication Tool (https://doi.org/10.5067/JCQB8QALDOYD).

Refer to Lachlan et al. (2019, https://doi.org/10.3389/feart.2019.00176)
for propagating radiometric data uncertainties through NASA Ocean Colour
algorithms.

Further references:

Hu et al. (2019). Improving Satellite Global Chlorophyll a Data
Products Through Algorithm Refinement and Data Recovery. Journal
of Geophysical Research: Oceans, 124(3), 1524-1543.
https://doi.org/10.1029/2019JC014941.

O'Reilly and Werdell (2019). Chlorophyll algorithms for ocean
color sensors - OC4, OC5 & OC6. Remote Sensing of Environment,
229, 32-47. https://doi.org/10.1016/j.rse.2019.04.021.
"""

from abc import ABC
from typing import Literal

import jax.numpy as jnp
import numpy as np
from jax import Array

from ..f.jax import ToF
from ..m.jax import ToM


class Cb(ToF):
    """
    The blended chlorophyll function.

    For chlorophyll retrievals between, e.g., 0.25 and 0.35 mg m-3. For
    details refer to https://doi.org/10.5067/JCQB8QALDOYD.

    The blending occurs when :class:`OCx` is between, e.g., 0.25-0.35 mg
    m-3, creating a smooth handover between :class:`OCi` (low chlorophyll
    specialist) and :class:`OCx` (baseline). The blending uses perfectly
    normalized weights. Low :class:`OCi` values favour :class:`OCi` while
    high :class:`OCi` values favour :class:`OCx` through self-weighting. A
    quadratic term creates curvature that eliminates boundary
    discontinuities while :class:`OCx` acts as regime detector.
    """

    def __init__(self, t1=0.25, t2=0.35):
        """
        Creates a new instance.

        :param t1: The lower threshold value (e.g., mg m-3).
        :param t2: The upper threshold value (e.g., mg m-3).
        """
        self.t1 = t1
        self.t2 = t2
        self.dt = t2 - t1

        def f(x):
            """
            Returns the blended chlorophyll concentration.

            The blending is self-adjusting.

            :param x: Chlorophyll concentrations obtained from :class:`OCi`
            and :class:`OCx` model functions, respectively, shape (2, ...).
            :returns: The blended chlorophyll concentration.
            """

            def blend(ci: Array, cx: Array) -> Array:
                """Returns the blended chlorophyll concentration."""
                ti = self.t2 - ci
                tx = ci - self.t1
                return (ci * ti + cx * tx) / self.dt

            ci = x[0]
            cx = x[1]
            return jnp.where(
                cx < self.t1,
                ci,
                jnp.where(cx > self.t2, cx, blend(ci, cx)),
            )

        super().__init__(f)


class OCI(ToM):
    """NASA's ocean colour chlorophyll index (OCI) model function."""

    def __init__(self):
        """Creates a new model function instance."""

        def f(p, x):
            r"""
            The model function.

            Let :math:`p = (a_0, a_1) \in \mathbb{R}^{2}` be the model
            parameters and let :math:`\lambda = (\lambda_\mathrm{b},
            \lambda_\mathrm{g}, \lambda_\mathrm{r}) \in \mathbb{R}^{3}`
            denote the central wavelengths of wavebands in the blue,
            green and red, respectively. Further let

            .. math::
                x = (\lambda, R_\mathrm{rs}(\lambda))
                \in \mathbb{R}^{2 \times 3}

            denote the function inputs. Then:

            :param p: The parameters :math:`p \in \mathbb{R}^{2}`.
            :param x: The inputs :math:`x \in \mathbb{R}^{2 \times 3}`.
            :returns: The chlorophyll concentration (mg m-3).
            """
            a0, a1 = p
            b = x[0, 0]
            g = x[1, 0]
            r = x[2, 0]
            B = x[1, 0]  # noqa: N806
            G = x[1, 1]  # noqa: N806
            R = x[1, 2]  # noqa: N806
            c = G - (B + (R - B) * (g - b) / (r - b))

            return jnp.power(10.0, a0 + a1 * c)

        super().__init__(f)

    def estimate(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        preset: str | None = None,
    ) -> np.ndarray:
        """Returns the OCI default parameter values (Hu et al., 2019)."""
        return np.array([-0.4287, 230.47])


class OCX(ToM):
    """NASA's OCX family of chlorophyll model functions."""

    n: Literal[2, 3, 4, 5, 6]
    """The number of polynomial coefficients."""

    def __init__(self, n: Literal[2, 3, 4, 5, 6]):
        """Creates a new model function instance."""

        def f(p, x):
            r"""
            The model function.

            Let :math:`p = (p_0, \dots p_4) \in \mathbb{R}^{5}` be the
            model parameters and let :math:`\lambda = (\lambda_1,\dots
            \lambda_\mathrm{m}) \in \mathbb{R}^{m}` denote the central
            wavelengths of :math:`m-1 \ge 2` wavebands in the blue, and
            a single waveband :math:`\lambda_{m}` in the green.
            Further let

            .. math::
                x = R_\mathrm{rs}(\lambda) \in \mathbb{R}^{m}

            denote the function inputs. Then:

            :param p: The parameters :math:`p \in \mathbb{R}^{5}`.
            :param x: The inputs :math:`x \in \mathbb{R}^{m}`.
            :returns: The chlorophyll concentration (mg m-3).
            """
            return jnp.power(
                10.0,
                jnp.polyval(
                    p[::-1], jnp.log10(jnp.maximum.reduce(x[:-1]) / x[-1])
                ),
            )

        super().__init__(f)

    def estimate(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        preset: Literal["CZCS", "OLCI"] | None = "OLCI",
    ) -> np.ndarray:
        pars = [0.42540, -3.21679, 2.86907, -0.62628, -1.09333]
        match preset:
            case "CZCS":
                pars = [0.31841, -4.56386, 8.63979, -8.41411, 1.91532]
            case "OLCI":
                pass
            case _:
                pass
        return np.array(pars)
