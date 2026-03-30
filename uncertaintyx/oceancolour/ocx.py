#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
"""
This module provides functions and empirical model functions for
NASA Ocean Colour algorithms.

Refer to Lachlan et al. (2019, https://doi.org/10.3389/feart.2019.00176)
for propagating radiometric data uncertainties through NASA Ocean Color
algorithms.
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


class Ci(ToF):
    """
    The chlorophyll index (CI) function.

    For chlorophyll retrievals below 0.25 mg m-3. For details refer
    to https://doi.org/10.5067/JCQB8QALDOYD.
    """

    def __init__(self, b, g, r):
        """Creates a new instance.

        :param b: The blue spectral waveband (nm).
        :param g: The green spectral waveband (nm).
        :param r: The red spectral waveband (nm).
        """
        self.t = (g - b) / (r - b)

        def f(x):
            """
            Returns the ocean colour index.

            :param x: Remote sensing reflectance, shape (n_wavebands, ...).
            The last waveband refers to green.
            :returns: The ocean colour index.
            """
            B = x[0]  # noqa: N806
            G = x[1]  # noqa: N806
            R = x[2]  # noqa: N806
            return G - (B + (R - B) * self.t)

        super().__init__(f)


class Mbr(ToF):
    """The maximum band ratio function."""

    def __init__(self):
        def f(x):
            """
            Returns the common logarithm of the maximum band ratio.

            :param x: Remote sensing reflectance, shape (n_wavebands, ...).
            The last waveband refers to green.
            :returns: The common logarithm of the maximum band ratio,
            shape(...).
            """
            B = jnp.maximum.reduce(x[:-1])  # noqa: N806
            G = x[-1]  # noqa: N806
            return jnp.log10(B / G)

        super().__init__(f)


class OCi(ToM):
    """
    NASA's ocean colour chlorophyll index (OCI) model function.

    Needs values of chlorophyll index (:class:`Ci`) as input. Used
    for chlorophyll retrievals below 0.25 mg m-3. For details refer
    to https://doi.org/10.5067/JCQB8QALDOYD. Further references:

    Hu et al. (2019). Improving Satellite Global Chlorophyll a Data
    Products Through Algorithm Refinement and Data Recovery. Journal
    of Geophysical Research: Oceans, 124(3), 1524-1543.
    https://doi.org/10.1029/2019JC014941.

    O'Reilly and Werdell (2019). Chlorophyll algorithms for ocean
    color sensors - OC4, OC5 & OC6. Remote Sensing of Environment,
    229, 32-47. https://doi.org/10.1016/j.rse.2019.04.021.
    """

    def __init__(self):
        def f(p, x):
            """
            The model function.

            :param p: The model parameters, shape (2,).
            :param x: The ocean colour index.
            :returns: The chlorophyll concentration.
            """
            a0, a1 = p
            return jnp.power(10.0, a0 + a1 * x)

        super().__init__(f)

    def estimate(
        self, x: np.ndarray | None = None, y: np.ndarray | None = None
    ) -> np.ndarray:
        """Returns the OCI default parameter values (Hu et al., 2019)."""
        return np.array([-0.4287, 230.47])


class OCx(ToM, ABC):
    """
    NASA's OCx family of chlorophyll model functions.

    Needs values of maximum common logarithm of remote sensing
    reflectance band ratios (:class:`Mbr`) as input. Used for
    chlorophyll retrievals above 0.35 mg m-3. For details refer
    to https://doi.org/10.5067/JCQB8QALDOYD. Further references:

    Hu et al. (2019). Improving Satellite Global Chlorophyll a Data
    Products Through Algorithm Refinement and Data Recovery. Journal
    of Geophysical Research: Oceans, 124(3), 1524-1543.
    https://doi.org/10.1029/2019JC014941.

    O'Reilly and Werdell (2019). Chlorophyll algorithms for ocean
    color sensors - OC4, OC5 & OC6. Remote Sensing of Environment,
    229, 32-47. https://doi.org/10.1016/j.rse.2019.04.021.
    """

    n: Literal[2, 3, 4, 5, 6]
    """The number of polynomial coefficients."""

    def __init__(self, n: Literal[2, 3, 4, 5, 6]):
        """
        Creates a new model function instance.

        :param n: The number of polynomial coefficients.
        """
        self.n = n

        def f(p, x):
            """
            The model function.

            :param p: The model parameters, shape (n_parameters,).
            :param x: The common logarithm of maximum band ratios.
            :returns: The chlorophyll concentration.
            """
            return jnp.power(10.0, jnp.polyval(p[::-1], x))

        super().__init__(f)


class Olci(OCx):
    """
    The OC4 model function for Sentinel-3 OLCI.

    Applicable to OLCI with maximum common logarithm of 443/560,
    490/560 or 510/560 remote sensing reflectance band ratio as
    input.
    """

    def __init__(self):
        """Creates a new model function instance."""
        super().__init__(5)

    def estimate(
        self, x: np.ndarray | None = None, y: np.ndarray | None = None
    ) -> np.ndarray:
        """Returns the OC4 default parameter values for OLCI."""
        return np.array([0.42540, -3.21679, 2.86907, -0.62628, -1.09333])
