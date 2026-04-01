#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
"""
This module provides functions and empirical model functions used
in the Quasi-Analytical Algorithm (QAA). For details refer to:

Melin & Sclep (2015). Band shifting for ocean color multi-spectral
reflectance data. https://doi.org/10.1364/OE.23.002262.

Lee et al. (2010). An Update of the Quasi-Analytical Algorithm.
https://www.ioccg.org/groups/Software_OCA/QAA_v5.pdf.
"""

import jax.numpy as jnp
import numpy as np

from ..m.jax import ToM


class U(ToM):
    def __init__(self):
        def f(par, x):
            g0, g1 = par
            return (jnp.sqrt(g0 * g0 + 4.0 * g1 * x) - g0) / (2.0 * g1)

        super().__init__(f)

    def estimate(
        self, x: np.ndarray | None = None, y: np.ndarray | None = None
    ) -> np.ndarray:
        return np.array([1.70, 0.52])


class Rrs(ToM):
    def __init__(self):
        def f(par, x):
            c0, c1 = par
            return x / (c0 + c1 * x)

        super().__init__(f)

    def estimate(
        self, x: np.ndarray | None = None, y: np.ndarray | None = None
    ) -> np.ndarray:
        return np.array([1.70, 0.52])


class Eta(ToM):
    """
    Empirical model function to fit data in Lee et al. (2010, Figure 2).
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
    Empirical model function to fit data in Lee et al. (2010, Figure 3).
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
