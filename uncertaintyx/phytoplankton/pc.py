#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import jax.numpy as jnp
import numpy as np

from ..m.jax import ToM


class Maranon(ToM):
    r"""
    The Maranon phytoplankton carbon algorithm.

    .. math::
        f: \mathbb{R}^{2} \times \mathbb{R}
        \to \mathbb{R}, \quad
        (p, x) \mapsto 10^{b} x^{a}

    Computes phytoplankton carbon concentration from a given
    chlorophyll concentration.
    """

    def __init__(self):
        """Creates a new model function instance."""

        def f(p, x):
            r"""
            The model function.

            Let :math:`p = (a, b) \in \mathbb{R}^{2}` be the model
            parameters and let :math:`x` denote the concentration
            of chlorophyll. Then:

            :param p: The parameters :math:`p \in \mathbb{R}^{2}`.
            :param x: :math:`x \in \mathbb{R}` (mg m-3).
            :returns: The phytoplankton carbon concentration (mg m-3).
            """
            a, b = p
            return 10.0**b * jnp.power(x, a)

        super().__init__(f)

    def prior(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        preset: str | None = None,
    ) -> np.ndarray:
        """
        Returns the OCX default parameter values for different sensors.

        The default parameter set returned is for SeaWiFS.
        """
        return np.asarray([0.89, 1.79])
