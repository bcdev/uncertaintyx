#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import jax.numpy as jnp
import numpy as np

from .ocx import OCI
from ..m.jax import ToM


class Maranon(ToM):
    r"""
    The Maranon et al. (2014) phytoplankton biomass algorithm:

    .. math::
        y = x^{a} \times 10^{b}

    Computes phytoplankton biomass concentration from a given
    chlorophyll concentration. For details refer to:

    Marañón et al. (2014). Resource Supply Overrides Temperature
    as a Controlling Factor of Marine Phytoplankton Growth.
    https://doi.org/10.1371/journal.pone.0099312.
    """

    def __init__(self, as_log10: bool = False):
        """
        Creates a new model function instance.

        :param as_log10: To return the logarithmic biomass concentration.
        """

        def f(p, x):
            r"""
            The phytoplankton biomass model function.

            :param p: The parameters :math:`p = (a, b)`.
            :param x: The chlorophyll concentration :math:`x` (mg m-3).
            :returns: The phytoplankton biomass :math:`y` (mg C m-3).
            """
            a, b = p
            pc = jnp.power(x, a) * 10.0**b
            return jnp.log10(pc) if as_log10 else pc

        super().__init__(f)

    def prior(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        preset: str | None = None,
    ) -> np.ndarray:
        return np.asarray([0.89, 1.79])


class PhytoplanktonCarbon(ToM):
    """
    Concatenation of an OCX chlorophyll algorithm with a phytoplankton
    biomass model.
    """

    def __init__(self, pc: Maranon, oc: OCI = OCI()):
        """
        Creates a new model function instance.

        :param pc: The phytoplankton biomass model.
        :param oc: The OCX chlorophyll model.
        """

        def f(p, x):
            """
            Returns the phytoplankton biomass (mg C m-3).
            """
            return pc.f(p[-2:], oc.f(p[:-2], x))

        super().__init__(f)
        self.oc: ToM = oc
        self.pc: ToM = pc

    def prior(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        preset: str | None = None,
    ) -> np.ndarray:
        return np.concatenate((self.oc.prior(x, y, preset), self.pc.prior()))
