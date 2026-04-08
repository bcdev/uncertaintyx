#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
"""
Interface adapters for pure JAX functions.

Adapters employ JAX algorithmic differentiation to compute
derivatives of generic functions.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..interface.core import F


@jax.jit(static_argnums=0)
def jac(f: Callable[[Array], Array], x: Array) -> Array:
    r"""
    Evaluates the Jacobian :math:`(G_x f)(X)`.

    Conducted in reverse mode.

    Let :math:`m, n` be shapes (natural numbers or tuples
    of natural numbers) and let

    .. math::
        f: \mathbb{R}^{m} \to \mathbb{R}^{n}

    be mappable over the batch dimension :math:`M`,
    then:

    :param f: The function :math:`f`.
    :param x: :math:`X \in \mathbb{R}^{M \times m}`.
    :returns: :math:`(G_x f)(X) \in \mathbb{R}^{M \times n \times m}`.
    """
    return jax.vmap(jax.jacrev(f))(x)


@jax.jit(static_argnums=0)
def vec(f: Callable[[Array], Array], x: Array) -> Array:
    r"""
    Evaluates :math:`f(X)`.

    Under the same notation as :meth:`jac`:

    :param f: The function :math:`f`.
    :param x: :math:`X \in \mathbb{R}^{M \times m}`.
    :returns: :math:`Y \in \mathbb{R}^{M \times n}`.
    """
    return jax.vmap(f)(x)


def jac_no_jit(  # no coverage
    f: Callable[[Array], Array], x: Array
) -> Array:
    """Noncompiled version of :meth:`jac` for debugging."""
    return jax.vmap(jax.jacrev(f))(x)


def vec_no_jit(  # no coverage
    f: Callable[[Array], Array], x: Array
) -> Array:
    """Noncompiled version of :meth:`vec` for debugging."""
    return jax.vmap(f)(x)


class ToF(F):
    r"""
    Adapts a pure function

    .. math::
        f: \mathbb{R}^{m} \to \mathbb{R}^{n}, \quad
        (x) \mapsto f(x)

    where :math:`m, n` are shapes (natural numbers or tuples
    of natural numbers) to the function interface ``F``.
    """

    def __init__(self, f: Callable[[Array], Array], jit: bool = True):
        """
        Creates a new instance of this class.

        :param f: The function :math:`f`.
        :param jit: Switches JIT compilation on and off (for debugging).
        """
        self._f = jax.jit(f) if jit else f
        self._jit = jit

    def eval(self, x: np.ndarray) -> np.ndarray:
        x_j = jnp.asarray(x)
        y_j = vec(self._f, x_j) if self._jit else vec_no_jit(self._f, x_j)
        return np.asarray(y_j)

    def jac(self, x: np.ndarray) -> np.ndarray:
        x_j = jnp.asarray(x)
        g_j = jac(self._f, x_j) if self._jit else jac_no_jit(self._f, x_j)
        return np.asarray(g_j)

    @property
    def f(self) -> Callable[[Array], Array]:
        return self._f
