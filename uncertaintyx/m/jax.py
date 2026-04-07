#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
"""
Interface adapters for pure JAX functions.

Adapters employ JAX algorithmic differentiation to compute
derivatives of generic model functions.
"""

from abc import ABC
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..interface.core import M


@jax.jit(static_argnums=0)
def jac_p(f: Callable[[Array, Array], Array], p: Array, x: Array) -> Array:
    r"""
    Evaluates the Jacobian :math:`(G_p f)(p, X)` with respect
    to model parameters :math:`p`.

    Conducted in forward mode.

    Let :math:`k, m, n` be shapes (natural numbers or tuples
    of natural numbers) and let

    .. math::
        f: \mathbb{R}^{k} \times \mathbb{R}^{m} \to
        \mathbb{R}^{n}

    be mappable over the batch dimension :math:`M`,
    then:

    :param f: The function :math:`f`.
    :param p: :math:`p \in \mathbb{R}^{k}`.
    :param x: :math:`X \in \mathbb{R}^{M \times m}`.
    :returns: :math:`(G_p f)(p, X) \in \mathbb{R}^{M \times n \times k}`.
    """
    return jax.jacfwd(f, argnums=0)(p, x)


@jax.jit(static_argnums=0)
def jac_x(f: Callable[[Array, Array], Array], p: Array, x: Array) -> Array:
    r"""
    Evaluates the Jacobian :math:`(G_x f)(p, X)` with respect
    to inputs.

    Conducted in reverse mode.

    Under the same notation as :meth:`jac_p`:

    :param f: The function :math:`f`.
    :param p: :math:`p \in \mathbb{R}^{k}`.
    :param x: :math:`X \in \mathbb{R}^{M \times m}`.
    :returns: :math:`(G_x f)(p, X) \in \mathbb{R}^{M \times n \times m}`.
    """
    return jax.vmap(jax.jacrev(f, argnums=1), in_axes=(None, 0))(p, x)


@jax.jit(static_argnums=0)
def vec_x(f: Callable[[Array, Array], Array], p: Array, x: Array) -> Array:
    r"""
    Evaluates :math:`f(p, X)`.

    Under the same notation as :meth:`jac_p`:

    :param f: The function :math:`f`.
    :param p: :math:`p \in \mathbb{R}^{k}`.
    :param x: :math:`X \in \mathbb{R}^{M \times m}`.
    :returns: :math:`Y \in \mathbb{R}^{M \times n}`.
    """
    return jax.vmap(f, in_axes=(None, 0))(p, x)


def jac_p_no_jit(
    f: Callable[[Array, Array], Array], p: Array, x: Array
) -> Array:
    """Noncompiled version of :meth:`jac_p` for debugging."""
    return jax.jacfwd(f, argnums=0)(p, x)


def jac_x_no_jit(
    f: Callable[[Array, Array], Array], p: Array, x: Array
) -> Array:
    """Noncompiled version of :meth:`jac_x` for debugging."""
    return jax.vmap(jax.jacrev(f, argnums=1), in_axes=(None, 0))(p, x)


def vec_x_no_jit(
    f: Callable[[Array, Array], Array], p: Array, x: Array
) -> Array:
    """Noncompiled version of :meth:`vec_x` for debugging."""
    return jax.vmap(f, in_axes=(None, 0))(p, x)


class ToM(M, ABC):
    r"""
    Adapts a pure function

    .. math::
        f: \mathbb{R}^{k} \times \mathbb{R}^{m} \to
        \mathbb{R}^{n},
        (p, x) \mapsto f(p, x)

    where :math:`k, m, n` are shapes (natural numbers or tuples
    of natural numbers) to the model function interface ``M``.
    """

    def __init__(self, f: Callable[[Array, Array], Array], jit: bool = True):
        """
        Creates a new instance of this class.

        :param f: The function :math:`f`.
        :param jit: Switches JIT compilation on and off (for debugging).
        """
        self._f = jax.jit(f) if jit else f
        self._jit = jit

    def eval(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        p_j = jnp.asarray(p)
        x_j = jnp.asarray(x)
        y_j = (
            vec_x(self._f, p_j, x_j)
            if self._jit
            else vec_x_no_jit(self._f, p_j, x_j)
        )
        return np.asarray(y_j)

    def jac_p(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        p_j = jnp.asarray(p)
        x_j = jnp.asarray(x)
        g_j = (
            jac_p(self._f, p_j, x_j)
            if self._jit
            else jac_p_no_jit(self._f, p_j, x_j)
        )
        return np.asarray(g_j)

    def jac_x(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        p_j = jnp.asarray(p)
        x_j = jnp.asarray(x)
        g_j = (
            jac_x(self._f, p_j, x_j)
            if self._jit
            else jac_x_no_jit(self._f, p_j, x_j)
        )
        return np.asarray(g_j)

    @property
    def f(self) -> Callable[[Array, Array], Array]:
        return self._f


class Exponential(ToM):
    """
    The exponential model function.
    """

    def __init__(self):
        def f(p, x):
            """The exponential function."""
            a, b, c = p
            return a * jnp.exp(b * x) + c

        super().__init__(f)

    def estimate(
        self, x: np.ndarray | None = None, y: np.ndarray | None = None
    ) -> np.ndarray:
        return np.array([1.0, 1.0, 0.0])


class Linear(ToM):
    """
    The linear model function.
    """

    def __init__(self):
        def f(p, x):
            """The linear function."""
            a, b = p
            return a * x + b

        super().__init__(f)

    def estimate(
        self, x: np.ndarray | None = None, y: np.ndarray | None = None
    ) -> np.ndarray:
        return np.array([1.0, 0.0])
