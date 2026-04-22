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


@jax.jit(static_argnums=(0, 2))
def jac(f: Callable[[Array], Array], x: Array, rev: bool = True) -> Array:
    r"""
    Evaluates the Jacobian :math:`(G_x f)(X)`.

    Let :math:`m, n` be shapes (natural numbers or tuples
    of natural numbers) and let

    .. math::
        f: \mathbb{R}^{m} \to \mathbb{R}^{n}

    be mappable over the batch dimension :math:`M`,
    then:

    :param f: The function :math:`f`.
    :param x: :math:`X \in \mathbb{R}^{M \times m}`.
    :param rev: Use reverse mode.
    :returns: :math:`(G_x f)(X) \in \mathbb{R}^{M \times n \times m}`.
    """
    return jax.vmap(jax.jacrev(f) if rev else jax.jacfwd(f))(x)


@jax.jit(static_argnums=(0, 3))
def lpu(d: int, g: Array, u: Array, diag: bool = False) -> Array:
    r"""
    Implementation of the law of propagation of uncertainty in
    general tensor form.

    Using Einstein's summation convention and the symmetry of the
    input uncertainty tensor :math:`U`:, the output uncertainty
    tensor reads:

    .. math::
        V_{\dots ij} = G_{\dots ik}U_{\dots lk}G_{\dots jl}

    with multi-indices :math:`k, l \in D \subset \mathbb{N}^d`
    for some :math:`d \in \mathbb{N}`. The summation is taken over
    all :math:`k, l \in D`.

    Under the same notation as :meth:`lpu_p`:

    :param d: The number of inner tensor dimensions.
    :param g: Jacobian :math:`G \in \mathbb{R}^{M \times \cdots \times D}`.
    :param u: Tensor :math:`U \in \mathbb{R}^{M \times \cdots \times D}`.
    :param diag: To return only variance elements of :math:`V`.
    :returns: Tensor :math:`V \in \mathbb{R}^{M \times \cdots}`.
    """
    return jax.vmap(make_lpu(d, diag), in_axes=(0, 0))(g, u)


def make_lpu(d: int, diag: bool = False) -> Callable[[Array, Array], Array]:
    """
    Returns the law of propagation of uncertainty.

    :param d: The number of inner tensor dimensions.
    :param diag: To return only variance elements .
    :returns: The law of propagation of uncertainty.
    """

    def lpu(g: Array, u: Array) -> Array:
        """The law of propagation of uncertainty."""
        dims = tuple(range(-d, 0))
        gu = jnp.tensordot(g, u, (dims, dims)) if u.ndim != d else g * u
        return (
            jnp.tensordot(gu, g, (dims, dims))
            if not diag
            else jnp.sum(gu * g, dims)
        )

    return lpu


@jax.jit(static_argnums=(0,))
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
    f: Callable[[Array], Array], x: Array, rev: bool = True
) -> Array:
    """Noncompiled version of :meth:`jac` for debugging."""
    return jax.vmap(jax.jacrev(f) if rev else jax.jacfwd(f))(x)


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

    def __init__(
        self, f: Callable[[Array], Array], rev: bool = True, jit: bool = True
    ):
        """
        Creates a new instance of this class.

        :param f: The function :math:`f`.
        :param rev: Use reverse mode for the Jacobian.
        :param jit: Switches JIT compilation on and off (for debugging).
        """
        self._f = jax.jit(f) if jit else f
        self._rev = rev
        self._jit = jit

    def eval(self, x: np.ndarray) -> np.ndarray:
        x_ = jnp.asarray(x)
        y_ = vec(self._f, x_) if self._jit else vec_no_jit(self._f, x_)
        return np.asarray(y_)

    def jac(self, x: np.ndarray) -> np.ndarray:
        x_ = jnp.asarray(x)
        g_ = (
            jac(self._f, x_, self._rev)
            if self._jit
            else jac_no_jit(self._f, x_, self._rev)
        )
        return np.asarray(g_)

    def lpu(
        self, x: np.ndarray, u: np.ndarray, diag: bool = False
    ) -> np.ndarray:
        x_ = jnp.asarray(x)
        u_ = jnp.asarray(u)
        v_ = lpu(x_.ndim - 1, jac(self._f, x_, self._rev), u_, diag)
        return np.asarray(v_)

    @property
    def f(self) -> Callable[[Array], Array]:
        return self._f
