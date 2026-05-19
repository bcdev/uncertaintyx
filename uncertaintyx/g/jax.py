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

from ..tyx import G


@jax.jit(static_argnums=(0, 2))
def jac(f: Callable[[Array], Array], p: Array, rev: bool = True) -> Array:
    """Returns the Jacobian (does not belong to public API)."""
    return jax.jacrev(f)(p) if rev else jax.jacfwd(f)(p)


@jax.jit(static_argnums=(0, 3))
def lpu(d: int, g: Array, u: Array, diag: bool = True) -> Array:
    r"""
    Implementation of the law of propagation of uncertainty in
    general tensor form.

    Using Einstein's summation convention and the symmetry of the
    parameter uncertainty tensor :math:`U`:, the output uncertainty
    tensor reads:

    .. math::
        V_{\dots ij} = G_{\dots ik}U_{\dots lk}G_{\dots jl}

    with multi-indices :math:`k, l \in D \subset \mathbb{N}^d`
    for some :math:`d \in \mathbb{N}`. The summation is taken over
    all :math:`k, l \in D`.

    Here, :math:`D` denotes the set of inner tensor indices
    (multi-indices of length :math:`d`), and the trailing tensor
    dimensions of :math:`G` and :math:`U` correspond to these
    indices.

    In what follows, we write :math:`\mathbb{R}^{\cdots \times D}`
    for a tensor space whose trailing indices are labelled by the
    index set :math:`D`.

    :param d: The number of inner tensor dimensions.
    :param g: Jacobian :math:`G \in \mathbb{R}^{M \times \cdots \times D}`.
    :param u: Tensor :math:`U \in \mathbb{R}^{\cdots \times D}`.
    :param diag: To return only variance elements of :math:`V`.
    :returns: Tensor :math:`V \in \mathbb{R}^{M \times \cdots}`.
    """
    return make_lpu(d, diag)(g, u)


def make_lpu(d: int, diag: bool = False) -> Callable[[Array, Array], Array]:
    """
    Returns the law of propagation of uncertainty.

    :param d: The number of inner tensor dimensions.
    :param diag: To return only the diagonal elements .
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


def jac_no_jit(  # pragma: no cover
    f: Callable[[Array], Array], p: Array, rev: bool = False
) -> Array:
    """Noncompiled version of :meth:`jac` for debugging."""
    return jax.jacrev(f)(p) if rev else jax.jacfwd(f)(p)


class ToG(G, ABC):
    r"""
    Adapts a pure function

    .. math::
        f: \mathbb{R}^{k} \to \mathbb{R}^{m}, \quad
        p \mapsto f(p)

    where :math:`k, m` are shapes (natural numbers or tuples
    of natural numbers) to the function interface ``G``.
    """

    def __init__(
        self,
        f: Callable[[Array], Array],
        rev: bool = True,
        jit: bool = True,
    ):
        """
        Creates a new instance of this class.

        :param f: The function :math:`f`.
        :param rev: Use reverse mode for the Jacobian.
        :param jit: Switches JIT compilation on and off (for debugging).
        """
        self._f = jax.jit(f) if jit else f
        self._jit = jit
        self._rev = rev

    def eval(self, p: np.ndarray) -> np.ndarray:
        p_ = jnp.asarray(p)
        y_ = self._f(p_)
        return np.asarray(y_)

    def jac(self, p: np.ndarray) -> np.ndarray:
        p_ = jnp.asarray(p)
        g_ = (
            jac(self._f, p_, self._rev)
            if self._jit
            else jac_no_jit(self._f, p_, self._rev)
        )
        return np.asarray(g_)

    def lpu(
        self, p: np.ndarray, u: np.ndarray, diag: bool = False
    ) -> np.ndarray:
        p_ = jnp.asarray(p)
        u_ = jnp.asarray(u)
        v_ = lpu(p_.ndim, jac(self._f, p_, self._rev), u_, diag)
        return np.asarray(v_)

    @property
    def f(self) -> Callable[[Array, Array], Array]:
        return self._f
