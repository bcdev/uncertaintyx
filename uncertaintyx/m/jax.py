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

from ..tyx import M


@jax.jit(static_argnums=(0, 3))
def jac_p(
    f: Callable[[Array, Array], Array], p: Array, x: Array, rev: bool = True
) -> Array:
    r"""
    Evaluates the Jacobian :math:`(G_p f)(p, X)` with respect
    to model parameters :math:`p`.

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
    :param rev: Use reverse mode.
    :returns: :math:`(G_p f)(p, X) \in \mathbb{R}^{M \times n \times k}`.
    """
    return jax.vmap(jac(f, 0, rev), in_axes=(None, 0))(p, x)


@jax.jit(static_argnums=(0, 3))
def jac_x(
    f: Callable[[Array, Array], Array], p: Array, x: Array, rev: bool = True
) -> Array:
    r"""
    Evaluates the Jacobian :math:`(G_x f)(p, X)` with respect
    to inputs.

    Under the same notation as :meth:`jac_p`:

    :param f: The function :math:`f`.
    :param p: :math:`p \in \mathbb{R}^{k}`.
    :param x: :math:`X \in \mathbb{R}^{M \times m}`.
    :param rev: Use reverse mode.
    :returns: :math:`(G_x f)(p, X) \in \mathbb{R}^{M \times n \times m}`.
    """
    return jax.vmap(jac(f, 1, rev), in_axes=(None, 0))(p, x)


def jac(f: Callable[[Array, Array], Array], arg: int, rev: bool = True):
    """Returns the Jacobian (does not belong to public API)."""
    return jax.jacrev(f, argnums=arg) if rev else jax.jacfwd(f, argnums=arg)


@jax.jit(static_argnums=(0, 3))
def lpu_p(d: int, g: Array, u: Array, diag: bool = False) -> Array:
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
    return jax.vmap(make_lpu(d, diag), in_axes=(0, None))(g, u)


@jax.jit(static_argnums=(0, 3))
def lpu_x(d: int, g: Array, u: Array, diag: bool = False) -> Array:
    r"""
    Implementation of the law of propagation of uncertainty in
    general tensor form (for input uncertainty tensors).

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


@jax.jit(static_argnums=(0,))
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


def jac_p_no_jit(  # pragma: no cover
    f: Callable[[Array, Array], Array], p: Array, x: Array, rev: bool = True
) -> Array:
    """Noncompiled version of :meth:`jac_p` for debugging."""
    return jax.vmap(jac(f, 0, rev), in_axes=(None, 0))(p, x)


def jac_x_no_jit(  # pragma: no cover
    f: Callable[[Array, Array], Array], p: Array, x: Array, rev: bool = True
) -> Array:
    """Noncompiled version of :meth:`jac_x` for debugging."""
    return jax.vmap(jac(f, 1, rev), in_axes=(None, 0))(p, x)


def vec_x_no_jit(  # pragma: no cover
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

    def __init__(
        self,
        f: Callable[[Array, Array], Array],
        rev_p: bool = True,
        rev_x: bool = True,
        jit: bool = True,
    ):
        """
        Creates a new instance of this class.

        :param f: The function :math:`f`.
        :param rev_p: Use reverse mode for the :math:`p`-Jacobian.
        :param rev_x: Use reverse mode for the :math:`x`-Jacobian.
        :param jit: Switches JIT compilation on and off (for debugging).
        """
        self._f = jax.jit(f) if jit else f
        self._jit = jit
        self._rev_p = rev_p
        self._rev_x = rev_x

    def eval(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        p_ = jnp.asarray(p)
        x_ = jnp.asarray(x)
        y_ = (
            vec_x(self._f, p_, x_)
            if self._jit
            else vec_x_no_jit(self._f, p_, x_)
        )
        return np.asarray(y_)

    def jac_p(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        p_ = jnp.asarray(p)
        x_ = jnp.asarray(x)
        g_ = (
            jac_p(self._f, p_, x_, self._rev_p)
            if self._jit
            else jac_p_no_jit(self._f, p_, x_, self._rev_p)
        )
        return np.asarray(g_)

    def jac_x(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        p_ = jnp.asarray(p)
        x_ = jnp.asarray(x)
        g_ = (
            jac_x(self._f, p_, x_, self._rev_x)
            if self._jit
            else jac_x_no_jit(self._f, p_, x_, self._rev_x)
        )
        return np.asarray(g_)

    def lpu_p(
        self, p: np.ndarray, u: np.ndarray, x: np.ndarray, diag: bool = False
    ) -> np.ndarray:
        p_ = jnp.asarray(p)
        u_ = jnp.asarray(u)
        x_ = jnp.asarray(x)
        v_ = lpu_p(p_.ndim, jac_p(self._f, p_, x_, self._rev_p), u_, diag)
        return np.asarray(v_)

    def lpu_x(
        self, p: np.ndarray, x: np.ndarray, u: np.ndarray, diag: bool = False
    ) -> np.ndarray:
        p_ = jnp.asarray(p)
        x_ = jnp.asarray(x)
        u_ = jnp.asarray(u)
        v_ = lpu_x(x_.ndim - 1, jac_x(self._f, p_, x_, self._rev_x), u_, diag)
        return np.asarray(v_)

    @property
    def f(self) -> Callable[[Array, Array], Array]:
        return self._f


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

    def prior(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        preset: str | None = None,
    ) -> np.ndarray:
        return np.asarray([1.0, 0.0])
