#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
"""
Interface adapters for pure PyTorch functions.

Adapters employ PyTorch algorithmic differentiation to compute
derivatives of generic functions.
"""

from typing import Callable

import numpy as np
import torch
from torch import Tensor

from ..interface.core import F


def jac(f: Callable[[Tensor], Tensor], x: Tensor, rev: bool = True) -> Tensor:
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
    return torch.vmap(torch.func.jacrev(f) if rev else torch.func.jacfwd(f))(
        x
    )


@torch.compile
def lpu(d: int, g: Tensor, u: Tensor, diag: bool = False) -> Tensor:
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
    return torch.vmap(make_lpu(d, diag), in_dims=(0, 0))(g, u)


def make_lpu(
    d: int, diag: bool = False
) -> Callable[[Tensor, Tensor], Tensor]:
    """
    Returns the law of propagation of uncertainty.

    :param d: The number of inner tensor dimensions.
    :param diag: To return only variance elements .
    :returns: The law of propagation of uncertainty.
    """

    def lpu(g: Tensor, u: Tensor) -> Tensor:
        """The law of propagation of uncertainty."""
        dims = list(range(-d, 0))
        gu = torch.tensordot(g, u, (dims, dims)) if u.ndim != d else g * u
        return (
            torch.tensordot(gu, g, (dims, dims))
            if not diag
            else torch.sum(gu * g, dim=dims)
        )

    return lpu


def vec(f: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    r"""
    Evaluates :math:`f(X)`.

    Under the same notation as :meth:`jac`:

    :param f: The function :math:`f`.
    :param x: :math:`X \in \mathbb{R}^{M \times m}`.
    :returns: :math:`Y \in \mathbb{R}^{M \times n}`.
    """
    return torch.vmap(f)(x)


def jac_no_jit(  # no coverage
    f: Callable[[Tensor], Tensor], x: Tensor, rev: bool = True
) -> Tensor:
    """Noncompiled version of :meth:`jac` for debugging."""
    return torch.vmap(torch.func.jacrev(f) if rev else torch.func.jacfwd(f))(
        x
    )


def vec_no_jit(  # no coverage
    f: Callable[[Tensor], Tensor], x: Tensor
) -> Tensor:
    """Noncompiled version of :meth:`vec` for debugging."""
    return torch.vmap(f)(x)


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
        self,
        f: Callable[[Tensor], Tensor],
        rev: bool = True,
        jit: bool = True,
    ):
        """
        Creates a new instance of this class.

        :param f: The function :math:`f`.
        :param rev: Use reverse mode for the Jacobian.
        :param jit: Switches JIT compilation on and off (for debugging).
        """
        self._f = torch.compile(f) if jit else f
        self._rev = rev
        self._jit = jit

    def eval(self, x: np.ndarray) -> np.ndarray:
        x_ = torch.from_numpy(x)
        y_t = vec(self._f, x_) if self._jit else vec_no_jit(self._f, x_)
        return y_t.detach().numpy()

    def jac(self, x: np.ndarray) -> np.ndarray:
        x_ = torch.from_numpy(x).requires_grad_(True)
        g_ = (
            jac(self._f, x_, self._rev)
            if self._jit
            else jac_no_jit(self._f, x_, self._rev)
        )
        return g_.detach().numpy()

    def lpu(
        self, x: np.ndarray, u: np.ndarray, diag: bool = False
    ) -> np.ndarray:
        x_ = torch.from_numpy(x).requires_grad_(True)
        u_ = torch.from_numpy(u)
        v_ = lpu(x_.ndim - 1, jac(self._f, x_, self._rev), u_, diag)
        return v_.detach().numpy()

    @property
    def f(self) -> Callable[[Tensor], Tensor]:
        return self._f
