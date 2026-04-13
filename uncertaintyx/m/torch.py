#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
"""
Interface adapters for pure PyTorch functions.

Adapters employ PyTorch algorithmic differentiation to compute
derivatives of generic model functions.
"""

from abc import ABC
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from ..interface.core import M


@torch.compile
def jac_p(
    f: Callable[[Tensor, Tensor], Tensor],
    p: Tensor,
    x: Tensor,
    rev: bool = True,
) -> Tensor:
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
    return torch.vmap(jac(f, 0, rev), in_dims=(None, 0))(p, x)


@torch.compile
def jac_x(
    f: Callable[[Tensor, Tensor], Tensor],
    p: Tensor,
    x: Tensor,
    rev: bool = True,
) -> Tensor:
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
    return torch.vmap(jac(f, 1, rev), in_dims=(None, 0))(p, x)


def jac(f: Callable[[Tensor, Tensor], Tensor], arg: int, rev: bool = True):
    """Returns the Jacobian (does not belong to public API)."""
    return (
        torch.func.jacrev(f, argnums=arg)
        if rev
        else torch.func.jacfwd(f, argnums=arg)
    )


@torch.compile
def vec_x(
    f: Callable[[Tensor, Tensor], Tensor], p: Tensor, x: Tensor
) -> Tensor:
    r"""
    Evaluates :math:`f(p, X)`.

    Under the same notation as :meth:`jac_p`:

    :param f: The function :math:`f`.
    :param p: :math:`p \in \mathbb{R}^{k}`.
    :param x: :math:`X \in \mathbb{R}^{M \times m}`.
    :returns: :math:`Y \in \mathbb{R}^{M \times n}`.
    """
    return torch.vmap(f, in_dims=(None, 0))(p, x)


def jac_p_no_jit(  # no coverage
    f, p: Tensor, x: Tensor, rev: bool = True
) -> Tensor:
    """Noncompiled version of :meth:`jac_p` for debugging."""
    return torch.vmap(jac(f, 0, rev), in_dims=(None, 0))(p, x)


def jac_x_no_jit(  # no coverage
    f, p: Tensor, x: Tensor, rev: bool = True
) -> Tensor:
    """Noncompiled version of :meth:`jac_x` for debugging."""
    return torch.vmap(jac(f, 1, rev), in_dims=(None, 0))(p, x)


def vec_x_no_jit(  # no coverage
    f: Callable[[Tensor, Tensor], Tensor], p: Tensor, x: Tensor
) -> Tensor:
    """Noncompiled version of :meth:`vec_x` for debugging."""
    return torch.vmap(f, in_dims=(None, 0))(p, x)


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
        f: Callable[[Tensor, Tensor], Tensor],
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
        self._f = torch.compile(f) if jit else f
        self._rev_p = rev_p
        self._rev_x = rev_x
        self._jit = jit

    def eval(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        p_t = torch.from_numpy(p)
        x_t = torch.from_numpy(x)
        y_t = (
            vec_x(self._f, p_t, x_t)
            if self._jit
            else vec_x_no_jit(self._f, p_t, x_t)
        )
        return y_t.detach().numpy()

    def jac_p(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        p_t = torch.from_numpy(p).requires_grad_(True)
        x_t = torch.from_numpy(x)
        g_t = (
            jac_p(self._f, p_t, x_t, self._rev_p)
            if self._jit
            else jac_p_no_jit(self._f, p_t, x_t, self._rev_p)
        )
        return g_t.detach().numpy()

    def jac_x(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        p_t = torch.from_numpy(p)
        x_t = torch.from_numpy(x).requires_grad_(True)
        g_t = (
            jac_x(self._f, p_t, x_t, self._rev_x)
            if self._jit
            else jac_x_no_jit(self._f, p_t, x_t, self._rev_x)
        )
        return g_t.detach().numpy()

    @property
    def f(self) -> Callable[[Tensor, Tensor], Tensor]:
        return self._f


class Exponential(ToM):
    """
    The exponential model function.
    """

    def __init__(self):
        def f(p, x):
            """The linear function."""
            a, b, c = p
            return a * torch.exp(b * x) + c

        super().__init__(f)

    def estimate(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        preset: str | None = None,
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
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        preset: str | None = None,
    ) -> np.ndarray:
        return np.array([1.0, 0.0])
