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


def jac(f: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
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
    return torch.vmap(torch.func.jacrev(f))(x)


def vec(f: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    r"""
    Evaluates :math:`f(X)`.

    Under the same notation as :meth:`jac`:

    :param f: The function :math:`f`.
    :param x: :math:`X \in \mathbb{R}^{M \times m}`.
    :returns: :math:`Y \in \mathbb{R}^{M \times n}`.
    """
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

    def __init__(self, f: Callable[[Tensor], Tensor]):
        """
        Creates a new instance of this class.

        :param f: The function :math:`f`.
        """
        self._f = f

    def eval(self, x: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(x)
        y_t = vec(self._f, x_t)
        return y_t.detach().numpy()

    def jac(self, x: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(x).requires_grad_(True)
        g_t = jac(self._f, x_t)
        return g_t.detach().numpy()

    @property
    def f(self) -> Callable[[Tensor], Tensor]:
        return self._f
