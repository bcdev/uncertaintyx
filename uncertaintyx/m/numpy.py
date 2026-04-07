#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
"""
Generic empirical model functions.
"""

from abc import ABC
from typing import Callable

import numpy as np

from ..interface.core import M


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

    def __init__(self, f: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        """
        Creates a new instance of this class.

        :param f: The function :math:`f`.
        """
        self._f = f

    def eval(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        r"""
        Evaluates :math:`f(p, X)`, where :math:`X` is a batch
        of inputs, under the assumption that :math:`f` can be
        applied directly to a batch.

        Clients must override this method unless this assumption
        holds.
        """
        return self._f(p, x)

    @property
    def f(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        return self._f


class Exponential(ToM):
    """
    The exponential model function.
    """

    def __init__(self):
        def f(p, x):
            """The exponential function."""
            a, b, c = p
            return a * np.exp(b * x) + c

        super().__init__(f)

    def jac_p(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        a, b, _ = p
        term = np.exp(b * x)
        return np.stack([term, a * x * term, np.ones_like(x)], axis=-1)

    def jac_x(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        a, b, _ = p
        return a * b * np.exp(b * x)

    def estimate(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        **kwargs,
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

    def jac_p(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.stack([x, np.ones_like(x)], axis=-1)

    def jac_x(self, p, x: np.ndarray) -> np.ndarray:
        a, _ = p
        return np.full_like(x, a)

    def estimate(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        return np.array([1.0, 0.0])
