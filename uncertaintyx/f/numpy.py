#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
"""
Generic functions.
"""

from abc import ABC
from typing import Callable

import numpy as np

from ..interface.core import F


class ToF(F, ABC):
    r"""
    Adapts a pure function

    .. math::
        f: \mathbb{R}^{m} \to \mathbb{R}^{n}, \quad
        (x) \mapsto f(x)

    where :math:`m, n` are shapes (natural numbers or tuples
    of natural numbers) to the function interface ``F``.
    """

    def __init__(self, f: Callable[[np.ndarray], np.ndarray]):
        """
        Creates a new instance of this class.

        :param f: The function :math:`f`.
        """
        self._f = f

    @property
    def f(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._f
