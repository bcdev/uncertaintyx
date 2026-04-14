#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
"""
This module provides regression methods.
"""

import numpy as np

from ..interface.core import Fitting
from ..interface.core import M
from ..interface.core import Result


class HomoscedasticRegression(Fitting):
    """
    Unit variance errors‑in‑variables regression.

    Use this regression when uncertainties of both x and y are
    unknown. Use the bootstrap method to propagate uncertainty
    of x and y to model parameters.
    """

    def __init__(self, eiv: Fitting):
        """
        Creates a new regression instance.

        :param eiv: The errors‑in‑variables method.
        """
        self._eiv = eiv

    def fit(
        self, f: M, x: np.ndarray, y: np.ndarray, *, max_iter: int = 100
    ) -> Result:
        r"""
        Fits the parameters of a model function to :math:`M`
        samples :math:`(x_i, y_i) \in \mathbb{R}^{m \times n}`
        of data.

        :param f: The model function.
        :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :param max_iter: The maximum number of iterations conducted.
        :returns: The fit result.
        """
        return self._eiv.fit(f, x, y)


class HomoHeteroscedasticRegression(Fitting):
    """
    Mixed errors‑in‑variables regression.

    Use this regression when uncertainties of x are unknown
    but uncertainties of y are known. Use the bootstrap method
    to propagate uncertainty of x and y to model parameters.
    """

    def __init__(self, eiv: Fitting):
        """
        Creates a new regression instance.

        :param eiv: The errors‑in‑variables method.
        """
        self._eiv = eiv

    def fit(  # noqa
        self,
        f: M,
        x: np.ndarray,
        y: np.ndarray,
        *,
        u: np.ndarray,
        max_iter: int = 100,
    ) -> Result:
        r"""
        Fits the parameters of a model function to :math:`M`
        samples :math:`(x_i, y_i)` of data.

        :param f: The model function.
        :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :param u: Uncertainties :math:`u(X) \in \mathbb{R}^{M \times m}`.
        :param max_iter: The maximum number of iterations conducted.
        :returns: The fit result.
        """
        return self._eiv.fit(f, x, y, uy=u)


class HeteroHomoscedasticRegression(Fitting):
    """
    Mixed errors‑in‑variables regression.

    Use this regression when uncertainties of x are known but
    uncertainties of y are unknown. Use the bootstrap method
    to propagate uncertainty of x and y to model parameters.
    """

    def __init__(self, eiv: Fitting):
        """
        Creates a new regression instance.

        :param eiv: The errors‑in‑variables method.
        """
        self._eiv = eiv

    def fit(  # noqa
        self,
        f: M,
        x: np.ndarray,
        y: np.ndarray,
        *,
        u: np.ndarray,
        max_iter: int = 100,
    ) -> Result:
        r"""
        Fits the parameters of a model function to :math:`M`
        samples :math:`(x_i, y_i)` of data.

        :param f: The model function.
        :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :param u: Uncertainties :math:`u(Y) \in \mathbb{R}^{M \times n}`.
        :param max_iter: The maximum number of iterations conducted.
        :returns: The fit result.
        """
        return self._eiv.fit(f, x, y, ux=u)


class HeteroscedasticRegression(Fitting):
    """
    General errors‑in‑variables regression.

    Use this regression when uncertainties of both x and y are
    known. Use the Monte Carlo method to propagate uncertainty
    of x and y to model parameters when the model function is
    significantly nonlinear within the range of errors or the
    error distribution is significantly nonnormal.
    """

    def __init__(self, eiv: Fitting):
        """
        Creates a new regression instance.

        :param eiv: The errors‑in‑variables method.
        """
        self._eiv = eiv

    def fit(  # noqa
        self,
        f: M,
        x: np.ndarray,
        y: np.ndarray,
        *,
        ux: np.ndarray,
        uy: np.ndarray,
        max_iter: int = 100,
    ) -> Result:
        r"""
        Fits the parameters of a model function to :math:`M`
        samples :math:`(x_i, y_i)` of data.

        :param f: The model function.
        :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :param ux: Uncertainties :math:`u(X) \in \mathbb{R}^{M \times m}`.
        :param uy: Uncertainties :math:`u(Y) \in \mathbb{R}^{M \times n}`.
        :param max_iter: The maximum number of iterations conducted.
        :returns: The fit result.
        """
        return self._eiv.fit(f, x, y, ux=ux, uy=uy)
