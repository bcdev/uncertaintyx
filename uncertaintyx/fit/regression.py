#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
"""
This module provides regression methods.
"""

import numpy as np

from ..tyx import Fitted
from ..tyx import Fitting
from ..tyx import M


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

    def fit(self, f: M, x: np.ndarray, y: np.ndarray, **kwargs) -> Fitted:
        r"""
        Fits the parameters of a model function to :math:`M`
        samples :math:`(x_i, y_i) \in \mathbb{R}^{m \times n}`
        of data.

        Under the same notation and remarks as :class:`M`:

        :param f: The model function.
        :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :returns: The fit result.
        """
        return self._eiv.fit(f, x, y, **kwargs)


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
        **kwargs,
    ) -> Fitted:
        r"""
        Fits the parameters of a model function to :math:`M`
        samples :math:`(x_i, y_i)` of data.

        Under the same notation and remarks as :class:`M`:

        :param f: The model function.
        :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :param u: Standard uncertainties :math:`u(X)`.
        :returns: The fit result.
        """
        return self._eiv.fit(f, x, y, uy=u, **kwargs)


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
        **kwargs,
    ) -> Fitted:
        r"""
        Fits the parameters of a model function to :math:`M`
        samples :math:`(x_i, y_i)` of data.

        Under the same notation and remarks as :class:`M`:

        :param f: The model function.
        :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :param u: Standard uncertainties :math:`u(Y)`.
        :returns: The fit result.
        """
        return self._eiv.fit(f, x, y, ux=u, **kwargs)


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
        **kwargs,
    ) -> Fitted:
        r"""
        Fits the parameters of a model function to :math:`M`
        samples :math:`(x_i, y_i)` of data.

        Under the same notation and remarks as :class:`M`:

        :param f: The model function.
        :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :param ux: Standard uncertainties :math:`u(X)`.
        :param uy: Standard uncertainties :math:`u(Y)`.
        :returns: The fit result.
        """
        return self._eiv.fit(f, x, y, ux=ux, uy=uy, **kwargs)
