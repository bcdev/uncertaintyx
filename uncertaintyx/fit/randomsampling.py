#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
"""
This module provides random sampling techniques like bootstrap
and Monte Carlo.
"""

from typing import Any

import numpy as np

from ..interface.core import Fitting
from ..interface.core import M
from ..interface.core import Perturbing
from ..interface.core import Result


class Bootstrap(Fitting):
    """
    The bootstrap method to fit a model function to x and y
    data with unknown (or crudely known) uncertainties.

    Use the bootstrap method to propagate uncertainties of x and y to
    model parameters through random sampling, e.g., when the model is
    not linear within the range of errors or errors are not normally
    distributed.

    Like Monte Carlo, bootstrapping generates variations of the data
    supplied. While Monte Carlo creates variants by random perturbation,
    bootstrapping creates variants by random sampling with replacement,
    resampling uncertainties, if known, paired with their corresponding
    data points.
    """

    def __init__(
        self, fitting: Fitting, seed: int = 5489, how_many: int = 1000
    ):
        """
        Creates a new bootstrap instance.

        Concrete implementations of :class:`Fitting` supplied as argument
        may accept keyword-only parameters for uncertainty propagation:

        fit(f, x, y, *, u: np.ndarray, **kwargs)
        fit(f, x, y, *, ux: np.ndarray, **kwargs)
        fit(f, x, y, *, uy: np.ndarray, **kwargs)
        fit(f, x, y, *, ux: np.ndarray, uy: np.ndarray, **kwargs)

        :param fitting: The fitting method.
        :param seed: The random seed used for the bootstrap.
        :param how_many: The number of bootstrap iterations.
        """
        self._fitting = fitting
        self._rng = np.random.default_rng(seed)
        self._how_many = how_many

    def fit(
        self,
        f: M,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ):
        r"""
        Fits the parameters of a model function to :math:`M`
        samples :math:`(x_i, y_i)` of data.

        :param f: The model function.
        :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :returns: The fit result.
        """

        def resampled_kwargs() -> dict[str, Any]:
            """
            Returns keyword-only arguments with uncertainty resampled,
            if applicable.
            """
            resampled = kwargs
            for kwarg in ["u", "ux", "uy"]:
                if kwarg in resampled:
                    resampled = {**kwargs, kwarg: kwargs[kwarg][idx]}
            return resampled

        def mean():
            """
            Returns the mean value of optimized model parameters.
            """
            return np.atleast_1d(np.mean(popt, axis=0))

        def rvar():
            """
            Returns the irreducible residual variance associated
            with the dependent variable.
            """
            return np.var(f.eval(mean, x) - y, axis=0, ddof=mean.size)

        popt = np.full((self._how_many, len(f.estimate())), np.nan)
        cost = np.full(self._how_many, np.nan)
        success_count = 0

        for i in range(self._how_many):
            idx = self._rng.choice(len(x), len(x), replace=True)
            res = self._fitting.fit(f, x[idx], y[idx], **resampled_kwargs())
            if not hasattr(res, "info") or res.info == 0:
                popt[i] = res.popt
                cost[i] = res.cost
                success_count += 1
        popt = popt[~np.isnan(popt).any(axis=1)]
        cost = cost[~np.isnan(cost)]
        mean = mean()
        rvar = rvar()

        return Result(
            f,
            popt=mean,
            punc=np.atleast_1d(np.std(popt, axis=0, ddof=1)),
            pcov=np.atleast_2d(np.cov(popt.T, ddof=1)),
            rvar=rvar,
            cost=np.mean(cost),
            info=0 if success_count > self._how_many // 2 else 1,
            cost_uncertainty=np.std(cost),
            success_count=success_count,
        )


class NormalPerturbator(Perturbing):
    """
    Perturbs x and y values with uncorrelated normally distributed errors.
    """

    def __init__(self, seed: int = 5489):
        self.rng = np.random.default_rng(seed)

    def perturb_x(self, x: np.ndarray, u: np.ndarray, **kwargs) -> np.ndarray:
        return x + self.rng.normal(0.0, u, x.shape)

    def perturb_y(self, y: np.ndarray, u: np.ndarray, **kwargs) -> np.ndarray:
        return y + self.rng.normal(0.0, u, y.shape)


class MonteCarlo(Fitting):
    """
    The Monte Carlo method to fit a model function to x and y
    data with known uncertainties.

    Use the Monte Carlo method to propagate uncertainties of x and y to
    model parameters through random perturbation, e.g., when the model is
    non-linear within the range of errors or errors are non-normally
    distributed.
    """

    def __init__(
        self,
        fitting: Fitting,
        perturbator: Perturbing = NormalPerturbator(5489),
        how_many: int = 1000,
    ):
        """
        Creates a new Monte Carlo instance.

        Concrete implementations of :class:`Fitting` supplied as argument
        must accept `ux` and `uy` keyword-only parameters for uncertainty
        propagation:

        fit(f, x, y, *, ux: np.ndarray, uy: np.ndarray, **kwargs)

        :param fitting: The fitting method.
        :param perturbator: To randomly perturb x and y values.
        :param how_many: The number of Monte Carlo simulations.
        """
        self._fitting = fitting
        self._perturbator = perturbator
        self._how_many = how_many

    def fit(  # noqa
        self,
        f: M,
        x: np.ndarray,
        y: np.ndarray,
        *,
        ux: np.ndarray,
        uy: np.ndarray,
        **kwargs,
    ):
        r"""
        Fits the parameters of a model function to :math:`M`
        samples :math:`(x_i, y_i)` of data.

        :param f: The model function.
        :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :param ux: Uncertainties :math:`u(X) \in \mathbb{R}^{M \times m}`.
        :param uy: Uncertainties :math:`u(Y) \in \mathbb{R}^{M \times n}`.
        :returns: The fit result.
        """

        def mean():
            """
            Returns the mean value of optimized model parameters.
            """
            return np.atleast_1d(np.mean(popt, axis=0))

        def rvar():
            """
            Returns the irreducible residual variance associated
            with the dependent variable.
            """
            return np.var(f.eval(mean, x) - y, axis=0, ddof=mean.size)

        popt = np.full((self._how_many, len(f.estimate())), np.nan)
        cost = np.full(self._how_many, np.nan)
        success_count = 0

        for i in range(self._how_many):
            dx = self._perturbator.perturb_x(x, ux, **kwargs)
            dy = self._perturbator.perturb_y(y, uy, **kwargs)
            res = self._fitting.fit(f, x + dx, y + dy, ux=ux, uy=uy, **kwargs)
            if not hasattr(res, "info") or res.info == 0:
                popt[i] = res.popt
                cost[i] = res.cost
                success_count += 1
        popt = popt[~np.isnan(popt).any(axis=1)]
        cost = cost[~np.isnan(cost)]
        mean = mean()
        rvar = rvar()

        return Result(
            f,
            popt=mean,
            punc=np.atleast_1d(np.std(popt, axis=0, ddof=1)),
            pcov=np.atleast_2d(np.cov(popt.T, ddof=1)),
            rvar=rvar,
            cost=np.mean(cost),
            info=0 if success_count > self._how_many // 2 else 1,
            cost_uncertainty=np.std(cost),
            success_count=success_count,
        )
