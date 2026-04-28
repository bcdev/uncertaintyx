#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any
from typing import Callable

import numpy as np


class F(ABC):
    r"""
    Interface for a differentiable function that is mappable
    over a batch dimension.

    Declares a function

    .. math::
        f: \mathbb{R}^{m} \to \mathbb{R}^{n}, \quad
        x \mapsto f(x)

    where :math:`m, n` are shapes (natural numbers or tuples of
    natural numbers). The function can be applied to a batch of
    input samples :math:`X \in \mathbb{R}^{M \times m}` over the
    outer batch dimension :math:`M` to produce a corresponding
    batch of output samples :math:`Y \in \mathbb{R}^{M \times n}`.

    By definition of this interface :math:`f` is separable along
    the batch dimension, i.e., an output sample :math:`y_i` depends
    only on the corresponding input sample :math:`x_i`.
    """

    @abstractmethod
    def eval(self, x: np.ndarray) -> np.ndarray:
        r"""
        Evaluates :math:`f(X)`.

        :param x: :math:`X \in \mathbb{R}^{M \times m}`.
        :returns: :math:`Y \in \mathbb{R}^{M \times n}`.
        """

    @abstractmethod
    def jac(self, x: np.ndarray) -> np.ndarray:
        r"""
        Evaluates the Jacobian :math:`(G_x f)(X)`.

        :param x: :math:`X \in \mathbb{R}^{M \times m}`.
        :returns: :math:`(G_x f)(X) \in \mathbb{R}^{M \times n \times m}`.
        """

    def lpu(
        self, x: np.ndarray, u: np.ndarray, diag: bool = False
    ) -> np.ndarray:
        r"""
        Propagates the input uncertainty tensor :math:`U(X)`
        through the Jacobian, producing the output uncertainty
        tensor :math:`U(Y)`.

        The uncertainty tensor may provide either full covariance,
        i.e., :math:`U(X) \in \mathbb{R}^{M \times m \times m}` or
        only variance, i.e., `U(X) \in \mathbb{R}^{M \times m}`.
        In both cases, the method propagates the input uncertainty
        tensor through the Jacobian, producing the output uncertainty
        tensor :math:`U(Y) \in \mathbb{R}^{M \times n \times n}`.

        Clients must override this method with a tensor-free
        implementation for large-scale problems.

        In what follows, a tensor space
        :math:`\mathbb{R}^{M \times \cdots \times m}` either denotes
        :math:`\mathbb{R}^{M \times m \times m}` or
        :math:`\mathbb{R}^{M \times m}` and no other shapes.

        :param x: :math:`X \in \mathbb{R}^{M \times m}`.
        :param u: :math:`U(X) \in \mathbb{R}^{M \times \cdots \times m}`.
        :param diag: To return only the diagonal elements of :math:`U(Y)`.
        :returns: :math:`U(Y) \in \mathbb{R}^{M \times n \times n}`.
        """
        return lpu_x(x.ndim - 1, self.jac(x), u, diag)

    @property
    @abstractmethod
    def f(self) -> Callable[[Any], Any]:
        r"""
        Returns the native function
        :math:`f: \mathbb{R}^{m} \to \mathbb{R}^{n}`.
        """


class M(ABC):
    r"""
    Interface for a differentiable model function that is mappable
    over a batch dimension.

    Declares a function

    .. math::
        f: \mathbb{R}^{k} \times \mathbb{R}^{m} \to
        \mathbb{R}^{n}, \quad (p, x) \mapsto f(p, x)

    where :math:`k, m, n` are shapes (natural numbers or tuples
    of natural numbers) and :math:`p \in \mathbb{R}^{k}` are model
    parameters. The function can be applied to a batch of input
    samples :math:`X \in \mathbb{R}^{M \times m}` over the outer
    batch dimension :math:`M` to produce a corresponding batch of
    output samples :math:`Y \in \mathbb{R}^{M \times n}`.

    By definition of this interface :math:`f` is separable along
    the batch dimension, i.e., an output sample :math:`y_i` depends
    only on the corresponding input sample :math:`x_i`.
    """

    @abstractmethod
    def eval(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        r"""
        Evaluates :math:`f(p, X)`.

        :param p: :math:`p \in \mathbb{R}^{k}`.
        :param x: :math:`X \in \mathbb{R}^{M \times m}`.
        :returns: :math:`Y \in \mathbb{R}^{M \times n}`.
        """

    @abstractmethod
    def jac_p(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        r"""
        Evaluates the Jacobian :math:`(G_p f)(p, X)` with respect
        to the model parameters.

        :param p: :math:`p \in \mathbb{R}^{k}`.
        :param x: :math:`X \in \mathbb{R}^{M \times m}`.
        :returns: :math:`(G_p f)(p, X) \in \mathbb{R}^{M \times n \times k}`.
        """

    @abstractmethod
    def jac_x(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        r"""
        Evaluates the Jacobian :math:`(G_x f)(p, X)` with respect
        to its input.

        :param p: :math:`p \in \mathbb{R}^{k}`.
        :param x: :math:`X \in \mathbb{R}^{M \times m}`.
        :returns: :math:`(G_x f)(p, X) \in \mathbb{R}^{M \times n \times m}`.
        """

    def lpu_p(
        self, p: np.ndarray, u: np.ndarray, x: np.ndarray, diag: bool = False
    ) -> np.ndarray:
        r"""
        Propagates the parameter uncertainty tensor :math:`U(p)`
        through the Jacobian, producing the output uncertainty
        tensor :math:`U(Y)`.

        Clients must override this method with a tensor-free
        implementation for large-scale problems.

        :param p: :math:`p \in \mathbb{R}^{k}`.
        :param u: :math:`U(p) \in \mathbb{R}^{k \times k}`.
        :param x: :math:`X \in \mathbb{R}^{M \times m}`.
        :param diag: To return only the diagonal elements of :math:`U(Y)`.
        :returns: :math:`U(Y) \in \mathbb{R}^{M \times n \times n}`.
        """
        return lpu_p(p.ndim, self.jac_p(p, x), u, diag)

    def lpu_x(
        self, p: np.ndarray, x: np.ndarray, u: np.ndarray, diag: bool = False
    ) -> np.ndarray:
        r"""
        Propagates the input uncertainty tensor :math:`U(X)`
        through the Jacobian, producing the output uncertainty
        tensor :math:`U(Y)`.

        The uncertainty tensor may provide either full covariance,
        i.e., :math:`U(X) \in \mathbb{R}^{M \times m \times m}` or
        only variance, i.e., `U(X) \in \mathbb{R}^{M \times m}`.
        In both cases, the method propagates the input uncertainty
        tensor through the Jacobian, producing the output uncertainty
        tensor :math:`U(Y) \in \mathbb{R}^{M \times n \times n}`.

        Clients must override this method with a tensor-free
        implementation for large-scale problems.

        In what follows, a tensor space
        :math:`\mathbb{R}^{M \times \cdots \times m}` either denotes
        :math:`\mathbb{R}^{M \times m \times m}` or
        :math:`\mathbb{R}^{M \times m}` and no other shapes.

        :param p: :math:`p \in \mathbb{R}^{k}`.
        :param x: :math:`X \in \mathbb{R}^{M \times m}`.
        :param u: :math:`U(X) \in \mathbb{R}^{M \times \cdots \times m}`.
        :param diag: To return only the diagonal elements of :math:`U(Y)`.
        :returns: :math:`U(Y) \in \mathbb{R}^{M \times n \times n}`.
        """
        return lpu_x(x.ndim - 1, self.jac_x(p, x), u, diag)

    @abstractmethod
    def prior(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        preset: str | None = None,
    ) -> np.ndarray:
        r"""
        Returns a prior estimate of the model parameters.

        :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :param preset: The name of a specific parameter preset.
        :returns: The prior estimate :math:`\check{p} \in \mathbb{R}^{k}`.
        """

    @property
    @abstractmethod
    def f(self) -> Callable[[Any, Any], Any]:
        r"""
        Returns the native function
        :math:`f: \mathbb{R}^{k} \times \mathbb{R}^{m} \to \mathbb{R}^{n}`.
        """


class Estimate(Mapping):
    r"""
    The estimate interface.

    An estimate is the result of an optimal  parameter fitting
    or retrieving estimation.

    Under the same notation and remarks as :class:`F` let

    .. math::
        f: \mathbb{R}^{M \times m} \to \mathbb{R}^{M \times n},
        \quad (X) \mapsto f(X)

    be a differentiable function. Then the result of an estimation
    includes:

    - The irreducible residual variance
      :math:`u^{2}(\zeta) \in \mathbb{R}^{M \times n}` of
      residuals :math:`\zeta \in \mathbb{R}^{M \times n}` with
      :math:`\nu \in \mathbb{N}` estimate-specific residual
      degrees of freedom.
    - The value of the objective function at its minimum.
    - The exit status of the estimation, a nonzero value
    indicating failure.

    Keyword-only arguments may be used to include custom properties
    with the estimate.
    """

    def __init__(
        self, zvar: np.ndarray, cost: np.floating, info: int, **kwargs
    ):
        """
        Creates a new instance of this class.

        :param zvar: The irreducible residual variance.
        :param cost: The value of the objective function at its minimum.
        :param info: The exit status of the estimation.
        """
        self._properties: dict[str, Any] = {
            "zvar": zvar,
            "cost": cost,
            "info": info,
        }
        self._properties |= {
            k: v for k, v in kwargs.items() if k not in self._properties
        }

    def __getitem__(self, key, /):
        return self._properties.__getitem__(key)

    def __len__(self):
        return self._properties.__len__()

    def __iter__(self):
        return self._properties.__iter__()

    @property
    def info(self) -> int:
        """
        Returns the exit status of the estimation, a nonzero value
        indicating failure.
        """
        return self.get("info")

    @property
    def cost(self) -> np.floating:
        """
        Returns the value of the objective function at its minimum.

        The standard convention in maximum likelihood and inverse
        problem theory implies, that the minimum cost is equal to
        half the degrees of freedom of the fitting problem, for a
        closed uncertainty budget.
        """
        return self.get("cost")

    @property
    def zvar(self) -> np.ndarray:
        """
        Returns the irreducible residual variance.
        """
        return self.get("zvar")


class Fit(Estimate):
    r"""
    The result of a model parameter fitting.

    Under the same notation and remarks as :class:`M` let

    .. math::
        f: \mathbb{R}^{k} \times \mathbb{R}^{M \times m} \to
        \mathbb{R}^{M \times n}, \quad (p, X) \mapsto f(X)

    be a differentiable model function. Then the result of a model
    fitting includes:

    - The model function :math:`f` itself.
    - The posterior parameter values :math:`\hat{p} \in \mathbb{R}^{k}`.
    - The posterior uncertainty tensor
      :math:`U(\hat{p}) \in \mathbb{R}^{k \times k}`.
    - The posterior standard uncertainty
      :math:`u(\hat{p}) \in \mathbb{R}^{k}`.
    - The irreducible residual variance
      :math:`u^{2}(Z) \in \mathbb{R}^{n}` with residuals
      :math:`Z = f(\hat{p}, X) - Y \in \mathbb{R}^{n}` and
      :math:`M - \|k\|` residual degrees of freedom.
    - The value of the objective function at its minimum.
    - The exit status, a nonzero value indicating failure.

    Keyword-only arguments may be used to include custom properties
    with the result. Besides properties, the class provides functions
    to propagate uncertainty.
    """

    def __init__(
        self,
        f: M,
        popt: np.ndarray,
        pcov: np.ndarray,
        punc: np.ndarray,
        zvar: np.ndarray,
        cost: Any,
        info: int,
        **kwargs,
    ):
        """
        Creates a new instance of this class.

        Keyword-only arguments may be used to include custom properties
        with the result.

        :param f: The model function.
        :param popt: The posterior parameter values.
        :param pcov: The posterior uncertainty tensor.
        :param punc: The posterior standard uncertainty.
        :param zvar: The irreducible residual variance.
        :param cost: The value of the objective function at its minimum.
        :param info: The exit status.
        """
        super().__init__(
            zvar, cost, info, popt=popt, pcov=pcov, punc=punc, **kwargs
        )
        self._f = f

    @property
    def popt(self) -> np.ndarray:
        r"""
        Returns the posterior parameter values
        :math:`\hat{p} \in \mathbb{R}^{k}`.
        """
        return self.get("popt")

    @property
    def pcov(self) -> np.ndarray:
        r"""
        Returns posterior uncertainty tensor
        :math:`U(\hat{p}) \in \mathbb{R}^{k \times k}`.
        """
        return self.get("pcov")

    @property
    def punc(self) -> np.ndarray:
        r"""
        Returns the posterior standard uncertainties
        :math:`u(\hat{p}) \in \mathbb{R}^{k}`.
        """
        return self.get("punc")

    def f(self, x: np.ndarray) -> np.ndarray:
        r"""
        Evaluates the fitted model function.

        Let :math:`M` denote the number of input samples and let
        :math:`m` and :math:`n` be arbitrary shapes, then:

        :param x: :math:`X \in \mathbb{R}^{M \times m}`.
        :returns: :math:`Y \in \mathbb{R}^{M \times n}`.
        """
        return self._f.eval(self.popt, x)

    def ycov_p(self, x: np.ndarray) -> np.ndarray:
        r"""
        Evaluates the uncertainty tensor of the fitted
        model function values due to the uncertainty of
        posterior parameters.

        Under the same notation as :meth:`f`:

        :param x: :math:`X \in \mathbb{R}^{M \times m}`.
        :returns: :math:`U(Y) \in \mathbb{R}^{M \times n \times n}`.
        """
        return self._f.lpu_p(self.popt, self.pcov, x)

    def yunc_p(self, x: np.ndarray) -> np.ndarray:
        r"""
        Evaluates the standard uncertainty of the fitted
        model function values due to the uncertainty of
        posterior parameters.

        Under the same notation as :meth:`f`:

        :param x: :math:`X \in \mathbb{R}^{M \times m}`.
        :returns: :math:`u(Y) \in \mathbb{R}^{M \times n}`.
        """
        return np.sqrt(self.yvar_p(x))

    def yvar_p(self, x: np.ndarray) -> np.ndarray:
        r"""
        Evaluates the variance of the fitted model function values
        due to the uncertainty of posterior parameters.

        Under the same notation as :meth:`f`:

        :param x: :math:`X \in \mathbb{R}^{M \times m}`.
        :returns: :math:`u^{2}(Y) \in \mathbb{R}^{M \times n}`.
        """
        return self._f.lpu_p(self.popt, self.pcov, x, True)

    def ycov_x(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        r"""
        Evaluates the uncertainty tensor of the fitted
        model function values due to the uncertainty of
        inputs.

        The uncertainty tensor may provide either full covariance,
        i.e., :math:`U(X) \in \mathbb{R}^{M \times m \times m}` or
        only variance, i.e., `U(X) \in \mathbb{R}^{M \times m}`.
        In both cases, the method propagates the input uncertainty
        tensor through the Jacobian, producing the output uncertainty
        tensor :math:`U(Y) \in \mathbb{R}^{M \times n \times n}`.

        In what follows, a tensor space
        :math:`\mathbb{R}^{M \times \cdots \times m}` either denotes
        :math:`\mathbb{R}^{M \times m \times m}` or
        :math:`\mathbb{R}^{M \times m}` and no other shapes.

        Otherwise, under the same notation as :meth:`f`:

        :param x: :math:`X \in \mathbb{R}^{M \times m}`.
        :param u: :math:`U(X) \in \mathbb{R}^{M \times \cdots \times m}`.
        :returns: :math:`U(Y) \in \mathbb{R}^{M \times n \times n}`.
        """
        return self._f.lpu_x(self.popt, x, u)

    def yunc_x(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        r"""
        Evaluates the standard uncertainty of the fitted
        model function values due to the uncertainty of
        inputs.

        Under the same notation as :meth:`f` and :meth:`ycov_x`:

        :param x: :math:`X \in \mathbb{R}^{M \times m}`.
        :param u: :math:`U(X) \in \mathbb{R}^{M \times \cdots \times m}`.
        :returns: :math:`u(Y) \in \mathbb{R}^{M \times n}`.
        """
        return np.sqrt(self.yvar_x(x, u))

    def yvar_x(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        r"""
        Evaluates the variance of the fitted model function
        values due to the uncertainty of inputs.

        Under the same notation as :meth:`f` and :meth:`ycov_x`:

        :param x: :math:`X \in \mathbb{R}^{M \times m}`.
        :param u: :math:`U(X) \in \mathbb{R}^{M \times \cdots \times m}`.
        :returns: :math:`u^{2}(Y) \in \mathbb{R}^{M \times n}`.
        """
        return self._f.lpu_x(self.popt, x, u, True)

    def yunc_t(
        self, x: np.ndarray, u: np.ndarray | None = None
    ) -> np.ndarray:
        r"""
        Evaluates the total standard uncertainty of the fitted
        model function values due to the uncertainty of posterior
        parameters, inputs, and residual variance.

        Under the same notation as :meth:`f` and :meth:`ycov_x`:

        :param x: :math:`X \in \mathbb{R}^{M \times m}`.
        :param u: :math:`U(X) \in \mathbb{R}^{M \times \cdots \times m}`.
        :returns: :math:`u(Y) \in \mathbb{R}^{M \times n}`.
        """
        return np.sqrt(self.yvar_t(x, u))

    def yvar_t(
        self, x: np.ndarray, u: np.ndarray | None = None
    ) -> np.ndarray:
        r"""
        Evaluates the total variance the fitted model function
        values due to the uncertainty of model parameters and
        inputs, and residual variance.

        Under the same notation as :meth:`f` and :meth:`ycov_x`:

        :param x: :math:`X \in \mathbb{R}^{M \times m}`.
        :param u: :math:`U(X) \in \mathbb{R}^{M \times \cdots \times m}`.
        :returns: :math:`u^{2}(Y) \in \mathbb{R}^{M \times n}`.
        """
        return (
            self.yvar_p(x)
            + (self.yvar_x(x, u) if u is not None else 0.0)
            + self.zvar
        )


class Fitting(ABC):
    """The fitting interface."""

    @abstractmethod
    def fit(self, f: M, x: np.ndarray, y: np.ndarray, **kwargs) -> Fit:
        r"""
        Fits the parameters of a model function to :math:`M`
        samples :math:`(x_i, y_i)` of data.

        Concrete implementations of :class:`Fitting` may accept
        keyword-only parameters for standard uncertainties:

            fit(f, x, y, *, u: np.ndarray, **kwargs)
            fit(f, x, y, *, ux: np.ndarray, **kwargs)
            fit(f, x, y, *, uy: np.ndarray, **kwargs)
            fit(f, x, y, *, ux: np.ndarray, uy: np.ndarray, **kwargs)

        Under the same notation and remarks as :class:`M`:

        :param f: The model function.
        :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :returns: The fit result.
        """


class Retrieval(Estimate):
    r"""
    The result of a retrieval.

    Under the same notation and remarks as :class:`F` let

    .. math::
        f: \mathbb{R}^{M \times m} \to \mathbb{R}^{M \times n},
        \quad (X) \mapsto f(X)

    be a differentiable function. Then the result of a retrieval
    includes:

    - The posterior estimate :math:`\hat{X} \in \mathbb{R}^{M \times m}`.
    - The posterior standard uncertainty
      :math:`u(\hat{X}) \in \mathbb{R}^{M \times m}`.
    - The posterior uncertainty tensor
      :math:`U(\hat{X}) \in \mathbb{R}^{M \times m \times m}`.
    - The irreducible residual variance
      :math:`u^{2}(Z) \in \mathbb{R}^{M \times n}` of residuals
      :math:`Z = f(\hat{X}) - Y \in \mathbb{R}^{M \times n}` with
      :math:`M(n - m)` residual degrees of freedom.
    - The value of the objective function at its minimum.
    - The exit status, a nonzero value indicating failure.

    Keyword-only arguments may be used to include custom properties
    with the retrieval.
    """

    def __init__(
        self,
        xopt: np.ndarray,
        xunc: np.ndarray,
        xcov: np.ndarray,
        zvar: np.ndarray,
        cost: np.floating,
        info: int,
        **kwargs,
    ):
        """
        Creates a new instance of this class.

        :param xopt: The posterior estimate.
        :param xcov: The posterior uncertainty tensor.
        :param xunc: The posterior uncertainty.
        :param zvar: The irreducible residual variance.
        :param cost: The value of the objective function at its minimum.
        :param info: The exit status.
        """
        super().__init__(
            zvar, cost, info, xopt=xopt, xcov=xcov, xunc=xunc, **kwargs
        )

    @property
    def xopt(self) -> np.ndarray:
        r"""
        Returns the posterior estimate
        :math:`\hat{X} \in \mathbb{R}^{M \time m}`.
        """
        return self.get("xopt")

    @property
    def xcov(self) -> np.ndarray:
        r"""
        Returns posterior uncertainty tensor
        :math:`U(\hat{X}) \in \mathbb{R}^{M \times m \times m}`.
        """
        return self.get("xcov")

    @property
    def xunc(self) -> np.ndarray:
        r"""
        Returns the posterior standard uncertainties
        :math:`u(\hat{X}) \in \mathbb{R}^{M \time m}`.
        """
        return self.get("xunc")


class Retrieving(ABC):
    """The retrieving interface."""

    @abstractmethod
    def retrieve(
        self, f: F, x: np.ndarray, y: np.ndarray, **kwargs
    ) -> Retrieval:
        r"""
        Solves an inverse problem of the form :math:`f(x) = y` for
        :math:`M` samples :math:`(\check{x}_i, y_i)` of data, where
        :math:`\check{x}_i` are prior estimates of the unknown
        solution :math:`\hat{x}_i`.

        Concrete implementations of :class:`Retrieving` may accept
        keyword-only parameters for standard uncertainties:

            retrieve(f, x, y, *, u: np.ndarray, **kwargs)
            retrieve(f, x, y, *, ux: np.ndarray, **kwargs)
            retrieve(f, x, y, *, uy: np.ndarray, **kwargs)
            retrieve(f, x, y, *, ux: np.ndarray, uy: np.ndarray, **kwargs)

        Under the same notation and remarks as :class:`F`:

        :param f: The function.
        :param x: Samples :math:`\check{X} \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :returns: The retrieval result.
        """


class Perturbing(ABC):
    """
    Perturbs samples of the independent and dependent variables
    with random errors complying with their associated standard
    uncertainty.
    """

    @abstractmethod
    def perturb_x(self, x: np.ndarray, u: np.ndarray, **kwargs) -> np.ndarray:
        r"""
        Perturbs :math:`M` samples of the independent variable
        :math:`x \in \mathbb{R}^{m}`.

        :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
        :param u: Standard uncertainties :math:`u(X)`.
        :returns: Perturbed samples :math:`X' \in \mathbb{R}^{M \times m}`.
        """

    @abstractmethod
    def perturb_y(self, y: np.ndarray, u: np.ndarray, **kwargs) -> np.ndarray:
        r"""
        Perturbs :math:`M` samples of the dependent variable
        :math:`y \in \mathbb{R}^{n}`.

        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :param u: Standard uncertainties :math:`u(Y)`.
        :returns: Perturbed samples :math:`Y' \in \mathbb{R}^{M \times n}`.
        """


def lpu_p(
    d: int, g: np.ndarray, u: np.ndarray, diag: bool = False
) -> np.ndarray:
    r"""
    Default implementation of the law of propagation of uncertainty
    in general tensor form (for parameter uncertainty tensors).

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

    Otherwise, under the same notation as :meth:`jac_p`:

    :param d: The number of inner tensor dimensions.
    :param g: Jacobian :math:`G \in \mathbb{R}^{M \times \cdots \times D}`.
    :param u: Tensor :math:`U \in \mathbb{R}^{\cdots \times D}`.
    :param diag: To return only the diagonal elements of :math:`V`.
    :returns: Tensor :math:`V \in \mathbb{R}^{M \times \cdots}`.
    """
    lpu = make_lpu(d, diag)
    return np.stack([lpu(g_, u) for g_ in g])


def lpu_x(
    d: int, g: np.ndarray, u: np.ndarray, diag: bool = False
) -> np.ndarray:
    r"""
    Default implementation of the law of propagation of uncertainty
    in general tensor form (for input uncertainty tensors)

    Using Einstein's summation convention and the symmetry of the
    input uncertainty tensor :math:`U`:, the output uncertainty
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

    Otherwise, under the same notation as :meth:`jac_p`:

    :param d: The number of inner tensor dimensions.
    :param g: Jacobian :math:`G \in \mathbb{R}^{M \times \cdots \times D}`.
    :param u: Tensor :math:`U \in \mathbb{R}^{M \times \cdots \times D}`.
    :param diag: To return only the diagonal elements of :math:`V`.
    :returns: Tensor :math:`V \in \mathbb{R}^{M \times \cdots}`.
    """
    lpu = make_lpu(d, diag)
    return np.stack([lpu(g_, u_) for g_, u_ in zip(g, u)])


def make_lpu(
    d: int, diag: bool = False
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns the law of propagation of uncertainty.

    :param d: The number of inner tensor dimensions.
    :param diag: To return only the diagonal elements.
    :returns: The law of propagation of uncertainty.
    """

    def lpu(g: np.ndarray, u: np.ndarray) -> np.ndarray:
        """The law of propagation of uncertainty."""
        dims = tuple(range(-d, 0))
        gu = np.tensordot(g, u, (dims, dims)) if u.ndim != d else g * u
        return (
            np.tensordot(gu, g, (dims, dims))
            if not diag
            else np.sum(gu * g, dims)
        )

    return lpu
