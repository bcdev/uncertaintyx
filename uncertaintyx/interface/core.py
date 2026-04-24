#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
from abc import ABC
from abc import abstractmethod
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
    def estimate(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        preset: str | None = None,
    ) -> np.ndarray:
        r"""
        Returns an initial estimate of the model parameters.

        :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :param preset: The name of a specific parameter preset.
        :returns: An initial estimate of :math:`p \in \mathbb{R}^{k}`.
        """

    @property
    @abstractmethod
    def f(self) -> Callable[[Any, Any], Any]:
        r"""
        Returns the native function
        :math:`f: \mathbb{R}^{k} \times \mathbb{R}^{m} \to \mathbb{R}^{n}`.
        """


class Result:
    r"""
    The result of a model fitting.

    Under the same notation and remarks as :class:`M` let

    .. math::
        f: \mathbb{R}^{k} \times \mathbb{R}^{M \times m} \to
        \mathbb{R}^{M \times n}, \quad (p, X) \mapsto f(X)

    be a differentiable model function. Then the result of a model
    fitting includes:

    - The model function :math:`f` itself
    - The optimized model parameter values :math:`p \in \mathbb{R}^{k}`
    - The standard uncertainty of the optimized parameter values
      :math:`u(p) \in \mathbb{R}^{k}`.
    - The uncertainty tensor of the optimized parameter values
      :math:`U(p) \in \mathbb{R}^{k \times k}`
    - The irreducible residual variance
      :math:`u^{2}(Z) \in \mathbb{R}^{n}` with residuals
      :math:`Z = f(p, X) - Y \in \mathbb{R}^{n}` and
      :math:`M - \|k\|` degrees of freedom.
    - The value of the objective function at its minimum.
    - The exit status, a nonzero value indicating failure

    Besides these properties, the class provides functions to
    propagate uncertainties.
    """

    def __init__(
        self,
        f: M,
        popt: np.ndarray,
        punc: np.ndarray,
        pcov: np.ndarray,
        zvar: np.ndarray,
        cost: Any,
        info: int,
        **kwargs,
    ):
        r"""
        Creates a new instance of this class.

        Keyword-only arguments may be used to include custom properties
        with the result.

        :param f: The model function.
        :param popt: The optimized model parameter values.
        :param punc: The uncertainty of the optimized model
        parameter values.
        :param pcov: The uncertainty tensor of the optimized model
        parameter values.
        :param zvar: The irreducible residual variance.
        :param cost: The value of the objective function at its minimum.
        :param info: The exit status, a nonzero value indicating failure.
        """
        self._f = f
        self._popt = popt
        self._punc = punc
        self._pcov = pcov
        self._zvar = zvar
        self._cost = cost
        self._info = info
        self._properties: dict[str, Any] = {}
        self._properties.update(kwargs)

    @property
    def cost(self) -> Any:
        r"""
        Returns the value of the objective function at its minimum.

        The standard convention in maximum likelihood and inverse
        problem theory implies, that the minimum cost is equal to
        half the degrees of freedom of the fitting problem, for a
        closed uncertainty budget.
        """
        return self._cost

    @property
    def info(self) -> int:
        """Returns the exit status."""
        return self._info

    @property
    def popt(self) -> np.ndarray:
        r"""
        Returns the optimized model parameter values
        :math:`p \in \mathbb{R}^{k}`.
        """
        return self._popt

    @property
    def pcov(self) -> np.ndarray:
        r"""
        Returns uncertainty tensor of the optimized model parameter
        values :math:`U(p) \in \mathbb{R}^{k \times k}`.
        """
        return self._pcov

    @property
    def punc(self) -> np.ndarray:
        r"""
        Returns the standard uncertainties of the optimized model
        parameter values :math:`u(p) \in \mathbb{R}^{k}`.
        """
        return self._punc

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
        model parameters.

        Under the same notation as :meth:`f`:

        :param x: :math:`X \in \mathbb{R}^{M \times m}`.
        :returns: :math:`U(Y) \in \mathbb{R}^{M \times n \times n}`.
        """
        return self._f.lpu_p(self.popt, self.pcov, x)

    def yunc_p(self, x: np.ndarray) -> np.ndarray:
        r"""
        Evaluates the standard uncertainty of the fitted
        model function values due to the uncertainty of
        model parameters.

        Under the same notation as :meth:`f`:

        :param x: :math:`X \in \mathbb{R}^{M \times m}`.
        :returns: :math:`u(Y) \in \mathbb{R}^{M \times n}`.
        """
        return np.sqrt(self.yvar_p(x))

    def yvar_p(self, x: np.ndarray) -> np.ndarray:
        r"""
        Evaluates the variance of the fitted model function
        values due to the uncertainty of model parameters.

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
        model function values due to the uncertainty of model
        parameters and the uncertainty of inputs (or residual
        variance).

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
        the uncertainty of inputs (or residual variance).

        Under the same notation as :meth:`f` and :meth:`ycov_x`:

        :param x: :math:`X \in \mathbb{R}^{M \times m}`.
        :param u: :math:`U(X) \in \mathbb{R}^{M \times \cdots \times m}`.
        :returns: :math:`u^{2}(Y) \in \mathbb{R}^{M \times n}`.
        """
        return (
            self.yvar_p(x) + self.zvar if u is None else self.yvar_x(x, u)
        )

    @property
    def zunc(self) -> np.ndarray:
        r"""
        Returns the irreducible residual standard uncertainty
        :math:`u(Z) \in \mathbb{R}^{n}`.
        """
        return np.sqrt(self._zvar)

    @property
    def zvar(self) -> np.ndarray:
        r"""
        Returns the irreducible residual variance
        :math:`u^{2}(Z) \in \mathbb{R}^{n}`.
        """
        return self._zvar

    def property(self, name: str) -> Any:
        """
        Returns the value of a property.

        :param name: The name of the property.
        :returns: The value of the property or `None` if the property
        is not defined.
        """
        return self._properties.get(name, None)


class Fitting(ABC):
    """The fitting interface."""

    @abstractmethod
    def fit(self, f: M, x: np.ndarray, y: np.ndarray, **kwargs) -> Result:
        r"""
        Fits the parameters of a model function to :math:`M`
        samples :math:`(x_i, y_i)` of data.

        Concrete implementations of :class:`Fitting` supplied as
        argument may accept keyword-only parameters for propagation
        of standard uncertainties::

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
