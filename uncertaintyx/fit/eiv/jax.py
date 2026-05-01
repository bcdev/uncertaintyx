#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
"""
Errors-in-variables implementation based on the effective
variance method (EVM). Refer to:

Giering, Quast, Mittaz et al. (2019). A Novel Framework to
Harmononise Satellite Data Series for Climate Applications.
Remote Sens., 11, 1002. https://doi.org/10.3390/rs11091002.

Tarantola (2005). Inverse Problem Theory and Methods for
Model Parameter Estimation. Society for Industrial and Applied
Mathematics. https://doi.org/10.1137/1.9780898717921

Watson et al. (1984). The effective variance weighting
for least squares calculations applied to the mass balance
receptor model. Atmospheric Environment (1967), 18, 1347-1355.
https://doi.org/10.1016/0004-6981(84)90043-X.

D. York (1968). Least squares fitting of a straight line
with correlated errors. Earth and Planetary Science Letters,
5, 320-324. https://doi.org/10.1016/S0012-821X(68)80059-7.

D. York (1966). Least squares fitting of a straight line.
Canadian Journal of Physics, 44, 1079-1086.
https://doi.org/10.1139/p66-090.
"""

from typing import Any
from typing import Callable

import jax
import numpy as np
import optax
import optimistix
from jax import Array
from jax import numpy as jnp
from jax.numpy import linalg as jli
from jax.scipy import linalg as jla

from ...tyx import Fitted
from ...tyx import Fitting
from ...tyx import M

DEFAULT_ATOL: Any = 1.0e-08
"""The absolute tolerance for terminating the optimization."""

DEFAULT_RTOL: Any = 1.0e-06
"""The relative tolerance for terminating the optimization."""

DEFAULT_MAX_STEPS: int = 2048
"""The maximum number of steps the optimizer can take."""


@jax.jit(static_argnums=(0,), static_argnames=("use_covar", "max_steps",))
def _batch(
    f: Callable[[Array, Array], Array],
    p: Array,
    x: Array,
    y: Array,
    ux: Array,
    uy: Array,
    up: Array | None = None,
    *,
    use_covar: bool = False,
    atol: Any = DEFAULT_ATOL,
    rtol: Any = DEFAULT_RTOL,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> tuple[Array, Array, Array, Array, Array]:
    r"""
    Bayesian errors-in-variables optimizer based on the effective
    variance method (EVM).

    The implementation accepts any combination of full-rank
    or diagonal-rank uncertainty tensors:
    :math:`U(X) \in \mathbb{R}^{M \times m \times m}`,
    :math:`U(X) \in \mathbb{R}^{M \times m}`,
    :math:`U(Y) \in \mathbb{R}^{M \times n \times n}`,
    :math:`U(Y) \in \mathbb{R}^{M \times n}`,
    :math:`U(\check{p}) \in \mathbb{R}^{k \times k}`, and
    :math:`U(\check{p}) \in \mathbb{R}^{k}`.

    Standard uncertainty is not accepted and must be squared to
    a variance (diagonal uncertainty tensor) before supplied as
    argument.

    :param f: The model function.
    :param p: Prior parameter values :math:`\check{p} \in \mathbb{R}^{k}`.
    :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
    :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
    :param ux: Uncertainty tensor :math:`U(X)`, full or diagonal.
    :param uy: Uncertainty tensor :math:`U(Y)`, full or diagonal.
    :param up: Uncertainty tensor :math:`U(\check{p})`, full or diagonal.
    :param use_covar: Consider covariance?
    :param atol: The absolute tolerance for terminating the optimization.
    :param rtol: The relative tolerance for terminating the optimization.
    :param max_steps: The maximum number of steps the optimizer can take.
    :returns: The fit result.
    """

    def g(P: Array, x: Array) -> Array:  # noqa : N806
        r"""
        The sample Jacobian.

        :param P: The parameters :math:`p \in \mathbb{R}^{k}`.
        :param x: The sample :math:`x \in \mathbb{R}^{m}`
        :returns: :math:`(G_x f)(q, x) \in \mathbb{R}^{n \times m}`
        """
        return (
            jax.jacrev(f, argnums=1)(P, x)
            if y.size < x.size
            else jax.jacfwd(f, argnums=1)(P, x)
        )

    def upc(d: int, G: Array, U: Array) -> Array:  # noqa: N806
        """
        The sample uncertainty propagation.

        :param d: The rank of the input sample tensor.
        :param G: The sample Jacobian.
        :param U: The input sample uncertainty tensor.
        :returns: The propagated uncertainty tensor.
        """
        dims = tuple(range(-d, 0))
        return jnp.tensordot(
            jnp.tensordot(G, U, axes=(dims, dims)), G, axes=(dims, dims)
        )

    def upd(d: int, G: Array, U: Array) -> Array:  # noqa: N806
        """
        The diagonalized sample uncertainty propagation.

        :param d: The rank of the input sample tensor.
        :param G: The sample Jacobian.
        :param U: The input sample uncertainty tensor.
        :returns: The diagonal of the propagated uncertainty tensor.
        """
        dims = tuple(range(-d, 0))
        return jnp.sum(jnp.tensordot(G, U, axes=(dims, dims)) * G, axis=dims)

    def loss(
        P: Array,  # noqa: N806
        x: Array,
        y: Array,
        ux: Array,
        uy: Array,
    ) -> Array:
        r"""
        The sample loss function.

        :param P: The parameters :math:`p \in \mathbb{R}^{k}`.
        :param x: The sample :math:`x \in \mathbb{R}^{m}`.
        :param y: The sample :math:`y \in \mathbb{R}^{n}`.
        :param ux: :math:`U(x) \in \mathbb{R}^{m \times m}`.
        :param uy: :math:`U(y) \in \mathbb{R}^{n \times n}`.
        :returns: The sample loss.
        """
        d = f(P, x) - y
        G = g(P, x)  # noqa: N806
        if not use_covar:
            U = upd(x.ndim, G, ux) + (  # noqa: N806
                uy
                if uy.ndim == y.ndim
                else jnp.diag(uy.reshape((y.size, -1))).reshape(y.shape)
            )
            b = d / U
        else:
            d = d.reshape(-1)
            U = upc(x.ndim, G, ux).reshape((y.size, -1)) + (  # noqa: N806
                jnp.diag(uy.reshape(-1))
                if uy.ndim == y.ndim
                else uy.reshape((y.size, -1))
            )
            b = jla.cho_solve(jla.cho_factor(U), d)
        return 0.5 * jnp.sum(d * b)

    def prior(P: Array) -> Array:  # noqa: N806
        r"""
        The prior loss function.

        :param P: The parameters :math:`p \in \mathbb{R}^{k}`.
        :returns: The prior term.
        """
        d = jnp.reshape(P - p, -1)
        b = hp * d if hp.ndim == p.ndim else hp @ d
        return 0.5 * jnp.sum(d * b)

    def misfit(P: Array, _: None = None) -> Array:  # noqa
        r"""
        The misfit function.

        :param P: The parameters :math:`p \in \mathbb{R}^{k}`.
        :returns: The total cost.
        """
        loss_term = jnp.sum(
            jax.vmap(loss, in_axes=(None, 0, 0, 0, 0))(P, x, y, ux, uy)
        )
        return loss_term if up is None else loss_term + prior(P)

    def post(p: Array) -> tuple[Array, Array]:
        r"""
        Computes posterior uncertainty.

        :param p: The posterior :math:`\hat{p} \in \mathbb{R}^{k}`.
        :returns: The posterior uncertainty tensor and standard uncertainty.
        """
        hess = jax.hessian(misfit)
        pcov = jli.pinv(hess(p).reshape(p.size, -1))
        punc = jnp.sqrt(jnp.diag(pcov))
        return pcov.reshape(p.shape + p.shape), punc.reshape(p.shape)

    def invert(u: Array, p: Array) -> Array:
        """
        Inverts an uncertainty tensor.

        :param u: The uncertainty tensor.
        :param p: The parameter tensor.
        :returns: The inverted uncertainty tensor (in matrix form).
        """
        return (
            jnp.where(u > 0.0, 1.0 / u, 0.0).reshape(-1)
            if u.ndim == p.ndim
            else jli.pinv(u.reshape(p.size, -1))
        )

    def make_minimizer():
        """Returns the minimizer."""
        return optimistix.OptaxMinimiser(
            optax.lbfgs(), atol=atol, rtol=rtol, norm=optimistix.max_norm
        )

    hp = invert(up, p) if up is not None else None
    if ux is None:
        ux = jnp.broadcast_to(1.0, x.shape)
    if uy is None:
        uy = jnp.broadcast_to(1.0, y.shape)
    optimum = optimistix.minimise(
        misfit, make_minimizer(), p, max_steps=max_steps, throw=False
    )
    popt = optimum.value
    pcov, punc = post(popt)
    cost = misfit(popt)
    info = jnp.where(optimum.result == optimistix.RESULTS.successful, 0, 1)

    return popt, pcov, punc, cost, info


class EIV(Fitting):
    """
    Bayesian errors-in-variables optimizer based on the effective
    variance method (EVM).

    This implementation is intended for large scale problems with
    up to millions of data points.
    """

    def fit(
        self,
        f: M,
        x: np.ndarray,
        y: np.ndarray,
        *,
        p: np.ndarray | None = None,
        ux: np.ndarray | None = None,
        uy: np.ndarray | None = None,
        up: np.ndarray | None = None,
        use_covar: bool = False,
        atol: Any = DEFAULT_ATOL,
        rtol: Any = DEFAULT_RTOL,
        max_steps: int = DEFAULT_MAX_STEPS,
        **kwargs,
    ) -> Fitted:
        r"""
        Fits the parameters of a model function to :math:`M` samples
        :math:`(x_i, y_i)` of data.

        Under the same notation and remarks as :class:`M`:

        :param f: The model function.
        :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :param p: Prior model parameter values :math:`\check{p}`.
        :param ux: Standard uncertainties :math:`u(X)`.
        :param uy: Standard uncertainties :math:`u(Y)`.
        :param up: Prior standard uncertainties :math:`u(\check{p})`.
        :param use_covar: Consider covariance?
        :param atol: The absolute tolerance for terminating the optimization.
        :param rtol: The relative tolerance for terminating the optimization.
        :param max_steps: The maximum number of steps the optimizer can take.
        :returns: The fit result.
        """
        popt, pcov, punc, cost, info = _batch(
            f.f,
            jnp.asarray(p if p is not None else f.prior(x, y)),
            jnp.asarray(x),
            jnp.asarray(y),
            jnp.square(ux) if ux is not None else None,
            jnp.square(uy) if uy is not None else None,
            jnp.square(up) if up is not None else None,
            use_covar=use_covar,
            atol=atol,
            rtol=rtol,
            max_steps=max_steps,
        )
        popt = np.asarray(popt)
        zvar = np.var(f.eval(popt, x) - y, axis=0, ddof=popt.size)

        return Fitted(
            f,
            popt=popt,
            pcov=np.asarray(pcov),
            punc=np.asarray(punc),
            zvar=zvar,
            cost=np.asarray(cost),
            info=info.item(),
        )
