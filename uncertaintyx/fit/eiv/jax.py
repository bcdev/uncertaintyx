#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
"""
Errors-in-variables implementation based on the effective
variance method (EVM). Refer to:

A. Tarantola (2005). Inverse Problem Theory and Methods for
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
import jax.numpy as jnp
import jax.numpy.linalg as jli
import jax.scipy.linalg as jla
import numpy as np
import optax
from jax import Array

from ...interface.core import Fitting
from ...interface.core import M
from ...interface.core import Result

DEFAULT_MAX_G: Any = 1.0e-08
"""The maximum gradient permitted."""

DEFAULT_MAX_I: int = 100
"""The maximum number of iterations permitted."""


@jax.jit(static_argnums=(0,), static_argnames=("covar",))
def evm(
    f: Callable[[Array, Array], Array],
    p: Array,
    x: Array,
    y: Array,
    ux: Array,
    uy: Array,
    up: Array | None = None,
    *,
    max_i: int = DEFAULT_MAX_I,
    max_g: Any = DEFAULT_MAX_G,
    covar: bool = False,
) -> tuple[Any, ...]:
    r"""
    Implementation of the effective variance method (EVM) with
    a limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)
    optimizer.

    This implementation accepts any combination of full-rank
    or diagonal-rank uncertainty tensors:

    .. math::
        U(X) \in \mathbb{R}^{M \times m \times m}, \quad
        U(X) \in \mathbb{R}^{M \times m},

        U(Y) \in \mathbb{R}^{M \times n \times n}, \quad
        U(Y) \in \mathbb{R}^{M \times n},

        U(p) \in \mathbb{R}^{k \times k}, \quad
        U(p) \in \mathbb{R}^{k},

    Standard uncertainty is not accepted and must be squared to
    a variance (diagonal uncertainty tensor) before supplied as
    argument. Otherwise, under the same notation as :class:`EIV`:

    :param f: The model function.
    :param p: Parameters :math:`p \in \mathbb{R}^{k}`.
    :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
    :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
    :param ux: Uncertainty tensor :math:`U(X)`, full or diagonal.
    :param uy: Uncertainty tensor :math:`U(Y)`, full or diagonal.
    :param up: Uncertainty tensor :math:`U(p)`, full or diagonal.
    :param max_i: The maximum number of iterations permitted.
    :param max_g: The maximum gradient permitted.
    :param covar: Use effective covariance, too.
    :returns: The fit result comprising: the optimized parameter
    values, the parameter uncertainty tensor, parameter standard
    uncertainties, the misfit, and the convergence status.
    """

    def g(q: Array, x: Array) -> Array:
        r"""
        The sample Jacobian.

        :param q: The parameters.
        :param x: A sample :math:`x \in \mathbb{R}^{m}`
        :returns: :math:`(G_x f)(q, x) \in \mathbb{R}^{n \times m}`
        """
        return (
            jax.jacrev(f, argnums=1)(q, x)
            if y.size < x.size
            else jax.jacfwd(f, argnums=1)(q, x)
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
        r"""
        The diagonalized sample uncertainty propagation.

        :param d: The rank of the input sample tensor.
        :param G: The sample Jacobian.
        :param U: The input sample uncertainty tensor.
        :returns: The diagonal of the propagated uncertainty tensor.
        """
        dims = tuple(range(-d, 0))
        return jnp.sum(jnp.tensordot(G, U, axes=(dims, dims)) * G, axis=dims)

    def loss(q: Array, x: Array, y: Array, ux: Array, uy: Array) -> Array:
        r"""
        The sample loss function.

        :param q: The parameters.
        :param x: The sample :math:`x \in \mathbb{R}^{m}`.
        :param y: The sample :math:`y \in \mathbb{R}^{n}`.
        :param ux: :math:`U(x) \in \mathbb{R}^{m \times m}`.
        :param uy: :math:`U(y) \in \mathbb{R}^{n \times n}`.

        :returns: The sample loss.
        """
        d = f(q, x) - y
        G = g(q, x)  # noqa: N806
        if not covar:
            U = upd(x.ndim, G, ux) + (  # noqa: N806
                uy
                if uy.ndim == y.ndim
                else jnp.diag(uy.reshape((d.size, -1))).reshape(d.shape)
            )
            b = d / U
        else:
            d = d.reshape(-1)
            U = upc(x.ndim, G, ux).reshape((d.size, -1)) + (  # noqa: N806
                jnp.diag(uy.reshape(-1))
                if uy.ndim == y.ndim
                else uy.reshape((d.size, -1))
            )
            b = jla.cho_solve(jla.cho_factor(U), d)
        return 0.5 * jnp.sum(d * b)

    def prior(q: Array) -> Array:
        """
        The prior loss function.

        :param q: The parameters.
        :returns: The prior loss.
        """
        d = jnp.reshape(q - p, -1)
        b = (
            jnp.where(up > 0.0, 1.0 / up, 0.0) * d
            if up.ndim == p.ndim
            else jli.pinv(up.reshape(d.size, -1)) @ d
        )
        return 0.5 * jnp.sum(d * b)

    def S(q: Array) -> Array:  # noqa: N806
        """
        The misfit function to minimize.

        :param q: The parameters.
        :returns: The misfit.
        """
        loss_term = jnp.sum(
            jax.vmap(loss, in_axes=(None, 0, 0, 0, 0))(q, x, y, ux, uy)
        )
        return loss_term if up is None else loss_term + prior(q)

    def cond(carry: tuple[Any, ...]) -> Array:
        """
        The function to check loop continuation.

        :param carry: The loop state carrier.
        :returns: A Boolean.
        """
        i, _, _, _, _, converged = carry
        return jnp.logical_and(i < max_i, jnp.logical_not(converged))

    def body(carry: tuple[Any, ...]) -> tuple[Any, ...]:
        """
        The loop body function.

        :param carry: The loop state carrier.
        :returns: The updated loop state carrier.
        """
        i, popt, tree, cost, grad, _ = carry
        u, tree = optimizer.update(
            grad, tree, popt, value=cost, grad=grad, value_fn=S
        )
        popt = optax.apply_updates(popt, u)
        cost, grad = cost_and_grad(popt, state=tree)
        converged = jli.norm(grad, ord=jnp.inf) < max_g  # noqa
        return i + 1, popt, tree, cost, grad, converged

    def optimize(p: Array) -> tuple[Array, ...]:
        tree = optimizer.init(p)
        cost, grad = cost_and_grad(p, state=tree)
        init = (0, p, tree, cost, grad, False)
        _, popt, _, cost, _, converged = jax.lax.while_loop(cond, body, init)
        return popt, cost, converged

    def post(p: Array) -> tuple[Array, Array]:
        hess = jax.hessian(S)
        pcov = jli.pinv(hess(p).reshape(p.size, -1))
        punc = jnp.sqrt(jnp.diag(pcov)).reshape(p.shape)
        return pcov.reshape(p.shape + p.shape), punc.reshape(p.shape)

    if ux is None:
        ux = jnp.broadcast_to(1.0, x.shape)
    if uy is None:
        uy = jnp.broadcast_to(1.0, y.shape)
    optimizer = optax.lbfgs()
    cost_and_grad = optax.value_and_grad_from_state(S)
    popt, cost, converged = optimize(p)
    pcov, punc = post(p)

    return popt, pcov, punc, cost, converged


class EIV(Fitting):
    """
    Errors-in-variables implementation based on the effective
    variance method (EVM).

    This implementation is intended for large scale problems
    with up to millions of data points.
    """

    def fit(
        self,
        f: M,
        x: np.ndarray,
        y: np.ndarray,
        *,
        ux: np.ndarray | None = None,
        uy: np.ndarray | None = None,
        up: np.ndarray | None = None,
        max_iter: int = 100,
        **kwargs,
    ) -> Result:
        r"""
        Fits the parameters of a model function to :math:`M`
        samples :math:`(x_i, y_i)` of data.

        Under the same notation and remarks as :class:`M`:

        :param f: The model function.
        :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :param ux: Standard uncertainties :math:`u(X)`.
        :param uy: Standard uncertainties :math:`u(Y)`.
        :param up: Standard uncertainties :math:`u(p)`.
        :param max_iter: The maximum number of iterations conducted.
        :returns: The fit result.
        """
        popt, pcov, punc, cost, converged = evm(
            f.f,
            jnp.asarray(f.estimate(x, y)),
            jnp.asarray(x),
            jnp.asarray(y),
            jnp.square(ux) if ux is not None else None,
            jnp.square(uy) if uy is not None else None,
            jnp.square(up) if up is not None else None,
            max_i=max_iter,
            **kwargs,
        )
        popt = np.asarray(popt)
        rvar = np.var(f.eval(popt, x) - y, axis=0, ddof=popt.size)

        return Result(
            f,
            popt=popt,
            punc=np.asarray(punc),
            pcov=np.asarray(pcov),
            rvar=rvar,
            cost=np.asarray(cost),
            info=0 if converged.item() else 1,
        )
