#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
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


@jax.jit(static_argnums=(0,), static_argnames=("diagonalize",))
def evm_fit(
    f: Callable[[Array, Array], Array],
    p: Array,
    x: Array,
    y: Array,
    ux: Array,
    uy: Array,
    up: Array | None = None,
    *,
    max_i: int = 100,
    max_g: Any = 1.0e-08,
    diagonalize: bool = True,
) -> tuple[Any, ...]:
    r"""
    Implementation of the effective variance method (EVM) with
    a limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)
    minimizer.

    This implementation accepts any combination of full-rank
    or diagonal-rank uncertainty tensors:

    .. math::
        U(X) \in \mathbb{R}^{M \times m \times m}, \quad
        U(X) \in \mathbb{R}^{M \times m},

        U(Y) \in \mathbb{R}^{M \times n \times n}, \quad
        U(Y) \in \mathbb{R}^{M \times n},

        U(p) \in \mathbb{R}^{k \times k}, \quad
        U(p) \in \mathbb{R}^{k},

    Otherwise, under the same notation as :class:`EIV`:

    :param f: The model function.
    :param p: Parameters :math:`p \in \mathbb{R}^{k}`.
    :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
    :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
    :param ux: Uncertainty tensor :math:`U(X)`, full or diagonal.
    :param uy: Uncertainty tensor :math:`U(Y)`, full or diagonal.
    :param up: Uncertainty tensor :math:`U(p)`, full or diagonal.
    :param max_i: The maximum number of iterations permitted.
    :param max_g: The maximum gradient permitted.
    :param diagonalize: Use only diagonal of uncertainty propagation output.
    :returns: The minimization loop state.
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
        if diagonalize:
            U = (  # noqa: N806
                jnp.diag(uy.reshape((y.size, y.size))).reshape(y.shape)
                if uy.ndim != y.ndim
                else uy
            ) + upd(x.ndim, G, ux)
            b = d / U
        else:
            d = d.reshape(y.size)
            U = (  # noqa: N806
                uy.reshape((y.size, y.size))
                if uy.ndim != y.ndim
                else jnp.diag(uy.reshape(y.size))
            ) + upc(x.ndim, G, ux).reshape((y.size, y.size))
            L = jla.cho_factor(U)  # noqa: N806
            b = jla.cho_solve(L, d)
        return 0.5 * jnp.sum(d * b)

    def prior(q: Array) -> Array:
        """
        The prior loss function.

        :param q: The parameters.
        :returns: The prior loss.
        """
        d = jnp.reshape(q - p, p.size)
        b = (
            jli.pinv(up.reshape(p.size, p.size)) @ d
            if up.ndim != p.ndim
            else jnp.where(up > 0.0, 1.0 / up, 0.0) * d
        )
        return 0.5 * jnp.sum(d * b)

    def S(q: Array) -> Array:  # noqa: N806
        """
        The misfit function to minimize.

        :param q: The parameters.
        :returns: The misfit.
        """
        term = jnp.sum(
            jax.vmap(loss, in_axes=(None, 0, 0, 0, 0))(q, x, y, ux, uy)
        )
        return term if up is None else term + prior(q)

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
        i, popt, state, cost, grad, _ = carry
        u, state = optim.update(
            grad, state, popt, value=cost, grad=grad, value_fn=S
        )
        popt = optax.apply_updates(popt, u)
        cost, grad = cost_and_grad(popt, state=state)
        converged = jli.norm(grad, ord=jnp.inf) < max_g  # noqa
        return i + 1, popt, state, cost, grad, converged

    optim = optax.lbfgs()
    state = optim.init(p)
    cost_and_grad = optax.value_and_grad_from_state(S)
    cost, grad = cost_and_grad(p, state=state)
    carry = (0, p, state, cost, grad, False)

    return jax.lax.while_loop(cond, body, carry)  # noqa


class EIV(Fitting):
    """
    Errors-in-variables implementation based on the effective
    variance method (EVM).

    This implementation is intended for large scale problems
    with up to millions of data points. Refer to:

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
        i, popt, state, cost, g, converged = evm_fit(
            f.f,
            jnp.asarray(f.estimate(x, y)),
            jnp.asarray(x),
            jnp.asarray(y),
            jnp.square(ux) if ux is not None else jnp.ones_like(x),
            jnp.square(uy) if uy is not None else jnp.ones_like(y),
            jnp.square(up) if up is not None else None,
            max_i=max_iter,
            **kwargs,
        )
        popt = np.asarray(popt)
        punc = np.zeros_like(popt)
        pcov = np.zeros_like(popt, shape=popt.shape + popt.shape)
        rvar = np.var(f.eval(popt, x) - y, axis=0, ddof=popt.size)
        cost = np.asarray(cost)
        conv = np.asarray(converged)

        return Result(
            f,
            popt=popt,
            punc=punc,
            pcov=pcov,
            rvar=rvar,
            cost=cost,
            info=0 if conv else 1,
        )
