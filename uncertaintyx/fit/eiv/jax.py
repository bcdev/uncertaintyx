#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
from typing import Any
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jcp
import numpy as np
import optax
from jax import Array

from ...interface.core import Fitting
from ...interface.core import M
from ...interface.core import Result
from ...m.jax import jac


@jax.jit(static_argnums=(0,))
def evm(
    f: Callable[[Array, Array], Array],
    p: Array,
    x: Array,
    y: Array,
    ux: Array,
    uy: Array,
    max_iter: int = 100,
    obj_g: Any = 1.0e-06,
) -> tuple[int, Array, Any, Array, Array, bool]:
    r"""
    Implementation of the effective variance method (EVM) with L-BFGS.

    Under the same notation as :class:`EIV`:

    :param f: The model function.
    :param p: Parameters :math:`p \in \mathbb{R}^{k}`.
    :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
    :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
    :param ux: Uncertainty :math:`U(X) \in \mathbb{R}^{M \times m}`.
    :param uy: Uncertainty :math:`U(Y) \in \mathbb{R}^{M \times n}`.
    :param max_iter: The maximum number of iterations permitted.
    :param obj_g: The gradient objective to achieve.
    :returns: A tuple carrying the state of the optimization.
    """

    def sample_loss(
        p: Array, x: Array, y: Array, ux: Array, uy: Array
    ) -> Array:
        r"""
        The sample loss function.

        :param p: The parameters.
        :param x: The sample :math:`x \in \mathbb{R}^{m}`.
        :param y: The sample :math:`y \in \mathbb{R}^{n}`.
        :param ux: :math:`U(x) \in \mathbb{R}^{m \times m}`.
        :param uy: :math:`U(y) \in \mathbb{R}^{n \times n}`.

        :returns: The loss.
        """

        def up_full(d: int, u: Array) -> Array:
            """Uncertainty propagation."""
            dims = tuple(range(-d, 0))
            return jnp.tensordot(
                jnp.tensordot(g, u, axes=(dims, dims)), g, axes=(dims, dims)
            )

        def up_diag(d: int, u: Array) -> Array:
            """Uncertainty propagation."""
            dims = tuple(range(-d, 0))
            return jnp.sum(
                jnp.tensordot(g, u, axes=(dims, dims)) * g, axis=dims
            )

        g = jac(f, 1, y.size < x.size)(p, x)
        d = jnp.reshape(f(p, x) - y, -1)
        if ux.shape == x.shape and uy.shape == y.shape:
            v = jnp.reshape(uy + up_diag(x.ndim, ux), -1)  # noqa: N806
            b = d / v
        elif uy.shape == y.shape:
            V = jnp.diag(uy.reshape(-1)) + jnp.reshape(  # noqa: N806
                up_full(x.ndim, ux), d.shape + d.shape
            )
            L = jcp.linalg.cho_factor(V)  # noqa: N806
            b = jcp.linalg.cho_solve(L, d)  # noqa
        else:
            V = jnp.reshape(  # noqa: N806
                uy + up_full(x.ndim, ux), d.shape + d.shape
            )
            L = jcp.linalg.cho_factor(V)  # noqa: N806
            b = jcp.linalg.cho_solve(L, d)  # noqa
        return 0.5 * jnp.sum(d * b)

    def loss(p: Array) -> Array:
        """
        The batch loss function.

        :param p: The parameters.
        :returns: The loss.
        """
        losses = jax.vmap(sample_loss, in_axes=(None, 0, 0, 0, 0))(
            p, x, y, ux, uy
        )
        return jnp.sum(losses)

    def cond(carry: tuple[int, Array, Any, Array, Array, bool]) -> Array:
        """
        The function to check loop continuation.

        :param carry: The tuple carrying the loop state.
        :returns: The loop continuation status.
        """
        i, _, _, _, _, converged = carry
        return jnp.logical_and(i < max_iter, jnp.logical_not(converged))

    def body(
        carry: tuple[int, Array, Any, Array, Array, bool],
    ) -> tuple[int, Array, Any, Array, Array, bool]:
        """
        The loop body function.

        :param carry: The tuple carrying the loop state.
        :returns: The updated loop state.
        """
        i, popt, state, cost, grad, _ = carry
        u, state = optim.update(
            grad, state, popt, value=cost, grad=grad, value_fn=loss
        )
        popt = optax.apply_updates(popt, u)
        cost, grad = cost_and_grad(popt, state=state)
        converged = jnp.linalg.norm(grad, ord=jnp.inf) < obj_g  # noqa
        return i + 1, popt, state, cost, grad, converged

    optim = optax.lbfgs()
    state = optim.init(p)
    cost_and_grad = optax.value_and_grad_from_state(loss)
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
        max_iter: int = 100,
        **kwargs,
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
        i, popt, state, cost, g, converged = evm(
            f.f,
            jnp.asarray(f.estimate(x, y)),
            jnp.asarray(x),
            jnp.asarray(y),
            jnp.asarray(ux * ux) if ux is not None else jnp.ones_like(x),
            jnp.asarray(uy * uy) if uy is not None else jnp.ones_like(y),
            max_iter,
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
