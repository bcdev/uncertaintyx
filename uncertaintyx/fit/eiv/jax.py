#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
from typing import Any
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array
from jax.numpy.linalg import norm
from jax.scipy.linalg import cho_factor
from jax.scipy.linalg import cho_solve
from optax import lbfgs

from ...interface.core import Fitting
from ...interface.core import M
from ...interface.core import Result


@jax.jit(static_argnums=(0,))
def evm(
    f: Callable[[Array, Array], Array],
    p: Array,
    x: Array,
    y: Array,
    ux: Array,
    uy: Array,
    *,
    max_i: int = 100,
    obj_g: Any = 1.0E-06,
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

    Otherwise, under the same notation as :class:`EIV`:

    :param f: The model function.
    :param p: Parameters :math:`p \in \mathbb{R}^{k}`.
    :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
    :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
    :param ux: Uncertainty tensor :math:`U(X)`, full or diagonal.
    :param uy: Uncertainty tensor :math:`U(Y)`, full or diagonal.
    :param diagonalize: Propagate full input uncertainty, use only
    the output diagonal.
    :param max_i: The maximum number of iterations permitted.
    :param max_g: The maximum gradient permitted.
    :returns: The minimization loop state.
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

        :returns: The sample loss.
        """

        def g(p: Array, x: Array) -> Array:
            """The Jacobian."""
            return jax.jacrev(f, argnums=1)(p, x) if y.size < x.size else jax.jacfwd(f, argnums=1)(p, x)

        def upc(d: int, G: Array, u: Array) -> Array:
            """Uncertainty propagation."""
            dims = tuple(range(-d, 0))
            return jnp.tensordot(
                jnp.tensordot(G, u, axes=(dims, dims)), G, axes=(dims, dims)
            )

        def upd(d: int, G: Array, u: Array) -> Array:
            """Uncertainty propagation."""
            dims = tuple(range(-d, 0))
            return jnp.sum(
                jnp.tensordot(G, u, axes=(dims, dims)) * G, axis=dims
            )

        d = f(p, x) - y
        G = g(p, x)  # noqa: N806
        if uy.shape == y.shape:
            V = uy + upd(x.ndim, G, ux)  # noqa: N806
            b = d / V
        else:
            d = d.reshape(-1)
            C = jnp.reshape(  # noqa: N806
                uy + upc(x.ndim, G, ux), d.shape + d.shape
            )
            L = cho_factor(C)  # noqa: N806
            b = cho_solve(L, d)
        return 0.5 * jnp.sum(d * b)

    def loss(p: Array) -> Array:
        """
        The batch loss function.

        :param p: The parameters.
        :returns: The batch loss.
        """
        losses = jax.vmap(sample_loss, in_axes=(None, 0, 0, 0, 0))(
            p, x, y, ux, uy
        )
        return jnp.sum(losses)

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
            grad, state, popt, value=cost, grad=grad, value_fn=loss
        )
        popt = optax.apply_updates(popt, u)
        cost, grad = cost_and_grad(popt, state=state)
        converged = norm(grad, ord=jnp.inf) < max_g  # noqa
        return i + 1, popt, state, cost, grad, converged

    optim = lbfgs()
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

        Under the same notation and remarks as :class:`M`:

        :param f: The model function.
        :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :param ux: Standard uncertainties :math:`u(X)`.
        :param uy: Standard uncertainties :math:`u(Y)`.
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
