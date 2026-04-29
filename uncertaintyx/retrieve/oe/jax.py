#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
"""
Optimal estimation (OE) implementation. Refer to:

Tarantola (2005). Inverse Problem Theory and Methods for
Model Parameter Estimation. Society for Industrial and Applied
Mathematics. https://doi.org/10.1137/1.9780898717921
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

from ...tyx import F
from ...tyx import Retrieved
from ...tyx import Retrieving

DEFAULT_MAX_D: Any = 1.0e-12
"""The maximum L2 norm of the parameter step allowed for convergence."""

DEFAULT_MAX_G: Any = 0.0
"""The maximum infinity norm of the gradient allowed for convergence."""

DEFAULT_MAX_I: int = 2000
"""The maximum number of iterations permitted."""


@jax.jit(static_argnums=(0,), static_argnames=("covar",))
def oe_sample(
    f: Callable[[Array], Array],
    x: Array,
    y: Array,
    ux: Array | None = None,
    uy: Array | None = None,
    *,
    covar: bool = False,
    max_i: int = DEFAULT_MAX_I,
    max_d: Any = DEFAULT_MAX_D,
    max_g: Any = DEFAULT_MAX_G,
) -> tuple[Array, Array, Array, Array, Array]:
    r"""
    Optimal estimation (OE) retrieval using a limited-memory
    Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimizer.

    This implementation accepts any combination of full-rank
    or diagonal-rank uncertainty tensors:
    :math:`U(\check{x}) \in \mathbb{R}^{m \times m}`,
    :math:`U(\check{x}) \in \mathbb{R}^{m}`,
    :math:`U(y) \in \mathbb{R}^{n \times n}`, and
    :math:`U(y) \in \mathbb{R}^{n}`.

    Standard uncertainty is not accepted and must be squared to
    a variance (diagonal uncertainty tensor) before supplied as
    argument.

    Otherwise, under the same notation as :class:`OE`:

    :param f: The function.
    :param x: Sample :math:`\check{x} \in \mathbb{R}^{m}`.
    :param y: Sample :math:`y \in \mathbb{R}^{n}`.
    :param ux: Uncertainty tensor :math:`U(\check{x})`, full or diagonal.
    :param uy: Uncertainty tensor :math:`U(y)`, full or diagonal.
    :param covar: Consider covariance?
    :param max_i: The maximum number of iterations allowed.
    :param max_d: The maximum norm of the update allowed for convergence.
    :param max_g: The maximum norm of the gradient allowed for convergence.
    :returns: The retrieval result.
    """

    def loss(q: Array, y: Array, u: Array) -> Array:
        r"""
        The loss function.

        :param q: The sample :math:`x \in \mathbb{R}^{m}`.
        :param y: The sample :math:`y \in \mathbb{R}^{n}`.
        :param u: :math:`U(y) \in \mathbb{R}^{n \times n}`.
        :returns: The sample loss.
        """
        d = f(q) - y
        if u.ndim == y.ndim or not covar:
            U = (  # noqa: N806
                u
                if u.ndim == y.ndim
                else jnp.diag(u.reshape((d.size, -1))).reshape(y.shape)
            )
            b = d / U
        else:
            d = d.reshape(-1)
            U = u.reshape((y.size, -1))  # noqa: N806
            b = jla.cho_solve(jla.cho_factor(U), d)
        return 0.5 * jnp.sum(d * b)

    def prior(q: Array) -> Array:
        """
        The prior loss function.

        :param q: The sample.
        :returns: The prior loss.
        """
        d = jnp.reshape(q - x, -1)
        b = (
            jnp.where(ux > 0.0, 1.0 / ux, 0.0) * d
            if ux.ndim == x.ndim
            else jli.pinv(ux.reshape(d.size, -1)) @ d
        )
        return 0.5 * jnp.sum(d * b)

    def S(q: Array) -> Array:  # noqa: N806
        """
        The cost (or misfit) function to minimize.

        :param q: The sample.
        :returns: The cost.
        """
        loss_term = loss(q, y, uy)
        return loss_term if ux is None else loss_term + prior(q)

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
        i, xopt, tree, cost, grad, _ = carry
        d, tree = optimizer.update(
            grad, tree, xopt, value=cost, grad=grad, value_fn=S
        )
        xopt = optax.apply_updates(xopt, d)
        cost, grad = cost_and_grad(xopt, state=tree)
        grad_norm = jli.norm(grad, ord=jnp.inf)  # noqa
        step_norm = jli.norm(d)
        converged = jnp.logical_or(grad_norm < max_g, step_norm < max_d)
        return i + 1, xopt, tree, cost, grad, converged

    def opti(x: Array) -> tuple[Array, Array, Array]:
        """
        Optimizes the sample.

        :param x: The prior sample.
        :returns: The posterior sample, the cost, and the convergence status.
        """
        tree = optimizer.init(x)
        cost, grad = cost_and_grad(x, state=tree)
        init = (0, x, tree, cost, grad, False)
        _, xopt, _, cost, _, converged = jax.lax.while_loop(cond, body, init)
        return xopt, cost, converged

    def post(x: Array) -> tuple[Array, Array]:
        """
        Computes posterior uncertainty.

        :param x: The posterior sample.
        :returns: The posterior uncertainty tensor and standard uncertainty.
        """
        hess = jax.hessian(S)
        xcov = jli.pinv(hess(x).reshape(x.size, -1))
        xunc = jnp.sqrt(jnp.diag(xcov))
        return xcov.reshape(x.shape + x.shape), xunc.reshape(x.shape)

    optimizer = optax.lbfgs()
    if uy is None:
        uy = jnp.broadcast_to(1.0, y.shape)
    cost_and_grad = optax.value_and_grad_from_state(S)
    xopt, cost, converged = opti(x)
    xcov, xunc = post(xopt)

    return xopt, xcov, xunc, cost, converged


@jax.jit(static_argnums=(0,), static_argnames=("covar",))
def oe_batch(
    f: Callable[[Array], Array],
    x: Array,
    y: Array,
    ux: Array | None = None,
    uy: Array | None = None,
    *,
    covar: bool = False,
    max_i: int = DEFAULT_MAX_I,
    max_d: Any = DEFAULT_MAX_D,
    max_g: Any = DEFAULT_MAX_G,
) -> tuple[Array, Array, Array, Array, Array]:
    r"""
    Optimal estimation (OE) retrieval using a limited-memory
    Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimizer.

    This implementation accepts any combination of full-rank
    or diagonal-rank uncertainty tensors:
    :math:`U(\check{X}) \in \mathbb{R}^{M \times m \times m}`,
    :math:`U(\check{X}) \in \mathbb{R}^{M \times m}`,
    :math:`U(Y) \in \mathbb{R}^{M \times n \times n}`, and
    :math:`U(Y) \in \mathbb{R}^{M \times n}`.

    Standard uncertainty is not accepted and must be squared to
    a variance (diagonal uncertainty tensor) before supplied as
    argument.

    Otherwise, under the same notation as :class:`OE`:

    :param f: The function.
    :param x: Samples :math:`\check{X} \in \mathbb{R}^{M \times m}`.
    :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
    :param ux: Uncertainty tensor :math:`U(\check{X})`, full or diagonal.
    :param uy: Uncertainty tensor :math:`U(Y)`, full or diagonal.
    :param covar: Consider covariance?
    :param max_i: The maximum number of iterations allowed.
    :param max_d: The maximum norm of the update allowed for convergence.
    :param max_g: The maximum norm of the gradient allowed for convergence.
    :returns: The retrieval result.
    """

    def oe(x, y, ux, uy):
        return oe_sample(
            f,
            x,
            y,
            ux,
            uy,
            covar=covar,
            max_i=max_i,
            max_d=max_d,
            max_g=max_g,
        )

    mapped = jax.vmap(
        oe,
        in_axes=(
            0,
            0,
            None if ux is None else 0,
            None if uy is None else 0,
        ),
    )
    return mapped(x, y, ux, uy)


class OE(Retrieving):
    """
    Optimal estimation (OE) retrieval.
    """

    def retrieve(
        self,
        f: F,
        x: np.ndarray,
        y: np.ndarray,
        *,
        ux: np.ndarray | None = None,
        uy: np.ndarray | None = None,
        max_i: int = DEFAULT_MAX_I,
        **kwargs,
    ) -> Retrieved:
        r"""
        Solves an inverse problem of the form :math:`f(x) = y` for
        :math:`M` samples :math:`(\check{x}_i, y_i)` of data, where
        :math:`\check{x}_i` are prior (or initial) estimates of the
        unknown posterior solution :math:`\hat{x}_i`.

        Only standard uncertainties are accepted. You must not
        supply uncertainty tensors (neither diagonal nor full).
        Standard uncertainties are squared to variances before
        passed to the OE optimizer.

        Under the same notation and remarks as :class:`F`:

        :param f: The function.
        :param x: Samples :math:`\check{X} \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :param ux: Standard uncertainties :math:`u(\check{X})`.
        :param uy: Standard uncertainties :math:`u(Y)`.
        :param max_i: The maximum number of iterations conducted.
        :returns: The retrieved result.
        """
        xopt, xcov, xunc, cost, converged = oe_batch(
            f.f,
            jnp.asarray(x),
            jnp.asarray(y),
            jnp.square(ux) if ux is not None else None,
            jnp.square(uy) if uy is not None else None,
            max_i=max_i,
            **kwargs,
        )
        xopt = np.asarray(xopt)
        zvar = np.var(
            f.eval(xopt) - y,
            axis=0,
            ddof=y.size - x.size if y.size > x.size else 0,
        )
        return Retrieved(
            xopt=xopt,
            xcov=np.asarray(xcov),
            xunc=np.asarray(xunc),
            zvar=zvar,
            cost=np.asarray(jnp.sum(cost)),
            info=np.asarray(jnp.where(converged, 0, 1)),
        )
