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
import numpy as np
import optax
import optimistix
from jax import Array

from ...tyx import F
from ...tyx import Retrieved
from ...tyx import Retrieving

DEFAULT_ATOL: Any = 1.0e-12
"""The absolute tolerance for terminating the optimization."""

DEFAULT_RTOL: Any = 1.0e-08
"""The relative tolerance for terminating the optimization."""

DEFAULT_MAX_STEPS: int = 2048
"""The maximum number of steps the optimizer can take."""


def _sample(
    f: Callable[[Array], Array],
    x: Array,
    y: Array,
    ux: Array | None = None,
    uy: Array | None = None,
    *,
    atol: Any = DEFAULT_ATOL,
    rtol: Any = DEFAULT_RTOL,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> tuple[Array, Array, Array, Array, Array]:
    r"""
    Optimal estimation (OE) retrieval.

    The implementation accepts any combination of full-rank
    or diagonal-rank uncertainty tensors:
    :math:`U(\check{x}) \in \mathbb{R}^{m \times m}`,
    :math:`U(\check{x}) \in \mathbb{R}^{m}`,
    :math:`U(y) \in \mathbb{R}^{n \times n}`, and
    :math:`U(y) \in \mathbb{R}^{n}`.

    Standard uncertainty is not accepted and must be squared to
    a variance (diagonal uncertainty tensor) before supplied as
    argument.

    :param f: The function.
    :param x: Sample :math:`\check{x} \in \mathbb{R}^{m}`.
    :param y: Sample :math:`y \in \mathbb{R}^{n}`.
    :param ux: Uncertainty tensor :math:`U(\check{x})`, full or diagonal.
    :param uy: Uncertainty tensor :math:`U(y)`, full or diagonal.
    :param atol: The absolute tolerance for terminating the optimization.
    :param rtol: The relative tolerance for terminating the optimization.
    :param max_steps: The maximum number of steps the optimizer can take.
    :returns: The retrieval result.
    """

    def loss(X: Array) -> Array:  # noqa : N806
        r"""
        The loss function.

        :param X: The sample :math:`x \in \mathbb{R}^{m}`.
        :returns: The sample loss.
        """
        d = jnp.reshape(f(X) - y, -1)
        b = hy * d if hy.ndim == y.ndim else hy @ d
        return 0.5 * jnp.sum(d * b)

    def prior(X: Array) -> Array:  # noqa : N806
        r"""
        The prior loss function.

        :param X: The sample :math:`x \in \mathbb{R}^{m}`.
        :returns: The prior term.
        """
        d = jnp.reshape(X - x, -1)
        b = hx * d if hx.ndim == x.ndim else hx @ d
        return 0.5 * jnp.sum(d * b)

    def misfit(X: Array, _: None = None) -> Array:  # noqa
        r"""
        The misfit function.

        :param X: The sample :math:`x \in \mathbb{R}^{m}`.
        :returns: The total cost.
        """
        return loss(X) if hx is None else loss(X) + prior(X)

    def post(x: Array) -> tuple[Array, Array]:  # noqa : N806
        r"""
        Computes the posterior uncertainty.

        :param x: The posterior :math:`\hat{x} \in \mathbb{R}^{m}`.
        :returns: The posterior uncertainty tensor and standard uncertainty.
        """
        hess = jax.hessian(misfit)
        xcov = jli.pinv(hess(x).reshape(x.size, -1))
        xunc = jnp.sqrt(jnp.diag(xcov))
        return xcov.reshape(x.shape + x.shape), xunc.reshape(x.shape)

    def invert(u: Array, t: Array) -> Array:
        """
        Inverts an uncertainty tensor.

        :param u: The uncertainty tensor.
        :param t: The sample tensor (a template).
        :returns: The inverted uncertainty tensor (in matrix form).
        """
        return (
            jnp.where(u > 0.0, 1.0 / u, 0.0).reshape(-1)
            if u.ndim == t.ndim
            else jli.pinv(u.reshape(t.size, -1))
        )

    def make_minimizer():
        """Returns the minimizer."""
        return optimistix.OptaxMinimiser(
            optax.lbfgs(), atol=atol, rtol=rtol, norm=optimistix.max_norm
        )

    if uy is None:
        uy = jnp.broadcast_to(1.0, y.shape)
    hx = invert(ux, x) if ux is not None else None
    hy = invert(uy, y)
    optimum = optimistix.minimise(
        misfit, make_minimizer(), x, max_steps=max_steps, throw=False
    )
    xopt = optimum.value
    xcov, xunc = post(xopt)
    cost = misfit(xopt)
    info = jnp.where(optimum.result == optimistix.RESULTS.successful, 0, 1)

    return xopt, xcov, xunc, cost, info


@jax.jit(static_argnums=(0,), static_argnames=("max_steps",))
def _batch(
    f: Callable[[Array], Array],
    x: Array,
    y: Array,
    ux: Array | None = None,
    uy: Array | None = None,
    *,
    atol: Any = DEFAULT_ATOL,
    rtol: Any = DEFAULT_RTOL,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> tuple[Array, Array, Array, Array, Array]:
    r"""
    Optimal estimation (OE) retrieval.

    The implementation accepts any combination of full-rank
    or diagonal-rank uncertainty tensors:
    :math:`U(\check{X}) \in \mathbb{R}^{M \times m \times m}`,
    :math:`U(\check{X}) \in \mathbb{R}^{M \times m}`,
    :math:`U(Y) \in \mathbb{R}^{M \times n \times n}`, and
    :math:`U(Y) \in \mathbb{R}^{M \times n}`.

    Standard uncertainty is not accepted and must be squared to
    a variance (diagonal uncertainty tensor) before supplied as
    argument.

    :param f: The function.
    :param x: Samples :math:`\check{X} \in \mathbb{R}^{M \times m}`.
    :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
    :param ux: Uncertainty tensor :math:`U(\check{X})`, full or diagonal.
    :param uy: Uncertainty tensor :math:`U(Y)`, full or diagonal.
    :param atol: The absolute tolerance for terminating the optimization.
    :param rtol: The relative tolerance for terminating the optimization.
    :param max_steps: The maximum number of steps the optimizer can take.
    :returns: The retrieval result.
    """

    def sample(x, y, ux, uy):
        """OE retrieval without keyword-only arguments."""
        return _sample(
            f, x, y, ux, uy, atol=atol, rtol=rtol, max_steps=max_steps
        )

    batch = jax.vmap(
        sample,
        in_axes=(
            0,
            0,
            None if ux is None else 0,
            None if uy is None else 0,
        ),
    )
    return batch(x, y, ux, uy)


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
        atol: Any = DEFAULT_ATOL,
        rtol: Any = DEFAULT_RTOL,
        max_steps: int = DEFAULT_MAX_STEPS,
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
        passed to an optimizer.

        Under the same notation and remarks as :class:`F`:

        :param f: The function.
        :param x: Samples :math:`\check{X} \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :param ux: Standard uncertainties :math:`u(\check{X})`.
        :param uy: Standard uncertainties :math:`u(Y)`.
        :param atol: The absolute tolerance for terminating the optimization.
        :param rtol: The relative tolerance for terminating the optimization.
        :param max_steps: The maximum number of steps the optimizer can take.
        :returns: The retrieved result.
        """
        xopt, xcov, xunc, cost, info = _batch(
            f.f,
            jnp.asarray(x),
            jnp.asarray(y),
            jnp.square(ux) if ux is not None else None,
            jnp.square(uy) if uy is not None else None,
            atol=atol,
            rtol=rtol,
            max_steps=max_steps,
        )
        xopt = np.asarray(xopt)
        zvar = np.var(f.eval(xopt) - y, axis=0)
        return Retrieved(
            xopt=xopt,
            xcov=np.asarray(xcov),
            xunc=np.asarray(xunc),
            zvar=zvar,
            cost=np.asarray(jnp.sum(cost)),
            info=np.asarray(info),
        )
