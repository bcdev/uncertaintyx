#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from ...interface.core import Fitting
from ...interface.core import M
from ...interface.core import Result
from ...m.jax import jac_x
from ...m.jax import vec_x


@jax.jit(static_argnums=0)
def evm_fit(f, p, x, y, ux, uy, max_iter: int = 100, tol_g: float = 1.0e-06):
    optim = optax.lbfgs()
    state = optim.init(p)

    def loss(p: Array):
        """The loss function."""
        return 0.5 * jnp.sum(
            (vec_x(f, p, x) - y) ** 2 / (uy**2 + (jac_x(f, p, x) * ux) ** 2)
        )

    cost_and_grad = optax.value_and_grad_from_state(loss)
    cost, grad = cost_and_grad(p, state=state)
    init = 0, p, state, cost, grad, False

    def cond(carry):
        """The function to check the loop condition."""
        i, _, _, _, _, converged = carry
        return jnp.logical_and(i < max_iter, jnp.logical_not(converged))

    def body(carry):
        """The loop body function."""
        i, popt, state, cost, grad, _ = carry
        u, state = optim.update(
            grad, state, popt, value=cost, grad=grad, value_fn=loss
        )
        popt = optax.apply_updates(popt, u)
        cost, grad = cost_and_grad(popt, state=state)
        converged = jnp.linalg.norm(grad, ord=jnp.inf) < tol_g  # noqa
        return i + 1, popt, state, cost, grad, converged

    return jax.lax.while_loop(cond, body, init)  # noqa


class EVM(Fitting):
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
        i, popt, state, cost, g, converged = evm_fit(
            f.f,
            jnp.asarray(f.estimate(x, y)),
            jnp.asarray(x),
            jnp.asarray(y),
            jnp.asarray(ux) if ux is not None else jnp.ones_like(x),
            jnp.asarray(uy) if uy is not None else jnp.ones_like(x),
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
