#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
from typing import Any
from typing import Literal
from typing import Self

import jax
import jax.lax.linalg as jla
import jax.numpy as jnp
import numpy as np
import optax
import optimistix
from jax import Array
from jax.scipy.special import gammaln

from uncertaintyx.g.jax import ToG
from uncertaintyx.m.jax import ToM


def _binom(i: Array, k: int) -> Array:
    """
    Returns the binomial coefficients for the Bernstein basis
    of degree :math:`k`.
    """
    return jnp.exp(gammaln(k + 1) - (gammaln(i + 1) + gammaln(k - i + 1)))


_b_basis = jax.custom_jvp(
    lambda k, x: (
        lambda i: (
            _binom(i, k)
            * jnp.power(x[jnp.newaxis, :], i)
            * jnp.power(1.0 - x[jnp.newaxis, :], k - i)
        )
    )(jnp.arange(k + 1)[:, jnp.newaxis]),
    nondiff_argnums=(0,),
)
r"""
Evaluates the Bernstein basis of degree :math:`n`.

Internal closure to enable registration of a custom
Jacobian-vector product (JVP).
"""


def _b_basis_jvp(
    k: int, inputs: tuple[Array], perturbations: tuple[Array]
) -> tuple[Array, Array]:
    r"""
    Custom forward-mode differentiation (JVP) for the Bernstein basis.

    This implementation overrides automatic differentiation using the
    mathematical identity:

    .. math::

        G_{x} B_{i,k}(x) =
        k \left(B_{i-1, k-1}(x) - B_{i, k-1}(x)\right) .

    In the context of automatic differentiation, inputs are usually
    referred to as `primals` while perturbations are referred to as
    `tangents`.

    :param k: The degree of the basis.
    :param inputs: A tuple containing :math:`x \in \mathbb{R}^{m}`.
    :param perturbations: A tuple containing the perturbation of :math:`x`.
    :returns: A tuple containing the basis and its perturbation.
    """
    (x,) = inputs
    (d,) = perturbations

    if k == 0:
        basis = _b_basis(k, x)
        return basis, jnp.zeros_like(basis)

    basis_minus = _b_basis(k - 1, x)
    a = jnp.pad(basis_minus, ((1, 0), (0, 0)))
    b = jnp.pad(basis_minus, ((0, 1), (0, 0)))
    g = k * (a - b)
    basis = b + (a - b) * x[jnp.newaxis, :]
    basis_perturbation = g * d[jnp.newaxis, :]

    return basis, basis_perturbation


_b_basis.defjvp(_b_basis_jvp)


@jax.jit(static_argnums=(0,))
def b_basis(k: int, x: Array) -> Array:
    r"""
    Evaluates the Bernstein basis of degree :math:`n`.

    :param k: The degree of the basis.
    :param x: The input :math:`x \in \mathbb{R}^{m}`.
    :returns: The basis :math:`B_{k}(x) \in \mathbb{R}^{(k + 1) \times m}`.
    """
    return _b_basis(k, x)


@jax.jit
def b_poly_grid(c: Array, x: tuple[Array, ...]) -> Array:
    r"""
    Evaluates an N-variate Bernstein polynomial on a regular
    grid of points.

    The function is designed to fit coefficients for the efficient
    approximation of N-dimensional lookup tables.

    Facilitated by the tensor product structure of a regular grid,
    the implementation uses sequential tensor contraction (Tucker
    product).

    Let :math:`N \in \mathbb{N}` be the arity of the Bernstein
    polynomial. Let :math:`k = (k_{1}, \dots, k_{N}) \in \mathbb{N}^{N}`
    denote its degrees and let :math:`\mathbb{R}^{k + 1}` denote the
    tensor space with dimensions :math:`(k_{1} + 1, \dots, k_{N} + 1)`.
    Likewise, let :math:`m = (m_{1}, \dots , m_{N}) \in \mathbb{N}^{N}`
    denote the dimensions of the grid coordinates
    :math:`x = (x_{1}, \dots, x_{N})` and let :math:`\mathbb{R}^{m}`
    denote the tensor space with dimensions :math:`m`. Then:

    :param c: The Bernstein coefficients :math:`c \in \mathbb{R}^{k + 1}`.
    :param x: The grid coordinates :math:`x`.
    :returns: The polynomial values :math:`B(x) \in \mathbb{R}^{m}`.
    """
    N = c.ndim  # noqa: N806
    k = tuple(c.shape[i] - 1 for i in range(N))

    for i in range(N):  # the Tucker product
        c = jnp.tensordot(c, b_basis(k[i], x[i]), (0, 0))

    return c


@jax.jit
def b_poly_point(c: Array, x: Array) -> Array:
    r"""
    Evaluates an N-variate Bernstein polynomial on a single point.

    Let :math:`N \in \mathbb{N}` be the arity of the Bernstein
    polynomial. Let :math:`k = (k_{1}, \dots, k_{N}) \in \mathbb{N}^{N}`
    denote its degrees and let :math:`\mathbb{R}^{k + 1}` denote the
    tensor space with dimensions :math:`(k_{1} + 1, \dots, k_{N} + 1)`.
    Let :math:`x \in \mathbb{R}^{N}` be a point. Then:

    :param c: The Bernstein coefficients :math:`c \in \mathbb{R}^{k + 1}`.
    :param x: The point :math:`x \in \mathbb{R}^{N}`.
    :returns: The polynomial value :math:`B(x) \in \mathbb{R}`.
    """
    N = c.ndim  # noqa: N806
    k = tuple(c.shape[i] - 1 for i in range(N))
    x = x[jnp.newaxis, :]

    for i in range(N):
        basis = b_basis(k[i], x[:, i])
        c = jnp.tensordot(c, basis[:, 0], (0, 0))

    return c


@jax.jit(static_argnums=(0,), static_argnames=("non_negative", "max_steps"))
def b_solve(
    k: tuple[int, ...],
    x: tuple[Array, ...],
    y: Array,
    *,
    non_negative: bool = False,
    atol: Any = 1.0e-08,
    rtol: Any = 1.0e-06,
    max_steps: int = 256,
) -> Array:
    """
    Solves the normal equation in the least squares sense to fit an
    N-variate Bernstein polynomial to an N-variate lookup table.

    :param k: The degrees of the Bernstein polynomial.
    :param x: The grid coordinates.
    :param y: The grid values.
    :param non_negative: Whether coefficients must be non-negative.
    :param atol: The absolute tolerance for terminating the solver.
    :param rtol: The relative tolerance for terminating the solver.
    :param max_steps: The maximum number of steps the solver can take.
    :returns: The Bernstein coefficients.
    """

    def hvp(c: Array):
        """The Hessian-vector product."""
        res = c
        for i in range(N):
            res = jnp.tensordot(res, R[i], axes=(0, 1))
        for i in range(N):
            res = jnp.tensordot(res, R[i], axes=(0, 0))
        return res

    def nnls(c: Array, rhs: Array):
        """
        Non-negative least-squares solver.

        Uses QR factorization to compute a stable unconstrained
        solution. Applies a softplus transformation and an L-BFGS
        optimizer to ensure non-negativity.
        """

        def forward(u: Array) -> Array:
            """The forward softplus transformation."""
            return jnp.log(1.0 + jnp.exp(u))

        def inverse(c_: Array) -> Array:
            """The inverse softplus transformation."""
            return jnp.log(jnp.expm1(c_))

        def misfit(u: Array, _: None = None) -> Array:
            """The misfit function with quadratic transformation."""
            c_ = forward(u)
            return 0.5 * jnp.sum(c_ * hvp(c_)) - jnp.sum(c_ * rhs)

        def make_minimizer():
            """Returns the minimizer."""
            return optimistix.OptaxMinimiser(
                optax.lbfgs(), atol=atol, rtol=rtol, norm=optimistix.max_norm
            )

        # compute the right hand side of the normal equation
        for i in range(N):
            rhs = jnp.tensordot(rhs, R[i], axes=(0, 0))

        u = inverse(jnp.abs(c))
        optimum = optimistix.minimise(
            misfit, make_minimizer(), u, max_steps=max_steps, throw=False
        )
        return forward(optimum.value)

    N = len(k)  # noqa: N806
    bases = [b_basis(k[i], x[i]) for i in range(N)]
    facts = [jla.qr(B.T, full_matrices=False) for B in bases]  # noqa: N806
    Q = [_[0] for _ in facts]  # noqa: N806
    R = [_[1] for _ in facts]  # noqa: N806

    # compute the right hand side of the triangular equation
    rhs = y
    for i in range(N):
        rhs = jnp.tensordot(rhs, Q[i], axes=(0, 0))
    # solve the triangular equation
    c_unconstrained = rhs
    for i in range(N):
        c_unconstrained = jla.triangular_solve(
            R[i], c_unconstrained, left_side=True
        )
        c_unconstrained = jnp.moveaxis(  # like the tensor dot product
            c_unconstrained, 0, -1
        )
    # solve iteratively with non-negativity constraint, if needed
    nnls_needed = non_negative and jnp.any(c_unconstrained < 0.0)
    return jax.lax.cond(
        nnls_needed,
        nnls,
        lambda c, _: c,
        c_unconstrained,
        rhs,
    )


def _lower_bounds(
    a: np.ndarray | Literal["auto"] | Any, x: tuple[np.ndarray, ...]
) -> np.ndarray:
    """Returns the lower coordinate bounds."""
    if not isinstance(a, np.ndarray):
        n = len(x)
        a = np.asarray([_.min() for _ in x]) if a == "auto" else np.full(n, a)
    return a


def _upper_bounds(
    b: np.ndarray | Literal["auto"] | Any, x: tuple[np.ndarray, ...]
) -> np.ndarray:
    """Returns the upper coordinate bounds."""
    if not isinstance(b, np.ndarray):
        n = len(x)
        b = np.asarray([_.max() for _ in x]) if b == "auto" else np.full(n, b)
    return b


class BernsteinGrid(ToG):
    r"""
    Evaluates an N-variate Bernstein polynomial on a regular
    grid of points.

    Encapsulates the multivariate Bernstein polynomial

    .. math::
        B: \mathbb{R}^{k + 1} \to \mathbb{R}^{m}, \quad
        c \mapsto B_{k}(c, x)

    where :math:`N \in \mathbb{N}` is the arity of the Bernstein
    polynomial, :math:`k = (k_{1}, \dots, k_{N}) \in \mathbb{N}^{N}`
    are its degrees, :math:`\mathbb{R}^{k + 1}` is the tensor space
    with dimensions :math:`(k_{1} + 1, \dots, k_{N} + 1)`,
    :math:`m = (m_{1}, \dots , m_{N}) \in \mathbb{N}^{N}` are the
    dimensions of the grid coordinates :math:`x = (x_{1}, \dots, x_{N})`,
    and :math:`\mathbb{R}^{m}` is the tensor space with dimensions
    :math:`m`.
    """

    def __init__(
        self,
        x: tuple[np.ndarray, ...],
        a: np.ndarray | Literal["auto"] | Any = 0.0,
        b: np.ndarray | Literal["auto"] | Any = 1.0,
    ):
        """
        Creates a new instance of this class.

        :param x: The grid coordinates :math:`x`.
        :param a: The lower bounds of the grid coordinates.
        :param b: The upper bounds of the grid coordinates.
        """
        a = _lower_bounds(a, x)
        b = _upper_bounds(b, x)
        x = tuple(jnp.asarray((x_ - a) / (b - a)) for x_ in x)

        def f(c: Array) -> Array:
            r"""
            The N-variate Bernstein polynomial on a regular grid
            of points.

            :param c: The coefficients :math:`c \in \mathbb{R}^{k + 1}`.
            """
            return b_poly_grid(c, x)

        super().__init__(f, rev=False)


class BernsteinPoly(ToM):
    r"""
    Evaluates an N-variate Bernstein polynomial on a batch of
    irregularly distributed points.

    Encapsulates the multivariate Bernstein polynomial

    .. math::
        B: \mathbb{R}^{k + 1} \times \mathbb{R}^{N}
        \to \mathbb{R}, \quad
        (c, x) \mapsto B(c, x)

    where :math:`N \in \mathbb{N}` is the arity of the Bernstein
    polynomial, :math:`k = (k_{1}, \dots, k_{N}) \in \mathbb{N}^{N}`
    denotes its degrees, and :math:`\mathbb{R}^{k + 1}` denotes the
    tensor space with dimensions :math:`(k_{1} + 1, \dots, k_{N} + 1)`.
    """

    def __init__(
        self,
        c: np.ndarray,
        a: np.ndarray | Any = 0.0,
        b: np.ndarray | Any = 1.0,
    ):
        r"""
        Creates a new instance of this class.

        :param c: The prior coefficients :math:`c \in \mathbb{R}^{k + 1}`.
        :param a: The lower bounds of the point coordinates.
        :param b: The upper bounds of the point coordinates.
        """
        N = c.ndim  # noqa: : N806
        a = jnp.asarray(a) if isinstance(a, np.ndarray) else jnp.full(N, a)
        b = jnp.asarray(b) if isinstance(b, np.ndarray) else jnp.full(N, b)

        def f(c: Array, x: Array) -> Array:
            r"""
            Evaluates an N-variate Bernstein polynomial on a single point.

            :param c: The coefficients :math:`c \in \mathbb{R}^{k + 1}`.
            :param x: The point :math:`x \in \mathbb{R}^{N}`.
            """
            return b_poly_point(c, (x - a) / (b - a))

        super().__init__(f)
        self._c = c

    @classmethod
    def from_lookup_table(
        cls,
        k: tuple[int, ...],
        x: tuple[np.ndarray, ...],
        y: np.ndarray,
        a: np.ndarray | Literal["auto"] | Any = 0.0,
        b: np.ndarray | Literal["auto"] | Any = 1.0,
        *,
        non_negative: bool = False,
        atol: Any = 1.0e-08,
        rtol: Any = 1.0e-06,
        max_steps: int = 256,
    ) -> Self:
        """
        Factory method to fit an N-variate Bernstein polynomial to
        an N-variate lookup table.

        Applies a linear least squares solver.

        :param k: The degrees of the Bernstein polynomial.
        :param x: The lookup table coordinates.
        :param y: The lookup table values.
        :param a: The lower bounds of the table coordinates.
        :param b: The upper bounds of the table coordinates.
        :param non_negative: Whether coefficients must be non-negative.
        :param atol: The absolute tolerance for terminating the solver.
        :param rtol: The relative tolerance for terminating the solver.
        :param max_steps: The maximum number of steps the solver can take.
        """
        a = _lower_bounds(a, x)
        b = _upper_bounds(b, x)
        x_ = tuple(jnp.asarray((x_ - a) / (b - a)) for x_ in x)
        y_ = jnp.asarray(y)
        c_ = b_solve(
            k,
            x_,
            y_,
            non_negative=non_negative,
            atol=atol,
            rtol=rtol,
            max_steps=max_steps,
        )
        return cls(np.array(c_), a, b)

    def prior(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        preset: str | None = None,
    ) -> np.ndarray:
        return self._c
