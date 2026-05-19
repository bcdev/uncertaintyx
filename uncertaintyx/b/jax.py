#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln


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
def b_poly(b: Array, x: Array) -> Array:
    r"""
    Evaluates a Bernstein polynomial.

    :param b: The Bernstein coefficients :math:`b \in \mathbb{R}^{(n + 1)}`.
    :param x: The input :math:`x \in \mathbb{R}^{m}`.
    :returns: The polynomial values :math:`B(x) \in \mathbb{R}^{m}`.
    """
    return jnp.dot(b, b_basis(b.shape[-1] - 1, x))


@jax.jit
def b_poly_grid(b: Array, x: tuple[Array, ...]) -> Array:
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

    :param b: The Bernstein coefficients :math:`b \in \mathbb{R}^{k + 1}`.
    :param x: The grid coordinates :math:`x`.
    :returns: The polynomial values :math:`B(x) \in \mathbb{R}^{m}`.
    """
    N = b.ndim  # noqa: N806
    k = tuple(b.shape[i] - 1 for i in range(N))

    for i in range(N):  # the Tucker product
        b = jnp.tensordot(b, b_basis(k[i], x[i]), (0, 0))

    return b


@jax.jit
def b_poly_point(b: Array, x: Array) -> Array:
    r"""
    Evaluates an N-variate Bernstein polynomial on a single point.

    Let :math:`N \in \mathbb{N}` be the arity of the Bernstein
    polynomial. Let :math:`k = (k_{1}, \dots, k_{N}) \in \mathbb{N}^{N}`
    denote its degrees and let :math:`\mathbb{R}^{k + 1}` denote the
    tensor space with dimensions :math:`(k_{1} + 1, \dots, k_{N} + 1)`.
    Let :math:`x \in \mathbb{R}^{N}` be a point. Then:

    :param b: The Bernstein coefficients :math:`b \in \mathbb{R}^{k + 1}`.
    :param x: The point :math:`x \in \mathbb{R}^{N}`.
    :returns: The polynomial value :math:`B(x) \in \mathbb{R}`.
    """
    N = b.ndim  # noqa: N806
    k = tuple(b.shape[i] - 1 for i in range(N))
    x = x[jnp.newaxis, :]

    for i in range(N):
        basis = b_basis(k[i], x[:, i])
        b = jnp.tensordot(b, basis[:, 0], (0, 0))

    return b


@jax.jit
def b_poly_points(b: Array, x: Array) -> Array:
    r"""
    Evaluates a multivariate Bernstein polynomial on a batch of
    irregularly distributed N-dimensional points.

    Under the same notation as :meth:`b_poly_point` let
    :math:`X \in \mathbb{R}^{M \times N}` be a batch of
    points over the outer batch dimension :math:`M `.
    Then:

    :param b: The Bernstein coefficients :math:`b \in \mathbb{R}^{k + 1}`.
    :param x: The points :math:`X \in \mathbb{R}^{M \times N}`.
    :returns: The polynomial values :math:`B(x) \in \mathbb{R}^{M}`.
    """
    return jax.vmap(b_poly_point, in_axes=(None, 0))(b, x)
