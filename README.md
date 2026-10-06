[![CodeQL Advanced](https://github.com/bcdev/uncertaintyx/actions/workflows/codeql.yml/badge.svg)](https://github.com/bcdev/uncertaintyx/actions/workflows/codeql.yml)
[![Python package](https://github.com/bcdev/uncertaintyx/actions/workflows/python-package.yml/badge.svg)](https://github.com/bcdev/uncertaintyx/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/bcdev/uncertaintyx/graph/badge.svg?token=742AWtYDCD)](https://codecov.io/gh/bcdev/uncertaintyx)

**Metrology space missions** such as NASA's [CLARREO Pathfinder](https://science.nasa.gov/mission/clarreo-pathfinder/)
will, for the first time, allow radiometric calibration that is
traceable to SI standards. This raises a fundamental question for
any remote‑sensing activity: if the measurements reach metrological
quality, can the processing algorithms keep up—or do they throw away
that precision because uncertainty is not properly propagated?
Expressing algorithm logic in a differentiation‑enabled framework
automatically tracks how uncertainties in inputs, calibration, and
model parameters affect the final products, operating directly on
images and data cubes with their spatial and temporal correlations
intact. Jacobians and covariance tensors become operational data
inside the algorithms, not external reports.
Such a framework makes it possible to deliver products whose uncertainty
is consistent with SI‑traceable measurements, supporting regulatory‑grade
use, high‑value decision making, and sensor‑to‑sensor consistency.
Strategically, it positions providers to offer truly uncertainty‑aware
services that fully exploit upcoming metrological missions, instead of
being limited by legacy ideas that ignore metrology.

**Everyone wants explainable AI**, but often “ML” in Remote Sensing
is a decoupled black box that ignores the physics we already know.
Our idea is to flip that around: by expressing existing, physics‑based
algorithms in a differentiation‑enabled framework, algorithmic
differentiation (AD) effectively translates the physics itself into
the mathematical form of machine learning. Through AD, physical equations
cease to be static code; they become dynamic, differentiable computational
graphs that speak the native language of ML. This structural transformation
creates an inherent, two-way bridge: physical models stay in charge of
structure and constraints, while learned components can be seamlessly
woven into the same differentiable program to fill in what the formulas
do not capture. Crucially, because the physics has been transformed into
a differentiable framework, Jacobians and covariance tensors become
operational data inside the algorithm. You can see and quantify exactly
how uncertainty propagates through each step. The result is a class of
differentiable programming (∂P) systems for Remote Sensing that are both
high‑performance and inherently explainable, because their learning
behavior is natively rooted in—and constrained by—the underlying physics.

# Synopsis

**Uncertaintyx** (or just **Tyx**) is a lightweight framework for
tensor‑level uncertainty propagation, inverse problems, and
metrology‑aware workflows. It produces uncertainty tensors by combining
tensor‑valued models with AD backends such as [JAX](https://docs.jax.dev/).
Conventional [NumPy](https://numpy.org) acts as a bidirectional interoperability layer,
enabling JAX‑based code to interoperate smoothly with existing workflows.

**Why tensors?** Remote sensing imagery provides 2D data, spectral imagers
deliver 3D data, and Earth climate records form 4D datasets—with ocean
and atmosphere data reaching up to 5D. Applying standard matrix-based
uncertainty propagation requires flattening these N-D arrays into 1D
vectors, which obscures the vital spatiotemporal structure of both the
data and the algorithms designed to analyse it. Tensors are the ideal
solution, and the law of propagation of uncertainty, when formulated
and coded in general tensor form, is elegantly beautiful.

**Why JAX?** Traditional methods like finite differences, manual Jacobians,
or Monte Carlo often struggle with scalability for high-dimensional
tensors, demanding extensive evaluations or approximations that compromise
fidelity. Frameworks like JAX, facilitating GPUs and TPUs besides CPUs,
make algorithmic differentiation a Game Changer, automatically generating
exact Jacobians, Hessians and higher order derivatives—even for complex,
nonlinear models.

**How does it work?** You define and code a function that maps from one
tensor space to another:

$$
f: \mathbb{R}^{m_1 \times \cdots \times m_{N_m}} \to 
\mathbb{R}^{n_1 \times \cdots \times n_{N_n}}, \quad
f(x) \mapsto y
$$

Here, $x$ and $y$ may be scalars, vectors, matrices, or higher-order
tensors of arbitrary shape. The function may also depend on parameters 
$p$, which themselves can be tensors of arbitrary shape:

$$
f: \mathbb{R}^{k_1 \times \cdots \times k_{N_k}} \times 
\mathbb{R}^{m_1 \times \cdots \times m_{N_m}} \to 
\mathbb{R}^{n_1 \times \cdots \times n_{N_n}}, \quad
f(p, x) \mapsto y
$$

**Tyx** extends this formulation by introducing a batch dimension
$M \in \mathbb{N}$ into the function signature:

$$
f: \mathbb{R}^{k_1 \times \cdots \times k_{N_k}} \times 
\mathbb{R}^{M \times m_1 \times \cdots \times m_{N_m}} \to 
\mathbb{R}^{M \times n_1 \times \cdots \times n_{N_n}}, \quad
f(p, X) \mapsto Y
$$

The main objective of Tyx is to provide efficient access to
uncertainty tensors for such functions. While Jacobians themselves
are obtained through automatic differentiation, Tyx delivers a
high-level interface, utilities, and structured handling for them.
These Jacobians form the foundation for parameter estimation,
sensitivity analysis, and uncertainty propagation within the
framework.

The **Single-Input Tensor Paradigm** is lightweight and modern,
following the design principles of leading machine learning frameworks.
By accepting a single input tensor of arbitrary shape, the model
remains both flexible and conceptually clean—supporting multiple
logical inputs without cluttering the function signature. Organizing
and assembling these logical inputs into a unified tensor structure is
the user’s responsibility. In this role, you serve as the *Thalamus*—the
interface channelling structured data into the computational core
of Tyx.

> **Note**
> The batch dimension $M$ enumerates independent samples (e.g.,
> sensor scans, simulations, ensemble members) but you get to define
> what “one sample” is: a single pixel value, a spectrum, a scan line,
> or a spatiotemporal cubelet. Tyx treats that single sample as
> a tensor $x$, and the framework scales it to a batch $X$ of $M$ such
> samples. Many remote‑sensing workflows implicitly assume “one sample
> is one pixel”, but this is often an oversimplification that obscures
> the full structure of the data and its uncertainties.

# Law of propagation of uncertainty

Using Einstein's summation convention and the symmetry of the
input uncertainty tensor $U$, the law of propagation of uncertainty
in general tensor form reads:

$$V_{\dots ij} = G_{\dots ik} U_{\dots lk} G_{\dots jl},$$

with multi-indices $k, l \in D \subset \mathbb{N}^d$ for some
$d \in \mathbb{N}$. The summation is taken over all $k, l \in D$.
Here, $D$ denotes the set of inner tensor indices (multi-indices
of length $d$), and the trailing tensor dimensions of the Jacobian
tensor $G$ and the input uncertainty tensor $U$ correspond to
these indices. The code below provides an implementation. 

```python
def make_lpu(d: int) -> Callable[[Array, Array], Array]:
    """
    Returns the law of propagation of uncertainty.

    :param d: The number of inner tensor dimensions.
    :returns: The law of propagation of uncertainty.
    """

    @jax.jit
    def lpu(g: Array, u: Array) -> Array:
        r"""
        The law of propagation of uncertainty.

        :param g: The Jacobian tensor :math:`G`.
        :param u: The uncertainty tensor :math:`U`.
        :returns: The uncertainty tensor :math:`V`.
        """
        dims = tuple(range(-d, 0))
        return jnp.tensordot(jnp.tensordot(g, u, (dims, dims)), g, (dims, dims))

    return lpu
```

Tyx hereby acts as a modern bridge, translating the rigorous logic
of the Law of Propagation of Uncertainty into the high-dimensional,
tensor-valued language of today’s computational frameworks.

> **Note**
> The mathematical approaches taken by Tyx for model parameter estimation
> and uncertainty propagation—specifically utilizing optimal estimation
> and errors-in-variables methods—are conceptually compliant with
> [JCGM Guides in Metrology](https://www.bipm.org/en/committees/jc/jcgm/publications).
> Tyx passes the JCGM example cases with explicit measurement
> models in [JCGM 102:2011](https://doi.org/10.59161/JCGM102-2011)
> (Examples 9.2, 9.3, and 9.4) which are implemented as unit‑level
> tests to verify correctness and accuracy to the last digit listed.

# Quickstart

The unit-test suite serves as the primary source of operational usage
examples. Refer to the `test` directory for practical examples. The
following example shows **how to propagate uncertainties through
a measurement model**:

```python
import jax.numpy as jnp
import numpy as np
from jax import Array

import uncertaintyx.f.jax as tyx

C = jnp.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
"""The sensitivity matrix."""


class AdditiveModel(tyx.ToF):
    """
    The additive measurement model (JCGM 102:2011,
    Example 9.2).
    """

    def __init__(self):
        def f(x: Array) -> Array:
            """The measurement function."""
            return C @ x

        super().__init__(f)


f = AdditiveModel()

X = np.array([[0.0, 0.0, 0.0]])
"""The input."""
U = np.array([[1.0, 1.0, 1.0]])
"""The input uncertainty matrix (diagonal)."""

Y = f.eval(X)
"""The output."""
V = f.lpu(X, U)
"""The output uncertainty matrix."""
G = f.jac(X)
"""The Jacobian matrix."""

np.testing.assert_allclose(Y, np.array([[0.0, 0.0]]))
"""Expect zero output for zero input."""

np.testing.assert_allclose(
    G, np.array([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]])
)
"""
Expect the Jacobian matrix to match the
sensitivity matrix.
"""

np.testing.assert_allclose(
    V, np.array([[[2.0, 1.0], [1.0, 2.0]]])
)
"""
Expect the propagated uncertainty matrix based on the
law of propagation of uncertainty.
"""
```

The following example illustrates **how to conduct an optimal estimation**:

```python
import numpy as np

from uncertaintyx.f.jax import Line
from uncertaintyx.retrieve.oe.jax import OE

M = 100
"""The dimension of a batch."""
m = 10
"""The dimension of a sample."""


def fuzzy(val) -> np.ndarray:
    """Returns a batch filled with fuzzy values."""
    return np.random.normal(val, 1.0, (M, m))


def sharp(val) -> np.ndarray:
    """Returns a batch filled with sharp values."""
    return np.broadcast_to(val, (M, m))


f = Line()
"""
The identity forward model.

Each state parameter is measured directly and
independently (y = x).

The identity model represents a direct observation
of the entire state where the instrument introduces
no structural information loss or cross-talk. The
only source of uncertainty is random measurement
noise.
"""

x = fuzzy(0.0)
"""The prior state parameter values."""
y = fuzzy(0.0)
"""The measurement values."""
ux = sharp(1.0)
"""
The prior state parameter uncertainty matrix, represented
as a vector of variances.
"""
uy = sharp(1.0)
"""
The measurement uncertainty, represented as a vector
of variances.
"""

retrieved = OE().retrieve(f, x, y, ux=ux, uy=uy)
"""
The retrieval result.

Following Tarantola's probabilistic framework, Tyx
finds the maximum a posteriori estimate using
quasi-Newton optimization. The posterior covariance
matrix is then obtained by inverting the Hessian of
the cost function at the minimum.
"""

np.testing.assert_array_equal(retrieved.info, 0)
"""Expect all OE iterations to converge successfully."""

np.testing.assert_allclose(retrieved.xopt, np.mean([x, y], axis=0))
"""
Expect the posterior state parameter values to match the
mean of the prior and the measurement.

Since prior and measurement have equal uncertainty
and a direct 1:1 mapping, the optimal posterior estimate
must be the exact arithmetic mean of the prior and
the measurement.
"""

np.testing.assert_allclose(
    [retrieved.xcov[:, i, j] for i in range(m) for j in range(m) if i == j],
    0.5,
)
"""
Expect 1/2 posterior variance.

Combining two independent sources of unit variance for
a direct measurement halves the variance.
"""

np.testing.assert_allclose(
    [retrieved.xcov[:, i, j] for i in range(m) for j in range(m) if i != j],
    0.0,
)
"""
Expect zero posterior covariance.

Since both the prior and measurement errors are uncorrelated,
the posterior covariance must be zero.
"""
```


# References

Quast, R., Baljeet Singh, Y. K. & Brandt, G. (2026). Turning Uncertainty
Into Knowledge: Inverse Problem Theory Lifted to the Computational
Top-Level [Graphic]. Zenodo. ESA Φnnovation Summit 2026, ESA ESRIN,
Frascati, Italy.  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21280786.svg)](https://doi.org/10.5281/zenodo.21280786)

<script>
  window.MathJax = {
    tex: {
      inlineMath: { '[+]': [['$', '$']] },
      processEscapes: true
    }
  };
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@4/tex-chtml.js"></script>
