![Bernstein basis](https://github.com/user-attachments/assets/80d2ee7c-3e1f-4242-ac07-6880516eaf8e "Bernstein basis")

Future satellite missions such as NASA's [CLARREO Pathfinder](https://science.nasa.gov/mission/clarreo-pathfinder/)
will, for the first time, allow radiometric calibration that is
traceable to SI metrological standards. This raises a fundamental
question for any remote‑sensing activity: if the measurements reach
metrological quality, can the processing algorithms keep up—or do
they throw away that precision because uncertainty is not properly
propagated?

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

Everyone wants explainable AI, but most “ML” in Remote Sensing still
remains a decoupled black box that ignores the physics we already know.

Our idea is to flip that around: start from existing, physics‑based
algorithms and express their logic in a differentiation‑enabled framework.
Algorithmic differentiation (AD) then provides exact sensitivities of every
output to every input and parameter, directly from the real code. Jacobians
and covariance tensors become operational data inside the algorithms, so
you can see and quantify how the physics drives the predictions and how
uncertainty propagates through each step.

Bringing physics‑based equations into a differentiation‑enabled framework
creates a natural bridge between physics and machine learning: physical
models stay in charge of structure and constraints, while learned components
fill in what the physics does not capture, all within one differentiable,
uncertainty‑aware program. The result is a class of physics‑informed ML
systems for Remote Sensing that are both high‑performance and inherently
explainable, because their behaviour is rooted in—and analysable through—the
underlying physics.

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
and coded in general tensor form, is elegantly beautiful. If you’re curious,
compare [NIST TN 1297](https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-appendix-law-propagation-uncertainty) (Equation A-3)
with the tensor equation and code further below.

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
> Tyx passes the GUM example cases with explicit measurement
> models in [JCGM 102:2011](https://doi.org/10.59161/JCGM102-2011)
> (Examples 9.2, 9.3, and 9.4) which are implemented as unit‑level
> tests to verify correctness and accuracy to the last digit
> listed.

# Further reading

Quast, R., Baljeet Singh, Y. K.& Brandt, G. (2026). Turning Uncertainty
Into Knowledge: Inverse Problem Theory Lifted to the Computational
Top-Level [Graphic]. Zenodo. ESA Phinnovation Summit 2026, ESA ESRIN, 
Frascati, Italy. <https://doi.org/10.5281/zenodo.21280786>.

[![CodeQL Advanced](https://github.com/bcdev/uncertaintyx/actions/workflows/codeql.yml/badge.svg)](https://github.com/bcdev/uncertaintyx/actions/workflows/codeql.yml)
[![Python package](https://github.com/bcdev/uncertaintyx/actions/workflows/python-package.yml/badge.svg)](https://github.com/bcdev/uncertaintyx/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/bcdev/uncertaintyx/graph/badge.svg?token=742AWtYDCD)](https://codecov.io/gh/bcdev/uncertaintyx)
![loc](https://img.shields.io/badge/loc-2.6k-blue)

<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']]
  }
};
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
</script>
