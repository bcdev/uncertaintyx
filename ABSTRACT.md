# Tensor-level uncertainty propagation for EO  

In contemporary Earth observation (EO), measurement models are implemented
as end‑to‑end data‑processing codes that transform raw satellite observations
into geophysical products. These codes evolve continuously as algorithms,
auxiliary data, and implementations change, which makes traditional,
hand‑crafted uncertainty bookkeeping brittle and difficult to maintain.
Classical GUM‑style treatments, based on fixed analytical expressions and
static data flows, provide important principles but do not naturally
accommodate this reality of evolving software systems.

A different route is to treat the data‑processing code itself as the
primary object of differentiation. Modern machine learning frameworks
already provide highly optimized algorithmic differentiation (AD) backends,
which can be repurposed to obtain Jacobians directly from the actual
implementation whenever needed. This allows local linearizations to
remain exactly consistent with the running code, rather than with a
separate analytical surrogate, and makes it possible to formulate
uncertainty propagation directly in terms of algorithmically
differentiable programs.

On top of these AD backends, tensor‑level formulations of the law of
propagation of uncertainty can be implemented that operate directly
on multidimensional EO data. Instead of flattening images, spectra,
and spatiotemporal cubes into vectors, one can work with Jacobian
and covariance tensors whose shapes reflect the underlying spatial
and temporal structure. This preserves correlations and aligns more
naturally with how EO scientists think about their data, while still
enabling efficient propagation of full covariance information.

A central novelty of this approach is that Jacobian and uncertainty
tensors become active citizens **inside** the data‑processing codes,
not just external annotations or post‑hoc diagnostics. Sensitivities
and uncertainties can be computed, transformed, and consumed as part
of the workflow itself—for example to steer parameter estimation,
drive adaptive algorithms, or expose uncertainty‑aware outputs at
multiple stages of a processing chain. The concept is currently being
explored on concrete EO sub‑algorithms as initial testbeds, but is
designed to be generic: any data‑processing pipeline that can be
expressed in an AD‑enabled framework (such as JAX) can, in principle,
be turned into an algorithmically differentiable measurement model
with automatic, tensor‑level uncertainty propagation built in.

> [!NOTE]
> Abstract submitted to [ESA $\Phi$innovation Summit 2026](https://philab.esa.int/phinnovation/).
