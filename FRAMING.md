# Metrology and uncertainty

Future satellite missions such as ESA [TRUTHS](https://www.esa.int/Applications/Observing_the_Earth/TRUTHS)
and NASA [CPF](https://science.nasa.gov/mission/clarreo-pathfinder/)
will, for the first time, allow radiometric calibration that is
traceable to SI metrological standards. This raises a fundamental
question for any remote‑sensing activity: if the measurements reach
metrological quality, can the processing algorithms keep up—or do
they throw away that precision because uncertainty is not properly
propagated?

Expressing EO algorithm logic in a differentiation‑enabled framework
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

# Differentiable programming

Everyone wants explainable AI, but most “ML” in Earth Observation still
acts as a black box that ignores the physics we already know.

Our idea is to flip that around: start from existing, physics‑based
EO algorithms and express their logic in a differentiation‑enabled framework.
Algorithmic differentiation then provides exact sensitivities of every
output to every input and parameter, directly from the real code. Jacobians
and covariance tensors become operational data inside the algorithms, so
you can see and quantify how the physics drives the predictions and how
uncertainty propagates through each step.

Bringing physics‑based equations into a differentiation‑enabled framework
creates a natural bridge between physics and machine learning: physical
models stay in charge of structure and constraints, while learned components
fill in what the physics does not capture, all within one differentiable,
uncertainty‑aware program. The result is a class of physics‑informed ML
systems for EO that are both high‑performance and inherently explainable,
because their behaviour is rooted in—and analysable through—the underlying
physics.
