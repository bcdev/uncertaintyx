#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

__version__ = "2026.1.0"

import jax

jax.config.update("jax_enable_x64", True)

import torch

torch.set_default_dtype(torch.float64)
