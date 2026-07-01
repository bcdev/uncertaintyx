#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from uncertaintyx.m.jax import ToM


class HSI(ToM):
    """The CHIME/HSI convolution model function."""

    def __init__(
        self,
        x_s: np.ndarray,
        x_t: np.ndarray,
        sigma: Any = 3.46,
    ):
        x_s = jnp.asarray(x_s)
        x_t = jnp.asarray(x_t)
        w_t = (x_t[-1] - x_t[0]) / (len(x_t) - 1)

        def f(p, y_s):
            """Simulates a CHIME/HSI spectrum."""

            def f(s, t):
                """The function :math:`f(t)`."""
                return jax.lax.erf(t / (jnp.sqrt(2.0) * s))

            def g(s, t):
                """The function :math:`g(t)`."""
                term = jnp.exp(-0.5 * (t / s) ** 2)
                return term * s / (jnp.sqrt(2.0 * jnp.pi))

            def h(s, t, u, x_i, y_i, m_i):
                """
                The antiderivative function.

                The antiderivative is a double convolution integral.
                The first convolution is a Gaussian filter reflecting
                the spectral grating, while the second convolution is
                a boxcar filter reflecting the detector element.

                The double integral is evaluated here: https://www.wolframalpha.com/input?i2d=true&i=Divide%5BIntegrate%5B%5C%2840%29Subscript%5Bm%2Ci%5D+%5C%2840%29t+-+Subscript%5Bx%2Ci%5D%5C%2841%29+%2B+Subscript%5By%2Ci%5D%5C%2841%29+exp%5C%2840%29-Divide%5B1%2C2%5D+Power%5B%5C%2840%29Divide%5B%5C%2840%29u+-+t%5C%2841%29%2Cs%5D%5C%2841%29%2C2%5D%5C%2841%29%2Ct%2Cu%5D%2Cs+sqrt%5C%2840%292+%CF%80%5C%2841%29%5D
                """
                a = 0.5 * f(s, t - u)
                b = 0.5 * g(s, t - u)

                c = m_i * (s**2 - t**2 + u**2 + 2.0 * (t - u) * x_i)
                d = m_i * (t + u - 2.0 * x_i) + 2.0 * y_i

                return 0.5 * a * (c - 2.0 * (t - u) * y_i) - b * d

            u_i = x_t[:, jnp.newaxis] - 0.5 * w_t
            u_j = x_t[:, jnp.newaxis] + 0.5 * w_t

            x_i = x_s[jnp.newaxis, :-1]
            y_i = y_s[jnp.newaxis, :-1]

            x_j = x_s[jnp.newaxis, 1:]
            y_j = y_s[jnp.newaxis, 1:]

            m_i = (y_j - y_i) / (x_j - x_i)

            A = h(p[0], x_j, u_j, x_i, y_i, m_i)  # noqa: N806
            B = h(p[0], x_i, u_j, x_i, y_i, m_i)  # noqa: N806
            C = h(p[0], x_j, u_i, x_i, y_i, m_i)  # noqa: N806
            D = h(p[0], x_i, u_i, x_i, y_i, m_i)  # noqa: N806

            return jnp.sum((A - B) - (C - D), axis=-1) / w_t

        super().__init__(f, jit=False)
        self._sigma = sigma

    def prior(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        preset: str | None = None,
    ) -> np.ndarray:
        """
        Returns the width of the Gaussian smoothing kernel (nm).
        """
        return np.asarray([self._sigma])
