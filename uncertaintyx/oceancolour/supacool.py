#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from uncertaintyx.m.jax import ToM


class HSI(ToM):
    """
    The CHIME/HSI convolution model class.

    The spectral convolution with the CHIME/HSI instrument spectral
    response function (reflecting the spectral grating) is conducted
    analytically by interpreting each source spectrum as a polyline and
    evaluating the convolution using piecewise integration by parts. The
    subsequent box-car resampling (reflecting the detector element) is
    also performed analytically, yielding a flux-conserving and fully
    closed-form solution for both the spectral convolution and the
    resampling across all CHIME/HSI and source spectral bands. The
    methodology borrows from techniques in computational astrophysics
    spectroscopy.

    The underlying indefinite convolution integral (or antiderivative)
    is defined and evaluated in this `WolframAlpha Query`_.

    .. _WolframAlpha Query: https://www.wolframalpha.com/input?i2d=true&i=Divide%5BIntegrate%5B%5C%2840%29Subscript%5Bm%2Ci%5D+%5C%2840%29t+-+Subscript%5Bx%2Ci%5D%5C%2841%29+%2B+Subscript%5By%2Ci%5D%5C%2841%29+exp%5C%2840%29-Divide%5B1%2C2%5D+Power%5B%5C%2840%29Divide%5B%5C%2840%29u+-+t%5C%2841%29%2Cs%5D%5C%2841%29%2C2%5D%5C%2841%29%2Ct%2Cu%5D%2Cs+sqrt%5C%2840%292+%CF%80%5C%2841%29%5D
    """

    def __init__(
        self,
        x_s: np.ndarray,
        x_t: np.ndarray,
        sigma: Any = 3.46,
    ):
        """
        Creates a new CHIME/HSI convolution model.

        Though the CHIME/HSI target sampling is presumably equidistant,
        the model function does not require equidistant samplings.

        :param x_s: The source spectral sampling (nm).
        :param x_t: The target spectral sampling (nm).
        :param sigma: The width of the Gaussian kernel (nm).
        """
        x_s = jnp.asarray(x_s)
        x_t = jnp.asarray(x_t)

        def f(p, y_s):
            """
            Simulates a CHIME/HSI spectrum.

            The only model parameter is the width of the Gaussian
            smoothing kernel (nm).

            :param p: The model parameters.
            :param y_s: The source remote sensing reflectance (sr-1).
            :returns: The target remote sensing reflectance (sr-1).
            """

            w_t = jnp.pad(jnp.diff(x_t), (0, 1), "edge")
            u_i = x_t[:, jnp.newaxis] - 0.5 * w_t[:, jnp.newaxis]
            u_j = x_t[:, jnp.newaxis] + 0.5 * w_t[:, jnp.newaxis]

            x_i = x_s[jnp.newaxis, :-1]
            y_i = y_s[jnp.newaxis, :-1]

            x_j = x_s[jnp.newaxis, 1:]
            y_j = y_s[jnp.newaxis, 1:]

            m_i = (y_j - y_i) / (x_j - x_i)

            def f(s, t):
                """An auxiliary function."""
                return jax.lax.erf(t / (jnp.sqrt(2.0) * s))

            def g(s, t):
                """An auxiliary function."""
                return (s / jnp.sqrt(2.0 * jnp.pi)) * jnp.exp(
                    -0.5 * (t / s) ** 2
                )

            def h(s, t, u):
                """
                The antiderivative function.

                The antiderivative is a double convolution integral.
                The first convolution is a Gaussian filter reflecting
                the spectral grating, while the second convolution is
                a boxcar filter reflecting the detector element.
                """
                a = 0.5 * f(s, t - u)
                b = 0.5 * g(s, t - u)

                c = m_i * (s**2 - t**2 + u**2 + 2.0 * (t - u) * x_i)
                d = m_i * (t + u - 2.0 * x_i) + 2.0 * y_i

                return 0.5 * a * (c - 2.0 * (t - u) * y_i) - b * d

            A = h(p[0], x_j, u_j)  # noqa: N806
            B = h(p[0], x_i, u_j)  # noqa: N806
            C = h(p[0], x_j, u_i)  # noqa: N806
            D = h(p[0], x_i, u_i)  # noqa: N806

            return jnp.sum((A - B) - (C - D), axis=-1) / w_t

        super().__init__(f)
        self._x_s = x_s
        self._x_t = x_t
        self._sigma = sigma

    def prior(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        preset: str | None = None,
    ) -> np.ndarray:
        """
        Returns the model parameters.

        The only model parameter is the width of the Gaussian
        smoothing kernel (nm).
        """
        return np.asarray([self._sigma])

    def kernel_matrix(self, p: np.ndarray) -> np.ndarray:
        """
        Returns the kernel matrix of the convolution model.

        Multiplying the kernel matrix with a spectrum vector (or a
        batch of spectrum vectors) returns the convoluted spectrum
        quickly.

        :param p: The model parameters.
        :return: The kernel matrix.
        """
        return np.squeeze(
            self.jac_x(
                p, np.ones_like(self._x_s, shape=(1, self._x_s.shape[0]))
            )
        )
