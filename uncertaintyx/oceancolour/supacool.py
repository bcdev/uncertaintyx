#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from uncertaintyx.f.jax import ToF
from uncertaintyx.m.jax import ToM


def P_a(m: Any = -1.0, g: Any = 0.7):  # noqa: N806
    """
    The Henyey-Greenstein aerosol phase function.

    :param m: The cosine of the scattering angle.
    :param g: The asymmetry factor.
    :returns: The aerosol scattering phase.
    """
    return (1.0 - g**2) / (1.0 + g**2 - 2.0 * g * m) ** 1.5


def P_r(m: Any = -1.0):  # noqa: N806
    """
    The Rayleigh phase function.

    :param m: The cosine of the scattering angle.
    :returns: The Rayleigh scattering phase.
    """
    return 0.75 * (1.0 + m**2)


def R_a(  # noqa: N806
    x,
    t: Any = 0.0,
    a: Any = 1.0,
    m_s: Any = 1.0,
    m_v: Any = 1.0,
    m_d: Any = 1.0,
    w: Any = 0.98,
):
    """
    Returns the aerosol path reflectance.

    :param x: The spectral wavelength (nm).
    :param t: The aerosol optical thickness at 550 nm.
    :param a: The aerosol Angstrom exponent.
    :param m_s: The cosine of the solar zenith angle.
    :param m_v: The cosine of the view zenith angle.
    :param m_d: The cosine of the azimuthal difference angle.
    :param w: The single scattering albedo.
    :returns: The aerosol path reflectance.
    """
    m = scattering_cosine(m_s, m_v, m_d)
    return (w * P_a(m) * tau_a(x, t, a)) / (4.0 * m_s * m_v)


def R_r(x, m_s: Any = 1.0, m_v: Any = 1.0, m_d: Any = 1.0):  # noqa: N806
    """
    Returns the Rayleigh path reflectance.

    :param x: The spectral wavelength (nm).
    :param m_s: The cosine of the solar zenith angle.
    :param m_v: The cosine of the view zenith angle.
    :param m_d: The cosine of the azimuthal difference angle.
    :returns: The Rayleigh path reflectance.
    """
    m = scattering_cosine(m_s, m_v, m_d)
    return (P_r(m) * tau_r(x)) / (4.0 * m_s * m_v)


def T_a(  # noqa: N806
    x, t: Any = 0.0, a: Any = 1.0, m_s: Any = 1.0, m_v: Any = 1.0
):
    """
    Returns the aerosol transmittance.

    :param x: The spectral wavelength (nm).
    :param t: The aerosol optical thickness at 550 nm.
    :param a: The aerosol Angstrom exponent.
    :param m_s: The cosine of the solar zenith angle.
    :param m_v: The cosine of the view zenith angle.
    :returns: The aerosol transmittance.
    """
    return jnp.exp(-tau_a(x, t, a) * (1.0 / m_s + 1.0 / m_v))


def T_r(x, m_s: Any = 1.0, m_v: Any = 1.0):  # noqa: N806
    """
    Returns the Rayleigh transmittance.

    :param x: The spectral wavelength (nm).
    :param m_s: The cosine of the solar zenith angle.
    :param m_v: The cosine of the view zenith angle.
    :returns: The Rayleigh transmittance.
    """
    return jnp.exp(-tau_r(x) * (1.0 / m_s + 1.0 / m_v))


def tau_a(x, t: Any = 0.0, a: Any = 1.0):
    """
    Returns the aerosol optical thickness of a clear maritime
    atmosphere.

    :param x: The spectral wavelength (nm).
    :param t: The aerosol optical thickness at 550 nm.
    :param a: The aerosol Angstrom exponent.
    :returns: The aerosol optical thickness.
    """
    return t * (550.0 / x) ** a


def tau_r(x):
    """
    Returns the Rayleigh optical thickness of the atmosphere.

    Uses the approximation of
    `Dutton et al., (1994) <https://doi.org/10.1029/93JD03520>`_
    at standard pressure.

    :param x: The spectral wavelength (nm).
    :returns: The Rayleigh optical depth.
    """
    return 0.00877 * (x / 1000.0) ** -4.05


def scattering_cosine(m_s: Any = 1.0, m_v: Any = 1.0, m_d: Any = 1.0):
    """
    Returns the cosine of the scattering angle.

    :param m_s: The cosine of the solar zenith angle.
    :param m_v: The cosine of the view zenith angle.
    :param m_d: The cosine of the azimuthal difference angle.
    :returns: The cosine of the scattering angle.
    """
    return jnp.sqrt(1.0 - m_s**2) * jnp.sqrt(1.0 - m_v**2) * m_d - m_s * m_v


def spherical_albedo(x, t: Any = 0.0, a: Any = 1.0, g: Any = 0.7):
    """
    Returns the spherical albedo of the atmosphere.

    :param x: The spectral wavelength (nm).
    :param t: The aerosol optical thickness at 550 nm.
    :param a: The aerosol Angstrom exponent.
    :param g: The Henyey-Greenstein asymmetry factor.
    :returns: The spherical albedo of the atmosphere.
    """
    return tau_r(x) / 3.0 + (1.0 - g) * tau_a(x, t, a)


class AtmosphericCorrection(ToF):
    """
    Propagates top-of-atmosphere reflectance to aquatic
    remote sensing reflectance.

    Only a pure Rayleigh scattering atmosphere and normal
    view and illumination conditions are considered.
    """

    def __init__(self, wav: np.ndarray):
        """
        Creates a new atmospheric correction function.

        :param wav: The spectral wavelength (nm).
        """
        wav_ = jnp.asarray(wav)

        def f(toa):
            """
            The atmospheric correction function.

            :param toa: The top-of-atmosphere reflectance.
            :returns: The remote sensing reflectance (sr-1)
            """
            r = R_r(wav_) + R_a(wav_)
            t = T_r(wav_) * T_a(wav_)
            s = spherical_albedo(wav_)
            return (toa - r) / (jnp.pi * (t + s * (toa - r)))

        super().__init__(f)


class AtmosphericSimulation(ToF):
    """
    Propagates aquatic remote sensing reflectance to
    top-of-atmosphere reflectance.

    Only a pure Rayleigh scattering atmosphere and normal
    view and illumination conditions are considered.
    """

    def __init__(self, wav: np.ndarray):
        """
        Creates a new atmospheric simulation function.

        :param wav: The spectral wavelength (nm).
        """
        wav_ = jnp.asarray(wav)

        def f(rrs):
            """
            The atmospheric simulation function.

            :param rrs: The remote sensing reflectance (sr-1)
            :returns: The top-of-atmosphere reflectance.
            """
            r = R_r(wav_) + R_a(wav_)
            t = T_r(wav_) * T_a(wav_)
            s = spherical_albedo(wav_)
            return r + (jnp.pi * t * rrs) / (1.0 - jnp.pi * s * rrs)

        super().__init__(f)


class HSI(ToM):
    r"""
    The CHIME/HSI spectral convolution model.

    The spectral convolution with the CHIME/HSI instrument spectral
    response function (reflecting the spectral grating) is conducted
    analytically by interpreting each source spectrum as a polyline and
    evaluating the convolution using piecewise integration by parts. The
    subsequent box-car resampling (reflecting the detector element) is
    also performed analytically.

    The underlying indefinite double integral (or antiderivative) is
    defined and evaluated in this `WolframAlpha Query`_.

    .. _WolframAlpha Query: https://www.wolframalpha.com/input?i2d=true&i=
        Divide%5BIntegrate%5B%5C%2840%29Subscript%5Bm%2Ck%5D+%5C%2840%29t+
        -+Subscript%5Bx%2Ck%5D%5C%2841%29+%2B+Subscript%5By%2Ck%5D%5C%2841
        %29+exp%5C%2840%29-Divide%5B1%2C2%5D+Power%5B%5C%2840%29Divide%5B%
        5C%2840%29u+-+t%5C%2841%29%2Cs%5D%5C%2841%29%2C2%5D%5C%2841%29%2Ct
        %2Cu%5D%2Cs+sqrt%5C%2840%292+%CF%80%5C%2841%29%5D

    The methodology borrows from techniques in computational astrophysics
    spectroscopy and yields a flux-conserving and fully closed-form solution
    for both the spectral convolution and the resampling across all target
    and source spectral bands.
    """

    def __init__(
        self,
        x_s: np.ndarray,
        x_t: np.ndarray,
        sigma: Any = 3.46,
    ):
        """
        Creates a new CHIME/HSI spectral convolution model.

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

            The algorithm applies the two-dimensional Fundamental
            Theorem of Calculus over a rectangular domain.

            The only model parameter is the width of the Gaussian
            smoothing kernel (nm).

            :param p: The model parameters.
            :param y_s: The source remote sensing reflectance (sr-1).
            :returns: The target remote sensing reflectance (sr-1).
            """
            w_t = jnp.diff(x_t)
            h_i = 0.5 * jnp.pad(w_t, (1, 0), "edge")
            h_j = 0.5 * jnp.pad(w_t, (0, 1), "edge")

            u_i = x_t[:, jnp.newaxis] - h_i[:, jnp.newaxis]
            u_j = x_t[:, jnp.newaxis] + h_j[:, jnp.newaxis]

            x_k = x_s[jnp.newaxis, :-1]
            y_k = y_s[jnp.newaxis, :-1]

            x_l = x_s[jnp.newaxis, 1:]
            y_l = y_s[jnp.newaxis, 1:]

            m_k = (y_l - y_k) / (x_l - x_k)

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
                The double antiderivative function.
                """
                a = 0.5 * f(s, t - u)
                b = 0.5 * g(s, t - u)

                c = m_k * (s**2 - t**2 + u**2 + 2.0 * (t - u) * x_k)
                d = m_k * (t + u - 2.0 * x_k) + 2.0 * y_k

                return 0.5 * a * (c - 2.0 * (t - u) * y_k) - b * d

            return jnp.sum(
                h(p[0], x_l, u_j)
                - h(p[0], x_k, u_j)
                - h(p[0], x_l, u_i)
                + h(p[0], x_k, u_i),
                axis=-1,
            ) / (h_i + h_j)

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
