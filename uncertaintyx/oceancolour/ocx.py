#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
"""
Empirical model functions for NASA Ocean Colour algorithms. Refer to the
Algorithm Publication Tool (https://doi.org/10.5067/JCQB8QALDOYD).

Refer to Lachlan et al. (2019, https://doi.org/10.3389/feart.2019.00176)
for propagating radiometric data uncertainties through NASA Ocean Colour
algorithms.

Further references:

Hu et al. (2019). Improving Satellite Global Chlorophyll a Data
Products Through Algorithm Refinement and Data Recovery. Journal
of Geophysical Research: Oceans, 124(3), 1524-1543.
https://doi.org/10.1029/2019JC014941.

O'Reilly and Werdell (2019). Chlorophyll algorithms for ocean
color sensors - OC4, OC5 & OC6. Remote Sensing of Environment,
229, 32-47. https://doi.org/10.1016/j.rse.2019.04.021.
"""

from typing import Literal

import jax.numpy as jnp
import numpy as np

from ..m.jax import ToM


class CI(ToM):
    """
    NASA's ocean colour chlorophyll index (CI) model function.

    The nearest wavebands to 443, 555, and 670 nm are used for the blue,
    green, and red, respectively, for all sensors.
    """

    def __init__(self):
        """Creates a new model function instance."""

        def f(p, x):
            r"""
            The model function.

            Let :math:`p = (a_0, a_1) \in \mathbb{R}^{2}` be the model
            parameters and let :math:`\lambda = (\lambda_\mathrm{b},
            \lambda_\mathrm{g}, \lambda_\mathrm{r}) \in \mathbb{R}^{3}`
            denote the central wavelengths of wavebands in the blue,
            green and red, respectively. Further let

            .. math::
                x = (\lambda, R_\mathrm{rs}(\lambda))
                \in \mathbb{R}^{2 \times 3}

            denote the function inputs. Then:

            :param p: The parameters :math:`p \in \mathbb{R}^{2}`.
            :param x: The inputs :math:`x \in \mathbb{R}^{2 \times 3}`.
            :returns: The chlorophyll concentration (mg m-3).
            """

            def maybe_convert(g, G):  # noqa: N806
                """
                Converts remote sensing reflectance in the green
                waveband, if its central wavelength is not within
                :math:`555\pm2 \mathrm{nm}`.
                """
                c = jnp.asarray(
                    [
                        [0.001723, 0.986, 0.081495, 1.031, 0.000216],
                        [0.001597, 0.988, 0.062195, 1.014, 0.000128],
                        [0.001148, 1.023, -0.103624, 0.979, -0.000121],
                        [0.000891, 1.039, -0.183044, 0.971, -0.000170],
                    ]
                )
                return jnp.where(
                    g < 547.5,
                    convert(G, c[0]),
                    jnp.where(
                        g < 553.0,
                        convert(G, c[1]),
                        jnp.where(
                            g > 562.5,
                            convert(G, c[2]),
                            jnp.where(
                                g > 557.0,
                                convert(G, c[2]),
                                G,
                            ),
                        ),
                    ),
                )

            def convert(x, c):
                sw, a1, b1, a2, b2 = c
                return jnp.where(
                    x < sw,
                    jnp.power(10.0, a1 * jnp.log10(x) - b1),
                    a2 * x - b2,
                )

            a0, a1 = p
            b = x[0, 0]
            g = x[0, 1]
            r = x[0, 2]
            B = x[1, 0]  # noqa: N806
            G = maybe_convert(g, x[1, 1])  # noqa: N806
            R = x[1, 2]  # noqa: N806
            c = G - (B + (R - B) * (g - b) / (r - b))

            return jnp.power(10.0, a0 + a1 * c)

        super().__init__(f)

    def estimate(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        preset: str | None = None,
    ) -> np.ndarray:
        """Returns the OCI default parameter values (Hu et al., 2019)."""
        return np.array([-0.4287, 230.47])


class OC4(ToM):
    """
    NASA's OC4 (and OC3) chlorophyll model function.

    The three (or two) nearest wavebands to 412, 443, 490 and 510 nm
    are used for the blue, while the nearest waveband to 555 nm is
    used for the green, for all sensors.
    """

    def __init__(self):
        """Creates a new model function instance."""

        def f(p, x):
            r"""
            The model function.

            Let :math:`p = (p_0, \dots p_4) \in \mathbb{R}^{5}` be the
            model parameters and let :math:`\lambda = (\lambda_1,\dots
            \lambda_\mathrm{m}) \in \mathbb{R}^{m}` denote the central
            wavelengths of :math:`2 \le m-1 \le 3` wavebands in the blue,
            and a single waveband :math:`\lambda_{m}` in the green.
            Further let

            .. math::
                x = R_\mathrm{rs}(\lambda) \in \mathbb{R}^{m}

            denote the function inputs. Then:

            :param p: The parameters :math:`p \in \mathbb{R}^{5}`.
            :param x: The inputs :math:`x \in \mathbb{R}^{m}`.
            :returns: The chlorophyll concentration (mg m-3).
            """
            b = jnp.log10(jnp.nanmax(x[:-1]) / x[-1])
            return jnp.power(10.0, jnp.polyval(p[::-1], b))

        super().__init__(f)

    def estimate(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        preset: Literal[
            "CZCS",
            "GOCI",
            "HAWKEYE",
            "MERIS",
            "MODIS",
            "OCTS",
            "OLCI",
            "PACE",
            "SEAWIFS",
            "VIIRS20",
            "VIIRS21",
        ]
        | None = None,
    ) -> np.ndarray:
        """
        Returns the OCX default parameter values for different sensors.

        The default parameter set returned is for SeaWiFS.
        """
        params = [0.32814, -3.20725, 3.22969, -1.36769, -0.81739]
        match preset:
            case "CZCS":
                params = [0.31841, -4.56386, 8.63979, -8.41411, 1.91532]
            case "GOCI":
                params = [0.28043, -2.49033, 1.53980, -0.09926, -0.68403]
            case "HAWKEYE":
                params = [0.32814, -3.20725, 3.22969, -1.36769, -0.81739]
            case "MERIS":
                params = [0.42487, -3.20974, 2.89721, -0.75258, -0.98259]
            case "MODIS":
                params = [0.26294, -2.64669, 1.28364, 1.08209, -1.76828]
            case "OCTS":
                params = [0.54655, -3.51799, 3.39128, -0.91567, -0.97112]
            case "OLCI":
                params = [0.42540, -3.21679, 2.86907, -0.62628, -1.09333]
            case "PACE":  # OCI
                params = [0.32814, -3.20725, 3.22969, -1.36769, -0.81739]
            case "SEAWIFS":
                params = [0.32814, -3.20725, 3.22969, -1.36769, -0.81739]
            case "VIIRS20":  # NOAA-20
                params = [0.28153, -2.65472, 1.30882, 1.31521, -2.08622]
            case "VIIRS21":  # NOAA-21
                params = [0.24765, -2.54926, 1.55323, 0.39485, -1.54632]
            case _:
                pass
        return np.array(params)


class OCI(ToM):
    """
    The blended OC4/CI model function.

    The two or three nearest wavebands to 412, 443, 490 and 510 nm
    are used for the blue, while the nearest waveband to 555 nm is
    used for the green, and the nearest waveband to 670 nm is used
    for the red.

    The blending occurs when :class:`OC4` is between, e.g.,
    0.25-0.35 mg m-3, creating a smooth handover between :class:`OCI`
    (low chlorophyll specialist) and :class:`OC4` (baseline). The
    blending uses perfectly normalized weights. Low :class:`CI`
    values favour :class:`CI` while high :class:`CI` values
    favour :class:`OC4` through self-weighting. A quadratic term
    creates curvature that eliminates boundary discontinuities while
    :class:`OC4` acts as regime detector.
    """

    def __init__(self, b: Literal[0, 1] = 0):
        """
        Creates a new model function instance.

        :param b: The index of the blue waveband near 443 nm.
        """
        self.m_ci: ToM = CI()
        self.m_oc: ToM = OC4()

        def f(p, x):
            r"""
            The blended OCI/OCX model function.

            The blending is self-adjusting.

            Let :math:`p = \in \mathbb{R}^{k}, k = 2 + 2 + 5` be the
            model parameters and let :math:`\lambda = (\lambda_1,\dots
            \lambda_\mathrm{m}) \in \mathbb{R}^{m}` denote the central
            wavelengths of :math:`2 \le m-2 \le 3` wavebands in the blue,
            and a single waveband :math:`\lambda_{m-1}` in the green, and
            a single waveband :math:`\lambda_{m}` in the red. Further let

            .. math::
                x = (\lambda, R_\mathrm{rs}(\lambda))
                \in \mathbb{R}^{2 \times m}

            denote the function inputs. Then:

            :param p: The parameters :math:`p \in \mathbb{R}^{k}`.
            :param x: The inputs :math:`x \in \mathbb{R}^{2 \times m}`.
            :returns: The chlorophyll concentration (mg m-3).
            """

            def blend(p, ci, cx):
                """Returns the blended chlorophyll concentration."""
                ti = p[1] - ci
                tx = ci - p[0]
                return (ci * ti + cx * tx) / (p[1] - p[0])

            ci = self.m_ci.f(p[2:4], x[:, [b, -2, -1]])
            oc = self.m_oc.f(p[4:9], x[1, 0:-1])

            return jnp.where(
                oc < p[0],
                ci,
                jnp.where(oc > p[1], oc, blend(p, ci, oc)),
            )

        super().__init__(f)

    def estimate(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        preset: Literal[
            "CZCS",
            "GOCI",
            "HAWKEYE",
            "MERIS",
            "MODIS",
            "OCTS",
            "OLCI",
            "PACE",
            "SEAWIFS",
            "VIIRS20",
            "VIIRS21",
        ]
        | None = None,
    ) -> np.ndarray:
        """
        Returns the blended OCI/OCX default parameter values.

        Elements ``[0:2]`` refer to the blending, elements ``[2:4]``
        refer to CI, and elements ``[4:9]`` refer to OC4.
        """
        return np.concatenate(
            (
                np.array([0.25, 0.35]),
                self.m_ci.estimate(x, y),
                self.m_oc.estimate(x, y, preset),
            )
        )
