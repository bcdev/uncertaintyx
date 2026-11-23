#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest
from importlib import resources
from typing import Any
from typing import Any

import numpy as np
import pandas as pd
from numpy import dtype
from numpy import float64
from numpy import ndarray

from uncertaintyx.f.jax import ToF
from uncertaintyx.m.jax import ToM
from uncertaintyx.oceancolour.supacool import AtmosphericCorrection
from uncertaintyx.oceancolour.supacool import AtmosphericSimulation
from uncertaintyx.oceancolour.supacool import HSI
from uncertaintyx.plot.plots import LinePlot

PAD_WITH_BOOST = 6
"""
The number of samples padded to the red edge of the
HyperBOOST sampling to avoid boundary effects.
"""

SAMPLING_BOOST = np.linspace(355.2, 800.7, 136)
"""The HyperBOOST spectral sampling."""

SAMPLING_BOOST_PADDED = np.pad(
    SAMPLING_BOOST, (0, PAD_WITH_BOOST), "linear_ramp", end_values=820.5
)
"""
The HyperBOOST spectral sampling padded at the red
edge to avoid boundary effects.

The padding here requires an equidistant sampling, which
applies to So-Rad but not to HTRSB radiometers.
"""

SAMPLING_CHIME = np.linspace(404.2, 799.0, 48)
"""The CHIME spectral sampling."""


def make_ramp_spectrum_batch(x: np.ndarray, div: Any = 1.0) -> np.ndarray:
    """Returns a batch with a single linear ramp spectrum."""
    return x[np.newaxis, :] / div


def make_unit_spectrum_batch(x: np.ndarray) -> np.ndarray:
    """Returns a batch with a single unit spectrum."""
    return np.ones_like(x)[np.newaxis, :]


def read_boost_data_padded(
    package: str, filename: str
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Returns a HyperBOOST spectrum data table.

    :param package: The package name.
    :param filename: The filename.
    :returns: The data table.
    """
    with resources.path(package, filename) as resource:
        with open(resource) as r:
            df = pd.read_csv(r, sep=";", header=None, index_col=None)
            m = df.values.shape[1] // 2
            n = df.values[:, 0:1]
            rrs = df.values[:, 1 : m + 1]
            std = df.values[:, m + 1 :]
            unc = std / np.sqrt(n)
            rrs = np.pad(rrs, ((0, 0), (0, 6)), "edge")
            unc = np.pad(unc, ((0, 0), (0, 6)), "edge")
    return rrs, unc, rrs.shape[0], rrs.shape[1]


def read_chime_data(
    package: str, filename: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns a CHIME table.

    :param package: The package name.
    :param filename: The filename.
    :returns: The data table.
    """
    with resources.path(package, filename) as resource:
        with open(resource) as r:
            df = pd.read_csv(r, sep=";")
    return df.values[:, 1], df.values[:, 2]


def get_rra(wav: np.ndarray, toa: np.ndarray) -> tuple[np.ndarray, ...]:
    """
    Returns the CHIME relative radiometric accuracy at the
    top of the atmosphere.
    """
    a, b, = read_chime_data(
        "test.resources.oceancolour", "chime_rra.csv"
    )
    ell = 3.95 - 0.57 * np.log(wav)
    """The aquatic-to-terrestrial reference radiance ratio."""
    terrest = 0.025 + 150.0 / (a + b)
    aquatic = 0.025 + 150.0 / (a + b * ell)
    ell = toa * ell / (1.0E06 * np.power(wav, -2.663))
    """The aquatic-to-terrestrial sample radiance ratio."""
    sample = 0.025 + 150.0 / (a + b * ell)
    return terrest, aquatic, sample


def get_snr(wav: np.ndarray, toa: np.ndarray) -> tuple[np.ndarray, ...]:
    """
    Returns the CHIME signal-to-noise ratio at the
    top of the atmosphere.
    """
    g, n, = read_chime_data(
        "test.resources.oceancolour", "chime_snr.csv"
    )
    ell = 3.95 - 0.57 * np.log(wav)
    """The aquatic-to-terrestrial reference radiance ratio."""
    terrest = g / np.sqrt(n + 1.0)
    aquatic = (g * ell) / np.sqrt(n + ell)
    ell = toa * ell / (1.0E06 * np.power(wav, -2.663))
    """The aquatic-to-terrestrial sample radiance ratio."""
    sample = (g * ell) / np.sqrt(n + ell)
    return terrest, aquatic, sample


class SpectrumTest(unittest.TestCase):
    """Tests the CHIME/HSI spectral convolution model."""

    def test_unit_spectrum(self):
        """
        Tests the unit spectrum.
        """
        x_s = SAMPLING_BOOST_PADDED
        x_t = SAMPLING_CHIME
        f = HSI(x_s, x_t)
        self.assertIsInstance(f, ToM)

        y_s = make_unit_spectrum_batch(x_s)
        y_t = f.eval(f.prior(), y_s)
        self.assertIsInstance(y_t, np.ndarray)
        self.assertEqual((1, 48), y_t.shape)
        self.assertTrue(np.allclose(y_t, 1.0))

        K = f.kernel_matrix(f.prior())  # noqa: N806
        y_k = (K @ y_s.T).T
        self.assertTrue(np.allclose(y_k, y_t))

    def test_ramp_spectrum(self):
        """
        Tests the linear ramp spectrum.
        """
        x_s = SAMPLING_BOOST_PADDED
        x_t = SAMPLING_CHIME
        f = HSI(x_s, x_t)
        self.assertIsInstance(f, ToM)

        y_s = make_ramp_spectrum_batch(x_s, 10.0)
        y_t = f.eval(f.prior(), y_s)
        self.assertIsInstance(y_t, np.ndarray)
        self.assertEqual((1, 48), y_t.shape)
        self.assertTrue(np.allclose(y_t, x_t / 10.0))

        K = f.kernel_matrix(f.prior())  # noqa: N806
        y_k = (K @ y_s.T).T
        self.assertTrue(np.allclose(y_k, y_t))

    def test_hyperboost_spectrum(self):
        """
        Tests a spectrum from the HyperBOOST dataset.
        """
        x_s = SAMPLING_BOOST_PADDED
        x_t = SAMPLING_CHIME
        f = HSI(x_s, x_t)
        self.assertIsInstance(f, ToM)

        y_s, u_s, _, _ = read_boost_data_padded(
            "test.resources.oceancolour", "hyperboost.csv"
        )
        y_t = f.eval(f.prior(), y_s)
        self.assertIsInstance(y_t, np.ndarray)
        self.assertEqual((1, 48), y_t.shape)

        LinePlot().plot(
            [x_s, x_t],
            [y_s[0], y_t[0]],
            xrange=(405.0, 795.0),
            yrange=(0.0, 1.5e-02),
            labels=["HyperBOOST", "CHIME/HSI"],
            title="Spectral convolution",
            xlabel=r"wavelength $\lambda$ (nm)",
            ylabel=r"remote sensing reflectance (sr$^{-1}$)",
            savefig="hsi_spectrum.png" if True else None,
        )

        u_c = np.sqrt(
            f.lpu_p(f.prior(), np.square(np.asarray([0.15])), y_s, diag=True)
        )
        u_t = np.sqrt(f.lpu_x(f.prior(), y_s, np.square(u_s), diag=True))

        g = AtmosphericSimulation(x_t)
        h = AtmosphericCorrection(x_t)
        self.assertIsInstance(g, ToF)
        self.assertIsInstance(h, ToF)

        toa = g.eval(y_t)
        snr_terrest, snr_aquatic, snr = get_snr(x_t, toa)
        rra_terrest, rra_aquatic, rra = get_rra(x_t, toa)
        u_n = np.sqrt(h.lpu(toa, np.square(toa / snr), diag=True))
        u_a = np.sqrt(
            h.lpu(toa, np.square(toa * rra / np.sqrt(12.0)), diag=True)
        )

        LinePlot().plot(
            [x_t, x_t, x_t],
            [rra[0], rra_terrest, rra_aquatic],
            xrange=(405.0, 795.0),
            yrange=(0.00, 0.05),
            labels=[
                "aquatic sample",
                "terrestrial reference",
                "aquatic reference",
            ],
            xlabel=r"wavelength $\lambda$ (nm)",
            ylabel=r"relative radiometric accuracy",
            semilogy=False,
            columns=3,
            savefig="hsi_rra.png" if True else None,
        )

        LinePlot().plot(
            [x_t, x_t, x_t],
            [snr[0], snr_terrest, snr_aquatic],
            xrange=(405.0, 795.0),
            yrange=(0.0, 450.0),
            labels=[
                "aquatic sample",
                "terrestrial reference",
                "aquatic reference",
            ],
            xlabel=r"wavelength $\lambda$ (nm)",
            ylabel=r"signal-to-noise ratio",
            semilogy=False,
            columns=3,
            savefig="hsi_snr.png" if True else None,
        )

        LinePlot().plot(
            [x_t, x_t, x_t],
            [np.pi * y_t[0], 1.0E06 * np.power(x_t, -2.663), toa[0]],
            xrange=(405.0, 795.0),
            yrange=(0.00, 0.15),
            labels=[
                "water-leaving reflectance (example)",
                "top-of-atmosphere aquatic reference",
                "top-of atmosphere reflectance (example)",
            ],
            xlabel=r"wavelength $\lambda$ (nm)",
            ylabel=r"reflectance",
            semilogy=False,
            columns=1,
            savefig="hsi_spectrum_toa.png" if True else None,
        )

        LinePlot().plot(
            [x_t, x_t, x_t, x_t],
            [y_t[0], u_n[0], u_a[0], u_c[0]],
            xrange=(405.0, 795.0),
            yrange=(1.5e-09, 1.5e-02),
            labels=[
                "Rrs",
                "Rrs radiometric uncertainty (random)",
                "Rrs radiometric uncertainty (systematic)",
                "Rrs convolution uncertainty",
            ],
            xlabel=r"wavelength $\lambda$ (nm)",
            ylabel=r"remote sensing reflectance (sr$^{-1}$)",
            semilogy=True,
            columns=1,
            savefig="hsi_spectrum_unc.png" if True else None,
        )

        LinePlot().plot(
            [x_t, x_t, x_t, x_t],
            [
                y_t[0] / y_t[0],
                u_n[0] / y_t[0],
                u_a[0] / y_t[0],
                u_c[0] / y_t[0],
            ],
            xrange=(405.0, 795.0),
            yrange=(1.5e-06, 1.5),
            labels=[
                "Rrs",
                "Rrs radiometric uncertainty (random)",
                "Rrs radiometric uncertainty (systematic)",
                "Rrs convolution uncertainty",
            ],
            xlabel=r"wavelength $\lambda$ (nm)",
            ylabel=r"relative reflectance",
            semilogy=True,
            columns=1,
            savefig="hsi_spectrum_unc_relative.png" if True else None,
        )

        rng = np.random.default_rng(42)
        y_t = rng.normal(y_t, np.sqrt(u_t**2 + u_c**2 + u_n**2))

        LinePlot().plot(
            [x_s, x_t],
            [y_s[0], y_t[0]],
            xrange=(405.0, 795.0),
            yrange=(0.0, 1.5e-02),
            labels=["HyperBOOST", "CHIME/HSI"],
            xlabel=r"wavelength $\lambda$ (nm)",
            ylabel=r"remote sensing reflectance (sr$^{-1}$)",
            savefig="hsi_spectrum_noisy.png" if True else None,
        )

        LinePlot().plot(
            [x_t, x_t, x_t, x_t, x_t],
            [y_t[0], u_n[0], u_a[0], u_c[0], u_t[0]],
            xrange=(405.0, 795.0),
            yrange=(1.5e-09, 1.5e-02),
            labels=[
                "Rrs CHIME/HSI",
                "Rrs radiometric uncertainty (random)",
                "Rrs radiometric uncertainty (systematic)",
                "Rrs convolution uncertainty",
                "Rrs HyperBOOST uncertainty propagated",
            ],
            xlabel=r"wavelength $\lambda$ (nm)",
            ylabel=r"remote sensing reflectance (sr$^{-1}$)",
            semilogy=True,
            columns=1,
            savefig="hsi_spectrum_noisy_unc.png" if True else None,
        )


if __name__ == "__main__":
    unittest.main()
