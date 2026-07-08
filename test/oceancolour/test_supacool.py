#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest
from importlib import resources
from typing import Any

import numpy as np
import pandas as pd

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
) -> tuple[np.ndarray, int, int]:
    """
    Returns a HyperBOOST spectrum data table.

    :param package: The package name.
    :param filename: The filename.
    :returns: The data table.
    """
    with resources.path(package, filename) as resource:
        with open(resource) as r:
            df = pd.read_csv(r, sep=";", header=None, index_col=None)
            rrs = np.pad(df.values, ((0, 0), (0, 6)), "edge")
    return rrs, rrs.shape[0], rrs.shape[1]


def read_chime_data(
    package: str, filename: str
) -> tuple[np.ndarray, int, int]:
    """
    Returns a CHIME table.

    :param package: The package name.
    :param filename: The filename.
    :returns: The data table.
    """
    with resources.path(package, filename) as resource:
        with open(resource) as r:
            df = pd.read_csv(r, sep=";", header=None, index_col=None)
            tab = df.values
    return tab, tab.shape[0], tab.shape[1]


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

        y_s, _, _ = read_boost_data_padded(
            "test.resources.oceancolour", "hyperboost.csv"
        )
        y_t = f.eval(f.prior(), y_s)
        self.assertIsInstance(y_t, np.ndarray)
        self.assertEqual((1, 48), y_t.shape)

        LinePlot().plot(
            [x_s, x_t],
            [y_s[0], y_t[0]],
            xrange=(405.0, 795.0),
            yrange=(1.5e-04, 1.5e-02),
            labels=["HyperBOOST", "CHIME/HSI"],
            title="Spectral convolution",
            xlabel=r"wavelength $\lambda$ (nm)",
            ylabel=r"remote sensing reflectance $\rho(\lambda)$ (sr$^{-1}$)",
            savefig="hsi_spectrum.png" if True else None,
        )

        u_c = np.sqrt(
            f.lpu_p(f.prior(), np.square(np.asarray([0.15])), y_s, diag=True)
        )

        g = AtmosphericSimulation(x_t)
        h = AtmosphericCorrection(x_t)
        self.assertIsInstance(g, ToF)
        self.assertIsInstance(h, ToF)

        snr, _, _ = read_chime_data(
            "test.resources.oceancolour", "chime_snr.csv"
        )
        rra, _, _ = read_chime_data(
            "test.resources.oceancolour", "chime_rra.csv"
        )

        toa = g.eval(y_t)
        u_n = np.sqrt(h.lpu(toa, np.square(toa / snr), diag=True))
        u_a = np.sqrt(
            h.lpu(toa, np.square(toa * rra / np.sqrt(12.0)), diag=True)
        )

        LinePlot().plot(
            [x_t, x_t, x_t, x_t],
            [y_t[0], u_n[0], u_a[0], u_c[0]],
            xrange=(405.0, 795.0),
            yrange=(1.5e-09, 1.5e-02),
            labels=[
                "RRS",
                "RRS radiometric uncertainty (random)",
                "RRS radiometric uncertainty (systematic)",
                "RRS convolution uncertainty (systematic)",
            ],
            xlabel=r"wavelength $\lambda$ (nm)",
            ylabel=r"reflectance (sr$^{-1}$)",
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
                "RRS",
                "RRS radiometric uncertainty (random)",
                "RRS radiometric uncertainty (systematic)",
                "RRS convolution uncertainty (systematic)",
            ],
            xlabel=r"wavelength $\lambda$ (nm)",
            ylabel=r"relative reflectance",
            semilogy=True,
            columns=1,
            savefig="hsi_spectrum_unc_relative.png" if True else None,
        )


if __name__ == "__main__":
    unittest.main()
