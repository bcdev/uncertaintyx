#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest
from importlib import resources
from typing import Any

import numpy as np
import pandas as pd

from uncertaintyx.m.jax import ToM
from uncertaintyx.oceancolour.supacool import HSI
from uncertaintyx.plot.plots import LinePlot

SAMPLING_CHIME = np.linspace(404.2, 799.0, 48)
"""The CHIME spectral sampling."""

SAMPLING_SORAD = np.linspace(355.2, 800.7, 136)
"""The SoRad spectral sampling."""


def make_triangle_spectrum_batch(x: np.ndarray, div: Any = 1.0) -> np.ndarray:
    """Returns a batch with a single triangle spectrum."""
    return x[np.newaxis, :] / div


def make_unit_spectrum_batch(x: np.ndarray) -> np.ndarray:
    """Returns a batch with a single unit spectrum."""
    return np.ones_like(x)[np.newaxis, :]


def read_hyperboost_data(
    package: str, filename: str
) -> tuple[np.ndarray, int, int]:
    """
    Returns hyperboost spectrum data table.

    :param package: The package name.
    :param filename: The filename.
    :returns: The data table.
    """
    with resources.path(package, filename) as resource:
        with open(resource) as r:
            df = pd.read_csv(r, sep=";", header=None, index_col=None)
            rrs = df.values
    return rrs, rrs.shape[0], rrs.shape[1]


class SpectrumTest(unittest.TestCase):
    """Tests the CHIME/HSI spectral convolution model."""

    def test_unit_spectrum(self):
        """
        Tests the unit spectrum.

        The test spectrum is padded beyond the HyperBOOST
        spectral range to avoid boundary effects.
        """
        x_s = np.pad(SAMPLING_SORAD, (0, 6), "linear_ramp", end_values=820.5)
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

    def test_triangle_spectrum(self):
        """
        Tests the triangle spectrum.

        The test spectrum is padded beyond the HyperBOOST
        spectral range to avoid boundary effects.
        """
        x_s = np.pad(SAMPLING_SORAD, (0, 6), "linear_ramp", end_values=820.5)
        x_t = SAMPLING_CHIME
        f = HSI(x_s, x_t)
        self.assertIsInstance(f, ToM)

        y_s = make_triangle_spectrum_batch(x_s, 10.0)
        y_t = f.eval(f.prior(), y_s)
        self.assertIsInstance(y_t, np.ndarray)
        self.assertEqual((1, 48), y_t.shape)
        self.assertTrue(np.allclose(y_t, x_t / 10.0))

        K = f.kernel_matrix(f.prior())  # noqa: N806
        y_k = (K @ y_s.T).T
        self.assertTrue(np.allclose(y_k, y_t))

    def test_hyperboost_spectrum(self):
        x_s = SAMPLING_SORAD
        x_t = SAMPLING_CHIME
        f = HSI(x_s, x_t)
        self.assertIsInstance(f, ToM)

        y_s, _, _ = read_hyperboost_data(
            "test.resources.oceancolour", "hyperboost.csv"
        )
        y_t = f.eval(f.prior(), y_s)
        self.assertIsInstance(y_t, np.ndarray)
        self.assertEqual((1, 48), y_t.shape)

        fig = LinePlot().plot(
            [x_s, x_t],
            [y_s[0], y_t[0]],
            xrange=(410.0, 790.0),
            yrange=(0.000, 0.015),
            labels=["HyperBOOST", "CHIME/HSI"],
            title="Spectral convolution",
            xlabel=r"wavelength $\lambda$ (nm)",
            ylabel=r"remote sensing reflectance $\rho(\lambda)$ (sr$^{-1}$)",
            savefig="hsi_spectrum.png" if True else None,
        )


if __name__ == "__main__":
    unittest.main()
