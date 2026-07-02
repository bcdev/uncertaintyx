#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest
from typing import Any

import numpy as np

from uncertaintyx.m.jax import ToM
from uncertaintyx.oceancolour.supacool import HSI

SAMPLING_CHIME = np.linspace(404.2, 799.0, 48)
"""The CHIME spectral sampling."""

SAMPLING_SORAD = np.linspace(355.2, 800.7, 136)
"""The SoRad spectral sampling."""


def make_triangle_spectrum_batch(x: np.ndarray, div: Any = 1.0) -> np.ndarray:
    """Returns a batch with a single triangle spectrum."""
    return np.broadcast_to(x / div, (1,) + x.shape)


def make_unit_spectrum_batch(x: np.ndarray) -> np.ndarray:
    """Returns a batch with a single unit spectrum."""
    return np.broadcast_to(np.ones_like(x), (1,) + x.shape)


class SpectrumTest(unittest.TestCase):
    """Tests the CHIME/HSI spectral convolution model."""

    def test_unit_spectrum(self):
        """
        Tests the unit spectrum.

        The test spectrum is padded beyond the HyperBOOST
        spectral range to avoid boundary effects.
        """
        x_s = np.pad(SAMPLING_SORAD, (0, 6), "linear_ramp", end_values=820.5)
        f = HSI(x_s, SAMPLING_CHIME)
        self.assertIsInstance(f, ToM)

        y_s = make_unit_spectrum_batch(x_s)
        y_t = f.eval(f.prior(), y_s)
        self.assertIsInstance(y_t, np.ndarray)
        self.assertEqual((1, 48), y_t.shape)
        self.assertTrue(np.allclose(y_t, 1.0))

    def test_triangle_spectrum(self):
        """
        Tests the triangle spectrum.

        The test spectrum is padded beyond the HyperBOOST
        spectral range to avoid boundary effects.
        """
        x_s = np.pad(SAMPLING_SORAD, (0, 6), "linear_ramp", end_values=820.5)
        f = HSI(x_s, SAMPLING_CHIME)
        self.assertIsInstance(f, ToM)

        y_s = make_triangle_spectrum_batch(x_s, 10.0)
        y_t = f.eval(f.prior(), y_s)
        self.assertIsInstance(y_t, np.ndarray)
        self.assertEqual((1, 48), y_t.shape)
        self.assertTrue(np.allclose(y_t, SAMPLING_CHIME / 10.0))


if __name__ == "__main__":
    unittest.main()
