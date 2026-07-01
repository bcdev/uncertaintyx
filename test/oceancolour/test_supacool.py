#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest

import numpy as np

from uncertaintyx.m.jax import ToM
from uncertaintyx.oceancolour.supacool import HSI


class SpectrumTest(unittest.TestCase):
    """Tests the CHIME/HSI spectral convolution model."""

    def test_unit_spectrum(self):
        """
        Tests the unit spectrum.

        The spectrum is padded beyond the HyperBOOST spectral
        range to avoid boundary effects.
        """
        x_s = np.pad(
            np.linspace(355.2, 800.7), (0, 6), "linear_ramp", end_values=820.5
        )
        x_t = np.linspace(404.2, 799.0, 48)
        f = HSI(x_s, x_t)
        self.assertIsInstance(f, ToM)

        y_s = np.broadcast_to(np.ones_like(x_s), (1,) + x_s.shape)
        y_t = f.eval(f.prior(), y_s)
        self.assertIsInstance(y_t, np.ndarray)
        self.assertEqual((1, 48), y_t.shape)
        self.assertTrue(np.allclose(y_t, 1.0))


if __name__ == "__main__":
    unittest.main()
