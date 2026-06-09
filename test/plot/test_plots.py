#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest
from importlib import resources

import numpy as np
import pandas as pd

from uncertaintyx.oceancolour.carbon import Maranon
from uncertaintyx.oceancolour.carbon import PhytoplanktonCarbon
from uncertaintyx.oceancolour.ocx import OCI
from uncertaintyx.plot.plots import BernsteinBasisPlot
from uncertaintyx.plot.plots import WaterClassLinePlot
from uncertaintyx.plot.plots import WaterClassScatterPlot


def read_owt_data(
    package: str, filename: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Returns the optical water types (OWT) data table.

    :param package: The package name.
    :param filename: The filename.
    :returns: The data table.
    """
    with resources.path(package, filename) as resource:
        rows = []
        with open(resource) as r:
            df = pd.read_csv(r, sep=";", header=None, index_col=0)
            for name, _ in df.items():
                rows.append(df[name].values)
            data = np.stack(rows, axis=-1)
    wav = data[0, :6]
    rrs = data[1:, :6]
    unc = data[1:, 6:]

    return wav, rrs, unc, rrs.shape[0], rrs.shape[1]


class BernsteinBasisPlotTest(unittest.TestCase):
    """Tests plotting Bernstein basis polynomials."""

    def test_plot_bernstein_basis(self):
        plot = BernsteinBasisPlot("paper")
        grid_size = 100
        x = np.linspace(0.0, 1.0, grid_size)
        y = np.linspace(0.0, 1.0, grid_size)

        caption = (
            r"Approximation with Bernstein polynomials smoothly and "
            r"simultaneously fits both the target function and its "
            r"derivatives"
        )
        fig = plot.plot(
            x,
            y,
            degree=2,
            caption=caption,
            cmap="viridis",
            savefig="bernstein_basis.png" if False else None,
        )
        self.assertIsNotNone(fig)


def elasticity(x: np.ndarray, y: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Returns the elasticity."""
    return g * (x / y)


class WaterClassLinePlotTest(unittest.TestCase):
    """Tests plotting water classes."""

    def test_plot_water_classes(self):
        w, R, u, _, _ = read_owt_data(  # noqa : N806
            "test.resources.oceancolour", "owt.csv"
        )

        fig = WaterClassLinePlot("paper").plot(
            w,
            R,
            u,
            xlabel=r"wavelength $\lambda$ (nm)",
            ylabel=r"remote sensing reflectance "
            r"$R_{\mathrm{rs}}(\lambda)$ (sr$^{-1}$)",
            yrange=(-0.002, 0.032),
            title="Water classes (Jackson et al., 2017)",
            savefig="water_classes.png" if True else None,
        )
        self.assertIsNotNone(fig)

    def test_plot_phytoplankton_elasticity(self):
        w, R, _, M, m = read_owt_data(  # noqa : N806
            "test.resources.oceancolour", "owt.csv"
        )
        W = np.broadcast_to(w, (M, m))  # noqa : N806

        f = PhytoplanktonCarbon(Maranon(True))
        x = np.stack([W[:, 1:], R[:, 1:]], axis=1)
        p = f.prior(preset="OC4_MERIS")
        y = f.eval(p, x)
        g = f.jac_x(p, x)

        fig = WaterClassLinePlot().plot(
            w[1:],
            elasticity(x[:, 1, :], y[:, np.newaxis], g[:, 1, :]),
            xlabel=r"wavelength $\lambda$ (nm)",
            ylabel=r"elasticity "
            r"$\epsilon(\log_{10} C_{\mathrm{phy}}, "
            r"R_{\mathrm{rs}}(\lambda))$",
            yrange=(-5.5, 1.5),
            title="Typical elasticity",
            savefig="phytoplankton_elasticity.png" if True else None,
        )
        self.assertIsNotNone(fig)


class WaterClassScatterPlotTest(unittest.TestCase):
    """Tests plotting water classes."""

    def test_plot_phytoplankton_uncertainty(self):
        w, R, _, M, m = read_owt_data(  # noqa : N806
            "test.resources.oceancolour", "owt.csv"
        )
        W = np.broadcast_to(w, (M, m))  # noqa : N806

        f_oc = OCI(True)
        f_pc = PhytoplanktonCarbon(Maranon(True))
        x = np.stack([W[:, 1:], R[:, 1:]], axis=1)
        u = np.stack(
            [
                np.broadcast_to(0.0, (M, 5)),
                np.asarray([[0.05, 0.05, 0.05, 0.05, 0.10]] * R[:, 1:]),
            ],
            axis=1,
        )
        p = f_pc.prior(preset="OC4_MERIS")
        y_oc = f_oc.eval(p, x)
        y_pc = f_pc.eval(p, x)

        U = np.square(u)  # noqa : N806
        U_oc = f_oc.lpu_x(p, x, U)  # noqa : N806
        u_oc = np.sqrt(U_oc)

        U_pc = f_pc.lpu_x(p, x, U)  # noqa : N806
        u_pc = np.sqrt(U_pc)

        fig = WaterClassScatterPlot().plot(
            y_oc,
            y_pc,
            u_oc,
            u_pc,
            xlabel=r"$\log_{10} C_{\mathrm{chl}}$ (mg m$^{-3}$)",
            ylabel=r"$\log_{10} C_{\mathrm{phy}}$ (mg C m$^{-3}$)",
            title="Typical uncertainty",
            savefig="phytoplankton_uncertainty.png" if True else None,
        )
        self.assertIsNotNone(fig)


if __name__ == "__main__":
    unittest.main()
