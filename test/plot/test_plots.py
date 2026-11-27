#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest
from importlib import resources
from typing import Any

import numpy as np
import pandas as pd

from uncertaintyx.fit.eiv.jax import EIV
from uncertaintyx.fit.randomsampling import Bootstrap
from uncertaintyx.fit.randomsampling import MonteCarlo
from uncertaintyx.fit.regression import HeteroscedasticRegression
from uncertaintyx.m.jax import Linear
from uncertaintyx.oceancolour.carbon import MaranonOCI
from uncertaintyx.oceancolour.ocx import OCI
from uncertaintyx.plot.plots import BernsteinBasisPlot
from uncertaintyx.plot.plots import MatrixPlot
from uncertaintyx.plot.plots import TrendPlot
from uncertaintyx.plot.plots import WaterClassLinePlot
from uncertaintyx.plot.plots import WaterClassScatterPlot
from uncertaintyx.tyx import Fitted


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


def matrix(result: Fitted, domain: tuple[Any, Any], n: int) -> np.ndarray:
    """
    Returns the variance-covariance matrix of the fitted curve.
    """
    return np.squeeze(
        result.ycov_p(np.linspace(domain[0], domain[1], n).reshape(1, n))
    )


def read_time_series_data(
    package: str, filename: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Returns the POC time series.

    :param package: The package name.
    :param filename: The filename.
    :returns: The time series.
    """
    with resources.path(package, filename) as resource:
        rows = []
        with open(resource) as r:
            df = pd.read_csv(r, sep=",")
            for name, _ in df.items():
                rows.append(df[name].values)
            data = np.stack(rows, axis=-1)
    yrs = data[:, 0] + (data[:, 1] - 0.5) / 12.0
    y = data[:, 2]
    u = data[:, 3]

    return yrs, y, u, len(yrs)


class TrendPlotTest(unittest.TestCase):
    """Tests plotting a time series."""

    def test_trend_poc(self):
        x, y, u, n = read_time_series_data(
            "test.resources.oceancolour",
            "global_v6_4km_mld_filled_poc_monthly.csv",
        )
        x = x.reshape((n, 1))
        y = y.reshape((n, 1))
        u = u.reshape((n, 1))

        method = "bootstrap-ols"

        if method == "bootstrap-ols":
            fitting = Bootstrap(
                HeteroscedasticRegression(EIV()), how_many=10000
            )
            result = fitting.fit(
                Linear(), x, y, ux=np.zeros_like(u), uy=np.ones_like(u)
            )
        elif method == "mc-wls":
            fitting = MonteCarlo(
                HeteroscedasticRegression(EIV()), how_many=10000
            )
            result = fitting.fit(Linear(), x, y, ux=np.zeros_like(u), uy=u)
        else:
            fitting = EIV()
            result = fitting.fit(Linear(), x, y, ux=np.zeros_like(u), uy=u)

        self.assertEqual(0, result.info)

        print()
        print("popt = ", result.popt)
        print("punc = ", result.punc)
        print("pcov = ", result.pcov)

        TrendPlot(result).plot(
            x,
            y,
            xlabel=r"year",
            ylabel=r"POC / (Gt C)",
            xrange=(1997.75, 2024.25),
            yrange=(0.7, 1.5),
            xticks=np.linspace(1999, 2023, 9),
            savefig="poc_trend.png" if False else None,
            title="Particulate organic carbon (1998 - 2024)",
        )
        years = (1998.0 + 1.0 / 12, 2024.0 - 1.0 / 12.0)
        MatrixPlot().plot(
            matrix(result, years, n),
            xlabel=r"year",
            ylabel=r"year",
            xrange=years,
            yrange=years,
            xticks=np.linspace(1999, 2023, 9),
            yticks=np.linspace(1999, 2023, 9),
            cbar_max=0.0002,
            cbar_min=-0.0001,
            cbar_label=r"trend variance-covariance / (Gt C)$^2$",
            cmap="cividis",
            savefig="poc_trend_cov.png" if False else None,
            title="Particulate organic carbon (1998 - 2024)",
        )

    def test_trend_pp(self):
        x, y, u, n = read_time_series_data(
            "test.resources.oceancolour",
            "global_v6_4km_zeu_filled_pp_monthly.csv",
        )
        x = x.reshape((n, 1))
        y = y.reshape((n, 1))
        u = u.reshape((n, 1))

        method = "bootstrap-ols"

        if method == "bootstrap-ols":
            fitting = Bootstrap(
                HeteroscedasticRegression(EIV()), how_many=10000
            )
            result = fitting.fit(
                Linear(), x, y, ux=np.zeros_like(u), uy=np.ones_like(u)
            )
        elif method == "mc-wls":
            fitting = MonteCarlo(
                HeteroscedasticRegression(EIV()), how_many=10000
            )
            result = fitting.fit(Linear(), x, y, ux=np.zeros_like(u), uy=u)
        else:
            fitting = EIV()
            result = fitting.fit(Linear(), x, y, ux=np.zeros_like(u), uy=u)

        self.assertEqual(0, result.info)

        print()
        print("popt = ", result.popt)
        print("punc = ", result.punc)
        print("pcov = ", result.pcov)

        TrendPlot(result).plot(
            x,
            y,
            xlabel=r"year",
            ylabel=r"primary production / (Gt C)",
            xrange=(1997.75, 2023.25),
            yrange=(51.5, 64.5),
            xticks=np.linspace(1998, 2023, 6),
            savefig="pp_trend.png" if False else None,
            title="Primary production (1998 - 2023)",
        )
        years = (1998.0 + 1.0 / 12, 2023.0 - 1.0 / 12.0)
        MatrixPlot().plot(
            matrix(result, years, n),
            xlabel=r"year",
            ylabel=r"year",
            xrange=years,
            yrange=years,
            xticks=np.linspace(1998, 2023, 6),
            yticks=np.linspace(1998, 2023, 6),
            cbar_max=0.05,
            cbar_min=-0.02,
            cbar_label=r"trend variance-covariance / (Gt C)$^2$",
            cmap="cividis",
            savefig="pp_trend_cov.png" if False else None,
            title="Primary production (1998 - 2023)",
        )


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
            yrange=(-0.002, 0.037),
            title="Water classes (Jackson et al., 2017)",
            savefig="water_classes.png" if False else None,
        )
        self.assertIsNotNone(fig)

    def test_plot_chlorophyll_elasticity(self):
        w, R, _, M, m = read_owt_data(  # noqa : N806
            "test.resources.oceancolour", "owt.csv"
        )
        W = np.broadcast_to(w, (M, m))  # noqa : N806
        U = np.asarray(  # noqa : N806
            [[0.05, 0.05, 0.05, 0.10, 0.20]] * R[:, 1:]
        )

        f = OCI()
        x = np.stack([W[:, 1:], R[:, 1:], U], axis=1)
        p = f.prior(preset="OC4_MERIS")
        y = f.eval(p, x)
        g = f.jac_x(p, x)

        fig = WaterClassLinePlot().plot(
            w[1:],
            elasticity(x[:, 1, :], y[:, np.newaxis], g[:, 1, :]),
            xlabel=r"wavelength $\lambda$ (nm)",
            ylabel=r"elasticity "
            r"$\epsilon(C_{\mathrm{chl}}, R_{\mathrm{rs}}(\lambda))$",
            yrange=(-4.5, 4.5),
            savefig="chlorophyll_elasticity.png" if True else None,
        )
        self.assertIsNotNone(fig)

    def test_plot_phytoplankton_elasticity(self):
        w, R, _, M, m = read_owt_data(  # noqa : N806
            "test.resources.oceancolour", "owt.csv"
        )
        W = np.broadcast_to(w, (M, m))  # noqa : N806
        U = np.asarray(  # noqa : N806
            [[0.05, 0.05, 0.05, 0.10, 0.20]] * R[:, 1:]
        )

        f = MaranonOCI()
        x = np.stack([W[:, 1:], R[:, 1:], U], axis=1)
        p = f.prior(preset="OC4_MERIS")
        y = f.eval(p, x)
        g = f.jac_x(p, x)

        fig = WaterClassLinePlot().plot(
            w[1:],
            elasticity(x[:, 1, :], y[:, np.newaxis], g[:, 1, :]),
            xlabel=r"wavelength $\lambda$ (nm)",
            ylabel=r"elasticity "
            r"$\epsilon(C_{\mathrm{phy}}, R_{\mathrm{rs}}(\lambda))$",
            yrange=(-4.5, 4.5),
            savefig="phytoplankton_elasticity.png" if True else None,
        )
        self.assertIsNotNone(fig)


class WaterClassScatterPlotTest(unittest.TestCase):
    """Tests plotting water classes."""

    def test_plot_chlorophyll_uncertainty(self):
        w, R, _, M, m = read_owt_data(  # noqa : N806
            "test.resources.oceancolour", "owt.csv"
        )
        W = np.broadcast_to(w, (M, m))  # noqa : N806
        U = np.asarray(  # noqa : N806
            [[0.05, 0.05, 0.05, 0.10, 0.20]] * R[:, 1:]
        )

        f_oc = OCI(True)
        x = np.stack([W[:, 1:], R[:, 1:], U], axis=1)
        u = np.stack(
            [np.broadcast_to(0.0, (M, 5)), U, np.broadcast_to(0.0, (M, 5))],
            axis=1,
        )
        p = f_oc.prior(preset="OC4_MERIS")
        x_rs = x[:, 1, 0]
        y_oc = f_oc.eval(p, x)

        U = np.square(u)  # noqa : N806
        U_oc = f_oc.lpu_x(p, x, U)  # noqa : N806
        u_rs = x_rs * 0.1
        u_oc = np.sqrt(U_oc)

        fig = WaterClassScatterPlot().plot(
            x_rs,
            y_oc,
            u_rs,
            u_oc,
            xlabel=r"$R_{\mathrm{rs}}(443~\text{nm})$ (sr$^{-1}$)",
            ylabel=r"$\log_{10} C_{\mathrm{chl}}$ (mg m$^{-3}$)",
            savefig="chlorophyll_uncertainty.png" if True else None,
        )
        self.assertIsNotNone(fig)

    def test_plot_phytoplankton_uncertainty(self):
        w, R, _, M, m = read_owt_data(  # noqa : N806
            "test.resources.oceancolour", "owt.csv"
        )
        W = np.broadcast_to(w, (M, m))  # noqa : N806
        U = np.asarray(  # noqa : N806
            [[0.05, 0.05, 0.05, 0.10, 0.20]] * R[:, 1:]
        )

        f_oc = OCI(True)
        f_pc = MaranonOCI(True)
        x = np.stack([W[:, 1:], R[:, 1:], U], axis=1)
        u = np.stack(
            [np.broadcast_to(0.0, (M, 5)), U, np.broadcast_to(0.0, (M, 5))],
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
            savefig="phytoplankton_uncertainty.png" if True else None,
        )
        self.assertIsNotNone(fig)


if __name__ == "__main__":
    unittest.main()
