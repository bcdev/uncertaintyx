#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
from typing import Any
from typing import List
from typing import Literal

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from ..plotting import Plotting
from ..tyx import Result


class MatrixPlot(Plotting):
    """A matrix plot."""

    def __init__(
        self,
        context: Literal["paper", "notebook", "talk", "poster"] = "paper",
    ):
        """
        Creates a new regression plot.

        :param context: The plot context.
        """
        self._context = context

    def plot(
        self,
        matrix: np.ndarray,
        *,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xrange: tuple[Any, Any] | None = None,
        yrange: tuple[Any, Any] | None = None,
        cbar_label: str | None = None,
        cbar_min: Any | None = None,
        cbar_max: Any | None = None,
        cmap: str | None = "viridis",
        savefig: str | None = None,
    ) -> Figure:
        """Plots a matrix."""
        sns.set_theme(context=self._context)
        sns.set_style("ticks")
        sns.set_palette(sns.color_palette("colorblind"))

        fig, ax = plt.subplots()
        dx = (
            None
            if xrange is None
            else (xrange[1] - xrange[0]) / (matrix.shape[1] - 1)
        )
        dy = (
            None
            if yrange is None
            else (yrange[1] - yrange[0]) / (matrix.shape[0] - 1)
        )
        extent = (
            None if xrange is None else xrange[0] - 0.5 * dx,
            None if xrange is None else xrange[1] + 0.5 * dx,
            None if yrange is None else yrange[0] - 0.5 * dy,
            None if yrange is None else yrange[1] + 0.5 * dy,
        )
        im = ax.imshow(
            matrix,
            cmap=cmap,
            extent=extent if None not in extent else None,
            origin="lower",
            vmin=cbar_min,
            vmax=cbar_max,
        )
        plt.colorbar(
            im, ax=ax, label=cbar_label, extend="both", extendfrac=0.05
        )
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if extent:
            ax.set_xlim(xmin=extent[0], xmax=extent[1])
            ax.set_ylim(ymin=extent[2], ymax=extent[3])
        if savefig:
            fig.savefig(savefig, dpi=300)
        plt.close(fig)

        return fig


class RegressionPlot(Plotting):
    """A regression plot."""

    def __init__(
        self,
        result: Result,
        context: Literal["paper", "notebook", "talk", "poster"] = "paper",
    ):
        """
        Creates a new regression plot.

        :param result: The regression result.
        :param context: The plot context.
        """
        self._result = result
        self._context = context

    @property
    def _colors(self) -> List[Any]:
        """Returns a list of colors available."""
        return sns.color_palette("colorblind")

    def plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xrange: tuple[Any, Any] | None = None,
        yrange: tuple[Any, Any] | None = None,
        savefig: str | None = None,
    ):
        """
        Plots the regression result along with data supplied as arguments.
        """
        sns.set_theme(context=self._context)
        sns.set_style("ticks")
        sns.set_palette(sns.color_palette("colorblind"))

        fig, ax = plt.subplots()
        ax.scatter(x, y, s=1, color=self._colors[0])

        if not xrange:
            xrange = np.min(x), np.max(x)
        if not yrange:
            yrange = np.min(y), np.max(y)
        x = np.linspace(xrange[0], xrange[1], 1000)
        y_opt = self._result.f(x)
        y_unp = self._result.yunc_p(x)
        y_unc = self._result.yunc_t(x)
        ax.plot(x, y_opt, "-", color=self._colors[1], label="regression")
        ax.plot(
            x,
            y_opt - y_unp,
            "--",
            color=self._colors[2],
            label="standard uncertainty of regression",
        )
        ax.plot(x, y_opt + y_unp, "--", color=self._colors[2])
        ax.plot(
            x,
            y_opt - y_unc,
            "-.",
            color=self._colors[3],
            label="standard uncertainty of residuals",
        )
        ax.plot(x, y_opt + y_unc, "-.", color=self._colors[3])
        ax.legend()

        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if xrange:
            ax.set_xlim(xrange)
        if yrange:
            ax.set_ylim(yrange)
        if savefig:
            fig.savefig(savefig, dpi=300)
        plt.close(fig)

        return fig


class ScatterPlot(Plotting):
    """A scatter plot."""

    def __init__(
        self,
        context: Literal["paper", "notebook", "talk", "poster"] = "paper",
    ):
        """
        Creates a new regression plot.

        :param context: The plot context.
        """
        self._context = context

    def plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xrange: tuple[Any, Any] | None = None,
        yrange: tuple[Any, Any] | None = None,
        savefig: str | None = None,
    ):
        """
        Plots the data supplied as arguments.
        """
        sns.set_theme(context=self._context)
        sns.set_style("ticks")
        sns.set_palette(sns.color_palette("colorblind"))

        fig, ax = plt.subplots()
        ax.scatter(x, y, s=1)

        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if xrange:
            ax.set_ylim(xrange)
        if yrange:
            ax.set_ylim(yrange)
        if savefig:
            fig.savefig(savefig, dpi=300)
        plt.close(fig)

        return fig
