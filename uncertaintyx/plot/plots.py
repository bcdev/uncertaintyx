#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
from typing import Any
from typing import List
from typing import Literal

import numpy as np
import seaborn as sns
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from ..b.jax import BernsteinGrid
from ..plotting import Plotting
from ..tyx import Fitted


class BernsteinBasisPlot(Plotting):
    """
    A plot to display bivariate Bernstein basis functions of a given degree.
    """

    def __init__(
        self,
        context: Literal["paper", "notebook"] = "paper",
    ):
        """
        Creates a new plot.

        :param context: The plot context.
        """
        self._context = context

    # noinspection PyUnresolvedReferences,PyTypeChecker
    def plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        degree: int = 2,
        figsize: tuple[int, int] = (14, 14),
        caption: str | None = None,
        cmap: Literal["cividis", "coolwarm", "viridis"] = "viridis",
        savefig: str | None = None,
        **kwargs,
    ) -> Figure:
        """
        Plots the bivariate Bernstein basis polynomials of the given degree.
        """

        def basis(i: int, j: int) -> np.ndarray:
            """
            Returns the Bernstein coefficients for only a specific basis
            polynomial.
            """
            c = np.zeros((n, n))
            c[i, j] = 1.0
            return c

        sns.set_theme(context=self._context)
        sns.set_style("ticks")
        sns.set_palette(sns.color_palette("colorblind"))

        n = degree + 1
        X, Y = np.meshgrid(x, y)  # noqa: N806
        g = BernsteinGrid((x, y))

        fig = plt.figure(figsize=figsize, facecolor="white")

        margin_l = 0.05
        margin_b = 0.05
        cell_w = 0.16
        cell_h = 0.16
        gap_x = 0.03
        gap_y = 0.03

        for i in range(n):
            for j in range(n):
                pos_x = margin_l + j * (cell_w + gap_x)
                pos_y = margin_b + (n - 1 - i) * (cell_h + gap_y)
                ax = fig.add_axes(
                    (pos_x, pos_y, cell_w, cell_h), projection="3d"
                )
                ax.plot_surface(
                    X,
                    Y,
                    g.eval(basis(i, j)),
                    cmap=cmap,
                    edgecolor="#444444",
                    linewidth=0.3,
                    rstride=5,
                    cstride=5,
                    antialiased=True,
                    alpha=1.0,
                )
                ax.set_box_aspect((1.0, 1.0, 0.5))
                ax.grid(False)
                ax.set_xlabel(r"$x$")
                ax.set_ylabel(r"$y$")
                ax.set_xticks([0.0, 0.5, 1.0])
                ax.set_yticks([0.0, 0.5, 1.0])
                ax.set_zticks([0.0, 0.5, 1.0])
                ax.tick_params(axis="both", which="major")

                ax.view_init(elev=20.0, azim=-60.0)
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

                ax.text2D(
                    0.50,
                    0.95,
                    rf"$B_{{{i},{j}}}^{{{degree}}}(x,y)$",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                )

        mid_x = margin_l + n // 2 * (cell_w + gap_x) + cell_w * 0.5
        top_y = margin_b + n * (cell_h + gap_y)
        fig.suptitle(
            rf"Bernstein basis polynomials $B^{{{degree}}}_{{i,j}}(x, y)$",
            x=mid_x,
            y=top_y,
            ha="center",
            va="top",
        )
        if caption is not None:
            plt.figtext(
                mid_x,
                0.02,
                caption,
                wrap=False,
                ha="center",
                va="bottom",
            )

        if savefig:
            fig.savefig(savefig, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

        return fig


class LinePlot(Plotting):
    """A line plot."""

    def __init__(
        self,
        context: Literal["paper", "notebook", "talk", "poster"] = "paper",
    ):
        """
        Creates a new line plot.

        :param context: The plot context.
        """
        self._context = context

    def plot(
        self,
        x: list[np.ndarray],
        y: list[np.ndarray],
        u: list[np.ndarray] | None = None,
        *,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xrange: tuple[Any, Any] | None = None,
        yrange: tuple[Any, Any] | None = None,
        xticks: tuple[Any, ...] | None = None,
        yticks: tuple[Any, ...] | None = None,
        labels: list[str] | None = None,
        savefig: str | None = None,
    ):
        """
        Plots the data supplied as arguments.
        """
        sns.set_theme(context=self._context)
        sns.set_style("ticks")
        sns.set_palette(sns.color_palette("colorblind"))

        styles = list(
            cycler("linestyle", self._styles) * cycler("color", self._colors)
        )

        fig, ax = plt.subplots()
        if u is not None:
            for k, (x_, y_, u_) in enumerate(zip(x, y, u)):
                style = styles[k]
                ax.plot(
                    x_,
                    y_,
                    **style,
                    label=f"{k + 1}" if labels is None else labels[k],
                )
                if u_ is not None:
                    ax.fill_between(x_, y_, y_ + u_, **style, alpha=0.3)
                    ax.fill_between(x_, y_, y_ - u_, **style, alpha=0.3)
        else:
            for k, (x_, y_) in enumerate(zip(x, y)):
                style = styles[k]
                ax.plot(
                    x_,
                    y_,
                    **style,
                    label=f"{k + 1}" if labels is None else labels[k],
                )
        ax.legend(ncol=2)

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
        if xticks:
            ax.set_xticks(xticks)
        if yticks:
            ax.set_yticks(yticks)
        if savefig:
            fig.savefig(savefig, dpi=300)
        plt.close(fig)

        return fig

    @property
    def _colors(self) -> List[Any]:
        """Returns a list of colours available."""
        return sns.color_palette("colorblind")

    @property
    def _styles(self) -> list[str]:
        """Returns a list of line styles available."""
        return ["-", "--", ":", "-."]


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
        **kwargs,
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
        result: Fitted,
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
        """Returns a list of colours available."""
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
        **kwargs,
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
        ax.plot(
            x, y_opt, linestyle="-", color=self._colors[1], label="regression"
        )
        ax.plot(
            x,
            y_opt - y_unp,
            "--",
            color=self._colors[2],
            label="standard uncertainty of regression",
        )
        ax.plot(x, y_opt + y_unp, linestyle="--", color=self._colors[2])
        ax.plot(
            x,
            y_opt - y_unc,
            "-.",
            color=self._colors[3],
            label="standard uncertainty of residuals",
        )
        ax.plot(x, y_opt + y_unc, linestyle="-.", color=self._colors[3])
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
        Creates a new scatter plot.

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
            ax.set_xlim(xrange)
        if yrange:
            ax.set_ylim(yrange)
        if savefig:
            fig.savefig(savefig, dpi=300)
        plt.close(fig)

        return fig


class WaterClassLinePlot(Plotting):
    """A water class line plot."""

    def __init__(
        self,
        context: Literal["paper", "notebook", "talk", "poster"] = "paper",
    ):
        """
        Creates a new line plot.

        :param context: The plot context.
        """
        self._context = context

    def plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray | None = None,
        *,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xrange: tuple[Any, Any] | None = None,
        yrange: tuple[Any, Any] | None = None,
        xticks: tuple[Any, ...] | None = None,
        yticks: tuple[Any, ...] | None = None,
        savefig: str | None = None,
    ):
        """
        Plots the data supplied as arguments.
        """
        sns.set_theme(context=self._context)
        sns.set_style("ticks")
        sns.set_palette(sns.color_palette("colorblind"))

        styles = list(
            cycler("linestyle", self._styles) * cycler("color", self._colors)
        )

        fig, ax = plt.subplots()
        if u is not None:
            for k, (y_, u_) in enumerate(zip(y, u)):
                style = styles[k]
                ax.plot(x, y_, **style, marker="|", label=f"class {k + 1}")
                ax.fill_between(x, y_, y_ + u_, **style, alpha=0.3)
                ax.fill_between(x, y_, y_ - u_, **style, alpha=0.3)
        else:
            for k, y_ in enumerate(y):
                style = styles[k]
                ax.plot(x, y_, **style, marker="|", label=f"class {k + 1}")
        ax.legend(ncol=3)

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
        if xticks:
            ax.set_xticks(xticks)
        else:
            ax.set_xticks(tuple(x))
        if yticks:
            ax.set_yticks(yticks)
        if savefig:
            fig.savefig(savefig, dpi=300)
        plt.close(fig)

        return fig

    @property
    def _colors(self) -> List[Any]:
        """Returns a list of colours available."""
        return sns.color_palette("colorblind")

    @property
    def _styles(self) -> list[str]:
        """Returns a list of line styles available."""
        return ["-", "--", ":", "-."]


class WaterClassScatterPlot(Plotting):
    """A water class scatter plot."""

    def __init__(
        self,
        context: Literal["paper", "notebook", "talk", "poster"] = "paper",
    ):
        """
        Creates a new line plot.

        :param context: The plot context.
        """
        self._context = context

    def plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        ux: np.ndarray,
        uy: np.ndarray,
        *,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xrange: tuple[Any, Any] | None = None,
        yrange: tuple[Any, Any] | None = None,
        xticks: tuple[Any, ...] | None = None,
        yticks: tuple[Any, ...] | None = None,
        savefig: str | None = None,
    ):
        """
        Plots the data supplied as arguments.
        """
        sns.set_theme(context=self._context)
        sns.set_style("ticks")
        sns.set_palette(sns.color_palette("colorblind"))

        styles = list(
            cycler("linestyle", self._styles) * cycler("color", self._colors)
        )

        fig, ax = plt.subplots()
        for k, (x_, y_, ux_, uy_) in enumerate(zip(x, y, ux, uy)):
            style = styles[k]
            ax.scatter(x_, y_, **style, label=f"class {k + 1}", s=6)
            ax.hlines(y_, xmin=x_ - ux_, xmax=x_ + ux_, **style)
            ax.vlines(x_, ymin=y_ - uy_, ymax=y_ + uy_, **style)
        ax.legend(ncol=3)

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
        if xticks:
            ax.set_xticks(xticks)
        if yticks:
            ax.set_yticks(yticks)
        if savefig:
            fig.savefig(savefig, dpi=300)
        plt.close(fig)

        return fig

    @property
    def _colors(self) -> List[Any]:
        """Returns a list of colours available."""
        return sns.color_palette("colorblind")

    @property
    def _styles(self) -> list[str]:
        """Returns a list of line styles available."""
        return ["-", "--", ":", "-."]
