#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
from abc import ABCMeta
from abc import abstractmethod
from typing import Any

from matplotlib.figure import Figure


class Plotting(metaclass=ABCMeta):
    """The plotting interface."""

    @abstractmethod
    def plot(self, *data: Any, **kwargs) -> Figure:
        """
        Plots the data.

        :param data: The data to be plotted.
        :returns: The figure plotted.
        """
