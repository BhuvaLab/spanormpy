"""Plotting (R ``plotSpatial`` and ``plotCovariate``)."""

from ._covariate import covariate_values, plot_covariate
from ._spatial import plot_spatial

__all__ = ["covariate_values", "plot_covariate", "plot_spatial"]
