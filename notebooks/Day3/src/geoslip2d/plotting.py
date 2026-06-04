"""Plotting helpers for GeoSlip2D core objects."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .geometry import InterfaceGeometry
from .greens import Greens2D


def plot_interface(interface: InterfaceGeometry, ax=None):
    """Plot the top and bottom edge points of the interface patches.

    Parameters
    ----------
    interface
        GeoSlip2D interface geometry.
    ax
        Optional matplotlib axes. If omitted, a new figure and axes are created.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    else:
        fig = ax.figure
    ax.plot(interface.topx, interface.topz, ".", label="patch top")
    ax.plot(interface.botx, interface.botz, ".", label="patch bottom")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.grid(True)
    ax.set_xlabel("distance from trench (km)")
    ax.set_ylabel("depth (km)")
    ax.set_title("Interface geometry")
    ax.legend()
    return fig


def plot_greens_summary(greens: Greens2D, ax=None):
    """Plot summed horizontal and vertical Green's functions versus xobs."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    else:
        fig = ax.figure
    ax.plot(greens.xobs, np.sum(greens.Ghor, axis=1), label="sum Ghor")
    if greens.Gvert is not None:
        ax.plot(greens.xobs, np.sum(greens.Gvert, axis=1), label="sum Gvert")
    ax.grid(True)
    ax.set_xlabel("xobs (km)")
    ax.set_ylabel(greens.units)
    ax.set_title(f"Green's summary: {greens.source_type}")
    ax.legend()
    return fig


def plot_greens_columns(
    greens: Greens2D,
    component: str = "horizontal",
    columns: Sequence[int] = (0, -1),
    ax=None,
):
    """Plot selected columns of a Green's matrix.

    Parameters
    ----------
    greens
        GeoSlip2D Green's-function object.
    component
        ``"horizontal"``/``"hor"`` or ``"vertical"``/``"vert"``.
    columns
        Column indices to plot. Negative indices follow normal Python indexing.
    ax
        Optional matplotlib axes.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    comp = component.lower()
    if comp in {"horizontal", "hor", "x", "ghor"}:
        G = greens.Ghor
        ylabel = "Ghor"
    elif comp in {"vertical", "vert", "z", "gvert"}:
        if greens.Gvert is None:
            raise ValueError("This Greens2D object does not contain vertical Green's functions.")
        G = greens.Gvert
        ylabel = "Gvert"
    else:
        raise ValueError("component must be 'horizontal' or 'vertical'.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    else:
        fig = ax.figure

    ncols = G.shape[1]
    for col in columns:
        idx = int(col)
        if idx < 0:
            idx = ncols + idx
        if idx < 0 or idx >= ncols:
            raise IndexError(f"Green's column index {col} is out of bounds for {ncols} patches.")
        ax.plot(greens.xobs, G[:, idx], label=f"patch {idx}")

    ax.grid(True)
    ax.set_xlabel("xobs (km)")
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig
