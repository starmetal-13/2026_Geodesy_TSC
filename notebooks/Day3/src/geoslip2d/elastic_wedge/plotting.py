import matplotlib.pyplot as plt


def plot_elastic_wedge_geometry(
    geom,
    shift=None,
    xlim=(-100.0, 400.0),
    W=None,
    x_trench=None,
    show_centers=True,
    show_segments=True,
    ax=None,
):
    """
    Plot elastic wedge boundary geometry and subduction interface.

    Parameters
    ----------
    geom : Geometry
        Geometry object from elastic_wedge_py.
        Expected to contain geom.B, where each boundary has:
            top, bot, center
        with arrays shaped (2, N).

    shift : float or None
        Vertical shift used in the elastic wedge construction.
        If provided, z is plotted as z + shift to recover physical depths.

    xlim : tuple[float, float] or None
        Explicit x-axis limits as (xmin, xmax).
        Default is (-100, 400). Set to None to disable explicit limits.

    W : float or None
        Model half-width parameter. If provided with x_trench, sets x limits
        similar to the original MATLAB code when xlim is None.

    x_trench : float or None
        Trench x-position. Used for x limits.

    show_centers : bool
        Plot patch centers.

    show_segments : bool
        Plot patch edges/segments.

    ax : matplotlib axis or None
        Existing axis. If None, create a new figure and axis.

    Returns
    -------
    fig, ax
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    if shift is None:
        shift = 0.0

    # Boundary colors roughly following the original MATLAB plot.
    colors = {
        0: "k",
        1: "k",
        2: "k",
        3: "k",
        4: "r",
        5: "g",
    }

    labels = {
        0: "surface left",
        1: "surface wedge",
        2: "surface upper plate",
        3: "wedge backstop",
        4: "subduction interface",
        5: "slab extension",
    }

    for k, B in enumerate(geom.B):
        color = colors.get(k, "k")
        label = labels.get(k, f"boundary {k}")

        top = B.top.copy()
        bot = B.bot.copy()
        center = B.center.copy()

        top_z = top[1, :] + shift
        bot_z = bot[1, :] + shift
        cen_z = center[1, :] + shift

        if show_segments:
            for i in range(top.shape[1]):
                ax.plot(
                    [top[0, i], bot[0, i]],
                    [top_z[i], bot_z[i]],
                    color=color,
                    linewidth=2,
                    alpha=0.9,
                )

        if show_centers:
            ax.plot(
                center[0, :],
                cen_z,
                ".",
                color=color,
                markersize=6,
                label=label,
            )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlabel("x distance")
    ax.set_ylabel("z")

    if xlim is not None:
        ax.set_xlim([xlim[0], xlim[1]])
    elif W is not None and x_trench is not None:
        ax.set_xlim([x_trench - 2 * W, x_trench + 4 * W])

    ax.legend(loc="best")
    fig.tight_layout()

    return fig, ax
