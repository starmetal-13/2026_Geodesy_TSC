"""Elastic/compliant wedge Green's-function backend for GeoSlip2D.

This module provides the GeoSlip2D-facing wrapper around the vendored
``geoslip2d.elastic_wedge`` solver.  The wrapper normalizes inputs/outputs so
wedge Green's functions look like every other GeoSlip2D backend.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d

from .geometry import InterfaceGeometry, interface_from_arrays
from .greens import Greens2D


@dataclass(slots=True)
class WedgeConfig:
    """Configuration for the elastic/compliant wedge backend.

    The field names intentionally mirror the original wedge notebook so that
    existing parameter cells can be translated with minimal editing.
    """

    # Interface / wedge geometry
    faultdip_trench: float | None = None
    faultdip_bottom: float | None = None
    x_trench: float | None = None
    x_bottom: float | None = None
    z_bottom: float | None = None
    wedge_bot: float = 12.0
    wedge_top_x: float = 90.0
    L_slab: float = 50.0
    pL: float | None = None

    # Numerical domain / solver settings
    W: float = 200.0
    shift: float = 1e5
    self_offset: float = 1e-3
    okada_length: float = 1e6
    use_sparse: bool = False
    compute_body: bool = False
    save_output: bool = False
    save_regression: bool = False

    # Elastic parameters
    mu1: float = 1.0
    mu2: float = 1.0
    mu3: float = 1.0
    nu: float = 0.25

    # GeoSlip2D wrapper controls
    sync_geometry_from_interface: bool = True
    output_sign: float = 1.0
    progress: bool = False
    verbose: bool = False

    # Geometry plotting controls.  ``plot_geometry`` is intentionally a
    # lightweight diagnostic option; the public helper ``plot_wedge_geometry``
    # can also be called directly from notebooks for more control.
    plot_geometry: bool = False
    plot_geometry_xlim: tuple[float, float] | None = None
    plot_geometry_show_centers: bool = True
    plot_geometry_show_segments: bool = True

    sign_convention: str = "wedge_native_forward_slip_positive"



def plot_wedge_geometry(
    config: WedgeConfig | None = None,
    interface: InterfaceGeometry | None = None,
    *,
    params: Any | None = None,
    ax: Any | None = None,
    xlim: tuple[float, float] | None = None,
    show_centers: bool = True,
    show_segments: bool = True,
):
    """Plot the full elastic/compliant wedge boundary geometry.

    This is the GeoSlip2D-facing wrapper around the vendored wedge plotting
    helper.  It reproduces the diagnostic geometry plot used in the original
    ``Run_Wedge.ipynb`` workflow: free surface, wedge backstop, megathrust,
    slab extension, and patch centers.

    Parameters
    ----------
    config
        Wedge backend configuration.  If omitted, ``WedgeConfig()`` is used.
    interface
        Optional GeoSlip2D interface.  When provided and
        ``config.sync_geometry_from_interface=True``, the wedge geometry is
        synchronized to this interface before plotting.
    params
        Optional already-created vendored parameter object.  This is mainly
        used internally by ``build_wedge_greens`` so the automatic plot shows
        the exact geometry used by the build.
    ax
        Optional matplotlib axis to draw into.
    xlim
        Optional x-axis limits.  If omitted, the vendored plotting helper uses
        the model-width settings.
    show_centers, show_segments
        Control whether patch centers and patch-edge segments are shown.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """
    if config is None:
        config = WedgeConfig()

    if params is None:
        params = _make_plot_params(config=config, interface=interface)

    from .elastic_wedge import build_greens_elastic_wedge
    from .elastic_wedge.plotting import plot_elastic_wedge_geometry

    geom = build_greens_elastic_wedge(params, geometry_only=True)
    fig, ax = plot_elastic_wedge_geometry(
        geom,
        shift=getattr(params, "shift", None),
        W=getattr(params, "W", None),
        x_trench=getattr(params, "x_trench", None),
        xlim=xlim,
        show_centers=show_centers,
        show_segments=show_segments,
        ax=ax,
    )
    ax.set_title("elastic-wedge boundary geometry")
    return fig, ax

def build_wedge_greens(
    interface: InterfaceGeometry,
    xobs: ArrayLike,
    config: WedgeConfig | None = None,
) -> Greens2D:
    """Build wedge Green's functions and return a canonical :class:`Greens2D`.

    This wrapper calls the vendored :mod:`geoslip2d.elastic_wedge` solver.
    The resulting native fields ``Gx``, ``Gz``, and ``Gtau`` are preserved in
    ``greens.metadata`` while ``Gx``/``Gz`` are exposed as canonical
    ``Ghor``/``Gvert``.
    """
    if config is None:
        config = WedgeConfig()

    params = _make_vendored_params(interface, np.asarray(xobs, dtype=float).reshape(-1), config)

    if config.plot_geometry:
        plot_wedge_geometry(
            config=config,
            interface=interface,
            params=params,
            xlim=config.plot_geometry_xlim,
            show_centers=config.plot_geometry_show_centers,
            show_segments=config.plot_geometry_show_segments,
        )

    from .elastic_wedge import build_greens_elastic_wedge

    out = build_greens_elastic_wedge(params)
    native_greens = getattr(out, "greens", out)
    return wedge_greens_from_native(native_greens, config=config, target_xobs=xobs)


def wedge_greens_from_native(
    native_greens: Any,
    *,
    config: WedgeConfig | None = None,
    target_xobs: ArrayLike | None = None,
) -> Greens2D:
    """Convert a native wedge output object/dictionary to :class:`Greens2D`.

    Parameters
    ----------
    native_greens
        Object or dict with at least ``Gx``, ``Gz``, ``xpos``/``xobs``, and
        interface endpoint arrays.  This matches the object used in the
        original ``Run_Wedge`` notebook.
    config
        Optional wrapper configuration.  ``output_sign`` is applied here.
    target_xobs
        Optional observation positions requested by the GeoSlip2D caller.
        The native wedge solver evaluates surface displacements on its own
        boundary mesh; when this is provided, ``Gx`` and ``Gz`` are
        interpolated from the native surface mesh onto ``target_xobs`` so the
        returned ``Greens2D`` object uses the same observation coordinates as
        the other backends.
    """
    if config is None:
        config = WedgeConfig()

    get = _getter(native_greens)
    xobs_val = get("xobs", default=None)
    if xobs_val is None:
        xobs_val = get("xpos")
    xobs = _as_1d(xobs_val, "xobs/xpos")

    Gx_val = get("Gx", default=None)
    if Gx_val is None:
        Gx_val = get("Ghor")
    Gz_val = get("Gz", default=None)
    if Gz_val is None:
        Gz_val = get("Gvert")
    Gx = np.asarray(Gx_val, dtype=float)
    Gz = np.asarray(Gz_val, dtype=float)

    topx = _as_1d(get("topx_interface"), "topx_interface")
    topz = _positive_depth(_as_1d(get("topz_interface"), "topz_interface"))
    botx = _as_1d(get("botx_interface"), "botx_interface")
    botz = _positive_depth(_as_1d(get("botz_interface"), "botz_interface"))

    centers = get("centers_interface", default=None)
    if centers is not None:
        centers = np.asarray(centers, dtype=float)
        # Historical wedge outputs may store centers as either (n_patch, 2)
        # or MATLAB-style (2, n_patch). Normalize to GeoSlip2D's (n, 2).
        if centers.ndim == 2 and centers.shape[0] == 2:
            centers = centers.T
        if centers.ndim == 2 and centers.shape[1] >= 2:
            centers = centers[:, :2]
            centers[:, 1] = _positive_depth(centers[:, 1])
        else:
            centers = None

    interface = interface_from_arrays(
        topx=topx,
        topz=topz,
        botx=botx,
        botz=botz,
        centers=centers,
        metadata={"builder": "elastic_wedge_py_native_output"},
    )

    metadata: dict[str, Any] = {
        "backend": "wedge",
        "native_backend": "geoslip2d.elastic_wedge",
        "config": asdict(config),
    }
    for name in ("Gtau", "patch_slips_all", "rhs"):
        val = get(name, default=None)
        if val is not None:
            metadata[name] = np.asarray(val)

    if target_xobs is not None:
        target = np.asarray(target_xobs, dtype=float).reshape(-1)
        if target.size == 0:
            raise ValueError("target_xobs must contain at least one observation position.")
        if not (target.shape == xobs.shape and np.allclose(target, xobs, rtol=0.0, atol=0.0)):
            metadata["native_xobs"] = xobs.copy()
            metadata["interpolated_to_requested_xobs"] = True
            Gx = _interp_surface_matrix(xobs, Gx, target, "Gx")
            Gz = _interp_surface_matrix(xobs, Gz, target, "Gz")
            xobs = target
        else:
            metadata["interpolated_to_requested_xobs"] = False

    return Greens2D(
        Ghor=config.output_sign * Gx,
        Gvert=config.output_sign * Gz,
        xobs=xobs,
        interface=interface,
        source_type="wedge",
        units="displacement_per_unit_slip",
        sign_convention=config.sign_convention,
        metadata=metadata,
    )



def _interp_surface_matrix(x_native: np.ndarray, G_native: np.ndarray, x_target: np.ndarray, name: str) -> np.ndarray:
    """Interpolate a native wedge surface Green's matrix onto requested x positions."""
    x_native = np.asarray(x_native, dtype=float).reshape(-1)
    G_native = np.asarray(G_native, dtype=float)
    x_target = np.asarray(x_target, dtype=float).reshape(-1)

    if G_native.ndim != 2:
        raise ValueError(f"{name} must be a 2-D Green's matrix.")
    if G_native.shape[0] != x_native.size:
        raise ValueError(f"{name} has {G_native.shape[0]} rows but native xobs has length {x_native.size}.")

    order = np.argsort(x_native)
    xs = x_native[order]
    Gs = G_native[order, :]

    # Remove duplicate native surface positions if any boundary joins produce
    # repeated x values.  For duplicated x positions, average the rows.
    xu, inverse = np.unique(xs, return_inverse=True)
    if xu.size != xs.size:
        Gavg = np.zeros((xu.size, Gs.shape[1]), dtype=float)
        counts = np.zeros(xu.size, dtype=float)
        for i, j in enumerate(inverse):
            Gavg[j, :] += Gs[i, :]
            counts[j] += 1.0
        Gs = Gavg / counts[:, None]
        xs = xu

    if x_target.min() < xs.min() or x_target.max() > xs.max():
        raise ValueError(
            f"Requested xobs for wedge backend extend outside native surface mesh: "
            f"requested [{x_target.min():.6g}, {x_target.max():.6g}], "
            f"native [{xs.min():.6g}, {xs.max():.6g}]."
        )

    f = interp1d(xs, Gs, axis=0, bounds_error=True, assume_sorted=True)
    out = np.asarray(f(x_target), dtype=float)
    if np.any(~np.isfinite(out)):
        raise ValueError(f"Interpolated {name} contains non-finite values.")
    return out



def _make_plot_params(config: WedgeConfig, interface: InterfaceGeometry | None = None) -> Any:
    """Create vendored wedge params for geometry-only plotting."""
    if interface is not None:
        return _make_vendored_params(interface, np.array([], dtype=float), config)

    try:
        from .elastic_wedge import default_elastic_wedge_params
        params = default_elastic_wedge_params()
    except Exception:
        params = SimpleNamespace()

    wrapper_only = {
        "sync_geometry_from_interface",
        "output_sign",
        "progress",
        "sign_convention",
        "plot_geometry",
        "plot_geometry_xlim",
        "plot_geometry_show_centers",
        "plot_geometry_show_segments",
    }
    for name, value in asdict(config).items():
        if name in wrapper_only:
            continue
        if value is not None:
            setattr(params, name, value)

    setattr(params, "verbose", bool(config.verbose or config.progress))
    setattr(params, "save_output", bool(config.save_output))
    setattr(params, "save_regression", bool(config.save_regression))
    setattr(params, "compute_body", bool(config.compute_body))
    return params

def _make_vendored_params(interface: InterfaceGeometry, xobs: np.ndarray, config: WedgeConfig) -> Any:
    """Create and populate the parameter object expected by the vendored wedge solver."""
    try:
        from .elastic_wedge import default_elastic_wedge_params
        params = default_elastic_wedge_params()
    except Exception:
        params = SimpleNamespace()

    if config.sync_geometry_from_interface:
        inferred = {
            "faultdip_trench": float(interface.dip[0]),
            "faultdip_bottom": float(interface.dip[-1]),
            "x_trench": float(interface.topx[0]),
            "x_bottom": float(interface.botx[-1]),
            "z_bottom": float(interface.botz[-1]),
            "pL": float(np.nanmedian(interface.patch_length)),
        }
    else:
        inferred = {}

    for name, value in asdict(config).items():
        if name in {
            "sync_geometry_from_interface",
            "output_sign",
            "progress",
            "sign_convention",
            "plot_geometry",
            "plot_geometry_xlim",
            "plot_geometry_show_centers",
            "plot_geometry_show_segments",
        }:
            continue
        if value is None:
            value = inferred.get(name)
        if value is not None:
            setattr(params, name, value)

    for name, value in inferred.items():
        if getattr(params, name, None) is None:
            setattr(params, name, value)

    # Different historical versions of the wedge code have used slightly
    # different observation-position names. Set all common aliases.
    setattr(params, "xobs", xobs)
    setattr(params, "xpos", xobs)
    setattr(params, "x_obs", xobs)

    setattr(params, "verbose", bool(config.verbose or config.progress))
    setattr(params, "save_output", bool(config.save_output))
    setattr(params, "save_regression", bool(config.save_regression))
    setattr(params, "compute_body", bool(config.compute_body))
    return params


def _getter(obj: Any):
    def get(name: str, *, default: Any = ...):
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            return getattr(obj, name)
        if default is ...:
            raise KeyError(f"Native wedge output is missing required field '{name}'.")
        return default

    return get


def _as_1d(a: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float).reshape(-1)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _positive_depth(z: ArrayLike) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    # Some legacy wedge/geometry paths used negative z for depth. If all finite
    # depths are non-positive, flip to GeoSlip2D's positive-down convention.
    if z.size and np.nanmax(z) <= 0.0:
        return -z
    return z


__all__ = ["WedgeConfig", "build_wedge_greens", "wedge_greens_from_native", "plot_wedge_geometry"]
