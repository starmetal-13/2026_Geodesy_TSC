"""
Full VECycle2D Green's-function builder.

This module ports the outer loop of MATLAB:

    vec_build_greens.m

using the already verified Python components:

    make_wedge_geometry
    build_raw_greens
    compile_raw_to_linear_system
    build_cycle_for_source

The main public function is:

    greens = build_greens(cfg)

It returns a VECycleGreens dataclass containing the Python equivalents of the
MATLAB Green's-function output fields:

    tau_e, tau_v, tau_rate_v
    sig_e, sig_v, sig_rate_v
    Dx_e, Dx_v, Vx_v
    Dz_e, Dz_v, Vz_v
    times, xpos, Geometry

Cell-array-like MATLAB outputs are represented as Python lists of NumPy arrays,
with one entry per interface source patch:

    greens.Dx_v[source_index]
    greens.Vx_v[source_index]
    greens.tau_v[source_index]

where source_index is zero-based.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import scipy.io as sio

from .geometry import make_wedge_geometry
from .raw_greens import build_raw_greens
from .compile_greens import compile_raw_to_linear_system, WedgeLinearSystem
from .cycle import build_cycle_for_source, SourceCycleResponse


Array = np.ndarray


@dataclass(slots=True)
class GeometryOutput:
    """MATLAB-compatible geometry metadata saved with Green's functions."""

    H_elastic_left: float
    H_elastic_right: float
    faultdip_trench: float
    x_trench: float
    x_bottom: float
    faultdip_bottom: float
    L_slab: float
    W: float
    pL: float
    wedge_bot: float
    wedge_top_x: float
    centers_interface: Array


@dataclass(slots=True)
class VECycleGreens:
    """Full VECycle2D Green's-function output."""

    tau_e: Array
    tau_v: list[Array]
    tau_rate_v: list[Array]

    sig_e: Array
    sig_v: list[Array]
    sig_rate_v: list[Array]

    Dx_e: Array
    Dx_v: list[Array]
    Vx_v: list[Array]

    Dz_e: Array
    Dz_v: list[Array]
    Vz_v: list[Array]

    xpos: Array
    times: Array
    Geometry: GeometryOutput

    # Optional internals useful for debugging and validation.
    model: WedgeLinearSystem | None = None
    raw: dict[str, Array] | None = None


def _get_geometry_config(cfg: Any) -> Any:
    """Return geometry config from either a dataclass-style object or dict."""
    if isinstance(cfg, dict):
        return cfg["geometry"]
    return getattr(cfg, "geometry")


def _get_io_config(cfg: Any) -> Any | None:
    if isinstance(cfg, dict):
        return cfg.get("io", None)
    return getattr(cfg, "io", None)


def _io_value(io: Any, name: str, default: Any = None) -> Any:
    if io is None:
        return default
    if isinstance(io, dict):
        return io.get(name, default)
    return getattr(io, name, default)


def _geometry_output(cfg: Any, geom: Any) -> GeometryOutput:
    gcfg = _get_geometry_config(cfg)
    get = (lambda name: gcfg[name]) if isinstance(gcfg, dict) else (lambda name: getattr(gcfg, name))

    return GeometryOutput(
        H_elastic_left=float(get("H_elastic_left")),
        H_elastic_right=float(get("H_elastic_right")),
        faultdip_trench=float(get("faultdip_trench")),
        x_trench=float(get("x_trench")),
        x_bottom=float(get("x_bottom")),
        faultdip_bottom=float(get("faultdip_bottom")),
        L_slab=float(get("L_slab")),
        W=float(get("W")),
        pL=float(get("pL")),
        wedge_bot=float(get("wedge_bot")),
        wedge_top_x=float(get("wedge_top_x")),
        centers_interface=np.asarray(geom.centers_interface, dtype=float),
    )


def build_greens(
    cfg: Any,
    *,
    keep_internals: bool = False,
    progress: bool = True,
) -> VECycleGreens:
    """
    Build full VECycle2D Green's functions.

    Parameters
    ----------
    cfg
        VECycle2D configuration object from ``default_config()``.
    keep_internals
        If True, attach the raw dictionary and compiled model to the returned
        object. This is useful for debugging but can increase memory use.
    progress
        If True, print progress during raw Green's construction and the
        per-source cycle loop.

    Returns
    -------
    VECycleGreens
        Full Green's-function structure.
    """
    geom = make_wedge_geometry(cfg)

    if progress:
        print("Building raw Green's-function blocks...")
    raw = build_raw_greens(geom, nu=cfg.constants.nu, progress=progress)

    if progress:
        print("Compiling linear-system matrices...")
    model = compile_raw_to_linear_system(raw, geom, cfg.constants)

    num_interface = int(model.num_interface)
    nobs = int(np.asarray(model.xpos).size)

    tau_e = np.zeros((num_interface, num_interface), dtype=float)
    sig_e = np.zeros((num_interface, num_interface), dtype=float)
    Dx_e = np.zeros((nobs, num_interface), dtype=float)
    Dz_e = np.zeros((nobs, num_interface), dtype=float)

    tau_v: list[Array] = [None] * num_interface
    tau_rate_v: list[Array] = [None] * num_interface
    sig_v: list[Array] = [None] * num_interface
    sig_rate_v: list[Array] = [None] * num_interface
    Dx_v: list[Array] = [None] * num_interface
    Vx_v: list[Array] = [None] * num_interface
    Dz_v: list[Array] = [None] * num_interface
    Vz_v: list[Array] = [None] * num_interface

    times = None

    for source_index in range(num_interface):
        if progress:
            print(f"Cycle source {source_index + 1} of {num_interface}")

        src = build_cycle_for_source(model, source_index)

        tau_e[:, source_index] = src.tau_e
        sig_e[:, source_index] = src.sig_e
        Dx_e[:, source_index] = src.Dx_e
        Dz_e[:, source_index] = src.Dz_e

        tau_v[source_index] = src.tau_v
        tau_rate_v[source_index] = src.tau_rate_v
        sig_v[source_index] = src.sig_v
        sig_rate_v[source_index] = src.sig_rate_v

        Dx_v[source_index] = src.Dx_v
        Vx_v[source_index] = src.Vx_v
        Dz_v[source_index] = src.Dz_v
        Vz_v[source_index] = src.Vz_v

        if times is None:
            times = src.times

    greens = VECycleGreens(
        tau_e=tau_e,
        tau_v=tau_v,
        tau_rate_v=tau_rate_v,
        sig_e=sig_e,
        sig_v=sig_v,
        sig_rate_v=sig_rate_v,
        Dx_e=Dx_e,
        Dx_v=Dx_v,
        Vx_v=Vx_v,
        Dz_e=Dz_e,
        Dz_v=Dz_v,
        Vz_v=Vz_v,
        xpos=np.asarray(model.xpos, dtype=float).reshape(-1),
        times=np.asarray(times, dtype=float).reshape(-1),
        Geometry=_geometry_output(cfg, geom),
        model=model if keep_internals else None,
        raw=raw if keep_internals else None,
    )

    io = _get_io_config(cfg)
    save_output = bool(_io_value(io, "save_output", False))
    output_file = _io_value(io, "output_file", None)

    if save_output and output_file:
        save_greens_mat(output_file, greens)
        if progress:
            print(f"Saved Green's functions to: {output_file}")

    return greens


def greens_summary(greens: VECycleGreens) -> dict[str, Any]:
    """Return a compact shape summary for a full Green's-function object."""
    return {
        "tau_e": greens.tau_e.shape,
        "sig_e": greens.sig_e.shape,
        "Dx_e": greens.Dx_e.shape,
        "Dz_e": greens.Dz_e.shape,
        "len(tau_v)": len(greens.tau_v),
        "tau_v[0]": None if not greens.tau_v else greens.tau_v[0].shape,
        "Dx_v[0]": None if not greens.Dx_v else greens.Dx_v[0].shape,
        "xpos": greens.xpos.shape,
        "times": greens.times.shape,
        "Geometry.centers_interface": greens.Geometry.centers_interface.shape,
    }


def _object_list_for_savemat(items: list[Array]) -> Array:
    """
    Convert a list of arrays to a MATLAB-cell-like object array.

    scipy.io.savemat writes object arrays as MATLAB cell arrays.
    """
    out = np.empty((1, len(items)), dtype=object)
    for k, item in enumerate(items):
        out[0, k] = np.asarray(item, dtype=float)
    return out


def _geometry_mat_struct(geometry: GeometryOutput) -> dict[str, Any]:
    return {
        "H_elastic_left": geometry.H_elastic_left,
        "H_elastic_right": geometry.H_elastic_right,
        "faultdip_trench": geometry.faultdip_trench,
        "x_trench": geometry.x_trench,
        "x_bottom": geometry.x_bottom,
        "faultdip_bottom": geometry.faultdip_bottom,
        "L_slab": geometry.L_slab,
        "W": geometry.W,
        "pL": geometry.pL,
        "wedge_bot": geometry.wedge_bot,
        "wedge_top_x": geometry.wedge_top_x,
        "centers_interface": geometry.centers_interface,
    }


def greens_to_mat_dict(greens: VECycleGreens, *, suffix: str = "") -> dict[str, Any]:
    """
    Convert a VECycleGreens object to a scipy.io.savemat dictionary.

    Parameters
    ----------
    suffix
        Optional suffix appended to variable names, e.g. ``"_py"``.
    """
    s = suffix
    return {
        f"tau_e{s}": greens.tau_e,
        f"tau_v{s}": _object_list_for_savemat(greens.tau_v),
        f"tau_rate_v{s}": _object_list_for_savemat(greens.tau_rate_v),
        f"sig_e{s}": greens.sig_e,
        f"sig_v{s}": _object_list_for_savemat(greens.sig_v),
        f"sig_rate_v{s}": _object_list_for_savemat(greens.sig_rate_v),
        f"Dx_e{s}": greens.Dx_e,
        f"Dx_v{s}": _object_list_for_savemat(greens.Dx_v),
        f"Vx_v{s}": _object_list_for_savemat(greens.Vx_v),
        f"Dz_e{s}": greens.Dz_e,
        f"Dz_v{s}": _object_list_for_savemat(greens.Dz_v),
        f"Vz_v{s}": _object_list_for_savemat(greens.Vz_v),
        f"xpos{s}": greens.xpos.reshape(-1, 1),
        f"times{s}": greens.times.reshape(-1, 1),
        f"Geometry{s}": _geometry_mat_struct(greens.Geometry),
    }


def save_greens_mat(filename: str, greens: VECycleGreens, *, suffix: str = "") -> None:
    """Save full Green's functions to a MATLAB-readable MAT file."""
    sio.savemat(filename, greens_to_mat_dict(greens, suffix=suffix), do_compression=True)


def load_greens_mat(filename: str) -> dict[str, Any]:
    """Load a MAT file containing VECycle2D Green's functions.

    This returns scipy's raw dictionary. A full MAT-to-dataclass converter can
    be added later if needed.
    """
    return sio.loadmat(filename, squeeze_me=False, struct_as_record=False)


__all__ = [
    "GeometryOutput",
    "VECycleGreens",
    "build_greens",
    "greens_summary",
    "greens_to_mat_dict",
    "save_greens_mat",
    "load_greens_mat",
]
