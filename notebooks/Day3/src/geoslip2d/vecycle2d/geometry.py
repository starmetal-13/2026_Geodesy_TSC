"""Geometry construction for the VECycle2D Python port.

This module ports the geometry layer from the verified MATLAB code:

* ``make_geometry_slab_wedge.m``
* ``vec_make_wedge_geometry.m``
* ``vec_make_boundary.m``

The goal of Step 1 is to reproduce the same patch coordinates, centers,
dip vectors, and normal vectors as MATLAB before porting the Okada and cycle
solvers.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from .boundaries import Boundary, make_boundary
from .config import Config, GeometryConfig, InternalConstants


@dataclass(slots=True)
class Surface:
    x: np.ndarray


@dataclass(slots=True)
class Geometry:
    shift: float
    interface: Boundary
    slab_bottom: Boundary
    slab_top: Boundary
    boundaries: list[Boundary]
    surface: Surface
    centers_interface: np.ndarray
    num_interface: int
    cfg: GeometryConfig
    legacy: dict[str, Any]


def matlab_colon(start: float, step: float, stop: float) -> np.ndarray:
    """Approximate MATLAB's ``start:step:stop`` behavior for scalar floats.

    MATLAB includes the final value only if it lands on the grid. This helper
    avoids ``np.arange`` surprises due to roundoff and supports negative steps.
    """

    start = float(start)
    step = float(step)
    stop = float(stop)
    if step == 0:
        raise ValueError("step must be non-zero")
    if step > 0 and start > stop:
        return np.array([], dtype=float)
    if step < 0 and start < stop:
        return np.array([], dtype=float)

    tol = 1.0e-10 * max(1.0, abs(start), abs(stop), abs(step))
    vals: list[float] = []
    x = start
    if step > 0:
        while x <= stop + tol:
            vals.append(x)
            x = start + len(vals) * step
    else:
        while x >= stop - tol:
            vals.append(x)
            x = start + len(vals) * step
    return np.asarray(vals, dtype=float)


def _row(*parts: np.ndarray | float) -> np.ndarray:
    """Concatenate values into a 1D float array."""

    arrays = [np.asarray(p, dtype=float).reshape(-1) for p in parts]
    return np.concatenate(arrays) if arrays else np.array([], dtype=float)


def _centers(topx: np.ndarray, topz: np.ndarray, botx: np.ndarray, botz: np.ndarray) -> np.ndarray:
    return np.vstack(((topx + botx) / 2.0, (topz + botz) / 2.0))


def _normalize(v: np.ndarray) -> np.ndarray:
    out = np.asarray(v, dtype=float).copy()
    norm = np.sqrt(out[0, :] ** 2 + out[1, :] ** 2)
    out[0, :] /= norm
    out[1, :] /= norm
    return out


def _normal_from_dipvec(dipvec: np.ndarray) -> np.ndarray:
    # MATLAB: -[dipvec(2,:); -dipvec(1,:)]
    return np.vstack((-dipvec[1, :], dipvec[0, :]))


def make_geometry_slab_wedge(
    H_elastic_left: float,
    H_elastic_right: float,
    faultdip_trench: float,
    x_trench: float,
    x_bottom: float,
    faultdip_bottom: float,
    L_slab: float,
    W: float,
    pL: float,
    shift: float,
    wedge_bot: float,
    wedge_top_x: float,
) -> dict[str, np.ndarray]:
    """Port of MATLAB ``make_geometry_slab_wedge.m``.

    Returns a dictionary containing the same named arrays as the MATLAB
    function outputs.
    """

    pi = np.pi

    # Subduction interface top: cubic with specified depths and endpoint dips.
    fault_bot_x = x_bottom
    top_fault = 0.0
    b = np.array([
        np.tan(-faultdip_bottom * pi / 180.0),
        np.tan(-faultdip_trench * pi / 180.0),
        -top_fault,
        -wedge_bot,
    ])
    A = np.array([
        [3 * fault_bot_x**2, 2 * fault_bot_x, 1.0, 0.0],
        [3 * x_trench**2, 2 * x_trench, 1.0, 0.0],
        [x_trench**3, x_trench**2, x_trench, 1.0],
        [fault_bot_x**3, fault_bot_x**2, fault_bot_x, 1.0],
    ])
    c = np.linalg.solve(A, b)

    depth = 0.0
    dist = 0.0
    x_coord = [float(x_trench)]
    z_coord = [depth]
    while abs(depth) < wedge_bot:
        dip = np.arctan(c[0] * 3 * dist**2 + c[1] * 2 * dist + c[2])
        dx = pL * np.cos(dip)
        dist = dist + dx
        z_coord.append(c[0] * dist**3 + c[1] * dist**2 + c[2] * dist + c[3])
        x_coord.append(x_trench + dist)
        depth = z_coord[-1]

    x_coord = np.asarray(x_coord, dtype=float)
    z_coord = np.asarray(z_coord, dtype=float)
    topx_interface = x_coord[:-1]
    topz_interface = z_coord[:-1]
    botx_interface = x_coord[1:]
    botz_interface = z_coord[1:]

    # Bottom of slab.
    dip = np.arctan2(botz_interface[-1] - topz_interface[-1], botx_interface[-1] - topx_interface[-1])
    angle = dip - pi / 2.0
    x_corner = botx_interface[-1] + H_elastic_left * np.cos(angle)
    z_corner = botz_interface[-1] + H_elastic_left * np.sin(angle)

    fault_bot_x = x_corner
    b = np.array([
        np.tan(dip),
        np.tan(-faultdip_trench * pi / 180.0),
        -H_elastic_left,
        z_corner,
    ])
    A = np.array([
        [3 * fault_bot_x**2, 2 * fault_bot_x, 1.0, 0.0],
        [3 * x_trench**2, 2 * x_trench, 1.0, 0.0],
        [x_trench**3, x_trench**2, x_trench, 1.0],
        [fault_bot_x**3, fault_bot_x**2, fault_bot_x, 1.0],
    ])
    c = np.linalg.solve(A, b)

    depth = -H_elastic_left
    dist = 0.0
    x_coord = [float(x_trench)]
    z_coord = [depth]
    while abs(depth) < abs(z_corner):
        dip = np.arctan(c[0] * 3 * dist**2 + c[1] * 2 * dist + c[2])
        dx = pL * np.cos(dip)
        dz = pL * np.sin(dip)
        depth = depth + dz
        dist = dist + dx
        z_coord.append(c[0] * dist**3 + c[1] * dist**2 + c[2] * dist + c[3])
        x_coord.append(dist)
        depth = z_coord[-1]

    x_coord = np.asarray(x_coord, dtype=float)
    z_coord = np.asarray(z_coord, dtype=float)
    topx_botslab = x_coord[:-1]
    topz_botslab = z_coord[:-1]
    botx_botslab = x_coord[1:]
    botz_botslab = z_coord[1:]

    # Extension of slab into mantle.
    dip = np.arctan2(botz_interface[-1] - topz_interface[-1], botx_interface[-1] - topx_interface[-1])
    topx_topslab = matlab_colon(botx_interface[-1], pL * np.cos(dip), botx_interface[-1] + (L_slab - pL) * np.cos(dip))
    topz_topslab = matlab_colon(botz_interface[-1], pL * np.sin(dip), botz_interface[-1] + (L_slab - pL) * np.sin(dip))
    botx_topslab = matlab_colon(botx_interface[-1] + pL * np.cos(dip), pL * np.cos(dip), botx_interface[-1] + L_slab * np.cos(dip))
    botz_topslab = matlab_colon(botz_interface[-1] + pL * np.sin(dip), pL * np.sin(dip), botz_interface[-1] + L_slab * np.sin(dip))

    topx_botslab = _row(topx_botslab, matlab_colon(botx_botslab[-1], pL * np.cos(dip), botx_botslab[-1] + (L_slab - pL) * np.cos(dip)))
    topz_botslab = _row(topz_botslab, matlab_colon(botz_botslab[-1], pL * np.sin(dip), botz_botslab[-1] + (L_slab - pL) * np.sin(dip)))
    botx_botslab = _row(botx_botslab, matlab_colon(botx_botslab[-1] + pL * np.cos(dip), pL * np.cos(dip), botx_botslab[-1] + L_slab * np.cos(dip)))
    botz_botslab = _row(botz_botslab, matlab_colon(botz_botslab[-1] + pL * np.sin(dip), pL * np.sin(dip), botz_botslab[-1] + L_slab * np.sin(dip)))

    # Lower edge of slab.
    length = np.sqrt((botx_botslab[-1] - botx_topslab[-1]) ** 2 + (botz_botslab[-1] - botz_topslab[-1]) ** 2)
    Nedge = int(np.round(length / pL))
    pL_edge = length / Nedge
    angle = np.arctan2(botz_botslab[-1] - botz_topslab[-1], botx_botslab[-1] - botx_topslab[-1])
    topx3 = matlab_colon(botx_botslab[-1], -pL_edge * np.cos(angle), botx_topslab[-1] + pL_edge * np.cos(angle))
    topz3 = matlab_colon(botz_botslab[-1], -pL_edge * np.sin(angle), botz_topslab[-1] + pL_edge * np.sin(angle))
    botx3 = matlab_colon(botx_botslab[-1] - pL_edge * np.cos(angle), -pL_edge * np.cos(angle), botx_topslab[-1])
    botz3 = matlab_colon(botz_botslab[-1] - pL_edge * np.sin(angle), -pL_edge * np.sin(angle), botz_topslab[-1])

    # Lower over-riding plate boundary.
    length = np.sqrt((wedge_top_x - botx_interface[-1]) ** 2 + (-botz_interface[-1] - H_elastic_right) ** 2)
    Nedge = int(np.round(length / pL))
    pLx = (wedge_top_x - botx_interface[-1]) / Nedge
    pLz = abs(-botz_interface[-1] - H_elastic_right) / Nedge
    topx1 = matlab_colon(botx_interface[-1], pLx, wedge_top_x - pLx)
    topz1 = matlab_colon(botz_interface[-1], pLz, -H_elastic_right - pLz)
    botx1 = matlab_colon(botx_interface[-1] + pLx, pLx, wedge_top_x)
    botz1 = matlab_colon(botz_interface[-1] + pLz, pLz, -H_elastic_right)

    topx1_2 = matlab_colon(wedge_top_x, pL, wedge_top_x + W * pL - pL)
    topz1_2 = -H_elastic_right * np.ones_like(topx1_2)
    botx1_2 = matlab_colon(wedge_top_x + pL, pL, wedge_top_x + W * pL)
    botz1_2 = -H_elastic_right * np.ones_like(topx1_2)
    topx1 = _row(topx1, topx1_2)
    botx1 = _row(botx1, botx1_2)
    topz1 = _row(topz1, topz1_2)
    botz1 = _row(botz1, botz1_2)

    # Lower under-riding plate boundary.
    topx2 = matlab_colon(topx_botslab[0] - pL, -pL, topx_botslab[0] - W * pL)
    topz2 = topz_botslab[0] * np.ones_like(topx2)
    botx2 = matlab_colon(topx_botslab[0], -pL, topx_botslab[0] - W * pL + pL)
    botz2 = topz_botslab[0] * np.ones_like(topx2)
    topx2 = topx2[::-1]
    topz2 = topz2[::-1]
    botx2 = botx2[::-1]
    botz2 = botz2[::-1]

    # Left edge.
    length = abs(topz2[0])
    Nedge = int(np.round(length / pL))
    pL_edge = length / Nedge
    topz4 = matlab_colon(0.0, -pL_edge, -length + pL_edge)
    topx4 = topx2[0] * np.ones_like(topz4)
    botz4 = matlab_colon(-pL_edge, -pL_edge, -length)
    botx4 = topx2[0] * np.ones_like(topz4)

    # Right edge.
    length = abs(topz1[-1])
    Nedge = int(np.round(length / pL))
    pL_edge = length / Nedge
    topz5 = matlab_colon(0.0, -pL_edge, -length + pL_edge)
    topx5 = botx1[-1] * np.ones_like(topz5)
    botz5 = matlab_colon(-pL_edge, -pL_edge, -length)
    botx5 = botx1[-1] * np.ones_like(topz5)

    # Free surface.
    length = topx5[0] - x_trench
    Nedge = int(np.round(length / pL))
    pL_edge = length / Nedge
    topx6 = matlab_colon(x_trench, pL_edge, topx5[0] - pL_edge)
    topz6 = np.zeros_like(topx6)
    botx6 = topx6 + pL_edge
    botz6 = np.zeros_like(topx6)

    length = x_trench - topx4[0]
    Nedge = int(np.round(length / pL))
    pL_edge = length / Nedge
    topx7 = matlab_colon(topx4[0], pL_edge, x_trench - pL_edge)
    topz7 = np.zeros_like(topx7)
    botx7 = topx7 + pL_edge
    botz7 = np.zeros_like(topx7)

    # Viscoelastic free surface.
    length = 5 * W
    Nedge = int(np.round(length / (1 * pL)))
    pL_edge = length / Nedge
    topx8_right = matlab_colon(topx5[0], pL_edge, topx5[0] + length - pL_edge)
    topz8_right = np.zeros_like(topx8_right)
    botx8_right = topx8_right + pL_edge
    botz8_right = np.zeros_like(topx8_right)

    topx9 = matlab_colon(topx4[0] - length, pL_edge, topx4[0] - pL_edge)
    topz9 = np.zeros_like(topx9)
    botx9 = topx9 + pL_edge
    botz9 = np.zeros_like(topx9)

    topx8 = _row(topx9, topx8_right)
    topz8 = _row(topz9, topz8_right)
    botx8 = _row(botx9, botx8_right)
    botz8 = _row(botz9, botz8_right)

    arrays = locals().copy()
    output_names = [
        "topx_interface", "topz_interface", "botx_interface", "botz_interface",
        "topx_botslab", "topz_botslab", "botx_botslab", "botz_botslab",
        "topx_topslab", "topz_topslab", "botx_topslab", "botz_topslab",
        "topx1", "topz1", "botx1", "botz1", "topx2", "topz2", "botx2", "botz2",
        "topx3", "topz3", "botx3", "botz3", "topx4", "topz4", "botx4", "botz4",
        "topx5", "topz5", "botx5", "botz5", "topx6", "topz6", "botx6", "botz6",
        "topx7", "topz7", "botx7", "botz7", "topx8", "topz8", "botx8", "botz8",
    ]
    out = {name: np.asarray(arrays[name], dtype=float).copy() for name in output_names}

    # Apply legacy vertical shift to all z arrays.
    for name in list(out):
        if "z" in name:
            out[name] = out[name] - shift
    return out


def make_wedge_geometry(
    cfg: GeometryConfig | Config | None = None,
    constants: InternalConstants | None = None,
) -> Geometry:
    """Build the explicit VECycle2D geometry object.

    Parameters
    ----------
    cfg
        Either a full ``Config`` or a ``GeometryConfig``. If omitted, defaults
        are used.
    constants
        Internal constants. If omitted, defaults are used.
    """

    if cfg is None:
        cfg = Config()
    if isinstance(cfg, Config):
        constants = cfg.constants if constants is None else constants
        geom_cfg = cfg.geometry
    else:
        geom_cfg = cfg
        constants = InternalConstants() if constants is None else constants

    g = make_geometry_slab_wedge(
        geom_cfg.H_elastic_left,
        geom_cfg.H_elastic_right,
        geom_cfg.faultdip_trench,
        geom_cfg.x_trench,
        geom_cfg.x_bottom,
        geom_cfg.faultdip_bottom,
        geom_cfg.L_slab,
        geom_cfg.W,
        geom_cfg.pL,
        constants.shift,
        geom_cfg.wedge_bot,
        geom_cfg.wedge_top_x,
    )

    # Boundary centers, dip vectors, and normals. These mirror
    # vec_make_wedge_geometry.m exactly, including special cases.
    topx1, topz1, botx1, botz1 = g["topx1"], g["topz1"], g["botx1"], g["botz1"]
    topx2, topz2, botx2, botz2 = g["topx2"], g["topz2"], g["botx2"], g["botz2"]
    topx3, topz3, botx3, botz3 = g["topx3"], g["topz3"], g["botx3"], g["botz3"]
    topx4, topz4, botx4, botz4 = g["topx4"], g["topz4"], g["botx4"], g["botz4"]
    topx5, topz5, botx5, botz5 = g["topx5"], g["topz5"], g["botx5"], g["botz5"]
    topx6, topz6, botx6, botz6 = g["topx6"], g["topz6"], g["botx6"], g["botz6"]
    topx7, topz7, botx7, botz7 = g["topx7"], g["topz7"], g["botx7"], g["botz7"]
    topx8, topz8, botx8, botz8 = g["topx8"], g["topz8"], g["botx8"], g["botz8"]

    centers1 = _centers(topx1, topz1, botx1, botz1)
    centers1e = centers1 + 1.0e-3 * _normal_from_dipvec(_normalize(np.vstack((botx1 - topx1, botz1 - topz1))))
    centers1v = centers1 - 1.0e-3 * _normal_from_dipvec(_normalize(np.vstack((botx1 - topx1, botz1 - topz1))))
    dipvec1 = _normalize(np.vstack((botx1 - topx1, botz1 - topz1)))
    normvec1 = _normal_from_dipvec(dipvec1)

    centers2 = np.vstack(((topx2 + botx2) / 2.0, topz2))
    centers2e = centers2.copy(); centers2e[1, :] = centers2[1, :] + 1.0e-3
    centers2v = centers2.copy(); centers2v[1, :] = centers2[1, :] - 1.0e-3
    dipvec2 = _normalize(np.vstack((botx2 - topx2, -(botz2 - topz2))))
    normvec2 = _normal_from_dipvec(dipvec2)

    centers3 = _centers(topx3, topz3, botx3, botz3)
    centers3e = centers3.copy(); centers3e[1, :] = centers3[1, :] + 1.0e-3
    centers3v = centers3.copy(); centers3v[1, :] = centers3[1, :] - 1.0e-3
    dipvec3 = _normalize(np.vstack((botx3 - topx3, botz3 - topz3)))
    normvec3 = _normal_from_dipvec(dipvec3)

    centers4 = _centers(topx4, topz4, botx4, botz4)
    centers4e = centers4.copy(); centers4e[0, :] = centers4[0, :] + 1.0e-3
    centers4v = centers4.copy(); centers4v[0, :] = centers4[0, :] - 1.0e-3
    dipvec4 = _normalize(np.vstack((botx4 - topx4, botz4 - topz4)))
    normvec4 = _normal_from_dipvec(dipvec4)

    centers5 = _centers(topx5, topz5, botx5, botz5)
    centers5e = centers5.copy(); centers5e[0, :] = centers5[0, :] - 1.0e-3
    centers5v = centers5.copy(); centers5v[0, :] = centers5[0, :] + 1.0e-3
    dipvec5 = _normalize(np.vstack((botx5 - topx5, botz5 - topz5)))
    normvec5 = _normal_from_dipvec(dipvec5)

    centers6 = _centers(topx6, topz6, botx6, botz6)
    centers6e = centers6.copy(); centers6e[1, :] = centers6[1, :] - 1.0e-3
    dipvec6 = _normalize(np.vstack((botx6 - topx6, botz6 - topz6)))
    normvec6 = _normal_from_dipvec(dipvec6)

    centers7 = _centers(topx7, topz7, botx7, botz7)
    centers7e = centers7.copy(); centers7e[1, :] = centers7[1, :] - 1.0e-3
    dipvec7 = _normalize(np.vstack((botx7 - topx7, botz7 - topz7)))
    normvec7 = _normal_from_dipvec(dipvec7)

    centers8 = _centers(topx8, topz8, botx8, botz8)
    centers8v = centers8.copy(); centers8v[1, :] = centers8[1, :] - 1.0e-3
    dipvec8 = _normalize(np.vstack((botx8 - topx8, botz8 - topz8)))
    normvec8 = _normal_from_dipvec(dipvec8)

    topx_interface, topz_interface = g["topx_interface"], g["topz_interface"]
    botx_interface, botz_interface = g["botx_interface"], g["botz_interface"]
    centers_interface = _centers(topx_interface, topz_interface, botx_interface, botz_interface)
    dipvec_interface = _normalize(np.vstack((botx_interface - topx_interface, botz_interface - topz_interface)))
    normvec_interface = np.vstack((dipvec_interface[1, :], -dipvec_interface[0, :]))

    topx_botslab, topz_botslab = g["topx_botslab"], g["topz_botslab"]
    botx_botslab, botz_botslab = g["botx_botslab"], g["botz_botslab"]
    centers_botslab = _centers(topx_botslab, topz_botslab, botx_botslab, botz_botslab)
    centers_botslabe = centers_botslab.copy(); centers_botslabe[1, :] = centers_botslab[1, :] + 1.0e-3
    centers_botslabv = centers_botslab.copy(); centers_botslabv[1, :] = centers_botslab[1, :] - 1.0e-3
    dipvec_botslab = _normalize(np.vstack((botx_botslab - topx_botslab, botz_botslab - topz_botslab)))
    normvec_botslab = _normal_from_dipvec(dipvec_botslab)

    topx_topslab, topz_topslab = g["topx_topslab"], g["topz_topslab"]
    botx_topslab, botz_topslab = g["botx_topslab"], g["botz_topslab"]
    centers_topslab = _centers(topx_topslab, topz_topslab, botx_topslab, botz_topslab)
    centers_topslabe = centers_topslab.copy(); centers_topslabe[1, :] = centers_topslab[1, :] - 1.0e-3
    centers_topslabv = centers_topslab.copy(); centers_topslabv[1, :] = centers_topslab[1, :] + 1.0e-3
    dipvec_topslab = _normalize(np.vstack((botx_topslab - topx_topslab, botz_topslab - topz_topslab)))
    normvec_topslab = _normal_from_dipvec(dipvec_topslab)

    centers = np.hstack((centers1, centers2, centers3, centers4, centers5, centers6, centers7, centers_botslab, centers_topslab, centers_interface))

    boundaries = [
        make_boundary("b1_overriding_left_side", topx1, topz1, botx1, botz1, centers1, dipvec1, normvec1),
        make_boundary("b2_overriding_surface", topx2, topz2, botx2, botz2, centers2, dipvec2, normvec2),
        make_boundary("b3_overriding_right_side", topx3, topz3, botx3, botz3, centers3, dipvec3, normvec3),
        make_boundary("b4_overriding_base", topx4, topz4, botx4, botz4, centers4, dipvec4, normvec4),
        make_boundary("b5_slab_left_side", topx5, topz5, botx5, botz5, centers5, dipvec5, normvec5),
        make_boundary("b6_left_surface", topx6, topz6, botx6, botz6, centers6, dipvec6, normvec6),
        make_boundary("b7_right_surface", topx7, topz7, botx7, botz7, centers7, dipvec7, normvec7),
        make_boundary("b8_wedge_boundary", topx8, topz8, botx8, botz8, centers8, dipvec8, normvec8),
    ]

    # Legacy dictionary preserves MATLAB field names for later port phases.
    legacy = dict(g)
    legacy.update(locals())
    # Remove bulky/irrelevant local aliases that are not MATLAB legacy fields.
    for key in ["cfg", "geom_cfg", "constants", "g", "boundaries", "legacy", "key"]:
        legacy.pop(key, None)

    # ------------------------------------------------------------------
    # Public/user-facing interface geometry
    # ------------------------------------------------------------------
    # The arrays in g and legacy preserve the MATLAB internal convention in
    # which all z coordinates have been shifted downward by constants.shift.
    # That shifted geometry is still needed by the raw Green's-function
    # builder through geom.legacy.
    #
    # For user-facing outputs, however, greens.Geometry.centers_interface
    # should contain the physical interface coordinates, with z negative
    # downward and no artificial shift. Undo the shift only for the public
    # interface object and public centers_interface field.
    topz_interface_public = topz_interface + constants.shift
    botz_interface_public = botz_interface + constants.shift
    centers_interface_public = _centers(
        topx_interface,
        topz_interface_public,
        botx_interface,
        botz_interface_public,
    )

    interface = make_boundary(
        "interface",
        topx_interface,
        topz_interface_public,
        botx_interface,
        botz_interface_public,
        centers_interface_public,
        dipvec_interface,
        normvec_interface,
    )

    slab_bottom = make_boundary("slab_bottom", topx_botslab, topz_botslab, botx_botslab, botz_botslab, centers_botslab, dipvec_botslab, normvec_botslab)
    slab_top = make_boundary("slab_top", topx_topslab, topz_topslab, botx_topslab, botz_topslab, centers_topslab, dipvec_topslab, normvec_topslab)

    return Geometry(
        shift=constants.shift,
        interface=interface,
        slab_bottom=slab_bottom,
        slab_top=slab_top,
        boundaries=boundaries,
        surface=Surface(x=np.concatenate((centers7[0, :], centers6[0, :]))),
        centers_interface=centers_interface_public,
        num_interface=int(centers_interface_public.shape[1]),
        cfg=geom_cfg,
        legacy=legacy,
    )


def geometry_summary(geom: Geometry) -> dict[str, int]:
    """Return patch counts for quick notebook checks."""

    out = {b.name: b.npatch for b in geom.boundaries}
    out["interface"] = geom.interface.npatch
    out["slab_bottom"] = geom.slab_bottom.npatch
    out["slab_top"] = geom.slab_top.npatch
    return out
