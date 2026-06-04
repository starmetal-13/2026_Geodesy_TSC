from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

from .params import (
    ElasticWedgeParams,
    BoundaryID,
    BOUNDARY_NAMES,
    MaterialID,
)


@dataclass
class Boundary:
    id: BoundaryID
    name: str
    top: np.ndarray      # shape (2, n)
    bot: np.ndarray      # shape (2, n)
    topx: np.ndarray
    topz: np.ndarray
    botx: np.ndarray
    botz: np.ndarray
    center: np.ndarray
    centerx: np.ndarray
    centerz: np.ndarray
    dipvec: np.ndarray
    normvec: np.ndarray
    length: np.ndarray
    center_t: np.ndarray
    center_b: np.ndarray

    @property
    def n_patch(self) -> int:
        return self.center.shape[1]


@dataclass
class SourceBlock:
    boundary: BoundaryID
    side: str
    domain: MaterialID
    label: str


@dataclass
class ReceiverSide:
    boundary: BoundaryID
    side: str
    domain: MaterialID
    label: str


@dataclass
class SourcePatch:
    q: int
    boundary: BoundaryID
    boundary_name: str
    patch: int
    m: np.ndarray
    length: float
    dip_deg: float
    center: np.ndarray
    top: np.ndarray
    bot: np.ndarray


@dataclass
class InterfaceGeometry:
    boundaries: List[BoundaryID]
    top: np.ndarray
    bot: np.ndarray
    center: np.ndarray

    @property
    def n_patch(self) -> int:
        return self.center.shape[1]


@dataclass
class Geometry:
    B: List[Boundary]
    centers: np.ndarray
    n_patch: int
    interface: InterfaceGeometry
    source_blocks: List[SourceBlock]
    receiver_sides: List[ReceiverSide]
    sources: List[SourcePatch]

    @property
    def num_boundaries(self) -> int:
        return len(self.B)


def _matlab_colon(start: float, step: float, stop: float) -> np.ndarray:
    """MATLAB-like colon including values <= stop for positive step and >= for negative."""
    if step == 0:
        raise ValueError("step must be nonzero")
    # generous eps to match MATLAB floating endpoint behavior
    n = int(np.floor((stop - start) / step + 1e-12)) + 1
    if n <= 0:
        return np.array([], dtype=float)
    return start + step * np.arange(n, dtype=float)


def make_geometry_elastic_wedge(params: ElasticWedgeParams):
    p = params
    # Curved subduction interface represented by a cubic.
    fault_bot_x = p.x_bottom
    top_fault = 0.0
    b = np.array([
        np.tan(-p.faultdip_bottom * np.pi / 180.0),
        np.tan(-p.faultdip_trench * np.pi / 180.0),
        -top_fault,
        -p.z_bottom,
    ], dtype=float)
    A = np.array([
        [3 * fault_bot_x**2, 2 * fault_bot_x, 1, 0],
        [3 * p.x_trench**2, 2 * p.x_trench, 1, 0],
        [p.x_trench**3, p.x_trench**2, p.x_trench, 1],
        [fault_bot_x**3, fault_bot_x**2, fault_bot_x, 1],
    ], dtype=float)
    c = np.linalg.solve(A, b)

    depth = 0.0
    dist = 0.0
    x_coord = [p.x_trench]
    z_coord = [depth]
    while abs(depth) < p.z_bottom:
        dip = np.arctan(c[0] * 3 * dist**2 + c[1] * 2 * dist + c[2])
        dx = p.pL * np.cos(dip)
        dist = dist + dx
        z_new = c[0] * dist**3 + c[1] * dist**2 + c[2] * dist + c[3]
        x_coord.append(p.x_trench + dist)
        z_coord.append(z_new)
        depth = z_new

    x_coord = np.asarray(x_coord)
    z_coord = np.asarray(z_coord)
    topx_interface = x_coord[:-1]
    topz_interface = z_coord[:-1]
    botx_interface = x_coord[1:]
    botz_interface = z_coord[1:]

    # Extension of slab into mantle.
    dip = np.arctan2(botz_interface[-1] - topz_interface[-1],
                     botx_interface[-1] - topx_interface[-1])
    topx_topslab = _matlab_colon(botx_interface[-1], p.pL*np.cos(dip),
                                 botx_interface[-1] + (p.L_slab-p.pL)*np.cos(dip))
    topz_topslab = _matlab_colon(botz_interface[-1], p.pL*np.sin(dip),
                                 botz_interface[-1] + (p.L_slab-p.pL)*np.sin(dip))
    botx_topslab = _matlab_colon(botx_interface[-1]+p.pL*np.cos(dip), p.pL*np.cos(dip),
                                 botx_interface[-1]+p.L_slab*np.cos(dip))
    botz_topslab = _matlab_colon(botz_interface[-1]+p.pL*np.sin(dip), p.pL*np.sin(dip),
                                 botz_interface[-1]+p.L_slab*np.sin(dip))

    # Free surface, right of trench.
    length = 4*p.W - p.x_trench
    Nedge = int(round(length / p.pL))
    pL_edge = length / Nedge
    topx2 = _matlab_colon(p.x_trench, pL_edge, 4*p.W-pL_edge)
    topz2 = np.zeros_like(topx2)
    botx2 = topx2 + pL_edge
    botz2 = np.zeros_like(topx2)

    # Free surface, left of trench.
    length = p.x_trench - 2*p.W
    Nedge = int(round(length / p.pL))
    pL_edge = length / Nedge
    topx1 = _matlab_colon(-2*p.W, pL_edge, p.x_trench-pL_edge)
    topz1 = np.zeros_like(topx1)
    botx1 = topx1 + pL_edge
    botz1 = np.zeros_like(topx1)

    ind = np.where(topz_interface > -p.wedge_bot)[0]
    if ind.size == 0:
        raise ValueError("No interface point found above wedge_bot; check geometry parameters.")
    wedge_bot_x = topx_interface[ind[-1]]
    wedge_bot_z = topz_interface[ind[-1]]

    ind = np.where(topx2 < p.wedge_top_x)[0]
    if ind.size == 0:
        raise ValueError("No free-surface point found left of wedge_top_x; check geometry parameters.")
    wedge_top_x = topx2[ind[-1]]
    wedge_top_z = topz2[ind[-1]]

    length = np.sqrt((wedge_top_x - wedge_bot_x)**2 + (wedge_top_z - wedge_bot_z)**2)
    Nedge = int(round(length / p.pL))
    pLx = abs(wedge_top_x - wedge_bot_x) / Nedge
    pLz = abs(wedge_top_z - wedge_bot_z) / Nedge

    topx4 = _matlab_colon(wedge_bot_x, pLx, wedge_top_x-pLx)
    topz4 = _matlab_colon(wedge_bot_z, pLz, wedge_top_z-pLz)
    botx4 = _matlab_colon(wedge_bot_x+pLx, pLx, wedge_top_x)
    botz4 = _matlab_colon(wedge_bot_z+pLz, pLz, wedge_top_z)

    # Split right free surface into wedge/upper plate pieces.
    ind = np.where(botx2 <= wedge_top_x)[0]
    if ind.size == 0:
        raise ValueError("Cannot split free surface at wedge_top_x; check geometry parameters.")
    last = ind[-1]
    topx3 = topx2[last+1:]
    botx3 = botx2[last+1:]
    topz3 = topz2[last+1:]
    botz3 = botz2[last+1:]

    topx2 = topx2[:last+1]
    botx2 = botx2[:last+1]
    topz2 = topz2[:last+1]
    botz2 = botz2[:last+1]

    topz5 = topz_interface - p.shift
    botz5 = botz_interface - p.shift
    topx5 = topx_interface
    botx5 = botx_interface

    topz6 = topz_topslab - p.shift
    botz6 = botz_topslab - p.shift
    topx6 = topx_topslab
    botx6 = botx_topslab

    topz1 = topz1 - p.shift; botz1 = botz1 - p.shift
    topz2 = topz2 - p.shift; botz2 = botz2 - p.shift
    topz3 = topz3 - p.shift; botz3 = botz3 - p.shift
    topz4 = topz4 - p.shift; botz4 = botz4 - p.shift

    return [
        (topx1, topz1, botx1, botz1),
        (topx2, topz2, botx2, botz2),
        (topx3, topz3, botx3, botz3),
        (topx4, topz4, botx4, botz4),
        (topx5, topz5, botx5, botz5),
        (topx6, topz6, botx6, botz6),
    ]


def elastic_wedge_source_blocks() -> List[SourceBlock]:
    return [
        SourceBlock(BoundaryID.SURFACE_LEFT,        "b", MaterialID.SUBDUCTING_SLAB,   "1b"),
        SourceBlock(BoundaryID.SURFACE_WEDGE,       "b", MaterialID.COMPLIANT_WEDGE,   "2b"),
        SourceBlock(BoundaryID.SURFACE_UPPER_PLATE, "b", MaterialID.CONTINENTAL_CRUST, "3b"),
        SourceBlock(BoundaryID.WEDGE_BACKSTOP,      "b", MaterialID.CONTINENTAL_CRUST, "4b"),
        SourceBlock(BoundaryID.WEDGE_BACKSTOP,      "t", MaterialID.COMPLIANT_WEDGE,   "4t"),
        SourceBlock(BoundaryID.MEGATHRUST,          "b", MaterialID.SUBDUCTING_SLAB,   "5b"),
        SourceBlock(BoundaryID.MEGATHRUST,          "t", MaterialID.COMPLIANT_WEDGE,   "5t"),
        SourceBlock(BoundaryID.SLAB_EXTENSION,      "b", MaterialID.SUBDUCTING_SLAB,   "6b"),
        SourceBlock(BoundaryID.SLAB_EXTENSION,      "t", MaterialID.CONTINENTAL_CRUST, "6t"),
    ]


def elastic_wedge_receiver_sides() -> List[ReceiverSide]:
    return [
        ReceiverSide(BoundaryID.SURFACE_LEFT,        "b", MaterialID.SUBDUCTING_SLAB,   "1"),
        ReceiverSide(BoundaryID.SURFACE_WEDGE,       "b", MaterialID.COMPLIANT_WEDGE,   "2"),
        ReceiverSide(BoundaryID.SURFACE_UPPER_PLATE, "b", MaterialID.CONTINENTAL_CRUST, "3"),
        ReceiverSide(BoundaryID.WEDGE_BACKSTOP,      "t", MaterialID.COMPLIANT_WEDGE,   "4t"),
        ReceiverSide(BoundaryID.WEDGE_BACKSTOP,      "b", MaterialID.CONTINENTAL_CRUST, "4b"),
        ReceiverSide(BoundaryID.MEGATHRUST,          "t", MaterialID.COMPLIANT_WEDGE,   "5t"),
        ReceiverSide(BoundaryID.MEGATHRUST,          "b", MaterialID.SUBDUCTING_SLAB,   "5b"),
        ReceiverSide(BoundaryID.SLAB_EXTENSION,      "t", MaterialID.CONTINENTAL_CRUST, "6t"),
        ReceiverSide(BoundaryID.SLAB_EXTENSION,      "b", MaterialID.SUBDUCTING_SLAB,   "6b"),
    ]


def make_elastic_wedge_geometry_struct(params: ElasticWedgeParams) -> Geometry:
    parts = make_geometry_elastic_wedge(params)

    # Preserve original special handling: move segment-5 patches below wedge to segment 6.
    topx4, topz4, botx4, botz4 = parts[BoundaryID.WEDGE_BACKSTOP]
    topx5, topz5, botx5, botz5 = parts[BoundaryID.MEGATHRUST]
    topx6, topz6, botx6, botz6 = parts[BoundaryID.SLAB_EXTENSION]
    ind = botz5 < topz4[0]
    parts[BoundaryID.SLAB_EXTENSION] = (
        np.concatenate([topx5[ind], topx6]),
        np.concatenate([topz5[ind], topz6]),
        np.concatenate([botx5[ind], botx6]),
        np.concatenate([botz5[ind], botz6]),
    )
    parts[BoundaryID.MEGATHRUST] = (topx5[~ind], topz5[~ind], botx5[~ind], botz5[~ind])

    B: List[Boundary] = []
    for k, (topx, topz, botx, botz) in enumerate(parts):
        top = np.vstack([topx, topz])
        bot = np.vstack([botx, botz])
        center = 0.5 * (top + bot)
        dipvec = bot - top
        length = np.sqrt(np.sum(dipvec**2, axis=0))
        dipvec = dipvec / length
        normvec = -np.vstack([dipvec[1, :], -dipvec[0, :]])
        center_t = center + params.self_offset * normvec
        center_b = center - params.self_offset * normvec
        B.append(Boundary(
            id=BoundaryID(k), name=BOUNDARY_NAMES[k], top=top, bot=bot,
            topx=topx, topz=topz, botx=botx, botz=botz,
            center=center, centerx=center[0, :], centerz=center[1, :],
            dipvec=dipvec, normvec=normvec, length=length,
            center_t=center_t, center_b=center_b,
        ))

    interface_top = np.hstack([B[BoundaryID.MEGATHRUST].top, B[BoundaryID.SLAB_EXTENSION].top])
    interface_bot = np.hstack([B[BoundaryID.MEGATHRUST].bot, B[BoundaryID.SLAB_EXTENSION].bot])
    interface = InterfaceGeometry(
        boundaries=[BoundaryID.MEGATHRUST, BoundaryID.SLAB_EXTENSION],
        top=interface_top,
        bot=interface_bot,
        center=0.5 * (interface_top + interface_bot),
    )
    source_blocks = elastic_wedge_source_blocks()
    receiver_sides = elastic_wedge_receiver_sides()
    geom = Geometry(
        B=B,
        centers=np.hstack([b.center for b in B]),
        n_patch=sum(b.n_patch for b in B),
        interface=interface,
        source_blocks=source_blocks,
        receiver_sides=receiver_sides,
        sources=[],
    )
    geom.sources = make_elastic_wedge_sources(geom, params)
    return geom


def make_elastic_wedge_sources(geom: Geometry, params: ElasticWedgeParams) -> List[SourcePatch]:
    source_boundary_ids = [
        BoundaryID.SURFACE_LEFT,
        BoundaryID.SURFACE_WEDGE,
        BoundaryID.SURFACE_UPPER_PLATE,
        BoundaryID.WEDGE_BACKSTOP,
        BoundaryID.MEGATHRUST,
        BoundaryID.SLAB_EXTENSION,
    ]
    sources: List[SourcePatch] = []
    q = 0
    for b_id in source_boundary_ids:
        B = geom.B[b_id]
        dip_deg = -180.0 / np.pi * np.arctan2(B.bot[1, :] - B.top[1, :],
                                               B.bot[0, :] - B.top[0, :])
        for k in range(B.n_patch):
            m = np.array([
                params.okada_length,
                B.length[k],
                -B.bot[1, k],
                dip_deg[k],
                0.0,
                B.bot[0, k],
                0.0,
            ], dtype=float)
            sources.append(SourcePatch(
                q=q, boundary=b_id, boundary_name=B.name, patch=k, m=m,
                length=float(B.length[k]), dip_deg=float(dip_deg[k]),
                center=B.center[:, k].copy(), top=B.top[:, k].copy(), bot=B.bot[:, k].copy(),
            ))
            q += 1
    return sources
