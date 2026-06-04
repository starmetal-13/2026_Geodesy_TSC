from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as splinalg

from .geometry import Geometry
from .assemble import System
from .params import ElasticWedgeParams, BoundaryID


@dataclass
class Greens:
    xpos: np.ndarray
    Gx: np.ndarray
    Gz: np.ndarray
    Gtau: np.ndarray
    patch_slips_all: np.ndarray
    rhs: np.ndarray
    topx_interface: np.ndarray
    botx_interface: np.ndarray
    topz_interface: np.ndarray
    botz_interface: np.ndarray
    centers_interface: np.ndarray
    Gx_body: np.ndarray | None = None
    Gz_body: np.ndarray | None = None
    Gexx_body: np.ndarray | None = None
    Gexz_body: np.ndarray | None = None
    Gezz_body: np.ndarray | None = None
    xloc_body: np.ndarray | None = None


def solve_interface_greens(system: System, geom: Geometry, params: ElasticWedgeParams) -> Greens:
    B = geom.B
    numpatch = geom.n_patch
    numdisp = system.Gd11.shape[0]
    num_backstop = B[BoundaryID.WEDGE_BACKSTOP].n_patch
    num_slip_patches = B[BoundaryID.MEGATHRUST].n_patch + B[BoundaryID.SLAB_EXTENSION].n_patch

    xpos = np.concatenate([
        B[BoundaryID.SURFACE_LEFT].center[0, :],
        B[BoundaryID.SURFACE_WEDGE].center[0, :],
        B[BoundaryID.SURFACE_UPPER_PLATE].center[0, :],
    ])[:, None]

    D = np.zeros((system.G.shape[0], num_slip_patches), dtype=float)
    for k in range(num_slip_patches):
        ds1 = np.zeros(numdisp, dtype=float)
        ds2 = np.zeros(numdisp, dtype=float)
        ds1[num_backstop + k] = 1.0
        D[:, k] = np.concatenate([np.zeros(numpatch * 2), ds1, ds2])

    if sparse.issparse(system.G):
        lu = splinalg.splu(system.G.tocsc())
        patch_slips_all = lu.solve(D)
    else:
        patch_slips_all = linalg.solve(system.G, D, assume_a="gen")

    Gx = system.surfaceUxMat @ patch_slips_all
    Gz = system.surfaceUzMat @ patch_slips_all
    Gtau = system.interfaceTauMat @ patch_slips_all
    if sparse.issparse(Gx):
        Gx = Gx.toarray()
    if sparse.issparse(Gz):
        Gz = Gz.toarray()
    if sparse.issparse(Gtau):
        Gtau = Gtau.toarray()

    topx_interface = np.concatenate([B[BoundaryID.MEGATHRUST].top[0, :], B[BoundaryID.SLAB_EXTENSION].top[0, :]])
    botx_interface = np.concatenate([B[BoundaryID.MEGATHRUST].bot[0, :], B[BoundaryID.SLAB_EXTENSION].bot[0, :]])
    topz_interface = np.concatenate([B[BoundaryID.MEGATHRUST].top[1, :], B[BoundaryID.SLAB_EXTENSION].top[1, :]]) + params.shift
    botz_interface = np.concatenate([B[BoundaryID.MEGATHRUST].bot[1, :], B[BoundaryID.SLAB_EXTENSION].bot[1, :]]) + params.shift
    centers_interface = np.vstack([(topx_interface + botx_interface) / 2, (topz_interface + botz_interface) / 2])

    if params.verbose:
        print(f"  solved {num_slip_patches} interface RHS vectors at once")

    return Greens(
        xpos=xpos,
        Gx=np.asarray(Gx),
        Gz=np.asarray(Gz),
        Gtau=np.asarray(Gtau),
        patch_slips_all=patch_slips_all,
        rhs=D,
        topx_interface=topx_interface,
        botx_interface=botx_interface,
        topz_interface=topz_interface,
        botz_interface=botz_interface,
        centers_interface=centers_interface,
        Gx_body=np.array([]),
        Gz_body=np.array([]),
        Gexx_body=np.array([]),
        Gexz_body=np.array([]),
        Gezz_body=np.array([]),
        xloc_body=np.array([]),
    )
