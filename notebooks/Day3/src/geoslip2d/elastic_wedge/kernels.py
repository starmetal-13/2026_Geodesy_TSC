from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from .geometry import Geometry, BoundaryID
from .params import ElasticWedgeParams
from .okada3d import disloc3d


@dataclass
class TractionDispResponse:
    sig11: np.ndarray
    sig21: np.ndarray
    sig12: np.ndarray
    sig22: np.ndarray
    u11: np.ndarray
    u21: np.ndarray
    u12: np.ndarray
    u22: np.ndarray


@dataclass
class KernelBlock:
    sig11: np.ndarray
    sig21: np.ndarray
    sig12: np.ndarray
    sig22: np.ndarray
    u11_t: np.ndarray
    u21_t: np.ndarray
    u12_t: np.ndarray
    u22_t: np.ndarray
    u11_b: np.ndarray
    u21_b: np.ndarray
    u12_b: np.ndarray
    u22_b: np.ndarray


@dataclass
class Kernels:
    boundary: List[List[KernelBlock]]
    sources: list
    kernel_type: str
    supports_opening: bool
    supports_dip_slip: bool


def make_traction_disp(m, xloc, nu, normvec, dipvec) -> TractionDispResponse:
    """Python port of make_traction_disp.m, preserving the Okada sign convention."""
    m = np.asarray(m, dtype=float).reshape(-1)
    xloc = np.asarray(xloc, dtype=float)
    normvec = np.asarray(normvec, dtype=float)
    dipvec = np.asarray(dipvec, dtype=float)
    if xloc.ndim == 1:
        xloc = xloc.reshape(3, 1)
    if normvec.ndim == 1:
        normvec = normvec.reshape(2, 1)
    if dipvec.ndim == 1:
        dipvec = dipvec.reshape(2, 1)

    U1, _, S1, _ = disloc3d(np.r_[m, 0.0, -1.0, 0.0], xloc, 1.0, nu)
    U2, _, S2, _ = disloc3d(np.r_[m, 0.0, 0.0, 1.0], xloc, 1.0, nu)

    trac1_x = S1[0, :] * normvec[0, :] + S1[2, :] * normvec[1, :]
    trac1_z = S1[2, :] * normvec[0, :] + S1[5, :] * normvec[1, :]
    sig11 = trac1_x * dipvec[0, :] + trac1_z * dipvec[1, :]
    sig21 = trac1_x * normvec[0, :] + trac1_z * normvec[1, :]

    trac2_x = S2[0, :] * normvec[0, :] + S2[2, :] * normvec[1, :]
    trac2_z = S2[2, :] * normvec[0, :] + S2[5, :] * normvec[1, :]
    sig12 = trac2_x * dipvec[0, :] + trac2_z * dipvec[1, :]
    sig22 = trac2_x * normvec[0, :] + trac2_z * normvec[1, :]

    u11 = U1[0, :] * dipvec[0, :] + U1[2, :] * dipvec[1, :]
    u21 = U1[0, :] * normvec[0, :] + U1[2, :] * normvec[1, :]
    u12 = U2[0, :] * dipvec[0, :] + U2[2, :] * dipvec[1, :]
    u22 = U2[0, :] * normvec[0, :] + U2[2, :] * normvec[1, :]
    return TractionDispResponse(sig11, sig21, sig12, sig22, u11, u21, u12, u22)


def _init_kernel_block(nrec: int, nsrc: int) -> KernelBlock:
    z = lambda: np.zeros((nrec, nsrc), dtype=float)
    return KernelBlock(z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z())


def compute_boundary_kernels(geom: Geometry, params: ElasticWedgeParams) -> Kernels:
    nb = geom.num_boundaries
    K = [[_init_kernel_block(geom.B[irec].n_patch, geom.B[isrc].n_patch)
          for isrc in range(nb)] for irec in range(nb)]

    last_boundary = None
    for src in geom.sources:
        isrc = src.boundary
        k = src.patch
        m = src.m
        for irec in range(nb):
            Br = geom.B[irec]
            xloc = np.vstack([Br.center[0, :], np.zeros(Br.n_patch), Br.center[1, :]])
            r = make_traction_disp(m, xloc, params.nu, Br.normvec, Br.dipvec)
            blk = K[irec][isrc]
            blk.sig11[:, k] = r.sig11
            blk.sig21[:, k] = r.sig21
            blk.sig12[:, k] = r.sig12
            blk.sig22[:, k] = r.sig22
            blk.u11_t[:, k] = r.u11
            blk.u21_t[:, k] = r.u21
            blk.u12_t[:, k] = r.u12
            blk.u22_t[:, k] = r.u22
            blk.u11_b[:, k] = r.u11
            blk.u21_b[:, k] = r.u21
            blk.u12_b[:, k] = r.u12
            blk.u22_b[:, k] = r.u22

            if isrc == irec:
                xloct = np.array([Br.center_t[0, k], 0.0, Br.center_t[1, k]])
                xlocb = np.array([Br.center_b[0, k], 0.0, Br.center_b[1, k]])
                rt = make_traction_disp(m, xloct, params.nu, Br.normvec[:, k], Br.dipvec[:, k])
                rb = make_traction_disp(m, xlocb, params.nu, Br.normvec[:, k], Br.dipvec[:, k])
                blk.u11_t[k, k] = rt.u11[0]
                blk.u21_t[k, k] = rt.u21[0]
                blk.u12_t[k, k] = rt.u12[0]
                blk.u22_t[k, k] = rt.u22[0]
                blk.u11_b[k, k] = rb.u11[0]
                blk.u21_b[k, k] = rb.u21[0]
                blk.u12_b[k, k] = rb.u12[0]
                blk.u22_b[k, k] = rb.u22[0]

        if params.verbose and src.boundary != last_boundary:
            print(f"  computing source boundary {int(src.boundary)+1} of {nb}: {src.boundary_name}")
            last_boundary = src.boundary

    return Kernels(K, geom.sources, params.kernel.type, params.kernel.supports_opening, params.kernel.supports_dip_slip)
