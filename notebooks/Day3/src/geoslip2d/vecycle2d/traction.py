"""
Traction/displacement projection utilities for VECycle2D.

This module ports the MATLAB function:

    make_traction_disp.m

It wraps the Okada half-space dislocation kernel `disloc3d` and projects
the resulting displacement and stress fields onto the local dip and normal
directions of receiver patches.

The output order intentionally matches MATLAB:

    sig11, sig21, sig12, sig22, u11, u21, u12, u22

where:
    sig11 : dip-direction traction due to unit dip slip
    sig21 : normal-direction traction due to unit dip slip
    sig12 : dip-direction traction due to unit opening
    sig22 : normal-direction traction due to unit opening

    u11   : dip-direction displacement due to unit dip slip
    u21   : normal-direction displacement due to unit dip slip
    u12   : dip-direction displacement due to unit opening
    u22   : normal-direction displacement due to unit opening

Array conventions follow the MATLAB code:
    m       : length-7 Okada base source vector [L, W, D, dip, strike, Xc, Yc]
    xloc    : shape (3, N)
    normvec : shape (2, N) or shape (2,)
    dipvec  : shape (2, N) or shape (2,)

Coordinates are assumed to be:
    xloc[0, :] = horizontal x
    xloc[1, :] = horizontal y, normally zeros for 2D profiles
    xloc[2, :] = vertical z, negative downward in the Okada convention

The stress component order returned by okada3d.disloc3d is assumed to be:
    S[0, :] = Sxx
    S[1, :] = Sxy
    S[2, :] = Szx
    S[3, :] = Syy
    S[4, :] = Syz
    S[5, :] = Szz

This matches the MATLAB use of S(1,:), S(3,:), and S(6,:) in the
x-z plane traction projection.
"""

from __future__ import annotations

import numpy as np

from .okada3d import disloc3d


def _as_column_source(m: np.ndarray) -> np.ndarray:
    """Return the base Okada source vector as a 1D length-7 array."""
    m = np.asarray(m, dtype=float).reshape(-1)
    if m.size != 7:
        raise ValueError(
            "m must contain 7 base Okada parameters: "
            "[L, W, D, dip, strike, Xc, Yc]"
        )
    return m


def _as_xloc(xloc: np.ndarray) -> np.ndarray:
    """Return receiver coordinates with shape (3, N)."""
    xloc = np.asarray(xloc, dtype=float)
    if xloc.ndim == 1:
        if xloc.size != 3:
            raise ValueError("1D xloc must contain exactly 3 values.")
        xloc = xloc.reshape(3, 1)
    if xloc.ndim != 2 or xloc.shape[0] != 3:
        raise ValueError("xloc must have shape (3, N).")
    return xloc


def _as_vec2(vec: np.ndarray, n: int, name: str) -> np.ndarray:
    """
    Return a 2-component vector array with shape (2, N).

    Accepts either:
        shape (2,)
        shape (2, 1)
        shape (2, N)
    """
    vec = np.asarray(vec, dtype=float)

    if vec.ndim == 1:
        if vec.size != 2:
            raise ValueError(f"{name} must have 2 components.")
        vec = vec.reshape(2, 1)

    if vec.ndim != 2 or vec.shape[0] != 2:
        raise ValueError(f"{name} must have shape (2,), (2,1), or (2,N).")

    if vec.shape[1] == 1 and n != 1:
        vec = np.repeat(vec, n, axis=1)

    if vec.shape[1] != n:
        raise ValueError(
            f"{name} has {vec.shape[1]} columns, but xloc has {n} receiver points."
        )

    return vec


def make_traction_disp(
    m: np.ndarray,
    xloc: np.ndarray,
    nu: float,
    normvec: np.ndarray,
    dipvec: np.ndarray,
    shear_m: float = 1.0,
):
    """
    Compute local traction and displacement components for one source.

    Parameters
    ----------
    m : array_like, shape (7,)
        Base Okada source vector:
            [length, width, depth, dip_deg, strike_deg, x_center, y_center]
        This is the same vector passed to MATLAB before appending slip
        components.
    xloc : array_like, shape (3, N)
        Receiver coordinates.
    nu : float
        Poisson's ratio.
    normvec : array_like, shape (2, N) or (2,)
        Unit normal vectors at receiver points in the x-z plane.
    dipvec : array_like, shape (2, N) or (2,)
        Unit dip/tangential vectors at receiver points in the x-z plane.
    shear_m : float, optional
        Shear modulus passed to the Okada kernel. The MATLAB code uses 1.

    Returns
    -------
    sig11, sig21, sig12, sig22, u11, u21, u12, u22 : np.ndarray
        Arrays of length N, matching MATLAB make_traction_disp.m.
    """
    m = _as_column_source(m)
    xloc = _as_xloc(xloc)
    nrec = xloc.shape[1]
    normvec = _as_vec2(normvec, nrec, "normvec")
    dipvec = _as_vec2(dipvec, nrec, "dipvec")

    # MATLAB:
    #   [U1,D,S1] = disloc3d([m;0;-1;0], xloc, 1, nu);
    #   [U2,D,S2] = disloc3d([m;0; 0;1], xloc, 1, nu);
    #
    # In the Python kernel, m must be length 10:
    #   [L, W, D, dip, strike, Xc, Yc, strike_slip, dip_slip, tensile]
    source_dipslip = np.r_[m, 0.0, -1.0, 0.0]
    source_opening = np.r_[m, 0.0,  0.0, 1.0]

    U1, _D1, S1, _flag1 = disloc3d(
        source_dipslip,
        xloc,
        shear_m=shear_m,
        poisson_ratio=nu,
    )
    U2, _D2, S2, _flag2 = disloc3d(
        source_opening,
        xloc,
        shear_m=shear_m,
        poisson_ratio=nu,
    )

    # Traction vector in the x-z plane:
    #   T_x = Sxx*n_x + Sxz*n_z
    #   T_z = Szx*n_x + Szz*n_z
    #
    # The MATLAB code uses:
    #   S(1,:) = Sxx
    #   S(3,:) = Szx/Sxz
    #   S(6,:) = Szz
    trac1_x = S1[0, :] * normvec[0, :] + S1[2, :] * normvec[1, :]
    trac1_z = S1[2, :] * normvec[0, :] + S1[5, :] * normvec[1, :]

    sig11 = trac1_x * dipvec[0, :] + trac1_z * dipvec[1, :]
    sig21 = trac1_x * normvec[0, :] + trac1_z * normvec[1, :]

    trac2_x = S2[0, :] * normvec[0, :] + S2[2, :] * normvec[1, :]
    trac2_z = S2[2, :] * normvec[0, :] + S2[5, :] * normvec[1, :]

    sig12 = trac2_x * dipvec[0, :] + trac2_z * dipvec[1, :]
    sig22 = trac2_x * normvec[0, :] + trac2_z * normvec[1, :]

    # Displacement projections in local dip and normal directions.
    u11 = U1[0, :] * dipvec[0, :] + U1[2, :] * dipvec[1, :]
    u21 = U1[0, :] * normvec[0, :] + U1[2, :] * normvec[1, :]

    u12 = U2[0, :] * dipvec[0, :] + U2[2, :] * dipvec[1, :]
    u22 = U2[0, :] * normvec[0, :] + U2[2, :] * normvec[1, :]

    return sig11, sig21, sig12, sig22, u11, u21, u12, u22


__all__ = ["make_traction_disp"]
