"""
Single-source viscoelastic earthquake-cycle response for VECycle2D.

This module ports the MATLAB Phase-6 function:

    vec_build_cycle_for_source.m

It computes the normalized postseismic relaxation response to unit coseismic
slip on one interface source patch.

Python indexing convention:
    source_index is zero-based.

MATLAB equivalent:
    source_index = sourcenum - 1

The numerical ordering of the unknown vector is intentionally preserved from
MATLAB because downstream matrices depend on this ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(slots=True)
class SourceCycleResponse:
    """Per-source VECycle2D cycle response.

    Field names mirror MATLAB ``vec_build_cycle_for_source.m`` outputs.
    """

    times: Array
    Delt: Array

    Dx_e: Array
    Dx_v: Array
    Vx_v: Array

    Dz_e: Array
    Dz_v: Array
    Vz_v: Array

    tau_e: Array
    tau_v: Array
    tau_rate_v: Array

    sig_e: Array
    sig_v: Array
    sig_rate_v: Array


def _as_array(x: Any, name: str) -> Array:
    """Fetch and convert a model field to a NumPy array."""
    arr = np.asarray(getattr(x, name), dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1)
    return arr


def build_cycle_for_source(
    model: Any,
    source_index: int,
    *,
    n_time_inc: int = 150,
    tR: float = 1.0,
    co: float = 1.0,
    tlong: float = 350.0,
    n_corrector: int = 10,
    return_slip_history: bool = False,
) -> SourceCycleResponse:
    """
    Build the relaxation response for one unit interface source.

    Parameters
    ----------
    model
        ``WedgeLinearSystem`` from ``compile_greens.py`` or any object with
        the same fields.
    source_index
        Zero-based interface source index.
    n_time_inc, tR, co, tlong, n_corrector
        Hardwired legacy cycle-kernel constants. These are keyword arguments
        mainly for testing/debugging; the default values match MATLAB.
    return_slip_history
        Reserved for future debugging. The default is False, matching the
        performance-oriented MATLAB refactor that does not return internal
        slip histories.

    Returns
    -------
    SourceCycleResponse
        Per-source surface displacement/velocity and interface traction
        response.
    """
    # Local copies reduce repeated attribute lookup and mirror MATLAB.
    G_part1 = _as_array(model, "G_part1")
    G_part2 = _as_array(model, "G_part2")
    Gs11_1 = _as_array(model, "Gs11_1")
    Gs12_1 = _as_array(model, "Gs12_1")
    num_v = int(getattr(model, "num_v"))
    xpos = _as_array(model, "xpos")
    num_interface = int(getattr(model, "num_interface"))

    if source_index < 0 or source_index >= num_interface:
        raise IndexError(
            f"source_index must be between 0 and {num_interface - 1}; "
            f"got {source_index}"
        )

    # Time vector, matching MATLAB:
    #   ts = logspace(-3, log10(tlong), Ntimeinc+1);
    #   Delt = diff(ts);
    #   times = cumsum(Delt);
    ts = np.logspace(-3.0, np.log10(tlong), n_time_inc + 1)
    Delt = np.diff(ts)
    times = np.cumsum(Delt)
    expfac = np.exp(-Delt / tR)

    # ------------------------------------------------------------------
    # Remove interface patches that are not the active source.
    # MATLAB:
    #   ind_i_inactive = true(num_interface,1);
    #   ind_i_inactive(sourcenum) = false;
    #
    # Python:
    #   source_index is zero-based.
    # ------------------------------------------------------------------
    ind_i_inactive = np.ones(num_interface, dtype=bool)
    ind_i_inactive[source_index] = False

    # MATLAB:
    #   first_interface_col = size(Gs11_1,2) - num_interface + 1;
    #   ind_inactive(first_interface_col:size(Gs11_1,2)) = ind_i_inactive;
    #
    # Python columns are zero-based, stop is exclusive.
    ind_inactive = np.zeros(G_part2.shape[1], dtype=bool)
    first_interface_col = Gs11_1.shape[1] - num_interface
    ind_inactive[first_interface_col:Gs11_1.shape[1]] = ind_i_inactive

    G_part1_active = G_part1[:, ~ind_inactive]
    G_part2_active = G_part2[:, ~ind_inactive]

    # MATLAB's local Gi is an interface-only block. In this Python port,
    # model.Gi is the full interface-equation matrix. For the one-source
    # solve, only the active interface row count matters here because the
    # coseismic patch is excluded from the non-slip interface equations.
    n_interface_active = int(np.sum(~ind_i_inactive))
    Gi_active = np.zeros((n_interface_active, n_interface_active), dtype=float)

    numpatch_active = Gs11_1.shape[1] - int(np.sum(ind_inactive))

    # The one remaining interface patch slips coseismically.
    ind_i_co = np.ones(1, dtype=bool)
    ind_co = np.zeros(G_part2_active.shape[1], dtype=bool)
    ind_co[numpatch_active - ind_i_co.size : numpatch_active] = ind_i_co

    ind_i_noco = ~ind_i_co

    # These Gi blocks have zero rows because the active interface patch is
    # removed from the non-coseismic interface equations. MATLAB allows
    # vertical concatenation with a 0 x 0 empty matrix, but NumPy requires
    # matching column counts, so we replace empty middle blocks with 0 x n.
    G_co_top = G_part1_active[:, ind_co]
    G_co_mid = Gi_active[ind_i_noco, :][:, ind_i_co]
    G_co_bot = G_part2_active[:, ind_co]

    if G_co_mid.size == 0:
        G_co_mid = np.zeros((0, G_co_top.shape[1]), dtype=float)

    G_co = np.vstack((G_co_top, G_co_mid, G_co_bot))

    G_noco_top = G_part1_active[:, ~ind_co]
    G_noco_mid = Gi_active[ind_i_noco, :][:, ind_i_noco]
    G_noco_bot = G_part2_active[:, ~ind_co]

    if G_noco_mid.size == 0:
        G_noco_mid = np.zeros((0, G_noco_top.shape[1]), dtype=float)

    G_noco = np.vstack((G_noco_top, G_noco_mid, G_noco_bot))

    # Match MATLAB performance behavior:
    #   invG = inv(G_noco)
    invG = np.linalg.inv(G_noco)

    coslip = -co
    bc_co = G_co.reshape(G_co.shape[0]) * coslip
    slip0 = invG @ (-bc_co)

    num_unknown = slip0.size
    SLIP = np.zeros((num_unknown, n_time_inc + 1), dtype=float)
    SLIP[:, 0] = slip0

    num_e = int(numpatch_active - np.sum(ind_i_co))
    num_e2 = Gs12_1.shape[1]
    Nstress = G_part1_active.shape[0] - int(np.sum(ind_i_co)) + Gi_active.shape[0]

    # MATLAB i1/i2/i3/i4 are one-based inclusive ranges.
    # Python slices are zero-based, stop-exclusive.
    i1 = slice(0, num_e)
    i2 = slice(num_e, num_e + num_v)
    i3 = slice(num_e + num_v, num_e + num_v + num_e2)
    i4 = slice(num_e + num_e2 + num_v, num_e + num_e2 + 2 * num_v)

    stress_rows = slice(0, Nstress)
    disp_rows = slice(Nstress, G_noco.shape[0])

    G_stress_i1 = G_noco[stress_rows, i1]
    G_stress_i2 = G_noco[stress_rows, i2]
    G_stress_i3 = G_noco[stress_rows, i3]
    G_stress_i4 = G_noco[stress_rows, i4]
    G_disp = G_noco[disp_rows, :]

    total_stress1_v = np.zeros(Nstress, dtype=float)
    total_stress2_v = np.zeros(Nstress, dtype=float)
    total_stress1_e = np.zeros(Nstress, dtype=float)
    total_stress2_e = np.zeros(Nstress, dtype=float)
    total_disp = np.zeros(G_disp.shape[0], dtype=float)

    for loopt in range(n_time_inc):
        slip_prev = SLIP[:, loopt]

        incremental_stress1_e = G_stress_i1 @ slip_prev[i1]
        incremental_stress1_v = G_stress_i2 @ slip_prev[i2]
        incremental_stress2_e = G_stress_i3 @ slip_prev[i3]
        incremental_stress2_v = G_stress_i4 @ slip_prev[i4]

        total_stress1_v = (total_stress1_v + incremental_stress1_v) * expfac[loopt]
        total_stress2_v = (total_stress2_v + incremental_stress2_v) * expfac[loopt]
        total_stress1_e = total_stress1_e + incremental_stress1_e
        total_stress2_e = total_stress2_e + incremental_stress2_e
        total_disp = total_disp + G_disp @ slip_prev

        bc_relax1 = total_stress1_e + total_stress1_v + total_stress2_e + total_stress2_v
        bc = bc_co + np.r_[bc_relax1, total_disp]

        slip = np.zeros(num_unknown, dtype=float)
        for _ in range(n_corrector):
            slip_inc = invG @ (-bc)
            bc = bc + G_noco @ slip_inc
            slip = slip + slip_inc

        SLIP[:, loopt + 1] = slip

    postslip = np.cumsum(SLIP, axis=1)
    slipco = SLIP[:, 0]

    # ------------------------------------------------------------------
    # Surface displacement/velocity response
    # ------------------------------------------------------------------
    Gd11_6 = _as_array(model, "Gd11_6")
    Gd12_6 = _as_array(model, "Gd12_6")
    Gd21_6 = _as_array(model, "Gd21_6")
    Gd22_6 = _as_array(model, "Gd22_6")

    Gd11_7 = _as_array(model, "Gd11_7")
    Gd12_7 = _as_array(model, "Gd12_7")
    Gd21_7 = _as_array(model, "Gd21_7")
    Gd22_7 = _as_array(model, "Gd22_7")

    Z6 = np.zeros((Gd11_6.shape[0], num_v), dtype=float)
    Z7 = np.zeros((Gd11_7.shape[0], num_v), dtype=float)

    Gsurf_hz = np.vstack(
        (
            np.hstack((Gd11_7, Z7, Gd12_7, Z7)),
            np.hstack((Gd11_6, Z6, Gd12_6, Z6)),
        )
    )
    Gsurf_vert = np.vstack(
        (
            np.hstack((Gd21_7, Z7, Gd22_7, Z7)),
            np.hstack((Gd21_6, Z6, Gd22_6, Z6)),
        )
    )

    Gsurf_hz = Gsurf_hz[:, ~ind_inactive]
    Gsurf_vert = Gsurf_vert[:, ~ind_inactive]

    Gsurf_hz_co = Gsurf_hz[:, ind_co]
    Gsurf_vert_co = Gsurf_vert[:, ind_co]
    Gsurf_hz_noco = Gsurf_hz[:, ~ind_co]
    Gsurf_vert_noco = Gsurf_vert[:, ~ind_co]

    Gsurf_hz_co_vec = Gsurf_hz_co.reshape(Gsurf_hz_co.shape[0])
    Gsurf_vert_co_vec = Gsurf_vert_co.reshape(Gsurf_vert_co.shape[0])

    Dx_co = Gsurf_hz_co_vec * coslip + Gsurf_hz_noco @ slipco
    Dz_co = Gsurf_vert_co_vec * coslip + Gsurf_vert_noco @ slipco

    nobs = xpos.size
    Dx_post = np.zeros((nobs, n_time_inc), dtype=float)
    Dz_post = np.zeros((nobs, n_time_inc), dtype=float)
    Vx_post = np.zeros((nobs, n_time_inc), dtype=float)
    Vz_post = np.zeros((nobs, n_time_inc), dtype=float)

    for k in range(n_time_inc):
        Dx_post[:, k] = (
            Gsurf_hz_co_vec * coslip
            + Gsurf_hz_noco @ postslip[:, k + 1]
            - Dx_co
        )
        Dz_post[:, k] = (
            Gsurf_vert_co_vec * coslip
            + Gsurf_vert_noco @ postslip[:, k + 1]
            - Dz_co
        )
        Vx_post[:, k] = (Gsurf_hz_noco @ SLIP[:, k + 1]) / Delt[k]
        Vz_post[:, k] = (Gsurf_vert_noco @ SLIP[:, k + 1]) / Delt[k]

    # ------------------------------------------------------------------
    # Interface tractions
    # ------------------------------------------------------------------
    Gs11_i = _as_array(model, "Gs11_i")
    Gs12_i = _as_array(model, "Gs12_i")
    Gs21_i = _as_array(model, "Gs21_i")
    Gs22_i = _as_array(model, "Gs22_i")

    Zi = np.zeros((Gs11_i.shape[0], num_v), dtype=float)

    Gi_s1 = np.hstack((Gs11_i, Zi, Gs12_i, Zi))
    Gi_s2 = np.hstack((Gs21_i, Zi, Gs22_i, Zi))

    Gi_s1 = Gi_s1[:, ~ind_inactive]
    Gi_s2 = Gi_s2[:, ~ind_inactive]

    Gi_s1_co = Gi_s1[:, ind_co]
    Gi_s2_co = Gi_s2[:, ind_co]
    Gi_s1_noco = Gi_s1[:, ~ind_co]
    Gi_s2_noco = Gi_s2[:, ~ind_co]

    Gi_s1_co_vec = Gi_s1_co.reshape(Gi_s1_co.shape[0])
    Gi_s2_co_vec = Gi_s2_co.reshape(Gi_s2_co.shape[0])

    S1_co = Gi_s1_co_vec * coslip + Gi_s1_noco @ slipco
    S2_co = Gi_s2_co_vec * coslip + Gi_s2_noco @ slipco

    S1_post = np.zeros((S1_co.size, n_time_inc), dtype=float)
    S2_post = np.zeros((S2_co.size, n_time_inc), dtype=float)
    Srate1_post = np.zeros((S1_co.size, n_time_inc), dtype=float)
    Srate2_post = np.zeros((S2_co.size, n_time_inc), dtype=float)

    for k in range(n_time_inc):
        S1_post[:, k] = (
            Gi_s1_co_vec * coslip
            + Gi_s1_noco @ postslip[:, k + 1]
            - S1_co
        )
        S2_post[:, k] = (
            Gi_s2_co_vec * coslip
            + Gi_s2_noco @ postslip[:, k + 1]
            - S2_co
        )
        Srate1_post[:, k] = (Gi_s1_noco @ SLIP[:, k + 1]) / Delt[k]
        Srate2_post[:, k] = (Gi_s2_noco @ SLIP[:, k + 1]) / Delt[k]

    # Internal histories are deliberately not returned by default. If needed,
    # this function can later be extended with a debug dataclass.
    _ = return_slip_history

    return SourceCycleResponse(
        times=times,
        Delt=Delt,
        Dx_e=Dx_co,
        Dx_v=Dx_post,
        Vx_v=Vx_post,
        Dz_e=Dz_co,
        Dz_v=Dz_post,
        Vz_v=Vz_post,
        tau_e=S1_co,
        tau_v=S1_post,
        tau_rate_v=Srate1_post,
        sig_e=S2_co,
        sig_v=S2_post,
        sig_rate_v=Srate2_post,
    )


def cycle_response_summary(src: SourceCycleResponse) -> dict[str, tuple[int, ...]]:
    """Return output array shapes for one source response."""
    return {
        "times": src.times.shape,
        "Delt": src.Delt.shape,
        "Dx_e": src.Dx_e.shape,
        "Dx_v": src.Dx_v.shape,
        "Vx_v": src.Vx_v.shape,
        "Dz_e": src.Dz_e.shape,
        "Dz_v": src.Dz_v.shape,
        "Vz_v": src.Vz_v.shape,
        "tau_e": src.tau_e.shape,
        "tau_v": src.tau_v.shape,
        "tau_rate_v": src.tau_rate_v.shape,
        "sig_e": src.sig_e.shape,
        "sig_v": src.sig_v.shape,
        "sig_rate_v": src.sig_rate_v.shape,
    }


__all__ = [
    "SourceCycleResponse",
    "build_cycle_for_source",
    "cycle_response_summary",
]
