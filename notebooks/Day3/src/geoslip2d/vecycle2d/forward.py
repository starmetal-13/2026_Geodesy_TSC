"""
Forward-cycle assembly for VECycle2D.

This module ports the MATLAB forward-cycle logic from:

    make_cycle_vels.m
    geo2dslip_forward_cycle.m / forward_cycle.m

Given a full VECycle2D Green's-function object, this module builds the
interseismic Green's matrices:

    Gx_ss, Gz_ss       : fully relaxed / steady backslip response
    Gx_inter, Gz_inter : total interseismic response including past earthquake cycles

and evaluates velocities for a user-specified backslip / slip-deficit-rate
vector.

Array convention:
    greens.Dx_v[source_index] has shape (n_obs, n_time)
    greens.Vx_v[source_index] has shape (n_obs, n_time)

Internally, the time-dependent lists are stacked into arrays with shape:

    (n_time, n_obs, n_source)

matching the intent of the MATLAB permute/interp1 workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(slots=True)
class ForwardConfig:
    """Forward-cycle settings.

    These defaults match the MATLAB run script values used during refactoring.
    """

    tR_scale: float = 10.0
    teq: float = 500.0
    T: float = 550.0
    locking_depth: float = 40.0
    backslip: Array | None = None


@dataclass(slots=True)
class InterseismicGreens:
    """Interseismic Green's matrices assembled from VECycle2D outputs."""

    visco_times: Array
    times: Array

    Gxv_vel: Array
    Gzv_vel: Array
    Gxv_d: Array
    Gzv_d: Array

    Gx_ss: Array
    Gz_ss: Array

    Gx_cycle: Array
    Gz_cycle: Array

    Gx_inter: Array
    Gz_inter: Array


@dataclass(slots=True)
class ForwardResult:
    """Forward velocities for a given backslip/slip-deficit vector."""

    backslip: Array
    depths: Array

    Vx_total: Array
    Vz_total: Array

    Vx_elastic: Array
    Vz_elastic: Array

    Vx_ss: Array
    Vz_ss: Array

    Vx_cycle: Array
    Vz_cycle: Array

    interseismic_greens: InterseismicGreens


def _get_forward_config(cfg_or_forward: Any | None) -> ForwardConfig:
    """
    Return a ForwardConfig from:
        * None
        * ForwardConfig
        * full config object with .forward
        * dict-like object
    """
    if cfg_or_forward is None:
        return ForwardConfig()

    # Full config object from config.py
    if hasattr(cfg_or_forward, "forward"):
        fwd = getattr(cfg_or_forward, "forward")
    else:
        fwd = cfg_or_forward

    if isinstance(fwd, ForwardConfig):
        return fwd

    if isinstance(fwd, dict):
        return ForwardConfig(
            tR_scale=float(fwd.get("tR_scale", 10.0)),
            teq=float(fwd.get("teq", 500.0)),
            T=float(fwd.get("T", 550.0)),
            locking_depth=float(fwd.get("locking_depth", 40.0)),
            backslip=fwd.get("backslip", None),
        )

    # Dataclass-style ForwardConfig from config.py
    return ForwardConfig(
        tR_scale=float(getattr(fwd, "tR_scale", 10.0)),
        teq=float(getattr(fwd, "teq", 500.0)),
        T=float(getattr(fwd, "T", 550.0)),
        locking_depth=float(getattr(fwd, "locking_depth", 40.0)),
        backslip=getattr(fwd, "backslip", None),
    )


def _stack_time_source(items: list[Array]) -> Array:
    """
    Stack list[source] of (n_obs, n_time) arrays into (n_time, n_obs, n_source).
    """
    arr = np.stack([np.asarray(a, dtype=float) for a in items], axis=2)
    # arr shape: (n_obs, n_time, n_source)
    return np.transpose(arr, (1, 0, 2))


def _interp_time_cube(times: Array, cube: Array, t: float) -> Array:
    """
    Interpolate a time cube with shape (n_time, n_obs, n_source) at scalar t.

    Returns shape (n_obs, n_source).

    numpy.interp is 1D only, so reshape the non-time dimensions into columns.
    """
    times = np.asarray(times, dtype=float).reshape(-1)
    nt, nobs, nsrc = cube.shape
    flat = cube.reshape(nt, nobs * nsrc)

    out = np.empty(nobs * nsrc, dtype=float)
    for j in range(flat.shape[1]):
        out[j] = np.interp(t, times, flat[:, j])

    return out.reshape(nobs, nsrc)


def assemble_interseismic_greens(
    greens: Any,
    forward: Any | None = None,
) -> InterseismicGreens:
    """
    Assemble interseismic Green's matrices.

    This ports MATLAB ``make_cycle_vels.m``.

    Parameters
    ----------
    greens
        VECycleGreens object from build_greens.py.
    forward
        ForwardConfig, full config object, dict, or None.

    Returns
    -------
    InterseismicGreens
        Contains Gx_ss, Gz_ss, Gx_inter, Gz_inter, and intermediate arrays.
    """
    fwd = _get_forward_config(forward)

    visco_times = np.asarray(greens.times, dtype=float).reshape(-1)
    times = visco_times * fwd.tR_scale

    # MATLAB:
    #   for k = 1:length(Dx_v)
    #       Gxv_vel(k,:,:) = Vx_v{k};
    #       ...
    #   end
    #   Gxv_vel = permute(Gxv_vel,[3 2 1]);
    #
    # Resulting intended Python shape:
    #   (time, obs, source)
    Gxv_vel = _stack_time_source(greens.Vx_v)
    Gzv_vel = _stack_time_source(greens.Vz_v)
    Gxv_d = _stack_time_source(greens.Dx_v)
    Gzv_d = _stack_time_source(greens.Dz_v)

    # Steady backslip contribution:
    # MATLAB:
    #   Gx_ss = squeeze(Gxv_d(:,:,end)) + Dx_e;
    #
    # Here Gxv_d[-1,:,:] is shape (obs, source).
    Gx_ss = Gxv_d[-1, :, :] + np.asarray(greens.Dx_e, dtype=float)
    Gz_ss = Gzv_d[-1, :, :] + np.asarray(greens.Dz_e, dtype=float)

    # Scale velocities because Gev=Ged/dt and dt is scaled by tR_scale.
    Gxv_vel = Gxv_vel / fwd.tR_scale
    Gzv_vel = Gzv_vel / fwd.tR_scale

    nobs, nsrc = Gx_ss.shape
    Gx_cycle = np.zeros((nobs, nsrc), dtype=float)
    Gz_cycle = np.zeros((nobs, nsrc), dtype=float)

    t = fwd.teq
    max_time = np.max(times)

    while t < max_time:
        if t < times[0]:
            # MATLAB:
            #   -squeeze(Gxv_vel(1,:,:))*T
            Gx_cycle += -Gxv_vel[0, :, :] * fwd.T
            Gz_cycle += -Gzv_vel[0, :, :] * fwd.T
        else:
            Gx_cycle += -_interp_time_cube(times, Gxv_vel, t) * fwd.T
            Gz_cycle += -_interp_time_cube(times, Gzv_vel, t) * fwd.T

        t += fwd.T

    Gx_inter = Gx_ss + Gx_cycle
    Gz_inter = Gz_ss + Gz_cycle

    return InterseismicGreens(
        visco_times=visco_times,
        times=times,
        Gxv_vel=Gxv_vel,
        Gzv_vel=Gzv_vel,
        Gxv_d=Gxv_d,
        Gzv_d=Gzv_d,
        Gx_ss=Gx_ss,
        Gz_ss=Gz_ss,
        Gx_cycle=Gx_cycle,
        Gz_cycle=Gz_cycle,
        Gx_inter=Gx_inter,
        Gz_inter=Gz_inter,
    )


def default_backslip_from_locking_depth(
    greens: Any,
    locking_depth: float = 40.0,
    value: float = -1.0,
) -> tuple[Array, Array]:
    """
    Make the default MATLAB-style backslip vector.

    MATLAB:
        depths = -Geometry.centers_interface(2,:)';
        ind = depths < Ldepth;
        backslip = zeros(numpatch,1);
        backslip(ind) = -1;
    """
    centers = np.asarray(greens.Geometry.centers_interface, dtype=float)
    depths = -centers[1, :].reshape(-1)

    backslip = np.zeros(depths.size, dtype=float)
    backslip[depths < locking_depth] = value

    return backslip, depths


def forward_cycle(
    greens: Any,
    forward: Any | None = None,
    *,
    backslip: Array | None = None,
) -> ForwardResult:
    """
    Compute forward interseismic velocities for a backslip vector.

    Parameters
    ----------
    greens
        VECycleGreens object.
    forward
        ForwardConfig, full config object, dict, or None.
    backslip
        Optional explicit slip-deficit/backslip vector. If omitted, use
        ``forward.backslip`` if present; otherwise build the default locking-
        depth vector.

    Returns
    -------
    ForwardResult
        Contains total, elastic, steady-state, and cycle velocity components.
    """
    fwd = _get_forward_config(forward)

    inter = assemble_interseismic_greens(greens, fwd)

    if backslip is None:
        backslip = fwd.backslip

    if backslip is None:
        backslip, depths = default_backslip_from_locking_depth(
            greens,
            locking_depth=fwd.locking_depth,
            value=-1.0,
        )
    else:
        backslip = np.asarray(backslip, dtype=float).reshape(-1)
        centers = np.asarray(greens.Geometry.centers_interface, dtype=float)
        depths = -centers[1, :].reshape(-1)

    if backslip.size != inter.Gx_inter.shape[1]:
        raise ValueError(
            f"backslip has length {backslip.size}, but expected "
            f"{inter.Gx_inter.shape[1]} interface patches."
        )

    Vx_total = inter.Gx_inter @ backslip
    Vz_total = inter.Gz_inter @ backslip

    Vx_elastic = np.asarray(greens.Dx_e, dtype=float) @ backslip
    Vz_elastic = np.asarray(greens.Dz_e, dtype=float) @ backslip

    Vx_ss = inter.Gx_ss @ backslip
    Vz_ss = inter.Gz_ss @ backslip

    Vx_cycle = inter.Gx_cycle @ backslip
    Vz_cycle = inter.Gz_cycle @ backslip

    return ForwardResult(
        backslip=backslip,
        depths=depths,
        Vx_total=Vx_total,
        Vz_total=Vz_total,
        Vx_elastic=Vx_elastic,
        Vz_elastic=Vz_elastic,
        Vx_ss=Vx_ss,
        Vz_ss=Vz_ss,
        Vx_cycle=Vx_cycle,
        Vz_cycle=Vz_cycle,
        interseismic_greens=inter,
    )


def forward_summary(result: ForwardResult) -> dict[str, tuple[int, ...]]:
    """Return compact output shapes for a forward result."""
    return {
        "backslip": result.backslip.shape,
        "depths": result.depths.shape,
        "Vx_total": result.Vx_total.shape,
        "Vz_total": result.Vz_total.shape,
        "Vx_elastic": result.Vx_elastic.shape,
        "Vz_elastic": result.Vz_elastic.shape,
        "Vx_ss": result.Vx_ss.shape,
        "Vz_ss": result.Vz_ss.shape,
        "Vx_cycle": result.Vx_cycle.shape,
        "Vz_cycle": result.Vz_cycle.shape,
        "Gx_inter": result.interseismic_greens.Gx_inter.shape,
        "Gz_inter": result.interseismic_greens.Gz_inter.shape,
    }


__all__ = [
    "ForwardConfig",
    "InterseismicGreens",
    "ForwardResult",
    "assemble_interseismic_greens",
    "default_backslip_from_locking_depth",
    "forward_cycle",
    "forward_summary",
]
