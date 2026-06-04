"""Homogeneous elastic half-space Green's functions for GeoSlip2D.

This module ports the MATLAB workflow built from ``make_interface_buildG.m``,
``Make_dispG_edge.m``, and ``EdgeDisp_finite.m`` into package functions.

Conventions
-----------
- x is horizontal distance from the trench, in km.
- z/depth is positive downward, in km.
- slip is unit slip on each finite edge dislocation unless changed in config.
- the raw finite-edge-dislocation displacement response is multiplied by
  ``HomogeneousConfig.output_sign`` before being stored in ``Ghor`` and
  ``Gvert``.
- output ``Ghor`` and ``Gvert`` have shape ``(n_obs, n_patch)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

from .geometry import InterfaceGeometry
from .greens import Greens2D


@dataclass(slots=True)
class HomogeneousConfig:
    """Settings for homogeneous finite-edge-dislocation Green's functions.

    Parameters
    ----------
    slip
        Slip applied to each patch when computing a Green's-function column.
    updip_depth_epsilon
        Small positive depth offset for patches whose updip edge is at the free
        surface.  This reproduces the MATLAB ``-topz_interface + .01`` behavior.
    length_override
        Optional constant down-dip dislocation length.  Leave as ``None`` to use
        each patch's geometric length.  To exactly mimic the legacy MATLAB
        driver, set this equal to the original ``pL`` value.
    output_sign
        Multiplier applied to the raw finite-edge-dislocation displacement
        response before storing ``Ghor`` and ``Gvert``.  The default ``-1``
        matches the GeoSlip2D slip convention used by the inversion examples.
    sign_convention
        Human-readable sign convention stored in the output metadata.
    metadata
        Optional extra metadata copied to the returned ``Greens2D`` object.
    """

    slip: float = 1.0
    updip_depth_epsilon: float = 0.01
    length_override: Optional[float] = None
    output_sign: float = -1.0
    sign_convention: str = "geoslip2d_forward_slip_positive"
    progress: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def edge_disp_finite(
    x_obs: ArrayLike,
    *,
    updip_depth: float,
    updip_x: float,
    length: float,
    dip_degrees: float,
    slip: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Surface displacement from a finite dipping edge dislocation.

    This is a vectorized Python port of ``EdgeDisp_finite.m``.  The original
    arguments ``disloc = [depth, x_updip, length, dip_deg, slip]`` are exposed
    as named keyword arguments for clarity.

    Returns
    -------
    u, v, exx
        Horizontal displacement, vertical displacement, and horizontal strain.
    """
    x_obs = np.asarray(x_obs, dtype=float).reshape(-1)
    depth = float(updip_depth)
    x0 = float(updip_x)
    length = float(length)
    dip = np.radians(float(dip_degrees))
    slip = float(slip)

    if depth <= 0:
        raise ValueError("updip_depth must be positive. Use updip_depth_epsilon for surface-breaking patches.")
    if length <= 0:
        raise ValueError("length must be positive.")

    d1 = depth
    d2 = depth + length * np.sin(dip)
    if d2 <= 0:
        raise ValueError("downdip depth must be positive.")

    s_v = slip * np.sin(dip)
    s_h = slip * np.cos(dip)

    x = x_obs - x0
    zeta_1 = x / d1
    zeta_2 = (x - length * np.cos(dip)) / d2
    denom_1 = 1.0 + zeta_1**2
    denom_2 = 1.0 + zeta_2**2

    v = (
        s_v * np.arctan(zeta_1)
        + (s_h + s_v * zeta_1) / denom_1
        - s_v * np.arctan(zeta_2)
        - (s_h + s_v * zeta_2) / denom_2
    ) / np.pi

    u = -(
        s_h * np.arctan(zeta_1)
        + (s_v - s_h * zeta_1) / denom_1
        - s_h * np.arctan(zeta_2)
        - (s_v - s_h * zeta_2) / denom_2
    ) / np.pi

    exx = (d2 / d1) * (s_v * zeta_1 - s_h * zeta_1**2) / denom_1**2
    exx -= (s_v * zeta_2 - s_h * zeta_2**2) / denom_2**2
    exx = 2.0 * exx / np.pi

    return u, v, exx


def build_homogeneous_greens(
    interface: InterfaceGeometry,
    xobs: ArrayLike,
    config: Optional[HomogeneousConfig] = None,
) -> Greens2D:
    """Build homogeneous finite-edge-dislocation Green's functions.

    Parameters
    ----------
    interface
        Canonical GeoSlip2D interface geometry, with positive-down depths.
    xobs
        Observation positions along the 1-D profile in km.
    config
        Homogeneous Green's-function settings.

    Returns
    -------
    Greens2D
        Canonical GeoSlip2D Green's-function object.
    """
    if config is None:
        config = HomogeneousConfig()

    xobs = np.asarray(xobs, dtype=float).reshape(-1)
    if xobs.size == 0 or np.any(~np.isfinite(xobs)):
        raise ValueError("xobs must be a non-empty finite vector.")

    nobs = xobs.size
    npatch = interface.n_patch
    Ghor = np.zeros((nobs, npatch), dtype=float)
    Gvert = np.zeros((nobs, npatch), dtype=float)

    for k in range(npatch):
        length = float(config.length_override) if config.length_override is not None else float(interface.patch_length[k])
        updip_depth = float(interface.topz[k]) + float(config.updip_depth_epsilon)
        u, v, _ = edge_disp_finite(
            xobs,
            updip_depth=updip_depth,
            updip_x=float(interface.topx[k]),
            length=length,
            dip_degrees=float(interface.dip[k]),
            slip=float(config.slip),
        )
        Ghor[:, k] = float(config.output_sign) * u
        Gvert[:, k] = float(config.output_sign) * v
        if config.progress:
            print(f"completed {k + 1} of {npatch} patches")

    metadata: dict[str, Any] = {
        "backend": "homogeneous_finite_edge_dislocation",
        "matlab_sources": ["EdgeDisp_finite.m", "Make_dispG_edge.m", "make_interface_buildG.m"],
        "slip": config.slip,
        "updip_depth_epsilon": config.updip_depth_epsilon,
        "length_override": config.length_override,
        "output_sign": config.output_sign,
        "progress": config.progress,
    }
    metadata.update(config.metadata)

    return Greens2D(
        Ghor=Ghor,
        Gvert=Gvert,
        xobs=xobs,
        interface=interface,
        source_type="homogeneous",
        units="displacement_per_unit_slip",
        sign_convention=config.sign_convention,
        metadata=metadata,
    )
