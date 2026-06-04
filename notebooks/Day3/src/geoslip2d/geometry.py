"""Interface geometry utilities for GeoSlip2D.

The package-wide convention is:
- x is horizontal distance from the trench, in km.
- z is depth, positive downward, in km.
- each 1-D interface patch is represented by its top and bottom edge points.

This module intentionally avoids any Green's-function calculation.  Green's
engines should consume :class:`InterfaceGeometry` and return a canonical
``Greens2D`` object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike


@dataclass(slots=True)
class InterfaceGeometry:
    """Canonical 1-D subduction-interface geometry.

    Parameters
    ----------
    topx, topz, botx, botz
        Patch endpoint coordinates in km.  z is positive downward.
    centers
        Optional ``(n_patch, 2)`` array of patch center coordinates ``[x, z]``.
    patch_length
        Optional length of each patch in km.
    dip
        Optional dip of each patch in degrees, positive downward.
    metadata
        Dictionary for construction settings, original file names, etc.
    """

    topx: np.ndarray
    topz: np.ndarray
    botx: np.ndarray
    botz: np.ndarray
    centers: Optional[np.ndarray] = None
    patch_length: Optional[np.ndarray] = None
    dip: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.topx = _as_1d(self.topx, "topx")
        self.topz = _as_1d(self.topz, "topz")
        self.botx = _as_1d(self.botx, "botx")
        self.botz = _as_1d(self.botz, "botz")
        n = self.topx.size
        for name in ("topz", "botx", "botz"):
            if getattr(self, name).size != n:
                raise ValueError("topx, topz, botx, and botz must have the same length.")
        if n == 0:
            raise ValueError("InterfaceGeometry must contain at least one patch.")

        if self.centers is None:
            self.centers = np.column_stack(((self.topx + self.botx) / 2.0, (self.topz + self.botz) / 2.0))
        else:
            self.centers = np.asarray(self.centers, dtype=float)
            if self.centers.shape != (n, 2):
                raise ValueError("centers must have shape (n_patch, 2).")

        if self.patch_length is None:
            self.patch_length = np.hypot(self.botx - self.topx, self.botz - self.topz)
        else:
            self.patch_length = _as_1d(self.patch_length, "patch_length")
            if self.patch_length.size != n:
                raise ValueError("patch_length must have length n_patch.")

        if self.dip is None:
            self.dip = np.degrees(np.arctan2(self.botz - self.topz, self.botx - self.topx))
        else:
            self.dip = _as_1d(self.dip, "dip")
            if self.dip.size != n:
                raise ValueError("dip must have length n_patch.")

    @property
    def n_patch(self) -> int:
        return int(self.topx.size)

    def summary(self) -> str:
        """Return a short human-readable geometry summary."""
        return (
            f"InterfaceGeometry(n_patch={self.n_patch}, "
            f"x=[{self.topx.min():.3g}, {self.botx.max():.3g}] km, "
            f"z=[{self.topz.min():.3g}, {self.botz.max():.3g}] km)"
        )

    @property
    def fault_patches(self) -> np.ndarray:
        """Return MATLAB-style ``[topx, topz, botx, botz]`` patch array."""
        return np.column_stack([self.topx, self.topz, self.botx, self.botz])

    def to_matdict(self) -> dict[str, np.ndarray]:
        """Return a MATLAB-compatible geometry dictionary."""
        return {
            "topx_interface": self.topx,
            "topz_interface": self.topz,
            "botx_interface": self.botx,
            "botz_interface": self.botz,
            "centers_interface": self.centers,
            "patch_length_interface": self.patch_length,
            "dip_interface": self.dip,
        }


@dataclass(slots=True)
class InterfaceConfig:
    """Parameters for a simple Hermite-polynomial interface builder.

    This builder is intended as the package-standard geometry input.  Existing
    notebooks can also bypass it by passing explicit arrays to
    :func:`interface_from_arrays`.
    """

    faultdip_trench: float = 10.0
    faultdip_bottom: float = 20.0
    x_trench: float = 0.0
    x_bottom: float = 238.0
    z_bottom: float = 45.0
    patch_length: float = 5.0
    z_trench: float = 0.0


def make_interface_geometry(cfg: InterfaceConfig) -> InterfaceGeometry:
    """Build a smooth 1-D interface from endpoint positions and endpoint dips.

    The depth curve is a cubic Hermite polynomial ``z(x)`` constrained by
    ``z(x_trench)``, ``z(x_bottom)``, and the dips at both ends.  Patch edges are
    spaced approximately by arclength, with maximum spacing ``patch_length``.
    """
    if cfg.patch_length <= 0:
        raise ValueError("patch_length must be positive.")
    if cfg.x_bottom <= cfg.x_trench:
        raise ValueError("x_bottom must be larger than x_trench.")

    x0, x1 = float(cfg.x_trench), float(cfg.x_bottom)
    z0, z1 = float(cfg.z_trench), float(cfg.z_bottom)
    Lx = x1 - x0
    m0 = np.tan(np.radians(cfg.faultdip_trench))
    m1 = np.tan(np.radians(cfg.faultdip_bottom))

    def z_of_x(x: np.ndarray) -> np.ndarray:
        t = (x - x0) / Lx
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        return h00 * z0 + h10 * Lx * m0 + h01 * z1 + h11 * Lx * m1

    # Oversample the curve, compute cumulative arclength, then resample edges.
    dense_n = max(1000, int(np.ceil((x1 - x0) / cfg.patch_length)) * 50)
    xd = np.linspace(x0, x1, dense_n)
    zd = z_of_x(xd)
    ds = np.hypot(np.diff(xd), np.diff(zd))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    n_patch = max(1, int(np.ceil(s[-1] / cfg.patch_length)))
    sedges = np.linspace(0.0, s[-1], n_patch + 1)
    xedges = np.interp(sedges, s, xd)
    zedges = np.interp(sedges, s, zd)

    geom = interface_from_arrays(
        topx=xedges[:-1],
        topz=zedges[:-1],
        botx=xedges[1:],
        botz=zedges[1:],
        metadata={
            "builder": "cubic_hermite_endpoint_dips",
            "faultdip_trench": cfg.faultdip_trench,
            "faultdip_bottom": cfg.faultdip_bottom,
            "x_trench": cfg.x_trench,
            "x_bottom": cfg.x_bottom,
            "z_trench": cfg.z_trench,
            "z_bottom": cfg.z_bottom,
            "requested_patch_length": cfg.patch_length,
        },
    )
    return geom


def interface_from_arrays(
    topx: ArrayLike,
    topz: ArrayLike,
    botx: ArrayLike,
    botz: ArrayLike,
    *,
    centers: Optional[ArrayLike] = None,
    patch_length: Optional[ArrayLike] = None,
    dip: Optional[ArrayLike] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> InterfaceGeometry:
    """Create :class:`InterfaceGeometry` from explicit patch endpoint arrays."""
    return InterfaceGeometry(
        topx=np.asarray(topx, dtype=float),
        topz=np.asarray(topz, dtype=float),
        botx=np.asarray(botx, dtype=float),
        botz=np.asarray(botz, dtype=float),
        centers=None if centers is None else np.asarray(centers, dtype=float),
        patch_length=None if patch_length is None else np.asarray(patch_length, dtype=float),
        dip=None if dip is None else np.asarray(dip, dtype=float),
        metadata={} if metadata is None else dict(metadata),
    )


def _as_1d(a: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float).reshape(-1)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def make_interface_geometry_legacy(
    faultdip_trench: float = 10.0,
    x_trench: float = 0.0,
    x_bottom: float = 238.0,
    faultdip_bottom: float = 20.0,
    z_bottom: float = 45.0,
    patch_length: float = 5.0,
) -> InterfaceGeometry:
    """Port of the legacy MATLAB ``make_geometry_interface.m`` routine.

    The MATLAB routine represented the interface with negative ``z`` values and
    stepped along the curve using ``dx = pL*cos(dip)``.  This Python version
    returns the package-standard positive-down depths while preserving the same
    cubic and stepping logic.
    """
    if patch_length <= 0:
        raise ValueError("patch_length must be positive.")
    if x_bottom <= x_trench:
        raise ValueError("x_bottom must be larger than x_trench.")
    if z_bottom <= 0:
        raise ValueError("z_bottom must be positive downward.")

    fault_bot_x = float(x_bottom)
    top_fault = 0.0

    # Legacy MATLAB solves for a cubic with z negative downward.
    b = np.array(
        [
            np.tan(-np.radians(faultdip_bottom)),
            np.tan(-np.radians(faultdip_trench)),
            -top_fault,
            -float(z_bottom),
        ],
        dtype=float,
    )
    A = np.array(
        [
            [3 * fault_bot_x**2, 2 * fault_bot_x, 1.0, 0.0],
            [3 * float(x_trench) ** 2, 2 * float(x_trench), 1.0, 0.0],
            [float(x_trench) ** 3, float(x_trench) ** 2, float(x_trench), 1.0],
            [fault_bot_x**3, fault_bot_x**2, fault_bot_x, 1.0],
        ],
        dtype=float,
    )
    c = np.linalg.solve(A, b)

    depth = 0.0
    dist = 0.0
    x_coord = [float(x_trench)]
    z_coord_legacy = [0.0]

    while abs(depth) < z_bottom:
        dip = np.arctan(c[0] * 3 * dist**2 + c[1] * 2 * dist + c[2])
        dx = patch_length * np.cos(dip)
        dist = dist + dx
        z_val = c[0] * dist**3 + c[1] * dist**2 + c[2] * dist + c[3]
        x_coord.append(float(x_trench) + dist)
        z_coord_legacy.append(z_val)
        depth = z_coord_legacy[-1]

    x_coord = np.asarray(x_coord, dtype=float)
    z_positive_down = -np.asarray(z_coord_legacy, dtype=float)

    return interface_from_arrays(
        topx=x_coord[:-1],
        topz=z_positive_down[:-1],
        botx=x_coord[1:],
        botz=z_positive_down[1:],
        metadata={
            "builder": "legacy_make_geometry_interface_port",
            "faultdip_trench": faultdip_trench,
            "faultdip_bottom": faultdip_bottom,
            "x_trench": x_trench,
            "x_bottom": x_bottom,
            "z_bottom": z_bottom,
            "patch_length": patch_length,
            "legacy_note": "MATLAB routine used negative z; this object uses positive-down z.",
        },
    )
