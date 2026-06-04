"""Canonical Green's-function containers and validation for GeoSlip2D."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d

from .geometry import InterfaceGeometry, interface_from_arrays


@dataclass(slots=True)
class Greens2D:
    """Canonical Green's-function object used by all GeoSlip2D backends.

    ``Ghor`` and ``Gvert`` are observation-by-patch matrices.  The observation
    coordinates are ``xobs`` in km along the 1-D profile.  ``Gvert`` may be
    ``None`` for horizontal-only workflows.
    """

    Ghor: np.ndarray
    Gvert: Optional[np.ndarray]
    xobs: np.ndarray
    interface: InterfaceGeometry
    source_type: str
    units: str = "displacement_per_unit_slip"
    sign_convention: str = "forward_slip_positive"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.Ghor = np.asarray(self.Ghor, dtype=float)
        if self.Ghor.ndim != 2:
            raise ValueError("Ghor must be a 2-D array with shape (n_obs, n_patch).")
        self.xobs = np.asarray(self.xobs, dtype=float).reshape(-1)
        if self.Ghor.shape[0] != self.xobs.size:
            raise ValueError("Ghor row count must match len(xobs).")
        if self.Ghor.shape[1] != self.interface.n_patch:
            raise ValueError("Ghor column count must match interface.n_patch.")
        if np.any(~np.isfinite(self.Ghor)) or np.any(~np.isfinite(self.xobs)):
            raise ValueError("Ghor and xobs must contain finite values.")

        if self.Gvert is not None:
            self.Gvert = np.asarray(self.Gvert, dtype=float)
            if self.Gvert.shape != self.Ghor.shape:
                raise ValueError("Gvert must have the same shape as Ghor.")
            if np.any(~np.isfinite(self.Gvert)):
                raise ValueError("Gvert must contain finite values.")

    @property
    def n_obs(self) -> int:
        return int(self.Ghor.shape[0])

    @property
    def n_patch(self) -> int:
        return int(self.Ghor.shape[1])

    def summary(self) -> str:
        """Return a short human-readable Green's-function summary."""
        has_vert = self.Gvert is not None
        return (
            f"Greens2D(source_type={self.source_type!r}, n_obs={self.n_obs}, "
            f"n_patch={self.n_patch}, has_vertical={has_vert}, units={self.units!r})"
        )

    # MATLAB-compatible aliases expected by the current inversion code.
    @property
    def topx_interface(self) -> np.ndarray:
        return self.interface.topx

    @property
    def topz_interface(self) -> np.ndarray:
        return self.interface.topz

    @property
    def botx_interface(self) -> np.ndarray:
        return self.interface.botx

    @property
    def botz_interface(self) -> np.ndarray:
        return self.interface.botz

    def to_matdict(self) -> dict[str, Any]:
        """Return a dictionary compatible with the old MATLAB-style Greens struct."""
        out: dict[str, Any] = {
            "Ghor": self.Ghor,
            "xobs": self.xobs,
            "source_type": self.source_type,
            "units": self.units,
            "sign_convention": self.sign_convention,
        }
        if self.Gvert is not None:
            out["Gvert"] = self.Gvert
        out.update(self.interface.to_matdict())
        return out

    def interp_to(self, xnew: ArrayLike) -> "Greens2D":
        """Return a new Greens2D object interpolated to new observation positions."""
        xnew = np.asarray(xnew, dtype=float).reshape(-1)
        Ghor_i = _interp_matrix(self.xobs, self.Ghor, xnew)
        Gvert_i = None if self.Gvert is None else _interp_matrix(self.xobs, self.Gvert, xnew)
        return Greens2D(
            Ghor=Ghor_i,
            Gvert=Gvert_i,
            xobs=xnew,
            interface=self.interface,
            source_type=self.source_type,
            units=self.units,
            sign_convention=self.sign_convention,
            metadata=dict(self.metadata),
        )


def greens_from_matdict(obj: dict[str, Any], *, source_type: str = "unknown") -> Greens2D:
    """Create :class:`Greens2D` from a MATLAB-style Green's structure dict."""
    required = ["Ghor", "xobs", "topx_interface", "topz_interface"]
    missing = [k for k in required if k not in obj]
    if missing:
        raise KeyError(f"Missing required Green's fields: {', '.join(missing)}")

    topx = np.asarray(obj["topx_interface"], dtype=float).reshape(-1)
    topz = np.asarray(obj["topz_interface"], dtype=float).reshape(-1)
    if "botx_interface" in obj and "botz_interface" in obj:
        botx = np.asarray(obj["botx_interface"], dtype=float).reshape(-1)
        botz = np.asarray(obj["botz_interface"], dtype=float).reshape(-1)
    else:
        # Compatibility fallback for older files that saved only top-edge points.
        # Infer bottom endpoints from the next top point.  The final patch is
        # extrapolated from the previous segment.
        if topx.size < 2:
            raise KeyError("botx_interface/botz_interface are required when fewer than two top points are available.")
        botx = np.empty_like(topx)
        botz = np.empty_like(topz)
        botx[:-1] = topx[1:]
        botz[:-1] = topz[1:]
        botx[-1] = topx[-1] + (topx[-1] - topx[-2])
        botz[-1] = topz[-1] + (topz[-1] - topz[-2])

    interface = interface_from_arrays(
        topx=topx,
        topz=topz,
        botx=botx,
        botz=botz,
        centers=obj.get("centers_interface"),
        patch_length=obj.get("patch_length_interface"),
        dip=obj.get("dip_interface"),
        metadata={"source": "matdict_compatibility"},
    )
    return Greens2D(
        Ghor=np.asarray(obj["Ghor"], dtype=float),
        Gvert=None if "Gvert" not in obj else np.asarray(obj["Gvert"], dtype=float),
        xobs=np.asarray(obj["xobs"], dtype=float),
        interface=interface,
        source_type=str(obj.get("source_type", source_type)),
        units=str(obj.get("units", "displacement_per_unit_slip")),
        sign_convention=str(obj.get("sign_convention", "forward_slip_positive")),
        metadata={"loaded_from": "matdict"},
    )


def _interp_matrix(xobs: np.ndarray, G: np.ndarray, xnew: np.ndarray) -> np.ndarray:
    order = np.argsort(xobs)
    f = interp1d(
        xobs[order],
        G[order, :],
        axis=0,
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )
    Gi = np.asarray(f(xnew), dtype=float)
    if np.any(~np.isfinite(Gi)):
        raise ValueError("Interpolated Green's matrix contains non-finite values; xnew may be outside xobs range.")
    return Gi
