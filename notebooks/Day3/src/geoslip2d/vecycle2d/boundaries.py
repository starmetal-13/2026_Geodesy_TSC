"""Boundary dataclasses used by the VECycle2D geometry builder."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


Array = np.ndarray


@dataclass(slots=True)
class Boundary:
    """A segmented 2D boundary or interface.

    Arrays follow the MATLAB convention: ``centers``, ``dipvec``, and
    ``normvec`` have shape ``(2, n_patch)``.
    """

    name: str
    topx: Array
    topz: Array
    botx: Array
    botz: Array
    centers: Array
    dipvec: Array
    normvec: Array

    @property
    def length(self) -> Array:
        return np.sqrt((self.topx - self.botx) ** 2 + (self.topz - self.botz) ** 2)

    @property
    def npatch(self) -> int:
        return int(self.topx.size)


def make_boundary(
    name: str,
    topx: Array,
    topz: Array,
    botx: Array,
    botz: Array,
    centers: Array,
    dipvec: Array,
    normvec: Array,
) -> Boundary:
    """Create a boundary object with float NumPy arrays."""

    return Boundary(
        name=name,
        topx=np.asarray(topx, dtype=float).reshape(-1),
        topz=np.asarray(topz, dtype=float).reshape(-1),
        botx=np.asarray(botx, dtype=float).reshape(-1),
        botz=np.asarray(botz, dtype=float).reshape(-1),
        centers=np.asarray(centers, dtype=float),
        dipvec=np.asarray(dipvec, dtype=float),
        normvec=np.asarray(normvec, dtype=float),
    )
