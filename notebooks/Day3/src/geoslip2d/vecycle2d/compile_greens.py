"""
Compile raw VECycle2D Green's-function blocks into linear-system matrices.

This module ports the MATLAB Phase-6 functions:

    vec_compile_plate_grav_surf_from_raw.m
    vec_build_wedge_linear_system.m   (matrix-assembly part only)

The workflow is:

    raw = build_raw_greens(geom, nu=cfg.constants.nu)
    C = compile_plate_grav_surf_from_raw(raw, cfg.constants)
    model = build_wedge_linear_system_from_compiled(C, geom)

The compiled object ``C`` stores MATLAB-equivalent fields such as:

    C.Gs11_1
    C.Gs11_1v
    C.Gd11_6
    C.Gs11_i

The linear-system model stores:

    model.G_part1
    model.G_part2
    model.Gi

plus selected matrices needed by the earthquake-cycle solver.

This file intentionally preserves the MATLAB matrix ordering. The numerical
solver downstream depends on this ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


Array = np.ndarray


ELASTIC_DIP_SOURCES = ("1", "2", "3", "4", "5", "6", "7", "b", "t", "i")
ELASTIC_NORMAL_SOURCES = ("1", "2", "3", "4", "5", "6", "7", "b", "t")
VISC_SOURCES = ("1", "2", "3", "4", "5", "8", "b", "t")


@dataclass(slots=True)
class CompiledGreens:
    """Container for compiled Green's matrices.

    Matrices are stored in a dictionary to preserve the dynamic MATLAB-style
    field names while still allowing attribute access:

        C["Gs11_1"]
        C.Gs11_1
    """

    matrices: dict[str, Array]

    def __getitem__(self, key: str) -> Array:
        return self.matrices[key]

    def __getattr__(self, key: str) -> Array:
        try:
            return self.matrices[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def keys(self):
        return self.matrices.keys()

    def items(self):
        return self.matrices.items()

    def as_dict(self) -> dict[str, Array]:
        return dict(self.matrices)


@dataclass(slots=True)
class WedgeLinearSystem:
    """Compiled linear system needed by the VECycle2D cycle solver."""

    G_part1: Array
    G_part2: Array
    Gi: Array

    Gs11_1: Array
    Gs12_1: Array

    Gd11_6: Array
    Gd12_6: Array
    Gd21_6: Array
    Gd22_6: Array

    Gd11_7: Array
    Gd12_7: Array
    Gd21_7: Array
    Gd22_7: Array

    Gs11_i: Array
    Gs12_i: Array
    Gs21_i: Array
    Gs22_i: Array

    num_v: int
    xpos: Array
    centers_interface: Array
    num_interface: int

    raw_greens: Mapping[str, Array] | None = None
    compiled_greens: CompiledGreens | None = None
    geom: Any | None = None
    internal_constants: Any | None = None


def _const(constants: Any, name: str) -> float:
    """Read a constant from a dataclass/object or dictionary."""
    if isinstance(constants, Mapping):
        return float(constants[name])
    return float(getattr(constants, name))


def raw_field_name(prefix: str, receiver: str, source: str, suffix: str = "") -> str:
    """Return MATLAB-compatible raw field name, e.g. ``sig11_ii``."""
    return f"{prefix}_{receiver}{source}{suffix}"


def get_raw(
    raw: Mapping[str, Array],
    prefix: str,
    receiver: str,
    source: str,
    suffix: str = "",
) -> Array:
    """Fetch one raw block by MATLAB-compatible name."""
    name = raw_field_name(prefix, receiver, source, suffix)
    if name not in raw:
        return np.zeros((0, 0), dtype=float)
    return np.asarray(raw[name], dtype=float)


def _hstack_nonempty(parts: list[Array]) -> Array:
    """Horizontally concatenate, ignoring truly empty missing blocks."""
    nonempty = [np.asarray(p, dtype=float) for p in parts if np.asarray(p).size > 0]
    if not nonempty:
        return np.zeros((0, 0), dtype=float)
    return np.hstack(nonempty)


def disp_cat(
    raw: Mapping[str, Array],
    prefix: str,
    receiver: str,
    source_codes: Iterable[str],
    use_v: bool,
) -> Array:
    """Port of MATLAB ``disp_cat`` helper."""
    parts: list[Array] = []
    for source in source_codes:
        suffix = ""
        if use_v and source in {"1", "2", "3", "4", "5", "8", "b", "t"} and receiver == source:
            suffix = "v"
        parts.append(get_raw(raw, prefix, receiver, source, suffix))
    return _hstack_nonempty(parts)


def sig_cat(
    raw: Mapping[str, Array],
    prefix: str,
    receiver: str,
    source_codes: Iterable[str],
) -> Array:
    """Port of MATLAB ``sig_cat`` helper."""
    return _hstack_nonempty([
        get_raw(raw, prefix, receiver, source, "")
        for source in source_codes
    ])


def compile_plate_grav_surf_from_raw(
    raw: Mapping[str, Array],
    constants: Any,
) -> CompiledGreens:
    """
    Compile raw displacement/traction blocks into MATLAB-equivalent matrices.

    Parameters
    ----------
    raw
        Raw dictionary from ``build_raw_greens``.
    constants
        ``InternalConstants`` object or dict with ``mu_1``, ``mu_2``, and
        ``rhog``.

    Returns
    -------
    CompiledGreens
        Container with MATLAB-equivalent fields such as ``Gs11_1`` and
        ``Gd11_6``.
    """
    mu_1 = _const(constants, "mu_1")
    mu_2 = _const(constants, "mu_2")
    rhog = _const(constants, "rhog")

    C: dict[str, Array] = {}

    # Solid elastic boundaries with both elastic and viscous comparison matrices.
    for rc in ("1", "2", "3", "4", "5"):
        C[f"Gd11_{rc}"] = disp_cat(raw, "u11", rc, ELASTIC_DIP_SOURCES, False)
        C[f"Gd12_{rc}"] = disp_cat(raw, "u12", rc, ELASTIC_NORMAL_SOURCES, False)
        C[f"Gd21_{rc}"] = disp_cat(raw, "u21", rc, ELASTIC_DIP_SOURCES, False)
        C[f"Gd22_{rc}"] = disp_cat(raw, "u22", rc, ELASTIC_NORMAL_SOURCES, False)

        C[f"Gd11_{rc}v"] = disp_cat(raw, "u11", rc, VISC_SOURCES, True)
        C[f"Gd12_{rc}v"] = disp_cat(raw, "u12", rc, VISC_SOURCES, True)
        C[f"Gd21_{rc}v"] = disp_cat(raw, "u21", rc, VISC_SOURCES, True)
        C[f"Gd22_{rc}v"] = disp_cat(raw, "u22", rc, VISC_SOURCES, True)

        C[f"Gs11_{rc}"] = mu_1 * sig_cat(raw, "sig11", rc, ELASTIC_DIP_SOURCES)
        C[f"Gs12_{rc}"] = mu_1 * sig_cat(raw, "sig12", rc, ELASTIC_NORMAL_SOURCES)
        C[f"Gs21_{rc}"] = mu_1 * sig_cat(raw, "sig21", rc, ELASTIC_DIP_SOURCES)
        C[f"Gs22_{rc}"] = mu_1 * sig_cat(raw, "sig22", rc, ELASTIC_NORMAL_SOURCES)

        C[f"Gs11_{rc}v"] = mu_2 * sig_cat(raw, "sig11", rc, VISC_SOURCES)
        C[f"Gs12_{rc}v"] = mu_2 * sig_cat(raw, "sig12", rc, VISC_SOURCES)
        C[f"Gs21_{rc}v"] = mu_2 * sig_cat(raw, "sig21", rc, VISC_SOURCES)
        C[f"Gs22_{rc}v"] = mu_2 * sig_cat(raw, "sig22", rc, VISC_SOURCES)

    # Free-surface receiver boundaries 6 and 7 with gravitational correction.
    for rc in ("6", "7"):
        C[f"Gd11_{rc}"] = disp_cat(raw, "u11", rc, ELASTIC_DIP_SOURCES, False)
        C[f"Gd12_{rc}"] = disp_cat(raw, "u12", rc, ELASTIC_NORMAL_SOURCES, False)
        C[f"Gd21_{rc}"] = disp_cat(raw, "u21", rc, ELASTIC_DIP_SOURCES, False)
        C[f"Gd22_{rc}"] = disp_cat(raw, "u22", rc, ELASTIC_NORMAL_SOURCES, False)

        C[f"Gs11_{rc}"] = mu_1 * sig_cat(raw, "sig11", rc, ELASTIC_DIP_SOURCES)
        C[f"Gs12_{rc}"] = mu_1 * sig_cat(raw, "sig12", rc, ELASTIC_NORMAL_SOURCES)
        C[f"Gs21_{rc}"] = (
            mu_1 * sig_cat(raw, "sig21", rc, ELASTIC_DIP_SOURCES)
            + mu_1 * rhog * C[f"Gd21_{rc}"]
        )
        C[f"Gs22_{rc}"] = (
            mu_1 * sig_cat(raw, "sig22", rc, ELASTIC_NORMAL_SOURCES)
            + mu_1 * rhog * C[f"Gd22_{rc}"]
        )

    # Boundary 8 is viscous-side only in the assembled linear system.
    C["Gd11_8v"] = disp_cat(raw, "u11", "8", VISC_SOURCES, True)
    C["Gd12_8v"] = disp_cat(raw, "u12", "8", VISC_SOURCES, True)
    C["Gd21_8v"] = disp_cat(raw, "u21", "8", VISC_SOURCES, True)
    C["Gd22_8v"] = disp_cat(raw, "u22", "8", VISC_SOURCES, True)
    C["Gs11_8v"] = mu_2 * sig_cat(raw, "sig11", "8", VISC_SOURCES)
    C["Gs12_8v"] = mu_2 * sig_cat(raw, "sig12", "8", VISC_SOURCES)
    C["Gs21_8v"] = mu_2 * sig_cat(raw, "sig21", "8", VISC_SOURCES) + mu_2 * rhog * C["Gd21_8v"]
    C["Gs22_8v"] = mu_2 * sig_cat(raw, "sig22", "8", VISC_SOURCES) + mu_2 * rhog * C["Gd22_8v"]

    # Slab bottom and top receiver boundaries.
    for rc in ("b", "t"):
        C[f"Gd11_{rc}"] = disp_cat(raw, "u11", rc, ELASTIC_DIP_SOURCES, False)
        C[f"Gd12_{rc}"] = disp_cat(raw, "u12", rc, ELASTIC_NORMAL_SOURCES, False)
        C[f"Gd21_{rc}"] = disp_cat(raw, "u21", rc, ELASTIC_DIP_SOURCES, False)
        C[f"Gd22_{rc}"] = disp_cat(raw, "u22", rc, ELASTIC_NORMAL_SOURCES, False)

        C[f"Gd11_{rc}v"] = disp_cat(raw, "u11", rc, VISC_SOURCES, True)
        C[f"Gd12_{rc}v"] = disp_cat(raw, "u12", rc, VISC_SOURCES, True)
        C[f"Gd21_{rc}v"] = disp_cat(raw, "u21", rc, VISC_SOURCES, True)
        C[f"Gd22_{rc}v"] = disp_cat(raw, "u22", rc, VISC_SOURCES, True)

        C[f"Gs11_{rc}"] = mu_1 * sig_cat(raw, "sig11", rc, ELASTIC_DIP_SOURCES)
        C[f"Gs12_{rc}"] = mu_1 * sig_cat(raw, "sig12", rc, ELASTIC_NORMAL_SOURCES)
        C[f"Gs21_{rc}"] = mu_1 * sig_cat(raw, "sig21", rc, ELASTIC_DIP_SOURCES)
        C[f"Gs22_{rc}"] = mu_1 * sig_cat(raw, "sig22", rc, ELASTIC_NORMAL_SOURCES)

        C[f"Gs11_{rc}v"] = mu_2 * sig_cat(raw, "sig11", rc, VISC_SOURCES)
        C[f"Gs12_{rc}v"] = mu_2 * sig_cat(raw, "sig12", rc, VISC_SOURCES)
        C[f"Gs21_{rc}v"] = mu_2 * sig_cat(raw, "sig21", rc, VISC_SOURCES)
        C[f"Gs22_{rc}v"] = mu_2 * sig_cat(raw, "sig22", rc, VISC_SOURCES)

    # Interface receiver: traction equations only.
    C["Gs11_i"] = mu_1 * sig_cat(raw, "sig11", "i", ELASTIC_DIP_SOURCES)
    C["Gs12_i"] = mu_1 * sig_cat(raw, "sig12", "i", ELASTIC_NORMAL_SOURCES)
    C["Gs21_i"] = mu_1 * sig_cat(raw, "sig21", "i", ELASTIC_DIP_SOURCES)
    C["Gs22_i"] = mu_1 * sig_cat(raw, "sig22", "i", ELASTIC_NORMAL_SOURCES)

    return CompiledGreens(C)


def _zlike(rows_from: Array, ncols: int) -> Array:
    """Return zeros with the row count of ``rows_from`` and ncols columns."""
    return np.zeros((np.asarray(rows_from).shape[0], int(ncols)), dtype=float)


def _h(*arrays: Array) -> Array:
    """Short horizontal stack helper."""
    return np.hstack([np.asarray(a, dtype=float) for a in arrays])


def _v(*arrays: Array) -> Array:
    """Short vertical stack helper, skipping intentionally empty blocks."""
    nonempty = [np.asarray(a, dtype=float) for a in arrays if np.asarray(a).size > 0]
    if not nonempty:
        return np.zeros((0, 0), dtype=float)
    return np.vstack(nonempty)


def build_wedge_linear_system_from_compiled(
    C: CompiledGreens,
    geom: Any,
    *,
    raw: Mapping[str, Array] | None = None,
    constants: Any | None = None,
) -> WedgeLinearSystem:
    """
    Assemble the VECycle2D linear-system matrices from compiled Green's blocks.

    This ports the matrix assembly in MATLAB ``vec_build_wedge_linear_system.m``.
    """
    num_v = int(C.Gs11_1v.shape[1])

    G1 = _h(C.Gs11_1, -C.Gs11_1v, C.Gs12_1, -C.Gs12_1v)
    G2 = _h(C.Gs21_1, -C.Gs21_1v, C.Gs22_1, -C.Gs22_1v)

    G3 = _h(C.Gs11_2, -C.Gs11_2v, C.Gs12_2, -C.Gs12_2v)
    G4 = _h(C.Gs21_2, -C.Gs21_2v, C.Gs22_2, -C.Gs22_2v)

    G5 = _h(C.Gs11_3, -C.Gs11_3v, C.Gs12_3, -C.Gs12_3v)
    G6 = _h(C.Gs21_3, -C.Gs21_3v, C.Gs22_3, -C.Gs22_3v)

    G7 = _h(C.Gs11_4, -C.Gs11_4v, C.Gs12_4, -C.Gs12_4v)
    G8 = _h(C.Gs21_4, -C.Gs21_4v, C.Gs22_4, -C.Gs22_4v)

    G9 = _h(C.Gs11_5, -C.Gs11_5v, C.Gs12_5, -C.Gs12_5v)
    G10 = _h(C.Gs21_5, -C.Gs21_5v, C.Gs22_5, -C.Gs22_5v)

    Z = _zlike(C.Gs11_6, num_v)
    G11 = _h(C.Gs11_6, Z, C.Gs12_6, Z)
    G12 = _h(C.Gs21_6, Z, C.Gs22_6, Z)

    Z = _zlike(C.Gs11_7, num_v)
    G13 = _h(C.Gs11_7, Z, C.Gs12_7, Z)
    G14 = _h(C.Gs21_7, Z, C.Gs22_7, Z)

    Z1 = np.zeros((C.Gs11_8v.shape[0], C.Gs11_7.shape[1]), dtype=float)
    Z2 = np.zeros((C.Gs11_8v.shape[0], C.Gs12_7.shape[1]), dtype=float)
    G15 = _h(Z1, C.Gs11_8v, Z2, C.Gs12_8v)
    G16 = _h(Z1, C.Gd21_8v, Z2, C.Gd22_8v)

    G17 = _h(C.Gs11_b, -C.Gs11_bv, C.Gs12_b, -C.Gs12_bv)
    G18 = _h(C.Gs21_b, -C.Gs21_bv, C.Gs22_b, -C.Gs22_bv)

    G19 = _h(C.Gs11_t, -C.Gs11_tv, C.Gs12_t, -C.Gs12_tv)
    G20 = _h(C.Gs21_t, -C.Gs21_tv, C.Gs22_t, -C.Gs22_tv)

    G_part1 = _v(
        G1, G2, G3, G4, G5, G6, G7, G8, G9, G10,
        G11, G12, G13, G14, G15, G16, G17, G18, G19, G20,
    )

    Gi = _h(
        C.Gs11_i,
        np.zeros((C.Gs11_i.shape[0], num_v), dtype=float),
        C.Gs12_i,
        np.zeros((C.Gs11_i.shape[0], num_v), dtype=float),
    )

    H19 = _h(C.Gd11_1, -C.Gd11_1v, C.Gd12_1, -C.Gd12_1v)
    H20 = _h(C.Gd21_1, -C.Gd21_1v, C.Gd22_1, -C.Gd22_1v)

    H21 = _h(C.Gd11_2, -C.Gd11_2v, C.Gd12_2, -C.Gd12_2v)
    H22 = _h(C.Gd21_2, -C.Gd21_2v, C.Gd22_2, -C.Gd22_2v)

    H23 = _h(C.Gd11_3, -C.Gd11_3v, C.Gd12_3, -C.Gd12_3v)
    H24 = _h(C.Gd21_3, -C.Gd21_3v, C.Gd22_3, -C.Gd22_3v)

    H25 = _h(C.Gd11_4, -C.Gd11_4v, C.Gd12_4, -C.Gd12_4v)
    H26 = _h(C.Gd21_4, -C.Gd21_4v, C.Gd22_4, -C.Gd22_4v)

    H27 = _h(C.Gd11_5, -C.Gd11_5v, C.Gd12_5, -C.Gd12_5v)
    H28 = _h(C.Gd21_5, -C.Gd21_5v, C.Gd22_5, -C.Gd22_5v)

    H31 = _h(C.Gd11_b, -C.Gd11_bv, C.Gd12_b, -C.Gd12_bv)
    H32 = _h(C.Gd21_b, -C.Gd21_bv, C.Gd22_b, -C.Gd22_bv)

    H33 = _h(C.Gd11_t, -C.Gd11_tv, C.Gd12_t, -C.Gd12_tv)
    H34 = _h(C.Gd21_t, -C.Gd21_tv, C.Gd22_t, -C.Gd22_tv)

    G_part2 = _v(
        H19, H20, H21, H22, H23, H24, H25, H26, H27, H28,
        H31, H32, H33, H34,
    )

    xpos = np.asarray(getattr(geom, "surface_x", getattr(getattr(geom, "surface", None), "x", np.array([]))), dtype=float).reshape(-1)
    centers_interface = np.asarray(getattr(geom, "centers_interface"), dtype=float)
    num_interface = int(getattr(geom, "num_interface"))

    return WedgeLinearSystem(
        G_part1=G_part1,
        G_part2=G_part2,
        Gi=Gi,
        Gs11_1=C.Gs11_1,
        Gs12_1=C.Gs12_1,
        Gd11_6=C.Gd11_6,
        Gd12_6=C.Gd12_6,
        Gd21_6=C.Gd21_6,
        Gd22_6=C.Gd22_6,
        Gd11_7=C.Gd11_7,
        Gd12_7=C.Gd12_7,
        Gd21_7=C.Gd21_7,
        Gd22_7=C.Gd22_7,
        Gs11_i=C.Gs11_i,
        Gs12_i=C.Gs12_i,
        Gs21_i=C.Gs21_i,
        Gs22_i=C.Gs22_i,
        num_v=num_v,
        xpos=xpos,
        centers_interface=centers_interface,
        num_interface=num_interface,
        raw_greens=raw,
        compiled_greens=C,
        geom=geom,
        internal_constants=constants,
    )


def compile_raw_to_linear_system(
    raw: Mapping[str, Array],
    geom: Any,
    constants: Any,
) -> WedgeLinearSystem:
    """Convenience wrapper: raw dictionary -> compiled linear system."""
    C = compile_plate_grav_surf_from_raw(raw, constants)
    return build_wedge_linear_system_from_compiled(C, geom, raw=raw, constants=constants)


def compiled_summary(C: CompiledGreens | WedgeLinearSystem) -> dict[str, tuple[int, ...] | int]:
    """Return a compact shape summary for compiled outputs."""
    if isinstance(C, CompiledGreens):
        return {key: np.asarray(val).shape for key, val in sorted(C.items())}
    if isinstance(C, WedgeLinearSystem):
        return {
            "G_part1": C.G_part1.shape,
            "G_part2": C.G_part2.shape,
            "Gi": C.Gi.shape,
            "num_v": C.num_v,
            "num_interface": C.num_interface,
        }
    raise TypeError("Expected CompiledGreens or WedgeLinearSystem.")


__all__ = [
    "CompiledGreens",
    "WedgeLinearSystem",
    "compile_plate_grav_surf_from_raw",
    "build_wedge_linear_system_from_compiled",
    "compile_raw_to_linear_system",
    "compiled_summary",
    "disp_cat",
    "sig_cat",
    "get_raw",
]
