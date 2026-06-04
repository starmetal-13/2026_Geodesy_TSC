"""
Loop-based raw Green's-function construction for VECycle2D.

This ports the MATLAB Phase-6 function `vec_build_greens_viscoelastic_surf_loop.m`.
It loops over source patches and receiver boundaries, calls `make_traction_disp`,
and returns a MATLAB-compatible dictionary of raw Green's-function blocks.

Example raw keys:
    raw["u11_11"]
    raw["u11_11v"]
    raw["sig22_i5"]
    raw["u12_7i"]

Each value has shape:
    (number_of_receiver_points, number_of_source_patches)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .traction import make_traction_disp

Array = np.ndarray


@dataclass(slots=True)
class RawEntity:
    """Internal source/receiver entity matching the MATLAB boundary codes."""

    code: str
    topx: Array
    topz: Array
    botx: Array
    botz: Array
    centers: Array
    dipvec: Array
    normvec: Array
    centers_e: Array | None = None
    centers_v: Array | None = None

    @property
    def npatch(self) -> int:
        return int(self.topx.size)


def _get_legacy(geom_or_legacy: Any) -> Mapping[str, Any]:
    if isinstance(geom_or_legacy, Mapping):
        return geom_or_legacy
    if hasattr(geom_or_legacy, "legacy"):
        return geom_or_legacy.legacy
    raise TypeError("Expected a Geometry object with .legacy or a legacy dict.")


def _arr(legacy: Mapping[str, Any], name: str) -> Array:
    if name not in legacy:
        return np.array([], dtype=float)
    return np.asarray(legacy[name], dtype=float)


def _optional_arr(legacy: Mapping[str, Any], name: str) -> Array | None:
    if name not in legacy:
        return None
    val = np.asarray(legacy[name], dtype=float)
    if val.size == 0:
        return None
    return val


def _make_entities(
    legacy: Mapping[str, Any],
    *,
    for_sources: bool,
    codes: Iterable[str] | None = None,
) -> list[RawEntity]:
    """Construct source or receiver entities using MATLAB boundary codes."""
    if codes is None:
        codes = ("1", "2", "3", "4", "5", "6", "7", "8", "b", "t", "i") if for_sources else (
            "1", "2", "3", "4", "5", "6", "7", "8", "i", "b", "t"
        )

    entities: list[RawEntity] = []

    for code in codes:
        if code in {"1", "2", "3", "4", "5", "6", "7", "8"}:
            ent = RawEntity(
                code=code,
                topx=_arr(legacy, f"topx{code}").reshape(-1),
                topz=_arr(legacy, f"topz{code}").reshape(-1),
                botx=_arr(legacy, f"botx{code}").reshape(-1),
                botz=_arr(legacy, f"botz{code}").reshape(-1),
                centers=_arr(legacy, f"centers{code}"),
                centers_e=_optional_arr(legacy, f"centers{code}e"),
                centers_v=_optional_arr(legacy, f"centers{code}v"),
                dipvec=_arr(legacy, f"dipvec{code}"),
                normvec=_arr(legacy, f"normvec{code}"),
            )
        elif code == "i":
            ent = RawEntity(
                code="i",
                topx=_arr(legacy, "topx_interface").reshape(-1),
                topz=_arr(legacy, "topz_interface").reshape(-1),
                botx=_arr(legacy, "botx_interface").reshape(-1),
                botz=_arr(legacy, "botz_interface").reshape(-1),
                centers=_arr(legacy, "centers_interface"),
                dipvec=_arr(legacy, "dipvec_interface"),
                normvec=_arr(legacy, "normvec_interface"),
            )
        elif code == "b":
            ent = RawEntity(
                code="b",
                topx=_arr(legacy, "topx_botslab").reshape(-1),
                topz=_arr(legacy, "topz_botslab").reshape(-1),
                botx=_arr(legacy, "botx_botslab").reshape(-1),
                botz=_arr(legacy, "botz_botslab").reshape(-1),
                centers=_arr(legacy, "centers_botslab"),
                centers_e=_optional_arr(legacy, "centers_botslabe"),
                centers_v=_optional_arr(legacy, "centers_botslabv"),
                dipvec=_arr(legacy, "dipvec_botslab"),
                normvec=_arr(legacy, "normvec_botslab"),
            )
        elif code == "t":
            ent = RawEntity(
                code="t",
                topx=_arr(legacy, "topx_topslab").reshape(-1),
                topz=_arr(legacy, "topz_topslab").reshape(-1),
                botx=_arr(legacy, "botx_topslab").reshape(-1),
                botz=_arr(legacy, "botz_topslab").reshape(-1),
                centers=_arr(legacy, "centers_topslab"),
                centers_e=_optional_arr(legacy, "centers_topslabe"),
                centers_v=_optional_arr(legacy, "centers_topslabv"),
                dipvec=_arr(legacy, "dipvec_topslab"),
                normvec=_arr(legacy, "normvec_topslab"),
            )
        else:
            raise ValueError(f"Unknown boundary/source code: {code!r}")

        entities.append(ent)

    return entities


def raw_name(prefix: str, receiver_code: str, source_code: str, suffix: str = "") -> str:
    return f"{prefix}_{receiver_code}{source_code}{suffix}"


def _xloc_from_centers(centers: Array) -> Array:
    centers = np.asarray(centers, dtype=float)
    if centers.ndim == 1:
        centers = centers.reshape(2, 1)
    if centers.shape[0] != 2:
        raise ValueError("centers must have shape (2, N)")
    return np.vstack((centers[0, :], np.zeros(centers.shape[1]), centers[1, :]))


def _store_field(raw: dict[str, Array], name: str, values: Array, col: int, ncols: int) -> None:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raw[name] = np.array([], dtype=float)
        return
    if name not in raw:
        raw[name] = np.zeros((values.size, ncols), dtype=float)
    if raw[name].shape[0] != values.size:
        raise ValueError(
            f"Field {name} has {raw[name].shape[0]} receiver rows, but new values have {values.size}."
        )
    raw[name][:, col] = values


def _store_all(
    raw: dict[str, Array],
    receiver_code: str,
    source_code: str,
    suffix: str,
    col: int,
    ncols: int,
    sig11: Array,
    sig21: Array,
    sig12: Array,
    sig22: Array,
    u11: Array,
    u21: Array,
    u12: Array,
    u22: Array,
) -> None:
    _store_field(raw, raw_name("u11", receiver_code, source_code, suffix), u11, col, ncols)
    _store_field(raw, raw_name("u12", receiver_code, source_code, suffix), u12, col, ncols)
    _store_field(raw, raw_name("u21", receiver_code, source_code, suffix), u21, col, ncols)
    _store_field(raw, raw_name("u22", receiver_code, source_code, suffix), u22, col, ncols)
    _store_field(raw, raw_name("sig11", receiver_code, source_code, suffix), sig11, col, ncols)
    _store_field(raw, raw_name("sig12", receiver_code, source_code, suffix), sig12, col, ncols)
    _store_field(raw, raw_name("sig21", receiver_code, source_code, suffix), sig21, col, ncols)
    _store_field(raw, raw_name("sig22", receiver_code, source_code, suffix), sig22, col, ncols)


def _patch_indices_for_source(
    source: RawEntity,
    patch_indices_by_source: Mapping[str, Iterable[int]] | None,
) -> np.ndarray:
    if patch_indices_by_source is None or source.code not in patch_indices_by_source:
        return np.arange(source.npatch, dtype=int)
    idx = np.asarray(list(patch_indices_by_source[source.code]), dtype=int)
    if idx.ndim != 1:
        raise ValueError("Patch indices must be a 1D sequence.")
    if np.any(idx < 0) or np.any(idx >= source.npatch):
        raise IndexError(f"Patch indices for source {source.code!r} must be between 0 and {source.npatch - 1}.")
    return idx


def build_raw_greens(
    geom_or_legacy: Any,
    nu: float = 0.49,
    *,
    source_codes: Iterable[str] | None = None,
    receiver_codes: Iterable[str] | None = None,
    patch_indices_by_source: Mapping[str, Iterable[int]] | None = None,
    progress: bool = True,
) -> dict[str, Array]:
    """
    Build raw Green's-function blocks.

    Parameters
    ----------
    geom_or_legacy
        A Step-1 Geometry object or its legacy dictionary.
    nu
        Poisson's ratio.
    source_codes, receiver_codes
        Optional subsets of source and receiver codes for testing.
    patch_indices_by_source
        Optional mapping from source code to zero-based patch indices, e.g.
        {"i": [0, 5, 10]}.
    progress
        Print progress messages.

    Returns
    -------
    raw
        Dictionary of MATLAB-compatible raw Green's blocks.
    """
    legacy = _get_legacy(geom_or_legacy)
    sources = _make_entities(legacy, for_sources=True, codes=source_codes)
    receivers = _make_entities(legacy, for_sources=False, codes=receiver_codes)

    raw: dict[str, Array] = {}

    for isrc, src in enumerate(sources):
        if src.npatch == 0:
            continue

        patch_indices = _patch_indices_for_source(src, patch_indices_by_source)
        ncols = int(patch_indices.size)

        if progress:
            print(f"Source {isrc + 1}/{len(sources)} code={src.code!r}: {ncols} of {src.npatch} patches")

        lengths = np.sqrt((src.topx - src.botx) ** 2 + (src.topz - src.botz) ** 2)
        dips = -180.0 / np.pi * np.arctan2(src.botz - src.topz, src.botx - src.topx)

        for out_col, k in enumerate(patch_indices):
            m = np.array([1.0e6, lengths[k], -src.botz[k], dips[k], 0.0, src.botx[k], 0.0], dtype=float)

            for rcv in receivers:
                use_self_shift = src.code == rcv.code and rcv.centers_e is not None and src.code != "8"
                use_v_receiver = src.code == "8" and rcv.code == "8"

                if use_self_shift:
                    centers_eval = rcv.centers_e
                elif use_v_receiver:
                    centers_eval = rcv.centers_v
                else:
                    centers_eval = rcv.centers

                if centers_eval is None or np.asarray(centers_eval).size == 0:
                    continue

                xloc = _xloc_from_centers(centers_eval)
                sig11, sig21, sig12, sig22, u11, u21, u12, u22 = make_traction_disp(
                    m=m,
                    xloc=xloc,
                    nu=nu,
                    normvec=rcv.normvec,
                    dipvec=rcv.dipvec,
                )

                if use_v_receiver:
                    _store_field(raw, raw_name("u11", rcv.code, src.code, "v"), u11, out_col, ncols)
                    _store_field(raw, raw_name("u12", rcv.code, src.code, "v"), u12, out_col, ncols)
                    _store_field(raw, raw_name("u21", rcv.code, src.code, "v"), u21, out_col, ncols)
                    _store_field(raw, raw_name("u22", rcv.code, src.code, "v"), u22, out_col, ncols)
                    _store_field(raw, raw_name("sig11", rcv.code, src.code, ""), sig11, out_col, ncols)
                    _store_field(raw, raw_name("sig12", rcv.code, src.code, ""), sig12, out_col, ncols)
                    _store_field(raw, raw_name("sig21", rcv.code, src.code, ""), sig21, out_col, ncols)
                    _store_field(raw, raw_name("sig22", rcv.code, src.code, ""), sig22, out_col, ncols)
                else:
                    _store_all(
                        raw,
                        rcv.code,
                        src.code,
                        "",
                        out_col,
                        ncols,
                        sig11,
                        sig21,
                        sig12,
                        sig22,
                        u11,
                        u21,
                        u12,
                        u22,
                    )

                need_diag_v = src.code == rcv.code and rcv.centers_v is not None and src.code in {"1", "2", "3", "4", "5", "b", "t"}
                if need_diag_v:
                    if k >= rcv.centers_v.shape[1]:
                        continue
                    xlocv = _xloc_from_centers(rcv.centers_v[:, k])
                    _, _, _, _, u11v, u21v, u12v, u22v = make_traction_disp(
                        m=m,
                        xloc=xlocv,
                        nu=nu,
                        normvec=rcv.normvec[:, k],
                        dipvec=rcv.dipvec[:, k],
                    )

                    u11mat = np.asarray(u11, dtype=float).copy()
                    u12mat = np.asarray(u12, dtype=float).copy()
                    u21mat = np.asarray(u21, dtype=float).copy()
                    u22mat = np.asarray(u22, dtype=float).copy()

                    u11mat[k] = u11v[0]
                    u12mat[k] = u12v[0]
                    u21mat[k] = u21v[0]
                    u22mat[k] = u22v[0]

                    _store_field(raw, raw_name("u11", rcv.code, src.code, "v"), u11mat, out_col, ncols)
                    _store_field(raw, raw_name("u12", rcv.code, src.code, "v"), u12mat, out_col, ncols)
                    _store_field(raw, raw_name("u21", rcv.code, src.code, "v"), u21mat, out_col, ncols)
                    _store_field(raw, raw_name("u22", rcv.code, src.code, "v"), u22mat, out_col, ncols)

    return raw


def raw_summary(raw: Mapping[str, Array]) -> dict[str, tuple[int, ...]]:
    """Return raw field shapes."""
    return {key: np.asarray(val).shape for key, val in sorted(raw.items())}


__all__ = ["RawEntity", "build_raw_greens", "raw_name", "raw_summary"]
