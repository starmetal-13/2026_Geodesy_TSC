"""VECycle2D adapter for GeoSlip2D.

This module converts native VECycle2D Green's functions into GeoSlip2D's
canonical :class:`geoslip2d.greens.Greens2D` container.

GeoSlip2D now vendors the Python VECycle2D solver in
``geoslip2d.vecycle2d``.  The adapter can either build Green's functions using
that vendored solver or load an already-built native VECycle object and convert
it to the common ``Greens2D`` format.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from numpy.typing import ArrayLike

from .geometry import InterfaceGeometry, interface_from_arrays
from .greens import Greens2D


@dataclass(slots=True)
class VECycleConfig:
    """Configuration for the VECycle2D adapter.

    Parameters
    ----------
    native_config
        Optional native VECycle2D configuration object.  If omitted,
        ``geoslip2d.vecycle2d.default_config()`` is used.
    config_overrides
        Optional dotted-path overrides applied to the native config before the
        native build.  For example ``{"io.save_output": False}``.
    greens_pickle_file
        Optional path to an already-built native VECycle Green's pickle file.
        If provided with ``mode='load'`` or ``mode='auto'`` and the file exists,
        it is loaded and converted rather than rebuilt.
    mode
        ``'build'``, ``'load'``, or ``'auto'``.  ``'auto'`` loads an existing
        pickle when available and otherwise builds with the native package.
    keep_internals
        Passed through to ``geoslip2d.vecycle2d.build_greens.build_greens``.
    component
        Green's component to expose as canonical ``Ghor``/``Gvert``.  Common
        values are ``'interseismic'``, ``'total'``, ``'cycle'``, ``'elastic'``,
        and ``'steady_state'``.  The adapter searches several likely field-name
        variants and uses the first matching pair.
    output_sign
        Optional sign multiplier applied to both horizontal and vertical Green's
        matrices after conversion. The default is -1.0 so VECycle output
        follows the GeoSlip2D velocity sign convention.
    progress
        Print native progress messages when the native builder supports it.
    """

    native_config: Any | None = None
    config_overrides: Mapping[str, Any] = field(default_factory=dict)
    greens_pickle_file: str | Path | None = None
    mode: str = "build"
    keep_internals: bool = False
    component: str = "interseismic"
    output_sign: float = -1.0
    progress: bool = False
    # Optional compatibility field for older notebooks/scripts. The dispatcher
    # already passes a canonical InterfaceGeometry separately, but accepting this
    # keyword avoids TypeError and lets direct calls provide a conversion fallback.
    fallback_interface: InterfaceGeometry | None = None


def _get_attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _as_1d(a: Any, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _as_matrix(a: Any, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix.")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _component_candidates(component: str) -> list[tuple[str, str]]:
    key = component.lower().replace("-", "_").replace(" ", "_")
    if key in {"interseismic", "inter", "total"}:
        return [("Gx_inter", "Gz_inter"), ("Gx_total", "Gz_total"), ("Gx", "Gz")]
    if key in {"cycle", "transient", "viscoelastic"}:
        return [("Gx_cycle", "Gz_cycle"), ("Gx_transient", "Gz_transient")]
    if key in {"elastic", "elastic_only"}:
        return [("Gx_elastic", "Gz_elastic"), ("Dx_e", "Dz_e"), ("Gx", "Gz")]
    if key in {"steady_state", "steadystate", "ss"}:
        return [("Gx_ss", "Gz_ss"), ("Gx_steady", "Gz_steady")]
    # Let users pass a raw suffix such as "mycase" -> Gx_mycase/Gz_mycase.
    return [(f"Gx_{key}", f"Gz_{key}"), ("Gx", "Gz")]


def _find_greens_pair(native: Any, component: str) -> tuple[np.ndarray, np.ndarray | None, str]:
    # Some forward-cycle outputs put cycle matrices in a nested object.
    containers = [native]
    nested = _get_attr(native, "interseismic_greens", "greens", default=None)
    if nested is not None and nested is not native:
        containers.insert(0, nested)

    for container in containers:
        for gx_name, gz_name in _component_candidates(component):
            gx = _get_attr(container, gx_name, default=None)
            if gx is not None:
                gz = _get_attr(container, gz_name, default=None)
                return _as_matrix(gx, gx_name), None if gz is None else _as_matrix(gz, gz_name), gx_name

    names = []
    for container in containers:
        names.extend([n for n in dir(container) if n.startswith(("Gx", "Gz"))])
    raise AttributeError(
        "Could not find a VECycle Green's matrix pair for component "
        f"{component!r}. Available G-like fields include: {sorted(set(names))[:20]}"
    )


def _find_xobs(native: Any) -> np.ndarray:
    containers = [native]
    nested = _get_attr(native, "interseismic_greens", "greens", default=None)
    if nested is not None and nested is not native:
        containers.insert(0, nested)
    for container in containers:
        x = _get_attr(container, "xobs", "xpos", "x", default=None)
        if x is not None:
            return _as_1d(x, "xobs")
    raise AttributeError("Could not find observation coordinates; expected xobs or xpos.")


def _interface_from_native(
    native: Any,
    fallback: InterfaceGeometry | None = None,
    expected_npatch: int | None = None,
) -> InterfaceGeometry:
    # Prefer geometry carried by the native VECycle object.  The common
    # GeoSlip2D ``interface`` argument may have a different number of patches
    # because VECycle uses its own native geometry configuration internally.
    # A fallback is used only if native geometry cannot be inferred.
    containers = [native]
    for name in ("Geometry", "geometry", "geom", "greens", "interseismic_greens"):
        obj = _get_attr(native, name, default=None)
        if obj is not None and obj is not native:
            containers.append(obj)

    def _matches_expected(geom: InterfaceGeometry) -> bool:
        return expected_npatch is None or geom.n_patch == int(expected_npatch)

    candidate_mismatches: list[tuple[str, int]] = []

    for container in containers:
        topx = _get_attr(container, "topx_interface", default=None)
        topz = _get_attr(container, "topz_interface", default=None)
        botx = _get_attr(container, "botx_interface", default=None)
        botz = _get_attr(container, "botz_interface", default=None)
        if topx is not None and topz is not None and botx is not None and botz is not None:
            geom = interface_from_arrays(
                topx=_as_1d(topx, "topx_interface"),
                topz=np.abs(_as_1d(topz, "topz_interface")),
                botx=_as_1d(botx, "botx_interface"),
                botz=np.abs(_as_1d(botz, "botz_interface")),
                centers=None,
                metadata={"source": "vecycle_native_fields"},
            )
            if _matches_expected(geom):
                return geom
            candidate_mismatches.append(("vecycle_native_fields", geom.n_patch))

    for container in containers:
        centers = _get_attr(container, "centers_interface", default=None)
        if centers is not None:
            cen = np.asarray(centers, dtype=float)
            if cen.shape[0] == 2:
                x = cen[0, :].reshape(-1)
                z = np.abs(cen[1, :].reshape(-1))
            elif cen.shape[1] == 2:
                x = cen[:, 0].reshape(-1)
                z = np.abs(cen[:, 1].reshape(-1))
            else:
                continue
            if x.size < 2:
                raise ValueError("centers_interface needs at least two points when endpoints are unavailable.")
            # Fallback endpoint reconstruction from center-to-center spacing.
            dx = np.gradient(x)
            dz = np.gradient(z)
            topx = x - 0.5 * dx
            topz = z - 0.5 * dz
            botx = x + 0.5 * dx
            botz = z + 0.5 * dz
            geom = interface_from_arrays(
                topx=topx,
                topz=topz,
                botx=botx,
                botz=botz,
                centers=np.column_stack([x, z]),
                metadata={"source": "vecycle_centers_inferred_endpoints"},
            )
            if _matches_expected(geom):
                return geom
            candidate_mismatches.append(("vecycle_centers_inferred_endpoints", geom.n_patch))

    if fallback is not None and _matches_expected(fallback):
        return fallback
    if fallback is not None:
        candidate_mismatches.append(("fallback_interface", fallback.n_patch))

    if expected_npatch is not None:
        # Last-resort compatibility geometry.  Some VECycle components can have a
        # source-column count that does not match the Geometry.centers_interface
        # metadata carried by older/native objects.  Do not attach a wrong-size
        # interface; instead create a simple placeholder with the correct number
        # of patches so inversion/saving can proceed and record the mismatch in
        # metadata.  This makes the geometry issue explicit without crashing the
        # unified workflow.
        n = int(expected_npatch)
        x = np.arange(n, dtype=float)
        return interface_from_arrays(
            topx=x,
            topz=np.zeros(n, dtype=float),
            botx=x + 1.0,
            botz=np.ones(n, dtype=float),
            centers=np.column_stack([x + 0.5, np.full(n, 0.5)]),
            metadata={
                "source": "vecycle_placeholder_index_geometry",
                "reason": "no native/fallback interface matched VECycle Green's column count",
                "expected_npatch": n,
                "candidate_mismatches": candidate_mismatches,
            },
        )

    raise AttributeError(
        "Could not infer interface geometry from native VECycle object. Pass "
        "fallback_interface=... to vecycle_greens_from_native()."
    )


def vecycle_greens_from_native(
    native: Any,
    *,
    component: str = "interseismic",
    fallback_interface: InterfaceGeometry | None = None,
    output_sign: float = -1.0,
    metadata: Mapping[str, Any] | None = None,
) -> Greens2D:
    """Convert a native VECycle2D Green's object to canonical ``Greens2D``."""
    Ghor, Gvert, field_name = _find_greens_pair(native, component)
    xobs = _find_xobs(native)

    if Ghor.shape[0] != xobs.size and Ghor.shape[1] == xobs.size:
        Ghor = Ghor.T
        if Gvert is not None:
            Gvert = Gvert.T

    interface = _interface_from_native(native, fallback_interface, expected_npatch=Ghor.shape[1])

    if Ghor.shape[1] != interface.n_patch:
        raise ValueError(
            f"Converted VECycle Ghor has {Ghor.shape[1]} columns, but inferred interface has "
            f"{interface.n_patch} patches. Pass fallback_interface if the native geometry inference is wrong."
        )

    md = {
        "native_source_type": type(native).__name__,
        "component": component,
        "native_field": field_name,
    }
    if metadata:
        md.update(dict(metadata))

    return Greens2D(
        Ghor=float(output_sign) * Ghor,
        Gvert=None if Gvert is None else float(output_sign) * Gvert,
        xobs=xobs,
        interface=interface,
        source_type="vecycle",
        units="velocity_per_unit_slip_rate",
        sign_convention="geoslip2d_vecycle_velocity_convention",
        metadata=md,
    )


def _install_vecycle2d_pickle_aliases() -> None:
    """Alias vendored modules so old pickles referencing ``vecycle2d`` can load."""
    try:
        from . import vecycle2d as vendored
    except Exception:
        return

    sys.modules.setdefault("vecycle2d", vendored)
    for name in (
        "api",
        "boundaries",
        "build_greens",
        "compile_greens",
        "config",
        "cycle",
        "forward",
        "geometry",
        "okada3d",
        "raw_greens",
        "traction",
    ):
        full = f"geoslip2d.vecycle2d.{name}"
        if full in sys.modules:
            sys.modules.setdefault(f"vecycle2d.{name}", sys.modules[full])


def load_native_vecycle_pickle(filename: str | Path) -> Any:
    """Load an already-built native VECycle2D Green's object from pickle."""
    _install_vecycle2d_pickle_aliases()
    with open(Path(filename), "rb") as f:
        return pickle.load(f)


def _vecycle_component_requires_forward(component: str) -> bool:
    key = component.lower().replace("-", "_").replace(" ", "_")
    return key in {
        "interseismic",
        "inter",
        "total",
        "cycle",
        "transient",
        "viscoelastic",
        "steady_state",
        "steadystate",
        "ss",
    }


def _has_direct_component(native: Any, component: str) -> bool:
    containers = [native]
    nested = _get_attr(native, "interseismic_greens", "greens", default=None)
    if nested is not None and nested is not native:
        containers.insert(0, nested)
    for container in containers:
        for gx_name, gz_name in _component_candidates(component):
            if _get_attr(container, gx_name, default=None) is not None:
                return True
    return False


def _wrap_with_interseismic(native: Any, inter: Any) -> Any:
    """Attach assembled interseismic matrices while preserving native metadata."""
    return SimpleNamespace(
        xpos=getattr(native, "xpos", None),
        Geometry=getattr(native, "Geometry", None),
        geometry=getattr(native, "geometry", None),
        geom=getattr(native, "geom", None),
        interseismic_greens=inter,
        greens=native,
    )


def _apply_dotted_overrides(cfg: Any, overrides: Mapping[str, Any]) -> None:
    for path, value in overrides.items():
        parts = str(path).split(".")
        obj = cfg
        for name in parts[:-1]:
            obj = getattr(obj, name)
        setattr(obj, parts[-1], value)


def build_vecycle_greens(
    interface: InterfaceGeometry | None,
    xobs: ArrayLike | None = None,
    config: VECycleConfig | None = None,
) -> Greens2D:
    """Build or load native VECycle2D Green's functions and convert to ``Greens2D``.

    The build path uses the vendored ``geoslip2d.vecycle2d`` solver, so no
    separate external ``vecycle2d`` package is required.
    """
    if config is None:
        config = VECycleConfig()

    mode = config.mode.lower()
    if mode not in {"build", "load", "auto"}:
        raise ValueError("VECycleConfig.mode must be 'build', 'load', or 'auto'.")

    native = None
    pickle_file = None if config.greens_pickle_file is None else Path(config.greens_pickle_file)

    if mode in {"load", "auto"} and pickle_file is not None and pickle_file.is_file():
        native = load_native_vecycle_pickle(pickle_file)
    elif mode == "load":
        raise FileNotFoundError(f"VECycle pickle file not found: {pickle_file}")

    native_cfg = config.native_config

    if native is None:
        from .vecycle2d import default_config
        from .vecycle2d.build_greens import build_greens as native_build_greens

        native_cfg = native_cfg if native_cfg is not None else default_config()
        _apply_dotted_overrides(native_cfg, config.config_overrides)
        native = native_build_greens(native_cfg, keep_internals=config.keep_internals, progress=config.progress)

        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(native, f)

    native_for_conversion = native
    if _vecycle_component_requires_forward(config.component) and not _has_direct_component(native, config.component):
        try:
            from .vecycle2d.forward import assemble_interseismic_greens

            forward_cfg = None if native_cfg is None else getattr(native_cfg, "forward", None)
            inter = assemble_interseismic_greens(native, forward_cfg)
            native_for_conversion = _wrap_with_interseismic(native, inter)
        except Exception as exc:
            if config.component.lower().replace("-", "_").replace(" ", "_") in {"interseismic", "inter", "total"}:
                raise RuntimeError(
                    "Built/loaded a VECycle object, but could not assemble the interseismic "
                    "Green's matrices needed for component='interseismic'."
                ) from exc
            # Other components may still be available directly on the native object.

    # VECycle builds its own native interface, and its source-patch count may
    # differ from the shared interface used by homogeneous/layered/wedge tests.
    # The converter now tries native geometry first; this fallback is used only
    # for legacy/load objects that do not carry enough geometry metadata.
    conversion_interface = config.fallback_interface if config.fallback_interface is not None else interface
    greens = vecycle_greens_from_native(
        native_for_conversion,
        component=config.component,
        fallback_interface=conversion_interface,
        output_sign=config.output_sign,
        metadata={"mode": mode},
    )

    if xobs is not None:
        xobs_arr = np.asarray(xobs, dtype=float).reshape(-1)
        if xobs_arr.size and (xobs_arr.size != greens.xobs.size or np.nanmax(np.abs(xobs_arr - greens.xobs)) > 0):
            greens = greens.interp_to(xobs_arr)
    return greens


__all__ = [
    "VECycleConfig",
    "build_vecycle_greens",
    "vecycle_greens_from_native",
    "load_native_vecycle_pickle",
]
