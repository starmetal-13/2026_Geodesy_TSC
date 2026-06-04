"""Unified Green's-function builder interface for GeoSlip2D.

This module provides a small dispatcher so notebooks and scripts can use the
same public call pattern for all Green's-function backends.  At present the
homogeneous backend is implemented.  Layered, wedge, and viscoelastic-cycle
backends are intentionally reserved names and will be wired in as their modules
are ported into the package.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

from numpy.typing import ArrayLike

from .geometry import InterfaceGeometry
from .greens import Greens2D
from .homogeneous import HomogeneousConfig, build_homogeneous_greens
from .layered import LayeredConfig, build_layered_greens
from .wedge import WedgeConfig, build_wedge_greens
from .vecycle import VECycleConfig, build_vecycle_greens

GreensMethod = Literal["homogeneous", "layered", "wedge", "vecycle", "viscoelastic"]

_METHOD_ALIASES: dict[str, str] = {
    "homogeneous": "homogeneous",
    "homogeneous_halfspace": "homogeneous",
    "homogeneous-halfspace": "homogeneous",
    "halfspace": "homogeneous",
    "elastic_halfspace": "homogeneous",
    "elastic-halfspace": "homogeneous",
    "layered": "layered",
    "multilayer": "layered",
    "multi_layer": "layered",
    "wedge": "wedge",
    "compliant_wedge": "wedge",
    "compliant-wedge": "wedge",
    "vecycle": "vecycle",
    "ve_cycle": "vecycle",
    "ve-cycle": "vecycle",
    "viscoelastic": "vecycle",
    "viscoelastic_cycle": "vecycle",
    "viscoelastic-cycle": "vecycle",
}


def normalize_greens_method(method: str) -> str:
    """Normalize a user-provided Green's-function backend name.

    Parameters
    ----------
    method
        Backend name or alias.  The comparison is case-insensitive and treats
        spaces as underscores.

    Returns
    -------
    str
        Canonical backend name.
    """
    key = str(method).strip().lower().replace(" ", "_")
    if key not in _METHOD_ALIASES:
        allowed = ", ".join(sorted(set(_METHOD_ALIASES.values())))
        raise ValueError(f"Unknown Green's-function method '{method}'. Available methods: {allowed}.")
    return _METHOD_ALIASES[key]


def build_greens(
    method: str,
    interface: InterfaceGeometry,
    xobs: ArrayLike,
    config: Any | None = None,
    *,
    progress: bool | None = None,
    **kwargs: Any,
) -> Greens2D:
    """Build Green's functions using a named backend.

    This is the package-level dispatcher.  It lets example notebooks and future
    workflows use one call signature regardless of backend::

        greens = build_greens("homogeneous", interface, xobs, config)

    For convenience, homogeneous keyword arguments can be passed directly when
    ``config`` is omitted::

        greens = build_greens("homogeneous", interface, xobs, length_override=5)

    Parameters
    ----------
    method
        Backend name.  Currently implemented: ``"homogeneous"``, ``"layered"``, ``"wedge"``, and the ``"vecycle"`` adapter.
    interface
        Canonical GeoSlip2D interface geometry.
    xobs
        Observation positions in km along the 1-D profile.
    config
        Backend-specific configuration object.  For ``"homogeneous"``, use
        :class:`geoslip2d.homogeneous.HomogeneousConfig`. For ``"layered"``, use
        :class:`geoslip2d.layered.LayeredConfig`. For ``"wedge"``, use
        :class:`geoslip2d.wedge.WedgeConfig`. For ``"vecycle"``, use
        :class:`geoslip2d.vecycle.VECycleConfig`.
    **kwargs
        Backend-specific configuration keywords used only when ``config`` is
        omitted.
    progress
        Optional override for backend progress printing. Set ``progress=True``
        to print messages such as ``completed 1 of 25 patches`` while
        Green's functions are assembled.

    Returns
    -------
    Greens2D
        Canonical Green's-function object.
    """
    backend = normalize_greens_method(method)

    if backend == "homogeneous":
        if config is None:
            if progress is not None:
                kwargs["progress"] = progress
            config = HomogeneousConfig(**kwargs)
        elif kwargs:
            raise ValueError("Pass either a config object or keyword settings, not both, except progress.")
        if not isinstance(config, HomogeneousConfig):
            raise TypeError("Homogeneous backend requires HomogeneousConfig.")
        if progress is not None:
            config = replace(config, progress=progress)
        return build_homogeneous_greens(interface, xobs, config)

    if backend == "layered":
        if config is None:
            if progress is not None:
                kwargs["progress"] = progress
            config = LayeredConfig(**kwargs)
        elif kwargs:
            raise ValueError("Pass either a config object or keyword settings, not both, except progress.")
        if not isinstance(config, LayeredConfig):
            raise TypeError("Layered backend requires LayeredConfig.")
        if progress is not None:
            config = replace(config, progress=progress)
        return build_layered_greens(interface, xobs, config)


    if backend == "wedge":
        if config is None:
            if progress is not None:
                kwargs["progress"] = progress
            config = WedgeConfig(**kwargs)
        elif kwargs:
            raise ValueError("Pass either a config object or keyword settings, not both, except progress.")
        if not isinstance(config, WedgeConfig):
            raise TypeError("Wedge backend requires WedgeConfig.")
        if progress is not None:
            config = replace(config, progress=progress)
        return build_wedge_greens(interface, xobs, config)

    if backend == "vecycle":
        if config is None:
            if progress is not None:
                kwargs["progress"] = progress
            config = VECycleConfig(**kwargs)
        elif kwargs:
            raise ValueError("Pass either a config object or keyword settings, not both, except progress.")
        if not isinstance(config, VECycleConfig):
            raise TypeError("VECycle backend requires VECycleConfig.")
        if progress is not None:
            config = replace(config, progress=progress)
        return build_vecycle_greens(interface, xobs, config)

    raise NotImplementedError(
        f"The '{backend}' Green's-function backend is reserved but not yet implemented in this package scaffold. "
        "Implemented methods are currently 'homogeneous', 'layered', 'wedge', and the 'vecycle' adapter."
    )


__all__ = ["GreensMethod", "build_greens", "normalize_greens_method"]
