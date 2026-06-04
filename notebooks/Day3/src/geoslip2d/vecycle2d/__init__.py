"""Vendored VECycle2D solver used by GeoSlip2D.

This subpackage contains the Python VECycle2D source so GeoSlip2D can build
viscoelastic earthquake-cycle Green's functions without requiring a separate
external ``vecycle2d`` installation.
"""

from .config import (
    Config,
    GeometryConfig,
    ForwardConfig as ConfigForwardConfig,
    IOConfig,
    PlotConfig,
    InternalConstants,
    default_config,
)
from .build_greens import (
    GeometryOutput,
    VECycleGreens,
    build_greens,
    greens_summary,
    greens_to_mat_dict,
    save_greens_mat,
    load_greens_mat,
)
from .forward import (
    ForwardConfig,
    InterseismicGreens,
    ForwardResult,
    assemble_interseismic_greens,
    default_backslip_from_locking_depth,
    forward_cycle,
    forward_summary,
)

__all__ = [
    "Config",
    "GeometryConfig",
    "ConfigForwardConfig",
    "IOConfig",
    "PlotConfig",
    "InternalConstants",
    "default_config",
    "GeometryOutput",
    "VECycleGreens",
    "build_greens",
    "greens_summary",
    "greens_to_mat_dict",
    "save_greens_mat",
    "load_greens_mat",
    "ForwardConfig",
    "InterseismicGreens",
    "ForwardResult",
    "assemble_interseismic_greens",
    "default_backslip_from_locking_depth",
    "forward_cycle",
    "forward_summary",
]
