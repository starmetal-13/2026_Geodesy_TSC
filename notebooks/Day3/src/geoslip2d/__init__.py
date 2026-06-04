"""GeoSlip2D core package scaffold."""

from .geometry import InterfaceConfig, InterfaceGeometry, interface_from_arrays, make_interface_geometry, make_interface_geometry_legacy
from .greens import Greens2D, greens_from_matdict
from .io import load_greens, save_greens
from .plotting import plot_interface, plot_greens_summary, plot_greens_columns
from .homogeneous import HomogeneousConfig, build_homogeneous_greens, edge_disp_finite
from .layered import LayeredConfig, MultiLayerOptions, build_layered_greens, make_dispG_multilayer, multi_layer_tapered, validate_layer_model
from .wedge import WedgeConfig, build_wedge_greens, wedge_greens_from_native, plot_wedge_geometry
from .vecycle import VECycleConfig, build_vecycle_greens, vecycle_greens_from_native, load_native_vecycle_pickle
from .build import GreensMethod, build_greens, normalize_greens_method

from .inversion import (
    ProfileObservations,
    ProfileProjectionConfig,
    SlipInversionConfig,
    fit_profile_slip,
    run_profile_slip_inversion,
    load_profile_data,
    project_data_to_profile,
    make_smoothing_matrix,
)

__all__ = [
    "InterfaceConfig",
    "InterfaceGeometry",
    "Greens2D",
    "interface_from_arrays",
    "make_interface_geometry",
    "make_interface_geometry_legacy",
    "greens_from_matdict",
    "load_greens",
    "save_greens",
    "plot_interface",
    "plot_greens_summary",
    "plot_greens_columns",
    "HomogeneousConfig",
    "build_homogeneous_greens",
    "edge_disp_finite",
    "LayeredConfig",
    "MultiLayerOptions",
    "build_layered_greens",
    "make_dispG_multilayer",
    "multi_layer_tapered",
    "validate_layer_model",
    "WedgeConfig",
    "build_wedge_greens",
    "wedge_greens_from_native",
    "plot_wedge_geometry",
    "VECycleConfig",
    "build_vecycle_greens",
    "vecycle_greens_from_native",
    "load_native_vecycle_pickle",
    "GreensMethod",
    "build_greens",
    "normalize_greens_method",
    "ProfileObservations",
    "ProfileProjectionConfig",
    "SlipInversionConfig",
    "fit_profile_slip",
    "run_profile_slip_inversion",
    "load_profile_data",
    "project_data_to_profile",
    "make_smoothing_matrix",
]
