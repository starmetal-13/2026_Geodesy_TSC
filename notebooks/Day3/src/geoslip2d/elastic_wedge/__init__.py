"""Vendored elastic-wedge solver used by GeoSlip2D's wedge backend."""

from .params import ElasticWedgeParams, KernelParams, default_elastic_wedge_params
from .build import ElasticWedgeOutput, build_greens_elastic_wedge
from .geometry import make_elastic_wedge_geometry_struct, make_geometry_elastic_wedge
from .plotting import plot_elastic_wedge_geometry

__all__ = [
    "ElasticWedgeParams",
    "KernelParams",
    "default_elastic_wedge_params",
    "ElasticWedgeOutput",
    "build_greens_elastic_wedge",
    "make_elastic_wedge_geometry_struct",
    "make_geometry_elastic_wedge",
    "plot_elastic_wedge_geometry",
]
