from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class BoundaryID(IntEnum):
    SURFACE_LEFT = 0
    SURFACE_WEDGE = 1
    SURFACE_UPPER_PLATE = 2
    WEDGE_BACKSTOP = 3
    MEGATHRUST = 4
    SLAB_EXTENSION = 5


BOUNDARY_NAMES = (
    "surface_left",
    "surface_wedge",
    "surface_upper_plate",
    "wedge_backstop",
    "megathrust",
    "slab_extension",
)


class MaterialID(IntEnum):
    CONTINENTAL_CRUST = 0
    COMPLIANT_WEDGE = 1
    SUBDUCTING_SLAB = 2


MATERIAL_NAMES = (
    "continental_crust",
    "compliant_wedge",
    "subducting_slab",
)


@dataclass
class KernelParams:
    type: str = "okada3d"
    supports_dip_slip: bool = True
    supports_opening: bool = True


@dataclass
class ElasticWedgeParams:
    # Geometry
    faultdip_trench: float = 3.0
    faultdip_bottom: float = 20.0
    x_trench: float = 0.0
    x_bottom: float = 200.0
    z_bottom: float = 50.0
    wedge_bot: float = 12.0
    wedge_top_x: float = 90.0
    L_slab: float = 50.0
    pL: float = 1.0

    # Domain and offsets
    W: float = 200.0
    shift: float = 1e5
    self_offset: float = 1e-3
    okada_length: float = 1e6

    # Elastic properties
    mu1: float = 1.0
    mu2: float = 1.0
    mu3: float = 1.0
    nu: float = 0.25

    # Optional body fields, not yet ported from missing MATLAB body script
    compute_body: bool = False
    minx: float = 0.0
    maxx: float = 250.0
    dx: float = 2.0
    minz: float = 1.0
    maxz: float = 50.0
    dz: float = 2.0

    # Linear algebra / performance
    use_sparse: bool = False
    solve_all_rhs: bool = True

    # I/O / diagnostics
    savename: str = "GFs_homogeneous_hik_geom_python"
    save_output: bool = False
    plot_geometry: bool = False
    verbose: bool = True
    save_regression: bool = False
    regression_filename: str = "elastic_wedge_regression_test_python.mat"

    kernel: KernelParams = field(default_factory=KernelParams)

    @property
    def mu_vector(self):
        return [self.mu1, self.mu2, self.mu3]


def default_elastic_wedge_params() -> ElasticWedgeParams:
    return ElasticWedgeParams()
