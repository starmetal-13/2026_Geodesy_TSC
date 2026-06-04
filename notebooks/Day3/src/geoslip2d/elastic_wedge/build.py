from __future__ import annotations

from dataclasses import dataclass
from scipy.io import savemat

from .params import ElasticWedgeParams, default_elastic_wedge_params
from .geometry import make_elastic_wedge_geometry_struct
from .kernels import compute_boundary_kernels
from .assemble import assemble_elastic_wedge_system
from .solve import solve_interface_greens
from .io import make_legacy_save_dict, save_regression


@dataclass
class ElasticWedgeOutput:
    params: ElasticWedgeParams
    geom: object
    kernels: object
    system: object
    greens: object
    legacySave: dict


def build_greens_elastic_wedge(params: ElasticWedgeParams | None = None, geometry_only: bool = False):
    """
    Build Green's functions and geometry for the elastic wedge model.
    If geometry_only is True, only the geometry object is returned (for plotting),
    and all other calculations are skipped.
    """
    if params is None:
        params = default_elastic_wedge_params()
    if params.compute_body:
        raise NotImplementedError("compute_body=True is not supported because build_elastic_wedge_body.m was not included in the MATLAB phase-2 refactor.")

    if params.verbose:
        print("Building elastic wedge geometry...")
    geom = make_elastic_wedge_geometry_struct(params)

    if geometry_only:
        return geom

    if params.verbose:
        print("Computing pairwise boundary kernels...")
    kernels = compute_boundary_kernels(geom, params)

    if params.verbose:
        print("Assembling boundary-condition system...")
    system = assemble_elastic_wedge_system(geom, kernels, params)

    if params.verbose:
        print("Solving interface-slip Green's functions...")
    greens = solve_interface_greens(system, geom, params)

    out = ElasticWedgeOutput(params, geom, kernels, system, greens, {})
    out.legacySave = make_legacy_save_dict(out)

    if params.save_regression:
        save_regression(out, params.regression_filename)
    if params.save_output:
        savemat(params.savename + ".mat", out.legacySave)
    return out
