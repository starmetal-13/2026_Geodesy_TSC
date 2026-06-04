from __future__ import annotations

import numpy as np
from scipy.io import savemat, loadmat


def make_legacy_save_dict(out) -> dict:
    p = out.params
    g = out.greens
    return {
        "faultdip_trench": p.faultdip_trench,
        "faultdip_bottom": p.faultdip_bottom,
        "x_trench": p.x_trench,
        "x_bottom": p.x_bottom,
        "z_bottom": p.z_bottom,
        "wedge_bot": p.wedge_bot,
        "wedge_top_x": p.wedge_top_x,
        "xpos": g.xpos,
        "Gz": g.Gz,
        "Gx": g.Gx,
        "Gtau": g.Gtau,
        "topx_interface": g.topx_interface,
        "botx_interface": g.botx_interface,
        "botz_interface": g.botz_interface,
        "topz_interface": g.topz_interface,
        "centers_interface": g.centers_interface,
        "Gx_body": g.Gx_body,
        "Gz_body": g.Gz_body,
        "Gexx_body": g.Gexx_body,
        "Gexz_body": g.Gexz_body,
        "Gezz_body": g.Gezz_body,
        "xloc_body": g.xloc_body,
        "mu1": p.mu1,
        "mu2": p.mu2,
        "mu3": p.mu3,
    }


def save_regression(out, filename: str):
    test = {
        "G": out.system.G.toarray() if hasattr(out.system.G, "toarray") else out.system.G,
        "rhs": out.greens.rhs,
        "patch_slips_all": out.greens.patch_slips_all,
        "Gx": out.greens.Gx,
        "Gz": out.greens.Gz,
        "Gtau": out.greens.Gtau,
        "centers_interface": out.greens.centers_interface,
        "xpos": out.greens.xpos,
    }
    savemat(filename, {"test": test})


def compare_with_matlab_regression(out, matlab_regression_file: str, rtol=1e-8, atol=1e-10) -> dict:
    """Compare key Python outputs against a MATLAB regression .mat file."""
    mat = loadmat(matlab_regression_file, squeeze_me=False, struct_as_record=False)
    test = mat["test"][0, 0]
    pairs = {
        "G": (out.system.G.toarray() if hasattr(out.system.G, "toarray") else out.system.G, test.G),
        "rhs": (out.greens.rhs, test.rhs),
        "patch_slips_all": (out.greens.patch_slips_all, test.patch_slips_all),
        "Gx": (out.greens.Gx, test.Gx),
        "Gz": (out.greens.Gz, test.Gz),
        "Gtau": (out.greens.Gtau, test.Gtau),
        "centers_interface": (out.greens.centers_interface, test.centers_interface),
        "xpos": (out.greens.xpos, test.xpos),
    }
    result = {}
    for name, (py, ml) in pairs.items():
        py = np.asarray(py)
        ml = np.asarray(ml)
        ok = np.allclose(py, ml, rtol=rtol, atol=atol, equal_nan=True)
        result[name] = {
            "ok": bool(ok),
            "max_abs_error": float(np.nanmax(np.abs(py - ml))) if py.shape == ml.shape else float("nan"),
            "python_shape": py.shape,
            "matlab_shape": ml.shape,
        }
    return result
