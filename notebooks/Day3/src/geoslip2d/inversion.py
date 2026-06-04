"""Profile slip-inversion utilities for GeoSlip2D.

This module is the package version of the original profile slip-inversion
workflow.  The key change is that the inversion consumes the canonical
:class:`geoslip2d.greens.Greens2D` object produced by any Green's-function
backend: homogeneous, layered, wedge, or viscoelastic-cycle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.optimize import lsq_linear, nnls
from scipy.io import savemat

from .greens import Greens2D
from .io import load_greens


@dataclass(slots=True)
class ProfileProjectionConfig:
    """Settings for projecting lon/lat velocity data onto a 1-D profile."""

    endpoints: tuple[float, float, float, float] = (-120.0, 45.0, -116.0, 44.09)
    project_dist: float = 20.0
    trench_llh: tuple[float, float] = (-120.0, 45.0)
    horizontal_component: str = "profile_parallel"  # or "horizontal_magnitude"


@dataclass(slots=True)
class SlipInversionConfig:
    """Settings for the damped 1-D slip inversion."""

    alpha: float = 1.0
    smoothing_order: str = "second"  # "first", "second", or "none"
    use_vertical: bool = True
    data_has_vertical: bool = True
    inversion_mode: str = "forward_slip"  # or "slip_deficit"
    solver_type: str = "nonnegative"  # "unbounded", "nonnegative", or "bounded"
    lower_bound: float | ArrayLike = 0.0
    upper_bound: float | ArrayLike = np.inf
    solve_horizontal_shift: bool = False
    solve_vertical_shift: bool = False
    save_results: bool = False
    results_filename: str | Path = "profile_slip_inversion_results.mat"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProfileObservations:
    """Profile observations used directly by the inversion.

    Horizontal observations are required.  Vertical observations are optional.
    ``x_hor`` and ``x_vert`` are distances from the trench in km, in the same
    coordinate system as ``Greens2D.xobs``.
    """

    x_hor: np.ndarray
    v_hor: np.ndarray
    sig_hor: np.ndarray
    x_vert: Optional[np.ndarray] = None
    v_vert: Optional[np.ndarray] = None
    sig_vert: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x_hor = _as_1d(self.x_hor, "x_hor")
        self.v_hor = _as_1d(self.v_hor, "v_hor")
        self.sig_hor = _as_1d(self.sig_hor, "sig_hor")
        if not (self.x_hor.size == self.v_hor.size == self.sig_hor.size):
            raise ValueError("x_hor, v_hor, and sig_hor must have the same length.")
        if np.any(self.sig_hor <= 0):
            raise ValueError("sig_hor must be positive.")

        has_any_vert = self.x_vert is not None or self.v_vert is not None or self.sig_vert is not None
        if has_any_vert:
            if self.x_vert is None or self.v_vert is None or self.sig_vert is None:
                raise ValueError("x_vert, v_vert, and sig_vert must all be provided for vertical data.")
            self.x_vert = _as_1d(self.x_vert, "x_vert")
            self.v_vert = _as_1d(self.v_vert, "v_vert")
            self.sig_vert = _as_1d(self.sig_vert, "sig_vert")
            if not (self.x_vert.size == self.v_vert.size == self.sig_vert.size):
                raise ValueError("x_vert, v_vert, and sig_vert must have the same length.")
            if np.any(self.sig_vert <= 0):
                raise ValueError("sig_vert must be positive.")

    @property
    def has_vertical(self) -> bool:
        return self.x_vert is not None and self.v_vert is not None and self.sig_vert is not None


def run_profile_slip_inversion(
    data_filename: str | Path,
    greens: Greens2D | str | Path,
    *,
    projection_config: ProfileProjectionConfig | None = None,
    inversion_config: SlipInversionConfig | None = None,
    mat_struct_name: str | None = None,
) -> dict[str, Any]:
    """Load lon/lat profile data, prepare Green's functions, and solve slip.

    This convenience wrapper preserves the original notebook workflow while
    allowing ``greens`` to be either a loaded :class:`Greens2D` object or a path
    to a saved ``.npz``/``.mat`` Green's file.
    """
    if projection_config is None:
        projection_config = ProfileProjectionConfig()
    if inversion_config is None:
        inversion_config = SlipInversionConfig()
    greens_obj = _coerce_greens(greens, mat_struct_name=mat_struct_name)

    data_raw = load_profile_data(data_filename, data_has_vertical=inversion_config.data_has_vertical)
    data, good_mask = filter_profile_data(data_raw, use_vertical=inversion_config.use_vertical)
    profile_dict = project_data_to_profile(data, projection_config, use_vertical=inversion_config.use_vertical)
    observations = observations_from_projected_profile(profile_dict, use_vertical=inversion_config.use_vertical)
    out = fit_profile_slip(observations, greens_obj, inversion_config)
    out.update({
        "data_raw": data_raw,
        "data": data,
        "good_mask_original": good_mask,
        "projection": profile_dict,
        "projection_config": asdict(projection_config),
    })
    if inversion_config.save_results:
        save_inversion_result(inversion_config.results_filename, out)
    return out


def fit_profile_slip(
    observations: ProfileObservations,
    greens: Greens2D | str | Path,
    cfg: SlipInversionConfig | None = None,
    *,
    mat_struct_name: str | None = None,
) -> dict[str, Any]:
    """Solve for slip using already-profiled observations and a ``Greens2D`` object."""
    if cfg is None:
        cfg = SlipInversionConfig()
    validate_inversion_config(cfg)
    greens_obj = _coerce_greens(greens, mat_struct_name=mat_struct_name)

    Ghor = greens_obj.interp_to(observations.x_hor).Ghor
    green_sign, model_label = _mode_sign_and_label(cfg.inversion_mode)
    Ghor = green_sign * Ghor

    data_blocks = [observations.v_hor]
    sig_blocks = [observations.sig_hor]
    G_blocks = [Ghor]

    use_vertical = cfg.use_vertical and observations.has_vertical
    if cfg.use_vertical and not observations.has_vertical:
        raise ValueError("use_vertical=True, but ProfileObservations has no vertical data.")
    if use_vertical:
        if greens_obj.Gvert is None:
            raise ValueError("use_vertical=True, but Greens2D.Gvert is None.")
        Gvert = greens_obj.interp_to(observations.x_vert).Gvert
        if Gvert is None:
            raise ValueError("Vertical Green's functions are not available.")
        G_blocks.append(green_sign * Gvert)
        data_blocks.append(observations.v_vert)
        sig_blocks.append(observations.sig_vert)

    G_slip = np.vstack(G_blocks)
    d = np.concatenate(data_blocks)
    sig = np.concatenate(sig_blocks)
    nel = greens_obj.n_patch
    n_h = observations.v_hor.size

    G, offset_names, offset_columns = _append_reference_shift_columns(
        G_slip,
        n_h=n_h,
        n_total=d.size,
        use_vertical=use_vertical,
        solve_horizontal_shift=cfg.solve_horizontal_shift,
        solve_vertical_shift=cfg.solve_vertical_shift,
    )

    L_slip = make_smoothing_matrix(nel, cfg.smoothing_order)
    L = _expand_regularization_for_offsets(L_slip, n_offset=len(offset_names))
    W = sparse.diags(1.0 / sig, 0, shape=(sig.size, sig.size), format="csr")
    GG = sparse.vstack([W @ sparse.csr_matrix(G), cfg.alpha * L], format="csr")
    dd = np.concatenate([W @ d, np.zeros(L.shape[0])])

    model_hat = _solve_augmented_system(GG, dd, cfg, n_slip=nel, n_offset=len(offset_names))
    slip_hat = model_hat[:nel]
    offset_hat = model_hat[nel:]
    offset_dict = {name: float(value) for name, value in zip(offset_names, offset_hat)}
    horizontal_shift = offset_dict.get("horizontal", 0.0)
    vertical_shift = offset_dict.get("vertical", 0.0)

    dhat_slip = G_slip @ slip_hat
    dhat = G @ model_hat
    residual = d - dhat
    wrms = float(np.sqrt(np.mean((residual / sig) ** 2)))
    roughness = float(np.linalg.norm(L_slip @ slip_hat))

    out = {
        "settings": asdict(cfg),
        "observations": observations,
        "greens": greens_obj,
        "G": G,
        "G_slip": G_slip,
        "d": d,
        "sig": sig,
        "GG": GG,
        "dd": dd,
        "L": L,
        "L_slip": L_slip,
        "model_hat": model_hat,
        "slip_hat": slip_hat,
        "offset_hat": offset_hat,
        "offset_names": offset_names,
        "offset_columns": offset_columns,
        "horizontal_shift": horizontal_shift,
        "vertical_shift": vertical_shift,
        "dhat": dhat,
        "dhat_slip": dhat_slip,
        "dhat_hor": dhat[:n_h],
        "dhat_hor_slip": dhat_slip[:n_h],
        "residual": residual,
        "residual_hor": residual[:n_h],
        "wrms": wrms,
        "roughness": roughness,
        "model_label": model_label,
    }
    if use_vertical:
        out.update({
            "dhat_vert": dhat[n_h:],
            "dhat_vert_slip": dhat_slip[n_h:],
            "residual_vert": residual[n_h:],
        })
    else:
        out.update({
            "dhat_vert": np.array([]),
            "dhat_vert_slip": np.array([]),
            "residual_vert": np.array([]),
        })
    if cfg.save_results:
        save_inversion_result(cfg.results_filename, out)
    return out


def load_profile_data(data_filename: str | Path, data_has_vertical: bool = True) -> dict[str, np.ndarray]:
    """Load lon/lat velocity data from a whitespace- or comma-delimited text file.

    Expected columns with vertical data are ``lon lat Ve Vn Vu Sige Sign Sigu``.
    Expected columns without vertical data are ``lon lat Ve Vn Sige Sign``.
    """
    data_filename = Path(data_filename)
    if not data_filename.is_file():
        raise FileNotFoundError(f"Data file does not exist: {data_filename}")
    try:
        A = np.loadtxt(data_filename, delimiter=",")
    except ValueError:
        A = np.loadtxt(data_filename)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if data_has_vertical and A.shape[1] < 8:
        raise ValueError("data_has_vertical=True, but the data file has fewer than 8 columns.")
    if not data_has_vertical and A.shape[1] < 6:
        raise ValueError("data_has_vertical=False, but the data file has fewer than 6 columns.")
    data = {"lon": A[:, 0], "lat": A[:, 1], "Ve": A[:, 2], "Vn": A[:, 3]}
    if data_has_vertical:
        data.update({"Vu": A[:, 4], "Sige": A[:, 5], "Sign": A[:, 6], "Sigu": A[:, 7]})
    else:
        data.update({"Vu": np.array([]), "Sige": A[:, 4], "Sign": A[:, 5], "Sigu": np.array([])})
    return data


def filter_profile_data(data: dict[str, np.ndarray], use_vertical: bool = True) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Remove rows with non-finite values or non-positive uncertainties."""
    good = (
        np.isfinite(data["lon"]) & np.isfinite(data["lat"]) &
        np.isfinite(data["Ve"]) & np.isfinite(data["Vn"]) &
        np.isfinite(data["Sige"]) & np.isfinite(data["Sign"]) &
        (data["Sige"] > 0) & (data["Sign"] > 0)
    )
    if use_vertical:
        good = good & np.isfinite(data["Vu"]) & np.isfinite(data["Sigu"]) & (data["Sigu"] > 0)
    filtered = {k: np.asarray(v)[good] if np.asarray(v).size == good.size else np.asarray(v) for k, v in data.items()}
    return filtered, good


def project_data_to_profile(
    data: dict[str, np.ndarray],
    cfg: ProfileProjectionConfig,
    *,
    use_vertical: bool = True,
) -> dict[str, np.ndarray | str]:
    """Project lon/lat velocity data onto a 1-D profile."""
    lon = data["lon"]
    lat = data["lat"]
    origin = np.array([np.nanmean(lon), np.nanmean(lat)])
    endpoints = np.asarray(cfg.endpoints, dtype=float)
    trench_llh = np.asarray(cfg.trench_llh, dtype=float)

    xy_gps = llh2local(np.column_stack([lon, lat]), origin)
    xy_endpoints = llh2local(np.array([[endpoints[0], endpoints[1]], [endpoints[2], endpoints[3]]]), origin)
    xy_trench = llh2local(trench_llh.reshape(1, 2), origin)[0]

    p1, p2 = xy_endpoints
    profile_vec = p2 - p1
    profile_length = np.hypot(profile_vec[0], profile_vec[1])
    if profile_length <= 0:
        raise ValueError("Profile endpoints must be distinct.")
    profile_hat = profile_vec / profile_length
    normal_hat = np.array([-profile_hat[1], profile_hat[0]])

    rel_xy = xy_gps - p1
    along_dist = rel_xy @ profile_hat
    perp_dist = rel_xy @ normal_hat
    trench_along = (xy_trench - p1) @ profile_hat
    in_profile = (np.abs(perp_dist) <= cfg.project_dist) & (along_dist >= 0) & (along_dist <= profile_length)
    if not np.any(in_profile):
        raise ValueError("No data are within project_dist of the profile segment.")

    xprof = along_dist[in_profile] - trench_along
    Ve = data["Ve"]
    Vn = data["Vn"]
    Sige = data["Sige"]
    Sign = data["Sign"]
    if cfg.horizontal_component.lower() == "profile_parallel":
        v_hor = Ve[in_profile] * profile_hat[0] + Vn[in_profile] * profile_hat[1]
        sig_hor = np.sqrt((Sige[in_profile] * profile_hat[0]) ** 2 + (Sign[in_profile] * profile_hat[1]) ** 2)
        label = "profile-parallel velocity"
    elif cfg.horizontal_component.lower() == "horizontal_magnitude":
        v_hor = np.hypot(Ve[in_profile], Vn[in_profile])
        speed = np.maximum(v_hor, np.finfo(float).eps)
        sig_hor = np.sqrt((Ve[in_profile] / speed * Sige[in_profile]) ** 2 + (Vn[in_profile] / speed * Sign[in_profile]) ** 2)
        label = "horizontal speed"
    else:
        raise ValueError("horizontal_component must be 'profile_parallel' or 'horizontal_magnitude'.")

    out: dict[str, Any] = {
        "origin": origin,
        "xy_gps": xy_gps,
        "xy_endpoints": xy_endpoints,
        "xy_trench": xy_trench,
        "profile_hat": profile_hat,
        "normal_hat": normal_hat,
        "along_dist": along_dist,
        "perp_dist": perp_dist,
        "in_profile": in_profile,
        "x_hor": xprof,
        "v_hor": v_hor,
        "sig_hor": sig_hor,
        "horizontal_label": label,
    }
    if use_vertical:
        out.update({"x_vert": xprof.copy(), "v_vert": data["Vu"][in_profile], "sig_vert": data["Sigu"][in_profile]})
    return out


def observations_from_projected_profile(profile: dict[str, Any], *, use_vertical: bool = True) -> ProfileObservations:
    """Convert a projected-profile dictionary to :class:`ProfileObservations`."""
    return ProfileObservations(
        x_hor=profile["x_hor"],
        v_hor=profile["v_hor"],
        sig_hor=profile["sig_hor"],
        x_vert=profile.get("x_vert") if use_vertical else None,
        v_vert=profile.get("v_vert") if use_vertical else None,
        sig_vert=profile.get("sig_vert") if use_vertical else None,
        metadata={"horizontal_label": profile.get("horizontal_label", "horizontal velocity")},
    )


def llh2local(llh: ArrayLike, origin: ArrayLike) -> np.ndarray:
    """Convert lon/lat coordinates to local x/y coordinates in km.

    This is a direct Python port of the ``llh2local.m`` helper commonly used in
    the original profile-inversion notebooks.
    """
    arr = np.asarray(llh, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 2 and arr.shape[1] != 2:
        arr = arr.T
    if arr.shape[1] < 2:
        raise ValueError("llh must contain longitude and latitude columns.")

    lon = arr[:, 0] * np.pi / 180.0
    lat = arr[:, 1] * np.pi / 180.0
    origin = np.asarray(origin, dtype=float).reshape(-1)
    if origin.size < 2:
        raise ValueError("origin must be [lon0, lat0].")
    origin_rad = origin[:2] * np.pi / 180.0

    a = 6378137.0
    e = 0.08209443794970
    xy = np.zeros((lat.size, 2), dtype=float)
    z = lat != 0.0
    dlambda = lon[z] - origin_rad[0]
    latz = lat[z]
    M = a * (
        (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256) * latz
        - (3 * e**2 / 8 + 3 * e**4 / 32 + 45 * e**6 / 1024) * np.sin(2 * latz)
        + (15 * e**4 / 256 + 45 * e**6 / 1024) * np.sin(4 * latz)
        - (35 * e**6 / 3072) * np.sin(6 * latz)
    )
    M0 = a * (
        (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256) * origin_rad[1]
        - (3 * e**2 / 8 + 3 * e**4 / 32 + 45 * e**6 / 1024) * np.sin(2 * origin_rad[1])
        + (15 * e**4 / 256 + 45 * e**6 / 1024) * np.sin(4 * origin_rad[1])
        - (35 * e**6 / 3072) * np.sin(6 * origin_rad[1])
    )
    N = a / np.sqrt(1 - e**2 * np.sin(latz) ** 2)
    E = dlambda * np.sin(latz)
    cot_lat = 1.0 / np.tan(latz)
    xy[z, 0] = N * cot_lat * np.sin(E)
    xy[z, 1] = M - M0 + N * cot_lat * (1 - np.cos(E))
    xy[~z, 0] = a * (lon[~z] - origin_rad[0])
    xy[~z, 1] = -M0
    return xy / 1000.0


def make_smoothing_matrix(nel: int, smoothing_order: str = "second") -> sparse.csr_matrix:
    """Create first- or second-difference smoothing matrix."""
    smoothing_order = smoothing_order.lower()
    if smoothing_order == "none":
        return sparse.csr_matrix((0, nel))
    if smoothing_order == "first":
        if nel < 2:
            return sparse.csr_matrix((0, nel))
        e = np.ones(nel)
        return sparse.diags([-e, e], [0, 1], shape=(nel - 1, nel), format="csr")
    if smoothing_order == "second":
        if nel < 3:
            return sparse.csr_matrix((0, nel))
        e = np.ones(nel)
        return sparse.diags([e, -2 * e, e], [0, 1, 2], shape=(nel - 2, nel), format="csr")
    raise ValueError("smoothing_order must be 'first', 'second', or 'none'.")


def save_inversion_result(filename: str | Path, out: dict[str, Any]) -> None:
    """Save a compact MATLAB-readable inversion result file."""
    filename = Path(filename)
    obs = out["observations"]
    greens = out["greens"]
    mdict = {
        "slip_hat": out["slip_hat"],
        "model_hat": out.get("model_hat", out["slip_hat"]),
        "offset_hat": out.get("offset_hat", np.array([])),
        "horizontal_shift": out.get("horizontal_shift", 0.0),
        "vertical_shift": out.get("vertical_shift", 0.0),
        "dhat": out["dhat"],
        "dhat_slip": out.get("dhat_slip", out["dhat"]),
        "dhat_hor": out["dhat_hor"],
        "dhat_hor_slip": out.get("dhat_hor_slip", out["dhat_hor"]),
        "dhat_vert": out["dhat_vert"],
        "dhat_vert_slip": out.get("dhat_vert_slip", out["dhat_vert"]),
        "residual": out["residual"],
        "residual_hor": out["residual_hor"],
        "residual_vert": out["residual_vert"],
        "wrms": out["wrms"],
        "roughness": out["roughness"],
        "x_hor": obs.x_hor,
        "v_hor": obs.v_hor,
        "sig_hor": obs.sig_hor,
        "x_vert": np.array([]) if obs.x_vert is None else obs.x_vert,
        "v_vert": np.array([]) if obs.v_vert is None else obs.v_vert,
        "sig_vert": np.array([]) if obs.sig_vert is None else obs.sig_vert,
        "topx_interface": greens.topx_interface,
        "topz_interface": greens.topz_interface,
        "botx_interface": greens.botx_interface,
        "botz_interface": greens.botz_interface,
    }
    savemat(filename, {"out_py": mdict})


def validate_inversion_config(cfg: SlipInversionConfig) -> None:
    """Validate inversion settings."""
    if cfg.alpha < 0:
        raise ValueError("alpha must be nonnegative.")
    if cfg.smoothing_order.lower() not in {"first", "second", "none"}:
        raise ValueError("smoothing_order must be 'first', 'second', or 'none'.")
    if cfg.inversion_mode.lower() not in {"forward_slip", "slip_deficit"}:
        raise ValueError("inversion_mode must be 'forward_slip' or 'slip_deficit'.")
    if cfg.solver_type.lower() not in {"unbounded", "nonnegative", "bounded"}:
        raise ValueError("solver_type must be 'unbounded', 'nonnegative', or 'bounded'.")


def _coerce_greens(greens: Greens2D | str | Path, *, mat_struct_name: str | None = None) -> Greens2D:
    if isinstance(greens, Greens2D):
        return greens
    return load_greens(greens, mat_struct_name=mat_struct_name)


def _mode_sign_and_label(inversion_mode: str) -> tuple[float, str]:
    mode = inversion_mode.lower()
    if mode == "forward_slip":
        return 1.0, "forward slip rate"
    if mode == "slip_deficit":
        return -1.0, "slip-deficit rate"
    raise ValueError("inversion_mode must be 'forward_slip' or 'slip_deficit'.")


def _append_reference_shift_columns(
    G_slip: np.ndarray,
    *,
    n_h: int,
    n_total: int,
    use_vertical: bool,
    solve_horizontal_shift: bool,
    solve_vertical_shift: bool,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    """Append constant reference-velocity columns to the Green's matrix.

    The horizontal shift column is one for horizontal observations and zero for
    vertical observations. The vertical shift column is zero for horizontal
    observations and one for vertical observations. Both shifts have the same
    velocity units as the observations.
    """
    G_slip = np.asarray(G_slip, dtype=float)
    if G_slip.shape[0] != n_total:
        raise ValueError("G_slip row count must match the total number of observations.")

    columns: list[np.ndarray] = []
    names: list[str] = []
    column_dict: dict[str, np.ndarray] = {}

    if solve_horizontal_shift:
        col = np.zeros(n_total, dtype=float)
        col[:n_h] = 1.0
        columns.append(col)
        names.append("horizontal")
        column_dict["horizontal"] = col

    if solve_vertical_shift:
        if not use_vertical:
            raise ValueError("solve_vertical_shift=True requires vertical data to be used in the inversion.")
        col = np.zeros(n_total, dtype=float)
        col[n_h:] = 1.0
        columns.append(col)
        names.append("vertical")
        column_dict["vertical"] = col

    if not columns:
        return G_slip, names, column_dict

    G = np.column_stack([G_slip, *columns])
    return G, names, column_dict


def _expand_regularization_for_offsets(L_slip: sparse.csr_matrix, n_offset: int) -> sparse.csr_matrix:
    """Pad the slip roughness matrix with zero columns for reference shifts."""
    if n_offset == 0:
        return L_slip
    zero_cols = sparse.csr_matrix((L_slip.shape[0], n_offset))
    return sparse.hstack([L_slip, zero_cols], format="csr")


def _solve_augmented_system(
    GG: sparse.csr_matrix,
    dd: np.ndarray,
    cfg: SlipInversionConfig,
    *,
    n_slip: int | None = None,
    n_offset: int = 0,
) -> np.ndarray:
    """Solve the weighted/damped system.

    When reference-shift columns are included, the slip part keeps the requested
    bounds/nonnegativity, but the reference shifts are always unbounded nuisance
    parameters. This avoids the common mistake of forcing velocity offsets to be
    nonnegative when ``solver_type='nonnegative'``.
    """
    solver = cfg.solver_type.lower()
    if n_slip is None:
        n_slip = GG.shape[1] - n_offset

    if solver == "unbounded":
        return np.linalg.lstsq(GG.toarray(), dd, rcond=None)[0]

    if solver == "nonnegative":
        if n_offset == 0:
            return nnls(GG.toarray(), dd)[0]
        lb = np.concatenate([np.zeros(n_slip), np.full(n_offset, -np.inf)])
        ub = np.full(n_slip + n_offset, np.inf)
        res = lsq_linear(GG, dd, bounds=(lb, ub), method="trf")
        if not res.success:
            raise RuntimeError(f"Nonnegative least-squares solver with reference shifts failed: {res.message}")
        return res.x

    if solver == "bounded":
        lb_slip = _expand_bound(cfg.lower_bound, n_slip)
        ub_slip = _expand_bound(cfg.upper_bound, n_slip)
        lb = np.concatenate([lb_slip, np.full(n_offset, -np.inf)])
        ub = np.concatenate([ub_slip, np.full(n_offset, np.inf)])
        res = lsq_linear(GG, dd, bounds=(lb, ub), method="trf")
        if not res.success:
            raise RuntimeError(f"Bounded least-squares solver failed: {res.message}")
        return res.x
    raise ValueError("solver_type must be 'unbounded', 'nonnegative', or 'bounded'.")


def _expand_bound(bound: float | ArrayLike, nel: int) -> np.ndarray:
    arr = np.asarray(bound, dtype=float)
    if arr.size == 1:
        return np.full(nel, float(arr.reshape(-1)[0]))
    arr = arr.reshape(-1)
    if arr.size != nel:
        raise ValueError("Bounds must be scalars or vectors with length equal to number of patches.")
    return arr


def _as_1d(a: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float).reshape(-1)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr
