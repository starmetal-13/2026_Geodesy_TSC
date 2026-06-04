"""Layered elastic Green's-function backend for GeoSlip2D.

This module wraps the layered plane-strain Green's-function workflow in the
canonical GeoSlip2D API.  The low-level numerical routines are a Python port of
Kaj Johnson's MATLAB layered-elastic workflow.  The public builder accepts the
package-wide :class:`InterfaceGeometry` convention: x in km and depth positive
downward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import expm
from numpy.typing import ArrayLike

from .geometry import InterfaceGeometry
from .greens import Greens2D

DEFAULT_TIP_DEPTH_EPSILON = 0.01
DEFAULT_INTERP_METHOD = "linear"
DEFAULT_EXTRAP_VALUE = 0.0


@dataclass(slots=True)
class MultiLayerOptions:
    """Numerical options for the layered elastic spectral calculation."""

    nps: int = 3
    kdz_cutoff: float = 150.0
    source_kz_cutoff: float = 15.0
    taper_kz_start: float = 30.0
    taper_kz_stop: float = 60.0
    taper_zmin: float = 0.05
    rcond_cutoff: float = 1.0e-14


@dataclass(slots=True)
class LayeredConfig:
    """Configuration for :func:`build_layered_greens`.

    Parameters
    ----------
    h
        Depths to layer interfaces in km.  Must be monotonically increasing.
        For ``N`` layer interfaces, ``mu`` and ``nu`` must each have length
        ``N + 1``.
    mu, nu
        Shear modulus and Poisson's ratio for each elastic layer.
    tip_depth_epsilon
        Small positive depth offset applied to the updip tip to avoid exactly
        surface-breaking singular behavior.
    interp_method, extrap_value
        Interpolation settings used to sample the layered displacement profiles
        at the requested observation positions.  Only linear interpolation is
        currently implemented.
    options
        Numerical spectral/quadrature settings.
    output_sign
        Optional final multiplier applied to both ``Ghor`` and ``Gvert``.
    progress
        If true, print patch-completion messages while Green's functions are assembled.
    """

    h: ArrayLike = field(default_factory=lambda: np.array([5.0, 10.0, 15.0]))
    mu: ArrayLike = field(default_factory=lambda: np.array([1.0, 1.0, 1.0, 1.0]))
    nu: ArrayLike = field(default_factory=lambda: np.array([0.25, 0.25, 0.25, 0.25]))
    tip_depth_epsilon: float = DEFAULT_TIP_DEPTH_EPSILON
    interp_method: str = DEFAULT_INTERP_METHOD
    extrap_value: float = DEFAULT_EXTRAP_VALUE
    options: MultiLayerOptions = field(default_factory=MultiLayerOptions)
    output_sign: float = 1.0
    sign_convention: str = "layered_legacy_forward_slip_positive"
    progress: bool = False

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = np.asarray(self.h, dtype=float).reshape(-1)
        mu = np.asarray(self.mu, dtype=float).reshape(-1)
        nu = np.asarray(self.nu, dtype=float).reshape(-1)
        return h, mu, nu


def validate_layer_model(h: ArrayLike, mu: ArrayLike, nu: ArrayLike) -> None:
    """Validate the layered elastic model vectors."""
    h = np.asarray(h, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    nu = np.asarray(nu, dtype=float).reshape(-1)

    if h.ndim != 1 or mu.ndim != 1 or nu.ndim != 1:
        raise ValueError("h, mu, and nu must be vectors.")
    if h.size == 0:
        raise ValueError("h must contain at least one layer-interface depth.")
    if mu.size != h.size + 1 or nu.size != h.size + 1:
        raise ValueError("len(mu) and len(nu) must both equal len(h)+1.")
    if np.any(np.diff(h) <= 0):
        raise ValueError("h must increase monotonically with depth.")
    if np.any(mu <= 0):
        raise ValueError("All shear moduli in mu must be positive.")
    if np.any(nu <= 0) or np.any(nu >= 0.5):
        raise ValueError("Poisson's ratios in nu should be between 0 and 0.5.")


def make_A_matrix(kj: float, muq: float, lamq: float, gq: float) -> np.ndarray:
    """Build the 4-by-4 first-order system matrix for one layer."""
    return np.array(
        [
            [0, kj, 1 / muq, 0],
            [-kj * lamq / gq, 0, 0, 1 / gq],
            [4 * kj**2 * muq * (lamq + muq) / gq, 0, 0, kj * lamq / gq],
            [0, 0, -kj, 0],
        ],
        dtype=complex,
    )


def safe_layer_propagator(
    A4x4: np.ndarray,
    dz: float,
    kj: float,
    kdz_cutoff: float,
) -> Tuple[np.ndarray, bool]:
    """Numerically safe layer propagator using ``expm(A*dz)``."""
    P = np.eye(4, dtype=complex)
    if (not np.isfinite(dz)) or (not np.isfinite(kj)) or np.any(~np.isfinite(A4x4)):
        return P, True
    if abs(kj * dz) > kdz_cutoff:
        return P, True
    P = expm(A4x4 * dz)
    if np.any(~np.isfinite(P)):
        return np.eye(4, dtype=complex), True
    return P, False


def matlab_row_mrdivide_same_size(a: np.ndarray, b: np.ndarray) -> complex:
    """Mimic MATLAB row-vector right division ``a / b`` for same-sized rows."""
    a = np.asarray(a, dtype=complex).reshape(-1)
    b = np.asarray(b, dtype=complex).reshape(-1)
    denom = np.vdot(b, b)
    if denom == 0:
        return np.nan + 0j
    return np.dot(a, np.conj(b)) / denom


def matlab_rcond_1norm(A: np.ndarray) -> float:
    """Approximate MATLAB ``rcond`` for a small dense matrix using 1-norm cond."""
    A = np.asarray(A, dtype=complex)
    if np.any(~np.isfinite(A)):
        return 0.0
    try:
        c = np.linalg.cond(A, p=1)
    except np.linalg.LinAlgError:
        return 0.0
    c = np.real_if_close(c, tol=1000)
    if np.iscomplexobj(c):
        c = abs(c)
    c = float(c)
    if not np.isfinite(c) or c == 0:
        return 0.0
    return float(1.0 / c)


def apply_k_taper(
    u: np.ndarray,
    k: np.ndarray,
    zs: float,
    kz_start: float,
    kz_stop: float,
    zmin: float,
) -> np.ndarray:
    """Apply a smooth high-wavenumber taper in Fourier space."""
    u = np.asarray(u, dtype=complex).reshape(-1)
    k = np.asarray(k, dtype=float).reshape(-1)
    if kz_stop <= kz_start:
        raise ValueError("apply_k_taper: kz_stop must be larger than kz_start.")

    zeff = max(abs(zs), zmin)
    kz = np.abs(k) * zeff
    w = np.ones_like(k, dtype=float)
    idx_taper = (kz > kz_start) & (kz < kz_stop)
    idx_zero = kz >= kz_stop
    theta = (kz[idx_taper] - kz_start) / (kz_stop - kz_start)
    w[idx_taper] = 0.5 * (1 + np.cos(np.pi * theta))
    w[idx_zero] = 0.0
    out = u * w
    out[~np.isfinite(out)] = 0.0
    return out


def _interp_complex(x: np.ndarray, y: np.ndarray, xi: np.ndarray, fill_value: complex = 0.0) -> np.ndarray:
    """1-D linear interpolation for complex arrays with scalar fill value."""
    yr = np.interp(xi, x, np.real(y), left=np.real(fill_value), right=np.real(fill_value))
    yi = np.interp(xi, x, np.imag(y), left=np.imag(fill_value), right=np.imag(fill_value))
    return yr + 1j * yi


def multi_layer_tapered(
    m: ArrayLike,
    h: ArrayLike,
    mu: ArrayLike,
    nu: ArrayLike,
    opts: Optional[MultiLayerOptions] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute layered elastic displacement profiles for one finite fault patch."""
    if opts is None:
        opts = MultiLayerOptions()

    m = np.asarray(m, dtype=float).reshape(-1)
    h = np.asarray(h, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    nu = np.asarray(nu, dtype=float).reshape(-1)

    H = h[-1]
    NL = len(h) + 1
    muh = mu[-1]

    normalize = np.max(h)
    h = h / normalize
    H = h[-1]

    ztip = m[1] / normalize
    xtip = m[0] / normalize
    L = m[2] / normalize
    dip = m[3] * np.pi / 180.0
    s = m[4]
    nps = int(opts.nps)

    lam = -2 * nu * mu / (2 * nu - 1)
    g = lam + 2 * mu
    lamh = matlab_row_mrdivide_same_size(-2 * nu * muh, 2 * nu - 1)
    D = ztip + L * np.sin(dip)

    U1 = np.zeros((nps, 2000), dtype=float)
    U2 = np.zeros((nps, 2000), dtype=float)

    bp, w_leg = np.polynomial.legendre.leggauss(nps)
    a = 0.0
    b = L
    wf = w_leg * (b - a) / 2.0
    xp = (a + b) / 2.0 + (b - a) / 2.0 * bp

    xmax = None

    for n in range(nps):
        xs = xtip + xp[n] * np.cos(dip)
        zs = ztip + xp[n] * np.sin(dip)
        xmax = 200 * D

        if zs < 0.5:
            N = 16000
        elif zs < 1.5:
            N = 8000
        elif zs < 2.5:
            N = 4000
        else:
            N = 2000

        delta = 2 * xmax / N
        kmax = 0.5 / delta
        nk = N // 4
        k = np.linspace(-kmax, kmax, nk) * 2 * np.pi
        ki = np.linspace(-kmax, kmax, N) * 2 * np.pi
        k[np.isclose(k, 0.0)] = 0.001

        zslay = int(np.sum(zs >= h)) + 1
        mus = mu[zslay - 1]
        t = zslay

        u1 = np.zeros(nk, dtype=complex)
        u2 = np.zeros(nk, dtype=complex)

        for j in range(nk):
            kj = k[j]
            kabs = abs(kj)
            if kabs * zs > opts.source_kz_cutoff:
                u1[j] = 0
                u2[j] = 0
                continue

            bad_k = False
            A = np.zeros((4, 4, NL), dtype=complex)
            P4x4 = np.zeros((4, 4, len(h)), dtype=complex)
            A[:, :, NL - 1] = make_A_matrix(kj, mu[NL - 1], lam[NL - 1], g[NL - 1])

            for q in range(len(h)):
                A[:, :, q] = make_A_matrix(kj, mu[q], lam[q], g[q])
                A4x4 = A[:, :, q]
                z = 0.0 if q == 0 else h[q - 1]
                z0 = h[q]
                dz = z - z0
                P4x4[:, :, q], bad_layer = safe_layer_propagator(A4x4, dz, kj, opts.kdz_cutoff)
                if bad_layer:
                    bad_k = True
                    break

            if bad_k:
                u1[j] = 0
                u2[j] = 0
                continue

            z = 0.0 if zslay == 1 else h[zslay - 2]
            dzs = z - zs
            A4x4 = A[:, :, t - 1]
            P4x4zs, bad_layer = safe_layer_propagator(A4x4, dzs, kj, opts.kdz_cutoff)
            if bad_layer:
                u1[j] = 0
                u2[j] = 0
                continue

            sourceP4x4 = np.eye(4, dtype=complex)
            if zslay > 1:
                for q in range(zslay - 1):
                    sourceP4x4 = sourceP4x4 @ P4x4[:, :, q]
                sourceP4x4 = sourceP4x4 @ P4x4zs
            else:
                sourceP4x4 = P4x4zs

            halfspaceP4x4 = np.eye(4, dtype=complex)
            for q in range(len(h)):
                halfspaceP4x4 = halfspaceP4x4 @ P4x4[:, :, q]
            if zs > H:
                halfspaceP4x4 = halfspaceP4x4 @ P4x4zs

            if np.any(~np.isfinite(sourceP4x4)) or np.any(~np.isfinite(halfspaceP4x4)):
                u1[j] = 0
                u2[j] = 0
                continue

            if kj < 0:
                a1 = np.array([-0.5 / muh, 0.5 / muh, -kj, kj], dtype=complex)
                b1 = np.array(
                    [
                        -(2 * lamh + 3 * muh) / (2 * muh * kj * (lamh + muh)),
                        lamh / (2 * muh * kj * (lamh + muh)),
                        -2,
                        1,
                    ],
                    dtype=complex,
                )
                d1 = a1
                d2 = b1 + a1 * (zs if zs > H else H)
            else:
                a2 = np.array([-0.5 / muh, -0.5 / muh, kj, kj], dtype=complex)
                b2 = np.array(
                    [
                        1 / (2 * kj * (lamh + muh)),
                        -(lamh + 2 * muh) / (2 * muh * kj * (lamh + muh)),
                        0,
                        1,
                    ],
                    dtype=complex,
                )
                d1 = a2
                d2 = b2 + a2 * (zs if zs > H else H)

            M11 = s * np.cos(dip) * np.sin(dip) * 2 * mus
            M12 = s * np.sin(dip) * np.sin(dip) * mus - s * np.cos(dip) * np.cos(dip) * mus
            M22 = -s * np.sin(dip) * np.cos(dip) * 2 * mus

            F1 = -np.array([0, 0, 1j * 1j * kj * M11, 1j * kj * M12], dtype=complex) * np.exp(
                -1j * xs * kj
            )
            F2 = -np.array([0, 0, 1j * M12, M22], dtype=complex) * np.exp(-1j * xs * kj)

            B = -sourceP4x4 @ (F1 + A[:, :, zslay - 1] @ F2)
            Pd1 = halfspaceP4x4 @ d1
            Pd2 = halfspaceP4x4 @ d2
            Mmat = np.array([[Pd1[2], Pd2[2]], [Pd1[3], Pd2[3]]], dtype=complex)
            bvec = -B[2:4]

            if (
                matlab_rcond_1norm(Mmat) < opts.rcond_cutoff
                or np.any(~np.isfinite(Mmat))
                or np.any(~np.isfinite(bvec))
            ):
                u1[j] = 0
                u2[j] = 0
                continue

            unknown = np.linalg.solve(Mmat, bvec)
            c1 = unknown[0]
            c2 = unknown[1]
            u1[j] = -1j * c1 * Pd1[0] - 1j * c2 * Pd2[0] - 1j * B[0]
            u2[j] = c1 * Pd1[1] + c2 * Pd2[1] + B[1]

        u1 = apply_k_taper(u1, k, zs, opts.taper_kz_start, opts.taper_kz_stop, opts.taper_zmin)
        u2 = apply_k_taper(u2, k, zs, opts.taper_kz_start, opts.taper_kz_stop, opts.taper_zmin)
        u1 = _interp_complex(k, u1, ki, 0.0)
        u2 = _interp_complex(k, u2, ki, 0.0)
        u1[~np.isfinite(u1)] = 0
        u2[~np.isfinite(u2)] = 0

        u1hat = u1 / delta
        u2hat = u2 / delta
        u1s = np.fft.ifft(np.fft.fftshift(u1hat))
        u2s = np.fft.ifft(np.fft.fftshift(u2hat))
        U1s = np.real(np.fft.fftshift(u1s))
        U2s = np.real(np.fft.fftshift(u2s))

        step = N // 2000
        U1[n, :] = U1s[::step][:2000]
        U2[n, :] = U2s[::step][:2000]

    if xmax is None:
        raise RuntimeError("Layered calculation did not produce an x-grid.")
    UU1 = np.sum(wf[:, None] * U1, axis=0)
    UU2 = np.sum(wf[:, None] * U2, axis=0)
    x = np.linspace(-xmax, xmax, 2001)[:-1] * normalize
    return x, UU1, UU2


def _unique_stable(x: np.ndarray, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """MATLAB-like ``unique(x,'stable')`` for x and companion arrays."""
    _, idx = np.unique(x, return_index=True)
    idx = np.sort(idx)
    if arrays:
        return (x[idx],) + tuple(np.asarray(a)[idx] for a in arrays)
    return (x[idx],)


def make_dispG_multilayer(
    xobs: ArrayLike,
    topx_interface: ArrayLike,
    topz_interface: ArrayLike,
    botx_interface: ArrayLike,
    botz_interface: ArrayLike,
    h: ArrayLike,
    mu: ArrayLike,
    nu: ArrayLike,
    tip_depth_epsilon: float = DEFAULT_TIP_DEPTH_EPSILON,
    interp_method: str = DEFAULT_INTERP_METHOD,
    extrap_value: float = DEFAULT_EXTRAP_VALUE,
    opts: Optional[MultiLayerOptions] = None,
    *,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build layered Green's matrices patch by patch.

    Unlike the older standalone script, this function expects ``topz`` and
    ``botz`` to be positive depths, matching the GeoSlip2D package convention.
    """
    if interp_method != "linear":
        raise NotImplementedError("Only linear interpolation is implemented, matching this MATLAB workflow.")
    if opts is None:
        opts = MultiLayerOptions()

    xobs = np.asarray(xobs, dtype=float).reshape(-1)
    topx_interface = np.asarray(topx_interface, dtype=float).reshape(-1)
    topz_interface = np.asarray(topz_interface, dtype=float).reshape(-1)
    botx_interface = np.asarray(botx_interface, dtype=float).reshape(-1)
    botz_interface = np.asarray(botz_interface, dtype=float).reshape(-1)

    nel = topz_interface.size
    Ghor = np.zeros((xobs.size, nel), dtype=float)
    Gvert = np.zeros((xobs.size, nel), dtype=float)
    patch_m = np.zeros((5, nel), dtype=float)

    dx = botx_interface - topx_interface
    dz_depth = botz_interface - topz_interface
    patch_length = np.hypot(dx, dz_depth)
    dip = np.degrees(np.arctan2(dz_depth, dx))

    xgrid_last = None
    for kk in range(nel):
        m = np.array(
            [
                topx_interface[kk],
                topz_interface[kk] + tip_depth_epsilon,
                patch_length[kk],
                dip[kk],
                1.0,
            ],
            dtype=float,
        )
        x_layer, u_hor, u_vert = multi_layer_tapered(m, h, mu, nu, opts=opts)
        xgrid_last = x_layer.copy()

        x_layer = np.asarray(x_layer, dtype=float).reshape(-1)
        u_hor = np.asarray(u_hor, dtype=float).reshape(-1)
        u_vert = np.asarray(u_vert, dtype=float).reshape(-1)
        x_layer, u_hor, u_vert = _unique_stable(x_layer, u_hor, u_vert)

        # Latest notebook convention: the user removed the original MATLAB
        # minus sign in the interpolation coordinate, so xp = x_layer.
        xp = x_layer
        order = np.argsort(xp)
        xp = xp[order]
        u_hor_sorted = u_hor[order]
        u_vert_sorted = u_vert[order]

        Ghor[:, kk] = -np.interp(xobs, xp, u_hor_sorted, left=extrap_value, right=extrap_value)
        Gvert[:, kk] = np.interp(xobs, xp, u_vert_sorted, left=extrap_value, right=extrap_value)
        patch_m[:, kk] = m
        if verbose:
            print(f"completed {kk + 1} of {nel} patches")

    return Ghor, Gvert, patch_m, xgrid_last


def build_layered_greens(
    interface: InterfaceGeometry,
    xobs: ArrayLike,
    config: Optional[LayeredConfig] = None,
) -> Greens2D:
    """Build layered elastic Green's functions for a GeoSlip2D interface."""
    if config is None:
        config = LayeredConfig()
    h, mu, nu = config.arrays()
    validate_layer_model(h, mu, nu)

    xobs = np.asarray(xobs, dtype=float).reshape(-1)
    Ghor, Gvert, patch_m, xgrid_last = make_dispG_multilayer(
        xobs,
        interface.topx,
        interface.topz,
        interface.botx,
        interface.botz,
        h,
        mu,
        nu,
        config.tip_depth_epsilon,
        config.interp_method,
        config.extrap_value,
        config.options,
        verbose=config.progress,
    )

    Ghor = config.output_sign * Ghor
    Gvert = config.output_sign * Gvert

    return Greens2D(
        Ghor=Ghor,
        Gvert=Gvert,
        xobs=xobs,
        interface=interface,
        source_type="layered",
        units="displacement_per_unit_slip",
        sign_convention=config.sign_convention,
        metadata={
            "backend": "layered",
            "layer_h": h,
            "layer_mu": mu,
            "layer_nu": nu,
            "tip_depth_epsilon": config.tip_depth_epsilon,
            "interp_method": config.interp_method,
            "extrap_value": config.extrap_value,
            "output_sign": config.output_sign,
            "progress": config.progress,
            "multi_layer_options": asdict(config.options),
            "patch_m": patch_m,
            "xgrid_last": xgrid_last,
        },
    )


__all__ = [
    "LayeredConfig",
    "MultiLayerOptions",
    "validate_layer_model",
    "multi_layer_tapered",
    "make_dispG_multilayer",
    "build_layered_greens",
]
