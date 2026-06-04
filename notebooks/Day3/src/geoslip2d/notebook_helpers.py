"""Reusable helper functions for GeoSlip2D notebook workflows."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .inversion import ProfileObservations, SlipInversionConfig, fit_profile_slip


def patch_center_depths(greens):
    return np.asarray(greens.interface.centers)[:, 1]


def patch_center_x(greens):
    return np.asarray(greens.interface.centers)[:, 0]


def uniform_backslip_above_depth(greens, locking_depth_km, backslip_rate=1.0):
    depths = patch_center_depths(greens)
    return backslip_rate * (depths <= locking_depth_km).astype(float)


def surface_velocity(greens, slip):
    slip = np.asarray(slip, dtype=float).reshape(-1)
    vh = greens.Ghor @ slip
    vv = greens.Gvert @ slip
    return vh, vv


def make_synthetic_observations(greens_truth, slip_truth, sigma_h=0.02, sigma_v=0.02, noise=False, seed=7):
    vh, vv = surface_velocity(greens_truth, slip_truth)
    rng = np.random.default_rng(seed)
    if noise:
        vh = vh + rng.normal(0.0, sigma_h, size=vh.size)
        vv = vv + rng.normal(0.0, sigma_v, size=vv.size)
    return ProfileObservations(
        x_hor=greens_truth.xobs,
        v_hor=vh,
        sig_hor=np.full_like(vh, sigma_h, dtype=float),
        x_vert=greens_truth.xobs,
        v_vert=vv,
        sig_vert=np.full_like(vv, sigma_v, dtype=float),
        metadata={"truth_backend": greens_truth.source_type},
    )


def run_pair_inversions(observations, greens_hom, greens_nonhom, alpha=0.5):
    cfg = SlipInversionConfig(
        alpha=alpha,
        smoothing_order="second",
        solver_type="nonnegative",
        inversion_mode="forward_slip",
        use_vertical=True,
    )
    out_hom = fit_profile_slip(observations, greens_hom, cfg)
    out_nonhom = fit_profile_slip(observations, greens_nonhom, cfg)
    print(f"Homogeneous inversion WRMS:     {out_hom['wrms']:.3f}")
    print(f"Non-homogeneous inversion WRMS: {out_nonhom['wrms']:.3f}")
    return out_hom, out_nonhom


def plot_surface_velocities(xobs, hom_vel, nonhom_vel, backend_label):
    vh_hom, vv_hom = hom_vel
    vh_non, vv_non = nonhom_vel
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    axes[0].plot(xobs, vh_hom, label="homogeneous")
    axes[0].plot(xobs, vh_non, label=backend_label)
    axes[0].set_ylabel("horizontal velocity")
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_title("Surface response to uniform backslip above locking depth")

    axes[1].plot(xobs, vv_hom, label="homogeneous")
    axes[1].plot(xobs, vv_non, label=backend_label)
    axes[1].set_xlabel("distance from trench, x (km)")
    axes[1].set_ylabel("vertical velocity")
    axes[1].grid(True)
    axes[1].legend()
    return fig


def plot_inversion_results(greens_hom, greens_nonhom, slip_true, out_hom, out_nonhom, backend_label):
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)

    axes[0].plot(greens_nonhom.xobs, out_nonhom["observations"].v_hor, "k.", label="synthetic data")
    axes[0].plot(greens_hom.xobs, out_hom["dhat_hor"], label="fit with homogeneous")
    axes[0].plot(greens_nonhom.xobs, out_nonhom["dhat_hor"], label=f"fit with {backend_label}")
    axes[0].set_ylabel("horizontal velocity")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(greens_nonhom.xobs, out_nonhom["observations"].v_vert, "k.", label="synthetic data")
    axes[1].plot(greens_hom.xobs, out_hom["dhat_vert"], label="fit with homogeneous")
    axes[1].plot(greens_nonhom.xobs, out_nonhom["dhat_vert"], label=f"fit with {backend_label}")
    axes[1].set_ylabel("vertical velocity")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(patch_center_x(greens_nonhom), slip_true, "k-", linewidth=2, label="true slip")
    axes[2].plot(patch_center_x(greens_hom), out_hom["slip_hat"], label="recovered with homogeneous")
    axes[2].plot(patch_center_x(greens_nonhom), out_nonhom["slip_hat"], label=f"recovered with {backend_label}")
    axes[2].set_xlabel("patch center x (km)")
    axes[2].set_ylabel("slip / backslip rate")
    axes[2].grid(True)
    axes[2].legend()
    return fig
