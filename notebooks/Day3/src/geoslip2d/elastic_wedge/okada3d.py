"""Partial direct Python port of the bundled MATLAB disloc3d.m.

This implementation covers the modes used by make_traction_disp:
  * dip slip:   disloc3d([m; 0; -1; 0], ...)
  * opening:    disloc3d([m; 0;  0; 1], ...)

Strike slip is not implemented because the elastic-wedge workflow never calls it.
The array convention intentionally follows MATLAB: coordinates have shape (3, N),
and outputs have shape (3, N), (9, N), and (6, N).
"""
from __future__ import annotations

import numpy as np


def disloc3d(m, coordinates, shear_m=1.0, poisson_ratio=0.25):
    m = np.asarray(m, dtype=float).reshape(-1)
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim == 1:
        coordinates = coordinates.reshape(3, 1)
    if coordinates.shape[0] != 3:
        raise ValueError("coordinates must have shape (3, N)")
    if m.size != 10:
        raise ValueError("m must contain 10 Okada source parameters")

    sin = np.sin; cos = np.cos; sqrt = np.sqrt; log = np.log; atan2 = np.arctan2
    pi = np.pi

    X = coordinates[0, :]
    Y = coordinates[1, :]
    Z = coordinates[2, :]

    L = m[0]
    W = m[1]
    D = m[2]
    angle = m[3]

    if not (D >= W * sin(angle / 180.0 * pi) and np.array_equal(Z, -np.abs(Z))):
        raise ValueError("Physically impossible source or coordinates: all z should be negative.")

    strikeAngle = m[4]
    Xc = m[5]
    Yc = m[6]
    slip_str = m[7]
    slip_dip = m[8]
    tensile = m[9]
    if slip_str != 0:
        raise NotImplementedError("Strike-slip mode is not implemented in this direct wedge port.")

    Gshear = shear_m
    nu = 0.4999 if poisson_ratio == 0.5 else poisson_ratio
    youngs = 2 * Gshear * (1 + nu)
    lam = nu * youngs / ((1 + nu) * (1 - 2 * nu))
    mu = Gshear

    delta = angle / 180.0 * pi
    angle_Str = -strikeAngle / 180.0 * pi
    c = D

    x = -sin(angle_Str) * (X - Xc) + cos(angle_Str) * (Y - Yc) + 0.5 * L
    y = -cos(angle_Str) * (X - Xc) - sin(angle_Str) * (Y - Yc)
    z = Z
    one4 = np.ones((4, 1))

    def common_terms(d):
        p = y * cos(delta) + d * sin(delta)
        xi = np.vstack([x, x, x - L, x - L])
        eta = np.vstack([p, p - W, p, p - W])
        q = one4 * y * sin(delta) - one4 * d * cos(delta)
        Rsquare = xi**2 + eta**2 + q**2
        R = sqrt(Rsquare)
        y_ = eta * cos(delta) + q * sin(delta)
        d_ = eta * sin(delta) - q * cos(delta)
        X11 = 1.0 / (R * (R + xi))
        X32 = (2 * R + xi) / (R**3 * (R + xi)**2)
        Y11 = 1.0 / (R * (R + eta))
        Y32 = (2 * R + eta) / (R**3 * (R + eta)**2)
        qsign = np.sign(q)
        theta = qsign * atan2(xi * eta, np.abs(q) * R)
        Xroot = sqrt(xi**2 + q**2)
        if abs(cos(delta)) < 1e-6:
            I3 = 0.5 * (eta / (R + d_) + y_ * q / (R + d_)**2 - log(R + eta))
            I4 = 0.5 * (xi * y_ / (R + d_)**2)
        else:
            I3 = (1 / cos(delta)) * y_ / (R + d_) - (1 / cos(delta)**2) * (
                log(R + eta) - sin(delta) * log(R + d_)
            )
            I4 = (sin(delta) / cos(delta)) * xi / (R + d_) + (2 / cos(delta)**2) * atan2(
                eta * (Xroot + q * cos(delta)) + Xroot * (R + Xroot) * sin(delta),
                xi * (R + Xroot) * cos(delta),
            )
        E = sin(delta) / R - y_ * q / R**3
        F = d_ / R**3 + xi**2 * Y32 * sin(delta)
        G = 2 * X11 * sin(delta) - y_ * q * X32
        H = d_ * q * X32 + xi * q * Y32 * sin(delta)
        E_ = cos(delta) / R + d_ * q / R**3
        F_ = y_ / R**3 + xi**2 * Y32 * cos(delta)
        G_ = 2 * X11 * cos(delta) + d_ * q * X32
        H_ = y_ * q * X32 + xi * q * Y32 * cos(delta)
        return dict(p=p, xi=xi, eta=eta, q=q, R=R, y_=y_, d_=d_, X11=X11, X32=X32,
                    Y11=Y11, Y32=Y32, theta=theta, I3=I3, I4=I4,
                    E=E, F=F, G=G, H=H, E_=E_, F_=F_, G_=G_, H_=H_)

    alpha = (lam + mu) / (lam + 2 * mu)

    # First image/source terms at z.
    d = c - z
    T0 = common_terms(d)
    xi = T0['xi']; eta = T0['eta']; q = T0['q']; R = T0['R']; y_ = T0['y_']; d_ = T0['d_']
    X11 = T0['X11']; X32 = T0['X32']; Y11 = T0['Y11']; Y32 = T0['Y32']; theta = T0['theta']
    I3 = T0['I3']; I4 = T0['I4']; E = T0['E']; F = T0['F']; G = T0['G']; H = T0['H']
    E_ = T0['E_']; F_ = T0['F_']; G_ = T0['G_']; H_ = T0['H_']
    X53 = (8 * R**2 + 9 * R * xi + 3 * xi**2) / (R**5 * (R + xi)**3)
    Y53 = (8 * R**2 + 9 * R * eta + 3 * eta**2) / (R**5 * (R + eta)**3)
    h = q * cos(delta) - one4 * z
    Z32 = sin(delta) / R**3 - h * Y32
    Z53 = 3 * sin(delta) / R**5 - h * Y53
    Y0 = Y11 - xi**2 * Y32
    Z0 = Z32 - xi**2 * Z53
    c_ = d_ + one4 * z

    I1 = -(xi / (R + d_)) * cos(delta) - I4 * sin(delta)
    I2 = log(R + d_) + I3 * sin(delta)
    D11 = 1.0 / (R * (R + d_))
    if abs(cos(delta)) < 1e-6:
        K1 = (xi * q) / (R + d_) * D11
        K3 = sin(delta) / (R + d_) * (xi**2 * D11 - 1)
    else:
        K1 = xi / cos(delta) * (D11 - Y11 * sin(delta))
        K3 = 1 / cos(delta) * (q * Y11 - y_ * D11)
    K2 = 1.0 / R + K3 * sin(delta)
    K4 = xi * Y11 * cos(delta) - K1 * sin(delta)
    J5 = -(d_ + y_**2 / (R + d_)) * D11
    J2 = xi * y_ / (R + d_) * D11
    if abs(cos(delta)) < 1e-6:
        J6 = -y_ / (R + d_)**2 * (xi**2 * D11 - 0.5)
        J3 = -xi / (R + d_)**2 * (q**2 * D11 - 0.5)
    else:
        J6 = 1 / cos(delta) * (K3 - J5 * sin(delta))
        J3 = 1 / cos(delta) * (K1 - J2 * sin(delta))
    J1 = J5 * cos(delta) - J6 * sin(delta)
    J4 = -xi * Y11 - J2 * cos(delta) + J3 * sin(delta)
    P = cos(delta) / R**3 + q * Y32 * sin(delta)
    Q = 3 * c_ * d_ / R**5 - (one4 * z * Y32 + Z32 + Z0) * sin(delta)
    P_ = sin(delta) / R**3 - q * Y32 * cos(delta)
    Q_ = (3 * c_ * y_) / R**5 + q * Y32 - (one4 * z * Y32 + Z32 + Z0) * cos(delta)

    if slip_dip != 0:
        Du1A = alpha / 2 * q / R
        Du2A = theta / 2 + alpha / 2 * eta * q * X11
        Du3A = (1 - alpha) / 2 * log(R + xi) - alpha / 2 * q**2 * X11
        Du1B = -q / R + (1 - alpha) / alpha * I3 * sin(delta) * cos(delta)
        Du2B = -eta * q * X11 - theta - (1 - alpha) / alpha * xi / (R + d_) * sin(delta) * cos(delta)
        Du3B = q**2 * X11 + (1 - alpha) / alpha * I4 * sin(delta) * cos(delta)
        Du1C = (1 - alpha) * cos(delta) / R - q * Y11 * sin(delta) - alpha * c_ * q / R**3
        Du2C = (1 - alpha) * y_ * X11 - alpha * c_ * eta * q * X32
        Du3C = -d_ * X11 - xi * Y11 * sin(delta) - alpha * c_ * (X11 - q**2 * X32)
        Dj1A = -alpha / 2 * xi * q / R**3
        Dj2A = -q / 2 * Y11 - alpha / 2 * eta * q / R**3
        Dj3A = (1 - alpha) / 2 * 1 / R + alpha / 2 * q**2 / R**3
        Dj1B = xi * q / R**3 + (1 - alpha) / alpha * J4 * sin(delta) * cos(delta)
        Dj2B = eta * q / R**3 + q * Y11 + (1 - alpha) / alpha * J5 * sin(delta) * cos(delta)
        Dj3B = -q**2 / R**3 + (1 - alpha) / alpha * J6 * sin(delta) * cos(delta)
        Dj1C = -(1 - alpha) * xi / R**3 * cos(delta) + xi * q * Y32 * sin(delta) + alpha * (3 * c_ * xi * q / R**5)
        Dj2C = -(1 - alpha) * y_ / R**3 + alpha * 3 * c_ * eta * q / R**5
        Dj3C = d_ / R**3 - Y0 * sin(delta) + alpha * c_ / R**3 * (1 - 3 * q**2 / R**2)
        Dk1A = alpha / 2 * E
        Dk2A = (1 - alpha) / 2 * d_ * X11 + xi / 2 * Y11 * sin(delta) + alpha / 2 * eta * G
        Dk3A = (1 - alpha) / 2 * y_ * X11 - alpha / 2 * q * G
        Dk1B = -E + (1 - alpha) / alpha * J1 * sin(delta) * cos(delta)
        Dk2B = -eta * G - xi * Y11 * sin(delta) + (1 - alpha) / alpha * J2 * sin(delta) * cos(delta)
        Dk3B = q * G + (1 - alpha) / alpha * J3 * sin(delta) * cos(delta)
        Dk1C = -(1 - alpha) * eta / R**3 + Y0 * sin(delta)**2 - alpha * ((c_ + d_) / R**3 * sin(delta) - 3 * c_ * y_ * q / R**5)
        Dk2C = (1 - alpha) * (X11 - y_**2 * X32) - alpha * c_ * ((d_ + 2 * q * cos(delta)) * X32 - y_ * eta * q * X53)
        Dk3C = xi * P * sin(delta) + y_ * d_ * X32 + alpha * c_ * ((y_ + 2 * q * sin(delta)) * X32 - y_ * q**2 * X53)
        Dl1A = alpha / 2 * E_
        Dl2A = (1 - alpha) / 2 * y_ * X11 + xi / 2 * Y11 * cos(delta) + alpha / 2 * eta * G_
        Dl3A = -(1 - alpha) / 2 * d_ * X11 - alpha / 2 * q * G_
        Dl1B = -E_ - (1 - alpha) / alpha * K3 * sin(delta) * cos(delta)
        Dl2B = -eta * G_ - xi * Y11 * cos(delta) - (1 - alpha) / alpha * xi * D11 * sin(delta) * cos(delta)
        Dl3B = q * G_ - (1 - alpha) / alpha * K4 * sin(delta) * cos(delta)
        Dl1C = -q / R**3 + Y0 * sin(delta) * cos(delta) - alpha * ((c_ + d_) / R**3 * cos(delta) + 3 * c_ * d_ * q / R**5)
        Dl2C = (1 - alpha) * y_ * d_ * X32 - alpha * c_ * ((y_ - 2 * q * sin(delta)) * X32 + d_ * eta * q * X53)
        Dl3C = -xi * P_ * sin(delta) + X11 - d_**2 * X32 - alpha * c_ * ((d_ - 2 * q * cos(delta)) * X32 - d_ * q**2 * X53)
    else:
        Du1A=Du2A=Du3A=Du1B=Du2B=Du3B=Du1C=Du2C=Du3C=0
        Dj1A=Dj2A=Dj3A=Dj1B=Dj2B=Dj3B=Dj1C=Dj2C=Dj3C=0
        Dk1A=Dk2A=Dk3A=Dk1B=Dk2B=Dk3B=Dk1C=Dk2C=Dk3C=0
        Dl1A=Dl2A=Dl3A=Dl1B=Dl2B=Dl3B=Dl1C=Dl2C=Dl3C=0

    if tensile != 0:
        Tu1A = -(1 - alpha) / 2 * log(R + eta) - alpha / 2 * q**2 * Y11
        Tu2A = -(1 - alpha) / 2 * log(R + xi) - alpha / 2 * q**2 * X11
        Tu3A = theta / 2 - alpha / 2 * q * (eta * X11 + xi * Y11)
        Tu1B = q**2 * Y11 - (1 - alpha) / alpha * I3 * sin(delta)**2
        Tu2B = q**2 * X11 + (1 - alpha) / alpha * xi / (R + d_) * sin(delta)**2
        Tu3B = q * (eta * X11 + xi * Y11) - theta - (1 - alpha) / alpha * I4 * sin(delta)**2
        Tu1C = -(1 - alpha) * (sin(delta) / R + q * Y11 * cos(delta)) - alpha * (one4 * z * Y11 - q**2 * Z32)
        Tu2C = (1 - alpha) * 2 * xi * Y11 * sin(delta) + d_ * X11 - alpha * c_ * (X11 - q**2 * X32)
        Tu3C = (1 - alpha) * (y_ * X11 + xi * Y11 * cos(delta)) + alpha * q * (c_ * eta * X32 + xi * Z32)
        Tj1A = -(1 - alpha) / 2 * xi * Y11 + alpha / 2 * xi * q**2 * Y32
        Tj2A = -(1 - alpha) / 2 * 1 / R + alpha / 2 * q**2 / R**3
        Tj3A = -(1 - alpha) / 2 * q * Y11 - alpha / 2 * q**3 * Y32
        Tj1B = -xi * q**2 * Y32 - (1 - alpha) / alpha * J4 * sin(delta)**2
        Tj2B = -q**2 / R**3 - (1 - alpha) / alpha * J5 * sin(delta)**2
        Tj3B = q**3 * Y32 - (1 - alpha) / alpha * J6 * sin(delta)**2
        Tj1C = (1 - alpha) * xi / R**3 * sin(delta) + xi * q * Y32 * cos(delta) + alpha * xi * (3 * c_ * eta / R**5 - 2 * Z32 - Z0)
        Tj2C = (1 - alpha) * 2 * Y0 * sin(delta) - d_ / R**3 + alpha * c_ / R**3 * (1 - 3 * q**2 / R**2)
        Tj3C = -(1 - alpha) * (y_ / R**3 - Y0 * cos(delta)) - alpha * (3 * c_ * eta * q / R**5 - q * Z0)
        Tk1A = -(1 - alpha) / 2 * (cos(delta) / R + q * Y11 * sin(delta)) - alpha / 2 * q * F
        Tk2A = -(1 - alpha) / 2 * y_ * X11 - alpha / 2 * q * G
        Tk3A = (1 - alpha) / 2 * (d_ * X11 + xi * Y11 * sin(delta)) + alpha / 2 * q * H
        Tk1B = q * F - (1 - alpha) / alpha * J1 * sin(delta)**2
        Tk2B = q * G - (1 - alpha) / alpha * J2 * sin(delta)**2
        Tk3B = -q * H - (1 - alpha) / alpha * J3 * sin(delta)**2
        Tk1C = (1 - alpha) * (q / R**3 + Y0 * sin(delta) * cos(delta)) + alpha * (one4 * z / R**3 * cos(delta) + 3 * c_ * d_ * q / R**5 - q * Z0 * sin(delta))
        Tk2C = -(1 - alpha) * 2 * xi * P * sin(delta) - y_ * d_ * X32 + alpha * c_ * ((y_ + 2 * q * sin(delta)) * X32 - y_ * q**2 * X53)
        Tk3C = -(1 - alpha) * (xi * P * cos(delta) - X11 + y_**2 * X32) + alpha * c_ * ((d_ + 2 * q * cos(delta)) * X32 - y_ * eta * q * X53) + alpha * xi * Q
        Tl1A = (1 - alpha) / 2 * (sin(delta) / R - q * Y11 * cos(delta)) - alpha / 2 * q * F_
        Tl2A = (1 - alpha) / 2 * d_ * X11 - alpha / 2 * q * G_
        Tl3A = (1 - alpha) / 2 * (y_ * X11 + xi * Y11 * cos(delta)) + alpha / 2 * q * H_
        Tl1B = q * F_ + (1 - alpha) / alpha * K3 * sin(delta)**2
        Tl2B = q * G_ + (1 - alpha) / alpha * xi * D11 * sin(delta)**2
        Tl3B = -q * H_ + (1 - alpha) / alpha * K4 * sin(delta)**2
        Tl1C = -eta / R**3 + Y0 * cos(delta)**2 - alpha * (one4 * z / R**3 * sin(delta) - 3 * c_ * y_ * q / R**5 - Y0 * sin(delta)**2 + q * Z0 * cos(delta))
        Tl2C = (1 - alpha) * 2 * xi * P_ * sin(delta) - X11 + d_**2 * X32 - alpha * c_ * ((d_ - 2 * q * cos(delta)) * X32 - d_ * q**2 * X53)
        Tl3C = (1 - alpha) * (xi * P_ * cos(delta) + y_ * d_ * X32) + alpha * c_ * ((y_ - 2 * q * sin(delta)) * X32 + d_ * eta * q * X53) + alpha * xi * Q_
    else:
        Tu1A=Tu2A=Tu3A=Tu1B=Tu2B=Tu3B=Tu1C=Tu2C=Tu3C=0
        Tj1A=Tj2A=Tj3A=Tj1B=Tj2B=Tj3B=Tj1C=Tj2C=Tj3C=0
        Tk1A=Tk2A=Tk3A=Tk1B=Tk2B=Tk3B=Tk1C=Tk2C=Tk3C=0
        Tl1A=Tl2A=Tl3A=Tl1B=Tl2B=Tl3B=Tl1C=Tl2C=Tl3C=0

    # Second image terms with Z_ = -z.
    T1 = common_terms(c - (-z))
    xi = T1['xi']; eta = T1['eta']; q = T1['q']; R = T1['R']; y_ = T1['y_']; d_ = T1['d_']
    X11 = T1['X11']; X32 = T1['X32']; Y11 = T1['Y11']; Y32 = T1['Y32']; theta = T1['theta']
    E = T1['E']; F = T1['F']; G = T1['G']; H = T1['H']; E_ = T1['E_']; F_ = T1['F_']; G_ = T1['G_']; H_ = T1['H_']

    if slip_dip != 0:
        Du1A_ = alpha / 2 * q / R
        Du2A_ = theta / 2 + alpha / 2 * eta * q * X11
        Du3A_ = (1 - alpha) / 2 * log(R + xi) - alpha / 2 * q**2 * X11
        Dj1A_ = -alpha / 2 * xi * q / R**3
        Dj2A_ = -q / 2 * Y11 - alpha / 2 * eta * q / R**3
        Dj3A_ = (1 - alpha) / 2 * 1 / R + alpha / 2 * q**2 / R**3
        Dk1A_ = alpha / 2 * E
        Dk2A_ = (1 - alpha) / 2 * d_ * X11 + xi / 2 * Y11 * sin(delta) + alpha / 2 * eta * G
        Dk3A_ = (1 - alpha) / 2 * y_ * X11 - alpha / 2 * q * G
        Dl1A_ = alpha / 2 * E_
        Dl2A_ = (1 - alpha) / 2 * y_ * X11 + xi / 2 * Y11 * cos(delta) + alpha / 2 * eta * G_
        Dl3A_ = -(1 - alpha) / 2 * d_ * X11 - alpha / 2 * q * G_
    else:
        Du1A_=Du2A_=Du3A_=Dj1A_=Dj2A_=Dj3A_=Dk1A_=Dk2A_=Dk3A_=Dl1A_=Dl2A_=Dl3A_=0

    if tensile != 0:
        Tu1A_ = -(1 - alpha) / 2 * log(R + eta) - alpha / 2 * q**2 * Y11
        Tu2A_ = -(1 - alpha) / 2 * log(R + xi) - alpha / 2 * q**2 * X11
        Tu3A_ = theta / 2 - alpha / 2 * q * (eta * X11 + xi * Y11)
        Tj1A_ = -(1 - alpha) / 2 * xi * Y11 + alpha / 2 * xi * q**2 * Y32
        Tj2A_ = -(1 - alpha) / 2 * 1 / R + alpha / 2 * q**2 / R**3
        Tj3A_ = -(1 - alpha) / 2 * q * Y11 - alpha / 2 * q**3 * Y32
        Tk1A_ = -(1 - alpha) / 2 * (cos(delta) / R + q * Y11 * sin(delta)) - alpha / 2 * q * F
        Tk2A_ = -(1 - alpha) / 2 * y_ * X11 - alpha / 2 * q * G
        Tk3A_ = (1 - alpha) / 2 * (d_ * X11 + xi * Y11 * sin(delta)) + alpha / 2 * q * H
        Tl1A_ = (1 - alpha) / 2 * (sin(delta) / R - q * Y11 * cos(delta)) - alpha / 2 * q * F_
        Tl2A_ = (1 - alpha) / 2 * d_ * X11 - alpha / 2 * q * G_
        Tl3A_ = (1 - alpha) / 2 * (y_ * X11 + xi * Y11 * cos(delta)) + alpha / 2 * q * H_
    else:
        Tu1A_=Tu2A_=Tu3A_=Tj1A_=Tj2A_=Tj3A_=Tk1A_=Tk2A_=Tk3A_=Tl1A_=Tl2A_=Tl3A_=0

    # Strike-slip contributions are zero in this port.
    Sux = Suy = Suz = 0
    Sduxdx = Sduydx = Sduzdx = 0
    Sduxdy = Sduydy = Sduzdy = 0
    Sduxdz = Sduydz = Sduzdz = 0

    if slip_dip != 0:
        z4 = one4 * z
        Dux = 1 / (2 * pi) * slip_dip * (Du1A - Du1A_ + Du1B + z4 * Du1C)
        Duy = 1 / (2 * pi) * slip_dip * ((Du2A - Du2A_ + Du2B + z4 * Du2C) * cos(delta) - (Du3A - Du3A_ + Du3B + z4 * Du3C) * sin(delta))
        Duz = 1 / (2 * pi) * slip_dip * ((Du2A - Du2A_ + Du2B - z4 * Du2C) * sin(delta) + (Du3A - Du3A_ + Du3B - z4 * Du3C) * cos(delta))
        Dduxdx = 1 / (2 * pi) * slip_dip * (Dj1A - Dj1A_ + Dj1B + z4 * Dj1C)
        Dduydx = 1 / (2 * pi) * slip_dip * ((Dj2A - Dj2A_ + Dj2B + z4 * Dj2C) * cos(delta) - (Dj3A - Dj3A_ + Dj3B + z4 * Dj3C) * sin(delta))
        Dduzdx = 1 / (2 * pi) * slip_dip * ((Dj2A - Dj2A_ + Dj2B - z4 * Dj2C) * sin(delta) + (Dj3A - Dj3A_ + Dj3B - z4 * Dj3C) * cos(delta))
        Dduxdy = 1 / (2 * pi) * slip_dip * (Dk1A - Dk1A_ + Dk1B + z4 * Dk1C)
        Dduydy = 1 / (2 * pi) * slip_dip * ((Dk2A - Dk2A_ + Dk2B + z4 * Dk2C) * cos(delta) - (Dk3A - Dk3A_ + Dk3B + z4 * Dk3C) * sin(delta))
        Dduzdy = 1 / (2 * pi) * slip_dip * ((Dk2A - Dk2A_ + Dk2B - z4 * Dk2C) * sin(delta) + (Dk3A - Dk3A_ + Dk3B - z4 * Dk3C) * cos(delta))
        Dduxdz = 1 / (2 * pi) * slip_dip * (Dl1A + Dl1A_ + Dl1B + Du1C + z4 * Dl1C)
        Dduydz = 1 / (2 * pi) * slip_dip * ((Dl2A + Dl2A_ + Dl2B + Du2C + z4 * Dl2C) * cos(delta) - (Dl3A + Dl3A_ + Dl3B + Du3C + z4 * Dl3C) * sin(delta))
        Dduzdz = 1 / (2 * pi) * slip_dip * ((Dl2A + Dl2A_ + Dl2B - Du2C - z4 * Dl2C) * sin(delta) + (Dl3A + Dl3A_ + Dl3B - Du3C - z4 * Dl3C) * cos(delta))
    else:
        Dux=Duy=Duz=0
        Dduxdx=Dduydx=Dduzdx=0
        Dduxdy=Dduydy=Dduzdy=0
        Dduxdz=Dduydz=Dduzdz=0

    if tensile != 0:
        z4 = one4 * z
        Tux = 1 / (2 * pi) * tensile * (Tu1A - Tu1A_ + Tu1B + z4 * Tu1C)
        Tuy = 1 / (2 * pi) * tensile * ((Tu2A - Tu2A_ + Tu2B + z4 * Tu2C) * cos(delta) - (Tu3A - Tu3A_ + Tu3B + z4 * Tu3C) * sin(delta))
        Tuz = 1 / (2 * pi) * tensile * ((Tu2A - Tu2A_ + Tu2B - z4 * Tu2C) * sin(delta) + (Tu3A - Tu3A_ + Tu3B - z4 * Tu3C) * cos(delta))
        Tduxdx = 1 / (2 * pi) * tensile * (Tj1A - Tj1A_ + Tj1B + z4 * Tj1C)
        Tduydx = 1 / (2 * pi) * tensile * ((Tj2A - Tj2A_ + Tj2B + z4 * Tj2C) * cos(delta) - (Tj3A - Tj3A_ + Tj3B + z4 * Tj3C) * sin(delta))
        Tduzdx = 1 / (2 * pi) * tensile * ((Tj2A - Tj2A_ + Tj2B - z4 * Tj2C) * sin(delta) + (Tj3A - Tj3A_ + Tj3B - z4 * Tj3C) * cos(delta))
        Tduxdy = 1 / (2 * pi) * tensile * (Tk1A - Tk1A_ + Tk1B + z4 * Tk1C)
        Tduydy = 1 / (2 * pi) * tensile * ((Tk2A - Tk2A_ + Tk2B + z4 * Tk2C) * cos(delta) - (Tk3A - Tk3A_ + Tk3B + z4 * Tk3C) * sin(delta))
        Tduzdy = 1 / (2 * pi) * tensile * ((Tk2A - Tk2A_ + Tk2B - z4 * Tk3C) * sin(delta) + (Tk3A - Tk3A_ + Tk3B - z4 * Tk3C) * cos(delta))
        # NOTE: previous line is intentionally corrected below to match MATLAB exactly.
        Tduzdy = 1 / (2 * pi) * tensile * ((Tk2A - Tk2A_ + Tk2B - z4 * Tk2C) * sin(delta) + (Tk3A - Tk3A_ + Tk3B - z4 * Tk3C) * cos(delta))
        Tduxdz = 1 / (2 * pi) * tensile * (Tl1A + Tl1A_ + Tl1B + Tu1C + z4 * Tl1C)
        Tduydz = 1 / (2 * pi) * tensile * ((Tl2A + Tl2A_ + Tl2B + Tu2C + z4 * Tl2C) * cos(delta) - (Tl3A + Tl3A_ + Tl3B + Tu3C + z4 * Tl3C) * sin(delta))
        Tduzdz = 1 / (2 * pi) * tensile * ((Tl2A + Tl2A_ + Tl2B - Tu2C - z4 * Tl2C) * sin(delta) + (Tl3A + Tl3A_ + Tl3B - Tu3C - z4 * Tl3C) * cos(delta))
    else:
        Tux=Tuy=Tuz=0
        Tduxdx=Tduydx=Tduzdx=0
        Tduxdy=Tduydy=Tduzdy=0
        Tduxdz=Tduydz=Tduzdz=0

    n = coordinates.shape[1]
    factor = np.vstack([np.ones(n), -np.ones(n), -np.ones(n), np.ones(n)])
    G1 = np.sum(factor * (Sux + Dux + Tux), axis=0)
    G2 = np.sum(factor * (Suy + Duy + Tuy), axis=0)
    G3 = np.sum(factor * (Suz + Duz + Tuz), axis=0)
    Dg11 = np.sum(factor * (Sduxdx + Dduxdx + Tduxdx), axis=0)
    Dg12 = np.sum(factor * (Sduxdy + Dduxdy + Tduxdy), axis=0)
    Dg13 = np.sum(factor * (Sduxdz + Dduxdz + Tduxdz), axis=0)
    Dg21 = np.sum(factor * (Sduydx + Dduydx + Tduydx), axis=0)
    Dg22 = np.sum(factor * (Sduydy + Dduydy + Tduydy), axis=0)
    Dg23 = np.sum(factor * (Sduydz + Dduydz + Tduydz), axis=0)
    Dg31 = np.sum(factor * (Sduzdx + Dduzdx + Tduzdx), axis=0)
    Dg32 = np.sum(factor * (Sduzdy + Dduzdy + Tduzdy), axis=0)
    Dg33 = np.sum(factor * (Sduzdz + Dduzdz + Tduzdz), axis=0)

    Gx = cos(angle_Str) * (-G2) - sin(angle_Str) * G1
    Gy = sin(angle_Str) * (-G2) + cos(angle_Str) * G1
    Gz = G3
    displacement = np.vstack([Gx, Gy, Gz])

    Dg11_ = Dg22; Dg12_ = -Dg21; Dg13_ = -Dg23
    Dg21_ = -Dg12; Dg22_ = Dg11; Dg23_ = Dg13
    Dg31_ = -Dg32; Dg32_ = Dg31; Dg33_ = Dg33

    Dgxx = (cos(angle_Str)*Dg11_ - sin(angle_Str)*Dg21_) * cos(angle_Str) + (cos(angle_Str)*Dg12_ - sin(angle_Str)*Dg22_) * (-sin(angle_Str))
    Dgyx = (sin(angle_Str)*Dg11_ + cos(angle_Str)*Dg21_) * cos(angle_Str) - (sin(angle_Str)*Dg12_ + cos(angle_Str)*Dg22_) * sin(angle_Str)
    Dgzx = Dg31_ * cos(angle_Str) - Dg32_ * sin(angle_Str)
    Dgxy = (cos(angle_Str)*Dg11_ - sin(angle_Str)*Dg21_) * sin(angle_Str) + (cos(angle_Str)*Dg12_ - sin(angle_Str)*Dg22_) * cos(angle_Str)
    Dgyy = (sin(angle_Str)*Dg11_ + cos(angle_Str)*Dg21_) * sin(angle_Str) + (sin(angle_Str)*Dg12_ + cos(angle_Str)*Dg22_) * cos(angle_Str)
    Dgzy = sin(angle_Str) * Dg31_ + cos(angle_Str) * Dg32_
    Dgxz = cos(angle_Str) * Dg13_ - sin(angle_Str) * Dg23_
    Dgyz = sin(angle_Str) * Dg13_ + cos(angle_Str) * Dg23_
    Dgzz = Dg33_
    gradient = np.vstack([Dgxx, Dgxy, Dgxz, Dgyx, Dgyy, Dgyz, Dgzx, Dgzy, Dgzz])

    Ex = Dgxx; Ey = Dgyy; Ez = Dgzz
    Exy = 0.5 * (Dgyx + Dgxy)
    Eyz = 0.5 * (Dgyz + Dgzy)
    Ezx = 0.5 * (Dgzx + Dgxz)
    Sx = youngs / ((1 + nu) * (1 - 2 * nu)) * (Ex + nu * (Ey + Ez - Ex))
    Sy = youngs / ((1 + nu) * (1 - 2 * nu)) * (Ey + nu * (Ex + Ez - Ey))
    Sz = youngs / ((1 + nu) * (1 - 2 * nu)) * (Ez + nu * (Ey + Ex - Ez))
    Sxy = 2 * Gshear * Exy
    Syz = 2 * Gshear * Eyz
    Szx = 2 * Gshear * Ezx
    stress = np.vstack([Sx, Sxy, Szx, Sy, Syz, Sz])
    return displacement, gradient, stress, None
