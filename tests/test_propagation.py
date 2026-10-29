"""T1 of the paper: Van Vleck kernel vs. Rayleigh-Sommerfeld kernel; angular spectrum basics."""

from __future__ import annotations

import numpy as np

from fieldtracing.fourier import centered_grid
from fieldtracing.propagation import (
    angular_spectrum,
    rayleigh_sommerfeld_kernel,
    van_vleck_kernel_homogeneous,
)
from fieldtracing.units import wavenumber


def test_van_vleck_is_leading_term_of_rs() -> None:
    # K_RS / K_VV = 1 + i/(k r) exactly, so the relative difference is 1/(k r).
    k = wavenumber(0.5)
    d = 10.0
    dx = np.linspace(0.0, 8.0, 9)
    dy = np.zeros_like(dx)
    rs = rayleigh_sommerfeld_kernel(dx, dy, d, k)
    vv = van_vleck_kernel_homogeneous(dx, dy, d, k)
    r = np.sqrt(d**2 + dx**2)
    assert np.allclose(np.abs(rs / vv - 1.0), 1.0 / (k * r), rtol=1e-9)


def test_angular_spectrum_conserves_power_and_inverts() -> None:
    k = wavenumber(0.5)
    x = centered_grid(1024, 0.002)
    field = np.exp(-((x / 0.1) ** 2) + 1j * 300.0 * x)
    forward = angular_spectrum(field, 0.002, k, 5.0)
    assert np.isclose(np.sum(np.abs(forward) ** 2), np.sum(np.abs(field) ** 2), rtol=1e-12)
    back = angular_spectrum(forward, 0.002, k, -5.0)
    assert np.max(np.abs(back - field)) < 1e-10
