"""T3 of the paper: exact phase cancellation of the spherical exit phase; Airy integral."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import j0

from fieldtracing.focus import (
    airy_integral,
    parabolic_residual,
    spherical_legendre,
    spherical_pft_amplitude,
    spherical_phase,
    stationary_point,
)
from fieldtracing.units import wavenumber

K, F = wavenumber(0.5), 25.0


def test_exact_cancellation() -> None:
    k_perp = np.linspace(0.0, 0.5 * K, 50)
    rho = stationary_point(k_perp, F, K)
    legendre = spherical_phase(rho, F, K) - k_perp * rho
    assert np.max(np.abs(legendre - spherical_legendre(k_perp, F, K))) < 1e-9 * K * F
    kz = np.sqrt(K**2 - k_perp**2)
    assert np.max(np.abs(spherical_legendre(k_perp, F, K) + kz * F - K * F)) < 1e-6


def test_parabolic_mismatch_is_large() -> None:
    assert np.isclose(parabolic_residual(np.array([0.5 * K]), F, K)[0], -2819.45, atol=0.1)


def test_pft_amplitude_on_axis() -> None:
    assert np.isclose(spherical_pft_amplitude(np.array([0.0]), F, K)[0], F / K)


def test_airy_integral_matches_hankel_quadrature() -> None:
    kmax = 0.5 * K
    for rho in (0.0, 1e-3, 3e-3):
        val, _ = quad(lambda kr, r=rho: 2 * np.pi * kr * j0(kr * r), 0.0, kmax, limit=400)
        assert np.isclose(airy_integral(np.array([rho]), kmax)[0], val, rtol=1e-8)
