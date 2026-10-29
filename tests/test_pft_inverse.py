"""Issue #4: inverse PFT, several stationary points, validity warning (paper, section 4)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from fieldtracing._types import ComplexArray, FloatArray
from fieldtracing.pft import (
    AsymptoticWarning,
    inverse_pft_1d,
    inverse_pft_2d,
    pft_1d,
    pft_1d_branches,
    pft_2d,
)
from fieldtracing.units import wavenumber

K = wavenumber(0.5)


def test_forward_inverse_roundtrip_spherical_2d() -> None:
    # Spherical exit phase (not quadratic): the roundtrip with the Legendre derivatives
    # ∇ψ̃ = -ρ_s, H̃ = -H⁻¹ is an algebraic identity.
    f = 20.0
    rng = np.random.default_rng(5)
    pts = rng.uniform(-5.0, 5.0, size=(50, 2))
    r = np.sqrt(f * f + np.sum(pts**2, axis=1))
    phase = -K * (r - f)
    grad = -K * pts / r[:, None]
    outer = np.einsum("ni,nj->nij", pts, pts)
    hess = -K * (np.eye(2)[None] / r[:, None, None] - outer / (r**3)[:, None, None])
    amp = (1.0 + 0.2 * rng.normal(size=(50, 3))).astype(np.complex128)
    k, res, spec_phase = pft_2d(pts, amp, phase, grad, hess)
    rho, back, back_phase = inverse_pft_2d(k, res, spec_phase, -pts, -np.linalg.inv(hess))
    assert np.allclose(rho, pts) and np.allclose(back_phase, phase, atol=1e-6)
    assert np.max(np.abs(back - amp)) < 1e-12


def test_roundtrip_1d() -> None:
    x = np.linspace(-0.5, 0.5, 11)
    c = 3000.0 + 500.0 * x  # varying curvature
    phase, grad = 0.5 * c * x**2, c * x
    amp = np.exp(-(x**2)).astype(np.complex128)
    k, res, spec_phase = pft_1d(x, amp, phase, grad, c)
    xb, back, back_phase = inverse_pft_1d(k, res, spec_phase, -x, -1.0 / c)
    assert np.allclose(xb, x) and np.allclose(back_phase, phase)
    assert np.max(np.abs(back - amp)) < 1e-14


def _inverse_error(b: float) -> float:
    """Inverse PFT of Ẽ = exp(-k²/s²)·exp(i b k²/2) against direct quadrature (rel. to max)."""
    s = 400.0  # 1/mm
    k = np.linspace(-8 * s, 8 * s, 400_001)
    xs = np.linspace(-0.3 * b * s, 0.3 * b * s, 9)  # x = -ψ̃'(k) = -b k  →  k = -x/b
    exact = np.array(
        [np.trapezoid(np.exp(-(k**2) / s**2 + 1j * (0.5 * b * k**2 + k * x)), k) for x in xs]
    )
    ks = -xs / b
    x_out, u, phase = inverse_pft_1d(
        ks,
        np.exp(-(ks**2) / s**2).astype(np.complex128),
        0.5 * b * ks**2,
        b * ks,
        np.full_like(ks, b),
    )
    assert np.allclose(x_out, xs)
    return float(np.max(np.abs(u * np.exp(1j * phase) - exact)) / np.max(np.abs(exact)))


def test_inverse_pft_error_scales_like_inverse_fresnel_number() -> None:
    e1, e2 = _inverse_error(2e-3), _inverse_error(8e-3)  # F = |b| s² = 320, 1280
    assert e1 < 1e-2
    assert 3.0 < e1 / e2 < 5.0


def test_two_branches_interfere() -> None:
    # ψ = a x³/3 has two stationary points x = ±sqrt(k/a) for k > 0 (fold at k = 0).
    a = 2000.0

    def amplitude(x: FloatArray) -> ComplexArray:
        return np.asarray(np.exp(-(x**2) / 0.5), dtype=np.complex128)

    def phase(x: FloatArray) -> FloatArray:
        return np.asarray(a * x**3 / 3)

    def gradient(x: FloatArray) -> FloatArray:
        return np.asarray(a * x**2)

    def curvature(x: FloatArray) -> FloatArray:
        return np.asarray(2 * a * x)

    k_out = np.array([300.0, 600.0, 1000.0, 2000.0])
    spectrum, count = pft_1d_branches(
        k_out, np.linspace(-3.0, 3.0, 6001), amplitude, phase, gradient, curvature
    )
    assert np.all(count == 2)
    x = np.linspace(-3.0, 3.0, 2_000_001)
    field = amplitude(x) * np.exp(1j * phase(x))
    exact = np.array([np.trapezoid(field * np.exp(-1j * kk * x), x) for kk in k_out]) / (2 * np.pi)
    assert np.max(np.abs(spectrum - exact)) < 2e-2 * np.max(np.abs(exact))
    # a single branch misses the interference by far more
    xs = np.sqrt(k_out / a)
    _, res, sp = pft_1d(xs, amplitude(xs), phase(xs), k_out, curvature(xs))
    assert np.max(np.abs(res * np.exp(1j * sp) - exact)) > 0.3 * np.max(np.abs(exact))


def test_validity_warning() -> None:
    x = np.array([0.0, 0.1])
    args = (x, np.ones(2, dtype=np.complex128), 0.5 * 50.0 * x**2, 50.0 * x, np.full(2, 50.0))
    with pytest.warns(AsymptoticWarning):
        pft_1d(*args, length_u=0.2)  # F = 50·0.04 = 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pft_1d(*args, length_u=2.0)  # F = 200
        pft_1d(*args)  # no check without length_u
