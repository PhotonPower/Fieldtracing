"""T2 and T7 of the paper: Jacobian law and Maslov index against the exact angular spectrum.

The geometric law is the leading order in 1/k: the deviation from the exact field must shrink
like 1/k (paper, section 3.4), and the phase behind a caustic must carry exp(-iπ/2).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from fieldtracing._types import FloatArray
from fieldtracing.fourier import centered_grid
from fieldtracing.propagation import angular_spectrum
from fieldtracing.semiclassical import GeometricResult, propagate_1d, propagate_2d
from fieldtracing.units import wavenumber

N_1D, LENGTH_1D = 2**16, 4.0


def _spherical_1d(x: FloatArray, radius: float, k: float) -> tuple[FloatArray, ...]:
    """Phase, gradient, curvature of sign(R)·k(sqrt(R² + x²) - |R|) (R > 0 diverging)."""
    r = np.sqrt(radius**2 + x**2)
    s = np.sign(radius)
    return k * s * (r - abs(radius)), k * s * x / r, k * s * radius**2 / r**3


def _compare_1d(
    wavelength_um: float, radius: float, d: float, width: float, x1: FloatArray
) -> tuple[FloatArray, GeometricResult]:
    """|exact - geometric| at the landing points of the samples x1."""
    k = wavenumber(wavelength_um)
    x = centered_grid(N_1D, LENGTH_1D / N_1D)
    psi, _, _ = _spherical_1d(x, radius, k)
    exact = angular_spectrum(np.exp(-((x / width) ** 8)) * np.exp(1j * psi), LENGTH_1D / N_1D, k, d)
    p1, g1, c1 = _spherical_1d(x1, radius, k)
    res = propagate_1d(x1, np.exp(-((x1 / width) ** 8)) + 0j, p1, g1, c1, k, d)
    numeric = np.interp(res.points, x, exact.real) + 1j * np.interp(res.points, x, exact.imag)
    return np.abs(numeric - res.amplitude * np.exp(1j * res.phase)), res


@pytest.mark.parametrize(
    ("radius", "d", "width", "mu"),
    [(50.0, 30.0, 0.6, 0), (-20.0, 40.0, 0.5, 1)],  # diverging; converging past the line focus
)
def test_jacobian_law_1d(radius: float, d: float, width: float, mu: int) -> None:
    x1 = np.array([0.1, 0.2, 0.3])
    err_coarse, res = _compare_1d(0.5, radius, d, width, x1)
    err_fine, _ = _compare_1d(0.125, radius, d, width, x1)
    assert np.all(res.maslov == mu)
    assert np.max(err_coarse) < 2e-2  # amplitudes are O(1); leading-order error
    # leading order in 1/k: four times the wavenumber, a quarter of the error (±25 %)
    ratio = err_coarse[-1] / err_fine[-1]
    assert 3.0 < ratio < 5.0


def test_jacobian_law_2d() -> None:
    k = wavenumber(0.5)
    n, length, radius, d, width = 512, 2.4, 50.0, 30.0, 0.4
    step = length / n
    x = centered_grid(n, step)
    xx, yy = np.meshgrid(x, x)
    rr = np.sqrt(radius**2 + xx**2 + yy**2)
    amp = np.exp(-((np.hypot(xx, yy) / width) ** 4))
    exact = angular_spectrum(amp * np.exp(1j * k * (rr - radius)), step, k, d)

    pts = np.array([[0.0, 0.0], [0.1, 0.05], [0.15, -0.1]])
    r1 = np.sqrt(radius**2 + np.sum(pts**2, axis=1))
    grad = k * pts / r1[:, None]
    outer = np.einsum("ni,nj->nij", pts, pts)
    hess = k * (np.eye(2)[None] / r1[:, None, None] - outer / (r1**3)[:, None, None])
    amp1 = np.exp(-((np.hypot(pts[:, 0], pts[:, 1]) / width) ** 4)) + 0j
    res = propagate_2d(pts, amp1, k * (r1 - radius), grad, hess, k, d)
    assert np.all(res.maslov == 0)
    assert np.allclose(res.jacobian, (1 + d / radius) ** 2, rtol=1e-12)
    interp = RegularGridInterpolator((x, x), np.abs(exact))  # |E| is smooth; axes (y, x)
    numeric = interp(np.stack([res.points[:, 1], res.points[:, 0]], axis=1))
    assert np.allclose(numeric, np.abs(res.amplitude), rtol=3e-3)


def test_new_hessian_for_non_commuting_hessian_and_ray_directions() -> None:
    # Tilted astigmatic quadratic phase: H and ∂(p/k_z)/∂p do not commute off axis. Reference:
    # H₂ = ∂∇ψ/∂ρ₂ = C·(∂ρ₂/∂ρ₁)⁻¹ with ∂ρ₂/∂ρ₁ from central differences of the landing points.
    k, d, eps = wavenumber(0.5), 10.0, 1e-5
    c = np.array([[-400.0, 60.0], [60.0, -250.0]])
    pts = np.array([[1.2, -0.7], [-0.9, 1.4], [0.3, 0.2]])

    def landing(p: FloatArray) -> FloatArray:
        g = p @ c
        return propagate_2d(
            p, np.ones(len(p), np.complex128), np.zeros(len(p)), g, np.tile(c, (len(p), 1, 1)), k, d
        ).points

    res = propagate_2d(
        pts,
        np.ones(len(pts), np.complex128),
        np.zeros(len(pts)),
        pts @ c,
        np.tile(c, (3, 1, 1)),
        k,
        d,
    )
    for i, p in enumerate(pts):
        fd = np.empty((2, 2))
        for j in range(2):
            step = np.zeros(2)
            step[j] = eps
            fd[:, j] = (landing((p + step)[None])[0] - landing((p - step)[None])[0]) / (2 * eps)
        assert res.curvature is not None
        assert np.allclose(res.curvature[i], c @ np.linalg.inv(fd), rtol=1e-7, atol=1e-6)
        assert np.allclose(res.curvature[i], res.curvature[i].T, rtol=1e-12)
        assert res.jacobian[i] == pytest.approx(np.linalg.det(fd), rel=1e-8)
