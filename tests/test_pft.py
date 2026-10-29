"""PFT (paper eq. pft_final): exact for quadratic phase and constant U, O(1/F) otherwise."""

from __future__ import annotations

import numpy as np

from fieldtracing.fourier import centered_grid
from fieldtracing.pft import pft_1d, pft_2d


def _pft_error(c: float) -> float:
    """Max deviation of the 1D PFT from an FFT for U = Gaussian, ψ = c x²/2 (rel. to max)."""
    n, dx = 2**15, 2.5e-4
    x = centered_grid(n, dx)
    u = np.exp(-(x**2) / 0.5)
    field = u * np.exp(0.5j * c * x**2)
    xs = np.linspace(-0.5, 0.5, 11)
    k, residual, phase = pft_1d(
        xs, np.exp(-(xs**2) / 0.5) + 0j, 0.5 * c * xs**2, c * xs, np.full_like(xs, c)
    )
    approx = residual * np.exp(1j * phase)
    # direct quadrature at the PFT frequencies (interpolating a chirped spectrum would alias)
    exact = np.exp(-1j * np.outer(k, x)) @ field * dx / (2 * np.pi)
    return float(np.max(np.abs(approx - exact)) / np.max(np.abs(exact)))


def test_pft_error_scales_like_inverse_fresnel_number() -> None:
    e1, e2 = _pft_error(500.0), _pft_error(2000.0)
    assert e1 < 1e-2
    assert 3.0 < e1 / e2 < 5.0


def test_pft_2d_signature_and_amplitude() -> None:
    # Saddle phase: sig = 0, |det H| = |c1 c2|.
    pts = np.array([[0.1, -0.2]])
    hess = np.array([[[300.0, 0.0], [0.0, -200.0]]])
    grad = np.einsum("nij,nj->ni", hess, pts)
    phase = 0.5 * np.einsum("ni,nij,nj->n", pts, hess, pts)
    k, residual, spec_phase = pft_2d(pts, np.array([1.0 + 0j]), phase, grad, hess)
    assert np.allclose(k, grad)
    assert np.isclose(residual[0], 1.0 / (2 * np.pi * np.sqrt(6.0e4)))
    assert np.isclose(spec_phase[0], phase[0] - grad[0] @ pts[0])
