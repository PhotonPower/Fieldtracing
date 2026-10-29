"""T6 of the paper: SFT against a brute-force quadrature; 2D separability; PFT limit."""

from __future__ import annotations

import numpy as np

from fieldtracing._types import FloatArray
from fieldtracing.fourier import centered_grid
from fieldtracing.sft import aperture_weights_1d, circular_aperture_weights, sft_1d, sft_2d_diagonal

C, A = 1.0e5, 1.0  # curvature 1/mm², slit half width mm (paper, SFT table)
K_TEST = np.array([0.0, 5e4, 0.99e5, 1.0e5, 1.01e5, 1.2e5])


def _amp(x: FloatArray) -> FloatArray:
    return np.asarray(np.exp(-(x**2) / 0.8), dtype=np.float64)


def _brute_force() -> FloatArray:
    x = np.linspace(-A, A, 4_000_001)
    dx = x[1] - x[0]
    return np.array(
        [np.trapezoid(_amp(x) * np.exp(1j * (C * x**2 / 2 - k * x)), dx=dx) for k in K_TEST]
    ) / (2 * np.pi)


REFERENCE = _brute_force()


def _sft_error(n: int, subpixel: bool) -> float:
    x = centered_grid(n, 4.0 / n)
    v = (_amp(x) * aperture_weights_1d(x, A, subpixel)).astype(np.complex128)
    k, res, _ = sft_1d(v, x, C)
    # interpolate the smooth residual only, then apply the analytic phase
    smooth = np.interp(K_TEST, k, res.real) + 1j * np.interp(K_TEST, k, res.imag)
    approx = smooth * np.exp(-0.5j * K_TEST**2 / C)
    return float(np.max(np.abs(approx - REFERENCE)) / np.max(np.abs(REFERENCE)))


def test_sft_matches_quadrature_with_first_order_convergence() -> None:
    hard = [_sft_error(n, False) for n in (4096, 16384, 65536)]
    soft = [_sft_error(n, True) for n in (4096, 16384)]
    assert hard[0] < 0.025 and hard[1] < 6e-3 and hard[2] < 2e-3  # paper: 1.9 %, 0.46 %, 0.13 %
    assert 2.5 < hard[0] / hard[1] < 5.5  # first order in the step (factor 4)
    assert soft[0] < hard[0] / 2 and soft[1] < hard[1] / 2  # subpixel weighting helps


def test_sft_2d_is_product_of_1d() -> None:
    n, step, cx, cy = 256, 0.01, 800.0, -500.0
    x = centered_grid(n, step)
    vx = np.exp(-((x / 0.5) ** 2)) + 0j
    vy = (np.abs(x) <= 0.6).astype(float) + 0j
    _, rx, px = sft_1d(vx, x, cx)
    _, ry, py = sft_1d(vy, x, cy)
    _, _, r2, p2 = sft_2d_diagonal(vy[:, None] * vx[None, :], step, cx, cy)
    assert np.allclose(p2, py[:, None] + px[None, :])
    assert np.max(np.abs(r2 - ry[:, None] * rx[None, :])) < 1e-12 * np.max(np.abs(r2))


def test_sft_reduces_to_pft_for_smooth_field() -> None:
    # Without edge, Ẽ(k) ≈ V(k/c)·exp(-ik²/(2c) + iπ/4)/sqrt(2π|c|).
    n, c = 4096, 2.0e4
    x = centered_grid(n, 4.0 / n)
    v = np.exp(-((x / 0.7) ** 2)) + 0j
    k, res, phase = sft_1d(v, x, c)
    sel = np.abs(x) < 0.5
    pft = v[sel] * np.exp(0.25j * np.pi) / np.sqrt(2 * np.pi * c)
    assert np.allclose(phase, -0.5 * k**2 / c)
    assert np.max(np.abs(res[sel] - pft)) < 1e-3 * np.max(np.abs(pft))


def test_circular_aperture_area() -> None:
    x = centered_grid(200, 0.01)
    w = circular_aperture_weights(x, x, 0.7, oversample=16)
    assert np.isclose(w.sum() * 0.01**2, np.pi * 0.49, rtol=1e-3)
