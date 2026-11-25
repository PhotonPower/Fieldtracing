"""Issue #3: minimax curvature for the SFT (paper, sections 7.4 and 7.5)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linprog

from fieldtracing import selection
from fieldtracing._types import FloatArray
from fieldtracing.selection import fit_curvature, fit_curvature_radial, sft_samples
from fieldtracing.units import wavenumber

K = wavenumber(0.5)
A = 12.5  # aperture radius, mm (D = 25 mm)
F = A / math.tan(math.asin(0.5))  # NA 0.5 → f = 21.65 mm


def _spherical_gradient(points: FloatArray) -> FloatArray:
    """∇ψ of ψ = -k(sqrt(f² + ρ²) - f) at (N, 2) points."""
    r = np.hypot(points[:, 0], points[:, 1])
    return np.asarray(-K * points / np.sqrt(F * F + r * r)[:, None])


def _disc_samples(n: int, rim: int) -> FloatArray:
    x = np.linspace(-A, A, n)
    xx, yy = np.meshgrid(x, x)
    inside = xx**2 + yy**2 <= A * A
    angles = np.linspace(0.0, 2 * np.pi, rim, endpoint=False)
    rim_points = A * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return np.concatenate([np.stack([xx[inside], yy[inside]], axis=1), rim_points])


def test_radial_fit_reproduces_section_7_5() -> None:
    rho = np.linspace(0.0, A, 20001)
    gradient = -K * rho / np.sqrt(F * F + rho * rho)
    c, residual = fit_curvature_radial(rho, gradient)
    assert c == pytest.approx(-521.0, rel=1e-2)  # f_eff = 24.1 mm
    assert -K / c == pytest.approx(24.1, abs=0.05)
    assert residual == pytest.approx(230.0, rel=1e-2)
    paraxial = float(np.max(np.abs(gradient + K / F * rho)))  # vertex curvature -k/f
    assert paraxial == pytest.approx(972.0, rel=1e-2)


def test_matrix_fit_on_disc_is_isotropic() -> None:
    points = _disc_samples(121, 720)
    c_matrix, residual = fit_curvature(points, _spherical_gradient(points))
    assert np.allclose(c_matrix, -521.04 * np.eye(2), atol=1.0)
    assert residual == pytest.approx(230.0, rel=1e-2)


def test_quadratic_phase_is_fitted_exactly() -> None:
    rng = np.random.default_rng(3)
    points = rng.uniform(-1.0, 1.0, size=(500, 2))
    c0 = np.array([[700.0, -150.0], [-150.0, -300.0]])
    c_matrix, residual = fit_curvature(points, points @ c0)  # ∇(½ρᵀC₀ρ) = C₀ρ
    assert np.allclose(c_matrix, c0, atol=1e-6)
    assert residual < 1e-6


def test_sample_counts_of_the_example() -> None:
    step, n = sft_samples(30.0, -521.0, 10.0, 230.0)
    assert step * 1e3 == pytest.approx(6.2, abs=0.05)  # limited by w_c/q
    assert n == pytest.approx(4840, rel=1e-2)
    step_par, n_par = sft_samples(30.0, -K / F, 10.0, 972.0)
    assert step_par == pytest.approx(math.pi / 972.0)  # limited by the residual phase
    assert n_par > 1.8 * n


DATA = Path(__file__).parent / "data" / "curvature_loop_86.npz"


def _full_lp(points: FloatArray, gradient: FloatArray) -> float:
    """Optimum of the minimax program over all samples, solved directly (independent of the
    subset loop): min t subject to |g_x - c11 x - c12 y| ≤ t, |g_y - c12 x - c22 y| ≤ t."""
    x, y = points[:, 0], points[:, 1]
    zero, one = np.zeros_like(x), np.ones_like(x)
    rx = np.stack([x, y, zero], axis=1)
    ry = np.stack([zero, x, y], axis=1)
    a = np.concatenate(
        [
            np.column_stack([-rx, -one]),
            np.column_stack([rx, -one]),
            np.column_stack([-ry, -one]),
            np.column_stack([ry, -one]),
        ]
    )
    b = np.concatenate([-gradient[:, 0], gradient[:, 0], -gradient[:, 1], gradient[:, 1]])
    bounds = [(None, None), (None, None), (None, None), (0.0, None)]
    res = linprog([0.0, 0.0, 0.0, 1.0], A_ub=a, b_ub=b, bounds=bounds)
    assert res.success
    return float(res.x[3])


def test_the_fit_ends_when_the_subset_stops_growing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue #86: 2650 samples of ∇ψ behind a tube lens (PhotonPower/RTT-Fusion#135) on which the
    subset stopped growing at 407 samples – every violated sample was in it already, HiGHS
    solving to about 1e-7 against the loop's 1e-9 – and the same program was solved forever.
    The fit ends after a few programs with a bound that holds on every sample and is the
    optimum of the whole program."""
    data = np.load(DATA)
    points, gradient = np.asarray(data["points"]), np.asarray(data["gradient"])
    calls: list[int] = []
    lp = selection._curvature_lp

    def counting(p: FloatArray, g: FloatArray) -> tuple[FloatArray, float]:
        calls.append(len(p))
        if len(calls) > 50:
            raise AssertionError(f"the fit does not end: subsets {calls[:3]} … {calls[-3:]}")
        return lp(p, g)

    monkeypatch.setattr(selection, "_curvature_lp", counting)
    c_matrix, bound = fit_curvature(points, gradient)
    assert len(calls) < 10, calls
    residual = np.max(np.abs(gradient - points @ c_matrix), axis=1)
    assert bound >= float(residual.max())  # the bound holds everywhere
    assert bound == pytest.approx(_full_lp(points, gradient), rel=1e-6)
