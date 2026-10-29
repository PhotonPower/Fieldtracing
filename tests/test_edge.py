"""T5 of the paper: uniform erfc edge formula vs. quadrature; asymptotic edge-wave limit."""

from __future__ import annotations

import numpy as np
import pytest

from fieldtracing._types import FloatArray
from fieldtracing.edge import edge_wave_1d, uniform_edge_factor

C, XE = 2000.0, 0.3  # ψ = C x²/2 (1/mm²), field kept for x ≤ XE (mm)


def _amp(x: FloatArray | float) -> FloatArray:
    return np.exp(-(np.asarray(x) ** 2) / 4)


def _pft(k: float) -> complex:
    xs = k / C
    return complex(
        np.sqrt(2 * np.pi / C)
        * np.exp(0.25j * np.pi)
        * _amp(xs)
        * np.exp(1j * (C * xs**2 / 2 - k * xs))
    ) / (2 * np.pi)


@pytest.mark.parametrize("offset", [-0.05, -0.01, 0.0, 0.01, 0.05])
def test_uniform_formula_matches_quadrature(offset: float) -> None:
    x = np.linspace(-3.0, XE, 2_000_001)
    k = C * (XE + offset)
    exact = np.trapezoid(_amp(x) * np.exp(1j * (C * x**2 / 2 - k * x)), dx=x[1] - x[0]) / (
        2 * np.pi
    )
    uniform = _pft(k) * complex(uniform_edge_factor(np.array([k / C]), XE, np.array([C]))[0])
    assert abs(uniform - exact) / abs(exact) < 1e-2  # paper: 0.1-0.9 %


def test_shadow_boundary_halves_amplitude() -> None:
    assert np.isclose(uniform_edge_factor(np.array([XE]), XE, np.array([C]))[0], 0.5)


def test_uniform_formula_tends_to_edge_wave_in_shadow() -> None:
    k = C * (XE + 0.5)  # deep shadow, |z| = sqrt(C/2)·0.5 ≈ 16
    xs = k / C
    uniform = _pft(k) * complex(uniform_edge_factor(np.array([xs]), XE, np.array([C]))[0])
    # same amplitude at the edge as the PFT uses at x_s (asymptotic statement for U ≈ const)
    wave = complex(edge_wave_1d(complex(_amp(xs)), C * XE**2 / 2, C * XE, XE, np.array([k]))[0])
    assert abs(uniform - wave) / abs(wave) < 5e-3
