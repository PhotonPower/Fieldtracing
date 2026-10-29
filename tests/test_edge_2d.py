"""Issue #5: 2D apertures - boundary-diffraction integral (eq. bdw) and critical points.

References: rectangle with separable quadratic phase = product of two Fresnel integrals in closed
form; disc with isotropic quadratic phase = Hankel quadrature. Scaling ψ → sψ, k → sk keeps the
rays and multiplies the wavenumber by s.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erfc, j0

from fieldtracing._types import ComplexArray, FloatArray
from fieldtracing.edge import (
    boundary_integral,
    circle_asymptotics,
    circle_nodes,
    polygon_asymptotics,
    polygon_nodes,
)

A, B = 0.5, 0.3  # rectangle half widths (mm)
RECT = np.array([[-A, -B], [A, -B], [A, B], [-A, B]])
R = 0.5  # disc radius (mm)


def _ones(p: FloatArray) -> ComplexArray:
    return np.ones(len(p), dtype=np.complex128)


def _slit(c: float, a: float, q: float) -> complex:
    """∫_{-a}^{a} exp(i c x²/2 - i q x) dx in closed form."""
    sg = np.sign(c)

    def primitive(x: float) -> complex:
        arg = -np.exp(-0.25j * np.pi * sg) * np.sqrt(abs(c) / 2) * (x - q / c)
        return complex(np.sqrt(2 * np.pi / abs(c)) * np.exp(0.25j * np.pi * sg) * 0.5 * erfc(arg))

    return complex(np.exp(-0.5j * q * q / c) * (primitive(a) - primitive(-a)))


def _rect_case(
    s: float, k_rel: tuple[float, float]
) -> tuple[complex, complex, tuple[complex, ...]]:
    """(exact, PFT + numerical boundary integral, asymptotic terms) for the saddle phase."""
    cx, cy = 2000.0 * s, -1500.0 * s
    k = np.array([k_rel[0] * cx * A, k_rel[1] * cy * B])

    def phase(p: FloatArray) -> FloatArray:
        return np.asarray(0.5 * (cx * p[:, 0] ** 2 + cy * p[:, 1] ** 2))

    def gradient(p: FloatArray) -> FloatArray:
        return np.asarray(p * np.array([cx, cy]))

    exact = _slit(cx, A, float(k[0])) * _slit(cy, B, float(k[1]))
    terms = polygon_asymptotics(RECT, _ones, phase, gradient, np.diag([cx, cy]), k)
    numeric = terms[0] + boundary_integral(polygon_nodes(RECT, 4000), _ones, phase, gradient, k)
    return exact, numeric, terms


def _disc_case(s: float, k_rel: float) -> tuple[complex, complex, tuple[complex, complex]]:
    c = 3000.0 * s
    kr = k_rel * c * R
    k = kr * np.array([np.cos(0.3), np.sin(0.3)])

    def phase(p: FloatArray) -> FloatArray:
        return np.asarray(0.5 * c * np.sum(p * p, axis=1))

    def gradient(p: FloatArray) -> FloatArray:
        return np.asarray(c * p)

    def part(f: object) -> float:
        return float(quad(f, 0.0, R, limit=2000)[0])

    exact = complex(
        part(lambda r: 2 * np.pi * r * j0(kr * r) * np.cos(0.5 * c * r * r))
        + 1j * part(lambda r: 2 * np.pi * r * j0(kr * r) * np.sin(0.5 * c * r * r))
    )
    terms = circle_asymptotics(R, _ones, phase, gradient, c * np.eye(2), k)
    numeric = terms[0] + boundary_integral(circle_nodes(R, 40000), _ones, phase, gradient, k)
    return exact, numeric, terms


@pytest.mark.parametrize("k_rel", [0.5, 0.9, 1.2])
def test_disc_boundary_formula_is_exact_for_isotropic_phase(k_rel: float) -> None:
    # For U = 1 and H = c·I the field ∇Φ/|∇Φ|² is divergence-free in 2D, so PFT term plus
    # boundary integral is exact (lit, near-boundary and shadow frequencies).
    exact, numeric, _ = _disc_case(1.0, k_rel)
    assert abs(numeric - exact) < 1e-10 * abs(exact)


@pytest.mark.parametrize(("k_rel", "tol"), [(0.5, 5e-4), (0.9, 3e-2), (1.2, 5e-2)])
def test_disc_critical_points_converge(k_rel: float, tol: float) -> None:
    errors = []
    for s in (1.0, 4.0):
        exact, _, (first, second) = _disc_case(s, k_rel)
        errors.append(abs(first + second - exact) / abs(exact))
    assert errors[0] < tol
    assert errors[0] / errors[1] > 3.5  # four times the wavenumber


def test_rectangle_lit_region() -> None:
    # stationary point inside: PFT + edges (2nd kind) + corners (3rd kind)
    errs_num, errs_asym = [], []
    for s in (1.0, 4.0):
        exact, numeric, (first, second, third) = _rect_case(s, (0.2, 0.3))
        errs_num.append(abs(numeric - exact) / abs(exact))
        errs_asym.append(abs(first + second + third - exact) / abs(exact))
        assert abs(first - exact) > 10 * abs(first + second + third - exact)  # edges matter
    assert errs_asym[0] < 2e-3 and errs_num[0] < 3e-3
    assert errs_asym[0] / errs_asym[1] > 5.0 and errs_num[0] / errs_num[1] > 5.0


def test_rectangle_shadow_regions() -> None:
    # (1.3, 0.2): only 2nd-kind points on the y edges and corners; (1.4, 1.5): corners only
    for k_rel, tol in (((1.3, 0.2), 3e-2), ((1.4, 1.5), 2e-2)):
        errors = []
        for s in (1.0, 4.0):
            exact, _, (first, second, third) = _rect_case(s, k_rel)
            assert first == 0j
            errors.append(abs(second + third - exact) / abs(exact))
        assert errors[0] < tol and errors[1] < errors[0]
    exact, _, (_, second, _) = _rect_case(1.0, (1.4, 1.5))
    assert abs(second - exact) > 0.9 * abs(exact)  # without the corners nothing is left
