"""Uniform solution of the 2D aperture integral near shadow boundaries (paper, section 6).

`polygon_asymptotics(..., uniform=True)` and `circle_asymptotics(..., uniform=True)` replace the
coalescing stationary point (1st kind) and critical point of the 2nd kind by the Fresnel factor of
eq. (fresnel_edge); for polygons also the 2nd-kind point passing a vertex (3rd kind).

References, all independent of edge.py:
- rectangle with separable quadratic phase: product of two Fresnel integrals in closed form;
- disc with isotropic phase and U = 1: Hankel quadrature (scipy.integrate.quad);
- disc and triangle with rotated anisotropic H and Gaussian amplitude: direct 2D Gauss quadrature
  (polar coordinates resp. Duffy map), converged to < 1e-10 at 1.4 times the nodes.
Scaling ψ → sψ, k → sk keeps the rays and multiplies the wavenumber by s.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import erfc, j0

from fieldtracing._types import ComplexArray, FloatArray
from fieldtracing.edge import circle_asymptotics, polygon_asymptotics

A, B = 0.5, 0.3  # rectangle half widths (mm)
RECT = np.array([[-A, -B], [A, -B], [A, B], [-A, B]])
R = 0.5  # disc radius (mm)
TRIANGLE = np.array([[-0.4, -0.3], [0.5, -0.2], [0.0, 0.45]])  # counter-clockwise (mm)


def _ones(p: FloatArray) -> ComplexArray:
    return np.ones(len(p), dtype=np.complex128)


def _gauss(p: FloatArray) -> ComplexArray:
    """Gaussian amplitude (σ = 0.3 mm) with a linear slope, so that U(ρ_b) ≠ U(ρ_s)."""
    return np.asarray(
        np.exp(-np.sum(p * p, axis=1) / (2 * 0.3**2)) * (1.0 + 0.3 * p[:, 0]), dtype=np.complex128
    )


def _slit(c: float, a: float, q: float) -> complex:
    """∫_{-a}^{a} exp(i c x²/2 - i q x) dx in closed form."""
    sg = np.sign(c)

    def primitive(x: float) -> complex:
        arg = -np.exp(-0.25j * np.pi * sg) * np.sqrt(abs(c) / 2) * (x - q / c)
        return complex(np.sqrt(2 * np.pi / abs(c)) * np.exp(0.25j * np.pi * sg) * 0.5 * erfc(arg))

    return complex(np.exp(-0.5j * q * q / c) * (primitive(a) - primitive(-a)))


def _rect(s: float, k_rel: tuple[float, float], uniform: bool) -> tuple[complex, complex]:
    """(exact, sum of the asymptotic terms) for the saddle phase c_x = 2000 s, c_y = -1500 s."""
    cx, cy = 2000.0 * s, -1500.0 * s
    k = np.array([k_rel[0] * cx * A, k_rel[1] * cy * B])

    def phase(p: FloatArray) -> FloatArray:
        return np.asarray(0.5 * (cx * p[:, 0] ** 2 + cy * p[:, 1] ** 2))

    def gradient(p: FloatArray) -> FloatArray:
        return np.asarray(p * np.array([cx, cy]))

    exact = _slit(cx, A, float(k[0])) * _slit(cy, B, float(k[1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = polygon_asymptotics(
            RECT, _ones, phase, gradient, np.diag([cx, cy]), k, uniform=uniform
        )
    return exact, sum(terms, 0j)


def _rel(exact: complex, approx: complex) -> float:
    return abs(approx - exact) / abs(exact)


@pytest.mark.parametrize("ky", [0.9, 0.95, 1.05, 1.1])
def test_rectangle_across_shadow_boundary_of_an_edge(ky: float) -> None:
    # stationary point near the edge y = B (shadow side for ky > 1)
    errors = [_rel(*_rect(s, (0.5, ky), uniform=True)) for s in (1.0, 4.0)]
    assert errors[0] < 5e-4
    assert errors[0] / errors[1] > 7.0  # better than O(1/k) here
    assert _rel(*_rect(1.0, (0.5, ky), uniform=False)) > 100 * errors[0]


def test_rectangle_double_transition() -> None:
    # ky = 1: the stationary point lies on the edge y = B while the second-kind points of the edges
    # x = ±A sit on the corners (H diagonal in the edge axes). Here the error is O(1/k) and the
    # result has a small jump across the shadow line; the plain formula divides by zero.
    for ky in (1.0 - 1e-6, 1.0, 1.0 + 1e-6):
        errors = [_rel(*_rect(s, (0.5, ky), uniform=True)) for s in (1.0, 4.0, 16.0)]
        assert errors[0] < 4e-3
        assert errors[0] / errors[1] > 3.5 and errors[1] / errors[2] > 3.5
    below = _rect(1.0, (0.5, 1.0 - 1e-12), uniform=True)[1]
    above = _rect(1.0, (0.5, 1.0 + 1e-12), uniform=True)[1]
    assert abs(above - below) < 1e-3 * abs(above)


@pytest.mark.parametrize(("ky", "tol"), [(0.97, 5e-2), (0.99, 0.2), (1.01, 0.2), (1.03, 7e-2)])
def test_rectangle_vertex_transition(ky: float, tol: float) -> None:
    # double shadow (kx = 1.3): only a second-kind point on x = A, passing the corner (A, B);
    # the field there is small, errors are relative to it
    errors = [_rel(*_rect(s, (1.3, ky), uniform=True)) for s in (1.0, 4.0, 16.0)]
    assert errors[0] < tol
    assert errors[0] / errors[1] > 7.0 and errors[1] / errors[2] > 7.0
    assert _rel(*_rect(1.0, (1.3, ky), uniform=False)) > 10 * errors[0]


@pytest.mark.parametrize("k_rel", [(0.2, 0.3), (1.3, 0.2), (1.4, 1.5), (0.5, 0.6)])
def test_rectangle_far_from_transitions_not_worse(k_rel: tuple[float, float]) -> None:
    for s in (1.0, 4.0):
        assert _rel(*_rect(s, k_rel, uniform=True)) <= _rel(*_rect(s, k_rel, uniform=False))


def _disc_isotropic(s: float, k_rel: float) -> tuple[complex, complex]:
    c = 3000.0 * s
    kr = k_rel * c * R
    k = kr * np.array([np.cos(0.3), np.sin(0.3)])

    def phase(p: FloatArray) -> FloatArray:
        return np.asarray(0.5 * c * np.sum(p * p, axis=1))

    def gradient(p: FloatArray) -> FloatArray:
        return np.asarray(c * p)

    def part(f: object) -> float:
        return float(quad(f, 0.0, R, limit=4000)[0])

    exact = complex(
        part(lambda r: 2 * np.pi * r * j0(kr * r) * np.cos(0.5 * c * r * r))
        + 1j * part(lambda r: 2 * np.pi * r * j0(kr * r) * np.sin(0.5 * c * r * r))
    )
    terms = circle_asymptotics(R, _ones, phase, gradient, c * np.eye(2), k, uniform=True)
    return exact, sum(terms, 0j)


@pytest.mark.parametrize("k_rel", [0.9, 0.97, 1.0, 1.03, 1.1])
def test_disc_across_shadow_boundary(k_rel: float) -> None:
    # non-uniform: 30 % (0.97) up to a division by zero (1.0); uniform: < 3e-5 everywhere
    errors = [_rel(*_disc_isotropic(s, k_rel)) for s in (1.0, 4.0)]
    assert errors[0] < 3e-5
    assert errors[0] / errors[1] > 2.0  # O(1/k) or better (the curvature term limits the ratio)


def test_disc_continuous_on_the_shadow_boundary() -> None:
    # the remainder S - PFT·tail is regularised for |z| < 1e-3: no jump at Δ = 0
    # (Δ = 0 at k_rel = 1, |z| ≈ 2e-4 at k_rel = 1 ± 1e-7): the curve is smooth, i.e. the second
    # difference is far below the first
    values = [_disc_isotropic(1.0, 1.0 + d)[1] for d in (-1e-7, 0.0, 1e-7)]
    first = abs(values[2] - values[0])
    second = abs(values[2] - 2 * values[1] + values[0])
    assert second < 1e-3 * first


# --- general case: rotated anisotropic H, Gaussian amplitude, direct 2D quadrature -------------


def _hessian(s: float) -> FloatArray:
    c, sn = np.cos(0.4), np.sin(0.4)
    q = np.array([[c, -sn], [sn, c]])
    return np.asarray(s * q @ np.diag([3000.0, 2000.0]) @ q.T)


def _phase_fns(h: FloatArray) -> tuple[object, object]:
    def phase(p: FloatArray) -> FloatArray:
        return np.asarray(0.5 * np.einsum("ni,ij,nj->n", p, h, p))

    def gradient(p: FloatArray) -> FloatArray:
        return np.asarray(p @ h)

    return phase, gradient


def _reference_disc(h: FloatArray, k: FloatArray, n_r: int, n_phi: int) -> complex:
    x, w = leggauss(n_r)
    r, wr = 0.5 * R * (x + 1.0), 0.5 * R * w
    angles = 2.0 * np.pi * np.arange(n_phi) / n_phi
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    total = 0j
    for ri, wi in zip(r, wr, strict=True):
        p = ri * ring
        integrand = _gauss(p) * np.exp(1j * (0.5 * np.einsum("ni,ij,nj->n", p, h, p) - p @ k))
        total += wi * ri * (2.0 * np.pi / n_phi) * complex(np.sum(integrand))
    return total


def _reference_triangle(h: FloatArray, k: FloatArray, n: int) -> complex:
    # Duffy map p = v0 + u (v1 - v0) + u v (v2 - v1), Jacobian u·|det|
    x, w = leggauss(n)
    u, wu = 0.5 * (x + 1.0), 0.5 * w
    uu, vv = np.meshgrid(u, u, indexing="ij")
    weights = np.outer(wu, wu).ravel() * uu.ravel()
    v0, v1, v2 = TRIANGLE
    e1, e2 = v1 - v0, v2 - v1
    det = abs(e1[0] * e2[1] - e1[1] * e2[0])
    p = (v0 + uu[..., None] * e1 + (uu * vv)[..., None] * e2).reshape(-1, 2)
    integrand = _gauss(p) * np.exp(1j * (0.5 * np.einsum("ni,ij,nj->n", p, h, p) - p @ k))
    return complex(det * np.sum(weights * integrand))


def _disc_general(s: float, kappa: float, n_r: int, n_phi: int) -> tuple[complex, complex]:
    h = _hessian(s)
    phase, gradient = _phase_fns(h)
    k = h @ (kappa * R * np.array([np.cos(0.3), np.sin(0.3)]))  # ρ_s at kappa·R
    approx = sum(circle_asymptotics(R, _gauss, phase, gradient, h, k, uniform=True), 0j)  # type: ignore[arg-type]
    return _reference_disc(h, k, n_r, n_phi), approx


def _triangle_general(s: float, offset: float, n: int) -> tuple[complex, complex, complex]:
    h = _hessian(s)
    phase, gradient = _phase_fns(h)
    a, b = TRIANGLE[0], TRIANGLE[1]
    t = (b - a) / np.linalg.norm(b - a)
    k = h @ (a + 0.45 * (b - a) + offset * np.array([t[1], -t[0]]))  # ρ_s near edge 0
    with np.errstate(divide="ignore", invalid="ignore"):
        plain = sum(polygon_asymptotics(TRIANGLE, _gauss, phase, gradient, h, k), 0j)  # type: ignore[arg-type]
    approx = sum(
        polygon_asymptotics(TRIANGLE, _gauss, phase, gradient, h, k, uniform=True),  # type: ignore[arg-type]
        0j,
    )
    return _reference_triangle(h, k, n), approx, plain


@pytest.mark.parametrize("kappa", [0.97, 1.0, 1.03])
def test_disc_general_phase_and_amplitude(kappa: float) -> None:
    exact, approx = _disc_general(1.0, kappa, 800, 1600)
    assert _rel(exact, approx) < 4e-4


@pytest.mark.parametrize("offset", [-0.02, 0.0, 0.02])
def test_triangle_general_phase_and_amplitude(offset: float) -> None:
    exact, approx, plain = _triangle_general(1.0, offset, 1200)
    assert _rel(exact, approx) < 4e-3
    assert not abs(plain - exact) < 30 * abs(approx - exact)  # nan at offset 0 also passes


@pytest.mark.slow
@pytest.mark.parametrize("kappa", [0.97, 1.0, 1.03])
def test_disc_general_converges(kappa: float) -> None:
    e1 = _rel(*_disc_general(1.0, kappa, 800, 1600))
    e4 = _rel(*_disc_general(4.0, kappa, 3000, 12000))
    assert e1 / e4 > 3.0


@pytest.mark.slow
@pytest.mark.parametrize("offset", [-0.02, 0.0, 0.02])
def test_triangle_general_converges(offset: float) -> None:
    e1 = _rel(*_triangle_general(1.0, offset, 1200)[:2])
    e4 = _rel(*_triangle_general(4.0, offset, 2400)[:2])
    assert e1 / e4 > 3.5
