"""Edge diffraction of a truncated hybrid field (paper, section 6.1).

- `edge_wave_1d`: boundary term of the integration by parts, eq. (keller_gtd).
- `uniform_edge_factor`: uniform Fresnel solution, eq. (fresnel_edge), Ẽ = Ẽ_PFT·factor.
- 2D apertures (section 6.2): boundary-diffraction integral, eq. (bdw), and its asymptotics by
  critical points of the second kind (edges) and third kind (corners).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.special import erfc

from fieldtracing._types import ComplexArray, FloatArray


def uniform_edge_factor(x_s: FloatArray, x_e: float, curvature: FloatArray) -> ComplexArray:
    """½·erfc(exp(-iπ/4·sgn ψ'')·sqrt(|ψ''|/2)·(x_s - x_e)) for a field truncated at x ≤ x_e.

    Args:
        x_s: stationary points (mm).
        x_e: edge position (mm); the field is kept for x ≤ x_e.
        curvature: ψ''(x_s) in 1/mm².
    """
    arg = (
        np.exp(-0.25j * np.pi * np.sign(curvature)) * np.sqrt(np.abs(curvature) / 2.0) * (x_s - x_e)
    )
    return np.asarray(0.5 * erfc(arg))


def edge_wave_1d(
    amplitude_e: complex, phase_e: float, gradient_e: float, x_e: float, k: FloatArray
) -> ComplexArray:
    """Edge wave 1/(2πi)·U(x_e)/(ψ'(x_e) - k)·exp(i(ψ(x_e) - k x_e)) of a field cut at x_e."""
    return np.asarray(
        amplitude_e / (2j * np.pi * (gradient_e - k)) * np.exp(1j * (phase_e - k * x_e))
    )


# ---------------------------------------------------------------------------------------------
# Two-dimensional apertures (paper, section 6.2, eq. bdw)
# ---------------------------------------------------------------------------------------------

PointFunction = Callable[[FloatArray], ComplexArray]
RealPointFunction = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True)
class BoundaryNodes:
    """Quadrature nodes on a closed boundary ∂Ω traversed counter-clockwise.

    Attributes:
        points: (M, 2) node positions in mm.
        normals: (M, 2) outward unit normals.
        weights: (M,) arc-length weights in mm.
    """

    points: FloatArray
    normals: FloatArray
    weights: FloatArray


@lru_cache(maxsize=8)
def _leggauss(n: int) -> tuple[FloatArray, FloatArray]:
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return np.asarray(nodes), np.asarray(weights)


def polygon_nodes(vertices: FloatArray, per_edge: int = 2000) -> BoundaryNodes:
    """Gauss-Legendre nodes on every edge of a counter-clockwise polygon (vertices (V, 2), mm)."""
    v = np.asarray(vertices, dtype=np.float64)
    t, w = _leggauss(per_edge)
    points, normals, weights = [], [], []
    for i in range(len(v)):
        a, b = v[i], v[(i + 1) % len(v)]
        edge = b - a
        length = float(np.hypot(edge[0], edge[1]))
        tangent = edge / length
        points.append(a + 0.5 * (t[:, None] + 1.0) * edge)
        normals.append(np.tile([tangent[1], -tangent[0]], (per_edge, 1)))
        weights.append(0.5 * length * w)
    return BoundaryNodes(np.concatenate(points), np.concatenate(normals), np.concatenate(weights))


def circle_nodes(radius: float, count: int = 20000) -> BoundaryNodes:
    """Equidistant nodes on the circle of `radius` (mm) about the origin (spectral accuracy)."""
    theta = 2.0 * np.pi * np.arange(count) / count
    normals = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return BoundaryNodes(radius * normals, normals, np.full(count, 2.0 * np.pi * radius / count))


def boundary_integral(
    nodes: BoundaryNodes,
    amplitude: PointFunction,
    phase: RealPointFunction,
    gradient: RealPointFunction,
    k: FloatArray,
) -> complex:
    """Boundary-diffraction integral ∮ U·(ν·∇Φ)/(i|∇Φ|²)·exp(iΦ) dl with Φ = ψ - k·ρ (eq. bdw).

    Together with the PFT contributions of the stationary points inside Ω it approximates
    ∬_Ω U exp(iΦ) d²ρ; the neglected term ∬ exp(iΦ) ∇·F is one order of 1/k smaller.

    Args:
        nodes: quadrature nodes of ∂Ω.
        amplitude, phase, gradient: U, ψ (rad), ∇ψ (1/mm) as functions of (N, 2) points.
        k: transverse frequency (2,) in 1/mm.
    """
    p = nodes.points
    grad_phi = gradient(p) - k[None, :]
    big_phi = phase(p) - p @ k
    integrand = (
        amplitude(p)
        * np.sum(nodes.normals * grad_phi, axis=1)
        / (1j * np.sum(grad_phi * grad_phi, axis=1))
        * np.exp(1j * big_phi)
    )
    return complex(np.sum(nodes.weights * integrand))


def _stationary_area_term(
    point: FloatArray,
    amplitude: PointFunction,
    phase: RealPointFunction,
    hessian: FloatArray,
    k: FloatArray,
) -> complex:
    eig = np.linalg.eigvalsh(hessian)
    big_phi = float(phase(point[None, :])[0] - point @ k)
    return complex(
        2.0
        * np.pi
        * amplitude(point[None, :])[0]
        / np.sqrt(abs(np.prod(eig)))
        * np.exp(1j * (big_phi + 0.25 * np.pi * np.sum(np.sign(eig))))
    )


def polygon_asymptotics(
    vertices: FloatArray,
    amplitude: PointFunction,
    phase: RealPointFunction,
    gradient: RealPointFunction,
    hessian: FloatArray,
    k: FloatArray,
) -> tuple[complex, complex, complex]:
    """Asymptotic contributions to ∬_Ω U exp(iΦ) d²ρ for a polygon and a quadratic phase.

    ψ must be quadratic with constant Hessian H (1/mm²); then ∇Φ is affine, every edge has at
    most one critical point of the second kind (t·∇Φ = 0), and the corners are the critical points
    of the third kind (van Kampen 1949).

    Returns:
        (first kind: PFT term of the interior stationary point or 0,
         second kind: sum over edge points A·sqrt(2π/|Φ_ll|)·exp(iΦ + iπ/4·sgn Φ_ll),
         third kind: sum over edges of [A·exp(iΦ)/(iΦ_l)] between the end points),
        with A = U (ν·∇Φ)/(i|∇Φ|²). Contributions of the second kind are relative order k^(-1/2),
        of the third kind k^(-1).
    """
    v = np.asarray(vertices, dtype=np.float64)
    h = np.asarray(hessian, dtype=np.float64)
    # interior stationary point: ∇ψ(ρ) = k, solved with one Newton step from the origin
    g0 = gradient(np.zeros((1, 2)))[0]
    rho_s = np.linalg.solve(h, k - g0)
    first = _stationary_area_term(rho_s, amplitude, phase, h, k) if _inside(v, rho_s) else 0j

    def a_factor(p: FloatArray, normal: FloatArray) -> ComplexArray:
        gp = gradient(p) - k[None, :]
        return np.asarray(
            amplitude(p) * (gp @ normal) / (1j * np.sum(gp * gp, axis=1)), dtype=np.complex128
        )

    second, third = 0j, 0j
    for i in range(len(v)):
        a, b = v[i], v[(i + 1) % len(v)]
        edge = b - a
        length = float(np.hypot(edge[0], edge[1]))
        t = edge / length
        normal = np.array([t[1], -t[0]])
        # Φ_l(l) = t·(∇ψ(a + l t) - k) is affine in l with slope tᵀHt
        slope = float(t @ h @ t)
        phi_l0 = float(t @ (gradient(a[None, :])[0] - k))
        ends = np.stack([a, b])
        phi_l_ends = np.array([phi_l0, phi_l0 + slope * length])
        terms = a_factor(ends, normal) * np.exp(1j * (phase(ends) - ends @ k)) / (1j * phi_l_ends)
        third += complex(terms[1] - terms[0])
        if slope != 0.0:
            l_s = -phi_l0 / slope
            if 0.0 < l_s < length:
                p = (a + l_s * t)[None, :]
                second += complex(
                    a_factor(p, normal)[0]
                    * np.sqrt(2.0 * np.pi / abs(slope))
                    * np.exp(1j * (phase(p)[0] - p[0] @ k + 0.25 * np.pi * np.sign(slope)))
                )
    return first, second, third


def circle_asymptotics(
    radius: float,
    amplitude: PointFunction,
    phase: RealPointFunction,
    gradient: RealPointFunction,
    hessian: FloatArray,
    k: FloatArray,
) -> tuple[complex, complex]:
    """Asymptotic contributions for the disc |ρ| ≤ radius and a quadratic phase (constant H).

    Second-kind points satisfy t·∇Φ = 0 on the circle; there Φ_ll = tᵀHt - (ν·∇Φ)/R.

    Returns:
        (first kind: PFT term or 0, second kind: sum over the critical points of the boundary).
    """
    h = np.asarray(hessian, dtype=np.float64)
    g0 = gradient(np.zeros((1, 2)))[0]
    rho_s = np.linalg.solve(h, k - g0)
    first = (
        _stationary_area_term(rho_s, amplitude, phase, h, k)
        if float(rho_s @ rho_s) < radius * radius
        else 0j
    )

    def tangential(theta: float) -> float:
        p = radius * np.array([[np.cos(theta), np.sin(theta)]])
        t = np.array([-np.sin(theta), np.cos(theta)])
        return float(t @ (gradient(p)[0] - k))

    grid = np.linspace(0.0, 2.0 * np.pi, 4097)
    values = np.array([tangential(th) for th in grid])
    second = 0j
    for j in np.nonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)[0]:
        theta = brentq(tangential, float(grid[j]), float(grid[j + 1]))
        nu = np.array([np.cos(theta), np.sin(theta)])
        t = np.array([-np.sin(theta), np.cos(theta)])
        p = (radius * nu)[None, :]
        gp = gradient(p)[0] - k
        phi_ll = float(t @ h @ t - (nu @ gp) / radius)
        a = amplitude(p)[0] * (nu @ gp) / (1j * (gp @ gp))
        second += complex(
            a
            * np.sqrt(2.0 * np.pi / abs(phi_ll))
            * np.exp(1j * (phase(p)[0] - p[0] @ k + 0.25 * np.pi * np.sign(phi_ll)))
        )
    return first, second


def _inside(vertices: FloatArray, point: FloatArray) -> bool:
    """Point-in-polygon test (ray casting)."""
    x, y = float(point[0]), float(point[1])
    inside = False
    for i in range(len(vertices)):
        a, b = vertices[i], vertices[(i + 1) % len(vertices)]
        if (a[1] > y) != (b[1] > y):
            x_cross = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < x_cross:
                inside = not inside
    return inside
