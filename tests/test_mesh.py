"""Issue #18: Jacobian factor from unstructured ray clouds (paper, section 3.4).

Reference: the analytic Jacobian of `semiclassical.propagate_2d`, which uses the analytic phase
Hessian (eq. jacobian_relation), never the landing points.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from fieldtracing._types import FloatArray
from fieldtracing.lpia import mesh_jacobian
from fieldtracing.mesh import (
    fit_derivatives,
    fit_second_derivatives,
    mesh_jacobian_cloud,
    triangle_area_ratio,
    triangulate,
)
from fieldtracing.semiclassical import GeometricResult, propagate_2d
from fieldtracing.units import wavenumber

K = wavenumber(0.5)
HALF = 2.0  # cloud half-width in mm


def _jittered(n: int, seed: int, jitter: float = 0.3) -> tuple[FloatArray, float]:
    """Regular n×n grid over ±HALF with uniform random offsets of ±jitter·h (fixed seed)."""
    h = 2.0 * HALF / (n - 1)
    x = np.linspace(-HALF, HALF, n)
    grid = np.stack(np.meshgrid(x, x), axis=-1).reshape(-1, 2)
    rng = np.random.default_rng(seed)
    return grid + rng.uniform(-jitter * h, jitter * h, grid.shape), h


def _astigmatic(points: FloatArray, fx: float, fy: float) -> tuple[FloatArray, ...]:
    """ψ = -k(sqrt(fx² + x²) - fx) - k(sqrt(fy² + y²) - fy), gradient and Hessian."""
    parts = []
    for coord, f in ((points[:, 0], fx), (points[:, 1], fy)):
        r = np.sqrt(f * f + coord * coord)
        parts.append((-K * (r - f), -K * coord / r, -K * f * f / r**3))
    hess = np.zeros((len(points), 2, 2))
    hess[:, 0, 0], hess[:, 1, 1] = parts[0][2], parts[1][2]
    return parts[0][0] + parts[1][0], np.stack([parts[0][1], parts[1][1]], axis=1), hess


def _propagate(points: FloatArray, fx: float, fy: float, d: float) -> GeometricResult:
    psi, grad, hess = _astigmatic(points, fx, fy)
    return propagate_2d(points, np.ones(len(points), np.complex128), psi, grad, hess, K, d)


# (fx, fy, d, μ): spherical (no caustic), between the line foci, behind a point focus and behind
# both line foci (as in test_maslov_2d.py)
CASES = [(25.0, 25.0, 10.0, 0), (20.0, 40.0, 30.0, 1), (20.0, 20.0, 40.0, 2), (20.0, 40.0, 60.0, 2)]


def _relative_error(n: int, fx: float, fy: float, d: float) -> tuple[float, float]:
    pts, _ = _jittered(n, seed=7)
    ref = _propagate(pts, fx, fy, d)
    cloud = mesh_jacobian_cloud(pts, ref.points)
    assert np.all(cloud.valid)
    assert np.all(cloud.maslov == ref.maslov)
    fit = float(np.max(np.abs(cloud.determinant - ref.jacobian) / np.abs(ref.jacobian)))
    area = float(np.sqrt(np.mean(((cloud.area_ratio - ref.jacobian) / ref.jacobian) ** 2)))
    return fit, area


@pytest.mark.parametrize(("fx", "fy", "d", "mu"), CASES)
def test_jacobian_from_jittered_cloud_is_second_order(
    fx: float, fy: float, d: float, mu: int
) -> None:
    coarse, area_coarse = _relative_error(41, fx, fy, d)
    fine, area_fine = _relative_error(81, fx, fy, d)
    assert np.all(_propagate(np.zeros((1, 2)), fx, fy, d).maslov == mu)
    assert coarse < 5e-4  # all nodes, including the border
    assert 3.0 < coarse / fine < 5.5  # O(h²)
    assert 1.6 < area_coarse / area_fine < 3.0  # triangle areas at the nodes: about O(h)


def test_uniform_random_cloud() -> None:
    errors = []
    for n in (400, 1600, 6400):
        pts = np.random.default_rng(3).uniform(-HALF, HALF, size=(n, 2))
        ref = _propagate(pts, 25.0, 25.0, 10.0)
        cloud = mesh_jacobian_cloud(pts, ref.points)
        assert np.all(cloud.valid)
        errors.append(float(np.max(np.abs(cloud.determinant / ref.jacobian - 1.0))))
    assert errors[0] < 3e-4
    assert errors[0] > errors[1] > errors[2]  # convergent (no fixed ratio on a Poisson cloud)


def test_affine_map_is_exact_and_flips_are_negative() -> None:
    pts, _ = _jittered(15, seed=1)
    a = np.array([[0.3, -1.2], [0.8, 0.5]])
    out = pts @ a.T + np.array([0.1, -0.2])
    tri = triangulate(pts)
    ratio, node = triangle_area_ratio(pts, out, tri)
    assert np.allclose(ratio, np.linalg.det(a), rtol=1e-12)
    assert np.allclose(node, np.linalg.det(a), rtol=1e-12)
    mirrored = out * np.array([-1.0, 1.0])
    assert np.all(triangle_area_ratio(pts, mirrored, tri)[0] < 0.0)
    deriv, valid = fit_derivatives(pts, out, tri)
    assert np.all(valid) and np.allclose(deriv, a, atol=1e-12)


def test_regular_grid_matches_lpia_mesh_jacobian() -> None:
    differences = []
    for n in (21, 41):
        x = np.linspace(-HALF, HALF, n)
        grid = np.stack(np.meshgrid(x, x), axis=-1)
        pts = grid.reshape(-1, 2)
        # quadratic mapping: both schemes are exact
        quad = pts + 0.01 * np.stack(
            [pts[:, 0] ** 2 + pts[:, 0] * pts[:, 1], 0.3 * pts[:, 1] ** 2 - pts[:, 0] * pts[:, 1]],
            axis=1,
        )
        ref = mesh_jacobian(grid, quad.reshape(n, n, 2)).reshape(-1, 2, 2)
        assert np.max(np.abs(mesh_jacobian_cloud(pts, quad).matrix - ref)) < 1e-12
        # general mapping: agreement to O(h²)
        landing = _propagate(pts, 25.0, 40.0, 10.0).points
        ref = mesh_jacobian(grid, landing.reshape(n, n, 2)).reshape(-1, 2, 2)
        differences.append(float(np.max(np.abs(mesh_jacobian_cloud(pts, landing).matrix - ref))))
    assert differences[0] < 5e-5
    assert 3.0 < differences[0] / differences[1] < 5.0


def test_central_obscuration_has_no_triangles_over_the_hole() -> None:
    inner, outer = 0.6, HALF
    pts, h = _jittered(41, seed=9)
    r = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts[(r > inner) & (r < outer)]

    def annulus(x: FloatArray, y: FloatArray) -> NDArray[np.bool_]:
        return np.asarray(np.hypot(x, y) > inner)

    plain = triangulate(pts)
    assert np.min(np.hypot(*pts[plain].mean(axis=1).T)) < inner / 2  # Delaunay fills the hole
    annulus_area = math.pi * (outer**2 - inner**2)
    for tri in (triangulate(pts, max_edge=2.0 * h), triangulate(pts, region=annulus)):
        centroid = pts[tri].mean(axis=1)
        assert np.min(np.hypot(centroid[:, 0], centroid[:, 1])) > inner
        a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
        u, v = b - a, c - a
        covered = float(np.sum(0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])))
        # the hole (area 1.13 mm²) stays open; the polygonal rims lose at most a strip of width h
        assert annulus_area - 2.0 * math.pi * (inner + outer) * h < covered < annulus_area
        ref = _propagate(pts, 25.0, 25.0, 10.0)
        cloud = mesh_jacobian_cloud(pts, ref.points, triangles=tri)
        assert np.all(cloud.valid)
        assert np.max(np.abs(cloud.determinant / ref.jacobian - 1.0)) < 5e-4


def _aberrated(points: FloatArray, f: float, b: float) -> tuple[FloatArray, ...]:
    """ψ = -k(ρ²/(2f) + bρ⁴/4): converging wave with spherical aberration."""
    r2 = np.sum(points * points, axis=1)
    psi = -K * (r2 / (2 * f) + b * r2**2 / 4)
    grad = -K * (1 / f + b * r2)[:, None] * points
    outer = np.einsum("ni,nj->nij", points, points)
    hess = -K * ((1 / f + b * r2)[:, None, None] * np.eye(2) + 2 * b * outer)
    return psi, grad, hess


@pytest.mark.parametrize("n", [41, 81])
def test_caustic_in_the_target_plane_is_flagged(n: int) -> None:
    # Marginal rays have crossed the caustic, paraxial ones not: J changes sign on a circle.
    f, d, b = 20.0, 18.0, 9.45e-4
    pts, h = _jittered(n, seed=7)
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < HALF]
    psi, grad, hess = _aberrated(pts, f, b)
    ref = propagate_2d(pts, np.ones(len(pts), np.complex128), psi, grad, hess, K, d)
    assert set(np.unique(ref.maslov)) == {0, 1}
    cloud = mesh_jacobian_cloud(pts, ref.points, threshold=0.0)
    r = np.hypot(pts[:, 0], pts[:, 1])
    r_caustic = float(r[np.argmin(np.abs(ref.jacobian))])
    assert np.all(np.abs(r[cloud.caustic] - r_caustic) < 2.0 * h)  # localised
    assert np.all(cloud.caustic[np.sign(cloud.determinant) != np.sign(ref.jacobian)])
    ok = ~cloud.caustic
    assert np.all(cloud.maslov[ok] == ref.maslov[ok])
    # the threshold flags exactly the small |J|
    flagged = mesh_jacobian_cloud(pts, ref.points, triangles=cloud.triangles, threshold=0.05)
    assert np.all(flagged.caustic[np.abs(flagged.determinant) < 0.05])


def test_invalid_input() -> None:
    with pytest.raises(ValueError):
        triangulate(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        mesh_jacobian_cloud(np.zeros((5, 2)), np.zeros((4, 2)))
    # a lone triangle has too few neighbours for the quadratic fit
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cloud = mesh_jacobian_cloud(pts, pts)
    assert not np.any(cloud.valid) and np.all(np.isnan(cloud.determinant))


def test_second_derivatives_of_a_quadratic_are_exact() -> None:
    pts, _ = _jittered(15, seed=2)
    q = np.array([[3.0, -1.5], [-1.5, 0.5]])
    values = np.stack([0.5 * np.einsum("ni,ij,nj->n", pts, q, pts) + pts[:, 0], pts[:, 1]], 1)
    hess, valid = fit_second_derivatives(pts, values, triangulate(pts))
    assert np.all(valid)
    assert np.allclose(hess[:, 0], q, atol=1e-10) and np.allclose(hess[:, 1], 0.0, atol=1e-10)
