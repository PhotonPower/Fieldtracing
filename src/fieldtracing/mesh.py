"""Jacobian factor from unstructured ray clouds (paper, section 3.4, eq. jacobian_amplitude).

The ray mapping ρ₁ ↦ ρ₂ is known only at the start points of the rays. Following the paper's
Erkenntnis on the neighbourhood meshes, the area ratio J = dA₂/dA₁ = det ∂ρ₂/∂ρ₁ is recovered
from the neighbours of each ray:

- `triangulate`: Delaunay triangulation of the start points (`scipy.spatial.Delaunay`); triangles
  spanning an obscuration or a gap are removed (longest edge or region test).
- `triangle_area_ratio`: signed ratio dA₂/dA₁ of every triangle (the determinant of the affine
  map of the triangle) and its start-area-weighted mean at the nodes. On irregular clouds this
  nodal value is only first order in the spacing h.
- `fit_derivatives`: derivative matrix of any vector field sampled on the cloud by a local
  quadratic least-squares fit over the two-ring of mesh neighbours; error O(h²).
- `mesh_jacobian_cloud`: ∂ρ₂/∂ρ₁, its determinant, the Maslov index μ (negative eigenvalues,
  docs/konventionen.md) and caustic flags (|J| below a threshold or a sign change of J to a mesh
  neighbour). Near a caustic the geometric law itself fails (J → 0); the flags mark where.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import Delaunay

from fieldtracing._types import FloatArray, IntArray

Region = Callable[[FloatArray, FloatArray], NDArray[np.bool_]]

MIN_NEIGHBOURS = 5  # unknowns of the quadratic fit per component: ∂x, ∂y, ∂xx, ∂xy, ∂yy


def _edges(triangles: IntArray) -> IntArray:
    """Unique undirected edges (E, 2) of a triangle list."""
    pairs = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    return np.asarray(np.unique(np.sort(pairs, axis=1), axis=0), dtype=np.int64)


def _signed_areas(points: FloatArray, triangles: IntArray) -> FloatArray:
    """Signed areas (M,) of the triangles, positive for counter-clockwise vertices."""
    u = points[triangles[:, 1]] - points[triangles[:, 0]]
    v = points[triangles[:, 2]] - points[triangles[:, 0]]
    return np.asarray(0.5 * (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]))


def triangulate(
    points: FloatArray,
    max_edge: float | None = None,
    region: Region | None = None,
    min_quality: float = 0.2,
) -> IntArray:
    """Delaunay triangulation of a 2D point cloud without triangles over holes.

    Args:
        points: (N, 2) start points of the rays in mm.
        max_edge: drop triangles whose longest edge exceeds this length (mm). Removes triangles
            spanning a central obscuration or a gap and long slivers at the convex hull.
        region: optional predicate inside(x, y) of the illuminated region (mm); triangles whose
            centroid or edge midpoints lie outside are dropped (exact for known obscurations).
        min_quality: peel boundary slivers: triangles with an edge on the mesh boundary and a
            shape quality 4√3·A/(a² + b² + c²) below this value (1 for an equilateral
            triangle) are removed repeatedly. Delaunay fills the convex hull of a ragged cloud
            with such slivers; they couple distant boundary rays.

    Returns:
        (M, 3) vertex indices, counter-clockwise in the start plane.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
        raise ValueError("points must have shape (N, 2) with N ≥ 3")
    tri = np.asarray(Delaunay(p).simplices, dtype=np.int64)
    flip = _signed_areas(p, tri) < 0.0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    keep = np.ones(len(tri), dtype=bool)
    if max_edge is not None:
        longest = np.max(
            np.stack(
                [
                    np.linalg.norm(b - a, axis=1),
                    np.linalg.norm(c - b, axis=1),
                    np.linalg.norm(a - c, axis=1),
                ]
            ),
            axis=0,
        )
        keep &= longest <= max_edge
    if region is not None:
        for probe in ((a + b + c) / 3.0, (a + b) / 2.0, (b + c) / 2.0, (c + a) / 2.0):
            keep &= np.asarray(region(probe[:, 0], probe[:, 1]), dtype=bool)
    tri = tri[keep]
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    squares = np.sum((b - a) ** 2 + (c - b) ** 2 + (a - c) ** 2, axis=1)
    quality = 4.0 * np.sqrt(3.0) * np.abs(_signed_areas(p, tri)) / squares
    while len(tri):
        poor = (quality < min_quality) & _on_boundary(tri)
        if not np.any(poor):
            break
        tri, quality = tri[~poor], quality[~poor]
    return np.asarray(tri, dtype=np.int64)


def _on_boundary(triangles: IntArray) -> NDArray[np.bool_]:
    """True for triangles with at least one edge that belongs to no other triangle."""
    pairs = np.sort(
        np.stack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=1),
        axis=2,
    )  # (M, 3, 2)
    flat = pairs.reshape(-1, 2)
    _, inverse, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    single = (counts[np.ravel(inverse)] == 1).reshape(-1, 3)
    return np.asarray(np.any(single, axis=1))


def triangle_area_ratio(
    start: FloatArray, landing: FloatArray, triangles: IntArray
) -> tuple[FloatArray, FloatArray]:
    """Signed area ratio dA₂/dA₁ per triangle and its start-area-weighted mean per node.

    The ratio of a triangle is the determinant of the affine map that carries its start
    vertices to their landing points; a negative value means the triangle is flipped (a caustic
    lies between the planes). Nodes without a triangle get NaN.

    Args:
        start: (N, 2) start points in mm; landing: (N, 2) landing points in mm.
        triangles: (M, 3) counter-clockwise triangles from `triangulate`.

    Returns:
        (per-triangle ratio (M,), per-node ratio (N,)), dimensionless.
    """
    a1 = _signed_areas(np.asarray(start, dtype=np.float64), triangles)
    a2 = _signed_areas(np.asarray(landing, dtype=np.float64), triangles)
    ratio = a2 / a1
    n = len(start)
    weight = np.zeros(n)
    total = np.zeros(n)
    for j in range(3):
        np.add.at(weight, triangles[:, j], a1)
        np.add.at(total, triangles[:, j], a1 * ratio)
    node = np.full(n, np.nan)
    has = weight > 0.0
    node[has] = total[has] / weight[has]
    return np.asarray(ratio), node


def _two_ring(n: int, triangles: IntArray) -> csr_matrix:
    """Sparse pattern of the mesh neighbours up to two edges away (without the node itself)."""
    e = _edges(triangles)
    ones = np.ones(2 * len(e))
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    adj = csr_matrix(coo_matrix((ones, (rows, cols)), shape=(n, n)))
    ring = (adj + adj @ adj).tocsr()
    ring.setdiag(0)
    ring.eliminate_zeros()
    return ring


def fit_derivatives(
    points: FloatArray, values: FloatArray, triangles: IntArray
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Derivative matrices ∂v/∂ρ (N, C, 2) of a field sampled on a triangulated cloud.

    For every node i the fit v(ρ_j) - v(ρ_i) ≈ D·d + ½ dᵀ Q_c d with d = ρ_j - ρ_i runs over
    the two-ring of mesh neighbours (unweighted least squares, 5 unknowns per component, the
    node value is kept exactly). The derivative error is O(h²) in the spacing h; quadratic
    mappings are reproduced to rounding.

    Args:
        points: (N, 2) sample positions in mm.
        values: (N,) or (N, C) samples (any unit u).
        triangles: (M, 3) mesh from `triangulate`; it defines the neighbourhoods, so no fit
            reaches across a removed obscuration.

    Returns:
        (derivatives (N, C, 2) in u/mm, rows = components, columns = (∂/∂x, ∂/∂y); valid (N,):
        False where fewer than five neighbours or a degenerate neighbourhood make the fit
        undetermined; those rows are NaN).
    """
    p = np.asarray(points, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    scalar = v.ndim == 1
    v2 = v[:, None] if scalar else v
    n = len(p)
    ring = _two_ring(n, triangles)
    counts = np.diff(ring.indptr)
    width = int(np.max(counts)) if n else 0
    # padded neighbour lists; padding gets weight 0
    slot = np.arange(width)[None, :]
    mask = slot < counts[:, None]
    idx = np.zeros((n, width), dtype=np.int64)
    idx[mask] = ring.indices
    d = p[idx] - p[:, None, :]
    scale = np.sqrt(np.max(np.where(mask, np.sum(d * d, axis=2), 0.0), axis=1))
    scale = np.where(scale > 0.0, scale, 1.0)
    u = d / scale[:, None, None]
    basis = (
        np.stack(
            [
                u[..., 0],
                u[..., 1],
                0.5 * u[..., 0] ** 2,
                u[..., 0] * u[..., 1],
                0.5 * u[..., 1] ** 2,
            ],
            axis=-1,
        )
        * mask[..., None]
    )
    rhs = (v2[idx] - v2[:, None, :]) * mask[..., None]
    normal = np.einsum("nmi,nmj->nij", basis, basis)
    right = np.einsum("nmi,nmc->nic", basis, rhs)
    eig = np.linalg.eigvalsh(normal)
    valid = (counts >= MIN_NEIGHBOURS) & (eig[:, 0] > 1e-10 * np.maximum(eig[:, -1], 1e-300))
    out = np.full((n, v2.shape[1], 2), np.nan)
    if np.any(valid):
        coef = np.linalg.solve(normal[valid], right[valid])  # (n_valid, 5, C)
        grad = coef[:, :2, :] / scale[valid][:, None, None]
        out[valid] = np.transpose(grad, (0, 2, 1))
    return np.asarray(out), np.asarray(valid)


@dataclass(frozen=True)
class CloudJacobian:
    """Jacobian of a ray mapping on an unstructured cloud (paper, eq. jacobian_amplitude).

    Attributes:
        triangles: (M, 3) mesh of the start points.
        triangle_ratio: (M,) signed area ratio dA₂/dA₁ per triangle.
        area_ratio: (N,) start-area-weighted mean of `triangle_ratio` at the nodes (O(h)).
        matrix: (N, 2, 2) ∂ρ₂/∂ρ₁ from the quadratic fit (O(h²)); NaN where not `valid`.
        determinant: (N,) J = det ∂ρ₂/∂ρ₁, signed.
        maslov: (N,) μ = number of negative eigenvalues of ∂ρ₂/∂ρ₁ (homogeneous medium).
        caustic: (N,) True where |J| < threshold or J changes sign to a mesh neighbour; the
            geometric amplitude U/sqrt|J| is not reliable there.
        valid: (N,) False where the fit is undetermined (isolated or too few neighbours).
    """

    triangles: IntArray
    triangle_ratio: FloatArray
    area_ratio: FloatArray
    matrix: FloatArray
    determinant: FloatArray
    maslov: IntArray
    caustic: NDArray[np.bool_]
    valid: NDArray[np.bool_]


def mesh_jacobian_cloud(
    start: FloatArray,
    landing: FloatArray,
    triangles: IntArray | None = None,
    threshold: float = 1e-2,
    max_edge: float | None = None,
    region: Region | None = None,
) -> CloudJacobian:
    """Jacobian ∂ρ₂/∂ρ₁, J, Maslov index and caustic flags of an unstructured ray cloud.

    Args:
        start: (N, 2) ray start points in mm (any arrangement).
        landing: (N, 2) landing points of the same rays on the target plane in mm.
        triangles: mesh of the start points; built with `triangulate(start, max_edge, region)`
            if None.
        threshold: nodes with |J| below this value are flagged as caustic (dimensionless).
        max_edge, region: passed to `triangulate`.

    Returns:
        CloudJacobian. On a regular grid `matrix` agrees with `lpia.mesh_jacobian` to O(h²)
        (both are exact for quadratic mappings).
    """
    s = np.asarray(start, dtype=np.float64)
    t = np.asarray(landing, dtype=np.float64)
    if s.shape != t.shape or s.ndim != 2 or s.shape[1] != 2:
        raise ValueError("start and landing must both have shape (N, 2)")
    tri = triangulate(s, max_edge, region) if triangles is None else np.asarray(triangles)
    tri_ratio, node_ratio = triangle_area_ratio(s, t, tri)
    matrix, valid = fit_derivatives(s, t, tri)
    det = np.full(len(s), np.nan)
    maslov = np.zeros(len(s), dtype=np.int64)
    if np.any(valid):
        det[valid] = np.linalg.det(matrix[valid])
        maslov[valid] = np.sum(np.linalg.eigvals(matrix[valid]).real < 0.0, axis=1)
    caustic = valid & (np.abs(np.nan_to_num(det)) < threshold)
    e = _edges(tri)
    both = valid[e[:, 0]] & valid[e[:, 1]]
    flip = both & (np.sign(det[e[:, 0]]) != np.sign(det[e[:, 1]]))
    caustic[e[flip, 0]] = True
    caustic[e[flip, 1]] = True
    return CloudJacobian(tri, tri_ratio, node_ratio, matrix, det, maslov, caustic, valid)
