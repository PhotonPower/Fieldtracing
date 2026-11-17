"""The C++ mesh operations (optional extension fieldtracing._mesh) give what the Python code
gives: the same edges and boundary triangles, and the quadratic fit (gradients, Hessians, valid
nodes) to rounding – on an irregular cloud with a hole and on nodes with too few neighbours."""

from __future__ import annotations

import numpy as np
import pytest

from fieldtracing import mesh
from fieldtracing._types import FloatArray, IntArray

pytest.importorskip("fieldtracing._mesh")


def _cloud() -> tuple[FloatArray, IntArray]:
    rng = np.random.default_rng(5)
    p = rng.uniform(-1.0, 1.0, size=(1500, 2))
    p = p[np.hypot(p[:, 0], p[:, 1]) > 0.25]  # a central obscuration
    tri = mesh.triangulate(p, max_edge=0.2)
    return p, tri


def _both(monkeypatch: pytest.MonkeyPatch, f, *args):  # type: ignore[no-untyped-def]
    fast = f(*args)
    monkeypatch.setattr(mesh, "USE_CPP", False)
    slow = f(*args)
    monkeypatch.setattr(mesh, "USE_CPP", True)
    return fast, slow


def test_edges_and_boundary_equal_python(monkeypatch: pytest.MonkeyPatch) -> None:
    _, tri = _cloud()
    fast, slow = _both(monkeypatch, mesh._edges, tri)
    assert np.array_equal(fast, slow)
    fast, slow = _both(monkeypatch, mesh._on_boundary, tri)
    assert np.array_equal(fast, slow) and fast.any() and not fast.all()


def test_triangulation_equals_python(monkeypatch: pytest.MonkeyPatch) -> None:
    p, _ = _cloud()
    fast, slow = _both(monkeypatch, mesh.triangulate, p, 0.2)
    assert np.array_equal(fast, slow)


@pytest.mark.parametrize("components", [1, 3])
def test_quadratic_fit_equals_python(monkeypatch: pytest.MonkeyPatch, components: int) -> None:
    p, tri = _cloud()
    x, y = p[:, 0], p[:, 1]
    v = np.stack([np.sin(2 * x) * np.cos(y) + k * x * y**2 for k in range(components)], axis=1)
    values = v[:, 0] if components == 1 else v
    tri = tri[: len(tri) - 40]  # a few nodes with too few neighbours at the end of the list
    (g1, h1, ok1), (g2, h2, ok2) = _both(monkeypatch, mesh._quadratic_fit, p, values, tri)
    assert np.array_equal(ok1, ok2) and ok1.any()
    assert np.allclose(g1[ok1], g2[ok2], rtol=1e-9, atol=1e-11)
    assert np.allclose(h1[ok1], h2[ok2], rtol=1e-8, atol=1e-9)
    assert np.isnan(g1[~ok1]).all() and np.isnan(h1[~ok1]).all()


def test_jacobian_of_a_lens_equals_python(monkeypatch: pytest.MonkeyPatch) -> None:
    p, _ = _cloud()
    landing = p * (1.0 - 0.3 * np.sum(p * p, axis=1))[:, None]  # spherical aberration
    fast, slow = _both(monkeypatch, mesh.mesh_jacobian_cloud, p, landing)
    assert np.array_equal(fast.triangles, slow.triangles)
    assert np.array_equal(fast.valid, slow.valid) and np.array_equal(fast.caustic, slow.caustic)
    assert np.allclose(fast.determinant[fast.valid], slow.determinant[slow.valid], rtol=1e-9)
