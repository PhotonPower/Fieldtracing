"""Issue #21: vectorised Fresnel operator and refraction against the per-ray functions (T8)."""

from __future__ import annotations

import numpy as np

from fieldtracing._types import FloatArray
from fieldtracing.lpia import (
    fresnel_coefficients,
    fresnel_coefficients_batch,
    fresnel_operator,
    fresnel_operator_batch,
    refract,
    refract_batch,
    trace_surface,
)
from fieldtracing.units import wavenumber

N1, N2 = 1.0, 1.5


def _rays(n: int, seed: int) -> tuple[FloatArray, FloatArray]:
    """Random forward directions (θ < 70°) and random tilted normals of either orientation."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 1.2, n)
    phi = rng.uniform(0.0, 2 * np.pi, n)
    k = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], 1)
    normal = np.concatenate([rng.normal(scale=0.3, size=(n, 2)), np.ones((n, 1))], axis=1)
    normal *= rng.choice([-1.0, 1.0], size=(n, 1)) * rng.uniform(0.5, 2.0, size=(n, 1))
    return k, normal


def test_batch_equals_per_ray_functions() -> None:
    k, normal = _rays(500, seed=3)
    k[:3] = [0.0, 0.0, 1.0]  # normal incidence, special choice of s
    normal[:3] = [[0.0, 0.0, 2.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]
    out, op, tir = fresnel_operator_batch(k, normal, N1, N2)
    assert not np.any(tir)
    kout, _ = refract_batch(N1 * k, normal, N2)
    for i in range(len(k)):
        ref_out, ref_op = fresnel_operator(k[i], normal[i], N1, N2)
        assert np.allclose(out[i], ref_out, rtol=0, atol=1e-14)
        assert np.allclose(op[i], ref_op, rtol=0, atol=1e-14)
        assert np.allclose(kout[i], refract(N1 * k[i], normal[i], N2), rtol=0, atol=1e-14)
    cos_i = np.linspace(0.05, 1.0, 50)
    batch = fresnel_coefficients_batch(N1, N2, cos_i)
    for i, c in enumerate(cos_i):
        ref = fresnel_coefficients(N1, N2, float(c))
        assert abs(batch.t_p[i] - ref.t_p) < 1e-15 and abs(batch.r_s[i] - ref.r_s) < 1e-15


def test_total_reflection_is_a_mask() -> None:
    k, normal = _rays(200, seed=5)
    out, op, tir = fresnel_operator_batch(k, normal, N2, N1)  # glass → air
    cos_i = np.abs(np.sum(k * normal, axis=1)) / np.linalg.norm(normal, axis=1)
    assert np.array_equal(tir, (N2 / N1) ** 2 * (1 - cos_i**2) > 1.0)
    assert 0 < np.sum(tir) < len(k)
    assert np.all(np.isnan(out[tir])) and np.all(op[tir] == 0.0)
    assert np.all(np.isfinite(out[~tir]))
    # beyond the critical angle |r_s| = |r_p| = 1 (no exception)
    co = fresnel_coefficients_batch(N2, N1, np.array([0.3, 0.5]))
    assert np.allclose(np.abs(co.r_s), 1.0) and np.allclose(np.abs(co.r_p), 1.0)


def test_trace_surface_flags_totally_reflected_rays() -> None:
    # tilted plane wave in glass hitting a plane surface: θ = 50° > critical angle 41.8°
    k0 = wavenumber(0.5)
    k1 = N2 * k0
    pts = np.stack(np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)), axis=-1)
    grad = np.zeros_like(pts)
    grad[:3, :, 0] = k1 * np.sin(np.radians(50.0))  # first three rows totally reflected
    grad[3:, :, 0] = k1 * np.sin(np.radians(20.0))
    amp = np.zeros((5, 5, 3), np.complex128)
    amp[..., 1] = 1.0  # s-polarised (y ⟂ plane of incidence)
    trace = trace_surface(
        pts,
        amp,
        np.zeros((5, 5)),
        grad,
        N2,
        N1,
        k0,
        lambda x, y: 2.0 + 0 * x,
        lambda x, y: (0 * x, 0 * y),
        5.0,
    )
    assert np.all(trace.total_reflection[:3]) and not np.any(trace.total_reflection[3:])
    assert np.all(trace.transmitted[:3] == 0.0) and np.all(np.isnan(trace.points[:3]))
    t_s = fresnel_coefficients(N2, N1, float(np.cos(np.radians(20.0)))).t_s
    assert np.allclose(trace.transmitted[3:, :, 1], t_s, rtol=1e-12)
