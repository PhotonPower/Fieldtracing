"""Issue #6: LPIA step through a curved surface with a ray mesh (paper, eq. interface_amplitude)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from fieldtracing._types import ComplexArray, FloatArray
from fieldtracing.lpia import (
    fresnel_coefficients,
    fresnel_operator,
    interface_amplitude,
    mesh_jacobian,
    trace_surface,
)
from fieldtracing.units import wavenumber

K0 = wavenumber(0.5)
N1, N2 = 1.0, 1.5
K1 = N1 * K0
T_NORMAL = fresnel_coefficients(N1, N2, 1.0).t_s

Sag = Callable[[FloatArray, FloatArray], FloatArray]
SagGradient = Callable[[FloatArray, FloatArray], tuple[FloatArray, FloatArray]]


def _grid(half: float, n: int) -> FloatArray:
    x = np.linspace(-half, half, n)
    xx, yy = np.meshgrid(x, x)
    return np.stack([xx, yy], axis=-1)


def _transversal_polarisation(gradient: FloatArray) -> ComplexArray:
    """Unit field vectors perpendicular to the rays (k⊥ = ∇ψ in medium n1)."""
    kz = np.sqrt(K1 * K1 - np.sum(gradient * gradient, axis=-1))
    s = np.concatenate([gradient, kz[..., None]], axis=-1) / K1
    pol = np.cross(s, np.array([1.0, 0.0, 0.0]))
    return np.asarray(pol / np.linalg.norm(pol, axis=-1, keepdims=True), dtype=np.complex128)


def _plane(z: float) -> tuple[Sag, SagGradient]:
    return (lambda x, y: z + 0 * x), (lambda x, y: (0 * x, 0 * y))


def _sphere(z_center: float, radius: float, sign: float) -> tuple[Sag, SagGradient]:
    """Sphere about (0, 0, z_center); sign = -1: the part facing the incoming light."""

    def sag(x: FloatArray, y: FloatArray) -> FloatArray:
        return np.asarray(z_center + sign * np.sqrt(radius**2 - x * x - y * y))

    def grad(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        root = np.sqrt(radius**2 - x * x - y * y)
        return np.asarray(-sign * x / root), np.asarray(-sign * y / root)

    return sag, grad


def test_tilted_plane_wave_through_plane_is_fresnel() -> None:
    pts = _grid(1.0, 21)
    alpha, beta = 0.4, 0.7
    direction = np.array(
        [np.sin(alpha) * np.cos(beta), np.sin(alpha) * np.sin(beta), np.cos(alpha)]
    )
    grad = np.broadcast_to(K1 * direction[:2], pts.shape).copy()
    u = np.cross(direction, [0.0, 0.0, 1.0])
    u /= np.linalg.norm(u)
    amp = np.broadcast_to(u.astype(np.complex128), (*pts.shape[:2], 3)).copy()
    sag, sag_grad = _plane(5.0)
    trace = trace_surface(pts, amp, pts @ grad[0, 0], grad, N1, N2, K0, sag, sag_grad, 12.0)
    jac = mesh_jacobian(pts, trace.points)
    out = interface_amplitude(trace, jac)
    _, op = fresnel_operator(direction, np.array([0.0, 0.0, 1.0]), N1, N2)
    assert np.allclose(np.linalg.det(jac), 1.0, atol=1e-12)
    assert np.max(np.abs(out - op @ u)) < 1e-12
    assert np.max(np.abs(np.sum(out * trace.direction_out, axis=-1))) < 1e-12  # transversal


def test_sphere_about_the_focus_is_hit_normally() -> None:
    # Converging wave to C = (0, 0, 30); surface = sphere of radius 20 about C. Every ray hits
    # normally, so |E| = |t|·|E₁|·r₁/r₂ (spherical wave) and the phase is k₀ times the path.
    z_c, radius, z_out = 30.0, 20.0, 20.0
    pts = _grid(3.0, 41)
    r1 = np.sqrt(np.sum(pts**2, axis=-1) + z_c**2)
    phase = -K1 * (r1 - z_c)
    grad = -K1 * pts / r1[..., None]
    sag, sag_grad = _sphere(z_c, radius, -1.0)
    trace = trace_surface(
        pts, _transversal_polarisation(grad), phase, grad, N1, N2, K0, sag, sag_grad, z_out
    )
    out = interface_amplitude(trace, mesh_jacobian(pts, trace.points))
    r2 = np.sqrt(np.sum(trace.points**2, axis=-1) + (z_c - z_out) ** 2)
    expected = abs(T_NORMAL) * r1 / r2
    assert np.max(np.abs(np.linalg.norm(out, axis=-1) - expected) / expected) < 1e-12
    path_phase = phase + K0 * (N1 * (r1 - radius) + N2 * (radius - r2))
    assert np.max(np.abs(trace.phase - path_phase)) < 1e-8 * np.max(np.abs(path_phase))


def _plane_interface_jacobian_error(n: int) -> float:
    """Mesh Jacobian against the analytic one for a diverging wave refracted at a plane."""
    r0, z_s, z_out = 30.0, 5.0, 25.0
    pts = _grid(4.0, n)
    rho = np.hypot(pts[..., 0], pts[..., 1])
    rr = np.sqrt(rho**2 + r0**2)
    grad = K1 * pts / rr[..., None]
    sag, sag_grad = _plane(z_s)
    trace = trace_surface(
        pts, _transversal_polarisation(grad), K1 * (rr - r0), grad, N1, N2, K0, sag, sag_grad, z_out
    )
    mesh = np.linalg.det(mesh_jacobian(pts, trace.points))
    s1 = rho / rr
    c1 = np.sqrt(1 - s1**2)
    s2 = N1 / N2 * s1
    c2 = np.sqrt(1 - s2**2)
    rho2 = rho + z_s * s1 / c1 + (z_out - z_s) * s2 / c2  # radial landing point
    d_rho2 = 1 + (z_s / c1**3 + (z_out - z_s) * (N1 / N2) / c2**3) * r0**2 / rr**3
    safe = np.where(rho > 0, rho, 1.0)
    analytic = np.where(rho > 0, rho2 / safe * d_rho2, d_rho2**2)  # J = (ρ₂/ρ₁)·dρ₂/dρ₁
    assert np.max(np.abs(np.hypot(trace.points[..., 0], trace.points[..., 1]) - rho2)) < 1e-12
    return float(np.max(np.abs(mesh - analytic) / analytic))


def test_mesh_jacobian_converges_second_order() -> None:
    coarse, fine = _plane_interface_jacobian_error(41), _plane_interface_jacobian_error(81)
    assert coarse < 1e-3
    assert 3.0 < coarse / fine < 5.0


@pytest.mark.parametrize(("d", "mu"), [(10.0, 0), (30.0, 0), (50.0, 0), (90.0, 2)])
def test_plane_wave_through_sphere_on_axis(d: float, mu: int) -> None:
    # Paraxial focal length f' = n₂R/(n₂ - n₁) behind the vertex; on the axis
    # U₂ = t·exp(-iπμ/2)/|1 - d/f'|, J = (1 - d/f')²; behind the focus μ = 2 (Gouy phase -π).
    radius, z_v = 20.0, 5.0
    f_image = N2 * radius / (N2 - N1)
    pts = _grid(0.2, 21)
    amp = np.zeros((*pts.shape[:2], 3), dtype=np.complex128)
    amp[..., 0] = 1.0
    sag, sag_grad = _sphere(z_v + radius, radius, -1.0)
    trace = trace_surface(
        pts, amp, np.zeros(pts.shape[:2]), np.zeros(pts.shape), N1, N2, K0, sag, sag_grad, z_v + d
    )
    jac = mesh_jacobian(pts, trace.points)
    out = interface_amplitude(trace, jac)
    expected = T_NORMAL * np.exp(-0.5j * np.pi * mu) / abs(1 - d / f_image)
    assert abs(np.linalg.det(jac[10, 10]) - (1 - d / f_image) ** 2) < 1e-5
    assert abs(out[10, 10, 0] - expected) < 1e-5 * abs(expected)
