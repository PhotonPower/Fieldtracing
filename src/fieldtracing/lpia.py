"""Local plane interface approximation (paper, section 3.2).

Fresnel coefficients in Convention A (Byrnes, arXiv:1603.02720v5, eq. (6)), as in Raytatouille:

    r_s = (n1 cos i - n2 cos t)/(n1 cos i + n2 cos t),
    r_p = (n2 cos i - n1 cos t)/(n2 cos i + n1 cos t),
    t_s = 2 n1 cos i/(n1 cos i + n2 cos t),
    t_p = 2 n1 cos i/(n2 cos i + n1 cos t),

basis s = k̂_in × N/|k̂_in × N|, p = k̂ × s for every wave, N pointing into medium 2.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fieldtracing._types import ComplexArray, FloatArray


@dataclass(frozen=True)
class FresnelCoefficients:
    """Amplitude coefficients (complex) and the cosine of the transmission angle."""

    r_s: complex
    r_p: complex
    t_s: complex
    t_p: complex
    cos_t: complex


def fresnel_coefficients(n1: float, n2: float, cos_i: float) -> FresnelCoefficients:
    """Fresnel amplitude coefficients for real indices, Convention A.

    Args:
        n1, n2: refractive indices of the incidence and transmission medium.
        cos_i: cosine of the angle of incidence (0 < cos_i ≤ 1).
    """
    sin_t_sq = (n1 / n2) ** 2 * (1.0 - cos_i * cos_i)
    cos_t = complex(np.sqrt(1.0 - sin_t_sq + 0j))
    a, b = n1 * cos_i, n2 * cos_t
    c, d = n2 * cos_i, n1 * cos_t
    return FresnelCoefficients(
        (a - b) / (a + b), (c - d) / (c + d), 2 * a / (a + b), 2 * a / (c + d), cos_t
    )


def _unit(v: FloatArray) -> FloatArray:
    return np.asarray(v / np.linalg.norm(v))


def refract(k_in: FloatArray, normal: FloatArray, k2: float) -> FloatArray:
    """Vector Snell law: k_out = k_in + (sqrt(k2² - |k_in|² + (k_in·N)²) - k_in·N)·N.

    Args:
        k_in: incident wave vector (1/mm).
        normal: unit surface normal; it is flipped to point into medium 2 if needed.
        k2: wavenumber n2·k0 of medium 2 (1/mm).

    Raises:
        ValueError: on total internal reflection.
    """
    n = _unit(normal)
    if np.dot(k_in, n) < 0.0:
        n = -n
    kn = float(np.dot(k_in, n))
    disc = k2 * k2 - float(np.dot(k_in, k_in)) + kn * kn
    if disc < 0.0:
        raise ValueError("total internal reflection")
    return np.asarray(k_in + (np.sqrt(disc) - kn) * n)


def fresnel_operator(
    k_in: FloatArray, normal: FloatArray, n1: float, n2: float
) -> tuple[FloatArray, ComplexArray]:
    """Transmission operator T = t_s s sᵀ + t_p p_out p_inᵀ acting on the full field vector.

    Args:
        k_in: incident direction or wave vector (only its direction is used).
        normal: surface normal (orientation is irrelevant).
        n1, n2: refractive indices.

    Returns:
        (unit direction of the transmitted wave, 3×3 complex T).
    """
    k_hat = _unit(k_in)
    n = _unit(normal)
    if np.dot(k_hat, n) < 0.0:
        n = -n
    out_hat = _unit(refract(n1 * k_hat, n, n2))
    cross = np.cross(k_hat, n)
    if np.linalg.norm(cross) < 1e-12:  # normal incidence: any s ⟂ N
        helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        cross = np.cross(helper, n)
    s = _unit(cross)
    p_in = np.cross(k_hat, s)
    p_out = np.cross(out_hat, s)
    co = fresnel_coefficients(n1, n2, float(np.dot(k_hat, n)))
    t = co.t_s * np.outer(s, s) + co.t_p * np.outer(p_out, p_in)
    return out_hat, np.asarray(t, dtype=np.complex128)


# ---------------------------------------------------------------------------------------------
# LPIA step through a curved surface with a ray mesh (paper, sections 3.2 and 3.5)
# ---------------------------------------------------------------------------------------------

SagFunction = Callable[[FloatArray, FloatArray], FloatArray]


@dataclass(frozen=True)
class SurfaceTrace:
    """Rays of a gridded hybrid field traced through one surface to an output plane.

    All arrays have the shape of the input grid (Ny, Nx) plus trailing components.

    Attributes:
        points: landing points (..., 2) on the output plane in mm.
        phase: ψ₂ = ψ₁ + k₀(n₁ L₁ + n₂ L₂) in rad.
        transmitted: T·U₁ (..., 3), the Fresnel operator applied at the surface point.
        direction_in, direction_out: unit ray directions (..., 3) before and after the surface.
        cos_in, cos_out: cos θ_e, cos θ_a against the surface normal.
    """

    points: FloatArray
    phase: FloatArray
    transmitted: ComplexArray
    direction_in: FloatArray
    direction_out: FloatArray
    cos_in: FloatArray
    cos_out: FloatArray


def trace_surface(
    points: FloatArray,
    amplitude: ComplexArray,
    phase: FloatArray,
    gradient: FloatArray,
    n1: float,
    n2: float,
    k0: float,
    sag: SagFunction,
    sag_gradient: Callable[[FloatArray, FloatArray], tuple[FloatArray, FloatArray]],
    z_out: float,
    newton_steps: int = 30,
) -> SurfaceTrace:
    """Trace the rays of U·exp(iψ) on the plane z = 0 (medium n1) through z = sag(x, y) to z_out.

    Args:
        points: (..., 2) ray start points in mm; amplitude: (..., 3) complex U (transversal);
            phase: ψ in rad; gradient: ∇ψ in 1/mm (transverse wave vector).
        n1, n2: refractive indices before and after the surface; k0 in 1/mm.
        sag, sag_gradient: surface z = h(x, y) (mm) and (∂h/∂x, ∂h/∂y); h > 0 in the aperture.
        z_out: output plane behind the surface (mm).

    Returns:
        SurfaceTrace; the amplitude on the output plane follows with `interface_amplitude`.
    """
    k1 = n1 * k0
    kz = np.sqrt(k1 * k1 - np.sum(gradient * gradient, axis=-1))
    s_in = np.concatenate([gradient, kz[..., None]], axis=-1) / k1
    # intersection: t·s_z = h(ρ + t s⊥), Newton from t₀ = h(ρ)/s_z
    t = sag(points[..., 0], points[..., 1]) / s_in[..., 2]
    for _ in range(newton_steps):
        px = points[..., 0] + t * s_in[..., 0]
        py = points[..., 1] + t * s_in[..., 1]
        hx, hy = sag_gradient(px, py)
        f = t * s_in[..., 2] - sag(px, py)
        t = t - f / (s_in[..., 2] - hx * s_in[..., 0] - hy * s_in[..., 1])
    px = points[..., 0] + t * s_in[..., 0]
    py = points[..., 1] + t * s_in[..., 1]
    hx, hy = sag_gradient(px, py)
    normal = np.stack([-hx, -hy, np.ones_like(hx)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)

    flat_in = s_in.reshape(-1, 3)
    flat_n = normal.reshape(-1, 3)
    flat_u = np.asarray(amplitude).reshape(-1, 3)
    s_out = np.empty_like(flat_in)
    transmitted = np.empty(flat_u.shape, dtype=np.complex128)
    for i in range(flat_in.shape[0]):
        s_out[i], op = fresnel_operator(flat_in[i], flat_n[i], n1, n2)
        transmitted[i] = op @ flat_u[i]
    s_out = s_out.reshape(s_in.shape)
    z_s = sag(px, py)
    length_out = (z_out - z_s) / s_out[..., 2]
    landing = np.stack([px + length_out * s_out[..., 0], py + length_out * s_out[..., 1]], axis=-1)
    new_phase = phase + k0 * (n1 * t + n2 * length_out)
    return SurfaceTrace(
        landing,
        np.asarray(new_phase),
        transmitted.reshape(np.shape(amplitude)),
        s_in,
        s_out,
        np.sum(s_in * normal, axis=-1),
        np.sum(s_out * normal, axis=-1),
    )


def mesh_jacobian(start: FloatArray, landing: FloatArray) -> FloatArray:
    """Jacobian matrices ∂ρ₂/∂ρ₁ (Ny, Nx, 2, 2) of a ray mapping on a regular input grid.

    Second-order differences of the landing points over the neighbouring rays of the mesh
    (central inside, one-sided of second order at the border); dA₂/dA₁ is the determinant
    (paper, Erkenntnis on the mesh areas).

    Args:
        start: (Ny, Nx, 2) regular input grid in mm (rows y, columns x).
        landing: (Ny, Nx, 2) landing points in mm.
    """
    dx = float(start[0, 1, 0] - start[0, 0, 0])
    dy = float(start[1, 0, 1] - start[0, 0, 1])
    jac = np.empty((*landing.shape[:2], 2, 2))
    for c in range(2):
        jac[..., c, 0] = np.gradient(landing[..., c], dx, axis=1, edge_order=2)
        jac[..., c, 1] = np.gradient(landing[..., c], dy, axis=0, edge_order=2)
    return jac


def interface_amplitude(trace: SurfaceTrace, jacobian: FloatArray) -> ComplexArray:
    """Amplitude on the output plane, eq. (interface_amplitude) of the paper.

    U₂ = sqrt(cos θ_a cos γ₁ / (cos θ_e cos γ₂ |J|))·exp(-iπμ/2)·T U₁ with J = det ∂ρ₂/∂ρ₁ and μ
    the number of negative eigenvalues of ∂ρ₂/∂ρ₁ (caustics between surface and plane).
    """
    det = np.linalg.det(jacobian)
    maslov = np.sum(np.linalg.eigvals(jacobian).real < 0.0, axis=-1)
    factor = np.sqrt(
        trace.cos_out
        * trace.direction_in[..., 2]
        / (trace.cos_in * trace.direction_out[..., 2] * np.abs(det))
    ) * np.exp(-0.5j * np.pi * maslov)
    return np.asarray(factor[..., None] * trace.transmitted)
