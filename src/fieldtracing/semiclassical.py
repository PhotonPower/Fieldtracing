"""Geometric field propagation in a homogeneous medium (paper, section 3.4).

Each sample of a hybrid field U·exp(iψ) is a ray with transverse wave vector ∇ψ. After a
distance d it lands at ρ2 = ρ1 + d·∇ψ/k_z and carries

    U2 = U1 / sqrt|J| · exp(-iπμ/2),   ψ2 = ψ1 + k·r,   r = d·k/k_z,

with the Jacobian J = det ∂ρ2/∂ρ1 (eq. jacobian_amplitude) and the Maslov index μ = number of
negative eigenvalues of ∂ρ2/∂ρ1 (caustics crossed). The law is the leading order in 1/k; it fails
near caustics (J → 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fieldtracing._types import ComplexArray, FloatArray, IntArray


@dataclass(frozen=True)
class GeometricResult:
    """Result of a geometric propagation; all arrays are per sample.

    Attributes:
        points: landing points ρ2 in mm, shape (N,) in 1D or (N, 2) in 2D.
        amplitude: U2 (complex, same trailing shape as the input amplitude).
        phase: ψ2 in rad.
        jacobian: J = det ∂ρ2/∂ρ1 (dimensionless, signed).
        maslov: μ, number of caustics crossed (0, 1 or 2).
        curvature: new ψ'' (1D) or Hessian (N, 2, 2) of ψ₂ at the landing points; the transverse
            wave vector is conserved along a ray, so H₂ = H₁·(∂ρ₂/∂ρ₁)⁻¹.
    """

    points: FloatArray
    amplitude: ComplexArray
    phase: FloatArray
    jacobian: FloatArray
    maslov: IntArray
    curvature: FloatArray | None = None


def _kz(p_sq: FloatArray, k: float) -> FloatArray:
    if np.any(p_sq >= k * k):
        raise ValueError("phase gradient reaches or exceeds k (evanescent sample)")
    return np.asarray(np.sqrt(k * k - p_sq), dtype=np.float64)


def propagate_1d(
    x: FloatArray,
    amplitude: ComplexArray,
    phase: FloatArray,
    gradient: FloatArray,
    curvature: FloatArray,
    k: float,
    d: float,
) -> GeometricResult:
    """Propagate a 1D hybrid field geometrically by d (mm) in a medium with wavenumber k (1/mm).

    Args:
        x: sample positions in mm.
        amplitude: U at the samples.
        phase, gradient, curvature: ψ (rad), ψ' (1/mm), ψ'' (1/mm²) at the samples.
    """
    kz = _kz(gradient * gradient, k)
    points = x + d * gradient / kz
    jacobian = 1.0 + d * curvature * k * k / kz**3
    maslov = (jacobian < 0.0).astype(np.int64)
    new_phase = phase + k * k * d / kz
    new_amp = amplitude / np.sqrt(np.abs(jacobian)) * np.exp(-0.5j * np.pi * maslov)
    return GeometricResult(
        points, np.asarray(new_amp), new_phase, jacobian, maslov, np.asarray(curvature / jacobian)
    )


def propagate_2d(
    points: FloatArray,
    amplitude: ComplexArray,
    phase: FloatArray,
    gradient: FloatArray,
    hessian: FloatArray,
    k: float,
    d: float,
) -> GeometricResult:
    """Propagate a 2D hybrid field geometrically by d (mm).

    Args:
        points: (N, 2) sample positions in mm.
        amplitude: (N,) or (N, 3) complex U.
        phase: (N,) ψ in rad.
        gradient: (N, 2) ∇ψ in 1/mm.
        hessian: (N, 2, 2) Hessian of ψ in 1/mm².
    """
    p_sq = np.sum(gradient * gradient, axis=1)
    kz = _kz(p_sq, k)
    landing = points + d * gradient / kz[:, None]
    # ∂(p/k_z)/∂p = I/k_z + p pᵀ/k_z³
    dir_jac = (
        np.eye(2)[None, :, :] / kz[:, None, None]
        + np.einsum("ni,nj->nij", gradient, gradient) / (kz**3)[:, None, None]
    )
    mapping = np.eye(2)[None, :, :] + d * np.einsum("nij,njk->nik", hessian, dir_jac)
    jacobian = np.linalg.det(mapping)
    eigen = np.linalg.eigvals(mapping).real
    maslov = np.sum(eigen < 0.0, axis=1).astype(np.int64)
    factor = np.exp(-0.5j * np.pi * maslov) / np.sqrt(np.abs(jacobian))
    new_amp = amplitude * (factor[:, None] if np.ndim(amplitude) == 2 else factor)
    new_phase = phase + k * k * d / kz
    new_hessian = np.einsum("nij,njk->nik", hessian, np.linalg.inv(mapping))
    return GeometricResult(landing, np.asarray(new_amp), new_phase, jacobian, maslov, new_hessian)
