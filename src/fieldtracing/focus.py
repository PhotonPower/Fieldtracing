"""Focusing chain PFT → angular spectrum → FFT (paper, section 5.3).

For the exactly spherical exit phase ψ = -k(sqrt(f² + ρ²) - f) the Legendre transform is
ψ̃(k⊥) = k f - f k_z(k⊥) (eq. exact_legendre), so the propagation phase k_z f cancels exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.special import j1, jv

from fieldtracing._types import ComplexArray, FloatArray
from fieldtracing.fourier import centered_grid, ift
from fieldtracing.sft import circular_aperture_weights


def spherical_phase(rho: FloatArray, f: float, k: float) -> FloatArray:
    """Phase -k(sqrt(f² + ρ²) - f) of a wave converging to a focus at distance f (mm)."""
    return np.asarray(-k * (np.sqrt(f * f + rho * rho) - f))


def stationary_point(k_perp: FloatArray, f: float, k: float) -> FloatArray:
    """Exit-plane point ρ_s = -f k⊥/k_z of the ray with transverse wave vector k⊥ (1D or radial)."""
    return np.asarray(-f * k_perp / np.sqrt(k * k - k_perp * k_perp))


def spherical_legendre(k_perp: FloatArray, f: float, k: float) -> FloatArray:
    """Exact Legendre transform ψ̃(k⊥) = k f - f k_z of the spherical phase (rad)."""
    return np.asarray(k * f - f * np.sqrt(k * k - k_perp * k_perp))


def spherical_pft_amplitude(k_perp: FloatArray, f: float, k: float) -> FloatArray:
    """|det H_ψ|^(-1/2) = f/(k cos²θ) of the spherical phase in 2D (mm²), cos θ = k_z/k."""
    cos_t = np.sqrt(k * k - k_perp * k_perp) / k
    return np.asarray(f / (k * cos_t * cos_t))


def parabolic_residual(k_perp: FloatArray, f: float, k: float) -> FloatArray:
    """Phase error f(k_z - k + k⊥²/(2k)) when a parabolic lens phase meets the exact k_z (rad)."""
    return np.asarray(f * (np.sqrt(k * k - k_perp * k_perp) - k + k_perp * k_perp / (2.0 * k)))


def airy_integral(rho: FloatArray, k_max: float) -> FloatArray:
    """Integral of exp(i k·ρ) over the disc |k| ≤ K: 2π K² J1(Kρ)/(Kρ); at ρ = 0 it is π K²."""
    arg = k_max * np.asarray(rho, dtype=np.float64)
    out = np.empty_like(arg)
    small = np.abs(arg) < 1e-12
    out[small] = np.pi * k_max * k_max
    out[~small] = 2.0 * np.pi * k_max * k_max * j1(arg[~small]) / arg[~small]
    return out


@dataclass(frozen=True)
class FocalField:
    """Field in a plane z = f + defocus behind the exit plane (paper, eq. debye).

    The physical field is values·exp(i·phase_offset); the constant phase k f is kept apart
    (ADR 0003).

    Attributes:
        x: centred coordinates of the output grid in mm (same for x and y).
        values: (M, M) or (M, M, 3) complex field, rows y, columns x.
        phase_offset: constant phase k·f in rad.
    """

    x: FloatArray
    values: ComplexArray
    phase_offset: float


def debye_focus(
    residual: Callable[[FloatArray, FloatArray], ComplexArray],
    f: float,
    k: float,
    sin_max: float,
    n: int = 256,
    pad: int = 4,
    defocus: float = 0.0,
    subpixel: bool = True,
    oversample: int = 8,
) -> FocalField:
    """Focal field of a converging wave by the chain PFT → angular spectrum → FFT (eq. debye).

    The exit-plane field is residual(ρ)·exp(iψ) with the exactly spherical phase
    ψ = -k(sqrt(f² + ρ²) - f) (eq. lens_exit_field without aberration; aberrations belong to
    the residual). Evaluated is

        E(ρ, f + δz) = -(i f e^{ikf}/(2π k)) ∬_{|k⊥| ≤ k sin_max} U(ρ_s(k⊥))/cos²θ
                        · exp(i(k⊥·ρ + k_z δz)) d²k⊥,   ρ_s = -f k⊥/k_z.

    Args:
        residual: U(x, y) on the exit plane (mm in, complex out); returns shape (...,) for a
            scalar or (..., 3) for a vector field.
        f: distance exit plane → focus in mm (> 0).
        k: wavenumber of the image-space medium in 1/mm.
        sin_max: sine of the marginal-ray angle, NA/n (0 < sin_max < 1).
        n: samples across the k-domain diameter (the focal window is n·λ/(2 sin_max)).
        pad: zero-padding factor; the output grid step is 2π/(n·pad·dk).
        defocus: δz in mm, positive away from the lens.
        subpixel: weight boundary cells by their covered fraction (recommended).
        oversample: sub-samples per cell and axis for the boundary weights.

    Returns:
        FocalField on an (n·pad)² grid. Validity: Fresnel number of the aperture ≫ 1
        (paper, section 5.3; Wolf & Li 1981).
    """
    if not 0.0 < sin_max < 1.0:
        raise ValueError("sin_max must lie in (0, 1)")
    if f <= 0.0 or n < 8 or pad < 1:
        raise ValueError("need f > 0, n ≥ 8 and pad ≥ 1")
    k_max = k * sin_max
    dk = 2.0 * k_max / (n - 2)
    kx = centered_grid(n, dk)
    weights = (
        circular_aperture_weights(kx, kx, k_max, oversample)
        if subpixel
        else (kx[None, :] ** 2 + kx[:, None] ** 2 <= k_max * k_max).astype(np.float64)
    )
    k_perp_sq = np.minimum(kx[None, :] ** 2 + kx[:, None] ** 2, (k * (1.0 + sin_max) / 2.0) ** 2)
    kz = np.sqrt(k * k - k_perp_sq)
    cos_sq = (kz / k) ** 2
    u = residual(-f * kx[None, :] / kz, -f * kx[:, None] / kz)
    factor = weights / cos_sq * np.exp(1j * kz * defocus)
    integrand = u * (factor[..., None] if np.ndim(u) == 3 else factor)

    m = n * pad
    padded = np.zeros((m, m) + np.shape(integrand)[2:], dtype=np.complex128)
    lo = m // 2 - n // 2
    padded[lo : lo + n, lo : lo + n] = integrand
    field = ift(padded, dk, axes=(0, 1))
    values = -1j * f / (2.0 * np.pi * k) * field
    return FocalField(centered_grid(m, 2.0 * np.pi / (m * dk)), np.asarray(values), k * f)


def richards_wolf(
    points: FloatArray,
    f: float,
    k: float,
    sin_max: float,
    defocus: float = 0.0,
    nodes: int | None = None,
) -> ComplexArray:
    """Focal field of an aplanatic lens for x-polarised illumination (Richards and Wolf).

    Field on the Gaussian reference sphere E_sphere = sqrt(cos θ)·[cos θ cos²φ + sin²φ,
    (cos θ - 1) sin φ cos φ, -sin θ cos φ] (unit input amplitude, φ the azimuth of k⊥), and

        E = -(i k f/2π) ∬ E_sphere exp(i k (ρ sin θ cos(φ - φ_p) + δz cos θ)) dΩ,

    reduced with ∫ cos(nφ) exp(i x cos(φ - φ_p)) dφ = 2π iⁿ J_n(x) cos(nφ_p) to the integrals
    I₀, I₁, I₂ over θ (Richards and Wolf 1959; paper, section 5.3). Composite Gauss-Legendre in θ.
    Independent of the FFT path; reference for `debye_focus` and `GridlessField.aplanatic`.

    Args:
        points: (N, 2) points in the plane z = f + δz (mm, relative to the axis).
        f: focal length in mm; k: wavenumber of the image space in 1/mm.
        sin_max: NA/n (0 < sin_max < 1); defocus: δz in mm (positive away from the lens).
        nodes: Gauss-Legendre nodes in θ (default from the largest phase k·(ρ + |δz|)).

    Returns:
        (N, 3) complex field, same normalisation as `debye_focus` with the exit-plane residual
        of `GridlessField.aplanatic` (constant phase k f not included).
    """
    p = np.asarray(points, dtype=np.float64)
    rho = np.hypot(p[:, 0], p[:, 1])
    phi = np.arctan2(p[:, 1], p[:, 0])
    t_max = float(np.arcsin(sin_max))
    if nodes is None:
        phase = k * (float(np.max(rho, initial=0.0)) * sin_max + abs(defocus) * (1 - np.cos(t_max)))
        nodes = int(max(200, 8 * phase))
    xg, wg = np.polynomial.legendre.leggauss(nodes)
    t = 0.5 * t_max * (xg + 1.0)
    w = 0.5 * t_max * wg
    c, s = np.cos(t), np.sin(t)
    common = w * np.sqrt(c) * np.exp(1j * k * defocus * c)
    x = k * rho[:, None] * s[None, :]
    i0 = (jv(0, x) * (common * s * (1 + c))[None]).sum(axis=1)
    i1 = (jv(1, x) * (common * s * s)[None]).sum(axis=1)
    i2 = (jv(2, x) * (common * s * (1 - c))[None]).sum(axis=1)
    ex = -0.5j * k * f * (i0 + i2 * np.cos(2 * phi))
    ey = -0.5j * k * f * i2 * np.sin(2 * phi)
    ez = -k * f * i1 * np.cos(phi)
    return np.asarray(np.stack([ex, ey, ez], axis=-1), dtype=np.complex128)
