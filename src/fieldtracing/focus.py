"""Focusing chain PFT → angular spectrum → FFT (paper, section 5.3).

For the exactly spherical exit phase ψ = -k(sqrt(f² + ρ²) - f) the Legendre transform is
ψ̃(k⊥) = k f - f k_z(k⊥) (eq. exact_legendre), so the propagation phase k_z f cancels exactly.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j1

from fieldtracing._types import FloatArray


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
