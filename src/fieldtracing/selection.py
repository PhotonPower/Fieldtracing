"""Local figures of merit and selection rules (paper, section 7).

w = |h|^(-1/2), F = |h| L_U², edge-wave ratio w/(sqrt(2π) d_e), SFT gain sqrt(2F_a)/(π q).
h is an eigenvalue of the phase Hessian (1/mm²); lengths in mm.
"""

from __future__ import annotations

import math


def fresnel_width(h: float) -> float:
    """Fresnel-zone width w = |h|^(-1/2) in mm for a phase-Hessian eigenvalue h (1/mm²)."""
    if h == 0.0:
        return math.inf
    return 1.0 / math.sqrt(abs(h))


def local_fresnel_number(h: float, length_u: float) -> float:
    """Local Fresnel number F = |h|·L_U² (L_U: variation length of U in mm)."""
    return abs(h) * length_u * length_u


def edge_wave_ratio(w: float, edge_distance: float) -> float:
    """|Ẽ_edge|/|Ẽ_PFT| ≈ w/(sqrt(2π) d_e), paper eq. (edge_ratio)."""
    return w / (math.sqrt(2.0 * math.pi) * edge_distance)


def min_edge_distance(w: float, tolerance: float) -> float:
    """Smallest edge distance d_e (mm) at which the edge wave stays below `tolerance`."""
    return w / (math.sqrt(2.0 * math.pi) * tolerance)


def sft_gain(fresnel_aperture: float, q: float) -> float:
    """Samples per direction, direct FFT over SFT: sqrt(2 F_a)/(π q), F_a = |h| a² = 2π N_F."""
    return math.sqrt(2.0 * fresnel_aperture) / (math.pi * q)


def sft_step(c: float, q: float, max_residual_gradient: float = 0.0) -> float:
    """SFT grid step min(w_c/q, π/max|∇ψ_res|) in mm, with w_c = sqrt(2/|c|)."""
    step = math.sqrt(2.0 / abs(c)) / q
    if max_residual_gradient > 0.0:
        step = min(step, math.pi / max_residual_gradient)
    return step


def nyquist_samples(window: float, max_gradient: float) -> float:
    """Samples per direction of a direct FFT: window·max|∇ψ|/π."""
    return window * max_gradient / math.pi
