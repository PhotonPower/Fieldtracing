"""Local figures of merit and selection rules (paper, section 7).

w = |h|^(-1/2), F = |h| L_U², edge-wave ratio w/(sqrt(2π) d_e), SFT gain sqrt(2F_a)/(π q).
h is an eigenvalue of the phase Hessian (1/mm²); lengths in mm. Minimax curvature fits for the SFT.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linprog

from fieldtracing._types import FloatArray


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


def fit_curvature_radial(rho: FloatArray, gradient: FloatArray) -> tuple[float, float]:
    """Minimax curvature c of a rotationally symmetric phase (paper, section 7.4).

    Minimises max_i |ψ'(ρ_i) - c·ρ_i| over the samples (a linear program, solved exactly).

    Args:
        rho: radial sample positions in mm (include the aperture rim).
        gradient: ψ'(ρ) at the samples in 1/mm.

    Returns:
        (c in 1/mm², max|ψ'_res| in 1/mm).
    """
    r = np.asarray(rho, dtype=np.float64)
    g = np.asarray(gradient, dtype=np.float64)
    ones = np.ones_like(r)
    # variables (c, t): minimise t subject to ±(g - c r) ≤ t
    a_ub = np.concatenate([np.stack([-r, -ones], axis=1), np.stack([r, -ones], axis=1)])
    b_ub = np.concatenate([-g, g])
    res = linprog([0.0, 1.0], A_ub=a_ub, b_ub=b_ub, bounds=[(None, None), (0.0, None)])
    if not res.success:
        raise RuntimeError(f"minimax fit failed: {res.message}")
    return float(res.x[0]), float(res.x[1])


def fit_curvature(points: FloatArray, gradient: FloatArray) -> tuple[FloatArray, float]:
    """Minimax curvature matrix C for an SFT on an axis-aligned grid (paper, section 7.4).

    Minimises the largest Cartesian component of the residual gradient ∇ψ - C·ρ over the samples,
    max_i max(|∂_xψ_res|, |∂_yψ_res|), with C symmetric. On an axis-aligned grid this component
    sets the Nyquist step π/max|∂ψ_res|.

    Args:
        points: (N, 2) sample positions in mm (include the aperture rim).
        gradient: (N, 2) ∇ψ at the samples in 1/mm.

    Returns:
        (C (2, 2) in 1/mm², largest residual gradient component in 1/mm).
    """
    p = np.asarray(points, dtype=np.float64)
    g = np.asarray(gradient, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    zeros, ones = np.zeros_like(x), np.ones_like(x)
    # variables (c11, c12, c22, t); residuals r_x = g_x - c11 x - c12 y, r_y = g_y - c12 x - c22 y
    rows_x = np.stack([x, y, zeros], axis=1)
    rows_y = np.stack([zeros, x, y], axis=1)
    a_ub = np.concatenate(
        [
            np.column_stack([-rows_x, -ones]),
            np.column_stack([rows_x, -ones]),
            np.column_stack([-rows_y, -ones]),
            np.column_stack([rows_y, -ones]),
        ]
    )
    b_ub = np.concatenate([-g[:, 0], g[:, 0], -g[:, 1], g[:, 1]])
    bounds = [(None, None), (None, None), (None, None), (0.0, None)]
    res = linprog([0.0, 0.0, 0.0, 1.0], A_ub=a_ub, b_ub=b_ub, bounds=bounds)
    if not res.success:
        raise RuntimeError(f"minimax fit failed: {res.message}")
    c11, c12, c22, t = res.x
    return np.array([[c11, c12], [c12, c22]]), float(t)


def sft_samples(
    window: float, c: float, q: float, max_residual_gradient: float
) -> tuple[float, int]:
    """Grid step (mm) and samples per direction of an SFT: (sft_step, ceil(window/step))."""
    step = sft_step(c, q, max_residual_gradient)
    return step, math.ceil(window / step)
