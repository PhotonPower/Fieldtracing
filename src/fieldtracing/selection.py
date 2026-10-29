"""Local figures of merit and selection rules (paper, section 7).

w = |h|^(-1/2), F = |h| L_U², edge-wave ratio w/(sqrt(2π) d_e), SFT gain sqrt(2F_a)/(π q).
h is an eigenvalue of the phase Hessian (1/mm²); lengths in mm. Minimax curvature fits for the SFT.
Selection rules of table tab:selection (`select_method`) and error terms of table tab:errors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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


# ----- selection rules of table tab:selection and error terms of table tab:errors ------------

DEBYE_ERROR_CONSTANT = 1.5
"""Debye error ≈ 1.5/N_F (test T4, on the axis at NA 0.1; paper, section 5.3)."""

SFT_EDGE_CONSTANT = 0.4
"""Hard edge on a grid (subpixel weights, 2D): error ≲ 0.4/q with q = w_c/Δx.

Measured constants: rotated rectangle (T6, #2) 0.20, 0.23, 0.26 at q = 5.4, 10.8, 21.6; circular
stop away from the axis (#20) 0.20, 0.35, 0.16 at q = 10, 15, 20 (irregular, grid effects of the
circle). 0.4 is the upper bound with margin. Near the axis of a circular stop the error is
amplified (caustic of the edge wave); see `chain.plan_propagation`.
"""

METHODS = ("lpia", "rigorous", "geometric", "sft", "fft", "mixed", "debye", "rayleigh_sommerfeld")


def debye_error(fresnel_number: float) -> float:
    """Relative error of the Debye integral ≈ 1.5/N_F, N_F = a²/(λf) (table tab:errors)."""
    return math.inf if fresnel_number <= 0.0 else DEBYE_ERROR_CONSTANT / fresnel_number


def grid_edge_error(q: float) -> float:
    """Relative error of a subpixel-weighted hard edge on a grid with q = w/Δx samples."""
    return math.inf if q <= 0.0 else SFT_EDGE_CONSTANT / q


def q_for_tolerance(tolerance: float, q_min: float = 4.0) -> float:
    """Samples per Fresnel width that keep `grid_edge_error` below `tolerance`."""
    return max(q_min, float(math.ceil(SFT_EDGE_CONSTANT / tolerance)))


@dataclass(frozen=True)
class Situation:
    """Figures of merit of one step of a chain (paper, section 7.1).

    Attributes:
        fresnel_min, fresnel_max: smallest and largest local Fresnel number F_j = |h_j|·L_U² over
            the illuminated samples (eigenvalues h_j of the phase Hessian).
        edge_ratio: largest edge-wave ratio w/(sqrt(2π) d_e) of the illuminated samples
            (eq. edge_ratio); 0 without a stop.
        target_fresnel: local Fresnel number of the field in the target plane, |h₂|·L_U²·|J|;
            small values mean the target lies near a caustic (focus).
        fresnel_number: Fresnel number N_F = F_max/(2π) of the aperture for focusing.
        fft_samples, sft_samples: samples per direction of a direct FFT and of the SFT (cost).
        interface: True for a surface (LPIA row of the table).
        curvature_radius_mm: smallest curvature radius of surface and wavefront (interface).
        wavelength_mm: wavelength in the medium (interface).
    """

    fresnel_min: float = math.inf
    fresnel_max: float = math.inf
    edge_ratio: float = 0.0
    target_fresnel: float = math.inf
    fresnel_number: float = math.inf
    fft_samples: float = math.inf
    sft_samples: float = math.inf
    interface: bool = False
    curvature_radius_mm: float = math.inf
    wavelength_mm: float = 0.0


def select_method(s: Situation, tolerance: float) -> tuple[str, str]:
    """Building block for one step by the selection rules of table tab:selection.

    Rules, in this order (ε = tolerance):

    1. Interface: LPIA if λ/R_c ≤ ε, else a locally rigorous solver ("rigorous").
    2. Weakly curved, F_max ≤ 1: direct FFT.
    3. Mixed, F_min ≤ 1 and F_max ≥ 1/ε: mixed chart.
    4. Target near a caustic, 1/F_target > ε: Debye integral if 1.5/N_F ≤ ε, else
       Rayleigh-Sommerfeld.
    5. Strongly curved and away from edges, 1/F_min ≤ ε and w/(sqrt(2π) d_e) ≤ ε: geometric
       propagation (PFT).
    6. Otherwise SFT, unless the direct FFT needs fewer samples (cost, eq. sft_gain).

    Returns:
        (method from METHODS, reason with the deciding figures).
    """
    eps = tolerance
    if s.interface:
        ratio = s.wavelength_mm / s.curvature_radius_mm
        if ratio <= eps:
            return "lpia", f"interface: λ/R_c = {ratio:.3g} ≤ ε"
        return "rigorous", f"interface: λ/R_c = {ratio:.3g} > ε, LPIA not valid"
    if s.fresnel_max <= 1.0:
        return "fft", f"weakly curved: F_max = {s.fresnel_max:.3g} ≤ 1"
    if s.fresnel_min <= 1.0:
        if s.fresnel_max >= 1.0 / eps:
            return "mixed", f"mixed: F_min = {s.fresnel_min:.3g} ≤ 1, F_max = {s.fresnel_max:.3g}"
        return "fft", f"weakly curved overall: F_min = {s.fresnel_min:.3g} ≤ 1"
    if s.target_fresnel * eps < 1.0:
        err = debye_error(s.fresnel_number)
        if err <= eps:
            return "debye", (
                f"target near a caustic (F_target = {s.target_fresnel:.3g}); "
                f"Debye error 1.5/N_F = {err:.3g} ≤ ε"
            )
        return "rayleigh_sommerfeld", (
            f"target near a caustic (F_target = {s.target_fresnel:.3g}); "
            f"Debye error 1.5/N_F = {err:.3g} > ε"
        )
    if 1.0 / s.fresnel_min <= eps and s.edge_ratio <= eps:
        return "geometric", (
            f"strongly curved, away from edges: 1/F_min = {1.0 / s.fresnel_min:.3g}, "
            f"edge ratio {s.edge_ratio:.3g} ≤ ε"
        )
    if s.fft_samples <= s.sft_samples:
        return "fft", (
            f"direct FFT cheaper: {s.fft_samples:.3g} ≤ {s.sft_samples:.3g} samples per direction"
        )
    return "sft", (
        f"strongly curved near edges or F_min = {s.fresnel_min:.3g} < 1/ε "
        f"(edge ratio {s.edge_ratio:.3g}); SFT {s.sft_samples:.3g} vs FFT "
        f"{s.fft_samples:.3g} samples per direction"
    )
