"""Edge diffraction of a truncated hybrid field (paper, section 6.1).

- `edge_wave_1d`: boundary term of the integration by parts, eq. (keller_gtd).
- `uniform_edge_factor`: uniform Fresnel solution, eq. (fresnel_edge), Ẽ = Ẽ_PFT·factor.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc

from fieldtracing._types import ComplexArray, FloatArray


def uniform_edge_factor(x_s: FloatArray, x_e: float, curvature: FloatArray) -> ComplexArray:
    """½·erfc(exp(-iπ/4·sgn ψ'')·sqrt(|ψ''|/2)·(x_s - x_e)) for a field truncated at x ≤ x_e.

    Args:
        x_s: stationary points (mm).
        x_e: edge position (mm); the field is kept for x ≤ x_e.
        curvature: ψ''(x_s) in 1/mm².
    """
    arg = (
        np.exp(-0.25j * np.pi * np.sign(curvature)) * np.sqrt(np.abs(curvature) / 2.0) * (x_s - x_e)
    )
    return np.asarray(0.5 * erfc(arg))


def edge_wave_1d(
    amplitude_e: complex, phase_e: float, gradient_e: float, x_e: float, k: FloatArray
) -> ComplexArray:
    """Edge wave 1/(2πi)·U(x_e)/(ψ'(x_e) - k)·exp(i(ψ(x_e) - k x_e)) of a field cut at x_e."""
    return np.asarray(
        amplitude_e / (2j * np.pi * (gradient_e - k)) * np.exp(1j * (phase_e - k * x_e))
    )
