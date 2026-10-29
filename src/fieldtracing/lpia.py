"""Local plane interface approximation (paper, section 3.2).

Fresnel coefficients in Convention A (Byrnes, arXiv:1603.02720v5, eq. (6)), as in Raytatouille:

    r_s = (n1 cos i - n2 cos t)/(n1 cos i + n2 cos t),
    r_p = (n2 cos i - n1 cos t)/(n2 cos i + n1 cos t),
    t_s = 2 n1 cos i/(n1 cos i + n2 cos t),
    t_p = 2 n1 cos i/(n2 cos i + n1 cos t),

basis s = k̂_in × N/|k̂_in × N|, p = k̂ × s for every wave, N pointing into medium 2.
"""

from __future__ import annotations

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
