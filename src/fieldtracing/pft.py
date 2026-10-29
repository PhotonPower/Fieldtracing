"""Pointwise Fourier transform (paper, section 4, eq. pft_final).

Stationary-phase evaluation of the Fourier integral of a hybrid field U·exp(iψ):

    k = ∇ψ(ρ_s),  ψ̃(k) = ψ(ρ_s) - k·ρ_s,
    Ẽ(k) = U(ρ_s)·exp(i ψ̃ + iπ/4·sig H) / ((2π)^(d/2)·sqrt|det H|)   (d = 1 or 2).

Valid for large local Fresnel numbers and away from edges; relative error O(1/F).
"""

from __future__ import annotations

import numpy as np

from fieldtracing._types import ComplexArray, FloatArray


def pft_1d(
    x: FloatArray,
    amplitude: ComplexArray,
    phase: FloatArray,
    gradient: FloatArray,
    curvature: FloatArray,
) -> tuple[FloatArray, ComplexArray, FloatArray]:
    """1D pointwise Fourier transform.

    Returns:
        (k, residual spectrum, spectral phase ψ̃): Ẽ(k) = residual·exp(iψ̃). k in 1/mm.
    """
    if np.any(curvature == 0.0):
        raise ValueError("PFT needs non-zero curvature (collimated samples: use an FFT)")
    k = gradient.copy()
    spectral_phase = phase - k * x
    residual = (
        amplitude
        * np.exp(0.25j * np.pi * np.sign(curvature))
        / np.sqrt(2.0 * np.pi * np.abs(curvature))
    )
    return k, np.asarray(residual), spectral_phase


def pft_2d(
    points: FloatArray,
    amplitude: ComplexArray,
    phase: FloatArray,
    gradient: FloatArray,
    hessian: FloatArray,
) -> tuple[FloatArray, ComplexArray, FloatArray]:
    """2D pointwise Fourier transform of samples (N, 2) with (N, 2, 2) Hessians.

    Returns:
        (k (N, 2), residual spectrum, spectral phase ψ̃), Ẽ = residual·exp(iψ̃).
    """
    det = np.linalg.det(hessian)
    if np.any(det == 0.0):
        raise ValueError("PFT needs a regular Hessian (use a mixed chart or an FFT)")
    eigen = np.linalg.eigvalsh(hessian)
    signature = np.sum(np.sign(eigen), axis=1)
    k = gradient.copy()
    spectral_phase = phase - np.sum(k * points, axis=1)
    factor = np.exp(0.25j * np.pi * signature) / (2.0 * np.pi * np.sqrt(np.abs(det)))
    residual = amplitude * (factor[:, None] if np.ndim(amplitude) == 2 else factor)
    return k, np.asarray(residual), spectral_phase
