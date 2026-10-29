"""Pointwise Fourier transform (paper, section 4, eq. pft_final).

Stationary-phase evaluation of the Fourier integral of a hybrid field U·exp(iψ):

    k = ∇ψ(ρ_s),  ψ̃(k) = ψ(ρ_s) - k·ρ_s,
    Ẽ(k) = U(ρ_s)·exp(i ψ̃ + iπ/4·sig H) / ((2π)^(d/2)·sqrt|det H|)   (d = 1 or 2).

The inverse transform E(ρ) = ∫ Ẽ(k) exp(ik·ρ) dᵈk is evaluated the same way with the roles of
ρ and k exchanged: ρ = -∇ψ̃(k_s), ψ = ψ̃ + k·ρ, factor (2π)^(d/2)/sqrt|det H̃|·exp(iπ/4·sig H̃).
For a spectrum produced by the PFT, ∇ψ̃ = -ρ_s and H̃ = -H⁻¹ (Legendre transform).

Valid for large local Fresnel numbers F = |h|·L_U² and away from edges; relative error O(1/F).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from scipy.optimize import brentq

from fieldtracing._types import ComplexArray, FloatArray, IntArray


class AsymptoticWarning(UserWarning):
    """A stationary-phase evaluation is used outside its range of validity (F too small)."""


def _check_fresnel(
    curvatures: FloatArray, length_u: float | None, min_fresnel: float, what: str
) -> None:
    """Warn if the smallest local Fresnel number |h|·L_U² falls below `min_fresnel`."""
    if length_u is None:
        return
    smallest = float(np.min(np.abs(curvatures))) * length_u * length_u
    if smallest < min_fresnel:
        warnings.warn(
            f"{what}: local Fresnel number {smallest:.3g} < {min_fresnel:.3g}; "
            "relative error O(1/F) is large (use the SFT or an FFT)",
            AsymptoticWarning,
            stacklevel=3,
        )


def pft_1d(
    x: FloatArray,
    amplitude: ComplexArray,
    phase: FloatArray,
    gradient: FloatArray,
    curvature: FloatArray,
    length_u: float | None = None,
    min_fresnel: float = 10.0,
) -> tuple[FloatArray, ComplexArray, FloatArray]:
    """1D pointwise Fourier transform.

    Args:
        x: stationary points (mm); amplitude: U there; phase, gradient, curvature: ψ, ψ', ψ''
            (rad, 1/mm, 1/mm²).
        length_u: variation length of U in mm; if given, warn when |ψ''|·L_U² < min_fresnel.

    Returns:
        (k, residual spectrum, spectral phase ψ̃): Ẽ(k) = residual·exp(iψ̃). k in 1/mm.
    """
    if np.any(curvature == 0.0):
        raise ValueError("PFT needs non-zero curvature (collimated samples: use an FFT)")
    _check_fresnel(curvature, length_u, min_fresnel, "pft_1d")
    k = gradient.copy()
    spectral_phase = phase - k * x
    residual = (
        amplitude
        * np.exp(0.25j * np.pi * np.sign(curvature))
        / np.sqrt(2.0 * np.pi * np.abs(curvature))
    )
    return k, np.asarray(residual), spectral_phase


def inverse_pft_1d(
    k: FloatArray,
    residual: ComplexArray,
    spectral_phase: FloatArray,
    spectral_gradient: FloatArray,
    spectral_curvature: FloatArray,
    length_u: float | None = None,
    min_fresnel: float = 10.0,
) -> tuple[FloatArray, ComplexArray, FloatArray]:
    """1D inverse PFT of Ẽ(k) = residual·exp(iψ̃(k)), E(x) = ∫ Ẽ exp(ikx) dk.

    Args:
        k: stationary frequencies (1/mm); spectral_phase, spectral_gradient, spectral_curvature:
            ψ̃, ψ̃' (mm), ψ̃'' (mm²) there.
        length_u: variation length of the residual in 1/mm (optional validity check).

    Returns:
        (x = -ψ̃'(k), residual U, phase ψ = ψ̃ + k·x), E(x) = U·exp(iψ).
    """
    if np.any(spectral_curvature == 0.0):
        raise ValueError("inverse PFT needs non-zero spectral curvature (focus: use an FFT)")
    _check_fresnel(spectral_curvature, length_u, min_fresnel, "inverse_pft_1d")
    x = -spectral_gradient
    phase = spectral_phase + k * x
    u = (
        residual
        * np.sqrt(2.0 * np.pi / np.abs(spectral_curvature))
        * np.exp(0.25j * np.pi * np.sign(spectral_curvature))
    )
    return np.asarray(x), np.asarray(u), np.asarray(phase)


def pft_2d(
    points: FloatArray,
    amplitude: ComplexArray,
    phase: FloatArray,
    gradient: FloatArray,
    hessian: FloatArray,
    length_u: float | None = None,
    min_fresnel: float = 10.0,
) -> tuple[FloatArray, ComplexArray, FloatArray]:
    """2D pointwise Fourier transform of samples (N, 2) with (N, 2, 2) Hessians.

    Args:
        length_u: variation length of U in mm; if given, warn when the smallest
            |eigenvalue|·L_U² is below `min_fresnel`.

    Returns:
        (k (N, 2), residual spectrum, spectral phase ψ̃), Ẽ = residual·exp(iψ̃).
    """
    det = np.linalg.det(hessian)
    if np.any(det == 0.0):
        raise ValueError("PFT needs a regular Hessian (use a mixed chart or an FFT)")
    eigen = np.linalg.eigvalsh(hessian)
    _check_fresnel(eigen, length_u, min_fresnel, "pft_2d")
    signature = np.sum(np.sign(eigen), axis=1)
    k = gradient.copy()
    spectral_phase = phase - np.sum(k * points, axis=1)
    factor = np.exp(0.25j * np.pi * signature) / (2.0 * np.pi * np.sqrt(np.abs(det)))
    residual = amplitude * (factor[:, None] if np.ndim(amplitude) == 2 else factor)
    return k, np.asarray(residual), spectral_phase


def inverse_pft_2d(
    k: FloatArray,
    residual: ComplexArray,
    spectral_phase: FloatArray,
    spectral_gradient: FloatArray,
    spectral_hessian: FloatArray,
    length_u: float | None = None,
    min_fresnel: float = 10.0,
) -> tuple[FloatArray, ComplexArray, FloatArray]:
    """2D inverse PFT of Ẽ(k) = residual·exp(iψ̃(k)), E(ρ) = ∬ Ẽ exp(ik·ρ) d²k.

    Args:
        k: (N, 2) stationary frequencies in 1/mm; spectral_gradient (N, 2) in mm;
            spectral_hessian (N, 2, 2) in mm².

    Returns:
        (ρ = -∇ψ̃ (N, 2), residual U, phase ψ = ψ̃ + k·ρ).
    """
    det = np.linalg.det(spectral_hessian)
    if np.any(det == 0.0):
        raise ValueError("inverse PFT needs a regular spectral Hessian (focus: use an FFT)")
    eigen = np.linalg.eigvalsh(spectral_hessian)
    _check_fresnel(eigen, length_u, min_fresnel, "inverse_pft_2d")
    signature = np.sum(np.sign(eigen), axis=1)
    rho = -spectral_gradient
    phase = spectral_phase + np.sum(k * rho, axis=1)
    factor = 2.0 * np.pi * np.exp(0.25j * np.pi * signature) / np.sqrt(np.abs(det))
    u = residual * (factor[:, None] if np.ndim(residual) == 2 else factor)
    return np.asarray(rho), np.asarray(u), np.asarray(phase)


def _shifted(gradient: Callable[[FloatArray], FloatArray], k: float) -> Callable[[float], float]:
    def f(t: float) -> float:
        return float(gradient(np.array([t]))[0]) - k

    return f


def pft_1d_branches(
    k: FloatArray,
    x_grid: FloatArray,
    amplitude: Callable[[FloatArray], ComplexArray],
    phase: Callable[[FloatArray], FloatArray],
    gradient: Callable[[FloatArray], FloatArray],
    curvature: Callable[[FloatArray], FloatArray],
) -> tuple[ComplexArray, IntArray]:
    """Fourier spectrum at frequencies k as the sum over all stationary points (1D).

    For a non-injective ∇ψ several points x_s satisfy ψ'(x_s) = k; their PFT contributions
    interfere. The stationary points are bracketed by sign changes of ψ'(x) - k on `x_grid`
    and refined with Brent's method.

    Args:
        k: output frequencies in 1/mm.
        x_grid: sorted search grid in mm, fine enough to separate neighbouring roots.
        amplitude, phase, gradient, curvature: U, ψ, ψ', ψ'' as functions of x.

    Returns:
        (Ẽ(k) as complex values, number of stationary points per k). Near a fold
        (ψ'' → 0, merging roots) the result is invalid; that is a caustic in k-space.
    """
    g = gradient(x_grid)
    spectrum = np.zeros(np.shape(k), dtype=np.complex128)
    count = np.zeros(np.shape(k), dtype=np.int64)
    for i, kk in enumerate(np.asarray(k, dtype=np.float64)):
        diff = g - kk
        brackets = np.nonzero(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)[0]
        refined = [
            brentq(_shifted(gradient, float(kk)), float(x_grid[j]), float(x_grid[j + 1]))
            for j in brackets
        ]
        on_grid = [float(x_grid[j]) for j in np.nonzero(diff == 0.0)[0]]
        roots = np.array(sorted(refined + on_grid), dtype=np.float64)
        if roots.size == 0:
            continue
        _, res, spec_phase = pft_1d(
            roots, amplitude(roots), phase(roots), np.full_like(roots, kk), curvature(roots)
        )
        spectrum[i] = np.sum(res * np.exp(1j * spec_phase))
        count[i] = roots.size
    return spectrum, count
