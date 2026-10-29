"""Continuous Fourier transform on centred grids (paper, eq. ft_convention).

Convention: F{f}(k) = (2π)^-d ∫ f(x) exp(-i k·x) d^d x,  f(x) = ∫ F(k) exp(i k·x) d^d k.

Grids are centred: x_n = (n - N//2)·dx, k_m = (m - N//2)·dk with dk = 2π/(N dx). With this
convention the convolution theorem reads F{f*g} = (2π)^d F{f}·F{g}.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from fieldtracing._types import ComplexArray, FloatArray


def centered_grid(n: int, step: float) -> FloatArray:
    """Return the centred grid (i - n//2)·step for i = 0..n-1 (unit of `step`)."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return (np.arange(n, dtype=np.float64) - n // 2) * step


def frequency_step(n: int, step: float) -> float:
    """Return the spatial-frequency step 2π/(n·step) of the grid conjugate to `centered_grid`."""
    return 2.0 * math.pi / (n * step)


def _axes(ndim: int, axes: Sequence[int] | None) -> tuple[int, ...]:
    return tuple(range(ndim)) if axes is None else tuple(axes)


def ft(
    f: ComplexArray | FloatArray, step: float, axes: Sequence[int] | None = None
) -> ComplexArray:
    """Approximate the continuous Fourier transform of samples on a centred grid.

    Args:
        f: samples f(x_n) on centred grids along `axes` (same `step` on every axis).
        step: grid step dx in mm.
        axes: transformed axes (default: all).

    Returns:
        F(k_m) on the conjugate centred grids, convention (2π)^-d ∫ f exp(-ikx) dx.
    """
    ax = _axes(np.ndim(f), axes)
    scale = (step / (2.0 * math.pi)) ** len(ax)
    spectrum = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(f, axes=ax), axes=ax), axes=ax)
    return np.asarray(scale * spectrum, dtype=np.complex128)


def ift(spectrum: ComplexArray, k_step: float, axes: Sequence[int] | None = None) -> ComplexArray:
    """Inverse of `ft`: f(x_n) = ∫ F(k) exp(i k x) dk, sampled on the conjugate centred grid.

    Args:
        spectrum: F(k_m) on centred grids along `axes`.
        k_step: frequency step dk in 1/mm.
        axes: transformed axes (default: all).
    """
    ax = _axes(np.ndim(spectrum), axes)
    count = math.prod(np.shape(spectrum)[a] for a in ax)
    field = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(spectrum, axes=ax), axes=ax), axes=ax)
    return np.asarray(count * k_step ** len(ax) * field, dtype=np.complex128)
