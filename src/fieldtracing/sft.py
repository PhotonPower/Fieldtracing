"""Semi-analytical Fourier transform (paper, section 6.4, eq. sft_formula).

For E = V·exp(i/2 ρᵀCρ) the exact identity

    Ẽ(k) = (2π)^-d · exp(-i/2 kᵀC⁻¹k) · W(C⁻¹k),   W = V * g_C,  g_C = exp(i/2 ρᵀCρ),

holds. W is a "virtual Fresnel propagation" of V and is computed with two FFTs and the transfer
function (2π)^(d/2)·exp(iπ/4·sig C)/sqrt|det C|·exp(-i/2 κᵀC⁻¹κ). The result lives on the grid
k = C·ρ' and is returned in hybrid form
(residual, phase). Sampling conditions: see docs/testfaelle.md and paper section 6.4.
"""

from __future__ import annotations

import numpy as np

from fieldtracing._types import ComplexArray, FloatArray
from fieldtracing.fourier import centered_grid


def sft_1d(v: ComplexArray, x: FloatArray, c: float) -> tuple[FloatArray, ComplexArray, FloatArray]:
    """1D SFT of V(x)·exp(i c x²/2) sampled on the centred grid x (mm), c in 1/mm².

    Returns:
        (k, residual, spectral phase) on the grid k = c·x, with Ẽ(k) = residual·exp(i·phase),
        convention (2π)^-1 ∫ E exp(-ikx) dx. The result stays hybrid: only the smooth residual
        may be interpolated, never Ẽ itself (its phase -k²/(2c) is not resolved by the grid).
    """
    if c == 0.0:
        raise ValueError("c must be non-zero (use an FFT)")
    n = x.size
    dx = float(x[1] - x[0])
    kappa = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    transfer = (
        np.sqrt(2.0 * np.pi / abs(c))
        * np.exp(0.25j * np.pi * np.sign(c))
        * np.exp(-0.5j * kappa**2 / c)
    )
    w = np.fft.fftshift(np.fft.ifft(np.fft.fft(np.fft.ifftshift(v)) * transfer))
    k = c * x
    return k, np.asarray(w / (2.0 * np.pi)), -0.5 * k**2 / c


def sft_2d_diagonal(
    v: ComplexArray, step: float, cx: float, cy: float
) -> tuple[FloatArray, FloatArray, ComplexArray, FloatArray]:
    """2D SFT for a diagonal curvature matrix C = diag(cx, cy) (1/mm²).

    Args:
        v: (Ny, Nx) samples of V on centred grids with equal step (mm).

    Returns:
        (kx (Nx,), ky (Ny,), residual (Ny, Nx), spectral phase (Ny, Nx)) with kx = cx·x,
        ky = cy·y and Ẽ = residual·exp(i·phase).
    """
    if cx == 0.0 or cy == 0.0:
        raise ValueError("curvatures must be non-zero (use a mixed chart)")
    ny, nx = np.shape(v)
    x = centered_grid(nx, step)
    y = centered_grid(ny, step)
    kapx = 2.0 * np.pi * np.fft.fftfreq(nx, d=step)
    kapy = 2.0 * np.pi * np.fft.fftfreq(ny, d=step)
    signature = np.sign(cx) + np.sign(cy)
    transfer = (
        2.0
        * np.pi
        * np.exp(0.25j * np.pi * signature)
        / np.sqrt(abs(cx * cy))
        * np.exp(-0.5j * (kapx[None, :] ** 2 / cx + kapy[:, None] ** 2 / cy))
    )
    w = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(v)) * transfer))
    kx, ky = cx * x, cy * y
    phase = -0.5 * (kx[None, :] ** 2 / cx + ky[:, None] ** 2 / cy)
    return kx, ky, np.asarray(w / (2.0 * np.pi) ** 2), phase


def aperture_weights_1d(x: FloatArray, half_width: float, subpixel: bool = True) -> FloatArray:
    """Weights of a hard slit |x| ≤ half_width on the centred grid x (mm).

    With `subpixel` each sample gets the covered fraction of its cell (area weighting); this
    shrinks the first-order discretisation error at the edge (paper, SFT table).
    """
    if not subpixel:
        return (np.abs(x) <= half_width).astype(np.float64)
    h = float(x[1] - x[0])
    return np.clip((half_width - np.abs(x)) / h + 0.5, 0.0, 1.0)


def circular_aperture_weights(
    x: FloatArray, y: FloatArray, radius: float, oversample: int = 8
) -> FloatArray:
    """Covered fraction of each cell of the grid (y, x) by the disc of `radius` (mm).

    Cells are squares of the grid step; the fraction is estimated with oversample² sub-samples.
    """
    h = float(x[1] - x[0])
    offsets = (np.arange(oversample) + 0.5) / oversample - 0.5
    sub_x = x[None, :, None, None] + h * offsets[None, None, None, :]
    sub_y = y[:, None, None, None] + h * offsets[None, None, :, None]
    inside = (sub_x**2 + sub_y**2) <= radius * radius
    return np.asarray(inside.mean(axis=(2, 3)), dtype=np.float64)
