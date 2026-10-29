"""Semi-analytical Fourier transform (paper, section 6.4, eq. sft_formula).

For E = V·exp(i/2 ρᵀCρ) the exact identity

    Ẽ(k) = (2π)^-d · exp(-i/2 kᵀC⁻¹k) · W(C⁻¹k),   W = V * g_C,  g_C = exp(i/2 ρᵀCρ),

holds. W is a "virtual Fresnel propagation" of V and is computed with two FFTs and the transfer
function (2π)^(d/2)·exp(iπ/4·sig C)/sqrt|det C|·exp(-i/2 κᵀC⁻¹κ). The result lives on the grid
k = C·ρ' and is returned in hybrid form
(residual, phase). Sampling conditions: see docs/testfaelle.md and paper section 6.4.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

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


def region_weights(
    x: FloatArray,
    y: FloatArray,
    inside: Callable[[FloatArray, FloatArray], NDArray[np.bool_]],
    oversample: int = 8,
) -> FloatArray:
    """Covered fraction of each cell of the grid (y, x) by the region `inside(x, y)`.

    Cells are squares of the x step around each sample; the fraction is estimated with
    oversample² sub-samples per cell (memory O(N²)).
    """
    h = float(x[1] - x[0])
    offsets = (np.arange(oversample) + 0.5) / oversample - 0.5
    total = np.zeros((y.size, x.size), dtype=np.float64)
    for dy in offsets:
        for dx in offsets:
            total += inside(x[None, :] + h * dx, y[:, None] + h * dy)
    return np.asarray(total / oversample**2, dtype=np.float64)


def circular_aperture_weights(
    x: FloatArray, y: FloatArray, radius: float, oversample: int = 8
) -> FloatArray:
    """Covered fraction of each cell of the grid (y, x) by the disc of `radius` (mm)."""

    def disc(px: FloatArray, py: FloatArray) -> NDArray[np.bool_]:
        return np.asarray(px * px + py * py <= radius * radius)

    return region_weights(x, y, disc, oversample)


class SamplingWarning(UserWarning):
    """A sampling condition of the SFT (paper, section 6.4) is violated."""


def _symmetric_2x2(matrix: FloatArray, name: str) -> FloatArray:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (2, 2) or not np.allclose(m, m.T):
        raise ValueError(f"{name} must be a symmetric 2x2 matrix")
    if abs(np.linalg.det(m)) == 0.0:
        raise ValueError(f"{name} must be invertible (use a mixed chart for zero curvature)")
    return m


def check_sft_sampling(n: int, step: float, c_matrix: FloatArray, q_min: float = 4.0) -> list[str]:
    """Return the violated SFT sampling conditions (empty list if all hold).

    Per eigenvalue c of C (paper, section 6.4):

    - edge resolution: step ≤ w_c/q_min with w_c = sqrt(2/|c|);
    - transfer chirp: window n·step ≥ 2π/(|c|·step); otherwise the virtual Fresnel
      propagation of the highest grid frequencies wraps around the periodic window.

    Args:
        n: samples per axis; step: grid step in mm; c_matrix: C in 1/mm².
        q_min: minimum number of samples per Fresnel width w_c.
    """
    c = _symmetric_2x2(c_matrix, "c_matrix")
    problems = []
    for value in np.linalg.eigvalsh(c):
        w_c = math.sqrt(2.0 / abs(value))
        if step > w_c / q_min:
            problems.append(
                f"edge resolution: step {step:.3g} mm > w_c/q = {w_c / q_min:.3g} mm "
                f"(c = {value:.4g})"
            )
        needed = 2.0 * math.pi / (abs(value) * step)
        if n * step < needed:
            problems.append(
                f"transfer chirp aliased: window {n * step:.3g} mm < {needed:.3g} mm "
                f"(c = {value:.4g})"
            )
    return problems


def _chirp_convolution(v: ComplexArray, step: float, q: FloatArray) -> ComplexArray:
    """∬ v(u)·exp(i/2 (u' - u)ᵀQ(u' - u)) d²u on the centred grid (Q symmetric, invertible).

    Transfer function 2π·exp(iπ/4·sig Q)/sqrt|det Q|·exp(-i/2 κᵀQ⁻¹κ); axes (y, x).
    """
    ny, nx = np.shape(v)
    kapx = 2.0 * np.pi * np.fft.fftfreq(nx, d=step)
    kapy = 2.0 * np.pi * np.fft.fftfreq(ny, d=step)
    q_inv = np.linalg.inv(q)
    quad_form = (
        q_inv[0, 0] * kapx[None, :] ** 2
        + 2.0 * q_inv[0, 1] * kapx[None, :] * kapy[:, None]
        + q_inv[1, 1] * kapy[:, None] ** 2
    )
    signature = float(np.sum(np.sign(np.linalg.eigvalsh(q))))
    transfer = (
        2.0
        * np.pi
        * np.exp(0.25j * np.pi * signature)
        / np.sqrt(abs(np.linalg.det(q)))
        * np.exp(-0.5j * quad_form)
    )
    return np.asarray(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(v)) * transfer)))


def sft_2d(
    v: ComplexArray, step: float, c_matrix: FloatArray, warn: bool = True
) -> tuple[FloatArray, ComplexArray, FloatArray]:
    """2D SFT of V(ρ)·exp(i/2 ρᵀCρ) for symmetric, invertible C in 1/mm² (eq. sft_formula).

    Args:
        v: (N, N) samples of V on the centred square grid; axes (y, x).
        step: grid step in mm.
        c_matrix: C in 1/mm², component order (x, y).
        warn: emit a SamplingWarning for each violated condition of `check_sft_sampling`.

    Returns:
        (k (N, N, 2) in 1/mm, residual (N, N), phase (N, N)) on the lattice k = C·ρ' with
        Ẽ(k) = residual·exp(i·phase) and phase = -½ ρ'ᵀCρ' = -½ kᵀC⁻¹k. Only the residual may
        be interpolated (ADR 0003); for an arbitrary k use ρ' = C⁻¹k.
    """
    c = _symmetric_2x2(c_matrix, "c_matrix")
    n = np.shape(v)[0]
    if np.shape(v) != (n, n):
        raise ValueError("v must be square")
    if warn:
        for problem in check_sft_sampling(n, step, c):
            warnings.warn(problem, SamplingWarning, stacklevel=2)
    x = centered_grid(n, step)
    rho = np.stack(np.meshgrid(x, x), axis=-1)
    k = np.einsum("ij,...j->...i", c, rho)
    phase = -0.5 * np.einsum("...i,ij,...j->...", rho, c, rho)
    residual = _chirp_convolution(v, step, c) / (2.0 * np.pi) ** 2
    return k, np.asarray(residual), np.asarray(phase)


def inverse_sft_2d(
    residual: ComplexArray,
    step: float,
    b_matrix: FloatArray,
    lattice: FloatArray | None = None,
) -> tuple[FloatArray, ComplexArray, FloatArray]:
    """Inverse SFT: E(ρ) = ∬ R(k)·exp(i/2 kᵀBk)·exp(i k·ρ) d²k, B symmetric, invertible (mm²).

    R is given on the lattice k = L·u, u the centred square grid with `step` (L = identity:
    a regular k grid with step in 1/mm). Completing the square gives

        E(ρ) = exp(-i/2 ρᵀB⁻¹ρ)·|det L|·∬ R(Lu)·exp(i/2 (u' - u)ᵀ LᵀBL (u' - u)) d²u,
        ρ = -B·L·u'.

    The output of `sft_2d` is inverted with B = -C⁻¹ and L = C.

    Returns:
        (ρ (N, N, 2) in mm, residual (N, N), phase (N, N)) with E(ρ) = residual·exp(i·phase).
    """
    b = _symmetric_2x2(b_matrix, "b_matrix")
    lat = np.eye(2) if lattice is None else np.asarray(lattice, dtype=np.float64)
    q = _symmetric_2x2(lat.T @ b @ lat, "L^T B L")
    n = np.shape(residual)[0]
    u = centered_grid(n, step)
    grid = np.stack(np.meshgrid(u, u), axis=-1)
    rho = -np.einsum("ij,...j->...i", b @ lat, grid)
    phase = -0.5 * np.einsum("...i,ij,...j->...", rho, np.linalg.inv(b), rho)
    out = abs(np.linalg.det(lat)) * _chirp_convolution(residual, step, q)
    return rho, np.asarray(out), np.asarray(phase)
