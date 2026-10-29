"""Exact propagation in a homogeneous medium (paper, section 2).

- `angular_spectrum`: plane-wave spectrum, eq. (spw), exact for forward-travelling fields.
- `rayleigh_sommerfeld_kernel`: Rayleigh-Sommerfeld kernel of the first kind, eq. (rs_kernel).
- `van_vleck_kernel_homogeneous`: its semiclassical form, eq. (van_vleck) with S = n r.
"""

from __future__ import annotations

import numpy as np

from fieldtracing._types import ComplexArray, FloatArray
from fieldtracing.fourier import centered_grid, frequency_step, ft, ift


def longitudinal_wavenumber(k_perp_sq: FloatArray, k: float) -> ComplexArray:
    """Return k_z = sqrt(k² - |k_perp|²) with Im k_z ≥ 0 (evanescent waves decay), in 1/mm."""
    return np.asarray(np.sqrt(k * k - k_perp_sq + 0j), dtype=np.complex128)


def angular_spectrum(field: ComplexArray, step: float, k: float, dz: float) -> ComplexArray:
    """Propagate samples of a forward field by dz with the plane-wave spectrum (exact).

    Args:
        field: 1D (N,) or 2D (Ny, Nx) samples on centred grids with equal step.
        step: grid step in mm.
        k: wavenumber n·k0 of the medium in 1/mm.
        dz: propagation distance in mm (may be negative).

    Returns:
        Field at z + dz on the same grid. The computation is periodic; the window must hold
        the propagated field (no zero padding is added).
    """
    shape = np.shape(field)
    if len(shape) not in (1, 2):
        raise ValueError("field must be 1D or 2D")
    spectrum = ft(field, step)
    dk = frequency_step(shape[-1], step)
    if len(shape) == 1:
        kx = centered_grid(shape[0], dk)
        k_perp_sq = kx * kx
    else:
        if shape[0] != shape[1]:
            raise ValueError("2D fields must be square")
        kx = centered_grid(shape[1], dk)
        k_perp_sq = kx[None, :] ** 2 + kx[:, None] ** 2
    kz = longitudinal_wavenumber(k_perp_sq, k)
    return ift(spectrum * np.exp(1j * kz * dz), dk)


def rayleigh_sommerfeld_kernel(dx: FloatArray, dy: FloatArray, d: float, k: float) -> ComplexArray:
    """Rayleigh-Sommerfeld kernel of the first kind K_d(Δ) = d/(2π r²)·(1/r - i k)·exp(i k r).

    Args:
        dx, dy: transverse offsets ρ2 - ρ1 in mm.
        d: plane distance in mm (> 0).
        k: wavenumber n·k0 in 1/mm.
    """
    r = np.sqrt(d * d + dx * dx + dy * dy)
    return np.asarray(d / (2.0 * np.pi * r * r) * (1.0 / r - 1j * k) * np.exp(1j * k * r))


def van_vleck_kernel_homogeneous(
    dx: FloatArray, dy: FloatArray, d: float, k: float
) -> ComplexArray:
    """Semiclassical kernel k d / (2π i r²)·exp(i k r): the leading term of the RS kernel."""
    r = np.sqrt(d * d + dx * dx + dy * dy)
    return np.asarray(k * d / (2j * np.pi * r * r) * np.exp(1j * k * r))
