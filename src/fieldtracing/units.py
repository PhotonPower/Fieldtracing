"""Units and constants, aligned with Raytatouille.

Conventions (see docs/konventionen.md):

- lengths in mm, vacuum wavelengths in µm, angles in rad;
- plane waves ~ exp(i(k·r - ωt)), optical axis +z;
- wavenumbers therefore in 1/mm.
"""

from __future__ import annotations

import math

MM_PER_UM = 1.0e-3


def wavenumber(wavelength_um: float, refractive_index: float = 1.0) -> float:
    """Return the wavenumber k = 2π n / λ in 1/mm.

    Args:
        wavelength_um: vacuum wavelength in µm (> 0).
        refractive_index: real refractive index of the medium (> 0).

    Raises:
        ValueError: if the wavelength or the index is not positive.
    """
    if not wavelength_um > 0.0:
        raise ValueError(f"wavelength_um must be positive, got {wavelength_um}")
    if not refractive_index > 0.0:
        raise ValueError(f"refractive_index must be positive, got {refractive_index}")
    return 2.0 * math.pi * refractive_index / (wavelength_um * MM_PER_UM)
