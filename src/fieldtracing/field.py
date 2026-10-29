"""Data containers of the hybrid field representation U·exp(iψ) (paper, section 1.3, 7.2).

Placeholders for milestone F2: only shape validation, no behaviour yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fieldtracing._types import ComplexArray, FloatArray


@dataclass(frozen=True)
class GridlessField:
    """Hybrid field sampled on a point cloud in a plane z = const.

    Attributes:
        points: (N, 2) transverse positions in mm.
        phase: (N,) ψ in rad.
        gradient: (N, 2) ∇ψ = k⊥ in 1/mm.
        hessian: (N, 2, 2) Hessian of ψ in 1/mm².
        amplitude: (N, 3) complex residual field U (full vector, U_z from transversality).
        wavelength_um: vacuum wavelength in µm.
        refractive_index: index of the medium.
    """

    points: FloatArray
    phase: FloatArray
    gradient: FloatArray
    hessian: FloatArray
    amplitude: ComplexArray
    wavelength_um: float
    refractive_index: float = 1.0

    def __post_init__(self) -> None:
        n = np.shape(self.points)[0]
        expected = {
            "points": (n, 2),
            "phase": (n,),
            "gradient": (n, 2),
            "hessian": (n, 2, 2),
            "amplitude": (n, 3),
        }
        for name, shape in expected.items():
            if np.shape(getattr(self, name)) != shape:
                raise ValueError(
                    f"{name} must have shape {shape}, got {np.shape(getattr(self, name))}"
                )


@dataclass(frozen=True)
class GriddedField:
    """Field on a square centred grid, optionally with an analytic quadratic phase.

    The physical field is values·exp(i/2·(cx x² + cy y²)); cx = cy = 0 is a plain grid.

    Attributes:
        step: grid step in mm (or 1/mm in the k domain).
        values: (Ny, Nx, 3) complex samples.
        cx, cy: analytic curvatures in 1/mm² (k domain: mm²).
        wavelength_um: vacuum wavelength in µm.
        refractive_index: index of the medium.
    """

    step: float
    values: ComplexArray
    wavelength_um: float
    refractive_index: float = 1.0
    cx: float = 0.0
    cy: float = 0.0

    def __post_init__(self) -> None:
        if np.ndim(self.values) != 3 or np.shape(self.values)[2] != 3:
            raise ValueError("values must have shape (Ny, Nx, 3)")
