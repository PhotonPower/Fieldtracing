"""Hybrid field representation U·exp(iψ) (paper, sections 1.3 and 7.2; ADR 0003).

`GridlessField` holds samples on a point cloud, either in the transverse plane (domain "space":
points ρ in mm, ∇ψ = k⊥ in 1/mm) or in the plane-wave spectrum (domain "frequency": points k⊥ in
1/mm, ∇ψ̃ = -ρ in mm). `GriddedField` holds samples on a square grid with an analytic quadratic
carrier phase. Only the smooth residual U is ever interpolated; the phase is evaluated from its
local Taylor expansion or analytically.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
from scipy.spatial import cKDTree

from fieldtracing._types import ComplexArray, FloatArray
from fieldtracing.fourier import centered_grid
from fieldtracing.pft import inverse_pft_2d, pft_2d
from fieldtracing.semiclassical import propagate_2d
from fieldtracing.units import wavenumber

Domain = Literal["space", "frequency"]


@dataclass(frozen=True)
class GridlessField:
    """Hybrid field sampled on a point cloud.

    Attributes:
        points: (N, 2) sample positions (mm, or 1/mm in the frequency domain).
        phase: (N,) ψ in rad.
        gradient: (N, 2) ∇ψ (1/mm in space, mm in frequency).
        hessian: (N, 2, 2) Hessian of ψ (1/mm² in space, mm² in frequency).
        amplitude: (N, 3) complex residual U (full field vector, transversal to the local k).
        wavelength_um: vacuum wavelength in µm.
        refractive_index: index of the medium.
        domain: "space" or "frequency".
    """

    points: FloatArray
    phase: FloatArray
    gradient: FloatArray
    hessian: FloatArray
    amplitude: ComplexArray
    wavelength_um: float
    refractive_index: float = 1.0
    domain: Domain = "space"

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
                got = np.shape(getattr(self, name))
                raise ValueError(f"{name} must have shape {shape}, got {got}")

    @property
    def k(self) -> float:
        """Wavenumber n·k₀ of the medium in 1/mm."""
        return wavenumber(self.wavelength_um, self.refractive_index)

    def local_wave_vector(self) -> FloatArray:
        """(N, 3) local wave vectors (k⊥, k_z) in 1/mm."""
        k_perp = self.gradient if self.domain == "space" else self.points
        kz = np.sqrt(self.k**2 - np.sum(k_perp * k_perp, axis=1))
        return np.asarray(np.concatenate([k_perp, kz[:, None]], axis=1))

    def transversality_error(self) -> float:
        """max |k̂·U| / max |U|: zero for a transversal field."""
        k_hat = self.local_wave_vector() / self.k
        dot = np.abs(np.sum(k_hat * self.amplitude, axis=1))
        scale = float(np.max(np.linalg.norm(self.amplitude, axis=1)))
        return float(np.max(dot)) / scale if scale > 0.0 else 0.0

    def validate(self, tolerance: float = 1e-6) -> GridlessField:
        """Return self after checking transversality; raise ValueError otherwise."""
        error = self.transversality_error()
        if error > tolerance:
            raise ValueError(f"amplitude not transversal to the local wave vector ({error:.3g})")
        return self

    # ----- construction ---------------------------------------------------------------------

    @classmethod
    def from_phase(
        cls,
        points: FloatArray,
        amplitude: ComplexArray,
        phase: Callable[[FloatArray], FloatArray],
        gradient: Callable[[FloatArray], FloatArray],
        hessian: Callable[[FloatArray], FloatArray],
        wavelength_um: float,
        refractive_index: float = 1.0,
    ) -> GridlessField:
        """Sample an analytic phase (functions of (N, 2) points) at `points` (space domain)."""
        p = np.asarray(points, dtype=np.float64)
        return cls(
            p,
            np.asarray(phase(p)),
            np.asarray(gradient(p)),
            np.asarray(hessian(p)),
            np.asarray(amplitude, dtype=np.complex128),
            wavelength_um,
            refractive_index,
        )

    @classmethod
    def spherical(
        cls,
        points: FloatArray,
        amplitude: ComplexArray,
        f: float,
        wavelength_um: float,
        refractive_index: float = 1.0,
    ) -> GridlessField:
        """Wave converging to (0, 0, f) for f > 0 (diverging from (0, 0, f) for f < 0).

        ψ = -sgn(f)·k(sqrt(f² + ρ²) - |f|), paper eq. (lens_exit_field) without aberration.
        """
        k = wavenumber(wavelength_um, refractive_index)
        s = -float(np.sign(f))
        a = abs(f)

        def radius(p: FloatArray) -> FloatArray:
            return np.asarray(np.sqrt(a * a + np.sum(p * p, axis=1)))

        def hessian(p: FloatArray) -> FloatArray:
            r = radius(p)
            outer = np.einsum("ni,nj->nij", p, p)
            return np.asarray(
                s * k * (np.eye(2)[None] / r[:, None, None] - outer / (r**3)[:, None, None])
            )

        return cls.from_phase(
            points,
            amplitude,
            lambda p: s * k * (radius(p) - a),
            lambda p: s * k * p / radius(p)[:, None],
            hessian,
            wavelength_um,
            refractive_index,
        )

    @classmethod
    def quadratic(
        cls,
        points: FloatArray,
        amplitude: ComplexArray,
        c_matrix: FloatArray,
        wavelength_um: float,
        refractive_index: float = 1.0,
    ) -> GridlessField:
        """Quadratic phase ψ = ½ρᵀCρ with symmetric C in 1/mm²."""
        c = np.asarray(c_matrix, dtype=np.float64)
        return cls.from_phase(
            points,
            amplitude,
            lambda p: 0.5 * np.einsum("ni,ij,nj->n", p, c, p),
            lambda p: p @ c,
            lambda p: np.broadcast_to(c, (len(p), 2, 2)).copy(),
            wavelength_um,
            refractive_index,
        )

    # ----- evaluation -------------------------------------------------------------------------

    def phase_at(self, query: FloatArray) -> FloatArray:
        """ψ at arbitrary points from the second-order Taylor expansion about the nearest sample.

        The error is third order in the distance to the nearest sample.
        """
        q = np.asarray(query, dtype=np.float64)
        _, idx = cKDTree(self.points).query(q)
        d = q - self.points[idx]
        return np.asarray(
            self.phase[idx]
            + np.sum(self.gradient[idx] * d, axis=1)
            + 0.5 * np.einsum("ni,nij,nj->n", d, self.hessian[idx], d)
        )

    def residual_at(
        self, query: FloatArray, method: Literal["linear", "cubic"] = "linear"
    ) -> ComplexArray:
        """Residual U at arbitrary points (scattered interpolation; zero outside the hull)."""
        q = np.asarray(query, dtype=np.float64)
        interp_cls = LinearNDInterpolator if method == "linear" else CloughTocher2DInterpolator
        out = np.empty((len(q), 3), dtype=np.complex128)
        for c in range(3):
            interp = interp_cls(self.points, self.amplitude[:, c], fill_value=0.0)
            out[:, c] = interp(q)
        return out

    def to_grid(
        self,
        n: int,
        step: float,
        curvature: FloatArray | None = None,
        method: Literal["linear", "cubic"] = "linear",
    ) -> GriddedField:
        """Resample on an n×n centred grid; the carrier ½ρᵀCρ stays analytic.

        The stored values are U·exp(i(ψ - ½ρᵀCρ)). Raises ValueError if the remaining phase is
        not resolved by the grid (Nyquist: |∇(ψ - ½ρᵀCρ)|·step ≤ π at the samples).
        """
        if self.domain != "space":
            raise ValueError("to_grid needs a space-domain field")
        c = np.zeros((2, 2)) if curvature is None else np.asarray(curvature, dtype=np.float64)
        residual_gradient = self.gradient - self.points @ c
        worst = float(np.max(np.abs(residual_gradient))) * step
        if worst > np.pi:
            raise ValueError(
                f"phase not resolved: max|∂(ψ - ½ρᵀCρ)|·step = {worst:.3g} > π; "
                "use a smaller step or a curvature carrier (SFT)"
            )
        x = centered_grid(n, step)
        grid = np.stack(np.meshgrid(x, x), axis=-1).reshape(-1, 2)
        carrier = 0.5 * np.einsum("ni,ij,nj->n", grid, c, grid)
        values = (
            self.residual_at(grid, method) * np.exp(1j * (self.phase_at(grid) - carrier))[:, None]
        )
        return GriddedField(
            step, values.reshape(n, n, 3), self.wavelength_um, self.refractive_index, c
        )

    # ----- thin adapters to the F1 building blocks --------------------------------------------

    def propagate(self, distance: float) -> GridlessField:
        """Geometric propagation by `distance` mm (paper eq. jacobian_amplitude)."""
        if self.domain != "space":
            raise ValueError("propagate needs a space-domain field")
        res = propagate_2d(
            self.points, self.amplitude, self.phase, self.gradient, self.hessian, self.k, distance
        )
        assert res.curvature is not None
        return GridlessField(
            res.points,
            res.phase,
            self.gradient.copy(),
            res.curvature,
            res.amplitude,
            self.wavelength_um,
            self.refractive_index,
        )

    def pft(self) -> GridlessField:
        """Pointwise Fourier transform into the frequency domain (paper eq. pft_final)."""
        if self.domain != "space":
            raise ValueError("pft needs a space-domain field")
        k, res, spec_phase = pft_2d(
            self.points, self.amplitude, self.phase, self.gradient, self.hessian
        )
        return GridlessField(
            k,
            spec_phase,
            -self.points.copy(),
            -np.linalg.inv(self.hessian),
            res,
            self.wavelength_um,
            self.refractive_index,
            "frequency",
        )

    def inverse_pft(self) -> GridlessField:
        """Inverse pointwise Fourier transform back into the space domain."""
        if self.domain != "frequency":
            raise ValueError("inverse_pft needs a frequency-domain field")
        rho, u, phase = inverse_pft_2d(
            self.points, self.amplitude, self.phase, self.gradient, self.hessian
        )
        return GridlessField(
            rho,
            phase,
            self.points.copy(),
            -np.linalg.inv(self.hessian),
            u,
            self.wavelength_um,
            self.refractive_index,
        )


@dataclass(frozen=True)
class GriddedField:
    """Field on a centred square grid with an analytic quadratic carrier phase.

    The samples sit at p = L·u with u on the centred n×n grid of `step` (`lattice` L = identity:
    a plain grid; the SFT produces the lattice k = C·ρ'). The physical field is
    values·exp(i/2·pᵀCp); C = 0 is a plain grid.

    Attributes:
        step: grid step of u (mm in space with L = I, 1/mm in frequency with L = I; with a
            lattice the unit of u is that of p divided by the unit of L).
        values: (N, N, 3) complex samples, rows y, columns x.
        wavelength_um: vacuum wavelength in µm.
        refractive_index: index of the medium.
        curvature: carrier matrix C (2, 2), 1/mm² in space, mm² in frequency.
        domain: "space" (p = ρ in mm) or "frequency" (p = k⊥ in 1/mm).
        lattice: L (2, 2), maps the grid u to the sample positions p.
    """

    step: float
    values: ComplexArray
    wavelength_um: float
    refractive_index: float = 1.0
    curvature: FloatArray = field(default_factory=lambda: np.zeros((2, 2)))
    domain: Domain = "space"
    lattice: FloatArray = field(default_factory=lambda: np.eye(2))

    def __post_init__(self) -> None:
        shape = np.shape(self.values)
        if len(shape) != 3 or shape[0] != shape[1] or shape[2] != 3:
            raise ValueError("values must have shape (N, N, 3)")
        if np.shape(self.curvature) != (2, 2):
            raise ValueError("curvature must be a 2x2 matrix")
        if np.shape(self.lattice) != (2, 2) or np.linalg.det(self.lattice) == 0.0:
            raise ValueError("lattice must be an invertible 2x2 matrix")

    @property
    def k(self) -> float:
        """Wavenumber n·k₀ of the medium in 1/mm."""
        return wavenumber(self.wavelength_um, self.refractive_index)

    @property
    def x(self) -> FloatArray:
        """Centred grid coordinates u (unit of `step`)."""
        return centered_grid(np.shape(self.values)[0], self.step)

    @property
    def plain(self) -> bool:
        """True for an identity lattice (samples on the centred grid itself), up to rounding."""
        return bool(np.allclose(self.lattice, np.eye(2), rtol=0.0, atol=1e-12))

    def sample_points(self) -> FloatArray:
        """(N, N, 2) sample positions p = L·u (mm in space, 1/mm in frequency)."""
        x = self.x
        grid = np.stack(np.meshgrid(x, x), axis=-1)
        return np.asarray(np.einsum("ij,...j->...i", self.lattice, grid))

    def carrier(self) -> FloatArray:
        """(N, N) carrier phase ½pᵀCp in rad."""
        p = self.sample_points()
        return np.asarray(0.5 * np.einsum("...i,ij,...j->...", p, self.curvature, p))

    def physical(self) -> ComplexArray:
        """values·exp(i/2·pᵀCp); raises ValueError if the carrier is not resolved by the grid.

        Resolution: the carrier changes by at most π per grid step, max|Lᵀ C p|·step ≤ π.
        """
        p = self.sample_points()
        slope = np.einsum("ji,jk,...k->...i", self.lattice, self.curvature, p)
        worst = float(np.max(np.abs(slope))) * self.step
        if worst > np.pi:
            raise ValueError(f"carrier phase not resolved on this grid ({worst:.3g} > π per step)")
        return np.asarray(self.values * np.exp(1j * self.carrier())[..., None])
