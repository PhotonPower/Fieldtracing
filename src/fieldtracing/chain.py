"""Operators on the hybrid representation with a common interface (paper, eq. system_chain).

Every operator maps a field to a field of the same form U·exp(iψ) (paper, section 7.2, table
tab:chaining) and returns a log entry with its figures of merit (`selection`) and validity
warnings:

    field, entry = operator(field, ...)

Operators are pure functions; fields are immutable. A system E_out = P_M T_M ⋯ P_1 T_1 P_0 E_in
(eq. system_chain) is a list of operators applied with `run_chain`, e.g.

    run_chain(field, [partial(lpia_surface, n2=1.5, ...), partial(aperture, stop=...),
                      partial(pft, length_u=2.0), partial(angular_spectrum, dz=10.0), ...])

Representations: `GridlessField` (point cloud in space or frequency) and `GriddedField` (grid with
an analytic quadratic carrier; space or frequency, plain or on the SFT lattice k = C·ρ').

| operator | input → output | accuracy (table tab:chaining) |
| --- | --- | --- |
| `geometric_propagation` | cloud (space) → cloud (space) | O(1/F) |
| `lpia_surface` | cloud (space) → cloud (space) | LPIA validity |
| `aperture` | cloud or grid (space) → same | Kirchhoff model |
| `pft`, `inverse_pft` | cloud ↔ cloud (frequency) | O(1/F) |
| `angular_spectrum` | frequency cloud or grid, space grid → same | exact |
| `sft`, `inverse_sft` | grid with carrier ↔ grid on the lattice k = C·ρ' | discretisation |
| `fft`, `inverse_fft` | plain grid ↔ plain grid | discretisation |
| `resample` | cloud (space) → grid with carrier (only U interpolated, ADR 0003) | interpolation |
| `debye` | cloud (space, exit plane) → grid in the focal region (eq. debye) | ≈ 1.5/N_F |
| `rayleigh_sommerfeld` | plain space grid → grid (eq. rs_kernel, direct sum) | discretisation |

Warnings are collected in the log entry instead of being emitted.

The planner (`plan_interface`, `plan_propagation`) picks the operators of a step by the rules of
table tab:selection (`selection.select_method`) and predicts the error budget of table
tab:errors; `Plan.run` applies them.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from fieldtracing._types import ComplexArray, FloatArray, IntArray
from fieldtracing.field import GriddedField, GridlessField
from fieldtracing.focus import debye_focus
from fieldtracing.fourier import centered_grid, frequency_step, ft, ift
from fieldtracing.lpia import SagFunction, interface_amplitude, trace_surface
from fieldtracing.mesh import Region, fit_derivatives, mesh_jacobian_cloud
from fieldtracing.pft import inverse_pft_2d, pft_2d
from fieldtracing.propagation import angular_spectrum as angular_spectrum_grid
from fieldtracing.propagation import longitudinal_wavenumber, rayleigh_sommerfeld_kernel
from fieldtracing.selection import (
    Situation,
    debye_error,
    edge_wave_ratio,
    fit_curvature,
    fresnel_width,
    grid_edge_error,
    local_fresnel_number,
    q_for_tolerance,
    select_method,
    sft_gain,
    sft_step,
)
from fieldtracing.semiclassical import propagate_2d
from fieldtracing.sft import check_sft_sampling, inverse_sft_2d, region_weights, sft_2d
from fieldtracing.units import wavenumber

Field: TypeAlias = "GridlessField | GriddedField"


@dataclass(frozen=True)
class LogEntry:
    """Protocol entry of one operator.

    Attributes:
        operator: operator name.
        metrics: figures of merit (names carry the unit, e.g. `fresnel_width_min_mm`).
        warnings: validity warnings (stationary phase, sampling, edges, caustics).
    """

    operator: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


Step: TypeAlias = "tuple[GridlessField | GriddedField, LogEntry]"
Operator: TypeAlias = "Callable[[GridlessField | GriddedField], Step]"


def run_chain(
    start: GridlessField | GriddedField, operators: Sequence[Operator]
) -> tuple[GridlessField | GriddedField, tuple[LogEntry, ...]]:
    """Apply operators in order (eq. system_chain); returns the final field and the protocol."""
    current = start
    log = []
    for op in operators:
        current, entry = op(current)
        log.append(entry)
    return current, tuple(log)


# ----- helpers --------------------------------------------------------------------------------


def _cloud(f: GridlessField | GriddedField, domain: str, what: str) -> GridlessField:
    if not isinstance(f, GridlessField) or f.domain != domain:
        raise ValueError(f"{what} needs a GridlessField in the {domain} domain")
    return f


def _grid(f: GridlessField | GriddedField, domain: str, what: str) -> GriddedField:
    if not isinstance(f, GriddedField) or f.domain != domain:
        raise ValueError(f"{what} needs a GriddedField in the {domain} domain")
    return f


def _curvature_metrics(
    hessian: FloatArray, length_u: float | None, min_fresnel: float, unit: str
) -> tuple[dict[str, float], list[str]]:
    """Fresnel widths w = |h|^(-1/2) and local Fresnel numbers F = |h|·L_U² (paper, 7.1)."""
    eigen = np.abs(np.linalg.eigvalsh(np.asarray(hessian)))
    h_min, h_max = float(np.min(eigen)), float(np.max(eigen))
    metrics = {
        f"fresnel_width_min_{unit}": fresnel_width(h_max),
        f"fresnel_width_max_{unit}": fresnel_width(h_min),
    }
    notes = []
    if length_u is not None:
        f_min = local_fresnel_number(h_min, length_u)
        metrics["fresnel_number_min"] = f_min
        if f_min < min_fresnel:
            notes.append(
                f"local Fresnel number {f_min:.3g} < {min_fresnel:.3g}: stationary phase "
                "error O(1/F) is large (use the SFT or an FFT)"
            )
    return metrics, notes


def _jacobian_metrics(
    det: FloatArray, maslov: NDArray[np.integer], threshold: float
) -> tuple[dict[str, float], list[str]]:
    finite = np.isfinite(det)
    small = float(np.min(np.abs(det[finite]))) if np.any(finite) else math.nan
    metrics = {"jacobian_min_abs": small, "maslov_max": float(np.max(maslov))}
    notes = []
    if not math.isnan(small) and small < threshold:
        notes.append(
            f"|J| = {small:.3g} < {threshold:.3g}: target plane near a caustic, U/sqrt|J| "
            "unreliable (use the angle chart and an FFT)"
        )
    if not np.all(finite):
        notes.append(f"{int(np.sum(~finite))} rays without a mesh Jacobian (isolated)")
    return metrics, notes


def _per_component(
    values: ComplexArray, transform: Callable[[ComplexArray], ComplexArray]
) -> ComplexArray:
    return np.asarray(np.stack([transform(values[..., c]) for c in range(3)], axis=-1))


# ----- geometric propagation and LPIA (paper, sections 3.4 and 3.5) ---------------------------


def geometric_propagation(
    f: GridlessField | GriddedField,
    distance: float,
    jacobian: Literal["hessian", "mesh"] = "hessian",
    length_u: float | None = None,
    min_fresnel: float = 10.0,
    caustic_threshold: float = 1e-2,
    max_edge: float | None = None,
    region: Region | None = None,
) -> Step:
    """Geometric propagation by `distance` mm in the medium of the field (eq. jacobian_amplitude).

    New phase ψ₁ + k·r, new residual U₁|J|^(-1/2)·exp(-iπμ/2), samples move with the rays.

    Args:
        f: GridlessField in the space domain.
        distance: plane distance d in mm.
        jacobian: "hessian": J from the stored phase Hessian (`semiclassical.propagate_2d`);
            "mesh": J from the landing points of the neighbouring rays (`mesh`), for clouds
            without a reliable Hessian; the new Hessian then uses ∇ψ fitted on the mesh.
        length_u: variation length of U in mm for the Fresnel-number check (optional).
        min_fresnel: warn below this local Fresnel number.
        caustic_threshold: warn when min|J| falls below it.
        max_edge, region: mesh options (`mesh.triangulate`) for jacobian = "mesh".
    """
    g = _cloud(f, "space", "geometric_propagation")
    res = propagate_2d(g.points, g.amplitude, g.phase, g.gradient, g.hessian, g.k, distance)
    metrics, notes = _curvature_metrics(g.hessian, length_u, min_fresnel, "mm")
    det, maslov, amplitude, hessian = res.jacobian, res.maslov, res.amplitude, res.curvature
    if jacobian == "mesh":
        cloud = mesh_jacobian_cloud(
            g.points, res.points, threshold=caustic_threshold, max_edge=max_edge, region=region
        )
        det, maslov = cloud.determinant, cloud.maslov
        factor = np.exp(-0.5j * np.pi * maslov) / np.sqrt(np.abs(det))
        amplitude = g.amplitude * factor[:, None]
        d_grad, _ = fit_derivatives(g.points, g.gradient, cloud.triangles)
        hessian = np.einsum("nij,njk->nik", d_grad, _safe_inverse(cloud.matrix))
        metrics["caustic_nodes"] = float(np.sum(cloud.caustic))
    assert hessian is not None
    jm, jn = _jacobian_metrics(det, maslov, caustic_threshold)
    out = GridlessField(
        res.points,
        res.phase,
        g.gradient.copy(),
        np.asarray(hessian),
        np.asarray(amplitude),
        g.wavelength_um,
        g.refractive_index,
    )
    entry = LogEntry(
        "geometric_propagation", {"distance_mm": distance, **metrics, **jm}, tuple(notes + jn)
    )
    return out, entry


def _safe_inverse(matrix: FloatArray) -> FloatArray:
    out = np.full_like(matrix, np.nan)
    ok = np.all(np.isfinite(matrix), axis=(1, 2)) & (
        np.abs(np.linalg.det(np.nan_to_num(matrix))) > 0
    )
    out[ok] = np.linalg.inv(matrix[ok])
    return out


def lpia_surface(
    f: GridlessField | GriddedField,
    n2: float,
    sag: SagFunction,
    sag_gradient: Callable[[FloatArray, FloatArray], tuple[FloatArray, FloatArray]],
    z_out: float,
    caustic_threshold: float = 1e-2,
    max_edge: float | None = None,
    region: Region | None = None,
) -> Step:
    """LPIA step through the surface z = h(x, y) to the plane z_out (eq. interface_amplitude).

    The field lies on the plane z = 0 in its medium (index n₁ = f.refractive_index). New phase
    ψ + k₀ℓ, residual by eq. interface_amplitude with the Jacobian from the neighbouring rays of
    the cloud (`mesh.mesh_jacobian_cloud`, any arrangement of the samples). The new Hessian is
    H₂ = (∂k⊥/∂ρ₁)·(∂ρ₂/∂ρ₁)⁻¹, both derivatives fitted on the mesh (k⊥ is conserved behind the
    surface).

    Args:
        f: GridlessField in the space domain.
        n2: refractive index behind the surface.
        sag, sag_gradient: surface h(x, y) in mm and its gradient (see `lpia.trace_surface`).
        z_out: output plane in mm (behind the surface).
        caustic_threshold, max_edge, region: mesh options (`mesh.mesh_jacobian_cloud`).
    """
    g = _cloud(f, "space", "lpia_surface")
    k0 = wavenumber(g.wavelength_um)
    trace = trace_surface(
        g.points,
        g.amplitude,
        g.phase,
        g.gradient,
        g.refractive_index,
        n2,
        k0,
        sag,
        sag_gradient,
        z_out,
    )
    cloud = mesh_jacobian_cloud(
        g.points, trace.points, threshold=caustic_threshold, max_edge=max_edge, region=region
    )
    amplitude = interface_amplitude(trace, cloud.matrix)
    k_perp = n2 * k0 * trace.direction_out[:, :2]
    d_k, _ = fit_derivatives(g.points, k_perp, cloud.triangles)
    hessian = np.einsum("nij,njk->nik", d_k, _safe_inverse(cloud.matrix))
    out = GridlessField(
        trace.points,
        trace.phase,
        np.asarray(k_perp),
        np.asarray(hessian),
        amplitude,
        g.wavelength_um,
        n2,
    )
    metrics, notes = _jacobian_metrics(cloud.determinant, cloud.maslov, caustic_threshold)
    finite = np.all(np.isfinite(hessian), axis=(1, 2))
    if np.any(finite):
        h_max = float(np.max(np.abs(np.linalg.eigvalsh(hessian[finite]))))
        # wavefront radius k/|h|; LPIA needs it ≫ λ (table tab:selection)
        metrics["wavefront_radius_min_mm"] = out.k / h_max if h_max > 0 else math.inf
    metrics["caustic_nodes"] = float(np.sum(cloud.caustic))
    metrics["cos_incidence_min"] = float(np.min(trace.cos_in))
    return out, LogEntry("lpia_surface", {"z_out_mm": z_out, **metrics}, tuple(notes))


# ----- aperture (Kirchhoff model) ---------------------------------------------------------------


class Aperture(Protocol):
    """A stop in the plane of the field: transmitting region and distance to its edge (mm)."""

    def inside(self, x: FloatArray, y: FloatArray) -> NDArray[np.bool_]: ...

    def edge_distance(self, x: FloatArray, y: FloatArray) -> FloatArray: ...


@dataclass(frozen=True)
class CircularAperture:
    """Circular stop of `radius` mm, optionally with a central obscuration of `inner_radius`."""

    radius: float
    inner_radius: float = 0.0

    def inside(self, x: FloatArray, y: FloatArray) -> NDArray[np.bool_]:
        r = np.hypot(x, y)
        return np.asarray((r <= self.radius) & (r >= self.inner_radius))

    def edge_distance(self, x: FloatArray, y: FloatArray) -> FloatArray:
        r = np.hypot(x, y)
        outer = np.abs(self.radius - r)
        if self.inner_radius <= 0.0:
            return np.asarray(outer)
        return np.asarray(np.minimum(outer, np.abs(r - self.inner_radius)))


@dataclass(frozen=True)
class RectangularAperture:
    """Rectangular stop |x| ≤ half_x, |y| ≤ half_y (mm)."""

    half_x: float
    half_y: float

    def inside(self, x: FloatArray, y: FloatArray) -> NDArray[np.bool_]:
        return np.asarray((np.abs(x) <= self.half_x) & (np.abs(y) <= self.half_y))

    def edge_distance(self, x: FloatArray, y: FloatArray) -> FloatArray:
        dx = self.half_x - np.abs(x)
        dy = self.half_y - np.abs(y)
        inner = np.minimum(dx, dy)
        outer = np.hypot(np.minimum(dx, 0.0), np.minimum(dy, 0.0))
        return np.asarray(np.where(inner >= 0.0, inner, outer))


def aperture(
    f: GridlessField | GriddedField,
    stop: Aperture,
    tolerance: float = 1e-2,
    oversample: int = 8,
) -> Step:
    """Multiply the residual by the stop's indicator χ_Ω (Kirchhoff model, table tab:chaining).

    Phase and samples are unchanged. On a point cloud the samples outside get U = 0; on a plain
    grid each sample is weighted by the covered fraction of its cell (`sft.region_weights`).
    For a cloud the edge-wave ratio w/(sqrt(2π) d_e) (eq. edge_ratio) of the transmitted samples
    is logged; a value above `tolerance` means the edge wave is not negligible.

    Args:
        f: GridlessField or plain GriddedField in the space domain.
        stop: e.g. `CircularAperture`, `RectangularAperture`.
        tolerance: ε of the selection rules (table tab:selection).
        oversample: sub-samples per cell and axis for the grid weights.
    """
    if isinstance(f, GridlessField):
        g = _cloud(f, "space", "aperture")
        x, y = g.points[:, 0], g.points[:, 1]
        inside = stop.inside(x, y)
        out: GridlessField | GriddedField = GridlessField(
            g.points,
            g.phase,
            g.gradient,
            g.hessian,
            np.where(inside[:, None], g.amplitude, 0.0),
            g.wavelength_um,
            g.refractive_index,
        )
        metrics = {"transmitted_samples": float(np.sum(inside))}
        notes = []
        if np.any(inside):
            d_e = stop.edge_distance(x[inside], y[inside])
            h = np.abs(np.linalg.eigvalsh(g.hessian[inside]))
            w = np.array([fresnel_width(float(v)) for v in np.min(h, axis=1)])
            ratio = np.array(
                [
                    edge_wave_ratio(float(wi), float(di)) if di > 0 else math.inf
                    for wi, di in zip(w, d_e, strict=True)
                ]
            )
            worst = float(np.max(ratio))
            metrics.update(
                {"edge_distance_min_mm": float(np.min(d_e)), "edge_wave_ratio_max": worst}
            )
            if worst > tolerance:
                notes.append(
                    f"edge wave ratio {worst:.3g} > {tolerance:.3g}: edge diffraction not "
                    "negligible (SFT or erfc correction)"
                )
        return out, LogEntry("aperture", metrics, tuple(notes))
    grid = _grid(f, "space", "aperture")
    if not grid.plain:
        raise ValueError("aperture needs a plain grid (identity lattice)")
    x = grid.x
    weights = region_weights(x, x, stop.inside, oversample)
    out = GriddedField(
        grid.step,
        np.asarray(grid.values * weights[..., None]),
        grid.wavelength_um,
        grid.refractive_index,
        grid.curvature,
    )
    return out, LogEntry(
        "aperture", {"transmitted_area_mm2": float(np.sum(weights)) * grid.step**2}
    )


# ----- pointwise Fourier transform (paper, section 4) ------------------------------------------


def _collect(
    call: Callable[[], tuple[FloatArray, ComplexArray, FloatArray]],
) -> tuple[tuple[FloatArray, ComplexArray, FloatArray], list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    return result, [str(w.message) for w in caught]


def pft(
    f: GridlessField | GriddedField, length_u: float | None = None, min_fresnel: float = 10.0
) -> Step:
    """Pointwise Fourier transform of a space-domain cloud (eq. pft_final, `pft.pft_2d`).

    New phase ψ(ρ_s) - k⊥·ρ_s, residual U·exp(iπ/4·sig H)/(2π sqrt|det H|), samples k⊥ = ∇ψ;
    the spectral gradient is -ρ_s (mm) and the spectral Hessian -H⁻¹ (mm²).

    Args:
        length_u: variation length of U in mm for the validity check (optional).
    """
    g = _cloud(f, "space", "pft")
    (k, res, phase), notes = _collect(
        lambda: pft_2d(g.points, g.amplitude, g.phase, g.gradient, g.hessian, length_u, min_fresnel)
    )
    metrics, more = _curvature_metrics(g.hessian, length_u, min_fresnel, "mm")
    out = GridlessField(
        k,
        phase,
        -g.points.copy(),
        -np.linalg.inv(g.hessian),
        res,
        g.wavelength_um,
        g.refractive_index,
        "frequency",
    )
    return out, LogEntry("pft", metrics, tuple(notes or more))


def inverse_pft(
    f: GridlessField | GriddedField, length_u: float | None = None, min_fresnel: float = 10.0
) -> Step:
    """Inverse pointwise Fourier transform of a frequency-domain cloud (`pft.inverse_pft_2d`).

    Args:
        length_u: variation length of the residual in 1/mm for the validity check (optional).
    """
    g = _cloud(f, "frequency", "inverse_pft")
    (rho, u, phase), notes = _collect(
        lambda: inverse_pft_2d(
            g.points, g.amplitude, g.phase, g.gradient, g.hessian, length_u, min_fresnel
        )
    )
    metrics, more = _curvature_metrics(g.hessian, length_u, min_fresnel, "per_mm")
    out = GridlessField(
        rho,
        phase,
        g.points.copy(),
        -np.linalg.inv(g.hessian),
        u,
        g.wavelength_um,
        g.refractive_index,
    )
    return out, LogEntry("inverse_pft", metrics, tuple(notes or more))


# ----- angular spectrum (exact, pointwise in k⊥) -------------------------------------------------


def angular_spectrum(f: GridlessField | GriddedField, dz: float) -> Step:
    """Plane-wave propagation by dz mm: Ẽ → Ẽ·exp(i k_z dz) (eq. spw), exact in every chart.

    - Frequency cloud: phase += k_z dz, spectral gradient += -dz k⊥/k_z, spectral Hessian +=
      -dz(I/k_z + k⊥k⊥ᵀ/k_z³); residual unchanged. Evanescent samples raise ValueError.
    - Frequency grid with zero carrier (FFT path): values·exp(i k_z dz).
    - Frequency grid with a carrier B (SFT lattice): the paraxial part -dz|k⊥|²/(2k) goes into
      the carrier (B - dz/k·I), the remainder exp(i(k_z + |k⊥|²/(2k))dz) into the values, so the
      values stay smooth for the inverse SFT. Evanescent grid samples decay (Im k_z ≥ 0).
    - Plain space grid without carrier: `propagation.angular_spectrum` per component (periodic).
    """
    k = f.k
    if isinstance(f, GridlessField):
        g = _cloud(f, "frequency", "angular_spectrum on a point cloud (transform first)")
        k_sq = np.sum(g.points * g.points, axis=1)
        if np.any(k_sq >= k * k):
            raise ValueError("evanescent samples in the angular spectrum of a point cloud")
        kz = np.sqrt(k * k - k_sq)
        d_kz = (
            np.eye(2)[None] / kz[:, None, None]
            + np.einsum("ni,nj->nij", g.points, g.points) / (kz**3)[:, None, None]
        )
        out: GridlessField | GriddedField = GridlessField(
            g.points,
            g.phase + kz * dz,
            g.gradient - dz * g.points / kz[:, None],
            g.hessian - dz * d_kz,
            g.amplitude,
            g.wavelength_um,
            g.refractive_index,
            "frequency",
        )
        sin_max = math.sqrt(float(np.max(k_sq))) / k
        return out, LogEntry("angular_spectrum", {"dz_mm": dz, "sin_theta_max": sin_max})
    if f.domain == "space":
        if not f.plain:
            raise ValueError("angular_spectrum on a space grid needs a plain grid")
        values = _per_component(f.physical(), lambda v: angular_spectrum_grid(v, f.step, k, dz))
        out = GriddedField(f.step, values, f.wavelength_um, f.refractive_index)
        return out, LogEntry("angular_spectrum", {"dz_mm": dz, "grid": 1.0})
    p = f.sample_points()
    k_sq = np.sum(p * p, axis=-1)
    kz = longitudinal_wavenumber(k_sq, k)
    metrics = {"dz_mm": dz, "evanescent_fraction": float(np.mean(k_sq > k * k))}
    if not np.any(f.curvature):
        values = f.values * np.exp(1j * kz * dz)[..., None]
        curvature = f.curvature
    else:
        values = f.values * np.exp(1j * (kz + 0.5 * k_sq / k) * dz)[..., None]
        curvature = f.curvature - dz / k * np.eye(2)
    out = GriddedField(
        f.step,
        np.asarray(values),
        f.wavelength_um,
        f.refractive_index,
        np.asarray(curvature),
        "frequency",
        f.lattice,
    )
    return out, LogEntry("angular_spectrum", metrics)


# ----- SFT and FFT (paper, section 6.4) ----------------------------------------------------------


def sft(f: GridlessField | GriddedField, q_min: float = 4.0) -> Step:
    """Semi-analytical Fourier transform of a plain space grid with carrier C (eq. sft_formula).

    The output lives on the lattice k = C·ρ' (same u grid) with carrier -C⁻¹ and residual
    W(C⁻¹k)/(2π)² (`sft.sft_2d`, per field component). Violated sampling conditions of
    `sft.check_sft_sampling` are logged as warnings.
    """
    g = _grid(f, "space", "sft")
    if not g.plain:
        raise ValueError("sft needs a plain space grid (identity lattice)")
    c = g.curvature
    if np.linalg.det(c) == 0.0:
        raise ValueError("sft needs an invertible carrier C (zero curvature: use fft)")
    n = np.shape(g.values)[0]
    residual = _per_component(g.values, lambda v: sft_2d(v, g.step, c, warn=False)[1])
    eigen = np.abs(np.linalg.eigvalsh(c))
    w_c = math.sqrt(2.0 / float(np.max(eigen)))
    metrics = {
        "fresnel_width_min_mm": fresnel_width(float(np.max(eigen))),
        "samples_per_w_c": w_c / g.step,
        "window_mm": n * g.step,
        "sft_gain": sft_gain(float(np.min(eigen)) * (n * g.step / 2.0) ** 2, w_c / g.step),
    }
    notes = tuple(check_sft_sampling(n, g.step, c, q_min))
    out = GriddedField(
        g.step,
        residual,
        g.wavelength_um,
        g.refractive_index,
        np.asarray(-np.linalg.inv(c)),
        "frequency",
        np.asarray(c),
    )
    return out, LogEntry("sft", metrics, notes)


def inverse_sft(f: GridlessField | GriddedField) -> Step:
    """Inverse SFT of a frequency grid with carrier B on the lattice k = L·u (`inverse_sft_2d`).

    Output: space grid on the lattice ρ = -B·L·u' with carrier -B⁻¹. After `sft` alone
    (B = -C⁻¹, L = C) this is the original plain grid.
    """
    g = _grid(f, "frequency", "inverse_sft")
    b, lat = g.curvature, g.lattice
    if np.linalg.det(b) == 0.0:
        raise ValueError("inverse_sft needs an invertible carrier (zero carrier: use inverse_fft)")
    values = _per_component(g.values, lambda v: inverse_sft_2d(v, g.step, b, lat)[1])
    q = lat.T @ b @ lat
    eigen = np.abs(np.linalg.eigvalsh(0.5 * (q + q.T)))
    w_c = math.sqrt(2.0 / float(np.max(eigen)))
    out = GriddedField(
        g.step,
        values,
        g.wavelength_um,
        g.refractive_index,
        np.asarray(-np.linalg.inv(b)),
        "space",
        np.asarray(-b @ lat),
    )
    return out, LogEntry("inverse_sft", {"samples_per_w_c": w_c / g.step})


def fft(f: GridlessField | GriddedField) -> Step:
    """Fourier transform of a plain space grid (`fourier.ft`, convention (2π)^-2).

    A carrier is folded into the values first (`GriddedField.physical`, which checks that the
    grid resolves it). Output: plain frequency grid with step 2π/(N·step), carrier 0.
    """
    g = _grid(f, "space", "fft")
    if not g.plain:
        raise ValueError("fft needs a plain grid (identity lattice)")
    n = np.shape(g.values)[0]
    spectrum = ft(g.physical(), g.step, axes=(0, 1))
    dk = frequency_step(n, g.step)
    out = GriddedField(dk, spectrum, g.wavelength_um, g.refractive_index, domain="frequency")
    metrics = {"window_mm": n * g.step, "k_max_over_k": math.pi / g.step / g.k}
    return out, LogEntry("fft", metrics)


def inverse_fft(f: GridlessField | GriddedField) -> Step:
    """Inverse Fourier transform of a plain frequency grid (`fourier.ift`)."""
    g = _grid(f, "frequency", "inverse_fft")
    if not g.plain:
        raise ValueError("inverse_fft needs a plain grid (identity lattice)")
    n = np.shape(g.values)[0]
    values = ift(g.physical(), g.step, axes=(0, 1))
    out = GriddedField(frequency_step(n, g.step), values, g.wavelength_um, g.refractive_index)
    return out, LogEntry("inverse_fft", {"window_per_mm": n * g.step})


# ----- point cloud → grid ------------------------------------------------------------------------


def resample(
    f: GridlessField | GriddedField,
    n: int,
    step: float,
    curvature: FloatArray | None = None,
    method: Literal["linear", "cubic"] = "linear",
) -> Step:
    """Resample a space cloud on an n×n grid; only U is interpolated (`GridlessField.to_grid`).

    Logs the largest phase change per step of the remaining phase ψ - ½ρᵀCρ (must be ≤ π).
    """
    g = _cloud(f, "space", "resample")
    c = np.zeros((2, 2)) if curvature is None else np.asarray(curvature, dtype=np.float64)
    grid = g.to_grid(n, step, c, method)
    rest = float(np.max(np.abs(g.gradient - g.points @ c))) * step
    return grid, LogEntry("resample", {"residual_phase_per_step": rest, "window_mm": n * step})


# ----- focus: Debye integral and Rayleigh-Sommerfeld (paper, sections 2 and 5.3) ----------------


def debye(
    f: GridlessField | GriddedField,
    distance: float,
    sin_max: float,
    n: int = 64,
    pad: int = 4,
) -> Step:
    """Field near the focus by the Debye integral (eq. debye, `focus.debye_focus`).

    The reference sphere is centred on the axis in the target plane (f = distance). Its phase
    -k(sqrt(f² + ρ²) - f) is removed from the cloud; the rest U·exp(i(ψ - ψ_sphere)) is smooth
    (aberration and defocus) and is the only interpolated quantity (ADR 0003). The k-domain
    |k⊥| ≤ k·sin_max is the image of the stop |ρ| ≤ f·sin_max/cos θ_max under the sphere's ray
    map, with subpixel-weighted boundary; boundary cells whose centre lies just outside the
    samples take the nearest sample (the disc weights alone set the edge, otherwise the edge
    would shrink by up to a cell and the error would be first order in the k-step).

    Args:
        f: GridlessField in the space domain (exit plane, stop not yet applied).
        distance: exit plane → target plane in mm.
        sin_max: sine of the marginal-ray angle (stop radius a: a/sqrt(a² + distance²)).
        n, pad: samples across the k-disc and zero-padding factor (`debye_focus`).

    Returns:
        GriddedField (space) in the target plane, constant phase k·f included.
    """
    g = _cloud(f, "space", "debye")
    k = g.k

    def residual(x: FloatArray, y: FloatArray) -> ComplexArray:
        q = np.stack([np.ravel(x), np.ravel(y)], axis=-1)
        sphere = -k * (np.sqrt(distance**2 + np.sum(q * q, axis=1)) - distance)
        v = g.residual_at(q, outside="nearest") * np.exp(1j * (g.phase_at(q) - sphere))[:, None]
        return np.asarray(v.reshape(*np.shape(x), 3))

    focal = debye_focus(residual, distance, k, sin_max, n=n, pad=pad)
    values = np.asarray(focal.values * np.exp(1j * focal.phase_offset))
    out = GriddedField(float(focal.x[1] - focal.x[0]), values, g.wavelength_um, g.refractive_index)
    n_f = (distance * sin_max) ** 2 / (1.0 - sin_max**2) * k / (2.0 * math.pi * distance)
    metrics = {"distance_mm": distance, "sin_max": sin_max, "fresnel_number": n_f}
    notes: tuple[str, ...] = ()
    if debye_error(n_f) > 0.1:
        notes = (f"Debye error 1.5/N_F = {debye_error(n_f):.3g} (N_F = {n_f:.3g})",)
    return out, LogEntry("debye", metrics, notes)


def rayleigh_sommerfeld(
    f: GridlessField | GriddedField, distance: float, n_out: int, step_out: float
) -> Step:
    """Direct Rayleigh-Sommerfeld integral (eq. rs_kernel) onto an n_out² grid at `distance` mm.

    Rectangle rule over a plain space grid (the carrier must be resolved); cost N_in·N_out.

    Args:
        f: plain GriddedField in the space domain.
        distance: plane distance in mm (> 0).
        n_out, step_out: centred output grid (mm).
    """
    g = _grid(f, "space", "rayleigh_sommerfeld")
    if not g.plain:
        raise ValueError("rayleigh_sommerfeld needs a plain grid")
    phys = g.physical().reshape(-1, 3)
    keep = np.any(phys != 0.0, axis=1)
    x = g.x
    src = np.stack(np.meshgrid(x, x), axis=-1).reshape(-1, 2)[keep]
    values = phys[keep] * g.step**2
    xo = centered_grid(n_out, step_out)
    tgt = np.stack(np.meshgrid(xo, xo), axis=-1).reshape(-1, 2)
    out = np.empty((len(tgt), 3), dtype=np.complex128)
    chunk = max(1, 2_000_000 // max(len(src), 1))
    for i in range(0, len(tgt), chunk):
        t = tgt[i : i + chunk]
        kern = rayleigh_sommerfeld_kernel(
            t[:, None, 0] - src[None, :, 0], t[:, None, 1] - src[None, :, 1], distance, g.k
        )
        out[i : i + chunk] = kern @ values
    grid = GriddedField(step_out, out.reshape(n_out, n_out, 3), g.wavelength_um, g.refractive_index)
    return grid, LogEntry(
        "rayleigh_sommerfeld", {"distance_mm": distance, "sources": float(len(src))}
    )


# ----- planner (paper, section 7: tables tab:selection, tab:errors; eq. sft_gain) ----------------


@dataclass(frozen=True)
class Plan:
    """Operators for one step with the reasons and the predicted error budget.

    Attributes:
        method: chosen building block (`selection.METHODS`).
        reason: deciding figures in words.
        situation: the figures of merit (`selection.Situation`).
        budget: predicted relative errors per source of table tab:errors (relative to the
            largest field amplitude in the target plane).
        operators: the chain to apply (`run`).
        warnings: known limitations of the plan.
        parameters: numerical parameters (grid size, step, q, zones; names carry units).
    """

    method: str
    reason: str
    situation: Situation
    budget: Mapping[str, float]
    operators: tuple[Operator, ...]
    warnings: tuple[str, ...] = ()
    parameters: Mapping[str, float] = field(default_factory=dict)

    @property
    def predicted_error(self) -> float:
        """Sum of the budget terms (conservative; the largest term dominates, paper 7.3)."""
        return float(sum(self.budget.values()))

    def run(
        self, start: GridlessField | GriddedField
    ) -> tuple[GridlessField | GriddedField, tuple[LogEntry, ...]]:
        """Apply the operators (`run_chain`)."""
        if not self.operators:
            raise ValueError(f"no operators for method {self.method!r} (not implemented)")
        return run_chain(start, self.operators)


def plan_interface(
    f: GridlessField,
    n2: float,
    sag: SagFunction,
    sag_gradient: Callable[[FloatArray, FloatArray], tuple[FloatArray, FloatArray]],
    z_out: float,
    surface_radius: float,
    tolerance: float = 0.02,
) -> Plan:
    """Plan an interface (row "Grenzfläche" of table tab:selection).

    The LPIA error is taken as λ/R_c with R_c the smaller of the surface radius and the incident
    wavefront radius k/|h|_max (order of magnitude only; paper cites von Pfeil et al. 2000).

    Args:
        surface_radius: smallest curvature radius of the surface in mm (inf for a plane).
    """
    g = _cloud(f, "space", "plan_interface")
    h_max = float(np.max(np.abs(np.linalg.eigvalsh(g.hessian)))) if len(g.points) else 0.0
    wavefront = g.k / h_max if h_max > 0.0 else math.inf
    radius = min(abs(surface_radius), wavefront)
    lam = 2.0 * math.pi / max(g.k, n2 * g.k / g.refractive_index)
    s = Situation(interface=True, curvature_radius_mm=radius, wavelength_mm=lam)
    method, reason = select_method(s, tolerance)
    ops: tuple[Operator, ...] = ()
    if method == "lpia":
        ops = (_bind(lpia_surface, n2=n2, sag=sag, sag_gradient=sag_gradient, z_out=z_out),)
    return Plan(method, reason, s, {"lpia": lam / radius}, ops)


def _bind(op: Callable[..., Step], **kwargs: object) -> Operator:
    def bound(f: GridlessField | GriddedField) -> Step:
        return op(f, **kwargs)

    bound.__name__ = op.__name__
    return bound


def _stop_size(stop: Aperture | None, points: FloatArray) -> float:
    if isinstance(stop, CircularAperture):
        return stop.radius
    if isinstance(stop, RectangularAperture):
        return min(stop.half_x, stop.half_y)
    return float(np.max(np.hypot(points[:, 0], points[:, 1])))


def _interpolation_error(g: GridlessField, mask: NDArray[np.bool_]) -> float:
    """Linear residual interpolation error, estimated against Clough-Tocher at edge midpoints."""
    pts = g.points[mask]
    if len(pts) < 4:
        return 0.0
    _, idx = cKDTree(g.points).query(pts, k=2)
    probes = 0.5 * (pts + g.points[idx[:, 1]])
    lin, cub = g.residual_at(probes, "linear"), g.residual_at(probes, "cubic")
    scale = float(np.max(np.abs(g.amplitude[mask])))
    return float(np.max(np.abs(lin - cub))) / scale if scale > 0.0 else 0.0


def _phase_taylor_error(g: GridlessField, mask: NDArray[np.bool_]) -> float:
    """Error of the second-order Taylor phase (`GridlessField.phase_at`) in rad.

    At the midpoint of each sample and its nearest neighbour the expansions about both samples
    are compared; their difference is of the order of the third-order remainder.
    """
    pts = g.points[mask]
    if len(pts) < 2:
        return 0.0
    i = np.nonzero(mask)[0]
    _, idx = cKDTree(g.points).query(pts, k=2)
    j = idx[:, 1]
    q = 0.5 * (pts + g.points[j])

    def taylor(n: IntArray) -> FloatArray:
        d = q - g.points[n]
        return np.asarray(
            g.phase[n]
            + np.sum(g.gradient[n] * d, axis=1)
            + 0.5 * np.einsum("ni,nij,nj->n", d, g.hessian[n], d)
        )

    return float(np.max(np.abs(taylor(i) - taylor(j))))


def airy_alias_error(period: float, window: float, k: float, sin_max: float) -> float:
    """Amplitude of the periodic copies of a focal spot at the edge of the target window.

    The FFT of the Debye integral is periodic with `period` (mm); a point at |x| ≤ window/2 sees
    the neighbouring copy at distance period - window/2. Envelope of 2J₁(x)/x:
    (8/π)^(1/2)·x^(-3/2), x = k·sin_max·r (aberration-free spot; a bound for well-corrected
    systems).
    """
    r = period - 0.5 * window
    if r <= 0.0:
        return math.inf
    return float(math.sqrt(8.0 / math.pi) * (k * sin_max * r) ** -1.5)


def plan_propagation(
    f: GridlessField | GriddedField,
    distance: float,
    stop: Aperture | None = None,
    tolerance: float = 0.02,
    length_u: float | None = None,
    focus_samples: int | None = None,
    method: str | None = None,
    focus_window: float | None = None,
) -> Plan:
    """Plan stop + propagation of a space cloud to the plane at `distance` mm (section 7).

    Figures (table tab:selection) over the samples inside the stop: F_j = |h_j|·L_U²
    (L_U: stop radius by default), edge ratio w/(sqrt(2π) d_e), target Fresnel number
    |h₂|·L_U²·|J| from the geometric propagation, N_F = F_max/(2π), and the samples per
    direction of the SFT and of the direct FFT for the same edge accuracy q (cost, eq. sft_gain).
    The chosen operators apply the stop themselves: on the cloud (geometric), on the grid with
    subpixel weights (SFT, FFT, Rayleigh-Sommerfeld) or as the k-disc (Debye).

    Error budget (table tab:errors), relative to the largest amplitude in the target plane:
    stationary phase 1/F_min, edge wave w/(sqrt(2π) d_e), Debye 1.5/N_F, hard edge on a grid
    0.4/q (`selection.grid_edge_error`), interpolation of U (linear against Clough-Tocher), k-disc
    boundary of the Debye FFT 0.02/n. For a circular stop on a grid the budget does not hold
    within `parameters["axis_zone_mm"]` of the axis (caustic of the edge wave, see warnings).

    Args:
        f: GridlessField in the space domain (the stop is not applied yet).
        distance: plane distance in mm.
        stop: aperture in the plane of the field (optional).
        tolerance: ε of the selection rules.
        length_u: variation length L_U of U in mm (default: stop radius or cloud radius).
        focus_samples: samples across the k-disc (Debye) or per direction of the focal grid;
            default: from `focus_window` so that the periodic copies stay below ε/10.
        focus_window: full width of the region of interest in the target plane (mm; default
            ten Airy diameters, 12.2 λ/sin θ).
        method: force a method instead of the selection rules (for comparisons).
    """
    g = _cloud(f, "space", "plan_propagation")
    x, y = g.points[:, 0], g.points[:, 1]
    lit = stop.inside(x, y) if stop is not None else np.ones(len(x), dtype=bool)
    if not np.any(lit):
        raise ValueError("no sample inside the stop")
    a = _stop_size(stop, g.points[lit])
    l_u = a if length_u is None else length_u
    eig = np.abs(np.linalg.eigvalsh(g.hessian[lit]))
    f_min = float(np.min(eig)) * l_u**2
    f_max = float(np.max(eig)) * l_u**2
    edge = 0.0
    if stop is not None:
        d_e = stop.edge_distance(x[lit], y[lit])
        w = np.array([fresnel_width(float(h)) for h in np.min(eig, axis=1)])
        edge = float(np.max(w / (math.sqrt(2.0 * math.pi) * np.maximum(d_e, 1e-300))))
    target_fresnel = _target_fresnel(g.gradient[lit], g.hessian[lit], g.k, distance, l_u)
    window = 2.0 * float(np.max(np.hypot(x[lit], y[lit])))
    max_grad = float(np.max(np.abs(g.gradient[lit])))
    q = q_for_tolerance(tolerance)
    fft_grid = _fft_grid(g, distance, max_grad, window, q)
    sft_grid: dict[str, float] = {"samples": math.inf}
    c_fit: FloatArray = np.zeros((2, 2))
    if f_max > 1.0:
        c_fit, res_grad = fit_curvature(g.points[lit], g.gradient[lit])
        if np.min(np.abs(np.linalg.eigvalsh(c_fit))) > 0.0:
            sft_grid = _sft_grid(g, distance, c_fit, res_grad, window, q)
    s = Situation(
        fresnel_min=f_min,
        fresnel_max=f_max,
        edge_ratio=edge,
        target_fresnel=target_fresnel,
        fresnel_number=f_max / (2.0 * math.pi),
        fft_samples=fft_grid["samples"],
        sft_samples=sft_grid["samples"],
    )
    chosen, reason = select_method(s, tolerance)
    if method is not None:
        chosen, reason = method, f"forced (rules: {chosen}; {reason})"
    interp = _interpolation_error(g, lit)
    taylor = _phase_taylor_error(g, lit)
    stop_ops: tuple[Operator, ...] = () if stop is None else (_bind(aperture, stop=stop),)
    edge_term = grid_edge_error(q) if stop is not None else 0.0
    notes: list[str] = []
    params: dict[str, float] = {"length_u_mm": l_u, "tolerance": tolerance}
    budget: dict[str, float] = {"interpolation": interp, "phase_taylor": taylor}
    ops: tuple[Operator, ...] = ()
    if chosen == "geometric":
        budget = {"stationary_phase": 1.0 / f_min, "edge_wave": edge}
        ops = (*stop_ops, _bind(geometric_propagation, distance=distance))
    elif chosen in ("sft", "fft"):
        grid = sft_grid if chosen == "sft" else fft_grid
        n, step = int(grid["samples"]), grid["step_mm"]
        params.update(grid)
        if chosen == "sft":
            ops = (
                _bind(resample, n=n, step=step, curvature=c_fit),
                *stop_ops,
                sft,
                _bind(angular_spectrum, dz=distance),
                inverse_sft,
            )
        else:
            ops = (
                _bind(resample, n=n, step=step),
                *stop_ops,
                fft,
                _bind(angular_spectrum, dz=distance),
                inverse_fft,
            )
        budget["hard_edge_grid"] = edge_term
    elif chosen == "debye":
        sin_max = a / math.hypot(a, distance)
        lam = 2.0 * math.pi / g.k
        window = 12.2 * lam / sin_max if focus_window is None else focus_window
        n_debye = focus_samples
        if n_debye is None:
            # period n·λ/(2 sin θ) (debye_focus) large enough for aliases below ε/10
            period = 0.5 * window
            while airy_alias_error(period, window, g.k, sin_max) > 0.1 * tolerance:
                period *= 1.25
            n_debye = max(32, _even(2.0 * sin_max * period / lam))
        period = n_debye * lam / (2.0 * sin_max)
        ops = (_bind(debye, distance=distance, sin_max=sin_max, n=n_debye),)
        budget.update(
            {
                "debye": debye_error(s.fresnel_number),
                "k_boundary": 0.02 / n_debye,
                "aliasing": airy_alias_error(period, window, g.k, sin_max),
            }
        )
        params.update({"samples": float(n_debye), "period_mm": period, "window_mm": window})
    elif chosen == "rayleigh_sommerfeld":
        step = fft_grid["step_mm"]
        n = _even(1.1 * window / step)
        sin_max = a / math.hypot(a, distance)
        step_out = 2.0 * math.pi / g.k / sin_max / 8.0  # λ/(8 sin θ): ~4 samples per Airy radius
        ops = (
            _bind(resample, n=n, step=step),
            *stop_ops,
            _bind(
                rayleigh_sommerfeld, distance=distance, n_out=focus_samples or 64, step_out=step_out
            ),
        )
        params.update({"step_mm": step, "samples": float(n), "step_out_mm": step_out})
        budget["hard_edge_grid"] = edge_term
    else:
        notes.append(f"method {chosen!r} has no operator chain yet")
    if chosen in ("sft", "fft") and isinstance(stop, CircularAperture):
        # Fresnel rings in the target plane are about sqrt(2|C'|) wide with the paraxial
        # spectral curvature C' = 1/|h| - d/k after the propagation (paper, section 7.5)
        h = float(np.max(eig))
        ring = (
            math.sqrt(2.0 * abs(1.0 / h - distance / g.k))
            if h > 0
            else math.sqrt(2.0 * distance / g.k)
        )
        params["axis_zone_mm"] = 3.0 * ring
        notes.append(
            "circular stop: the edge wave has a caustic on the axis (Poisson spot); within "
            f"{3.0 * ring:.3g} mm of the axis the grid edge error exceeds the budget"
        )
    return Plan(chosen, reason, s, budget, ops, tuple(notes), params)


def _target_fresnel(
    gradient: FloatArray, hessian: FloatArray, k: float, distance: float, l_u: float
) -> float:
    """Smallest local Fresnel number |h₂|·L_U²·|J| in the target plane (no division by J).

    With the ray map M = ∂ρ₂/∂ρ₁ = I + d·∂(p/k_z)/∂p·H₁ the new Hessian is H₂ = H₁M⁻¹, so
    H₂·det M = H₁·adj M stays finite at a focus; its smallest singular value times L_U² is the
    figure (the footprint of the beam scales with |J|, its curvature with H₂).
    """
    kz = np.sqrt(k * k - np.sum(gradient * gradient, axis=1))
    dir_jac = (
        np.eye(2)[None] / kz[:, None, None]
        + np.einsum("ni,nj->nij", gradient, gradient) / (kz**3)[:, None, None]
    )
    m = np.eye(2)[None] + distance * np.einsum("nij,njk->nik", dir_jac, hessian)
    adj = np.empty_like(m)
    adj[:, 0, 0], adj[:, 1, 1] = m[:, 1, 1], m[:, 0, 0]
    adj[:, 0, 1], adj[:, 1, 0] = -m[:, 0, 1], -m[:, 1, 0]
    sigma = np.linalg.svd(np.einsum("nij,njk->nik", hessian, adj), compute_uv=False)
    return float(np.min(sigma[:, -1])) * l_u**2


def _even(n: float) -> int:
    m = int(math.ceil(n))
    return m + (m % 2)


def _sft_grid(
    g: GridlessField, distance: float, c_fit: FloatArray, res_grad: float, window: float, q: float
) -> dict[str, float]:
    """Step and size of the SFT chain (paper, 7.5 step 3b).

    Step min(w_c/q, π/max|∇ψ_res|) (`selection.sft_step`); window ≥ 1.3 × illuminated width and
    ≥ 1.1 × 2π/(|c|Δx) for the chirps of the forward SFT (C) and of the inverse SFT
    (Q = Cᵀ(-C⁻¹ - d/k)C), so the same grid serves both. The chirp condition alone gives
    n ≈ π q²/m with the magnification m = 1 + d·c/k.
    """
    c_abs = np.abs(np.linalg.eigvalsh(c_fit))
    step = sft_step(float(np.max(c_abs)), q, res_grad)
    b = -np.linalg.inv(c_fit) - distance / g.k * np.eye(2)
    q_inv = c_fit.T @ b @ c_fit
    q_abs = np.abs(np.linalg.eigvalsh(0.5 * (q_inv + q_inv.T)))
    chirp = 2.0 * math.pi / (float(min(np.min(c_abs), np.min(q_abs))) * step)
    n = _even(max(1.3 * window, 1.1 * chirp) / step)
    eig = np.linalg.eigvalsh(c_fit)
    return {
        "samples": float(n),
        "step_mm": step,
        "q": q,
        "curvature_min_per_mm2": float(eig[0]),
        "curvature_max_per_mm2": float(eig[-1]),
        "residual_gradient_per_mm": res_grad,
        "memory_gb": 3 * 16 * float(n) ** 2 / 1e9,  # one (n, n, 3) complex128 array
    }


def _fft_grid(
    g: GridlessField, distance: float, max_grad: float, window: float, q: float
) -> dict[str, float]:
    """Step and size of the direct FFT chain: q samples per Fresnel width sqrt(2d/k) and twice
    the Nyquist rate of the full phase; window twice the illuminated width (periodic)."""
    w = math.sqrt(2.0 * distance / g.k)
    step = min(w / q, math.pi / max(max_grad, 1e-300) / 2.0)
    return {"samples": float(_even(2.0 * window / step)), "step_mm": step, "q": w / step}
