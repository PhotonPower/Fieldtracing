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

Warnings are collected in the log entry instead of being emitted.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from fieldtracing._types import ComplexArray, FloatArray
from fieldtracing.field import GriddedField, GridlessField
from fieldtracing.fourier import frequency_step, ft, ift
from fieldtracing.lpia import SagFunction, interface_amplitude, trace_surface
from fieldtracing.mesh import Region, fit_derivatives, mesh_jacobian_cloud
from fieldtracing.pft import inverse_pft_2d, pft_2d
from fieldtracing.propagation import angular_spectrum as angular_spectrum_grid
from fieldtracing.propagation import longitudinal_wavenumber
from fieldtracing.selection import edge_wave_ratio, fresnel_width, local_fresnel_number, sft_gain
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
