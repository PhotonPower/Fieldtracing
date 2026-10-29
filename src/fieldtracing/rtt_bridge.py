"""Bridge from Raytatouille ray data to field amplitudes (F4; ADR 0004).

Raytatouille traces with a POWER-NORMALISED polarization ray-tracing matrix P and a weight
(Raytatouille ADR 0021): every interface contributes P_q = √c·T_q on the transverse plane, with
the field transmission operator T_q = t_s s sᵀ + t_p p_out p_inᵀ (paper, eq. fresnel_operator,
Convention A) and the power factor c = n₂ cos θ_t/(n₁ cos θ_i) (Byrnes, arXiv:1603.02720v5,
eq. (21), (22), real indices), and P k₀ = k. The weight of an unpolarized unit source is
weight = s·½‖P_T‖²_F with P_T = (I - k kᵀ)P and s the polarization-independent factors (volume
absorption, absorbers, diffraction efficiency).

Field amplitudes on planes z = const follow from flux conservation n|U|²cos γ dA along a ray tube
(paper, eq. interface_amplitude): over any sequence of interfaces

    U_out = sqrt(s·n_in cos γ_in dA_in / (n_out cos γ_out dA_out))·exp(-iπμ/2)·P_T·U_in,

because the factors sqrt(cos θ_t/cos θ_i) of eq. interface_amplitude and the √c in P differ only
by sqrt(n₁/n₂) per interface, which telescopes to sqrt(n_in/n_out). γ is the angle between the ray
and the plane normal (+z), dA_out/dA_in = |J| the area ratio of the ray tube (`mesh`), μ the
Maslov index. Indices are absolute (Raytatouille: AIR = Ciddor air).

`field_from_rays` builds a `GridlessField` on a plane z = const from traced ray data (arrays);
`trace_field` does the tracing with Raytatouille (imported only there, lazily).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fieldtracing._types import ComplexArray, FloatArray, IntArray
from fieldtracing.field import GriddedField, GridlessField
from fieldtracing.mesh import CloudJacobian, Region, fit_derivatives, mesh_jacobian_cloud
from fieldtracing.units import wavenumber


def transverse_prt(prt: ComplexArray, direction_out: FloatArray) -> ComplexArray:
    """P_T = (I - k kᵀ)P (N, 3, 3) for final unit directions k (N, 3) (Raytatouille ADR 0021)."""
    k = np.asarray(direction_out, dtype=np.float64)
    proj = np.eye(3)[None] - np.einsum("ni,nj->nij", k, k)
    return np.asarray(np.einsum("nij,njk->nik", proj, prt))


def absorption_factor(
    prt: ComplexArray, weight: FloatArray, direction_out: FloatArray
) -> FloatArray:
    """s = weight/(½‖P_T‖²_F): the polarization-independent power factor of each ray (≤ 1).

    Rays with P_T = 0 (fully blocked) get s = 0.
    """
    norm = 0.5 * np.sum(np.abs(transverse_prt(prt, direction_out)) ** 2, axis=(1, 2))
    w = np.asarray(weight, dtype=np.float64)
    return np.asarray(np.where(norm > 0.0, w / np.where(norm > 0.0, norm, 1.0), 0.0))


def field_operator(
    prt: ComplexArray,
    weight: FloatArray,
    direction_in: FloatArray,
    direction_out: FloatArray,
    n_in: float,
    n_out: float,
    area_ratio: FloatArray | float = 1.0,
    maslov: IntArray | int = 0,
    normal_in: FloatArray | None = None,
    normal_out: FloatArray | None = None,
) -> ComplexArray:
    """Field amplitude operator M (N, 3, 3) with U_out = M·U_in on planes z = const (ADR 0004).

    M = sqrt(s·n_in cos γ_in/(n_out cos γ_out |J|))·exp(-iπμ/2)·P_T.

    Args:
        prt: (N, 3, 3) power-normalised P of the rays (Raytatouille `RayBatch.prt_matrices()`).
        weight: (N,) weight of the rays (unpolarized unit source).
        direction_in, direction_out: (N, 3) unit ray directions at the input and output plane
            (cos γ is their z-component).
        n_in, n_out: absolute refractive indices of the input and output media.
        area_ratio: J = dA_out/dA_in of the ray tube on the two planes (signed or absolute).
        maslov: Maslov index μ of the caustics between the planes.
        normal_in, normal_out: unit normals (3,) of the input and output planes (default +z);
            cos γ = |k·normal|.

    Returns:
        M; for a transverse input U_in ⟂ direction_in the output is transverse to direction_out.
    """
    p_t = transverse_prt(prt, direction_out)
    s = absorption_factor(prt, weight, direction_out)
    z = np.array([0.0, 0.0, 1.0])
    n_i = z if normal_in is None else np.asarray(normal_in, dtype=np.float64)
    n_o = z if normal_out is None else np.asarray(normal_out, dtype=np.float64)
    cos_in = np.abs(np.asarray(direction_in, dtype=np.float64) @ n_i)
    cos_out = np.abs(np.asarray(direction_out, dtype=np.float64) @ n_o)
    scale = np.sqrt(s * n_in * cos_in / (n_out * cos_out * np.abs(area_ratio)))
    phase = np.exp(-0.5j * np.pi * np.asarray(maslov))
    return np.asarray((scale * phase)[..., None, None] * p_t)


def _plane_basis(normal: FloatArray) -> FloatArray:
    """Orthonormal (e1, e2) (2, 3) of the plane ⟂ normal; e1 = x̂ projected (y for normal ∥ x)."""
    n = normal / np.linalg.norm(normal)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = seed - n * (seed @ n)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(n, e1)])


def field_from_rays(
    start: FloatArray,
    direction_in: FloatArray,
    position: FloatArray,
    direction_out: FloatArray,
    opl: FloatArray,
    prt: ComplexArray,
    weight: FloatArray,
    wavelength_um: float,
    n_in: float,
    n_out: float,
    z_plane: float,
    jones: ComplexArray,
    amplitude_in: float = 1.0,
    max_edge: float | None = None,
    region: Region | None = None,
) -> tuple[GridlessField, CloudJacobian]:
    """Hybrid field on the plane z = z_plane from traced rays of a plane wave (paper, 1.3, 3.4).

    The rays start on the input wavefront, a plane ⟂ k₀ through the start points where the OPL
    is zero (Raytatouille's start rays of a field at infinity), with field U_in = amplitude_in·
    jones. Each ray is extended in a straight line (image-space medium n_out) to z = z_plane:

    - phase ψ = k₀·OPL (k₀ = 2π/λ_vac, OPL with absolute indices, mm),
    - gradient ∇ψ = k₀ n_out k̂⊥ (1/mm),
    - amplitude U = M·U_in with `field_operator` (ADR 0004); J and μ from the mesh of the start
      points (`mesh.mesh_jacobian_cloud`, coordinates in the input wavefront plane),
    - Hessian H = (∂∇ψ/∂ρ₁)(∂ρ₂/∂ρ₁)⁻¹, both fitted on the mesh (1/mm²).

    Args:
        start, direction_in: (N, 3) start points (mm) and unit directions before the trace.
        position, direction_out, opl: (N, 3), (N, 3), (N,) after the trace (any surface in the
            final medium).
        prt, weight: (N, 3, 3) power-normalised P and (N,) weight (Raytatouille ADR 0021).
        wavelength_um: vacuum wavelength in µm; n_in, n_out: absolute indices of object and
            image space.
        z_plane: output plane in mm (straight extension in the image space, also backwards).
        jones: (3,) complex polarization of the input wave, transverse to k₀ (normalised here).
        amplitude_in: field amplitude of the input wave (same unit as the output).
        max_edge, region: mesh options in the input wavefront plane (obscurations).

    Returns:
        (GridlessField in the space domain on z = z_plane, CloudJacobian of the ray map).
    """
    k_in = np.asarray(direction_in, dtype=np.float64)
    normal = np.mean(k_in, axis=0)
    normal /= np.linalg.norm(normal)
    if np.max(np.linalg.norm(k_in - normal, axis=1)) > 1e-9:
        raise ValueError("field_from_rays needs a plane input wave (equal start directions)")
    pol = np.asarray(jones, dtype=np.complex128)
    pol = pol - normal * (normal @ pol)
    pol = pol / np.linalg.norm(pol)
    pos = np.asarray(position, dtype=np.float64)
    k_out = np.asarray(direction_out, dtype=np.float64)
    t = (z_plane - pos[:, 2]) / k_out[:, 2]
    landing = pos[:, :2] + t[:, None] * k_out[:, :2]
    path = np.asarray(opl, dtype=np.float64) + n_out * t
    basis = _plane_basis(normal)
    start_2d = np.asarray(start, dtype=np.float64) @ basis.T
    cloud = mesh_jacobian_cloud(start_2d, landing, max_edge=max_edge, region=region)
    op = field_operator(
        prt, weight, k_in, k_out, n_in, n_out, cloud.determinant, cloud.maslov, normal_in=normal
    )
    amplitude = amplitude_in * np.einsum("nij,j->ni", op, pol)
    k0 = wavenumber(wavelength_um)
    gradient = k0 * n_out * k_out[:, :2]
    d_grad, _ = fit_derivatives(start_2d, gradient, cloud.triangles)
    inv = np.full_like(cloud.matrix, np.nan)
    ok = cloud.valid & (np.abs(np.nan_to_num(cloud.determinant)) > 0.0)
    inv[ok] = np.linalg.inv(cloud.matrix[ok])
    hessian = np.einsum("nij,njk->nik", d_grad, inv)
    out = GridlessField(
        landing, k0 * path, gradient, hessian, np.asarray(amplitude), wavelength_um, n_out
    )
    return out, cloud


def trace_field(
    system: Any,
    pupil: Any,
    z_plane: float,
    field: int = 0,
    jones: ComplexArray | None = None,
    path: int | str = 0,
    max_edge: float | None = None,
    region: Region | None = None,
) -> tuple[GridlessField, CloudJacobian]:
    """Trace a Raytatouille system and return the field on z = z_plane (`field_from_rays`).

    Rays of field index `field` at the reference wavelength for the pupil sampling `pupil`
    (e.g. `raytatouille.trace.GridPupil(n)` or `PointListPupil`); rays that are not alive or
    carry weight 0 (vignetted, failed) are dropped. Object and image indices come from
    `raytatouille.paraxial.first_order` (absolute). Input polarization `jones` (default x̂,
    projected ⟂ k₀), amplitude 1.

    Args:
        system: `raytatouille.CompiledSystem`.
    """
    import raytatouille as rt

    rays = rt.trace.make_rays(system, pupil, path=path, fields=[field])
    start = np.stack([rays.pos_x, rays.pos_y, rays.pos_z], axis=1).copy()
    k_in = np.stack([rays.dir_x, rays.dir_y, rays.dir_z], axis=1).copy()
    rt.trace.trace(system, rays, path=path)
    keep = (np.asarray(rays.status) == int(rt.trace.RayStatus.ALIVE)) & (rays.weight > 0.0)
    first = rt.paraxial.first_order(system, path=path)
    wavelength = float(system.wavelengths_um[system.reference_wavelength])
    pos = np.stack([rays.pos_x, rays.pos_y, rays.pos_z], axis=1)
    k_out = np.stack([rays.dir_x, rays.dir_y, rays.dir_z], axis=1)
    pol = np.array([1.0, 0.0, 0.0], dtype=np.complex128) if jones is None else jones
    return field_from_rays(
        start[keep],
        k_in[keep],
        pos[keep],
        k_out[keep],
        np.asarray(rays.opl)[keep],
        rays.prt_matrices()[keep],
        np.asarray(rays.weight)[keep],
        wavelength,
        float(first.object_index),
        float(first.image_index),
        z_plane,
        pol,
        max_edge=max_edge,
        region=region,
    )


def debye_psf(
    system: Any,
    exit_z: float,
    rings: int = 40,
    n: int = 128,
    pad: int = 4,
    jones: ComplexArray | None = None,
    path: int | str = 0,
) -> tuple[GriddedField, float]:
    """Vector PSF of the axial field in the image plane by the Debye integral (paper, eq. debye).

    Rays of `raytatouille.trace.HexapolarPupil(rings)` (rim included) give the field on the exit
    plane z = exit_z in image space (`trace_field`); `chain.debye` integrates over the disc
    |k⊥| ≤ k·sin θ_max onto the image plane (z of the chief ray on the image surface), with the
    reference sphere centred there. The Strehl ratio divides by the same integral with the
    aberration removed (spherical phase, same amplitude). Axial field only (sphere on the axis).

    Returns:
        (field (N·pad)² × 3 in the image plane, Strehl ratio from |E|² at the centre).
    """
    import raytatouille as rt

    from fieldtracing.chain import debye

    chief = rt.trace.make_rays(system, rt.trace.SinglePupilPoint(), path=path, fields=[0])
    rt.trace.trace(system, chief, path=path)
    z_image = float(chief.pos_z[0])
    exit_field, _ = trace_field(
        system, rt.trace.HexapolarPupil(rings), exit_z, jones=jones, path=path
    )
    f = z_image - exit_z
    p = exit_field.points
    a = float(np.max(np.hypot(p[:, 0], p[:, 1])))
    sin_max = a / float(np.hypot(a, f))
    k = exit_field.k
    focal, _ = debye(exit_field, f, sin_max, n=n, pad=pad)
    r = np.sqrt(f * f + np.sum(p * p, axis=1))
    outer = np.einsum("ni,nj->nij", p, p)
    sphere = GridlessField(
        p,
        -k * (r - f),
        -k * p / r[:, None],
        -k * (np.eye(2)[None] / r[:, None, None] - outer / (r**3)[:, None, None]),
        exit_field.amplitude,
        exit_field.wavelength_um,
        exit_field.refractive_index,
    )
    perfect, _ = debye(sphere, f, sin_max, n=n, pad=pad)
    assert isinstance(focal, GriddedField) and isinstance(perfect, GriddedField)
    c = focal.values.shape[0] // 2
    strehl = float(
        np.sum(np.abs(focal.values[c, c]) ** 2) / np.sum(np.abs(perfect.values[c, c]) ** 2)
    )
    return focal, strehl
