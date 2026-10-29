"""Issue #24: Raytatouille rays → GridlessField (`rtt_bridge.field_from_rays`, `trace_field`).

Marker `rtt` (Raytatouille's plano-convex singlet `m1/singlet_const`, field on z = 60 mm between
lens and focus):

- the same system recomputed in Fieldtracing with two LPIA steps (`chain.lpia_surface`, absolute
  indices from the model): landing points and phase to rounding, amplitude and Hessian to the
  mesh accuracy;
- phase against Raytatouille's `opd_points`: the OPL of every ray extended to Raytatouille's
  reference sphere gives its OPD;
- power: the field integrated over the output mesh (landing triangles, no Jacobian) equals the
  input power times Raytatouille's `polar.transmission`, also for an oblique field.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fieldtracing import chain
from fieldtracing._types import FloatArray
from fieldtracing.field import GridlessField
from fieldtracing.rtt_bridge import field_from_rays, trace_field
from fieldtracing.units import wavenumber

Z_PLANE = 60.0
RADIUS, Z_VERTEX, Z_BACK = 51.68, 5.0, 9.0


def test_plane_wave_without_surfaces() -> None:
    # straight rays in a medium: plane wave, J = 1, phase k₀·OPL
    n, lam = 1.33, 0.5
    k = np.array([0.0, np.sin(0.2), np.cos(0.2)])
    xs = np.linspace(-1.0, 1.0, 9)
    grid = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
    e1, e2 = np.array([1.0, 0.0, 0.0]), np.cross(k, [1.0, 0.0, 0.0])
    start = grid[:, :1] * e1 + grid[:, 1:] * e2  # on the wavefront plane through 0
    t = (5.0 - start[:, 2]) / k[2]
    pos = start + t[:, None] * k
    count = len(start)
    ks = np.tile(k, (count, 1))
    eye = np.tile(np.eye(3, dtype=np.complex128), (count, 1, 1))
    field, cloud = field_from_rays(
        start, ks, pos, ks, n * t, eye, np.ones(count), lam, n, n, 7.0, np.array([0, 1j, 0])
    )
    # wavefront plane → plane z = const: dA grows by 1/cos γ, the flux factor cos γ compensates
    assert np.allclose(cloud.determinant, 1.0 / k[2], atol=1e-12)
    expected = np.array([0, 1j * k[2], -1j * k[1]])  # (0, i, 0) projected ⟂ k
    assert np.allclose(field.amplitude, expected, atol=1e-12)
    k0 = wavenumber(lam)
    assert np.allclose(field.gradient, k0 * n * k[:2], rtol=1e-14)
    phase_plane = k0 * n * (field.points @ k[:2] + 7.0 * k[2])  # plane wave exp(i k·r)
    assert np.allclose(field.phase, phase_plane, rtol=0, atol=1e-8)
    assert np.allclose(field.hessian, 0.0, atol=1e-6)


# ----- against Raytatouille ---------------------------------------------------------------


def _singlet(reference: Path) -> Any:
    import raytatouille as rt

    return rt.compile(rt.load(reference / "m1" / "singlet_const.rtt.json"))


def _lpia_chain(system: Any, n: int) -> GridlessField:
    """The singlet with two `chain.lpia_surface` steps from Raytatouille's start rays (z = -1)."""
    import raytatouille as rt

    rays = rt.trace.make_rays(system, rt.trace.GridPupil(n), fields=[0])
    pts = np.stack([rays.pos_x, rays.pos_y], axis=1).copy()
    z0 = float(rays.pos_z[0])
    n_air = float(system.media[0].index[system.reference_wavelength].real)
    n_glass = float(system.media[1].index[system.reference_wavelength].real)
    amp = np.zeros((len(pts), 3), np.complex128)
    amp[:, 0] = 1.0
    zeros = np.zeros(len(pts))
    lam = float(system.wavelengths_um[system.reference_wavelength])
    start = GridlessField(
        pts, zeros, np.zeros((len(pts), 2)), np.zeros((len(pts), 2, 2)), amp, lam, n_air
    )
    centre = Z_VERTEX + RADIUS - z0  # local coordinates: start plane at z = 0

    def sag(x: FloatArray, y: FloatArray) -> FloatArray:
        return np.asarray(centre - np.sqrt(RADIUS**2 - x * x - y * y))

    def sag_grad(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        root = np.sqrt(RADIUS**2 - x * x - y * y)
        return np.asarray(x / root), np.asarray(y / root)

    flat: Callable[[FloatArray, FloatArray], FloatArray] = lambda x, y: 0.0 * x + 1.0  # noqa: E731
    z_mid = Z_BACK - 1.0 - z0  # a plane in the glass, 1 mm before the back surface
    inside, _ = chain.lpia_surface(start, n_glass, sag, sag_grad, z_mid)
    out, _ = chain.lpia_surface(
        inside, n_air, flat, lambda x, y: (0.0 * x, 0.0 * y), Z_PLANE - (Z_BACK - 1.0)
    )
    assert isinstance(out, GridlessField)
    return out


@pytest.mark.rtt
def test_adapter_matches_the_lpia_chain(rtt_reference_dir: Path) -> None:
    import raytatouille as rt

    system = _singlet(rtt_reference_dir)
    errors = []
    for n in (21, 41):
        field, cloud = trace_field(system, rt.trace.GridPupil(n), Z_PLANE)
        ref = _lpia_chain(system, n)
        assert len(field.points) == len(ref.points) and not np.any(cloud.caustic)
        assert np.max(np.abs(field.points - ref.points)) < 1e-12
        assert np.max(np.abs(field.phase - ref.phase)) < 1e-8  # phases ~10⁶ rad
        assert np.allclose(field.gradient, ref.gradient, rtol=1e-12)
        scale = float(np.max(np.abs(ref.amplitude)))
        diff = np.linalg.norm(field.amplitude - ref.amplitude, axis=1) / scale
        inner = np.hypot(field.points[:, 0], field.points[:, 1]) < 0.8 * np.max(
            np.hypot(field.points[:, 0], field.points[:, 1])
        )
        errors.append((float(np.max(diff)), float(np.max(diff[inner]))))
        assert field.transversality_error() < 1e-12
    assert errors[1][0] < 2e-4  # both are mesh estimates of J: one map vs. two steps
    assert errors[0][1] / errors[1][1] > 2.5  # O(h²) inside


@pytest.mark.rtt
def test_phase_matches_raytatouille_opd(rtt_reference_dir: Path) -> None:
    import raytatouille as rt

    system = _singlet(rtt_reference_dir)
    pupil = rt.trace.GridPupil(21)
    field, _ = trace_field(system, pupil, Z_PLANE)
    rays = rt.trace.make_rays(system, pupil, fields=[0])
    px, py = np.asarray(rays.pupil_x), np.asarray(rays.pupil_y)
    opd = rt.analysis.opd_points(system, field=0, rays=rt.trace.PointListPupil(px, py))
    centre, radius = np.asarray(opd.sphere.centre), float(opd.sphere.radius)
    lam = float(system.wavelengths_um[system.reference_wavelength])
    k0 = wavenumber(lam)
    n_air = field.refractive_index
    # extend each ray backwards from z = 60 mm to the reference sphere (image space, straight)
    k_perp = field.gradient / (k0 * n_air)
    k_hat = np.column_stack([k_perp, np.sqrt(1.0 - np.sum(k_perp**2, axis=1))])
    p = np.column_stack([field.points, np.full(len(field.points), Z_PLANE)]) - centre
    b = np.sum(k_hat * p, axis=1)
    s = -b - np.sqrt(b * b - (np.sum(p * p, axis=1) - radius**2))
    opl_sphere = field.phase / k0 + n_air * s
    chief = int(np.argmin(px**2 + py**2))
    w = (opl_sphere[chief] - opl_sphere) / (lam * 1e-3)  # W = OPL_ref - OPL in waves
    assert np.max(np.abs(np.asarray(opd.points.w))) > 1.0  # several waves of aberration
    assert np.max(np.abs(w - np.asarray(opd.points.w))) < 1e-8


@pytest.mark.rtt
@pytest.mark.parametrize("field_index", [0, 2])
def test_power_is_conserved(rtt_reference_dir: Path, field_index: int) -> None:
    import raytatouille as rt

    system = _singlet(rtt_reference_dir)
    pupil = rt.trace.GridPupil(41)
    field, cloud = trace_field(system, pupil, Z_PLANE, field=field_index)
    rays = rt.trace.make_rays(system, pupil, fields=[field_index])
    k0 = np.stack([rays.dir_x, rays.dir_y, rays.dir_z], axis=1)[0]
    start = np.stack([rays.pos_x, rays.pos_y, rays.pos_z], axis=1).copy()
    rt.trace.trace(system, rays)
    assert np.all(np.asarray(rays.status) == int(rt.trace.RayStatus.ALIVE))
    jones = np.cross(k0, np.cross([1.0, 0.0, 0.0], k0)).astype(np.complex128)  # x̂ ⟂ k₀
    jones /= np.linalg.norm(jones)
    transmission = rt.polar.transmission(rays, np.tile(jones, (len(rays), 1)))
    tri = cloud.triangles
    n_in = n_out = field.refractive_index
    # input power per triangle: n|U_in|² dA in the wavefront plane, times the mean transmission
    a, b, c = start[tri[:, 0]], start[tri[:, 1]], start[tri[:, 2]]
    area_in = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    expected = float(np.sum(n_in * area_in * transmission[tri].mean(axis=1)))
    # output power from the field on the landing triangles (geometry only, no J)
    k_z = np.sqrt(
        1.0 - np.sum((field.gradient / (wavenumber(field.wavelength_um) * n_out)) ** 2, axis=1)
    )
    density = n_out * k_z * np.sum(np.abs(field.amplitude) ** 2, axis=1)
    pa, pb, pc = field.points[tri[:, 0]], field.points[tri[:, 1]], field.points[tri[:, 2]]
    u, v = pb - pa, pc - pa
    area_out = 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    got = float(np.sum(area_out * density[tri].mean(axis=1)))
    assert got == pytest.approx(expected, rel=3e-4)  # vertex-mean quadrature on both sides: O(h²)
