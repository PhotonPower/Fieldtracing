"""Issue #19: operators with a common interface on the hybrid representation (eq. system_chain).

Regression: every operator gives the result of its F1 building block. Exact operators (angular
spectrum, SFT and FFT round trips) are invertible to rounding. Independent cross-checks:
PFT → angular spectrum → inverse PFT equals the geometric propagation (`propagate_2d`), and
SFT → angular spectrum → inverse SFT equals `propagation.angular_spectrum` on a fine grid.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
import pytest

from fieldtracing import chain
from fieldtracing._types import ComplexArray, FloatArray
from fieldtracing.field import GriddedField, GridlessField
from fieldtracing.fourier import centered_grid, ft
from fieldtracing.lpia import (
    fresnel_coefficients,
    interface_amplitude,
    mesh_jacobian,
    trace_surface,
)
from fieldtracing.pft import inverse_pft_2d, pft_2d
from fieldtracing.propagation import angular_spectrum
from fieldtracing.semiclassical import propagate_2d
from fieldtracing.sft import circular_aperture_weights, sft_2d
from fieldtracing.units import wavenumber

LAMBDA_UM = 0.5
K = wavenumber(LAMBDA_UM)


def _grid_points(half: float, n: int) -> FloatArray:
    x = np.linspace(-half, half, n)
    return np.stack(np.meshgrid(x, x), axis=-1).reshape(-1, 2)


def _transversal(points: FloatArray, gradient: FloatArray, k: float = K) -> ComplexArray:
    """Gaussian envelope times a unit vector perpendicular to the local wave vector."""
    kz = np.sqrt(k * k - np.sum(gradient * gradient, axis=1))
    k_hat = np.concatenate([gradient, kz[:, None]], axis=1) / k
    pol = np.cross(k_hat, np.array([1.0, 0.0, 0.0]))
    pol /= np.linalg.norm(pol, axis=1, keepdims=True)
    env = np.exp(-np.sum(points**2, axis=1) / 4.0)
    return np.asarray(env[:, None] * pol, dtype=np.complex128)


def _sphere(n: int = 21, half: float = 2.0, f: float = 25.0) -> GridlessField:
    pts = _grid_points(half, n)
    probe = GridlessField.spherical(pts, np.zeros((len(pts), 3), np.complex128), f, LAMBDA_UM)
    return GridlessField.spherical(pts, _transversal(pts, probe.gradient), f, LAMBDA_UM)


# ----- regression against the F1 building blocks --------------------------------------------


def test_geometric_propagation_matches_propagate_2d() -> None:
    field = _sphere()
    out, entry = chain.geometric_propagation(field, 10.0, length_u=2.0)
    ref = propagate_2d(
        field.points, field.amplitude, field.phase, field.gradient, field.hessian, K, 10.0
    )
    assert isinstance(out, GridlessField) and ref.curvature is not None
    assert np.array_equal(out.points, ref.points) and np.array_equal(out.amplitude, ref.amplitude)
    assert np.array_equal(out.hessian, ref.curvature)
    assert entry.operator == "geometric_propagation" and entry.warnings == ()
    eigen = np.abs(np.linalg.eigvalsh(field.hessian))  # k f²/r³ (radial) … k/f (centre)
    assert entry.metrics["fresnel_number_min"] == pytest.approx(np.min(eigen) * 4.0, rel=1e-12)
    assert entry.metrics["fresnel_width_min_mm"] == pytest.approx(np.max(eigen) ** -0.5)
    # behind the focus: Maslov index 2 in the log
    _, behind = chain.geometric_propagation(field, 40.0)
    assert behind.metrics["maslov_max"] == 2.0


def test_geometric_propagation_from_the_mesh_agrees_to_second_order() -> None:
    # quadratic phase with a tilted, astigmatic C: the ray mapping is non-linear (exact k_z);
    # for a spherical wave it would be a linear scaling and the mesh trivially exact
    c = np.array([[-400.0, 60.0], [60.0, -250.0]])
    errors = []
    for n in (21, 41):
        pts = _grid_points(2.0, n)
        field = GridlessField.quadratic(pts, _transversal(pts, pts @ c), c, LAMBDA_UM)
        exact, _ = chain.geometric_propagation(field, 10.0)
        mesh, entry = chain.geometric_propagation(field, 10.0, jacobian="mesh")
        assert isinstance(exact, GridlessField) and isinstance(mesh, GridlessField)
        scale = np.max(np.abs(exact.amplitude))
        errors.append(float(np.max(np.abs(mesh.amplitude - exact.amplitude)) / scale))
        rel_h = np.max(np.abs(mesh.hessian - exact.hessian)) / np.max(np.abs(exact.hessian))
        errors.append(float(rel_h))
        assert entry.metrics["caustic_nodes"] == 0.0
    assert errors[0] < 1e-4 and errors[1] < 1e-4  # amplitude, Hessian (21² rays)
    assert 3.0 < errors[0] / errors[2] < 5.5 and 3.0 < errors[1] / errors[3] < 5.5


def test_pft_and_inverse_match_f1_and_collect_warnings() -> None:
    field = _sphere(11)
    spec, entry = chain.pft(field, length_u=2.0)
    k, res, phase = pft_2d(
        field.points, field.amplitude, field.phase, field.gradient, field.hessian
    )
    assert isinstance(spec, GridlessField) and spec.domain == "frequency"
    assert np.array_equal(spec.points, k) and np.array_equal(spec.amplitude, res)
    assert np.array_equal(spec.phase, phase) and entry.warnings == ()
    back, _ = chain.inverse_pft(spec)
    rho, u, psi = inverse_pft_2d(
        spec.points, spec.amplitude, spec.phase, spec.gradient, spec.hessian
    )
    assert isinstance(back, GridlessField)
    assert np.array_equal(back.points, rho) and np.array_equal(back.amplitude, u)
    assert np.max(np.abs(back.amplitude - field.amplitude)) < 1e-12
    _, weak = chain.pft(field, length_u=0.01)  # F = |h| L_U² ≈ 0.05
    assert len(weak.warnings) == 1 and "Fresnel number" in weak.warnings[0]


def test_sft_matches_sft_2d_and_round_trip_is_exact() -> None:
    n, step = 128, 0.01
    c = np.array([[-300.0, 80.0], [80.0, -200.0]])
    x = centered_grid(n, step)
    xx, yy = np.meshgrid(x, x)
    v = np.exp(-(xx**2 + 2 * yy**2) / 0.08)
    values = np.stack([v, 0.5j * v, 0 * v], axis=-1).astype(np.complex128)
    grid = GriddedField(step, values, LAMBDA_UM, curvature=c)
    spec, entry = chain.sft(grid)
    _, ref, _ = sft_2d(v.astype(np.complex128), step, c, warn=False)
    assert isinstance(spec, GriddedField) and spec.domain == "frequency"
    assert np.array_equal(spec.values[..., 0], ref)
    assert np.allclose(
        spec.sample_points(), sft_2d(v.astype(np.complex128), step, c, warn=False)[0]
    )
    assert entry.metrics["samples_per_w_c"] > 4.0
    back, _ = chain.inverse_sft(spec)
    assert isinstance(back, GriddedField) and back.plain
    assert np.allclose(back.curvature, c)
    assert np.max(np.abs(back.values - values)) < 1e-10
    # sampling violations end up in the log
    _, coarse = chain.sft(GriddedField(0.05, values[::4, ::4], LAMBDA_UM, curvature=c * 20))
    assert any("edge resolution" in w for w in coarse.warnings)


def test_fft_matches_ft_and_round_trip_is_exact() -> None:
    n, step = 64, 0.02
    rng = np.random.default_rng(4)
    values = rng.normal(size=(n, n, 3)) + 1j * rng.normal(size=(n, n, 3))
    grid = GriddedField(step, values, LAMBDA_UM)
    spec, _ = chain.fft(grid)
    assert isinstance(spec, GriddedField)
    assert np.allclose(spec.values, ft(values, step, axes=(0, 1)), rtol=0, atol=1e-15)
    back, _ = chain.inverse_fft(spec)
    assert isinstance(back, GriddedField)
    assert back.step == pytest.approx(step) and np.max(np.abs(back.values - values)) < 1e-12


def test_aperture_on_cloud_and_grid() -> None:
    field = _sphere(41)
    stop = chain.CircularAperture(1.45, inner_radius=0.27)
    out, entry = chain.aperture(field, stop)
    assert isinstance(out, GridlessField)
    r = np.hypot(field.points[:, 0], field.points[:, 1])
    inside = (r <= 1.45) & (r >= 0.27)
    assert np.array_equal(out.amplitude[inside], field.amplitude[inside])
    assert np.all(out.amplitude[~inside] == 0.0) and np.array_equal(out.phase, field.phase)
    # edge-wave ratio (eq. edge_ratio) at the transmitted sample closest to an edge
    d_e = np.minimum(1.45 - r[inside], r[inside] - 0.27)
    h_min = np.min(np.abs(np.linalg.eigvalsh(field.hessian[inside])), axis=1)
    expected = np.max(h_min**-0.5 / (np.sqrt(2 * np.pi) * d_e))
    assert entry.metrics["edge_wave_ratio_max"] == pytest.approx(expected, rel=1e-12)
    assert entry.warnings  # samples within w/(sqrt(2π)·1 %) of the edge
    x = centered_grid(64, 0.05)
    grid = GriddedField(0.05, np.ones((64, 64, 3), np.complex128), LAMBDA_UM)
    stopped, _ = chain.aperture(grid, chain.CircularAperture(1.0))
    assert isinstance(stopped, GriddedField)
    assert np.allclose(stopped.values[..., 1], circular_aperture_weights(x, x, 1.0))
    rect = chain.RectangularAperture(1.0, 0.5)
    assert rect.edge_distance(np.array([0.2, 2.0]), np.array([0.1, 1.5])) == pytest.approx(
        [0.4, np.hypot(1.0, 1.0)]
    )


def test_lpia_on_the_cloud_matches_the_grid_version() -> None:
    # converging wave through a sphere about its focus (normal incidence, test_lpia_surface.py)
    z_c, radius, z_out, n2 = 30.0, 20.0, 20.0, 1.5
    half, n = 3.0, 41

    def sag(x: FloatArray, y: FloatArray) -> FloatArray:
        return np.asarray(z_c - np.sqrt(radius**2 - x * x - y * y))

    def sag_grad(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        root = np.sqrt(radius**2 - x * x - y * y)
        return np.asarray(x / root), np.asarray(y / root)

    pts = _grid_points(half, n)
    probe = GridlessField.spherical(pts, np.zeros((len(pts), 3), np.complex128), z_c, LAMBDA_UM)
    field = GridlessField.spherical(pts, _transversal(pts, probe.gradient), z_c, LAMBDA_UM)
    out, entry = chain.lpia_surface(field, n2, sag, sag_grad, z_out)
    assert isinstance(out, GridlessField) and out.refractive_index == n2
    # F1 chain on the regular grid with lpia.mesh_jacobian
    grid = pts.reshape(n, n, 2)
    trace = trace_surface(
        grid,
        field.amplitude.reshape(n, n, 3),
        field.phase.reshape(n, n),
        field.gradient.reshape(n, n, 2),
        1.0,
        n2,
        K,
        sag,
        sag_grad,
        z_out,
    )
    ref = interface_amplitude(trace, mesh_jacobian(grid, trace.points)).reshape(-1, 3)
    assert np.max(np.abs(out.amplitude - ref)) < 1e-12  # radial scaling: both meshes exact
    assert np.allclose(out.phase, trace.phase.reshape(-1), rtol=1e-14)
    # analytic: |E₂| = |t|·|E₁|·r₁/r₂, still a sphere about the focus in medium n₂
    r1 = np.sqrt(z_c**2 + np.sum(pts**2, axis=1))
    r2 = np.sqrt((z_c - z_out) ** 2 + np.sum(out.points**2, axis=1))
    t = abs(fresnel_coefficients(1.0, n2, 1.0).t_s)
    expected = t * np.linalg.norm(field.amplitude, axis=1) * r1 / r2
    assert np.max(np.abs(np.linalg.norm(out.amplitude, axis=1) - expected)) < 1e-12
    sphere = GridlessField.spherical(out.points, out.amplitude, z_c - z_out, LAMBDA_UM, n2)
    assert np.allclose(out.gradient, sphere.gradient, rtol=1e-12)
    rel = np.max(np.abs(out.hessian - sphere.hessian)) / np.max(np.abs(sphere.hessian))
    assert rel < 1e-4  # fitted ∂k⊥/∂ρ₁: O(h²)
    assert out.transversality_error() < 1e-12
    assert entry.metrics["wavefront_radius_min_mm"] == pytest.approx(z_c - z_out, rel=1e-3)


# ----- exact operators are invertible ----------------------------------------------------------


def test_angular_spectrum_is_invertible_on_cloud_and_grid() -> None:
    spec, _ = chain.pft(_sphere(11))
    there, _ = chain.angular_spectrum(spec, 7.0)
    back, _ = chain.angular_spectrum(there, -7.0)
    assert isinstance(back, GridlessField) and isinstance(spec, GridlessField)
    assert np.allclose(back.phase, spec.phase, rtol=0, atol=1e-9)
    assert np.allclose(back.hessian, spec.hessian, rtol=1e-12)
    assert np.array_equal(back.amplitude, spec.amplitude)
    rng = np.random.default_rng(2)
    values = rng.normal(size=(32, 32, 3)) + 0j
    for curvature in (np.zeros((2, 2)), np.diag([-2e-3, -3e-3])):
        grid = GriddedField(30.0, values, LAMBDA_UM, curvature=curvature, domain="frequency")
        moved, _ = chain.angular_spectrum(grid, 5.0)
        again, _ = chain.angular_spectrum(moved, -5.0)
        assert isinstance(again, GriddedField)
        assert np.max(np.abs(again.values - values)) < 1e-12
        assert np.allclose(again.curvature, curvature, rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="transform first"):
        chain.angular_spectrum(_sphere(5), 1.0)


def test_fft_angular_spectrum_chain_equals_direct_propagation() -> None:
    n, step, dz = 128, 0.004, 0.3
    x = centered_grid(n, step)
    xx, yy = np.meshgrid(x, x)
    v = np.exp(-(xx**2 + yy**2) / 0.01) * np.exp(1j * 900.0 * xx)
    values = np.stack([v, 0 * v, 0.3 * v], axis=-1)
    start = GriddedField(step, values, LAMBDA_UM)
    out, log = chain.run_chain(
        start, [chain.fft, partial(chain.angular_spectrum, dz=dz), chain.inverse_fft]
    )
    assert [e.operator for e in log] == ["fft", "angular_spectrum", "inverse_fft"]
    ref = angular_spectrum(v, step, K, dz)
    assert isinstance(out, GriddedField)
    assert np.max(np.abs(out.values[..., 0] - ref)) < 1e-12
    direct, _ = chain.angular_spectrum(start, dz)
    assert isinstance(direct, GriddedField)
    assert np.max(np.abs(direct.values[..., 2] - 0.3 * ref)) < 1e-12


# ----- independent cross-checks of the bookkeeping ---------------------------------------------


def _tilted(n: int = 11) -> GridlessField:
    c = np.array([[-400.0, 60.0], [60.0, -250.0]])  # line foci near 25 and 50 mm, tilted
    pts = _grid_points(2.0, n)
    return GridlessField.quadratic(pts, _transversal(pts, pts @ c), c, LAMBDA_UM)


@pytest.mark.parametrize(
    ("make", "d"), [(_sphere, 10.0), (_sphere, 40.0), (_tilted, 10.0), (_tilted, 40.0)]
)
def test_pft_angular_spectrum_inverse_pft_is_geometric_propagation(
    make: Callable[[], GridlessField], d: float
) -> None:
    # Algebraic identity (Legendre transform): the chain reproduces eq. jacobian_amplitude,
    # including the Maslov phase behind the foci (d = 40 mm: μ = 2 for the sphere, 1 for the
    # tilted astigmatic wave).
    field = make()
    out, _ = chain.run_chain(
        field, [chain.pft, partial(chain.angular_spectrum, dz=d), chain.inverse_pft]
    )
    ref = propagate_2d(
        field.points, field.amplitude, field.phase, field.gradient, field.hessian, K, d
    )
    assert isinstance(out, GridlessField)
    assert np.allclose(out.points, ref.points, rtol=0, atol=1e-12)
    assert np.max(np.abs(out.amplitude - ref.amplitude)) < 1e-10
    assert np.allclose(out.phase, ref.phase, rtol=1e-12)
    assert ref.curvature is not None and np.allclose(out.hessian, ref.curvature, rtol=1e-9)


def test_sft_angular_spectrum_inverse_sft_equals_angular_spectrum() -> None:
    # Gaussian × converging chirp (f = 50 mm) propagated halfway to the focus: the output lattice
    # ρ = (1 + dz·c/k)·u' = u'/2 coincides with a reference grid of half the step.
    f, n, step = 50.0, 320, 0.0125
    c = -K / f
    dz = f / 2.0
    x = centered_grid(n, step)
    xx, yy = np.meshgrid(x, x)
    v = np.exp(-(xx**2 + yy**2) / 0.25).astype(np.complex128)
    values = np.stack([v, 0 * v, 0 * v], axis=-1)
    start = GriddedField(step, values, LAMBDA_UM, curvature=c * np.eye(2))
    out, log = chain.run_chain(
        start, [chain.sft, partial(chain.angular_spectrum, dz=dz), chain.inverse_sft]
    )
    assert all(not e.warnings for e in log)
    assert isinstance(out, GriddedField) and np.allclose(out.lattice, 0.5 * np.eye(2))
    field = out.physical()[..., 0]
    # reference: plain angular spectrum on the half-step grid
    xr = centered_grid(2 * n, step / 2)
    xr2, yr2 = np.meshgrid(xr, xr)
    phys = np.exp(-(xr2**2 + yr2**2) / 0.25) * np.exp(0.5j * c * (xr2**2 + yr2**2))
    ref = angular_spectrum(phys, step / 2, K, dz)[n // 2 : n // 2 + n, n // 2 : n // 2 + n]
    assert np.allclose(out.sample_points()[..., 0][0], xr[n // 2 : n // 2 + n])
    assert np.max(np.abs(field - ref)) < 1e-6 * np.max(np.abs(ref))


def test_resample_logs_the_residual_phase() -> None:
    field = _sphere(81)
    c = -K / 25.0 * np.eye(2)
    grid, entry = chain.resample(field, 64, 0.05, c)
    assert isinstance(grid, GriddedField)
    assert np.array_equal(grid.values, field.to_grid(64, 0.05, c).values)
    assert entry.metrics["residual_phase_per_step"] < np.pi


def test_wrong_representation_is_rejected() -> None:
    field = _sphere(5)
    spec, _ = chain.pft(field)
    with pytest.raises(ValueError):
        chain.pft(spec)
    with pytest.raises(ValueError):
        chain.inverse_pft(field)
    with pytest.raises(ValueError):
        chain.sft(field)
    with pytest.raises(ValueError, match="fft"):
        chain.sft(GriddedField(0.1, np.zeros((8, 8, 3), np.complex128), LAMBDA_UM))
