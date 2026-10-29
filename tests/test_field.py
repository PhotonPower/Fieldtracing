"""Issue #17: hybrid field representation (paper, sections 1.3 and 7.2; ADR 0003)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fieldtracing._types import ComplexArray, FloatArray
from fieldtracing.field import GriddedField, GridlessField
from fieldtracing.fourier import centered_grid
from fieldtracing.units import wavenumber

LAMBDA_UM = 0.5
K = wavenumber(LAMBDA_UM)


def _grid_points(half: float, n: int) -> FloatArray:
    x = np.linspace(-half, half, n)
    return np.stack(np.meshgrid(x, x), axis=-1).reshape(-1, 2)


def _transversal_amplitude(field_points: FloatArray, gradient: FloatArray) -> ComplexArray:
    """Smooth Gaussian envelope times a unit vector perpendicular to the local k."""
    kz = np.sqrt(K * K - np.sum(gradient * gradient, axis=1))
    k_hat = np.concatenate([gradient, kz[:, None]], axis=1) / K
    pol = np.cross(k_hat, np.array([1.0, 0.0, 0.0]))
    pol /= np.linalg.norm(pol, axis=1, keepdims=True)
    env = np.exp(-np.sum(field_points**2, axis=1) / 4.0)
    return np.asarray(env[:, None] * pol, dtype=np.complex128)


def _sphere(n: int, half: float = 2.0, f: float = 25.0) -> GridlessField:
    pts = _grid_points(half, n)
    probe = GridlessField.spherical(pts, np.zeros((len(pts), 3), np.complex128), f, LAMBDA_UM)
    return GridlessField.spherical(pts, _transversal_amplitude(pts, probe.gradient), f, LAMBDA_UM)


def _cell_centres(half: float, n: int) -> FloatArray:
    h = 2 * half / (n - 1)
    x = np.linspace(-half + h / 2, half - h / 2, n - 1)
    return np.stack(np.meshgrid(x, x), axis=-1).reshape(-1, 2)


def test_constructors_and_transversality() -> None:
    field = _sphere(21).validate()
    r = np.sqrt(25.0**2 + np.sum(field.points**2, axis=1))
    assert np.allclose(field.phase, -K * (r - 25.0))
    assert np.allclose(field.gradient, -K * field.points / r[:, None])
    bad = GridlessField(
        field.points,
        field.phase,
        field.gradient,
        field.hessian,
        np.tile(np.array([0, 0, 1], dtype=np.complex128), (len(field.points), 1)),
        LAMBDA_UM,
    )
    with pytest.raises(ValueError):
        bad.validate()
    c = np.array([[300.0, 50.0], [50.0, -200.0]])
    quad = GridlessField.quadratic(field.points, field.amplitude, c, LAMBDA_UM)
    assert np.allclose(quad.gradient, field.points @ c)


def test_taylor_phase_is_third_order() -> None:
    errors = []
    for n in (21, 41):
        field = _sphere(n)
        q = _cell_centres(2.0, n)  # farthest from the samples
        r = np.sqrt(25.0**2 + np.sum(q**2, axis=1))
        errors.append(float(np.max(np.abs(field.phase_at(q) + K * (r - 25.0)))))
    assert errors[0] < 0.05  # rad
    assert 6.0 < errors[0] / errors[1] < 10.0  # h³


def _residual_errors(method: str) -> list[float]:
    errors = []
    q = np.random.default_rng(11).uniform(-1.5, 1.5, size=(400, 2))
    exact = GridlessField.spherical(q, np.zeros((len(q), 3), np.complex128), 25.0, LAMBDA_UM)
    ref = _transversal_amplitude(q, exact.gradient)
    for n in (21, 41):
        res = _sphere(n).residual_at(q, method)  # type: ignore[arg-type]
        errors.append(float(np.max(np.abs(res - ref))))
    return errors


def test_residual_interpolation_order() -> None:
    linear = _residual_errors("linear")
    cubic = _residual_errors("cubic")
    assert 3.0 < linear[0] / linear[1] < 5.0  # second order
    assert (
        cubic[0] < linear[0] / 5 and cubic[1] < cubic[0] / 3
    )  # Clough-Tocher: smaller, converging


def test_interpolating_the_full_field_fails_adr_0003() -> None:
    # Curvature of the paper's example (f = 21.65 mm): the residual interpolates well, the full
    # field U·exp(iψ) does not, because ψ changes by far more than π between samples.
    f = 12.5 / math.tan(math.asin(0.5))
    pts = _grid_points(2.0, 81)
    field = GridlessField.spherical(pts, np.ones((len(pts), 3), np.complex128), f, LAMBDA_UM)
    q = _cell_centres(2.0, 81)[:200]
    exact = GridlessField.spherical(q, np.ones((len(q), 3), np.complex128), f, LAMBDA_UM)
    hybrid = field.residual_at(q)[:, 0] * np.exp(1j * field.phase_at(q))
    full = GridlessField(
        field.points,
        np.zeros(len(pts)),
        np.zeros((len(pts), 2)),
        np.zeros((len(pts), 2, 2)),
        field.amplitude * np.exp(1j * field.phase)[:, None],
        LAMBDA_UM,
    ).residual_at(q)[:, 0]
    truth = np.exp(1j * exact.phase)
    assert np.max(np.abs(hybrid - truth)) < 1e-3
    assert np.max(np.abs(full - truth)) > 0.5


def test_to_grid_with_and_without_carrier() -> None:
    field = _sphere(81)
    with pytest.raises(ValueError, match="not resolved"):
        field.to_grid(64, 0.05)  # |∇ψ| up to 400/mm: step 0.05 mm violates Nyquist
    c = -K / 25.0 * np.eye(2)
    gridded = field.to_grid(64, 0.05, curvature=c)
    x = centered_grid(64, 0.05)
    grid = np.stack(np.meshgrid(x, x), axis=-1).reshape(-1, 2)
    exact = GridlessField.spherical(grid, np.zeros((len(grid), 3), np.complex128), 25.0, LAMBDA_UM)
    ref = (
        _transversal_amplitude(grid, exact.gradient)
        * np.exp(1j * (exact.phase - 0.5 * np.einsum("ni,ij,nj->n", grid, c, grid)))[:, None]
    )
    inside = np.max(np.abs(grid), axis=1) < 1.9
    assert np.max(np.abs(gridded.values.reshape(-1, 3)[inside] - ref[inside])) < 2e-3
    with pytest.raises(ValueError, match="carrier"):
        gridded.physical()
    plain = GriddedField(0.001, gridded.values[:8, :8], LAMBDA_UM)
    assert np.allclose(plain.physical(), plain.values)


def test_adapters_pft_roundtrip_and_propagation() -> None:
    field = _sphere(11)
    back = field.pft().inverse_pft()
    assert np.allclose(back.points, field.points) and np.allclose(back.phase, field.phase)
    assert np.max(np.abs(back.amplitude - field.amplitude)) < 1e-12
    assert np.allclose(back.hessian, field.hessian)
    # geometric propagation keeps the field spherical: Hessian of a wave converging to f - d
    moved = field.propagate(10.0)
    ref = GridlessField.spherical(moved.points, moved.amplitude, 15.0, LAMBDA_UM)
    assert np.allclose(moved.hessian, ref.hessian, rtol=1e-9)
    assert np.allclose(moved.gradient, ref.gradient, rtol=1e-9)
