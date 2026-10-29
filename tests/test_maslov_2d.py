"""Issue #7 / T7: Maslov index of the geometric propagation in 2D (paper, section 3.4).

Point focus: μ = 2 (Gouy phase -π). Astigmatic wave: μ = 1 between the line foci, μ = 2 behind
both. Reference: exact 2D angular spectrum. The geometric field is compared after demodulating
the exact field with the local plane wave, so that only smooth quantities are interpolated
(ADR 0003).
"""

from __future__ import annotations

import numpy as np
import pytest

from fieldtracing._types import ComplexArray, FloatArray, IntArray
from fieldtracing.fourier import centered_grid
from fieldtracing.propagation import angular_spectrum
from fieldtracing.semiclassical import propagate_2d
from fieldtracing.units import wavenumber

WIDTH, LENGTH = 1.0, 4.0  # super-Gaussian width and window in mm
PROBES = np.array([[0.1, 0.0], [0.2, 0.15], [-0.3, 0.2], [0.0, -0.4]])


def _phase(points: FloatArray, fx: float, fy: float, k: float) -> tuple[FloatArray, ...]:
    """ψ = -k(sqrt(fx² + x²) - fx) - k(sqrt(fy² + y²) - fy), its gradient and Hessian."""
    parts = []
    for coord, f in ((points[..., 0], fx), (points[..., 1], fy)):
        r = np.sqrt(f * f + coord * coord)
        parts.append((-k * (r - f), -k * coord / r, -k * f * f / r**3))
    psi = parts[0][0] + parts[1][0]
    grad = np.stack([parts[0][1], parts[1][1]], axis=-1)
    hess = np.zeros((*points.shape[:-1], 2, 2))
    hess[..., 0, 0], hess[..., 1, 1] = parts[0][2], parts[1][2]
    return psi, grad, hess


def _amplitude(points: FloatArray) -> FloatArray:
    return np.asarray(np.exp(-((np.hypot(points[..., 0], points[..., 1]) / WIDTH) ** 4)))


def _demodulated_sample(
    field: ComplexArray, x: FloatArray, point: FloatArray, kv: FloatArray
) -> complex:
    """Bilinear interpolation of field·exp(-ik·ρ), then multiplied back by exp(ik·ρ)."""
    step = float(x[1] - x[0])
    j, i = (point[0] - x[0]) / step, (point[1] - x[0]) / step
    j0, i0 = int(np.floor(j)), int(np.floor(i))
    dj, di = j - j0, i - i0
    total = 0j
    for ii, jj, w in (
        (i0, j0, (1 - di) * (1 - dj)),
        (i0 + 1, j0, di * (1 - dj)),
        (i0, j0 + 1, (1 - di) * dj),
        (i0 + 1, j0 + 1, di * dj),
    ):
        total += w * field[ii, jj] * np.exp(-1j * (kv[0] * x[jj] + kv[1] * x[ii]))
    return complex(total * np.exp(1j * (kv[0] * point[0] + kv[1] * point[1])))


def _compare(
    wavelength_um: float, n: int, fx: float, fy: float, d: float
) -> tuple[IntArray, FloatArray, ComplexArray, ComplexArray]:
    k = wavenumber(wavelength_um)
    x = centered_grid(n, LENGTH / n)
    grid = np.stack(np.meshgrid(x, x), axis=-1)
    psi, _, _ = _phase(grid, fx, fy, k)
    exact = angular_spectrum(_amplitude(grid) * np.exp(1j * psi), LENGTH / n, k, d)
    p1, g1, h1 = _phase(PROBES, fx, fy, k)
    res = propagate_2d(PROBES, _amplitude(PROBES).astype(np.complex128), p1, g1, h1, k, d)
    predicted = res.amplitude * np.exp(1j * res.phase)
    numeric = np.array(
        [_demodulated_sample(exact, x, res.points[i], g1[i]) for i in range(len(PROBES))]
    )
    return res.maslov, np.abs(numeric - predicted), predicted, numeric


CASES = [(20.0, 20.0, 40.0, 2), (20.0, 40.0, 30.0, 1), (20.0, 40.0, 60.0, 2)]


@pytest.mark.parametrize(("fx", "fy", "d", "mu"), CASES)
def test_maslov_index_and_phase(fx: float, fy: float, d: float, mu: int) -> None:
    maslov, err, predicted, numeric = _compare(0.5, 1600, fx, fy, d)
    assert np.all(maslov == mu)
    assert np.max(err) < 3e-2  # amplitudes O(1); paper: leading order in 1/k
    # without the Maslov factor exp(-iπμ/2) the prediction would be far off
    wrong = predicted * np.exp(0.5j * np.pi * mu)
    assert np.min(np.abs(numeric - wrong)) > 0.5 * np.min(np.abs(predicted))


@pytest.mark.slow
@pytest.mark.parametrize(("fx", "fy", "d", "mu"), CASES)
def test_maslov_error_is_first_order_in_one_over_k(fx: float, fy: float, d: float, mu: int) -> None:
    _, coarse, _, _ = _compare(0.5, 1600, fx, fy, d)
    _, fine, _, _ = _compare(0.25, 3200, fx, fy, d)
    ratio = np.max(coarse) / np.max(fine)
    assert 1.6 < ratio < 2.6  # twice the wavenumber, half the error
