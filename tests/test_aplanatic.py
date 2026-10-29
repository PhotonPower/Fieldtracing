"""Issue #39: aplanatic lens as GridlessField and Richards-Wolf reference (paper, section 5.3).

References: closed-form on-axis integral I₀, energy conservation through the exit plane, and the
Debye chain (`chain.debye`) from the exit-plane cloud against `focus.richards_wolf`.
"""

from __future__ import annotations

import numpy as np
import pytest

from fieldtracing import chain
from fieldtracing._types import FloatArray
from fieldtracing.field import GriddedField, GridlessField
from fieldtracing.focus import richards_wolf
from fieldtracing.units import wavenumber

LAMBDA_UM = 0.5
K = wavenumber(LAMBDA_UM)
F = 2.0


def _polar_cloud(radius: float, rings: int) -> FloatArray:
    pts: list[FloatArray] = [np.zeros((1, 2))]
    for j in range(1, rings + 1):
        r = radius * j / rings
        phi = 2 * np.pi * np.arange(6 * j) / (6 * j)
        pts.append(np.column_stack([r * np.cos(phi), r * np.sin(phi)]))
    return np.concatenate(pts)


def test_richards_wolf_on_axis_closed_form() -> None:
    # I₀ = ∫₀^α sqrt(cos θ) sin θ (1 + cos θ) dθ = 2/3 + 2/5 - (2/3)c^(3/2) - (2/5)c^(5/2)
    for na in (0.3, 0.9):
        c = np.sqrt(1 - na * na)
        i0 = 2 / 3 + 2 / 5 - 2 / 3 * c**1.5 - 2 / 5 * c**2.5
        e = richards_wolf(np.zeros((1, 2)), F, K, na)[0]
        assert e[0] == pytest.approx(-0.5j * K * F * i0, rel=1e-12)
        assert abs(e[1]) < 1e-12 * abs(e[0]) and abs(e[2]) < 1e-12 * abs(e[0])


def test_energy_and_polarization_of_the_exit_field() -> None:
    na = 0.9
    a = F * np.tan(np.arcsin(na))
    # power through the exit plane, radial quadrature of the constructor's values
    r = np.linspace(0.0, a, 20001)
    field = GridlessField.aplanatic(np.column_stack([r, np.zeros_like(r)]), F, LAMBDA_UM)
    cos = F / np.sqrt(F * F + r * r)
    density = np.sum(np.abs(field.amplitude) ** 2, axis=1) * cos * 2 * np.pi * r
    power = float(np.sum(0.5 * (density[1:] + density[:-1]) * np.diff(r)))
    assert power == pytest.approx(np.pi * (F * na) ** 2, rel=1e-6)  # pupil radius h = f·NA
    # transversal; radial input stays meridional, azimuthal stays azimuthal
    pts = _polar_cloud(a, 10)
    assert GridlessField.aplanatic(pts, F, LAMBDA_UM).transversality_error() < 1e-14
    point = np.array([[0.6, 0.8]])
    e_r = np.array([0.6, 0.8, 0.0])
    radial = GridlessField.aplanatic(point, F, LAMBDA_UM, e_r.astype(np.complex128))
    assert abs(np.cross(e_r, [0, 0, 1]) @ radial.amplitude[0]) < 1e-15
    azim = GridlessField.aplanatic(point, F, LAMBDA_UM, np.array([-0.8, 0.6, 0.0], np.complex128))
    assert abs(azim.amplitude[0, 2]) < 1e-15 and abs(e_r @ azim.amplitude[0]) < 1e-15


@pytest.mark.parametrize(("na", "defocus"), [(0.5, 0.0), (0.9, 0.0), (0.9, 6e-4)])
def test_debye_from_the_aplanatic_cloud_matches_richards_wolf(na: float, defocus: float) -> None:
    a = F * np.tan(np.arcsin(na))
    cloud = GridlessField.aplanatic(_polar_cloud(1.02 * a, 120), F, LAMBDA_UM)
    # target plane z = F + δz; reference sphere centred there, stop |ρ| ≤ a as its k-disc
    out, _ = chain.debye(cloud, F + defocus, a / np.hypot(a, F + defocus), n=256)
    assert isinstance(out, GriddedField)
    m = out.values.shape[0] // 2
    x = out.x
    idx = [(0, 0), (0, 3), (0, 6), (3, 3), (5, 2), (-4, 4)]
    pts = np.array([[x[m + j], x[m + i]] for i, j in idx])
    got = np.array([out.values[m + i, m + j] for i, j in idx]) * np.exp(-1j * K * F)
    ref = richards_wolf(pts, F, K, na, defocus)  # physical field = exp(ikf)·RW
    peak = float(np.max(np.abs(ref)))
    # defocused: chain.debye centres the reference sphere on the target plane, Richards-Wolf on
    # the focus; the two Debye integrals differ slightly (measured 1.8e-4 at δz = 1.2 λ)
    tolerance = 1e-4 if defocus == 0.0 else 3e-4
    assert np.max(np.abs(got - ref)) < tolerance * peak
