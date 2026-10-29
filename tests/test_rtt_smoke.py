"""Issue #22: smoke test against Raytatouille (marker `rtt`, read only).

The plano-convex singlet `m1/singlet_const.rtt.json` of Raytatouille (stop at z = 0, EPD 20 mm,
sphere R = 51.68 mm at z = 5, plane at z = 9, constant index 1.5168, image plane at z = 106.363)
is loaded, compiled and traced for three rays of the axial field. Reference: the same rays traced
here in closed form with the vector Snell law (`lpia.refract`): landing point, direction and
optical path length from the start plane of Raytatouille's rays; on the axis the weight equals the
product of the two Fresnel power transmissions (`lpia.fresnel_coefficients`). Raytatouille's
indices are absolute (AIR = Ciddor air, n ≈ 1.00027); they are model input and read from the
compiled system.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fieldtracing._types import FloatArray
from fieldtracing.lpia import fresnel_coefficients, refract

pytestmark = pytest.mark.rtt

RADIUS, Z_VERTEX, Z_BACK, Z_IMAGE = 51.68, 5.0, 9.0, 106.363


def _closed_form(
    start: FloatArray, n_air: float, n_glass: float
) -> tuple[FloatArray, FloatArray, float]:
    """Ray parallel to +z from `start` (mm) to the image plane: (point, direction, OPL in mm)."""
    centre = np.array([0.0, 0.0, Z_VERTEX + RADIUS])
    x, y, z0 = start
    z_s = centre[2] - np.sqrt(RADIUS**2 - x * x - y * y)
    hit = np.array([x, y, z_s])
    d1 = refract(np.array([0.0, 0.0, n_air]), centre - hit, n_glass)  # |k| = n
    d1 = d1 / np.linalg.norm(d1)
    t1 = (Z_BACK - z_s) / d1[2]
    back = hit + t1 * d1
    d2 = refract(n_glass * d1, np.array([0.0, 0.0, 1.0]), n_air)
    d2 = d2 / np.linalg.norm(d2)
    t2 = (Z_IMAGE - Z_BACK) / d2[2]
    return back + t2 * d2, d2, float(n_air * (z_s - z0) + n_glass * t1 + n_air * t2)


def test_singlet_rays_match_the_closed_form(rtt_reference_dir: Path) -> None:
    import raytatouille as rt

    system = rt.compile(rt.load(rtt_reference_dir / "m1" / "singlet_const.rtt.json"))
    wl = system.reference_wavelength
    n_air, n_glass = (float(m.index[wl].real) for m in system.media)
    assert [m.reference for m in system.media] == ["AIR", "CONST:1.5168"]
    pupil = rt.trace.PointListPupil([0.0, 0.0, 0.3], [0.0, 0.5, 0.2])
    rays = rt.trace.make_rays(system, pupil, fields=[0])
    start = np.stack([rays.pos_x, rays.pos_y, rays.pos_z], axis=1).copy()
    assert np.allclose(start[:, :2], 10.0 * np.array([[0.0, 0.0], [0.0, 0.5], [0.3, 0.2]]))
    stats = rt.trace.trace(system, rays)
    assert stats.count(rt.trace.RayStatus.ALIVE) == 3
    for i in range(3):
        point, direction, opl = _closed_form(start[i], n_air, n_glass)
        got = np.array([rays.pos_x[i], rays.pos_y[i], rays.pos_z[i]])
        assert np.allclose(got, point, rtol=0, atol=1e-10)
        assert np.allclose([rays.dir_x[i], rays.dir_y[i], rays.dir_z[i]], direction, atol=1e-12)
        assert rays.opl[i] == pytest.approx(opl, abs=1e-10)
    # on the axis s and p coincide: weight = T₁·T₂ with T = (n₂ cos t)/(n₁ cos i)·|t|²
    t1 = fresnel_coefficients(n_air, n_glass, 1.0).t_s
    t2 = fresnel_coefficients(n_glass, n_air, 1.0).t_s
    power = (n_glass / n_air) * abs(t1) ** 2 * (n_air / n_glass) * abs(t2) ** 2
    assert rays.weight[0] == pytest.approx(power, rel=1e-12)
