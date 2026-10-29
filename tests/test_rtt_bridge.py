"""Issue #23: power-normalised PRT matrix and weight → field amplitude (ADR 0004).

Synthetic cases (no Raytatouille): P built from `lpia.fresnel_operator` with the power factors c
must give back the field operator. Marker `rtt`: Raytatouille's P at a plane interface (s and p,
four angles incl. Brewster) against √c·T, the field after a plate against T₂T₁ from
`lpia.fresnel_operator`, and Raytatouille's `polar.transmission` against the power of that field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fieldtracing._types import ComplexArray, FloatArray
from fieldtracing.lpia import fresnel_coefficients, fresnel_operator
from fieldtracing.rtt_bridge import absorption_factor, field_operator, transverse_prt

ANGLES_DEG = [0.0, 20.0, float(np.degrees(np.arctan(1.5168))), 70.0]  # incl. Brewster


def _direction(theta_deg: float) -> FloatArray:
    t = np.radians(theta_deg)
    return np.array([0.0, np.sin(t), np.cos(t)])


def _power_factor(n1: float, n2: float, k_in: FloatArray) -> float:
    """c = n₂ cos θ_t/(n₁ cos θ_i) (Byrnes, eq. (21), (22), real indices)."""
    co = fresnel_coefficients(n1, n2, float(k_in[2]))
    return float(n2 * co.cos_t.real / (n1 * k_in[2]))


def _plate(
    theta_deg: float, n_glass: float
) -> tuple[FloatArray, FloatArray, ComplexArray, ComplexArray, float]:
    """Plane-parallel plate in vacuum: k₀, k₁ (in glass), T₁, T₂T₁ and the product c₁c₂."""
    k0 = _direction(theta_deg)
    normal = np.array([0.0, 0.0, 1.0])
    k1, t1 = fresnel_operator(k0, normal, 1.0, n_glass)
    k2, t2 = fresnel_operator(k1, normal, n_glass, 1.0)
    assert np.allclose(k2, k0)
    c = _power_factor(1.0, n_glass, k0) * _power_factor(n_glass, 1.0, k1)
    return k0, k1, t1, t2 @ t1, c


@pytest.mark.parametrize("theta", ANGLES_DEG)
def test_synthetic_prt_gives_back_the_field_operator(theta: float) -> None:
    n = 1.5168
    k0, k1, t1, t21, _ = _plate(theta, n)
    c1 = _power_factor(1.0, n, k0)
    c2 = _power_factor(n, 1.0, k1)
    # P after the first interface, P = √c T + k kᵀ₀ (Raytatouille ADR 0021)
    p1 = np.sqrt(c1) * t1 + np.outer(k1, k0)
    w1 = 0.5 * np.sum(np.abs(transverse_prt(p1[None], k1[None])) ** 2)
    m1 = field_operator(p1[None], np.array([w1]), k0[None], k1[None], 1.0, n)[0]
    assert np.allclose(m1, t1, atol=1e-14)  # planes parallel to the interface: U₂ = T U₁
    # through the plate with 30 % absorption: M = sqrt(0.7)·T₂T₁
    p2 = np.sqrt(c1 * c2) * t21 + np.outer(k0, k0)
    w2 = 0.7 * 0.5 * np.sum(np.abs(transverse_prt(p2[None], k0[None])) ** 2)
    assert absorption_factor(p2[None], np.array([w2]), k0[None])[0] == pytest.approx(0.7)
    m2 = field_operator(p2[None], np.array([w2]), k0[None], k0[None], 1.0, 1.0)[0]
    assert np.allclose(m2, np.sqrt(0.7) * t21, atol=1e-14)
    # area ratio and Maslov index
    m3 = field_operator(p2[None], np.array([w2]), k0[None], k0[None], 1.0, 1.0, 4.0, 2)[0]
    assert np.allclose(m3, -0.5 * m2, atol=1e-14)


# ----- against Raytatouille -------------------------------------------------------------------


def _fresnel_model(tmp_path: Path, reference: Path) -> Path:
    """Raytatouille's M3 Fresnel plate with the fields of ANGLES_DEG (field y = angle in deg)."""
    model: dict[str, Any] = json.loads((reference / "m3" / "fresnel_bk7.rtt.json").read_text())
    model["fields"]["points"] = [{"y": a} if a else {} for a in ANGLES_DEG]
    path = tmp_path / "fresnel_angles.rtt.json"
    path.write_text(json.dumps(model))
    return path


@pytest.mark.rtt
def test_raytatouille_prt_at_one_interface_and_through_the_plate(
    tmp_path: Path, rtt_reference_dir: Path
) -> None:
    import raytatouille as rt

    system = rt.compile(rt.load(_fresnel_model(tmp_path, rtt_reference_dir)))
    n_glass = float(system.media[1].index[0].real)
    assert n_glass == 1.5168 and system.media[0].reference == "VACUUM"
    rays = rt.trace.make_rays(system, rt.trace.PointListPupil([0.0], [0.0]))
    _, paths = rt.trace.trace(system, rays, record_path=True, polarization=True)
    assert paths.prt is not None
    assert list(paths.event_surfaces) == [0, 1, 2, 3]  # STO, P.S1, P.S2, IMG
    final = rays.prt_matrices()
    k_end = np.stack([rays.dir_x, rays.dir_y, rays.dir_z], axis=1)
    k_start = np.asarray(rt.polar.initial_directions(rays))
    s_states = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.complex128), (len(ANGLES_DEG), 1))
    p_states = np.cross(k_start, s_states.real).astype(np.complex128)  # p = k × s
    power_s = rt.polar.transmission(rays, s_states)
    power_p = rt.polar.transmission(rays, p_states)
    for i, theta in enumerate(ANGLES_DEG):
        k0, k1, t1, t21, c = _plate(theta, n_glass)
        assert np.allclose(k_start[i], k0, atol=1e-14)
        p_after_s1 = np.asarray(paths.prt)[i, 2]  # slot 0 input, 1 stop, 2 after P.S1
        assert np.allclose(paths.direction[i, 2], k1, atol=1e-14)
        expected = np.sqrt(_power_factor(1.0, n_glass, k0)) * t1 + np.outer(k1, k0)
        assert np.allclose(p_after_s1, expected, atol=1e-13), theta
        # field after the plate (vacuum, planes z = const, no absorption): T₂T₁
        m = field_operator(
            final[i : i + 1], rays.weight[i : i + 1], k0[None], k_end[i : i + 1], 1.0, 1.0
        )
        assert np.allclose(m[0], t21, atol=1e-13), theta
        assert c == pytest.approx(1.0)
        # power for s and p: Raytatouille's transmission = |T₂T₁ E|² (same medium and angle)
        assert power_s[i] == pytest.approx(float(np.linalg.norm(t21 @ s_states[i]) ** 2), rel=1e-12)
        assert power_p[i] == pytest.approx(float(np.linalg.norm(t21 @ p_states[i]) ** 2), rel=1e-12)
    # Brewster angle: the p-wave passes both surfaces without loss
    assert power_p[2] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.rtt
def test_raytatouille_absorption_enters_the_amplitude(rtt_reference_dir: Path) -> None:
    import raytatouille as rt

    coatings = rt.CoatingLibrary()
    coatings.add_catalog(rtt_reference_dir.parent / "catalogs" / "coatings")
    system = rt.compile(
        rt.load(rtt_reference_dir / "m3" / "absorbing_ar_plate.rtt.json"), coatings=coatings
    )
    # the model has no stop (make_rays needs one): two rays set directly, the second tilted
    rays = rt.trace.RayBatch(2)
    rays.pos_x[:] = [0.0, 1.0]
    rays.dir_y[:] = [0.0, np.sin(0.3)]
    rays.dir_z[:] = [1.0, np.cos(0.3)]
    rays.wl[:] = 0
    stats = rt.trace.trace(system, rays)
    assert stats.count(rt.trace.RayStatus.ALIVE) == 2
    k = np.stack([rays.dir_x, rays.dir_y, rays.dir_z], axis=1)
    prt = rays.prt_matrices()
    s = absorption_factor(prt, rays.weight, k)
    # volume absorption exp(-4πκt/λ) over t = 10 mm (normal incidence, κ = 1e-6, λ = 0.55 µm)
    assert s[0] == pytest.approx(np.exp(-4 * np.pi * 1e-6 * 10.0 / 0.55e-3), rel=1e-12)
    m = field_operator(prt, rays.weight, k, k, 1.0, 1.0)
    e = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
    power = rt.polar.transmission(rays, e)
    assert np.allclose(np.linalg.norm(m @ e, axis=1) ** 2, power, rtol=1e-12)
