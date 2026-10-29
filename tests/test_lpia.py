"""T8 of the paper: Fresnel operator (Convention A) - energy, Brewster, transversality, Snell."""

from __future__ import annotations

import numpy as np
import pytest

from fieldtracing.lpia import fresnel_coefficients, fresnel_operator, refract

N1, N2 = 1.0, 1.5


@pytest.mark.parametrize("theta_deg", [0.0, 20.0, 56.31, 70.0])
def test_energy_conservation(theta_deg: float) -> None:
    ci = np.cos(np.radians(theta_deg))
    co = fresnel_coefficients(N1, N2, ci)
    factor = (N2 * co.cos_t.real) / (N1 * ci)
    assert np.isclose(abs(co.r_s) ** 2 + factor * abs(co.t_s) ** 2, 1.0)
    assert np.isclose(abs(co.r_p) ** 2 + factor * abs(co.t_p) ** 2, 1.0)


def test_normal_incidence_and_brewster() -> None:
    co = fresnel_coefficients(N1, N2, 1.0)
    assert np.isclose(co.t_s, 2 * N1 / (N1 + N2)) and np.isclose(co.t_p, co.t_s)
    assert np.isclose(co.r_p, -co.r_s)  # Convention A
    brewster = np.arctan(N2 / N1)
    assert abs(fresnel_coefficients(N1, N2, np.cos(brewster)).r_p) < 1e-12


def test_operator_is_transversal_and_obeys_snell() -> None:
    k_in = np.array([0.0, np.sin(0.6), np.cos(0.6)])
    normal = np.array([0.1, -0.2, 1.0])
    out, t = fresnel_operator(k_in, normal, N1, N2)
    n_hat = normal / np.linalg.norm(normal)
    for e in (np.array([1.0, 0.0, 0.0]), np.cross(k_in, [1.0, 0.0, 0.0])):
        e = e - k_in * (k_in @ e)
        assert abs(out @ (t @ e)) < 1e-12
    assert np.allclose(np.cross(N2 * out, n_hat), np.cross(N1 * k_in, n_hat))


def test_total_internal_reflection_raises() -> None:
    with pytest.raises(ValueError):
        refract(1.5 * np.array([0.0, np.sin(1.2), np.cos(1.2)]), np.array([0.0, 0.0, 1.0]), 1.0)
