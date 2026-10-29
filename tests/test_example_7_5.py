"""Milestone F3: the worked example of the paper, section 7.5 (D = 25 mm, NA 0.5, λ = 0.5 µm).

Issue #40, focus: aplanatic lens (`GridlessField.aplanatic`) with a hard stop of radius a in the
exit plane, operators chosen by `chain.plan_propagation`, reference `focus.richards_wolf`.

Issue #42, reduced system (same NA, smaller aperture, so f and N_F = a²/(λf) ∝ a shrink): plane
halfway to the focus and focal plane against a direct Rayleigh-Sommerfeld quadrature over the
stop of a closed-form aplanatic exit field (written out here, independent of the chain).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fieldtracing import chain
from fieldtracing._types import ComplexArray, FloatArray
from fieldtracing.field import GriddedField, GridlessField
from fieldtracing.focus import richards_wolf
from fieldtracing.propagation import rayleigh_sommerfeld_kernel
from fieldtracing.selection import fresnel_width, min_edge_distance
from fieldtracing.units import wavenumber

LAMBDA_UM = 0.5
K = wavenumber(LAMBDA_UM)
A = 12.5  # stop radius in mm
NA = 0.5
F = A / math.tan(math.asin(NA))  # 21.65 mm


def polar_cloud(radius: float, rings: int) -> FloatArray:
    """Centre plus rings j = 1..rings with 6j points on radius·j/rings."""
    pts: list[FloatArray] = [np.zeros((1, 2))]
    for j in range(1, rings + 1):
        phi = 2 * np.pi * np.arange(6 * j) / (6 * j)
        r = radius * j / rings
        pts.append(np.column_stack([r * np.cos(phi), r * np.sin(phi)]))
    return np.concatenate(pts)


def test_figures_of_the_example() -> None:
    cloud = GridlessField.aplanatic(polar_cloud(1.02 * A, 40), F, LAMBDA_UM)
    plan = chain.plan_propagation(cloud, F, chain.CircularAperture(A), 0.01, focus_window=0.02)
    s = plan.situation
    # paper, table of section 7.5: h = k₀/f = 580 mm⁻² (centre), radial 377 at the rim
    assert s.fresnel_max / A**2 == pytest.approx(K / F, rel=1e-12)
    assert K / F == pytest.approx(580.4, abs=0.1)
    assert s.fresnel_min / A**2 == pytest.approx(377.0, rel=0.01)  # samples just inside a
    assert fresnel_width(K / F) == pytest.approx(41.5e-3, rel=1e-3)
    assert s.fresnel_number == pytest.approx(1.443e4, rel=1e-3)  # N_F = a²/(λf)
    assert 2 * math.pi * s.fresnel_number == pytest.approx(9.07e4, rel=1e-3)  # F_a
    assert min_edge_distance(fresnel_width(K / F), 0.01) == pytest.approx(1.66, abs=0.01)
    assert plan.method == "debye"  # step 3a: Debye with the sharp k-boundary


def test_focus_against_richards_wolf() -> None:
    cloud = GridlessField.aplanatic(polar_cloud(1.02 * A, 80), F, LAMBDA_UM)
    plan = chain.plan_propagation(cloud, F, chain.CircularAperture(A), 0.01, focus_window=0.02)
    assert plan.method == "debye"
    field, log = plan.run(cloud)
    assert isinstance(field, GriddedField)
    m = field.values.shape[0] // 2
    x = field.x
    reach = int(0.01 / field.step)  # L_F = 20 µm
    js = np.arange(reach)
    pts = np.array(
        [[x[m + j], 0.0] for j in js]
        + [[0.0, x[m + j]] for j in js]
        + [[x[m + j], x[m + j]] for j in js[: int(reach / math.sqrt(2))]]
    )
    got = np.array(
        [field.values[m, m + j] for j in js]
        + [field.values[m + j, m] for j in js]
        + [field.values[m + j, m + j] for j in js[: int(reach / math.sqrt(2))]]
    ) * np.exp(-1j * K * F)
    ref = richards_wolf(pts, F, K, NA)
    error = np.max(np.linalg.norm(got - ref, axis=1)) / np.max(np.linalg.norm(ref, axis=1))
    assert error <= plan.predicted_error
    assert error < 1e-3
    assert plan.parameters["window_mm"] == 0.02 and log[0].operator == "debye"


# ----- issue #42: reduced system against direct Rayleigh-Sommerfeld ---------------------------


def _aplanatic_exit(x: FloatArray, y: FloatArray, f: float) -> ComplexArray:
    """Closed-form exit field of the aplanatic lens (x-polarised), written out independently of
    `GridlessField.aplanatic`: cos^(3/2)θ·[J_ρ(cos θ e_ρ + sin θ e_z) + J_φ e_φ]·e^{iψ}."""
    r = np.hypot(x, y)
    root = np.sqrt(f * f + r * r)
    c, s = f / root, r / root
    safe = np.where(r > 0, r, 1.0)
    cos_phi, sin_phi = np.where(r > 0, x / safe, 1.0), np.where(r > 0, y / safe, 0.0)
    j_r, j_phi = cos_phi, -sin_phi  # x̂ = cos φ e_ρ - sin φ e_φ
    ex = j_r * c * cos_phi - j_phi * sin_phi
    ey = j_r * c * sin_phi + j_phi * cos_phi
    ez = j_r * s
    u = (c**1.5)[..., None] * np.stack([ex, ey, ez], axis=-1)
    return np.asarray(u * np.exp(-1j * K * (root - f))[..., None])


def rs_over_the_stop(
    targets: FloatArray, d: float, a: float, f: float, oversample: float = 1.0
) -> ComplexArray:
    """Rayleigh-Sommerfeld integral (eq. rs_kernel) over the disc |ρ| ≤ a in polar coordinates:
    Gauss-Legendre panels in r (exact edge), trapezoid in φ, node counts from the largest phase
    change; panels in chunks to bound the memory."""
    r_t = float(np.max(np.hypot(targets[:, 0], targets[:, 1])))
    n_phi = int(oversample * (1.3 * K * a * r_t / d + 64))
    n_r = 20 * int(oversample * (K * a * (NA + (r_t + a) / d) / 10 + 4))
    xg, wg = np.polynomial.legendre.leggauss(20)
    edges = np.linspace(0.0, a, n_r // 20 + 1)
    phi = np.arange(n_phi) * 2 * np.pi / n_phi
    out = np.zeros((len(targets), 3), np.complex128)
    for p0 in range(0, len(edges) - 1, 50):
        e = edges[p0 : p0 + 51]
        half = 0.5 * (e[1:] - e[:-1])[:, None]
        r = (0.5 * (e[1:] + e[:-1])[:, None] + half * xg).ravel()
        w_r = (half * wg).ravel()
        x, y = r[:, None] * np.cos(phi), r[:, None] * np.sin(phi)
        src = _aplanatic_exit(x, y, f) * ((w_r * r)[:, None] * 2 * np.pi / n_phi)[..., None]
        for i in range(len(targets)):
            kern = rayleigh_sommerfeld_kernel(targets[i, 0] - x, targets[i, 1] - y, d, K)
            out[i] += np.einsum("ij,ijc->c", kern, src)
    return out


def _probes(
    grid: GriddedField, reach: float, count: int, seed: int
) -> tuple[FloatArray, ComplexArray]:
    """Lattice points on +x from the axis to `reach` and random ones in that disc; field there."""
    pts = grid.sample_points()
    values = grid.values * np.exp(1j * grid.carrier())[..., None]
    c = pts.shape[0] // 2
    step = float(pts[c, c + 1, 0] - pts[c, c, 0])
    cols = np.unique(np.linspace(c, c + reach / step, count // 2).astype(int))
    span = int(reach / step)
    rng = np.random.default_rng(seed)
    ii = np.concatenate(
        [np.full(len(cols), c), rng.integers(c - span, c + span, count - len(cols))]
    )
    jj = np.concatenate([cols, rng.integers(c - span, c + span, count - len(cols))])
    keep = np.hypot(pts[ii, jj, 0], pts[ii, jj, 1]) <= reach
    return pts[ii, jj][keep], values[ii, jj][keep]


def _reduced_case(
    a: float, tolerance: float, plane: str, count: int
) -> tuple[chain.Plan, float, float]:
    """Plan and run stop + propagation of the reduced system; errors relative to max |E_ref|
    outside and inside the axis zone of the plan."""
    f = a / math.tan(math.asin(NA))
    cloud = GridlessField.aplanatic(polar_cloud(1.05 * a, 60), f, LAMBDA_UM)
    d = f / 2 if plane == "half" else f
    plan = chain.plan_propagation(cloud, d, chain.CircularAperture(a), tolerance)
    field, _ = plan.run(cloud)
    assert isinstance(field, GriddedField)
    reach = 0.6 * a if plane == "half" else 3 * 0.61 * LAMBDA_UM * 1e-3 / NA
    pts, values = _probes(field, reach, count, seed=1)
    ref = rs_over_the_stop(pts, d, a, f)
    err = np.linalg.norm(values - ref, axis=1) / np.max(np.linalg.norm(ref, axis=1))
    zone = np.hypot(pts[:, 0], pts[:, 1]) < plan.parameters.get("axis_zone_mm", 0.0)
    return plan, float(np.max(err[~zone])), float(np.max(err[zone], initial=0.0))


def test_reduced_system_focus() -> None:
    plan, error, _ = _reduced_case(0.1, 0.04, "focus", 24)  # N_F ≈ 115
    assert plan.method == "debye"
    assert error <= plan.predicted_error


@pytest.mark.slow
@pytest.mark.parametrize(("a", "tolerance"), [(0.1, 0.04), (0.2, 0.02)])
def test_reduced_system_intermediate_plane(a: float, tolerance: float) -> None:
    plan, error, inside = _reduced_case(a, tolerance, "half", 20)
    assert plan.method == "sft"
    assert error <= plan.predicted_error
    assert inside < 0.1  # axis zone (Poisson spot): documented, not covered by the budget


@pytest.mark.slow
def test_reduced_system_focus_larger() -> None:
    plan, error, _ = _reduced_case(0.2, 0.02, "focus", 24)  # N_F ≈ 231
    assert plan.method == "debye"
    assert error <= plan.predicted_error
