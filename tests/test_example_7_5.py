"""Milestone F3: the worked example of the paper, section 7.5 (D = 25 mm, NA 0.5, λ = 0.5 µm).

Issue #40, focus: aplanatic lens (`GridlessField.aplanatic`) with a hard stop of radius a in the
exit plane, operators chosen by `chain.plan_propagation`, reference `focus.richards_wolf`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fieldtracing import chain
from fieldtracing._types import FloatArray
from fieldtracing.field import GriddedField, GridlessField
from fieldtracing.focus import richards_wolf
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
