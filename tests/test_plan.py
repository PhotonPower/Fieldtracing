"""Issue #20: planner by the selection rules (paper, section 7) and integration test.

Decision table: one case per row of table tab:selection. Integration test (acceptance of F2):
collimated beam → spherical air/glass surface (LPIA) → circular stop → plane halfway to the
focus and focal plane. Reference: direct Rayleigh-Sommerfeld quadrature (eq. rs_kernel) over the
stop in polar coordinates, with the exit field from closed-form meridional ray tracing (Snell,
Fresnel Convention A, eq. interface_amplitude with the radial Jacobian); no chain code involved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import CubicSpline

from fieldtracing import chain
from fieldtracing._types import ComplexArray, FloatArray
from fieldtracing.field import GriddedField, GridlessField
from fieldtracing.propagation import rayleigh_sommerfeld_kernel
from fieldtracing.selection import Situation, select_method
from fieldtracing.units import wavenumber

LAMBDA_UM = 0.5
K0 = wavenumber(LAMBDA_UM)


# ----- decision table (table tab:selection) -----------------------------------------------------

STRONG = Situation(fresnel_min=500.0, fresnel_max=600.0, fresnel_number=95.0)
NEAR_EDGE = replace(STRONG, edge_ratio=0.3, target_fresnel=300.0)
ROWS = [
    ("interface", Situation(interface=True, curvature_radius_mm=5.0, wavelength_mm=5e-4), "lpia"),
    (
        "interface, R ~ λ",
        Situation(interface=True, curvature_radius_mm=0.01, wavelength_mm=5e-4),
        "rigorous",
    ),
    ("strong, away from edges", replace(STRONG, target_fresnel=300.0), "geometric"),
    ("strong, near an edge", replace(NEAR_EDGE, fft_samples=3e3, sft_samples=500.0), "sft"),
    (
        "near an edge, FFT cheaper",
        replace(NEAR_EDGE, fft_samples=500.0, sft_samples=3e3),
        "fft",
    ),
    (
        "F between 1 and 1/ε",
        Situation(fresnel_min=20.0, fresnel_max=30.0, target_fresnel=300.0, sft_samples=500.0),
        "sft",
    ),
    ("weakly curved", Situation(fresnel_min=0.0, fresnel_max=0.5), "fft"),
    ("mixed", Situation(fresnel_min=0.0, fresnel_max=600.0), "mixed"),
    ("target near a caustic", replace(STRONG, target_fresnel=0.01), "debye"),
    (
        "focus, small Fresnel number",
        Situation(fresnel_min=60.0, fresnel_max=70.0, fresnel_number=11.0, target_fresnel=0.01),
        "rayleigh_sommerfeld",
    ),
]


@pytest.mark.parametrize(("row", "situation", "expected"), ROWS, ids=[r[0] for r in ROWS])
def test_decision_table(row: str, situation: Situation, expected: str) -> None:
    method, reason = select_method(situation, tolerance=0.02)
    assert method == expected, reason


# ----- system: plane wave → sphere (air/glass) → circular stop ----------------------------------


@dataclass(frozen=True)
class System:
    """Spherical surface R (centre behind the vertex z_v), glass n2, stop radius a at z_out."""

    a: float
    radius: float = 5.0
    z_v: float = 1.0
    n2: float = 1.5

    @property
    def z_out(self) -> float:  # behind the surface over the whole cloud
        return self.z_v + 1.3 * (1.1 * self.a) ** 2 / (2.0 * self.radius)

    @property
    def focus(self) -> float:  # paraxial focus behind the exit plane
        return self.n2 * self.radius / (self.n2 - 1.0) - (self.z_out - self.z_v)

    @property
    def k2(self) -> float:
        return self.n2 * K0

    def sag(self, x: FloatArray, y: FloatArray) -> FloatArray:
        return np.asarray(self.z_v + self.radius - np.sqrt(self.radius**2 - x * x - y * y))

    def sag_gradient(self, x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        root = np.sqrt(self.radius**2 - x * x - y * y)
        return np.asarray(x / root), np.asarray(y / root)

    def cloud(self, n: int) -> GridlessField:
        """x-polarised collimated beam on z = 0 (air), square grid cut to 1.1·a."""
        half = 1.1 * self.a
        xs = np.linspace(-half, half, n)
        pts = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
        pts = pts[np.hypot(pts[:, 0], pts[:, 1]) <= half]
        amp = np.zeros((len(pts), 3), np.complex128)
        amp[:, 0] = 1.0
        zero = np.zeros(len(pts))
        return GridlessField(
            pts, zero, np.zeros((len(pts), 2)), np.zeros((len(pts), 2, 2)), amp, LAMBDA_UM
        )


class ExitField:
    """Closed-form exit field: meridional rays, Snell, Convention A, radial Jacobian."""

    def __init__(self, s: System, n_fan: int = 20001) -> None:
        r1 = np.linspace(0.0, 1.1 * s.a, n_fan)
        z_s = s.z_v + s.radius - np.sqrt(s.radius**2 - r1**2)
        sin_i = r1 / s.radius
        cos_i = np.sqrt(1.0 - sin_i**2)
        sin_t = sin_i / s.n2
        cos_t = np.sqrt(1.0 - sin_t**2)
        theta = np.arcsin(sin_i) - np.arcsin(sin_t)  # ray angle to the axis (converging)
        rho2 = r1 - (s.z_out - z_s) * np.tan(theta)
        path = (s.z_out - z_s) / np.cos(theta)
        psi = K0 * (z_s + s.n2 * path)
        t_s = 2.0 * cos_i / (cos_i + s.n2 * cos_t)
        t_p = 2.0 * cos_i / (s.n2 * cos_i + cos_t)
        d_rho = np.gradient(rho2, r1, edge_order=2)
        jac = np.where(r1 > 0, rho2 / np.where(r1 > 0, r1, 1.0) * d_rho, d_rho**2)
        energy = np.sqrt(cos_t / (cos_i * np.cos(theta) * np.abs(jac)))
        self.psi = CubicSpline(rho2, psi)
        self.theta = CubicSpline(rho2, theta)
        self.a_s = CubicSpline(rho2, t_s * energy)
        self.a_p = CubicSpline(rho2, t_p * energy)

    def __call__(self, x: FloatArray, y: FloatArray) -> ComplexArray:
        """E = e^{iψ}(-A_s sin φ e_φ + A_p cos φ (cos θ e_r + sin θ e_z)) for x-polarised input."""
        r, phi = np.hypot(x, y), np.arctan2(y, x)
        th, a_s, a_p = self.theta(r), self.a_s(r), self.a_p(r)
        c, s = np.cos(phi), np.sin(phi)
        e = np.stack(
            [
                a_s * s * s + a_p * np.cos(th) * c * c,
                (a_p * np.cos(th) - a_s) * s * c,
                a_p * np.sin(th) * c,
            ],
            axis=-1,
        )
        return np.asarray(e * np.exp(1j * self.psi(r))[..., None])  # fmt: skip


def rs_reference(
    s: System, exit_field: ExitField, targets: FloatArray, d: float, oversample: float = 1.0
) -> ComplexArray:
    """Rayleigh-Sommerfeld integral over the disc |ρ| ≤ a: Gauss-Legendre panels in r (exact
    edge), trapezoid in φ (periodic, exponential convergence).

    Node counts from the largest phase change of the integrand: in φ the harmonics reach
    k·a·r_t/d, in r the phase changes by at most k·a·(sin θ_max + (r_t + a)/d).
    """
    r_t = float(np.max(np.hypot(targets[:, 0], targets[:, 1])))
    sin_max = s.a / math.hypot(s.a, s.focus)
    n_phi = int(oversample * (1.3 * s.k2 * s.a * r_t / d + 64))
    n_r = 20 * int(oversample * (s.k2 * s.a * (sin_max + (r_t + s.a) / d) / 10 + 4))
    xg, wg = leggauss(20)
    edges = np.linspace(0.0, s.a, n_r // 20 + 1)
    half = 0.5 * (edges[1:] - edges[:-1])[:, None]
    r = (0.5 * (edges[1:] + edges[:-1])[:, None] + half * xg).ravel()
    w_r = (half * wg).ravel()
    phi = np.arange(n_phi) * 2.0 * np.pi / n_phi
    x, y = r[:, None] * np.cos(phi), r[:, None] * np.sin(phi)
    src = exit_field(x, y) * ((w_r * r)[:, None] * 2.0 * np.pi / n_phi)[..., None]
    out = np.empty((len(targets), 3), np.complex128)
    for i in range(len(targets)):
        kern = rayleigh_sommerfeld_kernel(targets[i, 0] - x, targets[i, 1] - y, d, s.k2)
        out[i] = np.einsum("ij,ijc->c", kern, src)
    return out


def _probe_points(
    grid: GriddedField, reach: float, count: int, seed: int
) -> tuple[FloatArray, ComplexArray]:
    """Lattice points along +x from the axis to `reach` and random ones in the disc of that
    radius; field there."""
    pts = grid.sample_points()
    values = grid.values * np.exp(1j * grid.carrier())[..., None]
    n = pts.shape[0]
    c = n // 2
    step = float(pts[c, c + 1, 0] - pts[c, c, 0])
    cols = np.unique(np.linspace(c, c + reach / step, count // 2).astype(int))
    rng = np.random.default_rng(seed)
    span = int(reach / step)
    rows_r = rng.integers(c - span, c + span, count - len(cols))
    cols_r = rng.integers(c - span, c + span, count - len(cols))
    ii = np.concatenate([np.full(len(cols), c), rows_r])
    jj = np.concatenate([cols, cols_r])
    keep = np.hypot(pts[ii, jj, 0], pts[ii, jj, 1]) <= reach
    return pts[ii, jj][keep], values[ii, jj][keep]


@dataclass(frozen=True)
class Outcome:
    method: str
    budget: float
    error_outside: float  # max error outside the axis zone, relative to max |E_ref|
    error_inside: float  # max error inside the axis zone (not covered by the budget)


def _plane_case(s: System, tolerance: float, method: str | None, count: int) -> Outcome:
    exit_field = ExitField(s)
    start = s.cloud(61)
    interface = chain.plan_interface(
        start, s.n2, s.sag, s.sag_gradient, s.z_out, s.radius, tolerance
    )
    assert interface.method == "lpia"
    out, _ = interface.run(start)
    assert isinstance(out, GridlessField)
    d = s.focus / 2
    plan = chain.plan_propagation(out, d, chain.CircularAperture(s.a), tolerance, method=method)
    field, log = plan.run(out)
    assert isinstance(field, GriddedField) and not any(e.warnings for e in log)
    pts, values = _probe_points(field, 0.65 * s.a, count, seed=1)
    ref = rs_reference(s, exit_field, pts, d)
    err = np.linalg.norm(values - ref, axis=1) / np.max(np.linalg.norm(ref, axis=1))
    zone = np.hypot(pts[:, 0], pts[:, 1]) < plan.parameters["axis_zone_mm"]
    budget = plan.predicted_error + interface.predicted_error
    return Outcome(
        plan.method, budget, float(np.max(err[~zone])), float(np.max(err[zone], initial=0.0))
    )


SMALL = System(a=0.354)  # N_F ≈ 25, NA ≈ 0.036
LARGE = System(a=0.707)  # N_F ≈ 100, NA ≈ 0.071


@pytest.mark.parametrize("method", [None, "sft"])
def test_plane_before_focus_within_budget(method: str | None) -> None:
    result = _plane_case(SMALL, 0.04, method, count=30)
    assert result.method == (method or "fft")  # N_F ≈ 25: the direct FFT is cheaper
    assert result.error_outside <= result.budget


def _focus_case(s: System, tolerance: float, count: int) -> tuple[chain.Plan, float]:
    exit_field = ExitField(s)
    start = s.cloud(61)
    out, _ = chain.plan_interface(start, s.n2, s.sag, s.sag_gradient, s.z_out, s.radius).run(start)
    assert isinstance(out, GridlessField)
    plan = chain.plan_propagation(
        out, s.focus, chain.CircularAperture(s.a), tolerance, focus_samples=24 if s is SMALL else 64
    )
    field, _ = plan.run(out)
    assert isinstance(field, GriddedField)
    airy = 0.61 * 2.0 * math.pi / s.k2 / (s.a / math.hypot(s.a, s.focus))
    pts, values = _probe_points(field, 2.0 * airy, count, seed=2)
    ref = rs_reference(s, exit_field, pts, s.focus)
    err = np.linalg.norm(values - ref, axis=1) / np.max(np.linalg.norm(ref, axis=1))
    return plan, float(np.max(err))


def test_focus_small_fresnel_number_uses_rayleigh_sommerfeld() -> None:
    plan, error = _focus_case(SMALL, 0.04, count=16)
    assert plan.method == "rayleigh_sommerfeld"  # Debye error 1.5/N_F ≈ 6 % > ε
    assert error <= plan.predicted_error


def test_focus_within_budget_debye() -> None:
    plan, error = _focus_case(LARGE, 0.02, count=16)
    assert plan.method == "debye"  # 1.5/N_F ≈ 1.5 % ≤ ε
    assert error <= plan.predicted_error


@pytest.mark.slow
@pytest.mark.parametrize("method", [None, "sft"])
def test_plane_before_focus_large_system(method: str | None) -> None:
    result = _plane_case(LARGE, 0.02, method, count=40)
    assert result.method == (method or "fft")
    assert result.error_outside <= result.budget


def _sphere(f: float = 25.0, n: int = 21, half: float = 1.0) -> GridlessField:
    xs = np.linspace(-half, half, n)
    pts = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
    amp = np.zeros((len(pts), 3), np.complex128)
    amp[:, 0] = 1.0
    return GridlessField.spherical(pts, amp, f, LAMBDA_UM)


def test_planner_rows_on_real_fields() -> None:
    sphere = _sphere()
    geometric = chain.plan_propagation(sphere, 12.5)
    assert geometric.method == "geometric"
    out, _ = geometric.run(sphere)
    ref, _ = chain.geometric_propagation(sphere, 12.5)
    assert isinstance(out, GridlessField) and isinstance(ref, GridlessField)
    assert np.array_equal(out.amplitude, ref.amplitude)
    assert geometric.budget["stationary_phase"] == pytest.approx(
        1.0 / geometric.situation.fresnel_min
    )
    stopped = chain.plan_propagation(sphere, 12.5, chain.CircularAperture(0.8))
    assert stopped.method in ("sft", "fft") and stopped.budget["hard_edge_grid"] > 0.0
    assert stopped.warnings and "axis_zone_mm" in stopped.parameters
    focus = chain.plan_propagation(sphere, 25.0, chain.CircularAperture(0.8))
    assert focus.method in ("debye", "rayleigh_sommerfeld")
    # cylindrical wave: curved in x only → mixed chart, no operator chain yet
    pts = sphere.points
    cyl = GridlessField.quadratic(pts, sphere.amplitude, np.diag([-K0 / 25.0, 0.0]), LAMBDA_UM)
    mixed = chain.plan_propagation(cyl, 10.0)
    assert mixed.method == "mixed" and mixed.warnings
    with pytest.raises(ValueError, match="not implemented"):
        mixed.run(cyl)
    # collimated beam behind a stop: weakly curved → direct FFT
    flat = GridlessField.quadratic(pts, sphere.amplitude, np.zeros((2, 2)), LAMBDA_UM)
    assert chain.plan_propagation(flat, 10.0, chain.CircularAperture(0.8)).method == "fft"
    # interface with a curvature radius of a few wavelengths: LPIA not valid
    tiny = chain.plan_interface(flat, 1.5, SMALL.sag, SMALL.sag_gradient, 2.0, 0.002)
    assert tiny.method == "rigorous" and not tiny.operators
