"""Reference cases for the port to Raytatouille (#50): one function per case, each returning the
data of a `FieldReference` (fieldtracing.results). Used by tools/export_reference.py and by
tests/test_reference_files.py (which recomputes every case and compares it with its file).

Each case stores the outputs of this implementation (`expected`, tight tolerance: the port runs
the same algorithm) and an independent value (`reference`, analytic or exact) with the physical
tolerance measured in the tests named in `source`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from fieldtracing._types import ComplexArray, FloatArray
from fieldtracing.field import GridlessField
from fieldtracing.focus import airy_integral, debye_focus, richards_wolf
from fieldtracing.fourier import centered_grid, frequency_step, ft
from fieldtracing.lpia import (
    fresnel_coefficients_batch,
    fresnel_operator,
    interface_amplitude,
    mesh_jacobian,
    trace_surface,
)
from fieldtracing.mesh import mesh_jacobian_cloud
from fieldtracing.pft import pft_2d
from fieldtracing.propagation import angular_spectrum
from fieldtracing.results import field_reference
from fieldtracing.rtt_bridge import field_operator
from fieldtracing.selection import Situation, select_method
from fieldtracing.semiclassical import propagate_2d
from fieldtracing.sft import sft_2d
from fieldtracing.units import wavenumber

TIGHT = {"rel": 1e-12, "abs": 1e-14}
K = wavenumber(0.5)


def fourier_gaussian() -> dict[str, Any]:
    n, step, sigma = 128, 0.05, 0.3  # window ±10σ: truncation and aliasing below rounding
    x = centered_grid(n, step)
    f = np.exp(-(x**2) / (2 * sigma**2))
    spectrum = ft(f.astype(np.complex128), step)
    k = centered_grid(n, frequency_step(n, step))
    analytic = sigma / math.sqrt(2 * math.pi) * np.exp(-(k**2) * sigma**2 / 2)
    return field_reference(
        "fourier_gaussian",
        "1D Fourier transform (2π)^-1 ∫ f e^{-ikx} dx of a Gaussian on a centred grid.",
        "tests/test_units_fourier.py; paper eq. ft_convention",
        {"n": n, "step_mm": step, "sigma_mm": sigma},
        {"x_mm": x, "f": f},
        {"k_per_mm": k, "spectrum": spectrum},
        TIGHT,
        {"spectrum": analytic.astype(np.complex128)},
        {"abs": 1e-15},
    )


def angular_spectrum_tilted_wave() -> dict[str, Any]:
    n, step, dz = 16, 0.02, 2.5
    x = centered_grid(n, step)
    kx = 3 * frequency_step(n, step)  # on the frequency grid: exactly periodic
    ky = -2 * frequency_step(n, step)
    xx, yy = np.meshgrid(x, x)
    field = np.exp(1j * (kx * xx + ky * yy))
    out = angular_spectrum(field, step, K, dz)
    kz = math.sqrt(K * K - kx * kx - ky * ky)
    return field_reference(
        "angular_spectrum_tilted_wave",
        "Angular spectrum of a tilted plane wave on its own frequency grid (exact).",
        "tests/test_propagation.py; paper eq. spw",
        {"n": n, "step_mm": step, "dz_mm": dz, "k_per_mm": K, "kx_per_mm": kx, "ky_per_mm": ky},
        {"field": field},
        {"field": out},
        {"rel": 1e-10, "abs": 1e-12},
        {"field": field * np.exp(1j * kz * dz)},
        {"abs": 1e-10},
    )


def pft_sphere() -> dict[str, Any]:
    f = 25.0
    pts = np.array([[0.0, 0.0], [0.5, -0.3], [1.2, 0.7], [-2.0, 1.5]])
    g = GridlessField.spherical(pts, np.ones((4, 3), np.complex128), f, 0.5)
    k, res, phase = pft_2d(g.points, g.amplitude[:, 0], g.phase, g.gradient, g.hessian)
    k_perp = np.hypot(k[:, 0], k[:, 1])
    kz = np.sqrt(K * K - k_perp**2)
    legendre = K * f - f * kz  # exact Legendre transform of ψ = -k(r - f): k f - f k_z
    amplitude = f / (K * (kz / K) ** 2) / (2 * np.pi)  # |det H|^(-1/2)/(2π)
    return field_reference(
        "pft_sphere",
        "Pointwise Fourier transform of a converging spherical wave (f = 25 mm).",
        "tests/test_pft.py, tests/test_pft_inverse.py; paper eq. pft_final, exact_legendre",
        {"f_mm": f, "wavelength_um": 0.5},
        {"points_mm": pts, "phase": g.phase, "gradient": g.gradient, "hessian": g.hessian},
        {"k_per_mm": k, "residual": res, "phase": phase},
        TIGHT,
        {"abs_residual": amplitude, "phase": legendre},
        {"abs": 1e-9},
    )


def sft_gauss_chirp() -> dict[str, Any]:
    n, length = 512, 3.0
    step = length / n

    def rot(deg: float) -> FloatArray:
        a = math.radians(deg)
        return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

    c = rot(30.0) @ np.diag([8000.0, -5000.0]) @ rot(30.0).T
    a = rot(50.0) @ np.diag([1 / 0.3**2, 1 / 0.2**2]) @ rot(50.0).T
    x = centered_grid(n, step)
    rho = np.stack(np.meshgrid(x, x), axis=-1)
    v = np.exp(-0.5 * np.einsum("...i,ij,...j->...", rho, a, rho)).astype(np.complex128)
    k, res, phase = sft_2d(v, step, c, warn=False)
    sel = (slice(n // 2 - 64, n // 2 + 65, 16), slice(n // 2 - 64, n // 2 + 65, 16))
    m = a - 1j * c
    kk = k[sel]
    closed = np.exp(-0.5 * np.einsum("...i,ij,...j->...", kk, np.linalg.inv(m), kk)) / (
        2 * np.pi * np.prod(np.sqrt(np.linalg.eigvals(m)))
    )
    return field_reference(
        "sft_gauss_chirp",
        "2D SFT of a rotated Gaussian times a rotated saddle chirp; 9×9 subset of the output.",
        "tests/test_sft_2d.py; paper eq. sft_formula",
        {
            "n": n,
            "step_mm": step,
            "subset": "rows/cols n/2-64 … n/2+64 step 16",
            "note": "spectrum = residual·exp(i·phase)",
        },
        {"c_per_mm2": c, "a_per_mm2": a},
        {"k_per_mm": kk, "residual": res[sel], "phase": phase[sel]},
        {"rel": 1e-10, "abs": 1e-12 * float(np.max(np.abs(res)))},
        {"spectrum": closed},
        {"abs": 1e-6 * float(np.max(np.abs(closed)))},
    )


def debye_airy() -> dict[str, Any]:
    f, sin_max, n = 2.0, 0.5, 64

    def residual(x: FloatArray, y: FloatArray) -> ComplexArray:
        cos = f / np.sqrt(f * f + x * x + y * y)
        return np.asarray(cos**2, dtype=np.complex128)  # constant integrand: Airy pattern

    out = debye_focus(residual, f, K, sin_max, n=n, pad=4)
    m = len(out.x) // 2
    js = np.arange(0, 24, 3)
    values = out.values[m, m + js]
    rho = out.x[m + js]
    airy = -1j * f / (2 * np.pi * K) * airy_integral(rho, K * sin_max)
    return field_reference(
        "debye_airy",
        "Debye focus by FFT with subpixel k-boundary for U = cos²θ (Airy pattern), row y = 0.",
        "tests/test_debye.py (Airy case); paper eq. debye",
        {"f_mm": f, "sin_max": sin_max, "k_per_mm": K, "n": n, "pad": 4, "oversample": 8},
        {"x_mm": rho},
        {"field": values},
        TIGHT,
        {"field": airy.astype(np.complex128)},
        {"abs": 3e-4 * float(np.abs(airy[0]))},
    )


def richards_wolf_focus() -> dict[str, Any]:
    f, na = 2.0, 0.9
    pts = np.array([[0.0, 0.0], [2e-4, 0.0], [0.0, 2e-4], [1.5e-4, 1.5e-4], [-3e-4, 1e-4]])
    rw = richards_wolf(pts, f, K, na)

    def aplanatic(x: FloatArray, y: FloatArray) -> ComplexArray:
        g = GridlessField.aplanatic(np.stack([np.ravel(x), np.ravel(y)], axis=-1), f, 0.5)
        r = np.sqrt(f * f + np.ravel(x) ** 2 + np.ravel(y) ** 2)
        u = g.amplitude * np.exp(1j * (g.phase + K * (r - f)))[:, None]  # remove the sphere
        return np.asarray(u.reshape(*np.shape(x), 3))

    out = debye_focus(aplanatic, f, K, na, n=256, pad=4)
    m = len(out.x) // 2
    step = float(out.x[1] - out.x[0])
    idx = [(0, 0), (0, 3), (3, 0), (2, 2), (1, -4)]
    grid_pts = np.array([[out.x[m + j], out.x[m + i]] for i, j in idx])
    debye = np.array([out.values[m + i, m + j] for i, j in idx])
    return field_reference(
        "richards_wolf_focus",
        "Vector focus of an aplanatic lens (NA 0.9, x-polarised): Richards-Wolf quadrature and "
        "the Debye FFT of the aplanatic exit field at lattice points.",
        "tests/test_aplanatic.py, tests/test_debye.py; paper eq. debye",
        {"f_mm": f, "sin_max": na, "k_per_mm": K, "n": 256, "pad": 4, "step_mm": step},
        {"points_mm": pts, "lattice_points_mm": grid_pts},
        {"richards_wolf": rw, "debye_at_lattice_points": debye},
        {"rel": 1e-9, "abs": 1e-9 * float(np.max(np.abs(rw)))},
        {"debye_at_lattice_points": richards_wolf(grid_pts, f, K, na)},
        {"abs": 1e-4 * float(np.max(np.abs(rw)))},
    )


def fresnel_convention_a() -> dict[str, Any]:
    cos_i = np.cos(np.radians([0.0, 20.0, 56.31, 70.0, 85.0]))
    co = fresnel_coefficients_batch(1.0, 1.5, cos_i)
    big_t = 1.5 * co.cos_t.real / cos_i
    return field_reference(
        "fresnel_convention_a",
        "Fresnel amplitude coefficients air → n = 1.5 (Convention A, Byrnes eq. (6)).",
        "tests/test_lpia.py (T8); paper eq. fresnel_operator",
        {"n1": 1.0, "n2": 1.5},
        {"cos_i": cos_i},
        {"r_s": co.r_s, "r_p": co.r_p, "t_s": co.t_s, "t_p": co.t_p, "cos_t": co.cos_t},
        TIGHT,
        {
            "R_plus_T_s": np.abs(co.r_s) ** 2 + big_t * np.abs(co.t_s) ** 2,
            "R_plus_T_p": np.abs(co.r_p) ** 2 + big_t * np.abs(co.t_p) ** 2,
        },
        {"abs": 1e-14, "note_value": 1.0},
    )


def geometric_propagation_astigmatic() -> dict[str, Any]:
    c = np.array([[-400.0, 60.0], [60.0, -250.0]])
    pts = np.array([[0.0, 0.0], [1.2, -0.7], [-0.9, 1.4], [0.3, 0.2]])
    amp = np.ones(len(pts), np.complex128)
    out = {}
    for d in (10.0, 40.0):  # before and between the line foci
        r = propagate_2d(pts, amp, np.zeros(len(pts)), pts @ c, np.tile(c, (4, 1, 1)), K, d)
        assert r.curvature is not None
        out[f"d{int(d)}"] = {
            "points_mm": r.points,
            "amplitude": r.amplitude,
            "phase": r.phase,
            "jacobian": r.jacobian,
            "maslov": r.maslov.astype(np.int64),
            "hessian_per_mm2": r.curvature,
        }
    return field_reference(
        "geometric_propagation_astigmatic",
        "Geometric propagation (Jacobian law, Maslov index, new Hessian) of a tilted astigmatic "
        "quadratic phase, before and between the line foci.",
        "tests/test_semiclassical.py, tests/test_maslov_2d.py; paper eq. jacobian_amplitude",
        {"wavelength_um": 0.5, "distances_mm": [10.0, 40.0]},
        {"points_mm": pts, "c_per_mm2": c},
        out,
        TIGHT,
    )


def mesh_jacobian_jittered() -> dict[str, Any]:
    n, half = 21, 2.0
    h = 2 * half / (n - 1)
    x = np.linspace(-half, half, n)
    grid = np.stack(np.meshgrid(x, x), axis=-1).reshape(-1, 2)
    pts = grid + np.random.default_rng(7).uniform(-0.3 * h, 0.3 * h, grid.shape)
    g = GridlessField.spherical(pts, np.zeros((len(pts), 3), np.complex128), 25.0, 0.5)
    res = propagate_2d(
        pts, np.ones(len(pts), np.complex128), g.phase, g.gradient, g.hessian, K, 10.0
    )
    cloud = mesh_jacobian_cloud(pts, res.points)
    return field_reference(
        "mesh_jacobian_jittered",
        "Jacobian of a ray map on an unstructured cloud (Delaunay, boundary slivers peeled, "
        "quadratic least squares over the two-ring).",
        "tests/test_mesh.py; paper section 3.4",
        {"n": n, "half_mm": half, "jitter": 0.3, "seed": 7, "f_mm": 25.0, "d_mm": 10.0},
        {"start_mm": pts, "landing_mm": res.points},
        {
            "triangles": cloud.triangles,
            "determinant": cloud.determinant,
            "matrix": cloud.matrix,
            "maslov": cloud.maslov.astype(np.int64),
        },
        {"rel": 1e-10, "abs": 1e-12},
        {"determinant": res.jacobian},
        {"rel": 5e-4},
    )


def lpia_sphere_about_focus() -> dict[str, Any]:
    z_c, radius, z_out, n1, n2 = 30.0, 20.0, 20.0, 1.0, 1.5
    x = np.linspace(-3.0, 3.0, 9)
    pts = np.stack(np.meshgrid(x, x), axis=-1)
    r1 = np.sqrt(np.sum(pts**2, axis=-1) + z_c**2)
    grad = -n1 * K * pts / r1[..., None]
    kz = np.sqrt((n1 * K) ** 2 - np.sum(grad**2, axis=-1))
    s = np.concatenate([grad, kz[..., None]], axis=-1) / (n1 * K)
    pol = np.cross(s, [1.0, 0.0, 0.0])
    pol = (pol / np.linalg.norm(pol, axis=-1, keepdims=True)).astype(np.complex128)

    def sag(xx: FloatArray, yy: FloatArray) -> FloatArray:
        return np.asarray(z_c - np.sqrt(radius**2 - xx * xx - yy * yy))

    def sag_grad(xx: FloatArray, yy: FloatArray) -> tuple[FloatArray, FloatArray]:
        root = np.sqrt(radius**2 - xx * xx - yy * yy)
        return np.asarray(xx / root), np.asarray(yy / root)

    phase = -n1 * K * (r1 - z_c)
    trace = trace_surface(pts, pol, phase, grad, n1, n2, K, sag, sag_grad, z_out)
    amp = interface_amplitude(trace, mesh_jacobian(pts, trace.points))
    r2 = np.sqrt(np.sum(trace.points**2, axis=-1) + (z_c - z_out) ** 2)
    t_normal = 2 * n1 / (n1 + n2)
    return field_reference(
        "lpia_sphere_about_focus",
        "LPIA step: converging wave through a sphere about its focus (normal incidence), 9×9 "
        "rays, mesh Jacobian on the regular grid.",
        "tests/test_lpia_surface.py; paper eq. interface_amplitude",
        {"z_c_mm": z_c, "radius_mm": radius, "z_out_mm": z_out, "n1": n1, "n2": n2},
        {"start_mm": pts, "amplitude": pol, "phase": phase, "gradient_per_mm": grad},
        {"landing_mm": trace.points, "phase": trace.phase, "amplitude": amp},
        {"rel": 1e-11, "abs": 1e-12},
        {"abs_amplitude": t_normal * r1 / r2},
        {"rel": 1e-12},
    )


def field_operator_plate() -> dict[str, Any]:
    n = 1.5168
    out: dict[str, Any] = {"angles_deg": [], "prt": [], "weight": [], "k0": [], "operator": []}
    ref = []
    for theta in (0.0, 20.0, float(np.degrees(np.arctan(n))), 70.0):
        t = math.radians(theta)
        k0 = np.array([0.0, math.sin(t), math.cos(t)])
        k1, t1 = fresnel_operator(k0, np.array([0.0, 0.0, 1.0]), 1.0, n)
        _, t2 = fresnel_operator(k1, np.array([0.0, 0.0, 1.0]), n, 1.0)
        c1 = n * k1[2] / k0[2]
        p = np.sqrt(c1 * (1 / c1)) * (t2 @ t1) + np.outer(k0, k0)  # c₁c₂ = 1 for a plate
        w = 0.7 * 0.5 * np.sum(np.abs(p - np.outer(k0, k0)) ** 2)
        m = field_operator(p[None], np.array([w]), k0[None], k0[None], 1.0, 1.0)[0]
        out["angles_deg"].append(theta)
        out["prt"].append(p)
        out["weight"].append(w)
        out["k0"].append(k0)
        out["operator"].append(m)
        ref.append(math.sqrt(0.7) * (t2 @ t1))
    return field_reference(
        "field_operator_plate",
        "Field amplitude operator from a power-normalised PRT matrix and weight (ADR 0004): "
        "plane-parallel plate in vacuum with 30 % volume absorption.",
        "tests/test_rtt_bridge.py; ADR 0004",
        {"n_glass": n, "absorption_factor": 0.7},
        {
            "prt": np.array(out["prt"]),
            "weight": np.array(out["weight"]),
            "direction": np.array(out["k0"]),
        },
        {"operator": np.array(out["operator"])},
        TIGHT,
        {"operator": np.array(ref)},
        {"abs": 1e-14},
    )


def selection_rules() -> dict[str, Any]:
    strong = Situation(fresnel_min=500.0, fresnel_max=600.0, fresnel_number=95.0)
    cases = {
        "interface": Situation(interface=True, curvature_radius_mm=5.0, wavelength_mm=5e-4),
        "strong_away_from_edges": replace(strong, target_fresnel=300.0),
        "near_edge": replace(
            strong, edge_ratio=0.3, target_fresnel=300.0, fft_samples=3e3, sft_samples=500.0
        ),
        "weak": Situation(fresnel_min=0.0, fresnel_max=0.5),
        "mixed": Situation(fresnel_min=0.0, fresnel_max=600.0),
        "caustic": replace(strong, target_fresnel=0.01),
        "focus_small_nf": Situation(
            fresnel_min=60.0, fresnel_max=70.0, fresnel_number=11.0, target_fresnel=0.01
        ),
    }
    names = list(cases)
    methods = [select_method(cases[k], 0.02)[0] for k in names]
    fields = ["fresnel_min", "fresnel_max", "edge_ratio", "target_fresnel", "fresnel_number"]
    table = np.array([[float(getattr(cases[k], f)) for f in fields] for k in names])
    return field_reference(
        "selection_rules",
        "Selection rules of table tab:selection at ε = 2 % (one case per row).",
        "tests/test_plan.py (decision table); paper table tab:selection",
        {"tolerance": 0.02, "columns": fields},
        {"cases": names, "figures": table},
        {"methods": methods},
        {"rel": 0.0, "abs": 0.0},
    )


CASES: dict[str, Callable[[], dict[str, Any]]] = {
    f.__name__: f
    for f in (
        fourier_gaussian,
        angular_spectrum_tilted_wave,
        pft_sphere,
        sft_gauss_chirp,
        debye_airy,
        richards_wolf_focus,
        fresnel_convention_a,
        geometric_propagation_astigmatic,
        mesh_jacobian_jittered,
        lpia_sphere_about_focus,
        field_operator_plate,
        selection_rules,
    )
}
