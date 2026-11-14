"""Issue #1 / T4: Debye focus by FFT (paper, section 5.3, eq. debye).

References are independent of the FFT path: the analytic Airy integral, a closed-form on-axis
defocus integral, the exact Rayleigh-Sommerfeld integral on the axis, and a Bessel quadrature of
the Richards-Wolf integrals.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import jv

from fieldtracing._types import ComplexArray, FloatArray
from fieldtracing.focus import airy_integral, debye_focus
from fieldtracing.propagation import rayleigh_sommerfeld_kernel
from fieldtracing.units import wavenumber

LAMBDA_UM = 0.5
K = wavenumber(LAMBDA_UM)


def _cos_theta(x: FloatArray, y: FloatArray, f: float) -> FloatArray:
    """cos θ of the ray through the exit-plane point (x, y) of a wave converging at distance f."""
    return np.asarray(f / np.sqrt(f * f + x * x + y * y))


def _airy_error(sin_max: float, n: int, subpixel: bool) -> float:
    """Relative deviation from the Airy integral for U = cos²θ (constant integrand)."""
    f = 2.0

    def residual(x: FloatArray, y: FloatArray) -> ComplexArray:
        return np.asarray(_cos_theta(x, y, f) ** 2, dtype=np.complex128)

    out = debye_focus(residual, f, K, sin_max, n=n, pad=4, subpixel=subpixel)
    m = out.x.size
    sel = np.abs(out.x) < 3 * 0.61 * LAMBDA_UM * 1e-3 / sin_max
    ref = -1j * f / (2 * np.pi * K) * airy_integral(out.x[sel], K * sin_max)
    return float(np.max(np.abs(out.values[m // 2, sel] - ref)) / np.max(np.abs(ref)))


@pytest.mark.parametrize("sin_max", [0.5, 0.9])
def test_airy_converges_and_subpixel_helps(sin_max: float) -> None:
    hard = [_airy_error(sin_max, n, False) for n in (64, 256)]
    soft = [_airy_error(sin_max, n, True) for n in (64, 128, 256)]
    assert hard[1] < hard[0] < 1e-2
    assert soft[2] < soft[1] < soft[0] < 5e-4
    assert soft[2] < hard[1] / 20  # boundary weighting gains more than an order of magnitude


@pytest.mark.parametrize("defocus", [-5e-3, -2e-3, 1e-3, 2e-3, 5e-3])
def test_on_axis_defocus_matches_closed_form(defocus: float) -> None:
    # U = cos²θ: E(0, δz) = -(i f/k)·∫_{k cosθ_m}^{k} u·exp(i u δz) du (substitution u = k_z)
    f, sin_max = 2.0, 0.5

    def antiderivative(u: float) -> complex:
        return complex(np.exp(1j * defocus * u) * (u / (1j * defocus) + 1 / defocus**2))

    def residual(x: FloatArray, y: FloatArray) -> ComplexArray:
        return np.asarray(_cos_theta(x, y, f) ** 2, dtype=np.complex128)

    out = debye_focus(residual, f, K, sin_max, n=256, pad=1, defocus=defocus)
    m = out.x.size
    k_c = K * np.sqrt(1 - sin_max**2)
    ref = -1j * f / K * (antiderivative(K) - antiderivative(k_c))
    assert abs(out.values[m // 2, m // 2] - ref) / abs(ref) < 1e-3


def _rayleigh_sommerfeld_on_axis(f: float, a: float, defocus: float) -> complex:
    """Exact RS field on the axis at z = f + δz for U = 1 and spherical phase in |ρ| ≤ a."""
    rho = np.linspace(0.0, a, 200_001)
    exit_field = np.exp(-1j * K * (np.sqrt(f * f + rho * rho) - f))
    kernel = rayleigh_sommerfeld_kernel(rho, np.zeros_like(rho), f + defocus, K)
    return complex(np.trapezoid(2 * np.pi * rho * kernel * exit_field, rho))


def _peak(dz: FloatArray, intensity: FloatArray) -> float:
    j = int(np.argmax(intensity))
    y0, y1, y2 = intensity[j - 1 : j + 2]
    return float(dz[j] + 0.5 * (dz[1] - dz[0]) * (y0 - y2) / (y0 - 2 * y1 + y2))


def _t4(fresnel_number: float) -> tuple[float, float, float]:
    """Deviation Debye vs. RS on the axis and both focal shifts (in units of λ/NA²)."""
    na = 0.1
    tan_t = np.tan(np.arcsin(na))
    f = fresnel_number * LAMBDA_UM * 1e-3 / tan_t**2  # N_F = a²/(λ f) with a = f tan θ
    a = f * tan_t
    depth = LAMBDA_UM * 1e-3 / na**2
    dz = np.linspace(-1.5 * depth, 1.5 * depth, 41)

    def residual(x: FloatArray, y: FloatArray) -> ComplexArray:
        return np.ones_like(x, dtype=np.complex128)

    debye = []
    for d in dz:
        out = debye_focus(residual, f, K, na, n=128, pad=1, defocus=float(d))
        m = out.x.size
        debye.append(out.values[m // 2, m // 2] * np.exp(1j * out.phase_offset))
    deb = np.array(debye)
    rs = np.array([_rayleigh_sommerfeld_on_axis(f, a, float(d)) for d in dz])
    deviation = float(np.max(np.abs(rs - deb)) / np.max(np.abs(deb)))
    return deviation, _peak(dz, np.abs(rs) ** 2) / depth, _peak(dz, np.abs(deb) ** 2) / depth


def test_t4_debye_against_rayleigh_sommerfeld() -> None:
    numbers = (3.0, 10.0, 30.0, 100.0, 1000.0)
    results = [_t4(nf) for nf in numbers]
    deviation = [r[0] for r in results]
    shift_rs = [r[1] for r in results]
    shift_debye = [r[2] for r in results]
    # the Debye approximation improves like 1/N_F
    assert all(d1 > d2 for d1, d2 in pairwise(deviation))
    assert all(1.2 < d * nf < 2.0 for d, nf in zip(deviation[1:], numbers[1:], strict=True))
    assert deviation[-1] < 3e-3
    # focal shift towards the lens, visible for small N_F, absent in the Debye integral
    assert all(s < 0 for s in shift_rs)
    assert all(abs(s1) > abs(s2) for s1, s2 in pairwise(shift_rs))
    assert abs(shift_rs[0]) > 0.2 and abs(shift_rs[-1]) < 5e-3
    assert max(abs(s) for s in shift_debye) < 1e-3


def _richards_wolf(rho: float, phi: float, f: float, sin_max: float) -> ComplexArray:
    """Focal-plane field of an aplanatic lens, x-polarised input, by Bessel quadrature.

    E_sphere = sqrt(cos θ)·[cos θ cos²φ + sin²φ, (cos θ - 1) sin φ cos φ, -sin θ cos φ];
    E = -(i k f/2π) ∬ E_sphere exp(i k ρ sin θ cos(φ - φ_p)) dΩ, reduced with
    ∫ cos(nφ) exp(i x cos(φ - φ_p)) dφ = 2π iⁿ J_n(x) cos(nφ_p).
    """
    t_max = float(np.arcsin(sin_max))

    def integral(order: int, weight: str) -> float:
        def g(t: float) -> float:
            c, s = np.cos(t), np.sin(t)
            w = {"0": s * (1 + c), "1": s * s, "2": s * (1 - c)}[weight]
            return float(np.sqrt(c) * w * jv(order, K * rho * s))

        return float(quad(g, 0.0, t_max, limit=400)[0])

    i0, i1, i2 = integral(0, "0"), integral(1, "1"), integral(2, "2")
    ex = -0.5j * K * f * (i0 + i2 * np.cos(2 * phi))
    ey = -0.5j * K * f * i2 * np.sin(2 * phi)
    ez = -K * f * i1 * np.cos(phi)
    return np.array([ex, ey, ez], dtype=np.complex128)


def _aplanatic_x(f: float) -> Callable[[FloatArray, FloatArray], ComplexArray]:
    def residual(x: FloatArray, y: FloatArray) -> ComplexArray:
        c = _cos_theta(x, y, f)
        s = np.sqrt(1 - c * c)
        phi = np.arctan2(-y, -x)  # azimuth of k⊥ (antiparallel to the exit-plane point)
        cp, sp = np.cos(phi), np.sin(phi)
        sphere = np.sqrt(c)[..., None] * np.stack(
            [c * cp**2 + sp**2, (c - 1) * sp * cp, -s * cp], axis=-1
        )
        return np.asarray(c[..., None] * sphere, dtype=np.complex128)  # U = cos θ·E_sphere

    return residual


@pytest.mark.parametrize("sin_max", [0.5, 0.9])
def test_vector_focus_matches_richards_wolf(sin_max: float) -> None:
    f = 2.0
    out = debye_focus(_aplanatic_x(f), f, K, sin_max, n=256, pad=4)
    m = out.x.size
    peak = np.max(np.abs(out.values[m // 2, m // 2]))
    for i, j in [(0, 3), (0, 6), (3, 3), (5, 2), (-4, 4)]:
        rho = float(np.hypot(out.x[m // 2 + j], out.x[m // 2 + i]))
        phi = float(np.arctan2(out.x[m // 2 + i], out.x[m // 2 + j]))
        ref = _richards_wolf(rho, phi, f, sin_max)
        assert np.max(np.abs(out.values[m // 2 + i, m // 2 + j] - ref)) < 1e-4 * peak


def test_vector_focus_is_divergence_free() -> None:
    f, sin_max, h = 2.0, 0.9, 2e-6
    res = _aplanatic_x(f)
    planes = [debye_focus(res, f, K, sin_max, n=128, pad=4, defocus=d) for d in (-h, 0.0, h)]
    e = planes[1].values
    step = float(planes[1].x[1] - planes[1].x[0])
    kk = 2 * np.pi * np.fft.fftfreq(e.shape[0], d=step)
    dx = np.fft.ifft(1j * kk[None, :] * np.fft.fft(e[..., 0], axis=1), axis=1)
    dy = np.fft.ifft(1j * kk[:, None] * np.fft.fft(e[..., 1], axis=0), axis=0)
    dz = (planes[2].values[..., 2] - planes[0].values[..., 2]) / (2 * h)
    assert np.max(np.abs(dx + dy + dz)) < 1e-3 * K * np.max(np.abs(e))


def _rayleigh_sommerfeld_plane(
    f: float, a: float, defocus: float, rho: FloatArray, n_r: int = 200, n_phi: int = 128
) -> ComplexArray:
    """Exact RS field at radii `rho` (mm) in the plane z = f + δz, U = 1, spherical exit phase.

    The field is rotationally symmetric, so one azimuth suffices. Gauss-Legendre in r, trapezoid
    in φ (periodic, spectral); the common factor exp(ikf) is omitted. Changes by < 1e-11 at 1.5
    times the nodes for N_F ≤ 1000.
    """
    x, w = np.polynomial.legendre.leggauss(n_r)
    r, wr = 0.5 * a * (x + 1.0), 0.5 * a * w
    angles = 2.0 * np.pi * np.arange(n_phi) / n_phi
    rr, pp = np.meshgrid(r, angles, indexing="ij")
    weights = (wr * r)[:, None] * (2.0 * np.pi / n_phi)
    sphere = np.sqrt(f * f + rr * rr)
    d = f + defocus
    values = []
    for rho_t in rho:
        dist = np.sqrt(d * d + rr * rr - 2.0 * rr * rho_t * np.cos(pp) + rho_t * rho_t)
        kernel = (
            d / (2 * np.pi * dist**2) * (1.0 / dist - 1j * K) * np.exp(1j * K * (dist - sphere))
        )
        values.append(complex(np.sum(weights * kernel)))
    return np.array(values)


def _t4_off_axis(fresnel_number: float, defocus_rel: float) -> float:
    """Max deviation Debye vs. RS off the axis (x axis and diagonal, |ρ| ≤ 1.2 λ/NA, past the
    second dark ring at 1.12 λ/NA), relative to the largest Debye value; δz in units of λ/NA²."""
    na = 0.1
    tan_t = np.tan(np.arcsin(na))
    f = fresnel_number * LAMBDA_UM * 1e-3 / tan_t**2
    a = f * tan_t
    defocus = defocus_rel * LAMBDA_UM * 1e-3 / na**2

    def residual(x: FloatArray, y: FloatArray) -> ComplexArray:
        return np.ones_like(x, dtype=np.complex128)

    out = debye_focus(residual, f, K, na, n=128, pad=4, defocus=defocus)
    c = out.x.size // 2
    limit = 1.2 * LAMBDA_UM * 1e-3 / na
    idx = np.nonzero((out.x >= 0) & (out.x <= limit))[0]
    diag = idx[np.sqrt(2) * out.x[idx] <= limit]
    carrier = np.exp(1j * (out.phase_offset - K * f))
    debye = np.concatenate([out.values[c, idx], out.values[diag, diag]]) * carrier
    rho = np.concatenate([out.x[idx], np.sqrt(2) * out.x[diag]])
    rs = _rayleigh_sommerfeld_plane(f, a, defocus, rho)
    return float(np.max(np.abs(rs - debye)) / np.max(np.abs(debye)))


@pytest.mark.parametrize(
    ("defocus_rel", "low", "high"), [(0.0, 0.25, 0.5), (-1.0, 1.2, 2.0), (1.0, 1.2, 2.0)]
)
def test_t4_off_axis(defocus_rel: float, low: float, high: float) -> None:
    # T4 in 2D: off the axis the Debye integral deviates from RS like c/N_F as well
    numbers = (3.0, 10.0, 30.0, 100.0, 1000.0)
    deviation = [_t4_off_axis(nf, defocus_rel) for nf in numbers]
    assert all(d1 > d2 for d1, d2 in pairwise(deviation))
    assert all(low < d * nf < high for d, nf in zip(deviation[1:], numbers[1:], strict=True))
    assert deviation[-1] < 2e-3


# ----- issue #79: pupils that are not a centred disc (vignetting, a covered half, obstruction) --


def _half_disc_reference(x: float, y: float, k_max: float) -> complex:
    """∬ exp(i k⊥·ρ) d²k⊥ over the half disc |k⊥| ≤ k_max, k_y ≤ 0 (polar quadrature)."""

    def part(fn: Callable[[float], float]) -> float:
        val, _ = quad(lambda phi: fn(phi), np.pi, 2 * np.pi, limit=200)
        return float(val)

    def radial(phi: float, comp: Callable[[float], float]) -> float:
        a = x * np.cos(phi) + y * np.sin(phi)
        val, _ = quad(lambda kr: kr * comp(kr * a), 0.0, k_max, limit=200)
        return float(val)

    re = part(lambda phi: radial(phi, np.cos))
    im = part(lambda phi: radial(phi, np.sin))
    return complex(re, im)


def _half_disc_error(n: int) -> float:
    """Relative deviation of the focus of a pupil covered on one half (exit-plane support y ≥ 0)
    from the direct quadrature, U = cos²θ (constant integrand)."""
    f, sin_max = 2.0, 0.5

    def residual(x: FloatArray, y: FloatArray) -> ComplexArray:
        return np.asarray(_cos_theta(x, y, f) ** 2, dtype=np.complex128)

    out = debye_focus(
        residual, f, K, sin_max, n=n, pad=4, support=lambda x, y: np.asarray(y >= 0.0)
    )
    m = out.x.size
    airy = 0.61 * LAMBDA_UM * 1e-3 / sin_max
    picks = [(m // 2, j) for j in range(m // 2 - 6, m // 2 + 7, 3)] + [(m // 2 + 3, m // 2 + 2)]
    errors, peak = [], 0.0
    for i, j in picks:
        ref = -1j * f / (2 * np.pi * K) * _half_disc_reference(out.x[j], out.x[i], K * sin_max)
        errors.append(abs(out.values[i, j] - ref))
        peak = max(peak, abs(ref))
    assert np.max(np.abs(out.x[[p[1] for p in picks]])) < 3 * airy
    return float(max(errors) / peak)


def test_issue_79_half_covered_pupil_matches_quadrature() -> None:
    errors = [_half_disc_error(n) for n in (64, 128, 256)]
    assert errors[2] < errors[1] < errors[0]  # converges with the step (straight edge, weighted)
    assert errors[2] < 2e-3


def test_issue_79_full_support_changes_nothing() -> None:
    f, sin_max = 2.0, 0.5

    def residual(x: FloatArray, y: FloatArray) -> ComplexArray:
        return np.asarray(_cos_theta(x, y, f) ** 2, dtype=np.complex128)

    plain = debye_focus(residual, f, K, sin_max, n=64, pad=2)
    with_support = debye_focus(
        residual, f, K, sin_max, n=64, pad=2, support=lambda x, y: np.ones(np.shape(x), bool)
    )
    assert np.array_equal(plain.values, with_support.values)


@pytest.mark.parametrize("shape", ["half disc", "two offset circles", "central obstruction"])
def test_issue_79_chain_debye_keeps_the_pupil_shape(shape: str) -> None:
    """The Debye step of the chain integrates over the support of the field's samples: the power
    after it equals the power through the true pupil (the half disc gave twice that before, the
    vignetted pupil 25 % more, the obstruction was filled). Reference: midpoint quadrature of
    cos θ (|U| = 1) over the exact pupil shape on a fine grid, independent of the samples."""
    from fieldtracing.chain import debye
    from fieldtracing.field import GriddedField, GridlessField

    lam, dist, radius = 0.546, 164.5, 4.4
    k = wavenumber(lam)

    def pupil(x: FloatArray, y: FloatArray) -> NDArray[np.bool_]:
        r = np.hypot(x, y)
        if shape == "half disc":
            return np.asarray((r <= radius) & (y >= 0.0))
        if shape == "central obstruction":
            return np.asarray((r <= radius) & (r >= 1.5))
        return np.asarray((np.hypot(x, y - 0.8) <= radius) & (np.hypot(x, y + 0.8) <= radius))

    # the support's edge is known to half a sample spacing (first order, `chain._support`): with
    # 61 samples across the obstructed pupil give 2.5 % less power, 121 samples 1.5 % (n = 128)
    g = np.linspace(-radius, radius, 121)
    xx, yy = np.meshgrid(g, g)
    p = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    p = p[pupil(p[:, 0], p[:, 1])]
    rr = np.sqrt(np.sum(p * p, axis=1) + dist * dist)
    hess = -k * (np.eye(2)[None] - np.einsum("ni,nj->nij", p, p) / rr[:, None, None] ** 2)
    hess = hess / rr[:, None, None]
    u = np.zeros((len(p), 3), dtype=np.complex128)
    u[:, 0] = 1.0
    field = GridlessField(p, -k * rr, -k * p / rr[:, None], hess, u, lam, 1.0)

    h = 2e-3
    fine = np.arange(-radius + h / 2, radius, h)
    fx, fy = np.meshgrid(fine, fine)
    before = float(np.sum(pupil(fx, fy) * _cos_theta(fx, fy, dist)) * h * h)

    sin_max = radius / np.hypot(radius, dist)
    out, _ = debye(field, dist, sin_max, n=128, pad=2)
    assert isinstance(out, GriddedField)
    after = float(np.sum(np.abs(np.asarray(out.values)) ** 2) * out.step**2)
    assert after == pytest.approx(before, rel=2e-2)
