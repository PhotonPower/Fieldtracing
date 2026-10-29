"""Issue #2: SFT for a general symmetric curvature matrix C, inverse SFT, sampling checks."""

from __future__ import annotations

import warnings
from itertools import pairwise

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import erfc

from fieldtracing._types import ComplexArray, FloatArray
from fieldtracing.fourier import centered_grid
from fieldtracing.sft import (
    SamplingWarning,
    check_sft_sampling,
    inverse_sft_2d,
    region_weights,
    sft_2d,
    sft_2d_diagonal,
)


def _rotation(deg: float) -> FloatArray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


ROT = _rotation(30.0)
C1, C2 = 8000.0, -5000.0  # eigenvalues of C in 1/mm²
C = ROT @ np.diag([C1, C2]) @ ROT.T
LENGTH = 3.0  # window in mm


def _gaussian_case(n: int) -> tuple[FloatArray, ComplexArray, FloatArray, FloatArray]:
    """V = exp(-½ρᵀAρ) with A not commuting with C; returns grid, V, A and the step."""
    rot_a = _rotation(50.0)
    a = rot_a @ np.diag([1 / 0.3**2, 1 / 0.2**2]) @ rot_a.T
    step = LENGTH / n
    x = centered_grid(n, step)
    rho = np.stack(np.meshgrid(x, x), axis=-1)
    v = np.exp(-0.5 * np.einsum("...i,ij,...j->...", rho, a, rho)).astype(np.complex128)
    return x, v, a, np.array(step)


def _gaussian_chirp_spectrum(k: FloatArray, a: FloatArray) -> ComplexArray:
    """(2π)^-2 ∬ exp(-½ρᵀ(A - iC)ρ - ik·ρ) d²ρ = exp(-½kᵀM⁻¹k)/(2π sqrt det M), M = A - iC."""
    m = a - 1j * C
    sqrt_det = np.prod(np.sqrt(np.linalg.eigvals(m)))  # Re of the eigenvalues > 0
    quad_form = np.einsum("...i,ij,...j->...", k, np.linalg.inv(m), k)
    return np.asarray(np.exp(-0.5 * quad_form) / (2 * np.pi * sqrt_det))


def test_matches_diagonal_version() -> None:
    n, step = 256, 0.01
    x = centered_grid(n, step)
    v = (np.exp(-((x[None, :] / 0.5) ** 2)) * (np.abs(x[:, None]) <= 0.6)).astype(np.complex128)
    _, _, r_diag, p_diag = sft_2d_diagonal(v, step, 800.0, -500.0)
    k, res, phase = sft_2d(v, step, np.diag([800.0, -500.0]), warn=False)
    assert np.allclose(phase, p_diag)
    assert np.max(np.abs(res - r_diag)) < 1e-12 * np.max(np.abs(r_diag))
    assert np.allclose(k[..., 0], 800.0 * x[None, :]) and np.allclose(
        k[..., 1], -500.0 * x[:, None]
    )


def test_rotated_gaussian_matches_closed_form() -> None:
    _, v, a, _ = _gaussian_case(1024)
    k, res, phase = sft_2d(v, LENGTH / 1024, C)
    ref = _gaussian_chirp_spectrum(k, a)
    # round-off: phases reach 10⁴ rad, so the absolute error floor is ~10⁻¹² of the peak·10⁴
    assert np.max(np.abs(res * np.exp(1j * phase) - ref)) < 1e-6 * np.max(np.abs(ref))


def test_pft_limit_for_wide_smooth_field() -> None:
    # Without edge the residual approaches V(C⁻¹k)·exp(iπ/4·sig C)/(2π sqrt|det C|).
    x, v, _, _ = _gaussian_case(1024)
    _, res, _ = sft_2d(v, LENGTH / 1024, C)
    # sig C = 0 for the saddle C (eigenvalues 8000 and -5000), so the phase factor is 1
    pft = v / (2 * np.pi * np.sqrt(abs(np.linalg.det(C))))
    assert np.max(np.abs(res - pft)) < 2e-3 * np.max(np.abs(pft))


def test_forward_inverse_roundtrip_is_exact() -> None:
    _, v, _, _ = _gaussian_case(512)
    rng = np.random.default_rng(7)
    v = v * (1 + 0.1 * rng.normal(size=v.shape))  # not band-limited on purpose
    step = LENGTH / 512
    _, res, _ = sft_2d(v, step, C, warn=False)
    rho, back, phase = inverse_sft_2d(res, step, -np.linalg.inv(C), lattice=C)
    x = centered_grid(512, step)
    assert np.allclose(rho[..., 0], x[None, :]) and np.allclose(rho[..., 1], x[:, None])
    assert np.allclose(phase, 0.5 * np.einsum("...i,ij,...j->...", rho, C, rho))
    assert np.max(np.abs(back - v)) < 1e-10


def _slit(c: float, half_width: float, q: FloatArray) -> ComplexArray:
    """∫_{-a}^{a} exp(i c u²/2 - i q u) du in closed form (Fresnel integrals via erfc)."""
    sg = np.sign(c)
    u0 = q / c

    def primitive(xx: float) -> ComplexArray:
        arg = -np.exp(-0.25j * np.pi * sg) * np.sqrt(abs(c) / 2) * (xx - u0)
        return np.asarray(
            np.sqrt(2 * np.pi / abs(c)) * np.exp(0.25j * np.pi * sg) * 0.5 * erfc(arg)
        )

    return np.asarray(np.exp(-0.5j * q * q / c) * (primitive(half_width) - primitive(-half_width)))


def _rotated_rect_error(n: int) -> float:
    """Max deviation (relative to the largest reference value) for a rotated rectangle."""
    a1, a2 = 0.6, 0.4  # half widths along the eigenvectors of C

    def inside(px: FloatArray, py: FloatArray) -> NDArray[np.bool_]:
        u = ROT[0, 0] * px + ROT[1, 0] * py
        w = ROT[0, 1] * px + ROT[1, 1] * py
        return np.asarray((np.abs(u) <= a1) & (np.abs(w) <= a2))

    step = LENGTH / n
    x = centered_grid(n, step)
    k, res, phase = sft_2d(region_weights(x, x, inside).astype(np.complex128), step, C, warn=False)
    probes = np.array([[0, 0], [0.2, 0.1], [0.5, 0.2], [0.45, 0.45], [-0.3, 0.25], [0.7, 0.0]])
    idx = np.rint(probes / step).astype(int) + n // 2
    k_p = k[idx[:, 1], idx[:, 0]]
    value = res[idx[:, 1], idx[:, 0]] * np.exp(1j * phase[idx[:, 1], idx[:, 0]])
    q = k_p @ ROT  # components along the eigenvectors
    ref = _slit(C1, a1, q[:, 0]) * _slit(C2, a2, q[:, 1]) / (2 * np.pi) ** 2
    return float(np.max(np.abs(value - ref)) / np.max(np.abs(ref)))


def test_rotated_rectangle_against_fresnel_integrals() -> None:
    assert _rotated_rect_error(1024) < 5e-2


@pytest.mark.slow
def test_rotated_rectangle_converges_first_order() -> None:
    errors = [_rotated_rect_error(n) for n in (1024, 2048, 4096)]
    assert errors[2] < 1.5e-2
    assert all(1.4 < e1 / e2 < 2.6 for e1, e2 in pairwise(errors))


def test_sampling_warnings() -> None:
    assert check_sft_sampling(1024, LENGTH / 1024, C) == []
    problems = check_sft_sampling(256, LENGTH / 256, C)  # step 11.7 µm > w_c/4
    assert any("edge resolution" in p for p in problems)
    assert any("transfer chirp" in p for p in check_sft_sampling(4096, 0.5 / 4096, C))
    v = np.ones((256, 256), dtype=np.complex128)
    with pytest.warns(SamplingWarning):
        sft_2d(v, LENGTH / 256, C)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sft_2d(v, LENGTH / 256, C, warn=False)
