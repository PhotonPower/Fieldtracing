"""Fourier convention (paper eq. ft_convention) and units."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fieldtracing.fourier import centered_grid, frequency_step, ft, ift
from fieldtracing.units import wavenumber


def test_wavenumber_in_inverse_mm() -> None:
    assert wavenumber(0.5) == pytest.approx(2 * math.pi / 0.5e-3)
    assert wavenumber(0.5, 1.5) == pytest.approx(1.5 * 2 * math.pi / 0.5e-3)
    with pytest.raises(ValueError):
        wavenumber(0.0)


def test_gaussian_transform_matches_analytic() -> None:
    # F{exp(-x²/(2σ²))}(k) = σ/sqrt(2π)·exp(-σ²k²/2) in the (2π)^-1 convention
    sigma, n, dx = 0.3, 512, 0.01
    x = centered_grid(n, dx)
    spectrum = ft(np.exp(-(x**2) / (2 * sigma**2)), dx)
    k = centered_grid(n, frequency_step(n, dx))
    expected = sigma / math.sqrt(2 * math.pi) * np.exp(-(sigma**2) * k**2 / 2)
    assert np.max(np.abs(spectrum - expected)) < 1e-12


def test_roundtrip_2d() -> None:
    rng = np.random.default_rng(1)
    f = rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))
    dx = 0.05
    back = ift(ft(f, dx), frequency_step(32, dx))
    assert np.max(np.abs(back - f)) < 1e-12
