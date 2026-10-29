"""Section 7 of the paper: figures of merit of the worked example (D = 25 mm, NA 0.5, 500 nm)."""

from __future__ import annotations

import math

import pytest

from fieldtracing.selection import (
    edge_wave_ratio,
    fresnel_width,
    local_fresnel_number,
    min_edge_distance,
    nyquist_samples,
    sft_gain,
    sft_step,
)
from fieldtracing.units import wavenumber

K = wavenumber(0.5)
A = 12.5
F = A / math.tan(math.asin(0.5))


def test_example_figures() -> None:
    h = K / F
    w = fresnel_width(h)
    assert F == pytest.approx(21.65, abs=0.01)
    assert w * 1e3 == pytest.approx(41.5, abs=0.1)  # µm
    assert local_fresnel_number(h, A) == pytest.approx(2 * math.pi * A**2 / (0.5e-3 * F))
    assert min_edge_distance(w, 0.01) == pytest.approx(1.656, abs=0.01)
    assert edge_wave_ratio(w, min_edge_distance(w, 0.01)) == pytest.approx(0.01)


def test_sft_example_numbers() -> None:
    step = sft_step(-521.0, 10.0, 230.0)
    assert step * 1e3 == pytest.approx(6.2, abs=0.05)  # µm, limited by w_c/q
    assert 30.0 / step == pytest.approx(4840, rel=0.01)
    assert nyquist_samples(30.0, K * 0.5) == pytest.approx(60000, rel=1e-3)
    assert sft_gain(1.0e4, 10.0) == pytest.approx(math.sqrt(2e4) / (10 * math.pi))
