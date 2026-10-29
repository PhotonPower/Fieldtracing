"""Issue #25: vector Debye PSF from Raytatouille rays against Raytatouille's FFT PSF (marker rtt).

Small NA (plano-convex singlet f = 100 mm, NA 0.05, spherical aberration; Raytatouille
`m6/planoconvex_f10`): Strehl ratio and PSF profile agree with `psf`/`psf_at`. High NA (ideal
lens, NA 0.9): the differences are explained (docs/vergleich-psf.md): Raytatouille maps the pupil
affinely onto the direction sines with the slope tan θ_max of the collinear lens, so its PSF is
narrower by tan θ/sin θ; the vector field is elongated along the polarization.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fieldtracing._types import ComplexArray, FloatArray
from fieldtracing.focus import debye_focus
from fieldtracing.rtt_bridge import debye_psf

pytestmark = pytest.mark.rtt


def _rt_psf(system: Any, points: FloatArray, grid: int = 256) -> tuple[FloatArray, float]:
    import raytatouille as rt

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pupil.mapping_nonlinear at high NA
        pupil = rt.analysis.pupil_function(system, field=0, grid=grid)
        strehl = float(rt.analysis.psf(system, field=0, grid=grid).strehl)
    return np.asarray(rt.analysis.psf_at(pupil, points)), strehl


def _fwhm(x: FloatArray, y: FloatArray) -> float:
    half = y[0] / 2.0
    i = int(np.argmax(y < half))
    return float(2.0 * (x[i - 1] + (half - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])))


def test_small_na_strehl_and_profile(rtt_reference_dir: Path) -> None:
    import raytatouille as rt

    system = rt.compile(rt.load(rtt_reference_dir / "m6" / "planoconvex_f10.rtt.json"))
    focal, strehl = debye_psf(system, exit_z=10.0, rings=40, n=128, pad=4)
    x = focal.x
    c = len(x) // 2
    airy = 0.61 * 0.5876e-3 / 0.05
    js = np.arange(int(3 * airy / (x[1] - x[0])))
    pts = np.array([[x[c + j], 0.0] for j in js] + [[0.0, x[c + j]] for j in js])
    ref, rt_strehl = _rt_psf(system, pts)
    assert 0.7 < rt_strehl < 0.8  # about 0.4 waves of spherical aberration
    assert strehl == pytest.approx(rt_strehl, abs=1e-3)
    intensity = np.sum(np.abs(focal.values) ** 2, axis=-1)
    ours = np.array([intensity[c, c + j] for j in js] + [intensity[c + j, c] for j in js])
    assert np.max(np.abs(ours * strehl / intensity[c, c] - ref)) < 5e-4  # Strehl units


def _ideal_lens(tmp_path: Path, na: float, f: float = 2.0) -> Any:
    import raytatouille as rt

    a = f * np.tan(np.arcsin(na))
    circle = {"type": "circular", "radius": a}
    stop = {"type": "stop", "name": "stop", "surfaces": [{"id": "STO", "aperture": circle}]}
    lens = {
        "id": "IL",
        "aperture": {"type": "circular", "radius": 1.2 * a},
        "interaction": {"type": "ideal_lens", "focal_length": f},
    }
    image = {"id": "IMG"}
    model = {
        "schema_version": "0.7.0",
        "name": f"ideal lens NA {na}",
        "units": {"length": "mm", "wavelength": "um"},
        "environment": {"medium": "VACUUM"},
        "wavelengths": [{"um": 0.5, "reference": True}],
        "aperture": {"type": "epd", "value": 2 * a},
        "fields": {"points": [{}]},
        "root": {
            "type": "assembly",
            "name": "system",
            "children": [
                stop,
                {
                    "type": "thin_element",
                    "name": "IL",
                    "pose": {"position": [0, 0, 1.0]},
                    "surfaces": [lens],
                },
                {
                    "type": "detector",
                    "name": "image",
                    "pose": {"position": [0, 0, 1.0 + f]},
                    "surfaces": [image],
                },
            ],
        },
        "paths": [{"name": "main", "events": "auto"}],
    }
    path = tmp_path / "ideal.rtt.json"
    path.write_text(json.dumps(model))
    return rt.compile(rt.load(path))


def test_high_na_differences_are_explained(tmp_path: Path) -> None:
    na, lam, f = 0.9, 0.5e-3, 2.0
    system = _ideal_lens(tmp_path, na, f)
    focal, strehl = debye_psf(system, exit_z=1.0, rings=60, n=128, pad=8)
    assert strehl == pytest.approx(1.0, abs=1e-5)  # ideal lens: no aberration
    x = focal.x
    c = len(x) // 2
    span = int(np.argmax(x[c:] > 1.0e-3))
    r = x[c : c + span]
    inten = np.sum(np.abs(focal.values) ** 2, axis=-1)
    fw_x, fw_y = _fwhm(r, inten[c, c : c + span]), _fwhm(r, inten[c : c + span, c])
    ez = float(np.sum(np.abs(focal.values[..., 2]) ** 2) / np.sum(inten))
    assert fw_x > 1.3 * fw_y  # elongated along the polarization (x)
    assert 0.25 < ez < 0.40  # longitudinal component carries about a third of the energy
    # Raytatouille: rotationally symmetric, and as narrow as a uniform pupil of "NA" tan θ_max
    rx, _ = _rt_psf(system, np.column_stack([r, np.zeros(span)]))
    ry, _ = _rt_psf(system, np.column_stack([np.zeros(span), r]))
    fw_rt = _fwhm(r, rx)
    assert fw_rt == pytest.approx(_fwhm(r, ry), rel=1e-6)

    def uniform_in_sines(u: FloatArray, v: FloatArray) -> ComplexArray:
        return np.asarray((f / np.sqrt(f * f + u * u + v * v)) ** 2, dtype=np.complex128)

    uniform = debye_focus(uniform_in_sines, f, 2 * np.pi / lam, na, n=256, pad=8)
    cu = len(uniform.x) // 2
    prof = np.abs(uniform.values[cu, cu:]) ** 2
    fw_uniform = _fwhm(uniform.x[cu:], prof)  # scalar, uniform in direction sines up to NA
    tan_over_sin = 1.0 / np.cos(np.arcsin(na))
    assert fw_rt * tan_over_sin == pytest.approx(fw_uniform, rel=0.03)
