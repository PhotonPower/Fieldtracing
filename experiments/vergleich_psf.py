"""Issue #25: vector Debye PSF from Raytatouille rays against Raytatouille's scalar FFT PSF.

Needs raytatouille (commit in rtt-commit.txt) and RAYTATOUILLE_SOURCE. Writes the numbers of
docs/vergleich-psf.md to stdout and the figure to docs/img/vergleich-psf.png.

    RAYTATOUILLE_SOURCE=../raytatouille python experiments/vergleich_psf.py
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import raytatouille as rt

from fieldtracing.focus import debye_focus
from fieldtracing.rtt_bridge import debye_psf

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path(os.environ["RAYTATOUILLE_SOURCE"]) / "tests" / "reference"


def ideal_lens(na: float, f: float = 2.0, lam: float = 0.5) -> tuple[rt.CompiledSystem, str]:
    """Ideal (collinear) lens f in vacuum at z = 1, stop for NA in image space, x-polarised."""
    a = f * np.tan(np.arcsin(na))
    model = {
        "schema_version": "0.7.0",
        "name": f"ideal lens NA {na}",
        "units": {"length": "mm", "wavelength": "um"},
        "environment": {"medium": "VACUUM"},
        "wavelengths": [{"um": lam, "reference": True}],
        "aperture": {"type": "epd", "value": 2 * a},
        "fields": {"points": [{}]},
        "root": {
            "type": "assembly",
            "name": "system",
            "children": [
                {
                    "type": "stop",
                    "name": "stop",
                    "surfaces": [{"id": "STO", "aperture": {"type": "circular", "radius": a}}],
                },
                {
                    "type": "thin_element",
                    "name": "IL",
                    "pose": {"position": [0, 0, 1.0]},
                    "surfaces": [
                        {
                            "id": "IL",
                            "aperture": {"type": "circular", "radius": 1.2 * a},
                            "interaction": {"type": "ideal_lens", "focal_length": f},
                        }
                    ],
                },
                {
                    "type": "detector",
                    "name": "image",
                    "pose": {"position": [0, 0, 1.0 + f]},
                    "surfaces": [{"id": "IMG"}],
                },
            ],
        },
        "paths": [{"name": "main", "events": "auto"}],
    }
    path = Path(tempfile.mkdtemp()) / "ideal.rtt.json"
    path.write_text(json.dumps(model))
    return rt.compile(rt.load(path)), str(path)


def rt_psf(
    system: rt.CompiledSystem, points: np.ndarray, grid: int = 256
) -> tuple[np.ndarray, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pupil = rt.analysis.pupil_function(system, field=0, grid=grid)
        strehl = rt.analysis.psf(system, field=0, grid=grid).strehl
    return np.asarray(rt.analysis.psf_at(pupil, points)), float(strehl)


def fwhm(x: np.ndarray, y: np.ndarray) -> float:
    """Full width at half maximum of a peak at x = 0 sampled on x ≥ 0 (linear interpolation)."""
    half = y[0] / 2.0
    i = int(np.argmax(y < half))
    return float(2.0 * (x[i - 1] + (half - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])))


def scalar_debye(
    amplitude: str, na: float, f: float, k: float, n: int = 256, pad: int = 8
) -> np.ndarray:
    """Scalar Debye PSF |E|² on the x axis: U = cos²θ (uniform in direction sines, as Raytatouille's
    affine pupil) or U = cos^(-1/2) θ (collinear ideal lens with flux conservation)."""

    def residual(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        cos = f / np.sqrt(f * f + x * x + y * y)
        return cos**2 if amplitude == "uniform" else cos**-0.5

    focal = debye_focus(residual, f, k, na, n=n, pad=pad)
    c = len(focal.x) // 2
    return np.asarray(np.abs(focal.values[c, c:]) ** 2)


def main() -> None:
    # --- small NA: plano-convex singlet f = 100 mm, NA 0.05 (Raytatouille m6) -------------
    singlet = rt.compile(rt.load(REFERENCE / "m6" / "planoconvex_f10.rtt.json"))
    focal, strehl = debye_psf(singlet, exit_z=10.0, rings=60, n=256, pad=4)
    x = focal.x
    c = len(x) // 2
    js = np.arange(0, 40)
    pts = np.array([[x[c + j], 0.0] for j in js] + [[0.0, x[c + j]] for j in js])
    ref, rt_strehl = rt_psf(singlet, pts)
    intensity = np.sum(np.abs(focal.values) ** 2, axis=-1)
    peak_perfect = intensity[c, c] / strehl
    ours = (
        np.array([intensity[c, c + j] for j in js] + [intensity[c + j, c] for j in js])
        / peak_perfect
    )
    print(
        f"singlet NA 0.05: Strehl ours {strehl:.5f}, Raytatouille {rt_strehl:.5f}; "
        f"max |ΔI| = {np.max(np.abs(ours - ref)):.2e} (Strehl units) up to {x[c + js[-1]] * 1e3:.1f} µm"
    )
    profiles = {"singlet": (x[c : c + 40] * 1e3, ours[:40], ref[:40])}

    # --- high NA: ideal lens --------------------------------------------------------------
    rows = []
    for na in (0.5, 0.9):
        system, _ = ideal_lens(na)
        focal, strehl = debye_psf(system, exit_z=1.0, rings=80, n=256, pad=8)
        x = focal.x
        c = len(x) // 2
        inten = np.sum(np.abs(focal.values) ** 2, axis=-1)
        ex = np.abs(focal.values[..., 0]) ** 2
        ez_share = float(np.sum(np.abs(focal.values[..., 2]) ** 2) / np.sum(inten))
        xs = x[c:]
        span = int(np.argmax(xs > 1.5e-3))  # up to 1.5 µm
        pts_x = np.column_stack([xs[:span], np.zeros(span)])
        pts_y = np.column_stack([np.zeros(span), xs[:span]])
        rx, rt_strehl = rt_psf(system, pts_x)
        ry, _ = rt_psf(system, pts_y)
        k = 2 * np.pi / 0.5e-3
        uni = scalar_debye("uniform", na, 2.0, k)[:span]
        col = scalar_debye("collinear", na, 2.0, k)[:span]
        row = {
            "na": na,
            "strehl_ours": strehl,
            "strehl_rt": rt_strehl,
            "fwhm_rt_x": fwhm(xs[:span], rx),
            "fwhm_rt_y": fwhm(xs[:span], ry),
            "fwhm_uniform": fwhm(xs[:span], uni),
            "fwhm_collinear": fwhm(xs[:span], col),
            "fwhm_vec_x": fwhm(xs[:span], inten[c, c : c + span]),
            "fwhm_vec_y": fwhm(xs[:span], inten[c : c + span, c]),
            "fwhm_ex_x": fwhm(xs[:span], ex[c, c : c + span]),
            "ez_share": ez_share,
            "max_diff_uniform_rt": float(np.max(np.abs(uni / uni[0] - rx / rx[0]))),
        }
        rows.append(row)
        profiles[f"NA {na}"] = (
            xs[:span] * 1e3,
            inten[c, c : c + span] / inten[c, c],
            rx / rx[0],
            inten[c : c + span, c] / inten[c, c],
            col / col[0],
        )
        print({k_: round(v, 5) if isinstance(v, float) else v for k_, v in row.items()})

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    r, o, s = profiles["singlet"]
    axes[0].plot(r, s, "k-", label="Raytatouille FFT-PSF (psf_at)")
    axes[0].plot(r, o, "C1--", label="Debye (Fieldtracing)")
    axes[0].set(title="Einzellinse NA 0,05", xlabel="x (µm)", ylabel="I (Strehl-Einheiten)")
    for ax, key in zip(axes[1:], ("NA 0.5", "NA 0.9"), strict=True):
        r, vx, rx, vy, col = profiles[key]
        ax.plot(r, rx, "k-", label="Raytatouille (skalar)")
        ax.plot(r, col, "C2:", label="skalar, kollineare Apodisation")
        ax.plot(r, vx, "C1--", label="vektoriell, x (parallel zur Pol.)")
        ax.plot(r, vy, "C0-.", label="vektoriell, y (senkrecht zur Pol.)")
        ax.set(
            title=f"ideale Linse {key.replace('.', ',')}", xlabel="Abstand (µm)", ylabel="I/I(0)"
        )
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ROOT / "docs" / "img" / "vergleich-psf.png"
    fig.savefig(out, dpi=120)
    print("figure:", out)


if __name__ == "__main__":
    main()
