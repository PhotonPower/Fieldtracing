# Changelog

Format nach [Keep a Changelog](https://keepachangelog.com/de/1.1.0/), Versionen nach SemVer.

## [Unreleased]

### Hinzugefügt

- Projektgerüst: Dokumentation, Plan (`docs/roadmap.md`), Konventionen, ADR 0001–0003, CI
  (ruff, mypy, pytest unter Linux und Windows, LaTeX-Build des Papers).
- Python-Referenzbausteine mit Tests: Fourier-Konvention, Winkelspektrum, Rayleigh-Sommerfeld- und
  Van-Vleck-Kern, geometrische Propagation mit Jacobi-Faktor und Maslov-Index (1D/2D), PFT (1D/2D),
  SFT (1D, 2D diagonal), Kantenformeln, Fokuskette mit sphärischer Phase, Fresnel-Operator
  (Convention A), Kennzahlen und Auswahlregeln aus Kapitel 7.
- Paper `docs/paper/fieldtracing.tex`.

- `focus.debye_focus`: Fokusfeld nach Gl. (debye) per FFT mit subpixelgewichteter Berandung,
  skalar und vektoriell; Tests gegen Airy, geschlossene Defokus-Form, Rayleigh-Sommerfeld (T4) und
  Richards-Wolf (#1).

- `sft.sft_2d` und `sft.inverse_sft_2d` für allgemeine symmetrische Krümmungsmatrizen,
  `check_sft_sampling` mit `SamplingWarning`, `region_weights` für beliebige Blendenformen;
  pytest-Marker `slow` und nächtlicher Workflow (#2).

### Geändert

- Entscheidungen E1–E5 und ADR 0001–0003 angenommen (`docs/roadmap.md`).
