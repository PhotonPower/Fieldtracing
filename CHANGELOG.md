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

- `selection.fit_curvature_radial`, `selection.fit_curvature` (Minimax als lineares Programm) und
  `selection.sft_samples`; reproduziert die Werte aus Abschnitt 7.5 des Papers (#3).

- `pft.inverse_pft_1d/2d`, `pft.pft_1d_branches` (mehrere stationäre Punkte) und
  `AsymptoticWarning` bei kleiner lokaler Fresnel-Zahl (#4).

- Tests für den Maslov-Index in 2D (Punktfokus, astigmatische Brennlinien) gegen das
  Winkelspektrum, mit Konvergenz O(1/k) als `slow` (#7).

- `edge.boundary_integral` (Gl. bdw) mit Knoten für Polygone und Kreise sowie
  `edge.polygon_asymptotics` und `edge.circle_asymptotics` über kritische Punkte 2. und 3. Art
  (#5).

- `lpia.trace_surface`, `lpia.mesh_jacobian` und `lpia.interface_amplitude`: LPIA-Schritt durch
  gekrümmte Flächen mit Jacobi-Faktor aus dem Strahlnetz (#6). Damit ist Meilenstein F1
  abgeschlossen.

- `field.GridlessField` und `field.GriddedField` mit Verhalten: Konstruktoren (sphärisch,
  quadratisch, analytisch), Phase per Taylor-Entwicklung, Interpolation nur des Residuums,
  Umtasten mit analytischer Trägerphase, Adapter für Propagation und PFT;
  `GeometricResult.curvature` liefert die neue Hesse-Matrix (#17).

- `lpia.refract_batch`, `lpia.fresnel_coefficients_batch`, `lpia.fresnel_operator_batch`:
  vektorisiert über beliebig viele Strahlen, Totalreflexion als Maske; `trace_surface` nutzt sie
  (≈ 130-mal schneller) und liefert `SurfaceTrace.total_reflection` (#21).

- `mesh.triangulate`, `mesh.triangle_area_ratio`, `mesh.fit_derivatives` und
  `mesh.mesh_jacobian_cloud`: Jacobi-Faktor, Maslov-Index und Kaustik-Erkennung aus
  unstrukturierten Strahlwolken (Delaunay, Obskurationen, quadratische Ausgleichsrechnung,
  Fehler O(h²)) (#18).

- `chain`: einheitliche Operatoren mit Protokoll (Gl. system_chain, Tabelle tab:chaining):
  `geometric_propagation` (Hesse- oder Maschen-Jacobi), `lpia_surface` auf Punktwolken,
  `aperture` (`CircularAperture`, `RectangularAperture`), `pft`/`inverse_pft`,
  `angular_spectrum` (Punktwolke, Gitter, SFT-Gitter), `sft`/`inverse_sft`, `fft`/`inverse_fft`,
  `resample`, `run_chain`; `GriddedField` mit `domain` und `lattice` (#19).

- Planer nach den Auswahlregeln (#20): `selection.Situation`, `selection.select_method`,
  Fehlerterme `debye_error`, `grid_edge_error`, `q_for_tolerance`; `chain.plan_interface`,
  `chain.plan_propagation` mit Begründung, Fehlerbudget und Achszone (`Plan`), neue Operatoren
  `chain.debye` und `chain.rayleigh_sommerfeld`. Integrationstest Linse → Kreisblende → Ebene
  gegen direkte Rayleigh-Sommerfeld-Quadratur; Abweichungen zum Paper in #31.

### Behoben

- `semiclassical.propagate_2d`: Die Strahlabbildung ∂ρ₂/∂ρ₁ wurde transponiert gebildet
  (H·∂(p/k_z)/∂p statt ∂(p/k_z)/∂p·H). J und μ blieben richtig, die neue Hesse-Matrix aus #17
  aber nicht, sobald H und die Strahlrichtungen nicht vertauschen (gekippte astigmatische Phase,
  relativer Fehler ≈ 4·10⁻⁴); sphärische Wellen waren nicht betroffen. Gefunden beim Vergleich mit
  der Maschen-Jacobi-Matrix in #19, Regressionstest mit zentralen Differenzen.

### Geändert

- Paper: Gültigkeitsgrenze der Debye-Näherung quantifiziert (≈ 1,5/N_F statt N_F ≲ 10),
  Abschnitt 5.3, Tabellen 7.1 und 7.3, Testliste in Kapitel 8 (#8).

- Entscheidungen E1–E5 und ADR 0001–0003 angenommen (`docs/roadmap.md`).
