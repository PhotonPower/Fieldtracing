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

- Raytatouille als optionale Abhängigkeit (#22): Commit-Pin `rtt-commit.txt`, pytest-Marker
  `rtt` (ohne Paket übersprungen, mit `FIELDTRACING_REQUIRE_RTT=1` ein Fehler), Workflow
  `rtt.yml` mit vcpkg- und Rad-Cache, Rauchtest an der Einzellinse aus Raytatouilles
  Referenzmodellen.

- `rtt_bridge.field_operator`, `transverse_prt`, `absorption_factor`: Feldamplitude aus
  Raytatouilles leistungsnormierter PRT-Matrix und Gewicht, Herleitung in ADR 0004 (#23).

- `rtt_bridge.field_from_rays` und `rtt_bridge.trace_field`: hybrides Feld auf einer Ebene aus
  Raytatouille-Strahlen (Phase k₀·OPL, k⊥ aus der Richtung, Amplitude nach ADR 0004 mit
  Maschen-Jacobi, Hesse-Matrix aus den Nachbarstrahlen; vignettierte Strahlen ausgeschlossen);
  `field_operator` mit wählbaren Ebenennormalen (#24).

- `rtt_bridge.debye_psf`: vektorielle Debye-PSF des Achsfeldes aus Raytatouille-Strahlen mit
  Strehl; Vergleich mit Raytatouilles FFT-PSF in `docs/vergleich-psf.md` (Bild aus
  `experiments/vergleich_psf.py`), Issue-Entwurf für Raytatouille in
  `docs/rtt-entwurf-psf-hohe-na.md` (#25).
- `GridlessField.residual_at(..., outside="nearest")` (#25).

### Behoben

- `semiclassical.propagate_2d`: Die Strahlabbildung ∂ρ₂/∂ρ₁ wurde transponiert gebildet
  (H·∂(p/k_z)/∂p statt ∂(p/k_z)/∂p·H). J und μ blieben richtig, die neue Hesse-Matrix aus #17
  aber nicht, sobald H und die Strahlrichtungen nicht vertauschen (gekippte astigmatische Phase,
  relativer Fehler ≈ 4·10⁻⁴); sphärische Wellen waren nicht betroffen. Gefunden beim Vergleich mit
  der Maschen-Jacobi-Matrix in #19, Regressionstest mit zentralen Differenzen.

- `GridlessField.aplanatic` (Austrittsfeld einer aplanatischen Linse, vektoriell, Sinusbedingung)
  und `focus.richards_wolf` (Referenz nach Richards und Wolf, mit Defokus) (#39).

- Beispiel 7.5, Fokus (#40): `tests/test_example_7_5.py`, Bericht `docs/beispiel-7-5.md`;
  `chain.airy_alias_error`; `plan_propagation` mit `focus_window` und den Budgettermen
  `phase_taylor` und `aliasing`, `focus_samples` standardmäßig aus dem Fenster.

- Beispiel 7.5, verkleinertes System gegen direkte Rayleigh-Sommerfeld-Quadratur an Fokus und
  Zwischenebene (#42).

- Beispiel 7.5, Zwischenebene (#41): Plan des großen Systems gegen das Paper, Skalierung mit
  der Apertur (`experiments/beispiel_7_5_zwischenebene.py`), Vergleich mit der Geometrie;
  `plan_propagation` gibt für die SFT Krümmung, Restgradient und Speicherbedarf aus.

- Sollwertdateien für den Port (#50): `fieldtracing.results` (Kodierung nach Raytatouilles
  ADR 0023, Typ-Vorschlag `FieldReference`), `tools/reference_cases.py`,
  `tools/export_reference.py`, 12 Dateien unter `reference/field/`, Aktualitätstest.

### Geändert

- Paper, Abschnitt 7.5 (#43): periodische Kopien der Fokus-FFT (N ≈ 80 statt 40 für 10⁻³),
  nachgerechnete Punktzahlen und Speicher der Zwischenebene, Fehlerbudget mit Messwerten,
  Fresnel-Zahl des verkleinerten Systems (≈ 1440 statt 144) und Ergebnisse des Integrationstests.

- `chain.debye` nimmt am Rand der k-Scheibe den nächsten Strahlwert statt 0; der Rand der
  Scheibe kommt allein aus den Subpixel-Gewichten. Vorher konvergierte der Strehl-Wert nur mit
  erster Ordnung in der k-Schrittweite (#25).

- Paper, Kapitel 7 (#31): Kantenfehler auf 2D-Gittern ≲ 0,4/q, Verstärkung auf der Achse
  rotationssymmetrischer Blenden (Kaustik der Kantenwelle) in Tabelle tab:errors,
  Untergrenze N_SFT ≳ πq²/m im Abschnitt Aufwand, Fehlerbudget des Beispiels 2–4 % mit Achszone.

- Paper: Gültigkeitsgrenze der Debye-Näherung quantifiziert (≈ 1,5/N_F statt N_F ≲ 10),
  Abschnitt 5.3, Tabellen 7.1 und 7.3, Testliste in Kapitel 8 (#8).

- Entscheidungen E1–E5 und ADR 0001–0003 angenommen (`docs/roadmap.md`).
