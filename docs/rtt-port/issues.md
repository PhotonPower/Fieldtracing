# Issue-Entwürfe für den Port nach Raytatouille

> Entwürfe aus Fieldtracing (#53) im Format von Raytatouilles Vorlage
> `.github/ISSUE_TEMPLATE/aufgabe.md` bzw. `frage.md`. Nicht dort angelegt (Fieldtracing,
> AGENTS.md Regel 8); Meilenstein-Kürzel, Nummern und Reihenfolge legt der Maintainer fest
> („Mx“ steht als Platzhalter). Grundlage: ADR-Entwurf `adr-entwurf-rtt-field.md`, API-Skizze
> `api-skizze.md`, Sollwertdateien `reference/field/` (Fieldtracing-Commit beim Übernehmen
> angeben).

## Reihenfolge

| Nr. | Entwurf | braucht | Integrationspunkt |
| --- | --- | --- | --- |
| 0 | Frage: ADR `rtt-field` annehmen | – | – |
| 1 | Frage: Delaunay, Interpolation und KD-Baum in C++ | 0 | – |
| 2 | `rtt-field`: Gerüst, Typen, FFT-Kapselung, Sollwert-Leser | 0 | – |
| 3 | `rtt-field`: Fourier und Winkelspektrum | 2 | I1, I2 |
| 4 | `rtt-field`: Maschen-Jacobi auf Strahlwolken | 1, 2 | I1 |
| 5 | `rtt-field`: Feldamplitude aus P und Gewicht, Feld aus Strahlen | 4 | I1 |
| 6 | `rtt-field`: Debye-Fokus und Richards-Wolf | 3, 5 | I1 |
| 7 | `rtt-field`, `rtt-py`: vektorielle Debye-PSF (I1) und `raytatouille.field` | 6 | **I1** |
| 8 | `rtt-field`: PFT, SFT, geometrische Propagation | 3 | I2 |
| 9 | `rtt-field`: Operatoren, Auswahlregeln und Planer | 8 | **I2** |
| 10 | `rtt-field`: Kantenbeugung (Randintegral, kritische Punkte) | 8 | I2 |
| 11 | `rtt-field`: hybride Feldverfolgung durch das System | 9 | **I3** |

---

## 0 · [Frage] ADR `rtt-field` annehmen

**Kontext.** Fieldtracing hat die Bausteine des Field Tracing in Python geprüft und einen
ADR-Entwurf geschrieben (`docs/rtt-port/adr-entwurf-rtt-field.md` dort).

**Frage.** Wird das ADR wie vorgeschlagen angenommen (Bibliothek `rtt-field` in der Schicht
Auswertung ohne Abhängigkeit von `rtt-analysis`, erster Integrationspunkt vektorielle
Debye-PSF aus Strahlen)?

**Optionen.** (a) wie vorgeschlagen; (b) `rtt-field` über `rtt-analysis` mit Jones-Pupille als
Eingang (übernimmt die affine Pupillenabbildung, bei hoher NA ungeeignet, siehe
`docs/vergleich-psf.md` in Fieldtracing); (c) zurückstellen bis nach M7.

**Betroffene Aufgabe.** Alle folgenden.

---

## 1 · [Frage] Delaunay, Interpolation und KD-Baum in C++

**Kontext.** Fieldtracing nutzt SciPy: `Delaunay` (Maschen der Strahlstartpunkte), lineare und
Clough-Tocher-Interpolation auf der Triangulierung (nur das Restfeld U), `cKDTree` (nächste Probe
für die Taylor-Phase), `linprog` (Minimax-Krümmung, 4 Unbekannte), `scipy.special.jv`.

**Frage.** Welche Bibliotheken oder Eigenimplementierungen?

**Optionen.**
- Delaunay: eigene Bowyer-Watson-Implementierung (≈ 300 Zeilen, deterministisch) oder eine
  Header-Bibliothek (z. B. delaunator-cpp, MIT) über vcpkg; Randsplitter-Abschälen und
  Lochfilter wie in Fieldtracing `mesh.triangulate`.
- Interpolation: baryzentrisch auf den Dreiecken (linear) reicht für den Port; Clough-Tocher nur
  für Prüfungen.
- KD-Baum: nanoflann (BSD, header-only) oder Gitter-Bucketing.
- Minimax: geschlossene Lösung über die aktiven Nebenbedingungen (Simplex auf 4 Variablen).
- Bessel: `std::cyl_bessel_j` (C++17, MSVC vorhanden).

**Betroffene Aufgabe.** 4, 6, 9.

---

## 2 · [Mx] rtt-field: Gerüst, Typen, FFT-Kapselung, Sollwert-Leser

## Ziel

`libs/rtt-field` existiert mit den Typen der API-Skizze (`GridlessField`, `GriddedField`,
`SampleStatus`), einer gekapselten pocketfft-Übersetzungseinheit und einem Testhelfer, der
Sollwertdateien `FieldReference` mit `rtt-json` liest.

## Meilenstein und Bibliothek

- Meilenstein: Mx
- Bibliothek: `libs/rtt-field` (neu), Tests unter `libs/rtt-field/tests`, Sollwerte unter
  `tests/reference/field/` (aus Fieldtracing übernommen, Commit im Changelog-Fragment)

## Schnittstellen (nur lesen, nicht ändern)

`rtt/math`, `rtt/diagnostics`, `rtt-json`, Target `rtt::pocketfft` (ADR 0033 Punkt 1).

## Referenzfälle

- Leser: alle Dateien aus `tests/reference/field/` laden; Arrays mit dtype und shape; NaN/±∞ und
  komplexe Werte wie ADR 0023.

## Abnahmekriterien

- [ ] Bibliothek in der Schicht Auswertung, CMake-Target `rtt::field`, keine Abhängigkeit von
  `rtt-analysis`
- [ ] FFT-Kapselung deterministisch, eine Übersetzungseinheit
- [ ] Definition of Done aus AGENTS.md erfüllt

## Nicht Teil dieser Aufgabe

Physik-Bausteine (ab Aufgabe 3), Python-Bindungen (Aufgabe 7).

---

## 3 · [Mx] rtt-field: Fourier und Winkelspektrum

## Ziel

`fourier.hpp` und `propagation.hpp` rechnen die Fourier-Konvention (2π)^−d auf zentrierten
Gittern und die Winkelspektrum-Propagation (Im k_z ≥ 0) wie Fieldtracing.

## Meilenstein und Bibliothek

- Meilenstein: Mx; Bibliothek: `libs/rtt-field`

## Schnittstellen (nur lesen, nicht ändern)

Aufgabe 2.

## Referenzfälle

- `fourier_gaussian.result.json`: `expected` rel 1e−12; `reference` (Gauß analytisch) abs 1e−15.
- `angular_spectrum_tilted_wave.result.json`: `expected` rel 1e−10; `reference` (ebene Welle
  exakt) abs 1e−10.
- Rayleigh-Sommerfeld- gegen Van-Vleck-Kern: Verhältnis 1 + i/(kr) (Fieldtracing T1).

## Abnahmekriterien

- [ ] Referenzfälle grün, Fourier-Konvention im Header mit Gleichung dokumentiert
- [ ] Definition of Done aus AGENTS.md erfüllt

## Nicht Teil dieser Aufgabe

SFT (Aufgabe 8).

---

## 4 · [Mx] rtt-field: Maschen-Jacobi auf Strahlwolken

## Ziel

Aus Start- und Landepunkten beliebig angeordneter Strahlen entstehen ∂ρ₂/∂ρ₁, J, Maslov-Index und
Kaustik-Flags (Fieldtracing `mesh`), mit Lochfilter für Obskurationen und Abschälen der
Randsplitter.

## Meilenstein und Bibliothek

- Meilenstein: Mx; Bibliothek: `libs/rtt-field` (`mesh.hpp`)

## Schnittstellen (nur lesen, nicht ändern)

Aufgabe 2; Entscheidung aus Frage 1.

## Referenzfälle

- `mesh_jacobian_jittered.result.json`: `expected` (Determinante, Matrix, Maslov) rel 1e−10
  **bei gleicher Triangulierung** (die Datei enthält die Dreiecke; eine andere, ebenfalls gültige
  Delaunay-Zerlegung ändert die Werte im Rahmen von `reference`); `reference` (analytisches J)
  rel 5e−4.
- Konvergenz O(h²) auf verrauschten Gittern (Fieldtracing `tests/test_mesh.py`: 41² → 81²
  Verhältnis 3,1–3,5).
- Ringblende: kein Dreieck über dem Loch.

## Abnahmekriterien

- [ ] Referenzfälle grün, Ordnung belegt
- [ ] Definition of Done aus AGENTS.md erfüllt

## Nicht Teil dieser Aufgabe

Gleichmäßige Lösungen an Kaustiken.

---

## 5 · [Mx] rtt-field: Feldamplitude aus P und Gewicht, Feld aus Strahlen

## Ziel

`rays.hpp`: `field_operator` nach Fieldtracing ADR 0004 und `field_from_rays`/`trace_field`
bauen aus einem verfolgten `RayBatch` ein `GridlessField` auf einer Ebene (Phase k₀·OPL,
Gradient aus der Richtung, Amplitude mit Maschen-Jacobi, Hesse-Matrix aus Nachbarstrahlen).

## Meilenstein und Bibliothek

- Meilenstein: Mx; Bibliothek: `libs/rtt-field`

## Schnittstellen (nur lesen, nicht ändern)

`rtt/trace` (`RayBatch`, `make_rays`, `trace`, Pupillenabtastungen), `rtt/paraxial`
(`first_order`: absolute Indizes), `rtt/compile`; Aufgabe 4.

## Referenzfälle

- `field_operator_plate.result.json`: `expected` rel 1e−12; `reference` (√s·T₂T₁) abs 1e−14.
- Ebene Einzellinse `m1/singlet_const`: Phase gegen `opd_points` < 1e−8 Wellen; Leistung über
  die Lande-Dreiecke gegen `polar.transmission` rel 3e−4 (Fieldtracing
  `tests/test_rtt_adapter.py`).
- Fresnel-Platte `m3/fresnel_bk7` bei 0°, 20°, Brewster, 70°: Feldoperator = T₂T₁ < 1e−13.

## Abnahmekriterien

- [ ] Referenzfälle grün; vignettierte Strahlen als `SampleStatus::OutsideStop`
- [ ] Definition of Done aus AGENTS.md erfüllt

## Nicht Teil dieser Aufgabe

Felder aus Punktquellen (nicht ebene Eingangswellen).

---

## 6 · [Mx] rtt-field: Debye-Fokus und Richards-Wolf

## Ziel

`focus.hpp`: Debye-Integral per FFT mit subpixelgewichteter k-Scheibe (skalar und vektoriell),
Debye aus einem `GridlessField` (Referenzkugel auf der Zielebene, Restfeld am Rand der Wolke
nächster Strahl), Richards-Wolf-Referenz, Airy-Integral, Fehlerterm der periodischen Kopien.

## Meilenstein und Bibliothek

- Meilenstein: Mx; Bibliothek: `libs/rtt-field`

## Schnittstellen (nur lesen, nicht ändern)

Aufgaben 3, 5.

## Referenzfälle

- `debye_airy.result.json`: `expected` rel 1e−12; `reference` (Airy) abs 3e−4 des Maximums.
- `richards_wolf_focus.result.json`: Richards-Wolf rel 1e−9; Debye an Gitterpunkten gegen
  Richards-Wolf abs 1e−4 des Maximums (NA 0,9).
- Defokus auf der Achse gegen die geschlossene Form < 4,6e−4 (Fieldtracing `test_debye.py`).

## Abnahmekriterien

- [ ] Referenzfälle grün
- [ ] Definition of Done aus AGENTS.md erfüllt

## Nicht Teil dieser Aufgabe

Rayleigh-Sommerfeld für kleine Fresnel-Zahlen (Aufgabe 9).

---

## 7 · [Mx] rtt-field, rtt-py: vektorielle Debye-PSF aus Strahlen (I1) und `raytatouille.field`

## Ziel

`debye_psf(system, exit_z, …)` liefert die vektorielle PSF des Achsfeldes mit Strehl-Zahl;
Python-Modul `raytatouille.field` mit `debye_psf`, `trace_field`, `field_operator` und dem
Ergebnistyp `FieldReference` in `raytatouille.results` (Schema 0.1.14). Schließt H7 aus ADR 0033.

## Meilenstein und Bibliothek

- Meilenstein: Mx; Bibliotheken: `libs/rtt-field`, `libs/rtt-py`

## Schnittstellen (nur lesen, nicht ändern)

Aufgaben 5, 6; `raytatouille.results` (ADR 0023) wird um einen Typ erweitert.

## Referenzfälle

- `m6/planoconvex_f10` (NA 0,05): Strehl gegen `psf` auf 1e−3 (Fieldtracing: 0,73820 gegen
  0,73815), Profil gegen `psf_at` < 5e−4 Strehl-Einheiten bis 3 Airy-Radien.
- Ideale Linse NA 0,9: Strehl 1 (< 1e−5); FWHM längs der Polarisation > 1,3 × quer; E_z-Anteil
  25–40 %.
- Python gegen C++ bitgleich (Muster `*_cases.cpp`).

## Abnahmekriterien

- [ ] Referenzfälle grün, Diagnosecodes `field.*` in der Registry
- [ ] Definition of Done aus AGENTS.md erfüllt

## Nicht Teil dieser Aufgabe

Außeraxiale Felder (Referenzkugel um den Hauptstrahl, eigene Aufgabe); Korrektur der FFT-PSF bei
nichtlinearer Pupillenabbildung (Issue-Entwurf `docs/rtt-entwurf-psf-hohe-na.md` in Fieldtracing).

---

## 8 · [Mx] rtt-field: PFT, SFT, geometrische Propagation

## Ziel

`pft.hpp`, `sft.hpp`, `geometric.hpp` wie Fieldtracing: PFT/inverse PFT in 2D, SFT für
allgemeines symmetrisches C mit inverser SFT auf dem Gitter k = C·ρ′, Abtastprüfung,
Blendengewichte; geometrische Propagation mit Jacobi-Faktor, Maslov-Index und neuer
Hesse-Matrix (H₂ = H₁(I + d·D·H₁)⁻¹).

## Meilenstein und Bibliothek

- Meilenstein: Mx; Bibliothek: `libs/rtt-field`

## Schnittstellen (nur lesen, nicht ändern)

Aufgaben 2, 3.

## Referenzfälle

- `pft_sphere.result.json`: rel 1e−12; `reference` (Legendre, |det H|^(−½)) abs 1e−9.
- `sft_gauss_chirp.result.json`: `expected` rel 1e−10; `reference` (geschlossene Form)
  abs 1e−6 des Maximums.
- `geometric_propagation_astigmatic.result.json`: rel 1e−12 (vor und zwischen den Brennlinien).
- SFT hin und zurück < 1e−10; PFT → Winkelspektrum → inverse PFT = geometrische Propagation
  < 1e−10 (Fieldtracing `tests/test_chain.py`).

## Abnahmekriterien

- [ ] Referenzfälle grün
- [ ] Definition of Done aus AGENTS.md erfüllt

## Nicht Teil dieser Aufgabe

Gemischte Karten (PFT in einer Richtung, Ort in der anderen).

---

## 9 · [Mx] rtt-field: Operatoren, Auswahlregeln und Planer (I2)

## Ziel

`chain.hpp`/`selection.hpp`: Operatoren mit Protokoll, Auswahlregeln nach Tabelle tab:selection,
Planer mit Fehlerbudget nach Tabelle tab:errors (Kantenkonstante 0,4/q, Achszone der
Kreisblende, Taylor-Phase, periodische Kopien), Felder in beliebigen Ebenen.

## Meilenstein und Bibliothek

- Meilenstein: Mx; Bibliothek: `libs/rtt-field`

## Schnittstellen (nur lesen, nicht ändern)

Aufgaben 6, 8; Entscheidung aus Frage 1 (Minimax).

## Referenzfälle

- `selection_rules.result.json`: Methoden exakt.
- Linse → Kreisblende → Ebene f/2 und Fokus gegen direkte Rayleigh-Sommerfeld-Quadratur im
  Budget außerhalb der Achszone (Fieldtracing `tests/test_plan.py`, `tests/test_example_7_5.py`).
- Beispiel 7.5: Minimax-c = −521 mm⁻², Δx = 6,19 µm, 5234 Punkte; Fokus gegen Richards-Wolf
  6,5e−4.

## Abnahmekriterien

- [ ] Referenzfälle grün, Plan als Ergebnis mit `to_dict`
- [ ] Definition of Done aus AGENTS.md erfüllt

## Nicht Teil dieser Aufgabe

Gemischte Karte (Planer meldet sie, ohne Operatorkette).

---

## 10 · [Mx] rtt-field: Kantenbeugung

## Ziel

`edge.hpp`: gleichmäßige erfc-Lösung (1D), Randintegral (Gl. bdw) mit Knoten für Polygone und
Kreise, kritische Punkte 2. und 3. Art.

## Meilenstein und Bibliothek

- Meilenstein: Mx; Bibliothek: `libs/rtt-field`

## Referenzfälle

- Fieldtracing T5 (`test_edge.py`) und #5 (`test_edge_2d.py`): Kreisscheibe gegen
  Hankel-Quadratur < 1e−12 (Randintegral), Ordnungen k^(−½) bzw. k^(−1).

## Abnahmekriterien

- [ ] Referenzfälle grün
- [ ] Definition of Done aus AGENTS.md erfüllt

## Nicht Teil dieser Aufgabe

Gleichmäßige Lösung nahe der Schattengrenze in 2D (in Fieldtracing offen).

---

## 11 · [Mx] rtt-field: hybride Feldverfolgung durch das System (I3)

## Ziel

Ein Feld wird durch ein sequenzielles System verfolgt: Strahlen von Raytatouilles Tracer liefern
Phase, P und Gewicht; an Blenden und Zwischenebenen wechselt der Planer zu SFT/FFT und zurück
(Gl. system_chain).

## Meilenstein und Bibliothek

- Meilenstein: Mx; Bibliothek: `libs/rtt-field`

## Schnittstellen (nur lesen, nicht ändern)

Aufgaben 5, 9; `rtt/trace`.

## Referenzfälle

- Einzellinse aus Strahlen gegen die LPIA-Kette in Fieldtracing (`test_rtt_adapter.py`):
  Landepunkte < 1e−12 mm, Phase < 1e−8 rad, Amplitude 1,4e−4.
- System mit Zwischenblende gegen Winkelspektrum auf feinem Gitter.

## Abnahmekriterien

- [ ] Referenzfälle grün
- [ ] Definition of Done aus AGENTS.md erfüllt

## Nicht Teil dieser Aufgabe

Nicht sequenzielle Pfade, Geisterbilder, Kristalle.
