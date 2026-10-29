# Testfälle

Nummern wie in Kapitel 8 des Papers. Status: ✅ umgesetzt und grün, ◐ teilweise, ○ offen.
Gemessene Werte stammen aus den Tests bzw. aus `experiments/pruefrechnungen_1d.py`.

| Nr. | Prüfung | Referenz | Abnahme | Status | Test |
| --- | --- | --- | --- | --- | --- |
| T1 | Van-Vleck-Kern gegen Rayleigh-Sommerfeld | analytisch: K_RS/K_VV = 1 + i/(kr) | relative Abweichung = 1/(kr) | ✅ | `test_propagation.py` |
| T2 | Jacobi-Gesetz | Winkelspektrum (exakt) | 1D: Fehler < 2·10⁻², Ordnung O(1/k) (Verhältnis 3–5 bei k × 4); 2D: \|U\| auf 0,3 % | ✅ 1D und 2D | `test_semiclassical.py` |
| T3 | exakte Phasenkompensation | analytisch | Rest < 10⁻⁶ rad; parabolisch + exaktes k_z: −2819,45 rad | ✅ | `test_focus.py` |
| T4 | Debye gegen Rayleigh-Sommerfeld, N_F von 3 bis 10³ | exaktes RS-Integral auf der Achse (radiale Quadratur) | Abweichung fällt wie 1/N_F, Fokusverschiebung zur Linse, im Debye-Integral null | ✅ auf der Achse | `test_debye.py` |
| T5 | gleichmäßige Kantenformel | direkte Quadratur | relativer Fehler < 1 % für \|x_s − x_e\| ≤ 0,05 mm; Schattengrenze ½; Übergang in die Kantenwelle auf 0,5 % | ✅ 1D | `test_edge.py` |
| T6 | SFT | direkte Quadratur (1D); geschlossene Formen (2D: Gauß × Chirp, gedrehtes Rechteck über Fresnel-Integrale) | siehe Tabellen unten | ✅ 1D und 2D, allgemeines C | `test_sft.py`, `test_sft_2d.py` |
| T7 | Maslov-Index | Winkelspektrum | Brennlinie (1D): μ = 1, Phase −π/2; Punktfokus (2D): μ = 2; astigmatisch: μ = 1 zwischen, μ = 2 hinter den Brennlinien | ✅ 1D und 2D | `test_semiclassical.py`, `test_maslov_2d.py` |
| T8 | Fresnel-Operator | analytisch | R + T = 1, r_p = −r_s bei 0°, r_p = 0 bei Brewster, Transversalität, Snellius | ✅ | `test_lpia.py` |
| B | Beispiel Kapitel 7.5 | verkleinertes System gegen RS; Kennzahlen | Kennzahlen wie im Paper | ◐ Kennzahlen, Minimax-Krümmung | `test_selection.py`, `test_curvature_fit.py` |

## T6: SFT-Fehler (1D, c = 10⁵ mm⁻², a = 1 mm, Fenster 4 mm)

| N | Δx | harte Blende | subpixelgewichtet |
| --- | --- | --- | --- |
| 4096 | 0,98 µm | 1,86 % | 0,58 % |
| 16384 | 0,24 µm | 0,46 % | 0,18 % |
| 65536 | 0,06 µm | 0,13 % | 0,07 % |

Erste Ordnung in Δx an der harten Kante (Verhältnis ≈ 4 je Vervierfachung von N).

## PFT-Konvergenz (1D, Gauß-Amplitude, ψ = c x²/2)

| c (mm⁻²) | max. relativer Fehler |
| --- | --- |
| 500 | 4,0·10⁻³ |
| 2000 | 1,0·10⁻³ |

Der Fehler fällt wie 1/F, wie in Abschnitt 4 des Papers angegeben.

## Jacobi-Gesetz: Konvergenz (1D, Probe x = 0,3 mm)

| Fall | λ = 0,5 µm | λ = 0,125 µm |
| --- | --- | --- |
| divergent, R = 50 mm, d = 30 mm | 1,43·10⁻³ | 3,56·10⁻⁴ |
| hinter der Brennlinie, R = −20 mm, d = 40 mm (μ = 1) | 1,68·10⁻² | 4,02·10⁻³ |

Verhältnis 4,0 bzw. 4,2: Ordnung O(1/k) bestätigt; die Phase hinter der Brennlinie trägt den Faktor e^(−iπ/2).

## Debye-Fokus per FFT (Issue #1)

Alle Referenzen sind unabhängig vom FFT-Weg.

**Airy-Fall** (U = cos²θ, also konstanter Integrand; f = 2 mm, λ = 0,5 µm; maximaler relativer
Fehler bis 3 Airy-Radien, Fenster n Punkte über den k-Durchmesser, Nullauffüllung 4):

| n | harte Berandung | subpixelgewichtet |
| --- | --- | --- |
| 64 | 6,0·10⁻³ | 2,8·10⁻⁴ |
| 128 | 1,3·10⁻³ | 6,2·10⁻⁵ |
| 256 | 1,1·10⁻³ | 1,9·10⁻⁵ |
| 512 | 1,7·10⁻⁴ | 8,7·10⁻⁶ |

Gleiche Werte für NA 0,5 und 0,9. Die harte Berandung konvergiert unregelmäßig (Gittereffekte),
die subpixelgewichtete gleichmäßig und um den Faktor 20–60 genauer.

**Defokus auf der Achse** gegen die geschlossene Form (Substitution u = k_z): Fehler ≤ 4,6·10⁻⁴
für |δz| ≤ 5 µm (NA 0,5, n = 256).

**T4** (NA 0,1, U = 1 in der Austrittsebene, δz über ±1,5 λ/NA², Abweichung relativ zum
Debye-Maximum, Verschiebungen in Einheiten von λ/NA²):

| N_F | f (mm) | Abweichung Debye–RS | N_F · Abweichung | Fokusverschiebung RS | Debye |
| --- | --- | --- | --- | --- | --- |
| 3 | 0,149 | 0,504 | 1,51 | −0,313 | 0 |
| 10 | 0,495 | 0,172 | 1,72 | −0,118 | 0 |
| 30 | 1,485 | 0,0527 | 1,58 | −0,040 | 0 |
| 100 | 4,95 | 0,0153 | 1,53 | −0,012 | 0 |
| 1000 | 49,5 | 0,00152 | 1,52 | −0,0012 | 0 |

Die Abweichung fällt wie etwa 1,5/N_F; die Fokusverschiebung zeigt zur Linse. Für 1 % Genauigkeit
braucht die Debye-Näherung hier N_F ≳ 150; das Paper ist entsprechend korrigiert (#8). Off-axis-Vergleich in 2D steht noch aus (später als `slow`).

**Vektor-Fokus** (aplanatische Linse, x-polarisiert) gegen die Bessel-Quadratur der
Richards-Wolf-Integrale: maximaler Fehler 1,3·10⁻⁵ (NA 0,5) bzw. 1,9·10⁻⁵ (NA 0,9) relativ zum
Maximum; Verhältnis der Maxima |E_z|/|E_x| = 0,19 bzw. 0,39. Das Feld ist divergenzfrei
(∇·E relativ zu k·max|E| < 10⁻³, z-Ableitung über ±2 nm).

## SFT für allgemeines C (Issue #2)

C mit Eigenwerten 8000 und −5000 mm⁻², um 30° gedreht (Sattel, sig C = 0); Fenster 3 mm.

- **Gauß × Chirp** (Gauß-Matrix um 50° gedreht, kommutiert nicht mit C) gegen die geschlossene
  Form exp(−½kᵀM⁻¹k)/(2π√det M), M = A − iC: relativer Fehler 5·10⁻⁸ (Rundung bei Phasen bis
  10⁴ rad).
- **PFT-Grenzfall** (dieselbe glatte Gauß-Funktion): Abweichung < 2·10⁻³.
- **Hin- und Rück-SFT** (`inverse_sft_2d` mit B = −C⁻¹, Gitter L = C): exakt bis auf Rundung
  (< 10⁻¹⁰), auch für ein nicht bandbegrenztes Feld, weil sich die Transferfunktionen diskret
  aufheben.
- **Gedrehtes Rechteck** (Halbbreiten 0,6 und 0,4 mm längs der Eigenvektoren von C,
  subpixelgewichtet) gegen das Produkt zweier Fresnel-Integrale in geschlossener Form:

| N | Δx | w_c/Δx (kleinstes) | Fehler |
| --- | --- | --- | --- |
| 1024 | 2,93 µm | 5,4 | 3,7 % |
| 2048 | 1,46 µm | 10,8 | 2,1 % |
| 4096 | 0,73 µm | 21,6 | 1,2 % (`slow`, 40 s) |

Erste Ordnung (Verhältnis ≈ 1,8 je Halbierung von Δx). Ohne Subpixel-Gewichtung konvergiert der
Fehler unregelmäßig (4,1 %, 3,4 %, 1,7 %), weil die Kantenlage relativ zum Gitter springt.

## Minimax-Krümmung (Issue #3)

Beispiel aus Abschnitt 7.5 (D = 25 mm, NA 0,5, λ = 0,5 µm, f = 21,65 mm), gelöst als lineares
Programm (exakt, keine Iteration):

| Größe | Paper | Rechnung |
| --- | --- | --- |
| Minimax-Krümmung c | −521 mm⁻² | −521,04 mm⁻² (f_eff = 24,12 mm) |
| Restgradient | 230 mm⁻¹ | 229,83 mm⁻¹ |
| Restgradient bei Scheitelkrümmung −k/f | 972 mm⁻¹ | 972,0 mm⁻¹ |
| SFT-Punkte je Richtung (L = 30 mm, q = 10) | ≈ 4800 | 4843 (Scheitelkrümmung: 9282) |

Die 2D-Anpassung (Matrix C, maximale kartesische Komponente des Restgradienten, weil sie auf
einem achsparallelen Gitter die Nyquist-Schrittweite bestimmt) liefert auf der Kreisscheibe
C = −521,04·I. Rein quadratische Phasen werden exakt angepasst.

## Inverse PFT und mehrere Zweige (Issue #4)

- **Hin- und Rück-PFT** (sphärische Phase in 2D, vektorielle Amplitude; 1D mit veränderlicher
  Krümmung): exakt bis auf Rundung. Mit ∇ψ̃ = −ρ_s und H̃ = −H⁻¹ ist das eine algebraische
  Identität.
- **Inverse PFT gegen Quadratur** (Ẽ = Gauß in k mal exp(i b k²/2), s = 400 mm⁻¹): Fehler
  3,1·10⁻³ bei F = |b|·s² = 320 und 7,8·10⁻⁴ bei F = 1280, also O(1/F).
- **Zwei stationäre Punkte** (ψ = a x³/3, a = 2000 mm⁻³, Gauß-Amplitude): Summe beider Zweige
  gegen direkte Quadratur 5,3·10⁻³ bis 3,8·10⁻⁴ (relativ zum Maximum), abnehmend mit wachsendem
  Abstand vom Faltpunkt k = 0. Ein einzelner Zweig liegt um 170 % daneben.
- **Gültigkeitswarnung** `AsymptoticWarning`, wenn |h|·L_U² unter die Schwelle fällt (Standard 10).

## Maslov-Index in 2D (Issue #7)

Super-Gauß-Amplitude (Breite 1 mm, Ordnung 4), Phase ψ = −k(√(f_x² + x²) − f_x) − k(√(f_y² + y²) − f_y),
Fenster 4 mm; Vergleich mit dem 2D-Winkelspektrum an vier Proben (maximaler Fehler, Amplituden ≈ 1):

| f_x, f_y (mm) | d (mm) | μ | λ = 0,5 µm | λ = 0,25 µm |
| --- | --- | --- | --- | --- |
| 20, 20 (Punktfokus) | 40 | 2 | 3,9·10⁻³ | 1,9·10⁻³ |
| 20, 40 (zwischen den Brennlinien) | 30 | 1 | 2,0·10⁻² | 1,0·10⁻² |
| 20, 40 (hinter beiden) | 60 | 2 | 9,4·10⁻³ | 4,7·10⁻³ |

Ordnung O(1/k) (Verhältnis ≈ 2 bei doppelter Wellenzahl; λ = 0,25 µm als `slow`). Ohne den
Faktor exp(−iπμ/2) läge die Vorhersage um mehr als die halbe Amplitude daneben.

Wichtig für die Wahl der Testfälle: Die geometrische Näherung braucht eine große Fresnel-Zahl
des defokussierten Strahls, a²·δ/(λ f²) mit dem Abstand δ von der jeweiligen Brennlinie. Bei
einer ersten Wahl (a = 0,4 mm, δ = 10 mm, f = 50 mm) war sie ≈ 1, und die Fehler lagen bei 30 %.

## Blenden in 2D: Randintegral und kritische Punkte (Issue #5)

Skalierung ψ → sψ, k → sk (gleiche Strahlen, s-fache Wellenzahl), s = 1 und 4. Fehler relativ
zum exakten Wert.

**Kreisscheibe** (R = 0,5 mm, ψ = c|ρ|²/2, c = 3000 mm⁻², U = 1) gegen Hankel-Quadratur:

| \|k\|/(cR) | PFT + Randintegral (numerisch) | PFT + Punkte 2. Art, s = 1 | s = 4 |
| --- | --- | --- | --- |
| 0,5 (beleuchtet) | < 10⁻¹² | 1,6·10⁻⁴ | 1,9·10⁻⁵ |
| 0,9 (nahe Rand) | < 10⁻¹² | 1,6·10⁻² | 2,2·10⁻³ |
| 1,2 (Schatten) | < 10⁻¹² | 3,6·10⁻² | 8,7·10⁻³ |

Für U = 1 und isotropes H ist das Randintegral exakt: Das Feld ∇Φ/|∇Φ|² ist in 2D quellenfrei,
der vernachlässigte Term ∬ e^(iΦ) ∇·F verschwindet.

**Rechteck** (Halbbreiten 0,5 und 0,3 mm, Sattelphase c_x = 2000, c_y = −1500 mm⁻²) gegen das
Produkt zweier Fresnel-Integrale:

| k/(c·a) | Region | Randintegral numerisch, s = 1 → 4 | kritische Punkte, s = 1 → 4 |
| --- | --- | --- | --- |
| (0,2; 0,3) | beleuchtet | 1,7·10⁻³ → 1,9·10⁻⁴ | 7,3·10⁻⁴ → 7,9·10⁻⁵ |
| (1,3; 0,2) | Schatten in x, Punkte 2. Art | 4,4·10⁻² → 8,8·10⁻³ | 2,5·10⁻² → 5,0·10⁻³ |
| (1,4; 1,5) | nur Ecken (3. Art) | 7,3·10⁻³ → 2,3·10⁻³ | 1,2·10⁻² → 5,0·10⁻³ |
| (0,5; 0,9) | nahe Schattengrenze in y | 21 % → 4,7 % | 14 % → 2,3 % |

Im beleuchteten Bereich ist die PFT allein über 10-mal schlechter als mit Kanten und Ecken. Nahe
einer Schattengrenze versagt die Randdarstellung (|∇Φ| → 0 am Rand); dort braucht es die
gleichmäßige Lösung, in 2D noch offen.

## LPIA an gekrümmten Flächen mit Strahlnetz (Issue #6)

n₁ = 1, n₂ = 1,5, λ = 0,5 µm. Amplitude nach Gl. (interface_amplitude) mit Jacobi-Matrix aus den
Nachbarstrahlen (Differenzen zweiter Ordnung, auch am Rand).

- **Gekippte ebene Welle durch eine Planfläche:** U₂ = T·U₁ exakt (< 10⁻¹²), J = 1, Ausgangsfeld
  transversal.
- **Konvergente Welle durch eine Sphäre um ihren Fokus** (alle Strahlen senkrecht):
  |E₂| = |t|·|E₁|·r₁/r₂ und Phase = k₀ × optischer Weg, beides bis auf Rundung (< 10⁻¹²). Prüft
  Schnittpunkt, Normale, cos-Faktoren und Maschen-Jacobi zusammen.
- **Divergente Welle an einer Planfläche:** Maschen-Jacobi gegen die analytische Jacobi-Funktion
  J = (ρ₂/ρ₁)·dρ₂/dρ₁: Fehler 1,3·10⁻⁵ (41² Strahlen) bzw. 3,1·10⁻⁶ (81²), zweite Ordnung.
- **Ebene Welle durch eine Sphäre, auf der Achse:** U₂ = t·e^(−iπμ/2)/|1 − d/f′| mit
  f′ = n₂R/(n₂ − n₁) = 60 mm für d = 10 … 50 mm (μ = 0) und d = 90 mm hinter dem Fokus (μ = 2,
  Gouy-Phase −π): Fehler < 10⁻⁵.

## Hybride Felddarstellung (Issue #17)

Sphärische Welle (f = 25 mm, λ = 0,5 µm) auf einem regelmäßigen Gitter über ±2 mm, Gauß-Hülle,
transversale Polarisation.

| Größe | 21² Punkte | 41² Punkte | Ordnung |
| --- | --- | --- | --- |
| Phase aus Taylor-Entwicklung (Zellmitten) | 2,9·10⁻³ rad | 3,9·10⁻⁴ rad | h³ (Verhältnis 7,4) |
| Residuum, linear | 4,7·10⁻³ | 1,2·10⁻³ | h² (Verhältnis 3,9) |
| Residuum, Clough-Tocher | 6,3·10⁻⁴ | 1,0·10⁻⁴ | kleiner, konvergent |

Gegenprobe zu ADR 0003 (Krümmung aus Kapitel 7.5, 81² Punkte über ±2 mm): Residuum
interpolieren und Phase auswerten ergibt < 10⁻³ Fehler, lineare Interpolation des vollen Feldes
U·e^(iψ) über 50 %.

PFT hin und zurück über die Feldobjekte ist exakt; geometrische Propagation einer sphärischen
Welle um 10 mm liefert die Hesse-Matrix der Welle mit f = 15 mm (H₂ = H₁·(∂ρ₂/∂ρ₁)⁻¹).

## Jacobi-Faktor aus unstrukturierten Strahlwolken (Issue #18)

Sollwert: analytisches J aus `semiclassical.propagate_2d` (über die analytische Hesse-Matrix,
Gl. jacobian_relation), λ = 0,5 µm. Wolke: Gitter über ±2 mm mit zufälliger Verschiebung um
±0,3 h je Punkt (Seed 7), Delaunay-Triangulierung, Randsplitter (Formgüte < 0,2) abgeschält.
Maximaler relativer Fehler von J über **alle** Knoten einschließlich Rand:

| f_x, f_y (mm) | d (mm) | μ | 41² Strahlen | 81² Strahlen | Verhältnis |
| --- | --- | --- | --- | --- | --- |
| 25, 25 | 10 | 0 | 3,0·10⁻⁵ | 8,6·10⁻⁶ | 3,5 |
| 20, 40 (zwischen den Brennlinien) | 30 | 1 | 1,8·10⁻⁴ | 5,8·10⁻⁵ | 3,1 |
| 20, 20 (hinter dem Punktfokus) | 40 | 2 | 1,4·10⁻⁴ | 4,1·10⁻⁵ | 3,5 |
| 20, 40 (hinter beiden Brennlinien) | 60 | 2 | 1,8·10⁻⁴ | 5,8·10⁻⁵ | 3,1 |

Zweite Ordnung in h (quadratische Ausgleichsrechnung über den Zwei-Ring der Maschennachbarn).
Der Maslov-Index aus den Eigenwerten stimmt an allen Knoten. Das flächengewichtete Mittel der
Dreiecks-Flächenverhältnisse an den Knoten ist nur etwa erster Ordnung (RMS-Fehler sinkt um den
Faktor 2,3–2,5 je Halbierung von h). Ohne Abschälen der Randsplitter stagniert der Randfehler bei
≈ 4·10⁻⁴, weil Splitter weit entfernte Randstrahlen koppeln.

- **Gleichverteilte Zufallswolke** (Seed 3): 1,9·10⁻⁴ (400 Punkte), 6,8·10⁻⁵ (1600),
  2,5·10⁻⁵ (6400); konvergent, ohne feste Ordnung (Poisson-Wolke).
- **Regelmäßiges Gitter:** gegen `lpia.mesh_jacobian` bei quadratischer Abbildung bis auf
  Rundung (< 10⁻¹²), allgemein Abweichung O(h²) (2,3·10⁻⁵ bei 21², 5,7·10⁻⁶ bei 41²).
- **Affine Abbildung:** Dreiecksverhältnisse und Ableitungen exakt; gespiegelte Abbildung ergibt
  negative Verhältnisse.
- **Zentrale Obskuration** (Ringblende 0,6–2 mm): Delaunay allein füllt das Loch; mit
  `max_edge = 2h` oder Gebietsprädikat liegt kein Dreieck darüber, die Fläche des Netzes bleibt
  kleiner als die Ringfläche, J auf allen Knoten < 5·10⁻⁴ genau.
- **Kaustik in der Zielebene** (konvergente Welle mit sphärischer Aberration, f = 20 mm,
  d = 18 mm, ψ = −k(ρ²/(2f) + bρ⁴/4), b = 9,45·10⁻⁴ mm⁻²): J wechselt auf einem Kreis
  ρ ≈ 1,35 mm das Vorzeichen. Markiert werden nur Knoten innerhalb 2h vom Kaustikkreis, jeder
  Knoten mit falschem Vorzeichen ist markiert, an allen übrigen stimmt μ (0 innen, 1 außen).

## Operatorkette (Issue #19)

Jeder Operator in `chain.py` gibt `(Feld, LogEntry)` zurück; das Protokoll enthält die Kennzahlen
aus `selection.py` (w, F, Kantenverhältnis, SFT-Kennzahlen) und die Gültigkeitswarnungen.

**Regression gegen F1** (gleiches Ergebnis wie der Baustein): geometrische Propagation =
`propagate_2d`, PFT/inverse PFT = `pft_2d`/`inverse_pft_2d`, SFT = `sft_2d` je Komponente,
FFT = `fourier.ft`, Blende auf dem Gitter = `circular_aperture_weights`, LPIA auf der Punktwolke
= `trace_surface` + `interface_amplitude` mit `lpia.mesh_jacobian` (< 10⁻¹², Sphäre um den Fokus;
dort zusätzlich |E₂| = |t|·|E₁|·r₁/r₂ und ∇ψ₂ wie die Kugelwelle in n₂ bis auf Rundung, neue
Hesse-Matrix aus den Maschenableitungen auf 10⁻⁴).

**Exakte Operatoren invertierbar** (bis auf Rundung): Winkelspektrum ±dz auf Punktwolke
(Phase < 10⁻⁹ rad bei Phasen ~10⁵ rad), Frequenzgitter ohne und mit Träger (< 10⁻¹²);
SFT hin und zurück (< 10⁻¹⁰); FFT hin und zurück (< 10⁻¹²).

**Unabhängige Gegenproben der Buchführung:**

- PFT → Winkelspektrum → inverse PFT = geometrische Propagation (`propagate_2d`): algebraische
  Identität über die Legendre-Transformation, Amplitude < 10⁻¹⁰, auch hinter den Brennlinien
  (Maslov-Phase) und für eine gekippte astigmatische Phase.
- SFT → Winkelspektrum → inverse SFT gegen `propagation.angular_spectrum` auf einem Gitter halber
  Schrittweite (Gauß × konvergente Kugelwelle, f = 50 mm, halber Weg zum Fokus; das
  Ausgangsgitter ρ = (1 + dz·c/k)·u′ = u′/2 fällt auf das Referenzgitter): 9,6·10⁻⁸ relativ zum
  Maximum. Paraxialer Anteil des Winkelspektrums wandert in den Träger, der Rest in die Werte.
- FFT → Winkelspektrum → inverse FFT = `propagation.angular_spectrum` (< 10⁻¹²).
- Geometrische Propagation mit Maschen-Jacobi gegen Hesse-Weg (gekippte astigmatische
  quadratische Phase, 21²/41²/81² Strahlen): Amplitude 3,1·10⁻⁵ → 7,8·10⁻⁶ → 1,9·10⁻⁶,
  neue Hesse-Matrix 4,7·10⁻⁵ → 1,2·10⁻⁵ → 3,0·10⁻⁶, beides O(h²).

Beim letzten Vergleich fiel ein Fehler in `propagate_2d` auf (aus #17): die Strahlabbildung war
transponiert, die neue Hesse-Matrix daher für nicht vertauschende H und Strahlrichtungen um
≈ 4·10⁻⁴ falsch (J und μ waren richtig). Behoben, Regressionstest mit zentralen Differenzen in
`test_semiclassical.py`.

## Planer und Integrationstest Linse → Blende → Ebene (Issue #20)

**Entscheidungstabelle** (`select_method`, ε = 2 %): je Zeile von Tabelle tab:selection ein Fall,
dazu Totalausfall der LPIA (R ~ λ), „FFT billiger als SFT“ (Aufwand) und F zwischen 1 und 1/ε.
Mit echten Feldern: Kugelwelle ohne Blende → geometrisch (Ergebnis identisch mit
`geometric_propagation`), mit Blende → Gitterverfahren mit Achszone, Fokus → Debye bzw.
Rayleigh-Sommerfeld, Zylinderwelle → gemischte Karte (noch ohne Operatoren), kollimiert mit
Blende → FFT.

**System:** kollimierter, x-polarisierter Strahl (λ = 0,5 µm) → Kugelfläche R = 5 mm Luft/Glas
(n = 1,5, LPIA auf einer Wolke mit 61² Strahlen) → Kreisblende Radius a in der Austrittsebene →
Ebene auf halbem Weg zum paraxialen Fokus bzw. Fokusebene (f ≈ 14,9 mm).

**Referenz** (unabhängig von allen Kettenbausteinen): Austrittsfeld aus meridionaler
Strahlrechnung in geschlossener Form (Snellius, Convention A, Gl. interface_amplitude mit
radialem Jacobi-Faktor, Polarisation analytisch: E = e^{iψ}(−A_s sin φ e_φ + A_p cos φ
(cos θ e_r + sin θ e_z))), dann Rayleigh-Sommerfeld-Integral über die Kreisscheibe (Gauß-Legendre
in r mit exakter Kante, Trapez in φ). Knotenzahlen aus der größten Phasenänderung; bei 1,4-facher
Knotenzahl ändert sich nichts (< 10⁻¹⁰).

Fehler relativ zum größten |E| der Proben, außerhalb der Achszone (siehe unten):

| System | Ebene | ε | Planer wählt | q | gemessen | Budget |
| --- | --- | --- | --- | --- | --- | --- |
| a = 0,354 mm (N_F ≈ 25) | f/2 | 4 % | FFT (502 statt 692 Punkte) | 10 | 1,2 % | 4,0 % |
| a = 0,354 mm | f/2, SFT erzwungen | 4 % | SFT | 10 | 0,89 % | 4,0 % |
| a = 0,707 mm (N_F ≈ 100, `slow`) | f/2 | 2 % | FFT (2010 statt 2772) | 20 | 1,3 % | 2,0 % |
| a = 0,707 mm (`slow`) | f/2, SFT erzwungen | 2 % | SFT | 20 | 1,1 % | 2,0 % |
| a = 0,354 mm | Fokus | 4 % | Rayleigh-Sommerfeld (1,5/N_F = 6 % > ε) | | 4·10⁻⁵ | 4,0 % |
| a = 0,707 mm | Fokus | 2 % | Debye (1,5/N_F = 1,5 %) | | 0,30 % | 1,5 % |

Das Budget ist bei Rayleigh-Sommerfeld im Fokus sehr großzügig: Der Kantenterm 0,4/q ist für das
Fokusfeld nicht maßgeblich.

**Abweichungen zwischen Budget und Messung** (Issue #31, Label `paper`):

- *Achs-Kaustik der Kantenwelle:* Innerhalb von etwa drei Fresnel-Ringbreiten √(2|C′|) um die
  Achse einer Kreisblende ist der Gitterfehler der harten Kante kohärent verstärkt: 3,1 % (FFT)
  bzw. 6,0 % (SFT) bei N_F ≈ 25, 5,5 % bzw. 11 % bei N_F ≈ 100. Der Planer gibt die Zone als
  `parameters["axis_zone_mm"]` mit Warnung aus; das Budget gilt außerhalb. Prototyp-Messung auf
  der Achse (N_F ≈ 100, SFT): 69 %, 18 %, 10 % bei q = 10, 15, 20.
- *Konstante des Kantenfehlers in 2D:* gemessen bis 0,35/q (Kreis) bzw. 0,26/q (Rechteck, T6),
  das Paper nennt ≈ 0,1/q (1D). Der Planer rechnet mit 0,4/q.
- *Fenster der SFT:* Mit demselben Gitter für Hin- und Rück-SFT gilt allein aus der
  Chirp-Bedingung n ≈ π q²/m. Bei kleinen Aperturen ist deshalb die direkte FFT billiger; das
  bestätigt die Aussage des Papers (SFT lohnt ab N_F ≳ 10²).

## Vektorisierte LPIA (Issue #21)

`refract_batch`, `fresnel_coefficients_batch` und `fresnel_operator_batch` gegen die bisherigen
Einzelstrahl-Funktionen: 500 zufällige Richtungen (θ < 70°, Seed 3) mit gekippten Normalen
beliebiger Orientierung und Länge, dazu senkrechter Einfall: Abweichung < 10⁻¹⁴. Totalreflexion
(Glas → Luft) als Maske genau für (n₁/n₂)²·sin²θ > 1, Richtung NaN, T = 0, |r| = 1;
`trace_surface` markiert totalreflektierte Strahlen in `total_reflection`. Alle Tests aus
`test_lpia.py` und `test_lpia_surface.py` unverändert grün.

Laufzeit (ein Kern): Schleife 7,7·10³ Strahlen/s, vektorisiert 9,8·10⁵ Strahlen/s, Faktor ≈ 130;
`trace_surface` mit 10⁶ Strahlen durch eine Sphäre (30 Newton-Schritte) 3 s statt etwa 130 s.

## Rauchtest gegen Raytatouille (Issue #22, Marker `rtt`)

Plankonvexe Einzellinse `m1/singlet_const.rtt.json` von Raytatouille (Commit aus
`rtt-commit.txt`), drei Strahlen des Achsfeldes (Pupille (0, 0), (0, 0,5), (0,3, 0,2)), gegen
eine Strahlrechnung in geschlossener Form mit `lpia.refract` und den absoluten Brechzahlen des
kompilierten Systems: Landepunkt < 10⁻¹⁰ mm, Richtung < 10⁻¹², OPL ab der Startebene < 10⁻¹⁰ mm;
Gewicht auf der Achse = T₁·T₂ aus den Fresnel-Koeffizienten (Convention A), relativ < 10⁻¹².
Beim ersten Versuch mit Luft = 1 wich die OPL um 0,028 mm ab: Raytatouille rechnet mit
Ciddor-Luft (n = 1,000272 bei 0,5876 µm).
