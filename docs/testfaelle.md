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
| T7 | Maslov-Index | Winkelspektrum | Brennlinie (1D): μ = 1, Phase −π/2; Punktfokus (2D): μ = 2 | ◐ 1D | `test_semiclassical.py` |
| T8 | Fresnel-Operator | analytisch | R + T = 1, r_p = −r_s bei 0°, r_p = 0 bei Brewster, Transversalität, Snellius | ✅ | `test_lpia.py` |
| B | Beispiel Kapitel 7.5 | verkleinertes System gegen RS; Kennzahlen | Kennzahlen wie im Paper | ◐ Kennzahlen | `test_selection.py` |

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
braucht die Debye-Näherung hier N_F ≳ 150 (Issue mit Label `paper`, weil Tabelle 7.1 die Grenze
bei N_F ≲ 10 zieht). Off-axis-Vergleich in 2D steht noch aus (später als `slow`).

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
