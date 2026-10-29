# Testfälle

Nummern wie in Kapitel 8 des Papers. Status: ✅ umgesetzt und grün, ◐ teilweise, ○ offen.
Gemessene Werte stammen aus den Tests bzw. aus `experiments/pruefrechnungen_1d.py`.

| Nr. | Prüfung | Referenz | Abnahme | Status | Test |
| --- | --- | --- | --- | --- | --- |
| T1 | Van-Vleck-Kern gegen Rayleigh-Sommerfeld | analytisch: K_RS/K_VV = 1 + i/(kr) | relative Abweichung = 1/(kr) | ✅ | `test_propagation.py` |
| T2 | Jacobi-Gesetz | Winkelspektrum (exakt) | 1D: Fehler < 2·10⁻², Ordnung O(1/k) (Verhältnis 3–5 bei k × 4); 2D: \|U\| auf 0,3 % | ✅ 1D und 2D | `test_semiclassical.py` |
| T3 | exakte Phasenkompensation | analytisch | Rest < 10⁻⁶ rad; parabolisch + exaktes k_z: −2819,45 rad | ✅ | `test_focus.py` |
| T4 | Debye gegen Rayleigh-Sommerfeld, N_F von 3 bis 10³ | direkte 2D-Integration | Abweichung wächst mit fallendem N_F, Fokusverschiebung bei N_F ≲ 10 | ○ F1-4 | – |
| T5 | gleichmäßige Kantenformel | direkte Quadratur | relativer Fehler < 1 % für \|x_s − x_e\| ≤ 0,05 mm; Schattengrenze ½; Übergang in die Kantenwelle auf 0,5 % | ✅ 1D | `test_edge.py` |
| T6 | SFT | direkte Quadratur (4·10⁶ Punkte) | siehe Tabelle unten | ✅ 1D, 2D diagonal | `test_sft.py` |
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
