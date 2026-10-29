# Beispiel aus Abschnitt 7.5 (Meilenstein F3)

Linse D = 25 mm (a = 12,5 mm), NA 0,5, λ₀ = 0,5 µm, f = a/tan θ_max = 21,65 mm, aplanatisch
(`GridlessField.aplanatic`), x-polarisierte homogene Beleuchtung, harte Blende vom Radius a in
der Austrittsebene. Tests: `tests/test_example_7_5.py`.

## Kennzahlen (Planer gegen Paper)

| Größe | Paper | Planer (`plan_propagation`, ε = 1 %) |
| --- | --- | --- |
| h = k₀/f (Zentrum) | 580 mm⁻² | 580,4 mm⁻² |
| h am Rand, radial | 377 mm⁻² | 378,5 mm⁻² (Proben knapp innerhalb a) |
| w = h^(−1/2) | 41,5 µm | 41,5 µm |
| N_F = a²/(λ₀f) | 1,44·10⁴ | 1,443·10⁴ |
| F_a = 2πN_F | 9,07·10⁴ | 9,07·10⁴ |
| kantennahe Zone (ε = 1 %) | ≈ 1,7 mm | 1,66 mm |
| Baustein für den Fokus | Debye (Schritt 3a) | `debye` (1,5/N_F = 10⁻⁴ ≤ ε) |

## Fokus (Schritt 3a, #40)

Austrittsfeld auf einer Polarwolke mit 80 Ringen (19 441 Strahlen) über 1,02·a. Der Planer
wählt die Debye-Kette; die k-Abtastung folgt aus dem Zielfenster L_F = 20 µm, sodass die
periodischen Kopien des FFT-Ergebnisses unter ε/10 bleiben: n = 78 Punkte über den
k-Durchmesser (Periode 39 µm), Nullauffüllung 4, Ausgabegitter 312² mit 0,12 µm Schritt.

| Fehlerquelle (Budget) | Wert |
| --- | --- |
| Debye 1,5/N_F | 1,0·10⁻⁴ |
| k-Berandung 0,02/n | 2,6·10⁻⁴ |
| periodische Kopien (Airy-Hüllkurve bei Periode − L_F/2) | 6,5·10⁻⁴ |
| Interpolation von U (linear gegen Clough-Tocher) | 1,8·10⁻⁵ |
| Taylor-Phase (größte lokale Abweichung) | 4,3·10⁻³ |
| **Summe** | **5,4·10⁻³** |
| **gemessen** gegen Richards-Wolf im Fenster (x, y, Diagonale) | **6,5·10⁻⁴** |

Laufzeit (ein Kern): Planer 1,5 s (Minimax-Anpassung als lineares Programm über alle Proben),
Debye-Kette 0,6 s, Richards-Wolf-Referenz 0,25 s.

Das Paper nennt nominell N = 2·NA·L_F/λ = 40 Punkte und Nullauffüllung auf etwa 256. Mit 40
Punkten ist die Periode 20 µm, die Kopien fallen dann in das Fenster selbst; für 10⁻³ braucht
es etwa die doppelte Punktzahl. Der Taylor-Term ist eine lokale Höchstschätzung und im
Ergebnis um eine Größenordnung kleiner, weil sich die Phasenfehler im Integral mitteln.

Zwei neue Budgetterme in `plan_propagation` (#40): der Fehler der Taylor-Phase
(`phase_taylor`, Vergleich der Entwicklungen zweier Nachbarproben an ihrer Mitte) für alle
Wege, die die Phase zwischen Proben auswerten, und die periodischen Kopien beim Debye-Weg
(`aliasing`, aus `focus_window`). Bei 40 statt 80 Ringen ist die Taylor-Phase der größte reale
Fehler (4·10⁻⁴ im Fokus).
