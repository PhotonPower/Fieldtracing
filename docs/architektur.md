# Architektur des Pakets

## Module und Paper

| Modul | Inhalt | Paper |
| --- | --- | --- |
| `units` | Wellenzahl in 1/mm | Abschnitt 1.1 |
| `fourier` | Fourier-Konvention auf zentrierten Gittern | Gl. ft_convention |
| `propagation` | Winkelspektrum, Rayleigh-Sommerfeld-Kern, Van-Vleck-Kern | Gl. spw, rs_kernel, van_vleck |
| `semiclassical` | geometrische Propagation, Jacobi-Faktor, Maslov-Index | Abschnitt 3.4, Gl. jacobian_amplitude |
| `pft` | punktweise Fourier-Transformation, invers, Summe über mehrere stationäre Punkte (1D), Gültigkeitswarnung | Abschnitt 4, Gl. pft_final |
| `sft` | semi-analytische Fourier-Transformation (1D, 2D für allgemeines C, invers), Abtastprüfung, Blendengewichte für beliebige Gebiete | Abschnitt 6.4, Gl. sft_formula |
| `edge` | Kantenwelle, gleichmäßige erfc-Lösung | Abschnitt 6.1, Gl. keller_gtd, fresnel_edge |
| `focus` | sphärische Austrittsphase, exakte Legendre-Transformation, Airy-Integral, Debye-Fokus per FFT (`debye_focus`, skalar und vektoriell) | Abschnitt 5.3, Gl. debye |
| `lpia` | vektorielles Snellius-Gesetz, Fresnel-Koeffizienten, Operator T | Abschnitt 3.2 |
| `selection` | Kennzahlen w, F, Kantenverhältnis, Aufwand, Minimax-Krümmung für die SFT | Kapitel 7 |
| `field` | Datenstrukturen der hybriden Darstellung | Abschnitt 1.3, 7.2 |

Abhängigkeiten nur in einer Richtung: `units`, `fourier` < `propagation` < alle übrigen;
`selection` und `field` hängen von keinem anderen Baustein ab.

## Entwurfsregeln

- **Reine Funktionen auf Arrays.** Jede Funktion bekommt alle Größen als Argumente (mit Einheit im
  Docstring) und gibt neue Arrays zurück. Kein globaler Zustand, keine versteckten Caches.
- **Hybride Rückgabe.** Transformationen geben `(k, residual, phase)` zurück (ADR 0003). So bleibt
  die analytische Phase erhalten, und eine spätere Interpolation betrifft nur glatte Größen.
- **Vektorfelder.** Amplituden dürfen die Form `(N,)` (skalar) oder `(N, 3)` (vektoriell) haben;
  Faktoren werden über die letzte Achse gesendet.
- **Datenstrukturen** (`field.py`) sind unveränderliche Datenklassen mit einem Array je Größe.
  Das entspricht der Structure-of-Arrays-Form von Raytatouille (ADR 0005 dort) und lässt sich
  direkt nach C++ übertragen.

## Teststrategie

1. **Unabhängige Sollwerte:** analytisch (Gauß-Transformation, Airy-Integral, Fresnel-Formeln)
   oder durch eine exakte Rechnung, die den geprüften Baustein nicht benutzt (Winkelspektrum,
   direkte Quadratur, Hankel-Quadratur).
2. **Ordnung statt nur Absolutfehler:** Für Näherungen in 1/k wird die Konvergenzordnung geprüft,
   für Diskretisierungen die Ordnung in der Schrittweite.
3. **Ein Testmodul je Baustein,** Testfälle mit Nummer aus `docs/testfaelle.md` im Docstring.
4. **Schnell bleiben:** Die Standardtests laufen in Sekunden. Teure Studien gehören nach
   `experiments/` oder werden als `slow` markiert.

## Abbildung auf Raytatouille

Jedes Modul entspricht später einem Header in `libs/rtt-field/include/rtt/field/` (Vorschlag,
siehe `docs/integration-raytatouille.md`). Die Python-Tests liefern die Sollwerte des Ports.
