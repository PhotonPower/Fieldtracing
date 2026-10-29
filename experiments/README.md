# Experimente

Explorative Skripte und Studien, die nicht zu den Tests gehören: Parameterstudien, Plots,
Voruntersuchungen. Ergebnisse, die als Sollwert dienen sollen, wandern als Test nach `tests/`.

| Skript | Inhalt |
| --- | --- |
| `vergleich_psf.py` | Debye-PSF aus Raytatouille-Strahlen gegen Raytatouilles FFT-PSF; erzeugt `docs/img/vergleich-psf.png` und die Zahlen in `docs/vergleich-psf.md` (braucht raytatouille, `RAYTATOUILLE_SOURCE`) |
| `pruefrechnungen_1d.py` | erste 1D-Prüfrechnungen zum Paper (SFT, erfc-Kante, Phasenkompensation, Jacobi-Gesetz); Vorläufer der Tests T2, T3, T5, T6 |

Ausgaben (Plots, Daten) nach `experiments/out/` schreiben; das Verzeichnis ist nicht versioniert.
