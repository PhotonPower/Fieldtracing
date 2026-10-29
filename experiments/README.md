# Experimente

Explorative Skripte und Studien, die nicht zu den Tests gehören: Parameterstudien, Plots,
Voruntersuchungen. Ergebnisse, die als Sollwert dienen sollen, wandern als Test nach `tests/`.

| Skript | Inhalt |
| --- | --- |
| `vergleich_psf.py` | Debye-PSF aus Raytatouille-Strahlen gegen Raytatouilles FFT-PSF; erzeugt `docs/img/vergleich-psf.png` und die Zahlen in `docs/vergleich-psf.md` (braucht raytatouille, `RAYTATOUILLE_SOURCE`) |
| `beispiel_7_5_zwischenebene.py` | Zwischenebene des Beispiels 7.5 für mehrere Blendenradien: Plan, Laufzeit, Speicher, Vergleich mit der Geometrie (#41) |
| `run_beispiel_7_5.sh` | installiert Fieldtracing in einer venv und rechnet `beispiel_7_5_zwischenebene.py` für alle Radien einschließlich des vollen Systems, je Radius ein Prozess mit Spitzenspeicher (GNU time), Log-Datei; für Rechner mit ≥ 8 GB (AlmaLinux 10 und andere Linux) |
| `pruefrechnungen_1d.py` | erste 1D-Prüfrechnungen zum Paper (SFT, erfc-Kante, Phasenkompensation, Jacobi-Gesetz); Vorläufer der Tests T2, T3, T5, T6 |

Ausgaben (Plots, Daten) nach `experiments/out/` schreiben; das Verzeichnis ist nicht versioniert.
