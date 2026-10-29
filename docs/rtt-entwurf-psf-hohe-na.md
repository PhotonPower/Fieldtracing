# Issue-Entwurf für Raytatouille: FFT-PSF bei nichtlinearer Pupillenabbildung

> Entwurf aus Fieldtracing (#25). Nicht in Raytatouille angelegt; Entscheidung beim Maintainer
> (AGENTS.md, Regel 8). Stand Raytatouille: Commit aus `rtt-commit.txt`.

**Titel:** analysis: FFT-PSF bei hoher NA zu schmal, wenn die Pupille nichtlinear auf die
Richtungssinus abgebildet wird

**Beobachtung.** Ideale Linse (`ideal_lens`, f = 2 mm, Vakuum, λ = 0,5 µm, Blende für Bildraum-NA
0,9). `pupil_function(...).mapping.m` = −2,065·I (= tan θ_max), `delta_max` = 0,56,
Warnung `pupil.mapping_nonlinear`. Die FWHM von `psf`/`psf_at` ist 0,12 µm; die Airy-Formel für
NA 0,9 gibt 0,29 µm, eine vektorielle Debye-Rechnung aus denselben Strahlen 0,24–0,36 µm. Bei
NA 0,5: 0,45 µm statt 0,52 µm. Das Verhältnis ist in beiden Fällen tan θ_max/sin θ_max.

**Ursache (vermutet).** Die affine Abbildung Pupille → Richtungssinus (ADR 0032) übernimmt die
Steigung tan θ_max/Radius, und die FFT integriert dann über eine Scheibe mit „Sinus“ bis 2,06.

**Vorschläge.**
1. Bei `delta_max` über einer Schwelle die PSF verweigern oder die Pupillenzellen auf die
   tatsächlichen Richtungssinus der Strahlen legen (Interpolation des Residuums, nicht des
   Feldes).
2. Langfristig: vektorielle Debye-PSF aus den Strahlen (Integrationspunkt I1 in Fieldtracings
   `docs/integration-raytatouille.md`, Vergleich in `docs/vergleich-psf.md`).

**Referenzwerte.** `tests/test_rtt_psf.py` in Fieldtracing.
