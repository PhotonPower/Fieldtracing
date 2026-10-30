# Issue-Entwurf für Raytatouille: FFT-PSF bei verletzter Sinusbedingung

> Angelegt als [Raytatouille #286](https://github.com/PhotonPower/Raytatouille/issues/286) (E7).

> Entwurf aus Fieldtracing (#25, überarbeitet in #60). Nicht in Raytatouille angelegt; Entscheidung
> beim Maintainer (Fieldtracing AGENTS.md, Regel 8). Stand Raytatouille: Commit aus
> `rtt-commit.txt`. Vorlage: `.github/ISSUE_TEMPLATE/aufgabe.md` bzw. `frage.md`.
>
> **Freigegeben** vom Maintainer von Fieldtracing am 2026-10-30 (Entscheidung E6 in
> `docs/roadmap.md`); Übernahme nach Raytatouille dort nach dessen Prozess.

**Titel:** [Frage] analysis: FFT-PSF bei großem δ_max der Pupillenzuordnung – warnen, ablehnen
oder richtig zuordnen?

## Kontext

ADR 0032 Punkt 4 ordnet die Pupille affin den Richtungssinus zu, M = ∂s/∂p bei p = 0, und nennt
die Grenze: linear genau dann, wenn das System die Sinusbedingung erfüllt. Über
`mapping_warning` (10⁻²) warnt `pupil.mapping_nonlinear`, die PSF wird trotzdem gerechnet.

Messung aus Fieldtracing (`docs/vergleich-psf.md`, `tests/test_rtt_psf.py` dort):

| System | δ_max | FWHM FFT-PSF | FWHM richtig | Verhältnis |
| --- | --- | --- | --- | --- |
| ideale Linse (`ideal_lens`, kollinear), NA 0,5 | – | 0,45 µm | 0,52 µm | tan θ/sin θ = 1,15 |
| ideale Linse, NA 0,9 | 0,56 | 0,12 µm | 0,29 µm (skalar) | 2,29 |
| Mikroskopobjektiv 20x/0,45 (Beispiel) | 0,0021 | 0,672 µm | 0,672 µm | 1,00 |

Bei korrigierten Systemen stimmt die FFT-PSF (Strehl 0,99243 wie eine unabhängige Debye-Rechnung).
Betroffen sind Systeme, die die Sinusbedingung verletzen, vor allem Raytatouilles eigene ideale
Linse: Mit ihr modelliert man gern „perfekte“ Objektive hoher NA und bekommt eine um tan θ/sin θ
zu schmale PSF, nur mit einer Warnung.

## Frage

Wie soll die FFT-PSF mit großem δ_max umgehen?

## Optionen mit Vor- und Nachteilen

1. **Ablehnen über einer zweiten Schwelle** (z. B. δ_max > 0,05, `AnalysisError` mit Code):
   einfach, verhindert stille Fehler; manche Nutzer verlieren eine grobe PSF.
2. **Zellen auf die tatsächlichen Richtungssinus legen** (Interpolation des glatten Restfelds
   auf ein gleichmäßiges s-Gitter, nicht des Feldes; Fieldtracing ADR 0003): richtig für alle
   Systeme; mehr Aufwand, die Zellflächen-Gewichte ändern sich (Jacobi-Faktor ds/dp).
3. **Ideale Linse mit Sinusbedingung** (`ideal_lens` mit Option `mapping: "aplanatic"`,
   h = f sin θ): modelliert ideale Objektive richtig; ändert die bestehende ideale Linse nicht.
4. **Vektorielle Debye-PSF aus Strahlen** (Integrationspunkt I1 des ADR-Entwurfs `rtt-field`):
   braucht keine Pupillenzuordnung, enthält Schiefe-, Radiometrie- und Vektoranteile; längerer Weg.

Vorschlag aus Fieldtracing: kurzfristig 1 oder 3, langfristig 4.

## Betroffene Aufgabe

ADR 0032/0033 (`rtt-analysis`), ggf. ADR 0031 (ideale Linse).
