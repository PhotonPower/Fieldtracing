# Fieldtracing

Numerische Referenzimplementierung und Testprogramme zum Paper
[„Theoretische Grundlagen des Field Tracing“](docs/paper/fieldtracing.tex): hybride
Felddarstellung U·exp(iψ), geometrische Feldpropagation mit Jacobi-Faktor und Maslov-Index,
punktweise und semi-analytische Fourier-Transformation (PFT, SFT), lokale
Ebene-Grenzflächen-Näherung (LPIA), Kantenbeugung und Fokussierung.

**Ziel:** Die hier geprüften Bausteine sollen später [Raytatouille](https://github.com/PhotonPower/Raytatouille)
um Field Tracing erweitern. Deshalb gelten hier dieselben Konventionen (mm, µm, rad,
exp(i(k·r − ωt)), Fresnel-Convention A), und jede Formel wird gegen eine unabhängige Referenz
getestet, bevor sie portiert wird.

**Stand:** Die Meilensteine F0–F5 sind abgeschlossen: alle Bausteine in 1D und 2D, hybride
Darstellung mit Operatorkette und Planer, Integrationstest des Beispiels aus Abschnitt 7.5, lesende
Brücke zu Raytatouille und der vom Maintainer freigegebene Integrationsvorschlag (ADR-Entwurf,
API-Skizze, Sollwertdateien unter `reference/field/`). Der Port (F6) geschieht in Raytatouille nach
dessen Release 0.9.0. Plan und Nacharbeiten: [docs/roadmap.md](docs/roadmap.md).

## Installieren und testen

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
pytest                       # alle Referenztests
ruff check . && ruff format --check . && mypy
```

Python ≥ 3.10; Laufzeitabhängigkeiten nur NumPy und SciPy.

## Aufbau

```text
src/fieldtracing/        Python-Paket (Code und Kommentare englisch)
  fourier.py             Fourier-Konvention des Papers auf zentrierten Gittern
  propagation.py         Winkelspektrum, Rayleigh-Sommerfeld- und Van-Vleck-Kern
  semiclassical.py       geometrische Propagation: Jacobi-Gesetz, Maslov-Index
  pft.py, sft.py         punktweise und semi-analytische Fourier-Transformation
  edge.py                Kantenwelle und gleichmäßige erfc-Lösung
  focus.py               Fokuskette mit exakt sphärischer Phase, Airy-Integral
  lpia.py                vektorielles Snellius-Gesetz, Fresnel-Operator (3×3)
  selection.py           Kennzahlen und Auswahlregeln (Kapitel 7)
  field.py               hybride Darstellung: GridlessField, GriddedField
  mesh.py                Jacobi-Faktor und Kaustiken aus unstrukturierten Strahlwolken
  chain.py               Operatoren mit Protokoll, Planer, Operatorkette
  rtt_bridge.py          Strahldaten aus Raytatouille → hybrides Feld (ADR 0004)
  results.py             Sollwertdateien im Format raytatouille-result
tests/                   pytest, ein Testmodul je Baustein, Sollwerte unabhängig berechnet
reference/field/         Sollwertdateien für den Port
tools/                   Erzeugen der Sollwertdateien
experiments/             explorative Skripte (nicht Teil der Tests)
docs/                    Plan, Konventionen, Architektur, Testfälle, Quellen, ADRs, Paper
```

## Dokumentation

- [Plan und Meilensteine](docs/roadmap.md)
- [Konventionen](docs/konventionen.md) (verbindlich, abgestimmt mit Raytatouille)
- [Architektur des Pakets](docs/architektur.md)
- [Testfälle mit Sollwerten](docs/testfaelle.md)
- [Integration in Raytatouille](docs/integration-raytatouille.md)
- [Geprüfte Literaturquellen](docs/quellen.md)
- [Architekturentscheidungen](docs/adr/README.md)
- [Regeln für Programmieragenten](AGENTS.md)
- [Paper (LaTeX)](docs/paper/fieldtracing.tex), die CI baut daraus ein PDF

## Lizenz

MIT, siehe [LICENSE](LICENSE).
