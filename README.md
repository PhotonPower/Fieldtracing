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

**Stand:** Meilenstein F0 (Fundament). Die 1D-Bausteine und die Testfälle T1–T3, T5–T8 aus
Kapitel 8 des Papers sind umgesetzt und grün. Der Plan steht in [docs/roadmap.md](docs/roadmap.md).

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
  field.py               Datenstrukturen der hybriden Darstellung (Gerüst für F2)
tests/                   pytest, ein Testmodul je Baustein, Sollwerte unabhängig berechnet
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
