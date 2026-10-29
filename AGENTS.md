# Regeln für Programmieragenten

Diese Datei liest jeder Agent zu Beginn jeder Sitzung. Sie gilt zusammen mit
`docs/roadmap.md`, `docs/konventionen.md`, `docs/architektur.md` und `docs/adr/`.

## Projekt in einem Satz

Fieldtracing ist die Python-Referenzimplementierung der Field-Tracing-Bausteine aus dem Paper
`docs/paper/fieldtracing.tex`; geprüfte Bausteine werden später nach Raytatouille (C++20)
portiert.

## Bauen und testen

```bash
pip install -r requirements-dev.txt && pip install -e .
pytest
ruff check . && ruff format --check . && mypy
```

Vor jedem PR müssen lokal grün sein: pytest, ruff (check und format), mypy (strict). Die CI
testet zusätzlich Python 3.10 und Windows.

Teure Referenzrechnungen (> 5 s) bekommen den Marker `@pytest.mark.slow`. `pytest` überspringt
sie; `pytest -m slow` führt sie aus, ebenso der nächtliche Workflow `slow.yml`. Wer einen
`slow`-Test ändert, führt ihn vor dem PR lokal aus.

## Aufbau

Siehe `docs/architektur.md`. Ein Modul je Baustein unter `src/fieldtracing/`, ein Testmodul je
Baustein unter `tests/`. Explorative Skripte gehören nach `experiments/` und sind keine Tests.

## Arbeitsregeln

1. **Ein Auftrag = ein Baustein + ein Meilenstein-Ziel** aus `docs/roadmap.md`. Nichts darüber
   hinaus bauen.
2. **Erst Test, dann Code.** Sollwerte kommen aus einer analytischen Lösung oder aus einer
   unabhängigen exakten Rechnung (Winkelspektrum, direkte Quadratur), nie aus dem zu prüfenden
   Code. Bestehende Tests und Sollwerte nur mit Freigabe des Maintainers ändern; einen Test
   anzupassen, damit er grün wird, ist verboten.
3. **Asymptotische Aussagen prüft man über ihre Ordnung.** Eine Näherung führender Ordnung in
   1/k (PFT, Jacobi-Gesetz) bekommt neben dem Absolutfehler einen Konvergenztest: k vervierfachen,
   Fehler muss etwa auf ein Viertel fallen. Diskretisierungsfehler ebenso über die Schrittweite.
4. **Konventionen sind Gesetz** (`docs/konventionen.md`): mm, µm, rad, k in 1/mm,
   exp(i(k·r − ωt)), Fourier-Konvention (2π)^−d, Fresnel-Convention A. Jede öffentliche Funktion
   nennt Einheiten und Vorzeichen im Docstring.
5. **Physik mit Quelle:** jede Formel mit Verweis auf die Gleichung im Paper (Label) oder auf eine
   Quelle aus `docs/quellen.md`. Gleichungsnummern fremder Quellen nur aus dem Volltext.
6. **Hybride Darstellung:** Transformationen geben `(k, residual, phase)` zurück. Interpoliert wird
   nur das glatte Residuum, nie ein Feld oder Spektrum mit schneller Phase (ADR 0003).
7. **Determinismus:** Zufall nur mit explizitem Seed (`np.random.default_rng(seed)`).
8. **Raytatouille nur lesen.** Dieses Projekt ändert Raytatouille nicht. Integrationsvorschläge
   entstehen als Dokumente unter `docs/` (ADR-Entwürfe, Issue-Entwürfe) und gehen über den
   Maintainer.
9. **Keine Geheimnisse im Repository:** keine Tokens, Passwörter oder persönlichen Pfade in
   Dateien, Commits oder Remote-URLs.
10. **Unklarheit = Frage, nicht Annahme.** Offene Konventionen als Issue mit Label `frage`.

## Bekannte Fallstricke

- **Gechirpte Spektren nicht interpolieren.** Das Spektrum einer stark gekrümmten Welle hat die
  Phase −k²/(2c); auf dem Gitter k = c·x springt sie um Dutzende Radiant pro Schritt. Lineare
  Interpolation von Ẽ liefert dann Fehler von 40 % statt 2 %. Immer Residuum interpolieren und die
  Phase analytisch anwenden. Für Referenzwerte an beliebigen k direkt summieren statt eine FFT zu
  interpolieren.
- **Harte Kanten konvergieren nur mit erster Ordnung** in der Schrittweite. Subpixel-gewichtete
  Blenden (`aperture_weights_1d`, `circular_aperture_weights`) verkleinern die Konstante.
- **Referenzquadraturen brauchen genug Punkte pro Phasenperiode** (≥ 10). Bei c = 10⁵ mm⁻² und
  a = 1 mm sind das Millionen Punkte; Testlaufzeit im Blick behalten.
- **numpy-Typen unter mypy:** Eingaben aus `np.linspace` sind `floating[Any]`; die Aliasse in
  `fieldtracing._types` sind deshalb breit. Rückgaben mit `np.asarray(..., dtype=...)` festlegen.

## Sprache und Stil

- Code, Kommentare, Docstrings, Commit-Messages englisch; Projektdokumentation deutsch.
- ruff (Zeilenlänge 100), mypy strict. Funktionen und Variablen `snake_case`, Klassen
  `PascalCase`, Konstanten `UPPER_CASE`.
- Commits im Imperativ, ein Thema pro Commit, z. B. `sft: add 2D transform for general C`.

## Definition of Done (PR-Checkliste)

- [ ] CI grün (Lint, Typen, Tests unter Linux und Windows)
- [ ] Referenztests mit unabhängigem Sollwert, bei Näherungen mit Konvergenztest
- [ ] Testfall in `docs/testfaelle.md` eingetragen oder aktualisiert (Status, gemessene Werte)
- [ ] Neue Quellen in `docs/quellen.md`
- [ ] Docstrings mit Einheiten, Vorzeichen und Verweis auf die Gleichung im Paper
- [ ] Eintrag in `CHANGELOG.md` unter „Unreleased“
- [ ] Bei Abweichung zwischen Rechnung und Paper: Issue mit Label `paper`
