# Sollwertdateien für den Port nach Raytatouille

Je Testfall eine Datei `field/<fall>.result.json` im Ergebnisformat von Raytatouille
(`raytatouille-result`, dort ADR 0023; Entscheidung E2 in `docs/roadmap.md`). Erzeugt mit

```bash
python tools/export_reference.py            # alle Fälle
python tools/export_reference.py pft_sphere # einzelne Fälle
```

`tests/test_reference_files.py` rechnet jeden Fall neu und vergleicht mit der Datei: Wer einen
Baustein ändert, muss die Dateien neu schreiben, sonst schlägt der Test fehl.

## Hülle und Kodierung

Wie in Raytatouille (`raytatouille.results`, Stand `rtt-commit.txt`, Schema 0.1.13):
`{"format": "raytatouille-result", "schema_version": "0.1.13", "type": "FieldReference",
"data": {…}}`; Arrays als `{"dtype", "shape", "values"}` in C-Reihenfolge, komplexe Werte als
`[re, im]`, NaN/±∞ in Gleitkomma-Arrays als `"NaN"`, `"Infinity"`, `"-Infinity"`, als Skalar
`{"float": …}`, komplexe Skalare `{"complex": [re, im]}`. Die Kodierung ist mit Raytatouilles
`encode_json` identisch (Test mit Marker `rtt`).

## Ergebnistyp `FieldReference` (Vorschlag für Raytatouille)

Raytatouilles `load_json` kennt diesen Typ noch nicht; er ist Teil des ADR-Entwurfs für
`rtt-field` (`docs/rtt-port/`). Felder von `data`:

| Schlüssel | Inhalt |
| --- | --- |
| `case` | Name des Falls (Dateistamm) |
| `description` | ein Satz |
| `source` | Test(s) in diesem Repository und Gleichung im Paper |
| `parameters` | Skalare und Texte; Einheiten im Namen (`f_mm`, `k_per_mm`) |
| `inputs` | Arrays, die der Port in die Funktion gibt |
| `expected` | Ergebnis dieser Implementierung; der Port rechnet denselben Algorithmus |
| `tolerance` | `{"rel", "abs"}` für den Vergleich mit `expected` (Rundung, Plattform) |
| `reference` | unabhängiger Wert (analytisch oder exakt), soweit vorhanden |
| `reference_tolerance` | physikalische Toleranz für `reference`, gemessen in den Tests aus `source` |

## Fälle

| Fall | Baustein (Python) | Referenz |
| --- | --- | --- |
| `fourier_gaussian` | `fourier.ft` | Gauß analytisch, 10⁻¹⁵ |
| `angular_spectrum_tilted_wave` | `propagation.angular_spectrum` | ebene Welle exakt, 10⁻¹⁰ |
| `pft_sphere` | `pft.pft_2d` | exakte Legendre-Transformation, \|det H\|^(−½), 10⁻⁹ |
| `sft_gauss_chirp` | `sft.sft_2d` (512², Teilmenge 9×9) | geschlossene Form, 10⁻⁶ des Maximums |
| `debye_airy` | `focus.debye_focus` | Airy-Integral, 3·10⁻⁴ |
| `richards_wolf_focus` | `focus.richards_wolf`, `debye_focus` vektoriell | Richards-Wolf, 10⁻⁴ |
| `fresnel_convention_a` | `lpia.fresnel_coefficients_batch` | R + T = 1 |
| `geometric_propagation_astigmatic` | `semiclassical.propagate_2d` | – (Regression; physikalisch in T2/T7) |
| `mesh_jacobian_jittered` | `mesh.mesh_jacobian_cloud` | analytisches J, 5·10⁻⁴ |
| `lpia_sphere_about_focus` | `lpia.trace_surface`, `interface_amplitude` | \|t\|·r₁/r₂, 10⁻¹² |
| `field_operator_plate` | `rtt_bridge.field_operator` (ADR 0004) | √s·T₂T₁, 10⁻¹⁴ |
| `selection_rules` | `selection.select_method` | Entscheidungstabelle (exakt) |
