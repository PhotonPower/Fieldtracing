# Paper

`fieldtracing.tex` ist die maßgebliche Fassung des Papers. Die CI (`.github/workflows/paper.yml`)
baut bei jeder Änderung ein PDF und legt es als Artefakt `fieldtracing-paper` ab.

Lokal:

```bash
cd docs/paper
pdflatex fieldtracing.tex && pdflatex fieldtracing.tex
```

Benötigt die Pakete `babel` (ngerman), `lmodern`, `koma-script`, `amsmath`, `mathtools`, `bm`,
`booktabs`, `tabularx`, `xcolor`, `hyperref`.

Weicht eine Rechnung dieses Repositories vom Paper ab, wird ein Issue mit der Vorlage
„Abweichung zwischen Rechnung und Paper“ angelegt und das Paper im selben PR wie der Test
korrigiert.
