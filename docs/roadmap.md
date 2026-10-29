# Plan und Meilensteine

> Maßgebliche Fassung des Plans. Änderungen per PR; Entscheidungen zusätzlich als ADR.

## Ziel

Am Ende soll Raytatouille Felder mit der Field-Tracing-Methode berechnen können: hybride
Felddarstellung aus den Strahldaten, geometrische Feldpropagation mit Jacobi-Faktor, PFT, SFT
und FFT je nach den Auswahlregeln aus Kapitel 7 des Papers, vektorielle Fokusfelder und Felder in
beliebigen Ebenen einschließlich Kantenbeugung.

Der Weg dahin hat zwei Teile:

1. **Dieses Repository** prüft jeden Baustein in Python gegen unabhängige Referenzen, belegt
   die Fehlerordnungen und liefert Sollwerte (ADR 0001). Es hält das Paper und die Rechnung
   gegenseitig konsistent.
2. **Raytatouille** bekommt die geprüften Bausteine als C++-Port mit denselben Sollwerten. Dafür
   entstehen hier ADR- und Issue-Entwürfe; umgesetzt wird nach dem Prozess von Raytatouille
   (Koordinator, Freigabe durch den Maintainer).

Leitlinien, die sich aus dem Ziel ergeben:

- Konventionen von Raytatouille von Anfang an (ADR 0002), damit der Port keine Vorzeichen ändert.
- Vektorielle Datenstrukturen von Anfang an (U ∈ ℂ³), auch wenn die ersten Tests skalar sind:
  Raytatouille trägt Polarisation in jedem Strahl (3×3-PRT-Matrix).
- Datenstrukturen als Arrays je Größe (wie Raytatouilles Structure-of-Arrays), reine Funktionen
  ohne globalen Zustand, deterministisch.
- Jede Formel mit Gleichungsverweis, jeder Test mit unabhängigem Sollwert, bei Näherungen ein
  Konvergenztest. Das entspricht den Regeln 2 und 6 von Raytatouille.

## Meilensteine

| Meilenstein | Umfang | Abnahme |
| --- | --- | --- |
| **F0 Fundament** ✅ | Repository, Doku, CI, Konventionen, ADR 0001–0003; 1D-Bausteine; Testfälle T1–T3, T5–T8 | CI grün; Fehler der SFT reproduziert die Tabelle im Paper; PFT und Jacobi-Gesetz konvergieren mit O(1/k) |
| **F1 Bausteine 2D** ✅ | SFT für allgemeines C, inverse PFT/SFT, Minimax-Krümmung, 2D-Kanten (Randintegral, kritische Punkte 2. Art), Debye-Fokus per FFT, LPIA an gekrümmter Fläche mit Strahlnetz | T1–T8 vollständig in 2D; T4 (Debye gegen Rayleigh-Sommerfeld für N_F von 3 bis 10³) |
| **F2 Hybride Darstellung und Verkettung** (#17–#21) ✅ | `GridlessField`/`GriddedField` mit Verhalten, Übergänge Punktwolke ↔ Gitter, Jacobi-Faktor aus Maschenflächen, Auswahlregeln als Planer, Operatorkette | Kette Linse → Blende → Ebene stimmt mit dem Winkelspektrum innerhalb der Toleranzen aus Tabelle 7.3 |
| **F3 Integrationstest Kapitel 7.5** (#39–#43) ✅ | Beispiel D = 25 mm, NA 0,5: Fokus und Zwischenebene; verkleinertes System gegen direkte Rayleigh-Sommerfeld-Rechnung; vektoriell mit Richards-Wolf | Abweichungen und Punktzahlen dokumentiert; Paper bei Abweichungen korrigiert |
| **F4 Brücke zu Raytatouille (lesend)** (#22–#25) ✅ | Adapter: Strahldaten aus der Python-API von Raytatouille → `GridlessField` (Phase aus OPL, Amplitude aus weight und Maschen, Polarisation aus PRT) | Debye-Fokus aus Raytatouille-Strahlen = FFT-PSF von Raytatouille bei kleiner NA; Abweichungen bei hoher NA erklärt (√cos, Vektor) |
| **F5 Integrationsvorschlag** (#50–#53) | ADR-Entwurf `rtt-field`, API-Skizze, Sollwertdateien für den Port, Issue-Entwürfe im Format von Raytatouille | Freigabe durch den Maintainer |
| F6 Port (in Raytatouille) | C++-Bibliothek nach Freigabe, nach den Regeln von Raytatouille | dort |

F1 und F4 sind unabhängig voneinander; F2 braucht F1; F3 braucht F2; F5 braucht F3 und F4.

## F1 im Detail (nächste Arbeit)

Je ein Issue pro Auftrag für einen Agenten, Meilenstein „F1 Bausteine 2D“ auf GitHub.

| Nr. | Issue | Baustein | Inhalt | Referenz und Abnahme |
| --- | --- | --- | --- | --- |
| F1-1 | #2 | `sft.py` | `sft_2d` für symmetrisches C (Transferfunktion direkt mit C⁻¹, keine Gitterdrehung), inverse SFT, Marker `slow` | Separabler Fall = Produkt der 1D-SFT; gedrehtes C gegen direkte 2D-Quadratur; PFT-Grenzfall |
| F1-2 | #3 | `sft.py` / `selection.py` | Minimax-Anpassung der Krümmung c für gegebenes ψ (Kap. 7.4) | Beispiel aus 7.5: c = −521 mm⁻², Restgradient 230 mm⁻¹ |
| F1-3 | #4 | `pft.py` | inverse PFT, Summe über mehrere stationäre Punkte, Warnung bei kleinem F | Hin- und Rück-PFT einer sphärischen Welle; Konvergenz O(1/F) |
| F1-4 | #1 | `focus.py` | Debye-Integral (Gl. debye) per FFT mit subpixelgewichteter k-Berandung und Nullauffüllung | Airy-Integral; T4: Abweichung von Rayleigh-Sommerfeld wächst mit fallendem N_F |
| F1-5 | #5 | `edge.py` | 2D-Randintegral (Gl. bdw), kritische Punkte 2. Art, Ecken | Kreisblende und Rechteckblende gegen 2D-Quadratur; Ordnung k^−1/2 bzw. k^−1 |
| F1-6 | #6 | `lpia.py` | LPIA-Schritt an einer Sphäre mit Strahlnetz, Energiebilanz nach Gl. interface_amplitude | Ebene Welle durch eine Planfläche = Fresnel; Fluss bleibt mit R + T = 1 erhalten |
| F1-7 | #7 | `semiclassical.py` | Maslov-Index am Punktfokus (μ = 2) und an Brennlinien in 2D | 2D-Winkelspektrum: Phasensprung −π bzw. −π/2 |

## F2 im Detail

Meilenstein „F2 Hybride Darstellung und Verkettung“ auf GitHub.

| Nr. | Issue | Inhalt | Abnahme |
| --- | --- | --- | --- |
| F2-1 | #17 | `GridlessField`/`GriddedField` mit Verhalten: Konstruktoren, Phase per Taylor, Umtasten nur des Residuums | Ordnungen der Phasen- und Residuumsinterpolation; Gegenprobe zu ADR 0003 |
| F2-2 | #18 | Jacobi-Faktor aus unstrukturierten Wolken (Delaunay), Kaustik-Erkennung | analytisches J, O(h²), Obskuration, Maslov-Index |
| F2-3 | #19 | einheitliche Operatoren mit Protokoll (Gl. system_chain, Tabelle 7.2) | Regression gegen F1, exakte Operatoren invertierbar |
| F2-4 | #20 | Planer nach Tabelle 7.1, Integrationstest Linse → Blende → Ebene | Abweichung innerhalb des vorhergesagten Fehlerbudgets |
| F2-5 | #21 | Strahlschleife in `lpia` vektorisieren | Tests unverändert, ≥ 50-fach schneller |

Reihenfolge: #17, dann #18 und #19 parallel, dann #20. #21 ist unabhängig.

Abgeschlossen am 2026-10-29. Abnahme: Kette Linse → Kreisblende → Ebene gegen direkte
Rayleigh-Sommerfeld-Quadratur innerhalb des vom Planer vorhergesagten Budgets (Werte in
`docs/testfaelle.md`), ausgenommen eine schmale Zone um die Achse der Kreisblende (Kaustik der
Kantenwelle); die Abweichungen zum Paper stehen in #31.

## F3 im Detail

Meilenstein „F3 Integrationstest Kapitel 7.5“ auf GitHub.

| Nr. | Issue | Inhalt | Abnahme |
| --- | --- | --- | --- |
| F3-1 | #39 | aplanatische Linse als `GridlessField`, Richards-Wolf als Referenzfunktion | Debye gegen Richards-Wolf, Energie |
| F3-2 | #40 | Fokus des großen Systems (D = 25 mm, NA 0,5) | gegen Richards-Wolf im Budget; Punktzahlen, Laufzeit |
| F3-3 | #41 | Zwischenebene z = f/2 des großen Systems | Planer-Wahl, Gitter, Speicher gegen das Paper |
| F3-4 | #42 | verkleinertes System gegen direkte Rayleigh-Sommerfeld-Rechnung | im Budget außerhalb der Achszone |
| F3-5 | #43 | Paper Abschnitt 7.5 abgleichen | Werte mit Verweis auf `docs/beispiel-7-5.md` |

Reihenfolge: #39, dann #40, #41 und #42 (unabhängig voneinander), zuletzt #43.

Abgeschlossen am 2026-10-29: Bericht `docs/beispiel-7-5.md`. Die volle Rechnung der
Zwischenebene (≈ 5–6 GB) ist als Experiment vorbereitet und für kleinere Blenden bei gleichem q
gerechnet; Paper Abschnitt 7.5 nachgezogen.

## F5 im Detail

Meilenstein „F5 Integrationsvorschlag“ auf GitHub. Alles entsteht als Dokument unter
`docs/rtt-port/` bzw. als Sollwertdatei unter `reference/`; in Raytatouille wird nichts angelegt
(Regel 8).

| Nr. | Issue | Inhalt | Abnahme |
| --- | --- | --- | --- |
| F5-1 | #50 | Sollwertdateien im Format `raytatouille-result` (Typ-Vorschlag `FieldReference`) | Roundtrip, Kodierung wie Raytatouilles `encode_json`, Dateien aktuell |
| F5-2 | #51 | ADR-Entwurf `rtt-field` | vollständig, mit Belegen aus F1–F4 |
| F5-3 | #52 | API-Skizze C++ und Python | jede zu portierende Funktion zugeordnet |
| F5-4 | #53 | Issue-Entwürfe im Format von Raytatouille | Freigabe durch den Maintainer |

Reihenfolge: #50, dann #51 und #52, zuletzt #53.

## F4 im Detail

Meilenstein „F4 Brücke zu Raytatouille (lesend)“ auf GitHub; alle Issues mit Label `rtt`.

| Nr. | Issue | Inhalt | Abnahme |
| --- | --- | --- | --- |
| F4-1 | #22 | Raytatouille als optionale Abhängigkeit, Marker `rtt`, Workflow mit Build | Rauchtest im Workflow, Standard-CI unverändert |
| F4-2 | #23 | Umrechnung leistungsnormierte PRT-Matrix und Gewicht → Feldamplitude (ADR) | einzelne Grenzfläche gegen `lpia.fresnel_operator`, Energie gegen `polar.transmission` |
| F4-3 | #24 | Adapter Raytatouille-Strahlen → `GridlessField` | Phase gegen `opd_map`, Nachrechnung mit `lpia.trace_surface` |
| F4-4 | #25 | Debye-Fokus aus Raytatouille-Strahlen gegen deren FFT-PSF | Bericht `docs/vergleich-psf.md`, Übereinstimmung bei kleiner NA |

Reihenfolge: #22 → #23 → #24 (braucht auch #17 und #18) → #25.

Abgeschlossen am 2026-10-29: Brücke über ADR 0004 (`rtt_bridge`), Vergleich der PSF in
`docs/vergleich-psf.md`; Issue-Entwurf für Raytatouille in `docs/rtt-entwurf-psf-hohe-na.md`.

## Entscheidungen

Alle angenommen vom Maintainer am 2026-10-29.

| Nr. | Frage | Entscheidung |
| --- | --- | --- |
| E1 | Bleibt dieses Repository reines Python, oder entstehen C++-Kerne schon hier? | reines Python (NumPy, SciPy); C++ nur in Raytatouille (ADR 0001) |
| E2 | Format der Sollwertdateien für den Port | JSON im Ergebnisformat `raytatouille-result` (ADR 0023 von Raytatouille) |
| E3 | Ort in Raytatouille | neue Bibliothek `rtt-field` in der Schicht Auswertung, oberhalb von `rtt-analysis`; die Reihenfolge legt dort ein ADR fest (Begründung in `docs/integration-raytatouille.md`) |
| E4 | Erster Integrationspunkt in Raytatouille | Vektor-Debye-PSF aus der Jones-Pupille (schließt die in ADR 0033 ausgelassene Vektor-PSF und die √cos-Faktoren); danach Felder in beliebigen Ebenen mit SFT |
| E5 | GitHub-Issues für F1 | angelegt am 2026-10-29 (#1–#7); Reihenfolge siehe unten |

Weitere Festlegungen vom selben Tag:

- ADR 0001–0003 sind angenommen.
- Teure Referenzrechnungen bekommen den pytest-Marker `slow`; die Standard-CI überspringt sie, ein
  eigener Workflow führt sie nachts und manuell aus. Eingeführt mit F1-1.

### Reihenfolge der F1-Issues

1. F1-4 (#1, Debye-Fokus), weil der erste Integrationspunkt (E4) darauf aufbaut; parallel dazu
   F1-1 (#2) und F1-2 (#3) (allgemeine SFT, Minimax-Krümmung), weil davon die SFT bei hoher NA abhängt.
2. Danach F1-3 (#4), F1-5 (#5), F1-6 (#6) und F1-7 (#7).

## Risiken

- **Hohe NA und SFT:** Der nicht-quadratische Phasenrest begrenzt die Schrittweite (Kap. 7.4).
  F1-2 klärt, ob die Minimax-Anpassung reicht oder eine gemischte PFT/SFT-Strategie nötig ist.
- **Jacobi-Faktor aus Strahldaten:** Raytatouille liefert Strahlen auf Pupillengittern, aber
  keine Hesse-Matrix der Wellenfront. Der Faktor muss aus Nachbarstrahlen (Maschenflächen) kommen;
  die Genauigkeit an Kaustiken und am Rand wird in F2 gemessen.
- **Leistungsnormierte PRT-Matrix** (ADR 0021 von Raytatouille): Die Umrechnung von Strahlgewicht
  und PRT in Feldamplituden muss in F4 exakt festgelegt und getestet werden.
- **Laufzeit der Referenztests:** Brute-Force-Quadraturen sind teuer; große Fälle als
  `slow` markieren und nur nachts oder manuell laufen lassen.
