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
| **F1 Bausteine 2D** | SFT für allgemeines C, inverse PFT/SFT, Minimax-Krümmung, 2D-Kanten (Randintegral, kritische Punkte 2. Art), Debye-Fokus per FFT, LPIA an gekrümmter Fläche mit Strahlnetz | T1–T8 vollständig in 2D; T4 (Debye gegen Rayleigh-Sommerfeld für N_F von 3 bis 10³) |
| **F2 Hybride Darstellung und Verkettung** | `GridlessField`/`GriddedField` mit Verhalten, Übergänge Punktwolke ↔ Gitter, Jacobi-Faktor aus Maschenflächen, Auswahlregeln als Planer, Operatorkette | Kette Linse → Blende → Ebene stimmt mit dem Winkelspektrum innerhalb der Toleranzen aus Tabelle 7.3 |
| **F3 Integrationstest Kapitel 7.5** | Beispiel D = 25 mm, NA 0,5: Fokus und Zwischenebene; verkleinertes System gegen direkte Rayleigh-Sommerfeld-Rechnung; vektoriell mit Richards-Wolf | Abweichungen und Punktzahlen dokumentiert; Paper bei Abweichungen korrigiert |
| **F4 Brücke zu Raytatouille (lesend)** | Adapter: Strahldaten aus der Python-API von Raytatouille → `GridlessField` (Phase aus OPL, Amplitude aus weight und Maschen, Polarisation aus PRT) | Debye-Fokus aus Raytatouille-Strahlen = FFT-PSF von Raytatouille bei kleiner NA; Abweichungen bei hoher NA erklärt (√cos, Vektor) |
| **F5 Integrationsvorschlag** | ADR-Entwurf `rtt-field`, API-Skizze, Sollwertdateien für den Port, Issue-Entwürfe im Format von Raytatouille | Freigabe durch den Maintainer |
| F6 Port (in Raytatouille) | C++-Bibliothek nach Freigabe, nach den Regeln von Raytatouille | dort |

F1 und F4 sind unabhängig voneinander; F2 braucht F1; F3 braucht F2; F5 braucht F3 und F4.

## F1 im Detail (nächste Arbeit)

Vorgeschlagene Issues, je ein Auftrag für einen Agenten. Sie werden erst nach Freigabe angelegt.

| Nr. | Baustein | Inhalt | Referenz und Abnahme |
| --- | --- | --- | --- |
| F1-1 | `sft.py` | `sft_2d` für symmetrisches C (Hauptachsen, Rotation des Gitters), inverse SFT | Separabler Fall = Produkt der 1D-SFT; gedrehtes C gegen direkte 2D-Quadratur; PFT-Grenzfall |
| F1-2 | `sft.py` / `selection.py` | Minimax-Anpassung der Krümmung c für gegebenes ψ (Kap. 7.4) | Beispiel aus 7.5: c = −521 mm⁻², Restgradient 230 mm⁻¹ |
| F1-3 | `pft.py` | inverse PFT, Summe über mehrere stationäre Punkte, Warnung bei kleinem F | Hin- und Rück-PFT einer sphärischen Welle; Konvergenz O(1/F) |
| F1-4 | `focus.py` | Debye-Integral (Gl. debye) per FFT mit subpixelgewichteter k-Berandung und Nullauffüllung | Airy-Integral; T4: Abweichung von Rayleigh-Sommerfeld wächst mit fallendem N_F |
| F1-5 | `edge.py` | 2D-Randintegral (Gl. bdw), kritische Punkte 2. Art, Ecken | Kreisblende und Rechteckblende gegen 2D-Quadratur; Ordnung k^−1/2 bzw. k^−1 |
| F1-6 | `lpia.py` | LPIA-Schritt an einer Sphäre mit Strahlnetz, Energiebilanz nach Gl. interface_amplitude | Ebene Welle durch eine Planfläche = Fresnel; Fluss bleibt mit R + T = 1 erhalten |
| F1-7 | `semiclassical.py` | Maslov-Index am Punktfokus (μ = 2) und an Brennlinien in 2D | 2D-Winkelspektrum: Phasensprung −π bzw. −π/2 |

## Entscheidungen

Alle angenommen vom Maintainer am 2026-10-29.

| Nr. | Frage | Entscheidung |
| --- | --- | --- |
| E1 | Bleibt dieses Repository reines Python, oder entstehen C++-Kerne schon hier? | reines Python (NumPy, SciPy); C++ nur in Raytatouille (ADR 0001) |
| E2 | Format der Sollwertdateien für den Port | JSON im Ergebnisformat `raytatouille-result` (ADR 0023 von Raytatouille) |
| E3 | Ort in Raytatouille | neue Bibliothek `rtt-field` in der Schicht Auswertung, oberhalb von `rtt-analysis`; die Reihenfolge legt dort ein ADR fest (Begründung in `docs/integration-raytatouille.md`) |
| E4 | Erster Integrationspunkt in Raytatouille | Vektor-Debye-PSF aus der Jones-Pupille (schließt die in ADR 0033 ausgelassene Vektor-PSF und die √cos-Faktoren); danach Felder in beliebigen Ebenen mit SFT |
| E5 | GitHub-Issues für F1 | werden angelegt, sobald ein auf dieses Repository beschränktes Token vorliegt; Reihenfolge siehe unten |

Weitere Festlegungen vom selben Tag:

- ADR 0001–0003 sind angenommen.
- Teure Referenzrechnungen bekommen den pytest-Marker `slow`; die Standard-CI überspringt sie, ein
  eigener Workflow führt sie nachts und manuell aus. Eingeführt mit F1-1.

### Reihenfolge der F1-Issues

1. F1-4 (Debye-Fokus), weil der erste Integrationspunkt (E4) darauf aufbaut; parallel dazu
   F1-1 und F1-2 (allgemeine SFT, Minimax-Krümmung), weil davon die SFT bei hoher NA abhängt.
2. Danach F1-3, F1-5, F1-6 und F1-7.

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
