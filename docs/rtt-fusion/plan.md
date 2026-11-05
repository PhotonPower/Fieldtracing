# Plan: Raytatouille nach v1.0 – Port von Field Tracing und „Raytatouille Fusion“

> **Entwurf** (2026-11-05) zur Abstimmung mit dem Maintainer. Nichts davon ist in Raytatouille
> angelegt (Fieldtracing, AGENTS.md Regel 8); Entscheidungen fallen dort nach dessen Prozess.
> Stand Raytatouille: Commit `6078e55` (nach 0.9.0, M8 offen), gelesen am 2026-11-05.

## 1. Ziel

Raytatouille soll nach v1.0 optische Systeme rechnen können, in denen geometrische Optik,
Field Tracing und rigorose Wellenoptik zusammenwirken. Vorgesehene Module: sequenzielles und
später nicht-sequenzielles Raytracing (vorhanden bzw. M9), Field Tracing (`rtt-field`), ein
vektorielles BPM, RCWA für periodische Strukturen, BEM für kleine Partikel, FEM für allgemeine
Nahfeldprobleme und ein bio-optisches Modul. Mehrere davon liegen bereits als headless Programme
mit Python-Anbindung vor.

Daraus folgen drei Aufgaben, die dieser Plan nacheinander und mit Abhängigkeiten beschreibt:

1. **Port von Field Tracing** (F6, Bibliothek `rtt-field`) nach den freigegebenen Entwürfen in
   `docs/rtt-port/`. Er liefert die Brücke Strahl ↔ Feld, ohne die keine Fusion möglich ist.
2. **Wellenoptische Blackbox:** eine einheitliche Schnittstelle, über die externe Löser als
   Gebiete mit Ein- und Ausgangsports in eine Rechnung eingebunden werden, ohne dass ihr Code in
   den Kern von Raytatouille wandert.
3. **Pfadplanung:** eine Verallgemeinerung der expliziten Pfade von Raytatouille („Pfade sind
   Daten“) auf Pfade durch Gebiete verschiedener Verfahren, mit einem Planer, der je Abschnitt
   Verfahren, Darstellung, Abtastung und Fehlerbudget wählt, wie es Kapitel 7 des Papers für die
   Feldbausteine tut.

## 2. Ausgangslage

**Raytatouille** (C++20, Python-API, MIT, headless und deterministisch):

- Strahlen als Structure-of-Arrays mit leistungsnormierter PRT-Matrix P und Gewicht (ADR 0005,
  0021); Pfade als explizite Ereignislisten, mehrere benannte Pfade, Ghost-Generator (ADR 0027),
  Gitter als Phasenschicht mit Ordnung am Ereignis (ADR 0025).
- Ergebnisformat `raytatouille-result` (ADR 0023), Diagnosecodes (ADR 0022), Änderungen am Modell
  über JSON Patch (ADR 0024), Python-Callbacks nur über die API und nie im Dateiformat
  (Entscheidung zu O2/F3 in `docs/roadmap-erweiterungen.md`).
- Scope laut `docs/architecture.md`: **„Nicht geplant: rigorose EM-Solver (RCWA, FDTD) …
  Dafür sind Import-Schnittstellen (Jones-Tabellen, Coating-Dateien) vorgesehen.“** Keine GPU in
  v1 (ADR 0013). Nach v1.0 als erste Priorität M9 (nicht-sequenziell mit Streuung), dann M10
  (Python-GUI); Beamlets, GPU-Backend und globale Optimierung unter „v2+“.
- Field Tracing: ADR `rtt-field` angenommen (Raytatouille #284), Start nach v1.0 (M8); #285 (C++-
  Ersatz für SciPy-Bausteine) offen.

**Fieldtracing** (dieses Repository): alle Bausteine geprüft, Sollwertdateien unter
`reference/field/`, ADR-Entwurf, API-Skizze und Issue-Entwürfe I1–I3 unter `docs/rtt-port/`.

**Folgerung für die Fusion:** Der bestehende Scope schließt rigorose Löser *im Kern* aus, nicht
ihre *Kopplung*. Der Plan hält daran fest: Löser bleiben außerhalb, der Kern bekommt
Darstellungen, Umwandlungen, Schnittstelle und Planer. Das ändert trotzdem die Scope-Aussage in
`architecture.md` und braucht dort ein ADR (Abschnitt 9, Entscheidung D1).

## 3. Phasen im Überblick

| Phase | Inhalt | Ort | Voraussetzung | Abnahme |
| --- | --- | --- | --- | --- |
| **P0 Vorbereitung** | Inventar der Löser, ADR-Entwurf „Fusion“, Konformitätsprüfung, Prototyp-Repository, Pilotfälle | Fieldtracing / neues Repo | – (jetzt, parallel zu M8) | Entwürfe freigegeben |
| **P1 Port `rtt-field`** | I1 Vektor-Debye-PSF aus Strahlen, I2 Felder in Ebenen, I3 hybride Feldverfolgung | Raytatouille | v1.0, #285 entschieden | Sollwertdateien grün |
| **P2 Blackbox-Prototyp** | Schnittstelle, zwei Pilotlöser, Darstellungen, Umwandlungen, Pilotfälle in Python | Prototyp-Repo | P0; P1 nur für den Strahl-Feld-Teil | Pilotfälle gegen unabhängige Referenz |
| **P3 Kern der Fusion** | Darstellungen und Ports in C++, Gebiete im Modell, Tabellenkopplung im Tracer, Ausführung und Cache | Raytatouille | P1, P2, ADR Fusion | Pilotfälle aus P2 bitgleich bzw. in Toleranz |
| **P4 Pfadplanung** | Topologie, Darstellungswahl, Abtastung, Fehlerbudget, Kohärenz; Abstimmung mit M9 | Raytatouille | P3; M9-Entwurf | Entscheidungstabellen, End-to-End gegen Brute-Force |
| **P5 Ausbau** | weitere Löser, GPU, Optimierung durch gekoppelte Systeme, GUI | Raytatouille und Adapter | P4 | je Modul eigene Abnahme |

P0 kann sofort beginnen. P1 und P2 laufen nach v1.0 parallel, weil P2 nur für den
Strahl-Feld-Übergang auf P1 wartet und sich bis dahin mit der Python-Referenz aus Fieldtracing
behilft. Die Abfolge folgt dem Muster, das sich bei Field Tracing bewährt hat: Prototyp in Python
mit unabhängigen Referenzen und Sollwertdateien, dann Port.

## 4. Phase P0: Vorbereitung (jetzt bis v1.0)

1. **Inventar der Löser.** Für jedes vorhandene Programm (BPM, RCWA, BEM, FEM, bio-optisch) eine
   Zeile mit: Verfahren und Gültigkeitsbereich, Sprache, Lizenz, Python-Anbindung, Ein- und
   Ausgabedarstellungen, Zeit- und Fourier-Konvention, Polarisationsbasis, Einheiten, Linearität,
   Determinismus, GPU, typische Laufzeit und Speicher, vorhandene Validierung. Vorlage in Anhang A.
   Das Inventar entscheidet, welche Darstellungen der Kern braucht.
2. **Pilotfälle auswählen** (Abschnitt 8). Zwei Löser für P2, Vorschlag: **RCWA** (einfachste
   Ports: ebene Wellen und Beugungsordnungen, direkter Nutzen im sequenziellen Tracer) und
   **BEM** (geschlossene Oberfläche, fokussiertes Vektorfeld aus `rtt-field` als Einfall, Mie als
   Referenz). BPM folgt in P5, weil es die Feldkopplung über ausgedehnte Gitter braucht.
3. **ADR-Entwurf „Fusion“** (Abschnitt 9): Scope, Darstellungen, Schnittstelle, Ausführung,
   Modell, Planer, Kohärenz, Determinismus.
4. **Konformitätsprüfung** für Löser spezifizieren (Abschnitt 6.5), bevor ein Adapter entsteht.
5. **Prototyp-Repository** anlegen (Vorschlag: `PhotonPower/Fusion`, Python, gleiche Regeln wie
   Fieldtracing: Test zuerst, unabhängige Sollwerte, Sollwertdateien im Format
   `raytatouille-result`). Fieldtracing bleibt die Referenz für Feldbausteine und wird dort als
   Abhängigkeit genutzt, nicht erweitert.
6. **M9 im Blick:** Der Entwurf von M9 (BVH, Ray Splitting, Streumodelle) fällt in dieselbe Zeit.
   Wie beim GRIN-ADR (X1, „ADR früh, damit … nicht fest einbauen“) sollte der Fusion-Entwurf vor
   dem M9-ADR vorliegen, damit Pfade und Strahlteilung Gebiete mit Ports nicht ausschließen.

## 5. Phase P1: Port von Field Tracing

Nach `docs/rtt-port/issues.md` und dem angenommenen ADR, in dieser Reihenfolge:

1. **Start-Checkliste** (E8 in `docs/roadmap.md`): `rtt-commit.txt` auf v1.0 setzen,
   `pytest -m rtt` auf dem Rechner des Maintainers, Abweichungen durch M8 (Flächentypen,
   Polygon-Aperturen, Obskurationen) als Issues; #285 entscheiden (Delaunay, KD-Baum,
   Interpolation, Minimax-LP, Bessel); Meilenstein „Field Tracing“ in Raytatouille.
2. **Grundlagen:** `fourier`, `propagation`, `geometric`, `mesh`, `pft`, `sft`, `edge`
   (einschließlich der gleichmäßigen 2D-Lösung, Sollwertdatei `edge_uniform_rectangle`),
   `lpia`, `field`, `selection` – je Header ein Issue mit Sollwertdatei.
3. **I1 Vektor-Debye-PSF aus Strahlen** (erster Integrationspunkt, schließt H7 aus ADR 0033).
4. **I2 Felder in beliebigen Ebenen** mit SFT/FFT nach den Auswahlregeln, Plan als Ergebnis.
5. **I3 hybride Feldverfolgung** durch das System (`chain`), Strahlen von `rtt-trace`.

Für die Fusion wichtig sind aus P1 vor allem: `rays.hpp` (Strahlen → hybrides Feld), die
umgekehrte Richtung (Feld → Strahlen über die lokalen Gradienten, in Fieldtracing als PFT und
`GridlessField` vorhanden), die Auswahlregeln als reine Funktion und der Plan als Datenobjekt.
Diese Teile sollten in P1 so geschnitten werden, dass P3 sie ohne Umbau wiederverwendet.

## 6. Die wellenoptische Blackbox

### 6.1 Gebiete und Ports

Ein **Gebiet** ist ein räumlicher Bereich, in dem genau ein Verfahren rechnet: ein Abschnitt
sequenzieller Flächen (Raytracing), freie Ausbreitung (Field Tracing), oder ein externer Löser
(BPM, RCWA, BEM, FEM, bio-optisch). Gebiete tauschen Licht nur über **Ports** aus. Ein Port hat:

- eine Geometrie: Ebene mit lokalem Koordinatensystem und Ausdehnung, oder geschlossene Fläche
  (Kugel, Hüllquader) bei Streukörpern;
- eine Richtung (ein, aus, beides) und die Medien auf beiden Seiten;
- eine **Darstellung** (6.2) mit Abtastung, Wellenlänge und Polarisationsbasis;
- eine Kohärenzangabe: kohärentes Feld oder inkohärente Leistungsgröße.

### 6.2 Darstellungen

Der Kern kennt eine kleine, abgeschlossene Menge von Darstellungen; jede Umwandlung zwischen
ihnen ist ein geprüfter Operator. Neue Darstellungen brauchen ein ADR.

| Darstellung | Inhalt | Kohärenz | Herkunft |
| --- | --- | --- | --- |
| **R Strahlen** | `RayBatch`: Ort, Richtung, OPL, P, Gewicht, Status | kohärent über OPL und P, oder inkohärent | Raytatouille |
| **H hybrides Feld** | `GridlessField`: U ∈ ℂ³, ψ, ∇ψ, Hesse-Matrix | kohärent | `rtt-field` |
| **G Gitterfeld** | `GriddedField`: Werte ∈ ℂ³ auf Gitter mit analytischer Trägerphase | kohärent | `rtt-field` |
| **S Ebene-Wellen-Spektrum** | diskrete Liste (k∥, Amplitude ∈ ℂ², Basis s/p) oder Gitterspektrum; für periodische Ports Floquet-Ordnungen | kohärent | neu |
| **M Streumatrix / Tabelle** | linearer Operator eines Gebiets: S-Matrix in Ordnungen, Jones- oder Mueller-Tabelle über (Einfallsrichtung, λ), T-Matrix in Kugelwellen | kohärent (Jones, S, T) bzw. inkohärent (Mueller) | neu, Tabellen in X2 vorgesehen |
| **P radiometrisch** | Strahlen oder Gitter mit Stokes-Vektor bzw. Strahldichte, ohne Phase | inkohärent | neu (bio-optisch, M9-Streuung) |

Kugelwellen (T-Matrix) sind die natürliche Darstellung für BEM und FEM an Partikeln; sie kommen
erst mit dem BEM-Pilot und nur als Inhalt von M, nicht als allgemeiner Port.

### 6.3 Umwandlungen

| von → nach | Operator | Gültigkeit (Kennzahl) | Quelle |
| --- | --- | --- | --- |
| R → H | Phase k₀·OPL, Amplitude nach Fieldtracing ADR 0004, Jacobi aus dem Netz | Strahlnetz dicht genug, keine Kaustik am Port | `rtt-field` I3 |
| H → R | Strahlen entlang ∇ψ, Gewicht aus \|U\|² und Jacobi | lokale Fresnel-Zahl ≫ 1 | Fieldtracing `field` |
| H ↔ G | Umtasten nur des Restfelds (ADR 0003 von Fieldtracing) | Abtastung nach den Auswahlregeln | `rtt-field` |
| G ↔ S | FFT, SFT, PFT | Tabelle 7.1 des Papers | `rtt-field` |
| S → R | eine Strahlfamilie je Ordnung bzw. Spektralbereich | Ordnungen getrennt, Bündel groß gegen λ | neu |
| S ↔ M | Anwenden der Matrix auf Ein-Ordnungen; Tabellen über k∥ interpolieren (nur glatte Größen, analog ADR 0003) | Tabelle fein genug, keine Resonanzen zwischen Stützstellen | neu |
| H/G → Kugelwellen | Projektion des einfallenden Felds (Vektor-Kugelfunktionen) | Feld im Hüllkörper bekannt | neu (BEM-Pilot) |
| R ↔ P | Gewicht und Stokes aus P; Rückweg ohne Phase | nur inkohärente Weiterrechnung | neu |

Jeder Operator bekommt eine Sollwertdatei mit unabhängiger Referenz und, bei Näherungen, einen
Konvergenztest – dieselben Regeln wie in Fieldtracing.

### 6.4 Schnittstelle eines Lösers

Ein Löser wird über einen **Adapter** eingebunden, der vier Aufrufe bereitstellt:

1. `describe()` – Metadaten: Name, Version, Verfahren, unterstützte Darstellungen je Port,
   Linearität, Reziprozität, Konventionen (Zeitvorzeichen, Fourier, Polarisationsbasis,
   Einheiten), Determinismus (bitgleich / in Toleranz), Ressourcen (GPU, Speicher).
2. `prepare(structure, wavelengths)` – Struktur einlesen und vorrechnen (etwa Eigenmoden der
   RCWA-Schichten, BEM-Matrix); liefert einen Griff und einen Inhalts-Hash.
3. `apply(handle, port_in, field) → {port_out: field}` – ein Einfallsfeld durchrechnen.
4. `operator(handle, port_in, port_out, basis)` – optional für lineare Löser: die Streumatrix bzw.
   Tabelle in einer gewählten Basis. Damit kann der Kern vorrechnen, zwischenspeichern und
   dieselbe Rechnung für viele Strahlen, Felder und Optimierungsschritte nutzen.
5. `estimate(handle, request) → (Laufzeit, Speicher, Fehler)` – Kostenmodell für den Planer;
   anfangs grob, aus Messungen kalibriert.

Die Daten an der Grenze sind **Werte, keine Objekte**: Darstellungen nach 6.2, kodiert wie
`raytatouille-result` (ADR 0023). Damit ist dieselbe Schnittstelle in drei Ausführungsarten
nutzbar:

| Ausführung | Wann | Eigenschaften |
| --- | --- | --- |
| **in-process Python** | Prototyp, vorhandene Python-Anbindungen | schnell zu bauen; GIL und Fremdabhängigkeiten im selben Prozess |
| **Unterprozess** | Standard für Produktion | Isolation (Abstürze, Lizenzen, GPU-Treiber), Austausch über Dateien oder Shared Memory, Cache auf Platte |
| **entfernter Dienst** | große FEM-/BEM-Läufe auf dem Rechner mit viel Speicher | gleiche Daten, Übertragung über Netz; erst bei Bedarf |

Der Kern von Raytatouille hängt von keinem Löser ab und linkt keinen. Adapter liegen in eigenen
Paketen (`raytatouille-fusion-rcwa` usw.) mit eigener Lizenzprüfung. Das hält den MIT-Kern sauber
und erfüllt die Scope-Aussage „rigorose Löser nicht im Kern“.

### 6.5 Konformitätsprüfung

Ein Adapter wird erst aufgenommen, wenn er einen gemeinsamen Testsatz besteht. Er prüft vor
allem die Konventionen, denn hier entstehen die teuersten Fehler (viele RCWA- und FEM-Codes
rechnen mit exp(−i(k·r − ωt)) oder anderen Polarisationsbasen):

- **Ebene Welle durch eine homogene Platte:** Fresnel nach Convention A und Airy-Formel; prüft
  Zeitvorzeichen, s/p-Basis, Normierung der Leistung (Raytatouille ADR 0021: leistungsnormierte
  P) und die Phase der Durchgänge.
- **Energieerhaltung** über alle Ports bei verlustfreier Struktur, **Reziprozität** bei
  reziproker Struktur.
- **Verfahrensspezifisch, unabhängig:** RCWA dünnes Gitter gegen skalare Beugungseffizienz und
  Gittergleichung (ADR 0025); BEM und FEM Kugel gegen Mie; BPM im homogenen Medium gegen das
  Winkelspektrum, im Parabolprofil gegen den analytischen Gaußmodus.
- **Determinismus:** gleiche Eingabe → gleiche Ausgabe; wo ein Löser auf der GPU nicht bitgleich
  ist, eine dokumentierte Toleranz statt Bitgleichheit.

Die Konventionsanpassung (Vorzeichen, Basiswechsel, Einheiten) geschieht nur im Adapter, nie im
Kern.

## 7. Pfadplanung

### 7.1 Fusionspfade

Raytatouille beschreibt einen Pfad als explizite Folge von Ereignissen an Flächen. Ein
**Fusionspfad** verallgemeinert das: eine Folge von Abschnitten, jeder in genau einem Gebiet, und
Übergängen an Ports. Ein Abschnitt in einem Löser-Gebiet trägt, analog zur Beugungsordnung am
Ereignis (ADR 0025), den gewählten **Kanal**: Beugungsordnung, Reflexion oder Transmission,
Streurichtung bzw. Ausgangsport. Damit bleiben Pfade Daten, Strahlteilung durch einen Löser
erzeugt mehrere Pfade, und der Tracer braucht keinen Sonderfall.

### 7.2 Planer in vier Stufen

Der Planer ist eine reine Funktion vom Modell und den Anforderungen (Ausgaben, Toleranz ε,
Ressourcen) auf einen **Plan als Datenobjekt**: welche Abschnitte, welche Darstellung und
Abtastung je Port, welcher Operator je Übergang, welches Fehlerbudget, mit Begründung. Der
Nutzer kann jeden Punkt überschreiben (wie `plan_propagation` in Fieldtracing).

1. **Topologie.** Alle Fusionspfade von den Quellen zu den Ausgaben aufzählen: Folge der Gebiete,
   Kanäle je Löser (Ordnungen, Reflexion/Transmission), Mehrfachdurchgänge bis zu einer Tiefe;
   Pfade mit geschätzter Leistung unter einer Schwelle werden mit Diagnose verworfen (wie das
   Ghost-Ranking, ADR 0027). Ergebnis: ein Baum bzw. DAG gemeinsamer Präfixe, damit gemeinsame
   Abschnitte nur einmal gerechnet werden.
2. **Darstellungswahl.** Je Abschnitt und Port die Darstellung, je Übergang den Operator. Formal
   ein kürzester Weg im Graphen der Zustände (Abschnitt, Darstellung): Kanten sind Operatoren mit
   Kosten aus `estimate` und Fehlerbeiträgen, unzulässige Kanten fallen über die
   Gültigkeitskennzahlen weg. Kennzahlen: lokale Fresnel-Zahlen und Kantenverhältnis (Kapitel 7
   des Papers), Strukturgröße zu λ (Strahl- vs. Wellenoptik), Größenparameter x = 2πa/λ
   (Partikel: Rayleigh, BEM, FEM, Mie), Winkelbreite des Spektrums gegen die Paraxialität des BPM,
   Zahl der Ordnungen, Kohärenzlänge zu Wegunterschieden.
3. **Abtastung und Budget.** Gitter, Strahlzahlen und Tabellendichte je Port aus dem
   Fehlerbudget; das Budget wird nach dem Prinzip des Papers verteilt (der größte Term dominiert,
   die übrigen deutlich darunter). Speichergrenze als harte Nebenbedingung (Beispiel 7.5:
   ≈ 1,3 GB je vektoriellem Feld).
4. **Ausführungsplan.** DAG aus Löser-Aufrufen und Umwandlungen, parallel über Wellenlängen,
   Feldpunkte und Polarisationszustände, mit Cache je Inhalts-Hash (Struktur, Löser-Version, λ,
   Einfall). Lineare Löser mit `operator` werden einmal tabelliert und dann angewandt.

### 7.3 Kohärenz und Überlagerung

Regeln, die der Plan festhält und die Ausgabe dokumentiert:

- Amplituden addieren sich innerhalb einer **Kohärenzgruppe**: gleiche Wellenlänge, gleicher
  Quellpunkt bzw. gleiche Mode, gleicher Polarisationszustand der Quelle, Wegunterschiede unter
  der Kohärenzlänge. Über Gruppen hinweg addieren sich Intensitäten bzw. Stokes-Vektoren.
- Zwei Pfade, die in denselben Ausgangsport münden, werden nur kohärent überlagert, wenn beide in
  einer kohärenten Darstellung ankommen; sonst warnt der Planer.
- Das bio-optische Modul ist voraussichtlich inkohärent (Strahlungstransport); seine Ports sind
  radiometrisch (P), und der Planer schaltet hinter ihm auf inkohärente Weiterrechnung um.

### 7.4 Beziehung zu M9

M9 bringt BVH, Quellen, Detektoren, Strahlteilung und Streumodelle. Die Fusionspfade sollten
dieselbe Pfadbeschreibung nutzen: Ein Löser-Gebiet ist aus Sicht des nicht-sequenziellen Tracers
ein Körper mit Ports, an dem Strahlen enden (R → Einfall) und neue Strahlfamilien entstehen
(Ausgang → R). Ein Streumodell in M9 (Lambert, Gauß, ABg) und eine tabellierte Mueller- oder
BSDF-Antwort eines Lösers sind dann zwei Formen derselben Interaktion. Abstimmung über den
M9-ADR (Entscheidung D6).

## 8. Pilotfälle

Jeder Fall prüft eine Kopplungsart gegen eine Referenz, die nicht aus den gekoppelten Teilen
stammt.

| Nr. | Fall | Kopplung | Referenz | Phase |
| --- | --- | --- | --- | --- |
| K1 | Gitterspektrometer mit RCWA-Effizienzen je Ordnung und Polarisation | Tabelle (M) als Interaktion im sequenziellen Tracer | Gittergleichung und dünnes Gitter; Energie über die Ordnungen | P2/P3 |
| K2 | Metafläche bzw. DOE als Linse, RCWA-Tabelle über k∥ | Tabelle mit Phase im Feldpfad (H) | Fokus gegen Richards-Wolf bei idealer Phase | P2/P3 |
| K3 | Partikel im Fokus: Mikroskopobjektiv → fokussiertes Vektorfeld → BEM-Streuung → Rückweg durch das Objektiv | Feld (H/G) → Kugelwellen → S → R/H | Mie für die Kugel; Fokusfeld gegen Richards-Wolf | P2/P3 |
| K4 | Faser- bzw. Wellenleiterkopplung mit vektoriellem BPM | G → BPM → G, Überlapp mit Moden | Gaußmodus im Parabolprofil, Winkelspektrum im homogenen Teil | P5 |
| K5 | Gewebeprobe mit bio-optischem Modul, Abbildung des Streulichts | R → P → R (inkohärent) | Energiebilanz; Grenzfall ohne Streuung = reines Raytracing | P5 |
| K6 | Brute-Force-Vergleich eines kleinen Gesamtsystems (2D oder kleine Apertur) | gesamte Kette | ein einziger rigoroser Lauf (FEM bzw. FDTD) über alles | P4 |

K6 ist die Abnahme des Planers: Der Plan muss im vorhergesagten Budget liegen, wie es F2 und F3 in
Fieldtracing für die Feldbausteine gezeigt haben.

## 9. Entscheidungen, die vorher fallen müssen

| Nr. | Frage | Vorschlag |
| --- | --- | --- |
| D1 | Scope von Raytatouille: Kopplung rigoroser Löser aufnehmen? | Ja, als Kopplung über Ports; Löser bleiben extern (ADR „Fusion“, ändert den Satz „Nicht geplant“ in `architecture.md`) |
| D2 | Wo liegt der Prototyp? | Neues Repository (Python), Fieldtracing als Abhängigkeit; Regeln wie Fieldtracing |
| D3 | Ausführung der Löser | Unterprozess als Standard, in-process Python für den Prototyp, Dienst später |
| D4 | Wo läuft der Planer? | erst Python im Prototyp; im Kern als reine C++-Funktion in einer neuen Bibliothek der Schicht Workflows (`rtt-fusion`), die Ausführung in C++ mit Adapter-Aufrufen über die Unterprozess-Grenze |
| D5 | Modell und Format | Gebiete und Ports als neuer Abschnitt im Systemmodell mit Verweis auf Struktur-Dateien und Löser-Kennung (F, Schemasprung); Löser-Parameter als Daten, kein Code |
| D6 | Abstimmung mit M9 | Fusion-ADR vor dem M9-ADR; Löser-Gebiete als Körper mit Ports im nicht-sequenziellen Tracer |
| D7 | GPU | Kern weiter ohne GPU (ADR 0013), bis ein eigenes ADR folgt; Löser dürfen GPU nutzen. Erster GPU-Kandidat im Kern ist `rtt-field` (FFT/SFT), nach P1 und mit Messungen |
| D8 | Determinismus mit externen Lösern | Bitgleichheit nur für den Kern; je Löser die in `describe()` erklärte Toleranz, im Ergebnis vermerkt |
| D9 | Lizenzen | Adapter-Pakete getrennt vom MIT-Kern; Lizenz je Löser im Inventar |
| D10 | Reihenfolge der Löser | RCWA und BEM zuerst (K1–K3), dann BPM (K4), dann bio-optisch (K5), FEM bei Bedarf als Gegenprobe zu BEM und für Nahfelder |

## 10. Risiken

- **Konventionen:** Zeitvorzeichen, Polarisationsbasis und Normierung unterscheiden sich zwischen
  Codes. Gegenmittel: Konformitätstest vor jedem Adapter, Anpassung nur im Adapter.
- **Abtastung an Ports:** Periodische Ports (RCWA), Gitter (BPM, Felder) und Strahlnetze passen
  nicht von selbst zusammen; schlechte Wahl verschiebt den Fehler an die Übergänge. Gegenmittel:
  Umwandlungen als geprüfte Operatoren mit Fehlermodell, Abtastung aus dem Budget.
- **Kosten in Schleifen:** Optimierung und Toleranzierung rufen das System hunderte Male auf.
  Gegenmittel: lineare Löser einmal tabellieren (`operator`), Cache nach Inhalts-Hash,
  Ersatzmodelle für Parameterabhängigkeit erst in P5.
- **Leistungsnormierung:** Raytatouilles P ist leistungsnormiert, Felder sind Amplituden.
  Gegenmittel: Umrechnung nach Fieldtracing ADR 0004 an jedem Port, als Test im Konformitätssatz.
- **Umfang:** Fünf Löser, sechs Darstellungen und ein Planer sind ein großes Vorhaben neben M9
  und M10. Gegenmittel: Phasen mit eigener Abnahme, Pilotfälle statt Vollständigkeit, jede
  Darstellung nur mit einem konkreten Nutzer.
- **Wartung vieler Adapter:** Gegenmittel: schmale Schnittstelle (6.4), Konformitätstests in der
  CI der Adapter, Löser-Version im Cache-Schlüssel.

## 11. Nächste Schritte

1. Inventar der vorhandenen Löser nach Anhang A ausfüllen (Maintainer, mit Hilfe).
2. Entscheidungen D1, D2 und D10 treffen; danach den ADR-Entwurf „Fusion“ schreiben.
3. Prototyp-Repository anlegen und K1 als ersten Fall aufsetzen (RCWA-Tabelle im Tracer, über die
   Python-API von Raytatouille).
4. Nach v1.0: P1 nach `docs/rtt-port/`; P2 parallel mit K2 und K3.

## Anhang A: Vorlage für das Inventar der Löser

| Feld | Inhalt |
| --- | --- |
| Name, Repository, Version | |
| Verfahren, Gültigkeitsbereich | z. B. RCWA für 1D/2D-periodische Schichten, Zahl der Ordnungen |
| Sprache, Python-Anbindung | |
| Lizenz | |
| Eingaben | Struktur (Format), Einfall (ebene Welle, Feld, Moden) |
| Ausgaben | Ordnungen, Nahfeld, Fernfeld, S-/T-Matrix, Stokes/Mueller |
| Konventionen | Zeitvorzeichen, Fourier, Polarisationsbasis, Einheiten |
| Linear / reziprok / deterministisch | |
| Ressourcen | CPU/GPU, Speicher, typische Laufzeit für einen Pilotfall |
| Validierung | vorhandene Referenzfälle |
