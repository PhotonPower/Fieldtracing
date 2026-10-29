# ADR 00xx (Entwurf): Bibliothek `rtt-field` – Felder mit Field Tracing

> **Entwurf aus Fieldtracing (#51) für Raytatouille.** Nicht dort angelegt (Fieldtracing,
> AGENTS.md Regel 8); Nummer und Annahme vergibt der Maintainer von Raytatouille. Stand
> Raytatouille: Commit `f77b9cd` (Version 0.8.0). Belege in Fieldtracing: `docs/testfaelle.md`,
> `docs/vergleich-psf.md`, `docs/beispiel-7-5.md`, ADR 0004, Sollwertdateien `reference/field/`.

**Status:** vorgeschlagen

## Kontext

Raytatouille rechnet Beugung bisher über die Pupillenfunktion (ADR 0032) und die skalare FFT-PSF
(ADR 0033). Dort sind ausdrücklich ausgelassen: Vektor-PSF (H7) sowie Schiefe- und
Radiometriefaktoren (√cos). Felder in beliebigen Ebenen, Through-Focus mit Blendenbeugung,
Kopplung an Fasern und kohärente Bestrahlungsstärke stehen in `docs/roadmap-erweiterungen.md`
(Huygens-PSF, Beamlet-Propagation, Faserkopplung).

Field Tracing (Paper `docs/paper/fieldtracing.tex` in Fieldtracing) führt das Feld als
U·e^(iψ) mit glattem Restfeld U ∈ ℂ³ und analytischer Phase ψ durch das System und wählt je
Schritt den passenden Operator (geometrische Propagation mit Jacobi-Faktor, LPIA, PFT, SFT, FFT,
Winkelspektrum, Debye). Die Bausteine sind in Fieldtracing in Python gegen unabhängige
Referenzen geprüft (Meilensteine F0–F4) und am Beispiel aus Abschnitt 7.5 des Papers als Kette
erprobt (F3).

**Ist-Stand in Raytatouille, der genutzt wird:**
- `rtt-trace`: `RayBatch` als Structure-of-Arrays (ADR 0005) mit Position, Richtung, OPL,
  leistungsnormierter PRT-Matrix P und Gewicht (ADR 0021); Pupillenabtastungen inkl.
  `HexapolarPupil` und `PointListPupil` (ADR 0032).
- `rtt-compile`, `rtt-paraxial`: absolute Brechzahlen (Ciddor-Luft), Hauptstrahl, `first_order`.
- `rtt-analysis`: pocketfft als interne Abhängigkeit (`rtt::pocketfft`, ADR 0033 Punkt 1).
- Ergebnisformat `raytatouille-result` (ADR 0023), Diagnosecodes (ADR 0022), Fehler als
  Status-Flags in Schleifen (ADR 0009).

**Ergebnisse aus Fieldtracing, die die Entscheidungen begründen:**
- Feldamplitude aus Raytatouilles P und Gewicht ohne Änderung an Raytatouille
  (Fieldtracing ADR 0004): U_aus = √(s·n_ein cos γ_ein dA_ein/(n_aus cos γ_aus dA_aus))·
  e^(−iπμ/2)·P_T·U_ein, s = weight/(½‖P_T‖²); gegen Raytatouilles P und `polar.transmission`
  auf 10⁻¹² bis 10⁻¹³ geprüft.
- Feld aus Strahlen (Phase k₀·OPL, Jacobi-Faktor aus der Triangulierung der Startpunkte,
  Hesse-Matrix aus den Nachbarstrahlen): Phase gegen `opd_points` auf 10⁻⁸ Wellen, Leistung auf
  1,4·10⁻⁴, Jacobi-Faktor O(h²) auch auf unstrukturierten Wolken und am Rand.
- Vektorielle Debye-PSF aus Strahlen: bei NA 0,05 Strehl wie `psf` (0,73820 gegen 0,73815),
  bei NA 0,9 (ideale Linse) erklärbare Abweichungen; Raytatouilles FFT-PSF ist dort wegen der
  affinen Pupillenabbildung um tan θ/sin θ zu schmal (Issue-Entwurf
  `docs/rtt-entwurf-psf-hohe-na.md` in Fieldtracing).
- Beispiel 7.5 (D = 25 mm, NA 0,5): Fokus gegen Richards-Wolf 6,5·10⁻⁴; verkleinerte Systeme
  gegen direkte Rayleigh-Sommerfeld-Rechnung im vorhergesagten Fehlerbudget; Zwischenebene mit
  SFT 5234² Punkte, etwa 1,3 GB je vektoriellem Feld.
- Gefundene Grenzen: Kaustik der Kantenwelle auf der Achse von Kreisblenden (Gitterfehler der
  harten Kante dort verstärkt), Kantenkonstante 0,4/q in 2D, SFT-Untergrenze n ≳ πq²/m.

## Entscheidung

1. **Neue Bibliothek `libs/rtt-field`**, Namespace `rtt::field`, Target `rtt_field` mit Alias
   `rtt::field`, in der **Schicht Auswertung neben `rtt-analysis`**. Sie verwendet `rtt-trace`,
   `rtt-paraxial`, `rtt-compile`, `rtt-polar`, `rtt-math` und `rtt-diagnostics`, aber **nicht
   `rtt-analysis`**: Der erste Integrationspunkt rechnet aus Strahlen, nicht aus der
   Pupillenfunktion (Punkt 7). Damit braucht es keine Reihenfolge innerhalb der Schicht wie in
   ADR 0016. Soll später die Jones-Pupille als Eingang dienen, legt ein Nachtrag die Reihenfolge
   `rtt-analysis` < `rtt-field` fest.
2. **Header je Baustein** unter `libs/rtt-field/include/rtt/field/` (Zuordnung zu den
   Python-Modulen von Fieldtracing in der API-Skizze): `fourier.hpp`, `propagation.hpp`,
   `geometric.hpp`, `mesh.hpp`, `pft.hpp`, `sft.hpp`, `edge.hpp`, `focus.hpp`, `lpia.hpp`,
   `field.hpp`, `chain.hpp`, `selection.hpp`, `rays.hpp` (Brücke aus `RayBatch`).
3. **Datenstrukturen.** Structure-of-Arrays wie `RayBatch` (ADR 0005): `GridlessField` mit
   Punkten (N × 2), Phase (N), Gradient (N × 2), Hesse-Matrix (N × 2 × 2), Restfeld (N × 3,
   `std::complex<double>`), Wellenlänge, Brechzahl, Bereich (Ort/Frequenz) und einer
   **Status-Spalte** je Probe; `GriddedField` mit Schritt, Gitterabbildung (`lattice`, 2 × 2),
   Trägermatrix (2 × 2), Werten (N × N × 3). `double` und `std::complex<double>` (ADR 0003);
   Templates über `Real` nur, wo der Ableitungspfad es braucht (ADR 0006/0015), zunächst nicht.
4. **Hybride Rückgabe und Interpolationsregel** wie in Fieldtracing (dort ADR 0003):
   Transformationen geben (k, Restfeld, Phase) zurück; interpoliert wird nur das Restfeld, die
   Phase über ihre Taylor-Entwicklung oder analytisch.
5. **Fehler und Warnungen.** Eingabeprüfung an der API mit `std::invalid_argument` (ADR 0009);
   je Probe Status-Flags statt Exceptions: `Evanescent`, `Caustic` (|J| klein oder Vorzeichen-
   wechsel), `MeshUndetermined`, `TotalReflection`. Gültigkeitsbedingungen als Diagnosen mit
   stabilen Codes (ADR 0022), Erzeuger `analysis`: `field.asymptotic_small_fresnel` (lokale
   Fresnel-Zahl klein), `field.edge_wave` (Kantenverhältnis über ε), `field.sft_sampling`,
   `field.axis_zone` (Kaustik der Kantenwelle), `field.caustic`, `field.debye_small_fresnel`.
6. **FFT.** `rtt-field` kapselt pocketfft in einer eigenen Übersetzungseinheit nach dem Muster
   von ADR 0033 Punkt 1 (Target `rtt::pocketfft`, gleicher Pin), ohne Plan-Cache über Aufrufe
   hinweg. Eine gemeinsame interne FFT von `rtt-analysis` und `rtt-field` wäre eine eigene
   Basisbibliothek; das bleibt ein späterer Schritt (Verworfen, Punkt 3).
7. **Erster Integrationspunkt (Fieldtracing E4): vektorielle Debye-PSF aus Strahlen.** Feld auf
   einer Ebene im Bildraum aus `RayBatch` (Hexapolar mit Rand), Feldamplitude nach Fieldtracing
   ADR 0004, Debye-Integral per FFT mit subpixelgewichteter k-Scheibe, Strehl gegen dasselbe
   Integral ohne Aberration. Diese PSF braucht keine affine Pupillenabbildung und enthält
   Schiefe-, Radiometrie- und Vektoranteile; sie schließt H7 aus ADR 0033. Danach Felder in
   beliebigen Ebenen mit SFT/FFT nach den Auswahlregeln (I2) und die hybride Feldverfolgung durch
   das System (I3).
8. **Auswahlregeln** (Fieldtracing Tabelle tab:selection, `selection.select_method`,
   `chain.plan_propagation`) gehören in die Bibliothek als reine Funktion über Kennzahlen; die
   Analyse-Optionen erlauben, einen Baustein zu erzwingen. Der Plan (Baustein, Begründung,
   Fehlerbudget, Gitter) ist Teil des Ergebnisses.
9. **Sollwerte.** Die Sollwertdateien von Fieldtracing (`reference/field/*.result.json`, Kodierung
   nach ADR 0023, Typ `FieldReference`) werden mit dem Fieldtracing-Commit nach
   `tests/reference/field/` übernommen. Die C++-Tests lesen sie mit `rtt-json`, prüfen gegen
   `expected` (gleicher Algorithmus, enge Toleranz) und gegen `reference` (unabhängig,
   physikalische Toleranz); dazu die analytischen Tests nach den Regeln von Raytatouille.
   `raytatouille.results` bekommt den Typ `FieldReference` (Schema 0.1.14), damit Python dieselben
   Dateien laden kann.
10. **Python.** Bindungen über nanobind (ADR 0002) im Modul `raytatouille.field`; Ergebnisse als
    NumPy-Arrays mit `to_dict`/`to_json` (ADR 0023). Die bitweisen Python-C++-Vergleiche folgen
    dem Muster von `rtt-py` (`*_cases.cpp`).
11. **Parallelität und Speicher.** oneTBB deterministisch (ADR 0004): parallel über Proben,
    Gitterzeilen und Feldkomponenten. Die drei Feldkomponenten werden nacheinander transformiert,
    um den Spitzenspeicher großer SFT-Gitter etwa zu dritteln (Beispiel 7.5: ≈ 1,3 GB je
    vektoriellem Feld bei 5234²).

## Verworfen

- **Field Tracing in `rtt-analysis`:** zu groß, eigene Datenstrukturen und Prüfstrategie; die
  Bibliothek wäre nicht mehr getrennt testbar.
- **Debye-PSF aus der Jones-Pupille als erster Schritt:** übernimmt die affine Pupillenabbildung
  von ADR 0032; bei hoher NA ist sie nach den Messungen in Fieldtracing nicht tragfähig.
- **Gemeinsame FFT-Basisbibliothek jetzt:** verschiebt Code aus `rtt-analysis` und berührt
  ADR 0033; erst sinnvoll, wenn beide Bibliotheken dieselben Pläne nutzen.
- **Feldamplitude-P in `RayBatch`:** Raytatouille ADR 0021 hat sie verworfen; Fieldtracing ADR 0004
  zeigt, dass sie nicht nötig ist.
- **Port ohne Sollwertdateien, nur analytische Tests:** Für Maschen-Jacobi, Planer und die
  Debye-Kette gibt es keine geschlossenen Formen in allen Fällen; die Python-Referenz liefert sie.

## Folgen

- `docs/architecture.md`: neue Zeile in der Schicht Auswertung; Abschnitt „Felder (`rtt-field`)“
  mit Konventionen (identisch zu Raytatouille, dazu Fourier-Konvention (2π)^−d, Signatur,
  Maslov-Index; Fieldtracing `docs/konventionen.md`).
- Neue Diagnosecodes in `rtt-diagnostics` und `docs/diagnostics.md` (Punkt 5).
- `vcpkg.json` bleibt für FFT, Parallelität und Lineare Algebra unverändert (pocketfft, oneTBB,
  Eigen). Fieldtracing nutzt SciPy für Bausteine, die in C++ einen Ersatz brauchen (je eine eigene
  Entscheidung): Delaunay-Triangulierung und lineare bzw. Clough-Tocher-Interpolation auf der
  Triangulierung (`mesh`, `field`), Nächster-Nachbar-Suche (KD-Baum für die Taylor-Phase), das
  Minimax-Problem der SFT-Krümmung (lineares Programm mit vier Unbekannten) und
  Bessel-Funktionen J₀, J₁, J₂ (Richards-Wolf, Airy; `std::cyl_bessel_j` aus C++17).
- `raytatouille.results`: Typ `FieldReference`, Schema 0.1.14.
- Port in Aufgaben nach `docs/rtt-port/issues.md` (Fieldtracing), Reihenfolge I1 → I2 → I3.
