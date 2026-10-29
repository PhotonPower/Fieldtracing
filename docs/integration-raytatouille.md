# Integration in Raytatouille

Analyse von Raytatouille (nur gelesen, Stand Commit `f77b9cd`, Version 0.8.0, M6 erreicht) und
Vorschlag, wo und wie Field Tracing dort eingebaut wird. Dieses Dokument ändert nichts an
Raytatouille; Entscheidungen trifft dort der Maintainer (AGENTS.md, Regel 8).

## Was Raytatouille schon hat

| Bereich | Stand | Bedeutung für Field Tracing |
| --- | --- | --- |
| Sprache, Build | C++20, CMake ≥ 3.25, vcpkg; Eigen, `double`; oneTBB deterministisch (ADR 0001–0004) | Port in C++20 mit denselben Bibliotheken |
| Strahldaten | `rtt::trace::RayBatch`, Structure-of-Arrays: Position, Richtung, Wellennormale, OPL, 3×3-PRT-Matrix (komplex), Gewicht, Status, Pupillenkoordinaten (ADR 0005, 0021) | jeder Strahl ist ein Abtastpunkt der Lagrange-Fläche: Phase k₀·OPL, k⊥ aus der Richtung, Polarisation aus P |
| Pupillenabtastung | `GridPupil`, `PointListPupil` mit Zellen und Flächenanteil (ADR 0032) | liefert die Nachbarschaft für den Jacobi-Faktor aus Maschenflächen |
| Polarisation | Fresnel in Convention A, Coatings (Transfermatrix), P leistungsnormiert, weight = Leistung für unpolarisierte Quelle (ADR 0021) | LPIA-Schritt existiert faktisch schon je Strahl; Feldamplitude = Leistung je Fläche plus P |
| Beugung | Pupillenfunktion, FFT-PSF skalar in Richtungssinus, MTF, EE, Jones-Pupille (ADR 0032–0034); pocketfft intern in `rtt-analysis` | Debye-Fokus ist die Verallgemeinerung; ADR 0033 lässt Vektor-PSF und √cos-Faktoren ausdrücklich aus (H7) |
| Roadmap | v2+: Beamlet-Wellenoptik; Erweiterungen: Huygens-PSF, Beamlet-Propagation, Faserkopplung, kohärente Bestrahlungsstärke | Field Tracing deckt diese Punkte ab oder ist ihre Grundlage |
| Konventionen | mm, µm, rad; exp(i(k·r − ωt)); +z; Convention A | identisch zu `docs/konventionen.md` |
| Prozess | ADR je Entscheidung, Test zuerst mit analytischem Sollwert, Formel mit Quelle und Gleichungsnummer, Changelog-Fragmente, Koordinator und Implementierer je Issue | Port-Issues werden in diesem Format vorbereitet |

## Raytatouille lesend nutzen (F4, #22)

Raytatouille ist eine **optionale** Abhängigkeit für die Tests mit dem Marker `rtt`. Der Commit
steht an genau einer Stelle: `rtt-commit.txt` (derzeit `f77b9cd`, Version 0.8.0). Ohne
installiertes Paket werden diese Tests übersprungen; die Standard-CI bleibt unverändert.

Installation (Linux, wie Raytatouilles README und CI):

```bash
git clone https://github.com/PhotonPower/Raytatouille.git ../raytatouille
git -C ../raytatouille checkout "$(cat rtt-commit.txt)"
# a) mit vcpkg (wie die CI von Raytatouille; Commit = builtin-baseline in dessen vcpkg.json)
export CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake -G Ninja"
# b) oder mit Systempaketen (Ubuntu 24.04):
#    sudo apt-get install cmake ninja-build g++ libeigen3-dev nlohmann-json3-dev libtbb-dev
#    export CMAKE_ARGS="-G Ninja"   (pocketfft lädt CMake am festen Commit nach)
pip install -r ../raytatouille/libs/rtt-py/tests/requirements.txt
pip install ../raytatouille
RAYTATOUILLE_SOURCE=../raytatouille pytest -m rtt
```

`RAYTATOUILLE_SOURCE` zeigt auf den Checkout, damit Tests die Referenzmodelle aus
`tests/reference` laden können. Der Workflow `.github/workflows/rtt.yml` (nächtlich, manuell und
bei PRs, die die Brücke berühren) baut das Rad mit vcpkg, cacht vcpkg-Binärpakete und das Rad je
Commit und setzt `FIELDTRACING_REQUIRE_RTT=1`: dann ist ein fehlendes Paket ein Fehler statt
eines übersprungenen Tests.

Wichtig für die Brücke: Raytatouilles Brechzahlen sind **absolut** (`AIR` = Ciddor-Luft,
n ≈ 1,00027 bei 0,5876 µm); OPL und Weglängen in Luft tragen diesen Faktor. Fieldtracing rechnet
mit der Brechzahl des Mediums im Feld (`refractive_index`), also ebenfalls absolut, wenn man sie
aus dem kompilierten System übernimmt.

## Integrationspunkte, nach Nutzen und Aufwand

1. **I1 Vektorielle Debye-PSF** aus der Jones-Pupille (ADR 0034): Kette PFT → Winkelspektrum → FFT
   mit exakt sphärischer Referenz, Faktor 1/cos²θ bzw. Apodisation, vektoriell nach Richards und
   Wolf. Schließt die Lücke H7 aus ADR 0033. Bausteine: F1-4, F3. Kleinster Eingriff, weil
   Pupillenfunktion und FFT vorhanden sind.
2. **I2 Felder in beliebigen Ebenen**, Through-Focus mit Blendenbeugung: SFT mit
   Minimax-Krümmung, Auswahlregeln. Bausteine: F1-1, F1-2, F2.
3. **I3 Hybride Feldverfolgung durch das System:** Feld aus Strahlen mit Jacobi-Faktor aus
   Maschen, LPIA je Strahl (vorhanden), Übergänge zu SFT/FFT an Blenden und Zwischenebenen;
   Quellen als Felder (z. B. Gauß-Strahlen). Bausteine: F2, F4.
4. **I4 Anwendungen aus den Roadmap-Erweiterungen:** Faserkopplung (Überlappintegral auf dem Feld),
   kohärente Bestrahlungsstärke, Beamlets.

## Vorschlag zur Struktur (Entwurf für ein ADR dort)

- Neue Bibliothek `libs/rtt-field` (Namespace `rtt::field`) in der Schicht **Auswertung**, neben
  `rtt-analysis`; Abhängigkeiten auf `rtt-trace` (Strahldaten) und tiefere Schichten. Falls I1
  zuerst kommt und die Jones-Pupille aus `rtt-analysis` gebraucht wird, liegt `rtt-field` über
  `rtt-analysis`; das braucht eine Festlegung der Reihenfolge wie bei ADR 0016.
- Header je Baustein, entsprechend den Python-Modulen: `fourier.hpp`, `propagation.hpp`,
  `geometric.hpp`, `pft.hpp`, `sft.hpp`, `edge.hpp`, `focus.hpp`, `selection.hpp`, `field.hpp`.
- Datenstrukturen als Structure-of-Arrays wie `RayBatch`; `double` und `std::complex<double>`.
- **FFT:** pocketfft ist in Raytatouille nur intern in `rtt-analysis` gekapselt (eine
  Übersetzungseinheit ohne Threads, SIMD und Plan-Cache, ADR 0033). `rtt-field` braucht entweder
  einen eigenen Wrapper nach demselben Muster oder eine gemeinsame interne FFT; das ist eine
  Entscheidung mit ADR.
- **Fehler:** Strahl- und Probenprobleme (evaneszent, Kaustik, J → 0) als Status-Flags, nicht als
  Exceptions (ADR 0009 dort).
- **Sollwerte:** Jeder Python-Test dieses Repositories liefert Sollwerte als JSON im Ergebnisformat
  `raytatouille-result` (ADR 0023 dort); die C++-Tests prüfen gegen diese Dateien und zusätzlich
  gegen die analytischen Fälle.
- **Python:** Bindungen über nanobind wie die übrigen Bibliotheken; die Python-API von
  Raytatouille bekommt ein Modul `raytatouille.field`.

Der ausgearbeitete Vorschlag steht als ADR-Entwurf in `docs/rtt-port/adr-entwurf-rtt-field.md`
(F5, #51); er weicht in einem Punkt von der Skizze oben ab: `rtt-field` braucht `rtt-analysis`
nicht, weil der erste Integrationspunkt aus Strahlen rechnet.

## Offene Fragen für den Port

1. ~~Exakte Umrechnung von weight und leistungsnormierter P in Feldamplituden auf einer Ebene~~:
   festgelegt und getestet in ADR 0004 (#23); Raytatouille braucht dafür keine Änderung.
2. ~~Nachbarschaft der Strahlen: reicht die Zellstruktur von `GridPupil`/`PointListPupil`, oder
   braucht es eine Triangulierung (Randzellen, Obskurationen)?~~ Triangulierung der Startpunkte
   (#18): beliebige Abtastungen, Obskurationen, O(h²) auch am Rand.
3. Bezugsebenen: Wie verhalten sich die PSF-Ebene von ADR 0032 (senkrecht zum Hauptstrahl) und die
   Fokusebene der Debye-Kette bei Feldpunkten außerhalb der Achse?
4. ~~Gehört die Wahl des Operators (Auswahlregeln) in die Bibliothek oder in eine Analyse-Option?~~
   Vorschlag im ADR-Entwurf, Punkt 8: Regeln und Plan in die Bibliothek, Erzwingen als Option.
