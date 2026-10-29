# Konventionen

Verbindlich für allen Code. Sie stimmen mit Raytatouille überein (`docs/architecture.md`,
Abschnitt „Konventionen“, dort Stand Commit `f77b9cd`), damit der Port keine Vorzeichen ändert
(ADR 0002).

## Einheiten und Koordinaten

| Größe | Einheit | Bemerkung |
| --- | --- | --- |
| Längen | mm | wie Raytatouille |
| Vakuumwellenlänge | µm | `wavenumber(λ_µm, n)` liefert k = 2πn/λ in 1/mm |
| Wellenzahl, Ortsfrequenz k⊥ | 1/mm | |
| Phase ψ | rad | ψ = k₀·OPL (OPL in mm) |
| Hesse-Matrix der Phase | 1/mm² | |
| Winkel | rad | Grad nur mit Suffix `_deg` |

Koordinatensystem rechtshändig, optische Achse +z, y meridional. Transversale Koordinaten
ρ = (x, y) auf Ebenen z = const.

## Felder, Phase, Fourier-Transformation

- **Zeitkonvention:** e^(−iωt), ebene Wellen ~ exp(i(k·r − ωt)). Ein größeres Argument bedeutet
  Verzögerung.
- **Fourier-Transformation** (Paper, Gl. ft_convention):
  F{f}(k) = (2π)^−d ∫ f(x) e^(−ik·x) dᵈx, Rücktransformation f(x) = ∫ F(k) e^(ik·x) dᵈk.
  Faltungssatz: F{f∗g} = (2π)^d F{f}·F{g}. Zentrierte Gitter x_n = (n − N//2)·Δx,
  Δk = 2π/(N Δx) (`fieldtracing.fourier`).
- **Ausbreitung:** k_z = √(n²k₀² − |k⊥|²) mit Im k_z ≥ 0; evaneszente Anteile klingen ab.
- **Hybride Darstellung:** E = U·e^(iψ) mit U ∈ ℂ³ (volles Feld, U_z aus der Transversalität).
  Transformationen geben `(k, residual, phase)` zurück; interpoliert wird nur das Residuum
  (ADR 0003).
- **Signatur:** sig H = Anzahl positiver minus Anzahl negativer Eigenwerte; Phasenfaktor
  e^(iπ/4·sig H) der stationären Phase.
- **Maslov-Index:** μ = Zahl der negativen Eigenwerte von ∂ρ₂/∂ρ₁ (homogenes Medium), Faktor
  e^(−iπμ/2). Brennlinie: μ = 1, Punktfokus: μ = 2 (Gouy-Phase −π).
- **Verhältnis zur OPD von Raytatouille:** Raytatouille definiert W = OPL_ref − OPL (W > 0
  voreilend) und die Pupillenphase −2πW·λ_ref/λ, also k(OPL − OPL_ref). Das ist ψ bis auf eine
  Konstante.

## Fresnel und Polarisation

Convention A nach Byrnes (arXiv:1603.02720v5, Gl. (6)), wie Raytatouille:

- r_s = (n₁ cos i − n₂ cos t)/(n₁ cos i + n₂ cos t), r_p = (n₂ cos i − n₁ cos t)/(n₂ cos i + n₁ cos t),
- t_s = 2n₁ cos i/(n₁ cos i + n₂ cos t), t_p = 2n₁ cos i/(n₂ cos i + n₁ cos t),
- bei senkrechtem Einfall r_p = −r_s;
- Basis: N zeigt ins Medium 2, s = k̂_ein × N/|k̂_ein × N|, p = k̂ × s für jede Welle;
- Transmissionsoperator auf dem vollen Feldvektor: T = t_s s sᵀ + t_p p_aus p_einᵀ (Paper,
  Gl. fresnel_operator).

Raytatouille trägt je Strahl eine **leistungsnormierte** 3×3-Matrix P und ein Gewicht
(weight = Leistung für eine unpolarisierte Quelle, ADR 0021 dort). Dieses Paket rechnet mit
Feldamplituden; die Umrechnung legt ADR 0004 fest (`rtt_bridge.field_operator`):

U_aus = √(s·n_ein cos γ_ein dA_ein/(n_aus cos γ_aus dA_aus))·e^(−iπμ/2)·P_T·U_ein,
P_T = (I − k kᵀ)P, s = weight/(½‖P_T‖²_F).

Brechzahlen sind dabei absolut wie in Raytatouille (`AIR` = Ciddor-Luft, n ≈ 1,00027).

## Code

- Funktionen sind rein und deterministisch; Zufall nur mit explizitem Seed.
- Fehler in der Eingabe: `ValueError` mit klarer Meldung. Physikalisch unmögliche Proben (etwa
  evaneszente Strahlen in der geometrischen Propagation) ebenfalls `ValueError`; der Port nach
  Raytatouille macht daraus Status-Flags (ADR 0009 dort).
- Arrays als NumPy-Arrays je Größe, letzte Achse für Vektorkomponenten.
