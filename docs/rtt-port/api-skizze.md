# API-Skizze `rtt-field` (C++ und Python)

> Entwurf aus Fieldtracing (#52) zum ADR-Entwurf `adr-entwurf-rtt-field.md`. Signaturen sind
> Vorschläge; Einheiten wie in Raytatouille (mm, µm, rad, k in 1/mm). Nicht in Raytatouille
> angelegt (Regel 8).
>
> **Freigegeben** vom Maintainer von Fieldtracing am 2026-10-30 (Entscheidung E6 in
> `docs/roadmap.md`); Übernahme nach Raytatouille dort nach dessen Prozess.

## Gemeinsame Typen (`rtt/field/types.hpp`)

```cpp
namespace rtt::field {
using Complex = std::complex<double>;
using Vec2 = Eigen::Vector2d;
using Mat2 = Eigen::Matrix2d;
using CVec3 = Eigen::Matrix<Complex, 3, 1>;

enum class Domain : std::uint8_t { Space, Frequency };

// Status je Probe (ADR 0009 von Raytatouille: keine Exceptions in Schleifen)
enum class SampleStatus : std::uint8_t {
  Ok = 0, Evanescent, Caustic, MeshUndetermined, TotalReflection, OutsideStop
};

// Hybrides Feld U·exp(iψ) auf einer Punktwolke, Structure-of-Arrays (ADR 0005)
struct GridlessField {
  std::vector<Vec2> points;        // mm (Space) bzw. 1/mm (Frequency)
  std::vector<double> phase;       // rad
  std::vector<Vec2> gradient;      // 1/mm bzw. mm
  std::vector<Mat2> hessian;       // 1/mm² bzw. mm²
  std::vector<CVec3> amplitude;    // volles Feld, transversal zum lokalen k
  std::vector<SampleStatus> status;
  double wavelength_um;
  double refractive_index = 1.0;   // absolut (Ciddor-Luft wie Raytatouille)
  Domain domain = Domain::Space;
};

// Gitter mit analytischer Trägerphase: Proben bei p = L·u, physikalisch values·exp(i/2 pᵀCp)
struct GriddedField {
  double step;                     // Schritt von u
  Eigen::Index n;                  // n × n
  std::vector<CVec3> values;       // Zeilen y, Spalten x
  Mat2 curvature = Mat2::Zero();   // C
  Mat2 lattice = Mat2::Identity(); // L
  double wavelength_um;
  double refractive_index = 1.0;
  Domain domain = Domain::Space;
};
}
```

## Header je Baustein

| Header | Funktionen (Vorschlag) | Python (Fieldtracing) | Sollwertdatei |
| --- | --- | --- | --- |
| `fourier.hpp` | `centered_grid(n, step)`, `frequency_step(n, step)`, `ft(span, step, axes)`, `ift(...)` | `fourier.*` | `fourier_gaussian` |
| `propagation.hpp` | `angular_spectrum(field, step, k, dz)`, `rayleigh_sommerfeld_kernel(dx, dy, d, k)`, `van_vleck_kernel(dx, dy, d, k)`, `longitudinal_wavenumber(k_perp_sq, k)` | `propagation.*` | `angular_spectrum_tilted_wave` |
| `geometric.hpp` | `GeometricResult propagate(const GridlessField&, double distance)`; 1D-Form nur für Tests | `semiclassical.*` (`GeometricResult`, `propagate_2d`, `propagate_1d`) | `geometric_propagation_astigmatic` |
| `mesh.hpp` | `Triangulation triangulate(points, MeshOptions)`, `triangle_area_ratio(...)`, `fit_derivatives(points, values, tri)`, `CloudJacobian mesh_jacobian(start, landing, MeshOptions)` | `mesh.*` (`CloudJacobian`) | `mesh_jacobian_jittered` |
| `pft.hpp` | `pft(const GridlessField&, PftOptions) -> GridlessField`, `inverse_pft(...)`, 1D-Zweige `pft_branches_1d(...)` | `pft.*` | `pft_sphere` |
| `sft.hpp` | `sft(const GriddedField&) -> GriddedField`, `inverse_sft(...)`, `check_sampling(n, step, C, q_min)`, `region_weights(x, y, Region, oversample)`; Diagonal- und 1D-Form intern | `sft.*` | `sft_gauss_chirp` |
| `edge.hpp` | `uniform_edge_factor(...)`, `edge_wave_1d(...)`, `boundary_integral(...)`, `polygon_asymptotics(...)`, `circle_asymptotics(...)`, `BoundaryNodes` | `edge.*` (`BoundaryNodes`) | (Tests T5, #5) |
| `focus.hpp` | `debye(const GridlessField&, DebyeOptions) -> GriddedField`, `debye_focus(residual, f, k, sin_max, ...) -> FocalField`, `richards_wolf(points, f, k, sin_max, defocus)`, `airy_integral(rho, k_max)`, Kugelhilfen (`spherical_phase`, `spherical_legendre`, `stationary_point`, `spherical_pft_amplitude`, `parabolic_residual`) | `focus.*` (`FocalField`), `chain.debye` | `debye_airy`, `richards_wolf_focus` |
| `lpia.hpp` | `fresnel(n1, n2, cos_i)` (Arrays), `refract(...)`, `fresnel_operator(...)` (Arrays, Totalreflexion als Status), `SurfaceTrace trace_surface(...)`, `interface_amplitude(...)`, `mesh_jacobian_grid(...)` | `lpia.*` (`FresnelCoefficients`, `FresnelArrays`, `SurfaceTrace`; Einzel- und `*_batch`-Formen werden eine Array-Form) | `fresnel_convention_a`, `lpia_sphere_about_focus` |
| `field.hpp` | `GridlessField::aplanatic/spherical/quadratic/from_phase`, `phase_at`, `residual_at(query, Interp, Outside)`, `to_grid(n, step, C)`, `transversality_error`, `GriddedField::physical/sample_points/carrier` | `field.*` (`GridlessField`, `GriddedField`) | (über Ketten) |
| `chain.hpp` | Operatoren `geometric_propagation`, `lpia_surface`, `aperture`, `pft`, `inverse_pft`, `angular_spectrum`, `sft`, `inverse_sft`, `fft`, `inverse_fft`, `resample`, `debye`, `rayleigh_sommerfeld`; `run_chain`; Blenden `CircularAperture`, `RectangularAperture`; Protokoll `LogEntry` | `chain.*` (`LogEntry`, `CircularAperture`, `RectangularAperture`, `Plan`) | (Tests #19, #20) |
| `selection.hpp` | Kennzahlen (`fresnel_width`, `local_fresnel_number`, `edge_wave_ratio`, `min_edge_distance`, `sft_gain`, `sft_step`, `sft_samples`, `nyquist_samples`), `fit_curvature(points, gradient)`, `fit_curvature_radial`, Fehlerterme (`debye_error`, `grid_edge_error`, `q_for_tolerance`, `airy_alias_error`), `Situation`, `select_method`, `Plan plan_interface(...)`, `Plan plan_propagation(...)` | `selection.*` (`Situation`), `chain.plan_interface`, `chain.plan_propagation`, `chain.airy_alias_error` | `selection_rules` |
| `rays.hpp` | `transverse_prt`, `absorption_factor`, `field_operator(prt, weight, k_in, k_out, n_in, n_out, J, μ, normals)`, `GridlessField field_from_rays(start, end, ...)` aus zwei `trace::RayBatch`, `trace_field(const compile::CompiledSystem&, const trace::PupilSampling&, double z, ...)`, `DebyePsf debye_psf(...)` | `rtt_bridge.*` | `field_operator_plate` |

**Nicht portiert:**
- `units.*` (`wavenumber`): Raytatouille hat eigene Hilfen.
- `results.*`: die Kodierung lebt in `raytatouille.results`; C++ liest die Sollwertdateien mit
  `rtt-json`.
- Die Warnklassen `AsymptoticWarning` (`pft`) und `SamplingWarning` (`sft`) werden
  Diagnosecodes (ADR-Entwurf, Punkt 5).
- Das Protokoll `chain.Aperture` wird ein Interface `Region` mit `inside` und `edge_distance`.

## Optionen statt Schlüsselwortargumente

```cpp
struct MeshOptions { std::optional<double> max_edge_mm; std::function<bool(double, double)> region;
                     double min_quality = 0.2; double caustic_threshold = 1e-2; };
struct DebyeOptions { double distance_mm; double sin_max; int n = 0;      // 0: aus dem Fenster
                      int pad = 4; double window_mm = 0.0; int oversample = 8; };
struct PlanOptions { double tolerance = 0.02; std::optional<double> length_u_mm;
                     std::optional<std::string> force_method; double focus_window_mm = 0.0; };
```

## Ergebnis des Planers

```cpp
struct Plan {
  std::string method;                          // lpia, geometric, sft, fft, mixed, debye, rayleigh_sommerfeld
  std::string reason;
  Situation situation;                         // Kennzahlen (Tabelle tab:selection)
  std::map<std::string, double> budget;        // Fehlerterme (Tabelle tab:errors)
  std::map<std::string, double> parameters;    // n, step_mm, q, axis_zone_mm, memory_gb, ...
  std::vector<diagnostics::Diagnostic> warnings;
  double predicted_error() const;              // Summe der Terme
};
```

## Python (`raytatouille.field`)

Dünne Spiegelung wie die übrigen Module (Raytatouille ADR 0002), Ergebnisse mit
`to_dict`/`to_json` (ADR 0023):

```python
import raytatouille as rt

system = rt.compile(rt.load("planoconvex.rtt.json"))
psf = rt.field.debye_psf(system, exit_z=10.0, rings=40, n=128)      # I1
psf.values, psf.x, psf.strehl, psf.warnings

field = rt.field.trace_field(system, rt.trace.HexapolarPupil(40), z=60.0)
plan = rt.field.plan_propagation(field, distance=7.5, stop=rt.field.CircularAperture(0.7))
out, log = plan.run(field)                                           # I2
```

## Zuordnung geprüft

`tests/test_api_sketch.py` prüft, dass jedes öffentliche Modul von Fieldtracing in dieser Skizze
zugeordnet ist und jede öffentliche Klasse namentlich vorkommt; Module ohne Pauschalzuordnung
müssen jede Funktion nennen.
