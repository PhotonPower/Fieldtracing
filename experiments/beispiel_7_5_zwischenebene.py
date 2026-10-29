"""Issue #41: intermediate plane z = f/2 of the example of section 7.5, scaling with the aperture.

Same NA 0.5, ε = 4 % (q = 10) for stop radii a; prints the plan, run time, peak memory and the
deviation of the intensity from the geometric propagation away from edge and axis zones. The full
system (a = 12.5 mm) needs about 5-6 GB; pass the radii on the command line:

    python experiments/beispiel_7_5_zwischenebene.py 1.25 2.5 5 12.5
"""

from __future__ import annotations

import math
import resource
import sys
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from fieldtracing import chain
from fieldtracing.field import GriddedField, GridlessField
from fieldtracing.selection import fresnel_width
from fieldtracing.units import wavenumber

K = wavenumber(0.5)


def polar_cloud(radius: float, rings: int) -> np.ndarray:
    pts = [np.zeros((1, 2))]
    for j in range(1, rings + 1):
        phi = 2 * np.pi * np.arange(6 * j) / (6 * j)
        pts.append(radius * j / rings * np.column_stack([np.cos(phi), np.sin(phi)]))
    return np.concatenate(pts)


def run(a: float) -> None:
    f = a / math.tan(math.asin(0.5))
    cloud = GridlessField.aplanatic(polar_cloud(1.05 * a, 60), f, 0.5)
    plan = chain.plan_propagation(cloud, f / 2, chain.CircularAperture(a), 0.04)
    start = time.perf_counter()
    field, _ = plan.run(cloud)
    seconds = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # Linux: kB
    assert isinstance(field, GriddedField)
    lat = field.lattice
    intensity = np.sum(np.abs(field.values) ** 2, axis=-1)
    deep = np.hypot(cloud.points[:, 0], cloud.points[:, 1]) <= a - 40 * fresnel_width(K / f)
    geo = cloud.propagate(f / 2)
    pts = geo.points[deep]
    ref = np.sum(np.abs(geo.amplitude[deep]) ** 2, axis=1)
    keep = np.hypot(pts[:, 0], pts[:, 1]) > plan.parameters["axis_zone_mm"]
    got = RegularGridInterpolator((field.x * lat[1, 1], field.x * lat[0, 0]), intensity)(
        pts[keep][:, ::-1]
    )
    err = float(np.max(np.abs(got - ref[keep])) / np.max(ref))
    p = plan.parameters
    print(
        f"a = {a} mm: n = {p['samples']:.0f}, step = {p['step_mm'] * 1e3:.2f} µm, "
        f"c = {p['curvature_min_per_mm2']:.1f} mm⁻², run {seconds:.1f} s, peak {peak:.2f} GB, "
        f"intensity vs. geometry {err:.2e}"
    )


if __name__ == "__main__":
    for value in sys.argv[1:] or ["1.25", "2.5"]:
        run(float(value))
