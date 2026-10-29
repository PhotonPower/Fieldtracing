"""Bridge from Raytatouille ray data to field amplitudes (F4; ADR 0004).

Raytatouille traces with a POWER-NORMALISED polarization ray-tracing matrix P and a weight
(Raytatouille ADR 0021): every interface contributes P_q = √c·T_q on the transverse plane, with
the field transmission operator T_q = t_s s sᵀ + t_p p_out p_inᵀ (paper, eq. fresnel_operator,
Convention A) and the power factor c = n₂ cos θ_t/(n₁ cos θ_i) (Byrnes, arXiv:1603.02720v5,
eq. (21), (22), real indices), and P k₀ = k. The weight of an unpolarized unit source is
weight = s·½‖P_T‖²_F with P_T = (I - k kᵀ)P and s the polarization-independent factors (volume
absorption, absorbers, diffraction efficiency).

Field amplitudes on planes z = const follow from flux conservation n|U|²cos γ dA along a ray tube
(paper, eq. interface_amplitude): over any sequence of interfaces

    U_out = sqrt(s·n_in cos γ_in dA_in / (n_out cos γ_out dA_out))·exp(-iπμ/2)·P_T·U_in,

because the factors sqrt(cos θ_t/cos θ_i) of eq. interface_amplitude and the √c in P differ only
by sqrt(n₁/n₂) per interface, which telescopes to sqrt(n_in/n_out). γ is the angle between the ray
and the plane normal (+z), dA_out/dA_in = |J| the area ratio of the ray tube (`mesh`), μ the
Maslov index. Indices are absolute (Raytatouille: AIR = Ciddor air).

This module only converts arrays; it does not import raytatouille.
"""

from __future__ import annotations

import numpy as np

from fieldtracing._types import ComplexArray, FloatArray, IntArray


def transverse_prt(prt: ComplexArray, direction_out: FloatArray) -> ComplexArray:
    """P_T = (I - k kᵀ)P (N, 3, 3) for final unit directions k (N, 3) (Raytatouille ADR 0021)."""
    k = np.asarray(direction_out, dtype=np.float64)
    proj = np.eye(3)[None] - np.einsum("ni,nj->nij", k, k)
    return np.asarray(np.einsum("nij,njk->nik", proj, prt))


def absorption_factor(
    prt: ComplexArray, weight: FloatArray, direction_out: FloatArray
) -> FloatArray:
    """s = weight/(½‖P_T‖²_F): the polarization-independent power factor of each ray (≤ 1).

    Rays with P_T = 0 (fully blocked) get s = 0.
    """
    norm = 0.5 * np.sum(np.abs(transverse_prt(prt, direction_out)) ** 2, axis=(1, 2))
    w = np.asarray(weight, dtype=np.float64)
    return np.asarray(np.where(norm > 0.0, w / np.where(norm > 0.0, norm, 1.0), 0.0))


def field_operator(
    prt: ComplexArray,
    weight: FloatArray,
    direction_in: FloatArray,
    direction_out: FloatArray,
    n_in: float,
    n_out: float,
    area_ratio: FloatArray | float = 1.0,
    maslov: IntArray | int = 0,
) -> ComplexArray:
    """Field amplitude operator M (N, 3, 3) with U_out = M·U_in on planes z = const (ADR 0004).

    M = sqrt(s·n_in cos γ_in/(n_out cos γ_out |J|))·exp(-iπμ/2)·P_T.

    Args:
        prt: (N, 3, 3) power-normalised P of the rays (Raytatouille `RayBatch.prt_matrices()`).
        weight: (N,) weight of the rays (unpolarized unit source).
        direction_in, direction_out: (N, 3) unit ray directions at the input and output plane
            (cos γ is their z-component).
        n_in, n_out: absolute refractive indices of the input and output media.
        area_ratio: J = dA_out/dA_in of the ray tube on the two planes (signed or absolute).
        maslov: Maslov index μ of the caustics between the planes.

    Returns:
        M; for a transverse input U_in ⟂ direction_in the output is transverse to direction_out.
    """
    p_t = transverse_prt(prt, direction_out)
    s = absorption_factor(prt, weight, direction_out)
    cos_in = np.abs(np.asarray(direction_in, dtype=np.float64)[:, 2])
    cos_out = np.abs(np.asarray(direction_out, dtype=np.float64)[:, 2])
    scale = np.sqrt(s * n_in * cos_in / (n_out * cos_out * np.abs(area_ratio)))
    phase = np.exp(-0.5j * np.pi * np.asarray(maslov))
    return np.asarray((scale * phase)[..., None, None] * p_t)
