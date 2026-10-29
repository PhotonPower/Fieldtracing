"""Issue #50: reference files for the port (reference/field/*.result.json).

Every case of tools/reference_cases.py is recomputed and compared with its file (stale files
fail), the independent reference values are checked against the implementation within the
physical tolerance, and the encoding round-trips. Marker `rtt`: the encoding equals Raytatouille's
`raytatouille.results.encode_json`.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fieldtracing.results import FIELD_REFERENCE, decode, encode, load_json, to_json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from reference_cases import CASES  # noqa: E402


def _close(a: Any, b: Any, rel: float, abs_: float, where: str) -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), where
        for key in a:
            _close(a[key], b[key], rel, abs_, f"{where}/{key}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _close(x, y, rel, abs_, f"{where}/{i}")
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape, where
        if a.dtype.kind in "fc":
            assert np.allclose(a, b, rtol=max(rel, 1e-10), atol=abs_, equal_nan=True), where
        else:
            assert np.array_equal(a, b), where
    elif isinstance(a, float):
        assert b == pytest.approx(a, rel=max(rel, 1e-10), abs=abs_) or (
            math.isinf(a) and a == b
        ), where
    else:
        assert a == b, where


@pytest.mark.parametrize("name", list(CASES))
def test_file_is_up_to_date(name: str) -> None:
    path = ROOT / "reference" / "field" / f"{name}.result.json"
    type_name, stored = load_json(path.read_text(encoding="utf-8"))
    assert type_name == FIELD_REFERENCE
    fresh = decode(encode(CASES[name]()))
    tol = fresh["tolerance"]
    _close(fresh["expected"], stored["expected"], tol["rel"], tol["abs"], f"{name}/expected")
    for key in ("case", "description", "source", "parameters", "inputs", "tolerance"):
        _close(fresh[key], stored[key], 1e-12, 0.0, f"{name}/{key}")


def _reference_view(name: str, expected: dict[str, Any]) -> dict[str, Any]:
    """The implementation's values in the form of the case's independent reference."""
    if name == "pft_sphere":
        return {"abs_residual": np.abs(expected["residual"]), "phase": expected["phase"]}
    if name == "sft_gauss_chirp":
        return {"spectrum": expected["residual"] * np.exp(1j * expected["phase"])}
    if name == "lpia_sphere_about_focus":
        return {"abs_amplitude": np.linalg.norm(expected["amplitude"], axis=-1)}
    if name == "fresnel_convention_a":
        return {}  # the reference holds R + T, checked against 1 below
    if name == "richards_wolf_focus":
        return {"debye_at_lattice_points": expected["debye_at_lattice_points"]}
    return expected


@pytest.mark.parametrize("name", list(CASES))
def test_reference_values_hold(name: str) -> None:
    data = decode(encode(CASES[name]()))
    ref, tol = data["reference"], data["reference_tolerance"]
    if name == "fresnel_convention_a":
        for key in ("R_plus_T_s", "R_plus_T_p"):
            assert np.max(np.abs(ref[key] - 1.0)) < tol["abs"]
        return
    view = _reference_view(name, data["expected"])
    for key, value in ref.items():
        diff = np.abs(np.asarray(view[key]) - np.asarray(value))
        if "abs" in tol:
            assert np.max(diff) <= tol["abs"], (name, key, float(np.max(diff)))
        if "rel" in tol:
            assert np.max(diff / np.abs(value)) <= tol["rel"], (name, key)


def test_encoding_round_trip_and_special_values() -> None:
    data: dict[str, Any] = {
        "f": np.array([1.5, -0.0, np.nan, np.inf, -np.inf]),
        "c": np.array([[1 + 2j, np.nan + 1j]]),
        "i": np.array([1, 2, 3], dtype=np.uint8),
        "s": "text",
        "nan": float("nan"),
        "z": 1 - 2j,
        "nested": [np.int64(4), {"x": 0.1}],
    }
    type_name, back = load_json(to_json(FIELD_REFERENCE, data))
    assert type_name == FIELD_REFERENCE
    assert np.array_equal(back["f"], data["f"], equal_nan=True)
    assert np.signbit(back["f"][1]) and back["i"].dtype == np.uint8
    assert np.array_equal(back["c"], data["c"], equal_nan=True)
    assert math.isnan(back["nan"]) and back["z"] == data["z"] and back["nested"] == [4, {"x": 0.1}]
    text = to_json(FIELD_REFERENCE, data)
    assert "NaN" in text and json.loads(text)["format"] == "raytatouille-result"
    with pytest.raises(ValueError):
        load_json(text.replace("raytatouille-result", "other"))


@pytest.mark.rtt
def test_encoding_matches_raytatouille() -> None:
    import raytatouille.results as rt_results

    data = {
        "values": np.array([[0.1, np.nan], [np.inf, -0.0]]),
        "field": np.array([1 + 1j, -2.5j]),
        "status": np.array([0, 3], dtype=np.uint8),
        "scalar": float("-inf"),
        "z": 0.5 + 0.25j,
        "name": "x",
    }
    ours = json.loads(to_json("Psf", data))  # a type Raytatouille knows
    theirs = json.loads(rt_results.encode_json("Psf", data))
    assert ours == theirs
    assert ours["schema_version"] == rt_results.SCHEMA_VERSION
