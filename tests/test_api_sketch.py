"""Issue #52: every public function and class of Fieldtracing appears in the API sketch."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

SKETCH = Path(__file__).resolve().parents[1] / "docs" / "rtt-port" / "api-skizze.md"
MODULES = [
    "units",
    "fourier",
    "propagation",
    "semiclassical",
    "mesh",
    "pft",
    "sft",
    "edge",
    "focus",
    "lpia",
    "selection",
    "field",
    "chain",
    "rtt_bridge",
    "results",
]


def _public(module_name: str) -> list[str]:
    module = importlib.import_module(f"fieldtracing.{module_name}")
    return [
        name
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and getattr(obj, "__module__", "") == module.__name__
    ]


@pytest.mark.parametrize("module_name", MODULES)
def test_every_public_name_is_mapped(module_name: str) -> None:
    text = SKETCH.read_text(encoding="utf-8")
    covered = f"`{module_name}.*`" in text  # whole module mapped or not ported
    missing = [
        name
        for name in _public(module_name)
        if not covered and f"{module_name}.{name}" not in text and f"`{name}" not in text
    ]
    if covered:
        # a module mapped as a whole still has to name its classes somewhere
        missing = [n for n in _public(module_name) if n[0].isupper() and n not in text]
    assert not missing, f"{module_name}: not in the API sketch: {missing}"
