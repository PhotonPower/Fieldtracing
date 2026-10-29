"""Shared test parameters (units: mm, µm) and the optional Raytatouille marker `rtt`.

Tests marked `rtt` need the package raytatouille (read only, AGENTS.md rule 8). Without it they are
skipped, unless FIELDTRACING_REQUIRE_RTT=1 (workflow rtt.yml), where a missing package is an
error. RAYTATOUILLE_SOURCE points to a checkout at the commit in rtt-commit.txt (reference models).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

WAVELENGTH_UM = 0.5

RTT_COMMIT = (Path(__file__).resolve().parents[1] / "rtt-commit.txt").read_text().strip()


def rtt_available() -> bool:
    """True if the package raytatouille can be imported."""
    return importlib.util.find_spec("raytatouille") is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if rtt_available():
        return
    if os.environ.get("FIELDTRACING_REQUIRE_RTT") == "1":
        raise pytest.UsageError("FIELDTRACING_REQUIRE_RTT=1 but raytatouille is not installed")
    skip = pytest.mark.skip(reason=f"raytatouille not installed (commit {RTT_COMMIT[:7]})")
    for item in items:
        if "rtt" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def rtt_reference_dir() -> Path:
    """tests/reference of the Raytatouille checkout in RAYTATOUILLE_SOURCE."""
    source = os.environ.get("RAYTATOUILLE_SOURCE")
    if not source:
        if os.environ.get("FIELDTRACING_REQUIRE_RTT") == "1":
            raise RuntimeError("RAYTATOUILLE_SOURCE is not set")
        pytest.skip("RAYTATOUILLE_SOURCE not set (checkout of Raytatouille)")
    path = Path(source) / "tests" / "reference"
    if not path.is_dir():
        raise RuntimeError(f"no reference models in {path}")
    return path
