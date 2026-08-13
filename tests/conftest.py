"""Shared pytest fixtures. Adds src/ to the import path for the whole suite."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soc_triage.config import load_env_config  # noqa: E402  (path insert must run first)


@pytest.fixture(scope="session")
def cfg():
    """The real, calibrated environment config — tests run against what ships."""
    return load_env_config(Path(__file__).resolve().parents[1] / "config" / "env_default.yaml")
