"""
Shared pytest fixtures and sys.path bootstrap.

The repository does not ship a `pyproject.toml`/`setup.py`, so we insert the
repo root onto `sys.path` here so that `import src` works from any pytest run.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def data_path() -> str:
    """Path to the bundled CSV data directory (`data2/`)."""
    p = REPO_ROOT / "data2"
    if not p.exists():
        pytest.skip(f"data2 directory not found at {p}")
    return str(p)


@pytest.fixture(scope="session")
def small_date_range():
    """
    A short date range for fast tests. Picks a window covered by the bundled
    yield curve CSV, futures CSV and bond metadata.
    """
    return datetime(2021, 1, 1), datetime(2021, 12, 31)
