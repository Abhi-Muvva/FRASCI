"""Shared pytest fixtures for diff_mols tests."""
import pytest
from pathlib import Path


@pytest.fixture
def tmp_results_dir(tmp_path: Path) -> Path:
    """Throw-away results dir per test."""
    d = tmp_path / "results"
    d.mkdir()
    return d
