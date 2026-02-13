"""Shared test fixtures."""

import pytest

from docrag.config import Settings


@pytest.fixture
def settings(tmp_path):
    """Create a Settings instance with temporary directories."""
    return Settings(
        chroma_dir=tmp_path / "chroma",
        cache_dir=tmp_path / "cache",
        cache_ttl_hours=1,
    )