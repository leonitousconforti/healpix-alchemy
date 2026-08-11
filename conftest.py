"""Pytest configuration for running the doctests in README.md."""

from unittest.mock import Mock

import pytest
import sqlalchemy as sa
from pytest_postgresql import factories

# The test databases are disposable, so trade durability for speed.
postgresql_proc = factories.postgresql_proc(
    postgres_options="-c fsync=off -c synchronous_commit=off -c full_page_writes=off"
)


@pytest.fixture
def engine(postgresql):
    """Create an SQLAlchemy engine with a disposable PostgreSQL database."""
    return sa.create_engine(
        "postgresql+psycopg://",
        poolclass=sa.pool.StaticPool,
        pool_reset_on_return=None,
        creator=lambda: postgresql,
    )


@pytest.fixture(autouse=True)
def add_mock_create_engine(monkeypatch, request):
    """Monkey patch sqlalchemy.create_engine for doctests in README.md."""
    if request.node.name == "README.md":
        engine = request.getfixturevalue("engine")
        monkeypatch.setattr(sa, "create_engine", Mock(return_value=engine))
