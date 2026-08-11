"""Pytest configuration for running the doctests in README.md."""

import os
from unittest.mock import Mock

import psycopg
import pytest
import sqlalchemy as sa
from pytest_postgresql import factories

# Either way the fixture is called postgresql_proc, because that is the name
# that pytest-postgresql's own postgresql fixture depends on.
if os.environ.get("POSTGRESQL_NOPROC"):
    # Connect to a server that is already running on 127.0.0.1:5432. CI starts
    # one itself, because spawning a server per session needs initdb, which
    # cannot create a data directory under pytest's temporary directories on
    # Windows: Python 3.12 and later create those with a protected owner-only
    # ACL whose attributes initdb cannot read.
    postgresql_proc = factories.postgresql_noproc()
else:
    # The test databases are disposable, so trade durability for speed.
    postgresql_proc = factories.postgresql_proc(
        postgres_options="-c fsync=off -c synchronous_commit=off -c full_page_writes=off"
    )


@pytest.fixture
def engine(postgresql: psycopg.Connection) -> sa.Engine:
    """Create an SQLAlchemy engine with a disposable PostgreSQL database."""
    return sa.create_engine(
        "postgresql+psycopg://",
        poolclass=sa.pool.StaticPool,
        pool_reset_on_return=None,
        creator=lambda: postgresql,
    )


@pytest.fixture(autouse=True)
def add_mock_create_engine(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Monkey patch sqlalchemy.create_engine for doctests in README.md."""
    if request.node.name == "README.md":
        engine = request.getfixturevalue("engine")
        monkeypatch.setattr(sa, "create_engine", Mock(return_value=engine))
