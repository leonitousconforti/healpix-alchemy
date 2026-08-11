"""Pytest configuration for running the doctests in README.md."""

import pathlib
import subprocess
import sys
from unittest.mock import Mock

import psycopg
import pytest
import sqlalchemy as sa
from pytest_postgresql import factories
from pytest_postgresql.executor import PostgreSQLExecutor

if sys.platform == "win32":
    _init_directory = PostgreSQLExecutor.init_directory

    def _init_directory_readable_by_initdb(self: PostgreSQLExecutor) -> None:
        """Open up the data directory's parents, then initialize it.

        Python 3.12 and later implement mode=0o700 on Windows as a protected
        owner-only ACL, and pytest creates all of its temporary directories that
        way. initdb cannot even read the attributes of such a directory, so it
        refuses to create the data directory underneath one, reporting the
        parent as already existing. These are disposable directories under the
        temporary directory, so simply grant everyone access to them.
        """
        # The data directory is <basetemp>/pytest-postgresql-<fixture>/data-<port>.
        # Above the basetemp, only pytest's own directories need opening up:
        # with the default basetemp those are pytest-<n> and pytest-of-<user>.
        for depth, directory in enumerate(pathlib.Path(self.datadir).parents):
            if depth > 1 and not directory.name.startswith("pytest"):
                break
            subprocess.run(  # noqa: S603
                ["icacls", str(directory), "/grant", "*S-1-1-0:(OI)(CI)F", "/Q"],  # noqa: S607
                capture_output=True,
                check=True,
            )
        _init_directory(self)

    PostgreSQLExecutor.init_directory = _init_directory_readable_by_initdb

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
