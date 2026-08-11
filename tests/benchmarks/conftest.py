"""Fixtures that seed a disposable database for the benchmarks."""

from collections.abc import Iterator
from typing import cast

import numpy as np
import psycopg
import pytest
import sqlalchemy as sa
from astropy.coordinates import SkyCoord
from mocpy import MOC
from numpy.typing import NDArray
from sqlalchemy import orm

from . import data, models


@pytest.fixture
def session(engine: sa.engine.Engine) -> Iterator[orm.Session]:
    """Create an ORM session."""
    with orm.Session(engine) as session:
        yield session


@pytest.fixture
def cursor(session: orm.Session) -> psycopg.Cursor:
    """Expose the session's raw psycopg cursor, for COPY bulk loading."""
    # SQLAlchemy types the DBAPI connection generically; we know it's psycopg.
    return cast(psycopg.Cursor, session.connection().connection.cursor())


@pytest.fixture
def tables(engine: sa.engine.Engine) -> None:
    """Create the database schema."""
    # ty can't see the attributes that `orm.as_declarative` adds to `Base`.
    models.Base.metadata.create_all(engine)  # ty: ignore[unresolved-attribute]


@pytest.fixture
def random_galaxies(cursor: psycopg.Cursor, tables: None) -> SkyCoord:
    """Seed 40,000 random galaxies."""
    return data.get_random_galaxies(40_000, cursor)


@pytest.fixture(params=np.geomspace(1, 10_000, 10, dtype=int).tolist())
def random_fields(
    cursor: psycopg.Cursor, tables: None, request: pytest.FixtureRequest
) -> list[MOC]:
    """Seed N random telescope fields, at 10 logarithmically spaced sizes."""
    return data.get_random_fields(request.param, cursor)


@pytest.fixture
def random_sky_map(
    cursor: psycopg.Cursor, tables: None
) -> tuple[list[int], NDArray[np.float64]]:
    """Seed a random 20,000-tile probability sky map."""
    return data.get_random_sky_map(20_000, cursor)
