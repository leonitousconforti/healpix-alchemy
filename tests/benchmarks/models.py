"""SQLAlchemy ORM models for unit tests."""

import sqlalchemy as sa
from sqlalchemy import orm

from healpix_alchemy.types import Point, Tile


@orm.as_declarative()
class Base:
    """Declarative base with table names derived from class names."""

    @orm.declared_attr.directive
    def __tablename__(cls) -> str:
        """Derive the table name from the class name."""
        # `orm.declared_attr` passes the class, but ty assumes an instance.
        return cls.__name__.lower()  # ty: ignore[unresolved-attribute]


class Galaxy(Base):
    """A point source on the sky."""

    id = sa.Column(sa.Integer, primary_key=True)
    hpx = sa.Column(Point, index=True, nullable=False)


class Field(Base):
    """A telescope field of view."""

    id = sa.Column(sa.Integer, primary_key=True)


class FieldTile(Base):
    """A HEALPix tile within the footprint of a field."""

    id = sa.Column(sa.ForeignKey(Field.id), primary_key=True, index=True)
    hpx = sa.Column(Tile, primary_key=True, index=True)


class Skymap(Base):
    """A multiresolution probability sky map."""

    id = sa.Column(sa.Integer, primary_key=True)


class SkymapTile(Base):
    """A HEALPix tile within a sky map, with a probability density."""

    id = sa.Column(sa.ForeignKey(Skymap.id), primary_key=True)
    hpx = sa.Column(Tile, primary_key=True, index=True)
    probdensity = sa.Column(sa.Float, nullable=False)
