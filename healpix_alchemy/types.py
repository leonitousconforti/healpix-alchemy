"""SQLAlchemy types for multiresolution HEALPix data."""

from collections.abc import Iterator, Sequence

import numpy as np
import sqlalchemy as sa
from astropy.coordinates import SkyCoord
from astropy_healpix import uniq_to_level_ipix
from mocpy import MOC
from numpy.typing import ArrayLike
from sqlalchemy import event
from sqlalchemy.dialects.postgresql import INT8RANGE

from .constants import HPX, LEVEL, PIXEL_AREA_LITERAL

__all__ = ("Point", "Tile")


class Point(sa.TypeDecorator[int]):
    """A point on the sky, stored as the HEALPix index of a level-29 pixel."""

    cache_ok = True
    impl = sa.BigInteger

    def process_bind_param(
        self,
        value: SkyCoord | tuple[float, float] | np.integer | int | None,
        dialect: sa.Dialect,  # noqa: ARG002  (required by the TypeDecorator API)
    ) -> int | None:
        """Convert a sky coordinate or ``(lon, lat)`` pair to a HEALPix index."""
        if isinstance(value, SkyCoord):
            value = HPX.skycoord_to_healpix(value)
        elif isinstance(value, Sequence) and len(value) == 2:  # noqa: PLR2004  ((lon, lat) pair)
            value = HPX.lonlat_to_healpix(*value)
        if isinstance(value, np.integer):
            value = int(value)
        return value


class Tile(sa.TypeDecorator[str]):
    """A multiresolution HEALPix tile, stored as a range of level-29 pixels."""

    cache_ok = True
    impl = INT8RANGE

    def process_bind_param(
        self,
        value: int | np.integer | tuple[int, int] | str | None,
        dialect: sa.Dialect,  # noqa: ARG002  (required by the TypeDecorator API)
    ) -> str | None:
        """Convert a UNIQ index or ``(lo, hi)`` pair to a range string."""
        if isinstance(value, (int, np.integer)):
            level, ipix = uniq_to_level_ipix(value)
            shift = 2 * (LEVEL - level)
            value = (ipix << shift, (ipix + 1) << shift)
        if isinstance(value, Sequence) and len(value) == 2:  # noqa: PLR2004  ((lo, hi) pair)
            value = f"[{value[0]},{value[1]})"
        return value

    class comparator_factory(INT8RANGE.comparator_factory):  # noqa: N801  (name required by SQLAlchemy)
        """Comparison operators and derived expressions for Tile columns."""

        @property
        def lower(self) -> sa.ColumnElement[int]:
            """Lower bound of the tile as a Point expression."""
            return sa.func.lower(self, type_=Point)

        @property
        def upper(self) -> sa.ColumnElement[int]:
            """Upper bound of the tile as a Point expression."""
            return sa.func.upper(self, type_=Point)

        @property
        def length(self) -> sa.ColumnElement[int]:
            """Number of level-29 pixels contained in the tile."""
            return self.upper - self.lower

        @property
        def area(self) -> sa.ColumnElement[float]:
            """Area of the tile in steradians."""
            return sa.type_coerce(self.length * PIXEL_AREA_LITERAL, sa.Float)

    @classmethod
    def tiles_from(cls, obj: MOC | SkyCoord) -> Iterator[str]:
        """Generate tile range strings covering a MOC or a polygon."""
        if isinstance(obj, MOC):
            return cls.tiles_from_moc(obj)
        if isinstance(obj, SkyCoord):
            return cls.tiles_from_polygon_skycoord(obj)
        msg = f"Unknown type: {type(obj).__name__}"
        raise TypeError(msg)

    @classmethod
    def tiles_from_polygon_skycoord(cls, polygon: SkyCoord) -> Iterator[str]:
        """Generate tile range strings covering a polygon on the sky."""
        return cls.tiles_from_moc(
            MOC.from_polygon_skycoord(polygon.transform_to(HPX.frame))
        )

    @classmethod
    def tiles_from_moc(cls, moc: MOC) -> Iterator[str]:
        """Generate tile range strings covering a MOC."""
        return (f"[{lo},{hi})" for lo, hi in moc.to_depth29_ranges)

    @classmethod
    def tiles_from_uniq(cls, uniq: ArrayLike) -> Iterator[str]:
        """Convert an array of UNIQ indices to tile range strings."""
        level, ipix = uniq_to_level_ipix(np.asarray(uniq, dtype=np.int64))
        shift = 2 * (LEVEL - level)
        lo = ipix << shift
        hi = (ipix + 1) << shift
        return (f"[{a},{b})" for a, b in zip(lo.tolist(), hi.tolist(), strict=True))


@event.listens_for(sa.Index, "after_parent_attach")
def _create_indices(index: sa.Index, _parent: sa.Table) -> None:
    """Set index method to SP-GiST_ for any indexed Tile or Region columns.

    .. _SP-GiST: https://www.postgresql.org/docs/current/spgist.html
    """
    if (
        index._column_flag  # noqa: SLF001
        and len(index.expressions) == 1
        and isinstance(index.expressions[0], sa.Column)
        and isinstance(index.expressions[0].type, Tile)
    ):
        index.dialect_options["postgresql"]["using"] = "spgist"
