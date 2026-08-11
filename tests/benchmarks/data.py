"""SQLAlchemy sample data for unit tests.

Notes
-----
We use the psycopg ``copy`` rather than SQLAlchemy for fast insertion.

The data is deterministic, so generation is memoized: several benchmarks
request the same sizes, and only the ``COPY`` into each test's fresh database
needs to happen per test.

"""

from functools import cache

import numpy as np
import psycopg
import pytest
from astropy import units as u
from astropy.coordinates import (
    SkyCoord,
    UnitSphericalRepresentation,
    uniform_spherical_random_surface,
)
from mocpy import MOC
from numpy.typing import NDArray
from psycopg import sql

from healpix_alchemy.constants import HPX, LEVEL, PIXEL_AREA
from healpix_alchemy.types import Tile

from .models import Field, FieldTile, Galaxy, Skymap, SkymapTile


def _copy_from_stdin(table: str) -> sql.Composed:
    return sql.SQL("COPY {} FROM STDIN").format(sql.Identifier(table))


(RANDOM_GALAXIES_SEED, RANDOM_FIELDS_SEED, RANDOM_SKY_MAP_SEED) = (
    np.random.SeedSequence(12345).spawn(3)
)


def get_ztf_footprint_corners() -> tuple[u.Quantity, u.Quantity]:
    """Return the corner offsets of the ZTF footprint.

    Notes
    -----
    This polygon is smaller than the spatial extent of the true ZTF field of
    view, but has approximately the same area because the real ZTF field of
    view has chip gaps.

    For the real ZTF footprint, use the region file
    https://github.com/skyportal/skyportal/blob/main/data/ZTF_Region.reg.

    """
    x = 6.86 / 2
    return [-x, +x, +x, -x] * u.deg, [-x, -x, +x, +x] * u.deg


def get_footprints_grid(
    lon: u.Quantity, lat: u.Quantity, offsets: SkyCoord
) -> SkyCoord:
    """Get a grid of footprints for an equatorial-mount telescope.

    Parameters
    ----------
    lon : astropy.units.Quantity
        Longitudes of footprint vertices at the standard pointing.
        Should be an array of length N.
    lat : astropy.units.Quantity
        Latitudes of footprint vertices at the standard pointing.
        Should be an array of length N.
    offsets : astropy.coordinates.SkyCoord
        Pointings for the field grid.
        Should have length M.

    Returns
    -------
    astropy.coordinates.SkyCoord
        Footprints with dimensions (M, N).

    """
    lon_grid = np.repeat(lon[np.newaxis, :], len(offsets), axis=0)
    lat_grid = np.repeat(lat[np.newaxis, :], len(offsets), axis=0)
    result = SkyCoord(
        lon_grid, lat_grid, frame=offsets[:, np.newaxis].skyoffset_frame()
    )
    return result.icrs


def get_random_points(
    n: int, seed: np.random.SeedSequence
) -> UnitSphericalRepresentation:
    """Generate n points drawn uniformly from the celestial sphere."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(np, "random", np.random.default_rng(seed))
        return uniform_spherical_random_surface(n)


@cache
def _generate_galaxies(n: int) -> tuple[SkyCoord, str]:
    points = SkyCoord(get_random_points(n, RANDOM_GALAXIES_SEED))
    hpx = HPX.skycoord_to_healpix(points)
    return points, "\n".join(f"{i}" for i in hpx)


def get_random_galaxies(n: int, cursor: psycopg.Cursor) -> SkyCoord:
    """Load n random galaxies and return their coordinates."""
    points, rows = _generate_galaxies(n)

    with cursor.copy(
        sql.SQL("COPY {} (hpx) FROM STDIN").format(sql.Identifier(Galaxy.__tablename__))
    ) as copy:
        copy.write(rows)

    return points


@cache
def _generate_fields(n: int) -> tuple[list[MOC], str, str]:
    centers = SkyCoord(get_random_points(n, RANDOM_FIELDS_SEED))
    footprints = get_footprints_grid(*get_ztf_footprint_corners(), centers)
    mocs = [MOC.from_polygon_skycoord(footprint) for footprint in footprints]
    field_rows = "\n".join(f"{i}" for i in range(len(mocs)))
    tile_rows = "\n".join(
        f"{i}\t{hpx}" for i, moc in enumerate(mocs) for hpx in Tile.tiles_from(moc)
    )

    return mocs, field_rows, tile_rows


def get_random_fields(n: int, cursor: psycopg.Cursor) -> list[MOC]:
    """Load n random telescope fields and return their footprints."""
    mocs, field_rows, tile_rows = _generate_fields(n)

    with cursor.copy(_copy_from_stdin(Field.__tablename__)) as copy:
        copy.write(field_rows)

    with cursor.copy(_copy_from_stdin(FieldTile.__tablename__)) as copy:
        copy.write(tile_rows)

    return mocs


@cache
def _generate_sky_map(n: int) -> tuple[list[int], NDArray[np.float64], str]:
    rng = np.random.default_rng(RANDOM_SKY_MAP_SEED)
    # Make a randomly subdivided sky map
    # Use `range` rather than `np.arange`: the latter computes its length in
    # floating point, and npix + 1 is not representable in float64, so it
    # silently drops the last coarse tile and leaves 1/12 of the sky uncovered.
    npix = int(HPX.npix)
    tiles = list(range(0, npix + 1, 4**LEVEL))
    while len(tiles) < n:
        i = int(rng.integers(len(tiles)))
        lo = 0 if i == 0 else tiles[i - 1]
        hi = tiles[i]
        diff = (hi - lo) // 4
        if diff == 0:
            continue
        tiles.insert(i, hi - diff)
        tiles.insert(i, hi - 2 * diff)
        tiles.insert(i, hi - 3 * diff)

    probdensity = rng.uniform(0, 1, size=len(tiles) - 1)
    probdensity /= np.sum(np.diff(tiles) * probdensity) * PIXEL_AREA

    rows = "\n".join(
        f"1\t[{lo},{hi})\t{p}"
        for lo, hi, p in zip(tiles[:-1], tiles[1:], probdensity, strict=True)
    )

    return tiles, probdensity, rows


def get_random_sky_map(
    n: int, cursor: psycopg.Cursor
) -> tuple[list[int], NDArray[np.float64]]:
    """Load a random n-tile sky map; return tile bounds and densities."""
    tiles, probdensity, rows = _generate_sky_map(n)

    with cursor.copy(_copy_from_stdin(Skymap.__tablename__)) as copy:
        copy.write("1")

    with cursor.copy(_copy_from_stdin(SkymapTile.__tablename__)) as copy:
        copy.write(rows)

    return tiles, probdensity
