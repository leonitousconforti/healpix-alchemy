"""Benchmarks for common queries, at a range of data sizes."""

from collections.abc import Callable, Sequence
from functools import reduce
from typing import Any

import numpy as np
import pytest
import sqlalchemy as sa
from astropy.coordinates import SkyCoord
from mocpy import MOC
from numpy.typing import NDArray
from pytest_benchmark.fixture import BenchmarkFixture
from sqlalchemy import orm

from healpix_alchemy import func
from healpix_alchemy.constants import PIXEL_AREA, PIXEL_AREA_LITERAL
from healpix_alchemy.types import Tile

from .models import FieldTile, Galaxy, SkymapTile

CREDIBLE_LEVEL = 0.9

Bench = Callable[[sa.Select[Any]], Sequence[sa.Row[Any]]]
Expected = NDArray[Any] | tuple[tuple[float, ...], ...]
BenchAndCheck = Callable[[sa.Select[Any], Expected], None]


@pytest.fixture
def bench(benchmark: BenchmarkFixture, session: orm.Session) -> Bench:
    """Time a query, running ANALYZE first so statistics are current."""

    def _func(query: sa.Select[Any]) -> Sequence[sa.Row[Any]]:
        session.execute(sa.text("ANALYZE"))
        return benchmark(lambda: session.execute(query).all())

    return _func


@pytest.fixture
def bench_and_check(bench: Bench) -> BenchAndCheck:
    """Time a query and verify its result against an expected value."""

    def _func(query: sa.Select[Any], expected: Expected) -> None:
        np.testing.assert_almost_equal(bench(query), expected, decimal=6)

    return _func


def test_union_area(bench_and_check: BenchAndCheck, random_fields: list[MOC]) -> None:
    """Find the area of the union of N fields."""
    # Assemble query
    subquery = sa.select(func.union(FieldTile.hpx).label("hpx")).subquery()
    query = sa.select(sa.func.sum(subquery.columns.hpx.area))

    # Expected result
    union = reduce(lambda a, b: a.union(b), random_fields)
    result = union.sky_fraction * 4 * np.pi
    expected = ((result,),)

    # Run benchmark, check result
    bench_and_check(query, expected)


def test_crossmatch_galaxies_and_fields(
    bench_and_check: BenchAndCheck,
    random_fields: list[MOC],
    random_galaxies: SkyCoord,
) -> None:
    """Cross match N galaxies with M fields."""
    # Assemble query
    count = sa.func.count(Galaxy.id)
    query = (
        sa.select(count)
        .filter(FieldTile.hpx.contains(Galaxy.hpx))
        .group_by(FieldTile.id)
        .order_by(count.desc())
        .limit(5)
    )

    # Expected result
    points = random_galaxies
    fields = random_fields
    result = np.sum([moc.contains_skycoords(points) for moc in fields], axis=1)
    expected = np.flipud(np.sort(result))[:5].reshape(-1, 1)

    # Run benchmark, check result
    bench_and_check(query, expected)


def test_fields_in_90pct_credible_region(
    bench_and_check: BenchAndCheck,
    random_fields: list[MOC],
    random_sky_map: tuple[list[int], NDArray[np.float64]],
) -> None:
    """Find which of N fields overlap the 90% credible region."""
    # Assemble query
    cum_prob = (
        sa.func.sum(SkymapTile.probdensity * SkymapTile.hpx.area)
        .over(order_by=SkymapTile.probdensity.desc())
        .label("cum_prob")
    )
    subquery1 = (
        sa.select(SkymapTile.probdensity, cum_prob)
        .filter(SkymapTile.id == 1)
        .subquery()
    )
    min_probdensity = (
        sa.select(sa.func.min(subquery1.columns.probdensity))
        .filter(subquery1.columns.cum_prob <= CREDIBLE_LEVEL)
        .scalar_subquery()
    )

    # Materialize the credible region as merged tiles so that the planner
    # scans it first and probes the (much larger) field tile index, instead
    # of the reverse. The probdensity threshold comes from a subquery, so
    # the planner cannot estimate its selectivity and otherwise picks a
    # join order that is slower by orders of magnitude.
    credible = (
        sa.select(func.union(SkymapTile.hpx).label("hpx"))
        .filter(SkymapTile.probdensity >= min_probdensity)
        .cte("credible")
        .prefix_with("MATERIALIZED")
    )

    query = sa.select(sa.func.count(FieldTile.id.distinct())).filter(
        credible.columns.hpx.overlaps(FieldTile.hpx)
    )

    # Expected result
    tiles, probdensity = random_sky_map
    lo, hi = np.asarray(tiles[:-1]), np.asarray(tiles[1:])
    order = np.argsort(probdensity)[::-1]
    cum_prob = np.cumsum((probdensity * (hi - lo))[order]) * PIXEL_AREA
    threshold = float(np.min(probdensity[order][cum_prob <= CREDIBLE_LEVEL]))
    # The credible tiles are disjoint and sorted, so a field overlaps the
    # credible region if and only if, for any of its tile ranges [a, b),
    # the first credible tile ending after a starts before b.
    cred_lo, cred_hi = lo[probdensity >= threshold], hi[probdensity >= threshold]
    result = 0
    for moc in random_fields:
        a, b = np.transpose(moc.to_depth29_ranges.astype(np.int64))
        i = np.searchsorted(cred_hi, a, side="right")
        result += bool(
            np.any((i < len(cred_lo)) & (cred_lo[np.minimum(i, len(cred_lo) - 1)] < b))
        )
    expected = ((result,),)

    # Run benchmark, check result
    bench_and_check(query, expected)


def test_integrated_probability(
    bench_and_check: BenchAndCheck,
    random_fields: list[MOC],
    random_sky_map: tuple[list[int], NDArray[np.float64]],
) -> None:
    """Find the probability contained within the union of N fields."""
    # Assemble query. Feeding all field tiles straight into `func.union`
    # makes `range_agg` merge millions of mostly-overlapping ranges, and
    # joining the union against the sky map by range overlap needs one index
    # probe per union range, which dominates when the fields are sparse and
    # the union has tens of thousands of ranges. Instead:
    #
    # 1. Sort the field tiles by their lower bound and keep only the part of
    #    each tile that extends past the running maximum of the upper bounds
    #    seen so far. These "extension" segments are disjoint and cover
    #    exactly the union, so the remaining `range_agg` merges them cheaply.
    tile = sa.select(
        FieldTile.hpx.lower.label("lo"),
        FieldTile.hpx.upper.label("hi"),
        sa.func.max(FieldTile.hpx.upper)
        .over(order_by=FieldTile.hpx.lower, rows=(None, -1))
        .label("prev_hi"),
    ).subquery()
    segment = (
        sa.select(
            sa.func.int8range(
                sa.func.greatest(tile.columns.lo, tile.columns.prev_hi),
                tile.columns.hi,
                type_=Tile,
            ).label("hpx")
        )
        .filter(
            sa.or_(
                tile.columns.prev_hi.is_(None),
                tile.columns.hi > tile.columns.prev_hi,
            )
        )
        .subquery()
    )
    union = (
        sa.select(func.union(segment.columns.hpx).label("hpx"))
        .cte("field_union")
        .prefix_with("MATERIALIZED")
    )

    # 2. The integral over each union range [lo, hi) is F(hi) - F(lo), where
    #    F is the cumulative integral of the sky map: a continuous, piecewise
    #    linear function of the pixel index. Evaluate F at all the union
    #    boundaries at once by merging them with the sorted sky map tiles: a
    #    running count of sky map rows in the merged stream gives the sky map
    #    tile that governs each boundary, replacing per-range index probes
    #    with sorts.
    skymap = (
        sa.select(
            sa.func.row_number().over(order_by=SkymapTile.hpx.lower).label("rn"),
            SkymapTile.hpx.lower.label("x"),
            SkymapTile.probdensity.label("probdensity"),
            sa.func.coalesce(
                sa.func.sum(SkymapTile.probdensity * SkymapTile.hpx.length).over(
                    order_by=SkymapTile.hpx.lower, rows=(None, -1)
                ),
                0.0,
            ).label("cum"),
        )
        .filter(SkymapTile.id == 1)
        .cte("skymap_cum")
        .prefix_with("MATERIALIZED")
    )
    boundary = sa.union_all(
        sa.select(
            union.columns.hpx.lower.label("x"),
            sa.literal(-1).label("sign"),
            sa.false().label("is_skymap"),
        ),
        sa.select(union.columns.hpx.upper, sa.literal(1), sa.false()),
        sa.select(SkymapTile.hpx.lower, sa.literal(0), sa.true()).filter(
            SkymapTile.id == 1
        ),
    ).subquery()
    event = sa.select(
        boundary.columns.x,
        boundary.columns.sign,
        boundary.columns.is_skymap,
        sa.func.count()
        .filter(boundary.columns.is_skymap)
        .over(
            order_by=(boundary.columns.x, boundary.columns.is_skymap.desc()),
            rows=(None, 0),
        )
        .label("rn"),
    ).subquery()

    prob = (
        sa.func.sum(
            event.columns.sign
            * (
                skymap.columns.cum
                + skymap.columns.probdensity * (event.columns.x - skymap.columns.x)
            )
        )
        * PIXEL_AREA_LITERAL
    )
    query = (
        sa.select(prob)
        .select_from(event.join(skymap, event.columns.rn == skymap.columns.rn))
        .filter(~event.columns.is_skymap)
    )

    # Expected result: merge every field's tile ranges into a union, then
    # integrate the sky map's probability density over the merged ranges.
    ranges = np.concatenate([moc.to_depth29_ranges for moc in random_fields]).astype(
        np.int64
    )
    order = np.argsort(ranges[:, 0])
    lo, hi = ranges[order, 0], ranges[order, 1]
    running_hi = np.maximum.accumulate(hi)
    starts = np.flatnonzero(np.concatenate(([True], lo[1:] > running_hi[:-1])))
    union_lo = lo[starts]
    union_hi = running_hi[np.append(starts[1:] - 1, len(hi) - 1)]

    tiles, probdensity = random_sky_map
    bounds = np.asarray(tiles)
    cum = np.concatenate(([0.0], np.cumsum(probdensity * np.diff(bounds))))

    def integral(x: NDArray[np.int64]) -> NDArray[np.float64]:
        """Integrate the probability density over level-29 pixels [0, x)."""
        i = np.clip(
            np.searchsorted(bounds, x, side="right") - 1, 0, len(probdensity) - 1
        )
        return cum[i] + probdensity[i] * (x - bounds[i])

    result = np.sum(integral(union_hi) - integral(union_lo)) * PIXEL_AREA
    expected = ((result,),)

    # Run benchmark, check result
    bench_and_check(query, expected)
