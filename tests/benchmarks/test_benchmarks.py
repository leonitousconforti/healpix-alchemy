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
from healpix_alchemy.constants import PIXEL_AREA

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
    """Find how many of N fields overlap the 90% credible region."""
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
        a, b = np.transpose(moc.to_depth29_ranges)
        i = np.searchsorted(cred_hi, a, side="right")
        result += bool(
            np.any((i < len(cred_lo)) & (cred_lo[np.minimum(i, len(cred_lo) - 1)] < b))
        )
    expected = ((result,),)

    # Run benchmark, check result
    bench_and_check(query, expected)
