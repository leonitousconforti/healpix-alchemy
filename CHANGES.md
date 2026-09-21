# Changes

## 1.2.0 (2026-09-21)

### Features

#### Add `Tile.tiles_from_uniq`

Add ``Tile.tiles_from_uniq``, a vectorized helper that converts an array of
UNIQ indices to tile range strings for fast bulk ingestion of multi-order sky
maps via PostgreSQL ``COPY``.

### Fixes

#### Restore `healpix_alchemy.func` as a top-level attribute

Restore `healpix_alchemy.func` as an attribute of the top level package, so
that `ha.func.union()` works as documented in the README without a separate
`from healpix_alchemy import func` import.

### Other changes

- Require astropy-healpix >= 2

#### Speed up the 90% credible region benchmark query

Materialize the credible region as merged tiles in a CTE, so that the query
planner probes the field tile index instead of scanning all field tiles.

#### Add an integrated probability benchmark

Add a benchmark of the integrated probability within the union of a set of
fields, with the result verified against an independent numpy computation.

#### Speed up the integrated probability benchmark query

Reduce the field tiles to disjoint "extension" segments with a window function
before merging them with `range_agg`, and evaluate the sky map's cumulative
integral at the union boundaries with a sort-merge sweep instead of one overlap
index probe per union range. Also raise `work_mem` for the disposable test
database so the sorts stay in memory.

#### Fix the benchmark sky map to cover the whole sky

Fix the random sky map used by the benchmarks to cover the whole sky:
`np.arange` computes its length in floating point and silently dropped the
last coarse tile, leaving 1/12 of the sky without any sky map tiles.

#### Require sqlalchemy >= 2

Older versions were declared as supported but were no longer covered by the
test suite.

## 1.1.1 (2026-06-23)

- Support NumPy 2. Drop support for Python 3.10, which is no longer expected to
  be supported by scientific Python packages as per SPEC-0. Run unit tests for
  Python 3.11 through 3.14.

- Coerce ``LEVEL`` to a Python ``int``. Newer versions of mocpy return
  ``MOC.MAX_ORDER`` as a ``numpy.uint8``, which silently overflowed in
  expressions such as ``4**LEVEL``.

## 1.1.0 (2024-04-16)

- Don't include the unit tests in the installed Python package.

- Drop support for Python 3.9. Add support for, and run unit tests for,
  Python 3.12.

- Update build system for compatibility with the latest version of Poetry.

## 1.0.2 (2023-03-03)

- Track API changes in mocpy.

- Fix compatibility with SQLAlchemy 2.x.

## 1.0.1 (2021-12-08)

- Add benchmarks to unit test suite.

- Update the doctest examples in the README file to match the paper draft.

- Add code coverage analysis with Codecov.io.

## 1.0.0 (2021-11-22)

- First stable release.
