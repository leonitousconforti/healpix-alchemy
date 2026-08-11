Restore `healpix_alchemy.func` as an attribute of the top level package, so
that `ha.func.union()` works as documented in the README without a separate
`from healpix_alchemy import func` import.
