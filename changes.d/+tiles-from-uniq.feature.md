Add ``Tile.tiles_from_uniq``, a vectorized helper that converts an array of
UNIQ indices to tile range strings for fast bulk ingestion of multi-order sky
maps via PostgreSQL ``COPY``.
