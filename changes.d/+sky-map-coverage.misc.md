Fix the random sky map used by the benchmarks to cover the whole sky:
`np.arange` computes its length in floating point and silently dropped the
last coarse tile, leaving 1/12 of the sky without any sky map tiles.
