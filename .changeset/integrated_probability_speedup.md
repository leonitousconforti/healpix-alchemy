---
default: misc
---

# Speed up the integrated probability benchmark query

Reduce the field tiles to disjoint "extension" segments with a window function
before merging them with `range_agg`, and evaluate the sky map's cumulative
integral at the union boundaries with a sort-merge sweep instead of one overlap
index probe per union range. Also raise `work_mem` for the disposable test
database so the sorts stay in memory.
