Speed up the 90% credible region benchmark query by materializing the
credible region as merged tiles in a CTE, so that the query planner probes
the field tile index instead of scanning all field tiles.
