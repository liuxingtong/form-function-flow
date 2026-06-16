# Cluster Generator Tasks

## Immediate

1. Implement `generate_clusters` core pipeline in backend:
- load morphology template
- generate candidate footprints in zone mask
- resolve overlaps
- assign scenario properties

2. Add validation integration:
- run existing Phase 2 rule checks
- return diagnostics with rule IDs

3. Add backend endpoint:
- `POST /api/site-design/cluster/generate`

4. Wire frontend action:
- "Generate Cluster" panel
- template selector + intensity inputs + seed
- append accepted blocks into current scenario store

## Next

1. Add deterministic regression fixture (same seed -> same blocks).
2. Add performance benchmark script.
3. Add optional OSMnx context ingestion path (road/POI influence).
