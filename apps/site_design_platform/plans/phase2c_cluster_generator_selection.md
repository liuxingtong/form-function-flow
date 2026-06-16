# Phase 2C Technical Selection (Cluster Auto-Generation)

## Decision
Use a custom lightweight generator stack as the primary path:

- Data: OSMnx (optional external context), local SITE + district masks (mandatory)
- Geometry kernel: GeoPandas + Shapely
- Morphology helpers: momepy (block/tessellation utilities where useful)
- 3D export: trimesh (optional for mesh previews), frontend still consumes GeoJSON scenario blocks

Do not directly embed `bss116/citygenerator` as runtime dependency.
Treat it as algorithm reference only.

## Why

1. Better fit for current platform constraints (zone masks, rule IDs, scenario schema).
2. Lower coupling with existing editor and economics modules.
3. Cleaner licensing path for productization (avoid GPL runtime entanglement).

## Integration Principles

1. Keep generator isolated under `apps/site_design_platform/backend/cluster_generator`.
2. Generator output must be same scenario block schema used by frontend editor.
3. All generated blocks must pass Phase 2 rule validation hooks.
4. Deterministic generation via explicit seed.

## Pipeline (MVP)

1. Input parse
- site polygon
- target zone polygon(s)
- template id
- intensity params (target_gfa, avg_height, spacing)
- seed

2. Buildable envelope
- zone polygon minus setbacks/buffers
- optional split by access lines

3. Primitive placement
- template-driven placement (slab/bar/box/podium)
- collision check and clipping inside mask

4. Attribute assignment
- functionType
- Height/Base/Floors
- cost_params/revenue_params defaults

5. Validation and filtering
- geometry validity
- min area/aspect
- no overlap
- zone-specific caps (e.g. TOD ground height)

6. Scenario serialization
- output blocks[] directly consumable by frontend store

## Target API (Backend)

- `generate_clusters(request: ClusterGenerateRequest) -> ClusterGenerateResponse`

Request
- `scenario_name: str`
- `seed: int`
- `template_id: str`
- `zone_id: str`
- `site_geojson: FeatureCollection`
- `zone_geojson: FeatureCollection`
- `constraints: dict`
- `intensity: dict` (target_gfa, max_height, spacing, rotation_step)

Response
- `blocks: list[ScenarioBlock]`
- `diagnostics: {accepted, rejected, rejection_reasons}`
- `stats: {gfa, footprint, by_function}`

## Performance Budget

- Single-zone generation: <= 500 ms target for 50-150 blocks on laptop CPU.
- Validation pass: <= 300 ms.

## Milestones

- 2C.1: morphology library JSON + generator interface
- 2C.2: RES_SLAB_CLUSTER + OFFICE_CAMPUS_BOX templates
- 2C.3: validation hooks + diagnostics
- 2C.4: frontend "Generate Cluster" action
- 2C.5: economics panel auto-refresh from generated blocks
