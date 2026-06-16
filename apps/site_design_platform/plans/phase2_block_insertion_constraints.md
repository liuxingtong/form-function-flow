# Phase 2 Planning Constraints (CBD / OFC / TOD)

This document captures the confirmed intent and translates it into implementable constraints for block insertion.

## 1) Confirmed Planning Intent

- **CBD**: primary station-city integration area, led by central business functions.
- **OFC**: green + cultural/creative industry park direction.
- **TOD**:
  - Ground level: rail yard / maintenance context; only small leisure-commercial insertions.
  - Over-track deck: residential-led development with saleable supporting public facilities.

## 2) Spatial Scope and Zone Binding

Use existing zone layers as hard assignment masks:

- `data/district_dxf/crs84/Z_CBD.geojson`
- `data/district_dxf/crs84/Z_OFC.geojson`
- `data/district_dxf/crs84/Z_TOD.geojson`

Each inserted block must bind to exactly one zone:

- `zone_id in {"CBD","OFC","TOD_GROUND","TOD_DECK"}`
- `TOD_GROUND` and `TOD_DECK` both come from `Z_TOD`, but distinguished by `deck_level` (`ground` / `over_track`).

## 3) Recommended Hard Constraints (for editor validation)

## 3.1 General (all zones)

- **Inside mask**: inserted polygon must be fully within its zone polygon (allow tiny tolerance).
- **No overlap**: new blocks cannot overlap existing preserved blocks unless operation is "replace/split-merge".
- **Minimum parcel area**: reject parcels below minimum editable footprint (suggest 150-250 sqm for MVP).
- **Max aspect ratio**: avoid extremely thin parcels (e.g., long side / short side <= 8 for MVP).
- **Access edge**: each inserted block must touch a walkable edge or designated internal street edge.

## 3.2 CBD constraints

Target: business-led station-city integration.

- **Use mix by GFA (recommended ranges)**:
  - Office / business: **45-70%**
  - Commercial retail/F&B: **15-30%**
  - Public-support facilities: **10-20%**
  - Residential: **0-20%** (optional, not dominant)
- **Height**:
  - Core frontage: medium-high to high intensity (project-specific), but preserve key view corridors.
- **Ground-floor activation**:
  - Main pedestrian edges require active frontage ratio >= 60%.
- **Public continuity**:
  - Keep at least one continuous through-route connecting station and major destinations.

## 3.3 OFC constraints

Target: green + cultural/creative campus.

- **Use mix by GFA**:
  - Cultural/creative office/R&D: **35-60%**
  - Green/open-space related uses (incl. civic recreation): **20-40%**
  - Retail/F&B support: **10-20%**
  - Residential (if any): **0-20%**
- **Ecology/open-space control**:
  - Zone-level open space ratio target >= 30% (project configurable).
- **Height profile**:
  - Prefer low-mid rise gradient near green corridors.
- **Industrial-to-creative compatibility**:
  - Allow flexible floorplates and shared service yards, avoid heavy logistics conflict with public open space.

## 3.4 TOD ground constraints

Target: rail yard-compatible, light insertion.

- **Allowed uses only**:
  - Small leisure/commercial pods, transit services, station-support amenities.
- **Height cap (MVP recommendation)**:
  - Typical cap <= 12m (user preference from concept sketch).
- **Footprint cap**:
  - Single inserted pod footprint controlled (e.g., <= 1,500 sqm, configurable).
- **Safety setbacks**:
  - Reserve rail operation buffer (distance configured from rail/service boundaries).
- **No residential at ground within rail-maintenance envelope**.

## 3.5 TOD deck constraints

Target: residential-led over-track development with saleable public-support package.

- **Use mix by GFA**:
  - Residential: **50-75%**
  - Community/public-support facilities: **10-20%**
  - Local retail service: **10-20%**
  - Office: **0-15%**
- **Deck-specific rules**:
  - Apply over-track structural premium in cost model.
  - Enforce vertical evacuation + emergency access corridors.
- **Livability baseline**:
  - Daylight/spacing/parking checks should reference applicable local standards and project approvals.

## 4) "Concept Sketch" Granularity for Initial Insertion

The requested granularity is conceptual massing, not construction-detail BIM.

Recommended MVP block primitives:

- `MX_PODIUM`: mixed-use podium block (2-5 floors)
- `RES_SLAB`: slab residential bar (8-18 floors)
- `OFFICE_BOX`: office/campus box (4-12 floors)
- `CULTURE_BOX`: cultural/creative low-rise (2-6 floors)
- `PUBLIC_SUPPORT`: public-support module (1-4 floors)
- `LEISURE_KIOSK`: small TOD-ground leisure/commercial kiosk (1-2 floors)

Each primitive needs only:

- footprint polygon
- floor count / height
- use type
- optional frontage class

## 5) Validation Order (implementation-ready)

For each edit operation:

1. geometry validity
2. zone containment
3. overlap/topology check
4. zone-specific hard constraints
5. zone-level mix/ratio check (post-operation aggregate)
6. if fail: return exact violated rule id

## 6) Rule IDs for Codex Implementation

Examples:

- `GEO_INSIDE_ZONE`
- `GEO_NO_OVERLAP`
- `CBD_MIN_OFFICE_SHARE`
- `OFC_MIN_OPENSPACE_RATIO`
- `TODG_MAX_HEIGHT_12M`
- `TODD_MIN_RES_SHARE`
- `TODD_MIN_PUBLIC_SUPPORT_SHARE`

## 7) Reference Basis (for policy tuning)

- ITDP TOD Standard (Mix / Densify / Walk / Connect principles): <https://tod.itdp.org/tod-standard/tod-standard-framework.html>
- China TOD local implementation examples (station-area integrated development rules):
  - Guangzhou station complex integrated development notice: <https://www.gz.gov.cn/zwgk/gzsrmzfgbn/2025/7/content/post_10612097.html>
  - Hangzhou TOD implementation details: <https://zfgb.hangzhou.gov.cn/10/112220253/t126220253124/530030.shtml>
- Residential planning baseline reference (China): GB 50180-2018 overview resource:
  <http://m.jianbiaoku.com/webarbs/book/1095/3781397.shtml>


## 8) Next Planned Capability: Lightweight Cluster Auto-Generation

Add a new Phase 2C planning task after 2B constraints editor:

- Goal: quickly generate plausible building clusters in batch from parcel geometry constraints + target function/morphology type.
- Requirement: prioritize open-source lightweight algorithms with transparent heuristics and low runtime.

### 8.1 Candidate algorithm families (open-source, lightweight)

- **Polygon packing / rectangle packing heuristics**
  - for fast placement of slab/bar/box footprints inside constrained parcels.
- **Straight skeleton / medial-axis based subdivision**
  - for generating internal bands/courtyards and lane-aligned buildable strips.
- **Cellular tiling with rule filters (grid/hex/voronoi-lite)**
  - for rapid massing seeding + post-filter by frontage, area, aspect ratio, setback.
- **Shape grammar / pattern rules (minimal DSL)**
  - for translating morphology archetypes to repeatable placement rules.

### 8.2 Morphology Library (required)

Create a maintainable morphology library under `apps/site_design_platform` with templates such as:

- `RES_SLAB_CLUSTER`
- `MIXED_PODIUM_TOWER`
- `OFFICE_CAMPUS_BOX`
- `CULTURE_COURTYARD_LOWRISE`
- `TOD_DECK_RESIDENTIAL_BAND`

Each template should include:

- allowed zone ids
- footprint primitive set
- target floor/height ranges
- spacing/setback defaults
- use-mix defaults
- generation steps + tunable parameters

### 8.3 MVP deliverables

- `cluster-generator` module (deterministic seed support)
- `morphology-library.json`
- UI action: "Generate Cluster" with template + intensity inputs
- output blocks directly editable in 2B editor and fully serializable into scenario JSON
