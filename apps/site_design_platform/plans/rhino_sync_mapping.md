# Rhino Sync Mapping (MVP)

This document defines the minimal field mapping from Rhino objects/layers to
`site_design_platform` scenario JSON for `/api/site-design/rhino/sync`.

## 1) Recommended Rhino Layer Structure

- `SDP_BUILDING_OFFICE`
- `SDP_BUILDING_RESIDENTIAL`
- `SDP_BUILDING_COMMERCIAL`
- `SDP_BUILDING_MIXED_USE`

Each closed footprint curve/surface in these layers is treated as one block.

## 2) Object Property -> JSON Field Mapping

For each Rhino building object:

- `object_id` (Rhino GUID/string) -> `block.id`
- `layer/use` -> `block.function`
- footprint geometry (closed polyline/curve) -> `block.geometry` (GeoJSON Polygon)
- `UserText:height` -> `block.height` (meters)
- `UserText:base` -> `block.base` (meters, default `0`)
- `UserText:floors` -> `block.floors` (int, optional; if missing backend/frontend can derive)
- `UserText:cost_hard` -> `block.cost_params.hard_cost_per_sqm`
- `UserText:cost_soft` -> `block.cost_params.soft_cost_ratio`
- `UserText:cost_infra` -> `block.cost_params.infra_cost_per_sqm`
- `UserText:cost_cont` -> `block.cost_params.contingency_ratio`
- `UserText:sale_price` -> `block.revenue_params.sale_price_per_sqm`
- `UserText:rent_price` -> `block.revenue_params.rent_price_per_sqm_year`
- `UserText:occupancy` -> `block.revenue_params.occupancy`
- `UserText:opex_ratio` -> `block.revenue_params.opex_ratio`
- `UserText:cap_rate` -> `block.revenue_params.cap_rate`

## 3) Function Mapping Rules

Layer name or object user text -> `function`:

- contains `OFFICE` -> `OFFICE`
- contains `RES` -> `RESIDENTIAL`
- contains `COMM` or `RETAIL` -> `COMMERCIAL`
- otherwise -> `MIXED_USE`

## 4) Scenario JSON Envelope

```json
{
  "schema_version": "0.2.0",
  "scenario_name": "rhino_live",
  "generated_at": "2026-05-22T12:00:00Z",
  "blocks": []
}
```

Each `block`:

```json
{
  "id": "rh_xxx",
  "geometry": { "type": "Polygon", "coordinates": [[[121.45,31.24],[121.4502,31.24],[121.4502,31.2402],[121.45,31.2402],[121.45,31.24]]] },
  "function": "OFFICE",
  "height": 48,
  "base": 0,
  "floors": 12,
  "cost_params": {
    "hard_cost_per_sqm": 7000,
    "soft_cost_ratio": 0.2,
    "infra_cost_per_sqm": 650,
    "contingency_ratio": 0.08
  },
  "revenue_params": {
    "sale_price_per_sqm": 0,
    "rent_price_per_sqm_year": 3800,
    "occupancy": 0.9,
    "opex_ratio": 0.22,
    "cap_rate": 0.05
  }
}
```

## 5) Sync Procedure

1. Rhino side exports this JSON.
2. Push via:
   - `python apps/site_design_platform/tools/push_rhino_scenario.py --input <json_path>`
3. Frontend clicks `Load Rhino Scenario`.
4. Editor remains locked by default in Rhino-driven mode.

