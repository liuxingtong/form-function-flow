# Site Design Platform

This folder isolates the site design platform from generic repository scripts.

## Structure

- `frontend/`: static web UI (MapLibre + Carto Light basemap)
- `backend/`: lightweight static file server
- `tools/`: data preparation pipeline for frontend datasets
- `start.py`: one-command launcher (prepare data + run server + open browser)

## Run

From repository root:

```bash
python apps/site_design_platform/start.py
```

Or use one-click script:

```bash
run_site_design_platform.bat
```

## Rhino Sync (MVP)

The backend provides a lightweight Rhino sync API:

- `POST /api/site-design/rhino/sync`: save latest Rhino scenario
- `GET /api/site-design/rhino/latest`: fetch latest Rhino scenario

Saved file path:

- `data/site_design_platform/scenarios/rhino_live.json`

### Scenario JSON (minimum)

```json
{
  "schema_version": "0.2.0",
  "scenario_name": "rhino_live",
  "blocks": [
    {
      "id": "rh_1",
      "geometry": { "type": "Polygon", "coordinates": [[[121.45,31.24],[121.4502,31.24],[121.4502,31.2402],[121.45,31.2402],[121.45,31.24]]] },
      "function": "OFFICE",
      "height": 48,
      "base": 0,
      "floors": 12,
      "cost_params": { "hard_cost_per_sqm": 7000, "soft_cost_ratio": 0.2, "infra_cost_per_sqm": 650, "contingency_ratio": 0.08 },
      "revenue_params": { "sale_price_per_sqm": 0, "rent_price_per_sqm_year": 3800, "occupancy": 0.9, "opex_ratio": 0.22, "cap_rate": 0.05 }
    }
  ]
}
```

### Push script

Use this helper to push Rhino-exported scenario JSON:

```bash
python apps/site_design_platform/tools/push_rhino_scenario.py --input path/to/rhino_scenario.json
```
