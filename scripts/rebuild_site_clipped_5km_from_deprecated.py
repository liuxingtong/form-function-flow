"""
Rebuild data/site_clipped from data/deprecated using a 1 km buffer around SITE.json
(red line closed to polygon), write intermediate layout (geojson/shapefile/csv),
then run reorganize_site_clipped_by_theme.py pattern manually after — or chain commands:

  python scripts/rebuild_site_clipped_5km_from_deprecated.py
  python scripts/reorganize_site_clipped_by_theme.py
  python scripts/generate_vts_maps.py

Does not move or archive deprecated sources.

Run from repo root.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import geopandas as gpd
from shapely.prepared import prep

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from clip_data_to_site import (
    clip_csv_points,
    clip_gdf,
    clip_geojson_file,
    detect_csv_spec,
    load_clip_mask,
)

BUFFER_M = 1000.0
DEPRECATED_ROOT_NAME = "deprecated"


def iter_shapefiles_deprecated(data_root: Path, deprecated_root: Path) -> list[Path]:
    out = []
    for p in deprecated_root.rglob("*.shp"):
        if p.name.endswith(".shp.xml"):
            continue
        try:
            rel = p.relative_to(data_root)
        except ValueError:
            continue
        if "site_clipped" in rel.parts:
            continue
        out.append(p)
    return sorted(out)


def iter_csv_deprecated(data_root: Path, deprecated_root: Path) -> list[Path]:
    out = []
    for p in deprecated_root.rglob("*.csv"):
        try:
            rel = p.relative_to(data_root)
        except ValueError:
            continue
        if "site_clipped" in rel.parts:
            continue
        out.append(p)
    return sorted(out)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    data_root = repo / "data"
    site_path = data_root / "SITE.json"
    deprecated_root = data_root / DEPRECATED_ROOT_NAME

    if not site_path.exists():
        print("SITE.json not found:", site_path, file=sys.stderr)
        return 1
    if not deprecated_root.is_dir():
        print("Missing deprecated folder:", deprecated_root, file=sys.stderr)
        return 1

    mask = load_clip_mask(site_path, BUFFER_M)
    poly_wgs = mask.geometry.iloc[0]
    if poly_wgs.geom_type not in ("Polygon", "MultiPolygon"):
        print("Clip mask must be Polygon or MultiPolygon.", file=sys.stderr)
        return 1
    prepared = prep(poly_wgs)

    out_root = data_root / "site_clipped"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    geo_out = out_root / "geojson"
    geo_out.mkdir(parents=True, exist_ok=True)

    for name in ("AOI.json", "BUIDING.json", "RAIL.json", "ROAD.json"):
        hits = sorted(deprecated_root.rglob(name))
        if not hits:
            continue
        src = hits[0]
        print(f"[geojson] {src.relative_to(data_root)} …")
        n0, n1 = clip_geojson_file(src, mask, geo_out / name)
        print(f"    features {n0} -> {n1}")

    shutil.copy2(site_path, out_root / "SITE.json")

    buf_fc = json.loads(mask.to_json())
    buf_fc.setdefault("name", "SITE_buffer_1km")
    buf_fc["crs"] = {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}
    with (geo_out / "SITE.buffer_1km.meta.geojson").open("w", encoding="utf-8") as f:
        json.dump(buf_fc, f, ensure_ascii=False, separators=(",", ":"))

    shp_root_out = out_root / "shapefile"
    shp_root_out.mkdir(parents=True, exist_ok=True)
    shp_files = iter_shapefiles_deprecated(data_root, deprecated_root)
    print(f"[shapefile] {len(shp_files)} layers …")
    ok_shp = 0
    for shp in shp_files:
        rel = shp.relative_to(deprecated_root)
        dest = shp_root_out / rel
        gdf = None
        last_err: Exception | None = None
        for enc in ("utf-8", "gb18030", "gbk", "latin1"):
            try:
                gdf = gpd.read_file(shp, encoding=enc)
                break
            except Exception as e:
                last_err = e
        if gdf is None:
            print(f"    SKIP {rel}: {last_err}")
            continue
        n0 = len(gdf)
        clipped = clip_gdf(gdf, mask)
        n1 = len(clipped)
        if n1 == 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            clipped.to_file(dest, driver="ESRI Shapefile", encoding="utf-8")
        except Exception as e:
            alt = dest.with_suffix(".geojson")
            try:
                clipped.to_file(alt, driver="GeoJSON", encoding="utf-8")
                print(f"    [as GeoJSON] {rel}: shapefile failed ({e})")
            except Exception as e2:
                print(f"    SKIP {rel}: {e2}")
                continue
        ok_shp += 1
        if ok_shp <= 5 or n1 < 500:
            print(f"    {rel}: {n0} -> {n1}")
    print(f"[shapefile] wrote {ok_shp} non-empty layers")

    csv_out = out_root / "csv"
    csv_out.mkdir(parents=True, exist_ok=True)
    for csv_path in iter_csv_deprecated(data_root, deprecated_root):
        rel = csv_path.relative_to(deprecated_root)
        spec = detect_csv_spec(csv_path)
        if spec is None:
            continue
        lon_c, lat_c, mode = spec
        dest = csv_out / rel
        print(f"[csv] {rel} ({mode}) …")
        try:
            t0, t1 = clip_csv_points(csv_path, dest, prepared, poly_wgs, lon_c, lat_c, mode)
            print(f"    rows {t0} -> {t1}")
        except Exception as e:
            print(f"    FAIL {rel}: {e}")

    print("Done. Clip radius:", BUFFER_M, "m around SITE.json")
    print("Next: python scripts/reorganize_site_clipped_by_theme.py")
    print("Then: python scripts/generate_vts_maps.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
