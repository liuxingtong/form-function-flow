"""
Reorganize data/site_clipped from file-type folders (geojson/shapefile/csv)
into theme-based folders, converting vectors and point-CSVs to GeoJSON.

Raster (WorldPop): clip to SITE polygon, write as GeoTIFF under 13_demography/
(lossless; full raster -> GeoJSON would require polygonizing every pixel).

Run from repo root:
  python scripts/reorganize_site_clipped_by_theme.py

Requires: geopandas, pandas, shapely, pyogrio, rasterio (for WorldPop)
"""
from __future__ import annotations

import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Polygon, mapping, shape

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from clip_data_to_site import load_clip_mask

_BUFFER_STUDY_M = 1000.0

try:
    import rasterio
    from rasterio.mask import mask as rio_mask
except ImportError:
    rasterio = None


def _out_of_china(lng: float, lat: float) -> bool:
    return not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271)


def _transform_lat(lng: float, lat: float) -> float:
    ret = (
        -100.0
        + 2.0 * lng
        + 3.0 * lat
        + 0.2 * lat * lat
        + 0.1 * lng * lat
        + 0.2 * math.sqrt(abs(lng))
    )
    ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * math.pi) + 40.0 * math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * math.pi) + 320 * math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lng(lng: float, lat: float) -> float:
    ret = (
        300.0
        + lng
        + 2.0 * lat
        + 0.1 * lng * lng
        + 0.1 * lng * lat
        + 0.1 * math.sqrt(abs(lng))
    )
    ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * math.pi) + 40.0 * math.sin(lng / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * math.pi) + 300.0 * math.sin(lng / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def gcj02_to_wgs84(lng: float, lat: float) -> tuple[float, float]:
    if _out_of_china(lng, lat):
        return lng, lat
    a = 6378245.0
    ee = 0.00669342162296594323
    dlat = _transform_lat(lng - 105.0, lat - 35.0)
    dlng = _transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return lng * 2 - mglng, lat * 2 - mglat


def load_site_polygon(site_path: Path) -> tuple[Polygon, str]:
    with site_path.open(encoding="utf-8") as f:
        fc = json.load(f)
    geom = fc["features"][0]["geometry"]
    g = shape(geom)
    if isinstance(g, LineString):
        c = list(g.coords)
        if c[0] != c[-1]:
            c.append(c[0])
        poly = Polygon(c)
    elif isinstance(g, Polygon):
        poly = g
    else:
        raise TypeError(type(g))
    if not poly.is_valid:
        poly = poly.buffer(0)
    crs = fc.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    if "CRS84" in str(crs) or "WGS" in str(crs).upper():
        epsg = "EPSG:4326"
    else:
        epsg = "EPSG:4326"
    return poly, epsg


def safe_name(stem: str, max_len: int = 120) -> str:
    s = re.sub(r'[<>:"/\\|?*]', "_", stem)
    s = s.strip() or "layer"
    return s[:max_len]


def classify_shapefile(rel: Path) -> str:
    """Return theme folder name from path relative to shapefile root."""
    parts_lower = [p.lower() for p in rel.parts]
    s = "/".join(parts_lower)
    stem = rel.stem.lower()

    if "01-建筑" in s or "建筑" in stem or "building" in stem:
        return "01_buildings"
    if "铁路" in stem or "rail" in stem or "03_rail" in s:
        return "03_rail"
    if "道路" in stem or "road" in stem or "路网" in s or "street" in stem:
        return "02_roads"
    if "02-poi" in s or "poi" in stem or "poidata" in stem:
        return "04_poi"
    if "aoi" in stem or "2-aoi" in s or "landuse-webmap" in s:
        return "05_aoi"
    if "地块" in s or "block" in stem or "plot" in stem:
        return "06_parcels"
    if "风貌" in s or "heritage" in stem or "conservation" in stem:
        return "07_heritage"
    if "土地利用" in s or "landuse" in stem or "建设用地" in s:
        return "08_landuse"
    if "04-交通" in s or "交通设施" in s or "收费站" in stem or "停车场" in stem or "轮渡" in stem or "港口" in stem:
        return "09_transport_facilities"
    if "13-常用公服" in s or "公服" in s:
        return "10_public_services"
    if "美团" in s or "meituan" in stem:
        return "11_commerce_meituan"
    if "03-" in s and "行政" in s:
        return "14_admin_boundaries"
    return "99_misc"


def classify_csv(rel: Path) -> str:
    s = "/".join(rel.parts).lower()
    stem = rel.stem.lower()
    if "大众点评" in stem or "dianping" in stem:
        return "11_commerce_dianping"
    if "楼盘" in s or "二手房" in stem or "新楼盘" in stem:
        return "12_real_estate"
    if "13-常用公服" in s or "公服" in s:
        return "10_public_services"
    if "02-poi" in s or "poi" in stem:
        return "04_poi"
    return "99_misc"


def csv_to_gdf(path: Path) -> gpd.GeoDataFrame | None:
    cols = {c.lower(): c for c in pd.read_csv(path, nrows=0, encoding="utf-8", encoding_errors="replace").columns}
    lon = lat = None
    mode = "wgs84"
    if "wgs84_lon" in cols and "wgs84_lat" in cols:
        lon, lat = cols["wgs84_lon"], cols["wgs84_lat"]
    elif "wgs84lng" in cols and "wgs84lat" in cols:
        lon, lat = cols["wgs84lng"], cols["wgs84lat"]
    elif "gcj02_lng" in cols and "gcj02_lat" in cols:
        lon, lat = cols["gcj02_lng"], cols["gcj02_lat"]
        mode = "gcj02"
    elif "lng" in cols and "lat" in cols and "wgs84" not in str(cols):
        lon, lat = cols["lng"], cols["lat"]
    else:
        return None

    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace", low_memory=False)
    if lon not in df.columns or lat not in df.columns:
        return None
    sub = df.dropna(subset=[lon, lat])
    if sub.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    lngv = sub[lon].astype(float)
    latv = sub[lat].astype(float)
    if mode == "gcj02":
        wx, wy = zip(*(gcj02_to_wgs84(a, b) for a, b in zip(lngv, latv)))
        geom = gpd.points_from_xy(wx, wy, crs="EPSG:4326")
    else:
        geom = gpd.points_from_xy(lngv, latv, crs="EPSG:4326")
    return gpd.GeoDataFrame(sub, geometry=geom, crs="EPSG:4326")


def write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf2 = gdf.copy()
    for col in gdf2.columns:
        if col == "geometry":
            continue
        if pd.api.types.is_datetime64_any_dtype(gdf2[col]):
            gdf2[col] = gdf2[col].astype(str)
    gdf2.to_file(path, driver="GeoJSON", encoding="utf-8")


def find_worldpop_dirs(search_roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_dir():
                continue
            n = p.name
            if "Worldpop" in n or "worldpop" in n.lower() or "人口密度" in n:
                out.append(p)
    return sorted(set(out))


def clip_rasters_to_site(
    worldpop_dirs: list[Path], site_poly_wgs84: Polygon, out_dir: Path, out_root: Path
) -> list[str]:
    if rasterio is None:
        return ["rasterio not installed; skipped WorldPop clip"]
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    site_gs = gpd.GeoDataFrame(geometry=[site_poly_wgs84], crs="EPSG:4326")
    for d in worldpop_dirs:
        for tif in sorted(d.glob("*.tif")) + sorted(d.glob("*.tiff")) + sorted(d.glob("*.TIF")):
            try:
                with rasterio.open(tif) as src:
                    crs = src.crs
                    geom = site_gs.to_crs(crs).geometry.iloc[0]
                    gjson = mapping(geom)
                    data, transform = rio_mask(src, [gjson], crop=True)
                    meta = src.meta.copy()
                    meta.update(
                        {
                            "driver": "GTiff",
                            "height": int(data.shape[1]),
                            "width": int(data.shape[2]),
                            "transform": transform,
                        }
                    )
                    out_path = out_dir / f"{safe_name(tif.stem)}_site.tif"
                    with rasterio.open(out_path, "w", **meta) as dst:
                        dst.write(data)
                    written.append(str(out_path.relative_to(out_root)))
            except Exception as e:
                written.append(f"FAIL {tif.name}: {e}")
    return written


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    data = repo / "data"
    site_path = data / "SITE.json"
    src_root = data / "site_clipped"
    if not src_root.exists():
        print("Missing:", src_root, file=sys.stderr)
        return 1

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = data / f"site_clipped__by_filetype_{ts}"
    if backup.exists():
        shutil.rmtree(backup)
    shutil.move(str(src_root), str(backup))
    out_root = data / "site_clipped"
    out_root.mkdir()

    site_poly, _ = load_site_polygon(site_path)
    shutil.copy2(site_path, out_root / "SITE.json")

    study_poly = load_clip_mask(site_path, _BUFFER_STUDY_M).geometry.iloc[0]

    themes = {
        "00_boundary": out_root / "00_boundary",
        "01_buildings": out_root / "01_buildings",
        "02_roads": out_root / "02_roads",
        "03_rail": out_root / "03_rail",
        "04_poi": out_root / "04_poi",
        "05_aoi": out_root / "05_aoi",
        "06_parcels": out_root / "06_parcels",
        "07_heritage": out_root / "07_heritage",
        "08_landuse": out_root / "08_landuse",
        "09_transport_facilities": out_root / "09_transport_facilities",
        "10_public_services": out_root / "10_public_services",
        "11_commerce_dianping": out_root / "11_commerce_dianping",
        "11_commerce_meituan": out_root / "11_commerce_meituan",
        "12_real_estate": out_root / "12_real_estate",
        "13_demography": out_root / "13_demography",
        "14_admin_boundaries": out_root / "14_admin_boundaries",
        "99_misc": out_root / "99_misc",
    }
    for d in themes.values():
        d.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {k: 0 for k in themes}

    # SITE redline + 1 km analysis buffer (aligned with rebuild_site_clipped_5km_from_deprecated.py)
    g_site = gpd.GeoDataFrame({"name": ["site_redline"]}, geometry=[site_poly], crs="EPSG:4326")
    write_geojson(g_site, themes["00_boundary"] / "SITE.redline.geojson")
    g_buf = load_clip_mask(site_path, _BUFFER_STUDY_M)
    write_geojson(g_buf, themes["00_boundary"] / "SITE.buffer_1km.geojson")
    counts["00_boundary"] = 2

    shape_root = backup / "shapefile"
    geo_root = backup / "geojson"
    csv_root = backup / "csv"

    # --- Packaged GeoJSON from first clip ------------------------------------------------
    mapping_packaged = {
        "AOI.json": "05_aoi",
        "BUIDING.json": "01_buildings",
        "RAIL.json": "03_rail",
        "ROAD.json": "02_roads",
    }
    for fname, theme in mapping_packaged.items():
        p = geo_root / fname
        if p.exists():
            dest = themes[theme] / fname.replace(".json", ".geojson")
            shutil.copy2(p, dest)
            counts[theme] += 1

    # --- Shapefiles ---------------------------------------------------------------------
    seq: dict[str, int] = {k: 0 for k in themes}
    if shape_root.exists():
        for shp in sorted(shape_root.rglob("*.shp")):
            if shp.name.endswith(".shp.xml"):
                continue
            rel = shp.relative_to(shape_root)
            theme = classify_shapefile(rel)
            seq[theme] += 1
            out_name = f"{seq[theme]:04d}_{safe_name(rel.stem)}.geojson"
            dest = themes[theme] / out_name
            gdf = None
            for enc in ("utf-8", "gb18030", "gbk", "latin1"):
                try:
                    gdf = gpd.read_file(shp, encoding=enc)
                    break
                except Exception:
                    continue
            if gdf is None:
                continue
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            gdf = gdf.to_crs(4326)
            gdf["__source_shp__"] = str(rel).replace("\\", "/")
            write_geojson(gdf, dest)
            counts[theme] += 1

        # Vector layers saved as GeoJSON when Shapefile write failed (mixed geometry)
        for gj in sorted(shape_root.rglob("*.geojson")):
            rel = gj.relative_to(shape_root)
            theme = classify_shapefile(rel.with_suffix(".shp"))
            seq[theme] += 1
            out_name = f"{seq[theme]:04d}_{safe_name(rel.with_suffix('').stem)}.geojson"
            dest = themes[theme] / out_name
            try:
                gdf = gpd.read_file(gj)
            except Exception:
                continue
            if gdf is None or gdf.empty:
                continue
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            gdf = gdf.to_crs(4326)
            gdf["__source_shp__"] = str(rel).replace("\\", "/")
            write_geojson(gdf, dest)
            counts[theme] += 1

    # --- CSV -> GeoJSON -----------------------------------------------------------------
    if csv_root.exists():
        for csv_path in sorted(csv_root.rglob("*.csv")):
            rel = csv_path.relative_to(csv_root)
            theme = classify_csv(rel)
            seq[theme] += 1
            dest = themes[theme] / f"{seq[theme]:04d}_{safe_name(rel.stem)}.geojson"
            try:
                gdf = csv_to_gdf(csv_path)
            except Exception:
                continue
            if gdf is None or len(gdf) == 0:
                continue
            gdf["__source_csv__"] = str(rel).replace("\\", "/")
            write_geojson(gdf, dest)
            counts[theme] += 1

    # --- WorldPop: deprecated archives + optional loose folder under data --------------
    dem_notes: list[str] = []
    search_roots = [data]
    dep = data / "deprecated"
    if dep.exists():
        for archive in sorted(dep.glob("pre_site_clip_*")):
            search_roots.append(archive)
    wp_dirs = find_worldpop_dirs(search_roots)
    explicit = data / "deprecated" / "pre_site_clip_20260429_005231" / "15-Worldpop人口密度"
    if explicit.is_dir() and explicit not in wp_dirs:
        wp_dirs.append(explicit)
    wp_dirs = sorted(set(wp_dirs))
    if wp_dirs and rasterio:
        dem_notes = clip_rasters_to_site(wp_dirs, study_poly, themes["13_demography"], out_root)
    elif wp_dirs:
        dem_notes = ["rasterio missing; copying WorldPop .tif without clip"]
        for d in wp_dirs:
            for tif in sorted(d.glob("*.tif")) + sorted(d.glob("*.TIF")):
                shutil.copy2(tif, themes["13_demography"] / tif.name)

    n_dem = len(list(themes["13_demography"].glob("*.tif"))) + len(list(themes["13_demography"].glob("*.TIF")))
    if n_dem:
        counts["13_demography"] = n_dem

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "site_boundary": "SITE.json + 00_boundary/SITE.redline.geojson + SITE.buffer_1km.geojson (1 km study area)",
        "backup_layout": str(backup.relative_to(data)),
        "feature_counts_by_theme": counts,
        "demography_notes": dem_notes,
        "raster_policy": "WorldPop kept as GeoTIFF under 13_demography (lossless vs pixel polygonization).",
    }
    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Wrote:", out_root)
    print("Backup:", backup)
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
