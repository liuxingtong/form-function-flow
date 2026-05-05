"""
Export selected folders under data/deprecated/ to GeoJSON clipped to a 3 km
buffer around data/SITE.json (metric buffer in EPSG:32651, same as
clip_data_to_site.load_clip_mask).

Mirrors each source file's path under data/deprecated/<subdir>/... into
data/site_3km/<subdir>/... and uses the original basename with a .geojson
suffix (e.g. 上海市_铁路线.shp -> 上海市_铁路线.geojson).

Vectors: .shp, .geojson, vector .json (GeoJSON layers)
Point CSV: columns handled by detect_csv_spec (clip_data_to_site) and csv_to_gdf
(reorganize_site_clipped_by_theme)
Rasters (.tif): clipped to buffer, sampled to Point features (value + lon/lat),
capped for size (continuous grids are not polygonized).

Run from repo root:
  python scripts/export_deprecated_to_site_3km_geojson.py

Requires: geopandas, pandas, shapely, pyproj; rasterio optional (for .tif).
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from clip_data_to_site import clip_gdf, detect_csv_spec, load_clip_mask
from reorganize_site_clipped_by_theme import csv_to_gdf

try:
    import rasterio
    from rasterio.mask import mask as rio_mask
    from rasterio.warp import transform as rio_transform
except ImportError:
    rasterio = None
    rio_mask = None
    rio_transform = None

BUFFER_M = 3000.0

# Top-level folders under data/deprecated/ to scan (must match repo layout).
DEPRECATED_SUBDIRS: tuple[str, ...] = (
    "01-建筑轮廓",
    "02-POI&AOI",
    "03-行政区",
    "04-交通数据",
    "05-DEM高程数据",
    "08-房价数据",
    "09-开源土地利用",
    "10-AI解译数据",
    "11-蓝绿空间",
    "13-常用公服设施点",
    "14-城市建成区",
    "15-Worldpop人口密度",
    "16-城市环路",
    "17-主城范围",
    "19-其它",
    "上海地块",
    "上海道路",
    "内环边界",
    "楼盘",
    "美团",
    "风貌保护区范围",
)

VECTOR_SUFFIX = {".shp", ".geojson"}
JSON_VECTOR_NAMES = {".json"}  # try GeoJSON read
RASTER_SUFFIX = {".tif", ".tiff"}
CSV_SUFFIX = {".csv"}


def write_geojson_fc(gdf: gpd.GeoDataFrame, out_path: Path, stem_name: str, source_rel: str) -> None:
    gdf2 = gdf.copy()
    gdf2["__source_file__"] = source_rel.replace("\\", "/")
    for col in gdf2.columns:
        if col == "geometry":
            continue
        if pd.api.types.is_datetime64_any_dtype(gdf2[col]):
            gdf2[col] = gdf2[col].astype(str)
    fc = json.loads(gdf2.to_json())
    fc["name"] = stem_name
    fc["crs"] = {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, separators=(",", ":"))


def read_shapefile(path: Path) -> gpd.GeoDataFrame | None:
    last_err: Exception | None = None
    for enc in ("utf-8", "gb18030", "gbk", "latin1"):
        try:
            return gpd.read_file(path, encoding=enc)
        except Exception as e:
            last_err = e
    print(f"    SKIP shp read {path.name}: {last_err}", file=sys.stderr)
    return None


def try_read_geojson_like(path: Path) -> gpd.GeoDataFrame | None:
    try:
        return gpd.read_file(path)
    except Exception:
        return None


def csv_points_to_gdf(path: Path) -> gpd.GeoDataFrame | None:
    spec = detect_csv_spec(path)
    if spec is not None:
        lon_c, lat_c, mode = spec
        from clip_data_to_site import gcj02_to_wgs84

        df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace", low_memory=False)
        if lon_c not in df.columns or lat_c not in df.columns:
            return None
        sub = df.dropna(subset=[lon_c, lat_c])
        if sub.empty:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        lngv = sub[lon_c].astype(float)
        latv = sub[lat_c].astype(float)
        if mode == "gcj02":
            wx, wy = zip(*(gcj02_to_wgs84(a, b) for a, b in zip(lngv, latv)))
            geom = gpd.points_from_xy(wx, wy, crs="EPSG:4326")
        else:
            geom = gpd.points_from_xy(lngv, latv, crs="EPSG:4326")
        return gpd.GeoDataFrame(sub, geometry=geom, crs="EPSG:4326")
    return csv_to_gdf(path)


def raster_to_points_geojson(
    tif_path: Path,
    mask_wgs84: gpd.GeoDataFrame,
    out_path: Path,
    stem_name: str,
    source_rel: str,
    max_points: int = 50_000,
) -> str | None:
    if rasterio is None or rio_mask is None:
        return "rasterio not installed"
    try:
        with rasterio.open(tif_path) as src:
            mask_local = mask_wgs84.to_crs(src.crs)
            geom = mapping(mask_local.geometry.iloc[0])
            data, transform = rio_mask(src, [geom], crop=True, all_touched=False)
            band = data[0]
            nodata = src.nodata
            h, w = int(band.shape[0]), int(band.shape[1])
            if h == 0 or w == 0:
                return "empty raster window"
            ncell = h * w
            stride = max(1, int(math.ceil(math.sqrt(max(1, ncell / max_points)))))
            xs: list[float] = []
            ys: list[float] = []
            vals: list[float] = []
            m = np.ma.masked_invalid(band)
            if nodata is not None:
                m = np.ma.masked_where(band == nodata, m, copy=False)
            for row in range(0, h, stride):
                for col in range(0, w, stride):
                    v = m[row, col]
                    if np.ma.is_masked(v):
                        continue
                    x, y = rasterio.transform.xy(transform, row + 0.5, col + 0.5, offset="center")
                    xs.append(float(x))
                    ys.append(float(y))
                    vals.append(float(np.asarray(v).item()))
            if not xs:
                return "no valid pixels in clip"
            if src.crs:
                lons, lats = rio_transform(src.crs, "EPSG:4326", xs, ys)
            else:
                lons, lats = xs, ys
            gdf = gpd.GeoDataFrame(
                {"z": vals},
                geometry=gpd.points_from_xy(lons, lats, crs="EPSG:4326"),
                crs="EPSG:4326",
            )
            write_geojson_fc(gdf, out_path, stem_name, source_rel)
            return None
    except Exception as e:
        return str(e)


def process_file(src: Path, deprecated: Path, site_3km: Path, mask: gpd.GeoDataFrame) -> tuple[str, str]:
    """Return (status, detail) where status in ok, skip, fail."""
    rel = src.relative_to(deprecated)
    rel_posix = rel.as_posix()
    out_path = site_3km / rel.parent / f"{src.stem}.geojson"
    suffix = src.suffix.lower()

    if suffix in VECTOR_SUFFIX or suffix in JSON_VECTOR_NAMES:
        if suffix == ".shp":
            gdf = read_shapefile(src)
        else:
            gdf = try_read_geojson_like(src)
        if gdf is None:
            return "skip", "unreadable vector"
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        n0 = len(gdf)
        clipped = clip_gdf(gdf, mask)
        if clipped.empty:
            return "skip", f"empty clip ({n0} features)"
        write_geojson_fc(clipped, out_path, src.stem, rel_posix)
        return "ok", f"{n0}->{len(clipped)}"

    if suffix in CSV_SUFFIX:
        try:
            gdf = csv_points_to_gdf(src)
        except Exception as e:
            return "fail", f"csv: {e}"
        if gdf is None:
            return "skip", "not a lon/lat CSV"
        if gdf.empty:
            return "skip", "empty csv coords"
        n0 = len(gdf)
        clipped = clip_gdf(gdf, mask)
        if clipped.empty:
            return "skip", f"empty clip ({n0} points)"
        write_geojson_fc(clipped, out_path, src.stem, rel_posix)
        return "ok", f"{n0}->{len(clipped)}"

    if suffix in RASTER_SUFFIX:
        err = raster_to_points_geojson(src, mask, out_path, src.stem, rel_posix)
        if err:
            return "skip" if "not installed" in err or "empty" in err else "fail", err
        return "ok", "raster->points"

    return "skip", f"unsupported type {suffix}"


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    data = repo / "data"
    site_path = data / "SITE.json"
    deprecated = data / "deprecated"
    site_3km = data / "site_3km"

    if not site_path.exists():
        print("SITE.json not found:", site_path, file=sys.stderr)
        return 1
    if not deprecated.is_dir():
        print("Missing:", deprecated, file=sys.stderr)
        return 1

    mask = load_clip_mask(site_path, BUFFER_M)
    if site_3km.exists():
        shutil.rmtree(site_3km)
    site_3km.mkdir(parents=True)
    shutil.copy2(site_path, site_3km / "SITE.json")

    buf_fc = json.loads(mask.to_json())
    buf_fc.setdefault("name", "SITE_buffer_3km")
    buf_fc["crs"] = {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}
    with (site_3km / "SITE.buffer_3km.geojson").open("w", encoding="utf-8") as f:
        json.dump(buf_fc, f, ensure_ascii=False, separators=(",", ":"))

    manifest: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "buffer_m": BUFFER_M,
        "site": "data/SITE.json",
        "sources_roots": list(DEPRECATED_SUBDIRS),
        "results": {"ok": [], "skip": [], "fail": []},
    }

    for sub in DEPRECATED_SUBDIRS:
        root = deprecated / sub
        if not root.is_dir():
            manifest["results"]["skip"].append({"path": sub, "reason": "folder missing"})
            print("missing folder:", sub)
            continue
        for src in sorted(root.rglob("*")):
            if not src.is_file():
                continue
            suf = src.suffix.lower()
            if suf not in VECTOR_SUFFIX | JSON_VECTOR_NAMES | RASTER_SUFFIX | CSV_SUFFIX:
                continue
            status, detail = process_file(src, deprecated, site_3km, mask)
            entry = {"source": src.relative_to(deprecated).as_posix(), "detail": detail}
            manifest["results"][status].append(entry)
            if status == "fail":
                print("FAIL", entry["source"], detail, file=sys.stderr)

    # summary counts
    for k in ("ok", "skip", "fail"):
        manifest["counts"] = manifest.get("counts", {})
        manifest["counts"][k] = len(manifest["results"][k])

    with (site_3km / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Wrote:", site_3km)
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
