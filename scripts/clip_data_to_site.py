"""
Clip vector + table data under data/ to SITE.json polygon, write to data/site_clipped/,
then move original large assets to data/deprecated/<timestamp>/.

Run from repo root:
  python scripts/clip_data_to_site.py

Requires: geopandas, pandas, shapely, pyproj, pyogrio (recommended)
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point, Polygon, mapping, shape
from shapely.prepared import prep

# --- GCJ-02 <-> WGS84 (Mars) for CSV rows stored in GCJ ---------------------------------

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


def load_site_polygon(site_path: Path) -> gpd.GeoDataFrame:
    with site_path.open(encoding="utf-8") as f:
        fc = json.load(f)
    geom = fc["features"][0]["geometry"]
    g = shape(geom)
    if isinstance(g, LineString):
        coords = list(g.coords)
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        poly = Polygon(coords)
    elif isinstance(g, Polygon):
        poly = g
    else:
        raise TypeError(f"Unsupported SITE geometry: {type(g)}")
    if not poly.is_valid:
        poly = poly.buffer(0)
    return gpd.GeoDataFrame({"name": ["site_boundary"]}, geometry=[poly], crs="EPSG:4326")


def load_clip_mask(site_path: Path, buffer_m: float | None = None) -> gpd.GeoDataFrame:
    """SITE polygon in WGS84; optional metric buffer (meters) for study extent (uses EPSG:32651)."""
    base = load_site_polygon(site_path)
    if buffer_m is None or buffer_m <= 0:
        return base
    g = base.to_crs("EPSG:32651")
    buf_geom = g.geometry.iloc[0].buffer(float(buffer_m))
    if not buf_geom.is_valid:
        buf_geom = buf_geom.buffer(0)
    out = gpd.GeoDataFrame({"name": ["site_clip_buffer"]}, geometry=[buf_geom], crs="EPSG:32651")
    return out.to_crs("EPSG:4326")


def clip_gdf(gdf: gpd.GeoDataFrame, mask_wgs84: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    mask_use = mask_wgs84.to_crs(gdf.crs)
    try:
        bb = mask_use.total_bounds
        sub = gdf.cx[bb[0] : bb[2], bb[1] : bb[3]]
    except Exception:
        sub = gdf
    if sub.empty:
        return gdf.iloc[0:0].copy()
    clipped = gpd.clip(sub, mask_use)
    return clipped


def write_geojson(fc: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, separators=(",", ":"))


def clip_geojson_file(src: Path, mask: gpd.GeoDataFrame, out: Path) -> tuple[int, int]:
    gdf = gpd.read_file(src)
    n0 = len(gdf)
    clipped = clip_gdf(gdf, mask)
    n1 = len(clipped)
    fc = json.loads(clipped.to_json())
    if "name" not in fc:
        fc["name"] = src.stem
    fc["crs"] = {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}
    write_geojson(fc, out)
    return n0, n1


def iter_shapefiles(root: Path) -> list[Path]:
    out = []
    for p in root.rglob("*.shp"):
        rel = p.relative_to(root)
        if "site_clipped" in rel.parts or "deprecated" in rel.parts:
            continue
        if p.name.endswith(".shp.xml"):
            continue
        out.append(p)
    return sorted(out)


def clip_csv_points(
    src: Path,
    out: Path,
    prepared_wgs: prep,
    poly_wgs: Polygon,
    lon_col: str,
    lat_col: str,
    coord_mode: str,
    chunksize: int = 100_000,
) -> tuple[int, int]:
    """coord_mode: 'wgs84' or 'gcj02'."""
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    kept = 0
    first = True
    reader = pd.read_csv(src, chunksize=chunksize, encoding="utf-8", encoding_errors="replace", low_memory=False)
    for chunk in reader:
        total += len(chunk)
        if lon_col not in chunk.columns or lat_col not in chunk.columns:
            raise KeyError(f"{src}: missing {lon_col}/{lat_col}")
        sub = chunk.dropna(subset=[lon_col, lat_col])
        lngs = sub[lon_col].astype(float)
        lats = sub[lat_col].astype(float)
        if coord_mode == "gcj02":
            wx, wy = zip(*(gcj02_to_wgs84(lng, lat) for lng, lat in zip(lngs, lats)))
            pts = gpd.GeoSeries.from_xy(wx, wy, crs="EPSG:4326")
        else:
            pts = gpd.GeoSeries.from_xy(lngs, lats, crs="EPSG:4326")
        inside = pts.within(poly_wgs)
        out_chunk = sub.loc[inside.values].copy()
        kept += len(out_chunk)
        out_chunk.to_csv(out, mode="w" if first else "a", header=first, index=False, encoding="utf-8-sig")
        first = False
    if first:
        pd.DataFrame().to_csv(out, index=False)
    return total, kept


def detect_csv_spec(path: Path) -> tuple[str, str, str] | None:
    """Return (lon_col, lat_col, coord_mode) or None if not a point CSV we handle."""
    name = path.name.lower()
    try:
        head = pd.read_csv(path, nrows=2, encoding="utf-8", encoding_errors="replace")
    except Exception:
        return None
    cols = {c.lower(): c for c in head.columns}

    if "wgs84lng" in cols and "wgs84lat" in cols:
        return cols["wgs84lng"], cols["wgs84lat"], "wgs84"
    if "wgs84_lon" in cols and "wgs84_lat" in cols:
        return cols["wgs84_lon"], cols["wgs84_lat"], "wgs84"
    if "gcj02_lng" in cols and "gcj02_lat" in cols:
        return cols["gcj02_lng"], cols["gcj02_lat"], "gcj02"
    if name.endswith("poidata-2025-上海市.csv") or "poidata" in name:
        if "wgs84lng" in head.columns:
            return "wgs84Lng", "wgs84Lat", "wgs84"
    return None


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    data_root = repo / "data"
    site_path = data_root / "SITE.json"
    if not site_path.exists():
        print("SITE.json not found:", site_path, file=sys.stderr)
        return 1

    mask = load_site_polygon(site_path)
    poly_wgs = mask.geometry.iloc[0]
    prepared = prep(poly_wgs)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = data_root / "site_clipped"
    arch_root = data_root / "deprecated" / f"pre_site_clip_{ts}"

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    # --- GeoJSON (project exports) -------------------------------------------------
    geo_out = out_root / "geojson"
    geo_out.mkdir(parents=True, exist_ok=True)
    for name in ("AOI.json", "BUIDING.json", "RAIL.json", "ROAD.json"):
        src = data_root / name
        if not src.exists():
            continue
        print(f"[geojson] {name} …")
        n0, n1 = clip_geojson_file(src, mask, geo_out / name)
        print(f"    features {n0} -> {n1}")

    shutil.copy2(site_path, out_root / "SITE.json")

    # --- Shapefiles ----------------------------------------------------------------
    shp_root_out = out_root / "shapefile"
    shp_root_out.mkdir(parents=True, exist_ok=True)
    shp_files = iter_shapefiles(data_root)
    print(f"[shapefile] {len(shp_files)} layers …")
    ok_shp = 0
    for shp in shp_files:
        rel = shp.relative_to(data_root)
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
        clipped.to_file(dest, driver="ESRI Shapefile", encoding="utf-8")
        ok_shp += 1
        if ok_shp <= 5 or n1 < 500:
            print(f"    {rel}: {n0} -> {n1}")
    print(f"[shapefile] wrote {ok_shp} non-empty layers")

    # --- CSV point tables -----------------------------------------------------------
    csv_out = out_root / "csv"
    csv_out.mkdir(parents=True, exist_ok=True)
    for csv_path in sorted(data_root.rglob("*.csv")):
        rel = csv_path.relative_to(data_root)
        if "site_clipped" in rel.parts or "deprecated" in rel.parts:
            continue
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

    # --- Archive originals ----------------------------------------------------------
    arch_root.mkdir(parents=True, exist_ok=True)
    skip = {"site_clipped", "deprecated", "SITE.json"}
    moves: list[tuple[Path, Path]] = []
    for p in sorted(data_root.iterdir()):
        if p.name in skip:
            continue
        if p.is_file() and p.suffix.lower() == ".json":
            if p.name == "SITE.json":
                continue
            moves.append((p, arch_root / p.name))
        elif p.is_dir():
            if p.name in skip:
                continue
            moves.append((p, arch_root / p.name))

    print(f"[archive] moving {len(moves)} items -> {arch_root}")
    for src, dst in moves:
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(src), str(dst))

    print("Done. Clipped data:", out_root)
    print("Archived:", arch_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
