"""
将 ``大众点评数据.csv``（含 wgs84_lon / wgs84_lat）转为点 GeoJSON，便于 QGIS 查看或与 3km 裁剪脚本联用。

默认输入：仓库上一级 ``all/大众点评数据.csv``；输出默认写入 ``data/site_3km/02-POI&AOI/大众点评/``。

  python scripts/dianping_csv_to_geojson.py
  python scripts/dianping_csv_to_geojson.py --csv "F:/path/大众点评数据.csv" --out data/site_3km/02-POI&AOI/大众点评/dianping_points.geojson
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def read_dianping_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "gb18030", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise SystemExit(f"无法解码 CSV: {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="大众点评 CSV → GeoJSON 点层")
    default_csv = REPO.parent / "all" / "大众点评数据.csv"
    default_out = REPO / "data" / "site_3km" / "02-POI&AOI" / "大众点评" / "dianping_points_wgs84.geojson"
    ap.add_argument("--csv", type=Path, default=default_csv, help="大众点评门店表路径")
    ap.add_argument("--out", type=Path, default=default_out, help="输出 GeoJSON")
    args = ap.parse_args()
    src: Path = args.csv.resolve()
    if not src.is_file():
        print("找不到输入文件:", src)
        return 1
    df = read_dianping_csv(src)
    lon_c = "wgs84_lon" if "wgs84_lon" in df.columns else "lon"
    lat_c = "wgs84_lat" if "wgs84_lat" in df.columns else "lat"
    if lon_c not in df.columns or lat_c not in df.columns:
        print("缺少经纬度列（需要 wgs84_lon/wgs84_lat 或 lon/lat）")
        return 1
    sub = df.copy()
    sub["_x"] = pd.to_numeric(sub[lon_c], errors="coerce")
    sub["_y"] = pd.to_numeric(sub[lat_c], errors="coerce")
    sub = sub.dropna(subset=["_x", "_y"])
    geom = gpd.GeoSeries.from_xy(sub["_x"].to_numpy(), sub["_y"].to_numpy(), crs="EPSG:4326")
    g = gpd.GeoDataFrame(sub.drop(columns=["_x", "_y"], errors="ignore"), geometry=geom, crs="EPSG:4326")
    out: Path = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fc = json.loads(g.to_json())
    fc.setdefault("name", "dianping_points_wgs84")
    fc["crs"] = {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}
    out.write_text(json.dumps(fc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {out}  features={len(g)}  from {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
