"""
Subset MetroFlow (Scientific Data / Figshare) to the Shanghai Railway Station
study buffer for use with site_3km / CRS84 workflows.

Reads:
  data/all/MetroFlow/MetroFlow/stationInfo.csv
  data/all/MetroFlow/MetroFlow/metroData_InOutFlow.csv
  data/all/MetroFlow/MetroFlow/metroData_ODFlow.csv  (optional; large)

Writes under data/site_3km/metroflow/:
  stations_3km.geojson          — WGS84 points; stationID joins flow CSVs; station_idx is table row id
  inout_10min_3km.parquet       — 10-min in/out rows for stations in buffer
  od_internal_10min_3km.parquet — OD rows where BOTH ends in buffer (needs duckdb)
  manifest_metroflow.json       — paths, counts, CRS, column dictionary

Note: metroData_InOutFlow `station` and OD `originstation`/`destinationstation` use **stationID**
from stationInfo (same as column stationID), not the unnamed first column index.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
METRO = REPO / "data" / "all" / "MetroFlow" / "MetroFlow"
OUT = REPO / "data" / "site_3km" / "metroflow"


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def load_station_table(path: Path):
    import pandas as pd

    df = pd.read_csv(path)
    first = df.columns[0]
    df = df.rename(columns={first: "station_idx"})
    return df


def stations_in_buffer(df, lon0: float, lat0: float, buffer_m: float):
    dist = df.apply(lambda r: haversine_m(lon0, lat0, float(r["lon"]), float(r["lat"])), axis=1)
    sub = df.assign(dist_m=dist).loc[dist <= buffer_m].copy()
    return sub.sort_values("dist_m")


def write_stations_geojson(sub, path: Path) -> None:
    feats = []
    for _, r in sub.iterrows():
        feats.append(
            {
                "type": "Feature",
                "properties": {
                    "station_idx": int(r["station_idx"]),
                    "stationID": int(r["stationID"]),
                    "name": str(r["name"]),
                    "dist_m": round(float(r["dist_m"]), 2),
                },
                "geometry": {"type": "Point", "coordinates": [float(r["lon"]), float(r["lat"])]},
            }
        )
    fc = {"type": "FeatureCollection", "name": "metroflow_stations_3km", "features": feats}
    path.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")


def subset_inout_duckdb(in_path: Path, out_path: Path, idx_tuple: tuple[int, ...]) -> int:
    import duckdb

    ids = ",".join(str(i) for i in idx_tuple)
    con = duckdb.connect(database=":memory:")
    con.execute(
        f"""
        COPY (
          SELECT *
          FROM read_csv_auto('{in_path.as_posix()}', normalize_names=true)
          WHERE station IN ({ids})
        ) TO '{out_path.as_posix()}' (FORMAT PARQUET);
        """
    )
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
    con.close()
    return int(n)


def subset_od_internal_duckdb(in_path: Path, out_path: Path, idx_tuple: tuple[int, ...]) -> int:
    import duckdb

    ids = ",".join(str(i) for i in idx_tuple)
    con = duckdb.connect(database=":memory:")
    con.execute(
        f"""
        COPY (
          SELECT *
          FROM read_csv_auto('{in_path.as_posix()}', normalize_names=true)
          WHERE originstation IN ({ids}) AND destinationstation IN ({ids})
        ) TO '{out_path.as_posix()}' (FORMAT PARQUET);
        """
    )
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
    con.close()
    return int(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer-m", type=float, default=3000.0)
    ap.add_argument("--center-lon", type=float, default=121.451257271)
    ap.add_argument("--center-lat", type=float, default=31.249149419)
    ap.add_argument("--skip-od", action="store_true", help="Skip 12GB OD scan (faster).")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    st_path = METRO / "stationInfo.csv"
    inout_path = METRO / "metroData_InOutFlow.csv"
    od_path = METRO / "metroData_ODFlow.csv"

    df = load_station_table(st_path)
    sub = stations_in_buffer(df, args.center_lon, args.center_lat, args.buffer_m)
    # Flow CSVs use stationID in column "station" / OD legs (not stationInfo row index).
    station_ids = tuple(int(x) for x in sorted(sub["stationID"].unique()))

    write_stations_geojson(sub, OUT / "stations_3km.geojson")

    try:
        import duckdb  # noqa: F401
    except ImportError:
        raise SystemExit("duckdb is required. Install: pip install duckdb")

    inout_out = OUT / "inout_10min_3km.parquet"
    if inout_out.exists():
        inout_out.unlink()
    n_inout = subset_inout_duckdb(inout_path, inout_out, station_ids)

    n_od = 0
    od_out = OUT / "od_internal_10min_3km.parquet"
    if not args.skip_od:
        if od_out.exists():
            od_out.unlink()
        if od_path.exists():
            n_od = subset_od_internal_duckdb(od_path, od_out, station_ids)
        else:
            od_out = None
    else:
        od_out = None

    manifest = {
        "source": str(METRO).replace("\\", "/"),
        "crs": "EPSG:4326 (lon/lat same as stationInfo; compatible with SITE.json CRS84)",
        "center_wgs84": [args.center_lon, args.center_lat],
        "buffer_m": args.buffer_m,
        "station_id_in_flow_csv": (
            "metroData_InOutFlow `station` and metroData_ODFlow origin/destination use **stationID** "
            "(column stationInfo.stationID), not the leading row index."
        ),
        "stations_in_buffer": len(station_ids),
        "outputs": {
            "stations_3km_geojson": str((OUT / "stations_3km.geojson").as_posix()),
            "inout_10min_3km_parquet": str(inout_out.as_posix()),
            "inout_row_count": n_inout,
            "od_internal_10min_3km_parquet": str(od_out.as_posix()) if od_out else None,
            "od_internal_row_count": n_od if n_od else None,
        },
        "inout_columns_parquet": [
            "date",
            "timeslot",
            "starttime",
            "endtime",
            "station",
            "inflow",
            "outflow",
            "cinflow",
            "hboinflow",
            "nhbinflow",
            "coutflow",
            "hbooutflow",
            "nhboutflow",
        ],
        "od_columns_parquet": [
            "date",
            "timeslot",
            "starttime",
            "endtime",
            "originstation",
            "destinationstation",
            "flow",
            "cflow",
            "hboflow",
            "nhbflow",
        ],
        "time_semantics": {
            "period": "2017-05-01 to 2017-08-31 (dataset scope)",
            "timeslot": "10-minute bins; startTime/endTime like 060000,061000",
            "workday_calendar": str((METRO / "MetaData" / "workday_calendar.csv").as_posix()),
            "weather_hourly": str((METRO / "MetaData" / "shanghai_weatherHourly.csv").as_posix()),
        },
        "merge_with_site_3km": (
            "Spatial join: buffer stations to units via unit polygons; "
            "aggregate inFlow/outFlow by unit_id and your t_id windows (map timeslot hours to slices). "
            "2017 network vs 2025 POI: use for diurnal SHAPE calibration, not level matching."
        ),
    }
    (OUT / "manifest_metroflow.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # copy small metadata for offline use
    for rel in ("MetaData/workday_calendar.csv", "MetaData/shanghai_weatherHourly.csv"):
        src = METRO / rel
        if src.exists():
            dst = OUT / rel.replace("/", "_")
            shutil.copy2(src, dst)

    print("stations", len(station_ids), "inout rows", n_inout, "od rows", n_od)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
