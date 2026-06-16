"""
Write synthetic Baidu-style source CSVs (field names aligned with platform manual screenshots)
for testing build_sshmm_observations.py without a gdb export.

Uses real unit centroids from 01_units.gpkg so points fall inside polygons (spatial join works).

Run from repo root:
  python scripts/write_sshmm_synthetic_source_csvs.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError as e:
    raise SystemExit("pip install geopandas") from e

REPO = Path(__file__).resolve().parents[1]
DEFAULT_UNITS = REPO / "data" / "site_3km" / "01_units.gpkg"
OUT_DIR = REPO / "data" / "sshmm" / "sample_inputs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--n-units", type=int, default=5, help="How many parcel centroids to use as grid centers.")
    ap.add_argument("--hours", type=int, default=48, help="Consecutive hourly rows starting at --start")
    ap.add_argument("--start", default="2021-04-01 06:00:00")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    g = gpd.read_file(args.units)
    if g.crs is None:
        g.set_crs("EPSG:4326", inplace=True)
    elif g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    g["cx"] = g.geometry.representative_point().x
    g["cy"] = g.geometry.representative_point().y
    samp = g.head(args.n_units)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    p_flow = args.out_dir / "synthetic_point_flow_hour.csv"
    p_od_o = args.out_dir / "synthetic_point_OD_hour_O.csv"
    p_od_d = args.out_dir / "synthetic_point_OD_hour_D.csv"
    p_poi = args.out_dir / "synthetic_poi_static_per_unit.csv"
    t0 = datetime.fromisoformat(args.start.replace("Z", "+00:00"))

    # --- point_flow_hour style (screenshot: time, grid, x, y, count) ---
    rows_f = []
    for hi in range(args.hours):
        ts = t0 + timedelta(hours=hi)
        dow_amp = 1.15 if ts.weekday() < 5 else 0.9
        peak = np.exp(-0.5 * ((ts.hour - 8.5) / 2.2) ** 2) + np.exp(
            -0.5 * ((ts.hour - 18.0) / 2.5) ** 2
        )
        for i, r in samp.iterrows():
            gid = f"hm_{i}"
            base = 800 * dow_amp * (0.4 + peak * 1.2)
            cnt = max(10.0, float(rng.normal(base, base * 0.12)))
            rows_f.append(
                {
                    "时间": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "网格ID": gid,
                    "x": round(float(r["cx"]), 6),
                    "y": round(float(r["cy"]), 6),
                    "人数": round(cnt, 2),
                }
            )
    df_f = pd.DataFrame(rows_f)
    df_f.to_csv(p_flow, index=False, encoding="utf-8-sig")

    # --- OD origin / dest (screenshot semantics): same locations, split volume ---
    rows_o: list[dict] = []
    rows_d: list[dict] = []
    for hi in range(args.hours):
        ts = t0 + timedelta(hours=hi)
        for i, r in samp.iterrows():
            lon, lat = float(r["cx"]), float(r["cy"])
            base = 50 * (0.5 + rng.random())
            half = base * rng.uniform(0.35, 0.65)
            rows_o.append(
                {
                    "时间": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "ox": round(lon + 0.0001, 6),
                    "oy": round(lat + 0.0001, 6),
                    "数量": round(half, 2),
                }
            )
            rows_d.append(
                {
                    "时间": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "dx": round(lon - 0.00005, 6),
                    "dy": round(lat - 0.00005, 6),
                    "数量": round(base - half, 2),
                }
            )
    pd.DataFrame(rows_o).to_csv(p_od_o, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_d).to_csv(p_od_d, index=False, encoding="utf-8-sig")

    # --- static POI mix per unit_id (9 cols, paper-like) ---
    cats = ["商业", "办公", "交通", "居住", "公服", "教育", "医疗", "文体", "其它"]
    pr = []
    for _, r in samp.iterrows():
        w = rng.integers(1, 40, size=len(cats))
        row = {"unit_id": r["unit_id"]}
        for c, v in zip(cats, w, strict=False):
            row[c] = int(v)
        pr.append(row)
    pd.DataFrame(pr).to_csv(p_poi, index=False, encoding="utf-8-sig")

    print("Wrote:")
    print(f"  {p_flow}")
    print(f"  {p_od_o}")
    print(f"  {p_od_d}")
    print(f"  {p_poi}")
    print("\nThen run:")
    print(
        f'  python scripts/build_sshmm_observations.py --units "{args.units}" '
        f'--flow-hour "{p_flow}" '
        f'--od-o "{p_od_o}" '
        f'--od-d "{p_od_d}" '
        f'--poi-static "{p_poi}" '
        f'--out-dir data/sshmm/out_synthetic_from_csv'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
