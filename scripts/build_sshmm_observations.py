"""
Build paper-aligned mobility behaviour observations (Definition 1) for SSHMM.

Inputs: Baidu LBS–style CSV exports (hourly grid flow, hourly OD with O/D points) +
optional POI category counts per unit. Spatially assigns points to study units
(01_units.gpkg) in WGS84.

Outputs:
  - observation_long.csv: raw o_arrive, o_leave, o_stay, poi_cat_*
  - observation_normalized.csv: + norm_* per paper §4.1 (min–max on flows; TF–IDF
    per time slot on POI matrix, then min–max over time per unit)
  - sshmm_manifest.json

Demo mode (--demo) synthesizes trajectories when you have no LBS export yet.

Run from repo root:
  python scripts/build_sshmm_observations.py --demo --out-dir data/sshmm/out_demo
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

try:
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError as e:
    raise SystemExit("Install geopandas + shapely: pip install geopandas shapely") from e

try:
    from sklearn.feature_extraction.text import TfidfTransformer
except ImportError:
    TfidfTransformer = None  # type: ignore[misc, assignment]


def _utc_naive(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is not None:
        return ts.tz_convert(None).tz_localize(None)
    return ts


def _normalize_ts_series(s: pd.Series) -> pd.Series:
    """Parse mixed time formats → pandas.Timestamp (timezone stripped)."""
    out = pd.to_datetime(s, errors="coerce")
    return pd.Series([_utc_naive(x) if pd.notna(x) else pd.NaT for x in out], index=s.index)


def _pick_col(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols:
            return cols[key]
    # fuzzy: strip spaces
    stripped = {re.sub(r"\s+", "", c.lower()): c for c in df.columns}
    for cand in candidates:
        k = re.sub(r"\s+", "", cand.lower())
        if k in stripped:
            return stripped[k]
    return None


def load_units_gpkg(path: Path) -> gpd.GeoDataFrame:
    g = gpd.read_file(path)
    if g.crs is None:
        g.set_crs("EPSG:4326", inplace=True)
    elif g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    if "unit_id" not in g.columns:
        raise ValueError("units layer must contain column unit_id")
    return g[["unit_id", "geometry"]]


def assign_points_to_units(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    units: gpd.GeoDataFrame,
) -> pd.Series:
    """Return unit_id for each row (NaN if outside all polygons)."""
    geom = [Point(float(x), float(y)) for x, y in zip(df[x_col], df[y_col], strict=False)]
    pts = gpd.GeoDataFrame(df.index.to_series(), geometry=geom, crs="EPSG:4326")
    joined = pts.sjoin(units, predicate="within", how="left")
    # restore row order
    s = joined["unit_id"].reindex(df.index)
    return s


def aggregate_weighted_points(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    count_col: str,
    time_col: str,
    units: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Assign points to units and sum count by (unit_id, timestamp)."""
    d = df.copy()
    d["_ts"] = _normalize_ts_series(d[time_col])
    d = d.dropna(subset=["_ts"])
    d["_uid"] = assign_points_to_units(d, x_col, y_col, units)
    d = d.dropna(subset=["_uid"])
    g = (
        d.groupby(["_uid", "_ts"], as_index=False)[count_col]
        .sum()
        .rename(columns={"_uid": "unit_id", "_ts": "timestamp", count_col: "value"})
    )
    return g


def minmax_by_unit_time(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    value_cols: list[str],
    prefix: str = "norm_",
) -> pd.DataFrame:
    """Per unit, min–max each column over time (paper §4.1 flow part)."""
    out = df.copy()
    for c in value_cols:
        def _mm(g: pd.Series) -> pd.Series:
            lo, hi = float(g.min()), float(g.max())
            if hi <= lo + 1e-12:
                return pd.Series(np.full(len(g), 0.5), index=g.index)
            return (g - lo) / (hi - lo)

        out[prefix + c] = df.groupby(unit_col, sort=False)[c].transform(_mm)
    return out


def tfidf_then_minmax_poi(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    poi_cols: list[str],
    prefix: str = "norm_",
) -> pd.DataFrame:
    """
    For each time slot, TF–IDF across regions on the region × POI count matrix;
    then per POI column, min–max over time within each unit (paper §4.1).
    """
    out = df.copy()
    if not poi_cols:
        return out
    if TfidfTransformer is None:
        raise SystemExit("Install scikit-learn for TF–IDF: pip install scikit-learn")

    idx = df.index
    times = df[time_col].unique()
    tfidf_vals = np.zeros((len(df), len(poi_cols)), dtype=float)

    for t in times:
        mask = df[time_col] == t
        sub = df.loc[mask, poi_cols].astype(float).values
        if sub.size == 0:
            continue
        # rows=regions; smooth zeros
        sub = np.maximum(sub, 0.0)
        tr = TfidfTransformer(norm=None, smooth_idf=True, sublinear_tf=False)
        try:
            z = tr.fit_transform(sub).toarray()
        except ValueError:
            z = sub
        tfidf_vals[np.where(mask)[0], :] = z

    tmp = pd.DataFrame(tfidf_vals, columns=[f"_tfidf_{c}" for c in poi_cols], index=idx)
    merged = pd.concat([df[[unit_col, time_col]], tmp], axis=1)
    for raw, tc in zip(poi_cols, [f"_tfidf_{c}" for c in poi_cols], strict=True):
        out[f"_tfidf_{raw}"] = merged[tc]

    for raw in poi_cols:
        tc = f"_tfidf_{raw}"
        out[prefix + raw] = merged.groupby(unit_col, sort=False)[tc].transform(
            lambda g: (
                (g - g.min()) / (g.max() - g.min() + 1e-12)
                if float(g.max()) > float(g.min()) + 1e-12
                else 0.5
            )
        )
    # drop helper cols
    drop_cols = [c for c in out.columns if c.startswith("_tfidf_")]
    out = out.drop(columns=drop_cols, errors="ignore")
    return out


def build_demo_frame(
    units: gpd.GeoDataFrame,
    *,
    seed: int,
    days: int,
    start: datetime,
    poi_dims: int,
    max_units: int | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Synthetic in/out/stay + PoI proxy columns for pipeline smoke tests."""
    rng = np.random.default_rng(seed)
    uids = units["unit_id"].tolist()
    if max_units is not None:
        uids = uids[:max_units]
    poi_cols = [f"poi_cat_{i + 1}" for i in range(poi_dims)]

    rows: list[dict[str, Any]] = []
    for day in range(days):
        for hour in range(24):
            ts = start + timedelta(days=day, hours=hour)
            # crude weekday vs weekend amplitude
            dow = (start.weekday() + day) % 7
            wd = 1.3 if dow < 5 else 0.85
            amp_peak = np.exp(-0.5 * ((hour - 8.5) / 2.5) ** 2) + np.exp(
                -0.5 * ((hour - 17.5) / 2.5) ** 2
            )

            for ui, uid in enumerate(uids):
                base = (1.0 + 0.02 * ui) * wd * (1.0 + 2.0 * amp_peak)
                noise = rng.lognormal(0.0, 0.15)
                arrive = max(0.0, base * noise * rng.uniform(0.8, 1.2))
                leave = max(0.0, base * noise * rng.uniform(0.75, 1.15))
                present = max(0.0, base * 3.0 * noise * rng.uniform(0.9, 1.1))
                cats = rng.poisson(lam=base * rng.uniform(0.05, 0.4), size=poi_dims).astype(float)
                row: dict[str, Any] = {
                    "unit_id": uid,
                    "timestamp": ts,
                    "hour": hour,
                    "o_arrive": arrive,
                    "o_leave": leave,
                    "o_stay": present,
                }
                for name, val in zip(poi_cols, cats, strict=False):
                    row[name] = val
                rows.append(row)

    df = pd.DataFrame(rows)
    return df, poi_cols


def read_optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def consume_od_points(
    path: Path,
    units: gpd.GeoDataFrame,
    dest: dict[tuple[str, pd.Timestamp], float],
    *,
    role: str,
) -> None:
    """Aggregate OD rows into dest keyed by (unit_id, timestamp); uses endpoint coords."""
    d0 = pd.read_csv(path)
    tc = _pick_col(d0, ("time", "timestamp", "时间"))
    cc = _pick_col(d0, ("count", "人数", "数量"))
    if role == "origin":
        xc = _pick_col(
            d0,
            ("ox", "o_x", "origin_x", "jobx", "housingx", "起点经度", "起点_x"),
        )
        yc = _pick_col(
            d0,
            ("oy", "o_y", "origin_y", "joby", "housingy", "起点纬度", "起点_y"),
        )
    else:
        xc = _pick_col(
            d0,
            ("dx", "d_x", "dest_x", "jobx", "housingx", "终点经度", "终点_x"),
        )
        yc = _pick_col(
            d0,
            ("dy", "d_y", "dest_y", "joby", "housingy", "终点纬度", "终点_y"),
        )
    if not all([tc, cc, xc, yc]):
        raise SystemExit(
            f"OD CSV {path} missing columns; have {list(d0.columns)}. "
            "Need time, count, endpoint x/y (rename to ox/oy or dx/dy if needed)."
        )
    assert tc is not None and cc is not None and xc is not None and yc is not None
    agg = aggregate_weighted_points(d0, xc, yc, cc, tc, units)
    for _, row in agg.iterrows():
        k = (str(row["unit_id"]), pd.Timestamp(row["timestamp"]))
        dest[k] = dest.get(k, 0.0) + float(row["value"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Build SSHMM observation tables from LBS exports.")
    ap.add_argument(
        "--units",
        type=Path,
        default=REPO / "data" / "site_3km" / "01_units.gpkg",
        help="Study units polygon layer (unit_id + geometry, WGS84).",
    )
    ap.add_argument("--flow-hour", type=Path, help="point_flow_hour CSV")
    ap.add_argument("--od-o", type=Path, help="point_OD_hour_O CSV (origins)")
    ap.add_argument("--od-d", type=Path, help="point_OD_hour_D CSV (destinations)")
    ap.add_argument(
        "--poi-static",
        type=Path,
        help="Static POI counts per unit: unit_id + poi_cat_* columns (replicated hourly).",
    )
    ap.add_argument(
        "--poi-hour",
        type=Path,
        help="Optional hourly POI counts long/wide per unit (advanced).",
    )
    ap.add_argument("--out-dir", type=Path, default=REPO / "data" / "sshmm" / "out")
    ap.add_argument(
        "--stay-mode",
        choices=("flow", "od_balance"),
        default="flow",
        help="flow: use point_flow_hour as o_stay; od_balance: stay=max(0,2*P-in-out) needs present P from flow",
    )
    ap.add_argument("--demo", action="store_true", help="Ignore CSV inputs; synthesize demo series.")
    ap.add_argument("--demo-days", type=int, default=7)
    ap.add_argument("--demo-max-units", type=int, default=40)
    ap.add_argument("--demo-poi-dims", type=int, default=9, help="Semantic dimensions L-3 (paper uses 9).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    units = load_units_gpkg(args.units)
    poi_cols: list[str] = []

    if args.demo:
        start = datetime(2021, 4, 1, 0, 0, 0)
        long_df, poi_cols = build_demo_frame(
            units,
            seed=args.seed,
            days=args.demo_days,
            start=start,
            poi_dims=args.demo_poi_dims,
            max_units=args.demo_max_units,
        )
        manifest: dict[str, Any] = {
            "mode": "demo_synthetic",
            "units_gpkg": str(args.units.resolve()),
            "L": 3 + len(poi_cols),
            "poi_columns": poi_cols,
            "stay_mode": args.stay_mode,
            "time_start": long_df["timestamp"].min().isoformat(),
            "time_end": long_df["timestamp"].max().isoformat(),
            "n_rows": int(len(long_df)),
            "n_units": int(long_df["unit_id"].nunique()),
        }
    else:
        # Initialize empty time grid from first available source
        flows: dict[tuple[str, pd.Timestamp], float] = {}
        outs: dict[tuple[str, pd.Timestamp], float] = {}
        ins: dict[tuple[str, pd.Timestamp], float] = {}

        fh = read_optional_csv(args.flow_hour)

        if fh is not None:
            tc = _pick_col(fh, ("time", "timestamp", "时间"))
            xc = _pick_col(fh, ("x", "x坐标", "经度", "lon"))
            yc = _pick_col(fh, ("y", "y坐标", "纬度", "lat"))
            cc = _pick_col(fh, ("count", "人数", "数量"))
            if not all([tc, xc, yc, cc]):
                raise SystemExit(
                    f"flow-hour CSV missing columns; have {list(fh.columns)}. Need time,x,y,count."
                )
            assert tc is not None and xc is not None and yc is not None and cc is not None
            agg = aggregate_weighted_points(fh, xc, yc, cc, tc, units)
            for _, row in agg.iterrows():
                flows[(str(row["unit_id"]), pd.Timestamp(row["timestamp"]))] = float(row["value"])

        if args.od_o is not None:
            consume_od_points(args.od_o, units, outs, role="origin")
        if args.od_d is not None:
            consume_od_points(args.od_d, units, ins, role="dest")

        # Union of timestamps
        keys = set(flows) | set(outs) | set(ins)
        if not keys:
            raise SystemExit("No data after join — provide --flow-hour and/or --od-o / --od-d, or use --demo.")

        units_hit = sorted({k[0] for k in keys})
        times_hit = sorted({k[1] for k in keys})

        rows: list[dict[str, Any]] = []
        poi_cols = []
        poi_static_df = read_optional_csv(args.poi_static)
        static_cat_cols: list[str] = []
        if poi_static_df is not None and len(poi_static_df):
            poi_static_df = poi_static_df.dropna(subset=["unit_id"])
            for c in poi_static_df.columns:
                if c == "unit_id":
                    continue
                if str(c).lower() in ("geometry", "gridid", "grid_id"):
                    continue
                if pd.api.types.is_numeric_dtype(poi_static_df[c]):
                    static_cat_cols.append(str(c))
            poi_cols = list(static_cat_cols)
            poi_static_df = poi_static_df.set_index("unit_id")
            poi_static_df.index = poi_static_df.index.astype(str)
            pmap = poi_static_df
        else:
            pmap = None

        for uid in units_hit:
            for ts in times_hit:
                oa = ins.get((uid, ts), 0.0)
                ol = outs.get((uid, ts), 0.0)
                pr = flows.get((uid, ts), 0.0)
                if args.stay_mode == "flow":
                    st = pr
                else:
                    st = max(0.0, 2.0 * pr - oa - ol)
                rec = {
                    "unit_id": uid,
                    "timestamp": ts,
                    "hour": ts.hour,
                    "o_arrive": oa,
                    "o_leave": ol,
                    "o_stay": st,
                }
                if pmap is not None and str(uid) in pmap.index:
                    for c in poi_cols:
                        rec[c] = float(pmap.loc[str(uid), c])
                else:
                    for c in poi_cols:
                        rec[c] = 0.0
                rows.append(rec)

        long_df = pd.DataFrame(rows)
        if not poi_cols:
            long_df["poi_cat_1"] = 0.0
            poi_cols = ["poi_cat_1"]

        manifest = {
            "mode": "from_csv",
            "units_gpkg": str(args.units.resolve()),
            "flow_hour": str(args.flow_hour) if args.flow_hour else None,
            "od_o": str(args.od_o) if args.od_o else None,
            "od_d": str(args.od_d) if args.od_d else None,
            "poi_static": str(args.poi_static) if args.poi_static else None,
            "L": 3 + len(poi_cols),
            "poi_columns": poi_cols,
            "stay_mode": args.stay_mode,
            "time_start": str(long_df["timestamp"].min()),
            "time_end": str(long_df["timestamp"].max()),
            "n_rows": int(len(long_df)),
            "n_units": int(long_df["unit_id"].nunique()),
        }

    # Normalize (paper §4.1)
    flow_cols = ["o_arrive", "o_leave", "o_stay"]
    mm = minmax_by_unit_time(long_df, "unit_id", "timestamp", flow_cols)
    if TfidfTransformer is None:
        print("Warning: scikit-learn missing; skipping TF-IDF on POI columns.", file=sys.stderr)
        norm = mm.copy()
        for c in poi_cols:
            norm["norm_" + c] = mm.groupby("unit_id", sort=False)[c].transform(
                lambda g: (
                    (g - g.min()) / (g.max() - g.min() + 1e-12)
                    if float(g.max()) > float(g.min()) + 1e-12
                    else 0.5
                )
            )
    else:
        norm = tfidf_then_minmax_poi(mm, "unit_id", "timestamp", poi_cols)

    raw_path = args.out_dir / "observation_long.csv"
    nrm_path = args.out_dir / "observation_normalized.csv"
    long_df.to_csv(raw_path, index=False)
    norm.to_csv(nrm_path, index=False)

    manifest_path = args.out_dir / "sshmm_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Wrote {raw_path} ({len(long_df)} rows)")
    print(f"Wrote {nrm_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
