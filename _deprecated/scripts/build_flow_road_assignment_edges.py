#!/usr/bin/env python3
"""
从 flow 制图所用的 GeoJSON 矢量交通线构建 **可用于 synthetic_flow_od_gravity 交通分配** 的边表。

与 ``plot_flow_modality_networks.py`` 相同的文件名规则归入四类；折线离散为邻接边；
地块质心 snap 到最近路网端点（≤ ``--snap-max-m``）生成 connector。

写出 CSV 含 ``allow_N01_pedestrian`` 等列；合成脚本使用 ``--assignment-edges-csv`` 指向该文件。

专用自行车道（文件名 stem 含「自行车」）不参与慢行类。

与 enriched N01 步行网对齐：先用 ``plot_N01_pedestrian_enriched.py --emit-walk-assignment-segments-csv`` 导出纯步行线段，
再 ``--enriched-n01-segments-csv`` 合并；建议 ``--site-buffer-m 3000`` 与制图一致。

用法（仓库根目录）：
  python scripts/plot_N01_pedestrian_enriched.py --extended-road-fclass ^
    --emit-walk-assignment-segments-csv output/synthetic_flow/n01_walk_segments_from_enriched.csv

  python scripts/build_flow_road_assignment_edges.py ^
    --units output/function/数据包/01_units.gpkg ^
    --data-root data/site_3km ^
    --site-buffer-m 3000 ^
    --enriched-n01-segments-csv output/synthetic_flow/n01_walk_segments_from_enriched.csv ^
    --out-csv output/synthetic_flow/flow_road_assignment_edges.csv

  python scripts/synthetic_flow_od_gravity.py ^
    --assignment-edges-csv output/synthetic_flow/flow_road_assignment_edges.csv ^
    --edges output/function/数据包/02_edges.csv ^
    --out-dir output/synthetic_flow
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, box

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_flow_modality_networks import (  # noqa: E402
    DEFAULT_MAX_FILE_MB,
    _assign_modality,
    _collect_geojsons,
    _explode_lines,
    is_dedicated_bicycle_geojson_path,
)

from site_map_overlay import load_site_gdf, resolve_site_json_path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CRS_M = "EPSG:32651"
WALK_M_PER_MIN = 5000.0 / 60.0

ALLOW_COLS = (
    "allow_N01_pedestrian",
    "allow_N01_bike",
    "allow_N02_fast_auto",
    "allow_N03_slow_auto",
    "allow_N04_transit_proxy",
)


def _node_key(x: float, y: float) -> str:
    return f"r{int(round(float(x)))}_{int(round(float(y)))}"


def _allows_for_flow_class(flow_class: str | None) -> dict[str, int]:
    """四类制图图层 → 各交通方式是否可走（简化规则）。"""
    z = {c: 0 for c in ALLOW_COLS}
    if flow_class is None:
        for c in ALLOW_COLS:
            z[c] = 1
        return z
    if flow_class == "慢行与人行（原始矢量）":
        z["allow_N01_pedestrian"] = 1
        z["allow_N01_bike"] = 1
        return z
    if flow_class == "快速路主干（原始矢量）":
        z["allow_N02_fast_auto"] = 1
        return z
    if flow_class == "其它机动车道路（原始矢量）":
        z["allow_N03_slow_auto"] = 1
        return z
    if flow_class == "轨道与站点（原始矢量）":
        z["allow_N04_transit_proxy"] = 1
        return z
    for c in ALLOW_COLS:
        z[c] = 1
    return z


def _connector_allows() -> dict[str, int]:
    """接驳边：各方式均可从地块进入路网（小汽车亦假定为停车场/路边接入 proxy）。"""
    return {c: 1 for c in ALLOW_COLS}


def site_buffer_polygon_m(units: gpd.GeoDataFrame, site_path: Path | None, radius_m: float):
    """与 plot_N01_pedestrian_enriched 一致：SITE 缓冲（米，EPSG:32651）；无 SITE 时用单元 union 质心。"""
    r = float(radius_m)
    site = load_site_gdf(site_path)
    if site is not None and not site.empty:
        sm = site.to_crs(CRS_M)
        geom = sm.geometry
        core = geom.union_all() if hasattr(geom, "union_all") else geom.unary_union
    else:
        um = units.to_crs(CRS_M)
        geom = um.geometry
        core = geom.union_all() if hasattr(geom, "union_all") else geom.unary_union
        try:
            core = core.centroid
        except Exception:
            pass
    return core.buffer(r)


def _load_units(path: Path) -> gpd.GeoDataFrame:
    try:
        u = gpd.read_file(path, layer="units")
    except Exception:
        u = gpd.read_file(path)
    if "unit_id" not in u.columns:
        raise ValueError("units 缺少 unit_id")
    if u.crs is None:
        u = u.set_crs(4326)
    return u


def _segment_rows_from_lines(
    lines_m: gpd.GeoDataFrame,
    flow_class: str | None,
    *,
    allows_override: dict[str, int] | None = None,
) -> list[dict]:
    allows = dict(allows_override) if allows_override is not None else _allows_for_flow_class(flow_class)
    rows: list[dict] = []
    for geom in lines_m.geometry:
        if geom is None or geom.is_empty:
            continue
        if not isinstance(geom, LineString):
            continue
        c = np.asarray(geom.coords, dtype=float)
        if len(c) < 2:
            continue
        for i in range(len(c) - 1):
            x1, y1 = float(c[i][0]), float(c[i][1])
            x2, y2 = float(c[i + 1][0]), float(c[i + 1][1])
            a, b = _node_key(x1, y1), _node_key(x2, y2)
            if a == b:
                continue
            length_m = float(np.hypot(x2 - x1, y2 - y1))
            if length_m < 0.05:
                continue
            walk_time_min = max(length_m / WALK_M_PER_MIN, 1e-6)
            cond = max(1.0 / max(length_m, 8.0), 1e-9)
            row = {
                "source_id": a,
                "target_id": b,
                "edge_kind": "flow_road_segment",
                "shared_length_m": 0.0,
                "centroid_dist_m": length_m,
                "walk_dist_m": length_m,
                "walk_time_min": walk_time_min,
                "cross_arterial": 0,
                "cross_rail": 0,
                "cross_river": 0,
                "cross_elevated": 0,
                "has_crossing_facility": 0,
                "barrier_cost": 0.0,
                "angular_cost": 0.0,
                "edge_cost": walk_time_min,
                "edge_conductance": cond,
                "edge_weight_norm": 0.0,
                "flow_geojson_class": flow_class or "",
                **allows,
            }
            rows.append(row)
    return rows


def _merge_undirected(rows: list[dict]) -> list[dict]:
    """同无向边合并：conductance 求和，walk_time_min 取较小。"""
    acc: dict[tuple[str, str], dict] = {}
    for r in rows:
        a, b = str(r["source_id"]), str(r["target_id"])
        uk = (a, b) if a <= b else (b, a)
        if uk not in acc:
            acc[uk] = dict(r)
            acc[uk]["source_id"], acc[uk]["target_id"] = uk[0], uk[1]
        else:
            acc[uk]["edge_conductance"] = float(acc[uk]["edge_conductance"]) + float(r["edge_conductance"])
            acc[uk]["walk_time_min"] = min(float(acc[uk]["walk_time_min"]), float(r["walk_time_min"]))
            acc[uk]["walk_dist_m"] = min(float(acc[uk]["walk_dist_m"]), float(r["walk_dist_m"]))
            acc[uk]["edge_cost"] = acc[uk]["walk_time_min"]
            for c in ALLOW_COLS:
                acc[uk][c] = max(int(acc[uk][c]), int(r[c]))
    return list(acc.values())


def main() -> int:
    ap = argparse.ArgumentParser(description="从 flow 用 GeoJSON 构建交通分配边表")
    ap.add_argument("--units", type=Path, default=REPO / "output/function/数据包/01_units.gpkg")
    ap.add_argument("--data-root", type=Path, default=REPO / "data/site_3km")
    ap.add_argument("--out-csv", type=Path, default=REPO / "output/synthetic_flow/flow_road_assignment_edges.csv")
    ap.add_argument("--out-meta", type=Path, default=None, help="默认与 out-csv 同目录 flow_road_assignment_edges_meta.json")
    ap.add_argument("--snap-max-m", type=float, default=650.0, help="质心到路网端点最大接驳距离 [m]")
    ap.add_argument("--bbox-pad-m", type=float, default=280.0, help="在 units 外包框外扩裁剪路网 [m]（--site-buffer-m=0 时生效）")
    ap.add_argument(
        "--site-buffer-m",
        type=float,
        default=0.0,
        help=">0 时用 SITE（或单元质心）缓冲多边形裁剪矢量（与 enriched N01 一致）；0 表示仅用 bbox-pad-m",
    )
    ap.add_argument("--site-json", type=Path, default=None, help="SITE GeoJSON；默认与 site_map_overlay 解析一致")
    ap.add_argument(
        "--enriched-n01-segments-csv",
        type=Path,
        default=None,
        help="plot_N01_pedestrian_enriched.py --emit-walk-assignment-segments-csv 产出的步行线段表，合并进本边表",
    )
    ns = ap.parse_args()

    roots = [Path(ns.data_root) / "04-交通数据", Path(ns.data_root) / "metroflow"]
    paths = _collect_geojsons(roots)

    u = _load_units(ns.units)
    um = u.to_crs(CRS_M)
    bb = um.total_bounds
    pad = float(ns.bbox_pad_m)
    site_path: Path | None = Path(ns.site_json) if ns.site_json is not None and Path(ns.site_json).is_file() else resolve_site_json_path()
    if float(ns.site_buffer_m) > 0:
        clip_poly = site_buffer_polygon_m(u, site_path, float(ns.site_buffer_m))
    else:
        clip_poly = box(bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad)

    cents = um.copy()
    cents["geometry"] = cents.geometry.centroid
    cx = cents.geometry.x.to_numpy(dtype=float)
    cy = cents.geometry.y.to_numpy(dtype=float)
    uids = cents["unit_id"].astype(str).tolist()

    road_rows: list[dict] = []
    stats_files: dict[str, int] = {
        "used": 0,
        "skipped_size": 0,
        "skipped_empty": 0,
        "skipped_nonline": 0,
        "skipped_bicycle": 0,
    }

    for p in paths:
        if is_dedicated_bicycle_geojson_path(p):
            stats_files["skipped_bicycle"] += 1
            continue
        if p.stat().st_size > DEFAULT_MAX_FILE_MB * 1024 * 1024:
            stats_files["skipped_size"] += 1
            continue
        flow_class = _assign_modality(p)
        try:
            gdf = gpd.read_file(p)
        except Exception:
            stats_files["skipped_empty"] += 1
            continue
        if gdf.empty:
            stats_files["skipped_empty"] += 1
            continue
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        gdf = gdf.to_crs(CRS_M)
        gdf = gdf[gdf.geometry.intersects(clip_poly)].copy()
        if gdf.empty:
            stats_files["skipped_empty"] += 1
            continue
        ex = _explode_lines(gdf)
        if ex.empty:
            stats_files["skipped_nonline"] += 1
            continue
        seg = _segment_rows_from_lines(ex, flow_class)
        road_rows.extend(seg)
        stats_files["used"] += 1

    enc = Path(ns.enriched_n01_segments_csv) if ns.enriched_n01_segments_csv else None
    if enc is not None and enc.is_file():
        extra = pd.read_csv(enc, encoding="utf-8-sig")
        stats_files["enriched_n01_segment_rows"] = int(len(extra))
        road_rows.extend(extra.to_dict("records"))

    road_rows = _merge_undirected(road_rows)

    end_keys: set[str] = set()
    for r in road_rows:
        end_keys.add(str(r["source_id"]))
        end_keys.add(str(r["target_id"]))
    endpoint_xy: list[tuple[float, float]] = []
    for nk in end_keys:
        if not nk.startswith("r"):
            continue
        parts = nk[1:].split("_", 1)
        if len(parts) != 2:
            continue
        try:
            endpoint_xy.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue

    snap_max = float(ns.snap_max_m)
    conn_rows: list[dict] = []
    n_snap_fail = 0
    if endpoint_xy:
        pts = np.asarray(endpoint_xy, dtype=float)
        tree = cKDTree(pts)
        allows_c = _connector_allows()
        for i in range(len(uids)):
            d_vec, idx = tree.query([cx[i], cy[i]], k=1)
            dist = float(d_vec)
            if dist > snap_max:
                n_snap_fail += 1
                continue
            ex, ey = pts[idx]
            rn = _node_key(ex, ey)
            length_m = max(dist, 1.0)
            walk_time_min = max(length_m / WALK_M_PER_MIN, 1e-6)
            cond = max(1.0 / max(length_m, 5.0), 1e-9)
            conn_rows.append(
                {
                    "source_id": uids[i],
                    "target_id": rn,
                    "edge_kind": "flow_road_connector",
                    "shared_length_m": 0.0,
                    "centroid_dist_m": length_m,
                    "walk_dist_m": length_m,
                    "walk_time_min": walk_time_min,
                    "cross_arterial": 0,
                    "cross_rail": 0,
                    "cross_river": 0,
                    "cross_elevated": 0,
                    "has_crossing_facility": 0,
                    "barrier_cost": 0.0,
                    "angular_cost": 0.0,
                    "edge_cost": walk_time_min,
                    "edge_conductance": cond,
                    "edge_weight_norm": 0.0,
                    "flow_geojson_class": "connector",
                    **allows_c,
                }
            )

    all_rows = _merge_undirected(road_rows + conn_rows)
    out_df = pd.DataFrame(all_rows)
    out_csv = Path(ns.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    meta_p = Path(ns.out_meta) if ns.out_meta else out_csv.with_name("flow_road_assignment_edges_meta.json")
    meta = {
        "method": "flow_geojson_line_segments_plus_centroid_snap_connectors",
        "inputs": {
            "units": str(ns.units),
            "data_root": str(ns.data_root),
            "site_json": str(site_path) if site_path else None,
            "enriched_n01_segments_csv": str(enc.resolve()) if enc and enc.is_file() else None,
        },
        "parameters": {
            "snap_max_m": snap_max,
            "bbox_pad_m": pad,
            "site_buffer_m": float(ns.site_buffer_m),
        },
        "stats": {
            **stats_files,
            "road_segment_rows_before_merge": len(road_rows),
            "connector_rows": len(conn_rows),
            "final_edge_rows": len(out_df),
            "units_total": len(uids),
            "units_not_snapped": int(n_snap_fail),
        },
        "allow_columns": list(ALLOW_COLS),
        "notes": (
            "Motor modes do not use pedestrian-only layers; N01 does not use pure motorway layers in this builder. "
            "Fragmented networks increase unreachable OD."
        ),
    }
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {out_csv} ({len(out_df)} edges), meta {meta_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
