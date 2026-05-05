"""
Build 01_units.gpkg and 02_edges.csv for the Shanghai Railway Station 3 km study area.

Units: Shanghai plot parcels (plot_84_51N) clipped to SITE.buffer_3km.geojson.
The parcel GeoJSON is stored in projected coordinates but often mis-tagged as EPSG:4326;
this script assigns CRS from data/all/上海地块/plot_84_51N.prj (WGS_1984_UTM_Zone_51N, CM 123).

Edges: parcel pairs that touch (shared boundary); **knn_bridge** for units with zero
touch neighbors; **proximity_bridge** for centroid pairs within a metric radius that are
not already touch-linked (weak cross-gap / across-road coupling for fourth-person graph).

Optional barrier flags on centroid segments (v1 heuristic; refine with OSM later).

Run from repo root:
  python scripts/build_site_units_and_edges.py
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely import make_valid
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parents[1]
SITE_3KM = REPO / "data" / "site_3km"
DEFAULT_PLOTS = SITE_3KM / "上海地块" / "plot_84_51N.geojson"
DEFAULT_PRJ = REPO / "data" / "all" / "上海地块" / "plot_84_51N.prj"
DEFAULT_MASK = SITE_3KM / "SITE.buffer_3km.geojson"
OUT_UNITS = SITE_3KM / "01_units.gpkg"
OUT_EDGES = SITE_3KM / "02_edges.csv"
OUT_META = SITE_3KM / "build_units_edges_meta.json"

# Shanghai Railway Station (MetroFlow default); lon/lat WGS84
DEFAULT_STATION_LON = 121.451257271
DEFAULT_STATION_LAT = 31.249149419

RAIL_PATH = SITE_3KM / "04-交通数据" / "1-交通路网" / "1-合集" / "上海市_铁路线.geojson"
RIVER_PATH = SITE_3KM / "11-蓝绿空间" / "上海市_水系-开源.geojson"
EXPRESS_PATH = SITE_3KM / "04-交通数据" / "1-交通路网" / "2-分类图层" / "上海市_城市快速路.geojson"

# edge_cost = α * walk_time_min + Σ λ * crosses ...
ALPHA = 1.0
LAMBDA_ARTERIAL = 3.0
LAMBDA_RAIL = 5.0
LAMBDA_RIVER = 4.0
LAMBDA_ELEVATED = 3.0
MU_CROSSING = 0.0
GAMMA_ANGULAR = 0.0
THETA = 0.12

# barrier buffers (m) in projected CRS
BUF_RAIL_M = 22.0
BUF_RIVER_M = 35.0
BUF_EXPRESS_M = 28.0

# walk speed 5 km/h → m/min
WALK_M_PER_MIN = 5000.0 / 60.0
TORTUOSITY = 1.22
# Synthetic reachability edges (isolated parcels): weaker conductance multiplier
BRIDGE_CONDUCTANCE_MULT = 0.28
# Centroid-near pairs not already parcel_touch (cross road / gap coupling)
PROXIMITY_CONDUCTANCE_MULT = 0.4
DEFAULT_PROXIMITY_RADIUS_M = 160.0
DEFAULT_PROXIMITY_MAX_PER_NODE = 6


def load_plot_crs(prj_path: Path) -> CRS:
    return CRS.from_wkt(prj_path.read_text(encoding="utf-8"))


def ring_zone_for_dist_m(d: float) -> str:
    if d <= 1000.0:
        return "0_1km"
    if d <= 3000.0:
        return "1_3km"
    return "3_4km"


def load_barrier_union(path: Path, target_crs, mask_geom, buf_m: float):
    if not path.is_file():
        return None
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    g = g.to_crs(target_crs)
    g = g[g.intersects(mask_geom)].copy()
    if g.empty:
        return None
    geom = unary_union(g.geometry.buffer(buf_m).tolist())
    return geom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", type=Path, default=DEFAULT_PLOTS)
    ap.add_argument("--plot-prj", type=Path, default=DEFAULT_PRJ)
    ap.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    ap.add_argument("--station-lon", type=float, default=DEFAULT_STATION_LON)
    ap.add_argument("--station-lat", type=float, default=DEFAULT_STATION_LAT)
    ap.add_argument("--out-units", type=Path, default=OUT_UNITS)
    ap.add_argument("--out-edges", type=Path, default=OUT_EDGES)
    ap.add_argument("--out-meta", type=Path, default=OUT_META)
    ap.add_argument(
        "--no-bridge-isolated",
        action="store_true",
        help="Do not add knn_bridge edges for units with no parcel-touch neighbor.",
    )
    ap.add_argument(
        "--no-proximity-bridges",
        action="store_true",
        help="Do not add proximity_bridge edges (centroid pairs within radius, not touch).",
    )
    ap.add_argument(
        "--proximity-radius-m",
        type=float,
        default=DEFAULT_PROXIMITY_RADIUS_M,
        help="Max centroid distance (m, projected CRS) for proximity_bridge pairs.",
    )
    ap.add_argument(
        "--proximity-max-per-node",
        type=int,
        default=DEFAULT_PROXIMITY_MAX_PER_NODE,
        help="Cap undirected proximity_bridge edges per unit_id (both endpoints count).",
    )
    ap.add_argument(
        "--proximity-conductance-mult",
        type=float,
        default=PROXIMITY_CONDUCTANCE_MULT,
        help="Multiply conductance for proximity_bridge (0–1).",
    )
    args = ap.parse_args()

    plot_crs = load_plot_crs(args.plot_prj)
    plots = gpd.read_file(args.plots)
    plots = plots.set_crs(plot_crs, allow_override=True)

    mask = gpd.read_file(args.mask)
    if mask.crs is None:
        mask = mask.set_crs("EPSG:4326")
    mask = mask.to_crs(plot_crs)
    mask_geom = unary_union(mask.geometry)

    units = gpd.clip(plots, mask_geom).copy()
    units["geometry"] = units.geometry.map(lambda g: make_valid(g) if g is not None else g)
    units = units[~units.geometry.is_empty].reset_index(drop=True)

    if "id" in units.columns:
        units["unit_id"] = "plot_" + units["id"].astype(str)
    else:
        units["unit_id"] = ["plot_%05d" % i for i in range(len(units))]

    if units["unit_id"].duplicated().any():
        raise SystemExit("Duplicate unit_id after clip; check id column.")

    station_pt = gpd.GeoSeries(
        [Point(args.station_lon, args.station_lat)], crs="EPSG:4326"
    ).to_crs(plot_crs).iloc[0]

    units_metric = units.to_crs(plot_crs)
    # `area` / `dist_to_station`: square meters and meters (projected CRS), per 形+功+流 §0.2
    # Avoid exact-zero area on degenerate / sliver polygons (downstream density features).
    units_metric["area"] = (
        units_metric.geometry.map(lambda g: max(float(g.area), 0.01)).round(2)
    )
    centroids = units_metric.geometry.centroid
    dist_m = centroids.distance(station_pt)
    units_metric["dist_to_station"] = dist_m.round(2)
    units_metric["ring_zone"] = dist_m.map(ring_zone_for_dist_m)

    cents_wgs = gpd.GeoDataFrame(geometry=centroids, crs=plot_crs).to_crs("EPSG:4326")
    units_metric["centroid_x"] = cents_wgs.geometry.x.round(7)
    units_metric["centroid_y"] = cents_wgs.geometry.y.round(7)

    out_gdf = units_metric[
        ["unit_id", "geometry", "area", "centroid_x", "centroid_y", "dist_to_station", "ring_zone"]
    ].copy()
    # keep a few source attributes if present
    for col in ("PLOTNUMBER", "LANDAREA", "PLANLAND_1", "id"):
        if col in units_metric.columns and col not in out_gdf.columns:
            out_gdf[col] = units_metric[col]

    units_wgs84 = out_gdf.to_crs("EPSG:4326")
    args.out_units.parent.mkdir(parents=True, exist_ok=True)
    units_wgs84.to_file(args.out_units, driver="GPKG", layer="units")

    # --- edges ---
    base = units_metric[["unit_id", "geometry"]].copy()
    sj = gpd.sjoin(base, base, predicate="touches", how="inner")
    sj = sj[sj["unit_id_left"] != sj["unit_id_right"]]

    rail_u = load_barrier_union(RAIL_PATH, plot_crs, mask_geom, BUF_RAIL_M)
    river_u = load_barrier_union(RIVER_PATH, plot_crs, mask_geom, BUF_RIVER_M)
    express_u = load_barrier_union(EXPRESS_PATH, plot_crs, mask_geom, BUF_EXPRESS_M)

    geom_by_id = units_metric.set_index("unit_id")["geometry"]
    cents_map = geom_by_id.centroid

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for _, r in sj.iterrows():
        a, b = r["unit_id_left"], r["unit_id_right"]
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)

        ga = geom_by_id.loc[a]
        gb = geom_by_id.loc[b]
        bnd_a, bnd_b = ga.boundary, gb.boundary
        inter = bnd_a.intersection(bnd_b)
        if inter.is_empty:
            shared_len = 0.0
        else:
            shared_len = float(inter.length)

        ca, cb = cents_map[a], cents_map[b]
        centroid_dist_m = float(ca.distance(cb))
        walk_dist_m = round(centroid_dist_m * TORTUOSITY, 3)
        walk_time_min = walk_dist_m / WALK_M_PER_MIN

        line = LineString([ca, cb])
        cross_rail = int(rail_u is not None and line.intersects(rail_u))
        cross_river = int(river_u is not None and line.intersects(river_u))
        cross_arterial = int(express_u is not None and line.intersects(express_u))
        cross_elevated = 0
        has_crossing_facility = 0
        crossing_type = ""

        angular_cost = 0.0
        barrier_cost = (
            LAMBDA_ARTERIAL * cross_arterial
            + LAMBDA_RAIL * cross_rail
            + LAMBDA_RIVER * cross_river
            + LAMBDA_ELEVATED * cross_elevated
        )
        edge_cost = (
            ALPHA * walk_time_min
            + barrier_cost
            - MU_CROSSING * has_crossing_facility
            + GAMMA_ANGULAR * angular_cost
        )

        rows.append(
            {
                "source_id": a,
                "target_id": b,
                "edge_kind": "parcel_touch",
                "shared_length_m": round(shared_len, 3),
                "centroid_dist_m": round(centroid_dist_m, 3),
                "walk_dist_m": walk_dist_m,
                "walk_time_min": round(walk_time_min, 4),
                "cross_arterial": cross_arterial,
                "cross_rail": cross_rail,
                "cross_river": cross_river,
                "cross_elevated": cross_elevated,
                "has_crossing_facility": has_crossing_facility,
                "crossing_type": crossing_type,
                "barrier_cost": round(barrier_cost, 4),
                "angular_cost": angular_cost,
                "edge_cost": round(edge_cost, 4),
            }
        )

    nodes_touch: set[str] = set()
    for a, b in seen:
        nodes_touch.add(a)
        nodes_touch.add(b)
    all_uids = list(units_metric["unit_id"])
    isolated = [u for u in all_uids if u not in nodes_touch]
    n_bridge_pairs = 0
    seen_bridge: set[tuple[str, str]] = set()
    if not args.no_bridge_isolated and isolated:
        for u in isolated:
            cu = cents_map.loc[u]
            best_j: str | None = None
            best_d = float("inf")
            for v in all_uids:
                if v == u:
                    continue
                d = float(cu.distance(cents_map.loc[v]))
                if d < best_d:
                    best_d = d
                    best_j = v
            if best_j is None:
                continue
            key = tuple(sorted((u, best_j)))
            if key in seen or key in seen_bridge:
                continue
            seen_bridge.add(key)
            a, b = key[0], key[1]
            ga, gb = geom_by_id.loc[a], geom_by_id.loc[b]
            ca, cb = cents_map.loc[a], cents_map.loc[b]
            centroid_dist_m = float(ca.distance(cb))
            walk_dist_m = round(centroid_dist_m * TORTUOSITY, 3)
            walk_time_min = walk_dist_m / WALK_M_PER_MIN
            line = LineString([ca, cb])
            cross_rail = int(rail_u is not None and line.intersects(rail_u))
            cross_river = int(river_u is not None and line.intersects(river_u))
            cross_arterial = int(express_u is not None and line.intersects(express_u))
            barrier_cost = (
                LAMBDA_ARTERIAL * cross_arterial
                + LAMBDA_RAIL * cross_rail
                + LAMBDA_RIVER * cross_river
            )
            edge_cost = ALPHA * walk_time_min + barrier_cost
            rows.append(
                {
                    "source_id": a,
                    "target_id": b,
                    "edge_kind": "knn_bridge",
                    "shared_length_m": 0.0,
                    "centroid_dist_m": round(centroid_dist_m, 3),
                    "walk_dist_m": walk_dist_m,
                    "walk_time_min": round(walk_time_min, 4),
                    "cross_arterial": cross_arterial,
                    "cross_rail": cross_rail,
                    "cross_river": cross_river,
                    "cross_elevated": 0,
                    "has_crossing_facility": 0,
                    "crossing_type": "",
                    "barrier_cost": round(barrier_cost, 4),
                    "angular_cost": 0.0,
                    "edge_cost": round(edge_cost, 4),
                }
            )
            n_bridge_pairs += 1

    blocked: set[tuple[str, str]] = set(seen) | seen_bridge

    n_proximity_pairs = 0
    if not args.no_proximity_bridges and args.proximity_radius_m > 0:
        from scipy.spatial import cKDTree

        uids = list(units_metric["unit_id"])
        coords = np.array([[cents_map.loc[u].x, cents_map.loc[u].y] for u in uids], dtype=float)
        _tree = cKDTree(coords)
        raw_pairs = _tree.query_pairs(r=float(args.proximity_radius_m), output_type="ndarray")
        cand: list[tuple[float, tuple[str, str], int, int]] = []
        for i, j in raw_pairs:
            if i > j:
                i, j = j, i
            a, b = uids[int(i)], uids[int(j)]
            key = tuple(sorted((a, b)))
            if key in blocked:
                continue
            dij = float(np.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1]))
            cand.append((dij, key, int(i), int(j)))
        cand.sort(key=lambda x: x[0])
        chosen: set[tuple[str, str]] = set()
        prox_deg: dict[str, int] = defaultdict(int)
        cap = max(1, int(args.proximity_max_per_node))
        for _dij, key, _i, _j in cand:
            u, v = key
            if prox_deg[u] >= cap or prox_deg[v] >= cap:
                continue
            if key in chosen:
                continue
            chosen.add(key)
            prox_deg[u] += 1
            prox_deg[v] += 1
            a, b = key[0], key[1]
            ca, cb = cents_map.loc[a], cents_map.loc[b]
            centroid_dist_m = float(ca.distance(cb))
            walk_dist_m = round(centroid_dist_m * TORTUOSITY, 3)
            walk_time_min = walk_dist_m / WALK_M_PER_MIN
            line = LineString([ca, cb])
            cross_rail = int(rail_u is not None and line.intersects(rail_u))
            cross_river = int(river_u is not None and line.intersects(river_u))
            cross_arterial = int(express_u is not None and line.intersects(express_u))
            barrier_cost = (
                LAMBDA_ARTERIAL * cross_arterial
                + LAMBDA_RAIL * cross_rail
                + LAMBDA_RIVER * cross_river
            )
            edge_cost = ALPHA * walk_time_min + barrier_cost
            rows.append(
                {
                    "source_id": a,
                    "target_id": b,
                    "edge_kind": "proximity_bridge",
                    "shared_length_m": 0.0,
                    "centroid_dist_m": round(centroid_dist_m, 3),
                    "walk_dist_m": walk_dist_m,
                    "walk_time_min": round(walk_time_min, 4),
                    "cross_arterial": cross_arterial,
                    "cross_rail": cross_rail,
                    "cross_river": cross_river,
                    "cross_elevated": 0,
                    "has_crossing_facility": 0,
                    "crossing_type": "",
                    "barrier_cost": round(barrier_cost, 4),
                    "angular_cost": 0.0,
                    "edge_cost": round(edge_cost, 4),
                }
            )
            n_proximity_pairs += 1

    edges = pd.DataFrame(rows)
    edges["edge_conductance"] = edges["edge_cost"].map(lambda c: math.exp(-THETA * float(c)))
    edges.loc[edges["edge_kind"] == "knn_bridge", "edge_conductance"] *= BRIDGE_CONDUCTANCE_MULT
    pmult = float(max(0.0, min(1.0, args.proximity_conductance_mult)))
    edges.loc[edges["edge_kind"] == "proximity_bridge", "edge_conductance"] *= pmult

    # row-normalize conductance per source_id (undirected: symmetrize by also adding reverse?)
    # Doc: for each source_id, normalize outgoing edges. Graph is undirected; store both directions
    # for downstream that expects directed from source.
    fwd = edges.copy()
    rev = edges.rename(columns={"source_id": "target_id", "target_id": "source_id"})
    both = pd.concat([fwd, rev], ignore_index=True)
    sums = both.groupby("source_id")["edge_conductance"].transform("sum")
    both["edge_weight_norm"] = (both["edge_conductance"] / sums).round(6)

    out_edges = both.sort_values(["source_id", "target_id"]).reset_index(drop=True)
    out_edges.to_csv(args.out_edges, index=False, encoding="utf-8-sig")

    meta = {
        "n_units": int(len(units_wgs84)),
        "n_undirected_touch_pairs": len(seen),
        "n_units_isolated_touch_only": len(isolated),
        "n_undirected_knn_bridge_pairs": int(n_bridge_pairs),
        "n_undirected_proximity_bridge_pairs": int(n_proximity_pairs),
        "n_directed_edges_csv": int(len(out_edges)),
        "bridge_conductance_mult": BRIDGE_CONDUCTANCE_MULT,
        "proximity_radius_m": float(args.proximity_radius_m),
        "proximity_max_per_node": int(args.proximity_max_per_node),
        "proximity_conductance_mult": float(pmult),
        "bridge_isolated_default": not args.no_bridge_isolated,
        "plot_crs_wkt": plot_crs.to_wkt(),
        "units_crs_output": "EPSG:4326",
        "station_wgs84": [args.station_lon, args.station_lat],
        "mask": str(args.mask).replace("\\", "/"),
        "barrier_layers": {
            "rail": str(RAIL_PATH).replace("\\", "/") if RAIL_PATH.is_file() else None,
            "river": str(RIVER_PATH).replace("\\", "/") if RIVER_PATH.is_file() else None,
            "expressway": str(EXPRESS_PATH).replace("\\", "/") if EXPRESS_PATH.is_file() else None,
        },
        "parameters": {
            "alpha": ALPHA,
            "lambda_arterial": LAMBDA_ARTERIAL,
            "lambda_rail": LAMBDA_RAIL,
            "lambda_river": LAMBDA_RIVER,
            "lambda_elevated": LAMBDA_ELEVATED,
            "theta": THETA,
            "tortuosity": TORTUOSITY,
            "walk_m_per_min": WALK_M_PER_MIN,
        },
        "notes": (
            "Barrier flags use centroid–centroid segment vs buffered rail/river/expressway; "
            "walk_dist_m is centroid distance × tortuosity (no pedestrian network yet). "
            "knn_bridge edges connect units with zero parcel-touch neighbors; conductance scaled by "
            "bridge_conductance_mult. proximity_bridge adds centroid pairs within proximity_radius_m "
            "(excluding touch/knn pairs), capped per node, scaled by proximity_conductance_mult. "
            "Refine with OSM walk graph and crossing inventory when available."
        ),
    }
    args.out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Wrote", args.out_units, "units", len(units_wgs84))
    print("Wrote", args.out_edges, "directed edges", len(out_edges))
    print("Wrote", args.out_meta)


if __name__ == "__main__":
    main()
