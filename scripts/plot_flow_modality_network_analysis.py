"""
对四类原始矢量交通网（与 plot_flow_modality_networks 同源）做图论指标：
  - 中心性：Monte Carlo 边介数（近似最短路径负荷）
  - 连通性：端点节点加权度之和，边值取两端 min（薄弱端 proxy）
  - 绕行成本：随机 OD 在移除「与 SITE 相交的边束」后的相对路径延长 / 断联率
  - 冗余度：SITE 相交边束的「绕行比」= 去掉该束后两端点最短路 / 该束最短边长

图构建：折线按相邻顶点拆成链段；与 plot_flow_modality_networks 相同的四类 GeoJSON。
子图：取「与 SITE 缓冲面相交的任一边」所在连通分量之并（否则退回最大分量），避免站域落在错误分量。

用法（仓库根）：
  python scripts/plot_flow_modality_network_analysis.py ^
    --units data/site_3km/01_units.gpkg ^
    --data-root data/site_3km ^
    --out-dir output/flow/flow_modality_network_analysis
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from shapely.geometry import LineString, box
from shapely.ops import unary_union

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_flow_modality_networks import (  # noqa: E402
    MODALITY_RULES,
    _assign_modality,
    _collect_geojsons,
    _explode_lines,
    configure_cn_font,
    is_dedicated_bicycle_geojson_path,
)
from site_map_overlay import load_site_gdf, plot_site_boundary, resolve_site_json_path  # noqa: E402

DEFAULT_MAX_FILE_MB = 30.0
METRIC_CRS = "EPSG:32651"

OUT_FILES = {
    "轨道与站点（原始矢量）": "N04_transit_network_metrics.png",
    "慢行与人行（原始矢量）": "N01_pedestrian_network_metrics.png",
    "快速路主干（原始矢量）": "N02_fast_road_network_metrics.png",
    "其它机动车道路（原始矢量）": "N03_slow_road_network_metrics.png",
}


def _snap_xy(x: float, y: float, snap_m: float) -> tuple[float, float]:
    inv = 1.0 / snap_m
    return round(x * inv) / inv, round(y * inv) / inv


def _explode_lines_metric(lines_wgs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    Lm = lines_wgs.to_crs(METRIC_CRS)
    return _explode_lines(Lm)


def _uv_key(u: tuple[float, float], v: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    return (u, v) if u <= v else (v, u)


def _lines_to_bundle_graph(
    lines_metric: gpd.GeoDataFrame, snap_m: float
) -> tuple[nx.Graph, gpd.GeoDataFrame, dict[int, tuple[tuple[float, float], tuple[float, float]]]]:
    """折线按相邻坐标拆成链段 → 无向简单图；平行束合并，权重取最短段长。"""
    Lexp = _explode_lines(lines_metric) if len(lines_metric) else gpd.GeoDataFrame(geometry=[], crs=lines_metric.crs)
    bundles: dict[tuple, dict] = defaultdict(lambda: {"lengths": [], "seg_idx": []})
    seg_geoms: list[LineString] = []
    seg_idx = 0
    for geom in Lexp.geometry:
        if geom is None or geom.is_empty or not isinstance(geom, LineString):
            continue
        c = np.array(geom.coords)
        if len(c) < 2:
            continue
        for i in range(len(c) - 1):
            u = _snap_xy(float(c[i][0]), float(c[i][1]), snap_m)
            v = _snap_xy(float(c[i + 1][0]), float(c[i + 1][1]), snap_m)
            if u == v:
                continue
            ddx = float(c[i + 1][0]) - float(c[i][0])
            ddy = float(c[i + 1][1]) - float(c[i][1])
            w = float(np.hypot(ddx, ddy))
            if w <= 0:
                continue
            key = _uv_key(u, v)
            b = bundles[key]
            b["lengths"].append(w)
            b["seg_idx"].append(seg_idx)
            seg_geoms.append(
                LineString([(float(c[i][0]), float(c[i][1])), (float(c[i + 1][0]), float(c[i + 1][1]))])
            )
            seg_idx += 1

    Lseg = gpd.GeoDataFrame(geometry=seg_geoms, crs=Lexp.crs)
    G = nx.Graph()
    seg_to_uv: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}
    for (ua, ub), d in bundles.items():
        wmin = min(d["lengths"])
        G.add_edge(ua, ub, length=wmin, seg_indices=list(d["seg_idx"]), lengths=list(d["lengths"]))
        for si in d["seg_idx"]:
            seg_to_uv[int(si)] = (ua, ub)

    return G, Lseg, seg_to_uv


def _site_polygon_metric(site_path: Path | None, units: gpd.GeoDataFrame, pad_m: float) -> object | None:
    site = load_site_gdf(site_path)
    if site is None or site.empty:
        return None
    sg = site.to_crs(METRIC_CRS)
    geom = sg.geometry
    site_union = geom.union_all() if hasattr(geom, "union_all") else unary_union(list(geom))
    try:
        return site_union.buffer(pad_m)
    except Exception:
        return site_union.buffer(pad_m)


def _edge_intersects_site(geom: LineString, site_poly) -> bool:
    if site_poly is None:
        return False
    try:
        return geom.intersects(site_poly)
    except Exception:
        return False


def _largest_cc_subgraph(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G
    nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(nodes).copy()


def _site_relevant_subgraph(G: nx.Graph, Lseg: gpd.GeoDataFrame, site_poly_metric) -> nx.Graph:
    """含 SITE 缓冲面相交边的所有连通分量之并；若无相交边则退回最大连通分量。"""
    if site_poly_metric is None or Lseg.empty:
        return _largest_cc_subgraph(G)
    touch: set[object] = set()
    for u, v, data in G.edges(data=True):
        for si in data.get("seg_indices", []):
            if si < 0 or si >= len(Lseg):
                continue
            geom = Lseg.geometry.iloc[int(si)]
            if isinstance(geom, LineString) and _edge_intersects_site(geom, site_poly_metric):
                touch.add(u)
                touch.add(v)
                break
    if not touch:
        return _largest_cc_subgraph(G)
    keep: set[object] = set()
    for comp in nx.connected_components(G):
        if set(comp) & touch:
            keep |= set(comp)
    return G.subgraph(keep).copy() if keep else _largest_cc_subgraph(G)


def monte_carlo_edge_betweenness(
    G: nx.Graph, n_samples: int, weight: str = "length", seed: int = 42
) -> dict[tuple, float]:
    rng = random.Random(seed)
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return {}
    counts: defaultdict[tuple, float] = defaultdict(float)
    trials = 0
    for _ in range(n_samples * 4):
        if trials >= n_samples:
            break
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u == v:
            continue
        try:
            p = nx.shortest_path(G, u, v, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        trials += 1
        for a, b in zip(p[:-1], p[1:]):
            ek = _uv_key(a, b)
            counts[ek] += 1.0
    if trials == 0:
        return {}
    inv = 1.0 / float(trials)
    return {ek: c * inv for ek, c in counts.items()}


def endpoint_strength(G: nx.Graph, weight: str = "length") -> dict[tuple, float]:
    strength: dict[object, float] = defaultdict(float)
    for u, v, data in G.edges(data=True):
        w = float(data.get(weight, 0.0) or 0.0)
        strength[u] += w
        strength[v] += w
    out: dict[tuple, float] = {}
    for u, v, data in G.edges(data=True):
        out[_uv_key(u, v)] = float(min(strength[u], strength[v]))
    return out


def redundancy_remove_bundle(G: nx.Graph, u, v, weight: str = "length") -> float | None:
    data = G.get_edge_data(u, v) or {}
    base = float(data.get(weight, 0.0) or 0.0)
    if base <= 0:
        return None
    H = G.copy()
    try:
        H.remove_edge(u, v)
    except Exception:
        return None
    try:
        d = nx.shortest_path_length(H, u, v, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    if not np.isfinite(d):
        return None
    return float(d) / base


def removal_od_stretch(
    G: nx.Graph,
    site_edges: set[tuple],
    n_samples: int,
    weight: str = "length",
    seed: int = 42,
) -> dict[str, float]:
    rng = random.Random(seed + 3)
    nodes = list(G.nodes())
    if len(nodes) < 2 or not site_edges:
        return {
            "mean_rel_stretch": float("nan"),
            "p95_rel_stretch": float("nan"),
            "disconnect_rate": float("nan"),
            "n_ok": 0.0,
        }
    stretches: list[float] = []
    disc = 0
    ok = 0
    for _ in range(n_samples * 5):
        if ok >= n_samples:
            break
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u == v:
            continue
        try:
            d0 = nx.shortest_path_length(G, u, v, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if not np.isfinite(d0) or d0 <= 0:
            continue
        H = G.copy()
        for ua, ub in site_edges:
            if H.has_edge(ua, ub):
                H.remove_edge(ua, ub)
        try:
            d1 = nx.shortest_path_length(H, u, v, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            disc += 1
            ok += 1
            continue
        stretches.append((d1 - d0) / d0)
        ok += 1
    if ok == 0:
        return {
            "mean_rel_stretch": float("nan"),
            "p95_rel_stretch": float("nan"),
            "disconnect_rate": float("nan"),
            "n_ok": 0.0,
        }
    arr = np.array(stretches, dtype=float)
    return {
        "mean_rel_stretch": float(np.nanmean(arr)),
        "p95_rel_stretch": float(np.nanpercentile(arr, 95)),
        "disconnect_rate": float(disc / max(1, ok)),
        "n_ok": float(ok),
    }


def _norm01(x: np.ndarray, lo_q: float = 5, hi_q: float = 95) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = np.percentile(x, lo_q), np.percentile(x, hi_q)
    if hi <= lo + 1e-12:
        return np.zeros_like(x)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def _gray_network_underlay(Lexp: gpd.GeoDataFrame) -> LineCollection | None:
    segs = []
    for geom in Lexp.geometry:
        if geom is None or geom.is_empty or not isinstance(geom, LineString):
            continue
        c = np.array(geom.coords)
        if len(c) < 2:
            continue
        segs.append(c)
    if not segs:
        return None
    return LineCollection(segs, colors="#c8c8c8", linewidths=0.32, alpha=0.55, zorder=1)


def _linecollection_from_segments(
    Lexp: gpd.GeoDataFrame,
    seg_to_val: dict[int, float],
    cmap: str,
) -> LineCollection | None:
    segs = []
    vals = []
    for seg_idx, geom in enumerate(Lexp.geometry):
        if geom is None or geom.is_empty or not isinstance(geom, LineString):
            continue
        if seg_idx not in seg_to_val:
            continue
        c = np.array(geom.coords)
        if len(c) < 2:
            continue
        segs.append(c)
        vals.append(float(seg_to_val[seg_idx]))
    if not segs:
        return None
    arr = np.array(vals, dtype=float)
    cnorm = _norm01(arr)
    lc = LineCollection(segs, array=cnorm, cmap=cmap, linewidths=1.35, alpha=0.88)
    return lc


def _map_bundle_to_segments(
    G: nx.Graph, bundle_metric: dict[tuple, float], seg_to_uv: dict[int, tuple]
) -> dict[int, float]:
    out: dict[int, float] = {}
    for si, uv in seg_to_uv.items():
        k = _uv_key(uv[0], uv[1])
        if k in bundle_metric:
            out[si] = bundle_metric[k]
    return out


def _site_bundle_edges(
    G: nx.Graph, Lexp: gpd.GeoDataFrame, site_poly_metric
) -> set[tuple]:
    """与 SITE 面相交的任一线段所属的边束 (u,v)。"""
    site_b: set[tuple] = set()
    for u, v, data in G.edges(data=True):
        for si in data.get("seg_indices", []):
            if si >= len(Lexp):
                continue
            geom = Lexp.geometry.iloc[int(si)]
            if isinstance(geom, LineString) and _edge_intersects_site(geom, site_poly_metric):
                site_b.add(_uv_key(u, v))
                break
    return site_b


def plot_four_panel(
    title: str,
    G: nx.Graph,
    Lexp: gpd.GeoDataFrame,
    seg_to_uv: dict[int, tuple],
    site_path: Path | None,
    units: gpd.GeoDataFrame,
    between: dict[tuple, float],
    strength: dict[tuple, float],
    redundancy: dict[tuple, float | None],
    out_path: Path,
    *,
    n_mc: int,
    site_poly_metric,
) -> dict:
    u0 = units.copy()
    if u0.crs is None:
        u0.set_crs(4326, inplace=True)
    um = u0.to_crs(METRIC_CRS)
    bb = um.total_bounds
    pad = max(bb[2] - bb[0], bb[3] - bb[1]) * 0.06
    clip_geom = box(bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad)

    site_edges = _site_bundle_edges(G, Lexp, site_poly_metric)
    rem_stats = removal_od_stretch(G, site_edges, n_samples=min(n_mc, 600), seed=7)

    red_list = [redundancy[ek] for ek in site_edges if ek in redundancy and redundancy[ek] is not None]
    red_arr = np.array(red_list, dtype=float) if red_list else np.array([], dtype=float)
    n_bridge = sum(1 for ek in site_edges if ek in redundancy and redundancy[ek] is None)

    bet_site = [between[ek] for ek in site_edges if ek in between]
    bet_all = list(between.values()) if between else []
    summary = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_site_bundle_edges": len(site_edges),
        "mc_samples": n_mc,
        "betweenness_mean_site": float(np.mean(bet_site)) if bet_site else float("nan"),
        "betweenness_mean_global": float(np.mean(bet_all)) if bet_all else float("nan"),
        "betweenness_ratio_site_over_global": (
            float(np.mean(bet_site) / (np.mean(bet_all) + 1e-12)) if bet_site and bet_all else float("nan")
        ),
        "endpoint_strength_mean_site": float(np.mean([strength[ek] for ek in site_edges if ek in strength]))
        if site_edges
        else float("nan"),
        "redundancy_ratio_mean_site": float(np.mean(red_arr)) if red_arr.size else float("nan"),
        "redundancy_ratio_median_site": float(np.median(red_arr)) if red_arr.size else float("nan"),
        "site_bundle_bridge_count": float(n_bridge),
        "removal_od_mean_rel_stretch": rem_stats["mean_rel_stretch"],
        "removal_od_p95_rel_stretch": rem_stats["p95_rel_stretch"],
        "removal_od_disconnect_rate": rem_stats["disconnect_rate"],
    }

    seg_between = _map_bundle_to_segments(G, between, seg_to_uv)
    seg_strength = _map_bundle_to_segments(G, strength, seg_to_uv)
    seg_redund: dict[int, float] = {}
    for si, uv in seg_to_uv.items():
        ek = _uv_key(uv[0], uv[1])
        if ek not in site_edges:
            continue
        r = redundancy.get(ek)
        seg_redund[si] = 0.0 if r is None else float(r)
    seg_site_bet = {}
    for si, uv in seg_to_uv.items():
        ek = _uv_key(uv[0], uv[1])
        if ek in site_edges:
            seg_site_bet[si] = between.get(ek, 0.0)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 11.0))
    panels = [
        ("中心性：边介数（Monte Carlo）", seg_between, "plasma"),
        ("连通性：端点加权度（薄弱端 min）", seg_strength, "viridis"),
        ("冗余度：SITE 束 绕行比（0=结构断束）", seg_redund, "coolwarm"),
        ("SITE 束：介数（关键廊道）", seg_site_bet, "magma"),
    ]

    for ax, (ptitle, segdict, cmap) in zip(axes.ravel(), panels):
        um_clip = um[um.intersects(clip_geom)]
        um_clip.plot(ax=ax, facecolor="#f4f4f4", edgecolor="#bbbbbb", linewidth=0.12, zorder=0)
        gray = _gray_network_underlay(Lexp)
        if gray is not None:
            ax.add_collection(gray)
        lc = _linecollection_from_segments(Lexp, {k: v for k, v in segdict.items() if np.isfinite(v)}, cmap)
        if lc is not None:
            lc.set_zorder(2)
            ax.add_collection(lc)
            cb = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
            cb.ax.tick_params(labelsize=7)
        plot_site_boundary(ax, METRIC_CRS, site_path)
        ax.set_xlim(bb[0] - pad, bb[2] + pad)
        ax.set_ylim(bb[1] - pad, bb[3] + pad)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(ptitle, fontsize=10)

    ratio = summary["betweenness_ratio_site_over_global"]
    rem = summary["removal_od_mean_rel_stretch"]
    rmed = summary["redundancy_ratio_median_site"]
    p_disc = summary["removal_od_disconnect_rate"]
    suptitle = (
        f"{title}\n"
        f"关键性：SITE 束平均介数 / 全网 ≈ {ratio:.2f}；"
        f"绕行：去掉 SITE 束后随机 OD 平均相对延长 {rem:.3f}，断联率 {p_disc:.1%}；"
        f"薄弱性：SITE 束绕行冗余中位数 ≈ {rmed:.2f}（越低越缺替代径）"
    )
    fig.suptitle(suptitle, fontsize=9.2, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return summary


def load_merged_modality(
    mod_name: str, roots: list[Path], max_file_mb: float
) -> gpd.GeoDataFrame | None:
    all_paths = _collect_geojsons(roots)
    paths = [p for p in all_paths if _assign_modality(p) == mod_name and not is_dedicated_bicycle_geojson_path(p)]
    if not paths:
        return None
    frames = []
    max_bytes = int(max(1.0, max_file_mb) * 1024 * 1024)
    for p in paths:
        try:
            if p.stat().st_size > max_bytes:
                continue
            g = gpd.read_file(p)
            if len(g) == 0:
                continue
            if g.crs is None:
                g.set_crs(4326, inplace=True)
            frames.append(g)
        except Exception:
            continue
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(merged, crs=frames[0].crs)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    return obj


def main() -> int:
    configure_cn_font()
    ap = argparse.ArgumentParser(description="四类矢量交通网：中心性/连通/绕行/冗余 制图")
    ap.add_argument("--units", type=Path, default=Path("data/site_3km/01_units.gpkg"))
    ap.add_argument("--data-root", type=Path, default=Path("data/site_3km"))
    ap.add_argument("--scan-all-under-root", action="store_true")
    ap.add_argument("--extra-root", type=Path, nargs="*", default=[])
    ap.add_argument("--out-dir", type=Path, default=Path("output/flow/flow_modality_network_analysis"))
    ap.add_argument("--site-json", type=Path, default=None)
    ap.add_argument("--snap-m", type=float, default=2.0, help="端点合并容差（米）")
    ap.add_argument("--site-buffer-m", type=float, default=80.0, help="SITE 面缓冲（米），用于与边求交")
    ap.add_argument("--mc-samples", type=int, default=900, help="Monte Carlo 介数抽样次数")
    ap.add_argument("--max-file-mb", type=float, default=DEFAULT_MAX_FILE_MB)
    ns = ap.parse_args()

    roots: list[Path] = []
    if ns.scan_all_under_root:
        roots.append(ns.data_root.resolve())
    else:
        for sub in ("04-交通数据", "metroflow"):
            p = (ns.data_root / sub).resolve()
            if p.is_dir():
                roots.append(p)
        if not roots:
            roots.append(ns.data_root.resolve())
    roots.extend([Path(p).resolve() for p in ns.extra_root])

    site_path = ns.site_json if ns.site_json is not None and ns.site_json.is_file() else resolve_site_json_path()

    try:
        units = gpd.read_file(ns.units, layer="units")
    except Exception:
        units = gpd.read_file(ns.units)

    try:
        units_wgs = units if units.crs and str(units.crs).upper().endswith("4326") else units.to_crs(4326)
    except Exception:
        units_wgs = units.to_crs(4326)

    site_poly_m = _site_polygon_metric(site_path, units_wgs, pad_m=float(ns.site_buffer_m))

    all_summary: dict[str, dict] = {}

    for mod_name, _keys in MODALITY_RULES:
        merged_wgs = load_merged_modality(mod_name, roots, ns.max_file_mb)
        out_png = ns.out_dir / OUT_FILES[mod_name]
        if merged_wgs is None or merged_wgs.empty:
            fig, ax = plt.subplots(figsize=(9, 7))
            units_wgs.plot(ax=ax, facecolor="#eee", edgecolor="#ccc", linewidth=0.2)
            plot_site_boundary(ax, units_wgs.crs, site_path)
            ax.set_title(f"{mod_name}\n无可用线段数据", fontsize=11)
            ax.axis("off")
            ns.out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_png, dpi=140, bbox_inches="tight")
            plt.close(fig)
            all_summary[mod_name] = {"error": "no_line_data"}
            print("Placeholder:", out_png)
            continue

        gt = merged_wgs.geometry.geom_type
        Lw = merged_wgs[gt.isin(["LineString", "MultiLineString", "GeometryCollection"])].copy()
        if Lw.empty:
            all_summary[mod_name] = {"error": "no_line_geometry"}
            print("Skip (no lines):", mod_name)
            continue

        Lm = _explode_lines_metric(Lw)
        G, Lseg, seg_to_uv = _lines_to_bundle_graph(Lm, snap_m=float(ns.snap_m))
        G = _site_relevant_subgraph(G, Lseg, site_poly_m)
        if G.number_of_edges() < 2:
            all_summary[mod_name] = {"error": "graph_too_small", "n_edges": G.number_of_edges()}
            print("Skip (tiny graph):", mod_name)
            continue

        between = monte_carlo_edge_betweenness(G, n_samples=int(ns.mc_samples), seed=11)
        strength = endpoint_strength(G, weight="length")
        site_edges = _site_bundle_edges(G, Lseg, site_poly_m)
        redundancy: dict[tuple, float | None] = {}
        for ua, ub in site_edges:
            redundancy[_uv_key(ua, ub)] = redundancy_remove_bundle(G, ua, ub, weight="length")

        summ = plot_four_panel(
            mod_name,
            G,
            Lseg,
            seg_to_uv,
            site_path,
            units_wgs,
            between,
            strength,
            redundancy,
            out_png,
            n_mc=int(ns.mc_samples),
            site_poly_metric=site_poly_m,
        )
        all_summary[mod_name] = summ
        print("Wrote", out_png, summ)

    ns.out_dir.mkdir(parents=True, exist_ok=True)
    (ns.out_dir / "metrics_summary.json").write_text(
        json.dumps(_json_safe(all_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
