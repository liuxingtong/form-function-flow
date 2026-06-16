#!/usr/bin/env python3
"""
将 ``synthetic_edge_flow_aon_multimodal.csv``（或 ``synthetic_edge_flow_aon_ped.csv``）中的分配流量
绘制成线图（路网节点 ``r{x}_{y}`` 为 EPSG:32651 米坐标；``plot_*`` 为地块 unit_id，取质心）。

**读图注意**

- ``synthetic_flow_od_gravity.py`` 默认 ``--total-trips 1``：全 OD 矩阵归一为 **1**（份额），边流量是和约为模态比例的 **小数**，不是「人次」；若要比绝对出行量，请增大 ``--total-trips``。
- 若曾使用 ``--aon-max-origins 150`` 等截断，则只有部分起点参与分配，边流量会整体偏小（见 ``synthetic_od_meta.json`` 中 ``aon_max_origins`` / ``lost_*``）。
- 提供 ``--assignment-edges-csv`` 时默认 **modality_layer**：每张图只保留与该方式对应的 ``flow_geojson_class``（与 N01–N04 制图图层一致），不再把其它等级路网叠在同一底图上；接驳边默认不进灰底，着色仍可选（见 ``--hide-flow-connectors``）。
- 若仅绘制 **流量 > min-flow** 的边，无流量路段不会着色；加 ``--underlay road`` 可铺本图层灰底。

用法（仓库根目录）：
  python scripts/plot_synthetic_assignment_flow.py ^
    --units output/function/数据包/01_units.gpkg ^
    --flow-csv output/synthetic_flow_vector_fw/synthetic_edge_flow_aon_multimodal.csv ^
    --assignment-edges-csv output/synthetic_flow/flow_road_assignment_edges.csv ^
    --underlay road ^
    --out-dir output/synthetic_flow_vector_fw/assignment_maps
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
from shapely.geometry import LineString

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_flow_modality_networks import configure_cn_font  # noqa: E402
from site_map_overlay import plot_site_boundary, resolve_site_json_path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CRS_M = "EPSG:32651"

configure_cn_font()

_ROAD_NODE_RE = re.compile(r"^r(-?\d+)_(-?\d+)$")

FLOW_COLUMN_LABELS: dict[str, str] = {
    "flow_N01_pedestrian_aon": "N01 步行",
    "flow_N01_bike_aon": "N01 自行车",
    "flow_N02_fast_auto_aon": "N02 快速路机动车",
    "flow_N03_slow_auto_aon": "N03 其它道路机动车",
    "flow_N04_transit_proxy_aon": "N04 轨道 proxy",
    "flow_ped_aon": "步行 (AON)",
}

FLOW_COL_TO_ALLOW: dict[str, str] = {
    "flow_N01_pedestrian_aon": "allow_N01_pedestrian",
    "flow_N01_bike_aon": "allow_N01_bike",
    "flow_N02_fast_auto_aon": "allow_N02_fast_auto",
    "flow_N03_slow_auto_aon": "allow_N03_slow_auto",
    "flow_N04_transit_proxy_aon": "allow_N04_transit_proxy",
    "flow_ped_aon": "allow_N01_pedestrian",
}

# 与 plot_flow_modality_networks 制图分类一致（分配边表 flow_geojson_class）
FLOW_COL_TO_LAYER_CLASSES: dict[str, tuple[str, ...]] = {
    "flow_N01_pedestrian_aon": ("慢行与人行（原始矢量）",),
    "flow_N01_bike_aon": ("慢行与人行（原始矢量）",),
    "flow_N02_fast_auto_aon": ("快速路主干（原始矢量）",),
    "flow_N03_slow_auto_aon": ("其它机动车道路（原始矢量）",),
    "flow_N04_transit_proxy_aon": ("轨道与站点（原始矢量）",),
    "flow_ped_aon": ("慢行与人行（原始矢量）",),
}


def _edge_uk(a: object, b: object) -> tuple[str, str]:
    x, y = str(a), str(b)
    return (x, y) if x <= y else (y, x)


def build_edge_lookup(edges_df: pd.DataFrame) -> dict[tuple[str, str], dict[str, object]]:
    out: dict[tuple[str, str], dict[str, object]] = {}
    for _, row in edges_df.iterrows():
        uk = _edge_uk(row["source_id"], row["target_id"])
        out[uk] = row.to_dict()
    return out


def _edge_visible_for_modality(
    source_id: object,
    target_id: object,
    *,
    flow_col: str,
    edge_lookup: dict[tuple[str, str], dict[str, object]] | None,
    network_filter: str,
    allow_col: str,
    include_flow_connectors: bool,
) -> bool:
    """是否绘制该 OD 边上的分配流量。"""
    if network_filter == "none" or edge_lookup is None:
        return True
    uk = _edge_uk(source_id, target_id)
    meta = edge_lookup.get(uk)
    if meta is None:
        return False
    ek = str(meta.get("edge_kind") or "")
    is_conn = ek == "flow_road_connector"

    if network_filter == "allow_only":
        try:
            ok = int(meta.get(allow_col, 0) or 0) == 1
        except (TypeError, ValueError):
            ok = False
        if is_conn:
            return bool(ok and include_flow_connectors)
        return ok

    classes = FLOW_COL_TO_LAYER_CLASSES.get(flow_col, ())
    if is_conn:
        return include_flow_connectors
    fg = str(meta.get("flow_geojson_class") or "")
    return fg in classes


def _underlay_modality_segments_wgs(
    edges_df: pd.DataFrame,
    cents_m: dict[str, tuple[float, float]],
    *,
    flow_col: str,
    underlay: str,
    network_filter: str,
    allow_col: str,
    underlay_connectors: bool,
) -> list[np.ndarray]:
    """灰底：默认仅本方式对应图层；不含其它 flow_geojson_class。"""
    if underlay not in ("road", "road_connector"):
        return []
    segs_m: list[LineString] = []

    if network_filter == "none":
        return _underlay_allow_segments_wgs(
            edges_df, cents_m, allow_col=allow_col, underlay=underlay
        )

    classes = FLOW_COL_TO_LAYER_CLASSES.get(flow_col, ())

    for _, row in edges_df.iterrows():
        a, b = row["source_id"], row["target_id"]
        ec = _edge_class(a, b)
        ek = str(row.get("edge_kind") or "")
        is_conn = ek == "flow_road_connector"
        if is_conn and not underlay_connectors:
            continue
        if underlay == "road" and ec != "road":
            continue
        if underlay == "road_connector" and ec == "parcel":
            continue

        if network_filter == "allow_only":
            try:
                av = int(row.get(allow_col, 0) or 0)
            except (TypeError, ValueError):
                av = 0
            if av != 1:
                continue
        else:
            fg = str(row.get("flow_geojson_class") or "")
            if is_conn:
                if not underlay_connectors:
                    continue
            elif fg not in classes:
                continue

        seg = _segment_m(a, b, cents_m)
        if seg is None or seg.is_empty:
            continue
        segs_m.append(seg)

    if not segs_m:
        return []
    gdf = gpd.GeoDataFrame(geometry=segs_m, crs=CRS_M).to_crs(4326)
    out: list[np.ndarray] = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        c = np.asarray(geom.coords, dtype=float)
        if len(c) >= 2:
            out.append(c)
    return out


def _parse_road_node(s: str) -> tuple[float, float] | None:
    m = _ROAD_NODE_RE.match(str(s))
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _unit_centroids_m(units: gpd.GeoDataFrame) -> dict[str, tuple[float, float]]:
    if "unit_id" not in units.columns:
        raise ValueError("units 缺少 unit_id")
    um = units.to_crs(CRS_M)
    um = um.copy()
    um["geometry"] = um.geometry.centroid
    out: dict[str, tuple[float, float]] = {}
    for uid, geom in zip(um["unit_id"].astype(str), um.geometry):
        if geom is None or geom.is_empty:
            continue
        out[str(uid)] = (float(geom.x), float(geom.y))
    return out


def _segment_m(
    a: object,
    b: object,
    cents_m: dict[str, tuple[float, float]],
) -> LineString | None:
    pa = _parse_road_node(a) if _is_road_node(a) else cents_m.get(str(a))
    pb = _parse_road_node(b) if _is_road_node(b) else cents_m.get(str(b))
    if pa is None or pb is None:
        return None
    return LineString([pa, pb])


def _edge_class(a: object, b: object) -> str:
    ra, rb = _is_road_node(a), _is_road_node(b)
    if ra and rb:
        return "road"
    if ra ^ rb:
        return "connector"
    return "parcel"


def _underlay_allow_segments_wgs(
    edges_df: pd.DataFrame,
    cents_m: dict[str, tuple[float, float]],
    *,
    allow_col: str,
    underlay: str,
) -> list[np.ndarray]:
    """按 allow_* 的灰底（旧逻辑；接驳对各方式均为 1 时会显得「全网」）。"""
    if underlay not in ("road", "road_connector"):
        return []
    segs_m: list[LineString] = []
    if allow_col not in edges_df.columns:
        return []
    for row in edges_df.itertuples(index=False):
        try:
            av = int(getattr(row, allow_col))
        except (TypeError, ValueError):
            av = 0
        if av != 1:
            continue
        a, b = getattr(row, "source_id"), getattr(row, "target_id")
        ec = _edge_class(a, b)
        if underlay == "road" and ec != "road":
            continue
        if underlay == "road_connector" and ec == "parcel":
            continue
        seg = _segment_m(a, b, cents_m)
        if seg is None or seg.is_empty:
            continue
        segs_m.append(seg)
    if not segs_m:
        return []
    gdf = gpd.GeoDataFrame(geometry=segs_m, crs=CRS_M).to_crs(4326)
    out: list[np.ndarray] = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        c = np.asarray(geom.coords, dtype=float)
        if len(c) >= 2:
            out.append(c)
    return out


def _is_road_node(s: object) -> bool:
    return isinstance(s, str) and _ROAD_NODE_RE.match(s) is not None


def _combine_bounds(*bounds_list: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    arr = np.array([b for b in bounds_list if b is not None], dtype=float)
    if arr.size == 0:
        return (0.0, 0.0, 1.0, 1.0)
    return (
        float(np.min(arr[:, 0])),
        float(np.min(arr[:, 1])),
        float(np.max(arr[:, 2])),
        float(np.max(arr[:, 3])),
    )


def _detect_flow_columns(df: pd.DataFrame) -> list[str]:
    skip = {"source_id", "target_id"}
    cols: list[str] = []
    for c in df.columns:
        if c in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(str(c))
    return cols


def _plot_flow_column(
    *,
    df: pd.DataFrame,
    flow_col: str,
    cents_m: dict[str, tuple[float, float]],
    units_wgs: gpd.GeoDataFrame,
    site_path: Path | None,
    out_png: Path,
    edge_filter: str,
    percentile_cap: float,
    min_flow: float,
    log_scale: bool,
    dpi: int,
    figsize: tuple[float, float],
    underlay_segs_wgs: list[np.ndarray],
    edge_lookup: dict[tuple[str, str], dict[str, object]] | None,
    network_filter: str,
    allow_col: str,
    include_flow_connectors: bool,
) -> dict:
    geoms: list[LineString] = []
    vals: list[float] = []
    for row in df.itertuples(index=False):
        fv = float(getattr(row, flow_col))
        if not np.isfinite(fv) or fv < float(min_flow):
            continue
        ec = _edge_class(row.source_id, row.target_id)
        if edge_filter == "road_only" and ec != "road":
            continue
        if edge_filter == "connectors_only" and ec != "connector":
            continue
        if not _edge_visible_for_modality(
            row.source_id,
            row.target_id,
            flow_col=flow_col,
            edge_lookup=edge_lookup,
            network_filter=network_filter,
            allow_col=allow_col,
            include_flow_connectors=include_flow_connectors,
        ):
            continue
        seg = _segment_m(row.source_id, row.target_id, cents_m)
        if seg is None or seg.length <= 0:
            continue
        geoms.append(seg)
        vals.append(fv)

    summary = {
        "flow_column": flow_col,
        "edges_drawn": int(len(geoms)),
        "flow_sum_drawn": float(sum(vals)) if vals else 0.0,
        "edge_filter": edge_filter,
        "network_filter": network_filter,
        "underlay_segments": int(len(underlay_segs_wgs)),
        "include_flow_connectors": include_flow_connectors,
    }
    if not geoms:
        fig, ax = plt.subplots(figsize=figsize)
        units_wgs.plot(ax=ax, color="#f4f4f4", edgecolor="#d4d4d4", linewidth=0.08, zorder=0)
        if underlay_segs_wgs:
            ax.add_collection(
                LineCollection(
                    underlay_segs_wgs,
                    colors="#9ca3af",
                    linewidths=0.35,
                    alpha=0.45,
                    zorder=1,
                )
            )
        plot_site_boundary(ax, "EPSG:4326", site_path)
        if underlay_segs_wgs:
            bbs = [gpd.GeoSeries([LineString(s) for s in underlay_segs_wgs], crs="EPSG:4326").total_bounds]
            pad = 0.002
            ax.set_xlim(float(bbs[0][0] - pad), float(bbs[0][2] + pad))
            ax.set_ylim(float(bbs[0][1] - pad), float(bbs[0][3] + pad))
        else:
            ax.set_xlim(*units_wgs.total_bounds[[0, 2]])
            ax.set_ylim(*units_wgs.total_bounds[[1, 3]])
        ax.set_title(
            f"{FLOW_COLUMN_LABELS.get(flow_col, flow_col)} · 无流量边（min-flow / edge-filter）；灰线为分配拓扑底图"
            if underlay_segs_wgs
            else f"{FLOW_COLUMN_LABELS.get(flow_col, flow_col)} · 无可用线段（检查 min-flow / edge-filter）"
        )
        ax.axis("off")
        ax.set_aspect("equal")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        summary["note"] = "empty_flow"
        return summary

    gdf_m = gpd.GeoDataFrame({"geometry": geoms, "_v": vals}, crs=CRS_M)
    gdf = gdf_m.to_crs(4326)
    v = np.asarray(gdf["_v"].values, dtype=float)
    pos = v[v > 0]
    if pos.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo = float(np.min(pos))
        hi = float(np.percentile(pos, min(100.0, float(percentile_cap))))
        if hi <= lo:
            hi = float(np.max(pos))

    segs_wgs: list[np.ndarray] = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        c = np.asarray(geom.coords, dtype=float)
        if len(c) >= 2:
            segs_wgs.append(c)

    fig, ax = plt.subplots(figsize=figsize)
    units_wgs.plot(ax=ax, color="#f4f4f4", edgecolor="#d4d4d4", linewidth=0.08, zorder=0)
    if underlay_segs_wgs:
        ax.add_collection(
            LineCollection(
                underlay_segs_wgs,
                colors="#9ca3af",
                linewidths=0.35,
                alpha=0.42,
                zorder=1,
            )
        )

    if log_scale and lo > 0:
        norm: Normalize | LogNorm = LogNorm(vmin=max(lo, 1e-12), vmax=max(hi, lo * 1.01))
    else:
        norm = Normalize(vmin=0.0, vmax=max(hi, 1e-12), clip=True)

    lc = LineCollection(
        segs_wgs,
        cmap="magma",
        array=v,
        norm=norm,
        linewidths=1.05,
        alpha=0.88,
        zorder=2,
    )
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("分配流量" + (" (log 标度)" if log_scale else ""))

    plot_site_boundary(ax, "EPSG:4326", site_path)
    ttl = FLOW_COLUMN_LABELS.get(flow_col, flow_col)
    ax.set_title(
        f"{ttl}\n有流量边={len(segs_wgs)}  流量合计≈{summary['flow_sum_drawn']:.4g}  筛选={edge_filter}"
        + f"  路网={network_filter}"
        + (f"  灰底拓扑={len(underlay_segs_wgs)}" if underlay_segs_wgs else ""),
        fontsize=11,
    )
    pad = 0.002
    if underlay_segs_wgs:
        ub = gpd.GeoSeries([LineString(s) for s in underlay_segs_wgs], crs="EPSG:4326").total_bounds
        bb = _combine_bounds(gdf.total_bounds, tuple(ub))
    else:
        bb = tuple(gdf.total_bounds)
    ax.set_xlim(float(bb[0] - pad), float(bb[2] + pad))
    ax.set_ylim(float(bb[1] - pad), float(bb[3] + pad))
    ax.set_aspect("equal")
    ax.axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="合成分配边流量地图（矢量路网 + 接驳）")
    ap.add_argument("--units", type=Path, default=REPO / "output/function/数据包/01_units.gpkg")
    ap.add_argument("--flow-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None, help="默认：flow-csv 同目录下 assignment_maps/")
    ap.add_argument("--site-json", type=Path, default=None)
    ap.add_argument(
        "--flow-columns",
        nargs="*",
        default=None,
        help="要绘制的列名；默认自动识别除 source_id/target_id 外全部数值列",
    )
    ap.add_argument(
        "--edge-filter",
        choices=("all", "road_only", "connectors_only"),
        default="all",
        help="road_only：仅 r–r 路网边；connectors_only：地块–路网接驳",
    )
    ap.add_argument("--percentile-cap", type=float, default=98.0)
    ap.add_argument("--min-flow", type=float, default=1e-10)
    ap.add_argument("--log-scale", action="store_true", help="颜色用对数标度（仅正值）")
    ap.add_argument("--dpi", type=int, default=165)
    ap.add_argument("--figsize", type=float, nargs=2, default=(11.0, 10.0))
    ap.add_argument(
        "--assignment-edges-csv",
        type=Path,
        default=None,
        help="与 synthetic 使用的 flow_road_assignment_edges.csv；配合 --underlay 画灰底拓扑",
    )
    ap.add_argument(
        "--underlay",
        choices=("none", "road", "road_connector"),
        default="none",
        help="none：不画底图；road：仅 r–r；road_connector：含接驳（接驳是否入灰底见 --underlay-connectors）",
    )
    ap.add_argument(
        "--network-filter",
        choices=("auto", "modality_layer", "allow_only", "none"),
        default="auto",
        help="auto：边表含 flow_geojson_class 时用 modality_layer（按图层类过滤），否则 none；"
        "modality_layer：与 N01–N04 制图图层一致；allow_only：按 allow_*；none：不着色过滤",
    )
    ap.add_argument(
        "--underlay-connectors",
        action="store_true",
        help="灰底包含 flow_road_connector（仅在与 --underlay road_connector 联用时生效）",
    )
    ap.add_argument(
        "--hide-flow-connectors",
        action="store_true",
        help="着色线段不画接驳边（仅道路段上色）",
    )
    ns = ap.parse_args()

    flow_path = Path(ns.flow_csv)
    if not flow_path.is_file():
        raise SystemExit(f"Missing {flow_path}")

    out_dir = Path(ns.out_dir) if ns.out_dir else flow_path.parent / "assignment_maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    site_path = Path(ns.site_json) if ns.site_json is not None and Path(ns.site_json).is_file() else resolve_site_json_path()

    try:
        units = gpd.read_file(ns.units, layer="units")
    except Exception:
        units = gpd.read_file(ns.units)
    if units.crs is None:
        units = units.set_crs(4326)
    units_wgs = units.to_crs(4326)
    cents_m = _unit_centroids_m(units)

    df = pd.read_csv(flow_path, encoding="utf-8-sig")
    flow_cols = list(ns.flow_columns) if ns.flow_columns else _detect_flow_columns(df)
    if not flow_cols:
        raise SystemExit("未找到可绘制的流量列")

    edges_df: pd.DataFrame | None = None
    ae = Path(ns.assignment_edges_csv) if ns.assignment_edges_csv else None
    if ae is not None and ae.is_file():
        edges_df = pd.read_csv(ae, encoding="utf-8-sig")
    elif ns.underlay != "none":
        print("WARN: --underlay 需要有效的 --assignment-edges-csv", file=sys.stderr)

    nf = str(ns.network_filter)
    if nf == "auto":
        if edges_df is not None and "flow_geojson_class" in edges_df.columns:
            nf = "modality_layer"
        else:
            nf = "none"

    edge_lookup = build_edge_lookup(edges_df) if edges_df is not None else None

    manifest: dict = {
        "inputs": {
            "units": str(ns.units.resolve()),
            "flow_csv": str(flow_path.resolve()),
            "assignment_edges_csv": str(ae.resolve()) if ae and ae.is_file() else None,
        },
        "edge_filter": ns.edge_filter,
        "underlay": ns.underlay,
        "network_filter_resolved": nf,
        "underlay_connectors": bool(ns.underlay_connectors),
        "hide_flow_connectors": bool(ns.hide_flow_connectors),
        "flow_columns": flow_cols,
        "maps": {},
    }
    safe = lambda s: "".join(c if c.isalnum() or c in "-_" else "_" for c in s)
    for col in flow_cols:
        if col not in df.columns:
            print(f"Skip missing column: {col}", file=sys.stderr)
            continue
        out_png = out_dir / f"assignment_flow__{safe(col)}.png"
        allow_col = FLOW_COL_TO_ALLOW.get(col, "")
        ul: list[np.ndarray] = []
        if edges_df is not None and ns.underlay != "none":
            ul = _underlay_modality_segments_wgs(
                edges_df,
                cents_m,
                flow_col=col,
                underlay=str(ns.underlay),
                network_filter=nf,
                allow_col=allow_col,
                underlay_connectors=bool(ns.underlay_connectors),
            )
        summ = _plot_flow_column(
            df=df,
            flow_col=col,
            cents_m=cents_m,
            units_wgs=units_wgs,
            site_path=site_path,
            out_png=out_png,
            edge_filter=ns.edge_filter,
            percentile_cap=float(ns.percentile_cap),
            min_flow=float(ns.min_flow),
            log_scale=bool(ns.log_scale),
            dpi=int(ns.dpi),
            figsize=(float(ns.figsize[0]), float(ns.figsize[1])),
            underlay_segs_wgs=ul,
            edge_lookup=edge_lookup,
            network_filter=nf,
            allow_col=allow_col,
            include_flow_connectors=not bool(ns.hide_flow_connectors),
        )
        manifest["maps"][col] = {"png": str(out_png.resolve()), **summ}
        print(f"Wrote {out_png}")

    meta_path = out_dir / "assignment_maps_meta.json"
    meta_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
