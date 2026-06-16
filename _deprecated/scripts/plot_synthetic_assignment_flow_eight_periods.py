#!/usr/bin/env python3
"""
将 ``synthetic_edge_flow_period_long.csv``（分时段 × 方式 × 边）汇总后在 4×2 子图中绘制路网边流量。

- 各子图带 **色标（colorbar）**：与 ``plot_synthetic_assignment_flow`` 一致，标为「分配流量」。
- 默认另存 **按交通方式拆分** 的八时段图：每方式一张 PNG，灰底仅含该方式对应的
  ``flow_geojson_class`` 路网（与 N01–N04 制图一致），着色边经 modality_layer 过滤。

用法（仓库根目录）：
  python scripts/plot_synthetic_assignment_flow_eight_periods.py ^
    --out-png output/synthetic_flow_worldpop/assignment_network_eight_periods_total.png ^
    --out-dir output/synthetic_flow_worldpop
"""
from __future__ import annotations

import argparse
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

from plot_synthetic_assignment_flow import (  # noqa: E402
    FLOW_COL_TO_ALLOW,
    FLOW_COLUMN_LABELS,
    _combine_bounds,
    _edge_class,
    _edge_visible_for_modality,
    _segment_m,
    _underlay_modality_segments_wgs,
    _unit_centroids_m,
    build_edge_lookup,
)
from plot_flow_modality_networks import configure_cn_font  # noqa: E402
from site_map_overlay import plot_site_boundary, resolve_site_json_path  # noqa: E402
from time_slice_constants import T_IDS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CRS_M = "EPSG:32651"

# 长表 modality 列 → 与 plot_synthetic_assignment_flow 一致的虚拟列名（用于图层过滤）
MODALITY_TO_FLOW_COL: dict[str, str] = {
    "N01_pedestrian": "flow_N01_pedestrian_aon",
    "N01_bike": "flow_N01_bike_aon",
    "N02_fast_auto": "flow_N02_fast_auto_aon",
    "N03_slow_auto": "flow_N03_slow_auto_aon",
    "N04_transit_proxy": "flow_N04_transit_proxy_aon",
}

# 与 synthetic_flow_od_gravity.FLOW_MODAL_ASSIGN_KEYS 顺序一致
MODAL_ORDER: tuple[str, ...] = tuple(MODALITY_TO_FLOW_COL.keys())


def _underlay_road_rr_wgs(
    edges_df: pd.DataFrame,
    cents_m: dict[str, tuple[float, float]],
) -> list[np.ndarray]:
    """仅 r–r 路网边灰底。"""
    segs_m: list[LineString] = []
    for _, row in edges_df.iterrows():
        a, b = row["source_id"], row["target_id"]
        if _edge_class(a, b) != "road":
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


def _draw_panel(
    fig: plt.Figure,
    ax,
    agg: pd.DataFrame,
    *,
    flow_col: str,
    cents_m: dict[str, tuple[float, float]],
    units_wgs: gpd.GeoDataFrame,
    site_path: Path | None,
    underlay_segs_wgs: list[np.ndarray],
    edge_filter: str,
    percentile_cap: float,
    min_flow: float,
    log_scale: bool,
    bb_pad: float,
    edge_lookup: dict | None = None,
    modality_flow_key: str | None = None,
    include_flow_connectors: bool = False,
    show_colorbar: bool = True,
) -> dict:
    geoms: list[LineString] = []
    vals: list[float] = []
    allow_col = FLOW_COL_TO_ALLOW.get(modality_flow_key or "", "") if modality_flow_key else ""
    for row in agg.itertuples(index=False):
        fv = float(getattr(row, flow_col))
        if not np.isfinite(fv) or fv < float(min_flow):
            continue
        ec = _edge_class(row.source_id, row.target_id)
        if edge_filter == "road_only" and ec != "road":
            continue
        if edge_filter == "connectors_only" and ec != "connector":
            continue
        if modality_flow_key and edge_lookup is not None:
            if not _edge_visible_for_modality(
                row.source_id,
                row.target_id,
                flow_col=modality_flow_key,
                edge_lookup=edge_lookup,
                network_filter="modality_layer",
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
        "edges_drawn": int(len(geoms)),
        "flow_sum_drawn": float(sum(vals)) if vals else 0.0,
    }

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

    bb = units_wgs.total_bounds
    pad = float(bb_pad)

    if not geoms:
        plot_site_boundary(ax, "EPSG:4326", site_path)
        ax.set_xlim(float(bb[0] - pad), float(bb[2] + pad))
        ax.set_ylim(float(bb[1] - pad), float(bb[3] + pad))
        ax.set_aspect("equal")
        ax.axis("off")
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

    if log_scale and lo > 0:
        norm: LogNorm | Normalize = LogNorm(vmin=max(lo, 1e-12), vmax=max(hi, lo * 1.01))
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
    if show_colorbar:
        cbar_label = "分配流量" + (" (log)" if log_scale else "")
        if modality_flow_key:
            cbar_label += f"\n{FLOW_COLUMN_LABELS.get(modality_flow_key, modality_flow_key)}"
        cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.02, shrink=0.82)
        cb.set_label(cbar_label, fontsize=8)
        cb.ax.tick_params(labelsize=7)

    plot_site_boundary(ax, "EPSG:4326", site_path)
    if underlay_segs_wgs:
        ub = gpd.GeoSeries([LineString(s) for s in underlay_segs_wgs], crs="EPSG:4326").total_bounds
        bb2 = _combine_bounds(gdf.total_bounds, tuple(ub))
    else:
        bb2 = tuple(gdf.total_bounds)
    ax.set_xlim(float(bb2[0] - pad), float(bb2[2] + pad))
    ax.set_ylim(float(bb2[1] - pad), float(bb2[3] + pad))
    ax.set_aspect("equal")
    ax.axis("off")
    return summary


def main() -> int:
    configure_cn_font()
    ap = argparse.ArgumentParser(description="八时段路网分配流量总览（长表汇总为各时段总流量）")
    ap.add_argument("--units", type=Path, default=REPO / "output" / "function" / "数据包" / "01_units.gpkg")
    ap.add_argument(
        "--period-flow-long-csv",
        type=Path,
        default=REPO / "output" / "synthetic_flow_worldpop" / "synthetic_edge_flow_period_long.csv",
    )
    ap.add_argument(
        "--assignment-edges-csv",
        type=Path,
        default=REPO / "output" / "synthetic_flow_worldpop" / "flow_road_assignment_edges.csv",
    )
    ap.add_argument(
        "--out-png",
        type=Path,
        default=REPO / "output" / "synthetic_flow_worldpop" / "assignment_network_eight_periods_total.png",
    )
    ap.add_argument("--site-json", type=Path, default=None)
    ap.add_argument(
        "--edge-filter",
        choices=("all", "road_only", "connectors_only"),
        default="road_only",
        help="默认 road_only：仅 r–r 道路段上色（接驳边流量通常较小）",
    )
    ap.add_argument("--percentile-cap", type=float, default=98.0)
    ap.add_argument("--min-flow", type=float, default=1e-12)
    ap.add_argument("--log-scale", action="store_true")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--bb-pad", type=float, default=0.002)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "output" / "synthetic_flow_worldpop",
        help="按方式拆分的 PNG 输出目录（默认与 --out-png 同根目录）",
    )
    ap.add_argument("--no-split-modalities", action="store_true", help="不输出按交通方式拆分的八时段图")
    ap.add_argument("--no-colorbar", action="store_true", help="不绘制色标（colorbar）")
    ap.add_argument(
        "--hide-flow-connectors",
        action="store_true",
        help="按方式出图时不着色地块–路网接驳边（仅 r–r 段上色）",
    )
    ns = ap.parse_args()

    long_path = Path(ns.period_flow_long_csv)
    if not long_path.is_file():
        raise SystemExit(f"Missing {long_path}")
    ae = Path(ns.assignment_edges_csv)
    if not ae.is_file():
        raise SystemExit(f"Missing {ae}")

    site_path = Path(ns.site_json) if ns.site_json is not None and Path(ns.site_json).is_file() else resolve_site_json_path()

    try:
        units = gpd.read_file(ns.units, layer="units")
    except Exception:
        units = gpd.read_file(ns.units)
    if units.crs is None:
        units = units.set_crs(4326)
    units_wgs = units.to_crs(4326)
    cents_m = _unit_centroids_m(units)

    edges_df = pd.read_csv(ae, encoding="utf-8-sig")
    underlay = _underlay_road_rr_wgs(edges_df, cents_m)

    df = pd.read_csv(long_path, encoding="utf-8-sig")
    for c in ("t_id", "source_id", "target_id", "flow_aon"):
        if c not in df.columns:
            raise SystemExit(f"period-flow-long-csv 缺少列 {c}")
    df["flow_aon"] = pd.to_numeric(df["flow_aon"], errors="coerce").fillna(0.0)

    if not ns.no_split_modalities and "modality" not in df.columns:
        raise SystemExit("period-flow-long-csv 缺少列 modality（无法按方式拆分）；可加 --no-split-modalities 仅输出合计图")

    t_order = [t for t in T_IDS if t in set(df["t_id"].astype(str).unique())]
    if not t_order:
        t_order = sorted(df["t_id"].astype(str).unique().tolist())

    agg_by_t: dict[str, pd.DataFrame] = {}
    for t_id in t_order:
        sub = df[df["t_id"].astype(str).eq(str(t_id))]
        g = (
            sub.groupby(["source_id", "target_id"], as_index=False)["flow_aon"]
            .sum()
            .rename(columns={"flow_aon": "flow_total"})
        )
        agg_by_t[str(t_id)] = g

    n_panels = len(t_order)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig_w = 14.0
    fig_h = max(3.85 * nrows, 9.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=int(ns.dpi), layout="constrained")
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, t_id in zip(axes_flat, t_order):
        summ = _draw_panel(
            fig,
            ax,
            agg_by_t[str(t_id)],
            flow_col="flow_total",
            cents_m=cents_m,
            units_wgs=units_wgs,
            site_path=site_path,
            underlay_segs_wgs=underlay,
            edge_filter=str(ns.edge_filter),
            percentile_cap=float(ns.percentile_cap),
            min_flow=float(ns.min_flow),
            log_scale=bool(ns.log_scale),
            bb_pad=float(ns.bb_pad),
            edge_lookup=None,
            modality_flow_key=None,
            show_colorbar=not bool(ns.no_colorbar),
        )
        ax.set_title(
            f"{t_id}  ·  边数={summ['edges_drawn']}  ·  Σflow≈{summ['flow_sum_drawn']:.4g}",
            fontsize=10,
        )

    for j in range(len(t_order), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "分时段交通分配：各时段全方式边流量合计\n"
        "灰色线：全部 r–r 路网底图  ·  彩色线：边流量（各子图右侧色标；标度为各时段独立 p98 截断）",
        fontsize=11,
    )
    out_png = Path(ns.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(ns.dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_png}")

    if not ns.no_split_modalities:
        edge_lookup = build_edge_lookup(edges_df)
        out_dir = Path(ns.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        modalities = [m for m in MODAL_ORDER if m in set(df["modality"].astype(str).unique())]
        for mod in modalities:
            flow_key = MODALITY_TO_FLOW_COL.get(mod)
            if not flow_key:
                continue
            allow_col = FLOW_COL_TO_ALLOW[flow_key]
            ul_mod = _underlay_modality_segments_wgs(
                edges_df,
                cents_m,
                flow_col=flow_key,
                underlay="road",
                network_filter="modality_layer",
                allow_col=allow_col,
                underlay_connectors=False,
            )
            agg_mod: dict[str, pd.DataFrame] = {}
            sub_m = df[df["modality"].astype(str).eq(mod)]
            for t_id in t_order:
                g = (
                    sub_m[sub_m["t_id"].astype(str).eq(str(t_id))]
                    .groupby(["source_id", "target_id"], as_index=False)["flow_aon"]
                    .sum()
                    .rename(columns={"flow_aon": "flow_mod"})
                )
                agg_mod[str(t_id)] = g
            fig_m, axes_m = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=int(ns.dpi), layout="constrained")
            axes_m_flat = np.atleast_1d(axes_m).ravel()
            for ax, t_id in zip(axes_m_flat, t_order):
                summ = _draw_panel(
                    fig_m,
                    ax,
                    agg_mod[str(t_id)],
                    flow_col="flow_mod",
                    cents_m=cents_m,
                    units_wgs=units_wgs,
                    site_path=site_path,
                    underlay_segs_wgs=ul_mod,
                    edge_filter=str(ns.edge_filter),
                    percentile_cap=float(ns.percentile_cap),
                    min_flow=float(ns.min_flow),
                    log_scale=bool(ns.log_scale),
                    bb_pad=float(ns.bb_pad),
                    edge_lookup=edge_lookup,
                    modality_flow_key=flow_key,
                    include_flow_connectors=not bool(ns.hide_flow_connectors),
                    show_colorbar=not bool(ns.no_colorbar),
                )
                ax.set_title(
                    f"{t_id}  ·  边数={summ['edges_drawn']}  ·  Σflow≈{summ['flow_sum_drawn']:.4g}",
                    fontsize=10,
                )
            for j in range(len(t_order), len(axes_m_flat)):
                axes_m_flat[j].set_visible(False)
            label = FLOW_COLUMN_LABELS.get(flow_key, mod)
            fig_m.suptitle(
                f"分时段分配 · {label}\n"
                "灰色线：本方式对应矢量图层（与 N01–N04 一致）  ·  彩色线：该方式边流量（右侧色标）",
                fontsize=11,
            )
            safe_mod = "".join(c if c.isalnum() or c in "-_" else "_" for c in mod)
            out_m = out_dir / f"assignment_network_eight_periods__{safe_mod}.png"
            fig_m.savefig(out_m, dpi=int(ns.dpi), bbox_inches="tight", facecolor="white")
            plt.close(fig_m)
            print(f"Wrote {out_m}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
