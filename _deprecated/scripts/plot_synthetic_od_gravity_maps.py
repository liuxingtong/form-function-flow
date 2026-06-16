#!/usr/bin/env python3
"""
重力 OD（``synthetic_od_long.csv``）制图：出发合计、到达合计（地块 choropleth），
以及可选 Top-K 期望线（质心连线）。

默认 ``--total-trips 1`` 时流量为小数份额，图例数值同上。

用法（仓库根目录）：
  python scripts/plot_synthetic_od_gravity_maps.py ^
    --units output/function/数据包/01_units.gpkg ^
    --od-csv output/synthetic_flow_vector_fw/synthetic_od_long.csv ^
    --out-dir output/synthetic_flow_vector_fw
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

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_flow_modality_networks import configure_cn_font  # noqa: E402
from site_map_overlay import plot_site_boundary, resolve_site_json_path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CRS_M = "EPSG:32651"

configure_cn_font()


def _unit_centroids_wgs84(units: gpd.GeoDataFrame) -> dict[str, tuple[float, float]]:
    um = units.copy()
    if um.crs is None:
        um = um.set_crs(4326)
    um = um.to_crs(CRS_M)
    um["geometry"] = um.geometry.centroid
    uwgs = um.to_crs(4326)
    out: dict[str, tuple[float, float]] = {}
    for uid, geom in zip(uwgs["unit_id"].astype(str), uwgs.geometry):
        if geom is None or geom.is_empty:
            continue
        out[str(uid)] = (float(geom.x), float(geom.y))
    return out


def _merge_flow(units_wgs: gpd.GeoDataFrame, flow_by_uid: pd.Series, col: str) -> gpd.GeoDataFrame:
    u = units_wgs.copy()
    u[col] = u["unit_id"].astype(str).map(flow_by_uid).fillna(0.0).astype(float)
    return u


def _choropleth_ax(ax, gdf: gpd.GeoDataFrame, column: str, title: str, site_path: Path | None, *, log_scale: bool, pct_cap: float):
    v = gdf[column].to_numpy(dtype=float)
    pos = v[v > 0]
    if pos.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo = float(np.min(pos))
        hi = float(np.percentile(pos, min(100.0, float(pct_cap))))
        if hi <= lo:
            hi = float(np.max(pos))
    if log_scale and lo > 0:
        norm: LogNorm | Normalize = LogNorm(vmin=max(lo, 1e-18), vmax=max(hi, lo * 1.01))
    else:
        norm = Normalize(vmin=0.0, vmax=max(hi, 1e-18), clip=True)
    gdf.plot(
        ax=ax,
        column=column,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.06,
        legend=True,
        legend_kwds={"label": column, "shrink": 0.72},
        norm=norm,
    )
    plot_site_boundary(ax, gdf.crs, site_path)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    ax.set_aspect("equal")


def main() -> int:
    ap = argparse.ArgumentParser(description="重力 OD 长表制图（出发/到达 choropleth + 可选期望线）")
    ap.add_argument("--units", type=Path, default=REPO / "output/function/数据包/01_units.gpkg")
    ap.add_argument("--od-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None, help="默认与 od-csv 同目录")
    ap.add_argument("--site-json", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=165)
    ap.add_argument("--percentile-cap", type=float, default=98.0, help="choropleth 颜色上限分位（抗异常）")
    ap.add_argument("--log-scale", action="store_true", help="choropleth 用对数色标（仅正值）")
    ap.add_argument(
        "--top-desire-lines",
        type=int,
        default=4000,
        help="期望线条数上限（按 flow 降序）；0 表示不画期望线子图",
    )
    ap.add_argument(
        "--min-dist-m",
        type=float,
        default=0.0,
        help="期望线仅保留 dist_m≥该值的 OD；origin≠destination 始终保留。默认 0（若边表里 dist_m 单位异常可改用正值过滤极短 OD）",
    )
    ns = ap.parse_args()

    od_path = Path(ns.od_csv)
    if not od_path.is_file():
        raise SystemExit(f"Missing {od_path}")

    out_dir = Path(ns.out_dir) if ns.out_dir else od_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    site_path = Path(ns.site_json) if ns.site_json is not None and Path(ns.site_json).is_file() else resolve_site_json_path()

    try:
        units = gpd.read_file(ns.units, layer="units")
    except Exception:
        units = gpd.read_file(ns.units)
    if units.crs is None:
        units = units.set_crs(4326)
    if "unit_id" not in units.columns:
        raise SystemExit("units 缺少 unit_id")
    units_wgs = units.to_crs(4326)

    df = pd.read_csv(od_path, encoding="utf-8-sig")
    for c in ("origin_id", "destination_id", "flow"):
        if c not in df.columns:
            raise SystemExit(f"od-csv 缺少列 {c}")
    df["flow"] = pd.to_numeric(df["flow"], errors="coerce").fillna(0.0)

    out_sum = df.groupby(df["origin_id"].astype(str), sort=False)["flow"].sum()
    in_sum = df.groupby(df["destination_id"].astype(str), sort=False)["flow"].sum()

    g_out = _merge_flow(units_wgs, out_sum, "od_out_flow")
    g_in = _merge_flow(units_wgs, in_sum, "od_in_flow")

    n_desire = int(ns.top_desire_lines)
    pad = 0.002
    bb = units_wgs.total_bounds

    if n_desire > 0:
        fig = plt.figure(figsize=(12.5, 11.0), dpi=int(ns.dpi), layout="constrained")
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax_des = fig.add_subplot(gs[1, :])
    else:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 6.2), dpi=int(ns.dpi), layout="constrained")
        ax_des = None

    _choropleth_ax(
        ax0,
        g_out,
        "od_out_flow",
        "重力 OD · 出发量合计（按 origin 聚合）",
        site_path,
        log_scale=bool(ns.log_scale),
        pct_cap=float(ns.percentile_cap),
    )
    _choropleth_ax(
        ax1,
        g_in,
        "od_in_flow",
        "重力 OD · 到达量合计（按 destination 聚合）",
        site_path,
        log_scale=bool(ns.log_scale),
        pct_cap=float(ns.percentile_cap),
    )

    for ax in (ax0, ax1):
        ax.set_xlim(float(bb[0] - pad), float(bb[2] + pad))
        ax.set_ylim(float(bb[1] - pad), float(bb[3] + pad))

    if n_desire > 0 and ax_des is not None:
        md = float(ns.min_dist_m)
        sub = df[(df["origin_id"].astype(str) != df["destination_id"].astype(str))].copy()
        if "dist_m" in sub.columns:
            sub["dist_m"] = pd.to_numeric(sub["dist_m"], errors="coerce").fillna(0.0)
            sub = sub[sub["dist_m"] >= md]
        sub = sub.nlargest(min(n_desire, len(sub)), "flow", keep="first")

        cents = _unit_centroids_wgs84(units)
        segs: list[list[tuple[float, float]]] = []
        vals: list[float] = []
        for r in sub.itertuples(index=False):
            o, d = str(r.origin_id), str(r.destination_id)
            po, pd_ = cents.get(o), cents.get(d)
            if po is None or pd_ is None:
                continue
            segs.append([po, pd_])
            vals.append(float(r.flow))

        units_wgs.plot(ax=ax_des, color="#f0f0f0", edgecolor="#d8d8d8", linewidth=0.06)
        plot_site_boundary(ax_des, units_wgs.crs, site_path)

        if segs:
            v = np.asarray(vals, dtype=float)
            lo = float(np.min(v[v > 0])) if np.any(v > 0) else 0.0
            hi = float(np.percentile(v, min(100.0, float(ns.percentile_cap))))
            if hi <= lo:
                hi = float(np.max(v))
            if bool(ns.log_scale) and lo > 0:
                norm = LogNorm(vmin=max(lo, 1e-18), vmax=max(hi, lo * 1.01))
            else:
                norm = Normalize(vmin=0.0, vmax=max(hi, 1e-18), clip=True)
            lc = LineCollection(
                segs,
                cmap="magma",
                array=v,
                norm=norm,
                linewidths=0.55,
                alpha=0.55,
                zorder=2,
            )
            ax_des.add_collection(lc)
            cbar = fig.colorbar(lc, ax=ax_des, fraction=0.035, pad=0.02)
            cbar.set_label("OD flow（期望线）" + (" (log)" if ns.log_scale else ""))
        ax_des.set_xlim(float(bb[0] - pad), float(bb[2] + pad))
        ax_des.set_ylim(float(bb[1] - pad), float(bb[3] + pad))
        ax_des.set_aspect("equal")
        ax_des.axis("off")
        ax_des.set_title(
            f"重力 OD · Top-{len(segs)} 期望线（按 flow；origin≠destination；dist_m≥{md:g}m）",
            fontsize=11,
        )

    out_png = out_dir / "gravity_od_maps.png"
    fig.savefig(out_png, dpi=int(ns.dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
