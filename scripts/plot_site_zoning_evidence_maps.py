#!/usr/bin/env python3
"""
基于 ``parcel_radar_fields.csv`` + ``01_units.gpkg`` 计算四个「判断层」代理得分，
在地块上出图（Carto Light 底图 + SITE 边界），用于支撑功能分区叙事。

维度为 **数据代理**，非规划法定分区；公式见输出目录 ``plot_meta.json``。

运行（仓库根目录）::

  python scripts/plot_site_zoning_evidence_maps.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import Normalize

REPO = Path(__file__).resolve().parents[1]
SITE_3KM = REPO / "data" / "site_3km"
CRS_WGS = "EPSG:4326"
CRS_WM = "EPSG:3857"

# 从 plot_site_poi_density_by_category 复用底图与 SITE
import importlib.util

_poi_spec = importlib.util.spec_from_file_location("_site_poi", Path(__file__).parent / "plot_site_poi_density_by_category.py")
if _poi_spec is None or _poi_spec.loader is None:
    raise RuntimeError("无法加载 plot_site_poi_density_by_category.py")
_poi_mod = importlib.util.module_from_spec(_poi_spec)
_poi_spec.loader.exec_module(_poi_mod)


def configure_cn_font() -> None:
    preferred = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans")
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _z(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
    mu = x.mean(skipna=True)
    sig = x.std(skipna=True)
    if sig is None or sig < 1e-12:
        return x.fillna(0.0) * 0.0
    return ((x - mu) / sig).fillna(0.0)


def build_dimension_scores(df: pd.DataFrame) -> pd.DataFrame:
    """返回带 dim_* 列的 DataFrame（与 df 同 index）。"""
    out = df.copy()
    if "dist_to_station" in out.columns:
        d_station = pd.to_numeric(out["dist_to_station"], errors="coerce").fillna(
            float(pd.to_numeric(out["dist_to_station"], errors="coerce").median())
        )
    else:
        d_station = pd.Series(5000.0, index=out.index, dtype=float)
    # 距站越近越强
    near_station = 1.0 / (1.0 + d_station / 500.0)
    pt = (
        pd.to_numeric(out.get("poi_transport_density"), errors="coerce").fillna(0.0)
        + pd.to_numeric(out.get("poi_life_service_density"), errors="coerce").fillna(0.0) * 0.35
    )
    retail_food = pd.to_numeric(out.get("poi_retail_density"), errors="coerce").fillna(0.0) + pd.to_numeric(
        out.get("poi_food_density"), errors="coerce"
    ).fillna(0.0)
    out["dim_station_city"] = _z(near_station) + _z(pt) + _z(retail_food)

    fast = pd.to_numeric(out.get("fast_road_length_km_per_km2"), errors="coerce").fillna(0.0)
    slow = pd.to_numeric(out.get("slow_road_length_km_per_km2"), errors="coerce").fillna(0.0)
    office = pd.to_numeric(out.get("poi_office_density"), errors="coerce").fillna(0.0)
    sci = pd.to_numeric(out.get("poi_public_service_density"), errors="coerce").fillna(0.0)
    out["dim_business_access"] = _z(fast) + _z(slow) * 0.6 + _z(office) + 0.5 * _z(sci)

    green = pd.to_numeric(out.get("landuse_green_ratio"), errors="coerce").fillna(0.0)
    public_lu = pd.to_numeric(out.get("landuse_public_ratio"), errors="coerce").fillna(0.0)
    stay_poi = (
        pd.to_numeric(out.get("poi_food_density"), errors="coerce").fillna(0.0)
        + pd.to_numeric(out.get("poi_retail_density"), errors="coerce").fillna(0.0)
        + pd.to_numeric(out.get("poi_leisure_density"), errors="coerce").fillna(0.0)
    )
    walk_share = pd.to_numeric(out.get("walk_length_share"), errors="coerce").fillna(0.0)
    out["dim_public_stay"] = _z(green) * 1.1 + _z(public_lu) + _z(stay_poi) + _z(walk_share)

    perm = pd.to_numeric(out.get("permeability_index"), errors="coerce").fillna(0.0)
    edge = pd.to_numeric(out.get("edge_conductance_mean"), errors="coerce").fillna(0.0)
    walk_km = pd.to_numeric(out.get("walk_length_km_per_km2"), errors="coerce").fillna(0.0)
    barrier = pd.to_numeric(out.get("barrier_index"), errors="coerce").fillna(0.0)
    out["dim_slow_stitch"] = _z(perm) + _z(edge) + _z(walk_share) * 0.8 + _z(walk_km) * 0.5 - _z(barrier) * 0.7

    return out


def plot_panel(
    *,
    units_wm: gpd.GeoDataFrame,
    site_wm: gpd.GeoDataFrame,
    column: str,
    title: str,
    cmap: str,
    cbar_label: str,
    out_path: Path,
    vmax_q: float,
) -> None:
    vals = pd.to_numeric(units_wm[column], errors="coerce").fillna(0.0)
    vmax = float(vals.quantile(vmax_q))
    vmin = float(vals.quantile(1.0 - vmax_q))
    if vmax <= vmin:
        vmax = float(vals.max()) if len(vals) else 1.0
        vmin = float(vals.min()) if len(vals) else 0.0
    if vmax <= vmin:
        vmax = vmin + 1e-9
    norm = Normalize(vmin=vmin, vmax=vmax)

    bounds = _poi_mod.padded_bounds(units_wm)
    fig, ax = plt.subplots(figsize=(10.5, 9.5), dpi=200)
    ok, bm_label = _poi_mod.add_carto_light_basemap(ax, bounds)
    units_wm.plot(
        ax=ax,
        column=column,
        cmap=cmap,
        norm=norm,
        linewidth=0.2,
        edgecolor="#444444",
        alpha=0.9,
        legend=True,
        legend_kwds={"shrink": 0.52, "label": cbar_label},
        zorder=3,
    )
    site_wm.boundary.plot(ax=ax, color="#111111", linewidth=2.0, zorder=5)
    ax.set_title(title, fontsize=12, pad=8)
    if not ok:
        ax.text(0.02, 0.98, bm_label, transform=ax.transAxes, va="top", fontsize=8, color="#666666")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_four_panel(
    *,
    units_wm: gpd.GeoDataFrame,
    site_wm: gpd.GeoDataFrame,
    out_path: Path,
    vmax_q: float,
) -> None:
    panels = [
        ("dim_station_city", "站城界面强度（代理）", "OrRd", "综合 z 得分"),
        ("dim_business_access", "商务可达性（代理）", "Purples", "综合 z 得分"),
        ("dim_public_stay", "公共停留潜力（代理）", "YlGn", "综合 z 得分"),
        ("dim_slow_stitch", "慢行/形态缝合潜力（代理）", "PuBuGn", "综合 z 得分"),
    ]
    bounds = _poi_mod.padded_bounds(units_wm)
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), dpi=180)
    for ax, (col, title, cmap, clab) in zip(axes.ravel(), panels):
        vals = pd.to_numeric(units_wm[col], errors="coerce").fillna(0.0)
        vmax = float(vals.quantile(vmax_q))
        vmin = float(vals.quantile(1.0 - vmax_q))
        if vmax <= vmin:
            vmax = vmin + 1e-9
        norm = Normalize(vmin=vmin, vmax=vmax)
        ok, bm_label = _poi_mod.add_carto_light_basemap(ax, bounds)
        units_wm.plot(
            ax=ax,
            column=col,
            cmap=cmap,
            norm=norm,
            linewidth=0.15,
            edgecolor="#333333",
            alpha=0.88,
            legend=True,
            legend_kwds={"shrink": 0.45, "label": clab},
            zorder=3,
        )
        site_wm.boundary.plot(ax=ax, color="#111111", linewidth=1.6, zorder=5)
        ax.set_title(title, fontsize=11, pad=6)
        if not ok:
            ax.text(0.02, 0.98, bm_label, transform=ax.transAxes, va="top", fontsize=7, color="#666666")
    fig.suptitle("SITE 3km · 功能分区判断层 · 数据代理（非规划法定边界）", fontsize=14, y=0.995)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02, wspace=0.06, hspace=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    configure_cn_font()
    ap = argparse.ArgumentParser(description="四维度判断层代理得分 · 地块填色图")
    ap.add_argument("--units", type=Path, default=SITE_3KM / "01_units.gpkg")
    ap.add_argument("--radar-csv", type=Path, default=REPO / "output" / "radar_fields" / "parcel_radar_fields.csv")
    ap.add_argument("--site-json", type=Path, default=SITE_3KM / "SITE.json")
    ap.add_argument("--out-dir", type=Path, default=SITE_3KM / "qa" / "zoning_evidence")
    ap.add_argument("--vmax-quantile", type=float, default=0.97, help="色标上下限分位数（对称裁尾）")
    args = ap.parse_args()

    if not args.units.is_file():
        print(f"缺少: {args.units}", file=sys.stderr)
        return 1
    if not args.radar_csv.is_file():
        print(f"缺少: {args.radar_csv}（可先运行 scripts/build_parcel_radar_fields.py）", file=sys.stderr)
        return 1

    units = gpd.read_file(args.units)
    if units.crs is None:
        units = units.set_crs(CRS_WGS)
    units = units.to_crs(CRS_WGS)

    radar = pd.read_csv(args.radar_csv, encoding="utf-8-sig")
    if "unit_id" not in radar.columns or "unit_id" not in units.columns:
        print("CSV 或地块缺少 unit_id", file=sys.stderr)
        return 1

    merged = units.merge(radar, on="unit_id", how="left", suffixes=("", "_rad"))
    missing = merged["poi_food_density"].isna().sum() if "poi_food_density" in merged.columns else len(merged)
    if missing == len(merged):
        print("合并后无雷达字段，请检查 parcel_radar_fields.csv", file=sys.stderr)
        return 1

    scored = build_dimension_scores(merged)
    site = _poi_mod.load_site_polygon(args.site_json)
    site_wm = site.to_crs(CRS_WM)

    # 写属性表（WGS84 几何过大时可只写表格）
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table_cols = [
        "unit_id",
        "dist_to_station",
        "dim_station_city",
        "dim_business_access",
        "dim_public_stay",
        "dim_slow_stitch",
    ]
    ext_cols = [c for c in table_cols if c in scored.columns]
    scored[ext_cols].to_csv(args.out_dir / "zoning_evidence_unit_scores.csv", index=False, encoding="utf-8-sig")

    meta = {
        "inputs": {"units": str(args.units), "radar_csv": str(args.radar_csv)},
        "formula_zh": {
            "dim_station_city": "z(近站权重) + z(交通+0.35×生活POI密度) + z(餐饮+购物POI)",
            "dim_business_access": "z(快速路网长) + 0.6×z(慢速路) + z(办公POI) + 0.5×z(公服POI)",
            "dim_public_stay": "1.1×z(绿地用地比) + z(公服用地比) + z(餐饮+购物+休闲POI) + z(步行路长占比)",
            "dim_slow_stitch": "z(渗透性) + z(边传导) + 0.8×z(步行占比) + 0.5×z(步行网密度) - 0.7×z(阻隔指数)",
        },
        "note": "得分在全区地块内做 z 标准化后加权求和；用于与规划概念对照，非认定法定功能。",
        "outputs": [],
    }

    uw = scored.to_crs(CRS_WM)
    plot_four_panel(units_wm=uw, site_wm=site_wm, out_path=args.out_dir / "zoning_evidence_four_panel.png", vmax_q=args.vmax_quantile)
    meta["outputs"].append(str((args.out_dir / "zoning_evidence_four_panel.png").relative_to(REPO)))

    singles = [
        ("dim_station_city", "站城界面强度（代理）", "OrRd"),
        ("dim_business_access", "商务可达性（代理）", "Purples"),
        ("dim_public_stay", "公共停留潜力（代理）", "YlGn"),
        ("dim_slow_stitch", "慢行/形态缝合潜力（代理）", "PuBuGn"),
    ]
    for col, title, cmap in singles:
        p = args.out_dir / f"zoning_evidence__{col}.png"
        plot_panel(
            units_wm=uw,
            site_wm=site_wm,
            column=col,
            title=f"SITE 3km · {title}",
            cmap=cmap,
            cbar_label="综合 z 得分（裁尾分位显示）",
            out_path=p,
            vmax_q=args.vmax_quantile,
        )
        meta["outputs"].append(str(p.relative_to(REPO)))

    (args.out_dir / "plot_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写入:", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
