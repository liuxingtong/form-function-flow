#!/usr/bin/env python3
"""
用 ``plot_site_zoning_evidence_maps.build_dimension_scores`` 得到的四个 dim_*，
在 **与 SITE 红线面（可选缓冲）相交** 的地块子集内做分位归一化，再按预设权重得到五类「功能分区亲和度」，
取 argmax 作为 **数据推断分区**（非规划法定成果）。

五类与规划概念对齐为软标签：站城门户 / 商务核心 / 消费活力（轴线无矢量时代理）/
绿地缝合 / 城市过渡。SITE 外地块不赋分区，图中置灰。

红线面积极小时，相交地块可能很少，建议加 ``--buffer-m 200``～``400``（米）扩展「场地」范围再算分区。

示例::

  python scripts/assign_site_zones_from_four_dims.py
  python scripts/assign_site_zones_from_four_dims.py --buffer-m 400
  python scripts/assign_site_zones_from_four_dims.py --out-dir data/site_3km/qa/zoning_evidence
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import ListedColormap

REPO = Path(__file__).resolve().parents[1]
SITE_3KM = REPO / "data" / "site_3km"
CRS_WGS = "EPSG:4326"
CRS_WM = "EPSG:3857"

_DIMS = ("dim_station_city", "dim_business_access", "dim_public_stay", "dim_slow_stitch")

_ZONE_DEFS: list[tuple[str, str, np.ndarray]] = [
    ("gateway", "站城门户区", np.array([0.52, 0.18, 0.25, 0.05], dtype=float)),
    ("business", "商务核心区", np.array([0.12, 0.58, 0.18, 0.12], dtype=float)),
    ("consumption", "消费活力区", np.array([0.15, 0.22, 0.52, 0.11], dtype=float)),
    ("green_stitch", "绿地缝合区", np.array([0.04, 0.06, 0.38, 0.52], dtype=float)),
    # 过渡区公式：w0*(1-S)+w1*(1-B)+w2*P+w3*T（见 zone_affinities）
    ("transition", "城市过渡区", np.array([0.18, 0.22, 0.30, 0.32], dtype=float)),
]


def _load_zoning_module():
    p = Path(__file__).resolve().parent / "plot_site_zoning_evidence_maps.py"
    spec = importlib.util.spec_from_file_location("_zoning_evidence", p)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 plot_site_zoning_evidence_maps.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def configure_cn_font() -> None:
    preferred = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans")
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _robust_unit(s: pd.Series, lo: float = 0.05, hi: float = 0.95) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
    a = float(x.quantile(lo))
    b = float(x.quantile(hi))
    if b <= a + 1e-12:
        return pd.Series(0.5, index=x.index, dtype=float)
    return ((x - a) / (b - a)).clip(0.0, 1.0).fillna(0.5)


def normalize_dims_within_site(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in _DIMS:
        out[f"N_{c}"] = _robust_unit(out[c])
    return out


def zone_affinities(df_n: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """对每个地块计算五列亲和度 + argmax 标签与 margin。"""
    S = df_n["N_dim_station_city"].to_numpy(dtype=float)
    B = df_n["N_dim_business_access"].to_numpy(dtype=float)
    P = df_n["N_dim_public_stay"].to_numpy(dtype=float)
    T = df_n["N_dim_slow_stitch"].to_numpy(dtype=float)
    X = np.column_stack([S, B, P, T])

    cols_L: list[np.ndarray] = []
    keys: list[str] = []
    labels_zh: list[str] = []
    for key, zh, w in _ZONE_DEFS:
        keys.append(key)
        labels_zh.append(zh)
        if key == "transition":
            # 低站城、低商务枢纽性 + 慢行/公服/缝合
            v = w[0] * (1.0 - S) + w[1] * (1.0 - B) + w[2] * P + w[3] * T
        else:
            v = (X * w.reshape(1, -1)).sum(axis=1)
        cols_L.append(v)

    L = np.column_stack(cols_L)
    ix = np.argmax(L, axis=1)
    row_max = L[np.arange(L.shape[0]), ix]
    L2 = np.partition(L, -2, axis=1)[:, -2]
    margin = (row_max - L2) / (np.abs(row_max) + 1e-9)

    out = df_n.copy()
    for i, k in enumerate(keys):
        out[f"L_{k}"] = L[:, i]
    out["fz_site_key"] = [keys[i] for i in ix]
    out["fz_site_zh"] = [labels_zh[i] for i in ix]
    out["fz_site_margin"] = margin
    return out, keys


def plot_site_zones_map(
    *,
    units_wm: gpd.GeoDataFrame,
    site_wm: gpd.GeoDataFrame,
    out_path: Path,
    zone_col: str = "fz_site_zh",
    inside_col: str = "inside_site",
) -> None:
    _poi_spec = importlib.util.spec_from_file_location(
        "_site_poi", Path(__file__).resolve().parent / "plot_site_poi_density_by_category.py"
    )
    if _poi_spec is None or _poi_spec.loader is None:
        raise RuntimeError("plot_site_poi_density_by_category")
    _poi_mod = importlib.util.module_from_spec(_poi_spec)
    _poi_spec.loader.exec_module(_poi_mod)

    bounds = _poi_mod.padded_bounds(units_wm)
    fig, ax = plt.subplots(figsize=(12.5, 11), dpi=200)
    ok, bm_label = _poi_mod.add_carto_light_basemap(ax, bounds)

    u = units_wm.copy()
    categories = [zh for _, zh, _ in _ZONE_DEFS]
    colors = ["#d94801", "#54278f", "#cb181d", "#238b45", "#fec44f"]
    cmap = ListedColormap(colors)
    u["_plot_zone"] = u[zone_col].astype(object)
    u.loc[~u[inside_col].astype(bool), "_plot_zone"] = pd.NA

    u.plot(
        ax=ax,
        column="_plot_zone",
        categorical=True,
        categories=categories,
        cmap=cmap,
        linewidth=0.18,
        edgecolor="#333333",
        alpha=0.92,
        legend=False,
        missing_kwds={"color": "#e8e8e8", "edgecolor": "#bbbbbb", "linewidth": 0.12},
        zorder=3,
    )
    site_wm.boundary.plot(ax=ax, color="#111111", linewidth=2.2, zorder=6)
    ax.set_title("SITE 相交地块 · 数据推断五类功能分区（由四指标亲和度 argmax）", fontsize=12, pad=10)
    if not ok:
        ax.text(0.02, 0.98, bm_label, transform=ax.transAxes, va="top", fontsize=8, color="#666666")

    from matplotlib.patches import Patch

    leg = [Patch(facecolor=colors[i], edgecolor="#333333", label=categories[i]) for i in range(len(categories))]
    leg.append(Patch(facecolor="#e8e8e8", edgecolor="#bbbbbb", label="SITE 外（未分类）"))
    ax.legend(handles=leg, loc="lower left", fontsize=8, framealpha=0.92)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    configure_cn_font()
    ap = argparse.ArgumentParser(description="由四 dim 推断 SITE 内五类功能分区（数据代理）")
    ap.add_argument("--units", type=Path, default=SITE_3KM / "01_units.gpkg")
    ap.add_argument("--radar-csv", type=Path, default=REPO / "output" / "radar_fields" / "parcel_radar_fields.csv")
    ap.add_argument("--site-json", type=Path, default=SITE_3KM / "SITE.json")
    ap.add_argument("--out-dir", type=Path, default=SITE_3KM / "qa" / "zoning_evidence")
    ap.add_argument("--gpkg-name", type=str, default="site_function_zones_from_dims.gpkg", help="写入 out-dir 的 GPKG 文件名")
    ap.add_argument(
        "--buffer-m",
        type=float,
        default=0.0,
        help="对 SITE 面做缓冲（米，EPSG:32651）后再判「在内」；红线面积极小时可设 50–200 以覆盖更多地块",
    )
    args = ap.parse_args()

    zm = _load_zoning_module()
    site = zm._poi_mod.load_site_polygon(args.site_json)
    try:
        site_union = site.geometry.union_all()
    except Exception:
        site_union = site.geometry.unary_union
    if args.buffer_m and float(args.buffer_m) > 0:
        site_m = gpd.GeoDataFrame(geometry=[site_union], crs=CRS_WGS).to_crs("EPSG:32651")
        site_union = site_m.buffer(float(args.buffer_m)).to_crs(CRS_WGS).geometry.iloc[0]

    units = gpd.read_file(args.units)
    if units.crs is None:
        units = units.set_crs(CRS_WGS)
    units = units.to_crs(CRS_WGS)

    radar = pd.read_csv(args.radar_csv, encoding="utf-8-sig")
    merged = units.merge(radar, on="unit_id", how="left", suffixes=("", "_rad"))
    scored = zm.build_dimension_scores(merged)
    for c in _DIMS:
        if c not in scored.columns:
            print(f"缺少列 {c}，请先跑 plot_site_zoning_evidence_maps / build_dimension_scores", file=sys.stderr)
            return 1

    scored["inside_site"] = scored.geometry.intersects(site_union)
    n_in = int(scored["inside_site"].sum())
    if n_in == 0:
        print("没有与 SITE 相交的地块，请检查 SITE.json 与 01_units 范围", file=sys.stderr)
        return 1

    sub = scored.loc[scored["inside_site"]].copy()
    sub_n = normalize_dims_within_site(sub)
    sub_z, _zone_keys = zone_affinities(sub_n)

    full = scored.copy()
    merge_cols = [c for c in sub_z.columns if c.startswith("N_") or c.startswith("L_") or c.startswith("fz_")]
    for c in merge_cols:
        if c in ("fz_site_key", "fz_site_zh"):
            full[c] = pd.Series(pd.NA, index=full.index, dtype="string")
        else:
            full[c] = np.nan
    full.loc[sub_z.index, merge_cols] = sub_z[merge_cols]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gpkg_path = args.out_dir / args.gpkg_name

    keep = [
        "unit_id",
        "inside_site",
        *_DIMS,
        "N_dim_station_city",
        "N_dim_business_access",
        "N_dim_public_stay",
        "N_dim_slow_stitch",
        *[f"L_{k}" for k, _, _ in _ZONE_DEFS],
        "fz_site_key",
        "fz_site_zh",
        "fz_site_margin",
    ]
    keep = [c for c in keep if c in full.columns]
    out_gdf = full[keep + ["geometry"]].copy()
    out_gdf.to_file(gpkg_path, driver="GPKG")

    csv_tbl = full.loc[full["inside_site"], keep].copy()
    csv_path = args.out_dir / "site_function_zones_from_dims.csv"
    csv_tbl.drop(columns=[c for c in ("inside_site",) if c in csv_tbl.columns], errors="ignore").to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )

    meta = {
        "method": "Within SITE-intersecting parcels: robust 0–1 normalize each dim (p5–p95); "
        "five linear affinity scores; argmax -> fz_site_zh. Outside SITE: no label.",
        "zone_weights": [
            {
                "key": k,
                "label_zh": zh,
                "weights_on_N_station_business_public_stitch": w.tolist(),
                "note": (
                    "过渡区为 w0*(1-N_station)+w1*(1-N_business)+w2*N_public+w3*N_stitch"
                    if k == "transition"
                    else "亲和度为 sum(N_i * w_i)"
                ),
            }
            for k, zh, w in _ZONE_DEFS
        ],
        "caveats": [
            "消费活力轴在无轴线矢量时用「公共停留+消费相关 dim」代理，与规划「轴」几何不必一致。",
            "五类互斥 argmax 会掩盖混合用地；可看各 L_* 列与 fz_site_margin。",
        ],
        "n_site_parcels": n_in,
        "buffer_m": float(args.buffer_m),
        "outputs": {
            "gpkg": str(gpkg_path.relative_to(REPO)),
            "csv": str(csv_path.relative_to(REPO)),
            "map_png": str((args.out_dir / "site_function_zones_from_dims_map.png").relative_to(REPO)),
        },
    }
    (args.out_dir / "site_function_zones_from_dims_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_site_zones_map(
        units_wm=full.to_crs(CRS_WM),
        site_wm=site.to_crs(CRS_WM),
        out_path=args.out_dir / "site_function_zones_from_dims_map.png",
    )

    print(f"SITE 内地块数: {n_in}；已写入 {gpkg_path}、{csv_path}")
    if n_in < 80 and float(args.buffer_m) <= 0:
        print("提示: 与红线严格相交的地块可能很少，可加 --buffer-m 200~400（米）扩展场地后再算。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
