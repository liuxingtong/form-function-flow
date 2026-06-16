"""
站点周边（约 3km 裁剪范围）房价分布图：合并 ``楼盘`` 与 ``08-房价数据`` 中的 GeoJSON。

输出：
  - 总体一张：网格均价底图 + 小区/楼盘点位叠加 + 场地红线 + 3km 参考圈
  - H00 OSM+SVM：Carto Light（Positron）底图 + RBF-SVM 单价中位数决策边界（南北差异示意）
  - 每种地产类型一张：同范围，仅显示该类型的点位（底图为浅灰网格单价）

仓库根目录运行：
  python scripts/visualize_site_housing_prices.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import Normalize
from shapely.geometry import Point
from shapely import contains_xy as shapely_contains_xy
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parents[1]
SITE_3KM = REPO / "data" / "site_3km"
DEFAULT_OUT = REPO / "output" / "site_housing_prices"
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from site_map_overlay import plot_site_boundary  # noqa: E402

# GCJ-02 → WGS84（火星坐标纠偏）
_PI = math.pi
_A = 6378245.0
_EE = 0.00669342162296594323


def _out_of_china(lng: float, lat: float) -> bool:
    return not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271)


def _transform_lat(lng: float, lat: float) -> float:
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * math.sqrt(abs(lng))
    ret += (20.0 * math.sin(6.0 * lng * _PI) + 20.0 * math.sin(2.0 * lng * _PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * _PI) + 40.0 * math.sin(lat / 3.0 * _PI)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * _PI) + 320 * math.sin(lat * _PI / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lng(lng: float, lat: float) -> float:
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * math.sqrt(abs(lng))
    ret += (20.0 * math.sin(6.0 * lng * _PI) + 20.0 * math.sin(2.0 * lng * _PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * _PI) + 40.0 * math.sin(lng / 3.0 * _PI)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * _PI) + 300.0 * math.sin(lng / 30.0 * _PI)) * 2.0 / 3.0
    return ret


def gcj02_to_wgs84(lng: float, lat: float) -> tuple[float, float]:
    if _out_of_china(lng, lat):
        return lng, lat
    dlat = _transform_lat(lng - 105.0, lat - 35.0)
    dlng = _transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * _PI
    magic = math.sin(radlat)
    magic = 1 - _EE * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((_A * (1 - _EE)) / (magic * sqrtmagic) * _PI)
    dlng = (dlng * 180.0) / (_A / sqrtmagic * math.cos(radlat) * _PI)
    mglat = lat + dlat
    mglng = lng + dlng
    return lng * 2 - mglng, lat * 2 - mglat


def configure_cn_font() -> None:
    preferred = ("Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans")
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in avail:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _parse_price_yuan_per_sqm(val: object) -> float | None:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if isinstance(val, (int, float)):
        v = float(val)
        return v if 3000 <= v <= 600000 else None
    s = str(val).strip()
    if not s or s in ("暂无", "—", "-"):
        return None
    m = re.search(r"([\d.,]+)", s.replace(",", ""))
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return v if 3000 <= v <= 600000 else None


def _site_anchor_and_buffer_gdf(site_json: Path, buffer_m: float) -> tuple[gpd.GeoDataFrame, Point]:
    """场地几何形心 + EPSG:32651 下 buffer_m 米圆（作参考范围）。"""
    sg = gpd.read_file(site_json)
    if sg.crs is None:
        sg = sg.set_crs(4326)
    sg = sg.to_crs(4326)
    geom = unary_union(sg.geometry)
    anchor = geom.centroid
    g3857 = gpd.GeoDataFrame(geometry=[anchor], crs="EPSG:4326").to_crs(32651)
    disk = g3857.geometry.iloc[0].buffer(buffer_m)
    buf = gpd.GeoDataFrame(geometry=[disk], crs="EPSG:32651").to_crs(4326)
    return buf, anchor


def _resolve_paths(site_root: Path) -> dict[str, Path]:
    loupan = site_root / "楼盘"
    if not loupan.is_dir():
        loupan = next(p for p in site_root.iterdir() if p.is_dir() and "楼盘" in p.name)
    h08 = site_root / "08-房价数据"
    if not h08.is_dir():
        h08 = next(p for p in site_root.iterdir() if p.is_dir() and "08-" in p.name and "房价" in p.name)

    second = next(loupan.glob("*二手房*.geojson"))
    newp = next(loupan.glob("*新楼盘*.geojson"))
    xiaoqu = next(h08.rglob("*房价_WGS84.geojson"))
    grid_dir = h08 / "网格房价" / "geojson"
    grid = grid_dir / "上海房价数据.geojson"
    if not grid.is_file():
        grid = next(h08.rglob("上海房价数据.geojson"))

    return {"二手房": second, "新楼盘": newp, "小区房价_WGS84": xiaoqu, "网格": grid}


def load_points_frames(paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict] = []

    # 二手房（WGS84）
    g = gpd.read_file(paths["二手房"])
    if g.crs is None:
        g = g.set_crs(4326)
    for _, r in g.iterrows():
        lng = float(r["wgs84lng"])
        lat = float(r["wgs84lat"])
        price = _parse_price_yuan_per_sqm(r.get("均价"))
        pt = str(r.get("物业类型") or "").strip() or "未知"
        if price is None:
            continue
        rows.append(
            {
                "lng": lng,
                "lat": lat,
                "price": price,
                "prop_type": pt,
                "layer": "楼盘·二手房小区",
            }
        )

    # 新楼盘（GCJ-02 → WGS84）
    g = gpd.read_file(paths["新楼盘"])
    if g.crs is None:
        g = g.set_crs(4326)
    for _, r in g.iterrows():
        glng = float(r["gcj02_lng"])
        glat = float(r["gcj02_lat"])
        lng, lat = gcj02_to_wgs84(glng, glat)
        price = _parse_price_yuan_per_sqm(r.get("参考价格"))
        pt = str(r.get("楼盘属性") or "").strip() or "未知"
        if price is None:
            continue
        rows.append(
            {
                "lng": lng,
                "lat": lat,
                "price": price,
                "prop_type": pt,
                "layer": "楼盘·新房",
            }
        )

    # 08 小区房价点（WGS84）
    g = gpd.read_file(paths["小区房价_WGS84"])
    if g.crs is None:
        g = g.set_crs(4326)
    key_price = "小区（"
    key_type = "物业类"
    for _, r in g.iterrows():
        lng = float(r["POINT_X"])
        lat = float(r["POINT_Y"])
        price = _parse_price_yuan_per_sqm(r.get(key_price))
        if price is None:
            price = _parse_price_yuan_per_sqm(r.get("小区均"))
        pt = str(r.get(key_type) or "").strip() or "未知"
        if price is None:
            continue
        rows.append(
            {
                "lng": lng,
                "lat": lat,
                "price": price,
                "prop_type": pt,
                "layer": "08·小区房价点位",
            }
        )

    return pd.DataFrame(rows)


def load_grid(paths: dict[str, Path]) -> gpd.GeoDataFrame:
    grid = gpd.read_file(paths["网格"])
    if grid.crs is None:
        grid = grid.set_crs(4326)
    return grid.to_crs(4326)


def _slug_filename(s: str) -> str:
    out = re.sub(r'[\\/:*?"<>|]+', "_", s.strip())
    return out[:120] if len(out) > 120 else out


def plot_one(
    *,
    title: str,
    out_path: Path,
    grid_gdf: gpd.GeoDataFrame,
    pts: pd.DataFrame | None,
    buf_wgs84: gpd.GeoDataFrame,
    vmin: float,
    vmax: float,
    cmap: str,
    grid_column: str | None,
    grid_uniform_facecolor: str | None,
    grid_alpha: float,
    grid_edgecolor: str | None,
    show_points: bool,
    extent_pad_deg: float | None = None,
    extent_bounds: tuple[float, float, float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 11))
    ax.set_aspect("equal")

    if not grid_gdf.empty:
        if grid_uniform_facecolor is not None:
            grid_gdf.plot(
                ax=ax,
                facecolor=grid_uniform_facecolor,
                edgecolor=grid_edgecolor or "#cccccc",
                linewidth=0.12,
                alpha=grid_alpha,
            )
        elif grid_column:
            grid_gdf.plot(
                ax=ax,
                column=grid_column,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidth=0.15,
                edgecolor=grid_edgecolor or "none",
                alpha=grid_alpha,
                legend=False,
            )

    if show_points and pts is not None and not pts.empty:
        norm = Normalize(vmin=vmin, vmax=vmax)
        ax.scatter(
            pts["lng"],
            pts["lat"],
            c=pts["price"],
            cmap=cmap,
            norm=norm,
            s=28,
            alpha=0.92,
            edgecolors="white",
            linewidths=0.35,
            zorder=5,
        )

    buf_wgs84.plot(ax=ax, facecolor="none", edgecolor="#666666", linestyle=(0, (6, 4)), linewidth=1.2, zorder=8)
    plot_site_boundary(ax, match_crs=None)

    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.62, pad=0.02)
    cbar.set_label("单价（元/㎡）")

    ax.set_xlabel("经度（°）")
    ax.set_ylabel("纬度（°）")
    ax.set_title(title, fontsize=13, pad=12)
    if extent_bounds is not None:
        minx, miny, maxx, maxy = extent_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
    elif extent_pad_deg is not None and not grid_gdf.empty:
        bx, by = grid_gdf.total_bounds[0], grid_gdf.total_bounds[1]
        bx2, by2 = grid_gdf.total_bounds[2], grid_gdf.total_bounds[3]
        ax.set_xlim(bx - extent_pad_deg, bx2 + extent_pad_deg)
        ax.set_ylim(by - extent_pad_deg, by2 + extent_pad_deg)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_overview_osm_svm(
    *,
    pts_wgs: pd.DataFrame,
    grid_3857: gpd.GeoDataFrame,
    buf_wgs84: gpd.GeoDataFrame,
    site_json: Path,
    vmin: float,
    vmax: float,
    out_path: Path,
    buffer_m: float,
    median_price: float,
    cmap: str = "viridis",
) -> bool:
    """Carto Light（Positron）底图 + RBF-SVM 决策边界（特征：平面坐标；标签：单价相对样本中位数的高低）。"""
    try:
        import contextily as ctx
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError:
        return False

    if pts_wgs.empty or len(pts_wgs) < 40:
        return False

    pts_g = gpd.GeoDataFrame(
        pts_wgs.copy(),
        geometry=gpd.points_from_xy(pts_wgs["lng"], pts_wgs["lat"]),
        crs="EPSG:4326",
    )
    pts_m = pts_g.to_crs(3857)
    mx = pts_m.geometry.x.to_numpy()
    my = pts_m.geometry.y.to_numpy()
    y_bin = (pts_wgs["price"].to_numpy() >= median_price).astype(int)
    if y_bin.min() == y_bin.max():
        return False

    X = np.column_stack([mx, my])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = SVC(kernel="rbf", gamma="scale", C=1.0, class_weight="balanced")
    clf.fit(Xs, y_bin)

    buf3857_geom = buf_wgs84.to_crs(3857).geometry.iloc[0]
    b3857 = buf_wgs84.to_crs(3857).total_bounds
    pad_m = 120.0
    n_mesh = 260
    xs = np.linspace(b3857[0] - pad_m, b3857[2] + pad_m, n_mesh)
    ys = np.linspace(b3857[1] - pad_m, b3857[3] + pad_m, n_mesh)
    XM, YM = np.meshgrid(xs, ys)
    XYm = np.column_stack([XM.ravel(), YM.ravel()])
    XYs = scaler.transform(XYm)
    Z = clf.decision_function(XYs).reshape(XM.shape)
    inside = shapely_contains_xy(buf3857_geom, XM.ravel(), YM.ravel()).reshape(XM.shape)
    Z_masked = np.where(inside, Z, np.nan)

    fig, ax = plt.subplots(figsize=(12, 11))

    ax.set_xlim(b3857[0] - pad_m, b3857[2] + pad_m)
    ax.set_ylim(b3857[1] - pad_m, b3857[3] + pad_m)
    ax.set_aspect("equal")

    basemap_ok = False
    basemap_label = ""
    # Carto Light = Positron（light_all）；优先浅灰街廓，失败再退回 OSM
    tile_try: list[tuple[str, object]] = []
    try:
        tile_try.append(("Carto Light (Positron)", ctx.providers.CartoDB.Positron))
    except AttributeError:
        pass
    try:
        tile_try.append(("Carto Light · 无注记", ctx.providers.CartoDB.PositronNoLabels))
    except AttributeError:
        pass
    try:
        tile_try.append(("OpenStreetMap", ctx.providers.OpenStreetMap.Mapnik))
    except AttributeError:
        pass
    for label, src in tile_try:
        try:
            ctx.add_basemap(ax, crs="EPSG:3857", source=src, attribution_size=6)
            basemap_ok = True
            basemap_label = label
            break
        except Exception:
            continue
    if not basemap_ok:
        ax.set_facecolor("#e8ecf0")

    if not grid_3857.empty:
        grid_3857.plot(
            ax=ax,
            column="avgprice",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidth=0.12,
            edgecolor="#ffffff",
            alpha=0.42,
            legend=False,
            zorder=2,
        )

    norm = Normalize(vmin=vmin, vmax=vmax)
    ax.scatter(
        mx,
        my,
        c=pts_wgs["price"],
        cmap=cmap,
        norm=norm,
        s=22,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.3,
        zorder=5,
    )

    # SVM 等高线：decision_function = 0（RBF 非线性分界）
    ax.contour(
        XM,
        YM,
        Z_masked,
        levels=[0.0],
        colors=["#00e5ff"],
        linewidths=2.8,
        linestyles="solid",
        zorder=12,
    )

    buf_wgs84.to_crs(3857).plot(
        ax=ax,
        facecolor="none",
        edgecolor="#555555",
        linestyle=(0, (6, 4)),
        linewidth=1.3,
        zorder=14,
    )
    plot_site_boundary(ax, match_crs="EPSG:3857", site_path=site_json)

    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.62, pad=0.02)
    cbar.set_label("单价（元/㎡）")

    hi_lat = float(pts_wgs.loc[y_bin == 1, "lat"].mean())
    lo_lat = float(pts_wgs.loc[y_bin == 0, "lat"].mean())
    richer_side = "北侧" if hi_lat > lo_lat else "南侧"

    ttl = (
        f"场地周边房价分布（总体 · Carto Light 底图 + SVM 分界）· 参考 {buffer_m:.0f}m\n"
        f"标签：单价≥中位数 {median_price:,.0f} 元/㎡ 为高价类 · "
        f"高价样本平均更偏{richer_side}（纬度均值对比）"
    )
    ax.set_title(ttl, fontsize=11, pad=10)
    ax.set_xlabel("Web 墨卡托 x（m）")
    ax.set_ylabel("Web 墨卡托 y（m）")

    if basemap_ok:
        if "Carto" in basemap_label or "Positron" in basemap_label:
            attr_txt = "底图: Carto Light (Positron) · © OpenStreetMap contributors · © CARTO"
        else:
            attr_txt = "底图: © OpenStreetMap contributors"
    else:
        attr_txt = "无底图（contextily 不可用或下载失败）"
    ax.text(
        0.01,
        0.02,
        attr_txt + "\n曲线：RBF-SVM decision_function=0；仅在 3km 缓冲区内绘制",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
        zorder=20,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return True


def main() -> None:
    configure_cn_font()
    ap = argparse.ArgumentParser(description="站点周边房价分布图（楼盘 + 08-房价数据）")
    ap.add_argument("--site-root", type=Path, default=SITE_3KM, help="含 SITE.json、楼盘、08-房价数据的目录")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--buffer-m", type=float, default=3000.0, help="参考圈半径（米），与 manifest 一致")
    ap.add_argument("--skip-osm-svm", action="store_true", help="不生成 OSM+SVM 总体图（免联网）")
    args = ap.parse_args()

    site_json = args.site_root / "SITE.json"
    if not site_json.is_file():
        raise SystemExit(f"缺少 {site_json}")

    paths = _resolve_paths(args.site_root)
    df = load_points_frames(paths)
    grid = load_grid(paths)

    buf_wgs, anchor = _site_anchor_and_buffer_gdf(site_json, args.buffer_m)
    buf3857 = buf_wgs.to_crs(3857)

    # 限制在缓冲区内（数据虽已裁剪，仍做一次相交以保证图幅一致）
    grid_m = grid.to_crs(3857)
    buf_union = buf3857.geometry.iloc[0]
    grid_clip = grid_m[grid_m.intersects(buf_union)].copy()
    grid_plot = grid_clip.to_crs(4326)

    inside = df.copy()
    pts_g = gpd.GeoDataFrame(inside, geometry=gpd.points_from_xy(inside["lng"], inside["lat"]), crs="EPSG:4326")
    pts_m = pts_g.to_crs(3857)
    sel = pts_m.within(buf_union)
    pts_wgs = inside.loc[sel.values].reset_index(drop=True)

    # 图幅范围略大于缓冲区，便于阅读（仍使用 WGS84 绘制）
    bb = buf_wgs.total_bounds
    pad_deg = 0.002
    extent_bounds = (bb[0] - pad_deg, bb[1] - pad_deg, bb[2] + pad_deg, bb[3] + pad_deg)

    prices = np.concatenate(
        [
            pts_wgs["price"].values.astype(float),
            grid_clip["avgprice"].dropna().values.astype(float),
        ]
    )
    p_lo, p_hi = np.percentile(prices, [2, 98])
    vmin, vmax = float(p_lo), float(p_hi)
    median_price = float(np.median(pts_wgs["price"].values))

    cmap = "viridis"

    # —— 总体（经典白底）
    plot_one(
        title=f"场地周边房价分布（总体）· 参考 {args.buffer_m:.0f}m 范围圈\n"
        f"底图：网格均价；点：二手房 / 新房 / 小区房价点位（n={len(pts_wgs)}）",
        out_path=args.out_dir / "H00_房价分布_总体.png",
        grid_gdf=grid_plot,
        pts=pts_wgs,
        buf_wgs84=buf_wgs,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        grid_column="avgprice",
        grid_uniform_facecolor=None,
        grid_alpha=0.55,
        grid_edgecolor="#ffffff",
        show_points=True,
        extent_bounds=extent_bounds,
    )

    # —— 总体（OSM + SVM 南北差异分界线示意）
    if not args.skip_osm_svm:
        svm_path = args.out_dir / "H00_房价分布_总体_OSMSVM.png"
        ok = plot_overview_osm_svm(
            pts_wgs=pts_wgs,
            grid_3857=grid_clip,
            buf_wgs84=buf_wgs,
            site_json=site_json,
            vmin=vmin,
            vmax=vmax,
            out_path=svm_path,
            buffer_m=args.buffer_m,
            median_price=median_price,
            cmap=cmap,
        )
        if ok:
            print(f"Wrote OSM+SVM overview: {svm_path}")
        else:
            print("Skipped OSM+SVM overview (missing deps, too few points, or single class).")

    # —— 分类型（仅点位；类型来自合并字段 prop_type）
    types = sorted(pts_wgs["prop_type"].dropna().unique())

    for i, pt in enumerate(types):
        sub = pts_wgs[pts_wgs["prop_type"] == pt]
        if sub.empty:
            continue
        fn = f"H{i + 1:02d}_房价分布_{_slug_filename(pt)}.png"
        plot_one(
            title=f"场地周边房价分布 · {pt}\n（{args.buffer_m:.0f}m 参考圈；n={len(sub)}）",
            out_path=args.out_dir / fn,
            grid_gdf=grid_plot,
            pts=sub,
            buf_wgs84=buf_wgs,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            grid_column=None,
            grid_uniform_facecolor="#e8e8e8",
            grid_alpha=0.42,
            grid_edgecolor="#cccccc",
            show_points=True,
            extent_bounds=extent_bounds,
        )

    meta = {
        "out_dir": str(args.out_dir),
        "n_points_total": int(len(pts_wgs)),
        "n_grid_cells": int(len(grid_plot)),
        "price_vmin_vmax_percentile_2_98": [vmin, vmax],
        "median_price_yuan_per_sqm": median_price,
        "property_types": types,
        "anchor_lon_lat": [float(anchor.x), float(anchor.y)],
        "overview_osm_svm_png": str((args.out_dir / "H00_房价分布_总体_OSMSVM.png").resolve())
        if not args.skip_osm_svm
        else None,
    }
    (args.out_dir / "housing_price_maps_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote maps to {args.out_dir}")


if __name__ == "__main__":
    main()
