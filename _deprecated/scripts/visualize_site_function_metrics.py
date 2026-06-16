"""
站点周边功能指标 OSM 底图 + 地块热力 + RBF-SVM 中位数分界（与房价 H00 OSMSVM 同构）。

数据源：
  - ``output/function/func_state.csv``：按 ``unit_id`` 对八时段取均值
  - ``output/function/数据包/01_units.gpkg``：地块几何与质心

默认输出三张（文件名 ASCII，避免 Windows 编码问题）：
  - Func01_avg_rating_OSMSVM.png
  - Func02_poi_density_OSMSVM.png
  - Func03_service_accessibility_OSMSVM.png

仓库根目录运行：
  python scripts/visualize_site_function_metrics.py
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
from shapely import contains_xy as shapely_contains_xy

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FUNC_CSV = REPO / "output" / "function" / "func_state.csv"
DEFAULT_UNITS = REPO / "output" / "function" / "数据包" / "01_units.gpkg"
DEFAULT_SITE = REPO / "data" / "site_3km" / "SITE.json"
DEFAULT_OUT = REPO / "output" / "function" / "metric_maps_osm_svm"

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from site_map_overlay import plot_site_boundary  # noqa: E402


def configure_cn_font() -> None:
    preferred = ("Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans")
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in avail:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _site_buffer_wgs84(site_json: Path, buffer_m: float) -> gpd.GeoDataFrame:
    from shapely.ops import unary_union

    sg = gpd.read_file(site_json)
    if sg.crs is None:
        sg = sg.set_crs(4326)
    sg = sg.to_crs(4326)
    geom = unary_union(sg.geometry)
    anchor = geom.centroid
    g3857 = gpd.GeoDataFrame(geometry=[anchor], crs="EPSG:4326").to_crs(32651)
    disk = g3857.geometry.iloc[0].buffer(buffer_m)
    return gpd.GeoDataFrame(geometry=[disk], crs="EPSG:32651").to_crs(4326)


def load_units_with_metrics(
    func_csv: Path,
    units_gpkg: Path,
    metric_cols: list[str],
) -> gpd.GeoDataFrame:
    df = pd.read_csv(func_csv)
    miss = [c for c in metric_cols if c not in df.columns]
    if miss:
        raise SystemExit(f"func_state 缺少列: {miss}")

    agg = df.groupby("unit_id", as_index=False)[metric_cols].mean()
    units = gpd.read_file(units_gpkg)
    if units.crs is None:
        units = units.set_crs(4326)
    units = units.to_crs(4326)
    merged = units.merge(agg, on="unit_id", how="inner")
    return merged


def plot_metric_osm_svm(
    *,
    units_buf_wgs: gpd.GeoDataFrame,
    buf_wgs84: gpd.GeoDataFrame,
    site_json: Path,
    column: str,
    title_metric_zh: str,
    cbar_label: str,
    out_path: Path,
    buffer_m: float,
    cmap: str = "viridis",
) -> bool:
    try:
        import contextily as ctx
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError:
        return False

    if units_buf_wgs.empty or len(units_buf_wgs) < 40:
        return False

    vals = pd.to_numeric(units_buf_wgs[column], errors="coerce")
    if vals.notna().sum() < 40:
        return False

    units_plot = units_buf_wgs.loc[vals.notna()].copy()
    v = units_plot[column].to_numpy(dtype=float)
    median_v = float(np.median(v))
    y_bin = (v >= median_v).astype(int)
    if y_bin.min() == y_bin.max():
        return False

    # 色标：缓冲区内数值 2–98 分位
    p_lo, p_hi = np.percentile(v, [2, 98])
    vmin, vmax = float(p_lo), float(p_hi)
    if vmax <= vmin:
        vmax = vmin + 1e-9

    pts_g = gpd.GeoDataFrame(
        units_plot,
        geometry=gpd.points_from_xy(units_plot["centroid_x"], units_plot["centroid_y"]),
        crs="EPSG:4326",
    )
    pts_m = pts_g.to_crs(3857)
    mx = pts_m.geometry.x.to_numpy()
    my = pts_m.geometry.y.to_numpy()

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

    units_3857 = units_plot.to_crs(3857)
    units_3857.plot(
        ax=ax,
        column=column,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidth=0.1,
        edgecolor="#ffffff",
        alpha=0.42,
        legend=False,
        zorder=2,
    )

    norm = Normalize(vmin=vmin, vmax=vmax)
    ax.scatter(
        mx,
        my,
        c=v,
        cmap=cmap,
        norm=norm,
        s=18,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.28,
        zorder=5,
    )

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
    cbar.set_label(cbar_label)

    # 与房价脚本一致：用纬度均值对比提示南北侧（字段值高低）
    lat_arr = units_plot["centroid_y"].to_numpy()
    hi_lat = float(np.mean(lat_arr[y_bin == 1]))
    lo_lat = float(np.mean(lat_arr[y_bin == 0]))
    hi_side = "北侧" if hi_lat > lo_lat else "南侧"

    ttl = (
        f"场地周边{title_metric_zh}（Carto Light 底图 + SVM 分界）· 参考 {buffer_m:.0f}m\n"
        f"标签：{title_metric_zh}≥中位数 {median_v:.6g} 为高值类 · "
        f"高值样本平均更偏{hi_side}（纬度均值对比）"
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
        attr_txt + "\n地块：01_units.gpkg；指标：func_state 八时段均值\n"
        "曲线：RBF-SVM decision_function=0；仅在缓冲区内绘制",
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
    ap = argparse.ArgumentParser(description="功能指标 OSM + SVM 地图（地块 + func_state）")
    ap.add_argument("--func-csv", type=Path, default=DEFAULT_FUNC_CSV)
    ap.add_argument("--units-gpkg", type=Path, default=DEFAULT_UNITS)
    ap.add_argument("--site-json", type=Path, default=DEFAULT_SITE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--buffer-m", type=float, default=3000.0)
    args = ap.parse_args()

    if not args.site_json.is_file():
        raise SystemExit(f"缺少 {args.site_json}")
    if not args.func_csv.is_file():
        raise SystemExit(f"缺少 {args.func_csv}")
    if not args.units_gpkg.is_file():
        raise SystemExit(f"缺少 {args.units_gpkg}")

    metric_defs: list[tuple[str, str, str, str]] = [
        ("avg_rating", "平均评分", "平均评分（分）", "Func01_avg_rating_OSMSVM.png"),
        ("poi_density", "POI 密度", "POI 密度（相对，地块口径）", "Func02_poi_density_OSMSVM.png"),
        (
            "service_accessibility",
            "公共服务可达性",
            "公共服务可达性（相对）",
            "Func03_service_accessibility_OSMSVM.png",
        ),
    ]
    cols = [m[0] for m in metric_defs]
    units_merged = load_units_with_metrics(args.func_csv, args.units_gpkg, cols)

    buf_wgs = _site_buffer_wgs84(args.site_json, args.buffer_m)
    buf_union = buf_wgs.to_crs(3857).geometry.iloc[0]
    u_m = units_merged.to_crs(3857)
    cen = gpd.GeoDataFrame(units_merged, geometry=gpd.points_from_xy(u_m["centroid_x"], u_m["centroid_y"]), crs="EPSG:4326").to_crs(3857)
    inside = cen.geometry.within(buf_union)
    units_buf = units_merged.loc[inside.values].reset_index(drop=True)

    meta: dict = {
        "out_dir": str(args.out_dir),
        "n_units_in_buffer": int(len(units_buf)),
        "buffer_m": args.buffer_m,
        "metrics": {},
    }

    for column, title_zh, cbar_label, fname in metric_defs:
        out_png = args.out_dir / fname
        ok = plot_metric_osm_svm(
            units_buf_wgs=units_buf,
            buf_wgs84=buf_wgs,
            site_json=args.site_json,
            column=column,
            title_metric_zh=title_zh,
            cbar_label=cbar_label,
            out_path=out_png,
            buffer_m=args.buffer_m,
        )
        v = pd.to_numeric(units_buf[column], errors="coerce").dropna()
        median_v = float(np.median(v)) if len(v) else float("nan")
        if len(v) >= 5:
            p2, p98 = (float(x) for x in np.percentile(v, [2, 98]))
        else:
            p2, p98 = float("nan"), float("nan")
        meta["metrics"][column] = {
            "png": str(out_png.resolve()) if ok else None,
            "written": ok,
            "median": median_v,
            "p02_p98": [p2, p98],
            "n_valid": int(len(v)),
        }
        if ok:
            print(f"Wrote {out_png}")
        else:
            print(f"Skipped {column} (deps / too few units / single class).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "function_metric_osm_svm_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Meta -> {args.out_dir / 'function_metric_osm_svm_meta.json'}")


if __name__ == "__main__":
    main()
