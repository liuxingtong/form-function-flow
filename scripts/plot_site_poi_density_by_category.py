"""
按 POI 大类（或地块规划用地类别）输出多张图：在 ``01_units.gpkg`` 地块上填色 **POI 密度**
（点数 / 地块面积，单位：个/平方米），底图使用 Carto Light（contextily：CartoDB Positron）。

数据来源优先使用 ``02-POI&AOI/1-POI/25.05/CSV/分类/按类别/*.csv``（每个文件一个大类），
避免在规则网格上画密度；若该目录无 CSV，则尝试同路径下的 ``*.shp`` / ``*.geojson``。

运行（仓库根目录）:
  python scripts/plot_site_poi_density_by_category.py
  python scripts/plot_site_poi_density_by_category.py ^
    --poi-root "F:/Aworks/2026studio/shanghaistation/all/02-POI&AOI/1-POI/25.05/CSV/分类/按类别"
  python scripts/plot_site_poi_density_by_category.py --split-by parcel_landuse

``--poi-root`` 可为 ``…/CSV/分类`` 或直接进入 ``…/CSV/分类/按类别`` 文件夹。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import Normalize
from shapely.geometry import Polygon

REPO = Path(__file__).resolve().parents[1]
SITE_3KM = REPO / "data" / "site_3km"
CRS_WGS = "EPSG:4326"
CRS_WM = "EPSG:3857"
CRS_METRIC = "EPSG:32651"  # Shanghai UTM — 与 build_site_units_and_edges 一致

LON_CANDIDATES = ("wgs84Lng", "wgs84_lon", "WGS84_Lng", "longitude", "lng", "lon", "x", "LON", "经度")
LAT_CANDIDATES = ("wgs84Lat", "wgs84_lat", "WGS84_Lat", "latitude", "lat", "y", "LAT", "纬度")


def resolve_poi_classification_dirs(poi_root: Path) -> tuple[Path, Path, Path]:
    """
    解析 POI 输入路径。

    返回 (按类别目录, 按区县目录, 分类根目录)。
    ``poi_root`` 可以是 ``…/CSV/分类``，也可以直接是 ``…/CSV/分类/按类别``。
    """
    root = poi_root.expanduser().resolve()
    if root.name == "按类别" and root.is_dir():
        by_type = root
        classification_base = root.parent
        by_dist = classification_base / "按区县"
        return by_type, by_dist, classification_base
    by_type = root / "按类别"
    by_dist = root / "按区县"
    return by_type, by_dist, root


def configure_cn_font() -> None:
    preferred = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans")
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_site_polygon(site_path: Path) -> gpd.GeoDataFrame:
    site = gpd.read_file(site_path)
    if site.crs is None:
        site = site.set_crs(CRS_WGS)
    site = site.to_crs(CRS_WGS)
    geom = site.geometry.iloc[0]
    if geom.geom_type == "LineString":
        coords = list(geom.coords)
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        geom = Polygon(coords)
    return gpd.GeoDataFrame({"name": ["SITE"]}, geometry=[geom], crs=CRS_WGS)


def padded_bounds(gdf_wm: gpd.GeoDataFrame, pad_ratio: float = 0.06) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = gdf_wm.total_bounds
    dx = (xmax - xmin) * pad_ratio
    dy = (ymax - ymin) * pad_ratio
    return xmin - dx, ymin - dy, xmax + dx, ymax + dy


def add_carto_light_basemap(ax, bounds_wm: tuple[float, float, float, float]) -> tuple[bool, str]:
    xmin, ymin, xmax, ymax = bounds_wm
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    try:
        import contextily as ctx
    except ImportError:
        return False, "contextily 未安装"
    for src in (
        getattr(ctx.providers.CartoDB, "Positron", None),
        getattr(ctx.providers.CartoDB, "PositronNoLabels", None),
    ):
        if src is None:
            continue
        try:
            ctx.add_basemap(ax, crs=CRS_WM, source=src, zoom="auto", attribution_size=6)
            return True, "Carto Light (Positron)"
        except Exception:
            continue
    return False, "底图下载失败"


def _pick_xy_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    lower = {c.lower(): c for c in df.columns}
    lon_col = None
    lat_col = None
    for cand in LON_CANDIDATES:
        if cand in df.columns:
            lon_col = cand
            break
        if cand.lower() in lower:
            lon_col = lower[cand.lower()]
            break
    for cand in LAT_CANDIDATES:
        if cand in df.columns:
            lat_col = cand
            break
        if cand.lower() in lower:
            lat_col = lower[cand.lower()]
            break
    if lon_col and lat_col:
        return lon_col, lat_col
    return None


def _read_table(path: Path) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-8", "gbk", "gb18030")
    last_err: Exception | None = None
    if path.suffix.lower() == ".csv":
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except Exception as e:
                last_err = e
        raise RuntimeError(f"无法读取 CSV {path}: {last_err}")
    raise ValueError(f"不支持的表格格式: {path}")


def category_name_from_poi_filename(path: Path) -> str:
    stem = path.stem
    for prefix in ("上海市-", "上海市2025-1343026.csv-", "上海市-POI-2023.csv-", "Poidata-2025-"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
    return stem.strip() or path.stem


def load_poi_points_from_csv(path: Path, category_label: str) -> gpd.GeoDataFrame:
    df = _read_table(path)
    cols = _pick_xy_columns(df)
    if cols is None:
        raise ValueError(f"{path}: 未找到经纬度列（尝试 {LON_CANDIDATES[:4]}…）")
    lon_c, lat_c = cols
    sub = df[[lon_c, lat_c]].copy()
    sub["lon"] = pd.to_numeric(sub[lon_c], errors="coerce")
    sub["lat"] = pd.to_numeric(sub[lat_c], errors="coerce")
    sub = sub[sub["lon"].between(-180, 180) & sub["lat"].between(-90, 90)]
    sub = sub[sub["lon"].between(120.0, 122.5) & sub["lat"].between(30.5, 32.0)]
    if sub.empty:
        return gpd.GeoDataFrame({"poi_category": []}, geometry=gpd.GeoSeries([], crs=CRS_WGS), crs=CRS_WGS)
    n = len(sub)
    g = gpd.GeoDataFrame(
        {"poi_category": [category_label] * n},
        geometry=gpd.points_from_xy(sub["lon"], sub["lat"]),
        crs=CRS_WGS,
    )
    return g


def load_poi_points_from_vector(path: Path, category_label: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs(CRS_WGS)
    g = g.to_crs(CRS_WGS)
    g = g[g.geometry.notna()]
    g = g[g.geom_type.isin(["Point", "MultiPoint"])]
    if g.empty:
        return gpd.GeoDataFrame({"poi_category": []}, geometry=[], crs=CRS_WGS)
    g = g.explode(index_parts=False, ignore_index=True)
    g = g[g.geom_type == "Point"]
    g["poi_category"] = category_label
    return g[["poi_category", "geometry"]]


def discover_poi_category_inputs(poi_root: Path) -> list[tuple[str, Path]]:
    """返回 (类别名, 文件路径) 列表，按类别名排序。"""
    out: list[tuple[str, Path]] = []
    by_type, _, classification_base = resolve_poi_classification_dirs(poi_root)
    if by_type.is_dir():
        for p in sorted(by_type.glob("*.csv")):
            out.append((category_name_from_poi_filename(p), p))
        if not out:
            for ext in ("*.shp", "*.geojson", "*.gpkg"):
                for p in sorted(by_type.glob(ext)):
                    out.append((category_name_from_poi_filename(p), p))
    if not out:
        combined = list(classification_base.glob("**/Poidata*.csv")) + list(
            classification_base.glob("**/上海市*.csv")
        )
        seen: set[Path] = set()
        for p in sorted(combined, key=lambda x: str(x)):
            if p in seen or "按区县" in p.parts:
                continue
            seen.add(p)
            if "按类别" not in p.parts and "分类" not in p.parts:
                continue
            out.append((category_name_from_poi_filename(p), p))
    return out


def load_all_pois_flat(poi_root: Path) -> tuple[gpd.GeoDataFrame, list[str]]:
    """合并所有「按类别」点层；若只有按区县 CSV，则整表带 bigType 列。"""
    warnings: list[str] = []
    parts: list[gpd.GeoDataFrame] = []
    by_type, by_dist, _ = resolve_poi_classification_dirs(poi_root)
    if by_type.is_dir() and list(by_type.glob("*.csv")):
        for cat, p in discover_poi_category_inputs(poi_root):
            try:
                parts.append(load_poi_points_from_csv(p, cat))
            except Exception as e:
                warnings.append(f"{p.name}: {e}")
        if parts:
            merged = pd.concat(parts, ignore_index=True)
            return gpd.GeoDataFrame(merged, geometry=merged.geometry, crs=CRS_WGS), warnings
    # 按区县单表（含 bigType）
    if by_dist.is_dir():
        for p in sorted(by_dist.glob("*.csv")):
            try:
                df = _read_table(p)
                cols = _pick_xy_columns(df)
                if cols is None:
                    continue
                lon_c, lat_c = cols
                bt = "bigType"
                if bt not in df.columns:
                    bt = next((c for c in df.columns if str(c).lower() in ("bigtype", "大类", "一级类")), None)
                if bt is None:
                    warnings.append(f"{p.name}: 无 bigType，跳过按大类拆分")
                    continue
                df = df.assign(
                    lon=pd.to_numeric(df[lon_c], errors="coerce"),
                    lat=pd.to_numeric(df[lat_c], errors="coerce"),
                    poi_cat=df[bt].astype(str),
                )
                df = df[df["lon"].between(120.0, 122.5) & df["lat"].between(30.5, 32.0)]
                g = gpd.GeoDataFrame(
                    df[["poi_cat"]].rename(columns={"poi_cat": "poi_category"}),
                    geometry=gpd.points_from_xy(df["lon"], df["lat"]),
                    crs=CRS_WGS,
                )
                parts.append(g)
            except Exception as e:
                warnings.append(f"{p.name}: {e}")
    if parts:
        merged = pd.concat(parts, ignore_index=True)
        return gpd.GeoDataFrame(merged, geometry=merged.geometry, crs=CRS_WGS), warnings
    # vector fallbacks under 按类别
    for cat, p in discover_poi_category_inputs(poi_root):
        if p.suffix.lower() == ".csv":
            continue
        try:
            parts.append(load_poi_points_from_vector(p, cat))
        except Exception as e:
            warnings.append(f"{p.name}: {e}")
    if parts:
        merged = pd.concat(parts, ignore_index=True)
        return gpd.GeoDataFrame(merged, geometry=merged.geometry, crs=CRS_WGS), warnings
    return gpd.GeoDataFrame({"poi_category": []}, geometry=[], crs=CRS_WGS), warnings + ["未找到可用 POI 数据源"]


def counts_per_unit(pois_m: gpd.GeoDataFrame, units_m: gpd.GeoDataFrame, predicate: str = "intersects") -> pd.Series:
    """unit_id -> 点数（左：点，右：面）。"""
    if pois_m.empty:
        return pd.Series(dtype=int)
    sj = gpd.sjoin(
        pois_m[["geometry"]],
        units_m[["unit_id", "geometry"]],
        how="inner",
        predicate=predicate,
    )
    return sj.groupby("unit_id").size()


def plot_one_choropleth(
    *,
    units_wm: gpd.GeoDataFrame,
    site_wm: gpd.GeoDataFrame,
    column: str,
    title: str,
    out_path: Path,
    vmax_q: float,
) -> None:
    vals = pd.to_numeric(units_wm[column], errors="coerce").fillna(0.0)
    vmax = float(vals.quantile(vmax_q)) if vals.max() > 0 else 1.0
    vmax = max(vmax, 1e-9)
    norm = Normalize(vmin=0.0, vmax=vmax)

    bounds = padded_bounds(units_wm)
    fig, ax = plt.subplots(figsize=(11, 10), dpi=200)
    ok, bm_label = add_carto_light_basemap(ax, bounds)
    units_wm.plot(
        ax=ax,
        column=column,
        cmap="YlOrRd",
        norm=norm,
        linewidth=0.25,
        edgecolor="#555555",
        alpha=0.88,
        legend=True,
        legend_kwds={"shrink": 0.55, "label": "POI 密度（个/平方米）"},
        zorder=3,
    )
    site_wm.boundary.plot(ax=ax, color="#111111", linewidth=2.0, zorder=5)
    ax.set_title(title, fontsize=13, pad=10)
    if not ok:
        ax.text(0.02, 0.98, bm_label, transform=ax.transAxes, va="top", fontsize=8, color="#666666")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    configure_cn_font()
    ap = argparse.ArgumentParser(description="SITE 3km 地块 POI 密度分图（地块填色，Carto Light 底图）")
    ap.add_argument("--site-json", type=Path, default=SITE_3KM / "SITE.json")
    ap.add_argument("--units", type=Path, default=SITE_3KM / "01_units.gpkg")
    ap.add_argument(
        "--poi-root",
        type=Path,
        default=SITE_3KM / "02-POI&AOI" / "1-POI" / "25.05" / "CSV" / "分类",
        help="POI 路径：可为 …/CSV/分类 或 …/CSV/分类/按类别（后者即 all 下按类别文件夹）",
    )
    ap.add_argument(
        "--split-by",
        choices=("poi_category", "parcel_landuse"),
        default="poi_category",
        help="poi_category：每个 POI 大类一张图；parcel_landuse：每个规划用地类别一张图（密度为该类 POI 或见 --landuse-density）",
    )
    ap.add_argument(
        "--landuse-density",
        choices=("all_poi", "poi_inside_landuse_plots"),
        default="all_poi",
        help="仅当 split-by=parcel_landuse：all_poi=地块内全部 POI 计数；poi_inside=仅统计落在该用地类型地块 union 内的 POI",
    )
    ap.add_argument("--out-dir", type=Path, default=SITE_3KM / "qa" / "poi_density_by_category_units")
    ap.add_argument("--vmax-quantile", type=float, default=0.98, help="色标上限分位数（避免极值压扁）")
    args = ap.parse_args()

    if not args.units.is_file():
        print(f"缺少地块文件: {args.units}", file=sys.stderr)
        return 1
    units = gpd.read_file(args.units)
    if units.crs is None:
        units = units.set_crs(CRS_WGS)
    units = units.to_crs(CRS_WGS)
    units["area_m2"] = pd.to_numeric(units.get("area"), errors="coerce").clip(lower=1.0)

    site = load_site_polygon(args.site_json)
    units_m = units.to_crs(CRS_METRIC)
    all_pois, load_warnings = load_all_pois_flat(args.poi_root)
    for w in load_warnings:
        print(f"[warn] {w}", file=sys.stderr)

    site_wm = site.to_crs(CRS_WM)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {"outputs": [], "warnings": load_warnings, "split_by": args.split_by}

    if args.split_by == "poi_category":
        by_type, _, _ = resolve_poi_classification_dirs(args.poi_root)
        inputs = discover_poi_category_inputs(args.poi_root)
        if not inputs and all_pois.empty:
            print("未找到按类别 POI 文件；请将 CSV/SHP 放入:", by_type, file=sys.stderr)
            return 2
        if inputs:
            for cat, p in inputs:
                try:
                    suf = p.suffix.lower()
                    if suf == ".csv":
                        pts = load_poi_points_from_csv(p, cat).to_crs(CRS_METRIC)
                    else:
                        pts = load_poi_points_from_vector(p, cat).to_crs(CRS_METRIC)
                except Exception as e:
                    print(f"[skip] {p.name}: {e}", file=sys.stderr)
                    continue
                cnt = counts_per_unit(pts, units_m)
                u = units.copy()
                u["poi_n"] = u["unit_id"].map(cnt).fillna(0).astype(int)
                u["poi_density"] = u["poi_n"] / u["area_m2"]
                uw = u.to_crs(CRS_WM)
                safe = re.sub(r'[\\/:*?"<>|]+', "_", cat)[:80]
                outp = args.out_dir / f"poi_density_units__{safe}.png"
                plot_one_choropleth(
                    units_wm=uw,
                    site_wm=site_wm,
                    column="poi_density",
                    title=f"SITE 周边地块 · {cat} POI 密度（个/平方米）",
                    out_path=outp,
                    vmax_q=args.vmax_quantile,
                )
                meta["outputs"].append({"category": cat, "path": str(outp.relative_to(REPO))})
        else:
            if all_pois.empty:
                print("无 POI 点数据可绘制", file=sys.stderr)
                return 2
            pm = all_pois.to_crs(CRS_METRIC)
            for cat in sorted(pm["poi_category"].dropna().unique()):
                pts = pm[pm["poi_category"] == cat]
                cnt = counts_per_unit(pts, units_m)
                u = units.copy()
                u["poi_n"] = u["unit_id"].map(cnt).fillna(0).astype(int)
                u["poi_density"] = u["poi_n"] / u["area_m2"]
                uw = u.to_crs(CRS_WM)
                safe = re.sub(r'[\\/:*?"<>|]+', "_", str(cat))[:80]
                outp = args.out_dir / f"poi_density_units__{safe}.png"
                plot_one_choropleth(
                    units_wm=uw,
                    site_wm=site_wm,
                    column="poi_density",
                    title=f"SITE 周边地块 · {cat} POI 密度（个/平方米）",
                    out_path=outp,
                    vmax_q=args.vmax_quantile,
                )
                meta["outputs"].append({"category": str(cat), "path": str(outp.relative_to(REPO))})

    else:
        # parcel_landuse：列名 PLANLAND_1（上海地块）；缺失则退化为单张「全部地块 + 全 POI」
        lu_col = "PLANLAND_1" if "PLANLAND_1" in units.columns else None
        if lu_col is None:
            print("地块缺少 PLANLAND_1，改为输出单张「全 POI 密度」", file=sys.stderr)
            pm = all_pois.to_crs(CRS_METRIC)
            cnt = counts_per_unit(pm, units_m)
            u = units.copy()
            u["poi_n"] = u["unit_id"].map(cnt).fillna(0).astype(int)
            u["poi_density"] = u["poi_n"] / u["area_m2"]
            outp = args.out_dir / "poi_density_units__all_parcels_all_poi.png"
            plot_one_choropleth(
                units_wm=u.to_crs(CRS_WM),
                site_wm=site_wm,
                column="poi_density",
                title=f"SITE 周边地块 · 全类别 POI 密度（个/平方米）",
                out_path=outp,
                vmax_q=args.vmax_quantile,
            )
            meta["outputs"].append({"category": "_all_", "path": str(outp.relative_to(REPO))})
        else:
            pm = all_pois.to_crs(CRS_METRIC)
            for lu in units[lu_col].fillna("未分类").astype(str).unique():
                sub_u = units[units[lu_col].fillna("未分类").astype(str) == lu].copy()
                if sub_u.empty:
                    continue
                sub_m = sub_u.to_crs(CRS_METRIC)
                if args.landuse_density == "all_poi":
                    cnt = counts_per_unit(pm, units_m)
                else:
                    mask_union = sub_m.geometry.unary_union
                    pm_clip = pm[pm.geometry.intersects(mask_union)]
                    cnt = counts_per_unit(pm_clip, units_m)
                sub_u["poi_n"] = sub_u["unit_id"].map(cnt).fillna(0).astype(int)
                sub_u["poi_density"] = sub_u["poi_n"] / sub_u["area_m2"]
                uw = units.copy()
                uw["poi_density"] = 0.0
                uw.loc[sub_u.index, "poi_density"] = sub_u["poi_density"].values
                uw = uw.to_crs(CRS_WM)
                safe = re.sub(r'[\\/:*?"<>|]+', "_", lu)[:80]
                outp = args.out_dir / f"parcel_landuse_{safe}__poi_density.png"
                plot_one_choropleth(
                    units_wm=uw,
                    site_wm=site_wm,
                    column="poi_density",
                    title=f"SITE 周边 · 规划用地「{lu}」地块上的 POI 密度（个/平方米）",
                    out_path=outp,
                    vmax_q=args.vmax_quantile,
                )
                meta["outputs"].append({"landuse": lu, "path": str(outp.relative_to(REPO))})

    with (args.out_dir / "plot_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
