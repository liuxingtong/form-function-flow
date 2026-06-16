"""
消费 / 商务 / 娱乐 POI 在地块上算密度，沿南北向分箱得到剖面；绿地（``11-蓝绿空间`` 面要素与地块相交）
得到覆盖率，在 **右轴** 绘制。

剖面在 **SITE 形心纬度两侧分段** 高斯平滑（不跨南北整体抹平），断点处保留跃迁。

条形图与「类内相对变化」：按 **SITE 邻域若干剖面带**（``--breakpoint-bins``）的南北加权均值
计算 (南−北)/南，表征 **断点处** 南北差异比例。

红黑主图 + 浅绿辅轴；POI 为「个/km²」，绿地为「覆盖率 %」。

示例::

  python scripts/plot_site_function_ns_density_profile.py ^
    --poi-root "F:/Aworks/2026studio/shanghaistation/all/02-POI&AOI/1-POI/25.05/CSV/分类/按类别"
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
from shapely.geometry import Polygon

REPO = Path(__file__).resolve().parents[1]
SITE_3KM = REPO / "data" / "site_3km"
CRS_WGS = "EPSG:4326"
CRS_METRIC = "EPSG:32651"

# 高德大类文件名（去掉「上海市-」前缀后）→ 消费 / 商务 / 娱乐
CONSUMPTION = "消费"
BUSINESS = "商务"
LEISURE = "娱乐"

POI_GROUP: dict[str, str] = {
    "餐饮服务": CONSUMPTION,
    "购物服务": CONSUMPTION,
    "生活服务": CONSUMPTION,
    "住宿服务": CONSUMPTION,
    "医疗保健服务": CONSUMPTION,
    "金融保险服务": CONSUMPTION,
    "汽车销售": CONSUMPTION,
    "汽车服务": CONSUMPTION,
    "汽车维修": CONSUMPTION,
    "摩托车服务": CONSUMPTION,
    "公司企业": BUSINESS,
    "商务住宅": BUSINESS,
    "政府机构及社会团体": BUSINESS,
    "科教文化服务": BUSINESS,
    "交通设施服务": BUSINESS,
    "通行设施": BUSINESS,
    "室内设施": BUSINESS,
    "公共设施": BUSINESS,
    "地名地址信息": BUSINESS,
    "体育休闲服务": LEISURE,
    "风景名胜": LEISURE,
    "事件活动": LEISURE,
    "道路附属设施": LEISURE,
}


def _load_poi_helpers():
    p = Path(__file__).resolve().parent / "plot_site_poi_density_by_category.py"
    spec = importlib.util.spec_from_file_location("_site_poi_cat", p)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 plot_site_poi_density_by_category.py")
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


def default_poi_root() -> Path:
    cand = Path(r"F:\Aworks\2026studio\shanghaistation\all\02-POI&AOI\1-POI\25.05\CSV\分类\按类别")
    if cand.is_dir():
        return cand
    return SITE_3KM / "02-POI&AOI" / "1-POI" / "25.05" / "CSV" / "分类"


def collect_pois_grouped(poi_root: Path, poi_mod) -> tuple[gpd.GeoDataFrame, list[str]]:
    rows: list[gpd.GeoDataFrame] = []
    skipped: list[str] = []
    for cat_name, path in poi_mod.discover_poi_category_inputs(poi_root):
        grp = POI_GROUP.get(cat_name)
        if grp is None:
            skipped.append(cat_name)
            continue
        if path.suffix.lower() != ".csv":
            continue
        try:
            g = poi_mod.load_poi_points_from_csv(path, cat_name)
        except Exception as e:
            skipped.append(f"{cat_name}:{e}")
            continue
        if g.empty:
            continue
        g = g.assign(func_group=grp)
        rows.append(g[["func_group", "geometry"]])
    if not rows:
        raise RuntimeError("未读到任何已映射的 POI CSV；检查 --poi-root 与 POI_GROUP 映射。")
    out = pd.concat(rows, ignore_index=True)
    gdf = gpd.GeoDataFrame(out, geometry=out.geometry, crs=CRS_WGS)
    return gdf, skipped


def counts_per_unit_by_group(
    pois_m: gpd.GeoDataFrame, units_m: gpd.GeoDataFrame
) -> pd.DataFrame:
    """unit_id × func_group 计数。"""
    pm = pois_m.copy()
    pm["poi_uid"] = np.arange(len(pm), dtype=np.int64)
    sj = gpd.sjoin(
        pm[["poi_uid", "func_group", "geometry"]],
        units_m[["unit_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    sj = sj.drop_duplicates(subset=["poi_uid"], keep="first")
    ct = sj.groupby(["unit_id", "func_group"], observed=False).size().unstack(fill_value=0)
    for col in (CONSUMPTION, BUSINESS, LEISURE):
        if col not in ct.columns:
            ct[col] = 0
    return ct[[CONSUMPTION, BUSINESS, LEISURE]].astype(int)


def bin_weighted_profile(um: gpd.GeoDataFrame, rho_col: str, nbins: int) -> pd.Series:
    """按 ns_bin 对 rho（个/km²）做面积加权平均。"""
    um_w = um.assign(_wrho=um[rho_col] * um["area_m2"])
    num = um_w.groupby("ns_bin", observed=False)["_wrho"].sum()
    den = um.groupby("ns_bin", observed=False)["area_m2"].sum()
    return (num / den).reindex(range(nbins)).fillna(0.0).astype(float)


def north_south_delta(
    prof: pd.Series, area_bin: pd.Series, centers: np.ndarray, site_y: float
) -> tuple[float, float, float]:
    pv = prof.to_numpy()
    av = area_bin.to_numpy()
    south_mask = centers < site_y
    north_mask = centers > site_y

    def _wm(mask: np.ndarray) -> float:
        w = av[mask]
        v = pv[mask]
        s = float(w.sum())
        if s <= 0:
            return float("nan")
        return float(np.sum(v * w) / s)

    s_mean = _wm(south_mask)
    n_mean = _wm(north_mask)
    d = n_mean - s_mean if np.isfinite(s_mean) and np.isfinite(n_mean) else float("nan")
    return s_mean, n_mean, d


def smooth_1d(y: np.ndarray, sigma: float) -> np.ndarray:
    """沿剖面索引高斯平滑；sigma<=0 原样返回。"""
    y = np.asarray(y, dtype=float)
    if sigma <= 0 or len(y) < 3:
        return y
    try:
        from scipy.ndimage import gaussian_filter1d

        return gaussian_filter1d(y, sigma=float(sigma), mode="nearest")
    except Exception:
        k = max(3, int(round(2 * sigma)) * 2 + 1)
        k = min(k, len(y) // 2 * 2 + 1) if len(y) >= 5 else len(y) | 1
        k = max(3, k | 1)
        s = pd.Series(y).rolling(k, center=True, min_periods=1).mean().to_numpy()
        return np.asarray(s, dtype=float)


def smooth_segmented_ns(y: np.ndarray, sigma: float, j_split: int) -> np.ndarray:
    """以 j_split 为界：南侧 bins 与北侧 bins 分别平滑，避免跨 SITE 抹平跃迁。"""
    y = np.asarray(y, dtype=float)
    n = len(y)
    j = int(np.clip(j_split, 0, n))
    if sigma <= 0:
        return y.copy()
    out = y.copy()
    if j > 0:
        out[:j] = smooth_1d(y[:j], sigma)
    if j < n:
        out[j:] = smooth_1d(y[j:], sigma)
    return out


def slice_weighted_mean(prof: pd.Series, area_bin: pd.Series, lo: int, hi: int) -> float:
    if lo >= hi:
        return float("nan")
    w = area_bin.to_numpy()[lo:hi]
    v = prof.to_numpy()[lo:hi]
    s = float(np.sum(w))
    if s <= 0:
        return float("nan")
    return float(np.sum(v * w) / s)


def breakpoint_adjacent_means(
    prof: pd.Series, area_bin: pd.Series, j_split: int, k: int
) -> tuple[float, float]:
    """
    SITE 断点邻域：南侧取剖面带 [j_split-k, j_split)，北侧取 [j_split, j_split+k)。
    j_split = 形心以北第一条剖面带下标（centers >= site_y 的首索引）。
    """
    nb = len(prof)
    kk = max(1, int(k))
    if nb < 2:
        return float("nan"), float("nan")
    j = int(np.clip(j_split, 0, nb))
    if j <= 0:
        hi_s = min(kk, nb)
        lo_n = hi_s
        hi_n = min(nb, lo_n + kk)
        return slice_weighted_mean(prof, area_bin, 0, hi_s), slice_weighted_mean(prof, area_bin, lo_n, hi_n)
    if j >= nb:
        hi_n = nb
        lo_n = max(0, nb - kk)
        lo_s = max(0, lo_n - kk)
        south = slice_weighted_mean(prof, area_bin, lo_s, lo_n)
        north = slice_weighted_mean(prof, area_bin, lo_n, hi_n)
        return south, north
    lo_s = max(0, j - kk)
    hi_s = j
    lo_n = j
    hi_n = min(nb, j + kk)
    south = slice_weighted_mean(prof, area_bin, lo_s, hi_s)
    north = slice_weighted_mean(prof, area_bin, lo_n, hi_n)
    return south, north


def within_class_ratio_from_means(south: float, north: float) -> float:
    if not np.isfinite(south) or not np.isfinite(north) or south <= 1e-12:
        return float("nan")
    return float((south - north) / south)


GREEN_LABEL = "绿地"
GREEN_STEM_KEYS = ("绿地", "绿色区域", "公园广场面", "森林", "自然保护区")
GREEN_STEM_EXCLUDE = ("水系", "河流", "水路", "黄浦江", "供水站", "海洋", "虚拟", "栅格")


def load_green_cover_by_unit(units_m: gpd.GeoDataFrame, blue_dir: Path) -> tuple[pd.Series, list[str]]:
    """unit_id -> [0,1] 绿地覆盖率（绿地面与地块相交面积 / 地块面积）。"""
    logs: list[str] = []
    idx = units_m["unit_id"]
    if not blue_dir.is_dir():
        logs.append(f"missing_blue_green_dir:{blue_dir}")
        return pd.Series(0.0, index=idx), logs

    parts: list[gpd.GeoDataFrame] = []
    for p in sorted(blue_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".geojson", ".json", ".shp", ".gpkg"):
            continue
        stem = p.stem
        if any(x in stem for x in GREEN_STEM_EXCLUDE):
            continue
        if not any(k in stem for k in GREEN_STEM_KEYS):
            continue
        try:
            g = gpd.read_file(p)
        except Exception as e:
            logs.append(f"{p.name}:{e}")
            continue
        if g.empty:
            continue
        if g.crs is None:
            g = g.set_crs(CRS_WGS)
        g = g.to_crs(CRS_METRIC)
        g = g[g.geometry.notna()]
        g = g[g.geom_type.isin(["Polygon", "MultiPolygon"])]
        if g.empty:
            continue
        parts.append(g[["geometry"]])

    if not parts:
        logs.append("no_matching_green_layers")
        return pd.Series(0.0, index=idx), logs

    greens = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=CRS_METRIC)
    greens = greens[greens.geometry.notna()]
    greens = greens[greens.geom_type.isin(["Polygon", "MultiPolygon"])]
    if greens.empty:
        logs.append("green_geoms_empty_after_filter")
        return pd.Series(0.0, index=idx), logs
    greens = greens.explode(index_parts=False, ignore_index=True)
    greens = greens[greens.geom_type == "Polygon"]
    if greens.empty:
        logs.append("green_no_polygon_parts")
        return pd.Series(0.0, index=idx), logs
    greens = greens.dissolve(as_index=False)

    base = units_m[["unit_id", "geometry"]].copy()
    base["geometry"] = base.geometry.make_valid()
    base_parts = base.explode(index_parts=False, ignore_index=True)
    base_parts = base_parts[base_parts.geometry.notna() & (base_parts.geom_type == "Polygon")]
    if base_parts.empty:
        logs.append("units_no_polygon_parts")
        return pd.Series(0.0, index=idx), logs
    try:
        inter = gpd.overlay(base_parts, greens, how="intersection", keep_geom_type=True)
    except Exception as e:
        logs.append(f"overlay_failed:{e}")
        return pd.Series(0.0, index=idx), logs
    if inter.empty:
        return pd.Series(0.0, index=idx), logs
    inter["ix_area"] = inter.geometry.area
    agg = inter.groupby("unit_id", observed=False)["ix_area"].sum()
    u_area = base.set_index("unit_id").geometry.area
    ratio = (agg / u_area).reindex(idx).fillna(0.0).astype(float).clip(0.0, 1.0)
    return ratio, logs


def main() -> int:
    configure_cn_font()
    poi_mod = _load_poi_helpers()

    ap = argparse.ArgumentParser(description="SITE 南北向综合功能密度剖面（消费+商务+娱乐 POI）")
    ap.add_argument("--site-json", type=Path, default=SITE_3KM / "SITE.json")
    ap.add_argument("--units", type=Path, default=SITE_3KM / "01_units.gpkg")
    ap.add_argument("--poi-root", type=Path, default=default_poi_root())
    ap.add_argument("--nbins", type=int, default=56, help="南北向分箱数")
    ap.add_argument(
        "--smooth-sigma",
        type=float,
        default=2.8,
        help="剖面在 SITE 南北 **分段** 高斯平滑 σ（按箱索引）；0 关闭",
    )
    ap.add_argument(
        "--blue-green-dir",
        type=Path,
        default=SITE_3KM / "11-蓝绿空间",
        help="蓝绿空间目录（绿地/公园广场面/森林等面矢量）",
    )
    ap.add_argument(
        "--breakpoint-bins",
        type=int,
        default=3,
        help="SITE 断点两侧各取几条剖面带（邻域加权，用于条形图比例）",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=SITE_3KM / "qa" / "function_ns_profile" / "site_function_ns_combined_red.png",
    )
    args = ap.parse_args()

    if not args.units.is_file():
        print("缺少 units:", args.units, file=sys.stderr)
        return 1

    units = gpd.read_file(args.units)
    if units.crs is None:
        units = units.set_crs(CRS_WGS)
    units = units.to_crs(CRS_WGS)
    units["area_m2"] = pd.to_numeric(units.get("area"), errors="coerce").fillna(1.0).clip(lower=1.0)

    site = load_site_polygon(args.site_json)
    units_m = units.to_crs(CRS_METRIC)
    site_m = site.to_crs(CRS_METRIC)
    site_y = float(site_m.geometry.centroid.iloc[0].y)

    pois_wgs, skipped = collect_pois_grouped(args.poi_root, poi_mod)
    pois_m = pois_wgs.to_crs(CRS_METRIC)

    grp_counts = counts_per_unit_by_group(pois_m, units_m)
    green_series, green_logs = load_green_cover_by_unit(units_m, args.blue_green_dir)

    u = units.set_index("unit_id")
    u = u.join(grp_counts, how="left")
    u[CONSUMPTION] = u[CONSUMPTION].fillna(0)
    u[BUSINESS] = u[BUSINESS].fillna(0)
    u[LEISURE] = u[LEISURE].fillna(0)
    u = u.join(green_series.rename("green_ratio"), how="left")
    u["green_ratio"] = u["green_ratio"].fillna(0.0).clip(0.0, 1.0)

    u["poi_total"] = u[CONSUMPTION] + u[BUSINESS] + u[LEISURE]
    u["rho_km2"] = u["poi_total"] / u["area_m2"] * 1_000_000.0
    u["rho_km2_消费"] = u[CONSUMPTION] / u["area_m2"] * 1_000_000.0
    u["rho_km2_商务"] = u[BUSINESS] / u["area_m2"] * 1_000_000.0
    u["rho_km2_娱乐"] = u[LEISURE] / u["area_m2"] * 1_000_000.0

    um = u.to_crs(CRS_METRIC)
    um["cy"] = um.geometry.centroid.y

    ymin, ymax = float(um["cy"].min()), float(um["cy"].max())
    edges = np.linspace(ymin, ymax, args.nbins + 1)
    um["ns_bin"] = np.clip(np.digitize(um["cy"].to_numpy(), edges) - 1, 0, args.nbins - 1)
    centers = np.asarray(0.5 * (edges[:-1] + edges[1:]))

    area_bin = um.groupby("ns_bin", observed=False)["area_m2"].sum().reindex(range(args.nbins)).fillna(0.0)
    prof_total = bin_weighted_profile(um, "rho_km2", args.nbins)
    prof_c = bin_weighted_profile(um, "rho_km2_消费", args.nbins)
    prof_b = bin_weighted_profile(um, "rho_km2_商务", args.nbins)
    prof_l = bin_weighted_profile(um, "rho_km2_娱乐", args.nbins)
    prof_g = bin_weighted_profile(um, "green_ratio", args.nbins)

    x_km = (centers - ymin) / 1000.0
    x_site = (site_y - ymin) / 1000.0

    south_mean, north_mean, delta = north_south_delta(prof_total, area_bin, centers, site_y)
    s_c, n_c, d_c = north_south_delta(prof_c, area_bin, centers, site_y)
    s_b, n_b, d_b = north_south_delta(prof_b, area_bin, centers, site_y)
    s_l, n_l, d_l = north_south_delta(prof_l, area_bin, centers, site_y)
    s_g, n_g, d_g = north_south_delta(prof_g, area_bin, centers, site_y)

    j_split = int(np.sum(centers < site_y))
    kb = max(1, int(args.breakpoint_bins))
    sbp, nbp = breakpoint_adjacent_means(prof_total, area_bin, j_split, kb)
    sb_c, nb_c = breakpoint_adjacent_means(prof_c, area_bin, j_split, kb)
    sb_b, nb_b = breakpoint_adjacent_means(prof_b, area_bin, j_split, kb)
    sb_l, nb_l = breakpoint_adjacent_means(prof_l, area_bin, j_split, kb)
    sb_g, nb_g = breakpoint_adjacent_means(prof_g, area_bin, j_split, kb)

    r_total = within_class_ratio_from_means(sbp, nbp)
    r_c = within_class_ratio_from_means(sb_c, nb_c)
    r_b = within_class_ratio_from_means(sb_b, nb_b)
    r_l = within_class_ratio_from_means(sb_l, nb_l)
    r_g = within_class_ratio_from_means(sb_g, nb_g)

    ratios_pct = np.array(
        [
            r_total * 100 if np.isfinite(r_total) else np.nan,
            r_c * 100 if np.isfinite(r_c) else np.nan,
            r_b * 100 if np.isfinite(r_b) else np.nan,
            r_l * 100 if np.isfinite(r_l) else np.nan,
            r_g * 100 if np.isfinite(r_g) else np.nan,
        ],
        dtype=float,
    )

    sm = float(args.smooth_sigma)
    pt = smooth_segmented_ns(prof_total.to_numpy(), sm, j_split)
    pc = smooth_segmented_ns(prof_c.to_numpy(), sm, j_split)
    pb = smooth_segmented_ns(prof_b.to_numpy(), sm, j_split)
    pl = smooth_segmented_ns(prof_l.to_numpy(), sm, j_split)
    pg = smooth_segmented_ns(prof_g.to_numpy() * 100.0, sm, j_split)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # —— 红黑视觉 —— #
    BG = "#0a0a0a"
    RED = "#ff1744"
    RED_DIM = "#8b0000"
    AXES = "#1a0505"
    TEXT = "#f5f5f5"

    fig = plt.figure(figsize=(14, 9.2), dpi=200, facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.25, 1.05], hspace=0.32)
    ax = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])
    ax2 = ax.twinx()
    ax.set_facecolor(AXES)
    ax_bar.set_facecolor("#100505")

    # 分项：略浅的红色系 + 不同线型，保证在黑底上可辨
    RED_C = "#ff8a80"
    RED_B = "#ff5252"
    RED_L = "#d50000"
    GREEN_LINE = "#b9f6ca"

    ax2.fill_between(x_km, 0, pg, color=GREEN_LINE, alpha=0.12, linewidth=0, zorder=1)
    ax2.plot(
        x_km,
        pg,
        color=GREEN_LINE,
        linewidth=2.5,
        linestyle="-",
        label="绿地覆盖率 %",
        zorder=2,
        alpha=0.95,
    )
    ax2.set_ylabel("绿地剖面 覆盖率 %", color="#c8e6c9", fontsize=12, labelpad=10)
    ax2.tick_params(axis="y", colors="#c8e6c9", labelsize=10)
    ax2.spines["right"].set_color("#558b2f")
    ax2.spines["right"].set_linewidth(1.2)
    ax2.set_ylim(0, max(5.0, float(np.nanmax(pg)) * 1.2, 1.0))

    ax.plot(
        x_km,
        pc,
        color=RED_C,
        linewidth=2.0,
        linestyle=(0, (4, 2)),
        label="消费",
        zorder=4,
        alpha=0.95,
    )
    ax.plot(
        x_km,
        pb,
        color=RED_B,
        linewidth=2.0,
        linestyle=(0, (1, 1.2)),
        label="商务",
        zorder=4,
        alpha=0.95,
    )
    ax.plot(
        x_km,
        pl,
        color=RED_L,
        linewidth=2.0,
        linestyle="-.",
        label="娱乐",
        zorder=4,
        alpha=0.95,
    )

    ax.fill_between(x_km, 0, pt, color=RED, alpha=0.14, linewidth=0, zorder=3)
    ax.plot(
        x_km,
        pt,
        color=RED,
        linewidth=3.4,
        solid_capstyle="round",
        zorder=6,
        label="综合 POI",
    )

    ax.axvline(x_site, color=RED_DIM, linestyle="--", linewidth=2.0, alpha=0.95, zorder=5)
    ax.axhline(0, color="#2a0000", linewidth=0.8, zorder=1)

    ax.set_xlim(x_km.min(), x_km.max())
    y_hi = (
        max(
            float(np.nanmax(pt)),
            float(np.nanmax(pc)),
            float(np.nanmax(pb)),
            float(np.nanmax(pl)),
        )
        * 1.12
    )
    y_hi = max(y_hi, 1e-6)
    ax.set_ylim(0, y_hi)

    ax.set_xlabel("自南向北距离 (km)", color=TEXT, fontsize=13, labelpad=10)
    ax.set_ylabel("POI 密度 (个/km²，箱内面积加权)", color=TEXT, fontsize=13, labelpad=10)
    ax.tick_params(axis="x", colors=TEXT, labelsize=11)
    ax.tick_params(axis="y", colors=TEXT, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(RED_DIM)
        spine.set_linewidth(1.2)
    ax.grid(True, linestyle=":", alpha=0.25, color=RED_DIM)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = ax.legend(
        h1 + h2,
        l1 + l2,
        loc="upper right",
        framealpha=0.88,
        facecolor="#120505",
        edgecolor=RED,
        labelcolor=TEXT,
        fontsize=9,
    )
    for t in leg.get_texts():
        t.set_fontweight("bold")

    ax.set_title(
        "POI + 绿地 南北剖面（SITE 处分段平滑）",
        color=RED,
        fontsize=16,
        fontweight="bold",
        pad=16,
    )

    if np.isfinite(delta):

        def _rp(r: float) -> str:
            return f"{r * 100:+.1f}%" if np.isfinite(r) else "—"

        d_g_pp = (n_g - s_g) * 100.0 if np.isfinite(n_g) and np.isfinite(s_g) else float("nan")
        ann = (
            f"【全域·综合 POI】南 {south_mean:.0f} 北 {north_mean:.0f}  Δ {delta:+.0f} 个/km²\n"
            f"【断点邻域各±{kb}带】(南−北)/南：综合 {_rp(r_total)} ｜消费 {_rp(r_c)} ｜商务 {_rp(r_b)} ｜娱乐 {_rp(r_l)} ｜绿地 {_rp(r_g)}\n"
            f"【全域·绿地覆盖】南 {s_g * 100:.1f}% 北 {n_g * 100:.1f}%  Δ {d_g_pp:+.1f} pp"
        )
    else:
        ann = "无法计算南北差（剖面数据不足）"

    ax.text(
        0.03,
        0.97,
        ann,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color=TEXT,
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#140000", edgecolor=RED, linewidth=1.8, alpha=0.92),
    )
    ax.text(
        x_site,
        y_hi * 0.96,
        " SITE ",
        color=TEXT,
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=RED_DIM, edgecolor=RED, linewidth=1.0),
    )

    # —— 下方：断点邻域 (南−北)/南 条形图（双向） —— #
    bar_labels = ["综合", "消费", "商务", "娱乐", GREEN_LABEL]
    y_pos = np.arange(len(bar_labels))
    vmax = float(np.nanmax(np.abs(ratios_pct)))
    if not np.isfinite(vmax) or vmax < 5:
        vmax = 50.0
    vmax = min(max(vmax * 1.15, 15.0), 95.0)
    ax_bar.set_xlim(-vmax, vmax)
    ax_bar.set_ylim(-0.6, len(bar_labels) - 0.4)
    bar_colors = [RED, RED_C, RED_B, RED_L, "#7cb342"]
    for i, (v, bc) in enumerate(zip(ratios_pct, bar_colors)):
        if not np.isfinite(v):
            ax_bar.text(0, i, " — ", ha="center", va="center", color=TEXT, fontsize=11)
            continue
        left = min(0.0, v)
        w = abs(v)
        ax_bar.barh(
            i,
            w,
            left=left,
            height=0.58,
            color=bc,
            edgecolor=RED if v >= 0 else "#cccccc",
            linewidth=1.0,
            alpha=0.92 if v >= 0 else 0.65,
        )
        off = 0.02 * vmax
        ha = "left" if v >= 0 else "right"
        xt = v + off if v >= 0 else v - off
        xt = float(np.clip(xt, -vmax + 0.05 * vmax, vmax - 0.05 * vmax))
        ax_bar.text(
            xt,
            i,
            f"{v:+.1f}%",
            va="center",
            ha=ha,
            color=TEXT,
            fontsize=10,
            fontweight="bold",
        )
    ax_bar.axvline(0, color=TEXT, linewidth=1.0, alpha=0.85)
    ax_bar.set_yticks(y_pos, bar_labels, color=TEXT, fontsize=11)
    ax_bar.tick_params(axis="x", colors=TEXT, labelsize=10)
    ax_bar.tick_params(axis="y", length=0)
    for spine in ax_bar.spines.values():
        spine.set_color(RED_DIM)
        spine.set_linewidth(1.0)
    ax_bar.set_xlabel(
        f"断点邻域（SITE 两侧各 ±{kb} 剖面带）相对变化  (南−北)/南 ×100%  ·  正值=北侧更低",
        color=TEXT,
        fontsize=11,
        labelpad=8,
    )
    ax_bar.set_title("断点邻域南北对比（非全域平均）", color=RED, fontsize=12, pad=10, fontweight="bold")
    ax_bar.grid(True, axis="x", linestyle=":", alpha=0.28, color=RED_DIM)

    fig.subplots_adjust(left=0.08, right=0.90, top=0.93, bottom=0.07, hspace=0.32)
    fig.savefig(args.out, facecolor=BG, edgecolor="none", bbox_inches="tight")
    plt.close(fig)

    meta = {
        "out_png": str(args.out.relative_to(REPO)),
        "poi_root": str(args.poi_root),
        "blue_green_dir": str(args.blue_green_dir),
        "nbins": args.nbins,
        "smooth_sigma": args.smooth_sigma,
        "breakpoint_bins_each_side": kb,
        "ns_bin_split_index_j": j_split,
        "site_centroid_northing_m": site_y,
        "green_layer_notes": green_logs,
        "breakpoint_ratio_south_minus_north_over_south": {
            "combined": r_total,
            "consumption": r_c,
            "business": r_b,
            "leisure": r_l,
            "green_cover": r_g,
        },
        "breakpoint_adjacent_weighted_mean": {
            "combined": {"south": sbp, "north": nbp},
            "consumption": {"south": sb_c, "north": nb_c},
            "business": {"south": sb_b, "north": nb_b},
            "leisure": {"south": sb_l, "north": nb_l},
            "green_cover": {"south": sb_g, "north": nb_g},
        },
        "global_domain_weighted_mean": {
            "combined_poi_km2": {"south": south_mean, "north": north_mean, "delta": delta},
            "consumption": {"south": s_c, "north": n_c, "delta": d_c},
            "business": {"south": s_b, "north": n_b, "delta": d_b},
            "leisure": {"south": s_l, "north": n_l, "delta": d_l},
            "green_cover_frac": {"south": s_g, "north": n_g, "delta": (n_g - s_g) if np.isfinite(n_g) and np.isfinite(s_g) else None},
        },
        "skipped_poi_categories": skipped,
        "profile_csv": str((args.out.with_suffix(".csv")).relative_to(REPO)),
    }
    pd.DataFrame(
        {
            "x_km_from_south": x_km,
            "rho_km2_combined": prof_total.to_numpy(),
            "rho_km2_combined_smooth": pt,
            "rho_km2_消费": prof_c.to_numpy(),
            "rho_km2_消费_smooth": pc,
            "rho_km2_商务": prof_b.to_numpy(),
            "rho_km2_商务_smooth": pb,
            "rho_km2_娱乐": prof_l.to_numpy(),
            "rho_km2_娱乐_smooth": pl,
            "green_cover_frac": prof_g.to_numpy(),
            "green_cover_pct_smooth": pg,
            "bin_center_n_m": centers,
        }
    ).to_csv(args.out.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    def _json_sanitize(o: object) -> object:
        if isinstance(o, float):
            if np.isnan(o) or np.isinf(o):
                return None
            return float(o)
        if isinstance(o, dict):
            return {str(k): _json_sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_json_sanitize(v) for v in o]
        return o

    with (args.out.with_suffix(".json")).open("w", encoding="utf-8") as f:
        json.dump(_json_sanitize(meta), f, ensure_ascii=False, indent=2)

    print(json.dumps(_json_sanitize(meta), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
