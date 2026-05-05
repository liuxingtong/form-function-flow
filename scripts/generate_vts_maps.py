"""
Generate V / T / S analysis maps from data/site_clipped using CartoDB Positron basemap.

Indicator construction follows docs/VTS_data_methods.md using **data-available proxies**:
  V^flow: WorldPop ρ/ρ̄ × (1 + ln(1 + Flow/Flow̄)), Flow ~ transit facility KDE
  V^com:  Gaussian KDE over commerce POIs (Score-weighted when 评分 column exists)
  V^dwell: public POI KDE × exp(-λ · d_to_nearest transit POI)
  V composite: (α+β+γ)^{-1}(α Ṽ_flow + β Ṽ_com + γ Ṽ_dwell) after percentile normalization
  T:       Normalized Shannon entropy over coarse time slices (营业时间 proxy → T1–T5)
  S:       Built volume proxy; population; rough capacity residual (pop vs volume)

Run from repo root:
  python scripts/generate_vts_maps.py

Requires: geopandas, matplotlib, numpy, scipy, shapely, contextily, rasterio (optional)
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from shapely.geometry import mapping

warnings.filterwarnings("ignore", category=UserWarning)

CRS_WM = "EPSG:3857"
CRS_WGS = "EPSG:4326"

# Doc §2.2 composite weights (equal after normalization)
ALPHA_V, BETA_V, GAMMA_V = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
# §2.2 (3) distance decay λ [1/m Web Mercator], ~exp(-1) near 500 m
LAMBDA_TRANSIT_WM = 2.0e-6


def load_site_polygon(site_clipped: Path):
    p = site_clipped / "00_boundary" / "SITE.redline.geojson"
    if not p.exists():
        p = site_clipped.parent / "SITE.json"
    gdf = gpd.read_file(p)
    poly = gdf.geometry.iloc[0]
    if poly.geom_type == "LineString":
        c = list(poly.coords)
        if c[0] != c[-1]:
            c.append(c[0])
        from shapely.geometry import Polygon as ShPoly

        poly = ShPoly(c)
    return poly


def load_study_polygon(site_clipped: Path):
    for name in ("SITE.buffer_1km.geojson", "SITE.buffer_5km.geojson"):
        p = site_clipped / "00_boundary" / name
        if p.exists():
            gdf = gpd.read_file(p)
            return gdf.geometry.iloc[0]
    return load_site_polygon(site_clipped)


def collect_points(site_clipped: Path, subdirs: list[str], max_features: int = 120_000) -> gpd.GeoDataFrame:
    rows = []
    n = 0
    for sub in subdirs:
        d = site_clipped / sub
        if not d.exists():
            continue
        for f in sorted(d.glob("*.geojson")):
            if n >= max_features:
                break
            try:
                g = gpd.read_file(f)
            except Exception:
                continue
            if g.empty:
                continue
            g = g[g.geometry.notna()]
            g = g[g.geom_type.isin(["Point", "MultiPoint"])]
            if g.empty:
                continue
            g = g.explode(index_parts=False, ignore_index=True)
            g = g[g.geom_type == "Point"]
            g["__layer__"] = f.stem
            rows.append(g)
            n += len(g)
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs=CRS_WGS)
    out = pd.concat(rows, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry=out.geometry, crs=CRS_WGS)


def collect_polygons(site_clipped: Path, sub: str, max_files: int = 40) -> gpd.GeoDataFrame:
    d = site_clipped / sub
    if not d.exists():
        return gpd.GeoDataFrame(geometry=[], crs=CRS_WGS)
    parts = []
    for i, f in enumerate(sorted(d.glob("*.geojson"))):
        if i >= max_files:
            break
        try:
            g = gpd.read_file(f)
        except Exception:
            continue
        g = g[g.geometry.notna()]
        g = g[g.geom_type.isin(["Polygon", "MultiPolygon"])]
        if g.empty:
            continue
        g["__layer__"] = f.stem
        parts.append(g)
    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs=CRS_WGS)
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=CRS_WGS)


def add_wm_basemap_wm(ax, bounds_wm: tuple[float, float, float, float]):
    xmin_wm, ymin_wm, xmax_wm, ymax_wm = bounds_wm
    ax.set_xlim(xmin_wm, xmax_wm)
    ax.set_ylim(ymin_wm, ymax_wm)
    ax.set_aspect("equal")
    ax.axis("off")
    try:
        ctx.add_basemap(ax, crs=CRS_WM, source=ctx.providers.CartoDB.Positron, zoom="auto", attribution_size=6)
    except Exception:
        ctx.add_basemap(ax, crs=CRS_WM, source=ctx.providers.CartoDB.Positron, zoom=16, attribution_size=6)


def kde_grid(points_wm: gpd.GeoDataFrame, bounds_wm: tuple[float, float, float, float], nx: int = 180, ny: int = 180, max_pts: int = 6000, bw_method=0.11):
    minx, miny, maxx, maxy = bounds_wm
    xs = np.linspace(minx, maxx, nx)
    ys = np.linspace(miny, maxy, ny)
    X, Y = np.meshgrid(xs, ys)
    xy_full = np.vstack([points_wm.geometry.x.values, points_wm.geometry.y.values])
    if xy_full.shape[1] > max_pts:
        rng = np.random.default_rng(42)
        idx = rng.choice(xy_full.shape[1], size=max_pts, replace=False)
        xy = xy_full[:, idx]
    else:
        xy = xy_full
    if xy.shape[1] < 8:
        Z = np.zeros((ny, nx))
        return X, Y, Z
    kde = gaussian_kde(xy, bw_method=bw_method)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    return X, Y, Z


def kde_grid_weighted(
    points_wm: gpd.GeoDataFrame,
    weights: np.ndarray,
    bounds_wm: tuple[float, float, float, float],
    nx: int = 160,
    ny: int = 160,
    max_pts: int = 8000,
    bw_method=0.11,
):
    """Approximate score-weighted KDE by duplicating samples proportional to integer weights."""
    minx, miny, maxx, maxy = bounds_wm
    xs = np.linspace(minx, maxx, nx)
    ys = np.linspace(miny, maxy, ny)
    X, Y = np.meshgrid(xs, ys)
    wx = points_wm.geometry.x.values
    wy = points_wm.geometry.y.values
    w = np.asarray(weights, dtype=float).ravel()
    w = np.clip(np.nan_to_num(w, nan=np.nanmedian(w) if np.isfinite(np.nanmedian(w)) else 3.0), 0.5, None)
    mult = np.maximum(1, np.round(w * (5.0 / (np.nanmedian(w) + 1e-6))).astype(int))
    mult = np.clip(mult, 1, 8)
    sx = np.repeat(wx, mult)
    sy = np.repeat(wy, mult)
    if sx.size > max_pts:
        rng = np.random.default_rng(43)
        idx = rng.choice(sx.size, size=max_pts, replace=False)
        sx, sy = sx[idx], sy[idx]
    if sx.size < 8:
        return X, Y, np.zeros((ny, nx))
    kde = gaussian_kde(np.vstack([sx, sy]), bw_method=bw_method)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    return X, Y, Z


def mask_polygon_rasterio(Z, X, Y, poly_wm):
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds

    transform = from_bounds(X.min(), Y.min(), X.max(), Y.max(), X.shape[1], X.shape[0])
    inside = geometry_mask([mapping(poly_wm)], out_shape=Z.shape, transform=transform, invert=True)
    return np.where(inside, Z, np.nan)


def norm_pct(arr: np.ndarray, lo_pct=5.0, hi_pct=95.0) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if finite.size < 8:
        return np.zeros_like(arr)
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    return np.clip((arr - lo) / (hi - lo + 1e-12), 0, 1)


def cmap_yellow_red():
    return LinearSegmentedColormap.from_list("yh", ["#fff9d6", "#ffe169", "#ff9f1c", "#e63946", "#7f0f1e"])


def warp_raster_band_to_wm(
    tif_path: Path,
    bounds_wm: tuple[float, float, float, float],
    width: int = 420,
    height: int = 420,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling

    left, bottom, right, top = bounds_wm
    dst_transform = from_bounds(left, bottom, right, top, width, height)
    dst = np.full((height, width), np.nan, dtype=np.float32)
    with rasterio.open(tif_path) as src:
        kw = dict(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=CRS_WM,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )
        if src.nodata is not None:
            kw["src_nodata"] = src.nodata
        reproject(**kw)
    extent = (left, right, bottom, top)
    return dst, extent


def nearest_distance_grid(transit_xy: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Min distance from each grid node to any transit point [meters in WM plane]."""
    if transit_xy.shape[0] == 0:
        return np.full(X.shape, np.inf)
    tree = cKDTree(transit_xy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    d, _ = tree.query(pts, k=1)
    return d.reshape(X.shape)


def classify_hours_T15(s) -> str:
    """Map 营业时间 text to doc §3.3 slices T1–T5 (proxy)."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "T2_flat"
    t = str(s)
    if "周六" in t or "周日" in t or "周末" in t:
        return "T5_weekend"
    if any(x in t for x in ("24小时", "24h", "00:00", "全天")):
        return "T4_night"
    if re.search(r"2[23]:|21:.*-.*0[0-2]:|18:|19:|20:", t):
        return "T3_evening_peak"
    if re.search(r"0[7-9]:|早", t) or (re.search(r"0[6-9]:", t) and "夜" not in t):
        return "T1_morning_peak"
    if re.search(r"0[6-9]:|1[01]:", t):
        return "T1_morning_peak"
    return "T2_flat"


def load_commerce_gdf(site_clipped: Path) -> tuple[gpd.GeoDataFrame, np.ndarray | None]:
    """Dianping GeoJSON + optional score column for V^com."""
    dp = gpd.GeoDataFrame(geometry=[], crs=CRS_WGS)
    score_col = None
    for f in sorted((site_clipped / "11_commerce_dianping").glob("*.geojson")):
        try:
            dp = gpd.read_file(f)
            break
        except Exception:
            continue
    for c in ("星级", "avgscore", "rating", "Rating", "mean_score", "评分"):
        if len(dp) and c in dp.columns:
            score_col = c
            break

    def _coerce_scores(series: pd.Series) -> np.ndarray:
        out = np.full(len(series), np.nan, dtype=float)
        for i, v in enumerate(series.values):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if isinstance(v, (int, float)) and np.isfinite(float(v)):
                out[i] = float(v)
                continue
            m = re.search(r"(\d+\.?\d*)", str(v))
            if m:
                out[i] = float(m.group(1))
        return out

    scores = _coerce_scores(dp[score_col]) if score_col else None
    return dp, scores


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    sc = repo / "data" / "site_clipped"
    out_dir = repo / "outputs" / "vts_maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    site_wgs = load_site_polygon(sc)
    study_wgs = load_study_polygon(sc)
    site_gdf = gpd.GeoDataFrame({"name": ["SITE"]}, geometry=[site_wgs], crs=CRS_WGS)
    site_wm = site_gdf.to_crs(CRS_WM)
    site_poly_wm = site_wm.geometry.iloc[0]

    study_gdf = gpd.GeoDataFrame({"name": ["study_buffer"]}, geometry=[study_wgs], crs=CRS_WGS)
    study_wm = study_gdf.to_crs(CRS_WM)
    study_poly_wm = study_wm.geometry.iloc[0]

    pad_m = 120.0
    plot_bounds_wm = tuple(study_poly_wm.buffer(pad_m).bounds)
    xmin_wm, ymin_wm, xmax_wm, ymax_wm = plot_bounds_wm

    poi_all = collect_points(sc, ["04_poi", "10_public_services", "09_transport_facilities"], max_features=120_000)
    poi_wm = poi_all.to_crs(CRS_WM) if len(poi_all) else gpd.GeoDataFrame(geometry=[], crs=CRS_WM)

    transit_wm = collect_points(sc, ["09_transport_facilities"], max_features=40_000).to_crs(CRS_WM)
    public_wm = collect_points(sc, ["10_public_services"], max_features=80_000).to_crs(CRS_WM)

    dp, dp_scores = load_commerce_gdf(sc)
    dp_wm = dp.to_crs(CRS_WM) if len(dp) else gpd.GeoDataFrame(geometry=[], crs=CRS_WM)

    layer = poi_wm["__layer__"].fillna("").astype(str) if len(poi_wm) else pd.Series([], dtype=str)
    com_mask = layer.str.contains("餐饮|购物|商场|咖啡|酒店|商业|零售|娱乐", na=False)
    commercial_wm = poi_wm[com_mask] if len(poi_wm) else gpd.GeoDataFrame(geometry=[], crs=CRS_WM)
    if commercial_wm.empty and len(dp_wm) >= 5:
        commercial_wm = dp_wm

    roads = collect_polygons(sc, "02_roads", max_files=35)
    roads_wm = roads.to_crs(CRS_WM) if len(roads) else gpd.GeoDataFrame(geometry=[], crs=CRS_WM)

    heritage = collect_polygons(sc, "07_heritage", max_files=25)
    heritage_wm = heritage.to_crs(CRS_WM) if len(heritage) else gpd.GeoDataFrame(geometry=[], crs=CRS_WM)

    buildings = gpd.GeoDataFrame(geometry=[], crs=CRS_WGS)
    for name in ("BUIDING.geojson", "0001_上海市_建筑-AI影像解译-带高度.geojson"):
        p = sc / "01_buildings" / name
        if p.exists():
            try:
                buildings = gpd.read_file(p)
                break
            except Exception:
                pass
    if buildings.empty:
        buildings = collect_polygons(sc, "01_buildings", max_files=8)
    b_wm = buildings.to_crs(CRS_WM) if len(buildings) else gpd.GeoDataFrame(geometry=[], crs=CRS_WM)

    wp_tif = None
    for cand in sorted((sc / "13_demography").glob("*_site.tif")):
        wp_tif = cand
        break

    nx = ny = 200
    grid_bounds = plot_bounds_wm

    # --- Build aligned grids -----------------------------------------------------------
    rho_grid = None
    if wp_tif is not None:
        try:
            rho_grid, _ = warp_raster_band_to_wm(wp_tif, grid_bounds, width=nx, height=ny)
        except Exception:
            rho_grid = None

    Xk, Yk, Z_tr = kde_grid(transit_wm, grid_bounds, nx=nx, ny=ny, max_pts=5000) if len(transit_wm) >= 5 else (None, None, None)
    if Z_tr is None:
        Z_tr = np.zeros((ny, nx))
        Xk, Yk = np.meshgrid(np.linspace(grid_bounds[0], grid_bounds[2], nx), np.linspace(grid_bounds[1], grid_bounds[3], ny))

    if rho_grid is None:
        rho_grid = np.ones((ny, nx))
    else:
        rho_grid = np.asarray(rho_grid, dtype=float)
        if rho_grid.shape != (ny, nx):
            from scipy.ndimage import zoom

            sy, sx = ny / rho_grid.shape[0], nx / rho_grid.shape[1]
            med = float(np.nanmedian(rho_grid[np.isfinite(rho_grid)])) if np.any(np.isfinite(rho_grid)) else 1.0
            rho_grid = zoom(np.nan_to_num(rho_grid, nan=med), (sy, sx), order=1)
            rho_grid = rho_grid[:ny, :nx]

    rho_bar = float(np.nanmedian(rho_grid[np.isfinite(rho_grid) & (rho_grid > 0)]) or 1.0)
    flow_bar = float(np.nanmedian(Z_tr[np.isfinite(Z_tr)]) + 1e-12)
    V_flow_raw = (rho_grid / rho_bar) * (1.0 + np.log1p(Z_tr / flow_bar))
    V_flow_raw = np.where(np.isfinite(V_flow_raw), V_flow_raw, np.nan)
    V_flow_m = mask_polygon_rasterio(V_flow_raw, Xk, Yk, study_poly_wm)

    # V^com — prefer Dianping + 评分 when available (doc §2.2 (2))
    if len(dp_wm) >= 8 and dp_scores is not None:
        Xc, Yc, Z_com = kde_grid_weighted(dp_wm, dp_scores, grid_bounds, nx=nx, ny=ny)
    elif len(commercial_wm) >= 8:
        Xc, Yc, Z_com = kde_grid(commercial_wm, grid_bounds, nx=nx, ny=ny, max_pts=8000)
    else:
        Xc, Yc, Z_com = Xk, Yk, np.zeros((ny, nx))
    Z_com_m = mask_polygon_rasterio(Z_com, Xc, Yc, study_poly_wm)

    # V^dwell = F_pub KDE × exp(-λ d_transit)
    _, _, Z_pub = kde_grid(public_wm, grid_bounds, nx=nx, ny=ny, max_pts=8000) if len(public_wm) >= 8 else (Xk, Yk, np.zeros((ny, nx)))
    Z_pub_m = mask_polygon_rasterio(Z_pub, Xk, Yk, study_poly_wm)
    tr_xy = np.column_stack([transit_wm.geometry.x, transit_wm.geometry.y]) if len(transit_wm) else np.zeros((0, 2))
    d_grid = nearest_distance_grid(tr_xy, Xk, Yk)
    V_dwell_raw = np.nan_to_num(Z_pub_m, nan=0.0) * np.exp(-LAMBDA_TRANSIT_WM * d_grid)
    V_dwell_m = mask_polygon_rasterio(V_dwell_raw, Xk, Yk, study_poly_wm)

    Vf_n = norm_pct(np.nan_to_num(V_flow_m, nan=np.nan))
    Vc_n = norm_pct(np.nan_to_num(Z_com_m, nan=np.nan))
    Vd_n = norm_pct(np.nan_to_num(V_dwell_m, nan=np.nan))
    V_total = ALPHA_V * Vf_n + BETA_V * Vc_n + GAMMA_V * Vd_n
    V_total_m = mask_polygon_rasterio(V_total, Xk, Yk, study_poly_wm)

    finite_v = V_total_m[np.isfinite(V_total_m)]
    mu_v = float(np.nanmean(finite_v)) if finite_v.size else 0.0
    sig_v = float(np.nanstd(finite_v)) if finite_v.size else 1.0

    # --- V01 V^flow ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 10), dpi=160)
    add_wm_basemap_wm(ax, (xmin_wm, ymin_wm, xmax_wm, ymax_wm))
    Zplot = norm_pct(np.nan_to_num(V_flow_m, nan=np.nan))
    im = ax.imshow(
        Zplot,
        extent=(Xk.min(), Xk.max(), Yk.min(), Yk.max()),
        origin="lower",
        cmap=cmap_yellow_red(),
        alpha=0.82,
        zorder=2,
        interpolation="bilinear",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=r"$\tilde{V}^{flow}$ proxy (norm.)")
    site_wm.boundary.plot(ax=ax, color="#d90429", linewidth=3.2, zorder=6)
    ax.set_xlim(xmin_wm, xmax_wm)
    ax.set_ylim(ymin_wm, ymax_wm)
    ax.set_title(
        r"V$^{flow}$ proxy: $(\rho/\bar\rho)\cdot(1+\ln(1+Flow/\overline{Flow}))$ — Flow≈transit KDE",
        fontsize=11,
        pad=10,
    )
    fig.savefig(out_dir / "V01_h_index_contextual_influence.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- V02 V^com -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 10), dpi=160)
    add_wm_basemap_wm(ax, (xmin_wm, ymin_wm, xmax_wm, ymax_wm))
    Zplot = norm_pct(np.nan_to_num(Z_com_m, nan=np.nan))
    im = ax.imshow(
        Zplot,
        extent=(Xc.min(), Xc.max(), Yc.min(), Yc.max()),
        origin="lower",
        cmap="Oranges",
        alpha=0.82,
        zorder=2,
        interpolation="bilinear",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=r"$\tilde{V}^{com}$ proxy (norm.)")
    if len(commercial_wm):
        commercial_wm.plot(ax=ax, color="#ff006e", markersize=12, alpha=0.25, zorder=3)
    site_wm.boundary.plot(ax=ax, color="#d90429", linewidth=3.2, zorder=6)
    ax.set_xlim(xmin_wm, xmax_wm)
    ax.set_ylim(ymin_wm, ymax_wm)
    ttl = r"V$^{com}$ proxy: commerce POI KDE"
    if dp_scores is not None:
        ttl += " (评分-weighted duplicate sampling)"
    ax.set_title(ttl, fontsize=11, pad=10)
    fig.savefig(out_dir / "V02_commercial_vitality_kde.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- V03 POI pressure hex (facility density) -------------------------------------
    fig, ax = plt.subplots(figsize=(11, 10), dpi=160)
    add_wm_basemap_wm(ax, (xmin_wm, ymin_wm, xmax_wm, ymax_wm))
    if len(poi_wm):
        poi_wm.plot(ax=ax, color="#00d4ff", markersize=5, alpha=0.45, zorder=2)
        hb = ax.hexbin(
            poi_wm.geometry.x,
            poi_wm.geometry.y,
            gridsize=32,
            cmap="YlOrRd",
            mincnt=1,
            alpha=0.78,
            zorder=3,
            extent=(xmin_wm, xmax_wm, ymin_wm, ymax_wm),
        )
        plt.colorbar(hb, ax=ax, fraction=0.035, pad=0.02, label="POI count / hex")
    site_wm.boundary.plot(ax=ax, color="#d90429", linewidth=3.2, zorder=6)
    ax.set_xlim(xmin_wm, xmax_wm)
    ax.set_ylim(ymin_wm, ymax_wm)
    ax.set_title("V: combined POI facility pressure (hex density)", fontsize=12, pad=10)
    fig.savefig(out_dir / "V03_poi_pressure_hexbin.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- V04 composite V + μ±σ hint --------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 10), dpi=160)
    add_wm_basemap_wm(ax, plot_bounds_wm)
    Zplot = np.nan_to_num(V_total_m, nan=np.nan)
    im = ax.imshow(
        norm_pct(Zplot),
        extent=(Xk.min(), Xk.max(), Yk.min(), Yk.max()),
        origin="lower",
        cmap="RdYlGn_r",
        alpha=0.85,
        zorder=2,
        interpolation="bilinear",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=r"$\tilde{V}_i$ composite (norm.)")
    site_wm.boundary.plot(ax=ax, color="#d90429", linewidth=3.0, zorder=6)
    ax.set_xlim(xmin_wm, xmax_wm)
    ax.set_ylim(ymin_wm, ymax_wm)
    ax.set_title(
        rf"Composite $V_i$: $\alpha\tilde{{V}}^{{flow}}+\beta\tilde{{V}}^{{com}}+\gamma\tilde{{V}}^{{dwell}}$ "
        rf"($\mu_V$={mu_v:.3f}, $\sigma_V$={sig_v:.3f})",
        fontsize=10,
        pad=10,
    )
    fig.savefig(out_dir / "V04_vitality_conversion_hex.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- V05 explicit V^dwell ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 10), dpi=150)
    add_wm_basemap_wm(ax, (xmin_wm, ymin_wm, xmax_wm, ymax_wm))
    Zplot = norm_pct(np.nan_to_num(V_dwell_m, nan=np.nan))
    im = ax.imshow(
        Zplot,
        extent=(Xk.min(), Xk.max(), Yk.min(), Yk.max()),
        origin="lower",
        cmap="PuBuGn",
        alpha=0.82,
        zorder=2,
        interpolation="bilinear",
    )
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=r"$\tilde{V}^{dwell}$ proxy (norm.)")
    site_wm.boundary.plot(ax=ax, color="#d90429", linewidth=3.0, zorder=6)
    ax.set_xlim(xmin_wm, xmax_wm)
    ax.set_ylim(ymin_wm, ymax_wm)
    ax.set_title(r"V$^{dwell}$ proxy: public KDE $\times\,\exp(-\lambda d_{transit})$", fontsize=11, pad=10)
    fig.savefig(out_dir / "V05_V_dwell_public_transit_decay.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- T: Shannon entropy -----------------------------------------------------------
    if len(dp_wm) >= 20 and "营业时间" in dp.columns:
        dp = dp.copy()
        dp["__slice__"] = dp["营业时间"].apply(classify_hours_T15)
        cats = ["T1_morning_peak", "T2_flat", "T3_evening_peak", "T4_night", "T5_weekend"]
        vc = dp["__slice__"].value_counts().reindex(cats, fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
        vc.plot(kind="bar", ax=ax, color="#457b9d", edgecolor="k")
        ax.set_title(r"T: time-slice counts (proxy → $H_{T,i}$ inputs)")
        ax.set_ylabel("Count")
        fig.savefig(out_dir / "T01_temporal_class_bar.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # Hex entropy per doc §3.2 (5 bins)
        gwm = dp_wm.copy()
        gwm["_s"] = dp["__slice__"].values
        hx = []
        hy = []
        for c in cats:
            sub = gwm[gwm["_s"] == c]
            if len(sub):
                hx.append(sub.geometry.x.values)
                hy.append(sub.geometry.y.values)
            else:
                hx.append(np.array([]))
                hy.append(np.array([]))

        fig, ax = plt.subplots(figsize=(11, 9), dpi=140)
        add_wm_basemap_wm(ax, (xmin_wm, ymin_wm, xmax_wm, ymax_wm))
        # coarse grid counts for entropy
        gx = np.linspace(xmin_wm, xmax_wm, 28)
        gy = np.linspace(ymin_wm, ymax_wm, 28)
        H_map = np.full((len(gy) - 1, len(gx) - 1), np.nan)
        for i in range(len(gy) - 1):
            for j in range(len(gx) - 1):
                counts = []
                for k in range(5):
                    if hx[k].size == 0:
                        counts.append(0)
                        continue
                    m = (hx[k] >= gx[j]) & (hx[k] < gx[j + 1]) & (hy[k] >= gy[i]) & (hy[k] < gy[i + 1])
                    counts.append(int(np.sum(m)))
                ssum = sum(counts)
                if ssum < 3:
                    continue
                p = np.array(counts, dtype=float) / ssum
                p = np.clip(p, 1e-12, 1.0)
                h = -np.sum(p * np.log(p)) / np.log(5.0)
                H_map[i, j] = h
        im = ax.imshow(
            H_map,
            extent=(gx[0], gx[-1], gy[0], gy[-1]),
            origin="lower",
            cmap="viridis",
            alpha=0.88,
            zorder=2,
            vmin=0,
            vmax=1,
            aspect="auto",
        )
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=r"normalized $H_{T,i}$ (5 slices)")
        site_wm.boundary.plot(ax=ax, color="#d90429", linewidth=3, zorder=20)
        ax.set_xlim(xmin_wm, xmax_wm)
        ax.set_ylim(ymin_wm, ymax_wm)
        ax.set_title(r"T: Shannon temporal entropy $H_{T,i}$ (营业时间 coarse proxy)", fontsize=11)
        fig.savefig(out_dir / "T02_temporal_classes_map.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
        ax.text(0.5, 0.5, "T maps skipped: need Dianping GeoJSON with 营业时间 (≥20 pts)", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_dir / "T01_temporal_class_bar.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
        ax.text(0.5, 0.5, "T02 skipped: same condition", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_dir / "T02_temporal_classes_map.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # --- S layers ----------------------------------------------------------------------
    hcol = None
    for c in ("height", "Height", "HEIGHT", "建筑高度", "楼高"):
        if len(buildings) and c in buildings.columns:
            hcol = c
            break

    fig, ax = plt.subplots(figsize=(11, 10), dpi=160)
    add_wm_basemap_wm(ax, (xmin_wm, ymin_wm, xmax_wm, ymax_wm))
    if len(b_wm) and hcol:
        b_wm.plot(ax=ax, column=hcol, cmap="viridis", legend=True, alpha=0.65, edgecolor="#222", linewidth=0.2, zorder=3)
    elif len(b_wm):
        b_wm.plot(ax=ax, facecolor="#6c757d", edgecolor="#222", alpha=0.55, linewidth=0.2, zorder=3)
    site_wm.boundary.plot(ax=ax, color="#d90429", linewidth=3.2, zorder=6)
    ax.set_xlim(xmin_wm, xmax_wm)
    ax.set_ylim(ymin_wm, ymax_wm)
    ax.set_title("S: built footprints & height (ΔFAR / structure proxy — data partial)", fontsize=11)
    fig.savefig(out_dir / "S01_building_height.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 10), dpi=160)
    add_wm_basemap_wm(ax, (xmin_wm, ymin_wm, xmax_wm, ymax_wm))
    if len(b_wm) and hcol:
        tmp = b_wm.copy()
        tmp["__vol__"] = tmp.geometry.area * tmp[hcol].fillna(0).clip(lower=0)
        mx = tmp["__vol__"].quantile(0.95) + 1e-6
        tmp["__int__"] = (tmp["__vol__"] / mx).clip(0, 1)
        tmp.plot(ax=ax, column="__int__", cmap="magma", legend=True, alpha=0.7, edgecolor="#111", linewidth=0.15, zorder=3, legend_kwds={"label": "Built volume proxy"})
    site_wm.boundary.plot(ax=ax, color="#d90429", linewidth=3.2, zorder=6)
    ax.set_xlim(xmin_wm, xmax_wm)
    ax.set_ylim(ymin_wm, ymax_wm)
    ax.set_title("S: development intensity proxy (footprint × height)", fontsize=12)
    fig.savefig(out_dir / "S02_development_intensity_proxy.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if wp_tif is not None:
        try:
            fig, ax = plt.subplots(figsize=(11, 10), dpi=160)
            add_wm_basemap_wm(ax, plot_bounds_wm)
            arr, ext = warp_raster_band_to_wm(wp_tif, plot_bounds_wm, width=480, height=480)
            arr_f = np.asarray(arr, dtype=float)
            finite = arr_f[np.isfinite(arr_f)]
            if finite.size:
                vmin, vmax = np.percentile(finite, [5, 95])
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                    vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
            else:
                vmin, vmax = 0.0, 1.0
            im = ax.imshow(arr, extent=ext, origin="upper", cmap="YlGnBu", alpha=0.55, zorder=2, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=r"$\rho_i$ population density (WorldPop)")
            site_wm.boundary.plot(ax=ax, color="#d90429", linewidth=3.2, zorder=6)
            ax.set_xlim(xmin_wm, xmax_wm)
            ax.set_ylim(ymin_wm, ymax_wm)
            ax.set_title(r"S: $\rho_i$ — population density for $V^{flow}$ / capacity context", fontsize=11)
            fig.savefig(out_dir / "S03_worldpop_density.png", bbox_inches="tight", facecolor="white")
            plt.close(fig)
        except Exception as e:
            print("WorldPop map skip:", e, file=sys.stderr)

    if wp_tif is not None and len(b_wm) and hcol:
        try:
            import rasterio

            fig, ax = plt.subplots(figsize=(11, 10), dpi=140)
            add_wm_basemap_wm(ax, plot_bounds_wm)
            tmp = b_wm.copy()
            with rasterio.open(wp_tif) as src:
                tmp_native = tmp.to_crs(src.crs)
                coords = [(float(pt.x), float(pt.y)) for pt in tmp_native.geometry.centroid]
                vals = []
                for v in src.sample(coords):
                    arr = np.atleast_1d(np.asarray(v, dtype=float))
                    vals.append(float(arr[0]) if arr.size else np.nan)
            tmp["pop_s"] = np.array(vals, dtype=float)
            tmp["vol"] = tmp.geometry.area * tmp[hcol].fillna(1)
            tmp["cap"] = np.log1p(tmp["pop_s"]) / (np.log1p(tmp["vol"]) + 0.1)
            mx = tmp["cap"].quantile(0.92) + 1e-6
            tmp["cap_n"] = (tmp["cap"] / mx).clip(0, 1)
            tmp.plot(ax=ax, column="cap_n", cmap="Greens", legend=True, alpha=0.72, edgecolor="#222", linewidth=0.15, legend_kwds={"label": "Capacity residual proxy"})
            site_wm.boundary.plot(ax=ax, color="#d90429", linewidth=3, zorder=8)
            ax.set_xlim(xmin_wm, xmax_wm)
            ax.set_ylim(ymin_wm, ymax_wm)
            ax.set_title("S: pop vs built volume — capacity tension (doc §4.6 linkage)", fontsize=11)
            fig.savefig(out_dir / "S04_capacity_potential_proxy.png", bbox_inches="tight", facecolor="white")
            plt.close(fig)
        except Exception as e:
            print("S4 skip:", e, file=sys.stderr)

    # --- S5: heritage / conservation zones (doc P_risk proxy) ----------------------------
    if len(heritage_wm):
        fig, ax = plt.subplots(figsize=(11, 10), dpi=140)
        add_wm_basemap_wm(ax, (xmin_wm, ymin_wm, xmax_wm, ymax_wm))
        heritage_wm.plot(ax=ax, facecolor="#e9d5ff", edgecolor="#7c3aed", linewidth=1.2, alpha=0.55, zorder=3)
        site_wm.boundary.plot(ax=ax, color="#d90429", linewidth=3.2, zorder=8)
        ax.set_xlim(xmin_wm, xmax_wm)
        ax.set_ylim(ymin_wm, ymax_wm)
        ax.set_title("S: conservation / heritage scope (risk constraint proxy, doc sec. 4)", fontsize=11)
        fig.savefig(out_dir / "S05_heritage_conservation_scope.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 10), dpi=140)
    add_wm_basemap_wm(ax, (xmin_wm, ymin_wm, xmax_wm, ymax_wm))
    if len(roads_wm):
        roads_wm.plot(ax=ax, color="#495057", linewidth=0.6, alpha=0.85, zorder=2)
    if len(poi_wm):
        poi_wm.plot(ax=ax, color="#00d4ff", markersize=4, alpha=0.4, zorder=3)
    site_wm.boundary.plot(ax=ax, color="#d90429", linewidth=3.2, zorder=6)
    ax.set_xlim(xmin_wm, xmax_wm)
    ax.set_ylim(ymin_wm, ymax_wm)
    ax.set_title("Context: roads + POI (Carto Light)", fontsize=12)
    fig.savefig(out_dir / "00_context_roads_poi.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    meta = {
        "methods_doc": "docs/VTS_data_methods.md",
        "basemap": "CartoDB.Positron",
        "crs_plot": CRS_WM,
        "V_weights": {"alpha": ALPHA_V, "beta": BETA_V, "gamma": GAMMA_V},
        "lambda_transit_wm": LAMBDA_TRANSIT_WM,
        "outputs": [p.name for p in sorted(out_dir.glob("*.png"))],
        "notes": "Proxies per docs/VTS_data_methods.md sections 2-4; full Flow/FAR/Space Syntax need external datasets.",
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Wrote maps to:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
