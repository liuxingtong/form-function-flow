from __future__ import annotations

from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps, font_manager
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, box

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "output" / "site_texture_density"
CRS_WGS = "EPSG:4326"
CRS_WM = "EPSG:3857"
CELL = 12.0
SLICE_STEP = 3.0
MAX_HEIGHT = 50.0
FINAL_SLICE_TOP = 50.0

LANDUSE_BASE = {
    "公园与绿地用地": 0.10,
    "居住用地": 0.08,
    "医疗卫生用地": 0.16,
    "行政办公用地": 0.20,
    "工业用地": 0.28,
    "交通场站用地": 0.24,
    "商务办公用地": 0.28,
    "商业服务用地": 0.32,
}

LANDUSE_COLORS = {
    "公园与绿地用地": "#7cb342",
    "居住用地": "#f4a261",
    "医疗卫生用地": "#2a9d8f",
    "行政办公用地": "#457b9d",
    "工业用地": "#6d597a",
    "交通场站用地": "#4d4d4d",
    "商务办公用地": "#b56576",
    "商业服务用地": "#e76f51",
    "未分类": "#bdbdbd",
}

PROTOTYPE_STYLE = {
    "公园与绿地用地": {"mode": "cluster", "angle": 135, "color": "#7cb342"},
    "居住用地": {"mode": "cluster", "angle": 45, "color": "#f4a261"},
    "医疗卫生用地": {"mode": "band", "angle": 90, "color": "#2a9d8f"},
    "行政办公用地": {"mode": "band", "angle": 90, "color": "#457b9d"},
    "工业用地": {"mode": "bar", "angle": 20, "color": "#6d597a"},
    "交通场站用地": {"mode": "corridor", "angle": 20, "color": "#4d4d4d"},
    "商务办公用地": {"mode": "band", "angle": 0, "color": "#b56576"},
    "商业服务用地": {"mode": "corridor", "angle": 0, "color": "#e76f51"},
    "未分类": {"mode": "cluster", "angle": 0, "color": "#bdbdbd"},
}


def slice_bounds() -> list[tuple[float, float]]:
    bounds = []
    z = 0.0
    while z < FINAL_SLICE_TOP - 1e-9:
        z_high = min(z + SLICE_STEP, FINAL_SLICE_TOP)
        bounds.append((float(z), float(z_high)))
        z = z_high
    return bounds


SLICE_BOUNDS = slice_bounds()
HEIGHT_FACTORS = np.linspace(1.00, 0.22, len(SLICE_BOUNDS)).round(4).tolist()


def configure_cn_font() -> None:
    preferred = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans")
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_site_polygon() -> gpd.GeoDataFrame:
    site = gpd.read_file(DATA / "SITE.json")
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


def resolve_input_paths() -> tuple[Path, Path]:
    building = next((DATA / "site_3km").glob("01-*/*/*带高度.geojson"))
    landuse = next((DATA / "site_3km").glob("09-*/*/Data/*开源建设用地.geojson"))
    return building, landuse


def load_inputs(site: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    building_path, landuse_path = resolve_input_paths()
    buildings = gpd.read_file(building_path)
    landuse = gpd.read_file(landuse_path)
    if buildings.crs is None:
        buildings = buildings.set_crs(CRS_WGS)
    if landuse.crs is None:
        landuse = landuse.set_crs(CRS_WGS)
    site_wm = site.to_crs(CRS_WM)
    buildings = gpd.clip(buildings.to_crs(CRS_WM), site_wm)
    landuse = gpd.clip(landuse.to_crs(CRS_WM), site_wm)
    buildings["Height"] = pd.to_numeric(buildings["Height"], errors="coerce")
    buildings = buildings[buildings["Height"].notna()].copy()
    landuse["class2"] = landuse["class2"].fillna("未分类")
    return buildings, landuse


def make_grid(site_wm: gpd.GeoDataFrame, cell: float = CELL) -> gpd.GeoDataFrame:
    geom = site_wm.geometry.iloc[0]
    minx, miny, maxx, maxy = geom.bounds
    xs = np.arange(minx, maxx, cell)
    ys = np.arange(miny, maxy, cell)
    cells = []
    ids = []
    idx = 0
    for x in xs:
        for y in ys:
            sq = box(x, y, x + cell, y + cell)
            if sq.intersects(geom):
                cells.append(sq.intersection(geom))
                ids.append(idx)
                idx += 1
    grid = gpd.GeoDataFrame({"cell_id": ids}, geometry=cells, crs=CRS_WM)
    grid["cell_area"] = grid.geometry.area
    cent = grid.geometry.centroid
    grid["cx"] = cent.x
    grid["cy"] = cent.y
    return grid


def assign_landuse_base(grid: gpd.GeoDataFrame, landuse: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, float]]:
    ov = gpd.overlay(grid[["cell_id", "cell_area", "geometry"]], landuse[["class2", "geometry"]], how="intersection", keep_geom_type=False)
    ov["int_area"] = ov.geometry.area
    ov = ov.sort_values(["cell_id", "int_area"], ascending=[True, False])
    dominant = ov.groupby("cell_id", as_index=False).first()[["cell_id", "class2"]]
    out = grid.merge(dominant, on="cell_id", how="left")

    known = out[out["class2"].notna()].copy()
    unknown = out[out["class2"].isna()].copy()
    filled_count = 0
    if not unknown.empty and not known.empty:
        tree = cKDTree(known[["cx", "cy"]].to_numpy())
        _, idx = tree.query(unknown[["cx", "cy"]].to_numpy(), k=1)
        out.loc[unknown.index, "class2"] = known.iloc[idx]["class2"].to_numpy()
        filled_count = int(len(unknown))
    out["class2"] = out["class2"].fillna("未分类")
    out["base"] = out["class2"].map(LANDUSE_BASE).fillna(0.12)
    return out, {"cells_filled_by_nearest": filled_count, "total_cells": int(len(out))}


def occupancy_for_slice(grid: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame, z_low: float) -> np.ndarray:
    subset = buildings[buildings["Height"] > z_low][["geometry"]].copy()
    if subset.empty:
        return np.zeros(len(grid), dtype=float)
    ov = gpd.overlay(grid[["cell_id", "cell_area", "geometry"]], subset, how="intersection", keep_geom_type=False)
    if ov.empty:
        return np.zeros(len(grid), dtype=float)
    ov["int_area"] = ov.geometry.area
    cover = ov.groupby("cell_id", as_index=False)["int_area"].sum()
    vals = np.zeros(len(grid), dtype=float)
    frac = np.clip(cover["int_area"].to_numpy() / grid.set_index("cell_id").loc[cover["cell_id"], "cell_area"].to_numpy(), 0, 1)
    vals[cover["cell_id"].to_numpy().astype(int)] = frac
    return vals


def context_factor(grid: gpd.GeoDataFrame, occ: np.ndarray) -> np.ndarray:
    xs = np.sort(grid["cx"].unique())
    ys = np.sort(grid["cy"].unique())
    x_map = {v: i for i, v in enumerate(xs)}
    y_map = {v: i for i, v in enumerate(ys)}
    arr = np.full((len(ys), len(xs)), np.nan, dtype=float)
    mask = np.zeros((len(ys), len(xs)), dtype=bool)
    for cid, cx, cy in zip(grid["cell_id"], grid["cx"], grid["cy"]):
        iy = y_map[cy]
        ix = x_map[cx]
        arr[iy, ix] = occ[int(cid)]
        mask[iy, ix] = True
    base = np.where(mask, np.nan_to_num(arr, nan=0.0), 0.0)
    smooth = gaussian_filter(base, sigma=1.4)
    smooth[~mask] = np.nan
    finite = smooth[np.isfinite(smooth)]
    if finite.size == 0 or np.nanmax(finite) <= 1e-9:
        norm = np.zeros_like(smooth)
    else:
        p95 = np.nanpercentile(finite, 95)
        denom = p95 if p95 > 1e-9 else np.nanmax(finite)
        norm = np.clip(smooth / (denom + 1e-12), 0, 1)
    out = np.zeros(len(grid), dtype=float)
    for cid, cx, cy in zip(grid["cell_id"], grid["cx"], grid["cy"]):
        out[int(cid)] = 0.70 + 0.45 * float(norm[y_map[cy], x_map[cx]])
    return out


def build_slice_table(grid: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []
    for i, (z_low, z_high) in enumerate(SLICE_BOUNDS):
        occ = occupancy_for_slice(grid, buildings, z_low)
        ctxf = context_factor(grid, occ)
        t = np.clip(grid["base"].to_numpy() * HEIGHT_FACTORS[i] * ctxf, 0, 1)
        n = np.clip(t - occ, 0, 1)
        upper = float(np.nanmax(n)) if len(n) else 0.0
        rows.append(pd.DataFrame({
            "cell_id": grid["cell_id"],
            "z_low": z_low,
            "z_high": z_high,
            "existing": occ,
            "context": ctxf,
            "theoretical": t,
            "new_texture": n,
            "slice_upper": upper,
        }))
    return pd.concat(rows, ignore_index=True)


def plot_basemap(ax, bounds: tuple[float, float, float, float]) -> None:
    xmin, ymin, xmax, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    ctx.add_basemap(ax, crs=CRS_WM, source=ctx.providers.CartoDB.Positron, zoom="auto", attribution_size=5)


def padded_bounds(site_wm: gpd.GeoDataFrame, pad: float = 0.08) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = site_wm.total_bounds
    dx = (maxx - minx) * pad
    dy = (maxy - miny) * pad
    return minx - dx, miny - dy, maxx + dx, maxy + dy


def annotate_upper(ax, upper: float) -> None:
    ax.text(0.02, 0.98, f"本层新增肌理上限: {upper:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.92, "boxstyle": "round,pad=0.28"}, zorder=9)


def _slice_label(z_low: float, z_high: float) -> str:
    return f"{int(z_low):02d}_{int(z_high):02d}m"


def _prototype_segments(row: pd.Series, quota: float) -> list[tuple[tuple[float, float], tuple[float, float], str, float, str]]:
    style = PROTOTYPE_STYLE.get(row["class2"], PROTOTYPE_STYLE["未分类"])
    angle = np.deg2rad(style["angle"])
    ux, uy = np.cos(angle), np.sin(angle)
    vx, vy = -uy, ux
    x, y = float(row["cx"]), float(row["cy"])
    size = CELL * 0.44
    width = CELL * (0.12 + 0.28 * quota)
    color = style["color"]
    alpha = 0.28 + 0.58 * min(1.0, quota / 0.22)
    items = []
    mode = style["mode"]
    if mode == "corridor":
        offsets = [-0.28, 0.0, 0.28] if quota > 0.11 else [0.0]
        for off in offsets:
            ox, oy = vx * off * CELL, vy * off * CELL
            p0 = (x - ux * size + ox, y - uy * size + oy)
            p1 = (x + ux * size + ox, y + uy * size + oy)
            items.append((p0, p1, color, CELL * (0.11 + 0.12 * quota / 0.22), "solid"))
    elif mode == "band":
        offsets = [-0.22, 0.22] if quota > 0.10 else [0.0]
        for off in offsets:
            ox, oy = vx * off * CELL, vy * off * CELL
            p0 = (x - ux * size * 0.88 + ox, y - uy * size * 0.88 + oy)
            p1 = (x + ux * size * 0.88 + ox, y + uy * size * 0.88 + oy)
            items.append((p0, p1, color, CELL * (0.09 + 0.10 * quota / 0.22), "solid"))
    elif mode == "bar":
        offsets = [-0.18, 0.18] if quota > 0.12 else [0.0]
        for off in offsets:
            cx, cy = x + vx * off * CELL, y + vy * off * CELL
            p0 = (cx - ux * size * 0.75, cy - uy * size * 0.75)
            p1 = (cx + ux * size * 0.75, cy + uy * size * 0.75)
            items.append((p0, p1, color, CELL * (0.15 + 0.14 * quota / 0.22), "solid"))
    else:
        offsets = [(-0.18, -0.18), (0.18, 0.18), (-0.18, 0.18), (0.18, -0.18)]
        count = 4 if quota > 0.10 else 2
        for ox0, oy0 in offsets[:count]:
            cx, cy = x + ox0 * CELL, y + oy0 * CELL
            p0 = (cx - ux * width * 0.9, cy - uy * width * 0.9)
            p1 = (cx + ux * width * 0.9, cy + uy * width * 0.9)
            items.append((p0, p1, color, CELL * 0.09, "solid"))
    return [(p0, p1, c, lw, ls + f"|{alpha:.3f}") for p0, p1, c, lw, ls in items]


def export_slice_heatmaps(site_wm: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, slices: pd.DataFrame) -> list[Path]:
    bounds = padded_bounds(site_wm)
    norm = Normalize(vmin=0, vmax=max(0.16, float(slices["new_texture"].quantile(0.985))))
    out_paths = []
    for z_low, z_high in SLICE_BOUNDS:
        df = slices[slices["z_low"] == z_low]
        g = grid.merge(df[["cell_id", "new_texture", "slice_upper"]], on="cell_id", how="left")
        fig, ax = plt.subplots(figsize=(9, 9), dpi=220)
        plot_basemap(ax, bounds)
        g.plot(ax=ax, column="new_texture", cmap="YlOrRd", norm=norm, linewidth=0.0, alpha=0.78, zorder=3)
        site_wm.boundary.plot(ax=ax, color="#111111", linewidth=2.0, zorder=4)
        upper = float(df["slice_upper"].iloc[0])
        annotate_upper(ax, upper)
        ax.set_title(f"新增肌理密度场热力图 {int(z_low)}-{int(z_high)}m", fontsize=15, pad=10)
        sm = plt.cm.ScalarMappable(norm=norm, cmap="YlOrRd")
        cbar = fig.colorbar(sm, ax=ax, fraction=0.034, pad=0.01)
        cbar.set_label("可新增平面密度额度", fontsize=9)
        out = OUT / f"heatmap_slice_{_slice_label(z_low, z_high)}.png"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        out_paths.append(out)
    return out_paths


def export_slice_prototypes(site_wm: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, slices: pd.DataFrame) -> list[Path]:
    bounds = padded_bounds(site_wm)
    out_paths = []
    for z_low, z_high in SLICE_BOUNDS:
        df = slices[slices["z_low"] == z_low]
        g = grid.merge(df[["cell_id", "new_texture", "slice_upper"]], on="cell_id", how="left")
        active = g[g["new_texture"] > 0.012].copy()
        fig, ax = plt.subplots(figsize=(9, 9), dpi=220)
        plot_basemap(ax, bounds)
        g.plot(ax=ax, color="#f7f7f7", edgecolor="none", alpha=0.18, zorder=2)
        site_wm.boundary.plot(ax=ax, color="#111111", linewidth=2.0, zorder=5)
        for _, row in active.iterrows():
            quota = float(row["new_texture"])
            for p0, p1, color, lw, ls_alpha in _prototype_segments(row, quota):
                ls, alpha = ls_alpha.split("|")
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=lw, alpha=float(alpha), solid_capstyle="round", zorder=4)
        upper = float(df["slice_upper"].iloc[0])
        annotate_upper(ax, upper)
        ax.text(0.02, 0.02, "原型规则: 商业/交通=廊道, 办公/医疗=带状, 工业=条形, 居住/绿地=组团", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8.5, color="#222222",
                bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9, "boxstyle": "round,pad=0.24"}, zorder=9)
        ax.set_title(f"新增肌理原型图 {int(z_low)}-{int(z_high)}m", fontsize=15, pad=10)
        out = OUT / f"prototype_slice_{_slice_label(z_low, z_high)}.png"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        out_paths.append(out)
    return out_paths


def export_montage(site_wm: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, slices: pd.DataFrame, kind: str) -> Path:
    bounds = padded_bounds(site_wm)
    cols = 5
    rows = int(np.ceil(len(SLICE_BOUNDS) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(22, rows * 4.4), dpi=220)
    axes = np.atleast_1d(axes).ravel()
    norm = Normalize(vmin=0, vmax=max(0.16, float(slices["new_texture"].quantile(0.985))))
    for ax in axes[len(SLICE_BOUNDS):]:
        ax.axis("off")
    for ax, (z_low, z_high) in zip(axes, SLICE_BOUNDS):
        df = slices[slices["z_low"] == z_low]
        g = grid.merge(df[["cell_id", "new_texture", "slice_upper"]], on="cell_id", how="left")
        plot_basemap(ax, bounds)
        if kind == "heatmap":
            g.plot(ax=ax, column="new_texture", cmap="YlOrRd", norm=norm, linewidth=0.0, alpha=0.78, zorder=3)
        else:
            g.plot(ax=ax, color="#f7f7f7", edgecolor="none", alpha=0.18, zorder=2)
            active = g[g["new_texture"] > 0.018]
            for _, row in active.iterrows():
                for p0, p1, color, lw, ls_alpha in _prototype_segments(row, float(row["new_texture"])):
                    _, alpha = ls_alpha.split("|")
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=max(0.8, lw * 0.55), alpha=float(alpha), solid_capstyle="round", zorder=4)
        site_wm.boundary.plot(ax=ax, color="#111111", linewidth=1.2, zorder=5)
        upper = float(df["slice_upper"].iloc[0])
        ax.set_title(f"{int(z_low)}-{int(z_high)}m | 上限 {upper:.3f}", fontsize=10.5, pad=5)
    if kind == "heatmap":
        sm = plt.cm.ScalarMappable(norm=norm, cmap="YlOrRd")
        cbar = fig.colorbar(sm, ax=list(axes[:len(SLICE_BOUNDS)]), fraction=0.018, pad=0.01)
        cbar.set_label("可新增平面密度额度", fontsize=10)
        title = "SITE新增肌理密度场热力图总览"
        out = OUT / "texture_density_heatmap_montage.png"
    else:
        title = "SITE新增肌理原型图总览"
        out = OUT / "texture_density_prototype_montage.png"
    fig.suptitle(title, fontsize=18, y=0.995)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def export_3d(grid: gpd.GeoDataFrame, slices: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(12, 10), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    polys = []
    facecolors = []
    cmap = colormaps["YlOrRd"]
    vmax = max(0.16, float(slices["new_texture"].quantile(0.985)))
    norm = Normalize(vmin=0, vmax=vmax)
    step = CELL
    minx, miny, maxx, maxy = grid.total_bounds
    for z_low, z_high in SLICE_BOUNDS:
        df = slices[slices["z_low"] == z_low]
        merged = grid.merge(df[["cell_id", "new_texture"]], on="cell_id", how="left")
        pick = merged[merged["new_texture"] > 0.015]
        for _, row in pick.iterrows():
            x = row["cx"]
            y = row["cy"]
            h = row["new_texture"]
            sx = step * 0.42
            sy = step * 0.42
            polys.append([(x - sx, y - sy, z_high), (x + sx, y - sy, z_high), (x + sx, y + sy, z_high), (x - sx, y + sy, z_high)])
            facecolors.append(cmap(norm(h)))
    coll = Poly3DCollection(polys, facecolors=facecolors, edgecolors="none", alpha=0.76)
    ax.add_collection3d(coll)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_zlim(0, FINAL_SLICE_TOP)
    ax.set_box_aspect((maxx - minx, maxy - miny, FINAL_SLICE_TOP * 16))
    ax.view_init(elev=32, azim=-58)
    ax.set_title("SITE新增肌理密度场 3D分层图", pad=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height (m)")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.08)
    cbar.set_label("可新增平面密度额度", fontsize=9)
    out = OUT / "texture_density_3d.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def export_metadata(grid: gpd.GeoDataFrame, slices: pd.DataFrame, landuse_fill: dict[str, float]) -> Path:
    summ = slices.groupby(["z_low", "z_high"], as_index=False).agg(
        avg_new_texture=("new_texture", "mean"),
        max_new_texture=("new_texture", "max"),
        avg_existing=("existing", "mean"),
        slice_upper=("slice_upper", "max"),
    )
    meta = {
        "cell_size_m": CELL,
        "slice_step_m": SLICE_STEP,
        "max_height_m": MAX_HEIGHT,
        "slice_bounds": SLICE_BOUNDS,
        "landuse_base": LANDUSE_BASE,
        "height_factors": HEIGHT_FACTORS,
        "landuse_fill": landuse_fill,
        "site_cells": int(len(grid)),
        "slice_summary": summ.to_dict(orient="records"),
    }
    out = OUT / "texture_density_meta.json"
    pd.Series(meta).to_json(out, force_ascii=False, indent=2)
    return out


def main() -> None:
    configure_cn_font()
    OUT.mkdir(parents=True, exist_ok=True)
    site = load_site_polygon()
    site_wm = site.to_crs(CRS_WM)
    buildings, landuse = load_inputs(site)
    grid = make_grid(site_wm)
    grid, landuse_fill = assign_landuse_base(grid, landuse)
    slices = build_slice_table(grid, buildings)

    heatmaps = export_slice_heatmaps(site_wm, grid, slices)
    prototypes = export_slice_prototypes(site_wm, grid, slices)
    heatmap_montage = export_montage(site_wm, grid, slices, kind="heatmap")
    prototype_montage = export_montage(site_wm, grid, slices, kind="prototype")
    view3d = export_3d(grid, slices)
    meta = export_metadata(grid, slices, landuse_fill)

    print(f"Wrote {len(heatmaps)} heatmaps")
    print(f"Wrote {len(prototypes)} prototype maps")
    print(f"Wrote: {heatmap_montage}")
    print(f"Wrote: {prototype_montage}")
    print(f"Wrote: {view3d}")
    print(f"Wrote: {meta}")
    print(f"Grid cells: {len(grid)}")
    print(f"Nearest-filled landuse cells: {landuse_fill['cells_filled_by_nearest']}")
    print(f"Average new texture quota: {slices['new_texture'].mean():.4f}")


if __name__ == "__main__":
    main()
