from __future__ import annotations

from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import Polygon

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "output" / "site_realdata_maps"
CRS_WGS = "EPSG:4326"
CRS_WM = "EPSG:3857"

LANDUSE_COLORS = {
    "居住用地": "#f4a261",
    "商业服务用地": "#e76f51",
    "商务办公用地": "#b56576",
    "工业用地": "#6d597a",
    "行政办公用地": "#457b9d",
    "医疗卫生用地": "#2a9d8f",
    "交通场站用地": "#4d4d4d",
    "公园与绿地用地": "#7cb342",
}

HEIGHT_BINS = [0, 12, 24, 36, 60, 90, 999]
HEIGHT_LABELS = ["<=12m", "12-24m", "24-36m", "36-60m", "60-90m", ">90m"]
HEIGHT_COLORS = ["#fff4cc", "#fdd870", "#fca311", "#f77f00", "#d62828", "#6a040f"]


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


def clip_inputs(site_gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    building_path, landuse_path = resolve_input_paths()
    buildings = gpd.read_file(building_path)
    landuse = gpd.read_file(landuse_path)
    if buildings.crs is None:
        buildings = buildings.set_crs(CRS_WGS)
    if landuse.crs is None:
        landuse = landuse.set_crs(CRS_WGS)
    buildings = buildings.to_crs(CRS_WGS)
    landuse = landuse.to_crs(CRS_WGS)
    return gpd.clip(buildings, site_gdf), gpd.clip(landuse, site_gdf)


def plot_basemap(ax, bounds_wm: tuple[float, float, float, float]) -> None:
    xmin, ymin, xmax, ymax = bounds_wm
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    ctx.add_basemap(ax, crs=CRS_WM, source=ctx.providers.CartoDB.Positron, zoom="auto", attribution_size=6)


def padded_bounds(gdf_wm: gpd.GeoDataFrame, pad_ratio: float = 0.08) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = gdf_wm.total_bounds
    dx = (xmax - xmin) * pad_ratio
    dy = (ymax - ymin) * pad_ratio
    return xmin - dx, ymin - dy, xmax + dx, ymax + dy


def export_landuse(site_wm: gpd.GeoDataFrame, landuse: gpd.GeoDataFrame) -> Path:
    landuse = landuse.copy()
    landuse["class2"] = landuse["class2"].fillna("未分类")
    landuse["plot_color"] = landuse["class2"].map(LANDUSE_COLORS).fillna("#bdbdbd")
    landuse_wm = landuse.to_crs(CRS_WM)
    bounds = padded_bounds(site_wm)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=220)
    plot_basemap(ax, bounds)
    landuse_wm.plot(ax=ax, color=landuse_wm["plot_color"], edgecolor="white", linewidth=0.9, alpha=0.82, zorder=3)
    site_wm.boundary.plot(ax=ax, color="#111111", linewidth=2.2, zorder=4)

    area_df = landuse_wm.assign(area_sqm=landuse_wm.geometry.area).groupby("class2", as_index=False)["area_sqm"].sum()
    area_df = area_df.sort_values("area_sqm", ascending=False)
    total_area = area_df["area_sqm"].sum()
    handles = []
    for _, row in area_df.iterrows():
        label = f"{row['class2']} ({row['area_sqm'] / total_area:.0%})"
        handles.append(Patch(facecolor=LANDUSE_COLORS.get(row["class2"], "#bdbdbd"), edgecolor="none", label=label))
    handles.append(Line2D([0], [0], color="#111111", lw=2.2, label="SITE红线"))
    ax.legend(handles=handles, loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    ax.set_title("SITE内部用地类型图", fontsize=16, pad=12)

    out_path = OUT / "SITE_landuse_cartolight.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def export_height(site_wm: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> Path:
    buildings = buildings.copy()
    buildings["Height"] = pd.to_numeric(buildings["Height"], errors="coerce")
    buildings = buildings[buildings["Height"].notna()].copy()
    buildings_wm = buildings.to_crs(CRS_WM).sort_values("Height")
    bounds = padded_bounds(site_wm)

    cmap = ListedColormap(HEIGHT_COLORS)
    norm = BoundaryNorm(HEIGHT_BINS, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=220)
    plot_basemap(ax, bounds)
    buildings_wm.plot(ax=ax, column="Height", cmap=cmap, norm=norm, linewidth=0.08, edgecolor="#ffffff", alpha=0.92, zorder=3)
    site_wm.boundary.plot(ax=ax, color="#111111", linewidth=2.2, zorder=4)

    handles = [Patch(facecolor=c, edgecolor="none", label=l) for c, l in zip(HEIGHT_COLORS, HEIGHT_LABELS)]
    handles.append(Line2D([0], [0], color="#111111", lw=2.2, label="SITE红线"))
    ax.legend(handles=handles, loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    ax.set_title("SITE内部建筑高度图", fontsize=16, pad=12)

    out_path = OUT / "SITE_building_height_cartolight.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    configure_cn_font()
    OUT.mkdir(parents=True, exist_ok=True)

    site = load_site_polygon()
    buildings, landuse = clip_inputs(site)
    if buildings.empty:
        raise RuntimeError("SITE范围内未找到建筑高度数据。")
    if landuse.empty:
        raise RuntimeError("SITE范围内未找到用地类型数据。")

    site_wm = site.to_crs(CRS_WM)
    landuse_png = export_landuse(site_wm, landuse)
    height_png = export_height(site_wm, buildings)

    print(f"Wrote: {landuse_png}")
    print(f"Wrote: {height_png}")
    print(f"Landuse features in SITE: {len(landuse)}")
    print(f"Building features in SITE: {len(buildings)}")
    print(f"Landuse classes: {', '.join(sorted(map(str, landuse['class2'].dropna().unique().tolist())))}")
    print(f"Height range: {buildings['Height'].min():.2f}m - {buildings['Height'].max():.2f}m")


if __name__ == "__main__":
    main()
